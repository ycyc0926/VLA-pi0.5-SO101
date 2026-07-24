import cv2
import requests
import time
import numpy as np
import os
import sys
import json
from pathlib import Path
from types import SimpleNamespace
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

# ================= 配置区 =================
SERVER_URL = "http://192.168.1.110:8080/predict"
ROBOT_PORT = "/dev/ttyACM2"
# 自动定位您的校准文件
CALIB_FILE_PATH = os.path.expanduser("~/.cache/huggingface/lerobot/calibration/robots/so_follower/my_awesome_follower_arm.json")

print("="*60)
print(f"🤖 [Client] 正在启动 SO101 深度兼容模式...")

# 1. 预处理校准数据 (将字典转换为底层总线需要的对象格式)
if not os.path.exists(CALIB_FILE_PATH):
    print(f"❌ 错误: 找不到校准文件 {CALIB_FILE_PATH}")
    sys.exit()

with open(CALIB_FILE_PATH, 'r') as f:
    raw_calib_data = json.load(f)

formatted_calib = {}
for motor_name, values in raw_calib_data.items():
    # 将 [min, pos, max] 列表包装成带有 range_min/max 属性的对象
    if isinstance(values, list):
        formatted_calib[motor_name] = SimpleNamespace(range_min=values[0], range_max=values[2])
    else:
        formatted_calib[motor_name] = SimpleNamespace(**values)

# 2. 初始化机器人与驱动
CALIB_DIR = Path(os.path.dirname(CALIB_FILE_PATH))
config = SO101FollowerConfig(port=ROBOT_PORT, calibration_dir=CALIB_DIR)
robot = SO101Follower(config)

# 3. 强力 Hack：绕过只读属性限制，强制标记为已校准
type(robot).is_calibrated = property(lambda self: True)
robot.calibration = raw_calib_data

# 4. 连接并注入底层总线校准
try:
    robot.connect()
    if hasattr(robot, 'bus'):
        # 直接覆盖底层总线的校准属性，防止 sync_read 报错
        if hasattr(robot.bus, 'calibration'):
            robot.bus.calibration = formatted_calib
        else:
            robot.bus._calibration = formatted_calib
        print("✅ [Client] 底层总线校准对象注入成功。")
    print("✅ [Client] 物理链路全通！SO101 已就绪。")
except Exception as e:
    print(f"❌ [Client] 连接或注入失败: {e}")
    sys.exit()

# ================= 相机初始化 =================
print("📷 [Client] 正在打开相机 (4: 全景, 2: 腕部)...")
cap_env = cv2.VideoCapture(4)
cap_wrist = cv2.VideoCapture(2)

def preprocess(frame):
    # 改为模型期望的 480x640 (或者干脆不 resize，直接用原图尺寸)
    img = cv2.resize(frame, (640, 480)) 
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return (img.transpose(2, 0, 1) / 255.0).tolist()

# ================= 推理主循环 =================
try:
    chunk_count = 0
    # 定义驱动返回的具体 Key 顺序，用于组装状态向量
    joint_keys = [
        'shoulder_pan.pos', 
        'shoulder_lift.pos', 
        'elbow_flex.pos', 
        'wrist_flex.pos', 
        'wrist_roll.pos', 
        'gripper.pos'
    ]

    print("\n🚀 [Observe Mode] 实时观察模式启动！")
    print("="*60)

    while True:
        # 1. 采集图像
        ret1, frame_env = cap_env.read()
        ret2, frame_wrist = cap_wrist.read()
        if not ret1 or not ret2:
            continue

        # 2. 手动从散装字典中提取 7 维状态
        try:
            obs = robot.get_observation()
            current_state = []
            
            # 按顺序抓取每个关节的数值
            for key in joint_keys:
                val = obs.get(key)
                # 处理可能返回的 Tensor 或标量
                if hasattr(val, 'item'): val = val.item()
                current_state.append(float(val) if val is not None else 0.0)

            # 如果数据只有 6 个（无夹爪），补齐到 7 维以适配 PI0 输入
            while len(current_state) < 7:
                current_state.append(0.0)

        except Exception as e:
            print(f"⚠️ 状态读取异常: {e}")
            continue

        # 3. 构造推理请求
        payload = {
            "img_env": preprocess(frame_env),
            "img_wrist": preprocess(frame_wrist),
            "state": current_state[:7] # 仅取前 7 位
        }
        
        try:
            # 向 GPU 服务器发起请求
            req_start = time.time()
            r = requests.post(SERVER_URL, json=payload, timeout=5)
            r.raise_for_status()
            actions = r.json()['actions']
            rtt = (time.time() - req_start) * 1000
            
            # 实时打印物理状态与预测结果对比
            print(f"\n📡 [Chunk #{chunk_count}] RTT: {rtt:.1f}ms")
            print(f"📍 物理位置: {[round(s, 1) for s in current_state[:7]]}")
            print(f"🔮 模型建议: {[round(a, 1) for a in actions[0][:7]]}")

            # 4. 模拟序列展示 (15 步动作为一跳)
            for i, act in enumerate(actions):
                # --- 物理控制开关：确认安全后取消下面这行注释 ---
                # robot.send_action(np.array(act[:7])) 
                
                if i % 7 == 0:
                    print(f"   ∟ Step {i:02d} | 预测姿态: {[round(x, 1) for x in act[:6]]} | 夹爪: {act[6]:.2f}")
                time.sleep(0.033) 
                
        except Exception as e:
            print(f"❌ 推理请求出错: {e}")
            time.sleep(1)
        
        chunk_count += 1

except KeyboardInterrupt:
    print("\n👋 正在断开连接并释放资源...")
    robot.disconnect()
    cap_env.release()
    cap_wrist.release()
    print("✨ 已安全退出。")