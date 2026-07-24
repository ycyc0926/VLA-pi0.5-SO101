#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Standalone SO101 inference & control loop (no record.py).

Changes in this version:
- Enhanced get_state6() with retry mechanism to handle zero-state errors
- Modified pack_action_joint_abs() to allow gripper control with safety limits
- Added detailed debug for state and gripper actions
- Kept robust error handling from previous version
"""

import time
import argparse
import numpy as np
import cv2
import concurrent.futures
import traceback
import queue
import threading

from openpi_client import image_tools
from openpi_client.websocket_client_policy import WebsocketClientPolicy
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

# ------------------------- Camera helpers (OpenCV, 摄像头) -------------------------

def open_cam(index: int, w: int = 640, h: int = 480) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    # 🚨 终极天眼：强行把 Windows 摄像头缓存队列砍到 1！彻底告别残影！
    #cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera at index {index}")
    return cap


DEBUG_SAVE_VISION = True 

def get_rgb_224_from_cap(cap, save_name="debug_cam"):
    ret, frame_bgr = cap.read()
    if not ret or frame_bgr is None:
        print(f"[WARN] Failed to read {save_name}")
        return np.zeros((224, 224, 3), dtype=np.uint8)

    # 1. 严格按照 lkw 逻辑：BGR 转 RGB
    frame_rgb = frame_bgr[:, :, ::-1]
    
    # 2. 保持长宽比缩放并填黑边，保证距离感绝对正确
    frame_rgb = image_tools.resize_with_pad(frame_rgb, 224, 224)
    frame_rgb = image_tools.convert_to_uint8(frame_rgb)
    
    # 保存未经颜色污染的最终 RGB 图像供你排查
    if DEBUG_SAVE_VISION:
        # OpenCV 保存时用 BGR 才正常，所以我们翻转回去存一张
        cv2.imwrite(f"{save_name}.jpg", frame_rgb[:, :, ::-1])

    # 🚨 绝对核心：直接返回 Numpy 矩阵，绝不压缩成 tobytes！
    return frame_rgb

#旧版未压缩
#def get_rgb_frame(cap: cv2.VideoCapture):
#    # 丢弃缓冲区内的 4-5 帧旧图，确保 read() 拿到的是最新的硬件捕捉
#    for _ in range(8):
#        cap.grab()
#    ok, frame = cap.read()
#    if not ok:
#        raise RuntimeError("Camera Disconnected") # 强制触发主循环的 hold 逻辑
#    return frame[:, :, ::-1]

DEBUG_SAVE_VISION = True 

def get_rgb_224_from_cap(cap, save_name="debug_cam"):
    ret, frame = cap.read()
    if not ret or frame is None:
        print(f"[WARN] Failed to read {save_name}")
        return np.zeros((224, 224, 3), dtype=np.uint8)
        
    # 1. 保持长宽比缩放并填充黑边 (这步是对的，你看保存的图片就知道了)
    frame_padded = image_tools.resize_with_pad(frame, 224, 224)
    
    # 保存本地查看
    if DEBUG_SAVE_VISION:
        cv2.imwrite(f"{save_name}.jpg", frame_padded)

    # 2. BGR 转 RGB
    frame_rgb = cv2.cvtColor(frame_padded, cv2.COLOR_BGR2RGB)
    frame_rgb = image_tools.convert_to_uint8(frame_rgb)
    
    # 🚨 绝对核心：直接返回 numpy 矩阵！绝不能压缩成 tobytes()！
    return frame_rgb


# ------------------------- Action post-processing -------------------------
# 平滑滤波和速度限制
def blend_and_rate_limit_abs_q(
    q_curr: np.ndarray,
    q_tgt: np.ndarray,
    alpha: float,
    dq_limit_rad: float,
) -> np.ndarray:
    q_curr = np.asarray(q_curr, dtype=np.float32).reshape(-1)
    q_tgt = np.asarray(q_tgt, dtype=np.float32).reshape(-1)

    q_blend = (1.0 - alpha) * q_curr + alpha * q_tgt # 平滑滤波
    dq = np.clip(q_blend - q_curr, -dq_limit_rad, dq_limit_rad) # 速度限制
    return q_curr + dq

# ------------------------- Robot state helpers -------------------------
# 通过循环重试机制，确保能稳定获取到机械臂的 6 维状态向量（float32），避免通讯瞬间中断导致的零向量错误
def get_state6(robot: SO101Follower, retries: int = 3, delay: float = 0.1) -> np.ndarray:
    for attempt in range(retries):
        try:
            obs = robot.get_observation()
            
            # --- 修改核心逻辑：手动从具体键中提取 6 个关节的数值 ---
            # 顺序必须对应：pan, lift, elbow, flex, roll, gripper
            keys = [
                'shoulder_pan.pos', 'shoulder_lift.pos', 'elbow_flex.pos', 
                'wrist_flex.pos', 'wrist_roll.pos', 'gripper.pos'
            ]
            
            # 如果 obs 里包含这些键，提取它们
            if all(k in obs for k in keys):
                st = np.array([obs[k] for k in keys], dtype=np.float32)
                
                if all(k in obs for k in keys):
                    st = np.array([obs[k] for k in keys], dtype=np.float32)
                    if not np.allclose(st, 0.0):
                        # print(f"[DEBUG] Robot Observation (Raw): {st}")
                        # --- 关键修改：如果机器人返回的是角度（比如 -90），强制转为弧度（-1.57） ---
                        # if robot.config.use_degrees:
                        #     # 只对前 5 个轴转弧度，第 6 个轴（夹爪）保持原样
                        #     res = np.zeros(6, dtype=np.float32)
                        #     res[:5] = np.deg2rad(st[:5])
                        #     res[5] = st[5] # 夹爪直接赋值
                        #     return res
                        return st
            
            print(f"[WARN] get_state6 attempt {attempt + 1}/{retries} returned invalid or zero state")
            time.sleep(delay)
        except Exception as e:
            print(f"[ERROR] get_state6 attempt {attempt + 1}/{retries} failed: {e}")
            time.sleep(delay)

    print("[ERROR] get_state6 failed after retries; returning zeros")
    return np.zeros(6, dtype=np.float32)

# 将模型输出的 6 维动作向量映射到 SO101 机械臂的具体关节特征上，并且对**夹爪（ID=6）**实施了关键的安全限程保护
def pack_action_joint_abs(robot: SO101Follower, q_cmd: np.ndarray) -> dict:
    """
    Pack a 6D absolute joint target vector, including gripper with safety limits.
    Gripper range: [-1.0, 1.0] radians (approx -57 to 57 degrees).
    """
    feat_keys = list(robot.action_features.keys())
    obs = robot.get_observation()
    current_gripper = obs.get("gripper", obs.get("gripper.pos", 0.0))
    action = {}

    for i, k in enumerate(feat_keys[:5]):  # Joints 1-5
        action[k] = float(q_cmd[i])

    if len(feat_keys) >= 6:  # Gripper (ID=6)
        gripper_cmd = float(q_cmd[5])
        # Safety: clip gripper to safe range (approx -57 to 57 deg in radians)
        gripper_cmd = np.clip(gripper_cmd, 0, 100) # -1.0, 1.0
        action[feat_keys[5]] = gripper_cmd
        print(f"[DEBUG] gripper command: {gripper_cmd} (clipped to [0, 100] degrees)")

    for k in feat_keys[6:]:
        action[k] = float(current_gripper) if "gripper" in k else 0.0

    return action



# 在程序出现异常或需要紧急停顿时，通过发送当前位置或最后一次指令位置，让机械臂“保持原位”不动，从而防止因失控导致的意外移动
def send_hold_action(robot: SO101Follower, last_q_sent: np.ndarray = None):
    """
    Send safe hold action using last_q_sent or current state.
    """
    try:
        if last_q_sent is None:
            q_curr = get_state6(robot)
        else:
            q_curr = last_q_sent
        action_dict = pack_action_joint_abs(robot, q_curr)
        robot.send_action(action_dict)
    except Exception as e:
        print("[WARN] failed to send hold action:", e)

# ==========================================
# 🚀 官方神技 1：全局弹夹与可重入硬件锁
# ==========================================
robot_lock = threading.RLock()  # 使用 RLock 允许同一线程多次获取锁，防止死锁
action_queue = queue.Queue()    # 动作弹夹

# ------------------------- Main loop -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="192.168.1.110", help="OpenPI policy server IP (WebSocket)")
    ap.add_argument("--port", type=int, default=5000, help="OpenPI policy server port (端口)")
    ap.add_argument(
        "--prompt",
        type=str,
        default="Grab the black cube and place it in the white cup",
        help="Task instruction sent to the policy",
    )
    ap.add_argument("--serial", default="/dev/ttyACM2", help="S0101 serial port (串口)")
    ap.add_argument("--use_degrees", action="store_true", help="Send degrees instead of radians to the robot (度 or 弧度)")
    ap.add_argument("--hz", type=int, default=30, help="Control frequency (控制频率 Hz)")
    ap.add_argument("--cam_top", type=int, default=4, help="OpenCV index for top cam (顶部)")
    ap.add_argument("--cam_wrist", type=int, default=2, help="OpenCV index for wrist cam")
    ap.add_argument("--dq_limit_deg", type=float, default=2.0, help="Per-step max joint change (deg 每步最大关节变化度数)")
    ap.add_argument("--alpha", type=float, default=0.2, help="Blending factor for absolute targets (0~1 混合因子)")
    ap.add_argument("--infer_timeout", type=float, default=5.0, help="Infer timeout in seconds (推理超时，秒)")
    args = ap.parse_args()

    dt = 1.0 / float(args.hz)

    robot = SO101Follower(SO101FollowerConfig(
        port=args.serial,
        use_degrees=args.use_degrees,
        id="my_awesome_follower_arm",
    ))
    
    try:
        robot.connect()
        # 强制更新一次观察值，激活底层缓存
        _ = robot.get_observation()
        print("Robot connected. action_features =", list(robot.action_features.keys()))
    except Exception as e:
        print(f"[ERROR] Failed to connect robot: {e}")
        return
    
    cap_top = open_cam(args.cam_top)
    cap_wrist = open_cam(args.cam_wrist)
    print(f"Opened cameras: top index={args.cam_top}, wrist index={args.cam_wrist}")

    client = WebsocketClientPolicy(host=args.host, port=args.port)
    try:
        meta = client.get_server_metadata()
    except Exception as e:
        meta = {}
        print("[WARN] get_server_metadata() raised:", e)
    print("Connected to policy server. Meta:", meta)
    
    dq_limit_rad = np.deg2rad(args.dq_limit_deg)
    infer_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1) # 使用线程池处理异步推理
    INFER_TIMEOUT = float(args.infer_timeout)

    try:
        with robot_lock:
            init_state = get_state6(robot)
        print(f"[DEBUG] Robot State (Converted to Radians): {init_state}")
        last_q_sent = np.array(init_state, dtype=np.float32)
        if np.allclose(last_q_sent, 0.0):
            print("[WARN] initial robot state is all zeros; caching zeros as last_q_sent.")
    except Exception:
        last_q_sent = np.zeros(6, dtype=np.float32)
        print("[WARN] failed to read initial robot state; using zeros for last_q_sent.")

    # ====== 🚨 核心修复：请确保这两行在这里，且缩进与 try 块对齐！ ======
    q_curr_used = last_q_sent.copy()
    prev_actions_summary = None
    
    # ========================================================
    # 🚀 官方神技 2：后台推理线程 (融合了你所有的安全校验！)
    # ========================================================
    def inference_worker():
        nonlocal prev_actions_summary, last_q_sent
        print("[INFO] 🧠 后台推理大脑已启动, 包含严格数据校验...")
        while True:
            # ==========================================================
            # 核心修改：在这里控制一次推理执行的步数
            # 如果动作还有剩余（比如大于 20 步），就先别去推理新的
            # 让相对位置模型把积累的 Delta 动作走完，机械臂才能动起来！
            # ==========================================================
            if action_queue.qsize() > 15:  # <--- 修改这里的数字控制执行步数
                time.sleep(0.01)
                continue
            
            t_prep_start = time.perf_counter()
            
            # 1. 拍照并直接拿到 Numpy 矩阵
            rgb_env = get_rgb_224_from_cap(cap_top, "model_sees_top")
            rgb_wrist = get_rgb_224_from_cap(cap_wrist, "model_sees_wrist")
            
            with robot_lock:
                st = get_state6(robot)

            # 2. 直接把矩阵装进字典发给服务端
            obs = {
                "observation.images.images_env":   rgb_env,
                "observation.images.images_wrist": rgb_wrist,
                "observation.state":               st, 
                "prompt":                          args.prompt,
            }
            
            t_prep_ms = (time.perf_counter() - t_prep_start) * 1000
            
            # 2. 状态保护逻辑
            q_curr_raw = st
            state_is_zero = np.allclose(q_curr_raw, 0.0)
            if state_is_zero:
                print("[WARN] robot reported state all zeros - using cached fallback")

            # 3. 发送给服务器进行推理 (使用你原来的 future.submit)
            t_net_start = time.perf_counter()
            future = infer_executor.submit(lambda o=obs: client.infer(o))
            try:
                res = future.result(timeout=INFER_TIMEOUT)
                t_total_round_ms = (time.perf_counter() - t_net_start) * 1000
                server_infer_ms = res.get("server_timing", {}).get("infer_ms", 0)
                
                print("-" * 40)
                print(f"📊 1.本地处理: {t_prep_ms:.1f}ms | 2.网络往返: {t_total_round_ms - server_infer_ms:.1f}ms | 3.服务端推理: {server_infer_ms:.1f}ms")
                print(f"🚀 总响应时间: {t_prep_ms + t_total_round_ms:.1f} ms")
            
            # ——————————  异常拦截与保护逻辑  ——————————
            except concurrent.futures.TimeoutError:
                print(f"[WARN] 推理超时({INFER_TIMEOUT}s)！发送 Hold 指令")
                with robot_lock:
                    send_hold_action(robot, last_q_sent)
                try:
                    future.cancel()
                except Exception:
                    pass
                time.sleep(dt)
                continue
            except Exception as e:
                print("[ERROR] 推理异常:", e)
                traceback.print_exc()
                with robot_lock:
                    send_hold_action(robot, last_q_sent)
                time.sleep(1.0)
                continue

            if not isinstance(res, dict) or "actions" not in res:
                print("[WARN] infer() returned unexpected response:", res)
                with robot_lock:
                    send_hold_action(robot, last_q_sent)
                time.sleep(dt)
                continue
            
            try:
                actions_chunk = np.asarray(res["actions"], dtype=np.float32)
            except Exception as e:
                print("[ERROR] failed to parse actions:", e, "res:", res)
                with robot_lock:
                    send_hold_action(robot, last_q_sent)
                time.sleep(dt)
                continue

            if actions_chunk.size == 0:
                print("[WARN] actions_chunk is empty.")
                with robot_lock:
                    send_hold_action(robot, last_q_sent)
                time.sleep(dt)
                continue

            # Debug Summary
            first_steps = actions_chunk[:min(2, len(actions_chunk))]
            summary = {
                "shape": actions_chunk.shape,
                "first_mean": np.mean(first_steps),
                "first_min": np.min(first_steps),
                "first_max": np.max(first_steps),
                "nan_any": np.isnan(actions_chunk).any(),
                "inf_any": np.isinf(actions_chunk).any(),
            }
            print("[DEBUG] actions summary:", summary)

            if summary["nan_any"] or summary["inf_any"]:
                print("[ERROR] actions contain NaN/Inf. Sending hold action.")
                with robot_lock:
                    send_hold_action(robot, last_q_sent)
                time.sleep(dt)
                continue

            if np.allclose(actions_chunk, 0.0):
                print("[WARN] actions_chunk all zeros; sending hold.")
                with robot_lock:
                    send_hold_action(robot, last_q_sent)
                time.sleep(dt)
                continue
            
            if prev_actions_summary is not None and np.isclose(prev_actions_summary["first_mean"], summary["first_mean"]) \
               and np.isclose(prev_actions_summary["first_min"], summary["first_min"]) \
               and np.isclose(prev_actions_summary["first_max"], summary["first_max"]):
                print("[DEBUG] actions appear similar to previous cycle (may be steady point)")
            prev_actions_summary = summary
            # ——————————————————————————————————————————————————————————

            # 拿到大模型的 50 步绝对位置轨迹
            actions_chunk = np.asarray(res["actions"])
            
            # 只有当动作是合理的，才清空旧弹夹，装入新动作
            action_queue.queue.clear()
            for t in range(actions_chunk.shape[0]):
                action_queue.put(actions_chunk[t])

            

    # 启动后台线程
    threading.Thread(target=inference_worker, daemon=True).start()

    # ========================================================
    # 3：精准 30Hz 硬件控制循环 (消费者)
    # ========================================================
    print("[INFO] 🦾 主循环已启动，准备以严格 30Hz 执行动作...")
    step_idx = 0  # 恢复你的步数打印
    
    try:
        while True:
            t_start = time.perf_counter()

            if not action_queue.empty():
                a = action_queue.get().copy()

                # ====== 🚨 我们永远信任服务器发来的绝对位置，绝不累加！ ======
                target_joints = a[:5]

                # ====== 1. 身体关节：柔顺滤波与限速 ======
                alpha_arm = args.alpha 
                
                # 第一步：计算柔顺预测值
                q_expected = (1.0 - alpha_arm) * q_curr_used[:5] + alpha_arm * target_joints
                expected_delta = q_expected - q_curr_used[:5]
                
                # 第二步：获取限速值（弧度/度）
                max_dq = args.dq_limit_deg if args.use_degrees else np.deg2rad(args.dq_limit_deg)
                
                # 第三步：硬限速截断
                safe_delta = np.clip(expected_delta, -max_dq, max_dq)
                
                # 最终下发安全目标
                q_cmd_joints = q_curr_used[:5] + safe_delta
                
                # ====== 2. 夹爪保护 ======
                gripper_cmd = a[5]
                # GRIPPER_SAFE_MIN = 14.0  # 物理安全极限可按需解开
                # if gripper_cmd < GRIPPER_SAFE_MIN:
                #     gripper_cmd = GRIPPER_SAFE_MIN

                q_cmd = np.concatenate([q_cmd_joints, [gripper_cmd]])
                
                # 恢复你喜欢的多行打印详情
                print(f"\n--- 虚拟执行步 (步数剩余: {action_queue.qsize():02d}) ---")
                print(f"当前角度 (Actual): {[round(x, 3) for x in q_curr_used[:5]]}")
                print(f"模型预测 (Target): {[round(x, 3) for x in a[:5]]}")
                print(f"夹爪预测 (Gripper): {a[5]}")

                if args.use_degrees:
                    q_cmd_to_send = q_cmd.copy()
                else:
                    q_cmd_to_send = q_cmd
                    print(f"[UNIT CHECK] Sending RADIANS to robot: {q_cmd_to_send}")
                
                print(f"[STEP] a_mean={np.mean(a):.4f} q_curr_used={q_curr_used} q_cmd={q_cmd}")

                # ====== 3. 硬件加锁保护并发送 ======
                with robot_lock:
                    try:
                        action_dict = pack_action_joint_abs(robot, q_cmd_to_send)
                        robot.send_action(action_dict)

                        if args.use_degrees:
                            last_q_sent = q_cmd_to_send
                            print(f"[UNIT CHECK] last_q_sent degree to robot: {last_q_sent}")
                        else:
                            last_q_sent = np.asarray(q_cmd_to_send, dtype=np.float32).reshape(-1)
                            print(f"[UNIT CHECK] last_q_sent radiant to robot: {last_q_sent}")
                            
                        q_curr_used = last_q_sent.copy()
                    except Exception as e:
                        print(f"[ERROR] robot.send_action() raised:", e)
                        send_hold_action(robot, last_q_sent)

            # 官方级时钟同步
            dt_elapsed = time.perf_counter() - t_start
            sleep_time = max(0.0, dt - dt_elapsed)
            time.sleep(sleep_time)    
    
                
    except KeyboardInterrupt:
        print("Stopped by user.")
    finally:
        try:
            robot.disconnect()
        except Exception:
            pass
        try:
            cap_top.release()
            cap_wrist.release()
        except Exception:
            pass
        # cv2.destroyAllWindows()
        try:
            infer_executor.shutdown(wait=False)
        except Exception:
            pass

if __name__ == "__main__":
    main()