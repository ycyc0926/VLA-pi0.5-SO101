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

from openpi_client import image_tools
from openpi_client.websocket_client_policy import WebsocketClientPolicy
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

# ------------------------- Camera helpers (OpenCV, 摄像头) -------------------------

def open_cam(index: int, w: int = 640, h: int = 480) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera at index {index}")
    return cap


def get_rgb_224_from_cap(cap: cv2.VideoCapture, sz: int = 224) -> np.ndarray:
    try:
        ok, frame_bgr = cap.read()
    except Exception as e:
        ok = False
        frame_bgr = None
        print("[WARN] camera read raised exception:", e)

    # 如果摄像头读取失败（例如连接松动），返回一个全黑的 black 图像
    if not ok or frame_bgr is None:
        print(f"[WARN] Failed to read from camera; returning black placeholder frame ({sz}x{sz})")
        black = np.zeros((sz, sz, 3), dtype=np.uint8)
        return black

    frame_rgb = frame_bgr[:, :, ::-1]  # BGR -> RGB
    frame_rgb = image_tools.resize_with_pad(frame_rgb, sz, sz)
    frame_rgb = image_tools.convert_to_uint8(frame_rgb)  # HWC, uint8
    return frame_rgb

def get_rgb_frame(cap: cv2.VideoCapture):
    # 丢弃缓冲区内的 4-5 帧旧图，确保 read() 拿到的是最新的硬件捕捉
    for _ in range(8):
        cap.grab()
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError("Camera Disconnected") # 强制触发主循环的 hold 逻辑
    return frame[:, :, ::-1]

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

    # # ------------------ 新增：保存第一帧图片到本地 ------------------
    # try:
    #     print("[INFO] 正在抓取并保存第一帧图片...")
    #     first_frame_top = get_rgb_frame(cap_top)
    #     first_frame_wrist = get_rgb_frame(cap_wrist)
        
    #     # 注意：get_rgb_frame 返回的是 RGB，OpenCV 保存需要转回 BGR
    #     cv2.imwrite("first_frame_top.jpg", cv2.cvtColor(first_frame_top, cv2.COLOR_RGB2BGR))
    #     cv2.imwrite("first_frame_wrist.jpg", cv2.cvtColor(first_frame_wrist, cv2.COLOR_RGB2BGR))
        
    #     print("[INFO] 成功保存图片：first_frame_top.jpg 和 first_frame_wrist.jpg")
    # except Exception as e:
    #     print(f"[WARN] 无法保存第一帧图片: {e}")
    # # --------------------------------------------------------------
    
    dq_limit_rad = np.deg2rad(args.dq_limit_deg)
    infer_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1) # 使用线程池处理异步推理
    INFER_TIMEOUT = float(args.infer_timeout)

    try:
        init_state = get_state6(robot)
        print(f"[DEBUG] Robot State (Converted to Radians): {init_state}")
        last_q_sent = np.array(init_state, dtype=np.float32)
        if np.allclose(last_q_sent, 0.0):
            print("[WARN] initial robot state is all zeros; caching zeros as last_q_sent.")
    except Exception:
        last_q_sent = np.zeros(6, dtype=np.float32)
        print("[WARN] failed to read initial robot state; using zeros for last_q_sent.")

    prev_actions_summary = None
    
    
    try:
        while True:
            
            obs = {
                "observation.images.images_env":   get_rgb_frame(cap_top),
                "observation.images.images_wrist": get_rgb_frame(cap_wrist),
                "observation.state":               get_state6(robot), # rad
                "prompt":                          args.prompt,
            }
            
            # # --- 添加：实时显示给服务器发送的图像 ---
            # # 注意：obs 中的图像是 RGB 格式，OpenCV 显示需要转回 BGR
            # show_top = cv2.cvtColor(obs["observation/image"], cv2.COLOR_RGB2BGR)
            # show_wrist = cv2.cvtColor(obs["observation/wrist_image"], cv2.COLOR_RGB2BGR)
            
            # # 横向拼接两张图以便同时查看
            # combined_img = np.hstack((show_top, show_wrist))
            
            # cv2.imshow("Cameras (Left: Top, Right: Wrist) - 224x224", combined_img)
            
            # # 必须调用 waitKey，否则窗口不会刷新；1ms 延迟不影响控制频率
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
            # # --------------------------------------
            

            img0 = obs["observation.images.images_env"]
            img1 = obs["observation.images.images_wrist"]
            st = obs["observation.state"]
            print("DEBUG obs shapes:", img0.shape, img1.shape, "state:", st.shape, "state_sum:", float(np.sum(st)))
            
            q_curr_raw = st
            # if last_q_sent is not None and not np.allclose(last_q_sent, 0.0):
            #     # 如果读到的原始角度相对于上次指令跳变超过了 40 度，判定为硬件读取错误
            #     raw_diff = np.abs(np.rad2deg(q_curr_raw[:5]) - np.rad2deg(last_q_sent[:5]))
            #     if np.any(raw_diff > 40.0):
            #         print(f"[ERROR] 检测到硬件回传状态异常跳变: {np.rad2deg(q_curr_raw)}, 使用缓存值。")
            #         q_curr_raw = last_q_sent.copy()
            print(f"[UNIT CHECK] q_curr (from robot to radians): {q_curr_raw}")
            
            state_is_zero = np.allclose(q_curr_raw, 0.0)
            if state_is_zero:
                print("[WARN] robot reported state all zeros - using cached last_q_sent as fallback")
                q_curr_used = last_q_sent.copy()
            else:
                q_curr_used = q_curr_raw.copy()
            print(f"[UNIT CHECK] q_curr_used : {q_curr_used}")
            
            future = infer_executor.submit(lambda o=obs: client.infer(o))
            t0 = time.time()
            try:
                res = future.result(timeout=INFER_TIMEOUT)
                infer_time = time.time() - t0
                print(f"[INFO] infer() returned in {infer_time:.3f}s; type={type(res)}")
            except concurrent.futures.TimeoutError:
                print(f"[WARN] infer() timed out after {INFER_TIMEOUT}s. Sending hold action")
                send_hold_action(robot, last_q_sent)
                try:
                    future.cancel()
                except Exception:
                    pass
                time.sleep(dt)
                continue
            except Exception as e:
                print("[ERROR] infer() raised exception:", e)
                traceback.print_exc()
                send_hold_action(robot, last_q_sent)
                time.sleep(1.0)
                continue

            if not isinstance(res, dict) or "actions" not in res:
                print("[WARN] infer() returned unexpected response:", res)
                send_hold_action(robot, last_q_sent)
                time.sleep(dt)
                continue
            
            try:
                actions_chunk = np.asarray(res["actions"], dtype=np.float32)
            except Exception as e:
                print("[ERROR] failed to parse actions:", e, "res:", res)
                send_hold_action(robot, last_q_sent)
                time.sleep(dt)
                continue

            if actions_chunk.size == 0:
                print("[WARN] actions_chunk is empty.")
                send_hold_action(robot, last_q_sent)
                time.sleep(dt)
                continue

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
                send_hold_action(robot, last_q_sent)
                time.sleep(dt)
                continue

            if np.allclose(actions_chunk, 0.0):
                print("[WARN] actions_chunk all zeros; sending hold.")
                send_hold_action(robot, last_q_sent)
                time.sleep(dt)
                continue
            
            if prev_actions_summary is not None and np.isclose(prev_actions_summary["first_mean"], summary["first_mean"]) \
               and np.isclose(prev_actions_summary["first_min"], summary["first_min"]) \
               and np.isclose(prev_actions_summary["first_max"], summary["first_max"]):
                print("[DEBUG] actions appear similar to previous cycle (may be steady point)")
            prev_actions_summary = summary

            # 获取当前这一刻的真实起始位置，在循环中保持不变
            base_q = q_curr_raw.copy()
            print(f"[UNIT CHECK] base_q (inference base): {base_q}")
            
            for t in range(min(20, len(actions_chunk))): 
                a = actions_chunk[t] # 包含前 5 维 delta 和第 6 维 absolute
                print(f"actions_chunk 的完整形状: {actions_chunk.shape}")
                print(f"[UNIT CHECK] actions_chunk (inference result): {actions_chunk[t]}")
                # 1. 提取前 5 维增量和第 6 维绝对位置
                joint_tgt = a[:5]

                gripper_abs = a[5]
                
                # 2. 计算新的目标（弧度制）
                # 注意：这里 q_curr_used 已经是弧度了
                # 始终基于推理开始时的 base_q 进行偏移，而不是基于上一步
                # new_joint_tgt = base_q[:5] + joint_delta
                # new_joint_tgt = q_curr_used[:5] + joint_delta
                

                # 1. 只对关节进行平滑和限速
                dq_limit_deg = args.dq_limit_deg # 比如 2.0 度
                q_cmd_joints = blend_and_rate_limit_abs_q(
                    q_curr=q_curr_used[:5], 
                    q_tgt=joint_tgt,
                    alpha=args.alpha, 
                    dq_limit_rad=dq_limit_deg
                )
                
                # 2. 夹爪直接使用目标值，或者使用更大的 alpha，且不参与 dq_limit
                q_cmd = np.concatenate([q_cmd_joints, [gripper_abs]])
                
                # 【增强输出】：观察预测值与真实值的差异
                print(f"\n--- 虚拟执行步 t={t} ---")
                print(f"当前角度 (Actual): {q_curr_used[:5]}")
                print(f"模型预测 (Target): {joint_tgt}")
                print(f"指令增量 (Diff):   {joint_tgt - q_curr_used[:5]}")
                print(f"夹爪预测 (Gripper): {gripper_abs}")
    
                # -----------------------
                
                # q_cmd = blend_and_rate_limit_abs_q(
                #     q_curr=q_curr_used, q_tgt=a,
                #     alpha=args.alpha, dq_limit_rad=dq_limit_rad,
                # )
                
                

                if args.use_degrees:
                    q_cmd_to_send = q_cmd.copy()
                #     q_cmd_to_send[:5] = np.rad2deg(q_cmd[:5])
                #     q_cmd_to_send[5] = q_cmd[5]
                #     print(f"[UNIT CHECK] Sending DEGREES to robot: {q_cmd_to_send}")
                else:
                    q_cmd_to_send = q_cmd
                    print(f"[UNIT CHECK] Sending RADIANS to robot: {q_cmd_to_send}")
                print(f"[STEP] t={t} a_mean={np.mean(a):.4f} q_curr_used={q_curr_used} q_cmd={q_cmd}")

                action_dict = pack_action_joint_abs(robot, q_cmd_to_send)
                try:
                    robot.send_action(action_dict)
                    if args.use_degrees:
                        last_q_sent = q_cmd_to_send
                        # last_q_sent = np.deg2rad(q_cmd_to_send).astype(np.float32)
                        print(f"[UNIT CHECK] last_q_sent degree to robot(Trans2Rad): {last_q_sent}")
                    else:
                        last_q_sent = np.asarray(q_cmd_to_send, dtype=np.float32).reshape(-1)
                        print(f"[UNIT CHECK] last_q_sent radiant to robot: {last_q_sent}")
                except Exception as e:
                    print(f"[ERROR] robot.send_action() raised:", e)
                    send_hold_action(robot, last_q_sent)
                    break

                q_curr_used = last_q_sent.copy()
                time.sleep(dt)
                
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