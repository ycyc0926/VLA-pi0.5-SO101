#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SO101 + OpenPI / PI0 本地推理端：安全基线优化版。

设计目标：
1. 保留原脚本“后台推理 Brain + 30Hz 下发 Body”的总体结构。
2. 明确统一使用 degrees：前五轴为角度，夹爪为 [0, 100] 标度。
3. 避免黑图、零状态、NaN/Inf、错误 action shape 进入实机链路。
4. 使用线程安全动作缓冲区，避免 clear() 与 get() 并发竞态。
5. 当前 checkpoint 每次返回 10 步动作块，默认完整装入，并尽早触发下一轮补充。
6. 提供 max_run_sec、队列空时 Hold、WebSocket watchdog 等安全护栏。

常规运行使用 Ctrl+C 或预览窗口中的 q / Esc 安全停止。
仅在修改关节单位、动作语义或关节映射后，建议短时验证。
"""

from __future__ import annotations

import argparse
from collections import deque
import threading
import time
import traceback
from typing import Deque, Optional

import cv2
import numpy as np

from openpi_client import image_tools
from openpi_client.websocket_client_policy import WebsocketClientPolicy
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig


# ------------------------- 固定约定：SO101 六维关节顺序 -------------------------

JOINT_KEYS = (
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
)
ACTION_DIM = len(JOINT_KEYS)
DEFAULT_PROMPT = "Grab the black cube and place it in the white cup"  # 修改：默认值改成当前已审计训练集中的精确 task 字符串，避免 prompt 不一致。


# ------------------------- 线程安全动作缓冲区 -------------------------

class ActionBuffer:
    """保存等待执行的绝对关节目标；替换动作块时保证原子性。"""

    def __init__(self) -> None:
        self._queue: Deque[np.ndarray] = deque()
        self._lock = threading.Lock()

    def size(self) -> int:
        with self._lock:
            return len(self._queue)

    def pop_left(self) -> Optional[np.ndarray]:
        with self._lock:
            if not self._queue:
                return None
            return self._queue.popleft().copy()

    def replace(self, actions: np.ndarray) -> None:
        actions = np.asarray(actions, dtype=np.float32)
        with self._lock:  # 修改：替换动作块期间加锁，避免原脚本 queue.queue.clear() 与消费者并发竞态。
            self._queue.clear()
            self._queue.extend(row.copy() for row in actions)

    def clear(self) -> None:
        with self._lock:
            self._queue.clear()


class RobotStateCache:
    """分别保存最后一次有效实测状态和最后一次成功下发目标。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._last_valid_state: Optional[np.ndarray] = None
        self._last_q_sent: Optional[np.ndarray] = None

    def set_last_valid_state(self, q: np.ndarray) -> None:
        with self._lock:
            self._last_valid_state = np.asarray(q, dtype=np.float32).reshape(ACTION_DIM).copy()

    def get_last_valid_state(self) -> Optional[np.ndarray]:
        with self._lock:
            return None if self._last_valid_state is None else self._last_valid_state.copy()

    def set_last_q_sent(self, q: np.ndarray) -> None:
        with self._lock:
            self._last_q_sent = np.asarray(q, dtype=np.float32).reshape(ACTION_DIM).copy()

    def get_last_q_sent(self) -> Optional[np.ndarray]:
        with self._lock:
            return None if self._last_q_sent is None else self._last_q_sent.copy()


# ------------------------- Live preview helpers -------------------------

class LiveVisionPreview:
    """线程安全缓存模型实际输入图，并由 Body 主线程限频显示。"""

    def __init__(self, enabled: bool, display_hz: float, display_scale: int) -> None:
        self.enabled = enabled
        self.display_hz = float(display_hz)
        self.display_scale = int(display_scale)
        self._lock = threading.Lock()
        self._rgb_env: Optional[np.ndarray] = None
        self._rgb_wrist: Optional[np.ndarray] = None
        self._next_display_time = 0.0
        self._window_name = "OpenPI model input preview: ENV(top) | WRIST(hand)"  # 新增：窗口标题明确标注左右图来源，便于现场审计。

    def update(self, rgb_env: np.ndarray, rgb_wrist: np.ndarray) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._rgb_env = np.asarray(rgb_env, dtype=np.uint8).copy()
            self._rgb_wrist = np.asarray(rgb_wrist, dtype=np.uint8).copy()  # 新增：Brain 线程只写缓存，不在后台线程调用 imshow/waitKey。

    def show_if_due(self, now: float) -> bool:
        if not self.enabled:
            return False
        if self.display_hz <= 0:
            return False
        if now < self._next_display_time:
            return False

        self._next_display_time = now + 1.0 / self.display_hz
        with self._lock:
            if self._rgb_env is None or self._rgb_wrist is None:
                return False
            rgb_env = self._rgb_env.copy()
            rgb_wrist = self._rgb_wrist.copy()

        canvas_rgb = np.hstack([rgb_env, rgb_wrist])
        scale = max(1, self.display_scale)
        if scale != 1:
            canvas_rgb = cv2.resize(canvas_rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        canvas_bgr = cv2.cvtColor(canvas_rgb, cv2.COLOR_RGB2BGR)
        cv2.imshow(self._window_name, canvas_bgr)
        key = cv2.waitKey(1) & 0xFF
        return key in (ord("q"), 27)  # 新增：按 q 或 Esc 只设置退出意图，后续仍走统一 finally 安全退出流程。

    def close(self) -> None:
        if not self.enabled:
            return
        try:
            cv2.destroyWindow(self._window_name)
        except Exception as exc:
            print(f"[WARN] Failed to close preview window cleanly: {exc}")  # 新增：预览窗口清理失败不影响机械臂安全退出。


# ------------------------- Camera helpers -------------------------

def open_cam(index: int, width: int, height: int, request_buffer_size_one: bool) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if request_buffer_size_one:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 修改：尽力请求单帧缓存；部分 OpenCV 后端可能忽略该设置。
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera at index {index}")
    return cap


def get_rgb_224_from_cap(
    cap: cv2.VideoCapture,
    save_name: str,
    flush_grabs: int,
    save_vision_debug: bool,
) -> np.ndarray:
    """读取尽可能新的帧，保持长宽比缩放到 224x224，并转换为 RGB uint8。"""
    for _ in range(max(0, flush_grabs)):
        cap.grab()  # 修改：可配置地丢弃缓存旧帧，降低视觉残影；默认值较小以免额外阻塞。

    ok, frame_bgr = cap.read()
    if not ok or frame_bgr is None:
        raise RuntimeError(f"Failed to read camera frame: {save_name}")  # 修改：不再向模型发送全黑占位图。

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    frame_rgb = image_tools.resize_with_pad(frame_rgb, 224, 224)
    frame_rgb = image_tools.convert_to_uint8(frame_rgb)

    if save_vision_debug:
        cv2.imwrite(f"{save_name}.jpg", frame_rgb[:, :, ::-1])  # 修改：仅在显式开启 debug 时写盘，降低推理前处理开销。

    return frame_rgb


# ------------------------- Robot helpers -------------------------

def is_valid_state(q: np.ndarray) -> bool:
    q = np.asarray(q, dtype=np.float32).reshape(-1)
    return q.shape == (ACTION_DIM,) and np.isfinite(q).all() and not np.allclose(q, 0.0)


def get_state6(robot: SO101Follower, retries: int, delay_sec: float) -> Optional[np.ndarray]:
    """读取 SO101 六维状态。连续失败时返回 None，而不是伪造零向量。"""
    for attempt in range(1, retries + 1):
        try:
            obs = robot.get_observation()
            if not all(key in obs for key in JOINT_KEYS):
                missing = [key for key in JOINT_KEYS if key not in obs]
                raise KeyError(f"missing state keys: {missing}")

            state = np.asarray([obs[key] for key in JOINT_KEYS], dtype=np.float32)
            if is_valid_state(state):
                return state

            print(f"[WARN] get_state6 attempt {attempt}/{retries}: invalid or all-zero state: {state}")
        except Exception as exc:
            print(f"[WARN] get_state6 attempt {attempt}/{retries} failed: {exc}")

        if attempt < retries:
            time.sleep(delay_sec)

    return None  # 修改：读取失败时返回 None，让上层跳过本轮推理；绝不把零状态继续送入模型。


def validate_action_features(robot: SO101Follower) -> None:
    actual_keys = tuple(robot.action_features.keys())
    if actual_keys != JOINT_KEYS:
        raise RuntimeError(
            "Unexpected robot.action_features order. "
            f"Expected {JOINT_KEYS}, got {actual_keys}. Refuse to move the robot."
        )  # 修改：不再盲信字典前六项，防止关节顺序变化后误下发。


def pack_action_joint_abs(q_cmd_deg: np.ndarray) -> dict[str, float]:
    """将六维绝对目标打包为 SO101 action 字典；单位固定为 degrees / gripper [0,100]。"""
    q_cmd_deg = np.asarray(q_cmd_deg, dtype=np.float32).reshape(-1)
    if q_cmd_deg.shape != (ACTION_DIM,):
        raise ValueError(f"Expected action shape {(ACTION_DIM,)}, got {q_cmd_deg.shape}")
    if not np.isfinite(q_cmd_deg).all():
        raise ValueError(f"Action contains NaN/Inf: {q_cmd_deg}")

    safe = q_cmd_deg.copy()
    safe[5] = np.clip(safe[5], 0.0, 100.0)  # 修改：夹爪使用 LeRobot 的 [0,100] 标度，统一注释与实际行为。
    return {key: float(value) for key, value in zip(JOINT_KEYS, safe, strict=True)}


def send_hold_action(robot: SO101Follower, state_cache: RobotStateCache) -> None:
    """重复发送最后一次成功下发目标；若尚无目标，则使用最后一次有效实测状态。"""
    q_hold = state_cache.get_last_q_sent()
    if q_hold is None:
        q_hold = state_cache.get_last_valid_state()
    if q_hold is None:
        print("[WARN] No cached state available; cannot send hold action.")
        return
    robot.send_action(pack_action_joint_abs(q_hold))


# ------------------------- Policy helpers -------------------------

def connect_policy_client(host: str, port: int) -> WebsocketClientPolicy:
    client = WebsocketClientPolicy(host=host, port=port)
    print("Connected to policy server. Meta:", client.get_server_metadata())
    return client


def close_policy_connection(client: Optional[WebsocketClientPolicy]) -> None:
    """尽力关闭 OpenPI WebSocket。上游客户端当前未暴露 close()，因此这里兼容性地访问内部连接。"""
    if client is None:
        return
    ws = getattr(client, "_ws", None)  # 修改：上游 WebsocketClientPolicy 没有公开 close；这里只做 best-effort 清理。
    if ws is None:
        return
    try:
        ws.close()
    except Exception as exc:
        print(f"[WARN] Failed to close policy websocket cleanly: {exc}")


def infer_with_watchdog(
    client: WebsocketClientPolicy,
    observation: dict,
    timeout_sec: float,
) -> dict:
    """
    调用同步 infer()。若超过 timeout_sec，尽力关闭底层 WebSocket，以中断阻塞 recv()。

    说明：上游 WebsocketClientPolicy.infer() 是同步阻塞调用，未公开 timeout 参数。
    这里的 watchdog 是 best-effort 安全护栏，不应被理解为网络层的绝对硬实时保证。
    """
    timed_out = threading.Event()

    def abort_connection() -> None:
        timed_out.set()
        close_policy_connection(client)

    timer = threading.Timer(timeout_sec, abort_connection)
    timer.daemon = True
    timer.start()
    try:
        response = client.infer(observation)
    except Exception as exc:
        if timed_out.is_set():
            raise TimeoutError(f"Policy inference exceeded {timeout_sec:.2f}s") from exc
        raise
    finally:
        timer.cancel()

    if timed_out.is_set():
        raise TimeoutError(f"Policy inference exceeded {timeout_sec:.2f}s")
    return response


def validate_actions_chunk(
    response: dict,
    expected_chunk_size: int,
    max_abs_arm_deg: float,
) -> np.ndarray:
    if not isinstance(response, dict) or "actions" not in response:
        raise ValueError(f"Unexpected inference response: {response!r}")

    actions = np.asarray(response["actions"], dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != ACTION_DIM:
        raise ValueError(f"Expected actions shape (T, {ACTION_DIM}), got {actions.shape}")
    if actions.shape[0] < 1:
        raise ValueError("Inference returned an empty action chunk")
    if expected_chunk_size > 0 and actions.shape[0] != expected_chunk_size:
        raise ValueError(f"Expected chunk size {expected_chunk_size}, got {actions.shape[0]}")
    if not np.isfinite(actions).all():
        raise ValueError("Actions contain NaN or Inf")
    if np.max(np.abs(actions[:, :5])) > max_abs_arm_deg:
        raise ValueError(
            f"Arm action exceeds broad safety bound ±{max_abs_arm_deg} degrees: "
            f"max_abs={np.max(np.abs(actions[:, :5])):.3f}"
        )

    actions = actions.copy()
    actions[:, 5] = np.clip(actions[:, 5], 0.0, 100.0)  # 修改：整块动作进入缓冲区前先完成夹爪安全裁剪。
    return actions


def stitch_actions_to_anchor(actions: np.ndarray, anchor: Optional[np.ndarray], transition_steps: int) -> np.ndarray:
    """将新绝对动作块前若干步平滑拼接到最后一次成功下发目标。"""
    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != ACTION_DIM:
        raise ValueError(f"Expected actions shape (T, {ACTION_DIM}), got {actions.shape}")
    if not np.isfinite(actions).all():
        raise ValueError("Actions contain NaN or Inf before stitching")
    if transition_steps <= 0 or anchor is None or actions.shape[0] == 0:
        return actions.copy()  # 新增：关闭拼接或没有锚点时保持绝对动作语义，不做 delta 累加。

    anchor = np.asarray(anchor, dtype=np.float32).reshape(-1)
    if anchor.shape != (ACTION_DIM,) or not np.isfinite(anchor).all():
        raise ValueError(f"Invalid stitch anchor: shape={anchor.shape}, value={anchor}")

    stitched = actions.copy()
    n = min(int(transition_steps), stitched.shape[0])
    for idx in range(n):
        t = float(idx + 1) / float(n)
        weight = t * t * (3.0 - 2.0 * t)
        stitched[idx] = (1.0 - weight) * anchor + weight * actions[idx]  # 新增：smoothstep 只在短窗口内把绝对动作从当前下发锚点过渡到新块动作。

    stitched[:, 5] = np.clip(stitched[:, 5], 0.0, 100.0)
    return stitched  # 新增：返回仍然是绝对关节目标，后续 Body 原有限速、blend 和夹爪裁剪继续生效。


# ------------------------- Main -------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Safe SO101 OpenPI/PI0 local inference client")
    parser.add_argument("--host", default="192.168.1.110", help="OpenPI policy server IP")
    parser.add_argument("--port", type=int, default=5000, help="OpenPI policy server port")
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT, help="Exact training task string")  # 修改：原参数是 int 且完全未使用；现在改为真正传给模型的字符串。
    parser.add_argument("--serial", default="COM4", help="SO101 serial port")  # 修改：默认串口对齐当前 blacknew_43k Windows 实机配置。
    parser.add_argument("--use_degrees", action="store_true", help="Required: use degree units for body joints")
    parser.add_argument("--hz", type=float, default=30.0, help="Control frequency")
    parser.add_argument("--cam_top", type=int, default=2, help="OpenCV index for top camera")  # 修改：默认 top camera 对齐当前 blacknew_43k 相机编号。
    parser.add_argument("--cam_wrist", type=int, default=0, help="OpenCV index for wrist camera")  # 修改：默认值对齐当前 blacknew_43k 实机相机映射。
    parser.add_argument("--cam_width", type=int, default=640, help="Requested camera width")
    parser.add_argument("--cam_height", type=int, default=480, help="Requested camera height")
    parser.add_argument("--camera_flush_grabs", type=int, default=1, help="Frames to discard before read()")
    parser.add_argument("--request_camera_buffer_one", action="store_true", help="Request CAP_PROP_BUFFERSIZE=1")
    parser.add_argument("--save_vision_debug", action="store_true", help="Overwrite model_sees_*.jpg each inference")
    parser.add_argument("--show_camera", action="store_true", help="Show the exact resized model input images")
    parser.add_argument("--display_hz", type=float, default=10.0, help="Preview display refresh rate")
    parser.add_argument("--display_scale", type=int, default=2, help="Integer scale for 224x224 preview tiles")  # 新增：预览参数只影响显示，不改变模型输入图像。
    parser.add_argument("--dq_limit_deg", type=float, default=3.5, help="Max per-step body joint change in degrees")  # 修改：使用已验证较稳的 3.5°/step；不要超过 4° 后直接上实机。
    parser.add_argument("--max_dq_gripper", type=float, default=15.0, help="Max per-step gripper change in [0,100] scale")
    parser.add_argument("--alpha", type=float, default=0.2, help="Blend factor for absolute targets")
    parser.add_argument("--infer_timeout", type=float, default=5.0, help="Best-effort inference watchdog timeout")
    parser.add_argument("--state_retries", type=int, default=3, help="Robot observation retry count")
    parser.add_argument("--state_retry_delay", type=float, default=0.05, help="Delay between state retries")
    parser.add_argument("--expected_chunk_size", type=int, default=10, help="0 disables strict T check; blacknew_43k returns 10")  # 修改：当前 checkpoint 期望 10 步动作块，保留严格 shape 检查。
    parser.add_argument("--enqueue_steps", type=int, default=10, help="Only enqueue first N usable steps from each chunk; 0 means all")  # 修改：当前 checkpoint chunk 为 10 步，默认完整装入一个短动作块。
    parser.add_argument("--queue_refill_threshold", type=int, default=9, help="Start a new inference when buffer size <= threshold")  # 修改：10 步短 chunk 下提前补队列，降低空缓冲概率。
    parser.add_argument("--drop_stale_prefix", action="store_true", default=False, help="Drop RTT-aligned stale prefix before enqueueing")  # 修改：默认不丢弃短 chunk 前缀，避免 RTT 抖动时剩余轨迹过短。
    parser.add_argument("--stale_prefix_cap", type=int, default=6, help="Maximum number of stale actions to drop")
    parser.add_argument("--stitch_steps", type=int, default=3, help="Smoothstep transition steps at the start of each new chunk")  # 新增：默认短窗口拼接，避免新旧绝对动作块切换突跳。
    parser.add_argument("--max_abs_arm_deg", type=float, default=180.0, help="Broad sanity bound for model output")
    parser.add_argument("--hold_resend_sec", type=float, default=0.5, help="Resend Hold when action buffer remains empty")
    parser.add_argument("--max_run_sec", type=float, default=0.0, help="0 means unlimited; use 10 for the first raised-arm test")
    parser.add_argument("--log_every_n_steps", type=int, default=10, help="Reduce per-step console I/O overhead")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if not args.use_degrees:
        raise SystemExit(
            "[FATAL] This client intentionally requires --use_degrees. "
            "Without it, current LeRobot SOFollower body joints are not radians; refuse to move."
        )  # 修改：禁止进入混合单位路径。
    if args.hz <= 0:
        raise SystemExit("[FATAL] --hz must be > 0")
    if not (0.0 < args.alpha <= 1.0):
        raise SystemExit("[FATAL] --alpha must be in (0, 1]")
    if args.enqueue_steps < 0 or args.queue_refill_threshold < 0:
        raise SystemExit("[FATAL] queue parameters must be >= 0")
    if args.enqueue_steps > 0 and args.queue_refill_threshold >= args.enqueue_steps:
        raise SystemExit("[FATAL] --queue_refill_threshold must be smaller than --enqueue_steps")
    if args.stitch_steps < 0:
        raise SystemExit("[FATAL] --stitch_steps must be >= 0")
    if args.show_camera and args.display_hz <= 0:
        raise SystemExit("[FATAL] --display_hz must be > 0 when --show_camera is enabled")
    if args.show_camera and args.display_scale < 1:
        raise SystemExit("[FATAL] --display_scale must be >= 1")  # 新增：显示参数只在启用预览时校验，避免影响无头运行。

    dt = 1.0 / args.hz
    stop_event = threading.Event()
    robot_lock = threading.RLock()
    action_buffer = ActionBuffer()
    state_cache = RobotStateCache()
    live_preview = LiveVisionPreview(args.show_camera, args.display_hz, args.display_scale)  # 新增：预览对象只缓存模型实际输入图，避免新增摄像头读取路径。

    client: Optional[WebsocketClientPolicy] = None
    robot: Optional[SO101Follower] = None
    cap_top: Optional[cv2.VideoCapture] = None
    cap_wrist: Optional[cv2.VideoCapture] = None
    inference_thread: Optional[threading.Thread] = None

    try:
        # 修改：先连接 server，再连接并上电机械臂；server 未启动时不让机械臂长时间处于待命状态。
        client = connect_policy_client(args.host, args.port)

        robot = SO101Follower(
            SO101FollowerConfig(
                port=args.serial,
                use_degrees=True,  # 修改：与命令行门禁一致，明确写死 degrees，避免误读。
                id="my_awesome_follower_arm",
            )
        )
        robot.connect()
        validate_action_features(robot)
        print("Robot connected. action_features =", list(robot.action_features.keys()))

        cap_top = open_cam(args.cam_top, args.cam_width, args.cam_height, args.request_camera_buffer_one)
        cap_wrist = open_cam(args.cam_wrist, args.cam_width, args.cam_height, args.request_camera_buffer_one)
        print(f"Opened cameras: top={args.cam_top}, wrist={args.cam_wrist}")

        with robot_lock:
            initial_state = get_state6(robot, args.state_retries, args.state_retry_delay)
        if initial_state is None:
            raise RuntimeError("Initial robot state is invalid after retries; refuse to start")  # 修改：初始零状态直接阻断启动。

        state_cache.set_last_valid_state(initial_state)
        state_cache.set_last_q_sent(initial_state)
        q_curr_used = initial_state.copy()
        print("[INFO] Initial robot state in degrees / gripper scale:", initial_state)

        def inference_worker() -> None:
            nonlocal client
            print("[INFO] Brain thread started.")

            while not stop_event.is_set():
                if action_buffer.size() > args.queue_refill_threshold:
                    stop_event.wait(0.005)
                    continue

                try:
                    prep_start = time.perf_counter()
                    rgb_env = get_rgb_224_from_cap(
                        cap_top,
                        "model_sees_top",
                        args.camera_flush_grabs,
                        args.save_vision_debug,
                    )
                    rgb_wrist = get_rgb_224_from_cap(
                        cap_wrist,
                        "model_sees_wrist",
                        args.camera_flush_grabs,
                        args.save_vision_debug,
                    )
                    live_preview.update(rgb_env, rgb_wrist)  # 新增：Brain 线程只更新已送入 observation 的 224x224 RGB 图缓存。

                    with robot_lock:
                        measured_state = get_state6(robot, args.state_retries, args.state_retry_delay)

                    if measured_state is None:
                        measured_state = state_cache.get_last_valid_state()
                        if measured_state is None:
                            raise RuntimeError("No valid state available for policy input")
                        print("[WARN] Using cached last-valid state for this inference request.")
                    else:
                        state_cache.set_last_valid_state(measured_state)

                    observation = {
                        "observation.images.images_env": rgb_env,
                        "observation.images.images_wrist": rgb_wrist,
                        "observation.state": measured_state,
                        "prompt": args.prompt,
                    }
                    prep_ms = (time.perf_counter() - prep_start) * 1000.0

                    net_start = time.perf_counter()
                    response = infer_with_watchdog(client, observation, args.infer_timeout)
                    round_trip_ms = (time.perf_counter() - net_start) * 1000.0
                    actions = validate_actions_chunk(response, args.expected_chunk_size, args.max_abs_arm_deg)

                    drop_steps = 0
                    if args.drop_stale_prefix:
                        drop_steps = min(args.stale_prefix_cap, int(round(round_trip_ms / 1000.0 * args.hz)))
                    if drop_steps >= actions.shape[0]:
                        raise ValueError(f"drop_steps={drop_steps} removes the entire chunk of length {actions.shape[0]}")

                    usable = actions[drop_steps:]
                    if args.enqueue_steps > 0:
                        usable = usable[: args.enqueue_steps]
                    if usable.shape[0] == 0:
                        raise ValueError("No usable actions remain after slicing")

                    anchor = state_cache.get_last_q_sent()
                    usable = stitch_actions_to_anchor(usable, anchor, args.stitch_steps)  # 新增：在替换缓冲区前对绝对动作块前段做 smoothstep 拼接，不做 delta 累加。
                    action_buffer.replace(usable)
                    server_timing = response.get("server_timing", {}) if isinstance(response, dict) else {}
                    server_infer_ms = float(server_timing.get("infer_ms", 0.0)) if isinstance(server_timing, dict) else 0.0
                    network_ms = max(0.0, round_trip_ms - server_infer_ms)
                    print(
                        "[BRAIN] "
                        f"prep={prep_ms:.1f}ms | network≈{network_ms:.1f}ms | "
                        f"server={server_infer_ms:.1f}ms | RTT={round_trip_ms:.1f}ms | "
                        f"raw={actions.shape} | drop={drop_steps} | queued={usable.shape[0]} | "
                        f"prompt={args.prompt!r}"
                    )

                except TimeoutError as exc:
                    print(f"[WARN] {exc}; action buffer cleared and Hold requested.")
                    action_buffer.clear()
                    with robot_lock:
                        send_hold_action(robot, state_cache)
                    close_policy_connection(client)
                    if stop_event.wait(0.5):
                        return
                    client = connect_policy_client(args.host, args.port)  # 修改：超时后重建 WebSocket，避免复用失效连接。

                except Exception as exc:
                    print(f"[ERROR] Brain loop failed: {exc}")
                    traceback.print_exc()
                    action_buffer.clear()
                    with robot_lock:
                        send_hold_action(robot, state_cache)
                    close_policy_connection(client)
                    if stop_event.wait(0.5):
                        return
                    client = connect_policy_client(args.host, args.port)

        inference_thread = threading.Thread(target=inference_worker, name="pi0-brain", daemon=True)
        inference_thread.start()

        print("[INFO] Body loop started at", args.hz, "Hz")
        start_time = time.perf_counter()
        next_tick = start_time
        last_hold_send = 0.0
        step_idx = 0
        underrun_count = 0
        buffer_empty_active = False  # 新增：仅在动作缓冲区从非空进入空状态时计数并告警，避免空周期重复刷屏。

        while not stop_event.is_set():
            now = time.perf_counter()
            if args.max_run_sec > 0 and now - start_time >= args.max_run_sec:
                print(f"[INFO] Reached max_run_sec={args.max_run_sec:.1f}; stopping safely.")
                break

            if live_preview.show_if_due(now):
                print("[INFO] Preview requested shutdown by q/Esc; stopping safely.")
                break  # 新增：预览按键退出也进入统一 finally，保留 Hold、watchdog close 和资源释放流程。

            action = action_buffer.pop_left()
            if action is None:
                if not buffer_empty_active:
                    underrun_count += 1
                    buffer_empty_active = True
                    print(f"[WARN] Action buffer underrun #{underrun_count}; sending Hold until new actions arrive.")  # 新增：只在进入空状态的第一个周期打印一次 underrun。
                if now - last_hold_send >= args.hold_resend_sec:
                    with robot_lock:
                        send_hold_action(robot, state_cache)
                    last_hold_send = now
            else:
                buffer_empty_active = False  # 新增：拿到动作后清除空状态标志，下一次真正 underrun 才再次计数。
                target_arm = action[:5]
                target_gripper = float(action[5])

                blended_arm = (1.0 - args.alpha) * q_curr_used[:5] + args.alpha * target_arm
                arm_delta = np.clip(blended_arm - q_curr_used[:5], -args.dq_limit_deg, args.dq_limit_deg)
                q_cmd_arm = q_curr_used[:5] + arm_delta

                blended_gripper = (1.0 - args.alpha) * q_curr_used[5] + args.alpha * target_gripper
                gripper_delta = np.clip(blended_gripper - q_curr_used[5], -args.max_dq_gripper, args.max_dq_gripper)
                q_cmd_gripper = np.clip(q_curr_used[5] + gripper_delta, 0.0, 100.0)

                q_cmd = np.concatenate([q_cmd_arm, [q_cmd_gripper]]).astype(np.float32)
                with robot_lock:
                    robot.send_action(pack_action_joint_abs(q_cmd))

                state_cache.set_last_q_sent(q_cmd)
                q_curr_used = q_cmd.copy()
                step_idx += 1

                if args.log_every_n_steps > 0 and step_idx % args.log_every_n_steps == 0:
                    print(
                        f"[BODY] step={step_idx} | remain={action_buffer.size()} | "
                        f"q_cmd={np.round(q_cmd, 3).tolist()} | "
                        f"target={np.round(action, 3).tolist()}"
                    )

            next_tick += dt
            sleep_sec = next_tick - time.perf_counter()
            if sleep_sec > 0:
                time.sleep(sleep_sec)
            else:
                # 修改：使用绝对节拍，出现偶发超时时不会永久累积相位漂移。
                missed_ticks = int((-sleep_sec) // dt) + 1
                next_tick += missed_ticks * dt
                if step_idx % max(1, args.log_every_n_steps) == 0:
                    print(f"[WARN] Body loop missed {missed_ticks} control tick(s)")

    except KeyboardInterrupt:
        print("[INFO] Stopped by user.")
    finally:
        stop_event.set()
        action_buffer.clear()

        if robot is not None:
            try:
                with robot_lock:
                    send_hold_action(robot, state_cache)
            except Exception as exc:
                print(f"[WARN] Final Hold failed: {exc}")

        close_policy_connection(client)

        if inference_thread is not None:
            inference_thread.join(timeout=1.0)

        if cap_top is not None:
            cap_top.release()
        if cap_wrist is not None:
            cap_wrist.release()

        live_preview.close()

        if robot is not None:
            try:
                robot.disconnect()
            except Exception as exc:
                print(f"[WARN] robot.disconnect() failed: {exc}")

        print(f"[INFO] Action buffer underrun count: {underrun_count if 'underrun_count' in locals() else 0}")  # 新增：退出时汇总累计 underrun 次数，便于评估 RTT/queue 参数。
        print("[INFO] Client shutdown complete.")


if __name__ == "__main__":
    main()
