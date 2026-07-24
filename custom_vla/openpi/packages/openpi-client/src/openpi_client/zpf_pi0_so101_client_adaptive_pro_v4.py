#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SO101 + OpenPI / PI0 本地推理端：自适应低延迟增强版 v4。

核心目标：
1. 保留安全基线：degrees 门禁、六维关节顺序校验、零状态保护、NaN/Inf 检查、夹爪裁剪、Hold 与 WebSocket watchdog。
2. 使用独立且唯一的双目采集线程，持续覆盖最新模型输入图；Brain 不再被 cap.read() 阻塞，预览窗口也不会因网络等待而冻结。
3. 自动统计本地准备时间、RTT、服务端推理时间、图像年龄、缓冲区 underrun 与 Body missed ticks。
4. 根据当前网络延迟分布、图像真实年龄和机械臂最近下发目标，保守选择新动作块切入位置；不需要手工先测网络再改参数。
5. 对已经老于动作 horizon 的结果直接拒绝，不让机械臂执行明显过期轨迹。
6. 继续保留短窗口 smoothstep 拼接，让新旧绝对动作块切换更柔和。
7. 常规运行不设置时间上限；使用 Ctrl+C 或预览窗口 q / Esc 安全停止。

物理限制：
- 当前 checkpoint 每次仅返回 10 步动作；30Hz 下 horizon 约 333ms。
- 如果网络冻结超过 horizon，任何客户端都无法凭空生成新的可靠动作。
- 本脚本会优先 Hold 并等待新鲜结果，而不是无限盲走。
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
import math
import threading
import time
import traceback
from typing import Deque, Optional

import cv2
import numpy as np

from openpi_client import image_tools
from openpi_client.websocket_client_policy import WebsocketClientPolicy
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig


# ------------------------- 固定协议：SO101 六维顺序与默认任务 -------------------------

JOINT_KEYS = (
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
)
ACTION_DIM = len(JOINT_KEYS)
DEFAULT_PROMPT = "Grab the black cube and place it in the white cup"  # 修改：默认 Prompt 与当前 blacknew_43k 训练配置严格一致。


# ------------------------- 通用小工具 -------------------------

def now_sec() -> float:
    """统一使用单调时钟，避免系统时间校准影响延迟计算。"""
    return time.perf_counter()


def percentile_or_default(values: list[float], percentile: float, default: float) -> float:
    """对滚动窗口计算百分位；窗口为空时返回安全默认值。"""
    if not values:
        return float(default)
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def safe_cleanup_call(label: str, fn) -> None:
    """退出阶段尽力释放资源；再次按 Ctrl+C 也不让清理流程中断。"""
    try:
        fn()
    except BaseException as exc:  # 修改：KeyboardInterrupt 继承 BaseException；退出阶段也要吞掉并继续释放其余资源。
        print(f"[WARN] Cleanup step failed: {label}: {exc}")


# ------------------------- 线程安全动作缓冲区 -------------------------

class ActionBuffer:
    """保存等待执行的绝对动作；Brain 替换与 Body 消费均通过同一把锁完成。"""

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
        with self._lock:  # 修改：clear + extend 作为一个原子事务，Body 不会看见装了一半的新轨迹。
            self._queue.clear()
            self._queue.extend(row.copy() for row in actions)

    def clear(self) -> None:
        with self._lock:
            self._queue.clear()


# ------------------------- 机器人状态缓存 -------------------------

class RobotStateCache:
    """保存最后有效实测状态、最后成功下发目标与最近两次下发目标。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._last_valid_state: Optional[np.ndarray] = None
        self._last_q_sent: Optional[np.ndarray] = None
        self._prev_q_sent: Optional[np.ndarray] = None

    def set_last_valid_state(self, q: np.ndarray) -> None:
        with self._lock:
            self._last_valid_state = np.asarray(q, dtype=np.float32).reshape(ACTION_DIM).copy()

    def get_last_valid_state(self) -> Optional[np.ndarray]:
        with self._lock:
            return None if self._last_valid_state is None else self._last_valid_state.copy()

    def set_last_q_sent(self, q: np.ndarray) -> None:
        q = np.asarray(q, dtype=np.float32).reshape(ACTION_DIM).copy()
        with self._lock:
            self._prev_q_sent = None if self._last_q_sent is None else self._last_q_sent.copy()
            self._last_q_sent = q

    def get_last_q_sent(self) -> Optional[np.ndarray]:
        with self._lock:
            return None if self._last_q_sent is None else self._last_q_sent.copy()

    def get_recent_sent_pair(self) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        with self._lock:
            prev = None if self._prev_q_sent is None else self._prev_q_sent.copy()
            last = None if self._last_q_sent is None else self._last_q_sent.copy()
            return prev, last


# ------------------------- 双目最新帧缓存：唯一相机读取路径 -------------------------

@dataclass(frozen=True)
class FramePair:
    """一对已经处理为模型输入格式的双目图，以及对应采集时刻。"""

    frame_id: int
    captured_at: float
    rgb_env: np.ndarray
    rgb_wrist: np.ndarray
    capture_ms: float


class LatestFramePairBuffer:
    """相机线程持续写入最新帧；Brain 与预览只读取副本，不直接碰 VideoCapture。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._frame: Optional[FramePair] = None
        self._last_model_frame_id: Optional[int] = None
        self._capture_failures = 0

    def update(self, frame: FramePair) -> None:
        with self._lock:
            self._frame = FramePair(
                frame_id=frame.frame_id,
                captured_at=float(frame.captured_at),
                rgb_env=np.asarray(frame.rgb_env, dtype=np.uint8).copy(),
                rgb_wrist=np.asarray(frame.rgb_wrist, dtype=np.uint8).copy(),
                capture_ms=float(frame.capture_ms),
            )

    def note_capture_failure(self) -> int:
        with self._lock:
            self._capture_failures += 1
            return self._capture_failures

    def mark_model_used(self, frame_id: int) -> None:
        with self._lock:
            self._last_model_frame_id = int(frame_id)

    def get_latest(self, max_age_sec: Optional[float] = None) -> FramePair:
        with self._lock:
            if self._frame is None:
                raise RuntimeError("Camera buffer does not contain a valid frame pair yet")
            frame = FramePair(
                frame_id=self._frame.frame_id,
                captured_at=self._frame.captured_at,
                rgb_env=self._frame.rgb_env.copy(),
                rgb_wrist=self._frame.rgb_wrist.copy(),
                capture_ms=self._frame.capture_ms,
            )

        age_sec = now_sec() - frame.captured_at
        if max_age_sec is not None and age_sec > max_age_sec:
            raise RuntimeError(
                f"Latest camera frame is stale: age={age_sec * 1000.0:.1f}ms > "
                f"limit={max_age_sec * 1000.0:.1f}ms"
            )
        return frame

    def preview_snapshot(self) -> tuple[Optional[FramePair], Optional[int]]:
        with self._lock:
            if self._frame is None:
                return None, self._last_model_frame_id
            frame = FramePair(
                frame_id=self._frame.frame_id,
                captured_at=self._frame.captured_at,
                rgb_env=self._frame.rgb_env.copy(),
                rgb_wrist=self._frame.rgb_wrist.copy(),
                capture_ms=self._frame.capture_ms,
            )
            return frame, self._last_model_frame_id


class CameraPairCapture:
    """唯一负责 cap.read() 的后台线程；持续覆盖最新双目模型输入图。"""

    def __init__(
        self,
        cap_top: cv2.VideoCapture,
        cap_wrist: cv2.VideoCapture,
        frame_buffer: LatestFramePairBuffer,
        stop_event: threading.Event,
        capture_hz: float,
        flush_grabs: int,
    ) -> None:
        self.cap_top = cap_top
        self.cap_wrist = cap_wrist
        self.frame_buffer = frame_buffer
        self.stop_event = stop_event
        self.capture_hz = float(capture_hz)
        self.flush_grabs = int(flush_grabs)
        self.thread: Optional[threading.Thread] = None

    @staticmethod
    def _read_model_frame(cap: cv2.VideoCapture, flush_grabs: int, label: str) -> np.ndarray:
        for _ in range(max(0, flush_grabs)):
            cap.grab()  # 修改：仅由唯一相机线程丢弃缓存帧，避免多个线程抢相机。
        ok, frame_bgr = cap.read()
        if not ok or frame_bgr is None:
            raise RuntimeError(f"Failed to read camera frame: {label}")
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb = image_tools.resize_with_pad(frame_rgb, 224, 224)
        return image_tools.convert_to_uint8(frame_rgb)

    def start(self) -> None:
        self.thread = threading.Thread(target=self._run, name="camera-pair-capture", daemon=True)
        self.thread.start()

    def join(self, timeout_sec: float = 2.0) -> None:
        if self.thread is not None:
            self.thread.join(timeout=timeout_sec)

    def _run(self) -> None:
        frame_id = 0
        next_tick = now_sec()
        dt = 1.0 / self.capture_hz if self.capture_hz > 0 else 0.0
        print(f"[INFO] Camera capture thread started at target {self.capture_hz:.1f} Hz")

        while not self.stop_event.is_set():
            capture_start = now_sec()
            try:
                rgb_env = self._read_model_frame(self.cap_top, self.flush_grabs, "model_sees_top")
                rgb_wrist = self._read_model_frame(self.cap_wrist, self.flush_grabs, "model_sees_wrist")
                capture_end = now_sec()
                frame_id += 1
                self.frame_buffer.update(
                    FramePair(
                        frame_id=frame_id,
                        captured_at=(capture_start + capture_end) * 0.5,  # 修改：用双目读取中点近似这对图片的观测时刻。
                        rgb_env=rgb_env,
                        rgb_wrist=rgb_wrist,
                        capture_ms=(capture_end - capture_start) * 1000.0,
                    )
                )
            except Exception as exc:
                failures = self.frame_buffer.note_capture_failure()
                if failures <= 3 or failures % 30 == 0:
                    print(f"[WARN] Camera capture failure #{failures}: {exc}")
                self.stop_event.wait(0.02)

            if dt <= 0:
                continue
            next_tick += dt
            sleep_sec = next_tick - now_sec()
            if sleep_sec > 0:
                self.stop_event.wait(sleep_sec)
            else:
                missed = int((-sleep_sec) // dt) + 1
                next_tick += missed * dt


# ------------------------- 实时预览：读取统一最新帧缓存 -------------------------

class LiveVisionPreview:
    """显示最新双目模型输入图；不会额外调用 cap.read()。"""

    def __init__(self, enabled: bool, display_hz: float, display_scale: int, frame_buffer: LatestFramePairBuffer) -> None:
        self.enabled = bool(enabled)
        self.display_hz = float(display_hz)
        self.display_scale = int(display_scale)
        self.frame_buffer = frame_buffer
        self._next_display_time = 0.0
        self._window_name = "OpenPI latest model-input pipeline: ENV(top) | WRIST(hand)"

    def show_if_due(self, now: float) -> bool:
        if not self.enabled or now < self._next_display_time:
            return False
        self._next_display_time = now + 1.0 / self.display_hz

        frame, last_model_frame_id = self.frame_buffer.preview_snapshot()
        if frame is None:
            return False

        env_bgr = cv2.cvtColor(frame.rgb_env, cv2.COLOR_RGB2BGR)
        wrist_bgr = cv2.cvtColor(frame.rgb_wrist, cv2.COLOR_RGB2BGR)
        age_ms = (now_sec() - frame.captured_at) * 1000.0

        cv2.putText(env_bgr, f"ENV frame={frame.frame_id}", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(wrist_bgr, f"WRIST age={age_ms:.0f}ms", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(
            wrist_bgr,
            f"last model frame={last_model_frame_id}",
            (6, 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

        canvas_bgr = np.hstack([env_bgr, wrist_bgr])
        if self.display_scale != 1:
            canvas_bgr = cv2.resize(
                canvas_bgr,
                None,
                fx=self.display_scale,
                fy=self.display_scale,
                interpolation=cv2.INTER_NEAREST,
            )
        cv2.imshow(self._window_name, canvas_bgr)
        key = cv2.waitKey(1) & 0xFF
        return key in (ord("q"), 27)  # 修改：q 或 Esc 只提出退出请求，统一 finally 负责 Hold 与释放资源。

    def close(self) -> None:
        if not self.enabled:
            return
        try:
            cv2.destroyWindow(self._window_name)
        except Exception as exc:
            print(f"[WARN] Failed to close preview window cleanly: {exc}")


# ------------------------- 运行时统计与自适应延迟估计 -------------------------

class RuntimeStats:
    """线程安全滚动统计；用于自动选择补队列阈值与动作块切入位置。"""

    def __init__(self, window_size: int) -> None:
        self._lock = threading.Lock()
        self._window_size = int(window_size)
        self._prep_ms: Deque[float] = deque(maxlen=self._window_size)
        self._rtt_ms: Deque[float] = deque(maxlen=self._window_size)
        self._server_ms: Deque[float] = deque(maxlen=self._window_size)
        self._result_age_ms: Deque[float] = deque(maxlen=self._window_size)
        self._capture_ms: Deque[float] = deque(maxlen=self._window_size)
        self._adaptive_skip: Deque[float] = deque(maxlen=self._window_size)
        self.underrun_count = 0
        self.missed_ticks = 0
        self.rejected_stale_results = 0
        self.accepted_chunks = 0
        self._last_summary_time = now_sec()

    def record_chunk(
        self,
        prep_ms: float,
        rtt_ms: float,
        server_ms: float,
        result_age_ms: float,
        capture_ms: float,
        adaptive_skip: int,
    ) -> None:
        with self._lock:
            self._prep_ms.append(float(prep_ms))
            self._rtt_ms.append(float(rtt_ms))
            self._server_ms.append(float(server_ms))
            self._result_age_ms.append(float(result_age_ms))
            self._capture_ms.append(float(capture_ms))
            self._adaptive_skip.append(float(adaptive_skip))
            self.accepted_chunks += 1

    def note_underrun(self) -> None:
        with self._lock:
            self.underrun_count += 1

    def note_missed_ticks(self, count: int) -> None:
        with self._lock:
            self.missed_ticks += int(count)

    def note_rejected_stale(self) -> None:
        with self._lock:
            self.rejected_stale_results += 1

    def p90_cycle_ms(self, default_ms: float) -> float:
        with self._lock:
            cycle = [p + r for p, r in zip(self._prep_ms, self._rtt_ms)]
        return percentile_or_default(cycle, 90.0, default_ms)

    def recommended_refill_threshold(self, chunk_len: int, hz: float, safety_steps: int, fallback: int) -> int:
        """根据滚动 P90 周期自动决定何时补充动作；当前 10 步短 chunk 下通常会自动收敛到 9。"""
        if chunk_len <= 1:
            return 0
        dt_ms = 1000.0 / hz
        cycle_steps = int(math.ceil(self.p90_cycle_ms(default_ms=(fallback + 1) * dt_ms) / dt_ms))
        return int(np.clip(cycle_steps + safety_steps, 1, chunk_len - 1))

    def required_keep_steps(self, chunk_len: int, hz: float, min_keep_steps: int, safety_steps: int, current_cycle_ms: float) -> int:
        """保留足够动作覆盖下一轮 P90 周期；网络差时自动少丢前缀，优先防止断粮。"""
        dt_ms = 1000.0 / hz
        predicted_ms = max(float(current_cycle_ms), self.p90_cycle_ms(default_ms=current_cycle_ms))
        predicted_steps = int(math.ceil(predicted_ms / dt_ms)) + int(safety_steps)
        return int(np.clip(max(min_keep_steps, predicted_steps), 1, chunk_len))

    def maybe_print_summary(self, every_sec: float) -> None:
        now = now_sec()
        with self._lock:
            if now - self._last_summary_time < every_sec:
                return
            self._last_summary_time = now
            text = self._format_summary_unlocked(prefix="[STATS]")
        print(text)

    def final_summary(self) -> str:
        with self._lock:
            return self._format_summary_unlocked(prefix="[FINAL STATS]")

    def _format_summary_unlocked(self, prefix: str) -> str:
        rtt = list(self._rtt_ms)
        prep = list(self._prep_ms)
        server = list(self._server_ms)
        age = list(self._result_age_ms)
        capture = list(self._capture_ms)
        skip = list(self._adaptive_skip)

        def stat(values: list[float]) -> str:
            if not values:
                return "n/a"
            return f"median={np.median(values):.1f}, p90={np.percentile(values, 90):.1f}, max={np.max(values):.1f}"

        return (
            f"{prefix} accepted={self.accepted_chunks} | rejected_stale={self.rejected_stale_results} | "
            f"underrun={self.underrun_count} | missed_ticks={self.missed_ticks}\n"
            f"{prefix} capture_ms({stat(capture)}) | prep_ms({stat(prep)}) | RTT_ms({stat(rtt)})\n"
            f"{prefix} server_ms({stat(server)}) | result_age_ms({stat(age)}) | adaptive_skip({stat(skip)})"
        )


# ------------------------- Robot helpers -------------------------

def open_cam(index: int, width: int, height: int, request_buffer_size_one: bool) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if request_buffer_size_one:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 修改：尽力请求单帧缓存；具体是否生效由 OpenCV 后端决定。
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera at index {index}")
    return cap


def is_valid_state(q: np.ndarray) -> bool:
    q = np.asarray(q, dtype=np.float32).reshape(-1)
    return q.shape == (ACTION_DIM,) and np.isfinite(q).all() and not np.allclose(q, 0.0)


def get_state6(robot: SO101Follower, retries: int, delay_sec: float) -> Optional[np.ndarray]:
    """读取六维状态；连续失败时返回 None，绝不伪造零状态。"""
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
    return None


def validate_action_features(robot: SO101Follower) -> None:
    actual_keys = tuple(robot.action_features.keys())
    if actual_keys != JOINT_KEYS:
        raise RuntimeError(
            "Unexpected robot.action_features order. "
            f"Expected {JOINT_KEYS}, got {actual_keys}. Refuse to move the robot."
        )  # 修改：关节顺序不同立即拒绝运行，避免错误映射导致甩臂。


def pack_action_joint_abs(q_cmd_deg: np.ndarray) -> dict[str, float]:
    """打包绝对动作；前五轴 degrees，夹爪固定裁剪到 [0,100]。"""
    q_cmd_deg = np.asarray(q_cmd_deg, dtype=np.float32).reshape(-1)
    if q_cmd_deg.shape != (ACTION_DIM,):
        raise ValueError(f"Expected action shape {(ACTION_DIM,)}, got {q_cmd_deg.shape}")
    if not np.isfinite(q_cmd_deg).all():
        raise ValueError(f"Action contains NaN/Inf: {q_cmd_deg}")
    safe = q_cmd_deg.copy()
    safe[5] = np.clip(safe[5], 0.0, 100.0)
    return {key: float(value) for key, value in zip(JOINT_KEYS, safe, strict=True)}


def send_hold_action(robot: SO101Follower, state_cache: RobotStateCache) -> None:
    """优先重复最后成功下发目标；尚未下发时退回最后有效实测状态。"""
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
    """尽力关闭底层 WebSocket；上游当前未公开 close()。"""
    if client is None:
        return
    ws = getattr(client, "_ws", None)
    if ws is None:
        return
    try:
        ws.close()
    except Exception as exc:
        print(f"[WARN] Failed to close policy websocket cleanly: {exc}")


def infer_with_watchdog(client: WebsocketClientPolicy, observation: dict, timeout_sec: float) -> dict:
    """同步 infer() 外挂 watchdog；超时后关闭底层连接，随后由 Brain 重连。"""
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


def validate_actions_chunk(response: dict, expected_chunk_size: int, max_abs_arm_deg: float) -> np.ndarray:
    """校验服务端动作块并完成夹爪安全裁剪。"""
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
    actions[:, 5] = np.clip(actions[:, 5], 0.0, 100.0)
    return actions


# ------------------------- 绝对动作拼接与自适应切入 -------------------------

def stitch_actions_to_anchor(actions: np.ndarray, anchor: Optional[np.ndarray], transition_steps: int) -> np.ndarray:
    """
    将新绝对动作块前若干步平滑拼接到最近一次已成功下发目标。

    transition_steps=3 时：
    - 第 1 步权重约 0.259：B0' = 0.741 * anchor + 0.259 * B0
    - 第 2 步权重约 0.741：B1' = 0.259 * anchor + 0.741 * B1
    - 第 3 步权重为 1.000：B2' = B2
    """
    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != ACTION_DIM:
        raise ValueError(f"Expected actions shape (T, {ACTION_DIM}), got {actions.shape}")
    if not np.isfinite(actions).all():
        raise ValueError("Actions contain NaN or Inf before stitching")
    if transition_steps <= 0 or anchor is None or actions.shape[0] == 0:
        return actions.copy()

    anchor = np.asarray(anchor, dtype=np.float32).reshape(-1)
    if anchor.shape != (ACTION_DIM,) or not np.isfinite(anchor).all():
        raise ValueError(f"Invalid stitch anchor: shape={anchor.shape}, value={anchor}")

    stitched = actions.copy()
    n = min(int(transition_steps), stitched.shape[0])
    for idx in range(n):
        t = float(idx + 1) / float(n)
        weight = t * t * (3.0 - 2.0 * t)  # 修改：smoothstep 比线性插值更柔和，窗口边缘不会突然折一下。
        stitched[idx] = (1.0 - weight) * anchor + weight * actions[idx]
    stitched[:, 5] = np.clip(stitched[:, 5], 0.0, 100.0)
    return stitched


@dataclass(frozen=True)
class AlignmentDecision:
    """记录本轮自适应切入决策，便于日志审计。"""

    start_idx: int
    elapsed_steps: int
    required_keep_steps: int
    max_allowed_skip: int
    score: float


def choose_adaptive_start_idx(
    actions: np.ndarray,
    anchor: Optional[np.ndarray],
    result_age_ms: float,
    hz: float,
    stats: RuntimeStats,
    min_keep_steps: int,
    refill_safety_steps: int,
    max_adaptive_skip: int,
    current_cycle_ms: float,
    time_weight: float,
    state_weight: float,
) -> AlignmentDecision:
    """
    根据图像年龄、滚动网络统计和当前下发目标，自动选择从新块第几步切入。

    设计原则：
    1. 图像越旧，时间上越倾向跳过更多前缀。
    2. 网络越抖，越要保留更多动作储备，避免刚切进去又断粮。
    3. 候选动作越接近当前下发目标，切换越平滑。
    """
    actions = np.asarray(actions, dtype=np.float32)
    chunk_len = actions.shape[0]
    elapsed_steps = max(0, int(round(result_age_ms / 1000.0 * hz)))
    required_keep = stats.required_keep_steps(
        chunk_len=chunk_len,
        hz=hz,
        min_keep_steps=min_keep_steps,
        safety_steps=refill_safety_steps,
        current_cycle_ms=current_cycle_ms,
    )
    max_allowed_skip = max(0, min(int(max_adaptive_skip), chunk_len - required_keep))

    if anchor is None or max_allowed_skip <= 0:
        return AlignmentDecision(
            start_idx=0,
            elapsed_steps=elapsed_steps,
            required_keep_steps=required_keep,
            max_allowed_skip=max_allowed_skip,
            score=0.0,
        )

    anchor = np.asarray(anchor, dtype=np.float32).reshape(ACTION_DIM)
    arm_scale_deg = 10.0
    gripper_scale = 25.0
    best_idx = 0
    best_score = float("inf")

    for idx in range(max_allowed_skip + 1):
        arm_distance = float(np.mean(np.abs(actions[idx, :5] - anchor[:5])) / arm_scale_deg)
        gripper_distance = float(abs(actions[idx, 5] - anchor[5]) / gripper_scale)
        state_distance = arm_distance + 0.25 * gripper_distance
        time_distance = abs(float(idx - elapsed_steps))
        score = time_weight * time_distance + state_weight * state_distance
        if score < best_score:
            best_idx = idx
            best_score = score

    return AlignmentDecision(
        start_idx=best_idx,
        elapsed_steps=elapsed_steps,
        required_keep_steps=required_keep,
        max_allowed_skip=max_allowed_skip,
        score=best_score,
    )


class StalePolicyResult(RuntimeError):
    """推理结果已经老于动作 horizon；拒绝执行并立即请求新鲜观测。"""


# ------------------------- CLI -------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Adaptive SO101 OpenPI/PI0 local inference client")

    parser.add_argument("--host", default="192.168.1.110", help="OpenPI policy server IP")
    parser.add_argument("--port", type=int, default=5000, help="OpenPI policy server port")
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT, help="Exact training task string")
    parser.add_argument("--serial", default="COM4", help="SO101 serial port")
    parser.add_argument("--use_degrees", action="store_true", help="Required: use degree units for body joints")
    parser.add_argument("--hz", type=float, default=30.0, help="Body control frequency")

    parser.add_argument("--cam_top", type=int, default=2, help="OpenCV index for top camera")
    parser.add_argument("--cam_wrist", type=int, default=0, help="OpenCV index for wrist camera")
    parser.add_argument("--cam_width", type=int, default=640, help="Requested camera width")
    parser.add_argument("--cam_height", type=int, default=480, help="Requested camera height")
    parser.add_argument("--capture_hz", type=float, default=30.0, help="Dedicated camera-pair capture target Hz")
    parser.add_argument("--camera_flush_grabs", type=int, default=0, help="Frames to discard before read() in the sole capture thread")
    parser.add_argument("--no_camera_buffer_one", action="store_true", help="Do not request CAP_PROP_BUFFERSIZE=1")
    parser.add_argument("--max_camera_frame_age_ms", type=float, default=250.0, help="Reject inference request when latest camera pair is too old")

    parser.add_argument("--show_camera", action="store_true", help="Show latest frames from the exact model-input pipeline")
    parser.add_argument("--display_hz", type=float, default=15.0, help="Preview refresh rate")
    parser.add_argument("--display_scale", type=int, default=2, help="Integer scale for 224x224 preview tiles")

    parser.add_argument("--dq_limit_deg", type=float, default=1.0, help="Max per-step arm joint change in degrees")
    parser.add_argument("--max_dq_gripper", type=float, default=5.0, help="Max per-step gripper change in [0,100] scale")
    parser.add_argument("--alpha", type=float, default=0.2, help="Body low-pass blend factor")
    parser.add_argument("--max_abs_arm_deg", type=float, default=180.0, help="Broad arm sanity bound")

    parser.add_argument("--infer_timeout", type=float, default=5.0, help="Runtime best-effort WebSocket watchdog timeout")
    parser.add_argument("--startup_infer_timeout", type=float, default=30.0, help="Longer timeout for startup warmup inference")
    parser.add_argument("--startup_attempts", type=int, default=3, help="Warmup attempts before starting Body")
    parser.add_argument("--state_retries", type=int, default=3, help="Robot observation retry count")
    parser.add_argument("--state_retry_delay", type=float, default=0.05, help="Delay between state retries")

    parser.add_argument("--expected_chunk_size", type=int, default=10, help="Expected current checkpoint action horizon")
    parser.add_argument("--enqueue_steps", type=int, default=0, help="0 means enqueue all remaining usable actions")
    parser.add_argument("--fallback_refill_threshold", type=int, default=9, help="Used before latency statistics become available")
    parser.add_argument("--refill_safety_steps", type=int, default=1, help="Extra runway added to rolling P90 delay estimate")
    parser.add_argument("--min_keep_steps", type=int, default=4, help="Never skip so much that fewer than this many actions remain")
    parser.add_argument("--max_adaptive_skip", type=int, default=6, help="Maximum automatically skipped prefix steps")
    parser.add_argument("--reject_stale_ratio", type=float, default=1.10, help="Reject a result older than ratio * chunk horizon")
    parser.add_argument("--disable_adaptive_alignment", action="store_true", help="Keep start_idx=0 for A/B comparison")
    parser.add_argument("--alignment_time_weight", type=float, default=1.0, help="Weight of image-age alignment prior")
    parser.add_argument("--alignment_state_weight", type=float, default=1.0, help="Weight of current-command proximity")
    parser.add_argument("--stitch_steps", type=int, default=3, help="Smoothstep transition steps at each new chunk start")

    parser.add_argument("--hold_resend_sec", type=float, default=0.5, help="Resend Hold while action buffer remains empty")
    parser.add_argument("--stats_window", type=int, default=100, help="Rolling latency statistics window")
    parser.add_argument("--stats_every_sec", type=float, default=10.0, help="Periodic runtime summary interval")
    parser.add_argument("--max_run_sec", type=float, default=0.0, help="0 means unlimited")
    parser.add_argument("--log_every_n_steps", type=int, default=20, help="Body log interval")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if not args.use_degrees:
        raise SystemExit(
            "[FATAL] This client intentionally requires --use_degrees. "
            "Without it, current LeRobot SOFollower body joints are not radians; refuse to move."
        )
    if args.hz <= 0 or args.capture_hz <= 0:
        raise SystemExit("[FATAL] --hz and --capture_hz must be > 0")
    if not (0.0 < args.alpha <= 1.0):
        raise SystemExit("[FATAL] --alpha must be in (0, 1]")
    if args.expected_chunk_size < 1:
        raise SystemExit("[FATAL] --expected_chunk_size must be >= 1")
    if args.enqueue_steps < 0:
        raise SystemExit("[FATAL] --enqueue_steps must be >= 0")
    if args.min_keep_steps < 1 or args.min_keep_steps > args.expected_chunk_size:
        raise SystemExit("[FATAL] --min_keep_steps must be within [1, expected_chunk_size]")
    if args.max_adaptive_skip < 0 or args.stitch_steps < 0:
        raise SystemExit("[FATAL] skip and stitch parameters must be >= 0")
    if args.reject_stale_ratio <= 0:
        raise SystemExit("[FATAL] --reject_stale_ratio must be > 0")
    if args.show_camera and (args.display_hz <= 0 or args.display_scale < 1):
        raise SystemExit("[FATAL] preview parameters are invalid")


# ------------------------- 主流程 -------------------------

def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)

    dt = 1.0 / args.hz
    stop_event = threading.Event()
    robot_lock = threading.RLock()
    action_buffer = ActionBuffer()
    state_cache = RobotStateCache()
    frame_buffer = LatestFramePairBuffer()
    stats = RuntimeStats(window_size=args.stats_window)

    client: Optional[WebsocketClientPolicy] = None
    robot: Optional[SO101Follower] = None
    cap_top: Optional[cv2.VideoCapture] = None
    cap_wrist: Optional[cv2.VideoCapture] = None
    camera_capture: Optional[CameraPairCapture] = None
    inference_thread: Optional[threading.Thread] = None
    live_preview = LiveVisionPreview(args.show_camera, args.display_hz, args.display_scale, frame_buffer)

    def build_and_request_chunk(timeout_sec: float) -> tuple[np.ndarray, dict]:
        """读取最新缓存图和状态，完成推理、过期拒绝、自适应切入与 stitch。"""
        nonlocal client
        prep_start = now_sec()
        frame = frame_buffer.get_latest(max_age_sec=args.max_camera_frame_age_ms / 1000.0)
        frame_buffer.mark_model_used(frame.frame_id)

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
            "observation.images.images_env": frame.rgb_env,
            "observation.images.images_wrist": frame.rgb_wrist,
            "observation.state": measured_state,
            "prompt": args.prompt,
        }
        prep_ms = (now_sec() - prep_start) * 1000.0

        net_start = now_sec()
        response = infer_with_watchdog(client, observation, timeout_sec)
        arrived_at = now_sec()
        rtt_ms = (arrived_at - net_start) * 1000.0
        result_age_ms = (arrived_at - frame.captured_at) * 1000.0
        actions = validate_actions_chunk(response, args.expected_chunk_size, args.max_abs_arm_deg)

        horizon_ms = actions.shape[0] / args.hz * 1000.0
        if result_age_ms > args.reject_stale_ratio * horizon_ms:
            stats.note_rejected_stale()
            raise StalePolicyResult(
                f"Reject stale policy result: frame={frame.frame_id}, age={result_age_ms:.1f}ms, "
                f"horizon={horizon_ms:.1f}ms, ratio_limit={args.reject_stale_ratio:.2f}"
            )

        server_timing = response.get("server_timing", {}) if isinstance(response, dict) else {}
        server_ms = float(server_timing.get("infer_ms", 0.0)) if isinstance(server_timing, dict) else 0.0
        current_cycle_ms = prep_ms + rtt_ms
        anchor = state_cache.get_last_q_sent()

        if args.disable_adaptive_alignment:
            decision = AlignmentDecision(0, 0, actions.shape[0], 0, 0.0)
        else:
            decision = choose_adaptive_start_idx(
                actions=actions,
                anchor=anchor,
                result_age_ms=result_age_ms,
                hz=args.hz,
                stats=stats,
                min_keep_steps=args.min_keep_steps,
                refill_safety_steps=args.refill_safety_steps,
                max_adaptive_skip=args.max_adaptive_skip,
                current_cycle_ms=current_cycle_ms,
                time_weight=args.alignment_time_weight,
                state_weight=args.alignment_state_weight,
            )

        usable = actions[decision.start_idx:]
        if args.enqueue_steps > 0:
            usable = usable[: args.enqueue_steps]
        if usable.shape[0] == 0:
            raise RuntimeError("No usable actions remain after adaptive alignment")

        usable = stitch_actions_to_anchor(usable, anchor, args.stitch_steps)
        stats.record_chunk(
            prep_ms=prep_ms,
            rtt_ms=rtt_ms,
            server_ms=server_ms,
            result_age_ms=result_age_ms,
            capture_ms=frame.capture_ms,
            adaptive_skip=decision.start_idx,
        )

        info = {
            "frame_id": frame.frame_id,
            "capture_ms": frame.capture_ms,
            "prep_ms": prep_ms,
            "rtt_ms": rtt_ms,
            "server_ms": server_ms,
            "overhead_ms": max(0.0, rtt_ms - server_ms),
            "result_age_ms": result_age_ms,
            "raw_shape": actions.shape,
            "queued": usable.shape[0],
            "decision": decision,
        }
        return usable, info

    def print_brain_log(info: dict) -> None:
        decision: AlignmentDecision = info["decision"]
        print(
            "[BRAIN] "
            f"frame={info['frame_id']} | capture={info['capture_ms']:.1f}ms | prep={info['prep_ms']:.1f}ms | "
            f"overhead≈{info['overhead_ms']:.1f}ms | server={info['server_ms']:.1f}ms | RTT={info['rtt_ms']:.1f}ms | "
            f"age={info['result_age_ms']:.1f}ms | raw={info['raw_shape']} | "
            f"start={decision.start_idx}/{decision.max_allowed_skip} | elapsed={decision.elapsed_steps} | "
            f"keep>={decision.required_keep_steps} | queued={info['queued']}"
        )

    try:
        client = connect_policy_client(args.host, args.port)

        robot = SO101Follower(
            SO101FollowerConfig(
                port=args.serial,
                use_degrees=True,  # 修改：单位明确写死为 degrees，与命令行门禁双保险。
                id="my_awesome_follower_arm",
            )
        )
        robot.connect()
        validate_action_features(robot)
        print("Robot connected. action_features =", list(robot.action_features.keys()))

        cap_top = open_cam(args.cam_top, args.cam_width, args.cam_height, not args.no_camera_buffer_one)
        cap_wrist = open_cam(args.cam_wrist, args.cam_width, args.cam_height, not args.no_camera_buffer_one)
        print(f"Opened cameras: top={args.cam_top}, wrist={args.cam_wrist}")

        camera_capture = CameraPairCapture(
            cap_top=cap_top,
            cap_wrist=cap_wrist,
            frame_buffer=frame_buffer,
            stop_event=stop_event,
            capture_hz=args.capture_hz,
            flush_grabs=args.camera_flush_grabs,
        )
        camera_capture.start()

        # 修改：等待相机线程产出首帧，避免 Body 启动后才发现相机不可用。
        camera_deadline = now_sec() + 5.0
        while True:
            try:
                frame_buffer.get_latest(max_age_sec=args.max_camera_frame_age_ms / 1000.0)
                break
            except Exception:
                if now_sec() >= camera_deadline:
                    raise RuntimeError("Camera capture thread did not produce a valid frame pair within 5 seconds")
                time.sleep(0.02)

        with robot_lock:
            initial_state = get_state6(robot, args.state_retries, args.state_retry_delay)
        if initial_state is None:
            raise RuntimeError("Initial robot state is invalid after retries; refuse to start")
        state_cache.set_last_valid_state(initial_state)
        state_cache.set_last_q_sent(initial_state)
        q_curr_used = initial_state.copy()
        print("[INFO] Initial robot state in degrees / gripper scale:", initial_state)

        # 修改：Body 启动前先同步预热并拿到一块有效动作，避免刚启动就 underrun。
        bootstrap_ok = False
        for attempt in range(1, args.startup_attempts + 1):
            try:
                usable, info = build_and_request_chunk(timeout_sec=args.startup_infer_timeout)
                action_buffer.replace(usable)
                print(f"[INFO] Startup warmup accepted on attempt {attempt}/{args.startup_attempts}")
                print_brain_log(info)
                bootstrap_ok = True
                break
            except StalePolicyResult as exc:
                print(f"[WARN] Startup warmup stale on attempt {attempt}/{args.startup_attempts}: {exc}")
            except Exception as exc:
                print(f"[WARN] Startup warmup failed on attempt {attempt}/{args.startup_attempts}: {exc}")
                traceback.print_exc()
                close_policy_connection(client)
                if attempt < args.startup_attempts:
                    client = connect_policy_client(args.host, args.port)
        if not bootstrap_ok:
            raise RuntimeError("Failed to obtain a fresh startup action chunk; refuse to start Body")

        def inference_worker() -> None:
            nonlocal client
            print("[INFO] Brain thread started.")

            while not stop_event.is_set():
                refill_threshold = stats.recommended_refill_threshold(
                    chunk_len=args.expected_chunk_size,
                    hz=args.hz,
                    safety_steps=args.refill_safety_steps,
                    fallback=args.fallback_refill_threshold,
                )
                if action_buffer.size() > refill_threshold:
                    stop_event.wait(0.003)
                    continue

                try:
                    usable, info = build_and_request_chunk(timeout_sec=args.infer_timeout)
                    if stop_event.is_set():
                        return  # 修改：退出过程中返回的推理结果不再装入缓冲区。
                    action_buffer.replace(usable)
                    print_brain_log(info)
                    stats.maybe_print_summary(args.stats_every_sec)

                except StalePolicyResult as exc:
                    print(f"[WARN] {exc}; discard result and immediately request a fresher observation.")
                    continue  # 修改：过期结果不执行、不重连，直接请求最新图。

                except TimeoutError as exc:
                    print(f"[WARN] {exc}; clear actions, Hold and reconnect.")
                    action_buffer.clear()
                    with robot_lock:
                        send_hold_action(robot, state_cache)
                    close_policy_connection(client)
                    if stop_event.wait(0.5):
                        return
                    client = connect_policy_client(args.host, args.port)

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
        start_time = now_sec()
        next_tick = start_time
        last_hold_send = 0.0
        step_idx = 0
        buffer_empty_active = False

        while not stop_event.is_set():
            now = now_sec()
            if args.max_run_sec > 0 and now - start_time >= args.max_run_sec:
                print(f"[INFO] Reached max_run_sec={args.max_run_sec:.1f}; stopping safely.")
                break

            if live_preview.show_if_due(now):
                print("[INFO] Preview requested shutdown by q/Esc; stopping safely.")
                break

            action = action_buffer.pop_left()
            if action is None:
                if not buffer_empty_active:
                    buffer_empty_active = True
                    stats.note_underrun()
                    print(f"[WARN] Action buffer underrun #{stats.underrun_count}; sending Hold until fresh actions arrive.")
                if now - last_hold_send >= args.hold_resend_sec:
                    with robot_lock:
                        send_hold_action(robot, state_cache)
                    last_hold_send = now
            else:
                buffer_empty_active = False
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
                        f"q_cmd={np.round(q_cmd, 3).tolist()} | target={np.round(action, 3).tolist()}"
                    )

            next_tick += dt
            sleep_sec = next_tick - now_sec()
            if sleep_sec > 0:
                time.sleep(sleep_sec)
            else:
                missed_ticks = int((-sleep_sec) // dt) + 1
                next_tick += missed_ticks * dt
                stats.note_missed_ticks(missed_ticks)
                if args.log_every_n_steps > 0 and step_idx % args.log_every_n_steps == 0:
                    print(f"[WARN] Body loop missed {missed_ticks} control tick(s)")

    except KeyboardInterrupt:
        print("[INFO] Stopped by user.")
    finally:
        stop_event.set()
        action_buffer.clear()

        if robot is not None:
            safe_cleanup_call("final Hold", lambda: send_hold_action(robot, state_cache))

        close_policy_connection(client)  # 修改：先关闭 WebSocket，尽力打断仍阻塞在 infer() 的 Brain。
        if inference_thread is not None:
            safe_cleanup_call("join Brain", lambda: inference_thread.join(timeout=2.0))

        if camera_capture is not None:
            safe_cleanup_call("join camera thread", lambda: camera_capture.join(timeout_sec=2.0))

        if cap_top is not None:
            safe_cleanup_call("release top camera", cap_top.release)
        if cap_wrist is not None:
            safe_cleanup_call("release wrist camera", cap_wrist.release)

        safe_cleanup_call("close preview", live_preview.close)

        if robot is not None:
            safe_cleanup_call("robot.disconnect", robot.disconnect)

        print(stats.final_summary())
        print("[INFO] Client shutdown complete.")


if __name__ == "__main__":
    main()
