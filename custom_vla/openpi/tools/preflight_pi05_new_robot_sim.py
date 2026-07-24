#!/usr/bin/env python3
"""Static preflight for the PI05 new_robot_sim LoRA smoke config."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
DATASET_ROOT = Path("/home/likunwei/lerobot/dataset/new_robot_sim_20260608")
CONFIG_NAME = "pi05_new_robot_sim_smoke_lora"
COMPUTE_NORM_COMMAND = f"python3 scripts/compute_norm_stats.py --config-name={CONFIG_NAME}"
SMOKE_TRAIN_COMMAND = f"python3 scripts/train.py {CONFIG_NAME}"
# 预检脚本只做 CPU 静态读取；清空 CUDA_VISIBLE_DEVICES 且不导入 torch/jax/openpi，避免访问 GPU。


def _reexec_with_venv_if_needed() -> None:
    if importlib.util.find_spec("pyarrow") is not None:
        return
    venv_python = Path(__file__).resolve().parents[1] / ".venv/bin/python"
    if venv_python.exists() and Path(sys.executable).resolve() != venv_python.resolve():
        os.environ["PREFLIGHT_ORIGINAL_PYTHON"] = sys.executable
        os.execv(str(venv_python), [str(venv_python), *sys.argv])
    raise RuntimeError("pyarrow is required to read parquet schema; openpi .venv/bin/python was not available.")
    # 系统 python3 可能没有 pyarrow；这里自动切到仓库虚拟环境，保证用户指定命令可直接执行。


def _video_probe_with_original_python(path: Path, sample_frames: list[int]) -> dict | None:
    original_python = os.environ.get("PREFLIGHT_ORIGINAL_PYTHON")
    if not original_python or Path(original_python).resolve() == Path(sys.executable).resolve():
        return None

    code = """
import cv2, json, numpy as np, sys
path = sys.argv[1]
sample_frames = [int(x) for x in sys.argv[2:]]
cap = cv2.VideoCapture(path)
result = {
    "path": path,
    "opened": bool(cap.isOpened()),
    "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else 0,
    "fps": float(cap.get(cv2.CAP_PROP_FPS)) if cap.isOpened() else None,
    "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) if cap.isOpened() else None,
    "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if cap.isOpened() else None,
    "samples": [],
    "backend_python": sys.executable,
}
for frame_index in sample_frames:
    if not cap.isOpened():
        break
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    sample = {"frame_index": frame_index, "ok": bool(ok)}
    if ok and frame is not None:
        sample.update({
            "shape": tuple(frame.shape),
            "dtype": str(frame.dtype),
            "finite": bool(np.isfinite(frame).all()),
            "min": int(frame.min()),
            "max": int(frame.max()),
        })
    result["samples"].append(sample)
cap.release()
print(json.dumps(result))
"""
    completed = subprocess.run(
        [original_python, "-c", code, str(path), *[str(index) for index in sample_frames]],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return None
    return json.loads(completed.stdout)
    # parquet 读取需要 .venv，视频 AV1 抽帧优先委托给原始 python3，避免虚拟环境 OpenCV 解码噪声。


def _load_parquet_table(path: Path):
    import pyarrow.parquet as pq

    return pq.read_table(path), pq.ParquetFile(path).schema
    # parquet 只读取 schema 和少量列统计，不改写任何数据集文件。


def _as_float_matrix(values: list[list[float]]) -> "np.ndarray":
    import numpy as np

    return np.asarray(values, dtype=np.float32)
    # state/action 数值检查使用 numpy finite，不依赖训练数据 loader。


def _video_probe(path: Path, sample_frames: list[int]) -> dict:
    if external_probe := _video_probe_with_original_python(path, sample_frames):
        return external_probe

    import cv2
    import numpy as np

    cap = cv2.VideoCapture(str(path))
    result = {
        "path": str(path),
        "opened": bool(cap.isOpened()),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else 0,
        "fps": float(cap.get(cv2.CAP_PROP_FPS)) if cap.isOpened() else None,
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) if cap.isOpened() else None,
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if cap.isOpened() else None,
        "samples": [],
    }
    for frame_index in sample_frames:
        if not cap.isOpened():
            break
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        sample = {"frame_index": frame_index, "ok": bool(ok)}
        if ok and frame is not None:
            sample.update(
                {
                    "shape": tuple(frame.shape),
                    "dtype": str(frame.dtype),
                    "finite": bool(np.isfinite(frame).all()),
                    "min": int(frame.min()),
                    "max": int(frame.max()),
                }
            )
        result["samples"].append(sample)
    cap.release()
    return result
    # 视频检查只用 OpenCV 打开首个相机文件并抽样帧，确认 shape/dtype/finite。


def _print_section(title: str) -> None:
    print(f"\n## {title}")
    # 控制台输出按报告字段分节，便于复制到 preflight 记录。


def main() -> int:
    _reexec_with_venv_if_needed()
    import numpy as np

    info_path = DATASET_ROOT / "meta/info.json"
    stats_path = DATASET_ROOT / "meta/stats.json"
    data_path = DATASET_ROOT / "data/chunk-000/file-000.parquet"
    tasks_path = DATASET_ROOT / "meta/tasks.parquet"
    episodes_path = DATASET_ROOT / "meta/episodes/chunk-000/file-000.parquet"
    video_root = DATASET_ROOT / "videos"

    info = json.loads(info_path.read_text())
    data_table, data_schema = _load_parquet_table(data_path)
    tasks_table, _ = _load_parquet_table(tasks_path)
    episodes_table, _ = _load_parquet_table(episodes_path)

    features = info.get("features", {})
    camera_keys = [key for key, value in features.items() if value.get("dtype") == "video"]
    has_action = "action" in features and "action" in data_table.column_names
    has_task_column = "task" in data_table.column_names
    has_task_index = "task_index" in data_table.column_names
    state_feature = features.get("observation.state", {})
    action_feature = features.get("action")
    image_feature = features.get(camera_keys[0], {}) if camera_keys else {}

    states = _as_float_matrix(data_table["observation.state"].to_pylist())
    actions = _as_float_matrix(data_table["action"].to_pylist()) if "action" in data_table.column_names else None
    episodes = episodes_table.to_pydict()
    episode_lengths = episodes.get("length", [])

    first_episode_len = int(episode_lengths[0]) if episode_lengths else 0
    sample_indices = sorted({0, max(first_episode_len // 2, 0), max(first_episode_len - 1, 0)})
    sample_indices = [idx for idx in sample_indices if idx < len(states)]

    video_files = sorted(video_root.glob("*/*/*.mp4"))
    first_video = video_files[0] if video_files else None
    video_probe = _video_probe(first_video, sample_indices) if first_video else None

    frame_count_matches = info.get("total_frames") == data_table.num_rows
    finite_state = bool(np.isfinite(states).all())
    finite_action = bool(actions is not None and np.isfinite(actions).all())
    finite_video = bool(video_probe and all(sample.get("finite", False) for sample in video_probe["samples"]))
    has_required_keys = all(key in features for key in ["observation.state", *camera_keys]) and has_action
    can_compute_norm_stats = bool(has_required_keys and finite_state and finite_action and frame_count_matches)
    can_train_smoke = can_compute_norm_stats

    _print_section("Dataset Summary")
    print(f"dataset_root: {DATASET_ROOT}")
    print(f"lerobot_like_layout: {info_path.exists() and data_path.exists() and tasks_path.exists() and episodes_path.exists()}")
    print(f"directly_trainable_lerobot_dataset: {can_train_smoke}")
    print(f"episodes: {info.get('total_episodes')} (episode_lengths={episode_lengths})")
    print(f"frames_declared: {info.get('total_frames')}")
    print(f"frames_in_first_parquet: {data_table.num_rows}")
    print(f"fps: {info.get('fps')}")
    print(f"stats_json_exists: {stats_path.exists()}")

    _print_section("Schema")
    print(data_schema)
    print(f"columns: {data_table.column_names}")
    print(f"observation.state shape: {state_feature.get('shape')}")
    print(f"action shape: {action_feature.get('shape') if action_feature else None}")
    print(f"state_dim: {states.shape[-1] if states.ndim == 2 else None}")
    print(f"action_dim: {actions.shape[-1] if actions is not None and actions.ndim == 2 else None}")
    print(f"camera feature keys: {camera_keys}")
    print(f"image shape: {image_feature.get('shape')}")
    print(f"image dtype: {image_feature.get('dtype')}")
    print(f"state dtype: {state_feature.get('dtype')}")
    print(f"action dtype: {action_feature.get('dtype') if action_feature else None}")
    print(f"has task column: {has_task_column}")
    print(f"has language/task_index field: {has_task_index}")
    print(f"tasks: {tasks_table.to_pydict()}")

    _print_section("Finite Checks")
    print(f"state finite: {finite_state}, min={float(states.min())}, max={float(states.max())}")
    print(f"action finite: {finite_action} (missing action blocks this check)")
    print(f"sample indices checked in episode 0: {sample_indices}")
    for idx in sample_indices:
        print(f"state sample {idx}: shape={states[idx].shape}, finite={bool(np.isfinite(states[idx]).all())}")
    if video_probe:
        print(f"video probe: {video_probe}")
    else:
        print("video probe: no video files found")

    _print_section("Action Semantics")
    if not has_action:
        print("action_semantics: UNKNOWN - no action feature/column exists; do not guess absolute or delta.")
    else:
        print("action_semantics: UNKNOWN - action exists but semantics are not documented in metadata.")

    _print_section("Gate")
    print(f"required_keys_ok: {has_required_keys}")
    print(f"frame_count_matches: {frame_count_matches}")
    print(f"can_compute_norm_stats: {can_compute_norm_stats}")
    print(f"can_run_100_step_smoke_train: {can_train_smoke}")
    print(f"next_norm_stats_command: {COMPUTE_NORM_COMMAND}")
    print(f"next_smoke_train_command: {SMOKE_TRAIN_COMMAND}")
    if not can_train_smoke:
        print("BLOCKED: add a real action feature/column and fix the 363 declared frames vs 362 parquet rows mismatch first.")

    return 0
    # 返回 0 表示预检脚本自身完成；训练/归一化是否允许由 Gate 字段决定。


if __name__ == "__main__":
    raise SystemExit(main())
