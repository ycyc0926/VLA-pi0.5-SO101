#!/usr/bin/env python3
"""只读检查一个 LeRobot v3 数据集；不会改写 parquet、视频或 meta。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq


DEFAULT_ROOT = Path(__file__).resolve().parents[1] / "datasets/AlexFeng1/blacknew"


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def print_info(root: Path) -> None:
    info = load_json(root / "meta/info.json")
    print("\n=== info.json：全局清单 ===")
    for key in ("codebase_version", "robot_type", "fps", "total_episodes", "total_frames", "total_tasks"):
        print(f"{key:>16}: {info[key]}")

    print("\nfeatures：每个训练样本能取到的字段")
    for key, feature in info["features"].items():
        names = feature.get("names")
        print(f"- {key}: dtype={feature['dtype']}, shape={feature['shape']}, names={names}")


def print_tasks(root: Path) -> None:
    # tasks.parquet 是 task_index -> 自然语言指令的字典；主数据只保存较小的整数 task_index。
    table = pq.read_table(root / "meta/tasks.parquet")
    print("\n=== tasks.parquet：任务字典 ===")
    for row in table.to_pylist():
        print(f"task_index={row['task_index']}: {row['task']!r}")


def load_episode_rows(root: Path) -> list[dict]:
    path = root / "meta/episodes/chunk-000/file-000.parquet"
    columns = [
        "episode_index",
        "tasks",
        "length",
        "dataset_from_index",
        "dataset_to_index",
        "data/chunk_index",
        "data/file_index",
        "videos/observation.images.env/from_timestamp",
        "videos/observation.images.env/to_timestamp",
        "videos/observation.images.hand/from_timestamp",
        "videos/observation.images.hand/to_timestamp",
    ]
    return pq.read_table(path, columns=columns).to_pylist()


def print_episodes(root: Path, selected_episode: int, fps: int) -> dict:
    rows = load_episode_rows(root)
    if selected_episode < 0 or selected_episode >= len(rows):
        raise ValueError(f"episode 必须在 0..{len(rows) - 1}，收到 {selected_episode}")

    lengths = [row["length"] for row in rows]
    print("\n=== episodes：样本边界与视频时间段 ===")
    print(
        f"episodes={len(rows)}, frames={sum(lengths)}, "
        f"length(min/mean/max)={min(lengths)}/{sum(lengths) / len(lengths):.1f}/{max(lengths)}"
    )

    # 低维数据与视频不要求整个文件帧数相等；真正的对齐依据是每个 episode 的时间范围。
    mismatches = []
    for row in rows:
        video_frames = round(
            (row["videos/observation.images.env/to_timestamp"]
             - row["videos/observation.images.env/from_timestamp"])
            * fps
        )
        if video_frames != row["length"]:
            mismatches.append((row["episode_index"], row["length"], video_frames))
    print(f"视频片段帧数与 parquet 行数不等的 episode: {mismatches or '无'}")

    row = rows[selected_episode]
    print(f"\n选中的 episode {selected_episode}:")
    for key, value in row.items():
        print(f"  {key}: {value}")
    return row


def print_low_dimensional_rows(root: Path, episode: dict, row_count: int) -> None:
    path = root / "data/chunk-000/file-000.parquet"
    table = pq.read_table(path)

    # dataset_from_index/to_index 是左闭右开区间 [from, to)，可直接切主 parquet。
    start = episode["dataset_from_index"]
    stop = episode["dataset_to_index"]
    sample = table.slice(start, min(row_count, stop - start))

    print("\n=== 主 parquet：选中 episode 的前几帧 ===")
    print(sample.schema)
    for row in sample.to_pylist():
        print(row)


def print_stats(root: Path) -> None:
    stats = load_json(root / "meta/stats.json")
    print("\n=== stats.json：原始 LeRobot 字段统计摘要 ===")
    for key in ("observation.state", "action"):
        item = stats[key]
        print(f"{key}:")
        for stat_name in ("min", "max", "mean", "std", "q01", "q50", "q99", "count"):
            print(f"  {stat_name}: {item[stat_name]}")


def print_video_streams(root: Path) -> None:
    try:
        import av
    except ImportError:
        print("\n未安装 PyAV，跳过 MP4 容器检查。")
        return

    print("\n=== MP4：物理视频流 ===")
    for camera in ("env", "hand"):
        path = root / f"videos/observation.images.{camera}/chunk-000/file-000.mp4"
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            duration = float(stream.duration * stream.time_base) if stream.duration else None
            print(
                f"{camera}: codec={stream.codec_context.name}, {stream.width}x{stream.height}, "
                f"fps={stream.average_rate}, frames={stream.frames}, duration={duration:.3f}s"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="LeRobot v3 数据集根目录")
    parser.add_argument("--episode", type=int, default=0, help="展开查看哪个 episode")
    parser.add_argument("--rows", type=int, default=3, help="打印该 episode 开头多少行低维数据")
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    if not (root / "meta/info.json").is_file():
        raise FileNotFoundError(f"{root} 不是完整的 LeRobot v3 数据集目录")

    info = load_json(root / "meta/info.json")
    print(f"dataset root: {root}")
    print_info(root)
    print_tasks(root)
    episode = print_episodes(root, args.episode, info["fps"])
    print_low_dimensional_rows(root, episode, args.rows)
    print_stats(root)
    print_video_streams(root)


if __name__ == "__main__":
    main()
