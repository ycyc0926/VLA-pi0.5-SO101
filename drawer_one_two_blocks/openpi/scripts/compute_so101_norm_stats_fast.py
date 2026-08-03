#!/usr/bin/env python3
"""Compute SO101 state/action norm stats directly from LeRobot parquet files.

This is equivalent to the standard OpenPI norm-stat path for the SO101 config
used in this repository, but it avoids decoding camera videos that are not part
of the resulting ``state`` and ``actions`` statistics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

import openpi.shared.normalize as normalize
import openpi.training.config as config_lib


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", required=True)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def load_columns(root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tables = [
        pq.read_table(path, columns=["index", "observation.state", "action"])
        for path in sorted((root / "data").glob("chunk-*/*.parquet"))
    ]
    if not tables:
        raise FileNotFoundError(f"No data parquet files under {root / 'data'}")
    indices = np.concatenate(
        [np.asarray(table.column("index").to_numpy(), dtype=np.int64) for table in tables]
    )
    states = np.concatenate(
        [np.asarray(table.column("observation.state").to_pylist(), dtype=np.float32) for table in tables]
    )
    actions = np.concatenate(
        [np.asarray(table.column("action").to_pylist(), dtype=np.float32) for table in tables]
    )
    order = np.argsort(indices)
    indices = indices[order]
    if not np.array_equal(indices, np.arange(len(indices))):
        raise ValueError("Dataset index column is not contiguous from zero")
    return indices, states[order], actions[order]


def load_episode_bounds(root: Path) -> dict[int, tuple[int, int]]:
    tables = [
        pq.read_table(
            path,
            columns=["episode_index", "dataset_from_index", "dataset_to_index"],
        )
        for path in sorted((root / "meta" / "episodes").glob("chunk-*/*.parquet"))
    ]
    if not tables:
        raise FileNotFoundError(f"No episode metadata under {root}")
    bounds: dict[int, tuple[int, int]] = {}
    for table in tables:
        rows = table.to_pylist()
        for row in rows:
            bounds[int(row["episode_index"])] = (
                int(row["dataset_from_index"]),
                int(row["dataset_to_index"]),
            )
    return bounds


def kept_indices(filter_path: Path | None, bounds: dict[int, tuple[int, int]]) -> np.ndarray:
    if filter_path is None:
        return np.arange(max(end for _, end in bounds.values()), dtype=np.int64)

    ranges = json.loads(filter_path.read_text(encoding="utf-8"))
    result: list[np.ndarray] = []
    for episode_key, episode_ranges in ranges.items():
        episode_index = int(episode_key)
        episode_start, episode_end = bounds[episode_index]
        episode_length = episode_end - episode_start
        for start, end in episode_ranges:
            if not 0 <= start <= end <= episode_length:
                raise ValueError(
                    f"Invalid range for episode {episode_index}: [{start}, {end})"
                )
            result.append(np.arange(episode_start + start, episode_start + end, dtype=np.int64))
    return np.concatenate(result)


def main() -> None:
    args = parse_args()
    train_config = config_lib.get_config(args.config_name)
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    if data_config.repo_id is None:
        raise ValueError("Config has no LeRobot repo/root")
    if data_config.action_sequence_keys != ("action",):
        raise ValueError(f"Unsupported action keys: {data_config.action_sequence_keys}")

    root = Path(data_config.repo_id)
    _, states, absolute_actions = load_columns(root)
    bounds = load_episode_bounds(root)
    filter_path = (
        Path(data_config.lerobot_filter_dict_path)
        if data_config.lerobot_filter_dict_path is not None
        else None
    )
    starts = kept_indices(filter_path, bounds)
    batch_size = train_config.batch_size
    usable_count = len(starts) // batch_size * batch_size
    starts = starts[:usable_count]

    episode_end_by_index = np.empty(len(states), dtype=np.int64)
    for episode_start, episode_end in bounds.values():
        episode_end_by_index[episode_start:episode_end] = episode_end

    stats = {"state": normalize.RunningStats(), "actions": normalize.RunningStats()}
    horizon_offsets = np.arange(train_config.model.action_horizon, dtype=np.int64)
    for batch_start in range(0, len(starts), batch_size):
        batch_indices = starts[batch_start : batch_start + batch_size]
        state_batch = states[batch_indices]
        query_indices = batch_indices[:, None] + horizon_offsets[None, :]
        query_indices = np.minimum(query_indices, episode_end_by_index[batch_indices, None] - 1)
        action_batch = absolute_actions[query_indices].copy()
        action_batch[..., :5] -= state_batch[:, None, :5]
        stats["state"].update(state_batch)
        stats["actions"].update(action_batch)

    norm_stats = {key: running.get_statistics() for key, running in stats.items()}
    output_dir = args.output_dir or (train_config.assets_dirs / str(data_config.asset_id))
    normalize.save(output_dir, norm_stats)
    print(f"Dataset: {root}")
    print(f"Filter: {filter_path or 'disabled'}")
    print(f"Kept starts: {len(kept_indices(filter_path, bounds))}")
    print(f"Used starts after drop_last: {usable_count}")
    print(f"Wrote: {output_dir / 'norm_stats.json'}")


if __name__ == "__main__":
    main()
