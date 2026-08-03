#!/usr/bin/env python3
"""Convert an SO101 LeRobot dataset between calibrations of the same follower arm.

The source dataset is copied before any parquet data is changed. Body joints are
converted through the physical encoder coordinate, while the gripper is converted
through its calibrated 0-100 range. The global and per-episode numeric statistics
for ``action`` and ``observation.state`` are regenerated after conversion.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


BODY_JOINTS = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
)
GRIPPER = "gripper"
MOTOR_RESOLUTION = 4095.0
QUANTILES = (0.01, 0.10, 0.50, 0.90, 0.99)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-calibration", type=Path, required=True)
    parser.add_argument("--target-calibration", type=Path, required=True)
    return parser.parse_args()


def load_calibration(path: Path) -> dict[str, dict[str, int]]:
    calibration = json.loads(path.read_text(encoding="utf-8"))
    expected = (*BODY_JOINTS, GRIPPER)
    if tuple(calibration) != expected:
        raise ValueError(
            f"Unexpected motor order in {path}: {tuple(calibration)}; expected {expected}"
        )
    for motor, values in calibration.items():
        if values["drive_mode"] != 0:
            raise ValueError(f"Unsupported drive_mode for {motor} in {path}: {values['drive_mode']}")
        if values["range_max"] <= values["range_min"]:
            raise ValueError(f"Invalid range for {motor} in {path}: {values}")
    return calibration


def body_offsets(
    source: dict[str, dict[str, int]], target: dict[str, dict[str, int]]
) -> np.ndarray:
    offsets = []
    for motor in BODY_JOINTS:
        source_mid = (source[motor]["range_min"] + source[motor]["range_max"]) / 2
        target_mid = (target[motor]["range_min"] + target[motor]["range_max"]) / 2
        encoder_offset = (
            source_mid
            + source[motor]["homing_offset"]
            - target[motor]["homing_offset"]
            - target_mid
        )
        offsets.append(encoder_offset * 360.0 / MOTOR_RESOLUTION)
    return np.asarray(offsets, dtype=np.float64)


def convert_values(
    values: np.ndarray,
    source: dict[str, dict[str, int]],
    target: dict[str, dict[str, int]],
    offsets: np.ndarray,
) -> np.ndarray:
    converted = np.asarray(values, dtype=np.float64).copy()
    if converted.ndim != 2 or converted.shape[1] != 6:
        raise ValueError(f"Expected an [N, 6] array, got {converted.shape}")

    converted[:, :5] += offsets
    # Keep full-turn joints in the conventional degree interval.
    converted[:, 4] = (converted[:, 4] + 180.0) % 360.0 - 180.0

    source_gripper = source[GRIPPER]
    target_gripper = target[GRIPPER]
    source_raw = (
        source_gripper["range_min"]
        + converted[:, 5]
        / 100.0
        * (source_gripper["range_max"] - source_gripper["range_min"])
    )
    target_raw = (
        source_raw
        + source_gripper["homing_offset"]
        - target_gripper["homing_offset"]
    )
    converted[:, 5] = (
        100.0
        * (target_raw - target_gripper["range_min"])
        / (target_gripper["range_max"] - target_gripper["range_min"])
    )
    converted[:, 5] = np.clip(converted[:, 5], 0.0, 100.0)
    return converted.astype(np.float32)


def numeric_stats(values: np.ndarray) -> dict[str, np.ndarray]:
    values = np.asarray(values)
    result = {
        "min": np.min(values, axis=0),
        "max": np.max(values, axis=0),
        "mean": np.mean(values, axis=0),
        "std": np.std(values, axis=0),
        "count": np.asarray([len(values)], dtype=np.int64),
    }
    for quantile in QUANTILES:
        result[f"q{int(quantile * 100):02d}"] = np.quantile(values, quantile, axis=0)
    return result


def replace_vector_column(table: pa.Table, name: str, values: np.ndarray) -> pa.Table:
    index = table.schema.get_field_index(name)
    if index < 0:
        raise KeyError(f"Missing parquet column: {name}")
    field = table.schema.field(index)
    column = pa.array(values.tolist(), type=field.type)
    return table.set_column(index, field, column)


def update_episode_stats(
    episode_path: Path,
    episode_values: dict[int, dict[str, np.ndarray]],
) -> None:
    table = pq.read_table(episode_path)
    episode_indices = np.asarray(table.column("episode_index").to_numpy())

    for feature in ("action", "observation.state"):
        stats_by_episode = {
            episode_index: numeric_stats(values[feature])
            for episode_index, values in episode_values.items()
        }
        for stat_name in ("min", "max", "mean", "std", "count", "q01", "q10", "q50", "q90", "q99"):
            column_name = f"stats/{feature}/{stat_name}"
            field_index = table.schema.get_field_index(column_name)
            if field_index < 0:
                raise KeyError(f"Missing episode metadata column: {column_name}")
            field = table.schema.field(field_index)
            column_values = [stats_by_episode[int(index)][stat_name].tolist() for index in episode_indices]
            table = table.set_column(field_index, field, pa.array(column_values, type=field.type))

    pq.write_table(table, episode_path)


def update_global_stats(stats_path: Path, all_values: dict[str, list[np.ndarray]]) -> None:
    stats_document = json.loads(stats_path.read_text(encoding="utf-8"))
    for feature, arrays in all_values.items():
        computed = numeric_stats(np.concatenate(arrays, axis=0))
        stats_document[feature] = {key: value.tolist() for key, value in computed.items()}
    stats_path.write_text(json.dumps(stats_document, indent=4) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(source_root)
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_root}")

    source_calibration = load_calibration(args.source_calibration)
    target_calibration = load_calibration(args.target_calibration)
    offsets = body_offsets(source_calibration, target_calibration)

    print("Body offsets in degrees (source -> target):")
    for motor, offset in zip(BODY_JOINTS, offsets, strict=True):
        print(f"  {motor}: {offset:+.6f}")

    shutil.copytree(source_root, output_root, ignore=shutil.ignore_patterns(".cache"))

    episode_values: dict[int, dict[str, list[np.ndarray]]] = {}
    all_values: dict[str, list[np.ndarray]] = {"action": [], "observation.state": []}
    data_paths = sorted((output_root / "data").glob("chunk-*/*.parquet"))
    if not data_paths:
        raise FileNotFoundError(f"No data parquet files under {output_root / 'data'}")

    for data_path in data_paths:
        table = pq.read_table(data_path)
        episode_indices = np.asarray(table.column("episode_index").to_numpy(), dtype=np.int64)
        converted_columns: dict[str, np.ndarray] = {}
        for feature in ("action", "observation.state"):
            values = np.asarray(table.column(feature).to_pylist(), dtype=np.float32)
            converted = convert_values(values, source_calibration, target_calibration, offsets)
            table = replace_vector_column(table, feature, converted)
            converted_columns[feature] = converted
            all_values[feature].append(converted)

        for episode_index in np.unique(episode_indices):
            mask = episode_indices == episode_index
            episode = episode_values.setdefault(
                int(episode_index), {"action": [], "observation.state": []}
            )
            for feature in ("action", "observation.state"):
                episode[feature].append(converted_columns[feature][mask])

        pq.write_table(table, data_path)

    concatenated_episode_values = {
        episode_index: {
            feature: np.concatenate(chunks, axis=0)
            for feature, chunks in feature_values.items()
        }
        for episode_index, feature_values in episode_values.items()
    }
    episode_paths = sorted((output_root / "meta" / "episodes").glob("chunk-*/*.parquet"))
    if len(episode_paths) != 1:
        raise ValueError(
            "This converter currently expects one episode metadata parquet; "
            f"found {len(episode_paths)}"
        )
    update_episode_stats(episode_paths[0], concatenated_episode_values)
    update_global_stats(output_root / "meta" / "stats.json", all_values)
    print(f"Converted dataset written to: {output_root}")


if __name__ == "__main__":
    main()
