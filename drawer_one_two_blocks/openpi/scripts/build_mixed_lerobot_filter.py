#!/usr/bin/env python3
"""Build a keep-range filter for an old filtered dataset plus new full episodes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--merged-root", type=Path, required=True)
    parser.add_argument("--old-filter", type=Path, required=True)
    parser.add_argument("--old-episode-count", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing filter: {args.output}")

    old_filter = json.loads(args.old_filter.read_text(encoding="utf-8"))
    expected_old_keys = {str(index) for index in range(args.old_episode_count)}
    if set(old_filter) != expected_old_keys:
        raise ValueError(
            "Old filter episode keys do not match the expected contiguous range: "
            f"expected {sorted(expected_old_keys)}, got {sorted(old_filter)}"
        )

    episode_paths = sorted((args.merged_root / "meta" / "episodes").glob("chunk-*/*.parquet"))
    if not episode_paths:
        raise FileNotFoundError(f"No episode metadata under {args.merged_root}")
    episodes = pd.concat((pd.read_parquet(path) for path in episode_paths), ignore_index=True)
    episodes = episodes.sort_values("episode_index")

    output_filter = dict(old_filter)
    for row in episodes.itertuples(index=False):
        episode_index = int(row.episode_index)
        if episode_index < args.old_episode_count:
            continue
        length = int(row.length)
        output_filter[str(episode_index)] = [[0, length]]

    expected_all_keys = {str(index) for index in episodes["episode_index"].astype(int)}
    if set(output_filter) != expected_all_keys:
        raise ValueError("Mixed filter does not cover every merged episode exactly once")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output_filter, indent=2) + "\n", encoding="utf-8")
    kept = sum(end - start for ranges in output_filter.values() for start, end in ranges)
    print(f"Wrote {args.output}")
    print(f"Episodes: {len(output_filter)}")
    print(f"Kept sample starts: {kept}")


if __name__ == "__main__":
    main()
