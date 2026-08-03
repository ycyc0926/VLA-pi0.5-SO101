#!/usr/bin/env python3
"""Backfill or follow OpenPI TRAIN_METRICS lines into a W&B run.

This tool never modifies the training log or checkpoint. Repeated metric lines
caused by resumed training are de-duplicated by step; the last occurrence wins.
Use --dry-run first to inspect what would be uploaded.
"""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path


METRIC_RE = re.compile(
    r"TRAIN_METRICS Step (?P<step>\d+):\s*"
    r"grad_norm=(?P<grad_norm>[-+0-9.eE]+),\s*"
    r"loss=(?P<loss>[-+0-9.eE]+),\s*"
    r"param_norm=(?P<param_norm>[-+0-9.eE]+)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill OpenPI text-log metrics into a W&B run."
    )
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--entity", default=None)
    parser.add_argument("--run-name", required=True)
    parser.add_argument(
        "--run-id",
        default=None,
        help="Resume this existing W&B run ID instead of creating a new run.",
    )
    parser.add_argument("--group", default="so101_drawer_idle_filter_comparison")
    parser.add_argument("--tag", action="append", default=[])
    parser.add_argument("--description", default=None)
    parser.add_argument(
        "--max-step",
        type=int,
        default=None,
        help="Only upload metrics at or before this training step.",
    )
    parser.add_argument(
        "--run-id-file",
        type=Path,
        default=None,
        help="After W&B initializes, write the run ID to this file.",
    )
    parser.add_argument(
        "--follow",
        action="store_true",
        help="Keep following appended log lines and upload new metric steps live.",
    )
    parser.add_argument(
        "--skip-backfill",
        action="store_true",
        help="Do not re-upload existing log metrics; still de-duplicate them while following.",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=2.0,
        help="Seconds between log-file polls in --follow mode.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def metric_from_match(match: re.Match[str]) -> dict[str, float | int]:
    return {
        "train/step": int(match.group("step")),
        "train/loss": float(match.group("loss")),
        "train/grad_norm": float(match.group("grad_norm")),
        "train/param_norm": float(match.group("param_norm")),
    }


def load_metrics(path: Path) -> tuple[list[dict[str, float | int]], int]:
    if not path.is_file():
        raise FileNotFoundError(f"Training log does not exist: {path}")

    matches: list[dict[str, float | int]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = METRIC_RE.search(line)
        if match is None:
            continue
        matches.append(metric_from_match(match))

    # Resumed training can repeat a logged step. Keep the last real observation
    # for each step, then upload in strictly increasing order.
    by_step = {int(item["train/step"]): item for item in matches}
    metrics = [by_step[step] for step in sorted(by_step)]
    duplicate_count = len(matches) - len(metrics)
    return metrics, duplicate_count


def follow_log(path: Path, run, seen_steps: set[int], poll_interval: float) -> None:
    if poll_interval <= 0:
        raise ValueError("--poll-interval must be greater than zero")

    position = path.stat().st_size
    pending = ""
    print(f"Following new metrics from byte offset {position}", flush=True)

    while True:
        time.sleep(poll_interval)
        size = path.stat().st_size
        if size < position:
            print("Log was truncated; restarting follow from byte offset 0", flush=True)
            position = 0
            pending = ""
        if size == position:
            continue

        with path.open("r", encoding="utf-8", errors="replace") as stream:
            stream.seek(position)
            chunk = stream.read()
            position = stream.tell()

        pending += chunk
        lines = pending.splitlines(keepends=True)
        pending = ""
        if lines and not lines[-1].endswith(("\n", "\r")):
            pending = lines.pop()

        for line in lines:
            if "Waiting for checkpoint manager to finish" in line:
                print("Training completion marker observed; stopping follow.", flush=True)
                return
            match = METRIC_RE.search(line)
            if match is None:
                continue
            item = metric_from_match(match)
            step = int(item["train/step"])
            if step in seen_steps:
                continue
            run.log(item)
            seen_steps.add(step)
            print(
                f"Uploaded live step {step}: loss={item['train/loss']}, "
                f"grad_norm={item['train/grad_norm']}",
                flush=True,
            )


def main() -> None:
    args = parse_args()
    all_metrics, duplicate_count = load_metrics(args.log)
    metrics = all_metrics
    if args.max_step is not None:
        metrics = [item for item in metrics if int(item["train/step"]) <= args.max_step]
    if not metrics:
        raise RuntimeError(
            f"No TRAIN_METRICS entries found in {args.log}; "
            "historical loss cannot be reconstructed from this file."
        )

    print(f"Log: {args.log}")
    print(f"Unique metric points: {len(metrics)}")
    print(f"Duplicate lines removed: {duplicate_count}")
    print(
        "Step range: "
        f"{metrics[0]['train/step']}..{metrics[-1]['train/step']}"
    )
    print(
        "Last metrics: "
        f"loss={metrics[-1]['train/loss']}, "
        f"grad_norm={metrics[-1]['train/grad_norm']}, "
        f"param_norm={metrics[-1]['train/param_norm']}"
    )

    if args.dry_run:
        print("DRY RUN: nothing was uploaded.")
        return

    import wandb

    run = wandb.init(
        id=args.run_id,
        resume="must" if args.run_id is not None else None,
        project=args.project,
        entity=args.entity,
        name=args.run_name,
        group=args.group,
        tags=args.tag,
        notes=args.description,
        job_type="live_log_mirror" if args.follow else "historical_log_backfill",
        config={
            "source_log": str(args.log),
            "source": "OpenPI TRAIN_METRICS text log",
            "historical_backfill": True,
            "unique_metric_points": len(metrics),
            "duplicates_removed": duplicate_count,
            "live_follow": args.follow,
        },
    )
    run_id = run.id
    run_url = run.url

    if args.run_id_file is not None:
        if not args.run_id_file.parent.is_dir():
            run.finish(exit_code=1)
            raise FileNotFoundError(
                f"Run ID parent directory does not exist: {args.run_id_file.parent}"
            )
        args.run_id_file.write_text(f"{run_id}\n", encoding="utf-8")
        print(f"Wrote run ID to: {args.run_id_file}", flush=True)

    try:
        wandb.define_metric("train/step")
        wandb.define_metric("train/*", step_metric="train/step")
        if not args.skip_backfill:
            for item in metrics:
                run.log(item)
        print(f"Upload active: {run_url}", flush=True)
        if args.follow:
            follow_log(
                args.log,
                run,
                {int(item["train/step"]) for item in all_metrics},
                args.poll_interval,
            )
    finally:
        run.finish()

    print(f"Upload complete: {run_url}")


if __name__ == "__main__":
    main()
