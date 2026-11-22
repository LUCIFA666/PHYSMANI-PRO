"""Replay exact TRAIN/validation public RGB-D through public tracking and v671."""
from pathlib import Path
import argparse
import json
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from our_mani.hierarchy_20261030.data import SCHEMA, TASK_FAMILIES, atomic_json, file_sha256
from our_mani.hierarchy_20261030.tracking import (
    FORECAST_SHA256, BlindPublicMotionPipeline, collect_motion_case, iter_public_hdf5_frames,
)


def run(args):
    manifest_path = (ROOT / args.data_manifest).resolve()
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != SCHEMA or not manifest.get("complete") or manifest.get("TEST_read") or manifest.get("source_debug_target_positions_read"):
        raise ValueError("complete audited TRAIN/validation scene cache manifest required")
    if args.limit_per_task is not None and args.limit_per_task < 1 or args.max_frames is not None and args.max_frames < 1:
        raise ValueError("development case/frame limits must be positive")
    tasks = tuple(args.tasks or TASK_FAMILIES)
    if set(tasks) - set(TASK_FAMILIES):
        raise ValueError("only the frozen four tasks are allowed")
    splits = ("train", "validation") if args.split == "both" else (args.split,)
    if set(splits) - {"train", "validation"}:
        raise ValueError("TEST sources are forbidden")
    segment = Path(args.segment_checkpoint).resolve() if args.segment_checkpoint else None
    if segment is None or not segment.is_file():
        raise FileNotFoundError("fit the public segmenter before blind replay on any scene")
    output = (ROOT / args.out).resolve()
    permitted = ROOT / "artifacts/framework/multitask_r10/hierarchy_20261030"
    if not output.is_relative_to(permitted) or output.exists():
        raise ValueError("a fresh hierarchy motion-input output directory is required")
    rows = [row for row in manifest["episodes"] if row["task"] in tasks and row["split"] in splits]
    if args.limit_per_task is not None:
        rows = [row for split in splits for task in tasks
                for row in [item for item in rows if item["task"] == task and item["split"] == split][:args.limit_per_task]]
    if not rows:
        raise ValueError("no declared source cases selected")
    # Construction validates all checkpoints and common CAD before output.
    torch.set_num_threads(4)
    pipeline = BlindPublicMotionPipeline(ROOT, segment_checkpoint=segment)
    output.mkdir(parents=True)
    atomic_json(output / "canonical_prior_audit.json", pipeline.canonical_prior_audit)
    identity = dict(data_manifest_sha256=file_sha256(manifest_path),
                    segmentation_checkpoint_sha256=file_sha256(segment) if segment is not None else None)
    status = dict(schema="hierarchy_public_motion_inputs_20261030_v1", complete=False,
                  TEST_read=False, source_debug_target_positions_read=False,
                  expected_cases=len(rows), cases=[], identity=identity,
                  frontend="blind_parallel_fixed_common_TRAIN_priors",
                  source_task_selects_frontend=False,
                  smoke_only=args.limit_per_task is not None or args.max_frames is not None)
    atomic_json(output / "manifest.json", status)
    for row in rows:
        pipeline.reset()
        case_started = time.perf_counter()
        arrays, summary, receipts = collect_motion_case(pipeline, iter_public_hdf5_frames(row, max_frames=args.max_frames))
        summary["source_decode_and_tracking_seconds"] = time.perf_counter() - case_started
        destination = output / row["split"] / row["task"] / (row["raw_episode_id"] + ".npz")
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
        audit = destination.with_suffix(".json")
        atomic_json(audit, dict(summary=summary, frames=receipts))
        case = dict(split=row["split"], task=row["task"], raw_episode_id=row["raw_episode_id"],
                    parent_source_group=row["parent_source_group"], source_path=row["source_path"],
                    npz_path=str(destination.relative_to(ROOT)), sha256=file_sha256(destination),
                    audit_path=str(audit.relative_to(ROOT)), audit_sha256=file_sha256(audit), summary=summary)
        status["cases"].append(case)
        atomic_json(output / "manifest.json", status)
        print(json.dumps(dict(task=row["task"], split=row["split"], episode=row["raw_episode_id"],
                              completed=len(status["cases"]), expected=len(rows),
                              role_valid_ratio=summary["role_valid_ratio"],
                              tracking_seconds_per_frame=summary["tracking_seconds_per_frame_mean"])), flush=True)
    if file_sha256(manifest_path) != identity["data_manifest_sha256"] or segment is not None and file_sha256(segment) != identity["segmentation_checkpoint_sha256"]:
        raise ValueError("prepared source or segmentation identity changed")
    if any(file_sha256(path) != FORECAST_SHA256 for path in pipeline.checkpoint_paths):
        raise ValueError("frozen motion checkpoint changed during replay")
    status.update(complete=True, native_success_measured=False, candidate_network_trained=False)
    atomic_json(output / "manifest.json", status)
    return status


def parser():
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--data-manifest", default="artifacts/framework/multitask_r10/hierarchy_20261030/data_v1/manifest.json")
    command.add_argument("--segment-checkpoint")
    command.add_argument("--out", required=True)
    command.add_argument("--split", choices=("train", "validation", "both"), default="both")
    command.add_argument("--tasks", nargs="+", choices=tuple(TASK_FAMILIES))
    command.add_argument("--limit-per-task", type=int)
    command.add_argument("--max-frames", type=int)
    return command


if __name__ == "__main__":
    run(parser().parse_args())
