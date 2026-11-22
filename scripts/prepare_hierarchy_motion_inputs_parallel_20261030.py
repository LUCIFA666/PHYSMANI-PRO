"""Four fixed, disjoint CPU source shards for the full blind public replay.

No source is selected by tracking success. Every source retains an outcome
ledger; zero-valid cases are complete observations, not removed examples.
"""
from pathlib import Path
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import json
import multiprocessing
import os
import signal
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from our_mani.hierarchy_20261030.data import SCHEMA, TASK_FAMILIES, atomic_json, file_sha256
from our_mani.hierarchy_20261030.tracking import (
    FORECAST_RELATIVE, FORECAST_SHA256, BlindPublicMotionPipeline,
    audit_canonical_public_priors, collect_motion_case, iter_public_hdf5_frames,
)


CODE_PATHS = (
    "scripts/prepare_hierarchy_motion_inputs_parallel_20261030.py",
    "our_mani/hierarchy_20261030/tracking.py",
    "our_mani/hierarchy_20261030/segmentation.py",
    "our_mani/hierarchy_20261030/data.py",
    "our_mani/multitask/context_se3.py", "our_mani/benchmark_perception.py",
    "our_mani/multitask/perception.py", "our_mani/perception.py",
    "our_mani/multitask/factory.py", "our_mani/multitask/training_geometry.py",
    "our_mani/multitask/rotation_equivalence.py", "our_mani/contracts.py",
    "our_mani/context_dynamics.py",
)


class AdministrativeStop(RuntimeError):
    pass


def source_key(row):
    return (row["split"], row["task"], row["raw_episode_id"])


def split_source_plan(rows, workers=4):
    if workers < 1:
        raise ValueError("positive CPU worker count required")
    identities = [source_key(row) for row in rows]
    if len(set(identities)) != len(identities):
        raise ValueError("duplicate source identity in fixed shard plan")
    return [rows[index::workers] for index in range(workers)]


def process_identity():
    pid = os.getpid()
    return dict(pid=pid, start_ticks=int(Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]),
                boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip())


def verify_identity(root, identity):
    for name, expected in identity["code_sha256"].items():
        if file_sha256(root / name) != expected:
            raise ValueError("replay source code changed: " + name)
    for name, digest in ((identity["data_manifest_path"], identity["data_manifest_sha256"]),
                         (identity["segmentation_checkpoint_path"], identity["segmentation_checkpoint_sha256"]),
                         (FORECAST_RELATIVE, FORECAST_SHA256)):
        if file_sha256(root / name) != digest:
            raise ValueError("replay artifact identity changed: " + name)


def _stop_checked_frames(row, stop):
    for item in iter_public_hdf5_frames(row):
        if stop.exists():
            raise AdministrativeStop("new blind replay STOP requested")
        yield item


def replay_shard(root_string, output_string, shard_index, rows, identity, threads, affinity):
    root, output = Path(root_string), Path(output_string)
    os.sched_setaffinity(0, affinity)
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    verify_identity(root, identity)
    folder = output / f"shard_{shard_index:03d}"
    folder.mkdir(exist_ok=False)
    status = dict(schema="hierarchy_blind_motion_source_shard_20261030_v1", shard=shard_index,
                  process=process_identity(), expected_cases=len(rows), cases=[], stage="starting",
                  complete=False, all_sources_attempted=False, natural_completion=False,
                  source_task_selects_frontend=False, smoke_only=False, identity=identity,
                  resources=dict(cpu_affinity=sorted(os.sched_getaffinity(0)),
                                 torch_num_threads=torch.get_num_threads(),
                                 torch_num_interop_threads=torch.get_num_interop_threads(),
                                 OMP_NUM_THREADS=os.environ.get("OMP_NUM_THREADS"),
                                 MKL_NUM_THREADS=os.environ.get("MKL_NUM_THREADS"),
                                 OPENBLAS_NUM_THREADS=os.environ.get("OPENBLAS_NUM_THREADS"),
                                 RGBD_preload="one episode per worker; sequential cases; four workers maximum"))
    try:
        from threadpoolctl import threadpool_info
        status["resources"]["loaded_threadpools"] = threadpool_info()
    except ImportError:
        status["resources"]["loaded_threadpools"] = "threadpoolctl unavailable; explicit environment and torch limits recorded"
    atomic_json(folder / "manifest.json", status)
    started = time.perf_counter()
    try:
        pipeline = BlindPublicMotionPipeline(root, root / identity["segmentation_checkpoint_path"])
        status["stage"] = "running"
        atomic_json(folder / "canonical_prior_audit.json", pipeline.canonical_prior_audit)
        for row in rows:
            if (output / "STOP").exists():
                status["stage"] = "drained"
                break
            pipeline.reset()
            status["active"] = dict(zip(("split", "task", "raw_episode_id"), source_key(row)))
            atomic_json(folder / "manifest.json", status)
            case_started = time.perf_counter()
            case = {key: row[key] for key in ("split", "task", "raw_episode_id", "parent_source_group", "source_path", "frames")}
            case.update(attempt=1, source_order=rows.index(row))
            try:
                arrays, summary, receipts = collect_motion_case(pipeline, _stop_checked_frames(row, output / "STOP"))
                if len(arrays["timestamps"]) != row["frames"]:
                    raise ValueError("full replay did not preserve the original source frame count")
                summary["source_decode_and_tracking_seconds"] = time.perf_counter() - case_started
                destination = folder / row["split"] / row["task"] / (row["raw_episode_id"] + ".npz")
                destination.parent.mkdir(parents=True, exist_ok=True)
                temporary = destination.with_suffix(".npz.partial")
                with temporary.open("xb") as stream:
                    np.savez_compressed(stream, **arrays)
                temporary.replace(destination)
                audit = destination.with_suffix(".json")
                atomic_json(audit, dict(summary=summary, frames=receipts))
                case.update(state="complete", npz_path=str(destination.relative_to(root)),
                            sha256=file_sha256(destination), audit_path=str(audit.relative_to(root)),
                            audit_sha256=file_sha256(audit), summary=summary,
                            zero_valid_forecast_case=not bool(arrays["trajectory_mask"].any()))
            except AdministrativeStop as error:
                case.update(state="administrative_stop", error=repr(error), native_failure=False)
                status["stage"] = "drained"
            except BaseException as error:
                case.update(state="infrastructure_failure", error=repr(error),
                            traceback=traceback.format_exc(), native_failure=False)
            status["cases"].append(case)
            status.update(active=None, elapsed_seconds=time.perf_counter() - started)
            atomic_json(folder / "manifest.json", status)
            print(json.dumps(dict(shard=shard_index, completed=len(status["cases"]), expected=len(rows),
                                  task=row["task"], split=row["split"], episode=row["raw_episode_id"],
                                  state=case["state"], valid_ratio=case.get("summary", {}).get("role_valid_ratio"),
                                  seconds=round(time.perf_counter() - case_started, 2))), flush=True)
            if status["stage"] == "drained":
                break
        verify_identity(root, identity)
        status["all_sources_attempted"] = len(status["cases"]) == len(rows)
        status["complete"] = status["all_sources_attempted"] and all(case["state"] == "complete" for case in status["cases"])
        status["natural_completion"] = status["all_sources_attempted"] and status["stage"] != "drained"
        if status["stage"] != "drained":
            status["stage"] = "complete" if status["complete"] else "failed"
    except BaseException as error:
        status.update(stage="failed", error=repr(error), traceback=traceback.format_exc(), complete=False)
    status.update(active=None, elapsed_seconds=time.perf_counter() - started, worker_final_result_written=True)
    atomic_json(folder / "manifest.json", status)
    return dict(shard=shard_index, complete=status["complete"], stage=status["stage"],
                manifest=str((folder / "manifest.json").relative_to(root)),
                manifest_sha256=file_sha256(folder / "manifest.json"))


def aggregate(output, planned, identity, worker_results, started):
    by_key, shards = {}, []
    for index in range(4):
        path = output / f"shard_{index:03d}/manifest.json"
        if not path.is_file():
            continue
        value = json.loads(path.read_text())
        shards.append(dict(shard=index, stage=value["stage"], complete=value["complete"],
                           cases=len(value["cases"]), expected_cases=value["expected_cases"],
                           process=value["process"], resources=value.get("resources")))
        for case in value["cases"]:
            key = source_key(case)
            if key in by_key:
                raise ValueError("two replay shards produced the same source identity")
            by_key[key] = case
    cases = [by_key[source_key(row)] for row in planned if source_key(row) in by_key]
    all_attempted = len(cases) == len(planned)
    failures = [case for case in cases if case["state"] == "infrastructure_failure"]
    natural = len(worker_results) == 4 and all(result["complete"] for result in worker_results)
    complete = all_attempted and natural and not failures
    return dict(schema="hierarchy_public_motion_inputs_20261030_v1", process=process_identity(),
                complete=complete, all_sources_attempted=all_attempted, natural_completion=natural,
                source_task_selects_frontend=False, frontend="blind_parallel_fixed_common_TRAIN_priors",
                smoke_only=False, TEST_read=False, source_debug_target_positions_read=False,
                native_success_measured=False, candidate_network_trained=False,
                expected_cases=len(planned), attempted_cases=len(cases), cases=cases,
                zero_valid_cases=sum(case.get("zero_valid_forecast_case", False) for case in cases),
                infrastructure_failed=len(failures), identity=identity,
                stage="complete" if complete else "running" if len(worker_results) < 4 else "failed",
                planned_sources=[dict(zip(("split", "task", "raw_episode_id"), source_key(row))) for row in planned],
                unattempted_sources=[dict(zip(("split", "task", "raw_episode_id"), source_key(row)))
                                     for row in planned if source_key(row) not in by_key],
                shards=shards, worker_results=worker_results,
                elapsed_seconds=time.perf_counter() - started)


def run(args):
    root = ROOT.resolve()
    manifest_path = (root / args.data_manifest).resolve()
    source_manifest = json.loads(manifest_path.read_text())
    if source_manifest.get("schema") != SCHEMA or not source_manifest.get("complete") or source_manifest.get("smoke_only") or source_manifest.get("TEST_read") or source_manifest.get("source_debug_target_positions_read"):
        raise ValueError("complete non-smoke TRAIN/offline validation data identity required")
    planned = source_manifest["episodes"]
    expected = {(split, task): (100 if split == "train" else 18 if task == "insert_onto_rotating_peg" else 19 if task == "insert_onto_rotating_peg_high_speed" else 20)
                for split in ("train", "validation") for task in TASK_FAMILIES}
    if Counter((row["split"], row["task"]) for row in planned) != expected:
        raise ValueError("all exact 400 TRAIN and 77 offline-validation parents are required")
    shards = split_source_plan(planned, 4)
    if args.threads_per_worker < 1:
        raise ValueError("positive thread limit required")
    cpus = sorted(os.sched_getaffinity(0))
    if len(cpus) < 4:
        raise ValueError("four independently isolated CPU shards require at least four available CPUs")
    affinities = [cpus[index * len(cpus) // 4:(index + 1) * len(cpus) // 4] for index in range(4)]
    segment = (root / args.segment_checkpoint).resolve()
    if not segment.is_file() or not segment.is_relative_to(root):
        raise ValueError("frozen project segmentation checkpoint required")
    output = (root / args.out).resolve()
    if not output.is_relative_to(root / "artifacts/framework/multitask_r10/hierarchy_20261030") or output.exists():
        raise ValueError("fresh isolated full-motion output required")
    output.mkdir(parents=True)
    atomic_json(output / "canonical_prior_audit.json", audit_canonical_public_priors(root))
    identity = dict(data_manifest_path=str(manifest_path.relative_to(root)),
                    data_manifest_sha256=file_sha256(manifest_path),
                    segmentation_checkpoint_path=str(segment.relative_to(root)),
                    segmentation_checkpoint_sha256=file_sha256(segment),
                    motion_checkpoint_path=FORECAST_RELATIVE, motion_checkpoint_sha256=FORECAST_SHA256,
                    code_sha256={name: file_sha256(root / name) for name in CODE_PATHS})
    verify_identity(root, identity)
    atomic_json(output / "source_shard_plan.json", dict(identity=identity, shard_count=4,
                shards=[[dict(zip(("split", "task", "raw_episode_id"), source_key(row))) for row in shard] for shard in shards],
                automatic_retry=False, expected_cases=477, expected_frames=sum(row["frames"] for row in planned)))
    def request_stop(signum, _frame):
        with (output / "STOP").open("a") as stream:
            stream.write(f"Administrative signal {signum}\n")
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    started, results = time.perf_counter(), []
    os.environ.setdefault("OMP_NUM_THREADS", str(args.threads_per_worker))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    atomic_json(output / "manifest.json", aggregate(output, planned, identity, results, started))
    with ProcessPoolExecutor(max_workers=4, mp_context=multiprocessing.get_context("spawn")) as executor:
        pending = {executor.submit(replay_shard, str(root), str(output), index, shard, identity, args.threads_per_worker, affinities[index]): index
                   for index, shard in enumerate(shards)}
        while pending:
            ready, _ = wait(pending, timeout=10, return_when=FIRST_COMPLETED)
            for future in ready:
                index = pending.pop(future)
                try:
                    results.append(future.result())
                except BaseException as error:
                    results.append(dict(shard=index, complete=False, stage="owner_observed_worker_exception", error=repr(error)))
            status = aggregate(output, planned, identity, results, started)
            atomic_json(output / "manifest.json", status)
    verify_identity(root, identity)
    status = aggregate(output, planned, identity, results, started)
    status["actual_all_worker_returns_observed"] = True
    atomic_json(output / "manifest.json", status)
    if not status["complete"]:
        raise RuntimeError("full blind motion preparation did not complete; inspect retained source ledger")
    print(json.dumps(dict(stage="complete", cases=len(status["cases"]), zero_valid_cases=status["zero_valid_cases"],
                          elapsed_seconds=status["elapsed_seconds"])), flush=True)
    return status


def parser():
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--data-manifest", default="artifacts/framework/multitask_r10/hierarchy_20261030/data_v1/manifest.json")
    command.add_argument("--segment-checkpoint", default="artifacts/framework/multitask_r10/hierarchy_20261030/segmentation_v1/model.pt")
    command.add_argument("--out", default="artifacts/framework/multitask_r10/hierarchy_20261030/motion_inputs_blind_v1")
    command.add_argument("--threads-per-worker", type=int, default=4)
    return command


if __name__ == "__main__":
    run(parser().parse_args())
