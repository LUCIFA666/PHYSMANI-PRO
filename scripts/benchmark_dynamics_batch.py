"""Measure NGFF training throughput versus batch size on synthetic scenes."""

import argparse
import gc
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F

from scripts.smoke_dynamics import make_model, make_state, measure, world_points


def run_batch(batch, device, repeats):
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    model = make_model(device, keypoints=2048, hidden_dim=200, num_layers=4).train()
    state = make_state(batch, 3, 2048, device, contact=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    def step():
        optimizer.zero_grad(set_to_none=True)
        prediction = world_points(model(*state, pred_len=79))
        target = state[0].unsqueeze(1).expand_as(prediction)
        loss = F.mse_loss(prediction, target)
        if not torch.isfinite(loss).item():
            raise AssertionError("non-finite synthetic loss")
        loss.backward()
        optimizer.step()

    timing = measure(device, step, warmup=1, repeats=repeats)
    assert all(torch.isfinite(p).all().item() for p in model.parameters())
    assert all(p.grad is None or torch.isfinite(p.grad).all().item() for p in model.parameters())
    seconds = timing["median_ms"] / 1000
    return {
        "status": "passed", "batch": batch, **timing,
        "trajectories_per_second": batch / seconds,
        "compute_only_days_2700_scenes_1001_epochs": math.ceil(2700 / batch) * 1001 * seconds / 86400,
    }


def write_reports(report, output):
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = [
        "NGFF synthetic batch-size benchmark",
        "", "GPU: " + report["environment"]["gpu"],
        "PyTorch: " + report["environment"]["torch"],
        "Fixed workload: 3 objects; 2048 points/object; 80 frames;",
        "hidden_dim=200; 4 layers; float32; Euler dt=step_size=0.02;",
        "forward + adjoint backward + Adam; LR=1e-5.",
        "One warmup iteration; " + str(report["repeats"]) + " timed iterations per batch.",
        "", "Batch | seconds/step | trajectories/s | peak allocated GiB | peak reserved GiB | compute-only days",
    ]
    for row in report["results"]:
        if row["status"] == "passed":
            lines.append(
                f"{row['batch']:5d} | {row['median_ms']/1000:12.3f} | "
                f"{row['trajectories_per_second']:14.3f} | "
                f"{row['peak_allocated_mib']/1024:18.3f} | "
                f"{row['peak_reserved_mib']/1024:17.3f} | "
                f"{row['compute_only_days_2700_scenes_1001_epochs']:17.3f}"
            )
        else:
            lines.append(f"{row['batch']:5d} | {row['status']}: {row.get('reason','')}")
    if "selection" in report:
        lines += ["", "Selection: " + json.dumps(report["selection"], ensure_ascii=False)]
    lines += [
        "", "These measurements use synthetic contact scenes and random initialization.",
        "They establish execution, throughput and memory use, not convergence.",
        "Compute-only days exclude download, preprocessing, data loading, evaluation",
        "and checkpointing. Different batch sizes also change optimizer-update counts.",
        "Learning rate and number of epochs must be validated on held-out real data.",
        "Peak allocated memory excludes the CUDA context and other GPU processes.",
        "The chosen memory cap and estimates can skip batches that might otherwise fit.",
        "", "Rerun:",
        f"{sys.executable} {ROOT/'scripts/benchmark_dynamics_batch.py'}",
    ]
    (output / "SUMMARY.txt").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batches", nargs="+", type=int, default=[9, 16, 32, 64, 72, 96, 128])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--memory-cap-gib", type=float, default=26)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/dynamics_batch")
    args = parser.parse_args()
    if any(batch < 1 for batch in args.batches) or args.repeats < 1 or args.memory_cap_gib <= 0:
        parser.error("batch sizes, repeats, and memory cap must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0")
    torch.set_num_threads(4)
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    memory_cap = min(args.memory_cap_gib * 1024**3, free_bytes * 0.90)
    torch.cuda.set_per_process_memory_fraction(memory_cap / total_bytes, device)
    report = {
        "status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
        "repeats": args.repeats,
        "environment": {"gpu": torch.cuda.get_device_name(device), "torch": torch.__version__,
                        "cuda": torch.version.cuda, "total_gib": total_bytes/1024**3,
                        "free_gib_before": free_bytes/1024**3,
                        "allocator_cap_gib": memory_cap/1024**3,
                        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32},
        "results": [],
    }
    try:
        for batch in sorted(set(args.batches)):
            gc.collect()
            torch.cuda.empty_cache()
            previous = [row for row in report["results"] if row["status"] == "passed"]
            if previous:
                estimate = previous[-1]["peak_allocated_mib"] / previous[-1]["batch"] * batch * 1.10 + 256
                if estimate * 1024**2 > memory_cap:
                    report["results"].append({"batch": batch, "status": "skipped",
                                              "reason": "conservative memory estimate exceeds allocator cap",
                                              "estimated_allocated_mib": estimate})
                    write_reports(report, args.output)
                    continue
            print("START batch", batch, flush=True)
            start = time.perf_counter()
            try:
                row = run_batch(batch, device, args.repeats)
                row["elapsed_s_including_warmup"] = time.perf_counter()-start
                report["results"].append(row)
                print("BATCH", json.dumps(row), flush=True)
            except torch.cuda.OutOfMemoryError:
                report["results"].append({"batch": batch, "status": "out_of_memory",
                                          "reason": "allocator cap reached; larger batches not attempted"})
                print("OOM at batch", batch, flush=True)
                break
            write_reports(report, args.output)
        passed = [row for row in report["results"] if row["status"] == "passed"]
        if not passed:
            raise RuntimeError("no batch size completed")
        best = max(passed, key=lambda row: row["trajectories_per_second"])
        efficient = min((row for row in passed if row["trajectories_per_second"] >=
                         best["trajectories_per_second"] * 0.95), key=lambda row: row["batch"])
        baseline = passed[0]
        report["selection"] = {
            "fastest_measured_batch": best["batch"],
            "smallest_batch_within_5_percent_of_best": efficient["batch"],
            "fastest_speedup_vs_smallest_batch": best["trajectories_per_second"] / baseline["trajectories_per_second"],
            "basis": "synthetic throughput; validate batch and LR on actual training data",
        }
        report["status"] = "passed"
    finally:
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        write_reports(report, args.output)
    print("RESULT", report["status"], report.get("selection"), flush=True)


if __name__ == "__main__":
    main()
