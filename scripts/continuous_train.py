"""Unattended NGFF training with growing official data and resumable state.

The previously inspected smoke-test examples are development data. This job
never reads the reserved test groups. Changing validation cohorts resets the
checkpoint-selection score; errors from different cohorts are not compared.
"""
import hashlib
import json
import math
import os
import random
import sys
import time
import traceback
from collections import OrderedDict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from scripts.continuous_common import (
    DATA, OUT, STATE, atomic_json, event, read_json, setup, StopFlag,
)
from scripts.smoke_dynamics import world_points
from scripts.train_quality_smoke import inputs, new_model, predict, metric, baselines


def atomic_torch(path, state, rotate=False):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        torch.save(state, stream)
        stream.flush()
        os.fsync(stream.fileno())
    if rotate and path.exists():
        os.replace(path, path.with_name("previous.pt"))
    os.replace(temporary, path)


def source_signature(entries):
    return hashlib.sha256("\n".join(sorted(
        x["source"] + ":" + x["cache_sha256"] for x in entries
    )).encode()).hexdigest()


def available_data():
    entries = [read_json(p) for p in sorted((DATA / "cache").glob("*.json"))]
    entries = [x for x in entries if x and (ROOT / x["cache"]).is_file()]
    # No test group can silently become training data through a stray cache.
    for entry in entries:
        assert entry["group"] in {f"3_{i}" for i in range(9)}
        assert entry["revision"] == config_revision()
        assert entry["split"] in ("train", "validation")
    train = [x for x in entries if x["split"] == "train"]
    validation = [x for x in entries if x["split"] == "validation"]
    assert not ({x["source"] for x in train} & {x["source"] for x in validation})
    return train, validation


def config_revision():
    return "4db8e745c1c4f77cdfd74334110ba1156ac8270e"


class TrackCache:
    """Bounded CPU cache; verify provenance before the first use of each file."""
    def __init__(self, limit=64):
        self.limit = limit
        self.arrays = OrderedDict()
        self.verified = set()

    def get(self, entry, canonical):
        key = entry["cache"]
        if key in self.arrays:
            self.arrays.move_to_end(key)
            return self.arrays[key]
        path = ROOT / key
        if key not in self.verified:
            assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["cache_sha256"], key
            self.verified.add(key)
        with np.load(path) as source:
            points = source["points"].copy()
        shape = (80, 3, 2048, 3) if canonical else (31, 3, 256, 3)
        assert points.shape == shape, (key, points.shape, shape)
        assert points.dtype == np.float32 and np.isfinite(points).all()
        tensor = torch.from_numpy(points)
        self.arrays[key] = tensor
        if len(self.arrays) > self.limit:
            self.arrays.popitem(last=False)
        return tensor

    def batch(self, entries, canonical, device):
        return torch.stack([self.get(x, canonical) for x in entries]).to(device)


class Training:
    def __init__(self, config, stop):
        self.config, self.stop = config, stop
        torch.set_num_threads(4)
        torch.manual_seed(config["seed"])
        torch.cuda.manual_seed_all(config["seed"])
        random.seed(config["seed"])
        self.device = torch.device("cuda:0")
        total_memory = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(.95, config["memory_cap_gib"] * 1024**3 / total_memory))
        self.model = new_model(config, self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config["bootstrap_lr"])
        self.cache = TrackCache()
        self.state = {
            "phase": "bootstrap", "updates": 0, "bootstrap_updates": 0,
            "canonical_updates": 0, "canonical_examples_seen": 0,
            "sampling_epoch": 0, "queue": [], "queue_signature": "",
            "best_validation_mse": None, "best_update": None,
            "validation_signature": "", "validation": None,
            "training_wall_seconds": 0., "started_time": time.time(),
        }
        self.recovery = read_json(STATE / "recovery.json", {})
        self.batch_limit = min(config["batch_size"], self.recovery.get("batch_size", config["batch_size"]))
        self.lr_factor = self.recovery.get("lr_factor", 1.)
        self.history_path = OUT / "history.jsonl"
        self.last_checkpoint = self.last_validation = self.last_refresh = 0.
        self.last_train = {}
        self.train, self.validation = [], []
        manifest = read_json(ROOT / "data/quality_smoke/manifest.json")
        assert manifest["revision"] == config["data_revision"]
        self.bootstrap_train = [x for x in manifest["scenes"] if x["split"] == "train"]
        self.bootstrap_validation = [x for x in manifest["scenes"] if x["split"] == "validation"]
        assert len(self.bootstrap_train) == 8 and len(self.bootstrap_validation) == 2
        self.restore()
        self.set_phase_parameters()

    def restore(self):
        paths = [OUT / "latest.pt", OUT / "previous.pt"]
        for path in paths:
            if not path.exists():
                continue
            try:
                checkpoint = torch.load(path, map_location=self.device, weights_only=True)
                assert checkpoint["config"] == self.config, "configuration changed; use a separate output folder"
                self.model.load_state_dict(checkpoint["state_dict"], strict=True)
                self.optimizer.load_state_dict(checkpoint["optimizer"])
                self.state = checkpoint["training_state"]
                random.setstate(checkpoint["random_state"])
                torch.set_rng_state(checkpoint["torch_rng"].cpu())
                torch.cuda.set_rng_state_all([x.cpu() for x in checkpoint["cuda_rng"]])
                event("trainer", "resumed", checkpoint=str(path.relative_to(ROOT)),
                      phase=self.state["phase"], updates=self.state["updates"])
                return
            except Exception as exc:
                event("trainer", "checkpoint_read_failed", path=path.name, error=str(exc))
        if any(path.exists() for path in paths):
            raise RuntimeError("existing checkpoints could not be restored; refusing to reset training")
        atomic_torch(OUT / "initial.pt", {
            "state_dict": self.model.state_dict(), "config": self.config,
            "initialization": self.config["initialization"],
        })
        event("trainer", "initialized", hidden_dim=self.config["hidden_dim"],
              previous_128_wide_weights_loaded=False)

    def set_phase_parameters(self):
        self.model.dt = .04 if self.state["phase"] == "bootstrap" else self.config["sampled_dt"]
        self.model.step_size = self.config["ode_step_size"]

    def checkpoint(self, reason):
        payload = {
            "state_dict": self.model.state_dict(), "optimizer": self.optimizer.state_dict(),
            "config": self.config, "training_state": self.state,
            "random_state": random.getstate(), "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all(), "saved_time": time.time(),
        }
        atomic_torch(OUT / "latest.pt", payload, rotate=True)
        self.last_checkpoint = time.monotonic()
        event("trainer", "checkpoint_saved", reason=reason, phase=self.state["phase"],
              updates=self.state["updates"])

    def refresh_data(self):
        self.canonical_train, self.canonical_validation = available_data()
        ready = (len(self.canonical_train) >= self.config["minimum_canonical_training_scenes"]
                 and len(self.canonical_validation) >= self.config["minimum_canonical_validation_scenes"])
        if self.state["phase"] == "bootstrap" and ready:
            self.checkpoint("before_canonical_switch")
            self.state.update(phase="canonical", queue=[], queue_signature="",
                              validation_signature="", best_validation_mse=None,
                              best_update=None, validation=None)
            # Same architecture and weights, reset Adam for the new resolution,
            # time horizon, learning rate and growing training distribution.
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config["lr"])
            self.set_phase_parameters()
            event("trainer", "canonical_phase_started", train_scenes=len(self.canonical_train),
                  validation_scenes=len(self.canonical_validation), points=2048, frames=80)
            self.last_validation = 0.
        if self.state["phase"] == "bootstrap":
            self.train, self.validation = self.bootstrap_train, self.bootstrap_validation
        else:
            self.train, self.validation = self.canonical_train, self.canonical_validation
        self.last_refresh = time.monotonic()

    def status(self, status="running", **extra):
        canonical = self.state["phase"] == "canonical"
        atomic_json(STATE / "trainer_status.json", {
            "status": status, "time": time.time(), "pid": os.getpid(),
            **self.state, "actual_batch_size": min(self.batch_limit, len(self.train)),
            "target_batch_size": self.config["batch_size"],
            "keypoints_per_object": 2048 if canonical else 256,
            "frames": 80 if canonical else 31, "sampled_dt": self.model.dt,
            "training_scenes": len(self.train), "validation_scenes": len(self.validation),
            "canonical_training_scenes_ready": len(getattr(self, "canonical_train", [])),
            "canonical_validation_scenes_ready": len(getattr(self, "canonical_validation", [])),
            "last_train": self.last_train, "effective_batch_limit": self.batch_limit,
            "lr_factor": self.lr_factor, "last_checkpoint_time": (OUT / "latest.pt").stat().st_mtime
            if (OUT / "latest.pt").exists() else None, **extra,
        })

    def next_batch(self):
        signature = source_signature(self.train)
        by_source = {x["source"]: x for x in self.train}
        if self.state["queue_signature"] != signature or not self.state["queue"]:
            queue = list(by_source)
            random.shuffle(queue)
            self.state["queue"] = queue
            self.state["queue_signature"] = signature
            self.state["sampling_epoch"] += 1
        size = min(self.batch_limit, len(self.state["queue"]))
        sources = self.state["queue"][:size]
        self.state["queue"] = self.state["queue"][size:]
        return [by_source[x] for x in sources]

    def learning_rate(self):
        if self.state["phase"] == "bootstrap":
            return self.config["bootstrap_lr"] * self.lr_factor
        # Normalize by the committed final training split, not the current
        # small cache; repeating 2 scenes is not 1 full-paper epoch.
        plan = read_json(DATA / "plan.json", {"entries": []})
        count = max(1, sum(x["split"] == "train" for x in plan["entries"]))
        epoch_equivalents = self.state["canonical_examples_seen"] / count
        fraction = min(1., epoch_equivalents / self.config["decay_reference_epochs"])
        rate = self.config["min_lr"] + (self.config["lr"] - self.config["min_lr"]) * .5 * (1 + math.cos(math.pi * fraction))
        return rate * self.lr_factor

    def train_step(self):
        entries = self.next_batch()
        canonical = self.state["phase"] == "canonical"
        tracks = self.cache.batch(entries, canonical, self.device)
        lr = self.learning_rate()
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        prediction = world_points(self.model(*inputs(tracks), pred_len=tracks.shape[1] - 1))
        loss = F.mse_loss(prediction, tracks)
        if not torch.isfinite(loss).item():
            raise FloatingPointError("non-finite training loss")
        loss.backward()
        try:
            norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config["gradient_clip_norm"], error_if_nonfinite=True)
        except RuntimeError as exc:
            raise FloatingPointError("non-finite training gradient") from exc
        self.optimizer.step()
        torch.cuda.synchronize()
        elapsed = time.monotonic() - started
        self.state["updates"] += 1
        if canonical:
            self.state["canonical_updates"] += 1
            self.state["canonical_examples_seen"] += len(entries)
        else:
            self.state["bootstrap_updates"] += 1
        self.state["training_wall_seconds"] += elapsed
        self.last_train = {
            "train_mse_including_initial_frame": loss.item(), "lr": lr,
            "gradient_norm_before_clip": norm.item(), "seconds": elapsed,
            "actual_batch_size": len(entries), "trajectories_per_second": len(entries) / elapsed,
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
        }
        if self.state["updates"] % 10 == 0 or self.state["updates"] == 1:
            self.append_history({"kind": "training", **self.last_train})
            event("trainer", "update", updates=self.state["updates"], phase=self.state["phase"], **self.last_train)

    def append_history(self, row):
        row = {"time": time.time(), "update": self.state["updates"],
               "phase": self.state["phase"], **row}
        with self.history_path.open("a") as file:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")

    def evaluate(self):
        signature = self.state["phase"] + ":" + source_signature(self.validation)
        if signature != self.state["validation_signature"]:
            self.state.update(validation_signature=signature, best_validation_mse=None, best_update=None)
            event("trainer", "validation_cohort_changed", signature=signature, scenes=len(self.validation))
        canonical = self.state["phase"] == "canonical"
        rows, baseline_rows = [], []
        for entry in self.validation:
            if self.stop.requested():
                return
            tracks = self.cache.batch([entry], canonical, self.device)
            prediction = predict(self.model, tracks)
            rows.append({"source": entry["source"], **metric(prediction, tracks)["per_scene"][0]})
            baseline = baselines(tracks, self.model.dt)["gravity_with_floor_stop"]
            baseline_rows.append(metric(baseline, tracks)["per_scene"][0])
            del prediction, tracks, baseline
        fields = ("rmse", "fpe", "r", "centroid_rmse", "centroid_final_error")
        mean = lambda values: {key: float(np.mean([x[key] for x in values if x[key] is not None])) for key in fields}
        score = mean(rows)
        mse = float(np.mean([x["rmse"]**2 for x in rows]))
        result = {"time": time.time(), "update": self.state["updates"],
                  "phase": self.state["phase"], "cohort_signature": signature,
                  "scenes": len(rows), "horizon_s": (80 - 1) * .02 if canonical else (31 - 1) * .04,
                  "ngff": score, "gravity_with_floor_stop": mean(baseline_rows),
                  "mean_scene_mse_excluding_initial_frame": mse, "per_scene": rows,
                  "units": "original simulation coordinates"}
        if self.state["best_validation_mse"] is None or mse < self.state["best_validation_mse"]:
            self.state["best_validation_mse"] = mse
            self.state["best_update"] = self.state["updates"]
            payload = {"state_dict": self.model.state_dict(), "config": self.config,
                       "training_state": self.state, "validation": result}
            atomic_torch(OUT / "best.pt", payload)
            # Retain the best checkpoint for each cohort when more scenes arrive.
            atomic_torch(OUT / f"best_{signature[:9].replace(':', '_')}_{signature.split(':')[-1][:12]}.pt", payload)
        self.state["validation"] = result
        atomic_json(OUT / "validation_latest.json", result)
        self.append_history({"kind": "validation", **result})
        self.last_validation = time.monotonic()
        event("trainer", "validation", updates=self.state["updates"], scenes=len(rows),
              rmse=score["rmse"], baseline_rmse=result["gravity_with_floor_stop"]["rmse"],
              best_update=self.state["best_update"], cohort=signature)
        self.plot_history()
        self.checkpoint("validation")

    def plot_history(self):
        lines = self.history_path.read_text().splitlines()[-20000:]
        history = []
        for line in lines:
            try:
                history.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        figure, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
        for phase, color in (("bootstrap", "#d88020"), ("canonical", "#1768ac")):
            train = [x for x in history if x["kind"] == "training" and x["phase"] == phase]
            if train:
                axes[0].plot([x["update"] for x in train], [x["train_mse_including_initial_frame"] for x in train], label=phase, color=color)
        axes[0].set(xlabel="Optimizer updates", ylabel="Training point MSE", yscale="log")
        axes[0].legend()
        vals = [x for x in history if x["kind"] == "validation"]
        signatures = list(dict.fromkeys(x["cohort_signature"] for x in vals))
        for signature in signatures[-8:]:
            cohort = [x for x in vals if x["cohort_signature"] == signature]
            axes[1].plot([x["update"] for x in cohort], [x["ngff"]["rmse"] for x in cohort], "o-", markersize=3,
                         label=f'{cohort[0]["phase"]}: {cohort[0]["scenes"]} scenes / {signature[-5:]}')
        axes[1].set(xlabel="Optimizer updates", ylabel="Validation RMSE (simulation units)", yscale="log")
        if signatures:
            axes[1].legend(fontsize=7)
        figure.suptitle("Growing data: compare validation errors only within the same cohort")
        temporary = OUT / "learning_curves.tmp.png"
        figure.savefig(temporary, dpi=150)
        plt.close(figure)
        os.replace(temporary, OUT / "learning_curves.png")

    def run(self):
        self.refresh_data()
        self.checkpoint("startup")
        while not self.stop.requested():
            now = time.monotonic()
            if now - self.last_refresh >= 30:
                self.refresh_data()
            if not self.train or not self.validation:
                self.status("waiting_for_data")
                self.stop.sleep(5)
                continue
            if now - self.last_validation >= self.config["validation_interval_seconds"]:
                self.status("validating")
                self.evaluate()
            if self.stop.requested():
                break
            if self.state["phase"] == "bootstrap" and self.state["bootstrap_updates"] >= self.config["bootstrap_updates"]:
                self.status("waiting_for_full_resolution_data")
                self.stop.sleep(5)
                continue
            self.train_step()
            self.status()
            if time.monotonic() - self.last_checkpoint >= self.config["checkpoint_interval_seconds"]:
                self.checkpoint("periodic")
        self.checkpoint("graceful_stop")
        self.status("stopped")
        event("trainer", "stopped", updates=self.state["updates"])


def main():
    config = setup()
    stop = StopFlag()
    if stop.requested():
        return 0
    training = None
    try:
        training = Training(config, stop)
        training.run()
        return 0
    except Exception as exc:
        recovery = read_json(STATE / "recovery.json", {})
        oom = isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in str(exc).lower()
        if oom:
            recovery["batch_size"] = max(1, (training.batch_limit if training else config["batch_size"]) // 2)
        if isinstance(exc, FloatingPointError):
            recovery["lr_factor"] = max(1e-4, recovery.get("lr_factor", 1.) * .5)
        recovery.update(last_error_type=type(exc).__name__, time=time.time())
        atomic_json(STATE / "recovery.json", recovery)
        atomic_json(STATE / "trainer_status.json", {
            "status": "restart_required", "time": time.time(), "error_type": type(exc).__name__,
            "error": str(exc), "updates": training.state["updates"] if training else None,
            "recovery": recovery,
        })
        event("trainer", "restart_required", error=type(exc).__name__, recovery=recovery)
        traceback.print_exc()
        return 75


if __name__ == "__main__":
    raise SystemExit(main())
