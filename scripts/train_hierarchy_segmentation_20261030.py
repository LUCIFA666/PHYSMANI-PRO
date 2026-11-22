"""Fit current-pixel public RGBD segmentation from four fixed TRAIN captures.

The first source per variant is fitted; the second is TRAIN development data.
No original VAL/TEST images, annotations, results or task IDs are consumed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from our_mani.hierarchy_20261030.segmentation import (
    SCHEMA, CLASS_NAMES, FEATURE_NAMES, PointSemanticMLP, balanced_pixel_indices,
    capture_training_records, evaluate_pixels, fit_feature_normalization,
    load_fit_and_development_pixels, sha256,
)


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help="CPU-only source/feature/label audit without optimizer")
    mode.add_argument("--train", action="store_true")
    parser.add_argument("--plan", type=Path, default=ROOT / "artifacts/framework/multitask_r10/hierarchy_20261030/visual_train_v1/plan_v1.json")
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts/framework/multitask_r10/hierarchy_20261030/segmentation_v1")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--samples-per-class", type=int, default=20000)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=20261030)
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.samples_per_class, args.hidden) < 1 or args.learning_rate <= 0:
        raise ValueError("positive training parameters required")
    if args.seed != 20261030:
        raise ValueError("this fixed experiment uses seed 20261030")
    started = time.perf_counter()
    torch.set_num_threads(4)
    records, plan_ref = capture_training_records(ROOT, args.plan)
    data, counts = load_fit_and_development_pixels(records)
    source_refs = [{**{k: v for k, v in row.items() if k not in ("input_path", "label_path")},
                    "input_path": str(row["input_path"].relative_to(ROOT)),
                    "label_path": str(row["label_path"].relative_to(ROOT))} for row in records]
    audit = dict(schema=SCHEMA, capture_plan=plan_ref, source_frames=source_refs,
                 fit_cases=sorted({r["case_key"] for r in records if r["partition"] == "fit"}),
                 development_cases=sorted({r["case_key"] for r in records if r["partition"] == "development"}),
                 pixel_counts={key: {CLASS_NAMES[i]: int(n) for i, n in enumerate(value)} for key, value in counts.items()},
                 input_features=list(FEATURE_NAMES), class_names=list(CLASS_NAMES),
                 normalization_fit_split="first TRAIN source of each variant only",
                 fit_split="train", development_split="train", TEST_read=False,
                 original_VAL_read=False, native_or_instance_inputs_used=False,
                 rendered_instance_arrays_used_as_input=False, optimizer_steps=0)
    if args.check:
        print(json.dumps(dict(mode="CPU_check", **audit, elapsed_seconds=time.perf_counter() - started), ensure_ascii=False, allow_nan=False))
        return 0
    output = args.out.resolve()
    permitted = ROOT / "artifacts/framework/multitask_r10/hierarchy_20261030"
    if not output.is_relative_to(permitted) or output.exists():
        raise ValueError("fresh hierarchy artifact output namespace required")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA device unavailable")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    rng = np.random.default_rng(args.seed)
    train_x, train_y = data["fit"]
    dev_x, dev_y = data["development"]
    mean, std = fit_feature_normalization(train_x)
    model = PointSemanticMLP(args.hidden, mean, std).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    output.mkdir(parents=True)
    code_refs = {name: sha256(ROOT / name) for name in (
        "our_mani/hierarchy_20261030/segmentation.py",
        "scripts/train_hierarchy_segmentation_20261030.py",
        "our_mani/hierarchy_20261030/native_capture.py",
    )}
    manifest = dict(**audit, stage="training", training_complete=False,
                    started_utc=datetime.now(timezone.utc).isoformat(), code_sha256=code_refs,
                    hyperparameters=dict(epochs=args.epochs, batch_size=args.batch_size,
                        samples_per_class=args.samples_per_class, hidden=args.hidden,
                        learning_rate=args.learning_rate, seed=args.seed, device=str(device)),
                    feature_mean=mean.tolist(), feature_std=std.tolist(),
                    checkpoint_selection="final declared epoch; TRAIN development metrics are diagnostics only",
                    fit_sparse_classes=[CLASS_NAMES[i] for i in np.flatnonzero(counts["fit"] < 12)],
                    history=[])
    write_json(output / "manifest.json", manifest)
    try:
        for epoch in range(1, args.epochs + 1):
            indices = balanced_pixel_indices(train_y, rng, args.samples_per_class)
            model.train()
            loss_sum, seen = 0., 0
            for start in range(0, len(indices), args.batch_size):
                selected = indices[start:start + args.batch_size]
                x = torch.as_tensor(train_x[selected], dtype=torch.float32, device=device)
                y = torch.as_tensor(train_y[selected], dtype=torch.long, device=device)
                optimizer.zero_grad(set_to_none=True)
                loss = F.cross_entropy(model(x), y)
                if not torch.isfinite(loss):
                    raise ValueError("nonfinite segmentation loss")
                loss.backward()
                nnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10.)
                if not torch.isfinite(nnorm):
                    raise ValueError("nonfinite segmentation gradient")
                optimizer.step()
                manifest["optimizer_steps"] += 1
                loss_sum += float(loss.detach()) * len(selected)
                seen += len(selected)
            development = evaluate_pixels(model, dev_x, dev_y, device=device)
            row = dict(epoch=epoch, balanced_fit_loss=loss_sum / seen,
                       development=development, elapsed_seconds=time.perf_counter() - started)
            manifest["history"].append(row)
            write_json(output / "manifest.json", manifest)
            print(json.dumps(dict(stage="epoch", **row), allow_nan=False), flush=True)
        if sha256(args.plan) != plan_ref["sha256"] or any(sha256(ROOT / key) != value for key, value in code_refs.items()):
            raise ValueError("bound source code or capture plan changed during fit")
        for row in records:
            if sha256(row["input_path"]) != row["input_sha256"] or sha256(row["label_path"]) != row["label_sha256"]:
                raise ValueError("capture arrays changed during fit")
        metadata = dict(schema=SCHEMA, training_complete=True, fit_split="train", development_split="train",
                        TEST_read=False, original_VAL_read=False, input_features=list(FEATURE_NAMES),
                        class_names=list(CLASS_NAMES), native_or_instance_inputs_used=False,
                        fit_cases=audit["fit_cases"], development_cases=audit["development_cases"],
                        code_sha256=code_refs, capture_plan=plan_ref,
                        selected_epoch=args.epochs, selection="final declared epoch",
                        probabilities_are_calibrated=False)
        temporary = output / "model.partial.pt"
        torch.save({"spec": {"hidden": args.hidden},
                    "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                    "metadata": metadata}, temporary)
        temporary.replace(output / "model.pt")
        manifest.update(stage="complete", training_complete=True,
                        model={"path": str((output / "model.pt").relative_to(ROOT)), "sha256": sha256(output / "model.pt")},
                        elapsed_seconds=time.perf_counter() - started,
                        finished_utc=datetime.now(timezone.utc).isoformat())
        write_json(output / "manifest.json", manifest)
        print(json.dumps(dict(stage="complete", manifest=str(output / "manifest.json"),
                              elapsed_seconds=manifest["elapsed_seconds"])), flush=True)
        return 0
    except BaseException as error:
        manifest.update(stage="failed", training_complete=False, error=repr(error),
                        elapsed_seconds=time.perf_counter() - started)
        write_json(output / "manifest.json", manifest)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
