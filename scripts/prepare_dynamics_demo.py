"""Freeze a best checkpoint and export actual validation rollouts for a demo."""
import argparse
import base64
import hashlib
import json
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))

import numpy as np
import torch
from scripts.train_quality_smoke import new_model, predict, baselines, metric


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    frozen = args.output / "checkpoint.pt"
    if not frozen.exists():
        shutil.copyfile(ROOT / "artifacts/continuous/best.pt", frozen)
    saved = torch.load(frozen, map_location="cpu", weights_only=True)
    config = saved["config"]
    validation = saved["validation"]
    rows = sorted(validation["per_scene"], key=lambda row: row["source"])
    median = float(np.median([row["rmse"] for row in rows]))
    typical = min(rows, key=lambda row: abs(row["rmse"] - median))
    difficult = max(rows, key=lambda row: row["rmse"])
    selected = [("typical", "典型验证场景（误差接近中位数）", typical),
                ("difficult", "困难验证场景（本组误差最大）", difficult)]
    metadata = {row["source"]: row for path in (ROOT / "data/continuous/cache").glob("*.json")
                if (row := json.loads(path.read_text()))["split"] == "validation"}
    torch.set_num_threads(8)
    if args.device.startswith("cuda"):
        # One scene, inference only. Keep a small allocator budget while the
        # ongoing trainer retains its own CUDA context and cached memory.
        torch.cuda.set_per_process_memory_fraction(.5 * 1024**3 / torch.cuda.get_device_properties(0).total_memory)
    model = new_model(config, torch.device(args.device))
    model.load_state_dict(saved["state_dict"], strict=True)
    model.eval()
    frames = np.unique(np.append(np.arange(0, 80, 2), 79))
    report = {"created_local": datetime.now().isoformat(), "device": args.device,
              "checkpoint_update": saved["training_state"]["updates"],
              "checkpoint_sha256": hashlib.sha256(frozen.read_bytes()).hexdigest(),
              "checkpoint_validation_scenes": validation["scenes"],
              "selection": "closest to median RMSE and maximum RMSE in the frozen checkpoint's validation set",
              "data_revision": config["data_revision"], "frames": 80,
              "dt": config["sampled_dt"], "keypoints_per_object": 2048,
              "units": "simulation coordinates, not calibrated meters",
              "scope": "validation examples, not independent test or grasp success",
              "scenes": []}
    inline = {"checkpoint_update": report["checkpoint_update"], "frame_indices": frames.tolist(),
              "dt": config["sampled_dt"], "display_points_per_object": 128, "scenes": []}
    for kind, label, row in selected:
        entry = metadata[row["source"]]
        path = ROOT / entry["cache"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["cache_sha256"]
        with np.load(path) as data:
            array = data["points"].copy()
        assert array.shape == (80, 3, 2048, 3)
        truth = torch.from_numpy(array).unsqueeze(0).to(args.device)
        start = time.monotonic()
        predicted = predict(model, truth)
        baseline = baselines(truth, config["sampled_dt"])["gravity_with_floor_stop"]
        scores = {"ngff": metric(predicted, truth), "baseline": metric(baseline, truth)}
        trajectories = {"truth": truth[0].cpu().numpy(), "ngff": predicted[0].cpu().numpy(),
                        "baseline": baseline[0].cpu().numpy()}
        errors = {key: np.sqrt(np.square(values - array).mean(axis=(1, 2, 3)))
                  for key, values in trajectories.items() if key != "truth"}
        np.savez_compressed(args.output / (kind + ".npz"), **trajectories,
                            error_ngff=errors["ngff"], error_baseline=errors["baseline"])
        difference = scores["ngff"]["rmse"] - row["rmse"]
        scene = {"kind": kind, "label": label, "source": entry["source"],
                 "name": entry["name"], "objects": entry["objects"],
                 "cache_sha256": entry["cache_sha256"], "metrics": scores,
                 "saved_gpu_validation_rmse": row["rmse"],
                 "demo_minus_saved_gpu_rmse": difference,
                 "inference_seconds": time.monotonic() - start}
        report["scenes"].append(scene)
        visual = {key: values[frames, :, :128].astype("<f2") for key, values in trajectories.items()}
        all_points = np.concatenate([values.reshape(-1, 3) for values in visual.values()])
        low, high = all_points.min(axis=0), all_points.max(axis=0)
        low[2] = min(low[2], -.94)
        inline["scenes"].append({**scene, "bounds": [low.tolist(), high.tolist()],
                                "metrics": {key: {k: v for k, v in value.items() if k != "per_scene"}
                                            for key, value in scores.items()},
                                "error_ngff": errors["ngff"][frames].round(6).tolist(),
                                "error_baseline": errors["baseline"][frames].round(6).tolist(),
                                "tracks": {key: base64.b64encode(values.tobytes()).decode()
                                           for key, values in visual.items()}})
        print(json.dumps({"scene": entry["source"], "kind": kind,
                          "inference_seconds": scene["inference_seconds"], "ngff_rmse": scores["ngff"]["rmse"],
                          "baseline_rmse": scores["baseline"]["rmse"],
                          "difference_from_saved_gpu_rmse": difference}, ensure_ascii=False), flush=True)
    (args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    (args.output / "inline-data.json").write_text(json.dumps(inline, ensure_ascii=False, separators=(",", ":")))
    print("READY", args.output, flush=True)


if __name__ == "__main__":
    main()
