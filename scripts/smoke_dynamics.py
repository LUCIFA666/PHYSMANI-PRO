"""Exercise the official NGFF dynamics on synthetic point clouds.

This verifies execution and measures synthetic workloads. It does not evaluate
prediction accuracy on GSCollision or produce a pretrained model.
"""

import argparse
import hashlib
import importlib.metadata
import json
import platform
import statistics
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F

from dynamic_models import NGFFobj
from utils.transformation_utils import euler_xyz_to_matrix


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def make_state(batch, objects, keypoints, device, contact=False, padded=False):
    """World-space points; zero initial orientation; nonzero tracked velocities."""
    centers = torch.tensor(
        [[-0.65, 0.0, 1.0], [0.0, 0.0, 1.0], [0.65, 0.0, 1.0],
         [-0.65, 0.65, 1.0], [0.0, 0.65, 1.0], [0.65, 0.65, 1.0]],
        device=device,
    )[:objects].unsqueeze(0).repeat(batch, 1, 1)
    if contact:
        centers[:, 0] = torch.tensor([-0.02, 0.0, -0.91], device=device)
        if objects > 1:
            centers[:, 1] = torch.tensor([0.02, 0.0, -0.91], device=device)
    local = (torch.rand(batch, objects, keypoints, 3, device=device) - 0.5) * 0.07
    local -= local.mean(dim=-2, keepdim=True)
    points = local + centers.unsqueeze(-2)
    velocity = torch.zeros_like(centers)
    if not contact:
        velocity[..., 0] = 0.12
        velocity[..., 1] = -0.05
        velocity[..., 2] = 0.03
    angles = torch.zeros_like(centers)
    angle_velocity = torch.zeros_like(centers)
    if not contact:
        angle_velocity[..., 2] = 0.1
    padding = torch.ones(batch, objects, keypoints, device=device)
    if padded:
        padding[..., -8:] = 0
    knn = torch.zeros(batch, objects, keypoints, 8, dtype=torch.long, device=device)
    return points, centers, velocity, angles, angle_velocity, padding, knn


def make_model(device, keypoints=64, hidden_dim=32, num_layers=2):
    return NGFFobj(
        input_dim=3, hidden_dim=hidden_dim, output_dim=3,
        num_layers=num_layers, num_keypoints=keypoints, k=8,
        mass=0.1, dt=0.02, ode_method="euler", step_size=0.02,
        threshold=0.05,
    ).to(device)


def world_points(output):
    points, centers, angles = output
    return points @ euler_xyz_to_matrix(angles).transpose(-2, -1) + centers.unsqueeze(-2)


def validate_output(output, batch, frames, objects, keypoints):
    expected = [(batch, frames, objects, keypoints, 3),
                (batch, frames, objects, 3), (batch, frames, objects, 3)]
    for value, shape in zip(output, expected):
        assert tuple(value.shape) == shape, (tuple(value.shape), shape)
        assert torch.isfinite(value).all().item(), "non-finite prediction"


def check_ballistic(device):
    model = make_model(device).eval()
    state = make_state(1, 3, 64, device)
    with torch.no_grad():
        output = model(*state, pred_len=9)
    validate_output(output, 1, 10, 3, 64)
    torch.testing.assert_close(world_points(output)[:, 0], state[0], atol=2e-6, rtol=0)
    n = torch.arange(10, device=device, dtype=torch.float32).view(1, 10, 1, 1)
    gravity = torch.tensor([0., 0., -5.], device=device).view(1, 1, 1, 3)
    expected = state[1].unsqueeze(1) + n * 0.02 * state[2].unsqueeze(1)
    expected += n * (n - 1) / 2 * 0.02 ** 2 * gravity
    torch.testing.assert_close(output[1], expected, atol=3e-6, rtol=0)
    torch.testing.assert_close(
        output[2], n * 0.02 * state[4].unsqueeze(1), atol=3e-6, rtol=0,
    )
    return {"com_max_error": (output[1] - expected).abs().max().item(),
            "note": "explicit Euler analytic CoM trajectory; original gravity=-5"}


def check_shapes_and_masks(device):
    cases = []
    for batch, objects in [(1, 1), (2, 3), (1, 6)]:
        model = make_model(device).eval()
        state = make_state(batch, objects, 64, device, contact=True, padded=True)
        with torch.no_grad():
            output = model(*state, pred_len=9)
            intersections, overlaps = model.predictor._calc_intersection(
                state[0], state[5].unsqueeze(-1), threshold=0.05,
            )
            boundary, boundary_points = model.predictor._calc_boundary(
                state[0], state[5].unsqueeze(-1), model.boundary, threshold=0.05,
            )
        validate_output(output, batch, 10, objects, 64)
        assert boundary[:, 0].all().item(), "floor-contact branch was not exercised"
        if objects > 1:
            assert intersections[:, 0, 1].all().item(), "object-contact branch not exercised"
        assert not overlaps[..., -8:].any().item(), "padded points entered contact masks"
        assert not boundary_points[..., -8:, :].any().item(), "padded boundary points"
        torch.testing.assert_close(
            output[0][..., -8:, :],
            output[0][:, :1, :, -8:, :].expand_as(output[0][..., -8:, :]),
            atol=2e-6, rtol=0,
        )
        cases.append({"batch": batch, "objects": objects, "frames": 10,
                      "contact_entries": int(intersections.sum().item())})
    return cases


def check_training_and_checkpoint(device, out_dir):
    model = make_model(device)
    state = make_state(2, 3, 64, device, contact=True, padded=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    output = model(*state, pred_len=9)
    prediction = world_points(output)
    target = state[0].unsqueeze(1).expand_as(prediction)
    valid = state[5].unsqueeze(1).unsqueeze(-1).expand_as(prediction).bool()
    loss = F.mse_loss(prediction[valid], target[valid])
    assert torch.isfinite(loss).item()
    loss.backward()
    gradients = {}
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"missing gradient: {name}"
            assert torch.isfinite(p.grad).all().item(), f"non-finite gradient: {name}"
            group = name.split(".")[1]
            gradients[group] = gradients.get(group, 0.) + p.grad.detach().square().sum().item()
    for group in ("pointnet", "state_embedding", "brunch", "trunk", "output", "boundary_net", "stress_net"):
        assert gradients.get(group, 0) > 0, f"zero gradient for {group}"
    optimizer.step()
    changed = sum(not torch.equal(before[name], p.detach()) for name, p in model.named_parameters())
    assert changed > 0, "optimizer changed no parameters"
    model.eval()
    with torch.no_grad():
        reference = model(*state, pred_len=9)
    checkpoint = out_dir / "smoke_checkpoint.pt"
    torch.save({"state_dict": model.state_dict(), "optimizer": optimizer.state_dict(),
                "purpose": "smoke_only_random_initialization_and_one_synthetic_update",
                "config": {"hidden_dim": 32, "num_layers": 2, "num_keypoints": 64}}, checkpoint)
    restored = make_model(device).eval()
    saved = torch.load(checkpoint, map_location=device, weights_only=True)
    restored.load_state_dict(saved["state_dict"], strict=True)
    restored_optimizer = torch.optim.Adam(restored.parameters(), lr=1e-5)
    restored_optimizer.load_state_dict(saved["optimizer"])
    with torch.no_grad():
        actual = restored(*state, pred_len=9)
    for a, b in zip(actual, reference):
        torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-6)
    return {"synthetic_loss": loss.item(), "updated_parameter_tensors": changed,
            "gradient_l2_by_group": {k: v ** 0.5 for k, v in gradients.items()},
            "checkpoint_max_error": max((a-b).abs().max().item() for a,b in zip(actual,reference)),
            "checkpoint": str(checkpoint)}


def measure(device, function, warmup=2, repeats=5):
    for _ in range(warmup):
        function()
    synchronize(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    times = []
    for _ in range(repeats):
        synchronize(device)
        start = time.perf_counter()
        function()
        synchronize(device)
        times.append((time.perf_counter() - start) * 1000)
    ordered = sorted(times)
    return {"samples_ms": times, "median_ms": statistics.median(times),
            "min_ms": min(times), "max_ms": max(times),
            "p95_ms": ordered[min(len(ordered)-1, int(0.95*len(ordered)))],
            "peak_allocated_mib": torch.cuda.max_memory_allocated(device)/1024**2 if device.type=="cuda" else None,
            "peak_reserved_mib": torch.cuda.max_memory_reserved(device)/1024**2 if device.type=="cuda" else None}


def benchmark(device):
    results = []
    # Same hidden width/layers as the paper. Points are per object, not per scene.
    for keypoints in (512, 2048):
        model = make_model(device, keypoints=keypoints, hidden_dim=200, num_layers=4).eval()
        state = make_state(1, 3, keypoints, device, contact=True)
        for frames in (26, 80):
            with torch.no_grad():
                validate_output(model(*state, pred_len=frames-1), 1, frames, 3, keypoints)
            def infer():
                with torch.no_grad():
                    world_points(model(*state, pred_len=frames-1))
            result = {"kind": "inference", "batch": 1, "objects": 3,
                      "keypoints_per_object": keypoints, "frames": frames,
                      "future_horizon_s": (frames-1)*0.02, "hidden_dim": 200, "num_layers": 4}
            result.update(measure(device, infer))
            results.append(result)
            print("BENCH", json.dumps(result), flush=True)
        del model, state
    for batch in (1, 9):
        model = make_model(device, keypoints=2048, hidden_dim=200, num_layers=4).train()
        state = make_state(batch, 3, 2048, device, contact=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
        def train_step():
            optimizer.zero_grad(set_to_none=True)
            output = world_points(model(*state, pred_len=79))
            loss = F.mse_loss(output, state[0].unsqueeze(1).expand_as(output))
            if not torch.isfinite(loss).item():
                raise AssertionError("benchmark loss is not finite")
            loss.backward()
            optimizer.step()
        result = {"kind": "forward_backward_adam", "batch": batch, "objects": 3,
                  "keypoints_per_object": 2048, "frames": 80,
                  "hidden_dim": 200, "num_layers": 4}
        result.update(measure(device, train_step, warmup=1, repeats=3))
        assert all(torch.isfinite(p).all().item() for p in model.parameters()), "non-finite updated weights"
        assert all(p.grad is None or torch.isfinite(p.grad).all().item() for p in model.parameters()), "non-finite benchmark gradients"
        results.append(result)
        print("BENCH", json.dumps(result), flush=True)
        del model, state, optimizer
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/dynamics_smoke")
    parser.add_argument("--skip-benchmarks", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    report = {"status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
              "scope": "synthetic execution checks; not paper accuracy reproduction",
              "checks": {}, "benchmarks": []}
    try:
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        torch.manual_seed(0)
        torch.set_num_threads(4)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(0)
        report["environment"] = {
            "python": sys.version, "platform": platform.platform(),
            "device": str(device), "torch": torch.__version__, "cuda": torch.version.cuda,
            "packages": {name: importlib.metadata.version(name) for name in
                         ("numpy", "scipy", "torchdiffeq", "torch-cluster")},
            "gpu": torch.cuda.get_device_name(device) if device.type=="cuda" else None,
            "compute_capability": torch.cuda.get_device_capability(device) if device.type=="cuda" else None,
            "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "upstream_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "source_sha256": {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in
                              ("dynamic_models/ngff.py", "dynamic_models/interaction_net.py")},
        }
        unnecessary = [name for name in ("torch_geometric", "gsplat", "cv2", "h5py", "matplotlib", "pandas") if name in sys.modules]
        assert not unnecessary, f"unnecessary pipeline imports: {unnecessary}"
        report["checks"]["dynamics_only_import"] = {"status": "passed"}
        for name, check in [("ballistic_com", lambda: check_ballistic(device)),
                            ("shapes_contacts_padding", lambda: check_shapes_and_masks(device)),
                            ("backward_optimizer_checkpoint", lambda: check_training_and_checkpoint(device,args.output))]:
            start = time.perf_counter()
            result = check()
            synchronize(device)
            report["checks"][name] = {"status": "passed", "result": result,
                                       "elapsed_s": time.perf_counter()-start}
            print("PASS", name, json.dumps(result), flush=True)
        if not args.skip_benchmarks:
            report["benchmarks"] = benchmark(device)
        report["status"] = "passed"
    except Exception:
        report["status"] = "failed"
        report["error"] = traceback.format_exc()
        print(report["error"], file=sys.stderr, flush=True)
    finally:
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print("RESULT", report["status"], args.output / "report.json", flush=True)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
