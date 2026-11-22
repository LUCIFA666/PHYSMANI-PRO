"""Run contract tests and a real frozen-checkpoint NGFF adapter smoke on GPU."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from our_mani.contracts import ObjectState, Pose
from our_mani.dynamics import NGFFDynamics, NGFFFrame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT/"artifacts/framework/smoke")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    torch.set_num_threads(4)
    args.output.mkdir(parents=True, exist_ok=True)
    suite = unittest.defaultTestLoader.discover(str(ROOT/"tests"), pattern="test_framework.py")
    tests = unittest.TextTestRunner(verbosity=2).run(suite)
    if not tests.wasSuccessful():
        raise SystemExit(1)
    # Deliberately calibrated synthetic SI test world. This is not PhysMani data.
    frame = NGFFFrame(.25, .4, np.array([1., 2., 3.]),
                      np.array([1.94, -1.94, 1.94, -1.94, 2.94, -.94]), "synthetic_coordinate_roundtrip_only")
    model = NGFFDynamics(ROOT/"artifacts/framework/checkpoints/ngff_best.pt", frame, device=args.device, keypoints=128)
    rng = np.random.default_rng(7)
    objects = []
    for i, center in enumerate([[-.6, 0., .8], [0., 0., .8], [.6, 0., .8]]):
        local = rng.uniform(-.025, .025, (128, 3))
        local -= local.mean(0)
        rotation = Rotation.from_euler("xyz", [.2, -.1, .4]).as_matrix()
        pose = Pose(np.asarray(center)*.25+frame.world_origin, rotation)
        # Visible-cloud centroid differs from the tracked object frame origin.
        points = pose.apply((local+np.array([.04, 0., 0.]))*.25)
        objects.append(ObjectState(f"object_{i}", 5., pose, points, [.03, -.01, 0.], [0., 0., .1]))
    warmup = model.predict(objects, horizon=.1, dt=.02)
    forecast = model.predict(objects, horizon=.3, dt=.02)
    initial_points = np.stack([s.points_world for s in objects])
    initial_error = float(np.max(np.abs(forecast.points_world[0]-initial_points)))
    assert initial_error < 3e-6
    np.testing.assert_allclose(forecast.positions[0], [s.pose.position for s in objects], atol=3e-6)
    np.testing.assert_allclose(forecast.rotations[0], [s.pose.rotation for s in objects], atol=3e-6)
    assert np.isfinite(forecast.points_world).all()
    # World velocity at t0 must survive coordinate/time scaling unchanged.
    np.testing.assert_allclose(forecast.linear_velocities[0], [s.linear_velocity for s in objects])
    # Rigid centroid offset must not create a spurious origin velocity.
    initial_xy_velocity = (forecast.positions[1, :, :2]-forecast.positions[0, :, :2])/.02
    np.testing.assert_allclose(initial_xy_velocity, [s.linear_velocity[:2] for s in objects], atol=2e-5)
    np.savez_compressed(args.output/"real_ngff_adapter_rollout.npz", times=forecast.times,
                        points_world=forecast.points_world, positions=forecast.positions,
                        rotations=forecast.rotations, velocities=forecast.linear_velocities)
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "kind": "framework_contract_smoke_not_grasp_benchmark",
              "unit_tests": tests.testsRun, "failures": len(tests.failures), "errors": len(tests.errors),
              "ngff_real_weights": forecast.metadata, "initial_roundtrip_max_error": initial_error,
              "warmup_latency_seconds": warmup.metadata["latency_seconds"],
              "ngff_training_resumed": False, "faster_policy_inference_executed": False,
              "faster_transport_test": "scripted in-process test double; not trained model performance",
              "not_yet_validated": ["PhysMani closed-loop success", "FASTER conditioned weights", "learned local policy/value",
                                    "real camera calibration", "NGFF transfer to driven motion"]}
    (args.output/"report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
