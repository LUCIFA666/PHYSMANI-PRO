"""Owned headless Panda simulator probe; no connection to running baseline servers."""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import h5py
from pyrep import PyRep
from pyrep.robots.arms.panda import Panda
from our_mani.contracts import Pose, RobotState
from our_mani.reachability import PyRepRobotModel, TimedReachability


def main():
    import rlbench
    scene = Path(rlbench.__file__).parent/'task_design.ttt'
    simulator = PyRep()
    simulator.launch(str(scene), headless=True)
    try:
        simulator.start()
        arm = Panda(max_velocity=3., max_acceleration=12.)
        # task_design.ttt stores a straight, singular arm before RLBench resets.
        # Seed the owned simulator from an actual training demonstration pose.
        data_root = Path('/home/ubuntu/桌面/phsimani/PhysMani/data/rmt/physmani_bench/train_package_compressed')
        seed_path = sorted(data_root.glob('pick_moving_target_on_the_table/all_variations/episodes/*/data.h5'))[0]
        with h5py.File(seed_path, 'r') as f:
            arm.set_joint_positions(f['low_dim_obs/joint_positions'][0].tolist())
        model = PyRepRobotModel(arm)
        original_q = np.asarray(arm.get_joint_positions())
        original_target = np.asarray(arm._ik_target.get_pose())
        pose = model.forward(original_q)
        robot = RobotState(0., pose, np.zeros(3), np.zeros(3), original_q, np.zeros(7))
        target = Pose(pose.position+np.array([.015, 0., .005]), pose.rotation)
        check = TimedReachability(model).check(robot, target, np.array([.02, 0., 0.]), np.zeros(3), .7)
        np.testing.assert_allclose(arm.get_joint_positions(), original_q, atol=1e-7)
        restored_target = Pose.from_xyzw(arm._ik_target.get_pose())
        previous_target = Pose.from_xyzw(original_target)
        # q and -q denote the same orientation, and simulator storage is float32.
        np.testing.assert_allclose(restored_target.position, previous_target.position, atol=1e-6)
        np.testing.assert_allclose(restored_target.rotation, previous_target.rotation, atol=1e-6)
        if not check.feasible:
            print('diagnostic_start_joints', original_q.tolist(), 'pose', pose.xyzw().tolist(), flush=True)
            raise RuntimeError(f"owned Panda reachability smoke failed: {check.reason}, {check.evidence}")
        report = {"created_utc": datetime.now(timezone.utc).isoformat(), "kind": "owned_pyrep_reachability_smoke_not_grasp_benchmark",
                  "robot": model.name, "scene": str(scene), "initial_pose_xyzw": pose.xyzw().tolist(),
                  "initial_joint_seed_train_episode": str(seed_path),
                  "target_pose_xyzw": target.xyzw().tolist(), "feasible": check.feasible,
                  "reason": check.reason, "evidence": check.evidence,
                  "joint_limits_upper": model.upper.tolist(), "target_joint_velocity": check.joint_velocities.tolist(),
                  "simulator_state_restored_after_queries": True, "baseline_server_connections": 0}
        output = ROOT/'artifacts/framework/smoke/pyrep_report.json'
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
        print(json.dumps(report, ensure_ascii=False, indent=2))
    finally:
        simulator.stop()
        simulator.shutdown()


if __name__ == '__main__':
    main()
