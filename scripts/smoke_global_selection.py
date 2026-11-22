"""Real Panda global handover selection from the released home configuration."""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'scripts')]
import numpy as np
from our_mani.benchmark_perception import RedBoxTracker
from our_mani.contracts import Pose, HandoverRegion
from our_mani.dynamics import ConstantVelocityDynamics
from our_mani.planning import RegionSelector
from our_mani.pyrep_env import LocalSimulation
from our_mani.reachability import TimedReachability
from collect_local_rollouts import initialization


def main():
    rows = json.loads((ROOT/'artifacts/framework/protocol/train_manifest.json').read_text())['episodes']
    rows = [next(r for r in rows if r['task'] == task) for task in
            ('pick_moving_target_on_the_table', 'pick_moving_target_on_the_table_high_speed')]
    results = []
    for row in rows:
        environment = LocalSimulation(row['task'], row['path'], seed=421, servo_lead=1.4)
        try:
            x, v, a, marker = initialization(row['path'])
            environment.reset(target_position=x, target_velocity=v, target_acceleration=a, goal_position=marker)
            # This is the episode RESET. No control trajectory is teleported.
            environment.arm.set_joint_positions(environment.home_joints.tolist(), disable_dynamics=True)
            environment.arm.set_joint_target_positions(environment.home_joints.tolist())
            for _ in range(3):
                environment.pyrep.step()
            tracker = RedBoxTracker()
            for _ in range(6):
                obs, robot = environment.observe()
                obj, _ = tracker.observe(obs)
                environment.advance(robot.pose)
            obs, robot = environment.observe()
            obj, _ = tracker.observe(obs)
            forecast = ConstantVelocityDynamics().predict([obj], 1.1, .05)
            regions = [HandoverRegion('local_capture_development', 'target',
                        Pose.from_xyzw([0, 0, .07, 0, 1, 0, 0]), [.04, .04, .05],
                        max_relative_speed=.14, value=.5, source='configured_smoke_not_learned_region')]
            selector = RegionSelector(TimedReachability(environment.robot_model), latency_reserve=.05)
            tick = time.perf_counter()
            goal = selector.select(forecast, regions, robot, environment.time, stride=2)
            elapsed = time.perf_counter()-tick
            result = {'task': row['task'], 'kind': 'real_robot_model_region_selection_smoke_not_faster_inference',
                'home_pose': robot.pose.xyzw().tolist(), 'object_pose_from_rgbd': obj.pose.xyzw().tolist(),
                'object_velocity_from_rgbd': obj.linear_velocity.tolist(), 'selection_latency_seconds': elapsed,
                'feasible_goal_found': goal is not None, 'diagnostics': selector.last_diagnostics,
                'formal_test_used': False, 'baseline_jobs_connected': False}
            if goal is not None:
                result.update(arrival_seconds=goal.arrival_time-environment.time, target_pose=goal.pose.xyzw().tolist(),
                              terminal_velocity=goal.linear_velocity.tolist(), reachability=goal.reachability.evidence)
            results.append(result)
            print(json.dumps(result, ensure_ascii=False), flush=True)
        finally:
            environment.close()
    out = ROOT/'artifacts/framework/smoke/global_selection_report.json'
    out.write_text(json.dumps({'created_utc': datetime.now(timezone.utc).isoformat(), 'results': results}, ensure_ascii=False, indent=2)+'\n')


if __name__ == '__main__':
    main()
