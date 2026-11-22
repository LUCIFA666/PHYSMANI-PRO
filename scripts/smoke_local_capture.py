"""Real RGB-D local-controller smoke in an owned simulator, TRAIN initialization."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import h5py
import numpy as np

from our_mani.pyrep_env import LocalSimulation
from our_mani.benchmark_perception import RedBoxTracker
from our_mani.contracts import Pose
from our_mani.control import VisualMPC


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=4)
    parser.add_argument('--oracle-debug', action='store_true')
    args = parser.parse_args()
    rows = json.loads((ROOT/'artifacts/framework/protocol/train_manifest.json').read_text())['episodes']
    chosen = [r for r in rows if r['task'] == 'pick_moving_target_on_the_table'][:args.episodes]
    environment = LocalSimulation(chosen[0]['task'], chosen[0]['path'])
    controller = VisualMPC(max_speed=.8, max_acceleration=3.)
    grasp = Pose.from_xyzw([0., 0., -.015, 0., 1., 0., 0.])
    results = []
    out = ROOT/'artifacts/framework/local_policy'
    out.mkdir(parents=True, exist_ok=True)
    try:
        for row in chosen:
            with h5py.File(row['path'], 'r') as f:
                positions = f['low_dim_obs/expert_trajectory_target_position'][:5]
            coeff = np.linalg.lstsq(np.c_[np.ones(5), np.arange(5)*.05, .5*(np.arange(5)*.05)**2], positions, rcond=None)[0]
            x, v, a = coeff
            x[2], v[2], a[2] = .775, 0., 0.
            valid = environment.reset(target_position=x, target_velocity=v, target_acceleration=a, relative_ee=(0., 0., .07))
            if not valid:
                results.append({'episode': row['raw_episode_id'], 'reset_ik_failed': True, 'captured': False})
                continue
            tracker = RedBoxTracker()
            history, captured, lifted = [], False, False
            stage, lift_goal, close_time = 'track', None, None
            for step in range(90):
                if (ROOT/'runtime/framework/local_state/STOP').exists():
                    break
                tick = time.perf_counter()
                observation, robot = environment.observe()
                try:
                    obj, meta = tracker.observe(observation)
                    if args.oracle_debug:
                        from dataclasses import replace
                        obj = replace(obj, pose=Pose(environment.metric_target_position()))
                    error = np.linalg.norm(obj.pose.position-environment.metric_target_position())
                except ValueError as exception:
                    history.append({'step': step, 'time': environment.time, 'perception_failure': str(exception)})
                    environment.advance(robot.pose)
                    continue
                if robot.contact and not captured:
                    captured = True
                    stage = 'lift'
                    lift_goal = Pose(robot.pose.position+np.array([0., 0., .12]), robot.pose.rotation)
                if stage == 'lift':
                    delta = lift_goal.position-robot.pose.position
                    velocity = delta*3
                    velocity *= min(1., .2/max(np.linalg.norm(velocity), 1e-8))
                    desired = Pose(robot.pose.position+velocity*.05, robot.pose.rotation)
                    lifted = robot.contact and environment.metric_target_position()[2]-x[2] > .08
                    if lifted and step > 15:
                        break
                    opening = 0.
                else:
                    target = obj.pose.apply(grasp.position)
                    distance = np.linalg.norm(robot.pose.position-target)
                    relative_speed = np.linalg.norm(robot.linear_velocity-obj.linear_velocity)
                    if close_time is None and distance < .022 and relative_speed < .14:
                        close_time = environment.time
                    opening = 0. if close_time is not None else 1.
                    try:
                        desired, velocity, angular, details = controller.command(robot, obj, grasp)
                    except RuntimeError as exception:
                        history.append({'step': step, 'controller_failure': str(exception),
                                        'robot_velocity': robot.linear_velocity.tolist(), 'object_velocity': obj.linear_velocity.tolist(),
                                        'robot_pose': robot.pose.xyzw().tolist()})
                        break
                moved = environment.advance(desired, opening=opening)
                history.append({'step': step, 'time': environment.time, 'pose': robot.pose.xyzw().tolist(),
                                'target_estimate': obj.pose.position.tolist(), 'perception_error_m': float(error),
                                'contact': robot.contact, 'stage': stage, 'opening': opening,
                                'action_applied': moved, 'wall_step_seconds': time.perf_counter()-tick})
            result = {'episode': row['raw_episode_id'], 'task': row['task'], 'captured': captured,
                      'lifted': bool(lifted), 'steps': len(history), 'oracle_debug': args.oracle_debug,
                      'official_task_success': bool(environment.task.success()[0]), 'history': history,
                      'kind': 'training_initialization_local_control_smoke_not_formal_benchmark'}
            results.append(result)
            (out/'smoke_report.json').write_text(json.dumps({'created_utc': datetime.now(timezone.utc).isoformat(),
                    'results': results, 'formal_test_used': False}, ensure_ascii=False, indent=2)+'\n')
            print(json.dumps({k:v for k,v in result.items() if k != 'history'}), flush=True)
    finally:
        environment.close()


if __name__ == '__main__':
    main()
