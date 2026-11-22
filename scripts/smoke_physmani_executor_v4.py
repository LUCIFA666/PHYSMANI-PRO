"""TRAIN-only native executor protocol smoke; no policy/WM or success-rate claim."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'scripts')]

import numpy as np

from collect_local_rollouts import initialization, TASKS
from our_mani.aligned.simulation import AlignedLocalSimulation, physics_time
from our_mani.ablations.physmani_executor_v4 import PhysManiAdapter
from our_mani.benchmark_protocol import file_sha256


OUTPUT = ROOT/'artifacts/framework/aligned_r06/baseline_executor_v4_smoke'
CONFIG = ROOT/'config/framework/aligned_protocol_r06_baseline_v4.json'


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def check_case(env, row, config, *, points, stale_target=False):
    from pyrep.robots.configuration_paths.arm_configuration_path import ArmConfigurationPath
    x, velocity, acceleration, goal = initialization(row['path'])
    assert env.reset(target_position=x, target_velocity=velocity,
                     target_acceleration=acceleration, goal_position=goal)
    env.finish_home_reset()
    initial = np.asarray(env.arm.get_joint_positions())
    final = initial.copy()
    final[0] += .3
    path = ArmConfigurationPath(env.arm, np.linspace(initial, final, points).reshape(-1))
    if stale_target:
        env.arm.set_joint_target_positions((initial+.04).tolist())
    adapter = object.__new__(PhysManiAdapter)
    adapter.config = config
    adapter.observe = lambda observation, robot: None
    original_plan = env.map_service.plan_path
    env.map_service.plan_path = lambda *args, **kwargs: path
    before = physics_time()
    try:
        # This known workspace pose only passes the action-boundary check.
        # The native path is synthetic: no neural action or future data used.
        execution = adapter.execute(env, np.array([.2, 0., 1.02, 0., 1., 0., 0., 1.]))
    finally:
        env.map_service.plan_path = original_plan
    receipt = env.alignment_receipt()
    clock = receipt['clock']
    assert execution['ticks'] == 10
    assert clock['ticks'] == 10
    assert clock['phases'] == {'physmani_path': 9, 'physmani_gripper': 1}
    assert abs(receipt['actual_physics_seconds']-.5) < 2e-5
    assert abs(receipt['actual_task_seconds']-.5) < 2e-5
    assert abs(physics_time()-before-.5) < 2e-5
    assert execution['common_joint_command_clipping']['command_ticks'] == 10
    targets = np.asarray(env.arm.get_joint_target_positions())
    assert np.isfinite(targets).all()
    assert np.all(targets >= env.robot_model.lower-1e-6)
    assert np.all(targets <= env.robot_model.upper+1e-6)
    result = dict(passed=True, synthetic_raw_waypoints=points,
                  execution=execution, alignment_receipt=receipt,
                  native_target_validation='v4 checked every applied native servo target',
                  target_reset_max_delta_rad=None)
    if stale_target:
        assert not execution['path_found']
        assert execution['error'] == 'released_modified_path_too_short'
        difference = float(np.max(np.abs(targets-initial)))
        assert difference < 1e-6
        result['target_reset_max_delta_rad'] = difference
    else:
        assert execution['path_found']
        assert execution['waypoint_modification']['selected_waypoints'] >= 2
        assert execution['arm_error'] is None
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=TASKS)
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if not args.task:
        statuses = []
        for task in TASKS:
            process = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--task', task])
            statuses.append(process.returncode)
        reports = [json.loads((OUTPUT/(task+'.json')).read_text()) for task in TASKS]
        report = dict(created_utc=datetime.now(timezone.utc).isoformat(),
                      passed=all(status == 0 for status in statuses) and all(r['passed'] for r in reports),
                      split='train', formal_test_used=False,
                      scope='native executor protocol smoke only; not full policy evaluation or success rate',
                      neural_policy_used=False, world_model_used=False, GPU_used=False,
                      task_processes_isolated=True,
                      config_sha256=file_sha256(CONFIG),
                      executor_sha256=file_sha256(ROOT/'our_mani/ablations/physmani_executor_v4.py'),
                      simulator_actuator_sha256=file_sha256(ROOT/'our_mani/aligned/simulation.py'),
                      results=reports)
        write(OUTPUT/'report.json', report)
        print(json.dumps(report, ensure_ascii=False), flush=True)
        if not report['passed']:
            raise RuntimeError('native executor protocol smoke failed; inspect report')
        return
    config = json.loads(CONFIG.read_text())
    rows = json.loads((ROOT/'artifacts/framework/protocol/train_manifest.json').read_text())['episodes']
    row = next(row for row in rows if row['task'] == args.task)
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    env = None
    report = dict(task=args.task, source=row['raw_episode_id'], split='train',
                  formal_test_used=False, passed=False, cases=[])
    try:
        env = AlignedLocalSimulation(args.task, row['path'], protocol=config,
                                     seed=config['seed'], servo_lead=1.)
        report['cases'].append(check_case(env, row, config, points=100))
        report['cases'].append(check_case(env, row, config, points=8, stale_target=True))
        report['passed'] = True
    except Exception as exc:
        report['error'] = repr(exc)
        raise
    finally:
        write(OUTPUT/(args.task+'.json'), report)
        if env:
            env.close()


if __name__ == '__main__':
    main()
