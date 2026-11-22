"""Fixed TRAIN native v6 smoke; no neural policy, WM, GPU or TEST tuning."""
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'scripts')]
import numpy as np

from collect_local_rollouts import initialization
from our_mani.ablations import physmani_executor_v4 as arm_executor
from our_mani.sensor_alignment.author_rgbd import AlignedLocalSimulation, public_initial_conditions
from our_mani.sensor_alignment.physmani_executor_v5 import PhysManiAdapter as V5Adapter
from our_mani.sensor_alignment.physmani_executor_v6 import PhysManiAdapter, ProtocolInvariantError
from our_mani.benchmark_protocol import file_sha256

OUTPUT = ROOT/'artifacts/framework/aligned_r07/baseline_executor_v6_smoke'
CONFIG = ROOT/'config/framework/aligned_protocol_r07_baseline_v6.json'
SOURCE = ROOT/'artifacts/framework/aligned_r07/smoke_physmani_v5/pick_moving_target_on_the_table__episode0.json'


def main():
    config = json.loads(CONFIG.read_text())
    common_config = json.loads((ROOT/'config/framework/aligned_protocol_r07.json').read_text())
    different = {key for key in set(config)|set(common_config) if config.get(key) != common_config.get(key)}
    assert different <= {'baseline_executor','baseline_reference_filter','baseline_reference_filter_scope'}
    source = json.loads(SOURCE.read_text())
    assert source['split'] == 'train' and not source['formal_test_used']
    action = np.asarray(source['steps'][0]['action'], dtype=float)
    rows = json.loads((ROOT/'artifacts/framework/protocol/train_manifest.json').read_text())['episodes']
    row = next(row for row in rows if row['task'] == source['task'] and row['raw_episode_id'] == source['raw_episode_id'])
    random.seed(config['seed']);np.random.seed(config['seed'])
    report = dict(created_utc=datetime.now(timezone.utc).isoformat(), passed=False,
                  split='train', formal_test_used=False, neural_policy_used=False,
                  world_model_used=False, GPU_used=False, task=row['task'], source=row['raw_episode_id'],
                  scope='fixed TRAIN native execution protocol/speed measurements; not policy success rate',
                  config_sha256=file_sha256(CONFIG), source_action_sha256=file_sha256(SOURCE),
                  executor_sha256=file_sha256(ROOT/'our_mani/sensor_alignment/physmani_executor_v6.py'),
                  configuration_difference_keys=sorted(different), fixed_action=action.tolist(), decisions=[], ticks=[])
    env = None
    try:
        env = AlignedLocalSimulation(row['task'], row['path'], protocol=config, seed=config['seed'], servo_lead=1.)
        x, velocity, acceleration, goal = initialization(row['path'])
        assert env.reset(target_position=x, target_velocity=velocity, target_acceleration=acceleration, goal_position=goal)
        env.finish_home_reset()
        observation, robot = env.observe()
        report['public_initial_conditions'] = public_initial_conditions(env, observation, robot)
        initial_q = np.asarray(env.arm.get_joint_positions())
        native_limits = env.robot_model.velocity_limits.copy()
        report['native_joint_upper_velocity_limits_rad_per_s'] = native_limits.tolist()
        adapter = object.__new__(PhysManiAdapter)
        adapter.config = config
        adapter.observe = lambda observation, robot: None
        original_helper = arm_executor._command_with_clipping_receipt
        def observer(observation, robot):
            targets = np.asarray(env.arm.get_joint_target_positions())
            assert np.all(targets >= env.robot_model.lower-1e-6) and np.all(targets <= env.robot_model.upper+1e-6)
            report['ticks'].append(dict(timestamp=env.time, actual_q_rad=env.arm.get_joint_positions(),
                                         actual_native_qdot_rad_per_s=env.arm.get_joint_velocities(),
                                         applied_target_q_rad=targets.tolist(), actual_eef_pose=robot.pose.xyzw().tolist()))
        for index in range(2):
            execution = adapter.execute(env, action, observer=observer)
            assert arm_executor._command_with_clipping_receipt is original_helper
            assert execution['ticks'] == 10
            assert execution['reference_filter'] == 'author_direct_setpoint'
            assert execution['executor'] == config['baseline_executor']
            assert execution['common_joint_command_clipping']['velocity_clip_ticks'] == 0
            assert execution['actual_speed_receipt']['command_ticks'] == 10
            report['decisions'].append(execution)
        report['alignment_receipt'] = env.alignment_receipt()
        clock = report['alignment_receipt']['clock']
        assert clock['ticks'] == 20 and clock['phases'] == {'physmani_path':18,'physmani_gripper':2}
        assert abs(report['alignment_receipt']['actual_physics_seconds']-1.) < 2e-5
        assert abs(report['alignment_receipt']['actual_task_seconds']-1.) < 2e-5
        actual_q = np.asarray([initial_q, *[tick['actual_q_rad'] for tick in report['ticks']]])
        measured_average_max = float(np.max(np.abs(np.diff(actual_q, axis=0)/config['dt'])/native_limits))
        measured_native_max = float(np.max(np.abs(np.asarray([tick['actual_native_qdot_rad_per_s'] for tick in report['ticks']]))/native_limits))
        assert np.isclose(measured_average_max, max(execution['actual_speed_receipt']['maximum_actual_average_speed_fraction'] for execution in report['decisions']), rtol=0, atol=1e-9)
        assert np.isclose(measured_native_max, max(execution['actual_speed_receipt']['maximum_actual_native_reported_speed_fraction'] for execution in report['decisions']), rtol=0, atol=1e-9)
        report['independent_actual_average_speed_max_fraction'] = measured_average_max
        report['independent_actual_native_speed_max_fraction'] = measured_native_max
        report['actual_speed_receipts_independently_verified'] = True
        original_parent_execute = V5Adapter.execute
        def fail_parent(*args, **kwargs):
            raise ProtocolInvariantError('owned scope restoration probe')
        V5Adapter.execute = fail_parent
        try:
            try:
                adapter.execute(env, action)
            except ProtocolInvariantError:
                pass
            else:
                raise AssertionError('expected scope restoration probe error')
            assert arm_executor._command_with_clipping_receipt is original_helper
        finally:
            V5Adapter.execute = original_parent_execute
        report['reference_binding_restored_on_normal_and_error_exit'] = True
        report['passed'] = True
    except Exception as exc:
        report['error'] = repr(exc)
        raise
    finally:
        OUTPUT.mkdir(parents=True, exist_ok=True)
        (OUTPUT/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
        if env:
            env.close()
    print(json.dumps({key:report[key] for key in ('passed','independent_actual_average_speed_max_fraction',
                                                'independent_actual_native_speed_max_fraction',
                                                'reference_binding_restored_on_normal_and_error_exit')}), flush=True)


if __name__ == '__main__':
    main()
