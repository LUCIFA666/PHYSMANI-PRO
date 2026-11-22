"""TRAIN fixed-action paired execution diagnostic; no WM, policy inference or GPU.

Each arm command style runs in its own process from the same TRAIN reset.
Only this owned probe process temporarily replaces the v4 command helper;
frozen modules and other processes are never modified.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import subprocess
import sys
from types import MethodType

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'scripts')]
import numpy as np
from scipy.spatial.transform import Rotation

from collect_local_rollouts import initialization
from our_mani.ablations import physmani_executor_v4 as arm_executor
from our_mani.aligned.simulation import physics_time
from our_mani.sensor_alignment.author_rgbd import AlignedLocalSimulation, public_initial_conditions
from our_mani.sensor_alignment.physmani_executor_v5 import PhysManiAdapter
from our_mani.benchmark_protocol import file_sha256

OUTPUT = ROOT/'artifacts/framework/aligned_r07/execution_sensitivity_probe'
CONFIG = ROOT/'config/framework/aligned_protocol_r07.json'
ACTION_SOURCE = ROOT/'artifacts/framework/aligned_r07/smoke_physmani_v5/pick_moving_target_on_the_table__episode0.json'
STYLES = ('common_setpoint_clip', 'author_direct_target')


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def direct_command(self, joints, *, opening=None, phase='arm'):
    """Same counted/gripper actuator, omitting only current-q velocity clipping."""
    self.ledger.before()
    current = np.asarray(self.arm.get_joint_positions())
    requested = np.asarray(joints, dtype=float)
    if requested.shape != current.shape or not np.isfinite(requested).all():
        raise arm_executor.ProtocolInvariantError('invalid diagnostic joint command')
    bounded = np.clip(requested, self.robot_model.lower, self.robot_model.upper)
    self.arm.set_joint_target_positions(bounded.tolist())
    if opening is not None:
        requested_opening = float(opening >= .5)
        if requested_opening != self._requested_opening:
            if requested_opening:
                self.gripper.release()
                self.ledger.release_requests += 1
            else:
                for obj in self.task.get_graspable_objects():
                    self.gripper.grasp(obj)
                self.ledger.grasp_requests += 1
            self._requested_opening = requested_opening
        self.gripper.actuate(requested_opening, velocity=self.protocol['gripper_velocity'])
    self._tick(phase)
    return True


def actual_speed_metrics(ticks, limits):
    finite_difference = np.asarray([tick['actual_q_delta_over_dt_rad_per_s'] for tick in ticks])
    native = np.asarray([tick['actual_native_qdot_rad_per_s'] for tick in ticks])
    fractions_fd = np.abs(finite_difference)/limits
    fractions_native = np.abs(native)/limits
    return dict(max_actual_50ms_average_speed_fraction=float(np.max(fractions_fd)),
                max_actual_native_reported_speed_fraction=float(np.max(fractions_native)),
                ticks_average_speed_over_nominal=sum(bool(np.any(values > 1.+1e-6)) for values in fractions_fd),
                ticks_native_speed_over_nominal=sum(bool(np.any(values > 1.+1e-6)) for values in fractions_native),
                ticks_average_speed_over_nominal_plus_1percent=sum(bool(np.any(values > 1.01)) for values in fractions_fd),
                ticks_native_speed_over_nominal_plus_1percent=sum(bool(np.any(values > 1.01)) for values in fractions_native),
                measurement_scope='actual native qdot and q differences at 50ms ticks; no unobserved substep peak claim')


def run_style(style):
    config = json.loads(CONFIG.read_text())
    source = json.loads(ACTION_SOURCE.read_text())
    assert source['split'] == 'train' and source['formal_test_used'] is False
    action = np.asarray(source['steps'][0]['action'], dtype=float)
    rows = json.loads((ROOT/'artifacts/framework/protocol/train_manifest.json').read_text())['episodes']
    row = next(row for row in rows if row['task'] == source['task'] and row['raw_episode_id'] == source['raw_episode_id'])
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    report = dict(style=style, passed=False, split='train', formal_test_used=False,
                  neural_policy_used=False, world_model_used=False, GPU_used=False,
                  task=row['task'], source=row['raw_episode_id'], fixed_action=action.tolist(),
                  seed=config['seed'], action_source_sha256=file_sha256(ACTION_SOURCE),
                  config_sha256=file_sha256(CONFIG), ticks=[], paths=[], decisions=[])
    env = None
    original_helper = arm_executor._command_with_clipping_receipt
    original_modify = arm_executor.modify_released_path
    decision_index = -1
    try:
        env = AlignedLocalSimulation(row['task'], row['path'], protocol=config,
                                     seed=config['seed'], servo_lead=1.)
        x, velocity, acceleration, goal = initialization(row['path'])
        assert env.reset(target_position=x, target_velocity=velocity,
                         target_acceleration=acceleration, goal_position=goal)
        env.finish_home_reset()
        observation, robot = env.observe()
        report['public_initial_conditions'] = public_initial_conditions(env, observation, robot)
        limits = env.robot_model.velocity_limits.copy()
        report['native_joint_upper_velocity_limits_rad_per_s'] = limits.tolist()
        initial_eef = np.asarray(env.arm.get_tip().get_pose())
        report['initial_eef_pose'] = initial_eef.tolist()
        if style == 'author_direct_target':
            env.command_joints = MethodType(direct_command, env)

        def modified_path(path):
            clock = physics_time()
            initial_q = np.asarray(env.arm.get_joint_positions())
            raw = np.asarray(path._path_points).reshape(-1, env.arm.get_joint_count()).copy()
            receipt = original_modify(path)
            selected = np.asarray(path._path_points).reshape(-1, env.arm.get_joint_count()).copy()
            assert physics_time() == clock
            first = selected[0] if len(selected) else initial_q
            report['paths'].append(dict(decision=decision_index, initial_actual_q_rad=initial_q.tolist(),
                                         raw_joint_waypoints=raw.tolist(), selected_joint_waypoints=selected.tolist(),
                                         first_truncated_waypoint_jump_rad=(first-initial_q).tolist(),
                                         max_first_truncated_waypoint_jump_rad=float(np.max(np.abs(first-initial_q))),
                                         first_jump_as_one_tick_native_speed_fraction=float(np.max(np.abs(first-initial_q)/(limits*env.ledger.dt))),
                                         no_physics_advance=True, modification=receipt))
            return receipt

        def record_command(current_env, joints, *, opening, phase, receipt):
            before_q = np.asarray(current_env.arm.get_joint_positions())
            before_native_qdot = np.asarray(current_env.arm.get_joint_velocities())
            before_eef = np.asarray(current_env.arm.get_tip().get_pose())
            before_clock = physics_time()
            before_task = float(current_env.task.t)
            reference = np.asarray(joints, dtype=float).copy()
            would_clip = before_q + np.clip(reference-before_q, -limits*current_env.ledger.dt,
                                           limits*current_env.ledger.dt)
            would_clip = np.clip(would_clip, current_env.robot_model.lower, current_env.robot_model.upper)
            if style == 'common_setpoint_clip':
                original_helper(current_env, joints, opening=opening, phase=phase, receipt=receipt)
            else:
                bounded = np.clip(reference, current_env.robot_model.lower, current_env.robot_model.upper)
                current_env.command_joints(joints, opening=opening, phase=phase)
                if not np.allclose(current_env.arm.get_joint_target_positions(), bounded, rtol=0, atol=1e-6):
                    raise arm_executor.ProtocolInvariantError('direct diagnostic target verification failed')
                delta = float(np.max(np.abs(reference-bounded)))
                receipt['command_ticks'] += 1
                receipt['position_clip_ticks'] += int(delta > 1e-12)
                receipt['any_clip_ticks'] += int(delta > 1e-12)
                receipt['maximum_position_clip_rad'] = max(receipt['maximum_position_clip_rad'], delta)
                receipt['maximum_requested_minus_applied_rad'] = max(receipt['maximum_requested_minus_applied_rad'], delta)
            after_q = np.asarray(current_env.arm.get_joint_positions())
            after_eef = np.asarray(current_env.arm.get_tip().get_pose())
            tick = dict(index=len(report['ticks']), decision=decision_index,
                        timestamp=current_env.time, phase=phase, opening=float(opening),
                        before_actual_q_rad=before_q.tolist(), actual_q_rad=after_q.tolist(),
                        before_actual_native_qdot_rad_per_s=before_native_qdot.tolist(),
                        actual_native_qdot_rad_per_s=current_env.arm.get_joint_velocities(),
                        actual_q_delta_over_dt_rad_per_s=((after_q-before_q)/current_env.ledger.dt).tolist(),
                        raw_rml_reference_q_rad=reference.tolist(),
                        applied_servo_target_q_rad=current_env.arm.get_joint_target_positions(),
                        reference_minus_actual_before_rad=(reference-before_q).tolist(),
                        common_clip_counterfactual_delta_rad=(reference-would_clip).tolist(),
                        before_eef_pose=before_eef.tolist(), actual_eef_pose=after_eef.tolist(),
                        position_error_to_fixed_action_m=float(np.linalg.norm(after_eef[:3]-action[:3])),
                        eef_tick_displacement_m=float(np.linalg.norm(after_eef[:3]-before_eef[:3])),
                        physics_delta_seconds=physics_time()-before_clock,
                        task_delta_seconds=float(current_env.task.t)-before_task)
            assert abs(tick['physics_delta_seconds']-current_env.ledger.dt) < 2e-5
            assert abs(tick['task_delta_seconds']-current_env.ledger.dt) < 2e-5
            report['ticks'].append(tick)

        arm_executor.modify_released_path = modified_path
        arm_executor._command_with_clipping_receipt = record_command
        adapter = object.__new__(PhysManiAdapter)
        adapter.config = config
        adapter.observe = lambda observation, robot: None
        for decision_index in range(2):
            execution = adapter.execute(env, action)
            assert execution['ticks'] == config['baseline_decision_ticks'] == 10
            assert execution['path_found'] and execution['arm_error'] is None
            report['decisions'].append(dict(decision=decision_index, execution=execution,
                                           endpoint_eef_pose=env.arm.get_tip().get_pose().tolist(),
                                           endpoint_position_error_m=report['ticks'][-1]['position_error_to_fixed_action_m']))
        report['alignment_receipt'] = env.alignment_receipt()
        assert report['alignment_receipt']['clock']['ticks'] == 20
        assert report['alignment_receipt']['clock']['phases'] == {'physmani_path': 18, 'physmani_gripper': 2}
        assert abs(report['alignment_receipt']['actual_physics_seconds']-1.) < 2e-5
        assert abs(report['alignment_receipt']['actual_task_seconds']-1.) < 2e-5
        report['actual_speed_metrics'] = actual_speed_metrics(report['ticks'], limits)
        report['endpoint_error_05s_m'] = report['decisions'][0]['endpoint_position_error_m']
        report['endpoint_error_10s_m'] = report['decisions'][1]['endpoint_position_error_m']
        report['eef_net_displacement_10s_m'] = float(np.linalg.norm(
            np.asarray(report['ticks'][-1]['actual_eef_pose'])[:3]-initial_eef[:3]))
        report['reference_speed_note'] = 'Reference jumps are recorded separately; all reported actual-speed fractions use native state measurements.'
        report['passed'] = True
    except Exception as exc:
        report['error'] = repr(exc)
        raise
    finally:
        arm_executor._command_with_clipping_receipt = original_helper
        arm_executor.modify_released_path = original_modify
        write(OUTPUT/(style+'.json'), report)
        if env:
            env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--style', choices=STYLES)
    parser.add_argument('--aggregate-existing', action='store_true',
                        help='reuse completed owned probe executions; do not rerun physics')
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if args.style:
        run_style(args.style)
        return
    statuses = [0, 0] if args.aggregate_existing else [
        subprocess.run([sys.executable, str(Path(__file__).resolve()), '--style', style]).returncode
        for style in STYLES]
    results = [json.loads((OUTPUT/(style+'.json')).read_text()) for style in STYLES]
    tolerances_path = ROOT/'artifacts/framework/aligned_r07/acceptance_tolerances.json'
    tolerances = json.loads(tolerances_path.read_text())
    first, second = [result['public_initial_conditions'] for result in results]
    first_pose, second_pose = np.asarray(first['robot_pose']), np.asarray(second['robot_pose'])
    joint_delta = float(np.max(np.abs(np.asarray(first['robot_joints'])-second['robot_joints'])))
    position_delta = float(np.linalg.norm(first_pose[:3]-second_pose[:3]))
    orientation_delta = float((Rotation.from_quat(first_pose[3:]).inv()*Rotation.from_quat(second_pose[3:])).magnitude())
    same_initial = (joint_delta <= tolerances['robot_joints_rad'] and
                    position_delta <= tolerances['robot_position_m'] and
                    orientation_delta <= tolerances['robot_orientation_rad'])
    report = dict(created_utc=datetime.now(timezone.utc).isoformat(),
                  passed=all(status == 0 for status in statuses) and all(result['passed'] for result in results) and same_initial,
                  split='train', formal_test_used=False, neural_policy_used=False, world_model_used=False, GPU_used=False,
                  process_isolated_styles=True, initial_robot_states_match=same_initial,
                  initial_state_parity=dict(maximum_joint_delta_rad=joint_delta,
                                            eef_position_delta_m=position_delta,
                                            eef_orientation_delta_rad=orientation_delta,
                                            preregistered_tolerances=tolerances,
                                            tolerance_sha256=file_sha256(tolerances_path),
                                            joints_bitwise_identical=first['robot_joints']==second['robot_joints'],
                                            eef_pose_bitwise_identical=first['robot_pose']==second['robot_pose']),
                  script_sha256=file_sha256(Path(__file__).resolve()),
                  source_action_sha256=file_sha256(ACTION_SOURCE), config_sha256=file_sha256(CONFIG),
                  frozen_core_sha256={name:file_sha256(ROOT/name) for name in (
                      'our_mani/ablations/physmani_executor_v4.py', 'our_mani/aligned/simulation.py',
                      'our_mani/sensor_alignment/author_rgbd.py', 'our_mani/sensor_alignment/physmani_executor_v5.py')},
                  scope='two fixed TRAIN actions, execution sensitivity only; no policy success-rate or TEST tuning claim',
                  direct_style='released arm targets with shared position bounds, counted ticks and incremental gripper; no hidden physics restored',
                  results=results)
    write(OUTPUT/'report.json', report)
    print(json.dumps(dict(passed=report['passed'], initial_robot_states_match=same_initial,
                         results=[{key:result.get(key) for key in ('style','endpoint_error_05s_m','endpoint_error_10s_m',
                                                                  'eef_net_displacement_10s_m','actual_speed_metrics','error')}
                                  for result in results])), flush=True)
    if not report['passed']:
        raise RuntimeError('paired execution diagnostic failed; inspect report')


if __name__ == '__main__':
    main()
