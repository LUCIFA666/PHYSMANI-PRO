"""Released PhysMani arm rules with the common counted r06 actuator.

Keep the author's waypoint selection, servo-target reset and recoverable arm
failure behavior. Hidden physics steps remain removed. Common joint command
limits and incremental gripper actuation remain active and are disclosed.
"""
import numpy as np

from ..aligned.physmani import PhysManiAdapter as OriginalAdapter
from ..aligned.simulation import physics_time


class ProtocolInvariantError(Exception):
    """Protocol/state corruption must never be scored as a policy failure."""


class InfrastructureExecutionError(Exception):
    """Runtime infrastructure errors must never be scored as arm rejection."""


def raise_protected_error(exc):
    """Let known infrastructure and integrity failures escape arm recovery."""
    if isinstance(exc, (ProtocolInvariantError, InfrastructureExecutionError)):
        raise exc
    message = str(exc).lower()
    if isinstance(exc, (OSError, MemoryError)) or any(token in message for token in (
            'out of memory', 'illegal memory', 'device-side assert', 'cuda error',
            'compilation', 'driver error', 'cublas', 'cudnn', 'connection lost',
            'not connected', 'invalid handle', 'invalid object handle',
            'object does not exist', 'simulator unavailable')):
        raise InfrastructureExecutionError(str(exc)) from exc
    if any(token in message for token in (
            'advanced physics', 'unaccounted physics', 'clocks diverged',
            'shared tick', 'restore robot', 'restore servo', 'state invariant',
            'aligned physics budget exhausted')):
        raise ProtocolInvariantError(str(exc)) from exc


def _check_clock(clock, operation):
    if abs(physics_time() - clock) > 1e-6:
        raise ProtocolInvariantError(operation + ' advanced physics')


def path_poses_without_physics(path):
    """Extract FK points; restore q and reset targets to q as the author does."""
    arm = path._arm
    clock = physics_time()
    joints_before = arm.get_joint_positions()
    ik_before = arm._ik_target.get_pose()
    joint_waypoints = np.asarray(path._path_points, dtype=float).reshape(
        -1, arm.get_joint_count())[1:].copy()
    poses = []
    try:
        for joints in joint_waypoints:
            arm.set_joint_positions(joints.tolist())
            arm.set_joint_target_positions(joints.tolist())
            poses.append(arm.get_tip().get_pose())
    finally:
        try:
            arm.set_joint_positions(joints_before)
            # Released get_action_traj resets both joints and servo targets to
            # init_angles, including when no usable modified path remains.
            arm.set_joint_target_positions(joints_before)
            arm._ik_target.set_pose(ik_before)
        except Exception as exc:
            raise_protected_error(exc)
            raise ProtocolInvariantError('waypoint extraction failed to restore robot state') from exc
        _check_clock(clock, 'waypoint extraction')
        if not np.allclose(arm.get_joint_positions(), joints_before, rtol=0, atol=1e-6):
            raise ProtocolInvariantError('waypoint extraction did not restore robot joints')
        if not np.allclose(arm.get_joint_target_positions(), joints_before, rtol=0, atol=1e-6):
            raise ProtocolInvariantError('waypoint extraction did not restore servo targets')
        if not np.allclose(arm._ik_target.get_pose(), ik_before, rtol=0, atol=1e-6):
            raise ProtocolInvariantError('waypoint extraction did not restore robot IK target')
    return np.asarray(poses, dtype=float).reshape(-1, 7), joint_waypoints


def released_waypoint_indices(poses, *, discard=10, min_dist=.01):
    selected = []
    for index in range(discard, len(poses)):
        if not selected or np.linalg.norm(poses[index, :3] - poses[selected[-1], :3]) > min_dist:
            selected.append(index)
    return selected


def modify_released_path(path):
    poses, joints = path_poses_without_physics(path)
    selected = released_waypoint_indices(poses)
    path._path_points = joints[selected].reshape(-1)
    return dict(raw_joint_waypoints=len(joints) + 1, extracted_waypoints=len(joints),
                discarded_initial_waypoints=10, cartesian_spacing_m=.01,
                selected_waypoints=len(selected), selected_extracted_indices=selected,
                extraction_advance_physics=False, servo_targets_reset_to_actual_joints=True)


def _command_with_clipping_receipt(env, joints, *, opening, phase, receipt):
    current = np.asarray(env.arm.get_joint_positions(), dtype=float)
    requested = np.asarray(joints, dtype=float)
    if requested.shape != current.shape or not np.isfinite(requested).all():
        raise ProtocolInvariantError('invalid RML joint-command state invariant')
    velocity_bounded = current + np.clip(
        requested - current, -env.robot_model.velocity_limits * env.ledger.dt,
        env.robot_model.velocity_limits * env.ledger.dt)
    bounded = np.clip(velocity_bounded, env.robot_model.lower, env.robot_model.upper)
    velocity_delta = float(np.max(np.abs(requested - velocity_bounded)))
    position_delta = float(np.max(np.abs(velocity_bounded - bounded)))
    total_delta = float(np.max(np.abs(requested - bounded)))
    try:
        env.command_joints(joints, opening=opening, phase=phase)
    except Exception as exc:
        raise_protected_error(exc)
        raise ProtocolInvariantError('common actuator failed to apply a valid joint command') from exc
    if not np.allclose(env.arm.get_joint_target_positions(), bounded, rtol=0, atol=1e-6):
        raise ProtocolInvariantError('common actuator target state invariant failed')
    receipt['command_ticks'] += 1
    receipt['velocity_clip_ticks'] += int(velocity_delta > 1e-12)
    receipt['position_clip_ticks'] += int(position_delta > 1e-12)
    receipt['any_clip_ticks'] += int(total_delta > 1e-12)
    receipt['maximum_velocity_clip_rad'] = max(receipt['maximum_velocity_clip_rad'], velocity_delta)
    receipt['maximum_position_clip_rad'] = max(receipt['maximum_position_clip_rad'], position_delta)
    receipt['maximum_requested_minus_applied_rad'] = max(
        receipt['maximum_requested_minus_applied_rad'], total_delta)


class PhysManiAdapter(OriginalAdapter):
    def execute(self, env, action, *, observer=None):
        from pyrep.const import ConfigurationPathAlgorithms as Algos
        from pyrep.errors import ConfigurationPathError, IKError
        recoverable_arm_errors = (ConfigurationPathError, IKError, RuntimeError,
                                  ValueError, IndexError, AssertionError)
        action = np.asarray(action, dtype=float)
        if action.shape != (8,) or not np.isfinite(action).all():
            raise ValueError('invalid PhysMani action')
        clock = physics_time()
        path = None
        error = None
        modification = None
        # Same predicate as released assert_unit_quaternion (NumPy defaults).
        if not np.isclose(np.linalg.norm(action[3:7]), 1.0):
            error = 'nonunit_action_quaternion'
        elif not env.scene.check_target_in_workspace(action[:3]):
            error = 'outside_workspace'
        else:
            try:
                path = env.map_service.plan_path(
                    action[:3].tolist(), quaternion=action[3:7].tolist(),
                    ignore_collisions=True, trials=100, max_configs=10,
                    max_time_ms=10, trials_per_goal=5, algorithm=Algos.RRTConnect)
                modification = modify_released_path(path)
                if len(path._path_points) < 2 * env.arm.get_joint_count():
                    error = 'released_modified_path_too_short'
                    path = None
            except recoverable_arm_errors as exc:
                raise_protected_error(exc)
                error = repr(exc)
                path = None
        _check_clock(clock, 'baseline planning/extraction')
        done = False
        arm_error = None
        last_target = np.asarray(env.arm.get_joint_target_positions())
        ticks = int(self.config['baseline_decision_ticks'])
        executed = 0
        clipping = dict(command_ticks=0, velocity_clip_ticks=0, position_clip_ticks=0,
                        any_clip_ticks=0, maximum_velocity_clip_rad=0.,
                        maximum_position_clip_rad=0., maximum_requested_minus_applied_rad=0.)
        try:
            for tick in range(ticks):
                if env.ledger.remaining < env.ledger.dt - 1e-8 or env._native_success_latched:
                    break
                if tick < ticks - 1 and path is not None and not done and arm_error is None:
                    arm_clock = physics_time()
                    try:
                        done = path.step()
                        if path._joint_position_action is not None:
                            last_target = np.asarray(path._joint_position_action)
                    except recoverable_arm_errors as exc:
                        raise_protected_error(exc)
                        arm_error = repr(exc)
                        # An arm failure rests at the last commanded target;
                        # the final tick still applies the new gripper action.
                        last_target = np.asarray(env.arm.get_joint_target_positions())
                    _check_clock(arm_clock, 'RML arm target generation')
                opening = float(np.round(action[7])) if tick == ticks - 1 else env._requested_opening
                _command_with_clipping_receipt(
                    env, last_target, opening=opening,
                    phase='physmani_gripper' if tick == ticks - 1 else 'physmani_path',
                    receipt=clipping)
                executed += 1
                try:
                    observation, robot = env.observe()
                except Exception as exc:
                    raise_protected_error(exc)
                    raise InfrastructureExecutionError('common sensor observation failed') from exc
                if observer:
                    observer(observation, robot)
                if env._native_success_latched:
                    break
                self.observe(observation, robot)
        finally:
            if path is not None and path._rml_handle is not None and not path._path_done:
                from pyrep.backend import sim
                try:
                    sim.simRMLRemove(path._rml_handle)
                except Exception as exc:
                    raise InfrastructureExecutionError('RML handle cleanup failed') from exc
                path._rml_handle = None
        return dict(path_found=path is not None, error=error, arm_error=arm_error,
                    ignore_collisions=True, waypoint_modification=modification,
                    executor='released_waypoints_no_hidden_physics_v4', ticks=executed,
                    common_joint_command_clipping=clipping,
                    aligned_execution_differences=[
                        'all physical steps counted against common task budget',
                        'incremental common gripper actuation',
                        'common joint velocity and position command limits'])
