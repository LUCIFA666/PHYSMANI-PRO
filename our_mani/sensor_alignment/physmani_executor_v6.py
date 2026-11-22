"""Author arm reference commands, shared physical robot/counting/RGB-D.

The actual-q reference clip is our controller's algorithm, not a robot
hardware parameter. This baseline keeps the author's direct RML setpoints,
native joint position bounds, common counted ticks and incremental gripper.
The helper binding exists only in this adapter's owned execute scope.
"""
import threading

import numpy as np

from ..ablations import physmani_executor_v4 as arm_executor
from .physmani_executor_v5 import (
    PhysManiAdapter as V5Adapter, ProtocolInvariantError,
    InfrastructureExecutionError, raise_protected_error,
)
from .author_rgbd import SensorProtocolInvariantError


_reference_binding_lock = threading.RLock()
REFERENCE_FILTER = 'author_direct_setpoint'
EXECUTOR = 'released_waypoints_counted_actuator_author_rgbd_v6'


def _apply_direct_target(env, joints, *, opening, phase):
    """Preserve common physics/gripper; submit source-style arm reference."""
    env.ledger.before()
    requested = np.asarray(joints, dtype=float)
    current = np.asarray(env.arm.get_joint_positions())
    if requested.shape != current.shape or not np.isfinite(requested).all():
        raise ProtocolInvariantError('invalid direct arm reference')
    bounded = np.clip(requested, env.robot_model.lower, env.robot_model.upper)
    env.arm.set_joint_target_positions(bounded.tolist())
    if opening is not None:
        requested_opening = float(opening >= .5)
        if requested_opening != env._requested_opening:
            if requested_opening:
                env.gripper.release()
                env.ledger.release_requests += 1
            else:
                for obj in env.task.get_graspable_objects():
                    env.gripper.grasp(obj)
                env.ledger.grasp_requests += 1
            env._requested_opening = requested_opening
        env.gripper.actuate(requested_opening, velocity=env.protocol['gripper_velocity'])
    env._tick(phase)
    if not np.allclose(env.arm.get_joint_target_positions(), bounded, rtol=0, atol=1e-6):
        raise ProtocolInvariantError('direct arm target state invariant failed')
    return bounded


class PhysManiAdapter(V5Adapter):
    def execute(self, env, action, *, observer=None):
        limits = np.asarray(env.robot_model.velocity_limits, dtype=float)
        if not np.isfinite(limits).all() or np.any(limits <= 0):
            raise ProtocolInvariantError('invalid native joint velocity parameters')
        counterfactual = dict(command_ticks=0, velocity_clip_ticks=0, position_clip_ticks=0,
                              any_clip_ticks=0, maximum_velocity_clip_rad=0.,
                              maximum_position_clip_rad=0., maximum_requested_minus_applied_rad=0.)
        measured = dict(command_ticks=0, native_joint_upper_velocity_limits_rad_per_s=limits.tolist(),
                        maximum_actual_average_speed_fraction=0., maximum_actual_native_reported_speed_fraction=0.,
                        maximum_actual_average_speed_by_joint_rad_per_s=[0.]*len(limits),
                        maximum_actual_native_reported_speed_by_joint_rad_per_s=[0.]*len(limits),
                        ticks_average_speed_over_nominal=0, ticks_native_speed_over_nominal=0,
                        ticks_average_speed_over_nominal_plus_1percent=0,
                        ticks_native_speed_over_nominal_plus_1percent=0,
                        measurement='actual native qdot and actual q differences at counted 50ms ticks; no substep peak bound',
                        nominal_limit_note='native motor parameters are shared; small numerical/transient exceedances are measured, not suppressed or claimed impossible')

        def direct_command_receipt(current_env, joints, *, opening, phase, receipt):
            if current_env is not env:
                raise ProtocolInvariantError('direct arm helper escaped its owned adapter scope')
            before = np.asarray(env.arm.get_joint_positions(), dtype=float)
            requested = np.asarray(joints, dtype=float)
            if requested.shape != before.shape or not np.isfinite(requested).all():
                raise ProtocolInvariantError('invalid direct arm reference state')
            velocity_bounded = before + np.clip(requested-before, -limits*env.ledger.dt, limits*env.ledger.dt)
            full_bounded = np.clip(velocity_bounded, env.robot_model.lower, env.robot_model.upper)
            velocity_delta = float(np.max(np.abs(requested-velocity_bounded)))
            position_delta = float(np.max(np.abs(velocity_bounded-full_bounded)))
            total_delta = float(np.max(np.abs(requested-full_bounded)))
            try:
                applied = _apply_direct_target(env, joints, opening=opening, phase=phase)
                after = np.asarray(env.arm.get_joint_positions(), dtype=float)
                native = np.asarray(env.arm.get_joint_velocities(), dtype=float)
            except Exception as exc:
                raise_protected_error(exc)
                raise ProtocolInvariantError('direct counted arm/gripper application failed') from exc
            if not np.isfinite(after).all() or not np.isfinite(native).all():
                raise ProtocolInvariantError('nonfinite native actual joint state')
            applied_delta = float(np.max(np.abs(requested-applied)))
            receipt['command_ticks'] += 1
            receipt['position_clip_ticks'] += int(applied_delta > 1e-12)
            receipt['any_clip_ticks'] += int(applied_delta > 1e-12)
            receipt['maximum_position_clip_rad'] = max(receipt['maximum_position_clip_rad'], applied_delta)
            receipt['maximum_requested_minus_applied_rad'] = max(receipt['maximum_requested_minus_applied_rad'], applied_delta)
            counterfactual['command_ticks'] += 1
            counterfactual['velocity_clip_ticks'] += int(velocity_delta > 1e-12)
            counterfactual['position_clip_ticks'] += int(position_delta > 1e-12)
            counterfactual['any_clip_ticks'] += int(total_delta > 1e-12)
            counterfactual['maximum_velocity_clip_rad'] = max(counterfactual['maximum_velocity_clip_rad'], velocity_delta)
            counterfactual['maximum_position_clip_rad'] = max(counterfactual['maximum_position_clip_rad'], position_delta)
            counterfactual['maximum_requested_minus_applied_rad'] = max(counterfactual['maximum_requested_minus_applied_rad'], total_delta)
            average = (after-before)/env.ledger.dt
            average_fraction, native_fraction = np.abs(average)/limits, np.abs(native)/limits
            measured['command_ticks'] += 1
            measured['maximum_actual_average_speed_fraction'] = max(measured['maximum_actual_average_speed_fraction'], float(np.max(average_fraction)))
            measured['maximum_actual_native_reported_speed_fraction'] = max(measured['maximum_actual_native_reported_speed_fraction'], float(np.max(native_fraction)))
            measured['maximum_actual_average_speed_by_joint_rad_per_s'] = np.maximum(
                measured['maximum_actual_average_speed_by_joint_rad_per_s'], np.abs(average)).tolist()
            measured['maximum_actual_native_reported_speed_by_joint_rad_per_s'] = np.maximum(
                measured['maximum_actual_native_reported_speed_by_joint_rad_per_s'], np.abs(native)).tolist()
            measured['ticks_average_speed_over_nominal'] += int(np.any(average_fraction > 1.+1e-6))
            measured['ticks_native_speed_over_nominal'] += int(np.any(native_fraction > 1.+1e-6))
            measured['ticks_average_speed_over_nominal_plus_1percent'] += int(np.any(average_fraction > 1.01))
            measured['ticks_native_speed_over_nominal_plus_1percent'] += int(np.any(native_fraction > 1.01))

        with _reference_binding_lock:
            previous_helper = arm_executor._command_with_clipping_receipt
            arm_executor._command_with_clipping_receipt = direct_command_receipt
            try:
                receipt = super().execute(env, action, observer=observer)
            finally:
                arm_executor._command_with_clipping_receipt = previous_helper
        receipt['executor'] = EXECUTOR
        receipt['reference_filter'] = REFERENCE_FILTER
        receipt['inherited_arm_executor'] = 'v4 released waypoints/arm error recovery; v5 shared author RGB'
        receipt['actual_joint_command_filter'] = 'native joint position bounds only; direct RML targets'
        receipt['counterfactual_ours_reference_clip'] = counterfactual
        receipt['actual_speed_receipt'] = measured
        receipt['aligned_execution_differences'] = [
            'all physical steps counted against common task budget',
            'incremental common gripper actuation',
            'native joint position bounds retained; reference filtering follows each method controller']
        return receipt
