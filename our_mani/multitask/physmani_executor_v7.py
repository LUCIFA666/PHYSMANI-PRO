"""R10 terminal-aware version of the released V6 actuator.

The arm loop is a minimal copy of V4: its two success-only stop predicates
become native-terminal predicates. The multiple-inheritance order routes V5's
arm call through this loop while V6 retains its owned direct-setpoint scope,
actual-speed receipts, author RGB transforms and unchanged neural/WM methods.
No frozen class or method is patched by this module.
"""
import numpy as np

from ..ablations import physmani_executor_v4 as arm_executor
from ..sensor_alignment.physmani_executor_v6 import (
    PhysManiAdapter as V6Adapter, REFERENCE_FILTER,
    ProtocolInvariantError, InfrastructureExecutionError, raise_protected_error,
)
from .protocol import EpisodeTerminatedError

physics_time = arm_executor.physics_time
_check_clock = arm_executor._check_clock
modify_released_path = arm_executor.modify_released_path
EXECUTOR = 'released_waypoints_counted_actuator_author_rgbd_terminal_v7'


def _command_with_clipping_receipt(*args, **kwargs):
    # V6 binds this frozen module's helper only inside its owned RLock scope.
    return arm_executor._command_with_clipping_receipt(*args, **kwargs)


class TerminalArmAdapter(arm_executor.PhysManiAdapter):
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
                if env.ledger.remaining < env.ledger.dt - 1e-8 or env._native_terminal_latched:
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
                if env._native_terminal_latched:
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


class PhysManiAdapter(V6Adapter, TerminalArmAdapter):
    def execute(self, env, action, *, observer=None):
        if env._native_terminal_latched:
            raise EpisodeTerminatedError('native terminal episode cannot accept another baseline decision')
        env._before_action()
        receipt = super().execute(env, action, observer=observer)
        receipt['executor'] = EXECUTOR
        receipt['inherited_arm_executor'] = 'v4 two terminal-stop predicates; v5 native RGB; v6 direct RML targets'
        receipt['native_terminal_latched'] = env._native_terminal_latched
        receipt['native_terminal_failure_latched'] = env._native_failure_latched
        receipt['native_success_latched'] = env._native_success_latched
        receipt['partial_decision_due_to_native_terminal'] = (
            env._native_terminal_latched and receipt['ticks'] < self.config['baseline_decision_ticks'])
        return receipt
