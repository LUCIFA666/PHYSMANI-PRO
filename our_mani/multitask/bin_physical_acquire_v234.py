"""Use the original physical grasp gate and actual close acknowledgement for bins.

This explicitly replaces v68's extra fitted-regression intersection. Fitting
accuracy and physical capture are separate requirements; neither native physics
nor the final task success condition is changed.
"""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from types import MethodType
from .evaluation import file_sha
from .interaction import GripperTimingConfig
from .cup_physical_acquire_v167 import install_confirmation, geometry_passes

TASKS = ('put_rubbish_in_moving_bin', 'put_rubbish_in_moving_bin_high_speed')


def binding(root, protocol, task):
    if task not in TASKS:
        raise ValueError('only the two bin tasks')
    previous = protocol['close_completion_gate_v68'][task]
    original = asdict(GripperTimingConfig(.02, .1))
    if previous['original_grasp_gate'] != original or protocol['gripper_timing_config'] != original:
        raise ValueError('original physical grasp contract differs')
    paths = ('bin_physical_acquire_v234.py', 'cup_physical_acquire_v167.py')
    return dict(schema='r10_bin_original_physical_capture_actual_ack_v234', task=task,
        sources_sha256={f'our_mani/multitask/{name}': file_sha(Path(__file__).with_name(name)) for name in paths},
        previous_extra_controller_gate=deepcopy(previous['effective_close_AND_stopped_capture_gate']),
        actual_physical_close_gate=original, extra_controller_intersection_relaxed=True,
        acquire_completion_policy_changed=True,
        acquire_completion='original SE3, elapsed fitted duration, dispatch-confirmed stopped close, unique public held identity',
        original_fitting_caps_unchanged=True, original_other_phase_completion_unchanged=True,
        original_collision_motion_stop_and_global10s_gates_unchanged=True,
        native_physics_grasp_and_task_success_unchanged=True, native_labels_supplied_to_policy=False,
        formal_VAL_TEST=False)


def install_completion(controller, witnesses, events):
    original = controller._phase_done
    def done(instance, state, target, *, public_geometry=False):
        goal = instance.goal
        if goal.template.phase != 'acquire':
            return original(state, target, public_geometry=public_geometry)
        witness = witnesses.get(goal.candidate_id)
        if (witness is None or witness['phase_index'] != instance.phase_index
                or witness['manipulated_ids'] != list(goal.manipulated_ids)
                or witness['confirmed_after'] > state.robot.timestamp+1e-8
                or state.robot.timestamp-goal.selected_at < goal.template.duration_seconds-1e-8
                or not instance._public_acquire_contact(state)):
            return False
        config = instance.gripper_timing_config
        passed = geometry_passes(state.robot, target,
            config.close_position_tolerance_m, config.close_orientation_tolerance_rad)
        if passed:
            events.append(dict(event='public_physical_acquire_completed', time=float(state.robot.timestamp),
                candidate_id=goal.candidate_id, native_task_success_claimed=False))
        return passed
    controller._phase_done = MethodType(done, controller)


def install(actor, protocol, root, task):
    expected = binding(root, protocol, task)
    if protocol.get('bin_physical_acquire_v234', {}).get(task) != expected:
        raise ValueError('freeze the explicit physical acquire policy')
    controller, manager = actor.controller, actor.continuous_reference_manager
    if (manager.controller is not controller or controller.goal is not None
            or controller.feedback_step_calls or manager.confirmations
            or asdict(controller.gripper_timing_config) != expected['previous_extra_controller_gate']):
        raise ValueError('install before first actual step on the expected final controller')
    controller.gripper_timing_config = GripperTimingConfig(**expected['actual_physical_close_gate'])
    witnesses, events = {}, []
    install_confirmation(manager, witnesses, events)
    install_completion(controller, witnesses, events)
    previous = actor.metrics_report
    def metrics(instance):
        return dict(previous(), bin_physical_acquire_v234=dict(binding=expected,
            actual_close_witnesses=deepcopy(witnesses), events=deepcopy(events),
            v68_extra_intersection_superseded_by_explicit_policy=True))
    actor.metrics_report = MethodType(metrics, actor)
