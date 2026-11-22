"""Cup capture uses original physical grasp geometry, with an actual close witness.

The fitted curve's regression accuracy is not a physical grasp condition. This
policy explicitly replaces the extra 5mm/.03rad intersection installed in v120
and v136. It does not change native physics, training acceptance or task success.
Transport completion uses the already confirmed first-prefix clock, while the
candidate's real selection time and trajectory remain untouched.
"""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from types import MethodType
import numpy as np
from scipy.spatial.transform import Rotation
from .interaction import GripperTimingConfig, COMMAND_GOAL, _world_target
from .evaluation import file_sha

TASKS = ('place_cups_on_rotating_frame', 'place_cups_on_rotating_frame_high_speed')
SCHEMA = 'r10_cup_original_physical_capture_and_confirmed_transport_completion_v167'


def binding(root, protocol, task):
    if task not in TASKS:
        raise ValueError('only the two cup placement tasks')
    key = 'normal_cup_close_intersection_v120' if task == TASKS[0] else 'high_cup_close_intersection_v136'
    prior = protocol[key]
    original = asdict(GripperTimingConfig(.02, .1))
    if prior['original'] != original or prior['actual'] != protocol['gripper_timing_config']:
        raise ValueError('original physical grasp and previous extra intersection must be explicit')
    return dict(schema=SCHEMA, task=task,
        sources_sha256={'our_mani/multitask/cup_physical_acquire_v167.py': file_sha(__file__)},
        previous_extra_controller_gate=deepcopy(prior['actual']),
        actual_physical_close_gate=original,
        extra_controller_intersection_relaxed=True,
        acquire_completion_policy_changed=True,
        acquire_completion='original SE3, elapsed fitted duration, dispatch-confirmed stopped close, unique public held identity',
        transport_completion_elapsed='first dispatch-confirmed transport prefix; actual selected_at unchanged',
        model_fitting_caps_unchanged=True, native_physics_grasp_success_unchanged=True,
        collision_motion_stop_global10s_gates_unchanged=True,
        native_labels_supplied_to_policy=False, formal_VAL_TEST=False)


def geometry_passes(robot, target, position_cap, orientation_cap):
    return bool(np.linalg.norm(robot.pose.position-target.position) <= position_cap
        and Rotation.from_matrix(robot.pose.rotation @ target.rotation.T).magnitude() <= orientation_cap)


def capture_proposal(manager, command, execution):
    """Read only the current public state before the original acknowledgement."""
    controller = manager.controller
    goal, state = controller.goal, manager.current_state
    if (goal is None or goal.phase != 'acquire' or state is None
            or command.get('opening', 1.) >= .5 or state.robot.contact
            or (manager.last_confirmed or {}).get('opening', 0.) < .5):
        return None
    now = float(state.robot.timestamp)
    issued = execution.get('actual_issued_at')
    closed = controller._close_edge_issued_at
    if (issued is None or abs(float(issued)-now) > 1e-8 or closed is None
            or abs(float(closed)-now) > 1e-8 or not manager.capture_stopped(state.robot)):
        return None
    progress = (now-goal.selected_at)/goal.template.duration_seconds
    if goal.template.at(float(np.clip(progress, 0., 1.)))[3] >= .5:
        return None
    try:
        current = controller.generator.observed_reference(state, goal.reference_id, goal.template, controller.origin)
        target = _world_target(current.pose, np.zeros(3), np.zeros(3), goal.template, 1.)[0]
    except (KeyError, ValueError, IndexError):
        return None
    config = controller.gripper_timing_config
    if not geometry_passes(state.robot, target, config.close_position_tolerance_m, config.close_orientation_tolerance_rad):
        return None
    return dict(candidate_id=goal.candidate_id, phase_index=controller.phase_index,
        manipulated_ids=list(goal.manipulated_ids), issued_at=now,
        close_position_error_m=float(np.linalg.norm(state.robot.pose.position-target.position)),
        close_orientation_error_rad=float(Rotation.from_matrix(state.robot.pose.rotation @ target.rotation.T).magnitude()),
        stopped_original_physical_SE3_close=True, public_state_only=True)


def install_confirmation(manager, witnesses, events):
    original = manager.confirm_actual_issued
    def confirm(instance, **kwargs):
        proposal = capture_proposal(instance, kwargs['command'], kwargs['execution'])
        result = original(**kwargs)
        execution = kwargs['execution']
        if (proposal is not None and not instance.qualification_failed
                and execution.get('actual_window_velocity_within_limits') is True
                and execution.get('actual_window_acceleration_within_limits') is True):
            proposal['confirmed_after'] = float(kwargs['time'])
            witnesses[proposal['candidate_id']] = proposal
            events.append(dict(event='original_physical_close_actually_dispatched', **deepcopy(proposal)))
        return result
    manager.confirm_actual_issued = MethodType(confirm, manager)


def install_completion(controller, clock, witnesses, events):
    original = controller._phase_done
    def done(instance, state, target, *, public_geometry=False):
        goal, robot = instance.goal, state.robot
        template = goal.template
        if template.reference_time_basis == COMMAND_GOAL and not public_geometry:
            return False
        if template.phase == 'acquire':
            witness = witnesses.get(goal.candidate_id)
            if (witness is None or witness['phase_index'] != instance.phase_index
                    or witness['manipulated_ids'] != list(goal.manipulated_ids)
                    or witness['confirmed_after'] > robot.timestamp+1e-8
                    or robot.timestamp-goal.selected_at < template.duration_seconds-1e-8
                    or not instance._public_acquire_contact(state)):
                return False
            config = instance.gripper_timing_config
            passed = geometry_passes(robot, target, config.close_position_tolerance_m, config.close_orientation_tolerance_rad)
        elif template.phase == 'transport' and instance.phase_index in clock.epochs:
            epoch = clock.epochs[instance.phase_index]
            if (robot.timestamp > epoch['deadline']+1e-8
                    or robot.timestamp-epoch['issued_at'] < template.duration_seconds-1e-8):
                return False
            passed = geometry_passes(robot, target, template.position_tolerance_m, template.orientation_tolerance_rad)
        else:
            return original(state, target, public_geometry=public_geometry)
        if passed:
            events.append(dict(event='public_phase_completion_verified', phase=template.phase,
                time=float(robot.timestamp), candidate_id=goal.candidate_id,
                actual_selected_at=float(goal.selected_at), native_success_claimed=False))
        return passed
    controller._phase_done = MethodType(done, controller)


def install(actor, protocol, root, task):
    expected = binding(root, protocol, task)
    if protocol.get('cup_physical_acquire_v167', {}).get(task) != expected:
        raise ValueError('freeze actual physical capture policy')
    controller, manager = actor.controller, actor.continuous_reference_manager
    if (controller.goal is not None or controller.feedback_step_calls or manager.confirmations
            or asdict(controller.gripper_timing_config) != expected['previous_extra_controller_gate']):
        raise ValueError('install before first control step on the expected final controller')
    controller.gripper_timing_config = GripperTimingConfig(**expected['actual_physical_close_gate'])
    witnesses, events = {}, []
    install_confirmation(manager, witnesses, events)
    install_completion(controller, manager.transport_phase_clock, witnesses, events)
    previous = actor.metrics_report
    def metrics(instance):
        return dict(previous(), cup_physical_acquire_v167=dict(binding=expected,
            actual_close_witnesses=deepcopy(witnesses), events=deepcopy(events),
            original_template_regression_caps_not_claimed_as_physical_completion=True,
            earlier_intersection_unchanged_statements_superseded_by_this_explicit_policy=True))
    actor.metrics_report = MethodType(metrics, actor)
