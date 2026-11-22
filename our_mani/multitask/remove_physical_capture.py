"""Complete command-frame acquire from acknowledged physical capture evidence."""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from types import MethodType
import numpy as np
from scipy.spatial.transform import Rotation
from ..contracts import Pose
from .interaction import COMMAND_GOAL,GripperTimingConfig
from .cup_physical_acquire_v167 import geometry_passes
from .evaluation import file_sha
TASKS=('remove_cups_from_rotating_frame','remove_cups_from_rotating_frame_high_speed')


def binding(root,config,task):
    if task not in TASKS:raise ValueError('explicit original command-frame acquire task')
    physical=asdict(GripperTimingConfig(.02,.1))
    if config['gripper_timing_config']!=physical:raise ValueError('original physical grasp gate required')
    sources=('our_mani/multitask/remove_physical_capture.py','our_mani/multitask/cup_physical_acquire_v167.py')
    return dict(schema='r10_command_canonical_physical_capture_actual_dispatch_v1',task=task,
        physical_grasp_gate=physical,interaction=config['learned_artifacts'][task]['interaction'],
        sources_sha256={p:file_sha(Path(root)/p) for p in sources},
        acquire_completion_policy='elapsed fitted acquire duration; stopped canonical close actually dispatched; qualified motion receipt; unique current public held identity; original physical SE3',
        original_fitting_tolerances_are_retained_as_fit_quality=True,
        held_identity_is_manipulated_cup_not_spoke_reference=True,
        fitted_regression_tolerance_is_not_a_physical_grasp_gate=True,
        native_physics_grasp_success_and_collision_motion_clock_unchanged=True,formal_VAL_TEST=False)


def public_cup_capture(c,state):
    """A spoke defines the command frame; only the selected cup may be held."""
    goal=c.goal;now=float(state.robot.timestamp);closed=c._close_edge_issued_at
    ids=goal.manipulated_ids
    if (goal.template.phase!='acquire' or not state.robot.contact or closed is None
        or not np.isfinite(closed) or closed>now+1e-8 or len(ids)!=1):return False
    identity=ids[0];metadata=state.perception_metadata
    invalid=set(metadata.get('invalid_mode_ids',()))
    bodies=metadata.get('bodies',{})
    associated={body for body,info in bodies.items()
        if info.get('motion_mode')=='attached' and body not in invalid}
    if (not identity.startswith('cup') or identity in invalid or associated!={identity}
        or tuple(state.role_bindings.get(goal.template.manipulated_role,()))!=(identity,)
        or bodies.get(identity,{}).get('mode_provenance')!='caller_controller_touch_action_proprioception'):
        return False
    try:
        c.generator.observed_reference(state,identity,goal.template,c.origin)
        return True
    except (KeyError,ValueError):
        return c._causal_attachment_pose(state,identity) is not None


def proposal(m,command,execution):
    c=m.controller;goal,state=c.goal,m.current_state
    if (goal is None or goal.phase!='acquire' or goal.template.reference_time_basis!=COMMAND_GOAL
        or state is None or command.get('opening',1.)>=.5 or state.robot.contact
        or (m.last_confirmed or {}).get('opening',0.)<.5):return None
    now=float(state.robot.timestamp);issued=execution.get('actual_issued_at');closed=c._close_edge_issued_at
    if (issued is None or abs(float(issued)-now)>1e-8 or closed is None or abs(float(closed)-now)>1e-8
        or not m.capture_stopped(state.robot)):return None
    progress=(now-goal.selected_at)/goal.template.duration_seconds
    if goal.template.at(float(np.clip(progress,0.,1.)))[3]>=.5:return None
    try:
        current=c.generator.observed_reference(state,goal.reference_id,goal.template,c.origin)
        canonical=Pose.from_xyzw(goal.template.completion_geometry_xyzw)
        target=Pose(current.pose.position+current.pose.rotation@canonical.position,
            current.pose.rotation@canonical.rotation)
    except (KeyError,ValueError,IndexError):return None
    cfg=c.gripper_timing_config
    if not geometry_passes(state.robot,target,cfg.close_position_tolerance_m,cfg.close_orientation_tolerance_rad):return None
    return dict(candidate_id=goal.candidate_id,phase_index=c.phase_index,
        manipulated_ids=list(goal.manipulated_ids),issued_at=now,
        close_position_error_m=float(np.linalg.norm(state.robot.pose.position-target.position)),
        close_orientation_error_rad=float(Rotation.from_matrix(state.robot.pose.rotation@target.rotation.T).magnitude()),
        stopped_original_physical_SE3_close=True,completion_reference='fresh_public_TRAIN_canonical_geometry',public_state_only=True)


def install(actor,config,root,task):
    expected=binding(root,config,task)
    if config.get('remove_physical_capture',{}).get(task)!=expected:raise ValueError('freeze physical capture policy')
    c,m=actor.controller,actor.continuous_reference_manager
    if c.goal is not None or c.feedback_step_calls or m.confirmations:raise ValueError('install before first counted step')
    if asdict(c.gripper_timing_config)!=expected['physical_grasp_gate']:raise ValueError('physical gate changed')
    previous_contact=c._public_acquire_contact
    def contact(instance,state):
        if instance.goal is not None and instance.goal.phase=='acquire':return public_cup_capture(instance,state)
        return previous_contact(state)
    c._public_acquire_contact=MethodType(contact,c)
    witnesses,events={},[];previous_confirm=m.confirm_actual_issued;previous_done=c._phase_done
    def confirm(instance,**kwargs):
        proposed=proposal(instance,kwargs['command'],kwargs['execution'])
        result=previous_confirm(**kwargs);execution=kwargs['execution']
        if (proposed is not None and not instance.qualification_failed
            and execution.get('actual_window_velocity_within_limits') is True
            and execution.get('actual_window_acceleration_within_limits') is True):
            proposed['confirmed_after']=float(kwargs['time']);witnesses[proposed['candidate_id']]=proposed
            events.append(dict(event='canonical_physical_close_actually_dispatched',**deepcopy(proposed)))
        return result
    m.confirm_actual_issued=MethodType(confirm,m)
    def done(instance,state,target,*,public_geometry=False):
        goal=instance.goal
        if goal is None or goal.phase!='acquire':return previous_done(state,target,public_geometry=public_geometry)
        if goal.template.reference_time_basis!=COMMAND_GOAL or not public_geometry:return False
        witness=witnesses.get(goal.candidate_id)
        if (witness is None or witness['phase_index']!=instance.phase_index
            or witness['manipulated_ids']!=list(goal.manipulated_ids)
            or witness['confirmed_after']>state.robot.timestamp+1e-8
            or state.robot.timestamp-goal.selected_at<goal.template.duration_seconds-1e-8
            or not instance._public_acquire_contact(state)):return False
        cfg=instance.gripper_timing_config
        passed=geometry_passes(state.robot,target,cfg.close_position_tolerance_m,cfg.close_orientation_tolerance_rad)
        if passed:events.append(dict(event='canonical_public_physical_acquire_completed',time=float(state.robot.timestamp),
            candidate_id=goal.candidate_id,native_success_claimed=False))
        return passed
    c._phase_done=MethodType(done,c)
    previous_metrics=actor.metrics_report
    def metrics(instance):return dict(previous_metrics(),remove_physical_capture=dict(binding=expected,
        actual_close_witnesses=deepcopy(witnesses),events=deepcopy(events)))
    actor.metrics_report=MethodType(metrics,actor)
    return witnesses,events
