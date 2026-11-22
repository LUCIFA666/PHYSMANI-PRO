"""Public action-stage guards from the author, separate from regression error.

The final native success, original physical capture, counted motion and collision
gates remain independent. A prefix transition records a policy-stage transition;
it never fabricates a selected task candidate or a fitted-skill training label.
"""
import ast
from copy import deepcopy
from pathlib import Path
from types import MethodType
import numpy as np
from ..contracts import Pose
from .evaluation import file_sha
from .interaction import COMMAND_GOAL, _world_target
from .specs import get_task_spec

TASKS=('place_cups_on_rotating_frame','place_cups_on_rotating_frame_high_speed')
CAPS={'approach':.1,'transport':.1,'align':.02,'hang':.03}
AUTHOR={'th_w0':.1,'th_w3':.1,'th_w4':.02,'th_w5':.03}


def author_source(root,protocol,task):
    spec=get_task_spec(task);path=Path(protocol['baseline_repo'])/spec.native_source_relative_path
    if file_sha(path)!=spec.native_source_sha256:raise ValueError('original native cup task changed')
    tree=ast.parse(path.read_text());found={}
    for node in ast.walk(tree):
        if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name):
            name=node.targets[0].id
            if name in AUTHOR:
                value=ast.literal_eval(node.value)
                if name in found and found[name]!=value:raise ValueError('ambiguous author action guard')
                found[name]=value
    if found!=AUTHOR:raise ValueError('original author cup action thresholds differ')
    return dict(path=str(path.resolve()),sha256=file_sha(path),thresholds=found)


def binding(root,protocol,task):
    if task not in TASKS:raise ValueError('only the two cup placement tasks')
    return dict(schema='r10_cup_current_public_author_action_stage_guards_v251',task=task,
        sources_sha256={'our_mani/multitask/cup_author_action_guard_v251.py':file_sha(__file__)},
        original_author_source=author_source(root,protocol,task),actual_position_guards_m=CAPS,
        extra_fitted_error_phase_completion_gates_explicitly_replaced=True,
        phase_orientation_completion_guard_replaced_by_author_position_guard=True,
        current_public_reference_required_not_future_prediction_or_native_waypoint=True,
        original_fitted_minimum_elapsed_and_absolute_deadlines_retained=True,
        prefix_requires_actual_qualified_dispatch_then_original_verified_stop=True,
        prefix_transition_is_not_a_task_candidate_or_fitted_skill_training_label=True,
        hang_open_requires_current_author_geometry_and_original_verified_stop=True,
        original_fit_error_caps_physics_success_capture_collision_motion_and10s_unchanged=True,
        formal_VAL_TEST=False)


def current_target(controller,state,template,reference_id):
    reference=controller.generator.observed_reference(state,reference_id,template,controller.origin)
    if template.reference_time_basis==COMMAND_GOAL:
        canonical=Pose.from_xyzw(template.completion_geometry_xyzw)
        target=Pose(reference.pose.position+reference.pose.rotation@canonical.position,
            reference.pose.rotation@canonical.rotation)
    else:target=_world_target(reference.pose,np.zeros(3),np.zeros(3),template,1.)[0]
    return target


def geometry(controller,state,template,reference_id):
    try:target=current_target(controller,state,template,reference_id)
    except (ValueError,KeyError,IndexError):return None
    distance=float(np.linalg.norm(state.robot.pose.position-target.position))
    return dict(phase=template.phase,reference_id=reference_id,time=float(state.robot.timestamp),
        current_public_target_xyzw=target.xyzw().tolist(),position_error_m=distance,
        position_guard_m=CAPS[template.phase],passed=bool(distance<=CAPS[template.phase]),
        native_task_success_claimed=False)


def elapsed_and_deadline(controller,manager,template,now):
    goal=controller.goal
    epoch=manager.transport_phase_clock.epochs.get(controller.phase_index) if template.phase=='transport' else None
    if epoch is not None:start,deadline=epoch['issued_at'],epoch['deadline']
    elif goal is not None:start,deadline=goal.selected_at,goal.selected_at+goal.phase_budget
    else:return False
    return bool(now-start>=template.duration_seconds-1e-8 and now<=deadline+1e-8 and now<=10.+1e-8)


def install_completion(controller,manager,events):
    previous=controller._phase_done
    def done(instance,state,target,*,public_geometry=False):
        goal=instance.goal;template=goal.template
        if template.phase not in CAPS:return previous(state,target,public_geometry=public_geometry)
        if not elapsed_and_deadline(instance,manager,template,float(state.robot.timestamp)):return False
        row=geometry(instance,state,template,goal.reference_id)
        if row is None or not row['passed']:return False
        if template.phase=='hang' and template.at(1.)[3]>=.5 and (not state.robot.gripper_open or state.robot.contact):
            return False
        events.append(dict(row,event='current_public_author_action_guard_completed',
            candidate_id=goal.candidate_id,original_fitted_error_completion_not_claimed=True))
        return True
    controller._phase_done=MethodType(done,controller)


def install_prefix_confirmation(manager,witnesses,events):
    previous=manager.confirm_actual_issued
    def confirm(instance,**kwargs):
        active=deepcopy(instance.approach_transit.active)
        certificate=kwargs['command'].get('feedforward_execution')
        prefix=bool(active is not None and instance.plan_kind=='approach_transit'
            and certificate is not None and certificate.get('candidate_id')==instance.plan_candidate_id)
        result=previous(**kwargs);execution=kwargs['execution']
        if (prefix and not instance.qualification_failed
                and execution.get('actual_window_velocity_within_limits') is True
                and execution.get('actual_window_acceleration_within_limits') is True):
            index=active['phase_index'];issued=float(execution['actual_issued_at']);now=float(kwargs['time'])
            old=witnesses.get(index)
            if old is not None and (old['reference_id']!=active['reference_id']
                    or old['manipulated_ids']!=list(active['manipulated_ids'])):
                witnesses.pop(index,None)
                events.append(dict(event='prefix_reference_identity_changed_clock_restarted_without_deadline_extension',time=now))
                old=None
            witnesses[index]=dict(phase=active['phase'],phase_index=index,reference_id=active['reference_id'],
                manipulated_ids=list(active['manipulated_ids']),issued_at=issued if old is None else old['issued_at'],
                confirmed_after=now,deadline=min(active['deadline'],10.) if old is None else min(old['deadline'],active['deadline'],10.),
                actual_prefix_plan_id=certificate['candidate_id'])
        return result
    manager.confirm_actual_issued=MethodType(confirm,manager)


def prefix_guard(controller,manager,state,witness):
    now=float(state.robot.timestamp);index=controller.phase_index
    if (controller.goal is not None or manager.qualification_failed or witness is None
            or witness['phase_index']!=index or witness['phase'] not in ('approach','transport')
            or controller.library.phase_order[index]!=witness['phase']
            or not witness['confirmed_after']<=now+1e-8<=witness['deadline']+1e-8
            or manager.last_confirmed is None or abs(manager.last_confirmed['time']-now)>1e-8):return None
    templates=controller.library.for_phase(witness['phase'])
    if len(templates)!=1:return None
    template=templates[0]
    if now-witness['issued_at']<template.duration_seconds-1e-8:return None
    bodies=state.perception_metadata.get('bodies',{});reference=bodies.get(witness['reference_id'],{})
    if reference.get('motion_mode')!=template.reference_mode:return None
    bindings={**state.role_bindings,**controller.committed_roles}
    if witness['reference_id'] not in bindings.get(template.reference_role,()):return None
    if len(witness['manipulated_ids'])!=1 or tuple(bindings.get(template.manipulated_role,()))!=tuple(witness['manipulated_ids']):return None
    if template.phase=='transport':
        attached={key for key,value in bodies.items() if value.get('motion_mode')=='attached'}
        identity=witness['manipulated_ids'][0];info=bodies.get(identity,{})
        if (not state.robot.contact or attached!={identity}
                or identity in state.perception_metadata.get('invalid_mode_ids',())
                or info.get('mode_provenance')!='caller_controller_touch_action_proprioception'):return None
        epoch=manager.transport_phase_clock.epochs.get(index)
        if epoch is None or now>epoch['deadline']+1e-8:return None
    elif state.robot.contact:return None
    row=geometry(controller,state,template,witness['reference_id'])
    return (template,row) if row is not None and row['passed'] else None


def install_prefix_transition(controller,manager,witnesses,events):
    previous=controller.step
    def step(instance,state,forecast,remaining_seconds):
        witness=witnesses.get(instance.phase_index)
        ready=prefix_guard(instance,manager,state,witness)
        if ready is not None:
            template,row=ready
            manager.current_robot,manager.current_state,manager.current_forecast=state.robot,state,forecast
            cutoff=forecast.metadata.get('observed_until')
            manager.forecast_observed_until=float(cutoff) if type(cutoff) in (float,int) else None
            if not manager.capture_stopped(state.robot):
                if not manager.braking_active(state.robot.timestamp):manager.invalidate('author_action_guard_requires_original_verified_stop',state.robot.timestamp)
                events.append(dict(row,event='author_action_guard_geometry_ready_wait_for_original_verified_brake'))
                return instance._hold('author_action_guard_wait_original_verified_brake')
            before=instance.phase_index
            instance.committed_roles[template.reference_role]=(witness['reference_id'],)
            instance.committed_roles[template.manipulated_role]=tuple(witness['manipulated_ids'])
            manager.cancel_unissued_retarget();manager.pending=None
            manager.invalidate('stopped_public_author_action_guard_prefix_stage_transition',state.robot.timestamp)
            instance.phase_index+=1;instance.goal=instance.trajectory=instance.origin=None
            events.append(dict(row,event='actual_prefix_public_author_action_stage_transition',
                from_phase_index=before,to_phase_index=instance.phase_index,actual_prefix_witness=deepcopy(witness),
                original_verified_stop_passed=True,task_candidate_fabricated=False,fitted_skill_training_label_created=False))
        return previous(state,forecast,remaining_seconds)
    controller.step=MethodType(step,controller)


def install_release(controller,manager,events):
    previous=controller._gripper_opening
    def opening(instance,state,requested,final_target):
        result=previous(state,requested,final_target);goal=instance.goal
        if goal is None or goal.phase!='hang' or result<.5:return result
        if (manager.last_confirmed or {}).get('opening',0.)>=.5:return result
        row=geometry(instance,state,goal.template,goal.reference_id)
        if row is None or not row['passed'] or not manager.capture_stopped(state.robot):
            events.append(dict(event='hang_open_deferred_until_current_author_geometry_and_original_stop',
                time=float(state.robot.timestamp),geometry=row,actual_opening_request=0.))
            return 0.
        events.append(dict(row,event='hang_open_current_public_author_geometry_and_stop_verified'))
        return result
    controller._gripper_opening=MethodType(opening,controller)


def install(actor,protocol,root,task):
    expected=binding(root,protocol,task)
    if protocol.get('cup_author_action_guard_v251',{}).get(task)!=expected:raise ValueError('freeze actual author action guard policy')
    c,m=actor.controller,actor.continuous_reference_manager
    if c.goal is not None or c.feedback_step_calls or m.confirmations:raise ValueError('install before first control step')
    witnesses,events={},[]
    install_completion(c,m,events);install_prefix_confirmation(m,witnesses,events)
    install_prefix_transition(c,m,witnesses,events);install_release(c,m,events)
    previous=actor.metrics_report
    def metrics(instance):return dict(previous(),cup_author_action_guard_v251=dict(binding=expected,
        actual_prefix_witnesses=deepcopy(witnesses),events=deepcopy(events),
        inherited_fitted_completion_unchanged_claims_superseded_by_explicit_action_policy=True))
    actor.metrics_report=MethodType(metrics,actor)
