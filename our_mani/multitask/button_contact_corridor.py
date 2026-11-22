"""Admit descent from a complete moving-target trajectory, before dispatch.

The TCP can approach diagonally above the plate. Every sampled reference point
at plate height must stay inside the original TRAIN contact disk. This is a
nominal public-forecast proof; actual motion and native task success stay separate.
"""
from copy import deepcopy
from dataclasses import replace
from types import MethodType
import numpy as np
from .button_standoff import contact_radius
from .interaction import CURRENT
SCHEMA='r10_public_forecast_button_contact_corridor_v1'


def prove(model,plan,state,forecast,reference_id,asset,tolerance,*,dt=.05,maximum_age=.1):
    now=float(state.robot.timestamp)
    evidence=dict(nominal_reference_only=True,motor_tracking_guaranteed=False,
        extra_camera_physics_or_native_future_queries=0,samples=[])
    try:
        cutoff=forecast.metadata.get('observed_until')
        if type(cutoff) not in (float,int) or not -1e-8<=now-cutoff<=maximum_age:
            raise ValueError('current public forecast required')
        reference,plate=state.bodies[reference_id],state.bodies['plate']
        if any(not b.visible or not -1e-8<=now-b.timestamp<=maximum_age for b in (reference,plate)):
            raise ValueError('current visible button and plate required')
        radius=contact_radius(asset,tolerance)
        vertices=np.asarray(asset['bodies']['plate']['collision_vertices_body'],float)
        top=float(np.max((vertices@plate.pose.rotation.T+plate.pose.position)[:,2]))
        offset=plate.pose.position-reference.pose.position
        start=max(now,float(plan.start))
        if plan.end<start-1e-8:raise ValueError('unexpired complete reference required')
        contact_samples=0
        for t in np.linspace(start,plan.end,max(2,int(np.ceil((plan.end-start)/dt))+1)):
            q,_=plan.at(float(t));tcp=model.forward(q)
            predicted,_,_=forecast.at(reference_id,float(t))
            plate_center=predicted.position+offset
            height=top+predicted.position[2]-reference.pose.position[2]
            low=bool(tcp.position[2]<=height+tolerance)
            error=float(np.linalg.norm(tcp.position[:2]-plate_center[:2]))
            evidence['samples'].append(dict(time=float(t),TCP_position=tcp.position.tolist(),
                predicted_plate_center=plate_center.tolist(),plate_top_z=height,
                at_contact_height=low,horizontal_error_m=error))
            if low:
                contact_samples+=1
                if error>radius+1e-8:raise ValueError('reference_crosses_plate_height_outside_contact_disk')
        evidence.update(contact_samples=contact_samples,contact_radius_m=radius,
            plate_geometry_source='unchanged_shared_TRAIN_plate_mesh_and_current_visible_pose')
        if not contact_samples:raise ValueError('trajectory_does_not_reach_plate_height')
        return True,dict(evidence,reason='complete_nominal_contact_corridor')
    except (KeyError,IndexError,ValueError) as error:
        return False,dict(evidence,reason=str(error))


def install(actor,asset,*,replan_seconds=.2):
    c,m=actor.controller,actor.continuous_reference_manager
    if m.confirmations or c.feedback_step_calls or c.library.phase_order!=('approach','press'):
        raise ValueError('fresh actual button controller required')
    approach,press=c.library.for_phase('approach')[0],c.library.for_phase('press')[0]
    if press.reference_time_basis!=CURRENT:raise ValueError('explicit moving-frame press required')
    positions=press.positions.copy();positions[:,2]=approach.positions[-1,2]
    hover=replace(press,positions=positions)
    events=[];previous_roll=m.roll_current_reference
    def roll(instance,robot):
        goal=c.goal
        if goal is None or goal.phase!='press' or goal.template is not press:
            return previous_roll(robot)
        saved=(m.plan,m.plan_kind,m.plan_candidate_id,m.invalidated,c.trajectory)
        admitted=previous_roll(robot)
        if not admitted:return False
        accepted,proof=prove(m.native_gate.model,m.plan,m.current_state,m.current_forecast,
            goal.reference_id,asset,press.position_tolerance_m,dt=m.config.dt,
            maximum_age=c.generator.config.maximum_observation_age)
        events.append(dict(time=float(robot.timestamp),event='descent_candidate_contact_corridor',
            accepted=accepted,proof=proof,original_full_reference_and_actual_gates_passed=True,
            dispatched_by_this_module=False,phase_clock_reset=False))
        if not accepted:
            m.plan,m.plan_kind,m.plan_candidate_id,m.invalidated,c.trajectory=saved
            m.rolling_events[-1]['contact_corridor_rejected_before_dispatch']=True
            return False
        return True
    m.roll_current_reference=MethodType(roll,m)
    previous_step=c.step;last_attempt={'time':-np.inf}
    def set_template(template):
        c.library.templates=tuple(template if t.phase=='press' else t for t in c.library.templates)
        if c.goal is not None and c.goal.phase=='press':c.goal=replace(c.goal,template=template)
    def step(instance,state,forecast,remaining_seconds):
        now=float(state.robot.timestamp)
        if c.phase_index!=1:return previous_step(state,forecast,remaining_seconds)
        m.current_state,m.current_forecast,m.current_robot=state,forecast,state.robot
        cutoff=forecast.metadata.get('observed_until')
        m.forecast_observed_until=float(cutoff) if type(cutoff) in (float,int) else None
        goal=c.goal
        if goal is None:
            set_template(hover)
        elif goal.phase=='press' and not m.braking_active(now) and now>=last_attempt['time']+replan_seconds-1e-8:
            last_attempt['time']=now
            old_template=goal.template
            set_template(press)
            if not m.roll_current_reference(state.robot):
                c.goal=goal;set_template(old_template)
                if old_template is press:
                    # Reaching the contact plane requires a fresh corridor. A
                    # failed hover replan falls back to the ordinary full brake.
                    set_template(hover)
                    if not m.roll_current_reference(state.robot):
                        c.goal=goal;set_template(old_template)
                        m.invalidate('fresh_contact_corridor_and_hover_unavailable',now)
                events.append(dict(time=now,event='descent_not_admitted_keep_or_replan_standoff',
                    native_success_claimed=False,phase_clock_reset=False))
        return previous_step(state,forecast,remaining_seconds)
    c.step=MethodType(step,c)
    previous_metrics=actor.metrics_report
    def metrics(instance):
        return dict(previous_metrics(),button_contact_corridor=dict(schema=SCHEMA,
            events=deepcopy(events),replan_seconds=replan_seconds,
            native_task_success_and_actual_motion_criteria_unchanged=True,
            original_current_alignment_worst_case_policy_replaced=True,
            TRAIN_approach_offset_m=approach.positions[-1].tolist(),
            TRAIN_press_offset_m=press.positions[-1].tolist()))
    actor.metrics_report=MethodType(metrics,actor)
    return events
