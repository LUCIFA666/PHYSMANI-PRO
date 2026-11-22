"""Track moving command frames while retaining their fresh canonical completion."""
from copy import deepcopy
from dataclasses import replace
from types import MethodType
from ..contracts import Pose,Reachability
from .interaction import CURRENT,COMMAND_GOAL
SCHEMA='r10_live_command_control_original_canonical_completion_v1'


def install(actor,*,phases=('align','insert'),replan_seconds=.1,ready=lambda:True):
    c,m=actor.controller,actor.continuous_reference_manager
    if c.goal is not None or c.feedback_step_calls or m.confirmations:
        raise ValueError('install before control execution')
    if (not phases or any(p not in c.library.phase_order for p in phases)
        or not m.config.dt<=replan_seconds<=.5):raise ValueError('declared actual task phases required')
    originals={};events=[];last_attempt={};previous_done=c._phase_done
    def done(instance,state,target,*,public_geometry=False):
        goal=instance.goal
        source=None if goal is None else originals.get(id(goal.template))
        if source is None:return previous_done(state,target,public_geometry=public_geometry)
        try:
            current=instance.generator.observed_reference(state,goal.reference_id,source,instance.origin)
            canonical=Pose.from_xyzw(source.completion_geometry_xyzw)
            physical=Pose(current.pose.position+current.pose.rotation@canonical.position,
                current.pose.rotation@canonical.rotation)
            return previous_done(state,physical,public_geometry=True)
        except (ValueError,KeyError,IndexError):return False
    c._phase_done=MethodType(done,c)
    previous_step=c.step
    def step(instance,state,forecast,remaining):
        phase=instance.library.phase_order[instance.phase_index] if instance.phase_index<len(instance.library.phase_order) else None
        if phase in phases and instance.goal is None:
            if not ready():return instance._hold('fresh_measured_grasp_frame_required_before_live_command_tracking')
            converted=[]
            for t in instance.library.templates:
                if t.phase in phases and t.reference_time_basis==COMMAND_GOAL:
                    if t.reference_mode!='external' or t.reference_kind!='observed_se3':
                        raise ValueError('external observed moving command frames required')
                    live=replace(t,reference_time_basis=CURRENT,reference_time_rule_sha256=None,completion_geometry_xyzw=None)
                    originals[id(live)]=t;converted.append(live)
                    events.append(dict(time=float(state.robot.timestamp),event='live_control_with_original_canonical_completion',
                        phase=t.phase,original_completion_geometry_xyzw=t.completion_geometry_xyzw.tolist(),
                        original_position_tolerance_m=t.position_tolerance_m,
                        original_orientation_tolerance_rad=t.orientation_tolerance_rad,
                        native_success_claimed=False,phase_clock_reset=False))
                else:converted.append(t)
            instance.library.templates=tuple(converted)
        return previous_step(state,forecast,remaining)
    c.step=MethodType(step,c)
    previous_stream=m.check_stream
    def stream(instance,robot,duration):
        goal,plan,now=c.goal,m.plan,float(robot.timestamp)
        if (goal is not None and id(goal.template) in originals and not m.qualification_failed
            and not m.braking_active(now) and abs(duration-m.config.dt)<1e-8):
            due=(plan is None or now>=plan.end-1e-8 or
                now>=last_attempt.get(goal.candidate_id,goal.selected_at)+replan_seconds-1e-8)
            if due:
                last_attempt[goal.candidate_id]=now;renewed=m.roll_current_reference(robot)
                events.append(dict(time=now,event='live_command_reference_roll',phase=goal.phase,
                    renewed=bool(renewed),original_phase_epoch=float(goal.selected_at),
                    original_phase_deadline=float(goal.selected_at+goal.phase_budget),phase_clock_reset=False))
                if not renewed and (plan is None or now>=plan.end-1e-8):
                    return Reachability(False,'live_command_requires_fresh_complete_motion_proof')
        return previous_stream(robot,duration)
    m.check_stream=MethodType(stream,m)
    previous_metrics=actor.metrics_report
    def metrics(instance):return dict(previous_metrics(),moving_command_canonical=dict(schema=SCHEMA,
        phases=list(phases),replan_seconds=replan_seconds,events=deepcopy(events),
        control_reference_time_semantics_explicitly_changed=True,
        original_canonical_completion_tolerances_progress_clock_and_native_success_retained=True,
        additional_camera_physics_or_native_future_queries=0))
    actor.metrics_report=MethodType(metrics,actor)
    return events
