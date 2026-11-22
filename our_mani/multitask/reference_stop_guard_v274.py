"""Reject a moving next reference without an original-gate stopping continuation.

These are controller trajectory proofs, not motor guarantees. Actual counted
motion remains an independently measured outcome. Cached stopping trajectories
are admitted only after real dispatch and are always rechecked in fresh state.
"""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import MethodType
import numpy as np
from ..contracts import Reachability
from ..intercept_planner import JointTrajectory
from ..reachability import TimedReachability
from .evaluation import digest,file_sha

BASE_DURATIONS=(.05,.15,.2,.25,.3,.35,.4,.45,.5,.6,.7,.8,.9,1.,1.1)


def binding(root,protocol,task):
    return dict(schema='r10_next_reference_full_offset_stop_and_dense_fresh_brake_v274',task=task,
        sources_sha256={'our_mani/multitask/reference_stop_guard_v274.py':file_sha(__file__)},
        original_sparse_brake_durations_seconds=[.05,.15,.25,.4,.6,.9,1.1],
        brake_durations_seconds=list(BASE_DURATIONS),adaptive_durations='1.02/1.25/1.5 times maximum joint speed divided by original acceleration limit',
        next_reference_requires_complete_offset_reference_stop_and_current_actual_Hermite_to_same_endpoint=True,
        cached_plan_requires_actual_dispatch_ack_and_fresh_revalidation=True,
        original_6_joint_limits_complete_collision_sampling_and_global10s_unchanged=True,
        task_phase_and_suffix_deadlines_not_extended_by_safety_braking=True,
        neither_motor_tracking_nor_robust_future_obstacle_guarantee_claimed=True,
        extra_camera_physics_or_native_dynamic_truth_inputs=False,formal_VAL_TEST=False)


def duration_grid(velocities,limits,available,extra=()):
    limits=np.asarray(limits,float)
    if not np.allclose(limits,6.):raise ValueError('original six-unit native model required')
    lower=max(float(np.max(np.abs(np.asarray(v,float))/limits)) for v in velocities)
    values=(*BASE_DURATIONS,*(lower*f for f in (1.02,1.25,1.5)),*extra)
    return sorted({float(x) for x in values if np.isfinite(x) and .05-1e-8<=x<=available+1e-8})


class OffsetModel:
    def __init__(self,model,offset):self.model,self.offset=model,float(offset)
    def __getattr__(self,name):return getattr(self.model,name)
    def collision(self,joints,future_offset):return self.model.collision(joints,self.offset+float(future_offset))


def offset_gate(original,offset):
    return TimedReachability(OffsetModel(original.model,offset),max_joint_step=original.max_joint_step,
        minimum_samples=original.minimum_samples,twist_tolerance=original.twist_tolerance,
        position_tolerance=original.position_tolerance,orientation_tolerance=original.orientation_tolerance)


class NextReferenceStopGuard:
    def __init__(self,manager):
        self.manager=manager;self.events=[];self.queries=self.rejections=0
        self.pending_stop=self.confirmed_stop=None

    def record(self,**row):
        if len(self.events)<512:self.events.append(row)

    def available(self,now,remaining=10.):
        forecast=self.manager.current_forecast
        if forecast is None or len(forecast.times)==0:return 0.
        return max(0.,min(10.-now,float(remaining),float(forecast.times[-1])-now))

    def next_stop(self,robot,pending):
        m=self.manager;dt=float(m.config.dt);now=float(robot.timestamp)
        q=np.asarray(pending['reference_to_joints'],float);v=np.asarray(pending['reference_to_velocity'],float)
        future=replace(robot,timestamp=now+dt,joints=q,joint_velocities=v)
        available=self.available(now)-dt;trials=[]
        reference_gate=offset_gate(m.native_gate,dt)
        for duration in duration_grid((v,),m.native_gate.model.acceleration_limits,available):
            end=q+.5*v*duration;zero=np.zeros_like(v)
            reference=reference_gate.check_joints(future,end,zero,duration)
            actual=(m.native_gate.check_joints(robot,end,zero,dt+duration) if reference.feasible else None)
            trials.append(dict(duration=duration,offset_reference_reason=reference.reason,
                current_actual_to_same_stop_reason=None if actual is None else actual.reason))
            if reference.feasible and actual is not None and actual.feasible:
                plan=JointTrajectory(now+dt,now+dt+duration,q.copy(),v.copy(),end,zero)
                return plan,dict(trials=trials,reference_collision_time_offset=dt,
                    offset_reference_full_gate=deepcopy(reference.evidence),actual_full_gate=deepcopy(actual.evidence),
                    actual_proof_is_a_full_Hermite_from_current_state_not_a_motor_tracking_claim=True)
        return None,dict(trials=trials,reference_collision_time_offset=dt)

    def install_stream(self):
        m=self.manager;previous=m.check_stream
        def check(instance,robot,duration):
            self.pending_stop=None
            result=previous(robot,duration)
            if result is None or not result.feasible or instance.pending is None or instance.plan_kind=='braking':return result
            self.queries+=1;pending=instance.pending
            v=np.asarray(pending['reference_to_velocity'],float)
            if np.all(np.abs(v)<=1e-10):
                pending['next_reference_stop_guard_v274']=dict(nominal_endpoint_derivative_zero=True,actual_stopped_not_claimed=True)
                return result
            plan,proof=self.next_stop(robot,pending)
            if plan is None:
                self.rejections+=1;pending['next_reference_stop_guard_rejected']=True
                instance.pending=None
                self.record(time=float(robot.timestamp),event='moving_reference_rejected_no_complete_original_gate_stop',
                    proposal_id=pending['proposal_id'],proof=proof,original_one_step_proposal_not_issued=True)
                return Reachability(False,'next_reference_has_no_original_collision_checked_stop')
            pending['next_reference_stop_guard_v274']=dict(complete_nominal_offset_stop=True,
                complete_current_actual_Hermite_to_same_stop=True,stop_at=float(plan.end),proof=proof,
                motor_tracking_guaranteed=False)
            self.pending_stop=dict(proposal_id=pending['proposal_id'],plan=plan,proof=proof)
            self.record(time=float(robot.timestamp),event='next_reference_complete_stop_admitted',
                proposal_id=pending['proposal_id'],stop_at=float(plan.end),motor_tracking_guaranteed=False)
            return result
        m.check_stream=MethodType(check,m)

    def install_confirmation(self):
        m=self.manager;previous=m.confirm_actual_issued
        def confirm(instance,**kwargs):
            staged=self.pending_stop;certificate=kwargs['command'].get('feedforward_execution')
            result=previous(**kwargs);execution=kwargs['execution']
            if (staged is not None and certificate is not None
                    and certificate.get('proposal_id')==staged['proposal_id']
                    and abs(float(kwargs['time'])-staged['plan'].start)<1e-8
                    and not instance.qualification_failed
                    and execution.get('actual_window_velocity_within_limits') is True
                    and execution.get('actual_window_acceleration_within_limits') is True):
                self.confirmed_stop=staged
                self.record(time=float(kwargs['time']),event='stop_candidate_cached_after_actual_qualified_dispatch',
                    proposal_id=staged['proposal_id'],fresh_revalidation_still_required=True)
            self.pending_stop=None
            return result
        m.confirm_actual_issued=MethodType(confirm,m)

    def make_brake(self,robot,remaining_seconds,reason):
        m=self.manager;boundary=m.continuity_state(robot)
        if boundary is None:return False
        q0,v0,basis=boundary;now=float(robot.timestamp)
        available=self.available(now,remaining_seconds);extra=[]
        if self.confirmed_stop is not None:
            plan=self.confirmed_stop['plan']
            if plan.start-1e-8<=now<plan.end-1e-8:
                expected_q,expected_v=plan.at(now)
                if np.allclose(q0,expected_q,rtol=0,atol=1e-8) and np.allclose(v0,expected_v,rtol=0,atol=1e-8):
                    extra.append(plan.end-now)
        trials=[]
        for duration in duration_grid((v0,robot.joint_velocities),m.native_gate.model.acceleration_limits,available,extra):
            phase=(m.controller.library.phase_order[m.controller.phase_index]
                if m.controller.phase_index<len(m.controller.library.phase_order) else 'verify')
            intent=m.controller.goal if m.controller.goal is not None else m.retired_brake_goal
            same=intent is not None and intent.phase==phase
            m.controller.generator.bind_candidate_scene(m.current_state,m.current_forecast,phase,
                intent.reference_id if same else '',intent.manipulated_ids if same else (),now+duration)
            end=q0+.5*v0*duration;zero=np.zeros_like(v0)
            reference=m.native_gate.check_joints(replace(robot,joints=q0,joint_velocities=v0),end,zero,duration)
            actual=m.native_gate.check_joints(robot,end,zero,duration)
            trials.append(dict(duration=duration,reference_reason=reference.reason,actual_reason=actual.reason))
            if not(reference.feasible and actual.feasible):continue
            m.plan=JointTrajectory(now,now+duration,q0.copy(),v0.copy(),end,zero)
            m.plan_kind='braking';m.plan_candidate_id='fresh_dense_brake:'+digest(dict(time=now,q=q0.tolist(),v=v0.tolist(),duration=duration))[:16]
            m.invalidated=False
            row=dict(time=now,event='dense_fresh_original_dual_gate_brake_staged',reason=reason,
                boundary_basis=basis,trials=trials,cached_certificate_reused_without_revalidation=False,
                phase_clock_reset=False,total_budget_extended=False,safety_brake_is_not_task_phase_continuation=True)
            m.events.append(row);self.record(**deepcopy(row));return True
        self.record(time=now,event='no_dense_fresh_original_gate_stop',reason=reason,trials=trials,
            unsafe_hold_not_relabelled_acceleration_limited=True)
        m.events.append(dict(time=now,event='no_feasible_continuous_brake',reason=reason,trials=trials,
            acceleration_limited_stop_verified=False))
        return False


def install(actor,protocol,root,task):
    expected=binding(root,protocol,task)
    if protocol.get('reference_stop_guard_v274',{}).get(task)!=expected:raise ValueError('freeze next reference stoppability policy')
    c,m=actor.controller,actor.continuous_reference_manager
    if c.goal is not None or c.feedback_step_calls or m.confirmations:raise ValueError('install before first actual step')
    if not np.allclose(m.native_gate.model.acceleration_limits,6.):raise ValueError('original native six-unit model required')
    guard=NextReferenceStopGuard(m);guard.install_stream();guard.install_confirmation()
    m.make_brake=MethodType(lambda instance,*args,**kwargs:guard.make_brake(*args,**kwargs),m)
    previous=actor.metrics_report
    def metrics(instance):return dict(previous(),reference_stop_guard_v274=dict(binding=expected,
        queries=guard.queries,rejections=guard.rejections,events=deepcopy(guard.events),maximum_retained_events=512,
        actual_motor_motion_still_measured=True,unchanged_collision_guards_not_bypassed=True))
    actor.metrics_report=MethodType(metrics,actor)
