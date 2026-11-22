"""Dispatch-confirmed continuous reference streams and dual-proof retargeting.

Reference q/v is a controller state, not observed instantaneous motor velocity.
Only the counted backend callback advances the issued-reference ledger. All
new reference plans retain the original actual-state reachability proof and
add a complete reference-state joint/sampled-collision proof before selection.
"""
from copy import deepcopy
from dataclasses import replace
from types import MethodType
from pathlib import Path
import numpy as np

from .evaluation import digest,file_sha
from .persistent_reference_feedforward import PersistentReferenceConfig,PersistentReferenceManager
from .causal_retarget_interaction import CausalRetargetManager,retarget_binding
from ..contracts import Reachability
from ..intercept_planner import JointTrajectory
from ..reachability import TimedReachability

SCHEMA='r10_continuous_verified_reference_feedforward_v2'
ROOT=Path(__file__).resolve().parents[2]


def continuous_reference_binding(config=None):
    raw=dict(config or {});version=raw.pop('version',SCHEMA)
    if version!=SCHEMA:raise ValueError('wrong continuous-reference version')
    base=PersistentReferenceConfig(**raw)
    normalized=dict(version=SCHEMA,dt=base.dt,observed_velocity_tolerance=base.observed_velocity_tolerance,
        observed_acceleration_tolerance=base.observed_acceleration_tolerance)
    sources=('our_mani/multitask/persistent_reference_feedforward.py',
        'our_mani/multitask/causal_retarget_interaction.py','our_mani/multitask/feedback_interaction.py',
        'our_mani/multitask/interaction.py','our_mani/reachability.py','our_mani/intercept_planner.py')
    return dict(normalized_config=normalized,configuration_sha256=digest(normalized),
        source=dict(path=str(Path(__file__).relative_to(ROOT)),sha256=file_sha(__file__)),
        original_sources_sha256={p:file_sha(ROOT/p)for p in sources},actual_tracking_guaranteed=False)


class _DualProofReachability:
    def __init__(self,manager):self.manager=manager
    def __getattr__(self,name):return getattr(self.manager.native_gate,name)
    def check_joints(self,robot,q,v,duration):
        m=self.manager
        actual=TimedReachability.check_joints(m.native_gate,robot,q,v,duration)
        if not actual.feasible:return actual
        if m.braking_active(robot.timestamp):return Reachability(False,'continuous_braking_in_progress')
        boundary=m.continuity_state(robot)
        if boundary is None:return Reachability(False,'confirmed_reference_boundary_unavailable')
        q0,v0,basis=boundary
        reference=replace(robot,joints=q0,joint_velocities=v0)
        checked=TimedReachability.check_joints(m.native_gate,reference,q,v,duration)
        evidence=deepcopy(actual.evidence)
        evidence['actual_state_reachability_retained']=True
        evidence['continuous_reference_proof']=dict(start_time=float(robot.timestamp),duration=float(duration),
            start_joints=q0.tolist(),start_velocity=v0.tolist(),end_joints=np.asarray(q).tolist(),
            end_velocity=np.asarray(v).tolist(),boundary_basis=basis,feasible=bool(checked.feasible),
            reason=checked.reason,evidence=deepcopy(checked.evidence))
        return Reachability(bool(checked.feasible),actual.reason if checked.feasible else
            'reference_continuity_'+checked.reason,actual.minimum_duration,actual.joints,actual.joint_velocities,evidence)


class _ContinuousGate:
    def __init__(self,manager,feedback):self.manager,self.feedback=manager,feedback
    def __getattr__(self,name):return getattr(self.feedback,name)
    def check(self,robot,pose,velocity,angular,duration):
        if duration>self.manager.config.dt+1e-8:
            return TimedReachability.check(self.manager.dual_gate,robot,pose,velocity,angular,duration)
        result=self.manager.check_stream(robot,duration)
        return result if result is not None else self.feedback.check(robot,pose,velocity,angular,duration)
    def check_joints(self,robot,q,v,duration):
        if duration>self.manager.config.dt+1e-8:
            return self.manager.dual_gate.check_joints(robot,q,v,duration)
        result=self.manager.check_stream(robot,duration)
        return result if result is not None else self.feedback.check_joints(robot,q,v,duration)


class ContinuousReferenceManager(PersistentReferenceManager):
    def __init__(self,controller,config=None,retarget_config=None):
        binding=continuous_reference_binding(config)
        base=dict(binding['normalized_config']);base.pop('version')
        super().__init__(controller,base)
        self.binding=binding;self.native_gate=controller.original_reachability
        self.dual_gate=_DualProofReachability(self)
        self.last_confirmed=None;self.confirmations=[];self.pending_dispatch=None
        self.plan=None;self.plan_candidate_id=None;self.plan_kind=None;self.invalidated=False
        self.events=[];self.current_robot=None;self.current_forecast=None
        self.retarget=CausalRetargetManager(controller,retarget_config)
        self.retarget_pending=None;self.confirmed_reissues_by_phase_index={}
        self.retarget_previous=None;self.absolute_phase_deadline=None

    def continuity_state(self,robot):
        issued=self.last_confirmed
        if issued is None or abs(issued['time']-robot.timestamp)>1e-8:return None
        if issued['nominal_velocity'] is not None:
            return np.asarray(issued['joints']),np.asarray(issued['nominal_velocity']),'backend_confirmed_reference_q_and_nominal_derivative'
        # A confirmed hold is an explicitly stationary SETPOINT, not a claim
        # that the real motor or its observed velocity is zero.
        if issued['command']=='hold':
            return np.asarray(issued['joints']),np.zeros_like(robot.joints),'confirmed_constant_hold_setpoint_derivative_zero_not_actual_velocity'
        return None

    def braking_active(self,time):
        return self.plan_kind=='braking' and self.plan is not None and time<self.plan.end-1e-8

    def invalidate(self,reason,time):
        self.invalidated=True;self.plan=None;self.plan_candidate_id=None;self.plan_kind=None
        self.events.append(dict(time=float(time),event='reference_invalidated',reason=reason,
            old_wallclock_plan_will_not_resume=True,phase_clock_reset=False,total_budget_extended=False))

    def confirm_actual_issued(self,*,time,joints,nominal_velocity,command,execution):
        q=np.asarray(joints,float);cert=command.get('feedforward_execution')
        if (execution.get('schema')!='r10_continuous_reference_direct_counted_setpoint_v2'
                or execution.get('tick_after')!=execution.get('tick_before',-2)+1
                or abs(execution.get('time_after',-100)-time)>1e-8
                or not np.array_equal(q,np.asarray(execution.get('applied_joint_reference')))
                or (self.last_confirmed is not None and time<=self.last_confirmed['time']+1e-8)):
            raise ValueError('only one actual counted dispatch callback can confirm a reference')
        if cert is not None:
            pending=self.pending_dispatch
            if (pending is None or cert.get('schema')!=SCHEMA or cert.get('proposal_id')!=pending['proposal_id']
                    or not np.array_equal(q,np.asarray(pending['reference_to_joints']))
                    or not np.array_equal(np.asarray(nominal_velocity),np.asarray(pending['reference_to_velocity']))
                    or abs(time-pending['time']-self.config.dt)>1e-8):
                raise ValueError('actual dispatch differs from pending continuous reference')
            velocity=np.asarray(nominal_velocity,float).tolist()
            pending['dispatch_confirmed']=True;pending['issued']=True;pending['confirmed_after_counted_tick_at']=float(time)
            self.invalidated=False
            if self.retarget_pending is not None and cert['candidate_id']==self.retarget_pending['candidate_id']:
                index=str(self.controller.phase_index)
                self.confirmed_reissues_by_phase_index[index]=self.confirmed_reissues_by_phase_index.get(index,0)+1
                self.retarget_pending['actual_first_reference_dispatch_confirmed_at']=float(time)
                self.retarget_pending['actual_dispatch_confirmed']=True
                self.retarget_pending=None
            for event in getattr(self.controller,'gripper_timing_events',[]):
                if event.get('time')==pending['time']:
                    event.update(actual_opening_request_issued=True,actual_opening_request=float(command['opening']),
                        actual_request_confirmed_after_counted_tick_at=float(time))
            for event in getattr(self.controller,'feedback_events',[]):
                if event.get('observed_time')==pending['time']:
                    event.update(actual_continuous_reference_issued=True,actual_reference_dispatch_confirmed_at=float(time))
        else:
            if nominal_velocity is not None:raise ValueError('ordinary commands cannot declare a verified nominal derivative')
            velocity=None
            if self.plan is not None:self.invalidate('ordinary_dispatch_breaks_verified_stream',time)
        self.last_confirmed=dict(time=float(time),joints=q.tolist(),nominal_velocity=velocity,
            command=command['command'],proposal_id=None if cert is None else cert['proposal_id'],
            reference_timestamp_is_after_counted_tick=True,nominal_derivative_is_not_actual_endpoint_velocity=True)
        self.confirmations.append(deepcopy(self.last_confirmed));self.pending_dispatch=None
        if (execution.get('actual_window_velocity_within_limits') is False
                or execution.get('actual_window_acceleration_within_limits') is False):
            self.qualification_failed=True
            if self.braking_active(time):
                self.events.append(dict(time=time,event='observed_violation_during_verified_braking',
                    qualification_failed=True,continue_same_continuous_braking_reference=True))
            else:self.invalidate('backend_observed_motion_limit_violation',time)

    def cancel_unissued_retarget(self):
        event=self.retarget_pending
        if event is None:return
        index=self.controller.phase_index
        counts=self.retarget.successful_reissues_by_phase_index
        counts[index]=max(0,counts.get(index,0)-1)
        if not counts[index]:counts.pop(index,None)
        for epoch in self.retarget.epochs.values():
            if event['candidate_id'] in epoch['known_candidate_ids']:
                epoch['known_candidate_ids'].discard(event['candidate_id']);epoch['issues']=max(0,epoch['issues']-1)
                previous=[e.get('issued_at',-1.)for e in self.retarget.events
                    if e is not event and e.get('actual_first_reference_dispatch_confirmed_at')is not None
                    and e['progress_epoch']==epoch['progress_epoch']]
                epoch['last_issue_at']=max([epoch['progress_epoch'],*previous])
        goal=self.controller.goal
        if goal is not None and goal.candidate_id==event['candidate_id'] and self.retarget_previous is not None:
            self.controller.goal,self.controller.trajectory=self.retarget_previous
        event.update(action='new_candidate_intent_not_issued',actual_dispatch_confirmed=False,
            reason='final_reference_not_dispatched_successful_reissue_not_consumed')
        self.retarget_pending=None;self.retarget_previous=None

    def adopt_selected_plan(self,robot):
        goal=self.controller.goal
        if goal is None:return False
        if self.plan is not None and self.plan_candidate_id==goal.candidate_id and not self.invalidated:return True
        proof=goal.reachability.evidence.get('continuous_reference_proof',{})
        if not proof.get('feasible') or abs(proof.get('start_time',-100)-robot.timestamp)>1e-8:return False
        if (not np.array_equal(np.asarray(proof['end_joints']),goal.reachability.joints)
                or not np.array_equal(np.asarray(proof['end_velocity']),goal.reachability.joint_velocities)):
            raise ValueError('selected proof belongs to a different retarget trial')
        self.plan=JointTrajectory(float(robot.timestamp),float(goal.arrival_time),np.asarray(proof['start_joints']),
            np.asarray(proof['start_velocity']),goal.reachability.joints.copy(),goal.reachability.joint_velocities.copy())
        self.plan_candidate_id=goal.candidate_id;self.plan_kind='tracking';self.controller.trajectory=self.plan
        self.invalidated=False
        return True

    def check_stream(self,robot,duration):
        if abs(duration-self.config.dt)>1e-8:return None
        if not self.braking_active(robot.timestamp):self.adopt_selected_plan(robot)
        plan=self.plan
        if plan is None or self.invalidated:return Reachability(False,'invalidated_reference_requires_continuous_replan')
        if robot.timestamp>=plan.end-1e-8 and np.any(abs(plan.v1)>1e-10):return None
        boundary=self.continuity_state(robot)
        if boundary is None:return Reachability(False,'confirmed_reference_boundary_unavailable')
        q0,v0,basis=boundary
        if robot.timestamp>=plan.end-1e-8:q1,v1=plan.q1.copy(),np.zeros_like(plan.v1)
        else:q1,v1=plan.at(min(robot.timestamp+self.config.dt,plan.end))
        expected_q,expected_v=plan.at(min(robot.timestamp,plan.end))
        if (not np.allclose(q0,expected_q,rtol=0,atol=1e-9)
                or not np.allclose(v0,expected_v,rtol=0,atol=1e-9)):
            self.invalidate('issued_reference_does_not_match_plan_boundary',robot.timestamp)
            return Reachability(False,'issued_reference_plan_discontinuity')
        if self.forecast_observed_until is None or not -1e-8<=robot.timestamp-self.forecast_observed_until<=self.controller.generator.config.maximum_observation_age:
            return Reachability(False,'current_causal_forecast_required')
        result=TimedReachability.check_joints(self.native_gate,replace(robot,joints=q0,joint_velocities=v0),q1,v1,self.config.dt)
        event=dict(schema=SCHEMA,time=float(robot.timestamp),candidate_id=str(self.plan_candidate_id),
            reference_gate_passed=bool(result.feasible),reference_gate_reason=result.reason,
            reference_gate_evidence=deepcopy(result.evidence),reference_from_joints=q0.tolist(),reference_from_velocity=v0.tolist(),
            reference_to_joints=q1.tolist(),reference_to_velocity=v1.tolist(),boundary_basis=basis,
            plan_start=self.plan.start,plan_end=self.plan.end,reference_kind=self.plan_kind,
            actual_tracking_guaranteed=False,phase_clock_reset=False,total_budget_extended=False,
            issued=False,dispatch_confirmed=False)
        event['proposal_id']=digest(dict(time=event['time'],candidate=event['candidate_id'],q=q1.tolist(),v=v1.tolist()))
        self.reference_events.append(event)
        if result.feasible:self.pending=event
        return result

    def make_brake(self,robot,remaining_seconds,reason):
        boundary=self.continuity_state(robot)
        if boundary is None:return False
        q0,v0,basis=boundary;goal=self.controller.goal
        available=min(float(remaining_seconds),10.-float(robot.timestamp))
        if goal is not None:available=min(available,max(0.,goal.selected_at+goal.phase_budget-robot.timestamp))
        elif self.absolute_phase_deadline is not None:
            available=min(available,max(0.,self.absolute_phase_deadline-robot.timestamp))
        trials=[]
        for duration in (.05,.15,.25,.4,.6,.9,1.1):
            if duration>available+1e-8:continue
            q1=q0+.5*v0*duration;v1=np.zeros_like(v0)
            reference=TimedReachability.check_joints(self.native_gate,replace(robot,joints=q0,joint_velocities=v0),q1,v1,duration)
            actual=TimedReachability.check_joints(self.native_gate,robot,q1,v1,duration)
            trials.append(dict(duration=duration,reference_reason=reference.reason,actual_reason=actual.reason))
            if reference.feasible and actual.feasible:
                self.plan=JointTrajectory(robot.timestamp,robot.timestamp+duration,q0.copy(),v0.copy(),q1,v1)
                self.plan_kind='braking';self.plan_candidate_id='continuous_brake:'+digest(dict(time=robot.timestamp,q=q0.tolist(),v=v0.tolist()))[:16]
                self.invalidated=False
                self.events.append(dict(time=robot.timestamp,event='continuous_braking_staged',reason=reason,
                    boundary_basis=basis,trials=trials,phase_clock_reset=False,total_budget_extended=False))
                return True
        self.events.append(dict(time=robot.timestamp,event='no_feasible_continuous_brake',reason=reason,trials=trials,
            acceleration_limited_stop_verified=False))
        return False

    def metrics(self):
        result=super().metrics();result.update(schema=SCHEMA,last_actual_issued_reference=deepcopy(self.last_confirmed),
            actual_dispatch_confirmations=deepcopy(self.confirmations),state_events=deepcopy(self.events),
            confirmed_reissues_by_phase_index=dict(self.confirmed_reissues_by_phase_index),
            reference_invalidated=self.invalidated,dispatch_confirmation_required=True)
        return result


def install_continuous_reference(actor,config=None,*,retarget_config=None):
    from .feedback_interaction import ObservedFeedbackController
    controller=actor.controller
    if (type(controller)is not ObservedFeedbackController or controller.feedback_step_calls or controller.goal is not None
            or hasattr(actor,'causal_retarget_manager')or hasattr(actor,'servo_response_compensator')
            or hasattr(actor,'continuous_reference_manager')or controller.ranking!='geometry'
            or controller.library.task_spec.task_name!='moving_basketball_in_hoop'):
        raise ValueError('install unified continuous reference once after feedback and before first act')
    manager=ContinuousReferenceManager(controller,config,retarget_config)
    old_step,old_act,old_metrics=controller.step,actor.act,actor.metrics_report
    controller.original_reachability=manager.dual_gate;controller.generator.reachability=manager.dual_gate
    controller.feedback_reachability=_ContinuousGate(manager,controller.feedback_reachability)

    def step(instance,state,forecast,remaining_seconds):
        manager.current_robot=state.robot;manager.current_forecast=forecast
        cutoff=forecast.metadata.get('observed_until')
        manager.forecast_observed_until=float(cutoff)if type(cutoff)in(float,int)else None
        event=None
        if not manager.braking_active(state.robot.timestamp) and not manager.invalidated:
            manager.retarget_previous=(instance.goal,instance.trajectory)
            event=manager.retarget.before_step(state,forecast,remaining_seconds)
            if event is not None and event['action']=='new_candidate_issue':
                event['reference_dual_gate_passed_before_commit']=True;event['actual_dispatch_confirmed']=False
                manager.retarget_pending=event;manager.adopt_selected_plan(state.robot)
        command=old_step(state,forecast,remaining_seconds)
        if instance.goal is not None:manager.absolute_phase_deadline=instance.goal.selected_at+instance.goal.phase_budget
        manager.retarget.register_current_goal()
        return command

    def act(instance,observation,robot,remaining_seconds):
        guard=manager.observe(robot);manager.pending=None;manager.pending_dispatch=None
        if guard['violations']:
            if manager.braking_active(robot.timestamp):
                manager.events.append(dict(time=robot.timestamp,event='observed_violation_during_verified_braking',
                    qualification_failed=True,continue_same_continuous_braking_reference=True))
            else:manager.invalidate('observed_window_limit_violation',robot.timestamp)
        old_opening=getattr(instance,'requested_opening',None)
        if old_opening is None:old_opening=float(robot.gripper_open)
        old_goal=controller.goal;old_close=controller._close_edge_issued_at
        if old_goal is not None:manager.absolute_phase_deadline=old_goal.selected_at+old_goal.phase_budget
        command=old_act(observation,robot,remaining_seconds)
        pending=manager.pending
        ready=instance.last_diagnostics.get('collision_scene_readiness',{}).get('ready')is True
        # Every non-stream hold/correction after a stream uses a continuous
        # reference brake, rather than an actual-q stop then virtual resume.
        if pending is None or command.get('command')!='joint_trajectory':
            if ready and manager.last_confirmed is not None:
                if manager.braking_active(robot.timestamp):
                    manager.check_stream(robot,.05);pending=manager.pending
                if pending is None:
                    manager.invalidate('ordinary_command_or_rejected_reference',robot.timestamp)
                    if manager.make_brake(robot,remaining_seconds,'final_command_requires_continuous_braking'):
                        manager.check_stream(robot,.05);pending=manager.pending
                if pending is not None:
                    command=dict(command='joint_trajectory',phase=command.get('phase'),opening=old_opening,
                        public_command_kind='joint_trajectory',reason='continuous_reference_braking')
                    instance.requested_opening=old_opening
                    controller._close_edge_issued_at=old_close if controller.goal is old_goal else None
        if (manager.retarget_pending is not None and
                (pending is None or pending['candidate_id']!=manager.retarget_pending['candidate_id'])):
            manager.cancel_unissued_retarget()
        if pending is not None and command.get('command')=='joint_trajectory':
            manager.pending_dispatch=pending
            certificate=deepcopy(pending);certificate.update(issued=False,dispatch_authorized=True,
                dispatch_confirmation_required=True,issued_at=float(robot.timestamp),
                reference_target_at=float(robot.timestamp+manager.config.dt))
            command=dict(command,joints=np.asarray(pending['reference_to_joints']),joint_velocities=np.asarray(pending['reference_to_velocity']),
                persistent_reference_verified=True,feedforward_execution=certificate,measured_feedback_command=False,
                actual_motor_tracking_guaranteed=False)
        elif ready and manager.last_confirmed is not None:
            command=dict(command='hold',phase=command.get('phase'),public_command_kind='hold',
                reason='no_feasible_continuous_reference_or_brake',acceleration_limited_stop_verified=False)
            instance.requested_opening=old_opening
            controller._close_edge_issued_at=old_close if controller.goal is old_goal else None
        feedback=[e for e in controller.feedback_events if e.get('observed_time')==float(robot.timestamp)]
        gripper=[e for e in controller.gripper_timing_events if e.get('time')==float(robot.timestamp)]
        if pending is not None:
            pending['inner_feedback_proposals']=deepcopy(feedback);pending['inner_gripper_proposals']=deepcopy(gripper)
        for event in feedback:
            proposed=bool(event.get('command_issued'))
            same=(command['command']=='joint_trajectory'and np.array_equal(command['joints'],event['bounded_joints']))
            event.update(inner_command_proposed=proposed,command_issued=bool(proposed and same),
                actual_continuous_reference_authorized=pending is not None,
                actual_returned_reference=None if command['command']!='joint_trajectory'else np.asarray(command['joints']).tolist())
        for event in gripper:
            event.update(inner_opening_proposal=event.get('commanded_opening'),
                commanded_opening=None if command['command']!='joint_trajectory'else float(command['opening']),
                actual_opening_request_authorized=command['command']=='joint_trajectory',
                actual_opening_request_issued=False,dispatch_confirmation_required=True)
        instance.last_command=command
        events=getattr(instance,'events',None)
        if events and events[-1].get('time')==float(robot.timestamp):
            events[-1].update(command=command['command'],phase=command.get('phase'),reason=command.get('reason'),
                public_command_kind=command.get('public_command_kind',command['command']))
        return command

    def metrics(instance):
        result=dict(old_metrics());result['continuous_reference_feedforward']=manager.metrics()
        result['causal_retarget']=manager.retarget.metrics();return result
    controller.step=MethodType(step,controller);actor.act=MethodType(act,actor);actor.metrics_report=MethodType(metrics,actor)
    actor.continuous_reference_manager=manager
    return actor
