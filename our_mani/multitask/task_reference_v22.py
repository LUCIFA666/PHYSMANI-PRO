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
from .interaction import _reference_at,_world_target
from .specs import TASK_SPECS
from ..contracts import Reachability,Forecast
from ..intercept_planner import JointTrajectory
from ..reachability import TimedReachability
from scipy.spatial.transform import Rotation

SCHEMA='r10_continuous_verified_reference_feedforward_v22'
ROOT=Path(__file__).resolve().parents[2]


def continuous_reference_binding(config=None):
    raw=dict(config or {});version=raw.pop('version',SCHEMA)
    if version!=SCHEMA:raise ValueError('wrong continuous-reference version')
    from .bounded_approach_v19 import configuration as approach_configuration
    approach=approach_configuration(raw.pop('approach_transit',None))
    delays=raw.pop('rolling_arrival_delays_seconds',[.15,.25,.4,.6,.9,1.1])
    if (not isinstance(delays,(list,tuple))or not delays or any(isinstance(x,bool)for x in delays)or not np.isfinite(delays).all()
            or list(sorted(set(delays)))!=list(delays)or min(delays)<.05 or max(delays)>1.1):
        raise ValueError('rolling horizons must be declared within the unchanged causal forecast horizon')
    base=PersistentReferenceConfig(**raw)
    normalized=dict(version=SCHEMA,dt=base.dt,observed_velocity_tolerance=base.observed_velocity_tolerance,
        observed_acceleration_tolerance=base.observed_acceleration_tolerance,
        rolling_arrival_delays_seconds=[float(x)for x in delays],approach_transit=approach)
    sources=('our_mani/multitask/persistent_reference_feedforward.py',
        'our_mani/multitask/causal_retarget_interaction.py','our_mani/multitask/feedback_interaction.py',
        'our_mani/multitask/interaction.py','our_mani/reachability.py','our_mani/intercept_planner.py',
        'our_mani/multitask/bounded_approach_v19.py')
    return dict(normalized_config=normalized,configuration_sha256=digest(normalized),
        source=dict(path=str(Path(__file__).relative_to(ROOT)),sha256=file_sha(__file__)),
        original_sources_sha256={p:file_sha(ROOT/p)for p in sources},actual_tracking_guaranteed=False)


class _DualProofReachability:
    def __init__(self,manager):self.manager=manager
    def __getattr__(self,name):return getattr(self.manager.native_gate,name)
    def check_joints(self,robot,q,v,duration):
        m=self.manager
        phase=m.controller.library.phase_order[m.controller.phase_index] if m.controller.phase_index<len(m.controller.library.phase_order) else None
        if phase in ('approach','acquire','lift','deposit') and duration>m.config.dt+1e-8:
            v=np.zeros_like(np.asarray(v,float))  # explicit stop-at-intercept boundary, same complete gates
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
            c=self.manager.controller
            if c.phase_index<len(c.library.phase_order) and c.library.phase_order[c.phase_index]=='deposit':
                if duration<min(t.duration_seconds for t in c.library.for_phase('deposit'))-1e-8:
                    return Reachability(False,'deposit_arrival_precedes_complete_TRAIN_curve')
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
        approach_configuration=base.pop('approach_transit')
        self.rolling_delays=base.pop('rolling_arrival_delays_seconds')
        super().__init__(controller,base)
        self.binding=binding;self.native_gate=controller.original_reachability
        self.dual_gate=_DualProofReachability(self)
        self.last_confirmed=None;self.confirmations=[];self.pending_dispatch=None
        self.plan=None;self.plan_candidate_id=None;self.plan_kind=None;self.invalidated=False
        self.events=[];self.current_robot=None;self.current_forecast=None
        self.retarget=CausalRetargetManager(controller,retarget_config)
        self.retarget_pending=None;self.confirmed_reissues_by_phase_index={}
        self.retarget_previous=None;self.absolute_phase_deadline=None
        self.current_state=None;self.rolling_events=[];self.handover_events=[];self.retired_brake_goal=None
        from .bounded_approach_v19 import BoundedApproach
        self.approach_transit=BoundedApproach(self,approach_configuration)

    def capture_stopped(self,robot):
        boundary=self.continuity_state(robot)
        if boundary is None:return False
        _,velocity,_=boundary
        bound=np.asarray(self.native_gate.model.acceleration_limits)*self.config.dt
        return bool(np.all(np.abs(velocity)<=self.config.observed_velocity_tolerance)
            and np.all(np.abs(robot.joint_velocities)<=bound-self.config.observed_velocity_tolerance))

    def capture_at_current_grasp(self,robot):
        goal=self.controller.goal;state=self.current_state;config=self.controller.gripper_timing_config
        if goal is None or goal.phase!='acquire' or state is None or config is None:return False
        if self.controller._close_edge_issued_at is not None:return True
        progress=float(np.clip((robot.timestamp-goal.selected_at)/goal.template.duration_seconds,0.,1.))
        if goal.template.at(progress)[3]>=.5:return False
        try:
            current=self.controller.generator.observed_reference(state,goal.reference_id,goal.template,self.controller.origin)
            target=_world_target(current.pose,np.zeros(3),np.zeros(3),goal.template,1.)[0]
            return bool(np.linalg.norm(robot.pose.position-target.position)<=config.close_position_tolerance_m
                and Rotation.from_matrix(robot.pose.rotation@target.rotation.T).magnitude()<=config.close_orientation_tolerance_rad)
        except (ValueError,KeyError,IndexError):return False

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
        if (execution.get('schema')!='r10_continuous_reference_direct_counted_setpoint_v22'
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
        previous_opening=(self.last_confirmed or {}).get('opening',0.)
        self.last_confirmed=dict(time=float(time),joints=q.tolist(),nominal_velocity=velocity,
            opening=float(command.get('opening',previous_opening)),
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
        if (self.plan is not None and robot.timestamp>=self.plan.end-1e-8
                and not self.invalidated and self.controller.goal is not None
                and self.controller.goal.template.reference_time_basis=='current'
                and self.controller.goal.phase in ('approach','acquire')):
            # A completed safety brake cannot replace a moving-target policy
            # indefinitely. Every renewed stream needs both complete proofs.
            captured=(self.capture_at_current_grasp(robot) if self.controller.goal.phase=='acquire'
                else self.capture_at_current_endpoint(robot))
            if not (self.capture_stopped(robot) and captured):
                if not self.roll_current_reference(robot):return None
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

    def roll_current_reference(self,robot):
        controller=self.controller;goal=controller.goal;state=self.current_state;forecast=self.current_forecast
        if goal is None or state is None or forecast is None or self.braking_active(robot.timestamp):return False
        now=float(robot.timestamp)
        available=min(goal.selected_at+goal.phase_budget-now,10.-now-float(goal.suffix_budget))
        event=dict(time=now,event='current_acquire_reference_roll',candidate_id=goal.candidate_id,
            phase_epoch=goal.selected_at,absolute_phase_deadline=goal.selected_at+goal.phase_budget,
            phase_clock_reset=False,total_budget_extended=False,trials=[],committed=False)
        accepted=[]
        try:
            if not isinstance(forecast,Forecast):raise ValueError('typed current causal forecast required')
            cutoff=forecast.metadata.get('observed_until')
            if (type(cutoff)not in(float,int)or not np.isfinite(cutoff)
                    or not -1e-8<=now-cutoff<=controller.generator.config.maximum_observation_age):
                raise ValueError('current causal forecast cutoff required before rolling commitment')
            current=controller.generator.observed_reference(state,goal.reference_id,goal.template,controller.origin)
            mode=state.perception_metadata.get('bodies',{}).get(goal.reference_id,{}).get('motion_mode')
            if mode!=goal.template.reference_mode or mode=='attached':
                raise ValueError('current reference mode differs; original public completion must resolve contact')
            for delay in self.rolling_delays:
                if delay>available+1e-8:continue
                arrival=now+delay
                ref,v,w=_reference_at(forecast,goal.reference_id,goal.template,
                    arrival+goal.template.lag_seconds,controller.origin)
                progress=float(np.clip((arrival-goal.selected_at)/goal.template.duration_seconds,0.,1.))
                pose,velocity,angular,_=_world_target(ref,v,w,goal.template,progress)
                trial=dict(delay=delay,arrival=arrival,target_xyzw=pose.xyzw().tolist(),progress=progress)
                event['trials'].append(trial)
                limits=controller.generator.config
                if np.linalg.norm(velocity)>limits.maximum_linear_speed+1e-8 or np.linalg.norm(angular)>limits.maximum_angular_speed+1e-8:
                    trial.update(feasible=False,reason='original_target_twist_limit');continue
                controller.generator.bind_candidate_scene(state,forecast,goal.phase,goal.reference_id,goal.manipulated_ids,arrival)
                checked=TimedReachability.check(self.dual_gate,robot,pose,velocity,angular,delay)
                trial.update(feasible=bool(checked.feasible),reason=checked.reason,evidence=deepcopy(checked.evidence))
                if not checked.feasible:continue
                collision=controller.generator.collision_check(pose=pose,time=arrival,phase=goal.phase,
                    observed_bodies=state.bodies,forecast=forecast,manipulated_ids=goal.manipulated_ids)
                trial['dynamic_collision']=deepcopy(collision)
                if collision.get('feasible')is not True:
                    trial.update(feasible=False,reason='original_dynamic_collision_rejected');continue
                utility=-limits.geometry_time_penalty*delay-limits.geometry_joint_effort_penalty*float(np.linalg.norm(checked.joints-robot.joints))
                accepted.append((utility,delay,checked,pose))
            if accepted:
                _,delay,checked,pose=max(accepted,key=lambda item:(item[0],-item[1]))
                controller.generator.bind_candidate_scene(state,forecast,goal.phase,goal.reference_id,goal.manipulated_ids,now+delay)
                proof=checked.evidence['continuous_reference_proof']
                self.plan=JointTrajectory(now,now+delay,np.asarray(proof['start_joints']),np.asarray(proof['start_velocity']),
                    checked.joints.copy(),checked.joint_velocities.copy())
                self.plan_kind='current_acquire_rolling';self.plan_candidate_id='rolling_acquire:'+digest(dict(
                    parent=goal.candidate_id,time=now,delay=delay,q=checked.joints.tolist()))[:16]
                self.invalidated=False;controller.trajectory=self.plan
                event.update(committed=True,reference_stream_id=self.plan_candidate_id,arrival=now+delay,
                    target_xyzw=pose.xyzw().tolist(),actual_and_reference_full_gate_feasible=True,
                    fresh_current_reference_observed_at=float(current.timestamp))
        except (ValueError,KeyError,IndexError)as error:event['error']=str(error)
        self.rolling_events.append(event)
        return event['committed']

    def capture_at_current_endpoint(self,robot):
        goal=self.controller.goal;state=self.current_state
        if goal is None or state is None:return False
        try:
            reference=self.controller.generator.observed_reference(state,goal.reference_id,goal.template,self.controller.origin)
            target=_world_target(reference.pose,np.zeros(3),np.zeros(3),goal.template,1.)[0]
            return bool(np.linalg.norm(robot.pose.position-target.position)<=goal.template.position_tolerance_m
                and Rotation.from_matrix(robot.pose.rotation@target.rotation.T).magnitude()<=goal.template.orientation_tolerance_rad)
        except (ValueError,KeyError,IndexError):return False

    def make_brake(self,robot,remaining_seconds,reason):
        boundary=self.continuity_state(robot)
        if boundary is None:return False
        q0,v0,basis=boundary;goal=self.controller.goal
        available=min(float(remaining_seconds),10.-float(robot.timestamp))
        # Safety braking is not a task candidate or phase continuation. A failed
        # phase cannot remove the remaining physical time needed to stop.
        # Task selection/rolling deadlines are unchanged; global 10s still caps braking.
        trials=[]
        for duration in (.05,.15,.25,.4,.6,.9,1.1):
            if duration>available+1e-8:continue
            if self.current_state is not None and self.current_forecast is not None:
                phase=(self.controller.library.phase_order[self.controller.phase_index]
                    if self.controller.phase_index<len(self.controller.library.phase_order) else 'verify')
                intent=goal if goal is not None else self.retired_brake_goal
                same=intent is not None and intent.phase==phase
                self.controller.generator.bind_candidate_scene(self.current_state,self.current_forecast,phase,
                    intent.reference_id if same else '',intent.manipulated_ids if same else (),robot.timestamp+duration)
            q1=q0+.5*v0*duration;v1=np.zeros_like(v0)
            reference=TimedReachability.check_joints(self.native_gate,replace(robot,joints=q0,joint_velocities=v0),q1,v1,duration)
            actual=TimedReachability.check_joints(self.native_gate,robot,q1,v1,duration)
            trials.append(dict(duration=duration,reference_reason=reference.reason,actual_reason=actual.reason))
            if reference.feasible and actual.feasible:
                self.plan=JointTrajectory(robot.timestamp,robot.timestamp+duration,q0.copy(),v0.copy(),q1,v1)
                self.plan_kind='braking';self.plan_candidate_id='continuous_brake:'+digest(dict(time=robot.timestamp,q=q0.tolist(),v=v0.tolist()))[:16]
                self.invalidated=False
                self.events.append(dict(time=robot.timestamp,event='continuous_braking_staged',reason=reason,
                    boundary_basis=basis,trials=trials,phase_clock_reset=False,total_budget_extended=False,
                    safety_brake_is_not_task_phase_continuation=True,task_candidate_phase_deadline_unchanged=True,
                    collision_binding='current public frame and same-phase intent only'))
                return True
        self.events.append(dict(time=robot.timestamp,event='no_feasible_continuous_brake',reason=reason,trials=trials,
            acceleration_limited_stop_verified=False))
        return False

    def metrics(self):
        result=super().metrics();result.update(schema=SCHEMA,last_actual_issued_reference=deepcopy(self.last_confirmed),
            actual_dispatch_confirmations=deepcopy(self.confirmations),state_events=deepcopy(self.events),
            confirmed_reissues_by_phase_index=dict(self.confirmed_reissues_by_phase_index),
            reference_invalidated=self.invalidated,dispatch_confirmation_required=True)
        result['current_acquire_rolling_events']=deepcopy(self.rolling_events)
        result['acquire_lift_handover_events']=deepcopy(self.handover_events)
        result['bounded_approach_transit_v18']=self.approach_transit.metrics()
        return result


def install_continuous_reference(actor,config=None,*,retarget_config=None):
    from .feedback_interaction import ObservedFeedbackController
    controller=actor.controller
    if (type(controller)is not ObservedFeedbackController or controller.feedback_step_calls or controller.goal is not None
            or hasattr(actor,'causal_retarget_manager')or hasattr(actor,'servo_response_compensator')
            or hasattr(actor,'continuous_reference_manager')or controller.ranking!='geometry'
            or controller.library.task_spec.task_name not in TASK_SPECS):
        raise ValueError('install unified continuous reference once after feedback and before first act')
    manager=ContinuousReferenceManager(controller,config,retarget_config)
    old_step,old_act,old_metrics=controller.step,actor.act,actor.metrics_report
    old_gripper_opening=controller._gripper_opening
    controller.original_reachability=manager.dual_gate;controller.generator.reachability=manager.dual_gate
    controller.feedback_reachability=_ContinuousGate(manager,controller.feedback_reachability)

    def step(instance,state,forecast,remaining_seconds):
        manager.current_robot=state.robot;manager.current_forecast=forecast;manager.current_state=state
        cutoff=forecast.metadata.get('observed_until')
        manager.forecast_observed_until=float(cutoff)if type(cutoff)in(float,int)else None
        event=None
        if not manager.braking_active(state.robot.timestamp) and not manager.invalidated:
            manager.retarget_previous=(instance.goal,instance.trajectory)
            event=manager.retarget.before_step(state,forecast,remaining_seconds)
            if event is not None and event['action']=='new_candidate_issue':
                event['reference_dual_gate_passed_before_commit']=True;event['actual_dispatch_confirmed']=False
                manager.retarget_pending=event;manager.adopt_selected_plan(state.robot)
        previous_goal=instance.goal
        previous_phase_index=instance.phase_index
        command=old_step(state,forecast,remaining_seconds)
        # Public acquire completion clears the goal and returns a bookkeeping
        # hold. Select lift now, from the confirmed qref/vref, instead of sending
        # that actual-q hold to a moving native motor for one physical tick.
        # This second controller decision neither observes nor advances physics.
        if (previous_goal is not None and previous_goal.phase=='acquire'
                and instance.phase_index==previous_phase_index+1
                and instance.phase_index<len(instance.library.phase_order)
                and instance.library.phase_order[instance.phase_index]=='lift'
                and command.get('reason')=='advance_to_next_TRAIN_fitted_skill'):
            manager.cancel_unissued_retarget()
            manager.pending=None
            manager.invalidate('public_acquire_completed_requires_fresh_lift_plan',state.robot.timestamp)
            command=old_step(state,forecast,remaining_seconds)
            manager.handover_events.append(dict(time=float(state.robot.timestamp),
                from_phase='acquire',to_phase='lift',public_acquire_completion_required=True,
                retired_phase_deadline=float(previous_goal.selected_at+previous_goal.phase_budget),
                next_candidate_id=None if instance.goal is None else instance.goal.candidate_id,
                next_phase_deadline=None if instance.goal is None else float(instance.goal.selected_at+instance.goal.phase_budget),
                command=command.get('command'),reason=command.get('reason'),
                controller_diagnostics=deepcopy(instance.last_diagnostics),
                same_current_state_and_forecast=True,additional_physics_ticks=0,
                existing_phase_deadline_reset=False,total_budget_extended=False))
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
        if old_goal is not None:manager.retired_brake_goal=old_goal
        if old_goal is not None:manager.absolute_phase_deadline=old_goal.selected_at+old_goal.phase_budget
        command=old_act(observation,robot,remaining_seconds)
        pending=manager.pending
        ready=instance.last_diagnostics.get('collision_scene_readiness',{}).get('ready')is True
        if (pending is None and command.get('command')=='hold'
                and command.get('reason')=='no_feasible_fitted_interaction_candidate' and ready
                and manager.approach_transit.try_approach(robot,remaining_seconds)):
            pending=manager.pending
            command=dict(command='joint_trajectory',phase=command.get('phase'),opening=old_opening,
                public_command_kind='joint_trajectory',reason='bounded_approach_before_full_arrival')
            instance.requested_opening=old_opening
            controller._close_edge_issued_at=old_close if controller.goal is old_goal else None
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
    def gripper_opening(instance,state,requested,final_target):
        goal=instance.goal;learned=None;progress=None
        if goal is not None:
            progress=float(np.clip((state.robot.timestamp-goal.selected_at)/goal.template.duration_seconds,0.,1.))
            learned=goal.template.at(progress)[3]
            if goal.phase=='acquire':
                if learned<.5 and requested>=.5:requested=learned
            else:
                # The contact sensor's boolean opening proxy can remain true
                # after a close request has attached the object. It is not an
                # actuator intent. Lift/deposit retain the TRAIN opening curve.
                requested=learned
        capture_ready=manager.capture_stopped(state.robot)
        deferred=(goal is not None and goal.phase=='acquire' and requested<.5
            and instance._close_edge_issued_at is None and not state.robot.contact and not capture_ready)
        if deferred:requested=1.
        release_position_error=release_rotation_error=None
        release_ready=None
        already_open=bool(manager.last_confirmed is not None and manager.last_confirmed.get('opening',0.)>=.5)
        if goal is not None and goal.phase=='deposit' and requested>=.5 and not already_open:
            release_ready=False
            # The legacy final_target argument is acquire-only. Deposit owns
            # its separately committed TRAIN command-goal reference instead.
            release_target=final_target
            if release_target is None and getattr(goal,'reference_pose',None) is not None:
                release_target=_world_target(goal.reference_pose,np.zeros(3),np.zeros(3),goal.template,1.)[0]
            if release_target is not None:
                release_position_error=float(np.linalg.norm(state.robot.pose.position-release_target.position))
                release_rotation_error=float(Rotation.from_matrix(state.robot.pose.rotation@release_target.rotation.T).magnitude())
                release_ready=bool(capture_ready
                    and release_position_error<=goal.template.position_tolerance_m
                    and release_rotation_error<=goal.template.orientation_tolerance_rad)
            if not release_ready:requested=0.
        result=old_gripper_opening(state,requested,final_target)
        if goal is not None and goal.phase=='deposit' and instance.gripper_timing_events:
            instance.gripper_timing_events[-1].update(release_stop_and_TRAIN_endpoint_gate_passed=release_ready,
                release_position_error_m=release_position_error,release_orientation_error_rad=release_rotation_error,
                release_position_tolerance_m=goal.template.position_tolerance_m,
                release_orientation_tolerance_rad=goal.template.orientation_tolerance_rad,
                release_deferred_until_actual_endpoint_and_verified_stop=release_ready is False)
        if goal is not None and goal.phase=='acquire' and instance.gripper_timing_events:
            instance.gripper_timing_events[-1].update(capture_stop_gate_passed=capture_ready,close_deferred_until_verified_stop=deferred,
                original_SE3_close_gate_retained=True,measured_stop_velocity_bound_rad_s=(np.asarray(manager.native_gate.model.acceleration_limits)*manager.config.dt).tolist())
        if goal is not None and instance.gripper_timing_events:
            instance.gripper_timing_events[-1].update(learned_phase_progress=progress,
                learned_phase_opening=learned,opening_intent_source='TRAIN_phase_curve_with_original_acquire_gate')
        if goal is not None and goal.phase=='acquire'and instance.gripper_timing_events:
            instance.gripper_timing_events[-1].update(learned_acquire_progress=progress,learned_opening=learned,
                original_observed_close_gate_unchanged=True,active_reference_does_not_force_open_after_learned_close=True)
        return result
    controller._gripper_opening=MethodType(gripper_opening,controller)
    controller.step=MethodType(step,controller);actor.act=MethodType(act,actor);actor.metrics_report=MethodType(metrics,actor)
    actor.continuous_reference_manager=manager
    return actor
