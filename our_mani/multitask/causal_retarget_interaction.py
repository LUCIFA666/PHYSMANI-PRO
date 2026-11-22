"""Independent causal reissues of one committed basketball approach reference.

A reissue changes the committed forecast pose and nominal arrival trajectory.
The original template, phase/progress epoch, absolute deadline, suffix reserve
and attempt remain unchanged. New issue time is explicit and separate from the
legacy candidate.selected_at phase epoch. No physics, future truth or critic.
"""
from copy import deepcopy
from dataclasses import asdict,dataclass,replace
from pathlib import Path
from types import MethodType
import numpy as np

from ..contracts import Forecast
from ..intercept_planner import JointTrajectory
from ..reachability import TimedReachability
from .evaluation import digest,file_sha
from .interaction import _reference_at,_world_target
from .reference_time import COMMAND_ISSUE

SCHEMA='r10_causal_command_issue_retarget_v1'
ROOT=Path(__file__).resolve().parents[2]
TASK='moving_basketball_in_hoop'


@dataclass(frozen=True)
class CausalRetargetConfig:
    version:str=SCHEMA
    dt:float=.05
    minimum_issue_interval_seconds:float=.25
    scheduled_issue_interval_seconds:float=.4
    reference_shift_threshold_m:float=.03
    arrival_delays_seconds:tuple=(1.1,.9,.7,.5,.3,.15,.05)
    maximum_reissues_per_phase:int=4

    def __post_init__(self):
        values=(self.dt,self.minimum_issue_interval_seconds,self.scheduled_issue_interval_seconds,
            self.reference_shift_threshold_m,*self.arrival_delays_seconds)
        if (self.version!=SCHEMA or any(isinstance(value,bool) for value in values)
                or not np.isfinite(values).all() or self.dt!=.05
                or self.minimum_issue_interval_seconds<2*self.dt
                or self.scheduled_issue_interval_seconds<self.minimum_issue_interval_seconds
                or self.reference_shift_threshold_m<=0 or not self.arrival_delays_seconds
                or min(self.arrival_delays_seconds)<self.dt
                or tuple(sorted(set(self.arrival_delays_seconds),reverse=True))!=tuple(self.arrival_delays_seconds)
                or type(self.maximum_reissues_per_phase) is not int or not 1<=self.maximum_reissues_per_phase<=16):
            raise ValueError('declare finite causal retarget intervals, horizons and phase issue cap')


def retarget_binding(config=None):
    config=config if isinstance(config,CausalRetargetConfig) else CausalRetargetConfig(**(config or {}))
    normalized=asdict(config);normalized['arrival_delays_seconds']=list(normalized['arrival_delays_seconds'])
    sources=('our_mani/multitask/interaction.py','our_mani/multitask/feedback_interaction.py',
        'our_mani/multitask/reference_time.py','our_mani/reachability.py','our_mani/intercept_planner.py')
    return dict(normalized_config=normalized,configuration_sha256=digest(normalized),
        source=dict(path=str(Path(__file__).relative_to(ROOT)),sha256=file_sha(__file__)),
        original_sources_sha256={p:file_sha(ROOT/p) for p in sources},geometry_only=True,Value_required=False,
        instantaneous_endpoint_velocity_claimed=False,formal_release_issued=False)


class CausalRetargetManager:
    def __init__(self,controller,config=None):
        self.controller=controller
        self.config=config if isinstance(config,CausalRetargetConfig) else CausalRetargetConfig(**(config or {}))
        if controller.generator.config.dt!=self.config.dt or controller.mpc.config.dt!=self.config.dt:
            raise ValueError('retarget must use the original measured feedback/control tick')
        self.binding=retarget_binding(self.config);self.epochs={};self.events=[];self.last_time=None
        self.successful_reissues_by_phase_index={}

    def _epoch(self,goal):
        active=self.controller.attempts.active_by_phase.get(goal.phase)
        if active is None:raise ValueError('retarget must remain associated with an existing phase attempt')
        key=(goal.phase_index,goal.selected_at,active)
        if key not in self.epochs:
            self.epochs[key]=dict(root_candidate_id=goal.candidate_id,attempt_decision_id=active,
                progress_epoch=float(goal.selected_at),phase_deadline=float(goal.selected_at+goal.phase_budget),
                phase_budget=float(goal.phase_budget),suffix_budget=float(goal.suffix_budget),
                last_issue_at=float(goal.selected_at),last_attempt_at=float(goal.selected_at),issues=0,
                learned_template=goal.template,known_candidate_ids={goal.candidate_id})
        epoch=self.epochs[key]
        if (goal.template is not epoch['learned_template'] or goal.candidate_id not in epoch['known_candidate_ids']
                or goal.selected_at!=epoch['progress_epoch'] or goal.phase_budget!=epoch['phase_budget']
                or goal.suffix_budget!=epoch['suffix_budget']):
            raise ValueError('active retarget phase/template/budget/attempt changed unexpectedly')
        return epoch

    def register_current_goal(self):
        goal=self.controller.goal
        if goal is not None and self._eligible(goal):self._epoch(goal)

    @staticmethod
    def _eligible(goal):
        return (goal.phase=='approach' and goal.template.phase=='approach'
            and goal.template.reference_time_basis==COMMAND_ISSUE
            and goal.template.reference_kind=='translation_only' and goal.template.reference_role=='ball')

    def before_step(self,state,forecast,remaining_seconds):
        now=float(state.robot.timestamp);c=self.config;controller=self.controller;goal=controller.goal
        if not np.isfinite(remaining_seconds) or not 0<=remaining_seconds<=10.+1e-8:
            raise ValueError('retarget keeps the original counted task budget')
        if self.last_time is not None and now<=self.last_time+1e-8:
            raise ValueError('retarget observes exactly one monotonically increasing controller step per tick')
        self.last_time=now
        if goal is None or not self._eligible(goal):return None
        epoch=self._epoch(goal)
        if (self.successful_reissues_by_phase_index.get(goal.phase_index,0)>=c.maximum_reissues_per_phase
                or now-epoch['last_attempt_at']<c.minimum_issue_interval_seconds-1e-8):return None
        available=min(epoch['phase_deadline']-now,float(remaining_seconds)-epoch['suffix_budget'])
        delays=[float(d) for d in c.arrival_delays_seconds if d<=available+1e-8]
        if not delays:return None  # original controller retains its ordinary timeout handling
        event=dict(schema=SCHEMA,observed_at=now,root_candidate_id=epoch['root_candidate_id'],
            parent_candidate_id=goal.candidate_id,attempt_decision_id=epoch['attempt_decision_id'],
            phase=goal.phase,reference_id=goal.reference_id,progress_epoch=epoch['progress_epoch'],
            original_phase_deadline=epoch['phase_deadline'],original_suffix_budget=epoch['suffix_budget'],
            learned_reference_time_basis=goal.template.reference_time_basis,
            learned_reference_time_rule_sha256=goal.template.reference_time_rule_sha256,
            original_template_unchanged=True,phase_completion_claimed=False,phase_clock_reset=False,
            total_budget_extended=False,actual_motor_tracking_guaranteed=False,
            joint_velocity_basis='existing observed backward-window proxy; Hermite boundary is nominal',
            nominal_sampled_collision_semantics_unchanged=True,Value_used=False,trials=[])
        try:
            if not isinstance(forecast,Forecast):raise ValueError('typed causal forecast required')
            observed_until=forecast.metadata.get('observed_until')
            if (type(observed_until) not in (float,int) or not np.isfinite(observed_until)
                    or not -1e-8<=now-observed_until<=controller.generator.config.maximum_observation_age):
                raise ValueError('forecast must declare a current causal observation cutoff')
            current=controller.generator.observed_reference(state,goal.reference_id,goal.template,controller.origin)
            mode=state.perception_metadata.get('bodies',{}).get(goal.reference_id,{}).get('motion_mode')
            if mode!=goal.template.reference_mode or mode=='attached':raise ValueError('reference motion mode changed')
            probe=_reference_at(forecast,goal.reference_id,goal.template,now+delays[0]+goal.template.lag_seconds,controller.origin)[0]
            shift=float(np.linalg.norm(probe.position-goal.reference_pose.position))
            scheduled=now-epoch['last_issue_at']>=c.scheduled_issue_interval_seconds-1e-8
            if not scheduled and shift<c.reference_shift_threshold_m:return None
            epoch['last_attempt_at']=now
            event.update(trigger='scheduled_interval' if scheduled else 'forecast_reference_shift',reference_shift_m=shift,
                forecast_observed_until=float(observed_until),current_reference_observed_at=float(current.timestamp))
            accepted=[]
            for delay in delays:
                arrival=now+delay
                reference=_reference_at(forecast,goal.reference_id,goal.template,
                    arrival+goal.template.lag_seconds,controller.origin)[0]
                progress=float(np.clip((arrival-epoch['progress_epoch'])/goal.template.duration_seconds,0.,1.))
                pose,velocity,angular,_=_world_target(reference,np.zeros(3),np.zeros(3),goal.template,progress)
                trial=dict(arrival_time=arrival,delay=delay,template_progress_at_arrival=progress,
                    requested_pose_xyzw=pose.xyzw().tolist(),requested_linear_velocity=velocity.tolist(),
                    requested_angular_velocity=angular.tolist())
                event['trials'].append(trial)
                limits=controller.generator.config
                if (np.linalg.norm(velocity)>limits.maximum_linear_speed+1e-8
                        or np.linalg.norm(angular)>limits.maximum_angular_speed+1e-8):
                    trial.update(feasible=False,reason='original_target_twist_limit');continue
                controller.generator.bind_candidate_scene(state,forecast,goal.phase,goal.reference_id,goal.manipulated_ids,arrival)
                gate=TimedReachability.check(controller.original_reachability,state.robot,pose,velocity,angular,delay)
                trial.update(original_joint_gate=dict(feasible=bool(gate.feasible),reason=gate.reason,evidence=deepcopy(gate.evidence)))
                if not gate.feasible:
                    trial.update(feasible=False,reason='original_IK_joint_gate_rejected');continue
                collision=controller.generator.collision_check(pose=pose,time=arrival,phase=goal.phase,
                    observed_bodies=state.bodies,forecast=forecast,manipulated_ids=goal.manipulated_ids)
                if type(collision.get('feasible')) is not bool:raise ValueError('explicit original dynamic-collision feasibility required')
                trial['dynamic_collision']=deepcopy(collision)
                if not collision['feasible']:
                    trial.update(feasible=False,reason='original_dynamic_collision_rejected');continue
                utility=-limits.geometry_time_penalty*delay-limits.geometry_joint_effort_penalty*float(np.linalg.norm(gate.joints-state.robot.joints))
                trial.update(feasible=True,geometry_utility=utility)
                accepted.append((utility,delay,reference,pose,velocity,angular,current,gate,collision))
            if not accepted:
                event.update(action='reissue_rejected_original_candidate_retained',reason='no_original_gate_feasible_reissue')
            else:
                utility,delay,reference,pose,velocity,angular,current,gate,collision=max(accepted,key=lambda item:(item[0],-item[1]))
                issue_id='retarget:'+digest(dict(parent=goal.candidate_id,root=epoch['root_candidate_id'],issued_at=now,
                    issue_number=epoch['issues']+1,reference=reference.xyzw().tolist(),arrival=now+delay))[:20]
                new=replace(goal,candidate_id=issue_id,arrival_time=now+delay,pose=pose,linear_velocity=velocity,
                    angular_velocity=angular,reference_pose=reference,reference_velocity=np.zeros(3),
                    reference_angular_velocity=np.zeros(3),current_reference=current,reachability=gate,
                    geometry_utility=float(utility),collision_evidence=deepcopy(collision.get('evidence',{})))
                controller.goal=new
                controller.trajectory=JointTrajectory(now,new.arrival_time,state.robot.joints.copy(),
                    state.robot.joint_velocities.copy(),gate.joints.copy(),gate.joint_velocities.copy())
                # A new issue ID must not grant feedback slack a second time.
                controller._phase_slack_tokens.add((new.candidate_id,new.selected_at,len(controller.attempts.records)))
                epoch['known_candidate_ids'].add(issue_id);epoch['issues']+=1;epoch['last_issue_at']=now
                self.successful_reissues_by_phase_index[goal.phase_index]=self.successful_reissues_by_phase_index.get(goal.phase_index,0)+1
                event.update(action='new_candidate_issue',issued_at=now,candidate_id=issue_id,arrival_time=new.arrival_time,
                    phase_successful_reissue_number=self.successful_reissues_by_phase_index[goal.phase_index],
                    attempt_successful_reissue_number=epoch['issues'],
                    selected_at_is_preserved_phase_epoch=True,legacy_selected_at=new.selected_at,
                    phase_deadline_after=new.selected_at+new.phase_budget,template_progress_now=float(np.clip(
                        (now-new.selected_at)/new.template.duration_seconds,0.,1.)),reference_pose=reference.xyzw().tolist(),
                    reference_is_new_causal_forecast_commitment=True,original_attempt_record_rewritten=False)
        except (ValueError,KeyError,IndexError) as error:
            epoch['last_attempt_at']=now
            event.update(action='reissue_rejected_original_candidate_retained',reason=str(error))
        self.events.append(event)
        return event

    def metrics(self):
        return dict(schema=SCHEMA,**deepcopy(self.binding),events=deepcopy(self.events),
            successful_reissues_by_phase_index=dict(self.successful_reissues_by_phase_index),
            installed_before_first_step=True,native_or_GPU_launched_by_module=False,
            future_native_state_used=False,phase_completion_claimed=False,total_budget_extended=False,
            legacy_selection_events_not_relabelled=True,
            formal_trace_consumer_for_new_issue_IDs_still_requires_separate_review=True)


def install_causal_retarget(actor,config=None):
    """Install after measured feedback and before the outer servo/braking layer."""
    from .feedback_interaction import ObservedFeedbackController
    controller=actor.controller
    if (type(controller) is not ObservedFeedbackController or controller.feedback_step_calls or controller.attempts.records
            or controller.goal is not None or hasattr(actor,'servo_response_compensator')
            or hasattr(actor,'causal_retarget_manager') or controller.ranking!='geometry'
            or controller.success_model is not None or controller.library.task_spec.task_name!=TASK
            or getattr(controller.step,'__func__',None) is not ObservedFeedbackController.step):
        raise ValueError('install once on normal basketball geometry after feedback, before outer servo and first step')
    manager=CausalRetargetManager(controller,config);original_step=controller.step;original_metrics=actor.metrics_report
    def step(instance,state,forecast,remaining_seconds):
        event=manager.before_step(state,forecast,remaining_seconds)
        command=original_step(state,forecast,remaining_seconds)
        manager.register_current_goal()
        if event is not None:instance.last_diagnostics['causal_retarget']=deepcopy(event)
        return command
    def metrics(instance):
        result=dict(original_metrics());result['causal_retarget']=manager.metrics();return result
    controller.step=MethodType(step,controller);actor.metrics_report=MethodType(metrics,actor)
    actor.causal_retarget_manager=manager
    return actor
