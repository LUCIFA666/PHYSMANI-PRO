"""Independent v4 active-visible-grasp development protocol, no force proof.

Legacy estimation is diagnostic only. FREE uses frozen public GRU; unknown
contact uses no FREE forecast. A one-shot fixed-deadline, model-selected probe
can establish visible vertical co-motion, followed by current visual hold.
"""
from copy import deepcopy
from dataclasses import dataclass,asdict
from pathlib import Path
import time
import numpy as np
from ..contracts import ObjectState
from ..local_mpc_20261102.records import append,write_new,quantiles
from .public_motion_bridge import PublicRobotSensors
from .public_shadow import PublicShadowBackend,sanitized_robot
from .public_grasp import (AttachmentAwarePublicBridge,PublicGraspConfig,PublicLiftHoldMetric,qualify_public_grasp_candidates)
from .public_grasp_state import PublicGraspStateAssembler
from .active_visual_attachment import ActiveVisualAttachment,ActiveVisualConfig
from .public_grasp_probe import VerificationProbePlanner,qualify_probe_candidates
from .response_factory import build_planner as build_response_planner
from .forecast_control import BoundForecastExecutionModel,ForecastJointGoalCEMPlanner
from .joint_goal_cem import JointGoalCEMPlanner
from .temporal_candidates import CurrentWindowCandidates

ROOT=Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ActiveVisualGraspConfig(PublicGraspConfig):
    verification_probe_enabled:bool=True
    def __post_init__(self):
        super().__post_init__()
        if self.verification_probe_enabled is not True or self.immediate_expiry is not True:
            raise ValueError('v4 explicitly requires active verification and current-window FREE candidates')


class ActiveVisualPublicBridge(AttachmentAwarePublicBridge):
    """Same real tracker/cache/GRU; only independent v4 evidence controls mode."""
    def __init__(self,*,active_config=None):
        super().__init__();self.active=ActiveVisualAttachment(active_config)
        self.latest_legacy_diagnostic=None
    def reset(self):
        super().reset()
        if hasattr(self,'active'):self.active.reset()
        self.latest_legacy_diagnostic=None
    def observe(self,observation,robot,*,close_requested):
        if not isinstance(robot,PublicRobotSensors):robot=PublicRobotSensors.from_mapping(robot)
        if type(close_requested) is not bool:raise TypeError('actual observed command intent required')
        target=None;diagnostic=None
        try:target,diagnostic=self.tracker.observe(observation)
        except (ValueError,RuntimeError):pass
        self.cache.stage_public_target(target,timestamp=robot.timestamp,diagnostic=diagnostic)
        if close_requested and not self.active.attempt_active:
            previous=self._last_open_visual
            self.close_visual_baseline=(previous if self._last_close_requested is False and previous is not None
                and abs(robot.timestamp-previous.timestamp-.05)<=1e-6 else None)
        # Both consume the same real target/robot sample; legacy emits no control
        # evidence, even when its old hypotheses happen to be positive.
        legacy=self.estimator.observe(target,robot,close_requested=close_requested,orientation_observable=False)
        active=self.active.observe(target,robot,close_requested=close_requested)
        result=self.bridge.observe(observation,robot,close_requested=close_requested,evidence=active['evidence'])
        result.update(attachment_estimate=active,legacy_attachment_diagnostic=legacy,
            active_visual_protocol='active_vertical_visual_comotion_v4',legacy_attachment_controls_bridge=False,
            public_preclose_visual_baseline=self.close_visual_baseline,approved_tracker_cache=self.binding.receipt(),
            tracker_injection_fact_retained=True)
        self.binding.bind_current(result,target)
        if not close_requested and isinstance(target,ObjectState) and target.visible and target.confidence>=.5:self._last_open_visual=target
        self._last_close_requested=close_requested;self._last_time=robot.timestamp
        self.latest_attachment=active;self.latest_legacy_diagnostic=legacy;self.latest_result=result
        return result


class ObservedPublicGraspV4Backend(PublicShadowBackend):
    def __init__(self,backend,output,*,public_bridge=None,recorder=None,save_frames=True,grasp_config=None):
        wrapper=public_bridge if public_bridge is not None else ActiveVisualPublicBridge()
        if type(wrapper) is not ActiveVisualPublicBridge:raise TypeError('real independently scoped v4 bridge required')
        wrapper.binding.validate_dependencies()
        self.assembler=PublicGraspStateAssembler(wrapper.binding);self.latest_robot=None
        self.public_hold_metric=PublicLiftHoldMetric(grasp_config or ActiveVisualGraspConfig())
        super().__init__(backend,output,bridge=wrapper,recorder=recorder,save_frames=save_frames,
            controller_uses_public_forecast=True,observer_role='grasp_prototype',attachment_estimator_used=True)
    def sample(self,state=None):
        current=self.backend.read_state() if state is None else state;robot=sanitized_robot(current)
        result=super().sample(current);self.assembler.observe_robot(robot);self.latest_robot=robot
        self.public_hold_metric.observe(result,robot,self.bridge.latest_attachment,self.bridge.close_visual_baseline)
        return result
    def receipt(self):
        result=super().receipt();result.update(active_visual_protocol='active_vertical_visual_comotion_v4',
            legacy_attachment_controls_bridge=False,attachment_is_visible_operational_comotion=True,
            force_or_frictional_grasp_proof=False,Native_labels_used_by_policy=False,
            external_synchronous_Z_motion_ruled_out=False,common_visual_bias_ruled_out=False)
        return result


class ActiveVisualGraspCoordinator:
    def __init__(self,core,helper,assembler,active,*,config=None):
        if type(active) is not ActiveVisualAttachment:raise TypeError('independent active visual state machine required')
        self.core,self.helper,self.assembler,self.active=core,helper,assembler,active
        self.config=config or ActiveVisualGraspConfig();self.geometry=CurrentWindowCandidates(horizon=.8)
        self.mode='UNAVAILABLE';self.warm=None;self.phase=None;self.candidates=None
        self.pending_started=None;self.reopening_active=False;self.hold_tcp_quaternion=None;self.mode_events=[]
    def _mode(self,new,now):
        if new!=self.mode:
            self.mode_events.append(dict(time=now,previous=self.mode,current=new,warm_cleared=True))
            self.warm=None;self.mode=new
            if new!='ATTACHED_PUBLIC':self.hold_tcp_quaternion=None
    def note_executed(self,action,robot):
        if action['opening']==0. and self.pending_started is None and self.mode=='FREE_GRU':self.pending_started=robot.timestamp
    def _verification_components(self,backend,state,planning,execution,spec):
        candidates,qualification=qualify_probe_candidates(backend,state,planning,self.helper,spec)
        planner=VerificationProbePlanner(self.core.config,execution_model=execution,goal_helper=self.helper,
            constraint_cost=self.core.planner.constraint_cost,terminal_config=self.core.terminal_config,probe_spec=spec)
        return candidates,qualification,planner
    def decide(self,backend,result,robot,h):
        if self.assembler.robot_twist is None or not self.assembler.robot_twist[2]:
            self._mode('UNAVAILABLE',robot.timestamp);return dict(mode=self.mode,state=None,reason='no_real_robot_twist')
        state=self.assembler.build(result,robot,h,phase_state=self.phase)
        spec=None
        if state is not None and state['grasp_motion_mode']=='ATTACHED_PUBLIC':
            mode='ATTACHED_PUBLIC';self.pending_started=None;self.reopening_active=False
        elif state is not None and state['grasp_motion_mode']=='FREE_GRU':
            mode='FREE_GRU';self.pending_started=None;self.reopening_active=False
        else:
            state=self.assembler.build_pending(result,robot,h,phase_state=self.phase,reopening=self.reopening_active or h['gripper_target']==1.)
            if state is None:
                self._mode('UNAVAILABLE',robot.timestamp)
                return dict(mode=self.mode,state=None,reason='current_public_object_unknown_hold_actual_q_keep_request',
                    active_visual_attachment=deepcopy(self.active.latest),GRU_used=False,Native_labels_used_by_policy=False)
            if self.pending_started is None:self.pending_started=self.active.attempt_started if self.active.attempt_started is not None else robot.timestamp
            settle_deadline=self.pending_started+self.config.pending_timeout_seconds
            if (h['gripper_target']==0. and self.active.spec is None and not self.active.failed
                    and robot.timestamp<settle_deadline-1e-8):
                self.active.begin_probe(self.assembler.binding.latest_visual_target,robot,h)
            spec=deepcopy(self.active.spec)
            deadline=spec['deadline'] if spec is not None else settle_deadline
            far=np.linalg.norm(state['object_position']-state['tcp_position'])>self.config.pending_far_distance_m
            expired=robot.timestamp>=deadline-1e-8
            if h['gripper_target']==1. or self.reopening_active or far or expired:
                mode='REOPENING_UNKNOWN_CONTACT';self.reopening_active=True;spec=None
            elif self.active.failed:
                self._mode('ACTIVE_VISUAL_PROBE_FAILED_HOLD',robot.timestamp)
                return dict(mode=self.mode,state=None,action=None,
                    reason='failed_active_attempt_hold_actual_q_keep_request_until_unchanged_deadline',
                    active_visual_attachment=deepcopy(self.active.latest),probe_deadline=deadline,
                    GRU_used=False,Native_labels_used_by_policy=False)
            elif spec is not None:mode='ACTIVE_VISUAL_VERIFICATION_PROBE'
            else:mode='PENDING_CLOSE_UNKNOWN_CONTACT'
        previous=self.mode;self._mode(mode,robot.timestamp)
        if mode=='FREE_GRU':
            if previous!=mode:state['phase_state']=None;self.phase=None
            generation=self.geometry.generation;self.candidates=self.geometry.update(state)
            if self.geometry.generation!=generation:self.warm=None
            execution=BoundForecastExecutionModel.from_public_bridge(self.core.execution_model,result);planner_class=ForecastJointGoalCEMPlanner
        else:
            if self.candidates is None:self.candidates=self.geometry.update(state)
            execution=self.core.execution_model;planner_class=JointGoalCEMPlanner
            if mode in ('PENDING_CLOSE_UNKNOWN_CONTACT','ACTIVE_VISUAL_VERIFICATION_PROBE'):
                state['phase_state']=dict(phase='close',dwell=0.,lost_hold=0.,updated_at=robot.timestamp,
                    close_entered_at=self.pending_started,reopening=False)
            elif mode=='REOPENING_UNKNOWN_CONTACT':
                state['phase_state']=dict(phase='approach',dwell=0.,lost_hold=0.,updated_at=robot.timestamp,reopening=True)
            else:
                if self.hold_tcp_quaternion is None:self.hold_tcp_quaternion=robot.tcp_quaternion.copy()
                state['phase_state']=dict(phase='hold',dwell=0.,lost_hold=0.,updated_at=robot.timestamp,
                    lift_goal_position=state['lift_goal_position'].copy(),hold_tcp_quaternion=self.hold_tcp_quaternion.copy())
        planning=deepcopy(self.candidates)
        if mode=='REOPENING_UNKNOWN_CONTACT':
            for c in planning:c['pregrasp_offset']=max(c['pregrasp_offset'],self.config.reopen_pregrasp_offset_m)
        if mode=='ACTIVE_VISUAL_VERIFICATION_PROBE':
            candidates,qualification,planner=self._verification_components(backend,state,planning,execution,spec)
        else:
            candidates,qualification=qualify_public_grasp_candidates(backend,state,planning,self.helper,mode=mode,
                bound=execution if mode=='FREE_GRU' else None)
            planner=planner_class(self.core.config,execution_model=execution,goal_helper=self.helper,
                constraint_cost=self.core.planner.constraint_cost,terminal_config=self.core.terminal_config)
        plan=planner.plan(state,candidates,warm_start=self.warm)
        self.phase=deepcopy(plan['phase_state']);self.warm=plan.get('warm_start');action=deepcopy(plan.get('action'))
        if action is not None:action['candidate_id']=plan['selected_candidate_id']
        return dict(mode=mode,state=state,plan=plan,action=action,qualification=qualification,
            active_visual_attachment=deepcopy(self.active.latest),probe=deepcopy(spec),
            GRU_used=mode=='FREE_GRU',free_motion_asserted=mode=='FREE_GRU',
            legacy_attachment_controls_mode=False,contact_response_qualified=False,force_grasp_proof=False)


def run_public_grasp_v4(backend,executor,output,config,*,recorder=None):
    if type(backend) is not ObservedPublicGraspV4Backend or executor.backend is not backend:
        raise ValueError('v4 observer must wrap every real50ms executor tick')
    if config.get('public_grasp_active_visual_v4') is not True:raise ValueError('explicit independently versioned v4 protocol required')
    if float(config.get('control_dt',.1))!=.1 or float(config.get('physics_dt',.05))!=.05:raise ValueError('100ms/2actual50ms budget required')
    if backend.samples==0:backend.sample()
    output=Path(output);settings=backend.public_hold_metric.config
    core,helper,provenance=build_response_planner(ROOT,executor.limits,controller='coupled',action_space='joint',
        joint_goal_config={'lower':tuple(config.get('workspace_lower',[-.1,-.65,.755]))},
        terminal_config={'enabled':bool(config.get('terminal_goal_cost',True))},reference_frequency=executor.config.reference_frequency)
    coordinator=ActiveVisualGraspCoordinator(core,helper,backend.assembler,backend.bridge.active,config=settings)
    provenance=deepcopy(provenance);provenance.update(active_visual_protocol='active_vertical_visual_comotion_v4',
        criteria=asdict(settings),active_attachment_criteria=asdict(backend.bridge.active.config),
        tracker_cache_binding=backend.bridge.binding.receipt(),legacy_attachment_controls_bridge=False,
        contact_response_qualified=False,force_or_frictional_grasp_proof=False,Native_labels_used_by_policy=False,
        previous_finger_response_checkpoint_requalified_for_completion_latch=False,
        public_hold_requires_actual_visible_4cm_and_halfsecond=True,formal_qualification=False,
        development_assumption='table_supported_target_without_independent_external_vertical_drive')
    write_new(output/'public_grasp_v4_provenance.json',provenance)
    maximum=int(config.get('public_grasp_max_decisions',np.ceil(float(config.get('simulation_seconds',6.))/.1)))
    if maximum<=0:raise ValueError('positive decision budget required')
    cycles=0;times=[];ever=False;stop='budget_exhausted'
    while cycles<maximum and backend.remaining_ticks()>=2:
        if any(p.exists() for p in (output/'STOP',output.parent/'STOP',ROOT/'runtime/framework/local_mpc_20261102/STOP')):stop='administrative_stop';break
        if backend.public_hold_metric.passed:stop='visible_public_lift_hold_observed';break
        started=time.perf_counter();robot=backend.latest_robot;h=deepcopy(executor.state())
        if not isinstance(robot,PublicRobotSensors) or abs(robot.timestamp-backend.clock())>1e-8:raise ValueError('fresh public robot frame required')
        if h['gripper_target'] not in (0.,1.) and not ever:
            decision=dict(mode='UNAVAILABLE',state=None,reason='initial_actual_open_warmup');action=None;opening=1.
        else:
            decision=coordinator.decide(backend,backend.latest,robot,h);action=decision.get('action');opening=float(h['gripper_target'])
        rejected=None;qualification=None
        if action is None:
            coordinator.warm=None;goal=robot.q.copy()
            if not ever:opening=1.
            executed=dict(joint_goal=goal.copy(),opening=opening,source='actual_q_hold_keep_requested_gripper')
        else:
            opening=float(action['opening'])
            if opening not in (0.,1.):raise ValueError('binary modeled opening required')
            if decision['mode']=='REOPENING_UNKNOWN_CONTACT' and opening!=1.:rejected='reopen_model_proposed_close'
            if decision['mode']=='ACTIVE_VISUAL_VERIFICATION_PROBE' and opening!=0.:rejected='probe_model_proposed_open'
            goal,reason,qualification=helper.qualify_action(backend,decision['state'],action,require_predicted_joint_goal=True)
            rejected=rejected or reason
            if rejected:
                goal=robot.q.copy();opening=float(h['gripper_target']);coordinator.warm=None
                executed=dict(joint_goal=goal.copy(),opening=opening,source='rejected_model_action_hold_actual_q_keep_request')
            else:
                if not np.array_equal(goal,np.asarray(action['joint_goal'])):raise ValueError('selected modeled joint goal changed before dispatch')
                executed=dict(joint_goal=goal.copy(),opening=opening,source='exact_selected_modeled_q_and_opening')
                coordinator.note_executed(action,robot)
        receipt=executor.execute_joint_target(goal,opening=opening,phase='P3_active_visual_v4_'+decision['mode'])
        ever=True;after=backend.latest_robot
        append(output/'public_grasp_v4_decisions.jsonl',dict(index=cycles,time=robot.timestamp,public_robot=robot,
            actuator_state_before=h,decision=decision,action_rejected_reason=rejected,action_qualification=qualification,
            executed_action=executed,executor=receipt,public_robot_after=after,actuator_state_after=deepcopy(executor.state()),
            active_visual_after=deepcopy(backend.bridge.active.latest),public_lift_hold=backend.public_hold_metric.receipt(),Native_labels_used_by_policy=False))
        cycles+=1;times.append(time.perf_counter()-started)
        if after is None or after.timestamp<=robot.timestamp+1e-8:stop='no_actual_public_tick';break
        if backend.public_hold_metric.passed:stop='visible_public_lift_hold_observed';break
    return dict(stage='P3_active_visual_grasp_v4_development',complete=True,cycles=cycles,canary_stop_reason=stop,
        public_lift_hold_observed=backend.public_hold_metric.passed,public_lift_hold=backend.public_hold_metric.receipt(),
        Native_task_success_claimed=False,Native_success_exit_used=False,Native_labels_used_by_policy=False,
        contact_response_qualified=False,formal_qualification=False,precision_qualified=False,
        force_or_frictional_grasp_proof=False,mode_events=coordinator.mode_events,controller_provenance=provenance,
        active_visual_final=deepcopy(backend.bridge.active.latest),full_cycle_wall_seconds=quantiles(times),real_time_guaranteed=False)
