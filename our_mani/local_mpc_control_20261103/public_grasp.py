"""Public close-pending/attachment prototype; no full-grasp qualification.

FREE uses the frozen public GRU. Pending uses a separately labelled current-
visual CV proxy with UNKNOWN contact, never a manufactured FREE result or held
claim. Confirmed public relative-motion evidence alone enables controlled hold.
No Native object/held/success label enters policy state or success termination.
"""
from copy import deepcopy
from collections import deque
from dataclasses import dataclass,asdict
from pathlib import Path
import time
import numpy as np

from ..benchmark_perception import RedBoxTracker
from ..contracts import ObjectState,Pose
from ..local_mpc_20261102 import mpc
from ..local_mpc_20261102.records import append,write_new,quantiles
from ..local_mpc_20261102.candidates import GeometricCandidates
from .public_motion_bridge import PublicMotionBridge,PublicRobotSensors
from .public_attachment import PublicAttachmentEstimator,PublicTargetFrameCache,AttachmentConfig
from .public_shadow import PublicShadowBackend,sanitized_robot
from .public_grasp_state import ApprovedTrackerCacheBinding,PublicGraspStateAssembler
from .response_factory import build_planner as build_response_planner
from .forecast_control import BoundForecastExecutionModel,ForecastJointGoalCEMPlanner
from .joint_goal_cem import JointGoalCEMPlanner
from .temporal_candidates import CurrentWindowCandidates
from .public_grasp_probe import BoundedVerificationProbe,VerificationProbePlanner,qualify_probe_candidates


ROOT=Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class PublicGraspConfig:
    pending_timeout_seconds: float = .8
    pending_far_distance_m: float = .12
    minimum_public_lift_m: float = .04
    public_hold_seconds: float = .5
    maximum_relative_drift_m: float = .01
    reopen_pregrasp_offset_m: float = .10
    sensor_dt: float = .05
    immediate_expiry: bool = True
    verification_probe_enabled: bool = False
    def __post_init__(self):
        for key,value in asdict(self).items():
            if key in ('immediate_expiry','verification_probe_enabled'):
                if type(value) is not bool:raise ValueError('explicit expiry-policy boolean required')
                continue
            if not np.isfinite(value) or value<=0:raise ValueError('positive finite grasp prototype threshold: '+key)
        if self.sensor_dt!=.05 or self.minimum_public_lift_m<.04 or self.public_hold_seconds<.5:
            raise ValueError('real50ms observations, >=4cm lift and >=.5s hold required')
        if self.reopen_pregrasp_offset_m<.09:
            raise ValueError('explicit release retreat must clear the .07m public separation gate; original .055m TCP offset is insufficient')


class AttachmentAwarePublicBridge:
    """Real tracker once -> cache -> estimator -> unchanged bridge once/frame."""
    def __init__(self,*,attachment_config=None):
        self.tracker=RedBoxTracker()
        self.cache=PublicTargetFrameCache(self.tracker)
        self.bridge=PublicMotionBridge(tracker=self.cache)
        self.estimator=PublicAttachmentEstimator(attachment_config or AttachmentConfig())
        self.binding=ApprovedTrackerCacheBinding(self.bridge,self.tracker,self.cache)
        self.latest_attachment=None;self.latest_result=None
        self._last_open_visual=None;self._last_close_requested=None;self._last_time=None
        self.close_visual_baseline=None
    def reset(self):
        self.bridge.reset();self.cache.reset_episode();self.estimator.reset()
        self.latest_attachment=self.latest_result=None
        self._last_open_visual=None;self._last_close_requested=None;self._last_time=None;self.close_visual_baseline=None
    def observe(self,observation,robot,*,close_requested):
        if not isinstance(robot,PublicRobotSensors):robot=PublicRobotSensors.from_mapping(robot)
        if type(close_requested) is not bool:raise ValueError('actual close intent must be bool')
        target=None;diagnostic=None
        try:target,diagnostic=self.tracker.observe(observation)
        except (ValueError,RuntimeError):pass
        self.cache.stage_public_target(target,timestamp=robot.timestamp,diagnostic=diagnostic)
        # Lift baseline is an actual visual pose from the previous open-intent
        # real frame, never an assumed table height or first post-close pose.
        first_close=close_requested and not self.estimator.attempt_active
        if first_close:
            previous=self._last_open_visual
            self.close_visual_baseline=(previous if self._last_close_requested is False and previous is not None
                and abs(robot.timestamp-previous.timestamp-.05)<=1e-6 else None)
        attachment=self.estimator.observe(target,robot,close_requested=close_requested,orientation_observable=False)
        result=self.bridge.observe(observation,robot,close_requested=close_requested,evidence=attachment['evidence'])
        result.update(attachment_estimate=attachment,public_preclose_visual_baseline=self.close_visual_baseline,
                      approved_tracker_cache=self.binding.receipt(),tracker_injection_fact_retained=True)
        self.binding.bind_current(result,target)
        if not close_requested and isinstance(target,ObjectState) and target.visible and target.confidence>=.5:
            self._last_open_visual=target
        self._last_close_requested=close_requested;self._last_time=robot.timestamp
        self.latest_attachment=attachment;self.latest_result=result
        return result


class PublicLiftHoldMetric:
    """Visible measured lift and relative stability, not controlled-pose lift."""
    def __init__(self,config=None):
        self.config=config or PublicGraspConfig();self.rows=deque();self.latest=None;self.last_time=None
    def observe(self,result,robot,attachment,baseline):
        now=robot.timestamp
        if self.last_time is not None and now<=self.last_time:raise ValueError('one increasing actual50ms hold metric sample required')
        consecutive=self.last_time is not None and abs(now-self.last_time-.05)<=1e-6
        self.last_time=now
        target=result.get('object_observation') if isinstance(result,dict) else None
        evidence=attachment.get('evidence') if isinstance(attachment,dict) else None
        confirmed=(isinstance(attachment,dict) and result.get('mode')=='ATTACHED_PUBLIC' and result.get('controlled_mode_available') is True
            and attachment.get('mode')=='ATTACHED_PUBLIC' and getattr(evidence,'event',None)=='attachment_confirmed'
            and abs(float(attachment.get('timestamp',np.inf))-now)<=1e-8)
        visible=(isinstance(target,ObjectState) and target.visible and target.confidence>=.5 and abs(target.timestamp-now)<=1e-8)
        reference=(isinstance(baseline,ObjectState) and baseline.timestamp<now)
        lift=float(target.pose.position[2]-baseline.pose.position[2]) if visible and reference else None
        good=bool(confirmed and visible and reference and lift>=self.config.minimum_public_lift_m-1e-12)
        if not good or not consecutive:self.rows.clear()
        relative=None;duration=0.;drift=None
        if good:
            relative=robot.tcp_pose.inverse_apply(target.pose.position)
            self.rows.append((now,relative.copy()))
            # Keep the complete current qualifying streak. Relative drift must
            # be small throughout it, not just at two endpoints.
            values=np.stack([row[1] for row in self.rows]);drift=float(np.linalg.norm(values-values[0],axis=-1).max())
            if drift>self.config.maximum_relative_drift_m:
                self.rows.clear();self.rows.append((now,relative.copy()));drift=0.
            duration=now-self.rows[0][0]
        self.latest=dict(timestamp=now,current_visual_target=bool(visible),current_public_attachment_confirmed=bool(confirmed),
            actual_preclose_visual_baseline_available=bool(reference),visual_lift_m=lift,public_hold_seconds=duration,
            relative_drift_m=drift,public_lift_hold_observed=bool(good and duration>=self.config.public_hold_seconds-1e-8),
            controlled_pose_used_as_visual_label=False,Native_labels_used=False,force_contact_or_task_success_qualified=False)
        return deepcopy(self.latest)
    @property
    def passed(self):return bool(self.latest and self.latest['public_lift_hold_observed'])
    def receipt(self):return dict(criteria=asdict(self.config),latest=deepcopy(self.latest),formal_qualification=False)


class ObservedPublicGraspBackend(PublicShadowBackend):
    """A new observer; frozen shadow/pregrasp classes are not modified."""
    def __init__(self,backend,output,*,public_bridge=None,recorder=None,save_frames=True,grasp_config=None):
        wrapper=public_bridge if public_bridge is not None else AttachmentAwarePublicBridge()
        if type(wrapper) is not AttachmentAwarePublicBridge:raise TypeError('real attachment-aware public bridge binding required')
        wrapper.binding.validate_dependencies()
        self.assembler=PublicGraspStateAssembler(wrapper.binding)
        self.latest_robot=None;self.public_hold_metric=PublicLiftHoldMetric(grasp_config)
        self.verification_probe_monitor=None
        super().__init__(backend,output,bridge=wrapper,recorder=recorder,save_frames=save_frames,
            controller_uses_public_forecast=True,observer_role='grasp_prototype',attachment_estimator_used=True)
    def sample(self,state=None):
        current=self.backend.read_state() if state is None else state
        robot=sanitized_robot(current)
        result=super().sample(current)
        self.assembler.observe_robot(robot);self.latest_robot=robot
        self.public_hold_metric.observe(result,robot,self.bridge.latest_attachment,self.bridge.close_visual_baseline)
        if self.verification_probe_monitor is not None:
            self.verification_probe_monitor.observe_feedback(result,robot,close_requested=float(current['requested_opening'])<.5)
        return result
    def receipt(self):
        result=super().receipt()
        result.update(observer_role='grasp_prototype',attachment_estimator_used=True,
            attachment_is_public_consistency_hypothesis=True,contact_response_qualified=False,
            actual_visible_lift_hold_required=True,Native_labels_used_by_policy=False)
        return result


def qualify_public_grasp_candidates(backend,state,candidates,helper,*,mode,bound=None):
    """Phase-matched public target qualifications, with explicit source scope."""
    values=[mpc.validate_candidate(c) for c in candidates]
    if len(values)!=8 or len({c['id'] for c in values})!=8:raise ValueError('K8 ledger required')
    phase=(state.get('phase_state') or {}).get('phase','approach');poses=[];valid=[]
    for candidate in values:
        at=candidate['deadline']
        # A representative which could not be defined at generation time
        # stays unavailable. Expired FREE timing masks, however, do not forbid
        # the existing pool's later robot-only probe or attached transport.
        representative_defined=('parallel_gripper_symmetry_choice' not in candidate
            or candidate['parallel_gripper_symmetry_choice'] is not None)
        ok=representative_defined
        if mode=='FREE_GRU':
            if bound is None:raise ValueError('FREE requires the current public GRU binding')
            availability=candidate.get('proposal_available',True)
            if not isinstance(availability,(bool,np.bool_)):raise ValueError('explicit boolean proposal availability required')
            ok=(ok and bool(availability)
                and bound.anchor_time-1e-9<=at<=bound.forecast.times[-1]+1e-9)
            object_pose=bound.sample(at if ok else bound.anchor_time)[0]
            offset=candidate['object_tcp_position']+(candidate['approach_direction']*candidate['pregrasp_offset'] if phase=='approach' else 0.)
            p=object_pose.apply(offset);q=mpc.quaternion(mpc.quat_multiply(object_pose.xyzw()[3:],candidate['object_tcp_quaternion']))
        elif mode=='ATTACHED_PUBLIC':
            p=state['lift_goal_position']+mpc.quat_matrix(state['object_quaternion'])@candidate['object_tcp_position']
            q=(state.get('phase_state') or {}).get('hold_tcp_quaternion',state['tcp_quaternion'])
        else:
            # A labelled hypothesis for a single receding100ms follow, not a
            # free-body GRU or a learned contact response.
            target=mpc.contact_target(state,candidate,max(state['time'],at),pregrasp=phase=='approach')
            p,q=target['position'],target['quaternion']
        poses.append((p,q));valid.append(ok)
    result=helper.adapter.solve(np.broadcast_to(state['q'],(8,7)).copy(),np.stack([p for p,q in poses]),np.stack([q for p,q in poses]))
    receipts=[];qualified=[]
    for i,candidate in enumerate(values):
        available=bool(valid[i] and result['joint_goal_feasible'][i]);collision=None
        if available:collision=bool(backend.probe_joint_collision(result['joint_goal'][i]))
        value=deepcopy(candidate);value['native_pose_qualified']=bool(available and not collision);qualified.append(value)
        receipts.append(dict(id=candidate['id'],deadline=candidate['deadline'],deadline_shifted=False,qualified=value['native_pose_qualified'],
            candidate_representation_defined=('parallel_gripper_symmetry_choice' not in candidate or candidate['parallel_gripper_symmetry_choice'] is not None),
            free_proposal_mask_respected=mode=='FREE_GRU',
            finite_placeholder_only=not valid[i],target_position=poses[i][0],target_quaternion=poses[i][1],target_source=mode,
            Native_static_collision_oracle_execution_check=available,static_collision=collision,Native_IK_calls=0,physical_steps=0,
            contact_object_response_qualified=False))
    return qualified,dict(cases=receipts,batch_DLS_calls=1,poses=8,mode=mode,formal_qualification=False)


class PublicGraspCoordinator:
    """A public-only decision machine. It emits proposals, not physical ticks."""
    def __init__(self,core,helper,assembler,*,config=None):
        self.core,self.helper,self.assembler=core,helper,assembler
        self.config=config or PublicGraspConfig()
        candidate_class=CurrentWindowCandidates if self.config.immediate_expiry else GeometricCandidates
        self.geometry=candidate_class(horizon=core.config.dt*core.config.horizon)
        self.mode='UNAVAILABLE';self.warm=None;self.phase=None;self.candidates=None
        self.pending_started=None;self.close_candidate=None;self.mode_events=[]
        self.reopening_active=False;self.hold_tcp_quaternion=None
        self.verification_probe=BoundedVerificationProbe()
    def _mode(self,new,now):
        if new!=self.mode:
            self.mode_events.append(dict(time=now,previous=self.mode,current=new,warm_cleared=True))
            self.warm=None;self.mode=new
            if new!='ATTACHED_PUBLIC':self.hold_tcp_quaternion=None
    def note_executed(self,action,robot):
        # Actual dispatch intent is control memory, not attachment evidence.
        if float(action['opening'])==0. and self.pending_started is None and self.mode=='FREE_GRU':
            self.pending_started=robot.timestamp;self.close_candidate=action.get('candidate_id')
            self.verification_probe.reset()
    def decide(self,backend,result,robot,h):
        if self.assembler.robot_twist is None or not self.assembler.robot_twist[2]:
            self.verification_probe.abort('no_current_real_robot_twist')
            self._mode('UNAVAILABLE',robot.timestamp);return dict(mode=self.mode,state=None,reason='no_current_real_robot_twist')
        state=self.assembler.build(result,robot,h,phase_state=self.phase)
        if state is not None and state['grasp_motion_mode']=='ATTACHED_PUBLIC':
            self.verification_probe.abort('current_public_attachment_confirmed_no_probe_needed')
            mode='ATTACHED_PUBLIC';self.pending_started=None;self.reopening_active=False
        elif state is not None and state['grasp_motion_mode']=='FREE_GRU':
            self.verification_probe.reset()
            mode='FREE_GRU';self.pending_started=None;self.reopening_active=False
        else:
            reopening=self.reopening_active
            state=self.assembler.build_pending(result,robot,h,phase_state=self.phase,reopening=reopening)
            if state is None:
                self.verification_probe.abort('current_visual_or_contact_state_unavailable')
                self._mode('UNAVAILABLE',robot.timestamp)
                return dict(mode=self.mode,state=None,reason='current_visual_or_contact_state_unknown_hold_actual_q_keep_request')
            if h['gripper_target']==0.:
                if self.pending_started is None:self.pending_started=robot.timestamp
                far=np.linalg.norm(state['object_position']-state['tcp_position'])>self.config.pending_far_distance_m
                expired=robot.timestamp-self.pending_started>=self.config.pending_timeout_seconds-1e-8
                mode='REOPENING_UNKNOWN_CONTACT' if far or expired else 'PENDING_CLOSE_UNKNOWN_CONTACT'
            else:mode='REOPENING_UNKNOWN_CONTACT'
            if mode=='REOPENING_UNKNOWN_CONTACT':
                self.reopening_active=True;self.verification_probe.abort('original_pending_timeout_far_or_reopen_intent')
        probe_spec=None
        if mode=='PENDING_CLOSE_UNKNOWN_CONTACT' and self.config.verification_probe_enabled:
            probe_spec=self.verification_probe.consider(result,robot,h)
            if probe_spec is not None:
                mode='PENDING_VERIFICATION_PROBE';state['grasp_motion_mode']=mode
            elif self.verification_probe.used:
                # Once an admitted probe loses its current safety gates, do
                # not resume a different moving policy under an unknown grasp.
                # The original absolute pending timeout still triggers reopen.
                self._mode('PENDING_PROBE_ABORTED_HOLD',robot.timestamp)
                return dict(mode=self.mode,state=None,action=None,
                    reason='verification_probe_invalidated_hold_actual_q_keep_closed_request_until_original_timeout',
                    verification_probe=self.verification_probe.receipt(),GRU_used=False,
                    free_motion_asserted=False,contact_response_qualified=False)
        previous_mode=self.mode
        self._mode(mode,robot.timestamp)
        if mode=='FREE_GRU':
            if previous_mode!=mode:
                state['phase_state']=None;self.phase=None
            previous_generation=self.geometry.generation
            self.candidates=self.geometry.update(state)
            if self.geometry.generation!=previous_generation:self.warm=None
            execution=BoundForecastExecutionModel.from_public_bridge(self.core.execution_model,result)
            planner_class=ForecastJointGoalCEMPlanner
        else:
            if self.candidates is None:self.candidates=self.geometry.update(state)
            execution=self.core.execution_model;planner_class=JointGoalCEMPlanner
            if mode in ('PENDING_CLOSE_UNKNOWN_CONTACT','PENDING_VERIFICATION_PROBE'):
                state['phase_state']=dict(phase='close',dwell=0.,lost_hold=0.,updated_at=robot.timestamp,
                    close_entered_at=self.pending_started,reopening=False)
            elif mode=='REOPENING_UNKNOWN_CONTACT':
                state['phase_state']=dict(phase='approach',dwell=0.,lost_hold=0.,updated_at=robot.timestamp,reopening=True)
            else:
                if self.hold_tcp_quaternion is None:self.hold_tcp_quaternion=robot.tcp_quaternion.copy()
                state['phase_state']=dict(phase='hold',dwell=0.,lost_hold=0.,updated_at=robot.timestamp,
                    lift_goal_position=state['lift_goal_position'].copy(),hold_tcp_quaternion=self.hold_tcp_quaternion.copy())
        planning_candidates=deepcopy(self.candidates)
        if mode=='REOPENING_UNKNOWN_CONTACT':
            for candidate in planning_candidates:
                candidate['pregrasp_offset']=max(candidate['pregrasp_offset'],self.config.reopen_pregrasp_offset_m)
                candidate['recovery_geometry']='explicit_release_retreat_original_ids_and_deadlines_unchanged'
        if probe_spec is not None:
            candidates,qualification=qualify_probe_candidates(backend,state,planning_candidates,self.helper,probe_spec)
            planner=VerificationProbePlanner(self.core.config,execution_model=execution,goal_helper=self.helper,
                constraint_cost=self.core.planner.constraint_cost,terminal_config=self.core.terminal_config,probe_spec=probe_spec)
        else:
            candidates,qualification=qualify_public_grasp_candidates(backend,state,planning_candidates,self.helper,mode=mode,
                bound=execution if mode=='FREE_GRU' else None)
            planner=planner_class(self.core.config,execution_model=execution,goal_helper=self.helper,
                constraint_cost=self.core.planner.constraint_cost,terminal_config=self.core.terminal_config)
        plan=planner.plan(state,candidates,warm_start=self.warm)
        self.phase=deepcopy(plan['phase_state']);self.warm=plan.get('warm_start')
        action=deepcopy(plan.get('action'))
        if action is not None:action['candidate_id']=plan['selected_candidate_id']
        return dict(mode=mode,state=state,plan=plan,action=action,qualification=qualification,
            GRU_used=mode=='FREE_GRU',free_motion_asserted=mode=='FREE_GRU',
            pending_visual_CV_is_unqualified_contact_proxy=mode in ('PENDING_CLOSE_UNKNOWN_CONTACT','PENDING_VERIFICATION_PROBE','REOPENING_UNKNOWN_CONTACT'),
            verification_probe=self.verification_probe.receipt(),
            contact_response_learned=False,contact_response_qualified=False)


def run_public_grasp(backend,executor,output,config,*,recorder=None):
    """Prototype runner for later explicit integration; this module starts none."""
    if not isinstance(backend,ObservedPublicGraspBackend) or getattr(executor,'backend',None) is not backend:
        raise ValueError('executor must use ObservedPublicGraspBackend for every real50ms callback')
    if config.get('public_grasp_prototype') is not True:raise ValueError('explicit separately versioned public_grasp_prototype required')
    if float(config.get('control_dt',.1))!=.1 or float(config.get('physics_dt',.05))!=.05:
        raise ValueError('two real50ms ticks per100ms decision required')
    if backend.samples==0:backend.sample()
    output=Path(output);settings=backend.public_hold_metric.config
    core,helper,provenance=build_response_planner(ROOT,executor.limits,controller='coupled',action_space='joint',
        joint_goal_config={'lower':tuple(config.get('workspace_lower',[-.1,-.65,.755]))},
        terminal_config={'enabled':bool(config.get('terminal_goal_cost',True))},reference_frequency=executor.config.reference_frequency)
    coordinator=PublicGraspCoordinator(core,helper,backend.assembler,config=settings)
    if settings.verification_probe_enabled:backend.verification_probe_monitor=coordinator.verification_probe
    provenance=deepcopy(provenance);provenance.update(public_grasp_prototype=True,GRU_FREE_only=True,
        tracker_cache_binding=backend.bridge.binding.receipt(),pending_held_false_is_unknown_not_free=True,
        contact_response_learned=False,contact_response_qualified=False,Native_labels_used_by_policy=False,
        success_uses_visible_public_lift_and_relative_hold_only=True,criteria=asdict(settings),formal_qualification=False)
    write_new(output/'public_grasp_provenance.json',provenance)
    maximum=int(config.get('public_grasp_max_decisions',np.ceil(float(config.get('simulation_seconds',6.))/.1)))
    if maximum<=0:raise ValueError('positive actual public decision budget required')
    cycles=0;stop='budget_exhausted';times=[];ever_dispatched=False
    while cycles<maximum and backend.remaining_ticks()>=2:
        if any(p.exists() for p in (output/'STOP',output.parent/'STOP',ROOT/'runtime/framework/local_mpc_20261102/STOP')):
            stop='administrative_stop';break
        if backend.public_hold_metric.passed:stop='visible_public_lift_hold_observed';break
        started=time.perf_counter();robot=backend.latest_robot;h=deepcopy(executor.state())
        if not isinstance(robot,PublicRobotSensors) or abs(robot.timestamp-backend.clock())>1e-8:
            raise ValueError('fresh current real robot sensor frame required')
        # Only the first open warmup normalizes an uninitialized aperture-derived
        # target. After any dispatch, visual loss preserves the actual request.
        if h['gripper_target'] not in (0.,1.) and not ever_dispatched:
            decision=dict(mode='UNAVAILABLE',state=None,reason='initial_real_open_warmup');opening=1.;action=None
        else:
            decision=coordinator.decide(backend,backend.latest,robot,h);action=decision.get('action');opening=float(h['gripper_target'])
        reject=None;qualification=None
        if action is None:
            coordinator.warm=None;goal=robot.q.copy()
            if not ever_dispatched:opening=1.
            executed=dict(joint_goal=goal.copy(),opening=opening,source='actual_q_hold_keep_requested_gripper_no_object_pose_fabricated')
        else:
            opening=float(action['opening'])
            if opening not in (0.,1.):raise ValueError('binary modeled opening required')
            if decision['mode']=='REOPENING_UNKNOWN_CONTACT' and opening!=1.:
                reject='reopen_mode_model_proposed_close'
            if decision['mode']=='PENDING_VERIFICATION_PROBE' and opening!=0.:
                reject='verification_probe_model_proposed_open'
            goal,reject_goal,qualification=helper.qualify_action(backend,decision['state'],action,require_predicted_joint_goal=True)
            reject=reject or reject_goal
            if reject:
                goal=robot.q.copy();opening=float(h['gripper_target']);coordinator.warm=None
                executed=dict(joint_goal=goal.copy(),opening=opening,source='rejected_model_action_actual_q_hold_keep_request')
            else:
                if not np.array_equal(goal,np.asarray(action['joint_goal'])):raise ValueError('modeled selected joint goal changed before dispatch')
                executed=dict(joint_goal=np.asarray(goal).copy(),opening=opening,source='exact_selected_modeled_joint_goal_and_opening')
                coordinator.note_executed(action,robot)
        receipt=executor.execute_joint_target(goal,opening=opening,phase='P3_public_grasp_'+decision['mode'])
        ever_dispatched=True;after=backend.latest_robot
        append(output/'public_grasp_decisions.jsonl',dict(index=cycles,time=robot.timestamp,public_robot=robot,
            actuator_state_before=h,decision=decision,action_rejected_reason=reject,action_qualification=qualification,
            executed_action=executed,executor=receipt,public_robot_after=after,actuator_state_after=deepcopy(executor.state()),
            verification_probe_after=coordinator.verification_probe.receipt(),
            public_lift_hold=backend.public_hold_metric.receipt(),Native_labels_used_by_policy=False))
        cycles+=1;times.append(time.perf_counter()-started)
        if after is None or after.timestamp<=robot.timestamp+1e-8:stop='executor_produced_no_real_public_tick';break
        if backend.public_hold_metric.passed:stop='visible_public_lift_hold_observed';break
    return dict(stage='P3_public_grasp_development_prototype',complete=True,cycles=cycles,canary_stop_reason=stop,
        public_lift_hold_observed=backend.public_hold_metric.passed,public_lift_hold=backend.public_hold_metric.receipt(),
        Native_task_success_claimed=False,Native_success_exit_used=False,Native_labels_used_by_policy=False,
        contact_response_learned=False,contact_response_qualified=False,formal_qualification=False,precision_qualified=False,
        mode_events=coordinator.mode_events,controller_provenance=provenance,
        verification_probe_final=coordinator.verification_probe.receipt(),
        full_cycle_wall_seconds=quantiles(times),real_time_guaranteed=False)
