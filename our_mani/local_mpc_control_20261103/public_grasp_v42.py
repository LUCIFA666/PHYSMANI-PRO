"""v42 perception-only candidate: explicit goal-shell public tracker binding.

The v41 recovery, active-visual attachment thresholds and MPC/dispatch are
unchanged. Sphere-fit failures and insufficient remaining observed support are
UNKNOWN. Raw sensor files are never filtered; private RGB copies only.
"""
from copy import deepcopy
from pathlib import Path
import time
import numpy as np
from ..contracts import ObjectState
from ..benchmark_perception import RedBoxTracker
from .goal_aware_perception import GoalAwareRedBoxTracker
from .public_attachment import PublicTargetFrameCache,PublicAttachmentEstimator,AttachmentConfig
from .public_motion_bridge import PublicMotionBridge,PublicRobotSensors
from .active_visual_attachment import ActiveVisualAttachment
from .public_shadow import PublicShadowBackend,sanitized_robot
from .public_grasp import PublicLiftHoldMetric
from .public_grasp_v41 import (
    ActiveVisualGraspV41Config,ActiveVisualGraspV41Coordinator,ConfirmedAttemptRecovery,
    ROOT,append,write_new,quantiles,asdict,build_response_planner)
from .goal_aware_binding_v42 import (
    ApprovedGoalAwareTrackerCacheBinding,GoalAwarePublicGraspStateAssembler,RAW_TRACKER_KEYS)

ActiveVisualGraspV42Config=ActiveVisualGraspV41Config
RUNNER_BASE_SOURCE_SHA256='039336bbba4fca69d2bff6e3e29fbb53df2db2e02e9881ff2f1ddb67693c23aa'
RUNNER_APPROVED_RENAMES={
 'run_public_grasp_v41':'run_public_grasp_v42',
 'ObservedPublicGraspV41Backend':'ObservedPublicGraspV42Backend',
 'public_grasp_active_visual_v41':'public_grasp_active_visual_v42',
 'public_grasp_v41_provenance.json':'public_grasp_v42_provenance.json',
 'public_grasp_v41_decisions.jsonl':'public_grasp_v42_decisions.jsonl',
 'P3_active_visual_v41_':'P3_active_visual_v42_',
 'P3_active_visual_grasp_v41_development':'P3_goal_aware_public_grasp_v42_development'}


class GoalAwareActiveVisualPublicBridge:
    def __init__(self,*,active_config=None):
        self.tracker=GoalAwareRedBoxTracker();self.cache=PublicTargetFrameCache(self.tracker)
        self.bridge=PublicMotionBridge(tracker=self.cache)
        self.estimator=PublicAttachmentEstimator(AttachmentConfig());self.active=ActiveVisualAttachment(active_config)
        self.binding=ApprovedGoalAwareTrackerCacheBinding(self.bridge,self.tracker,self.cache)
        self.latest_attachment=self.latest_legacy_diagnostic=self.latest_result=None
        self._last_open_visual=None;self._last_close_requested=None;self._last_time=None;self.close_visual_baseline=None
    def reset(self):
        self.tracker.reset_episode();self.bridge.reset();self.cache.reset_episode();self.estimator.reset();self.active.reset()
        self.binding.reset_episode()
        self.latest_attachment=self.latest_legacy_diagnostic=self.latest_result=None
        self._last_open_visual=None;self._last_close_requested=None;self._last_time=None;self.close_visual_baseline=None
    def observe(self,observation,robot,*,close_requested):
        if not isinstance(robot,PublicRobotSensors):robot=PublicRobotSensors.from_mapping(robot)
        if type(close_requested) is not bool:raise TypeError('actual observed command intent required')
        self.binding.begin_frame(observation)
        target=None;raw=None;all_metadata=None;error=None
        try:
            target,all_metadata=self.tracker.observe(observation)
            raw={key:all_metadata[key] for key in RAW_TRACKER_KEYS}
        except (ValueError,RuntimeError) as exc:error=type(exc).__name__+': '+str(exc)
        trace=deepcopy(self.tracker.last_filter_metadata)
        strict_available=(self.tracker.sphere is not None and isinstance(trace,dict) and trace.get('filter_available') is True)
        if not strict_available:
            target=None;raw=None
            if trace is None:trace={}
            trace['strict_initial_fit_unavailable_output_rejected_for_policy']=True
        self.cache.stage_public_target(target,timestamp=robot.timestamp,diagnostic=raw)
        if close_requested and not self.active.attempt_active:
            previous=self._last_open_visual
            self.close_visual_baseline=(previous if self._last_close_requested is False and previous is not None
                and abs(robot.timestamp-previous.timestamp-.05)<=1e-6 else None)
        legacy=self.estimator.observe(target,robot,close_requested=close_requested,orientation_observable=False)
        active=self.active.observe(target,robot,close_requested=close_requested)
        result=self.bridge.observe(observation,robot,close_requested=close_requested,evidence=active['evidence'])
        result.update(attachment_estimate=active,legacy_attachment_diagnostic=legacy,
            active_visual_protocol='active_vertical_visual_comotion_v4',legacy_attachment_controls_bridge=False,
            public_preclose_visual_baseline=self.close_visual_baseline,goal_aware_binding_version='v42',
            goal_aware_filter_trace=trace,goal_aware_tracker_error=error,
            observed_target_available=target is not None,strict_initial_goal_sphere_available=strict_available,
            goal_aware_extra_metadata_separate_from_raw7_cache=True,tracker_injection_fact_retained=True)
        self.binding.bind_current(result,target)
        result['approved_tracker_cache']=self.binding.receipt()
        if not close_requested and isinstance(target,ObjectState) and target.visible and target.confidence>=.5:self._last_open_visual=target
        self._last_close_requested=close_requested;self._last_time=robot.timestamp
        self.latest_attachment=active;self.latest_legacy_diagnostic=legacy;self.latest_result=result
        return result


class ObservedPublicGraspV42Backend(PublicShadowBackend):
    def __init__(self,backend,output,*,public_bridge=None,recorder=None,save_frames=True,grasp_config=None):
        wrapper=public_bridge if public_bridge is not None else GoalAwareActiveVisualPublicBridge()
        if type(wrapper) is not GoalAwareActiveVisualPublicBridge:raise TypeError('explicit exact v42 goal-aware bridge required')
        wrapper.binding.validate_dependencies()
        self.assembler=GoalAwarePublicGraspStateAssembler(wrapper.binding);self.latest_robot=None
        self.recovery=ConfirmedAttemptRecovery();self.public_hold_metric=PublicLiftHoldMetric(grasp_config or ActiveVisualGraspV42Config())
        super().__init__(backend,output,bridge=wrapper,recorder=recorder,save_frames=save_frames,
            controller_uses_public_forecast=True,observer_role='grasp_prototype',attachment_estimator_used=True)
    def sample(self,state=None):
        current=self.backend.read_state() if state is None else state;robot=sanitized_robot(current)
        result=super().sample(current);self.assembler.observe_robot(robot);self.latest_robot=robot
        self.public_hold_metric.observe(result,robot,self.bridge.latest_attachment,self.bridge.close_visual_baseline)
        self.recovery.observe(result,robot);return result
    def receipt(self):
        result=super().receipt();result.update(perception_revision='v42_goal_sphere_surface_only',
            approved_goal_aware_binding=self.bridge.binding.receipt(),recovery=self.recovery.receipt(),
            post_confirmation_recovery_revision='v41',legacy_attachment_controls_bridge=False,
            raw_RGBD_saved_without_filtering=True,unknown_not_filled_from_GRU_or_controlled_prediction=True)
        return result


def run_public_grasp_v42(backend,executor,output,config,*,recorder=None):
    if type(backend) is not ObservedPublicGraspV42Backend or executor.backend is not backend:
        raise ValueError('v4 observer must wrap every real50ms executor tick')
    if config.get('public_grasp_active_visual_v42') is not True:raise ValueError('explicit independently versioned v4 protocol required')
    if float(config.get('control_dt',.1))!=.1 or float(config.get('physics_dt',.05))!=.05:raise ValueError('100ms/2actual50ms budget required')
    if backend.samples==0:backend.sample()
    output=Path(output);settings=backend.public_hold_metric.config
    core,helper,provenance=build_response_planner(ROOT,executor.limits,controller='coupled',action_space='joint',
        joint_goal_config={'lower':tuple(config.get('workspace_lower',[-.1,-.65,.755]))},
        terminal_config={'enabled':bool(config.get('terminal_goal_cost',True))},reference_frequency=executor.config.reference_frequency)
    coordinator=ActiveVisualGraspV41Coordinator(core,helper,backend.assembler,backend.bridge.active,config=settings,recovery=backend.recovery)
    provenance=deepcopy(provenance);provenance.update(active_visual_protocol='active_vertical_visual_comotion_v4',
        post_confirmation_unknown_recovery='v41_actualq_hold_keep_request_no_lift_no_expired_probe_reopen',
        criteria=asdict(settings),active_attachment_criteria=asdict(backend.bridge.active.config),
        tracker_cache_binding=backend.bridge.binding.receipt(),legacy_attachment_controls_bridge=False,
        contact_response_qualified=False,force_or_frictional_grasp_proof=False,Native_labels_used_by_policy=False,
        previous_finger_response_checkpoint_requalified_for_completion_latch=False,
        public_hold_requires_actual_visible_4cm_and_halfsecond=True,formal_qualification=False,
        development_assumption='table_supported_target_without_independent_external_vertical_drive')
    write_new(output/'public_grasp_v42_provenance.json',provenance)
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
            if not ever and not backend.recovery.preserves_closed_attempt(backend.bridge.active.attempt_id):opening=1.
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
        receipt=executor.execute_joint_target(goal,opening=opening,phase='P3_active_visual_v42_'+decision['mode'])
        ever=True;after=backend.latest_robot
        append(output/'public_grasp_v42_decisions.jsonl',dict(index=cycles,time=robot.timestamp,public_robot=robot,
            actuator_state_before=h,decision=decision,action_rejected_reason=rejected,action_qualification=qualification,
            executed_action=executed,executor=receipt,public_robot_after=after,actuator_state_after=deepcopy(executor.state()),
            active_visual_after=deepcopy(backend.bridge.active.latest),recovery_after=backend.recovery.receipt(),public_lift_hold=backend.public_hold_metric.receipt(),Native_labels_used_by_policy=False))
        cycles+=1;times.append(time.perf_counter()-started)
        if after is None or after.timestamp<=robot.timestamp+1e-8:stop='no_actual_public_tick';break
        if backend.public_hold_metric.passed:stop='visible_public_lift_hold_observed';break
    return dict(stage='P3_goal_aware_public_grasp_v42_development',complete=True,cycles=cycles,canary_stop_reason=stop,
        public_lift_hold_observed=backend.public_hold_metric.passed,public_lift_hold=backend.public_hold_metric.receipt(),
        Native_task_success_claimed=False,Native_success_exit_used=False,Native_labels_used_by_policy=False,
        contact_response_qualified=False,formal_qualification=False,precision_qualified=False,
        force_or_frictional_grasp_proof=False,mode_events=coordinator.mode_events,controller_provenance=provenance,
        active_visual_final=deepcopy(backend.bridge.active.latest),recovery_final=backend.recovery.receipt(),full_cycle_wall_seconds=quantiles(times),real_time_guaranteed=False)
