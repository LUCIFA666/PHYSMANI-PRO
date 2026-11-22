"""Independent V62 position-measurement producer; no live integration by default.

Only independently passing V7 position certificates receive unmodified RGBD.
Generic/raw-mask model qualification is false and never promoted here.
The legacy goal-filter bool is only an initial-sphere/far-candidate capability
contract. The far branch is an explicitly falsified development hypothesis,
never an ownership certificate. Original probe, maintenance, MPC and exact
dispatch are unchanged. A committed measurement-source change resets only
motion history, requiring new real frames while retaining attachment anchors.
"""
from copy import deepcopy
from pathlib import Path
import time
import numpy as np
import torch

from ..contracts import ObjectState,Pose
from .public_motion_bridge import PublicRobotSensors,PublicMotionEvidence
from .public_shadow import PublicShadowBackend,sanitized_robot
from .public_grasp import PublicLiftHoldMetric
from .public_grasp_v4 import ActiveVisualPublicBridge,ActiveVisualGraspConfig
from .active_visual_attachment import ActiveVisualAttachment
from .public_grasp_v41 import ConfirmedAttemptRecovery,ROOT,append,write_new,quantiles,asdict,build_response_planner
from .public_grasp_state import ApprovedTrackerCacheBinding,PublicGraspStateAssembler
from .public_state import PublicStateAssembler
from .depth_position_goal_perception_v62 import CurrentDepthPositionTrackerV62
from .public_attachment import PublicTargetFrameCache,PublicAttachmentEstimator,AttachmentConfig
from .public_motion_bridge import PublicMotionBridge
from .goal_aware_position_binding_v62 import ApprovedDepthPositionTrackerCacheBindingV62,RAW_TRACKER_KEYS
from .postconfirmed_maintenance_candidate_20261103 import PostConfirmedPositionMaintenance
from .public_grasp_candidates_v5 import CandidateGraspV5Coordinator,_public_completion_ready,_public_completion_reason
from ..local_mpc_p4_20261103.runtime import pack_public_row,PublicRowCache,FrozenPilotInference

RUNNER_BASE_SOURCE_SHA256='029cd2fc85ca326c6ddddad5fa4dce48004c5d662cca3707e130a0929e0ef438'

MAINTENANCE_PROTOCOL='original_probe_v4_then_current_visible_position_maintenance_v56'
ROUTE_CHANGE_PROTOCOL='accepted_public_measurement_source_change_resets_motion_history_only_v62'


def position_producer_source_map():
    """Explicit implementation closure, never a directory glob or raw dataset."""
    import hashlib
    from .goal_aware_position_binding_v62 import APPROVED_COMPONENT_SHA256
    from ..cube_position_measurement_20261103_v7.qualification import qualification_source_map
    names=('depth_position_goal_perception_v62.py','goal_aware_position_binding_v62.py',
           'public_candidate_position_v62.py')
    paths=[Path(__file__).with_name(name) for name in names]
    paths.extend(ROOT/name for name in APPROVED_COMPONENT_SHA256)
    result=qualification_source_map()
    result.update({str(path.resolve()):hashlib.sha256(path.read_bytes()).hexdigest() for path in paths})
    return result


def _reset_motion_history_for_route_change(bridge,trace):
    if isinstance(trace,dict) and trace.get('accepted_history_reset_for_route_change') is True:
        bridge._clear_motion(clear_tracker=False)
        trace.update(motion_history_reset_for_route_change=True,
            motion_history_reset_protocol=ROUTE_CHANGE_PROTOCOL,
            attachment_probe_maintenance_goal_Sphere_and_cache_reset=False)


def v62_enabled(*,goal_sphere_filter=True,postconfirmed_maintenance=True):
    if type(goal_sphere_filter) is not bool or type(postconfirmed_maintenance) is not bool:
        raise ValueError('explicit boolean v62 feature modifiers required')
    if not goal_sphere_filter or not postconfirmed_maintenance:
        raise ValueError('v62 requires goal sphere filtering and postconfirmed position maintenance')
    return True


class PositionGoalAwareActiveVisualPublicBridgeV62:
    def __init__(self,*,refiner_checkpoint,refiner_checkpoint_sha256,active_config=None):
        torch.set_num_threads(2)
        self.tracker=CurrentDepthPositionTrackerV62(refiner_checkpoint,refiner_checkpoint_sha256)
        self.cache=PublicTargetFrameCache(self.tracker)
        self.bridge=PublicMotionBridge(tracker=self.cache)
        self.estimator=PublicAttachmentEstimator(AttachmentConfig());self.active=ActiveVisualAttachment(active_config)
        self.binding=ApprovedDepthPositionTrackerCacheBindingV62(self.bridge,self.tracker,self.cache)
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
        trace=deepcopy(self.tracker.last_measurement_metadata)
        _reset_motion_history_for_route_change(self.bridge,trace)
        strict_available=(self.tracker.sphere is not None and isinstance(trace,dict) and trace.get('producer_ready') is True)
        if not strict_available:
            target=None;raw=None
            if trace is None:trace={}
            trace['strict_refiner_or_initial_fit_unavailable_output_rejected_for_policy']=True
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
            public_preclose_visual_baseline=self.close_visual_baseline,goal_aware_binding_version='v62',
            depth_refiner_trace=trace,depth_refiner_tracker_error=error,
            observed_target_available=target is not None,strict_initial_goal_sphere_available=strict_available,
            candidate_position_measurement_v62=True,NN_input_sphere_shell_filtered=False,
            goal_sphere_filter_flag_is_initialSphere_and_far_legacy_compatibility=True,
            refiner_extra_metadata_separate_from_raw7_cache=True,tracker_injection_fact_retained=True)
        self.binding.bind_current(result,target)
        result['approved_tracker_cache']=self.binding.receipt()
        if not close_requested and isinstance(target,ObjectState) and target.visible and target.confidence>=.5:self._last_open_visual=target
        self._last_close_requested=close_requested;self._last_time=robot.timestamp
        self.latest_attachment=active;self.latest_legacy_diagnostic=legacy;self.latest_result=result
        return result


class PublicCandidatePositionBridgeV62:
    def __init__(self,*,refiner_checkpoint,refiner_checkpoint_sha256,
                 goal_sphere_filter=True,postconfirmed_maintenance=True,active_config=None):
        v62_enabled(goal_sphere_filter=goal_sphere_filter,postconfirmed_maintenance=postconfirmed_maintenance)
        self.goal_sphere_filter=goal_sphere_filter;self.postconfirmed_maintenance=postconfirmed_maintenance
        self._configured_flags=(goal_sphere_filter,postconfirmed_maintenance)
        cls=PositionGoalAwareActiveVisualPublicBridgeV62
        self._configured_refiner=(Path(refiner_checkpoint).resolve(strict=True),refiner_checkpoint_sha256)
        self.delegate=cls(refiner_checkpoint=refiner_checkpoint,
            refiner_checkpoint_sha256=refiner_checkpoint_sha256,active_config=active_config)
        # These exact primitive objects remain covered by the existing binding.
        for name in ('tracker','cache','bridge','binding','estimator','active'):setattr(self,name,getattr(self.delegate,name))
        self.maintenance=PostConfirmedPositionMaintenance();self.retired_maintenance=[]
        self.latest_attachment=self.latest_raw_active=self.latest_result=None
        self.last_initial_admission=None
        self._validate_components()

    def _validate_components(self):
        if (self.goal_sphere_filter,self.postconfirmed_maintenance)!=self._configured_flags:
            raise ValueError('feature modifiers cannot change within an episode')
        expected=PositionGoalAwareActiveVisualPublicBridgeV62
        if type(self.delegate) is not expected or type(self.active) is not ActiveVisualAttachment:
            raise TypeError('exact original probe owner and selected public producer required')
        for name in ('tracker','cache','bridge','binding','estimator','active'):
            if getattr(self,name) is not getattr(self.delegate,name):
                raise ValueError('outer public producer cannot detach from its authenticated '+name)
        if (self.binding.tracker is not self.tracker or self.binding.cache is not self.cache
                or self.binding.bridge is not self.bridge or type(self.maintenance) is not PostConfirmedPositionMaintenance):
            raise ValueError('authenticated actual producer/cache and exact maintenance owner required')
        if (self.tracker.refiner_checkpoint,self.tracker.expected_refiner_sha256)!=self._configured_refiner:
            raise ValueError('configured refiner checkpoint identity cannot change within an episode')
        self.tracker.validate_refiner()
        self.binding.validate_dependencies()

    @property
    def close_visual_baseline(self):return self.delegate.close_visual_baseline

    def reset(self):
        self.delegate.reset();self.maintenance=PostConfirmedPositionMaintenance();self.retired_maintenance=[]
        self.latest_attachment=self.latest_raw_active=self.latest_result=None;self.last_initial_admission=None
        self._validate_components()

    def _track_once(self,observation,robot):
        if self.goal_sphere_filter:self.binding.begin_frame(observation)
        else:self.binding.validate_dependencies()
        target=raw=trace=error=None
        try:
            target,metadata=self.tracker.observe(observation)
            raw={key:metadata[key] for key in RAW_TRACKER_KEYS} if self.goal_sphere_filter else metadata
        except (ValueError,RuntimeError) as exc:error=type(exc).__name__+': '+str(exc)
        if self.goal_sphere_filter:
            trace=deepcopy(self.tracker.last_measurement_metadata)
            _reset_motion_history_for_route_change(self.bridge,trace)
            if self.tracker.sphere is None or not isinstance(trace,dict) or trace.get('producer_ready') is not True:
                target=raw=None;trace=trace or {};trace['strict_refiner_or_initial_fit_unavailable_output_rejected_for_policy']=True
        self.cache.stage_public_target(target,timestamp=robot.timestamp,diagnostic=raw)
        return target,trace,error

    def observe(self,observation,robot,*,close_requested):
        self._validate_components()
        if not isinstance(robot,PublicRobotSensors):robot=PublicRobotSensors.from_mapping(robot)
        if type(close_requested) is not bool:raise TypeError('actual current closed intent bool required')
        if not self.postconfirmed_maintenance:
            result=self.delegate.observe(observation,robot,close_requested=close_requested)
            self.latest_attachment=self.delegate.latest_attachment;self.latest_raw_active=self.active.latest
            self.latest_result=result
            result.update(candidate_goal_sphere_filter_v56=self.goal_sphere_filter,postconfirmed_position_maintenance_v56=False)
            return result
        target,trace,error=self._track_once(observation,robot);d=self.delegate;now=robot.timestamp
        if close_requested and not self.active.attempt_active:
            previous=d._last_open_visual
            d.close_visual_baseline=(previous if d._last_close_requested is False and previous is not None
                and abs(now-previous.timestamp-.05)<=1e-6 else None)
        legacy=self.estimator.observe(target,robot,close_requested=close_requested,orientation_observable=False)
        raw_active=self.active.observe(target,robot,close_requested=close_requested)
        raw_event=raw_active['evidence']
        release=(isinstance(raw_event,PublicMotionEvidence) and raw_event.event=='release_confirmed'
            and abs(raw_event.timestamp-now)<=1e-8 and raw_active['mode']=='FREE'
            and raw_active['attempt_active'] is False and not close_requested
            and np.all(robot.finger_open_amounts>=.6))
        score=None
        if self.maintenance.started and not release:
            # A raw current measurement envelope: no successful availability,
            # controlled pose or forecast is manufactured for helper admission.
            measurement=dict(timestamp=now,object_observation=target,attachment_estimate=raw_active,
                Native_inputs_used=False,future_observations_used=False,observations_fabricated=False)
            score=self.maintenance.observe(measurement,robot,close_requested=close_requested)
            evidence=score['evidence'] if score['current_position_maintenance_passed'] else PublicMotionEvidence(
                'manipulation_unresolved',now,('public_robot_pose','public_gripper_opening'))
        else:evidence=raw_event
        # Exactly one bridge call consumes exactly one staged actual target.
        result=self.bridge.observe(observation,robot,close_requested=close_requested,evidence=evidence)
        effective=raw_active
        if score is not None:
            passed=score['current_position_maintenance_passed']
            if passed and not (result.get('mode')=='ATTACHED_PUBLIC' and result.get('available') is True
                    and result.get('controlled_mode_available') is True):
                raise RuntimeError('real bridge rejected maintenance evidence; never promote unavailable state')
            effective=dict(timestamp=now,protocol=MAINTENANCE_PROTOCOL,
                mode='ATTACHED_PUBLIC' if passed else 'UNRESOLVED',evidence=evidence,
                attempt_id=raw_active['attempt_id'],attempt_active=raw_active['attempt_active'],
                probe=deepcopy(raw_active['probe']),initial_probe_proof=deepcopy(score['initial_probe_proof']),
                maintenance=deepcopy(score),current_position_only=True,object_orientation_observable=False,
                force_or_frictional_grasp_proof=False)
        result['attachment_estimate']=effective
        if (not self.maintenance.started and not release and raw_active['mode']=='ATTACHED_PUBLIC'):
            # start() validates the ACTUAL bridge result, including its current
            # available/controlled/body flags. Initial v4 envelope remains on
            # this frame so unchanged recovery records the genuine first proof.
            self.last_initial_admission=self.maintenance.start(result,robot,close_requested=close_requested)
        if release and result.get('mode')=='FREE' and raw_active['attempt_active'] is False:
            if self.maintenance.started:
                self.retired_maintenance.append(dict(attempt_id=self.maintenance.attempt_id,release_time=now,
                    final_receipt=deepcopy(self.maintenance.latest),retirement='fresh_original_typed_public_release'))
            self.maintenance=PostConfirmedPositionMaintenance();self.last_initial_admission=None
        result.update(original_probe_estimator_diagnostic=raw_active,legacy_attachment_diagnostic=legacy,
            active_visual_protocol='original_probe_v4_with_optional_position_maintenance_v56',
            legacy_attachment_controls_bridge=False,public_preclose_visual_baseline=self.close_visual_baseline,
            candidate_goal_sphere_filter_v56=self.goal_sphere_filter,postconfirmed_position_maintenance_v56=True,
            maintenance_is_position_only_candidate_not_contact_certificate=True,
            maintenance_admission=deepcopy(self.last_initial_admission),tracker_injection_fact_retained=True)
        if self.goal_sphere_filter:
            result.update(goal_aware_binding_version='v62',depth_refiner_trace=trace,depth_refiner_tracker_error=error,
                observed_target_available=target is not None,strict_initial_goal_sphere_available=self.tracker.sphere is not None,
                candidate_position_measurement_v62=True,NN_input_sphere_shell_filtered=False,
                goal_sphere_filter_flag_is_initialSphere_and_far_legacy_compatibility=True,
                refiner_extra_metadata_separate_from_raw7_cache=True)
        self.binding.bind_current(result,target);result['approved_tracker_cache']=self.binding.receipt()
        if not close_requested and isinstance(target,ObjectState) and target.visible and target.confidence>=.5:d._last_open_visual=target
        d._last_close_requested=close_requested;d._last_time=now
        d.latest_attachment=effective;d.latest_legacy_diagnostic=legacy;d.latest_result=result
        self.latest_attachment=effective;self.latest_raw_active=raw_active;self.latest_result=result
        return result

    def receipt(self):
        self._validate_components()
        return dict(protocol='public_candidate_position_v62',goal_sphere_filter=self.goal_sphere_filter,
            postconfirmed_maintenance=self.postconfirmed_maintenance,initial_probe_changed=False,
            current_maintenance=deepcopy(self.maintenance.latest),retired_maintenance=deepcopy(self.retired_maintenance),
            current_visual_position_only=True,object_orientation_qualified=False,force_or_contact_qualified=False,
            tracker_cache_binding=self.binding.receipt(),
            color_variant='R100_for_far_legacy_candidate_only',fixed_minimum_red=100,original_minimum_red=170,
            refiner_checkpoint=str(self.tracker.refiner_checkpoint),
            refiner_checkpoint_sha256=self.tracker.expected_refiner_sha256,
            position_measurement_qualified=self.tracker.refiner.position_measurement_qualified,
            model_qualified=False,raw_mask_model_qualified=False,current_refiner_qualified=False,
            NN_input_RGBD_unmodified=True,NN_input_sphere_shell_filtered=False,
            goal_sphere_filter_flag_is_initialSphere_and_far_legacy_compatibility=True,
            NN_first=True,no_near_shell_fallback=True,
            far_consistency_development_only=True,far_semantic_ownership_or_physical_certificate=False,
            far_consistency_known_counterexamples=73,motion_history_route_change_protocol=ROUTE_CHANGE_PROTOCOL,
            box_prior_unchanged=False,box_geometry_prior_unchanged=True,color_prior_changed=True,
            phase_dependent_color_switch_used=False,preclose_bitexact_with_original_R170=False)


class PublicCandidatePositionStateAssemblerV62(PublicGraspStateAssembler):
    def __init__(self,binding):
        PublicStateAssembler.__init__(self)
        if type(binding) is not ApprovedDepthPositionTrackerCacheBindingV62:
            raise TypeError('only the exact v62 dark-red tracker-cache capability is accepted')
        binding.validate_dependencies();self.binding=binding


class ObservedPublicCandidatePositionV62Backend(PublicShadowBackend):
    def __init__(self,backend,output,*,checkpoint,checkpoint_sha256,
                 refiner_checkpoint,refiner_checkpoint_sha256,goal_sphere_filter=True,
                 postconfirmed_maintenance=True,active_config=None,public_bridge=None,recorder=None,
                 save_frames=True,grasp_config=None,require_public_goal_completion=False):
        v62_enabled(goal_sphere_filter=goal_sphere_filter,postconfirmed_maintenance=postconfirmed_maintenance)
        if type(require_public_goal_completion) is not bool:raise ValueError('explicit public completion flag required')
        torch.set_num_threads(2)
        bridge=public_bridge if public_bridge is not None else PublicCandidatePositionBridgeV62(
            refiner_checkpoint=refiner_checkpoint,refiner_checkpoint_sha256=refiner_checkpoint_sha256,
            goal_sphere_filter=goal_sphere_filter,postconfirmed_maintenance=postconfirmed_maintenance,active_config=active_config)
        if type(bridge) is not PublicCandidatePositionBridgeV62 or (bridge.goal_sphere_filter,bridge.postconfirmed_maintenance)!=(goal_sphere_filter,postconfirmed_maintenance):
            raise TypeError('exact v62 bridge with matching feature modifiers required')
        if (bridge.tracker.refiner_checkpoint!=Path(refiner_checkpoint).resolve(strict=True)
                or bridge.tracker.expected_refiner_sha256!=refiner_checkpoint_sha256):
            raise ValueError('external public bridge refiner checkpoint differs from explicit constructor identity')
        bridge._validate_components()
        bridge.binding.validate_dependencies();self.assembler=PublicCandidatePositionStateAssemblerV62(bridge.binding)
        self.latest_robot=None;self.recovery=ConfirmedAttemptRecovery()
        self.public_hold_metric=PublicLiftHoldMetric(grasp_config or ActiveVisualGraspConfig())
        self.require_public_goal_completion=require_public_goal_completion;self.public_goal_completion=None
        torch.set_num_threads(2);self.candidate_cache=PublicRowCache()
        self.pilot_inference=FrozenPilotInference(checkpoint,expected_sha256=checkpoint_sha256)
        super().__init__(backend,output,bridge=bridge,recorder=recorder,save_frames=save_frames,
            controller_uses_public_forecast=True,observer_role='grasp_prototype',attachment_estimator_used=True)

    def sample(self,state=None):
        current=self.backend.read_state() if state is None else state;robot=sanitized_robot(current)
        result=super().sample(current);self.assembler.observe_robot(robot);self.latest_robot=robot
        self.public_hold_metric.observe(result,robot,self.bridge.latest_attachment,self.bridge.close_visual_baseline)
        self.recovery.observe(result,robot)
        index=self.samples-1;close=float(current['requested_opening'])<.5
        self.candidate_cache.observe(pack_public_row(result,robot,close,index,f'rgbd/{index:04d}.npz'))
        if self.require_public_goal_completion:
            if self.public_goal_completion is None and isinstance(result.get('goal_pose'),Pose):
                from .public_grasp_goal_completion_v52 import PublicGraspGoalCompletion
                self.public_goal_completion=PublicGraspGoalCompletion(result['goal_pose'])
            if self.public_goal_completion is not None:
                self.public_goal_completion.observe(result,robot,self.public_hold_metric.receipt(),close_requested=close)
        return result

    def receipt(self):
        result=super().receipt();result.update(candidate_transformer=self.pilot_inference.receipt(),candidate_cache=self.candidate_cache.receipt(),
            actual_public50ms_capture_is_producer=True,live_RGBD_reference_file_check=False,
            live_file_check_scope='sameframe original observer, asynchronous writer may not have drained',
            require_public_goal_completion=self.require_public_goal_completion,
            public_goal_completion=None if self.public_goal_completion is None else self.public_goal_completion.receipt(),
            optional_public_producer=self.bridge.receipt(),geometry_fallback_used=False)
        return result


class CandidatePositionV62Coordinator(CandidateGraspV5Coordinator):
    def decide(self,backend,result,robot,h):
        decision=super().decide(backend,result,robot,h)
        decision['original_probe_estimator_diagnostic']=deepcopy(self.active.latest)
        decision['effective_attachment_estimate']=deepcopy(result.get('attachment_estimate'))
        decision['optional_public_producer']=backend.bridge.receipt()
        return decision


def _validate_v62_runner(backend,config):
    if config.get('candidate_position_measurement_v62',False) is not True:
        raise ValueError('explicit boolean V62 depth-refiner producer flag required')
    goal=config.get('candidate_goal_sphere_filter_v56',False);maintenance=config.get('postconfirmed_position_maintenance_v56',False)
    v62_enabled(goal_sphere_filter=goal,postconfirmed_maintenance=maintenance)
    if config.get('public_grasp_active_visual_v42',False):
        raise ValueError('v42 primary-controller flag cannot replace the candidate-v5 route')
    if (goal,maintenance)!=(backend.bridge.goal_sphere_filter,backend.bridge.postconfirmed_maintenance):
        raise ValueError('runner/observer feature modifiers must match exactly')


def run_public_candidate_position_v62(backend,executor,output,config,*,recorder=None):
    if type(config.get('finger_response_fastpath',False)) is not bool:
        raise ValueError('explicit boolean finger-response computation modifier required')
    _validate_v62_runner(backend,config)
    if type(backend) is not ObservedPublicCandidatePositionV62Backend or executor.backend is not backend:
        raise ValueError('v4 observer must wrap every real50ms executor tick')
    if config.get('public_grasp_candidate_transformer_v5') is not True:raise ValueError('explicit independently versioned v4 protocol required')
    if float(config.get('control_dt',.1))!=.1 or float(config.get('physics_dt',.05))!=.05:raise ValueError('100ms/2actual50ms budget required')
    if backend.samples==0:backend.sample()
    output=Path(output);settings=backend.public_hold_metric.config
    core,helper,provenance=build_response_planner(ROOT,executor.limits,controller='coupled',action_space='joint',
        joint_goal_config={'lower':tuple(config.get('workspace_lower',[-.1,-.65,.755]))},
        terminal_config={'enabled':bool(config.get('terminal_goal_cost',True))},reference_frequency=executor.config.reference_frequency)
    if config.get('finger_response_fastpath',False):
        from .finger_fastpath_integration_20261103 import enable_finger_fastpath
        provenance=enable_finger_fastpath(core.execution_model,provenance)
    coordinator=CandidatePositionV62Coordinator(core,helper,backend.assembler,backend.bridge.active,config=settings,recovery=backend.recovery,inference=backend.pilot_inference,cache=backend.candidate_cache,pi_symmetry_representative=config.get('candidate_pi_symmetry_representative',False),cache_fixed_probe_nominal_ik=config.get('cache_fixed_probe_nominal_ik',False),probe_terminal_at_fixed_deadline=config.get('probe_terminal_at_fixed_deadline',False),probe_onpolicy_joint_response=config.get('probe_onpolicy_joint_response',False))
    provenance=deepcopy(provenance);provenance.update(active_visual_protocol='active_vertical_visual_comotion_v4',
        post_confirmation_unknown_recovery='v41_actualq_hold_keep_request_no_lift_no_expired_probe_reopen',
        criteria=asdict(settings),active_attachment_criteria=asdict(backend.bridge.active.config),
        tracker_cache_binding=backend.bridge.binding.receipt(),legacy_attachment_controls_bridge=False,
        contact_response_qualified=False,force_or_frictional_grasp_proof=False,Native_labels_used_by_policy=False,
        previous_finger_response_checkpoint_requalified_for_completion_latch=False,
        public_hold_requires_actual_visible_4cm_and_halfsecond=True,formal_qualification=False,
        development_assumption='table_supported_target_without_independent_external_vertical_drive')
    provenance.update(candidate_transformer=backend.pilot_inference.receipt(),geometry_fallback_used=False,task_classifier_trained=False,calibrated_candidate_success_probability=False,candidate_pi_symmetry_representative=coordinator.pi_symmetry_representative,cache_fixed_probe_nominal_ik=coordinator.cache_fixed_probe_nominal_ik,probe_terminal_at_fixed_deadline=coordinator.probe_terminal_at_fixed_deadline,probe_onpolicy_joint_response=coordinator.probe_onpolicy_joint_response)
    provenance.update(optional_public_producer=backend.bridge.receipt())
    write_new(output/'public_candidate_position_v62_provenance.json',provenance)
    maximum=int(config.get('public_grasp_max_decisions',np.ceil(float(config.get('simulation_seconds',6.))/.1)))
    if maximum<=0:raise ValueError('positive decision budget required')
    cycles=0;times=[];ever=False;stop='budget_exhausted'
    while cycles<maximum and backend.remaining_ticks()>=2:
        if any(p.exists() for p in (output/'STOP',output.parent/'STOP',ROOT/'runtime/framework/local_mpc_20261102/STOP')):stop='administrative_stop';break
        if _public_completion_ready(backend):stop=_public_completion_reason(backend);break
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
        receipt=executor.execute_joint_target(goal,opening=opening,phase='P4_candidate_transformer_v62_'+decision['mode'])
        ever=True;after=backend.latest_robot
        append(output/'public_candidate_position_v62_decisions.jsonl',dict(index=cycles,time=robot.timestamp,public_robot=robot,
            actuator_state_before=h,decision=decision,action_rejected_reason=rejected,action_qualification=qualification,
            executed_action=executed,executor=receipt,public_robot_after=after,actuator_state_after=deepcopy(executor.state()),
            active_visual_after=deepcopy(backend.bridge.active.latest),recovery_after=backend.recovery.receipt(),public_lift_hold=backend.public_hold_metric.receipt(),Native_labels_used_by_policy=False))
        cycles+=1;times.append(time.perf_counter()-started)
        if after is None or after.timestamp<=robot.timestamp+1e-8:stop='no_actual_public_tick';break
        if _public_completion_ready(backend):stop=_public_completion_reason(backend);break
    return dict(stage='P4_public_candidate_position_v62_development',complete=True,cycles=cycles,canary_stop_reason=stop,
        public_lift_hold_observed=backend.public_hold_metric.passed,public_lift_hold=backend.public_hold_metric.receipt(),
        Native_task_success_claimed=False,Native_success_exit_used=False,Native_labels_used_by_policy=False,
        contact_response_qualified=False,formal_qualification=False,precision_qualified=False,
        force_or_frictional_grasp_proof=False,mode_events=coordinator.mode_events,controller_provenance=provenance,
        Transformer_used=backend.pilot_inference.calls>0,candidate_transformer=backend.pilot_inference.receipt(),candidate_events=coordinator.geometry.events,candidate_public_cache=backend.candidate_cache.receipt(),geometry_fallback_used=False,
        require_public_goal_completion=backend.require_public_goal_completion,public_goal_completion=None if backend.public_goal_completion is None else backend.public_goal_completion.receipt(),
        optional_public_producer=backend.bridge.receipt(),effective_attachment_final=deepcopy(backend.bridge.latest_attachment),
        active_visual_final=deepcopy(backend.bridge.active.latest),recovery_final=backend.recovery.receipt(),full_cycle_wall_seconds=quantiles(times),full_cycle_wall_seconds_samples=times,real_time_guaranteed=False)

