"""One natural fresh-public verification epoch; frozen V64 perception/control.

The failed original owner is retained. Current visibility only permits one
decision-time witness admission; it never asserts attachment. A new owner is
created only after a complete actual50ms co-motion witness and explicit commit.
Complete proof/spec trees are logged privately, never sent through P4 inputs.
"""
from copy import deepcopy
from pathlib import Path
import hashlib
import time
import numpy as np
import torch

from ..contracts import ObjectState,Pose
from . import public_candidate_position_v64 as base
from .public_motion_bridge import PublicRobotSensors,PublicMotionEvidence
from .active_visual_attachment import ActiveVisualAttachment
from .postconfirmed_maintenance_candidate_20261103 import PostConfirmedPositionMaintenance
from .public_reverification_epoch_prototype_20261103 import PublicReverificationEpochPrototype
from .public_reverified_maintenance_owner_20261103_v1 import PublicReverifiedMaintenanceOwner
from .public_grasp_v41 import ROOT,append,write_new,quantiles,asdict,build_response_planner
from .public_grasp_candidates_v5 import _public_completion_ready,_public_completion_reason

PROTOCOL=ROOT/'artifacts/framework/local_mpc_20261102/V65_reverification_development_protocol_v1/protocol.json'
PROTOCOL_SHA256='4ddd3fb870148c2f0eebc3c8ee563d54843d79900104ed7d76bccf7b1fca20df'
EXTRA_SOURCES={
    'our_mani/local_mpc_control_20261103/public_reverification_epoch_prototype_20261103.py':'1f6809f705cdb6288e45ca876c63d5626e49e7d4808719f7f447a648ead18173',
    'our_mani/local_mpc_control_20261103/public_reverified_maintenance_owner_20261103_v1.py':'b3c18a24d09ab49619c5b8124584ae9ff55c7e687b19638080cf20b00e3c0dc4',
    str(PROTOCOL.relative_to(ROOT)):PROTOCOL_SHA256,
}
EPOCH_PROTOCOL='explicit_reverified_epoch_current_public_position_maintenance_v1'


def _sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def _stat(path):
    s=Path(path).stat();return(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)


def position_producer_source_map():
    sources=base.position_producer_source_map()
    for name,expected in EXTRA_SOURCES.items():
        path=ROOT/name
        if _sha(path)!=expected:raise ValueError('Exact natural re-verification dependency required: '+name)
        sources[str(path.resolve())]=expected
    sources[str(Path(__file__).resolve())]=_sha(__file__)
    return sources


def _same_robot_values(left,right):
    return (type(left) is PublicRobotSensors and type(right) is PublicRobotSensors
        and left.timestamp==right.timestamp
        and all(np.array_equal(getattr(left,k),getattr(right,k))for k in
            ('q','dq','tcp_position','tcp_quaternion','finger_open_amounts')))


def _unresolved(now):
    return PublicMotionEvidence('manipulation_unresolved',now,('public_robot_pose','public_gripper_opening'))


class PublicCandidateReverificationBridgeV65:
    def __init__(self,*,refiner_checkpoint,refiner_checkpoint_sha256,
                 goal_sphere_filter=True,postconfirmed_maintenance=True,active_config=None):
        self.delegate=base.PublicCandidatePositionBridgeV64(refiner_checkpoint=refiner_checkpoint,
            refiner_checkpoint_sha256=refiner_checkpoint_sha256,goal_sphere_filter=goal_sphere_filter,
            postconfirmed_maintenance=postconfirmed_maintenance,active_config=active_config)
        for name in ('tracker','cache','bridge','binding','estimator','active'):
            setattr(self,name,getattr(self.delegate,name))
        self.goal_sphere_filter=goal_sphere_filter;self.postconfirmed_maintenance=postconfirmed_maintenance
        sources=position_producer_source_map();self._source_stats={p:_stat(p)for p in sources}
        self.witness=None;self.epoch_owner=None;self.used_attempt_ids=set();self.retired_reverification=[]
        self.latest_result=self.latest_attachment=self.latest_raw_active=None
        self._frame=None;self._inner_robot=None;self._last_close=None;self.last_admission=None
        self._validate_components()

    @property
    def maintenance(self):return self.delegate.maintenance
    @property
    def close_visual_baseline(self):return self.delegate.close_visual_baseline
    @property
    def verification_epoch(self):return 0 if self.epoch_owner is None else 1

    def _validate_components(self):
        if type(self.delegate) is not base.PublicCandidatePositionBridgeV64:
            raise TypeError('Exact unchanged V64 primitive producer required')
        self.delegate._validate_components()
        if (self.goal_sphere_filter,self.postconfirmed_maintenance)!=(True,True):
            raise ValueError('Frozen V64 measurement and original maintenance are required')
        for name in ('tracker','cache','bridge','binding','estimator','active'):
            if getattr(self,name)is not getattr(self.delegate,name):raise ValueError('Detached original primitive: '+name)
        if getattr(self.active.observe,'__func__',None)is not ActiveVisualAttachment.observe:
            raise ValueError('Original initial probe/attempt owner method required')
        if self.witness is not None and type(self.witness)is not PublicReverificationEpochPrototype:
            raise TypeError('Exact isolated natural witness required')
        if self.witness is not None:
            for name in ('observe','begin_witness','commit_verified_epoch'):
                if getattr(getattr(self.witness,name),'__func__',None)is not getattr(PublicReverificationEpochPrototype,name):
                    raise ValueError('Original natural witness method required: '+name)
        if self.epoch_owner is not None and type(self.epoch_owner)is not PublicReverifiedMaintenanceOwner:
            raise TypeError('Exact explicitly committed current owner required')
        if (self.epoch_owner is not None and getattr(self.epoch_owner.observe,'__func__',None)
                is not PublicReverifiedMaintenanceOwner.observe):
            raise ValueError('Original committed current maintenance method required')
        for name,value in self._source_stats.items():
            if _stat(name)!=value:raise ValueError('Natural re-verification source changed: '+name)

    def reset(self):
        if self.active.attempt_active:raise ValueError('An active failed physical attempt cannot be reset by re-verification')
        self.delegate.reset();self.witness=None;self.epoch_owner=None;self.used_attempt_ids=set();self.retired_reverification=[]
        self.latest_result=self.latest_attachment=self.latest_raw_active=None
        self._frame=self._inner_robot=self._last_close=self.last_admission=None

    def _frame_from_observation(self,target,raw_active,robot):
        if (self.binding.last_frame_timestamp!=robot.timestamp
            or self.tracker._last_observed_timestamp!=robot.timestamp
            or target is not self.tracker.last_current_target):
            raise ValueError('Witness must use this exact current tracker target and robot frame')
        # Explicit public physical-attempt envelope, with no full spec/proof.
        attempt={k:deepcopy(raw_active[k])for k in ('timestamp','protocol','attempt_id','attempt_active')}
        return dict(timestamp=robot.timestamp,object_observation=target,
            original_probe_estimator_diagnostic=attempt,Native_inputs_used=False,
            future_observations_used=False,observations_fabricated=False)

    def _ensure_witness(self):
        old=self.maintenance
        if self.witness is None and old.started and old.failed and self.close_visual_baseline is not None:
            self.witness=PublicReverificationEpochPrototype(old,self.close_visual_baseline)

    def try_begin_natural_witness(self,result,robot,h):
        """Only a decision with the current exact bound result and fresh h admits."""
        self._validate_components();self.binding.approve(result)
        if result is not self.latest_result or not _same_robot_values(robot,self._inner_robot):
            raise ValueError('Same observed inner robot values/current bound result required')
        p=self.witness;target=self.binding.latest_visual_target
        if (p is None or self.epoch_owner is not None or self.active.attempt_id in self.used_attempt_ids
            or self._last_close is not True or type(target)is not ObjectState or not target.visible
            or target.confidence<.5 or target.timestamp!=robot.timestamp or robot.timestamp<=p.failure_time+1e-8):
            return None
        # First currently visible decision is used once even if some original
        # reference or hardware admission check prevents beginning the witness.
        self.used_attempt_ids.add(self.active.attempt_id)
        if (self._frame is not p.last_result or self._inner_robot is not p.last_robot
            or self._frame['object_observation']is not target):
            raise ValueError('Exact observed raw envelope/inner robot identity required')
        spec=p.begin_witness(self._frame,self._inner_robot,h,mode='natural')
        self.last_admission=dict(timestamp=robot.timestamp,physical_attempt_id=self.active.attempt_id,
            admitted=spec is not None,natural_motion_only=True,active_test_action_executed=False)
        return deepcopy(self.last_admission)

    def public_summary(self):
        p=self.witness;o=self.epoch_owner
        return dict(protocol='one_natural_public_reverification_epoch_v65',
            timestamp=None if self._inner_robot is None else self._inner_robot.timestamp,
            physical_attempt_id=self.active.attempt_id,verification_epoch=self.verification_epoch,
            admission_attempted=self.active.attempt_id in self.used_attempt_ids,
            witness_started=p is not None and p.spec is not None,
            witness_failed=False if p is None else bool(p.witness_failed),
            epoch_owner_failed=False if o is None else bool(o.failed),
            current_reverified_position_maintenance_passed=bool(o is not None and o.latest and o.latest['current_reverified_position_maintenance_passed']),
            natural_motion_only=True,active_test_action_executed=False,
            old_failed_owner_reset=False,original_references_rolled=False,
            initial_probe_changed=False,force_or_contact_qualified=False)

    def private_diagnostics(self):
        return dict(public_summary=self.public_summary(),last_admission=deepcopy(self.last_admission),
            original_maintenance=deepcopy(self.maintenance.latest),
            witness=None if self.witness is None else dict(latest=deepcopy(self.witness.latest),
                spec=deepcopy(self.witness.spec),trace=deepcopy(self.witness.trace),committed=deepcopy(self.witness.committed)),
            epoch_owner=None if self.epoch_owner is None else deepcopy(self.epoch_owner.latest),
            retired=deepcopy(self.retired_reverification))

    def observe(self,observation,robot,*,close_requested):
        self._validate_components()
        if not isinstance(robot,PublicRobotSensors):robot=PublicRobotSensors.from_mapping(robot)
        if type(close_requested)is not bool:raise TypeError('Actual close intent required')
        d=self.delegate;primitive=d.delegate;now=robot.timestamp
        target,trace,error=d._track_once(observation,robot)
        if close_requested and not self.active.attempt_active:
            previous=primitive._last_open_visual
            primitive.close_visual_baseline=(previous if primitive._last_close_requested is False and previous is not None
                and abs(now-previous.timestamp-.05)<=1e-6 else None)
        legacy=self.estimator.observe(target,robot,close_requested=close_requested,orientation_observable=False)
        raw_active=self.active.observe(target,robot,close_requested=close_requested);raw_event=raw_active['evidence']
        release=(isinstance(raw_event,PublicMotionEvidence)and raw_event.event=='release_confirmed'
            and abs(raw_event.timestamp-now)<=1e-8 and raw_active['mode']=='FREE'
            and raw_active['attempt_active']is False and not close_requested and np.all(robot.finger_open_amounts>=.6))
        frame=self._frame_from_observation(target,raw_active,robot)
        self._frame=frame;self._inner_robot=robot;self._last_close=close_requested
        old_score=None;epoch_score=None
        if self.maintenance.started and not release:
            old_measurement=dict(timestamp=now,object_observation=target,attachment_estimate=raw_active,
                Native_inputs_used=False,future_observations_used=False,observations_fabricated=False)
            old_score=self.maintenance.observe(old_measurement,robot,close_requested=close_requested)
            evidence=old_score['evidence']if old_score['current_position_maintenance_passed']else _unresolved(now)
        else:evidence=raw_event
        if not release:
            self._ensure_witness()
            if self.epoch_owner is not None:
                epoch_score=self.epoch_owner.observe(frame,robot,close_requested=close_requested)
            elif self.witness is not None:
                receipt=self.witness.observe(frame,robot,close_requested=close_requested)
                if receipt['current_fresh_witness_passed']:
                    self.witness.commit_verified_epoch(frame,robot)
                    self.epoch_owner=PublicReverifiedMaintenanceOwner(self.witness)
                    # The constructor owns the commit frame. Calling observe
                    # again here would be a duplicate actual sensor frame.
                    epoch_score=deepcopy(self.epoch_owner.latest)
            if epoch_score is not None:
                evidence=epoch_score['evidence']if epoch_score['current_reverified_position_maintenance_passed']else _unresolved(now)
        result=self.bridge.observe(observation,robot,close_requested=close_requested,evidence=evidence)
        effective=raw_active
        if epoch_score is not None:
            passed=epoch_score['current_reverified_position_maintenance_passed']
            if passed and not(result.get('available')is True and result.get('controlled_mode_available')is True and result.get('mode')=='ATTACHED_PUBLIC'):
                raise RuntimeError('Real bridge rejected current epoch evidence; never fabricate availability')
            effective=dict(timestamp=now,protocol=EPOCH_PROTOCOL,mode='ATTACHED_PUBLIC'if passed else'UNRESOLVED',
                evidence=evidence,attempt_id=raw_active['attempt_id'],attempt_active=raw_active['attempt_active'],
                verification_epoch=1,current_position_only=True,object_orientation_observable=False,
                force_or_frictional_grasp_proof=False)
        elif old_score is not None:
            passed=old_score['current_position_maintenance_passed']
            if passed and not(result.get('available')is True and result.get('controlled_mode_available')is True and result.get('mode')=='ATTACHED_PUBLIC'):
                raise RuntimeError('Real bridge rejected original current maintenance evidence')
            effective=dict(timestamp=now,protocol=base.MAINTENANCE_PROTOCOL,
                mode='ATTACHED_PUBLIC'if passed else'UNRESOLVED',evidence=evidence,
                attempt_id=raw_active['attempt_id'],attempt_active=raw_active['attempt_active'],
                probe=deepcopy(raw_active['probe']),initial_probe_proof=deepcopy(old_score['initial_probe_proof']),
                maintenance=deepcopy(old_score),current_position_only=True,object_orientation_observable=False,
                force_or_frictional_grasp_proof=False)
        result['attachment_estimate']=effective
        if not self.maintenance.started and not release and raw_active['mode']=='ATTACHED_PUBLIC':
            d.last_initial_admission=self.maintenance.start(result,robot,close_requested=close_requested)
        if release and result.get('mode')=='FREE'and raw_active['attempt_active']is False:
            if self.maintenance.started:
                d.retired_maintenance.append(dict(attempt_id=self.maintenance.attempt_id,release_time=now,
                    final_receipt=deepcopy(self.maintenance.latest),retirement='fresh_original_typed_public_release'))
            if self.witness is not None:self.retired_reverification.append(self.private_diagnostics())
            d.maintenance=PostConfirmedPositionMaintenance();d.last_initial_admission=None
            self.witness=None;self.epoch_owner=None;self.last_admission=None
        result.update(original_probe_estimator_diagnostic=raw_active,legacy_attachment_diagnostic=legacy,
            active_visual_protocol='original_probe_then_current_position_maintenance_with_one_natural_epoch_v65',
            legacy_attachment_controls_bridge=False,public_preclose_visual_baseline=self.close_visual_baseline,
            candidate_goal_sphere_filter_v56=True,postconfirmed_position_maintenance_v56=True,
            maintenance_is_position_only_candidate_not_contact_certificate=True,
            maintenance_admission=deepcopy(d.last_initial_admission),tracker_injection_fact_retained=True,
            goal_aware_binding_version='v64',depth_refiner_trace=trace,depth_refiner_tracker_error=error,
            observed_target_available=target is not None,strict_initial_goal_sphere_available=self.tracker.sphere is not None,
            candidate_position_measurement_v64=True,NN_input_sphere_shell_filtered=False,
            goal_sphere_filter_flag_is_initialSphere_and_far_legacy_compatibility=True,
            refiner_extra_metadata_separate_from_raw7_cache=True,
            public_reverification_epoch_v65=True,public_reverification=self.public_summary())
        self.binding.bind_current(result,target);result['approved_tracker_cache']=self.binding.receipt()
        if not close_requested and isinstance(target,ObjectState)and target.visible and target.confidence>=.5:primitive._last_open_visual=target
        primitive._last_close_requested=close_requested;primitive._last_time=now
        for owner in (primitive,d,self):
            owner.latest_attachment=effective;owner.latest_legacy_diagnostic=legacy;owner.latest_result=result
        d.latest_raw_active=self.latest_raw_active=raw_active
        return result

    def receipt(self):
        self._validate_components()
        return dict(protocol='public_candidate_reverification_v65',public_reverification=self.public_summary(),
            tracker_cache_binding=self.binding.receipt(),initial_probe_changed=False,MPC_cost_and_budget_changed=False,
            current_visual_position_only=True,object_orientation_qualified=False,force_or_contact_qualified=False,
            position_measurement_qualified=True,model_qualified=False,raw_mask_model_qualified=False,
            refiner_checkpoint=str(self.tracker.refiner_checkpoint),refiner_checkpoint_sha256=self.tracker.expected_refiner_sha256,
            full_witness_proof_logged_privately=True,old_failed_owner_reset=False,original_references_rolled=False)


class ObservedPublicCandidateReverificationV65Backend(base.PublicShadowBackend):
    def __init__(self,backend,output,*,checkpoint,checkpoint_sha256,refiner_checkpoint,refiner_checkpoint_sha256,
                 goal_sphere_filter=True,postconfirmed_maintenance=True,active_config=None,public_bridge=None,
                 recorder=None,save_frames=True,grasp_config=None,require_public_goal_completion=False):
        if type(require_public_goal_completion)is not bool:raise ValueError('Explicit public goal flag required')
        torch.set_num_threads(2)
        bridge=public_bridge if public_bridge is not None else PublicCandidateReverificationBridgeV65(
            refiner_checkpoint=refiner_checkpoint,refiner_checkpoint_sha256=refiner_checkpoint_sha256,
            goal_sphere_filter=goal_sphere_filter,postconfirmed_maintenance=postconfirmed_maintenance,active_config=active_config)
        if type(bridge)is not PublicCandidateReverificationBridgeV65:
            raise TypeError('Exact independent natural-epoch bridge required')
        if (bridge.goal_sphere_filter,bridge.postconfirmed_maintenance)!=(goal_sphere_filter,postconfirmed_maintenance):
            raise ValueError('Matching frozen measurement modifiers required')
        if (bridge.tracker.refiner_checkpoint!=Path(refiner_checkpoint).resolve(strict=True)
            or bridge.tracker.expected_refiner_sha256!=refiner_checkpoint_sha256):raise ValueError('Exact configured checkpoint identity required')
        bridge._validate_components();self.assembler=base.PublicCandidatePositionStateAssemblerV64(bridge.binding)
        self.latest_robot=None;self.recovery=base.ConfirmedAttemptRecovery()
        self.public_hold_metric=base.PublicLiftHoldMetric(grasp_config or base.ActiveVisualGraspConfig())
        self.require_public_goal_completion=require_public_goal_completion;self.public_goal_completion=None
        self.candidate_cache=base.PublicRowCache();self.pilot_inference=base.FrozenPilotInference(checkpoint,expected_sha256=checkpoint_sha256)
        super().__init__(backend,output,bridge=bridge,recorder=recorder,save_frames=save_frames,
            controller_uses_public_forecast=True,observer_role='grasp_prototype',attachment_estimator_used=True)

    def sample(self,state=None):
        current=self.backend.read_state()if state is None else state;robot=base.sanitized_robot(current)
        result=super().sample(current);self.assembler.observe_robot(robot);self.latest_robot=robot
        self.public_hold_metric.observe(result,robot,self.bridge.latest_attachment,self.bridge.close_visual_baseline)
        self.recovery.observe(result,robot)
        append(self.output/'position_diagnostics_extra.jsonl',dict(index=self.samples-1,timestamp=robot.timestamp,
            tracker_diagnostics=deepcopy(self.bridge.tracker.last_measurement_metadata)))
        append(self.output/'reverification_diagnostics_private.jsonl',dict(index=self.samples-1,timestamp=robot.timestamp,
            diagnostics=self.bridge.private_diagnostics()))
        index=self.samples-1;close=float(current['requested_opening'])<.5
        self.candidate_cache.observe(base.pack_public_row(result,robot,close,index,f'rgbd/{index:04d}.npz'))
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
            require_public_goal_completion=self.require_public_goal_completion,
            public_goal_completion=None if self.public_goal_completion is None else self.public_goal_completion.receipt(),
            optional_public_producer=self.bridge.receipt(),geometry_fallback_used=False,
            full_reverification_diagnostics_private=True)
        return result


class CandidateReverificationV65Coordinator(base.CandidatePositionV64Coordinator):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self._seen_reverification_epoch=None
    def decide(self,backend,result,robot,h):
        if type(backend.bridge)is not PublicCandidateReverificationBridgeV65:raise TypeError('Exact V65 observer bridge required')
        self.assembler._inputs(result,robot,h,self.phase)
        admission=backend.bridge.try_begin_natural_witness(result,robot,h)
        epoch=(backend.bridge.active.attempt_id,backend.bridge.verification_epoch)
        if self._seen_reverification_epoch!=epoch and epoch[1]:
            self.warm=None;self.phase=None;self.hold_tcp_quaternion=None
        self._seen_reverification_epoch=epoch
        decision=super().decide(backend,result,robot,h)
        decision.update(public_reverification=backend.bridge.public_summary(),natural_witness_admission=admission,
                        active_new_reverification_probe_executed=False)
        return decision


def _validate_v65_runner(backend,config):
    if config.get('public_reverification_epoch_v65')is not True:raise ValueError('Explicit independent natural re-verification flag required')
    base._validate_v64_runner(backend,config)


def run_public_candidate_reverification_v65(backend,executor,output,config,*,recorder=None):
    if type(config.get('finger_response_fastpath',False)) is not bool:
        raise ValueError('explicit boolean finger-response computation modifier required')
    _validate_v65_runner(backend,config)
    if type(backend) is not ObservedPublicCandidateReverificationV65Backend or executor.backend is not backend:
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
    coordinator=CandidateReverificationV65Coordinator(core,helper,backend.assembler,backend.bridge.active,config=settings,recovery=backend.recovery,inference=backend.pilot_inference,cache=backend.candidate_cache,pi_symmetry_representative=config.get('candidate_pi_symmetry_representative',False),cache_fixed_probe_nominal_ik=config.get('cache_fixed_probe_nominal_ik',False),probe_terminal_at_fixed_deadline=config.get('probe_terminal_at_fixed_deadline',False),probe_onpolicy_joint_response=config.get('probe_onpolicy_joint_response',False))
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
    write_new(output/'public_candidate_reverification_v65_provenance.json',provenance)
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
        receipt=executor.execute_joint_target(goal,opening=opening,phase='P4_candidate_transformer_v65_'+decision['mode'])
        ever=True;after=backend.latest_robot
        append(output/'public_candidate_position_v65_decisions.jsonl',dict(index=cycles,time=robot.timestamp,public_robot=robot,
            actuator_state_before=h,decision=decision,action_rejected_reason=rejected,action_qualification=qualification,
            executed_action=executed,executor=receipt,public_robot_after=after,actuator_state_after=deepcopy(executor.state()),
            active_visual_after=deepcopy(backend.bridge.active.latest),recovery_after=backend.recovery.receipt(),public_lift_hold=backend.public_hold_metric.receipt(),Native_labels_used_by_policy=False))
        cycles+=1;times.append(time.perf_counter()-started)
        if after is None or after.timestamp<=robot.timestamp+1e-8:stop='no_actual_public_tick';break
        if _public_completion_ready(backend):stop=_public_completion_reason(backend);break
    return dict(stage='P4_public_candidate_reverification_v65_development',complete=True,cycles=cycles,canary_stop_reason=stop,
        public_lift_hold_observed=backend.public_hold_metric.passed,public_lift_hold=backend.public_hold_metric.receipt(),
        Native_task_success_claimed=False,Native_success_exit_used=False,Native_labels_used_by_policy=False,
        contact_response_qualified=False,formal_qualification=False,precision_qualified=False,
        force_or_frictional_grasp_proof=False,mode_events=coordinator.mode_events,controller_provenance=provenance,
        Transformer_used=backend.pilot_inference.calls>0,candidate_transformer=backend.pilot_inference.receipt(),candidate_events=coordinator.geometry.events,candidate_public_cache=backend.candidate_cache.receipt(),geometry_fallback_used=False,
        require_public_goal_completion=backend.require_public_goal_completion,public_goal_completion=None if backend.public_goal_completion is None else backend.public_goal_completion.receipt(),
        optional_public_producer=backend.bridge.receipt(),effective_attachment_final=deepcopy(backend.bridge.latest_attachment),
        active_visual_final=deepcopy(backend.bridge.active.latest),recovery_final=backend.recovery.receipt(),full_cycle_wall_seconds=quantiles(times),full_cycle_wall_seconds_samples=times,real_time_guaranteed=False)
