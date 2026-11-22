"""Independent fresh-public re-verification owner, NEVER wired to V64.

The old failed owner remains immutable. Reappearance only permits a test;
current actual co-motion is required before an explicit new epoch commit.
No simulator, actuator execution, forecast, controlled-pose input or learning.
"""
from copy import deepcopy
import hashlib,json
import numpy as np

from ..contracts import ObjectState,Pose
from ..local_mpc_20261102 import mpc
from ..local_mpc_p4_20261103.runtime import _walk_public
from .public_motion_bridge import PublicRobotSensors,PublicMotionEvidence
from .active_visual_attachment import ActiveVisualAttachment,ActiveVisualConfig
from .postconfirmed_maintenance_candidate_20261103 import PostConfirmedPositionMaintenance
from .public_grasp_probe import VerificationProbeConfig

VERSION='independent_public_reverification_epoch_prototype_20261103_v1'


def _digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False,
        default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x.item() if isinstance(x,np.generic) else None).encode()).hexdigest()


class PublicReverificationEpochPrototype:
    """One fixed witness per instance; no reset of any previous failure.

    Input source binding must be authenticated by a future outer integration;
    this CPU prototype validates typed current/public contracts, not physical
    sensor origin. A returned proposal is NOT an executed or feasible action.
    """
    def __init__(self,failed_maintenance,original_preclose_baseline):
        old=failed_maintenance
        if (type(old) is not PostConfirmedPositionMaintenance or old.started is not True or old.failed is not True
            or type(old.attempt_id) is not int or old.attempt_id<=0 or old.failure_time is None
            or not np.isfinite(old.failure_time) or not isinstance(old.proof,dict)):
            raise ValueError('Exact already failed original maintenance owner required')
        if (type(original_preclose_baseline) is not ObjectState or original_preclose_baseline.object_id!='target'
            or not original_preclose_baseline.visible or original_preclose_baseline.confidence<.5
            or original_preclose_baseline.timestamp>=old.proof['probe_trigger_time']):
            raise ValueError('Original preclose visible public body baseline, never a current reset anchor')
        self.old=old;self.attempt_id=old.attempt_id;self.failure_time=float(old.failure_time)
        self.r0=np.asarray(old.r0,float).copy();self.object_anchor=np.asarray(old.object_anchor,float).copy()
        if self.r0.shape!=(3,) or self.object_anchor.shape!=(3,) or not np.isfinite([self.r0,self.object_anchor]).all():
            raise ValueError('Finite original fixed references required')
        self.original_proof=deepcopy(old.proof);self._proof_digest=_digest(old.proof)
        self._baseline=deepcopy(original_preclose_baseline)
        self._baseline_digest=_digest(dict(timestamp=self._baseline.timestamp,position=self._baseline.pose.position))
        self.inner=None;self.spec=None;self._spec_digest=None;self.witness_mode=None;self.witness_failed=False;self.first_failure=None
        self.trace=[]
        self.last_timestamp=None;self.last_result=None;self.last_robot=None;self.last_target=None
        self.last_close=None;self._frame_digest=None;self.verification_epoch=0;self.committed=None;self.latest=None

    @property
    def original_preclose_baseline(self):return deepcopy(self._baseline)

    def _unchanged(self):
        if (self.old.started is not True or self.old.failed is not True or self.old.attempt_id!=self.attempt_id
            or self.old.failure_time!=self.failure_time or not np.array_equal(self.old.r0,self.r0)
            or not np.array_equal(self.old.object_anchor,self.object_anchor) or _digest(self.old.proof)!=self._proof_digest
            or _digest(dict(timestamp=self._baseline.timestamp,position=self._baseline.pose.position))!=self._baseline_digest):
            raise ValueError('Original failed proof and preclose references must remain unchanged')
        if self.spec is not None and (_digest(self.spec)!=self._spec_digest
                or self.inner is None or _digest(self.inner.spec)!=self._spec_digest):
            raise ValueError('Fixed witness identity, anchors and deadline cannot change')

    def _current(self,result,robot,close_requested):
        self._unchanged()
        if type(robot) is not PublicRobotSensors or type(close_requested) is not bool:
            raise TypeError('Actual typed robot and known current close intent required')
        if not isinstance(result,dict):raise TypeError('Actual public result dictionary required')
        _walk_public(result,typed=True)
        now=robot.timestamp
        if (not isinstance(result.get('timestamp'),(int,float)) or isinstance(result.get('timestamp'),bool)
            or not np.isfinite(result['timestamp']) or abs(result['timestamp']-now)>1e-8
            or now<=self.failure_time+1e-8
            or any(result.get(k) is not False for k in ('Native_inputs_used','future_observations_used','observations_fabricated'))):
            return None,'not_fresh_current_public_after_original_failure'
        a=result.get('original_probe_estimator_diagnostic',result.get('attachment_estimate'))
        if (not isinstance(a,dict) or a.get('protocol')!='active_vertical_visual_comotion_v4'
            or type(a.get('attempt_id')) is not int or a['attempt_id']!=self.attempt_id
            or a.get('attempt_active') is not True or not isinstance(a.get('timestamp'),(int,float))
            or isinstance(a.get('timestamp'),bool) or not np.isfinite(a['timestamp'])
            or abs(a['timestamp']-now)>1e-8):
            return None,'same_current_physical_close_attempt_required'
        if not close_requested:return None,'known_open_intent_no_reverification'
        target=result.get('object_observation')
        if (type(target) is not ObjectState or target.object_id!='target' or target.visible is not True
            or target.confidence<.5 or not np.isfinite(target.timestamp) or abs(target.timestamp-now)>1e-8):
            return None,'current_visible_body_missing_stale_or_lowconfidence'
        relative=robot.tcp_pose.inverse_apply(target.pose.position)
        if np.linalg.norm(relative-self.r0)>.010+1e-10:return None,'fixed_original_r0_over10mm_no_anchor_laundering'
        if target.pose.position[2]-self.object_anchor[2]<.008-1e-10:return None,'original_probe_height_baseline_not_preserved'
        return target,None

    def _receipt(self,now,reason,passed=False):
        self.latest=dict(timestamp=float(now),version=VERSION,reason=reason,
            current_fresh_witness_passed=bool(passed),verification_epoch=self.verification_epoch,
            physical_attempt_id=self.attempt_id,witness_started=self.spec is not None,
            witness_mode=self.witness_mode,witness_failed=self.witness_failed,first_failure=deepcopy(self.first_failure),
            witness=deepcopy(None if self.inner is None else self.inner.latest),probe_spec=deepcopy(self.spec),
            current_whole_trace_metrics=self._metrics(),
            original_failed_owner_unchanged=True,original_fixed_r0=self.r0.copy(),
            original_object_anchor=self.object_anchor.copy(),original_preclose_baseline=deepcopy(self._baseline),
            original_success_height_baseline_not_rolled=True,static_reappearance_is_not_attachment=True,
            controlled_or_predicted_pose_used=False,Native_inputs_used=False,
            production_integrated=False,outer_source_binding_required_for_production=True,
            force_or_frictional_proof=False,full_SE3_attachment_qualified=False,
            continuous_physical_safety_certified=False,proposal_is_executed_action=False)
        return deepcopy(self.latest)

    def _metrics(self):
        if not self.trace:return dict(real_probe_frames=0,entire_probe_trace_used=True)
        first,last=self.trace[0],self.trace[-1]
        return dict(real_probe_frames=len(self.trace),duration_seconds=last['time']-first['time'],
            TCP_signed_vertical_m=float(last['tcp'][2]-first['tcp'][2]),
            object_signed_vertical_m=float(last['object'][2]-first['object'][2]),
            maximum_fixed_r0_deviation_m=max(float(np.linalg.norm(x['relative']-first['relative'])) for x in self.trace),
            entire_probe_trace_used=True,reference_is_probe_start_not_rolling_or_mean=True)

    def _frame_identity(self,result,robot):
        target=result.get('object_observation');a=result.get('original_probe_estimator_diagnostic',result.get('attachment_estimate'))
        body=None if type(target) is not ObjectState else dict(object_id=target.object_id,timestamp=target.timestamp,
            position=target.pose.position,rotation=target.pose.rotation,points=target.points_world,
            confidence=target.confidence,visible=target.visible)
        return _digest(dict(timestamp=result['timestamp'],target=body,attempt=a,
            robot={k:getattr(robot,k) for k in ('timestamp','tcp_position','tcp_quaternion','q','dq','finger_open_amounts')},
            close=self.last_close))

    def _fail(self,now,reason):
        self.witness_failed=True
        if self.first_failure is None:self.first_failure=dict(time=float(now),reason=reason)
        return self._receipt(now,reason)

    def observe(self,result,robot,*,close_requested):
        target,error=self._current(result,robot,close_requested);now=robot.timestamp
        if self.last_timestamp is not None and now<=self.last_timestamp:
            raise ValueError('Unique increasing actual feedback required')
        gap=None if self.last_timestamp is None else now-self.last_timestamp
        self.last_timestamp=now;self.last_result=result;self.last_robot=robot
        self.last_target=target;self.last_close=close_requested
        self._frame_digest=self._frame_identity(result,robot)
        if self.verification_epoch:return self._receipt(now,'epoch_already_committed_no_second_reset')
        if self.witness_failed:return self._receipt(now,'failed_witness_latched_no_silent_restart')
        if self.spec is None:
            return self._receipt(now,error or 'fresh_reappearance_unknown_requires_new_motion_witness')
        if gap is not None and abs(gap-.05)>1e-6:return self._fail(now,'missing_non50ms_frame_no_interpolation')
        if error:return self._fail(now,error)
        # Continue the ORIGINAL probe bounds until explicit commit, even if
        # the isolated observer has already produced its first confirmation.
        if not self.inner._inside_probe(robot):return self._fail(now,'actual_witness_15mm_or004rad_bound_exceeded')
        if now>self.spec['deadline']+1e-8:return self._fail(now,'fixed_witness_deadline_expired_no_rollforward')
        self.trace.append(dict(time=now,tcp=robot.tcp_position.copy(),object=target.pose.position.copy(),
            relative=robot.tcp_pose.inverse_apply(target.pose.position)))
        metrics=self._metrics()
        if metrics['maximum_fixed_r0_deviation_m']>.006+1e-10:
            return self._fail(now,'all_current_witness_trace_fixed_r0_over6mm')
        current=self.inner.observe(target,robot,close_requested=close_requested)
        if current['probe_failed']:return self._fail(now,current['reason'])
        passed=(current['mode']=='ATTACHED_PUBLIC' and type(current['evidence']) is PublicMotionEvidence
            and current['evidence'].event=='attachment_confirmed'
            and abs(current['evidence'].timestamp-now)<1e-8
            and metrics['real_probe_frames']>=6 and metrics['duration_seconds']>=.25-1e-8
            and metrics['TCP_signed_vertical_m']>=.010-1e-10 and metrics['object_signed_vertical_m']>=.008-1e-10)
        return self._receipt(now,'fresh_public_comotion_pending_explicit_epoch_commit' if passed else current['reason'],passed)

    def begin_witness(self,result,robot,h,*,mode='natural'):
        if mode not in ('natural','active_probe_proposal'):raise ValueError('Only natural or a bounded unexecuted test proposal')
        target,error=self._current(result,robot,self.last_close)
        if (result is not self.last_result or robot is not self.last_robot or self.last_target is not target
            or abs(robot.timestamp-self.last_timestamp)>1e-8 or self._frame_identity(result,robot)!=self._frame_digest):
            raise ValueError('Begin only after the exact same observed public frame')
        if error or self.witness_failed or self.verification_epoch:return None
        if self.spec is not None:
            if mode!=self.witness_mode:raise ValueError('Witness mode cannot change after fixed admission')
            return deepcopy(self.spec)
        observer=ActiveVisualAttachment(ActiveVisualConfig())
        observer.observe(target,robot,close_requested=True)
        spec=observer.begin_probe(target,robot,h)
        if spec is None:return None
        self.inner=observer;self.spec=spec;self._spec_digest=_digest(spec);self.witness_mode=mode
        self.trace=deepcopy(observer.rows)
        self._receipt(robot.timestamp,'new_isolated_witness_admitted_no_attachment_or_epoch_reset')
        return deepcopy(self.spec)

    def active_test_proposal(self):
        self._unchanged()
        if self.spec is None or self.witness_failed or self.verification_epoch:return None
        return dict(proposal_kind='fixed12mm_public_reverification_test_only_not_executed',
            probe_spec=deepcopy(self.spec),tcp_target_position=np.asarray(self.spec['target_position']).copy(),
            tcp_target_quaternion=np.asarray(self.spec['target_quaternion']).copy(),opening=0.,
            requires_robot_model_MPC_optimization=True,requires_original_collision_and_50ms_checks=True,
            exact_selected_q_opening_dispatch_required=True,actuator_done_is_contact=False,
            Native_inputs_used=False,executed=False)

    def commit_verified_epoch(self,result,robot):
        target,error=self._current(result,robot,self.last_close)
        if (error or result is not self.last_result or robot is not self.last_robot or target is not self.last_target
            or self.witness_failed or self.verification_epoch or self.latest is None
            or self.latest.get('current_fresh_witness_passed') is not True
            or abs(self.latest['timestamp']-robot.timestamp)>1e-8
            or self._frame_identity(result,robot)!=self._frame_digest):
            raise ValueError('Only current fresh full co-motion witness permits explicit new epoch')
        self.verification_epoch=1
        relative=robot.tcp_pose.inverse_apply(target.pose.position)
        event=PublicMotionEvidence('attachment_confirmed',robot.timestamp,
            ('rgbd_relative_pose_tracking','public_robot_pose','public_gripper_opening'),
            Pose(relative,robot.tcp_pose.rotation.T@target.pose.rotation),orientation_observable=False)
        self.committed=dict(version=VERSION,verification_epoch=1,physical_attempt_id=self.attempt_id,
            timestamp=robot.timestamp,evidence=event,fresh_witness=deepcopy(self.inner.latest),
            current_whole_trace_confirmation=dict(timestamp=robot.timestamp,**self._metrics()),
            original_failed_proof=deepcopy(self.original_proof),original_fixed_r0=self.r0.copy(),
            original_object_anchor=self.object_anchor.copy(),original_preclose_baseline=deepcopy(self._baseline),
            old_failed_owner_reset=False,maintenance_replaced=False,production_integrated=False)
        return deepcopy(self.committed)
