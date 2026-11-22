"""Operational visible co-motion under an active, bounded vertical test.

This independent v4 protocol does not certify frictional/two-pad contact. Pose
observations cannot rule out synchronized external vertical manipulation or a
common visual bias; the development setting assumes a table-supported target
with no independent external Z drive. Legacy attachment gates are diagnostics.
"""
from copy import deepcopy
from dataclasses import dataclass,asdict
import hashlib,json
import numpy as np
from ..contracts import ObjectState,Pose
from .public_motion_bridge import PublicRobotSensors,PublicMotionEvidence
from .public_grasp_probe import VerificationProbeConfig
from .public_state import ACTUATOR_KEYS
from ..local_mpc_20261102 import mpc


@dataclass(frozen=True)
class ActiveVisualConfig:
    sensor_dt:float=.05
    minimum_confidence:float=.5
    maximum_TCP_object_distance_m:float=.10
    probe_seconds:float=.8
    minimum_TCP_vertical_m:float=.010
    minimum_object_vertical_m:float=.008
    maximum_relative_deviation_m:float=.006
    confirmation_frames:int=6
    confirmation_seconds:float=.25
    release_opening_min:float=.60
    release_separation_m:float=.070
    release_relative_change_m:float=.020
    release_frames:int=4
    release_seconds:float=.15
    enforce_probe_anchor_orientation_after_confirmation:bool=True
    def __post_init__(self):
        for key,value in asdict(self).items():
            if key=='enforce_probe_anchor_orientation_after_confirmation':
                if type(value) is not bool:raise ValueError('explicit post-confirmation orientation-scope bool required')
            elif key in ('confirmation_frames','release_frames'):
                if type(value) is not int or value<3:raise ValueError('real consecutive-frame count required')
            elif not np.isfinite(value) or value<=0:raise ValueError('positive finite active visual threshold')
        # The 1.2s development ablation changes only the fixed execution
        # budget. It never moves the admitted deadline or relaxes evidence.
        if (self.sensor_dt!=.05 or self.probe_seconds not in (.8,1.2) or self.minimum_TCP_vertical_m<.010
                or self.minimum_object_vertical_m<.008 or self.maximum_relative_deviation_m>.006
                or self.confirmation_frames<6 or self.confirmation_seconds<.25
                or self.confirmation_seconds>(self.confirmation_frames-1)*.05+1e-8
                or not .5<=self.minimum_confidence<=1. or self.maximum_TCP_object_distance_m>.10
                or not .60<=self.release_opening_min<=1. or self.release_separation_m<.070
                or self.release_relative_change_m<.020 or self.release_frames<4 or self.release_seconds<.15
                or self.release_seconds>(self.release_frames-1)*.05+1e-8):
            raise ValueError('v4 declares 50ms, one fixed .8s or explicit1.2s development probe, >=10/8mm signed Z, <=6mm fixed-r0, >=6 frames/.25s')


class ActiveVisualAttachment:
    """observe once/real50ms; begin_probe only after fresh100ms executor state.

    API order: observe(target,robot,close_requested=actual_intent), then optional
    begin_probe(same target,same robot,fresh_h). Every subsequent observe checks
    the whole trace against frozen r0. Failed attempts never silently restart.
    """
    def __init__(self,config=None):
        self.config=config or ActiveVisualConfig()
        if type(self.config) is not ActiveVisualConfig:raise TypeError('explicit ActiveVisualConfig required')
        self.reset()
    def reset(self):
        self.mode='FREE';self.attempt_active=False;self.attempt_id=0;self.last_timestamp=None
        self.last_robot=None;self.last_target=None;self.last_close_requested=False
        self.attempt_started=None;self.spec=None;self.rows=[];self.failed=False;self.reason='no_close_attempt'
        self.confirmation=None;self.latest=None;self._release_rows=[];self._release_reference=None
        self.failure_time=None
    def _valid_target(self,target,now):
        return (isinstance(target,ObjectState) and target.object_id=='target' and target.visible
            and target.confidence>=self.config.minimum_confidence and abs(target.timestamp-now)<1e-8)
    def _relative(self,target,robot):return robot.tcp_pose.inverse_apply(target.pose.position)
    def _evidence(self,event,now,relative=None):
        if event=='attachment_confirmed':return PublicMotionEvidence(event,now,
            ('rgbd_relative_pose_tracking','public_robot_pose','public_gripper_opening'),relative,orientation_observable=False)
        if event=='release_confirmed':return PublicMotionEvidence(event,now,('rgbd_object_release','public_gripper_opening'))
        return PublicMotionEvidence('manipulation_unresolved',now,('public_robot_pose','public_gripper_opening'))
    def _result(self,now,reason,evidence=None):
        self.reason=reason
        self.latest=dict(timestamp=now,mode=self.mode,evidence=evidence,reason=reason,attempt_active=self.attempt_active,
            attempt_id=self.attempt_id,attempt_started=self.attempt_started,probe=deepcopy(self.spec),
            probe_failed=self.failed,probe_failure_time=self.failure_time,diagnostic=self._metrics(),
            protocol='active_vertical_visual_comotion_v4',config=asdict(self.config),legacy_estimator_controls_mode=False,
            fixed_probe_budget_protocol=('original_fixed_08s' if self.config.probe_seconds==.8 else 'development_fixed_12s_timing_ablation'),
            public_observation_scope_only=True,Native_held_or_success_used=False,Native_object_pose_used=False,
            actuator_done_is_admission_only=True,attachment_is_visible_operational_comotion=True,
            object_orientation_observable=False,full_SE3_attachment_qualified=False,
            probe_anchor_orientation_guard_scope=(
                'probe_and_post_confirmation_maintenance'
                if self.config.enforce_probe_anchor_orientation_after_confirmation else 'probe_only'),
            post_confirmation_evidence_semantics=(
                'current_visible_center_relative_translation_with_probe_anchor_TCP_orientation_guard'
                if self.config.enforce_probe_anchor_orientation_after_confirmation else
                'current_visible_center_relative_translation_only_object_orientation_unobserved'),
            force_or_frictional_grasp_proof=False,two_pad_contact_proven=False,
            external_synchronous_Z_motion_ruled_out=False,common_visual_bias_ruled_out=False,
            development_assumption='table_supported_target_no_independent_external_vertical_drive',
            bound_scope='command_and_discrete_predicted_FK_constraints_real50ms_detect_then_next100ms_control',
            continuous_physical_bound_guaranteed=False,second_committed_tick_cancellation_guaranteed=False)
        return self.latest
    def _metrics(self):
        if not self.rows:return dict(real_probe_frames=0,entire_probe_trace_used=True)
        first,last=self.rows[0],self.rows[-1]
        drift=max(float(np.linalg.norm(r['relative']-first['relative'])) for r in self.rows)
        return dict(real_probe_frames=len(self.rows),duration_seconds=last['time']-first['time'],
            TCP_signed_vertical_m=float(last['tcp'][2]-first['tcp'][2]),
            object_signed_vertical_m=float(last['object'][2]-first['object'][2]),
            maximum_fixed_r0_deviation_m=drift,entire_probe_trace_used=True,
            reference_is_probe_start_not_rolling_or_mean=True,confirmation=deepcopy(self.confirmation))
    def _fail(self,now,reason):
        self.mode='UNRESOLVED';self.failed=True;self.failure_time=now
        return self._result(now,reason,self._evidence('manipulation_unresolved',now))
    def _inside_probe(self,robot):
        cfg=VerificationProbeConfig(**self.spec['config'])
        angle=np.linalg.norm(mpc.quat_log(mpc.quat_multiply(robot.tcp_quaternion,mpc.quat_inverse(self.spec['anchor_quaternion']))))
        return (np.linalg.norm(robot.tcp_position-self.spec['anchor_position'])<=cfg.maximum_displacement_m+1e-8
            and angle<=cfg.maximum_rotation_rad+1e-8)
    def begin_probe(self,target,robot,h):
        """Completion permits a test; it is never a contact/held observation."""
        if not isinstance(robot,PublicRobotSensors):raise TypeError('real robot sensors required')
        if not self.attempt_active or self.failed or self.mode!='UNRESOLVED':return None
        if not self.last_close_requested or self.last_robot is None or robot.timestamp!=self.last_timestamp:return None
        for key in ('tcp_position','tcp_quaternion','q','dq','finger_open_amounts'):
            if not np.array_equal(getattr(robot,key),getattr(self.last_robot,key)):raise ValueError('exact observed actual robot frame required')
        if target is not self.last_target or not self._valid_target(target,robot.timestamp):return None
        if np.linalg.norm(target.pose.position-robot.tcp_position)>self.config.maximum_TCP_object_distance_m:return None
        if not isinstance(h,dict) or set(h)!=ACTUATOR_KEYS or h['reference_is_measured_state'] is not False:
            raise ValueError('actual whitelisted executor state required for admission')
        if (h['reference_clock'] is None or not np.isfinite(float(h['reference_clock']))
                or abs(float(h['reference_clock'])-robot.timestamp)>1e-8):
            raise ValueError('fresh same-time executor completion state required')
        if type(h['reference_valid']) is not bool or not h['reference_valid']:
            raise ValueError('explicit valid initialized actual arm reference required')
        for key in ('reference_positions','reference_velocities','reference_accelerations','target_joints'):
            value=np.asarray(h[key],float)
            if value.shape!=(7,) or not np.isfinite(value).all():raise ValueError('initialized actual executor reference required')
        if h['gripper_target']!=0. or not isinstance(h['gripper_done'],(bool,np.bool_)) or not bool(h['gripper_done']):return None
        if self.spec is not None:return deepcopy(self.spec) # validated identity/deadline never roll
        cfg=VerificationProbeConfig();anchor=robot.tcp_position.copy();quat=robot.tcp_quaternion.copy()
        identity=hashlib.sha256(json.dumps(dict(attempt=self.attempt_id,time=robot.timestamp,anchor=anchor.tolist(),quaternion=quat.tolist()),sort_keys=True).encode()).hexdigest()
        self.spec=dict(identity=identity,attempt_id=self.attempt_id,trigger_time=robot.timestamp,
            deadline=robot.timestamp+self.config.probe_seconds,anchor_position=anchor,anchor_quaternion=quat,
            target_position=anchor+[0.,0.,cfg.lift_m],target_quaternion=quat.copy(),config=asdict(cfg),
            object_anchor_position=target.pose.position.copy(),relative_r0=self._relative(target,robot),
            admission='same_attempt_current_actual_close_target0_doneTrue_current_publicbody_and_robot',
            actuator_done_is_contact=False,held=False,attachment_asserted=False,
            anchor_source='actual_TCP_at_active_visual_probe_admission',object_follows_probe_assumed=False)
        self.rows=[dict(time=robot.timestamp,tcp=anchor,object=target.pose.position.copy(),relative=self.spec['relative_r0'].copy())]
        return deepcopy(self.spec)
    def observe(self,target,robot,*,close_requested):
        if not isinstance(robot,PublicRobotSensors):raise TypeError('actual PublicRobotSensors required')
        if type(close_requested) is not bool:raise TypeError('known current closed intent bool required')
        now=robot.timestamp;gap=None if self.last_timestamp is None else now-self.last_timestamp
        if gap is not None and gap<=0:raise ValueError('unique increasing actual50ms observations required')
        self.last_timestamp=now;self.last_robot=robot;self.last_target=target;self.last_close_requested=close_requested
        first_close=close_requested and not self.attempt_active
        if first_close:
            self.attempt_id+=1;self.attempt_active=True;self.attempt_started=now;self.mode='UNRESOLVED'
            self.spec=None;self.rows=[];self.failed=False;self.confirmation=None;self.failure_time=None;self._release_rows=[]
            self._release_reference=self._relative(target,robot) if self._valid_target(target,now) else None
        if gap is not None and abs(gap-.05)>1e-6:
            if self.attempt_active:return self._fail(now,'missing_non50ms_real_feedback_no_interpolation')
            self.mode='STALE';return self._result(now,'public_feedback_gap_before_attempt')
        if not self._valid_target(target,now):
            if self.attempt_active:return self._fail(now,'current_visual_missing_lowconfidence_or_identity_invalid')
            self.mode='STALE';return self._result(now,'current_visual_unavailable')
        if not self.attempt_active:
            self.mode='FREE';return self._result(now,'no_close_attempt_no_attachment_claim')
        relative=self._relative(target,robot)
        if self._release_reference is None:self._release_reference=relative.copy()
        if not close_requested:
            if not self.failed:
                self.failed=True;self.failure_time=now
            self.mode='UNRESOLVED';self._release_rows.append((now,robot.finger_open_amounts.copy(),relative.copy()))
            self._release_rows=self._release_rows[-self.config.release_frames:]
            release=self._release_rows
            separated=(len(release)==self.config.release_frames and release[-1][0]-release[0][0]>=self.config.release_seconds-1e-8
                and all(np.all(r[1]>=self.config.release_opening_min) and np.linalg.norm(r[2])>=self.config.release_separation_m
                    and np.linalg.norm(r[2]-self._release_reference)>=self.config.release_relative_change_m for r in release))
            if separated:
                self.attempt_active=False;self.mode='FREE'
                return self._result(now,'public_open_and_separation_reestablish_free',self._evidence('release_confirmed',now))
            return self._result(now,'open_requires_real_visual_release_not_automatic_FREE',self._evidence('manipulation_unresolved',now))
        self._release_rows=[]
        if self.failed:return self._result(now,'failed_attempt_requires_public_release_before_retry',self._evidence('manipulation_unresolved',now))
        if self.spec is None:return self._result(now,'closed_pending_current_actuator_done_for_probe',self._evidence('manipulation_unresolved',now))
        if self.mode=='ATTACHED_PUBLIC':
            drift=np.linalg.norm(relative-self.spec['relative_r0'])
            angle=np.linalg.norm(mpc.quat_log(mpc.quat_multiply(robot.tcp_quaternion,mpc.quat_inverse(self.spec['anchor_quaternion']))))
            # The bounded admission probe always retains its original .04rad
            # guard. A declared post-confirmation scope may instead maintain
            # only the current visible center's TCP-relative translation;
            # neither setting observes object orientation or proves contact.
            orientation_lost=(self.config.enforce_probe_anchor_orientation_after_confirmation and angle>.04+1e-8)
            if drift>self.config.maximum_relative_deviation_m+1e-10 or orientation_lost:
                return self._fail(now,'current_attached_relative_or_orientation_consistency_lost')
            pose=Pose(relative,robot.tcp_pose.rotation.T@target.pose.rotation)
            return self._result(now,'current_visible_relative_consistency_maintained',self._evidence('attachment_confirmed',now,pose))
        # Safety/identity/timing precede any positive co-motion evidence.
        if not self._inside_probe(robot):return self._fail(now,'actual_probe_15mm_or_orientation_bound_exceeded')
        if now>self.spec['deadline']+1e-8:return self._fail(now,'fixed_probe_deadline_expired_no_rollforward')
        self.rows.append(dict(time=now,tcp=robot.tcp_position.copy(),object=target.pose.position.copy(),relative=relative.copy()))
        metrics=self._metrics()
        if metrics['maximum_fixed_r0_deviation_m']>self.config.maximum_relative_deviation_m+1e-10:
            return self._fail(now,'all_probe_trace_fixed_r0_consistency_failed')
        confirmed=(len(self.rows)>=self.config.confirmation_frames
            and metrics['duration_seconds']>=self.config.confirmation_seconds-1e-8
            and metrics['TCP_signed_vertical_m']>=self.config.minimum_TCP_vertical_m-1e-10
            and metrics['object_signed_vertical_m']>=self.config.minimum_object_vertical_m-1e-10)
        if confirmed:
            self.confirmation=dict(timestamp=now,**{k:v for k,v in metrics.items() if k!='confirmation'})
            self.mode='ATTACHED_PUBLIC';pose=Pose(relative,robot.tcp_pose.rotation.T@target.pose.rotation)
            return self._result(now,'active_signed_vertical_visual_comotion_confirmed',self._evidence('attachment_confirmed',now,pose))
        return self._result(now,'active_probe_current_comotion_not_yet_sufficient',self._evidence('manipulation_unresolved',now))
