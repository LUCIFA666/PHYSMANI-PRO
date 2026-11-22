"""Independent position-only maintenance candidate; not wired into production.

The10mm limit matches the magnitude of the existing PublicLiftHoldMetric cap.
It is a phase-consistency hypothesis, NOT a calibrated error/noise bound. Its
reference remains the original probe r0; the hold metric instead references a
qualifying hold streak. These are not identical predicates. A real7mm relative
change can pass this candidate, so force/contact/stability are not certified.

Admission requires the unchanged initial real6mm/.04rad/15mm probe proof.
Later observations are current typed visible RGBD centres only. Forecasts and
controlled poses are never observations. Failure latches until explicit reset;
neither an anchor nor a proof can roll forward to hide an excursion.
"""
from copy import deepcopy
from dataclasses import asdict,dataclass
import numpy as np

from ..contracts import ObjectState,Pose,array
from .public_motion_bridge import PublicRobotSensors,PublicMotionEvidence
from .active_visual_attachment import ActiveVisualConfig
from .public_grasp_probe import VerificationProbeConfig


@dataclass(frozen=True)
class PostConfirmedMaintenanceConfig:
    sensor_dt:float=.05
    minimum_confidence:float=.5
    maximum_fixed_r0_deviation_m:float=.010
    minimum_visual_height_above_probe_anchor_m:float=.008

    def __post_init__(self):
        if (any(isinstance(v,(bool,np.bool_)) or not np.isfinite(v) for v in asdict(self).values())
                or self.sensor_dt!=.05 or not .5<=self.minimum_confidence<=1.
                or self.maximum_fixed_r0_deviation_m!=.010
                or self.minimum_visual_height_above_probe_anchor_m!=.008):
            raise ValueError('explicit candidate requires real50ms, confidence>=.5, fixed10mm and visual rise8mm')


def _fresh(value,now):
    return (not isinstance(value,(bool,np.bool_)) and isinstance(value,(int,float,np.integer,np.floating))
            and np.isfinite(value) and abs(float(value)-now)<=1e-8)


def _finite_number(value):
    return (not isinstance(value,(bool,np.bool_)) and isinstance(value,(int,float,np.integer,np.floating))
            and np.isfinite(value))


class PostConfirmedPositionMaintenance:
    """start once on first genuine proof; observe once per subsequent real50ms.

    This class deliberately ignores the old estimator's later6mm maintenance
    decisions when scoring a counterfactual. It never ignores its own failure.
    It accepts a public result for source/time/attempt/current-vision provenance,
    not its controlled or forecast pose. A new episode needs explicit reset.
    """
    def __init__(self,config=None):
        self.config=config or PostConfirmedMaintenanceConfig()
        if type(self.config) is not PostConfirmedMaintenanceConfig:raise TypeError('explicit maintenance config required')
        self.reset_episode()

    def reset_episode(self):
        self.started=False;self.failed=False;self.failure_time=None;self.failure_reason=None
        self.attempt_id=None;self.last_timestamp=None;self.r0=None;self.object_anchor=None
        self.proof=None;self.latest=None

    def _visual(self,result,robot):
        if not isinstance(robot,PublicRobotSensors):raise TypeError('real typed public robot sensors required')
        now=robot.timestamp;data=result if isinstance(result,dict) else {}
        if (not _fresh(data.get('timestamp'),now) or any(data.get(k) is not False for k in
                ('Native_inputs_used','future_observations_used','observations_fabricated'))):
            return None,'not_current_public_observation_scope'
        target=data.get('object_observation')
        if (not isinstance(target,ObjectState) or target.object_id!='target' or target.visible is not True
                or target.confidence<self.config.minimum_confidence or not _fresh(target.timestamp,now)):
            return None,'current_visible_typed_RGBD_body_missing_lowconfidence_or_stale'
        return target,None

    def _receipt(self,now,reason,passed,*,target=None,robot=None,drift=None,rise=None):
        evidence=None
        if passed:
            relative=robot.tcp_pose.inverse_apply(target.pose.position)
            evidence=PublicMotionEvidence('attachment_confirmed',now,
                ('rgbd_relative_pose_tracking','public_robot_pose','public_gripper_opening'),
                Pose(relative,robot.tcp_pose.rotation.T@target.pose.rotation),orientation_observable=False)
        self.latest=dict(timestamp=float(now),protocol='postconfirmed_public_position_maintenance_candidate_v1',
            current_position_maintenance_passed=bool(passed),evidence=evidence,
            reason=reason,attempt_id=self.attempt_id,started=self.started,failed=self.failed,
            failure_time=self.failure_time,first_failure_reason=self.failure_reason,
            current_fixed_r0_deviation_m=drift,current_visual_height_above_probe_anchor_m=rise,
            fixed_probe_r0=None if self.r0 is None else self.r0.tolist(),
            fixed_visual_probe_object_anchor=None if self.object_anchor is None else self.object_anchor.tolist(),
            criteria=asdict(self.config),initial_probe_proof=deepcopy(self.proof),
            initial_probe_6mm_004rad_15mm_changed=False,anchor_rolled_forward=False,
            current_body_is_RGBD_position=True,controlled_or_forecast_used_as_observation=False,
            Native_inputs_used=False,object_orientation_observable=False,full_SE3_attachment_qualified=False,
            force_or_frictional_grasp_proof=False,public_goal_or_Native_success_claimed=False,
            threshold_is_phase_consistency_candidate_not_calibration=True,production_integrated=False)
        return deepcopy(self.latest)

    def _fail(self,now,reason,**kwargs):
        if not self.failed:self.failure_time=float(now);self.failure_reason=reason
        self.failed=True
        return self._receipt(now,reason,False,**kwargs)

    def start(self,result,robot,*,close_requested):
        """Only original sameframe first confirmation can bind the fixed proof."""
        if type(close_requested) is not bool:raise TypeError('explicit current closed intent required')
        if self.started:raise ValueError('proof/attempt cannot be re-certified without explicit episode reset')
        target,error=self._visual(result,robot);now=robot.timestamp
        a=result.get('attachment_estimate',{}) if isinstance(result,dict) else {}
        if not isinstance(a,dict):raise ValueError('original active probe record must be a dictionary')
        evidence=a.get('evidence');probe=a.get('probe');config=a.get('config',{});diagnostic=a.get('diagnostic',{})
        diagnostic=diagnostic if isinstance(diagnostic,dict) else {}
        confirmation=diagnostic.get('confirmation') or {}
        confirmation=confirmation if isinstance(confirmation,dict) else {}
        try:
            if not isinstance(config,dict) or not isinstance(probe,dict):raise ValueError('probe config missing')
            ActiveVisualConfig(**config)
            VerificationProbeConfig(**probe.get('config',{}))
            valid_configs=True
        except (TypeError,ValueError):valid_configs=False
        frames=confirmation.get('real_probe_frames');duration=confirmation.get('duration_seconds')
        finite_proof=(type(frames) is int and frames>=6
            and all(_finite_number(confirmation.get(k)) for k in ('duration_seconds','maximum_fixed_r0_deviation_m',
                'TCP_signed_vertical_m','object_signed_vertical_m'))
            and confirmation.get('entire_probe_trace_used') is True
            and confirmation.get('reference_is_probe_start_not_rolling_or_mean') is True)
        if finite_proof:
            finite_proof=(abs(duration-(frames-1)*.05)<=1e-6
                and isinstance(probe,dict) and _finite_number(probe.get('trigger_time'))
                and _finite_number(probe.get('deadline'))
                and _fresh(probe['trigger_time']+duration,now)
                and valid_configs and _fresh(probe['trigger_time']+config['probe_seconds'],probe['deadline'])
                and now<=probe['deadline']+1e-8)
        valid=(target is not None and close_requested and result.get('mode')=='ATTACHED_PUBLIC'
            and result.get('available') is True and result.get('controlled_mode_available') is True
            and a.get('mode')=='ATTACHED_PUBLIC' and a.get('protocol')=='active_vertical_visual_comotion_v4'
            and _fresh(a.get('timestamp'),now) and a.get('attempt_active') is True
            and type(a.get('attempt_id')) is int and a['attempt_id']>0 and a.get('probe_failed') is False
            and isinstance(evidence,PublicMotionEvidence) and evidence.event=='attachment_confirmed'
            and _fresh(evidence.timestamp,now) and _fresh(confirmation.get('timestamp'),now)
            and valid_configs and finite_proof and isinstance(probe,dict) and probe.get('attempt_id')==a['attempt_id']
            and config.get('sensor_dt')==.05 and config.get('maximum_relative_deviation_m')==.006
            and config.get('minimum_TCP_vertical_m')==.01 and config.get('minimum_object_vertical_m')==.008
            and config.get('confirmation_frames',0)>=6 and config.get('confirmation_seconds',0.)>=.25
            and confirmation.get('real_probe_frames',0)>=6 and confirmation.get('duration_seconds',0.)>=.25-1e-8
            and 0.<=confirmation.get('maximum_fixed_r0_deviation_m',1.)<=.006+1e-10
            and confirmation.get('TCP_signed_vertical_m',0.)>=.01-1e-10
            and confirmation.get('object_signed_vertical_m',0.)>=.008-1e-10
            and probe.get('config',{}).get('maximum_displacement_m')==.015
            and probe.get('config',{}).get('maximum_rotation_rad')==.04)
        if not valid:raise ValueError('fresh unchanged original public probe proof required: '+str(error or 'proof_invalid'))
        self.r0=array(probe['relative_r0'],(3,));self.object_anchor=array(probe['object_anchor_position'],(3,))
        self.attempt_id=a['attempt_id'];self.started=True
        self.proof=dict(attempt_id=self.attempt_id,first_confirmation_timestamp=float(now),
            probe_trigger_time=probe['trigger_time'],probe_deadline=probe['deadline'],
            original_probe_config=deepcopy(probe['config']),original_confirmation=deepcopy(confirmation),
            public_evidence_sources=list(evidence.sources))
        self.last_timestamp=None
        return self.observe(result,robot,close_requested=close_requested)

    def observe(self,result,robot,*,close_requested):
        if type(close_requested) is not bool:raise TypeError('explicit current closed intent required')
        target,error=self._visual(result,robot);now=robot.timestamp
        if not self.started:return self._receipt(now,'no_original_probe_confirmation',False)
        if self.last_timestamp is not None and now<=self.last_timestamp:
            return self._fail(now,'duplicate_or_nonincreasing_real_frame')
        gap=None if self.last_timestamp is None else now-self.last_timestamp;self.last_timestamp=now
        if self.failed:return self._receipt(now,'failed_candidate_requires_explicit_reset_no_recertification',False)
        if gap is not None and abs(gap-.05)>1e-6:return self._fail(now,'missing_non50ms_real_frame_no_interpolation')
        if not close_requested:return self._fail(now,'known_open_intent_no_current_attachment_claim')
        a=result.get('attachment_estimate',{}) if isinstance(result,dict) else {}
        if (not isinstance(a,dict) or a.get('protocol')!='active_vertical_visual_comotion_v4'
                or type(a.get('attempt_id')) is not int or a.get('attempt_id')!=self.attempt_id
                or a.get('attempt_active') is not True or not _fresh(a.get('timestamp'),now)):
            return self._fail(now,'same_current_original_active_attempt_required')
        if error:return self._fail(now,error)
        relative=robot.tcp_pose.inverse_apply(target.pose.position)
        drift=float(np.linalg.norm(relative-self.r0));rise=float(target.pose.position[2]-self.object_anchor[2])
        if drift>self.config.maximum_fixed_r0_deviation_m+1e-10:
            return self._fail(now,'fixed_original_probe_r0_deviation_over10mm',drift=drift,rise=rise)
        if rise<self.config.minimum_visual_height_above_probe_anchor_m-1e-10:
            return self._fail(now,'current_visual_body_below_original_probe_anchor_plus8mm',drift=drift,rise=rise)
        return self._receipt(now,'current_visible_position_maintenance_only',True,target=target,robot=robot,drift=drift,rise=rise)
