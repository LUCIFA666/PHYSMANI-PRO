"""Passive public attachment evidence; no Native truth, actions or fake lift.

50ms integration order (all at the SAME real sensor timestamp):
  1. Read synchronized public RGBD/calibration and sanitized robot sensors.
  2. Run the public RGBD tracker ONCE; stage its ObjectState in the frame cache.
  3. estimator.observe(target_or_None, robot, close_requested=actual_intent).
  4. bridge.observe(rgbd, robot, close_requested=..., evidence=result['evidence'])
     with tracker=that cache. Never call bridge twice at one timestamp.

A low-level stationary close or proximity alone cannot confirm attachment.
These predeclared development thresholds express a conservative consistency
hypothesis, not force/contact proof or Native task success. In particular an
empty-endpoint fully closed gripper is unresolved even if a simulator's hidden
parent attachment might have occurred. No action, lift, or held flag is created.
"""
from __future__ import annotations
from collections import deque
from dataclasses import asdict,dataclass,replace

import numpy as np

from ..contracts import ObjectState,Pose
from ..benchmark_perception import RedBoxTracker
from .public_motion_bridge import PublicRobotSensors,PublicMotionEvidence,MotionMode,PUBLIC_OBSERVATION_KEYS


@dataclass(frozen=True)
class AttachmentConfig:
    sensor_dt_seconds:float=.05
    maximum_gap_seconds:float=.075
    minimum_confidence:float=.5
    confirmation_frames:int=6
    confirmation_duration_seconds:float=.25
    relative_position_deviation_m:float=.006
    relative_rotation_deviation_rad:float=.08
    maximum_TCP_rotation_if_object_rotation_unobservable_rad:float=.04
    minimum_TCP_motion_m:float=.010
    minimum_visual_object_motion_m:float=.008
    maximum_TCP_object_distance_m:float=.10
    opening_nonempty_min:float=.06
    opening_grasp_max:float=.90
    maximum_finger_asymmetry:float=.12
    maximum_stalled_opening_range:float=.015
    maximum_stalled_opening_speed_per_second:float=.03
    minimum_observed_closure:float=.04
    maximum_attached_relative_drift_m:float=.010
    release_opening_min:float=.60
    release_frames:int=4
    release_duration_seconds:float=.15
    release_separation_m:float=.070
    minimum_release_relative_change_m:float=.020

    def __post_init__(self):
        for name,value in asdict(self).items():
            if name in ('confirmation_frames','release_frames'):
                if type(value) is not int or value<3:raise ValueError('at least three real frames required: '+name)
            elif not np.isfinite(value) or value<=0:raise ValueError('positive finite threshold required: '+name)
        if (self.sensor_dt_seconds!=.05 or not self.sensor_dt_seconds<=self.maximum_gap_seconds<=.075
                or not 0<self.opening_nonempty_min<self.opening_grasp_max<1
                or not self.opening_nonempty_min<self.release_opening_min<1
                or not 0<self.minimum_confidence<=1):
            raise ValueError('50ms public sensor and normalized-opening thresholds required')
        if (self.confirmation_duration_seconds>(self.confirmation_frames-1)*.05+1e-8
                or self.release_duration_seconds>(self.release_frames-1)*.05+1e-8):
            raise ValueError('configured duration must fit the declared consecutive-frame window')


def rotation_angle(first,second):
    relative=first.T@second
    return float(np.arccos(np.clip((np.trace(relative)-1)/2,-1,1)))


class PublicAttachmentEstimator:
    """Only public tracked geometry, TCP/fingers/time and control intent enter."""
    def __init__(self,config=None):
        self.config=AttachmentConfig() if config is None else config
        if not isinstance(self.config,AttachmentConfig):raise TypeError('AttachmentConfig required')
        self.reset()

    def reset(self):
        self.mode=MotionMode.FREE;self.last_timestamp=None;self.attempt_active=False
        self._initial_opening=None;self._attachment_reference=None;self._release_reference=None;self._public_attachment_seen=False
        self._previous_public_opening=None;self._previous_close_requested=None
        self._closure_baseline_timestamp=None;self._closure_baseline_reason='no_active_close_attempt'
        self._rows=deque(maxlen=max(self.config.confirmation_frames,self.config.release_frames))

    def _result(self,now,reason,*,evidence=None,diagnostic=None):
        return dict(timestamp=float(now),mode=self.mode.value,evidence=evidence,reason=reason,
            consecutive_valid_frames=len(self._rows),attempt_active=self.attempt_active,
            diagnostic={} if diagnostic is None else diagnostic,config=asdict(self.config),
            closure_baseline=dict(available=self._initial_opening is not None,
                timestamp=self._closure_baseline_timestamp,
                finger_open_amounts=None if self._initial_opening is None else self._initial_opening.tolist(),
                source='previous_real_50ms_public_fingers_with_open_intent' if self._initial_opening is not None else None,
                reason=self._closure_baseline_reason),
            threshold_provenance='predeclared_development_sensor_and_geometry_consistency_not_fit_to_Native_labels',
            native_held_or_success_used=False,Native_object_pose_used=False,actions_emitted=False,
            observed_lift_or_contact_fabricated=False,force_or_frictional_grasp_proof=False,
            attachment_is_public_consistency_hypothesis=True)

    def _unresolved_evidence(self,now):
        return PublicMotionEvidence('manipulation_unresolved',now,('public_robot_pose','public_gripper_opening'))

    def _invalid(self,now,reason):
        self._rows.clear();self._attachment_reference=None;self.mode=MotionMode.STALE
        return self._result(now,reason,evidence=self._unresolved_evidence(now) if self.attempt_active else None)

    def _window_metrics(self,rows):
        c=self.config;n=len(rows)
        times=np.array([r['timestamp'] for r in rows]);relative=np.stack([r['relative'].position for r in rows])
        tcp=np.stack([r['tcp'].position for r in rows]);objects=np.stack([r['target'].pose.position for r in rows])
        openings=np.stack([r['opening'] for r in rows]);duration=float(times[-1]-times[0])
        pos_deviation=float(np.linalg.norm(relative-relative.mean(0),axis=-1).max())
        tcp_rotation=max(rotation_angle(rows[0]['tcp'].rotation,r['tcp'].rotation) for r in rows)
        observable=all(r['orientation_observable'] for r in rows)
        relative_rotation=max(rotation_angle(rows[0]['relative'].rotation,r['relative'].rotation) for r in rows) if observable else None
        ranges=np.ptp(openings,axis=0)
        speed=np.max(np.abs(np.diff(openings,axis=0))/np.diff(times)[:,None],axis=0) if n>1 else np.zeros(2)
        closure=None if self._initial_opening is None else self._initial_opening-openings.mean(0)
        tcp_motion=float(np.linalg.norm(tcp-tcp[0],axis=-1).max())
        object_motion=float(np.linalg.norm(objects-objects[0],axis=-1).max())
        distances=np.linalg.norm(relative,axis=-1)
        gates=dict(real_consecutive_feedback_window=n>=c.confirmation_frames and duration>=c.confirmation_duration_seconds-1e-8,
            stable_relative_translation=pos_deviation<=c.relative_position_deviation_m,
            orientation_consistency=(relative_rotation<=c.relative_rotation_deviation_rad) if observable else tcp_rotation<=c.maximum_TCP_rotation_if_object_rotation_unobservable_rad,
            actual_TCP_motion_witness=tcp_motion>=c.minimum_TCP_motion_m,
            visual_object_motion_witness=object_motion>=c.minimum_visual_object_motion_m,
            target_in_declared_grasp_neighborhood=bool((distances<=c.maximum_TCP_object_distance_m).all()),
            nonempty_grasp_opening=bool(((openings>=c.opening_nonempty_min)&(openings<=c.opening_grasp_max)).all()),
            fingers_consistent=bool((np.abs(openings[:,0]-openings[:,1])<=c.maximum_finger_asymmetry).all()),
            stalled_opening=bool(n>1 and (ranges<=c.maximum_stalled_opening_range).all() and (speed<=c.maximum_stalled_opening_speed_per_second).all()),
            closure_actually_observed=bool(closure is not None and (closure>=c.minimum_observed_closure).all()))
        return dict(gates=gates,duration_seconds=duration,relative_position_deviation_m=pos_deviation,
            TCP_rotation_excursion_rad=tcp_rotation,relative_rotation_deviation_rad=relative_rotation,
            rotation_observable=observable,TCP_motion_m=tcp_motion,visual_object_motion_m=object_motion,
            TCP_object_distance_m=distances.tolist(),opening_range=ranges.tolist(),opening_speed_per_second=speed.tolist() if n>1 else None,
            observed_closure=None if closure is None else closure.tolist(),
            closure_baseline_available=self._initial_opening is not None,
            failing_gates=[key for key,value in gates.items() if not value])

    def observe(self,target,robot,*,close_requested,orientation_observable=False):
        if not isinstance(robot,PublicRobotSensors):robot=PublicRobotSensors.from_mapping(robot)
        if type(close_requested) is not bool or type(orientation_observable) is not bool:
            raise ValueError('explicit control-intent and public orientation-observability booleans required')
        now=float(robot.timestamp);c=self.config
        if self.last_timestamp is not None and now<=self.last_timestamp:
            raise ValueError('unique increasing actual public sensor timestamps required')
        gap=None if self.last_timestamp is None else now-self.last_timestamp
        fresh_previous=(gap is not None and gap<=c.maximum_gap_seconds
            and np.isclose(gap,c.sensor_dt_seconds,atol=1e-6,rtol=0))
        previous_opening=self._previous_public_opening
        previous_close_requested=self._previous_close_requested
        previous_timestamp=self.last_timestamp
        # Store actual public feedback once even when the visual target is
        # missing. Never insert a synthetic pre-command opening or timestamp.
        self.last_timestamp=now
        self._previous_public_opening=robot.finger_open_amounts.copy()
        self._previous_close_requested=close_requested
        if close_requested and not self.attempt_active:
            self.attempt_active=True
            if fresh_previous and previous_close_requested is False and previous_opening is not None:
                self._initial_opening=previous_opening.copy()
                self._closure_baseline_timestamp=previous_timestamp
                self._closure_baseline_reason='fresh_real_open_intent_frame_before_first_close'
            else:
                self._initial_opening=None;self._closure_baseline_timestamp=None
                self._closure_baseline_reason='first_close_without_fresh_50ms_open_intent_preframe'
            self._rows.clear();self._attachment_reference=None;self._release_reference=None;self._public_attachment_seen=False;self.mode=MotionMode.UNRESOLVED
        if gap is not None and (gap>c.maximum_gap_seconds or not np.isclose(gap,c.sensor_dt_seconds,atol=1e-6,rtol=0)):
            if self.attempt_active:
                self._initial_opening=None;self._closure_baseline_timestamp=None
                self._closure_baseline_reason='missing_or_non_50ms_feedback_invalidates_closure_baseline'
            return self._invalid(now,'missing_or_non_50ms_real_public_feedback_no_interpolation')
        if target is not None and not isinstance(target,ObjectState):
            raise ValueError('only public RGBD tracker ObjectState or explicit None is accepted; no Native state dictionary')
        if target is None or not target.visible or target.confidence<c.minimum_confidence:
            return self._invalid(now,'visual_target_missing_or_low_confidence_no_attachment_fabricated')
        if target.object_id!='target' or abs(target.timestamp-now)>1e-8:
            raise ValueError('fresh same-time public target required; no stale or future pose')
        tcp=robot.tcp_pose
        relative=Pose(tcp.inverse_apply(target.pose.position),tcp.rotation.T@target.pose.rotation)
        if not self.attempt_active:
            self.mode=MotionMode.FREE
            return self._result(now,'no_close_attempt_no_attachment_claim')
        if self._release_reference is None:self._release_reference=relative
        self._rows.append(dict(timestamp=now,target=target,tcp=tcp,relative=relative,
            opening=robot.finger_open_amounts.copy(),orientation_observable=orientation_observable))
        rows=list(self._rows);diagnostic=self._window_metrics(rows)
        if not close_requested:
            self.mode=MotionMode.UNRESOLVED;self._attachment_reference=None
            release=rows[-c.release_frames:]
            open_window=(len(release)>=c.release_frames
                and release[-1]['timestamp']-release[0]['timestamp']>=c.release_duration_seconds-1e-8
                and all(bool((r['opening']>=c.release_opening_min).all()) for r in release))
            outside_neighborhood=all(np.linalg.norm(r['relative'].position)>c.maximum_TCP_object_distance_m for r in release)
            relative_changed=all(np.linalg.norm(r['relative'].position-self._release_reference.position)>=c.minimum_release_relative_change_m for r in release)
            separated=(open_window and all(np.linalg.norm(r['relative'].position)>=c.release_separation_m for r in release)
                       and (relative_changed or (not self._public_attachment_seen and outside_neighborhood)))
            diagnostic['release_relative_change_observed']=bool(relative_changed)
            diagnostic['release_window_open']=bool(open_window);diagnostic['release_visually_separated']=bool(separated)
            if separated:
                evidence=PublicMotionEvidence('release_confirmed',now,('rgbd_object_release','public_gripper_opening'))
                self.mode=MotionMode.FREE;self.attempt_active=False;self._initial_opening=None;self._release_reference=None;self._public_attachment_seen=False;self._rows.clear()
                self._closure_baseline_timestamp=None;self._closure_baseline_reason='released_no_active_close_attempt'
                diagnostic['release_means_public_free_reestablished_not_proof_prior_attachment']=True
                return self._result(now,'public_open_and_visually_separated_free_motion_reestablished',evidence=evidence,diagnostic=diagnostic)
            return self._result(now,'opening_intent_requires_real_open_and_visual_separation',evidence=self._unresolved_evidence(now),diagnostic=diagnostic)
        maintaining=self.mode==MotionMode.ATTACHED_PUBLIC and self._attachment_reference is not None
        gates=diagnostic['gates']
        if maintaining:
            drift=float(np.linalg.norm(relative.position-self._attachment_reference.position))
            diagnostic['fixed_attachment_relative_drift_m']=drift
            accepted=(drift<=c.maximum_attached_relative_drift_m and all(gates[key] for key in
                ('real_consecutive_feedback_window','stable_relative_translation','orientation_consistency',
                 'target_in_declared_grasp_neighborhood','nonempty_grasp_opening','fingers_consistent','stalled_opening','closure_actually_observed')))
        else:accepted=all(gates.values())
        if accepted:
            if not maintaining:
                self._attachment_reference=relative;self._release_reference=relative;self._public_attachment_seen=True
            estimate=Pose(np.mean([r['relative'].position for r in rows],axis=0),relative.rotation)
            evidence=PublicMotionEvidence('attachment_confirmed',now,
                ('rgbd_relative_pose_tracking','public_robot_pose','public_gripper_opening'),estimate,
                orientation_observable=diagnostic['rotation_observable'])
            self.mode=MotionMode.ATTACHED_PUBLIC
            return self._result(now,'public_relative_motion_and_stalled_nonempty_opening_consistent',evidence=evidence,diagnostic=diagnostic)
        self.mode=MotionMode.UNRESOLVED;self._attachment_reference=None
        return self._result(now,'post_close_public_evidence_not_sufficient',evidence=self._unresolved_evidence(now),diagnostic=diagnostic)


class PublicTargetFrameCache:
    """Pass one real public-tracker result to bridge without tracking twice.

    bridge.reset during a same-frame release preserves an unconsumed staged
    observation. Old unconsumed frames may be discarded, never replayed as a
    later sensor sample. Pass public_tracker=the RedBoxTracker so bridge resets
    also restart its causal velocity history; call reset_episode at boundaries.
    """
    def __init__(self,public_tracker=None):
        if public_tracker is not None and not isinstance(public_tracker,RedBoxTracker):
            raise ValueError('optional history reset hook must be the public RedBoxTracker instance')
        self.public_tracker=public_tracker
        self._pending=None;self._last_staged=None;self.discarded_unconsumed_frames=0

    def reset(self):
        if self.public_tracker is not None:
            self.public_tracker.history.clear();self.public_tracker.last_velocity=np.zeros(3)
            if self._pending is not None and not self._pending['consumed'] and self._pending['target'] is not None:
                # Same real measured pose and timestamp; derived velocity is
                # the causal zero-history initialization, not a fake frame.
                self._pending['target']=replace(self._pending['target'],linear_velocity=np.zeros(3),angular_velocity=np.zeros(3))
                self._pending['velocity_history_reset']=True
        if self._pending is None or self._pending['consumed']:self._pending=None

    def reset_episode(self):
        self.reset();self._pending=None;self._last_staged=None;self.discarded_unconsumed_frames=0

    def stage_public_target(self,target,*,timestamp,diagnostic=None):
        if not np.isfinite(timestamp) or timestamp<0:raise ValueError('real public timestamp required')
        if self._last_staged is not None and timestamp<=self._last_staged:
            raise ValueError('do not duplicate/interpolate/reorder cached public frames')
        if target is not None and (not isinstance(target,ObjectState) or abs(target.timestamp-timestamp)>1e-8):
            raise ValueError('current public tracker ObjectState or explicit unknown required')
        allowed={'backend','measured_points','visible_views','geometry_completion','simulator_pose_used',
                 'future_observations_used','known_dimensions_m'}
        if diagnostic is not None and (not isinstance(diagnostic,dict) or set(diagnostic)-allowed
                or diagnostic.get('simulator_pose_used',False) is not False
                or diagnostic.get('future_observations_used',False) is not False):
            raise ValueError('only frozen public RGBD tracker diagnostics allowed; no Native state channels')
        if self._pending is not None and not self._pending['consumed']:self.discarded_unconsumed_frames+=1
        self._pending=dict(target=target,timestamp=float(timestamp),diagnostic={} if diagnostic is None else dict(diagnostic),consumed=False,velocity_history_reset=False)
        self._last_staged=float(timestamp)

    def observe(self,observation):
        if not isinstance(observation,dict) or set(observation)!=PUBLIC_OBSERVATION_KEYS:
            raise ValueError('only current public RGBD/calibration keys accepted')
        if self._pending is None or self._pending['consumed'] or abs(observation['timestamp']-self._pending['timestamp'])>1e-8:
            raise ValueError('one staged same-time real public observation is required')
        self._pending['consumed']=True
        if self._pending['target'] is None:raise ValueError('explicit missing public target; no pose fabricated')
        return self._pending['target'],dict(self._pending['diagnostic'],public_tracker_frame_cache=True,
            one_real_current_observation=True,Native_inputs_used=False,
            public_velocity_history_reset=self._pending['velocity_history_reset'])
