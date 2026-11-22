"""Public, current-frame goal dwell for the optional v52 stop protocol.

This is an engineering completion predicate, not the Native task judge or a
contact/orientation certificate.  The initial RGBD-marker goal is fixed here;
only the current visible RGBD body centre can satisfy it.  Controlled poses,
forecasts, Native object state and historical attachment confirmation are not
read.  Call once after each actual 50ms public observation, including unknown
frames; a caller must not subsample or synthesize missing observations.
"""
from copy import deepcopy
from dataclasses import asdict, dataclass

import numpy as np

from ..contracts import ObjectState, Pose
from .public_motion_bridge import PublicMotionEvidence, PublicRobotSensors


@dataclass(frozen=True)
class PublicGraspGoalCompletionConfig:
    sensor_dt: float = .05
    minimum_confidence: float = .5
    maximum_goal_distance_m: float = .02
    goal_dwell_seconds: float = .1
    minimum_public_hold_seconds: float = .5

    def __post_init__(self):
        if any(isinstance(value, (bool, np.bool_)) or not np.isfinite(value)
               for value in asdict(self).values()):
            raise ValueError('finite explicit public completion parameters required')
        if (self.sensor_dt != .05 or not .5 <= self.minimum_confidence <= 1.
                or not 0 < self.maximum_goal_distance_m <= .02
                or self.goal_dwell_seconds < .1 or self.minimum_public_hold_seconds < .5):
            raise ValueError('real50ms, confidence>=.5, distance<=2cm, dwell>=.1s and public hold>=.5s required')


def _same_time(value, now):
    return (not isinstance(value, (bool, np.bool_))
            and isinstance(value, (float, int, np.floating, np.integer))
            and np.isfinite(value) and abs(float(value) - now) <= 1e-8)


class PublicGraspGoalCompletion:
    """Bind one initial public marker Pose, then observe actual public frames.

    ``public_hold`` accepts ``PublicLiftHoldMetric.receipt()``, its ``latest``
    dict, or an explicit bool from that metric's current ``passed`` property.
    The bool form is a caller assertion of the declared >=.5s public metric;
    a receipt permits independent freshness/duration checks and is preferred.
    ``close_requested`` is known current command intent, not finger position:
    an active attachment attempt can remain active while an open command awaits
    visual release.  No current completion is sticky across invalid frames.
    """

    def __init__(self, goal_pose, config=None):
        if not isinstance(goal_pose, Pose):
            raise TypeError('initial RGBD-marker public goal must be a typed Pose')
        self.goal_pose = Pose(goal_pose.position, goal_pose.rotation)
        self.config = config or PublicGraspGoalCompletionConfig()
        if type(self.config) is not PublicGraspGoalCompletionConfig:
            raise TypeError('explicit PublicGraspGoalCompletionConfig required')
        self.reset()

    def reset(self):
        """Reset episode/attempt timing; retain the explicitly bound public goal."""
        self.last_timestamp = None
        self.attempt_id = None
        self.dwell_started = None
        self.qualifying_frames = 0
        self.ever_passed = False
        self.latest = None

    @property
    def passed(self):
        return bool(self.latest and self.latest['public_grasp_goal_complete'])

    def _clear_dwell(self):
        self.dwell_started = None
        self.qualifying_frames = 0

    def _hold(self, public_hold, now):
        if type(public_hold) is bool:
            return public_hold, None, 'explicit_current_public_metric_passed_bool'
        if not isinstance(public_hold, dict):
            return False, None, 'public_hold_receipt_or_explicit_bool_missing'
        latest = public_hold.get('latest', public_hold)
        if not isinstance(latest, dict):
            return False, None, 'current_public_hold_receipt_missing'
        duration = latest.get('public_hold_seconds')
        numeric = (not isinstance(duration, (bool, np.bool_))
                   and isinstance(duration, (float, int, np.floating, np.integer))
                   and np.isfinite(duration))
        good = (latest.get('public_lift_hold_observed') is True
                and latest.get('current_visual_target') is True
                and latest.get('current_public_attachment_confirmed') is True
                and latest.get('actual_preclose_visual_baseline_available') is True
                and latest.get('controlled_pose_used_as_visual_label') is False
                and latest.get('Native_labels_used') is False
                and _same_time(latest.get('timestamp'), now) and numeric
                and duration >= self.config.minimum_public_hold_seconds - 1e-8)
        return bool(good), float(duration) if numeric else None, 'fresh_visible_public_hold_receipt'

    def observe(self, result, robot, public_hold, *, close_requested):
        """Return a JSON-safe current receipt; input failures never earn dwell."""
        if not isinstance(robot, PublicRobotSensors):
            self._clear_dwell()
            self.latest = None
            raise TypeError('current typed PublicRobotSensors required')
        if type(close_requested) is not bool:
            self._clear_dwell()
            self.latest = None
            raise TypeError('known current close_requested bool required')
        now = float(robot.timestamp)
        if self.last_timestamp is not None and now <= self.last_timestamp:
            self._clear_dwell()
            self.latest = None
            raise ValueError('unique increasing actual50ms frames required')
        gap = None if self.last_timestamp is None else now - self.last_timestamp
        consecutive = gap is None or abs(gap - self.config.sensor_dt) <= 1e-6
        self.last_timestamp = now
        data = result if isinstance(result, dict) else {}
        attachment = data.get('attachment_estimate')
        attachment = attachment if isinstance(attachment, dict) else {}
        target = data.get('object_observation')
        evidence = attachment.get('evidence')
        attempt = attachment.get('attempt_id')
        valid_attempt = (type(attempt) is int and attempt > 0
                         and attachment.get('attempt_active') is True)
        attempt_changed = self.attempt_id is not None and attempt != self.attempt_id
        self.attempt_id = attempt if valid_attempt else None
        same_frame = _same_time(data.get('timestamp'), now)
        visible = bool(isinstance(target, ObjectState) and target.object_id == 'target'
                       and target.visible is True and target.confidence >= self.config.minimum_confidence
                       and _same_time(target.timestamp, now))
        fresh_attachment = bool(data.get('mode') == 'ATTACHED_PUBLIC'
                                and attachment.get('mode') == 'ATTACHED_PUBLIC'
                                and _same_time(attachment.get('timestamp'), now)
                                and isinstance(evidence, PublicMotionEvidence)
                                and evidence.event == 'attachment_confirmed'
                                and _same_time(evidence.timestamp, now))
        hold_passed, hold_duration, hold_scope = self._hold(public_hold, now)
        distance = float(np.linalg.norm(target.pose.position - self.goal_pose.position)) if visible else None
        goal_near = distance is not None and distance <= self.config.maximum_goal_distance_m + 1e-12
        public_current_scope = (data.get('available') is True
                                and data.get('controlled_mode_available') is True
                                and data.get('Native_inputs_used') is False
                                and data.get('future_observations_used') is False
                                and data.get('observations_fabricated') is False)
        gates = dict(same_current_public_frame=bool(same_frame),
                     current_available_public_observation_scope=bool(public_current_scope),
                     consecutive_actual50ms=bool(consecutive),
                     current_visible_typed_body=visible,
                     current_typed_public_attachment=fresh_attachment,
                     close_requested=close_requested, close_attempt_active=bool(valid_attempt),
                     current_public_lift_hold_passed=hold_passed,
                     current_visual_body_at_public_goal=bool(goal_near))
        if attempt_changed:
            self._clear_dwell()
        eligible = all(gates.values())
        if not eligible:
            self._clear_dwell()
        else:
            if self.dwell_started is None:
                self.dwell_started = now
            self.qualifying_frames += 1
        duration = 0. if self.dwell_started is None else now - self.dwell_started
        required_frames = int(np.ceil(self.config.goal_dwell_seconds / self.config.sensor_dt - 1e-10)) + 1
        complete = bool(eligible and duration >= self.config.goal_dwell_seconds - 1e-8
                        and self.qualifying_frames >= required_frames)
        self.ever_passed = self.ever_passed or complete
        failing = [key for key, value in gates.items() if not value]
        self.latest = dict(timestamp=now, protocol='public_visible_goal_completion_v52',
                           public_grasp_goal_complete=complete, ever_public_grasp_goal_complete=self.ever_passed,
                           reason=('current_visible_goal_and_public_hold_complete' if complete else
                                   (';'.join(failing) if failing else 'goal_dwell_not_yet_complete')),
                           gates=gates, attempt_id=self.attempt_id, attempt_changed=attempt_changed,
                           goal_distance_m=distance, goal_dwell_seconds=duration,
                           goal_dwell_started=self.dwell_started, consecutive_goal_frames=self.qualifying_frames,
                           required_goal_frames=required_frames, observed_gap_seconds=gap,
                           public_lift_hold_passed=hold_passed, public_hold_seconds=hold_duration,
                           hold_input_scope=hold_scope, public_goal_position_m=self.goal_pose.position.tolist(),
                           goal_source='explicitly_bound_initial_RGBD_marker_pose',
                           criteria=asdict(self.config), Native_inputs_used=False,
                           controlled_or_forecast_pose_used_as_visible_body=False,
                           Native_task_judge_replaced=False, formal_benchmark_qualification=False,
                           current_public_position_only=True, object_orientation_qualified=False,
                           engineering_starting_parameters=True)
        return deepcopy(self.latest)

    def receipt(self):
        return dict(criteria=asdict(self.config), latest=deepcopy(self.latest),
                    formal_qualification=False, Native_task_judge_replaced=False)
