"""Controlled carry geometry from public RGBD/TCP, separately from FREE history.

Native membership is an explicit caller feedback signal. It is neither a force
certificate nor the task judge. No Native object state is accepted here.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from ..contracts import Forecast, ObjectState, Pose
from ..local_mpc_control_20261103.public_motion_bridge import PublicRobotSensors


class NativeContactCarryEstimate:
    """A TCP-relative estimate with explicit measured/predicted provenance.

    A currently visible target can refresh the relative pose. On first contact
    without current vision, the last FREE forecast may initialize it at the
    actual contact observation time, strictly inside its existing <=.8s range.
    The estimate then follows actual public TCP poses; it never becomes a
    current visual measurement and must never be put in the FREE GRU history.
    """

    def __init__(self):
        self._last_robot = None
        self._was_held = False
        self._clear_offset()

    def _clear_offset(self):
        self._contact_time = None
        self._contact_tcp = None
        self._offset = None
        self._source = None
        self._source_time = None
        self._forecast_sha = None
        self._forecast_observed_until = None

    @staticmethod
    def _check_robot(robot):
        if type(robot) is not PublicRobotSensors:
            raise TypeError('Exact sanitized public robot sensors required')
        if isinstance(robot.timestamp, (bool, np.bool_)) or not np.isfinite(robot.timestamp) or robot.timestamp < 0:
            raise ValueError('Actual finite nonnegative public robot clock required')
        # Revalidate public geometry without retaining caller-owned buffers.
        return robot.tcp_pose

    @staticmethod
    def _check_target(target, now):
        if target is None:
            return
        if type(target) is not ObjectState or type(target.pose) is not Pose:
            raise TypeError('Current target must be an exact public ObjectState/Pose')
        if (target.object_id != 'target' or target.visible is not True
                or isinstance(target.timestamp, (bool, np.bool_))
                or not np.isfinite(target.timestamp) or abs(float(target.timestamp) - now) > 1e-8):
            raise ValueError('Same-frame actually visible public target required')

    def _robot_twist(self, robot, pose):
        if self._last_robot is None:
            return np.zeros(3), np.zeros(3), False
        previous = self._last_robot
        dt = float(robot.timestamp) - float(previous.timestamp)
        if dt <= 0:
            raise ValueError('Each increasing actual public robot timestamp consumed once')
        before = previous.tcp_pose
        v = (pose.position - before.position) / dt
        w = Rotation.from_matrix(pose.rotation @ before.rotation.T).as_rotvec() / dt
        return v, w, True

    @staticmethod
    def _forecast_contact_pose(forecast, contact_time):
        if forecast is None:
            return None, 'no_public_forecast'
        if type(forecast) is not Forecast:
            raise TypeError('Exact public Forecast required; no injected state mapping')
        if forecast.object_ids != ('target',):
            raise ValueError('Target-only public forecast required')
        observed = forecast.metadata.get('observed_until')
        digest = forecast.metadata.get('checkpoint_sha256')
        if (isinstance(observed, (bool, np.bool_)) or not isinstance(observed, (float, int))
                or not np.isfinite(observed) or abs(float(observed) - float(forecast.times[0])) > 1e-8
                or not isinstance(digest, str) or len(digest) != 64
                or any(c not in '0123456789abcdef' for c in digest)):
            raise ValueError('Causal frozen public forecast provenance required')
        # Strict inclusion: Forecast.at permits tiny clamping; this initializer
        # explicitly forbids using any point outside the recorded forecast.
        if (observed > contact_time or contact_time < forecast.times[0]
                or contact_time > forecast.times[-1]
                or forecast.times[-1] - forecast.times[0] > .8 + 1e-8):
            return None, 'contact_time_outside_causal_existing_0p8_forecast'
        return forecast.at('target', contact_time)[0], None

    def observe(self, robot: PublicRobotSensors, *, held: bool,
                current_target: ObjectState | None,
                last_public_forecast: Forecast | None) -> dict:
        pose = self._check_robot(robot)
        if type(held) is not bool:
            raise TypeError('Actual membership feedback must be an explicit Boolean')
        now = float(robot.timestamp)
        self._check_target(current_target, now)
        v_tcp, w_tcp, twist_valid = self._robot_twist(robot, pose)
        released = self._was_held and not held
        reason = None
        if not held:
            self._clear_offset()
        else:
            if not self._was_held:
                self._clear_offset()
                self._contact_time = now
                self._contact_tcp = pose
            if current_target is not None:
                self._offset = Pose(pose.inverse_apply(current_target.pose.position),
                                    pose.rotation.T @ current_target.pose.rotation)
                self._source = 'current_visual_TCP_relative_estimate'
                self._source_time = now
                self._forecast_sha = self._forecast_observed_until = None
            elif self._offset is None:
                predicted, reason = self._forecast_contact_pose(last_public_forecast, self._contact_time)
                if predicted is not None:
                    anchor = self._contact_tcp
                    self._offset = Pose(anchor.inverse_apply(predicted.position),
                                        anchor.rotation.T @ predicted.rotation)
                    self._source = 'in_horizon_public_forecast_at_actual_contact'
                    self._source_time = self._contact_time
                    self._forecast_sha = last_public_forecast.metadata['checkpoint_sha256']
                    self._forecast_observed_until = float(last_public_forecast.metadata['observed_until'])

        estimate = None
        v_obj = w_obj = np.zeros(3)
        if held and self._offset is not None:
            lever = pose.rotation @ self._offset.position
            estimate = Pose(pose.position + lever, pose.rotation @ self._offset.rotation)
            v_obj = v_tcp + np.cross(w_tcp, lever)
            w_obj = w_tcp.copy()
        current_visual = held and current_target is not None
        result = dict(
            status=('FREE' if not held else 'HELD_CURRENT_VISUAL' if current_visual
                    else 'HELD_CONTROLLED_ESTIMATE' if estimate is not None else 'HELD_OFFSET_UNKNOWN'),
            timestamp=now, held_feedback=held, estimated_pose=estimate,
            linear_velocity=v_obj, angular_velocity=w_obj,
            offset_source=self._source,
            offset_age=None if self._source_time is None else now - self._source_time,
            offset_source_timestamp=self._source_time, contact_observation_time=self._contact_time,
            TCP_relative_pose=self._offset,
            forecast_checkpoint_sha256=self._forecast_sha,
            forecast_observed_until=self._forecast_observed_until,
            current_visual_measurement=bool(current_visual),
            proprioceptive_propagation=bool(estimate is not None and not current_visual),
            predicted_offset_initialization=self._source == 'in_horizon_public_forecast_at_actual_contact',
            velocity_source='finite_difference_public_TCP_plus_rigid_lever',
            velocity_estimate_valid=bool(estimate is not None and twist_valid),
            orientation_observed=False, controlled_estimate_only=True,
            FREE_history_eligible=False, Native_target_pose_used=False,
            task_success_claimed=False, force_contact_claimed=False,
            membership_loss_observed=bool(released), reason=reason)
        self._last_robot = robot
        self._was_held = held
        return result
