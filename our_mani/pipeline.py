"""The user's approach -> recheck -> local control -> grasp/lift closed loop."""
from dataclasses import dataclass
from enum import Enum

import numpy as np
from scipy.spatial.transform import Rotation

from .contracts import Pose
from .control import HandoverGate


class Stage(str, Enum):
    SEEK = "seek"
    APPROACH = "approach_faster"
    LOCAL = "local_visual_control"
    CLOSE = "close_gripper"
    LIFT = "stable_lift"
    SUCCESS = "verified_success"
    FAILED = "failed"


@dataclass(frozen=True)
class Decision:
    stage: Stage
    command: str
    goal_id: int | None = None
    pose: Pose | None = None
    linear_velocity: np.ndarray | None = None
    angular_velocity: np.ndarray | None = None
    details: dict | None = None


class ManipulationPipeline:
    def __init__(self, dynamics, selector, local_controller, regions, local_grasp_pose,
                 *, horizon=1.2, forecast_dt=0.05, maximum_target_jump=0.08,
                 maximum_target_rotation=0.5, grasp_distance=0.025, grasp_relative_speed=0.08,
                 lift_height=0.08, stable_duration=0.3, close_timeout=1.0,
                 lift_timeout=4.0, require_task_success=True, invalidate_actions=None):
        self.dynamics, self.selector, self.local_controller = dynamics, selector, local_controller
        self.regions, self.local_grasp_pose = regions, local_grasp_pose
        self.horizon, self.forecast_dt = horizon, forecast_dt
        self.maximum_target_jump, self.maximum_target_rotation = maximum_target_jump, maximum_target_rotation
        self.grasp_distance, self.grasp_relative_speed = grasp_distance, grasp_relative_speed
        self.lift_height, self.stable_duration = lift_height, stable_duration
        self.close_timeout, self.lift_timeout = close_timeout, lift_timeout
        self.require_task_success = require_task_success
        self.invalidate_actions = invalidate_actions or (lambda: None)
        self.gate = HandoverGate()
        self.stage, self.goal = Stage.SEEK, None
        self.events = []
        self.stable_since = None
        self.stage_since = None
        self.close_height = None
        self.lift_pose = None

    def _transition(self, stage, now, reason):
        self.events.append({"time": float(now), "from": self.stage.value, "to": stage.value,
                            "reason": reason, "goal_id": self.goal.goal_id if self.goal else None})
        self.stage, self.stage_since = stage, now

    def _replan(self, now, reason):
        self.invalidate_actions()
        self._transition(Stage.SEEK, now, reason)
        self.goal = None
        self.stable_since, self.close_height, self.lift_pose = None, None, None
        return Decision(self.stage, "hold_and_replan", details={"reason": reason})

    def step(self, objects, robot, now, *, forecast=None):
        if self.stage in (Stage.SUCCESS, Stage.FAILED):
            return Decision(self.stage, "hold")
        if not objects:
            return self._replan(now, "no_target_observation")
        if now-robot.timestamp > self.gate.max_age or robot.timestamp > now+1e-6:
            return self._replan(now, "stale_or_future_robot_state")
        if any(not s.visible or s.confidence < self.gate.min_confidence or now-s.timestamp > self.gate.max_age or s.timestamp > now+1e-6 for s in objects):
            return self._replan(now, "stale_or_unreliable_target")
        if any(abs(s.timestamp-robot.timestamp) > self.gate.max_skew for s in objects):
            return self._replan(now, "unsynchronized_observation")
        if self.stage == Stage.SEEK:
            supplied = forecast is not None
            if not supplied:
                forecast = self.dynamics.predict(objects, self.horizon, self.forecast_dt)
            else:
                # An asynchronous result is validated against CURRENT geometry.
                for obj in objects:
                    try:
                        predicted, _, _ = forecast.at(obj.object_id, obj.timestamp)
                    except (ValueError, IndexError):
                        return Decision(self.stage, "hold_and_replan", details={"reason": "async_forecast_expired_or_identity_changed"})
                    turn = Rotation.from_matrix(predicted.rotation@obj.pose.rotation.T).magnitude()
                    if np.linalg.norm(predicted.position-obj.pose.position) > self.maximum_target_jump or turn > self.maximum_target_rotation:
                        return Decision(self.stage, "hold_and_replan", details={"reason": "async_forecast_target_mismatch"})
            # Async latency is already reflected in now; do not charge it twice.
            planning_now = now if supplied else now + forecast.metadata.get("latency_seconds", 0.)
            goal = self.selector.select(forecast, self.regions, robot, planning_now)
            if goal is None:
                return Decision(self.stage, "hold", details=self.selector.last_diagnostics)
            self.goal = goal
            self.invalidate_actions()
            self._transition(Stage.APPROACH, now, "feasible_region_time_and_velocity_selected")
            return Decision(self.stage, "request_faster", goal.goal_id, goal.pose,
                            goal.linear_velocity, goal.angular_velocity, self.selector.last_diagnostics)
        obj = next((s for s in objects if s.object_id == self.goal.region.object_id), None)
        if obj is None:
            return self._replan(now, "target_identity_changed")
        if self.stage == Stage.APPROACH:
            try:
                predicted_pose, _, _ = self.goal.forecast.at(obj.object_id, obj.timestamp)
            except ValueError:
                return self._replan(now, "forecast_expired")
            jump = np.linalg.norm(obj.pose.position-predicted_pose.position)
            turn = Rotation.from_matrix(obj.pose.rotation@predicted_pose.rotation.T).magnitude()
            if jump > self.maximum_target_jump or turn > self.maximum_target_rotation:
                return self._replan(now, "target_changed_since_prediction")
            check = self.gate.check(self.goal, obj, robot, now)
            if check.allowed:
                self.invalidate_actions()  # queued VLA chunks cannot outlive handover
                self._transition(Stage.LOCAL, now, "latest_observation_passed_handover")
            elif now > self.goal.arrival_time+self.gate.time_tolerance:
                return self._replan(now, "missed_handover")
            else:
                return Decision(self.stage, "continue_faster", self.goal.goal_id, details=check.__dict__)
        if self.stage == Stage.LOCAL:
            grasp_pose, grasp_v = self._grasp_target(obj)
            distance = np.linalg.norm(robot.pose.position-grasp_pose.position)
            relative_speed = np.linalg.norm(robot.linear_velocity-grasp_v)
            turn = Rotation.from_matrix(robot.pose.rotation@grasp_pose.rotation.T).magnitude()
            if distance <= self.grasp_distance and relative_speed <= self.grasp_relative_speed and turn <= self.goal.region.max_orientation_error:
                self.close_height = obj.pose.position[2]
                self._transition(Stage.CLOSE, now, "pose_and_relative_velocity_ready_for_grasp")
                return Decision(self.stage, "close_gripper", self.goal.goal_id, robot.pose)
            try:
                pose, v, w, details = self.local_controller.command(robot, obj, self.local_grasp_pose)
            except RuntimeError as error:
                return self._replan(now, str(error))
            check = self.selector.reachability.check(robot, pose, v, w, self.local_controller.dt)
            if not check.feasible:
                return self._replan(now, "local_command_"+check.reason)
            return Decision(self.stage, "local_pose", self.goal.goal_id, pose, v, w, details)
        if self.stage == Stage.CLOSE:
            if not robot.gripper_open and robot.contact:
                self.lift_pose = Pose(robot.pose.position+np.array([0., 0., self.lift_height+0.02]), robot.pose.rotation)
                self._transition(Stage.LIFT, now, "gripper_contact_confirmed")
            elif now-self.stage_since > self.close_timeout:
                return self._replan(now, "grasp_contact_timeout")
            else:
                # Continue tracking while the fingers close on a moving object.
                # Holding the last pose would abandon relative-velocity matching.
                try:
                    pose, v, w, details = self.local_controller.command(robot, obj, self.local_grasp_pose)
                except RuntimeError as error:
                    return self._replan(now, str(error))
                return Decision(self.stage, "close_gripper", self.goal.goal_id, pose, v, w, details)
        if self.stage == Stage.LIFT:
            if robot.gripper_open or not robot.contact:
                return self._replan(now, "grasp_lost_during_lift")
            if now-self.stage_since > self.lift_timeout:
                self._transition(Stage.FAILED, now, "lift_verification_timeout")
                return Decision(self.stage, "hold", details={"reason": "lift_verification_timeout"})
            lifted = obj.pose.position[2]-self.close_height >= self.lift_height
            slow = np.linalg.norm(obj.linear_velocity-robot.linear_velocity) < self.grasp_relative_speed
            valid = lifted and slow and (robot.task_success or not self.require_task_success)
            self.stable_since = (self.stable_since if self.stable_since is not None else now) if valid else None
            if self.stable_since is not None and now-self.stable_since >= self.stable_duration:
                self._transition(Stage.SUCCESS, now, "contact_lift_stability_and_task_success_verified")
                return Decision(self.stage, "hold", self.goal.goal_id)
            return Decision(self.stage, "lift_pose", self.goal.goal_id, self.lift_pose,
                            details={"lifted": bool(lifted), "task_success": robot.task_success})
        raise RuntimeError(f"unhandled stage {self.stage}")

    def _grasp_target(self, obj):
        offset = obj.pose.rotation@self.local_grasp_pose.position
        return Pose(obj.pose.position+offset, obj.pose.rotation@self.local_grasp_pose.rotation), obj.linear_velocity+np.cross(obj.angular_velocity, offset)
