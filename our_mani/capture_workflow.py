"""Forecast/Value selection, real FASTER approach, fresh handover and capture.

The local controller owns closing, lift, and visual goal transport so its Value
uses the same controller version as the local rollout experiments.
"""
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy.spatial.transform import Rotation

from .control import HandoverGate
from .local_capture import LocalCaptureController
from .pipeline import Decision, Stage


class CaptureWorkflow:
    def __init__(self, dynamics, selector, regions, approach, buffer, goal_pose,
                 local_config, *, learned_policy=None, horizon=1.1, capture_budget=3., collision_scene=None,
                 local_controller_factory=None):
        self.dynamics, self.selector, self.regions = dynamics, selector, regions
        self.approach, self.buffer, self.goal_pose = approach, buffer, goal_pose
        self.local_config, self.learned_policy = local_config, learned_policy
        self.local_controller_factory = local_controller_factory
        self.horizon, self.capture_budget = horizon, capture_budget
        self.collision_scene = collision_scene
        self.gate = HandoverGate()
        self.stage, self.goal, self.local = Stage.SEEK, None, None
        self.events, self.capture_start, self.stable_since = [], None, None
        self.close_height = None
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='our-mani-workflow-forecast')
        self.future = None

    def transition(self, stage, now, reason):
        self.events.append({'time': float(now), 'from': self.stage.value, 'to': stage.value,
                            'reason': reason, 'goal_id': self.goal.goal_id if self.goal else None})
        self.stage = stage

    def replan(self, now, reason):
        self.buffer.invalidate()
        self.transition(Stage.SEEK, now, reason)
        self.goal, self.local = None, None
        return Decision(self.stage, 'hold_and_replan', details={'reason': reason})

    def step(self, observation, robot, obj, *, tick):
        now = robot.timestamp
        if self.stage in (Stage.SUCCESS, Stage.FAILED):
            return Decision(self.stage, 'hold')
        if (obj is None or not obj.visible or now-obj.timestamp > self.gate.max_age or obj.timestamp > now+1e-6
                or obj.confidence < .5 or abs(obj.timestamp-robot.timestamp) > self.gate.max_skew
                or abs(float(observation['timestamp'])-robot.timestamp) > self.gate.max_skew):
            return self.replan(now, 'missing_or_stale_current_target')
        if hasattr(self.dynamics, 'observe'):
            self.dynamics.observe([obj])
        if self.stage == Stage.SEEK:
            if self.future is None:
                self.future = self.executor.submit(self.dynamics.predict, [obj], self.horizon, .05)
                return Decision(self.stage, 'hold', details={'reason': 'forecast_in_progress'})
            if not self.future.done():
                return Decision(self.stage, 'hold', details={'reason': 'forecast_in_progress'})
            future, self.future = self.future, None
            try:
                forecast = future.result()
                predicted, _, _ = forecast.at(obj.object_id, now)
            except (ValueError, IndexError) as error:
                return Decision(self.stage, 'hold', details={'reason': 'forecast_unavailable', 'error': str(error)})
            if np.linalg.norm(predicted.position-obj.pose.position) > .08:
                return Decision(self.stage, 'hold', details={'reason': 'forecast_no_longer_matches_observation'})
            if self.collision_scene is not None:
                self.collision_scene.bind(forecast, now)
            self.goal = self.selector.select(forecast, self.regions, robot, now, stride=2)
            if self.goal is None:
                return Decision(self.stage, 'hold', details=self.selector.last_diagnostics)
            self.buffer.activate(self.goal.goal_id)
            self.transition(Stage.APPROACH, now, 'reachable_region_time_twist_and_controller_value_selected')
        if self.stage == Stage.APPROACH:
            if self.collision_scene is not None:
                self.collision_scene.bind(self.goal.forecast, now)
            handover = self.gate.check(self.goal, obj, robot, now)
            if handover.allowed:
                probability = self.goal.region.value
                if self.selector.success_value is not None:
                    probability = self.selector.success_value(region=self.goal.region, object_pose=obj.pose,
                        object_velocity=obj.linear_velocity, object_angular_velocity=obj.angular_velocity,
                        robot_pose=robot.pose, robot_velocity=robot.linear_velocity,
                        robot_angular_velocity=robot.angular_velocity)
                if not np.isfinite(probability) or probability < self.selector.minimum_success_probability:
                    return self.replan(now, 'fresh_state_value_below_capture_threshold')
                self.buffer.invalidate()
                self.local = (self.local_controller_factory(self.goal_pose,self.local_config)
                    if self.local_controller_factory is not None else
                    LocalCaptureController(self.goal_pose, self.local_config, learned_policy=self.learned_policy))
                self.capture_start, self.close_height = now, obj.pose.position[2]
                self.transition(Stage.LOCAL, now, 'fresh_rgbd_proprioception_and_value_passed_handover')
            elif now > self.goal.arrival_time+self.gate.time_tolerance:
                return self.replan(now, 'missed_handover')
            else:
                try:
                    predicted, _, _ = self.goal.forecast.at(obj.object_id, now)
                except ValueError:
                    # A fresh capture remains valid within the permitted time
                    # window. Hold instead of extrapolating the old forecast.
                    return Decision(self.stage, 'hold', self.goal.goal_id,
                                    details={'reason':'forecast_ended_waiting_for_fresh_handover',
                                             'handover':handover.__dict__})
                if np.linalg.norm(predicted.position-obj.pose.position) > .08:
                    return self.replan(now, 'target_changed_since_prediction')
                causal = {k:v for k,v in observation.items() if k in ('timestamp', 'prompt', 'observation/images', 'observation/state')}
                self.approach.submit(causal, self.goal, now=now, start_tick=tick+1)
                action = self.buffer.pop(tick, now=now)
                return Decision(self.stage, 'faster_action' if action is not None else 'hold', self.goal.goal_id,
                                details={'action': action, 'handover': handover.__dict__})
        if self.stage == Stage.LOCAL:
            if now-self.capture_start > self.capture_budget and not robot.contact:
                return self.replan(now, 'local_capture_budget_exceeded')
            if now-self.capture_start > self.capture_budget+4.5:
                self.transition(Stage.FAILED, now, 'visual_goal_transport_timeout')
                return Decision(self.stage, 'hold')
            valid = (robot.contact and not robot.gripper_open and robot.task_success
                     and obj.pose.position[2]-self.close_height >= .08)
            self.stable_since = (self.stable_since if self.stable_since is not None else now) if valid else None
            if self.stable_since is not None and now-self.stable_since >= .3-1e-6:
                self.transition(Stage.SUCCESS, now, 'contact_stable_lift_and_official_goal_success_verified')
                return Decision(self.stage, 'hold')
            try:
                pose, opening, v, w, details = self.local.command(robot, obj)
            except RuntimeError as error:
                return self.replan(now, str(error))
            return Decision(self.stage, 'local_capture', self.goal.goal_id, pose, v, w,
                            {**details, 'gripper_open': opening})
        raise RuntimeError('unhandled capture workflow stage')

    def close(self):
        self.buffer.invalidate()
        self.approach.close()
        self.executor.shutdown(wait=True)
