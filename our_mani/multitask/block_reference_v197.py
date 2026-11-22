"""Continuous local joint references around the original block geometry policy.

Original native parameters, grasp calls and counted actuator are retained.
Proposals use original public observations, both actual-state six-unit and
reference four-unit full Hermite checks, original collision queries and an
offset braking proof. Actual motor compliance remains a measured outcome.
"""
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
import numpy as np
from scipy.spatial.transform import Rotation
from ..contracts import Pose
from ..intercept_planner import JointTrajectory
from ..reachability import TimedReachability
from .joint16_block_episode_v98 import PublicCountedBlockEnvironment
from .bounded_transport_v159 import check_offset_brake

SCHEMA = 'r10_block_global_local_continuous_joint_reference_v197'


class ReferenceModel:
    def __init__(self, model):
        self.model = model
        self.acceleration_limits = np.full_like(model.acceleration_limits, 4.)
    def __getattr__(self, name): return getattr(self.model, name)
    @property
    def collision_query(self): return self.model.collision_query
    @collision_query.setter
    def collision_query(self, value): self.model.collision_query = value


class SmoothBlockEnvironment(PublicCountedBlockEnvironment):
    def __init__(self, env, context=None):
        super().__init__(env, context)
        self.public_robot = self.public_body = self.local_dynamics = self.current_decision = None
        self.targets = []
        self.reference_q = self.reference_v = None
        self.local_active = False
        self.backup = None
        self.reference_events = []

    def observe(self):
        result = super().observe()
        self.public_robot = result[1]
        return result

    def supplied_decision(self, robot, obj, decision, dynamics):
        if abs(robot.timestamp-self.env.time) > 1e-8:
            raise ValueError('local actuator needs the current already-observed public frame')
        self.public_robot, self.public_body, self.current_decision, self.local_dynamics = robot, obj, decision, dynamics

    def _target(self):
        return np.asarray(self.env.arm.get_joint_target_positions(), float)

    def command_joints(self, *args, **kwargs):
        decision = self.current_decision or {}
        if decision.get('command') == 'joint_trajectory' and 'reference_velocities' in decision:
            self.local_active = True
            self._initialize_reference()
            return self._issue(np.asarray(decision['reference_joints']), np.asarray(decision['reference_velocities']),
                kwargs.get('opening', 1.), decision['global_reference_proof_v197'])
        result = PublicCountedBlockEnvironment.command_joints(self, *args, **kwargs)
        self.targets.append(self._target().copy())
        self.reference_q, self.reference_v = self._target().copy(), np.zeros_like(self._target())
        return result

    def _initialize_reference(self):
        if self.reference_q is not None: return
        self.reference_q = self._target().copy()
        self.reference_v = ((self.targets[-1]-self.targets[-2])/self.env.ledger.dt
            if len(self.targets) >= 2 else np.zeros_like(self.reference_q))
        self.reference_events.append(dict(event='local_reference_initialized', time=float(self.env.time),
            position_basis='last_actual_applied_joint_target',
            initial_velocity_basis='backward_difference_of_last_two_applied_global_targets',
            initial_instantaneous_velocity_guaranteed=False,
            actual_motor_motion_guaranteed=False))

    @contextmanager
    def _collision_scope(self):
        model, body, robot = self.env.robot_model, self.public_body, self.public_robot
        if (body is None or not body.visible or body.timestamp > robot.timestamp+1e-8
                or robot.timestamp-body.timestamp > .1):
            raise ValueError('fresh public target required for original local collision service')
        previous = model.collision_query
        if robot.contact:
            local_p = robot.pose.rotation.T @ (body.pose.position-robot.pose.position)
            local_r = robot.pose.rotation.T @ body.pose.rotation
            def query(offset):
                # The original restoring collision probe has already set its
                # queried robot joints. This is kinematics, not dynamic truth.
                tool = Pose.from_xyzw(self.env.arm.get_tip().get_pose())
                return self.env.map_service.check(Pose(tool.position+tool.rotation@local_p, tool.rotation@local_r))
        else:
            forecast = self.local_dynamics.predict([body], 1.1, .05)
            def query(offset):
                try: return self.env.map_service.check(forecast.at('target', robot.timestamp+float(offset))[0])
                except (KeyError, ValueError, IndexError): return True
        model.collision_query = query
        try: yield
        finally: model.collision_query = previous

    def _gates(self):
        model = self.env.robot_model
        if not np.allclose(model.acceleration_limits, np.full_like(model.acceleration_limits, 6.)):
            raise ValueError('original native six-unit acceleration limits changed')
        return TimedReachability(model), TimedReachability(ReferenceModel(model))

    def _issue(self, q, v, opening, evidence):
        now = float(self.env.time)
        actual = np.asarray(self.env.arm.get_joint_positions(), float)
        if np.any(abs(q-actual) > self.env.robot_model.velocity_limits*self.env.ledger.dt+1e-7):
            return self._stationary_existing_target('original_actuator_would_clip_reference')
        result = super().command_joints(q, opening=opening, phase='local_continuous_reference')
        applied = self._target()
        if not np.allclose(applied, q, rtol=0, atol=1e-6):
            raise ValueError('original actuator did not apply the qualified reference')
        self.reference_q, self.reference_v = applied.copy(), np.asarray(v).copy()
        self.targets.append(applied.copy())
        self.reference_events.append(dict(event='local_reference_actually_dispatched', time=now,
            confirmed_after=float(self.env.time), reference_joints=np.asarray(q).tolist(),
            applied_joint_target=applied.tolist(), nominal_velocity=np.asarray(v).tolist(),
            evidence=evidence, actual_motion_guaranteed=False))
        return result

    def _stationary_existing_target(self, reason):
        target = self._target()
        result = super().command_joints(target, opening=self.env._requested_opening, phase='local_reference_unavailable_hold')
        self.targets.append(self._target().copy())
        self.reference_q, self.reference_v, self.backup = self._target().copy(), np.zeros_like(target), None
        self.reference_events.append(dict(event='existing_target_retained_without_new_moving_reference',
            time=float(self.env.time), reason=reason, actual_stopped_not_claimed=True,
            actual_motion_guaranteed=False))
        return result

    def _brake(self, opening, reason):
        now, dt = float(self.env.time), float(self.env.ledger.dt)
        actual_gate, reference_gate = self._gates()
        robot = self.public_robot
        reference_robot = replace(robot, joints=self.reference_q, joint_velocities=self.reference_v)
        if self.backup is None or now < self.backup.start-1e-8 or now > self.backup.end+1e-8:
            duration = max(.2, 1.5*float(np.max(abs(self.reference_v)/4.)))
            if now+duration > 10.+1e-8:
                return self._stationary_existing_target('no_qualified_brake_within_original_budget')
            end = self.reference_q+.5*self.reference_v*duration
            plan = JointTrajectory(now, now+duration, self.reference_q, self.reference_v, end, np.zeros_like(end))
        else: plan = self.backup
        duration = max(dt, plan.end-now)
        if now >= plan.end-1e-8:
            plan = JointTrajectory(now, now+dt, self.reference_q, np.zeros_like(self.reference_v), self.reference_q, np.zeros_like(self.reference_v))
        actual = actual_gate.check_joints(robot, plan.q1, plan.v1, duration)
        reference = reference_gate.check_joints(reference_robot, plan.q1, plan.v1, duration)
        if not (actual.feasible and reference.feasible):
            return self._stationary_existing_target('fresh_brake_actual_or_reference_gate_rejected')
        self.backup = plan
        q, v = plan.at(min(now+dt, plan.end))
        return self._issue(q, v, opening, dict(kind='original_collision_checked_continuous_brake',
            reason=reason, actual_gate=actual.reason, reference_gate=reference.reason))

    def advance(self, pose, *, opening=1.):
        decision, robot = self.current_decision, self.public_robot
        if (decision is None or decision.get('command') != 'local_capture' or robot is None
                or abs(robot.timestamp-self.env.time) > 1e-8
                or not np.allclose(pose.xyzw(), decision['pose'].xyzw(), rtol=0, atol=1e-9)):
            raise ValueError('original current local decision must reach the actual wrapper unchanged')
        self.local_active = True
        self._initialize_reference()
        now, dt = float(self.env.time), float(self.env.ledger.dt)
        actual_gate, reference_gate = self._gates()
        failures = []
        before_tick = int(self.env.ledger.ticks)
        try:
            with self._collision_scope():
                goal = self.env.robot_model.inverse(pose, robot.joints)
                if goal is not None:
                    solved = self.env.robot_model.forward(goal)
                    if (np.linalg.norm(solved.position-pose.position) > actual_gate.position_tolerance
                            or Rotation.from_matrix(solved.rotation @ pose.rotation.T).magnitude() > actual_gate.orientation_tolerance):
                        goal = None
                if goal is not None:
                    desired = np.clip((goal-robot.joints)/dt, -self.env.robot_model.velocity_limits, self.env.robot_model.velocity_limits)
                    reference_robot = replace(robot, joints=self.reference_q, joint_velocities=self.reference_v)
                    for duration in (.15, .25, .4, .5, .7, .9):
                        if now+duration > 10.+1e-8: continue
                        for scale in (1., .5, .25, .125):
                            terminal = clipped_terminal(self.reference_v, desired, duration, scale)
                            end = self.reference_q+.5*(self.reference_v+terminal)*duration
                            actual = actual_gate.check_joints(robot, end, terminal, duration)
                            reference = reference_gate.check_joints(reference_robot, end, terminal, duration)
                            failures.append(dict(duration=duration, scale=scale, actual=actual.reason, reference=reference.reason))
                            if not (actual.feasible and reference.feasible): continue
                            plan = JointTrajectory(now, now+duration, self.reference_q, self.reference_v, end, terminal)
                            q, v = plan.at(now+dt)
                            brake_duration = max(.2, 1.5*float(np.max(abs(v)/4.)))
                            if now+dt+brake_duration > 10.+1e-8: continue
                            endpoint = replace(robot, timestamp=now+dt, joints=q, joint_velocities=v)
                            backup = check_offset_brake(reference_gate, endpoint, dt, brake_duration)
                            if not backup.feasible: continue
                            self.backup = JointTrajectory(now+dt, now+dt+brake_duration, q, v,
                                q+.5*v*brake_duration, np.zeros_like(v))
                            return self._issue(q, v, opening, dict(kind='original_public_local_velocity_rolling_joint_Hermite',
                                actual_gate=actual.reason, reference_gate=reference.reason, reference_acceleration_cap=4.,
                                native_acceleration_cap=6., duration=duration, scale=scale,
                                offset_backup_brake=backup.reason, full_gate_trials=failures,
                                original_servo_lead_extrapolation_removed=True))
                return self._brake(self.env._requested_opening, 'local_goal_or_full_trajectory_rejected')
        except (ValueError, IndexError, KeyError) as error:
            if self.env.ledger.ticks != before_tick:
                raise
            self.reference_events.append(dict(event='public_local_reference_unavailable', time=now, error=str(error)))
            return self._stationary_existing_target('current_public_collision_forecast_unavailable')

    def hold_tick(self):
        if not self.local_active: return self.command_joints(self.env.arm.get_joint_positions(), opening=self.env._requested_opening, phase='hold')
        self._initialize_reference()
        before_tick = int(self.env.ledger.ticks)
        try:
            with self._collision_scope(): return self._brake(self.env._requested_opening, 'original_policy_requested_hold')
        except (ValueError, IndexError, KeyError):
            if self.env.ledger.ticks != before_tick:
                raise
            return self._stationary_existing_target('public_collision_unavailable_for_brake')

    def receipt(self):
        return dict(super().receipt(), global_local_continuous_reference_v197=dict(schema=SCHEMA,
            events=deepcopy(self.reference_events), reference_acceleration_limit=4., native_acceleration_limit=6.,
            original_native_actuator_physics_grasp_calls_unchanged=True,
            original_local_servo_lead_extrapolation_removed=True, actual_motion_guaranteed=False,
            no_extra_camera_or_uncounted_physics_calls=True, native_outcomes_supplied_to_actuator=False))


def clipped_terminal(current, desired, duration, scale):
    return current + np.clip(desired-current, -4.*duration, 4.*duration)*scale


class DualReferenceGate:
    def __init__(self, actual_model, reference_model):
        self.actual_gate = TimedReachability(actual_model)
        self.reference_gate = TimedReachability(reference_model)
    def __getattr__(self, name): return getattr(self.reference_gate, name)
    def _check_actual(self, robot, proof, duration):
        if not proof.feasible: return proof
        actual = self.actual_gate.check_joints(robot, proof.joints, proof.joint_velocities, duration)
        if not actual.feasible: return actual
        return replace(proof, evidence=dict(proof.evidence, original_actual_six_gate=actual.evidence,
            planning_reference_acceleration_cap=4., original_native_acceleration_cap=6.))
    def check(self, robot, pose, velocity, angular, duration):
        return self._check_actual(robot, self.reference_gate.check(robot, pose, velocity, angular, duration), duration)
    def check_joints(self, robot, q, v, duration):
        return self._check_actual(robot, self.reference_gate.check_joints(robot, q, v, duration), duration)


@contextmanager
def public_decision_scope(evaluator, boundary):
    original = evaluator.CandidateInterception
    def construct(*args, **kwargs):
        actual_model = args[1]
        reference_model = ReferenceModel(actual_model)
        amended = list(args)
        amended[1] = reference_model
        workflow = original(*amended, **kwargs)
        gate = DualReferenceGate(actual_model, reference_model)
        workflow.reachability = workflow.selector.reachability = gate
        def joint_command(robot):
            boundary._initialize_reference()
            now = float(robot.timestamp)
            plan = workflow.plan
            if plan is None or plan.end-now < boundary.env.ledger.dt-1e-8:
                return dict(command='hold', reason='original_global_reference_end_reached')
            reference_robot = replace(robot, joints=boundary.reference_q, joint_velocities=boundary.reference_v)
            duration = plan.end-now
            previous_query = actual_model.collision_query
            origin = workflow.collision_scene.origin
            if previous_query is not None and origin is not None:
                actual_model.collision_query = lambda offset: previous_query(now-float(origin)+float(offset))
            try:
                actual = gate.actual_gate.check_joints(robot, plan.q1, plan.v1, duration)
                reference = gate.reference_gate.check_joints(reference_robot, plan.q1, plan.v1, duration)
            finally:
                actual_model.collision_query = previous_query
            if not (actual.feasible and reference.feasible):
                return dict(command='hold', reason='global_actual_or_continuous_reference_gate_rejected',
                    actual_reason=actual.reason, reference_reason=reference.reason)
            workflow.plan = JointTrajectory(now, plan.end, boundary.reference_q, boundary.reference_v, plan.q1, plan.v1)
            q, v = workflow.plan.at(now+boundary.env.ledger.dt)
            return dict(command='joint_trajectory', joints=q, opening=1.,
                reference_joints=q, reference_velocities=v, trajectory_end=plan.end,
                nominal_constraints_verified=True, actual_tracking_guaranteed=False,
                global_reference_proof_v197=dict(kind='global_confirmed_reference_boundary_Hermite',
                    actual_gate=actual.reason, reference_gate=reference.reason,
                    native_acceleration_cap=6., reference_acceleration_cap=4.,
                    original_servo_lead_extrapolation_removed=True,
                    original_absolute_arrival_deadline=plan.end,
                    dynamic_forecast_collision_times_rebased_to_current_public_time=True))
        workflow.joint_command = joint_command
        previous = workflow.step
        def step(robot, obj, remaining):
            boundary.supplied_decision(robot, obj, dict(command='planning'), workflow.dynamics)
            decision = previous(robot, obj, remaining)
            boundary.supplied_decision(robot, obj, decision, workflow.dynamics)
            return decision
        workflow.step = step
        return workflow
    evaluator.CandidateInterception = construct
    try: yield
    finally: evaluator.CandidateInterception = original
