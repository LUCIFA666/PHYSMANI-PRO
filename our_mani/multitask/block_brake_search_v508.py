"""Search collision-checked stopping endpoints only after the old brake fails.

Keep the previously functional global/local controller, its full native6 and
reference4 checks, and its exact next reference collision check. A servo target
is not asserted to be the actual robot position one tick later. Actual motor
motion remains an independently measured outcome.
"""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import numpy as np
from ..intercept_planner import JointTrajectory
from .block_fresh_tail_v448 import SmoothBlockEnvironment as OriginalBoundary
from .block_clip_brake_v315 import public_decision_scope
from .evaluation import file_sha

SOURCES = ('our_mani/multitask/block_brake_search_v508.py',
           'our_mani/multitask/block_fresh_tail_v448.py',
           'our_mani/multitask/block_clip_brake_v315.py',
           'our_mani/multitask/block_reference_v197.py')


def binding(root):
    return dict(schema='r10_original_block_control_additional_stopping_endpoints_v508',
        sources_sha256={p:file_sha(Path(root)/p) for p in SOURCES},
        original448_global_and_local_control_retained=True,
        additional_search_only_after_original_dense_stop_rejection=True,
        extra455_473_484_488_constraints_not_installed=True,
        native6_reference4_full_paths_and_original_exact_collision_retained=True,
        reference_target_not_assumed_reached_by_actual_robot_in_one_tick=True,
        no_extra_physics_or_camera_queries=True, original10s_budget_retained=True,
        native_motion_guaranteed=False, formal_VAL_TEST=False)


def endpoint_interval(q, v, acceleration, duration):
    """Exact endpoint interval for both endpoint accelerations of a cubic stop."""
    # A cubic's acceleration is affine in time, so its absolute maximum is at
    # one of its endpoints. Position/speed extrema and collisions are checked
    # separately by the complete existing TimedReachability gate.
    scale = np.asarray(acceleration)*duration**2
    left = q + np.maximum(4*v*duration-scale, 2*v*duration-scale)/6
    right = q + np.minimum(4*v*duration+scale, 2*v*duration+scale)/6
    return left, right


def stopping_endpoints(actual_gate, reference_gate, robot, qref, vref, duration):
    lower, upper = endpoint_interval(robot.joints, robot.joint_velocities,
                                    actual_gate.model.acceleration_limits, duration)
    rlower, rupper = endpoint_interval(qref, vref,
                                     reference_gate.model.acceleration_limits, duration)
    lower = np.maximum.reduce([lower, rlower, actual_gate.model.lower, reference_gate.model.lower])
    upper = np.minimum.reduce([upper, rupper, actual_gate.model.upper, reference_gate.model.upper])
    if np.any(lower > upper):
        return []
    nominal = np.clip(qref + .5*vref*duration, lower, upper)
    center = (lower+upper)/2
    candidates = [nominal, center,
                  np.clip(robot.joints+.5*robot.joint_velocities*duration, lower, upper)]
    for anchor in (nominal, center):
        for joint in range(len(qref)):
            for bound in (lower, upper):
                endpoint = anchor.copy()
                # Stay inside the analytic interval. The full original gates
                # remain authoritative, including all speed/position extrema.
                endpoint[joint] = (anchor[joint]+bound[joint])/2
                candidates.append(endpoint)
    unique = []
    for q in candidates:
        if not any(np.allclose(q, p, rtol=0, atol=1e-10) for p in unique):
            unique.append(q)
    return unique


def find_stop(actual_gate, reference_gate, robot, qref, vref, durations, *, dt, deadline,
              command_velocity_limits):
    if (abs(dt-.05)>1e-9 or deadline>10.+1e-9
            or not np.allclose(actual_gate.model.acceleration_limits, 6.)
            or not np.allclose(reference_gate.model.acceleration_limits, 4.)):
        raise ValueError('retain original50ms/10s/actual6/reference4 contract')
    now = float(robot.timestamp)
    reference = replace(robot, joints=np.asarray(qref), joint_velocities=np.asarray(vref))
    zero = np.zeros_like(qref)
    trials = []
    for duration in durations:
        if duration < dt-1e-9 or duration > 1.1+1e-9 or now+duration > deadline+1e-9:
            continue
        for endpoint in stopping_endpoints(actual_gate, reference_gate, robot, qref, vref, duration):
            ref = reference_gate.check_joints(reference, endpoint, zero, duration)
            if not ref.feasible:
                trials.append(dict(duration=duration, reference=ref.reason))
                continue
            actual = actual_gate.check_joints(robot, endpoint, zero, duration)
            if not actual.feasible:
                trials.append(dict(duration=duration, reference=ref.reason, actual=actual.reason))
                continue
            plan = JointTrajectory(now, now+duration, np.asarray(qref).copy(),
                                   np.asarray(vref).copy(), endpoint.copy(), zero.copy())
            q, _ = plan.at(now+dt)
            unclipped = np.all(abs(q-robot.joints) <= command_velocity_limits*dt+1e-7)
            exact = unclipped and not actual_gate.model.collision(q, dt)
            trials.append(dict(duration=duration, reference=ref.reason, actual=actual.reason,
                               original_actuator_unclipped=bool(unclipped), exact_next_collision_clear=bool(exact)))
            if exact:
                return plan, dict(trials=trials, terminal_velocity_zero=True,
                    full_reference4_and_actual6_stop_proved=True,
                    actual_reaches_next_reference_target_not_assumed=True)
    return None, dict(trials=trials, no_full_two_branch_collision_free_stop_found=True)


class SmoothBlockEnvironment(OriginalBoundary):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stopping_search_active = False
        self.stopping_search_events = []

    def _stationary_existing_target(self, reason):
        if reason != 'no_fresh_actual_reference_dense_block_brake' or self.stopping_search_active:
            return super()._stationary_existing_target(reason)
        robot = self.public_robot
        if robot is None or abs(robot.timestamp-self.env.time) > 1e-8:
            return super()._stationary_existing_target(reason)
        self.stopping_search_active = True
        try:
            actual, reference = self._gates()
            # The parent has already tried the unchanged straight stops. This
            # search uses the same current public collision scope and horizon.
            durations = (.1, .15, .2, .25, .3, .4, .5, .6, .8, 1., 1.1)
            plan, proof = find_stop(actual, reference, robot, self.reference_q, self.reference_v,
                durations, dt=float(self.env.ledger.dt), deadline=10.,
                command_velocity_limits=self.env.robot_model.velocity_limits)
            self.stopping_search_events.append(dict(time=float(self.env.time),
                original_reason=reason, admitted=plan is not None, proof=proof))
            if plan is None:
                return super()._stationary_existing_target(reason)
            self.backup = plan
            q, v = plan.at(robot.timestamp+self.env.ledger.dt)
            return self._issue(q, v, self.env._requested_opening,
                dict(kind='additional_full_two_branch_stopping_endpoint_search', proof=proof))
        finally:
            self.stopping_search_active = False

    def receipt(self):
        return dict(super().receipt(), block_brake_search_v508=dict(
            binding=binding(Path(__file__).resolve().parents[2]),
            events=deepcopy(self.stopping_search_events)))
