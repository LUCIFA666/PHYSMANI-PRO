"""Public-observation partial approach, with independent complete motion proofs.

The static-target Hermite time estimate only triggers approach. It is not a
reachability certificate or a prediction outside the causal forecast horizon.
Each emitted segment still passes actual-state and confirmed-reference gates.
"""
from copy import deepcopy
from dataclasses import dataclass, replace
import numpy as np
from scipy.spatial.transform import Rotation

from ..contracts import Pose
from ..intercept_planner import JointTrajectory
from ..reachability import TimedReachability, hermite
from .evaluation import digest
from .interaction import _reference_at, _world_target

SCHEMA = 'r10_bounded_3D_transport_confirmed_phase_clock_v159'


@dataclass(frozen=True)
class ApproachConfig:
    trigger_seconds: float = 1.1
    segment_seconds: float = .5
    maximum_TCP_step_m: float = .08
    minimum_error_reduction: float = .001
    orientation_weight_m_per_rad: float = .05

    def __post_init__(self):
        values = tuple(self.__dict__.values())
        if (any(isinstance(v, bool) for v in values) or not np.isfinite(values).all()
                or not .05 <= self.segment_seconds <= self.trigger_seconds <= 1.1
                or not 0 < self.minimum_error_reduction < self.maximum_TCP_step_m <= .1
                or not 0 < self.orientation_weight_m_per_rad <= .1):
            raise ValueError('invalid bounded approach configuration')


def configuration(raw=None):
    raw = dict(raw or {})
    if raw.pop('version', SCHEMA) != SCHEMA:
        raise ValueError('wrong bounded approach version')
    config = ApproachConfig(**raw)
    return dict(version=SCHEMA, **config.__dict__)


def static_target_time_estimate(robot, target_q, model, threshold):
    """First admissible nominal Hermite duration on a declared bounded grid.

    This pure estimate does not query or waive collision checks. A real segment
    is authorized separately by the unchanged complete TimedReachability gates.
    """
    q = np.asarray(target_q, float)
    if (q.shape != robot.joints.shape or not np.isfinite(q).all()
            or np.any(q < model.lower) or np.any(q > model.upper)):
        return None
    durations = sorted(set([.05, .15, .3, .5, .7, .9, float(threshold),
                            1.25, 1.5, 2., 2.5, 3., 4., 6., 8., 10.]))
    for duration in durations:
        path, speed, acceleration = hermite(robot.joints, robot.joint_velocities,
            q, np.zeros_like(q), duration, 21)
        if (np.all(path >= model.lower - 1e-7) and np.all(path <= model.upper + 1e-7)
                and np.all(np.abs(speed) <= model.velocity_limits + 1e-7)
                and np.all(np.abs(acceleration) <= model.acceleration_limits + 1e-7)):
            return float(duration)
    return None


def pose_error(pose, target, weight):
    return float(np.linalg.norm(pose.position - target.position)
                 + weight * Rotation.from_matrix(pose.rotation @ target.rotation.T).magnitude())


def transit_waypoint(current,target,maximum_step):
    """Allow descent within the same bounded, fully checked 3D segment."""
    return Pose(target.position.copy(),target.rotation.copy())


class OffsetCollisionModel:
    def __init__(self,model,offset):self.model,self.offset=model,float(offset)
    def __getattr__(self,name):return getattr(self.model,name)
    def collision(self,joints,future_offset):return self.model.collision(joints,self.offset+float(future_offset))


def check_offset_brake(gate,endpoint,offset,duration):
    model=OffsetCollisionModel(gate.model,offset)
    shifted=TimedReachability(model,max_joint_step=gate.max_joint_step,minimum_samples=gate.minimum_samples,
        twist_tolerance=gate.twist_tolerance,position_tolerance=gate.position_tolerance,orientation_tolerance=gate.orientation_tolerance)
    q=endpoint.joints+.5*endpoint.joint_velocities*duration
    return shifted.check_joints(endpoint,q,np.zeros_like(q),duration)


class BoundedApproach:
    def __init__(self, manager, raw=None):
        self.manager = manager
        normalized = configuration(raw)
        self.config = ApproachConfig(**{k: v for k, v in normalized.items() if k != 'version'})
        self.normalized = normalized
        self.events = []
        self.deadlines = {}
        self.active = None

    def _bind(self, state, forecast, phase, reference_id, manipulated, arrival):
        self.manager.controller.generator.bind_candidate_scene(
            state, forecast, phase, reference_id, manipulated, arrival)

    def _remaining_proof(self, robot, state, forecast):
        m, active = self.manager, self.active
        duration = m.plan.end - robot.timestamp
        self._bind(state, forecast, active['phase'], active['reference_id'],
                   active['manipulated_ids'], m.plan.end)
        boundary = m.continuity_state(robot)
        if boundary is None or duration < m.config.dt - 1e-8:
            return False
        q, v, _ = boundary
        actual = TimedReachability.check_joints(m.native_gate, robot, m.plan.q1, m.plan.v1, duration)
        reference = TimedReachability.check_joints(m.native_gate,
            replace(robot, joints=q, joint_velocities=v), m.plan.q1, m.plan.v1, duration)
        self.events.append(dict(time=float(robot.timestamp), event='remaining_segment_rechecked',
            actual_reason=actual.reason, reference_reason=reference.reason,
            actual_and_reference_full_gate_feasible=bool(actual.feasible and reference.feasible),
            deadline=active['deadline']))
        return bool(actual.feasible and reference.feasible)

    def try_approach(self, robot, remaining_seconds):
        m = self.manager
        c, state, forecast = m.controller, m.current_state, m.current_forecast
        now = float(robot.timestamp)
        if (c.goal is not None or c.phase_index >= len(c.library.phase_order)
                or state is None or forecast is None or m.qualification_failed
                or m.braking_active(now)):
            return False
        cutoff = forecast.metadata.get('observed_until')
        if (type(cutoff) not in (int, float) or not 0 <= now - cutoff <= c.generator.config.maximum_observation_age
                or abs(state.robot.timestamp - now) > 1e-8 or m.continuity_state(robot) is None):
            return False
        phase = c.library.phase_order[c.phase_index]
        if phase not in ('approach', 'acquire', 'lift', 'deposit','transport'):
            return False
        suffix = sum(max(min(c.generator.config.arrival_offsets_seconds),
                         max(t.duration_budget_seconds for t in c.library.for_phase(name))) + m.config.dt
                     for name in c.library.phase_order[c.phase_index + 1:])
        deadline = min(10., now + float(remaining_seconds)) - suffix
        retired = m.retired_brake_goal
        if retired is not None and retired.phase == phase:
            deadline = min(deadline, retired.selected_at + retired.phase_budget)
        deadline = min(deadline, self.deadlines.get(c.phase_index, deadline))
        if phase=='transport':deadline=m.transport_phase_clock.provisional_deadline(c.phase_index,now,deadline)
        if self.active is not None:
            same = self.active['phase_index'] == c.phase_index and m.plan_kind == 'approach_transit'
            if same and m.plan is not None and not m.invalidated and now < m.plan.end - 1e-8:
                if m.plan.end > deadline + 1e-8 or not self._remaining_proof(robot, state, forecast):
                    m.invalidate('partial_approach_remaining_proof_rejected', now)
                    return False
                check = m.check_stream(robot, m.config.dt)
                return bool(check is not None and check.feasible and m.pending is not None)
            if m.plan_kind == 'approach_transit':
                m.invalidate('partial_approach_segment_ended_or_phase_changed', now)
            self.active = None
        duration = self.config.segment_seconds
        if now + duration > deadline + 1e-8 or forecast.times[-1] < now + duration - 1e-8:
            return False
        boundary = m.continuity_state(robot)
        q0, v0, basis = boundary
        reference_robot = replace(robot, joints=q0, joint_velocities=v0)
        reference_pose = m.native_gate.model.forward(q0)
        bindings = {**state.role_bindings, **getattr(c, 'committed_roles', {})}
        trials = []
        for template in c.library.for_phase(phase):
            ids = ((template.reference_kind + ':' + phase,) if template.reference_kind in ('phase_origin', 'world')
                   else tuple(bindings.get(template.reference_role, ())))
            manipulated = tuple(bindings.get(template.manipulated_role, ()))
            for reference_id in ids:
                if template.manipulated_role == template.reference_role:
                    manipulated = (reference_id,)
                if len(manipulated) != 1:
                    continue
                try:
                    current = c.generator.observed_reference(state, reference_id, template, c.origin)
                    if template.reference_kind not in ('phase_origin', 'world'):
                        mode = state.perception_metadata.get('bodies', {}).get(reference_id, {}).get('motion_mode')
                        if mode != template.reference_mode or mode == 'attached':
                            continue
                    # Fitted positive lag consumes the same causal forecast horizon.
                    lookahead = min(self.config.segment_seconds, float(forecast.times[-1] - now) - template.lag_seconds)
                    if lookahead < m.config.dt - 1e-8:
                        continue
                    ref, velocity, angular = _reference_at(forecast, reference_id, template,
                        now + lookahead + template.lag_seconds, c.origin)
                    target = _world_target(ref, velocity, angular, template, 1.)[0]
                    no_complete_candidate = (
                        getattr(c.generator, 'last_diagnostics', {}).get('feasible_candidates') == 0)
                    goal_q = m.native_gate.model.inverse(target, robot.joints)
                    if goal_q is not None:
                        actual_goal = m.native_gate.model.forward(goal_q)
                        if (np.linalg.norm(actual_goal.position - target.position) > m.native_gate.position_tolerance
                                or Rotation.from_matrix(actual_goal.rotation @ target.rotation.T).magnitude() > m.native_gate.orientation_tolerance):
                            goal_q=None
                    estimates = ([static_target_time_estimate(r, goal_q, m.native_gate.model,
                                  self.config.trigger_seconds) for r in (robot, reference_robot)]
                                 if goal_q is not None else [None,None])
                    slow = all(t is not None for t in estimates) and max(estimates) > self.config.trigger_seconds + 1e-8
                    # A zero-terminal static ETA omits the fitted candidate's
                    # endpoint twist and curve progress. A fast ETA therefore
                    # cannot establish that the actual complete candidate is
                    # feasible. Only an actual empty candidate search permits
                    # this alternative trigger; both full motion proofs below
                    # remain mandatory and an existing full goal wins above.
                    if not slow and not no_complete_candidate:
                        continue
                    before = pose_error(reference_pose, target, self.config.orientation_weight_m_per_rad)
                    # A joint-space interpolation can initially move the TCP
                    # away from the target and sweep an elbow through the bin.
                    # Request a bounded Cartesian advance, solve local IK from
                    # the confirmed boundary, and retain both original complete
                    # joint-space collision/derivative checks for its execution.
                    waypoint=transit_waypoint(reference_pose,target,self.config.maximum_TCP_step_m)
                    delta = waypoint.position-reference_pose.position
                    rotation_delta = Rotation.from_matrix(target.rotation@reference_pose.rotation.T).as_rotvec()
                    waypoint_error=pose_error(reference_pose,waypoint,self.config.orientation_weight_m_per_rad)
                    fraction_limit=min(1.,self.config.maximum_TCP_step_m/max(waypoint_error,1e-12))
                    for scale in (1., .5, .25, .125, .0625, .03125, .015625):
                        fraction=fraction_limit*scale
                        desired=Pose(reference_pose.position+fraction*delta,
                            Rotation.from_rotvec(fraction*rotation_delta).as_matrix()@reference_pose.rotation)
                        nominal_q=m.native_gate.model.inverse(desired,q0)
                        if nominal_q is None:continue
                        solved=m.native_gate.model.forward(nominal_q)
                        if (np.linalg.norm(solved.position-desired.position)>m.native_gate.position_tolerance
                                or Rotation.from_matrix(solved.rotation@desired.rotation.T).magnitude()>m.native_gate.orientation_tolerance):
                            continue
                        admitted=None
                        for nominal_duration in (.5,.7,.9,1.1,1.5,2.,3.,4.):
                            q1,v1=JointTrajectory(now,now+nominal_duration,q0,v0,nominal_q,np.zeros_like(nominal_q)).at(now+duration)
                            end_pose=m.native_gate.model.forward(q1)
                            after=pose_error(end_pose,target,self.config.orientation_weight_m_per_rad)
                            if (np.linalg.norm(end_pose.position-reference_pose.position)>self.config.maximum_TCP_step_m
                                    or before-after<self.config.minimum_error_reduction):continue
                            self._bind(state,forecast,phase,reference_id,manipulated,now+duration)
                            actual=TimedReachability.check_joints(m.native_gate,robot,q1,v1,duration)
                            reference=TimedReachability.check_joints(m.native_gate,reference_robot,q1,v1,duration)
                            row=dict(fraction=fraction,nominal_duration=nominal_duration,actual_reason=actual.reason,reference_reason=reference.reason,
                                requested_Cartesian_xyzw=desired.xyzw().tolist(),local_IK_seed='confirmed_reference_joints')
                            trials.append(row)
                            if not (actual.feasible and reference.feasible):continue
                            brake_duration=max(.2,1.5*float(np.max(np.abs(v1)/m.native_gate.model.acceleration_limits)))
                            if (now+duration+brake_duration>min(deadline,float(forecast.times[-1]))+1e-8):
                                row['backup_brake_reason']='original_deadline_or_forecast_horizon';continue
                            self._bind(state,forecast,phase,reference_id,manipulated,now+duration+brake_duration)
                            endpoint=replace(robot,timestamp=now+duration,joints=q1,joint_velocities=v1)
                            backup=check_offset_brake(m.native_gate,endpoint,duration,brake_duration)
                            row['backup_brake_reason']=backup.reason
                            if not backup.feasible:continue
                            self._bind(state,forecast,phase,reference_id,manipulated,now+duration)
                            admitted=(q1,v1,end_pose,nominal_duration,backup.evidence);break
                        if admitted is None:continue
                        q1,v1,end_pose,nominal_duration,backup_evidence=admitted
                        collision = c.generator.collision_check(pose=end_pose, time=now + duration,
                            phase=phase, observed_bodies=state.bodies, forecast=forecast, manipulated_ids=manipulated)
                        if collision.get('feasible') is not True:
                            continue
                        plan_id = 'partial_approach:' + digest(dict(time=now, phase=phase,
                            reference_id=reference_id, q=q1.tolist()))[:20]
                        m.plan = JointTrajectory(now, now + duration, q0.copy(), v0.copy(), q1, v1)
                        m.plan_kind, m.plan_candidate_id, m.invalidated = 'approach_transit', plan_id, False
                        self.deadlines[c.phase_index] = deadline
                        self.active = dict(phase=phase, phase_index=c.phase_index, reference_id=reference_id,
                                           manipulated_ids=manipulated, deadline=deadline)
                        self.events.append(dict(time=now, event='partial_approach_staged', phase=phase,
                            plan_id=plan_id, trigger_seconds=self.config.trigger_seconds,
                            static_target_Hermite_time_estimates_seconds=estimates,
                            time_estimate_is_not_collision_certificate=True, segment_seconds=duration,
                            trigger_reason='slow_static_target' if slow else 'actual_complete_candidate_search_empty',
                            complete_candidate_search=deepcopy(getattr(c.generator, 'last_diagnostics', {})),
                            target_xyzw=target.xyzw().tolist(), end_xyzw=end_pose.xyzw().tolist(),
                            predicted_reference_time=now + lookahead + template.lag_seconds,
                            approach_path_basis='bounded_Cartesian_transit_prefix_and_offset_backup_brake_with_original_dual_gates',
                            nominal_duration=nominal_duration,endpoint_velocity=v1.tolist(),backup_brake_evidence=backup_evidence,
                            source_observed_at=float(current.timestamp), boundary_basis=basis,
                            full_gate_trials=trials, actual_and_reference_full_gate_feasible=True,
                            endpoint_collision_gate=deepcopy(collision), deadline=deadline,
                            no_task_candidate_or_completion_claimed=True, gripper_intent_preserved=True,
                            phase_clock_reset=False, total_budget_extended=False))
                        check = m.check_stream(robot, m.config.dt)
                        if check is not None and check.feasible and m.pending is not None:
                            return True
                        m.invalidate('partial_approach_first_tick_rejected', now)
                        self.active = None
                        return False
                except (ValueError, KeyError, IndexError):
                    continue
        self.events.append(dict(time=now, event='no_safe_slow_target_approach', phase=phase, trials=trials))
        return False

    def metrics(self):
        return dict(schema=SCHEMA, normalized_config=self.normalized, events=deepcopy(self.events),
                    deadline_by_phase_index=dict(self.deadlines), stage_completion_claimed=False,
                    extra_physics_or_camera_calls=False, ETA_used_as_safety_certificate=False)
