"""World-intent to counted arm/then-gripper sequencing, using robot feedback.

No two-tick completion, RML completion or predicted pose is treated as measured
arm arrival, grasp, or task success. All tasks share these mechanical checks.
"""
from dataclasses import dataclass
import math
import numpy as np
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class SequenceConfig:
    arrival_position_m: float = .01
    arrival_rotation_rad: float = .1
    max_settle_seconds: float = 1.
    ignore_collisions: bool = True  # original EndEffectorPoseViaPlanning default

    def __post_init__(self):
        if any(isinstance(v, bool) or not math.isfinite(v) or v <= 0 for v in
               (self.arrival_position_m, self.arrival_rotation_rad,
                self.max_settle_seconds)):
            raise ValueError('Positive finite sequencing bounds required')


def arrival_error(reference, measured_tcp_pose):
    pose = np.asarray(measured_tcp_pose, dtype=float)
    if pose.shape != (7,) or not np.isfinite(pose).all() or abs(np.linalg.norm(pose[3:]) - 1) > 1e-5:
        raise ValueError('Current measured public TCP position/unit quaternion required')
    return (float(np.linalg.norm(reference.position - pose[:3])),
            float((Rotation.from_quat(reference.quaternion).inv() * Rotation.from_quat(pose[3:])).magnitude()))


def requested_opening_after_arrival(reference, measured_tcp_pose, retained_opening, config):
    position, angle = arrival_error(reference, measured_tcp_pose)
    arrived = position <= config.arrival_position_m and angle <= config.arrival_rotation_rad
    opening = float(reference.opening_intent > .5) if arrived else float(retained_opening)
    return opening, dict(measured_arm_arrived=arrived, position_error_m=position,
                         rotation_error_rad=angle, future_opening_intent=reference.opening_intent,
                         gripper_intent_authorized_by='measured_TCP_arrival' if arrived else None,
                         task_success_inferred=False, object_alignment_proven=False)


def sample_native_rml_path(backend, reference, config):
    """Original RRT/RML joint samples, without motor writes or physics steps.

    Uses original RML scalar parameterization/interpolation; does not reinterpret
    a sampled reference as a smoothed joint destination. The executor checks
    reference differences and transmits samples unchanged at actual50ms.
    """
    from pyrep.backend import sim
    from pyrep.const import ConfigurationPathAlgorithms as Algos
    before = backend._clock_tuple()
    actual = np.asarray(backend.arm.get_joint_positions(), float)
    servo = np.asarray(backend.arm.get_joint_target_positions(), float)
    ik_pose = backend.arm._ik_target.get_pose()
    try:
        path = backend.arm.get_path(reference.position, quaternion=reference.quaternion,
                                    ignore_collisions=config.ignore_collisions, trials=100,
                                    max_configs=10, max_time_ms=10, trials_per_goal=5,
                                    algorithm=Algos.RRTConnect)
    finally:
        backend.arm._ik_target.set_pose(ik_pose)
        backend.arm.set_joint_target_positions(servo)
        if (backend._clock_tuple() != before or
                not np.allclose(backend.arm.get_joint_positions(), actual, atol=1e-6, rtol=0)):
            raise RuntimeError('Native path planning changed actual robot state/clock')
    points = np.asarray(path._path_points, float).reshape(-1, 7)
    if len(points) < 2:
        return [actual.copy()]
    lengths = np.asarray(path._get_path_point_lengths(), float)
    if lengths[-1] < 1e-10:
        return [actual.copy()]
    handle = path._get_rml_handle()
    samples = []
    try:
        for _ in range(10000):
            state, position_velocity_acceleration = sim.simRMLStep(handle, .05, 1)
            if state < 0:
                raise RuntimeError('Native RML trajectory generation failed: ' + str(state))
            s = float(position_velocity_acceleration[0])
            index = min(max(int(np.searchsorted(lengths, s, side='right')) - 1, 0), len(points) - 2)
            while index < len(points) - 2 and lengths[index + 1] - lengths[index] < 1e-12:
                index += 1
            span = lengths[index + 1] - lengths[index]
            fraction = 1. if span < 1e-12 else (s - lengths[index]) / span
            samples.append(points[index] + (points[index + 1] - points[index]) * fraction)
            if state == 1:
                break
        else:
            raise RuntimeError('Native RML did not finish within original10000-step guard')
    finally:
        sim.simRMLRemove(handle)
    if backend._clock_tuple() != before:
        raise RuntimeError('Offline RML sampling advanced task/physics time')
    return samples


def retime_reference_path(points, executor, measured_q, measured_dq, now):
    """Explicit uniform time dilation of RML samples; never loosen limits.

    Linear interpolation stays on the sampled joint-space polyline. Two final
    hold samples include stopping in the preflight. Native RML's velocity-only
    audit is not a certificate for the executor's discrete acceleration/jerk.
    """
    from types import SimpleNamespace
    from ..local_mpc_20261102.executor import ReferenceLimitError
    raw = np.asarray(points, float)
    if raw.ndim != 2 or raw.shape[1] != 7 or len(raw) < 1 or not np.isfinite(raw).all():
        raise ValueError('Finite original sampled seven-joint path required')
    # Continue the actual persistent command reference. Actual measured q is
    # included as the connector to the original RRT path, not teleported.
    previous = executor._last_direct
    initial = np.asarray(previous[1] if previous is not None else measured_q)
    vertices = np.vstack((initial, np.asarray(measured_q), raw))
    times = np.arange(len(vertices), dtype=float)
    scale = 1.
    for _ in range(24):
        count = max(1, int(np.ceil(times[-1] * scale)))
        locations = np.arange(1, count + 1) * times[-1] / count
        candidate = np.stack([np.interp(locations, times, vertices[:, j]) for j in range(7)], axis=-1)
        candidate = np.vstack((candidate, [raw[-1], raw[-1]]))
        try:
            executor._trajectory_references(SimpleNamespace(positions=candidate), measured_q, measured_dq, now)
        except ReferenceLimitError:
            scale *= 1.25
            continue
        return list(candidate), dict(original_RML_samples=len(raw), retimed_samples=len(candidate),
                                     time_dilation=scale, geometric_polyline_preserved=True,
                                     final_stopping_preflight=True, limits_relaxed=False,
                                     actual_q_connector_included=True, offline_physics_steps=0)
    raise ReferenceLimitError('No bounded retiming of model RRT/RML path found')
