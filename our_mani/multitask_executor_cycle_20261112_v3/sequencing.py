"""Correct native-equivalent RML previews without carrying state across trials."""
import numpy as np
from ..multitask_executor_cycle_20261112_v1.sequencing import (
    SequenceConfig, arrival_error, requested_opening_after_arrival, retime_reference_path)


def interpolate(points, lengths, distance):
    index = min(max(int(np.searchsorted(lengths, distance, side='right'))-1, 0), len(points)-2)
    span = lengths[index+1]-lengths[index]
    if span <= 0:
        raise ValueError('Strictly increasing nonzero joint-polyline lengths required')
    fraction = (distance-lengths[index])/span
    return points[index] + (points[index+1]-points[index])*fraction


def sample_velocity_checked_rml(points, native_velocity_limits, mechanical_limits, sim, *, dt=.05):
    points = np.asarray(points, float)
    limits = np.asarray(native_velocity_limits, float)
    if (points.ndim != 2 or points.shape[1] != 7 or len(points) < 2
            or limits.shape != (7,) or not np.isfinite(points).all()
            or not np.isfinite(limits).all() or np.any(limits <= 0)):
        raise ValueError('Finite seven-joint path and original positive native velocities required')
    keep = np.r_[True, np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-12]
    points = points[keep]
    if len(points) < 2:
        return [points[0].copy()], dict(velocity_trials=[], zero_length_path=True)
    lengths = np.r_[0., np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    correction, trials = 1., []
    max_velocity, max_acceleration, max_jerk = mechanical_limits
    for trial in range(12):
        # Critical correction: each independent preview starts at the path's
        # beginning, never at the previous preview's terminal joint position.
        previous_q = points[0].copy()
        samples, maximum_ratio = [], 0.
        handle = sim.simRMLPos(1, .0001, -1, [0., 0., 0.],
                               [max_velocity*correction, max_acceleration, max_jerk], [1], [lengths[-1], 0.])
        try:
            for _ in range(10000):
                state, result = sim.simRMLStep(handle, dt, 1)
                if state < 0:
                    raise RuntimeError('Original RML primitive failed')
                q = interpolate(points, lengths, float(result[0]))
                ratio = float(np.max(np.abs((q-previous_q)/dt)/limits))
                maximum_ratio = max(maximum_ratio, ratio)
                previous_q = q
                samples.append(q.copy())
                if state == 1:
                    break
            else:
                raise RuntimeError('RML primitive exceeded original10000-step guard')
        finally:
            sim.simRMLRemove(handle)
        trials.append(dict(trial=trial, correction=correction, max_native_velocity_ratio=maximum_ratio,
                           preview_starts_at_original_first_point=True, samples=len(samples)))
        if maximum_ratio <= 1.001:
            return samples, dict(velocity_trials=trials, Native_mechanical_parameters_unchanged=True,
                                  preview_state_reset_between_trials=True, offline_physics_steps=0)
        correction /= maximum_ratio
    raise RuntimeError('Native velocity preview failed to converge in12 independent trials')


def sample_native_rml_path(backend, reference, config):
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
    samples, audit = sample_velocity_checked_rml(points, backend.arm.get_joint_upper_velocity_limits(),
                                                 (backend.arm.max_velocity, backend.arm.max_acceleration,
                                                  backend.arm.max_jerk), sim)
    backend.last_rml_sampling_audit = audit
    if backend._clock_tuple() != before:
        raise RuntimeError('Offline RML sampling advanced task/physics time')
    return samples
