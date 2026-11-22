"""Causal public features and separately privileged, posthoc contact labels.

No label function is called by the feature builder. An actual Native attachment
observation is a quantized supervision event, not a force-contact certificate.
Only later verified goal plus lifted hold in that same uninterrupted attachment
segment makes the realized contact pose an accepted demonstration.
"""
import numpy as np
from scipy.spatial.transform import Rotation

CHECKPOINT_SHA256 = 'ebec3d1b3a83ceb0d4fb3bc889f8b62376cb4581071f53de29e0b6bd71c162ef'
ROBOT_KEYS = {'timestamp', 'tcp_position', 'tcp_quaternion', 'q', 'dq', 'finger_open_amounts'}
PUBLIC_ROW_KEYS = {'index', 'result', 'public_robot', 'close_requested', 'rgbd'}


def finite(value, shape, name):
    out = np.asarray(value, dtype=np.float64)
    if out.shape != shape or not np.isfinite(out).all():
        raise ValueError('finite ' + name + ' with shape ' + str(shape) + ' required')
    return out


def rot6(matrix):
    return np.concatenate((matrix[..., :, 0], matrix[..., :, 1]), axis=-1)


def unit_quaternion(value, name):
    quat = finite(value, (4,), name)
    if not np.isclose(np.linalg.norm(quat), 1., atol=1e-4, rtol=0):
        raise ValueError('unit xyzw quaternion required')
    return Rotation.from_quat(quat)


def public_features(row, history):
    """Read only approved current/past public fields; output SI tensors.

    ``history`` contains the actual rows up to and including ``row``. It is
    never interpolated. File existence and NPZ timestamp checks belong to the
    dataset compiler; this function verifies the causal temporal contract.
    """
    if not isinstance(row, dict) or set(row) != PUBLIC_ROW_KEYS:
        raise ValueError('exact public observer record required, without label channels')
    r, robot = row['result'], row['public_robot']
    if not isinstance(robot, dict) or set(robot) != ROBOT_KEYS:
        raise ValueError('whitelisted public robot fields required')
    now = float(robot['timestamp'])
    if not np.isfinite(now) or now < 0 or abs(float(r['timestamp']) - now) > 1e-8:
        raise ValueError('same-time public record required')
    if (r.get('mode') != 'FREE' or r.get('available') is not True
            or r.get('free_forecast_available') is not True or row['close_requested'] is not False):
        raise ValueError('pre-close available FREE forecast required')
    for flag in ('Native_inputs_used', 'future_observations_used', 'observations_fabricated',
                 'forecaster_is_injected_test_double'):
        if r.get(flag) is not False:
            raise ValueError('public causal frozen forecaster provenance required: ' + flag)
    forecast = r['forecast']; metadata = forecast['metadata']
    if (metadata.get('checkpoint_sha256') != CHECKPOINT_SHA256
            or abs(float(metadata['observed_until']) - now) > 1e-8
            or forecast['object_ids'] != ['target']):
        raise ValueError('frozen target-only causal forecast required')
    n = metadata['history_frames']['target']
    if type(n) is not int or not 5 <= n <= 20 or len(history) < n:
        raise ValueError('5..20 actual causal history frames required')
    selected = history[-n:]
    times = finite([v['public_robot']['timestamp'] for v in selected], (n,), 'history times')
    if abs(times[-1] - now) > 1e-8 or not np.allclose(np.diff(times), .05, atol=1e-6, rtol=0):
        raise ValueError('real contiguous 50ms history ending at anchor required')
    for item, timestamp in zip(selected, times):
        obj = item['result'].get('object_observation')
        if (not isinstance(obj, dict) or obj.get('object_id') != 'target'
                or not obj.get('visible', True) or float(obj.get('confidence', 1.)) < .5
                or abs(float(obj['timestamp']) - timestamp) > 1e-8 or not item.get('rgbd')):
            raise ValueError('actual visible timestamped public RGBD history required')
    expected = now + np.arange(17) * .05
    if not np.allclose(finite(forecast['times'], (17,), 'forecast times'), expected, atol=1e-8, rtol=0):
        raise ValueError('current plus16 frozen future samples required')
    p = finite(forecast['positions'], (17, 1, 3), 'future positions')[:, 0]
    matrices = finite(forecast['rotations'], (17, 1, 3, 3), 'future rotations')[:, 0]
    if (not np.allclose(matrices.transpose(0, 2, 1) @ matrices, np.eye(3), atol=1e-5, rtol=0)
            or not np.allclose(np.linalg.det(matrices), 1., atol=1e-5, rtol=0)):
        raise ValueError('proper forecast rotations required')
    v = finite(forecast['linear_velocities'], (17, 1, 3), 'future velocity')[:, 0]
    w = finite(forecast['angular_velocities'], (17, 1, 3), 'future angular velocity')[:, 0]
    obj = r['object_observation']
    current = finite(obj['pose']['position'], (3,), 'current public body')
    if not np.allclose(current, p[0], atol=1e-6, rtol=0):
        raise ValueError('forecast current state differs from real public observation')
    points = np.asarray(obj['points_world'], dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 3 or not np.isfinite(points).all():
        raise ValueError('public geometry points required')
    c = finite(metadata['contexts']['target'], (16,), 'motion context')
    if np.any(c < 0) or not np.allclose(c.reshape(2, 8).sum(1), 1., atol=1e-5, rtol=0):
        raise ValueError('frozen SimNorm motion context required')
    q = finite(robot['q'], (7,), 'robot q'); dq = finite(robot['dq'], (7,), 'robot dq')
    tcp = finite(robot['tcp_position'], (3,), 'public TCP')
    tcp_rotation = unit_quaternion(robot['tcp_quaternion'], 'TCP quaternion').as_matrix()
    fingers = finite(robot['finger_open_amounts'], (2,), 'two fingers')
    if np.any((fingers < 0) | (fingers > 1)):
        raise ValueError('normalized actual two-finger sensors required')
    features = dict(
        future_tokens=np.concatenate(((expected-now)[:, None]/.8, p-current, rot6(matrices), v, w), axis=1),
        robot=np.r_[q, dq, tcp-current, rot6(tcp_rotation), fingers],
        geometry=np.r_[np.ptp(points, axis=0), float(obj.get('confidence', 1.))],
        context_c=c,
        # One semantic task cannot train a task classifier. This explicit
        # constant is a pilot placeholder, distinct from the motion context.
        task_e=np.full(16, 1./8),
    )
    if (not all(np.isfinite(value).all() for value in features.values())
            or not .5 <= features['geometry'][-1] <= 1.):
        raise ValueError('finite public features and valid confidence required')
    return {k: v.astype(np.float32) for k, v in features.items()}, dict(
        timestamp=now, real_history_timestamps=times.tolist(),
        real_history_rgbd=[item['rgbd'] for item in selected],
        checkpoint_sha256=CHECKPOINT_SHA256, task_e_source='fixed_uniform_single_task_pilot_not_task_classifier',
        geometry_source='public_tracker_proxy_points_not_observed_full_shape', orientation_observed=False)


def successful_contacts(initial, ticks):
    """Offline supervision only, with later evidence confined to one attempt."""
    from ..local_mpc_20261102.records import HoldMetric
    if initial['held']:
        raise ValueError('initially free teacher episode required')
    previous = float(initial['time']); segments = []; active = []
    for tick in ticks:
        now = float(tick['time'])
        if not np.isfinite(now) or abs(now-previous-.05) > 1e-6:
            raise ValueError('complete actual 50ms label stream required')
        previous = now
        if tick['held']:
            active.append(tick)
        elif active:
            segments.append(active); active = []
    if active:
        segments.append(active)
    accepted = []
    for segment in segments:
        metric = HoldMetric(seconds=.5, height=.04, max_relative_drift=.02)
        metric.update(initial)
        for tick in segment:
            metric.update(tick)
        if not metric.passed or not any(t['native_success'] for t in segment):
            continue
        first = segment[0]
        object_rotation = unit_quaternion(first['object_quaternion'], 'label object quaternion')
        tcp_rotation = unit_quaternion(first['tcp_quaternion'], 'label TCP quaternion')
        delta = finite(first['tcp_position'], (3,), 'label TCP') - finite(first['object_position'], (3,), 'label body')
        accepted.append(dict(
            tau=float(first['time']), object_tcp_position=object_rotation.inv().apply(delta),
            object_tcp_rotation=(object_rotation.inv()*tcp_rotation).as_matrix(),
            acceptance='later_original_goal_and_lifted_halfsecond_hold_in_same_uninterrupted_attachment_segment',
            label_event='first_actual_50ms_Native_attachment_observation_not_exact_force_contact',
            segment_end=float(segment[-1]['time']), hold=metric.receipt()))
    return accepted
