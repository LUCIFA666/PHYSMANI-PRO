"""TRAIN observed-pose histories under the old recorded-frame-time assumption.

Legacy cache times are frame_index*.05, not native HDF clock observations.
They are inherited for offline teacher-context pretraining only. Every cache
time array is checked; all invalid observations/parent exclusions are reported.
"""
from dataclasses import dataclass
from pathlib import Path
from collections import Counter
import hashlib
import json

import numpy as np
import torch

from ..multitask.context_se3 import finite_difference_twist

ROOT = Path(__file__).resolve().parents[2]
TASKS = ('pick_moving_target_on_the_table', 'pick_moving_target_on_the_table_high_speed')
REGISTRY_SCHEMA = 'context_gap_dropout_Cube_TRAIN_parent_partition_20261103_v1'
TEACHER_SOURCES = ROOT/'artifacts/framework/multitask_r10/context_joint16_after_executor_v671/training_sources.json'
MAXIMUM_FRAMES = 20
MAXIMUM_AGE = 1.
MAXIMUM_GAP = .25


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def source_key(entry):
    return (entry['task'], str(entry.get('parent_raw_episode_id', entry['raw_episode_id'])))


@dataclass(frozen=True)
class Window:
    parent: tuple
    role: str
    anchor: int
    times: np.ndarray
    positions: np.ndarray
    rotations: np.ndarray
    velocities: np.ndarray
    angular_velocities: np.ndarray
    confidence: np.ndarray
    observable: np.ndarray
    future_positions: np.ndarray
    future_rotations: np.ndarray
    future_valid: np.ndarray


def load_registry(path):
    path = Path(path).resolve(strict=True)
    registry = json.loads(path.read_text())
    if registry.get('schema') != REGISTRY_SCHEMA or registry.get('VAL_or_TEST_read') is not False:
        raise ValueError('Exact TRAIN-only frozen partition registry required')
    reference = json.loads(TEACHER_SOURCES.read_text())['entries']
    original = {source_key(e): e for e in reference if e['task'] in TASKS}
    rows = registry.get('rows')
    if not isinstance(rows, list) or len(rows) != len(original) or len(original) != 200:
        raise ValueError('All original200 Cube TRAIN parents must remain in registry, including exclusions')
    seen, roles = set(), Counter()
    for row in rows:
        entry = row.get('entry')
        if not isinstance(entry, dict) or entry.get('split') != 'train' or entry.get('task') not in TASKS:
            raise ValueError('No VAL/TEST or unrelated task arrays may enter this student')
        key = source_key(entry)
        if (key in seen or key not in original or canonical(entry) != canonical(original[key])
                or row.get('task') != key[0] or str(row.get('parent_raw_episode_id')) != key[1]
                or row.get('role') not in ('fit', 'selection', 'reserved_fresh_centre_VAL', 'excluded_consumed_calibration')):
            raise ValueError('Exact unique teacher TRAIN parent/whole-entry/role binding required')
        seen.add(key)
        roles[(key[0], row['role'])] += 1
    if seen != set(original) or any(roles[(task, role)] == 0 for task in TASKS for role in ('fit', 'selection')):
        raise ValueError('Both tasks need independent nonempty fit and selection parent groups')
    expected_roles = dict(fit=140, selection=16, reserved_fresh_centre_VAL=8, excluded_consumed_calibration=36)
    actual_roles = dict(Counter(row['role'] for row in rows))
    hdf_groups = [str(Path(row['source_HDF5']['path']).resolve()) for row in rows]
    if (registry.get('complete') is not True or registry.get('all200_HDF_and_teacher_cache_byte_hashes_verified') is not True
            or actual_roles != expected_roles or registry.get('counts') != expected_roles
            or len(set(hdf_groups)) != 200
            or any(row.get('teacher_exposed') is not True
                   or str(Path(row['entry']['recorded_transition_labels']['path']).resolve()) != group
                   for row, group in zip(rows, hdf_groups))):
        raise ValueError('Predeclared original-source/calibration overlap audit required')
    return registry


def load_windows(registry, *, split='train'):
    if split not in ('train', 'validation'):
        raise ValueError('TEST tracks are never permitted')
    windows, ledger, files = [], [], {}
    for row in registry['rows']:
        e, role = row['entry'], row['role']
        if e.get('split') != split or e.get('task') not in TASKS:
            raise ValueError('Exact declared original task/source split required')
        path = (ROOT/e['cache']).resolve(strict=True)
        digest = sha(path)
        if digest != e['cache_sha256']:
            raise ValueError('Original TRAIN cache changed')
        files[str(path)] = digest
        hdf = row.get('source_HDF5')
        if hdf is not None:
            if sha(hdf['path']) != hdf['sha256']:
                raise ValueError('Original source-episode HDF byte identity changed')
            files[str(Path(hdf['path']).resolve())] = hdf['sha256']
        if role in ('excluded_consumed_calibration', 'reserved_fresh_centre_VAL'):
            ledger.append(dict(parent=list(source_key(e)), role=role, decoded=False, reason='whole_parent_exclusion'))
            continue
        with np.load(path, allow_pickle=False) as stored:
            required = ('times', 'body_ids', 'positions', 'rotations', 'linear_velocities', 'angular_velocities',
                        'confidence', 'valid', 'modes', 'rotation_observable')
            if any(name not in stored for name in required):
                raise ValueError('Incomplete causal body track')
            a = {name: np.array(stored[name], copy=True) for name in required}
        t = a['times']
        if (t.ndim != 1 or len(t) < 5 or len(a['body_ids']) != 1 or str(a['body_ids'][0]) != 'target'
                or not np.isfinite(t).all() or np.any(np.diff(t) <= 0)
                or not np.allclose(np.diff(t), .05, atol=1e-6, rtol=0)):
            raise ValueError('Exact legacy recorded-frame-time Cube cache required; no physical50ms claim')
        valid, modes = a['valid'][:, 0], a['modes'][:, 0]
        if valid.dtype != np.bool_ or a['rotation_observable'].dtype != np.bool_:
            raise ValueError('Original validity/observability must remain boolean')
        for name, shape in (('positions', (len(t), 1, 3)), ('rotations', (len(t), 1, 3, 3)),
                            ('linear_velocities', (len(t), 1, 3)), ('angular_velocities', (len(t), 1, 3))):
            if a[name].shape != shape or not np.isfinite(a[name][valid]).all():
                raise ValueError('Finite valid original pose/twist required')
        parent_windows, skipped = 0, Counter()
        for anchor in range(len(t)):
            if not valid[anchor] or modes[anchor] != 0:
                skipped['original_invalid_or_nonexternal_anchor'] += 1
                continue
            begin = anchor
            while (begin > max(0, anchor-19) and valid[begin-1] and modes[begin-1] == 0
                   and t[anchor]-t[begin-1] <= MAXIMUM_AGE+1e-8 and t[begin]-t[begin-1] <= .075):
                begin -= 1
            if anchor-begin+1 < 5:
                skipped['fewer_than5_dense_real_observations'] += 1
                continue
            end = min(len(t), anchor+17)
            count = end-anchor
            fvalid = np.zeros(17, bool)
            fvalid[:count] = ~np.maximum.accumulate(~valid[anchor:end] | (modes[anchor:end] != 0))
            fvalid[0] = False
            fp, fr = np.zeros((17, 3), np.float32), np.tile(np.eye(3), (17, 1, 1)).astype(np.float32)
            fp[:count], fr[:count] = a['positions'][anchor:end, 0], a['rotations'][anchor:end, 0]
            sl = slice(begin, anchor+1)
            windows.append(Window(source_key(e), role, anchor, t[sl], a['positions'][sl, 0], a['rotations'][sl, 0],
                a['linear_velocities'][sl, 0], a['angular_velocities'][sl, 0], a['confidence'][sl, 0],
                a['rotation_observable'][sl, 0], fp, fr, fvalid))
            parent_windows += 1
        if parent_windows == 0:
            raise ValueError('Do not silently drop a preassigned fit/selection parent with no eligible real history')
        ledger.append(dict(parent=list(source_key(e)), role=role, decoded=True, frames=len(t),
            recorded_frame_time_min=float(t[0]), recorded_frame_time_max=float(t[-1]),
            inherited_frame_dt_min=float(np.diff(t).min()), inherited_frame_dt_max=float(np.diff(t).max()),
            physical_timestamp_observed=False,
            original_invalid_frames=int((~valid).sum()), external_frames=int((modes == 0).sum()),
            eligible_anchors=parent_windows, excluded_anchor_reasons=dict(skipped), task_outcome_used=False))
    if any(sha(Path(name)) != digest for name, digest in files.items()):
        raise ValueError('TRAIN caches changed while preparing')
    return windows, ledger, files


def load_development_validation():
    """Old source-disjoint VAL is development evaluation only, never fitting.

    It was already used by the original teacher's model selection, and is not
    fresh TEST. The eight new reserved TRAIN parents are not loaded here.
    """
    path = ROOT/'artifacts/framework/multitask_r10/joint16_body_track_adapter_v31/block_manifest.json'
    manifest = json.loads(path.read_text())
    selected = [e for e in manifest['entries'] if e['task'] in TASKS and e['split'] == 'validation']
    original = [e for e in json.loads(TEACHER_SOURCES.read_text())['entries'] if e['task'] in TASKS]
    if (len(selected) != 40 or len(set(source_key(e) for e in selected)) != 40
            or set(source_key(e) for e in selected) & set(source_key(e) for e in original)
            or set(e['cache_sha256'] for e in selected) & set(e['cache_sha256'] for e in original)
            or set(e['recorded_transition_labels']['path'] for e in selected)
               & set(e['recorded_transition_labels']['path'] for e in original)):
        raise ValueError('Exact40 old source-disjoint development VAL parents required')
    windows, ledger, files = load_windows(dict(rows=[dict(entry=e, role='development_validation') for e in selected]), split='validation')
    files[str(path)] = sha(path)
    return windows, ledger, files


def dropout_indices(window, rng, mode=None):
    n = len(window.times)
    if n < 5 or np.any(np.diff(window.times) <= 0):
        raise ValueError('At least5 causal true observations required')
    mode = ('isolated' if rng.integers(2) == 0 else 'burst') if mode is None else mode
    if mode not in ('isolated', 'burst'):
        raise ValueError('One fixed isolated/burst augmentation family only')
    k = min(int(rng.integers(1, 5)), n-5)
    if k == 0:
        return np.arange(n), dict(mode=mode, removed=0, reason='minimum5_real_observations')
    if mode == 'isolated':
        remove = rng.choice(n-1, size=k, replace=False)
    else:
        start = int(rng.integers(0, n-k))
        remove = np.arange(start, start+k)
    keep = np.flatnonzero(~np.isin(np.arange(n), remove))
    times = window.times[keep]
    if (len(keep) < 5 or keep[-1] != n-1 or times[-1]-times[0] > MAXIMUM_AGE+1e-8
            or np.any(np.diff(times) > MAXIMUM_GAP+1e-8)):
        raise ValueError('Augmentation must preserve current/5 real poses/1s age/.25s gap')
    return keep, dict(mode=mode, removed=len(remove), removed_indices=sorted(map(int, remove)),
                      retained_timestamps=times.tolist())


def features(window, model, indices=None, *, recompute_twist=False):
    keep = np.arange(len(window.times)) if indices is None else np.asarray(indices)
    if (keep.ndim != 1 or keep.dtype.kind not in 'iu' or len(keep) < 5 or keep[-1] != len(window.times)-1
            or np.any(np.diff(keep) <= 0) or keep[0] < 0):
        raise ValueError('Ordered retained real observations must include the current anchor')
    t, p, r = window.times[keep], window.positions[keep], window.rotations[keep]
    if t[-1]-t[0] > MAXIMUM_AGE+1e-8 or np.any(np.diff(t) > MAXIMUM_GAP+1e-8):
        raise ValueError('Outside fixed student gap/age support')
    v, w = window.velocities[keep].copy(), window.angular_velocities[keep].copy()
    if recompute_twist:
        v[:] = 0.
        w[:] = 0.
        for index in range(1, len(keep)):
            v[index], w[index] = finite_difference_twist(t[:index+1], p[:index+1], r[:index+1], window=5)
            if not window.observable[keep[index]]:
                w[index] = 0.
    with torch.no_grad():
        values = [torch.tensor(x, dtype=torch.float32) for x in (p, r, v, w)]
        state = model.normalized(*values, torch.zeros(len(keep), dtype=torch.long),
                                 torch.tensor(window.observable[keep], dtype=torch.float32))
    data = np.c_[state.numpy(), np.r_[.05, np.diff(t)]/.05, window.confidence[keep]].astype(np.float32)
    if not np.isfinite(data).all() or data.shape != (len(keep), 21):
        raise ValueError('Exact finite21D observed history required')
    history = np.zeros((20, 21), np.float32)
    history[:len(keep)] = data
    return dict(history=history, length=len(keep), state=state[-1].numpy().copy())
