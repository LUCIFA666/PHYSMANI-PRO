"""Grouped TRAIN/VAL SE3 tracks; future arrays are supervision, never history.

Track manifests are separate from TEST reset manifests and from expert TCP
demonstrations. In particular, an expert debug target is not a body track.
"""
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from ..benchmark_protocol import file_sha256
from .context_se3 import history_inputs


def validate_motion_label_schema(arrays, manifest_schema, entry_schema):
    from .motion_labels import SCHEMA, RAW_LINEAR, RAW_ANGULAR, backward_pose_supervision
    if manifest_schema != entry_schema or manifest_schema not in (None, SCHEMA):
        raise ValueError('body-track manifest/entry dynamics label schemas differ or are unsupported')
    if manifest_schema is None:
        if any(name in arrays for name in (RAW_LINEAR, RAW_ANGULAR, 'target_twist_valid')):
            raise ValueError('derived dynamics labels require their explicit manifest/entry schema')
        return
    required = (RAW_LINEAR, RAW_ANGULAR, 'target_twist_valid', 'target_pose_valid_before_twist_mask',
                'native_pose_valid', 'mode_valid', 'dynamics_segment_valid')
    if any(name not in arrays for name in required):
        raise ValueError('derived dynamics labels must retain raw native velocities and boundary masks')
    original = {name: value for name, value in arrays.items() if name not in (RAW_LINEAR, RAW_ANGULAR)}
    original['target_valid'] = arrays['target_pose_valid_before_twist_mask']
    original['target_linear_velocities'] = arrays[RAW_LINEAR]
    original['target_angular_velocities'] = arrays[RAW_ANGULAR]
    expected, _ = backward_pose_supervision(original)
    for name in ('target_twist_valid', 'target_valid'):
        if arrays[name].dtype != np.bool_ or not np.array_equal(arrays[name], expected[name]):
            raise ValueError('derived dynamics boundary mask differs: ' + name)
    for name in ('target_linear_velocities', 'target_angular_velocities'):
        if arrays[name].shape != expected[name].shape or not np.allclose(arrays[name], expected[name], atol=1e-8, rtol=1e-8):
            raise ValueError('dynamics labels differ from causal backward GT pose difference: ' + name)


@dataclass
class BodyTrackEpisode:
    entry: dict
    arrays: dict

    @property
    def source_key(self):
        return (self.entry['split'], self.entry['task'],
                self.entry.get('parent_raw_episode_id', self.entry['raw_episode_id']))


def read_tracks(root, manifest_path):
    root = Path(root)
    manifest = json.loads(Path(manifest_path).read_text())
    if manifest.get('version') != 'r10_causal_body_tracks_v1':
        raise ValueError('need body-pose tracks, not expert target-position labels')
    episodes, seen = [], set()
    required = ('times', 'body_ids', 'positions', 'rotations', 'linear_velocities',
                'angular_velocities', 'confidence', 'valid', 'modes', 'rotation_observable')
    for entry in manifest['entries']:
        if entry['split'] not in ('train', 'validation'):
            raise ValueError('TEST tracks cannot enter fitting or model selection')
        key = entry['split'], entry['task'], entry['raw_episode_id']
        if key in seen:
            raise ValueError('duplicate source episode')
        seen.add(key)
        parent = entry.get('parent_raw_episode_id', entry['raw_episode_id'])
        if (entry.get('parent_source_group') is not None
                and entry['parent_source_group'] != f"{entry['split']}:{entry['task']}:{parent}"):
            raise ValueError('augmentation must retain its real parent source group')
        if entry.get('observation_source') != 'causal_RGBD_and_proprioception':
            raise ValueError('model history must contain policy-observable pose estimates')
        if entry.get('target_source') not in ('tracked_future_observations', 'TRAIN_VAL_native_supervision'):
            raise ValueError('future label provenance must be explicit')
        cache = root / entry['cache']
        if file_sha256(cache) != entry['cache_sha256']:
            raise ValueError('body-track cache changed')
        with np.load(cache, allow_pickle=False) as stored:
            if any(name not in stored for name in required):
                raise ValueError('incomplete body-state cache')
            arrays = {name: np.array(stored[name], copy=True) for name in stored.files}
        times, bodies = arrays['times'], arrays['body_ids']
        t, b = len(times), len(bodies)
        if (times.ndim != 1 or t < 5 or b < 1 or len(set(map(str, bodies))) != b
                or not np.isfinite(times).all() or np.any(np.diff(times) <= 0)):
            raise ValueError('invalid episode time/body axis')
        for name in ('positions', 'linear_velocities', 'angular_velocities'):
            if arrays[name].shape != (t, b, 3):
                raise ValueError('invalid xyz shape: ' + name)
        if arrays['rotations'].shape != (t, b, 3, 3):
            raise ValueError('invalid rotation shape')
        for name in ('valid', 'modes', 'confidence', 'rotation_observable'):
            if arrays[name].shape != (t, b):
                raise ValueError('invalid state-mask shape: ' + name)
        if arrays['valid'].dtype != np.bool_ or arrays['rotation_observable'].dtype != np.bool_:
            raise ValueError('visibility/observability masks must be boolean')
        valid = arrays['valid']
        if (not np.isin(arrays['modes'], (0, 1, 2)).all()
                or not np.isfinite(arrays['confidence'][valid]).all()
                or np.any((arrays['confidence'][valid] < 0) | (arrays['confidence'][valid] > 1))):
            raise ValueError('invalid observed mode/confidence')
        for name in ('positions', 'rotations', 'linear_velocities', 'angular_velocities'):
            if not np.isfinite(arrays[name][valid]).all():
                raise ValueError('nonfinite visible pose/twist')
        rotations = arrays['rotations'][valid]
        if (not np.allclose(np.swapaxes(rotations, -1, -2) @ rotations, np.eye(3), atol=1e-4)
                or np.any(np.linalg.det(rotations) < .9999)):
            raise ValueError('visible orientation is not SO3')
        for name, fallback in (('target_positions', 'positions'), ('target_rotations', 'rotations'),
                               ('target_linear_velocities', 'linear_velocities'),
                               ('target_angular_velocities', 'angular_velocities')):
            if name not in arrays:
                if entry['target_source'] != 'tracked_future_observations':
                    raise ValueError('native-supervised targets must not silently use tracked poses')
                arrays[name] = arrays[fallback].copy()
            if arrays[name].shape != arrays[fallback].shape:
                raise ValueError('target shape differs')
        arrays.setdefault('target_valid', valid.copy())
        arrays.setdefault('target_rotation_observable', arrays['rotation_observable'].copy())
        for name in ('target_valid', 'target_rotation_observable'):
            if arrays[name].shape != (t, b) or arrays[name].dtype != np.bool_:
                raise ValueError('invalid future label mask')
        validate_motion_label_schema(arrays, manifest.get('dynamics_target_label_schema'),
                                     entry.get('dynamics_target_label_schema'))
        episodes.append(BodyTrackEpisode(dict(entry), arrays))
    if not episodes:
        raise ValueError('empty track manifest')
    return manifest, episodes


def train_normalization(episodes):
    """Fit all normalization statistics to TRAIN observations only."""
    train = [episode for episode in episodes if episode.entry['split'] == 'train']
    if not train:
        raise ValueError('TRAIN observations required for normalization')
    pooled = {name: [] for name in ('positions', 'linear_velocities', 'angular_velocities')}
    source_counts = {}
    for episode in train:
        arrays = episode.arrays
        valid = arrays['valid'] & (arrays['modes'] != 1)
        source_counts[episode.source_key] = source_counts.get(episode.source_key, 0) + int(valid.sum())
        for name in pooled:
            pooled[name].append(arrays[name][valid])
    if any(count == 0 for count in source_counts.values()):
        raise ValueError('some TRAIN sources have no observed external/released body states')
    pooled = {name: np.concatenate(values) for name, values in pooled.items()}
    center = np.median(pooled['positions'], axis=0)
    return dict(position_center=center.tolist(),
        position_scale=np.maximum(np.quantile(np.abs(pooled['positions'] - center), .95, axis=0), .05).tolist(),
        velocity_scale=np.maximum(np.quantile(np.abs(pooled['linear_velocities']), .95, axis=0), .1).tolist(),
        angular_velocity_scale=np.maximum(np.quantile(np.abs(pooled['angular_velocities']), .95, axis=0), .2).tolist()), source_counts


def examples(episodes, model, steps=22, stride=2):
    """Build causal windows and separate future labels, retaining source groups."""
    result, coverage, body_groups, parent_groups = [], {}, {}, {}
    for episode in episodes:
        arrays = episode.arrays
        times = arrays['times']
        coverage.setdefault(episode.source_key, 0)
        source_index = parent_groups.setdefault(episode.source_key, len(parent_groups))
        for body, body_id in enumerate(arrays['body_ids']):
            body_key = (episode.entry['split'], episode.entry['task'],
                        episode.entry['raw_episode_id'], str(body_id))
            body_group = body_groups.setdefault(body_key, len(body_groups))
            valid, modes = arrays['valid'][:, body], arrays['modes'][:, body]
            for anchor in range(4, len(times) - 1, stride):
                mode = int(modes[anchor])
                if mode == 1 or not valid[anchor]:
                    continue
                begin = anchor
                while (begin > max(0, anchor - 19) and valid[begin - 1]
                       and modes[begin - 1] == mode and times[begin] - times[begin - 1] <= .075):
                    begin -= 1
                if anchor - begin + 1 < 5:
                    continue
                rows = [dict(timestamp=times[k], position=arrays['positions'][k, body],
                    rotation=arrays['rotations'][k, body], velocity=arrays['linear_velocities'][k, body],
                    angular_velocity=arrays['angular_velocities'][k, body],
                    confidence=arrays['confidence'][k, body], mode=mode,
                    rotation_observable=bool(arrays['rotation_observable'][k, body])) for k in range(begin, anchor + 1)]
                history, lengths, state, _ = history_inputs(rows, model)
                end = min(len(times), anchor + steps + 1)
                count = end - anchor
                mode_changes = np.maximum.accumulate(modes[anchor:end] != mode)
                # An unknown attachment/identity interval can hide a mode change.
                # A pre-boundary anchor cannot resume supervision after it clears.
                unknown_boundary = np.maximum.accumulate(~arrays['target_valid'][anchor:end, body])
                mask = np.zeros(steps + 1, dtype=bool)
                mask[:count] = (~unknown_boundary & ~mode_changes
                    & np.isclose(times[anchor:end], times[anchor] + np.arange(count) * .05, atol=1e-6))
                mask[0] = False
                if not mask.any():
                    continue
                targets = {}
                for name, shape in (('positions', (3,)), ('rotations', (3, 3)),
                                     ('linear_velocities', (3,)), ('angular_velocities', (3,))):
                    target = np.zeros((steps + 1, *shape), dtype=np.float32)
                    target[:count] = arrays['target_' + name][anchor:end, body]
                    if not np.isfinite(target[mask]).all():
                        raise ValueError('nonfinite supervised future labels')
                    targets[name] = target
                rotation_mask = np.zeros(steps + 1, dtype=bool)
                rotation_mask[:count] = arrays['target_rotation_observable'][anchor:end, body]
                rotation_mask &= mask & bool(arrays['rotation_observable'][anchor, body])
                result.append(dict(history=history[0].numpy(), length=int(lengths[0]), state=state[0].numpy(),
                    targets=targets, mask=mask, rotation_mask=rotation_mask, source_index=source_index,
                    source_key=episode.source_key, body_group=body_group, body_id=str(body_id), anchor=anchor,
                    rotation_observable=bool(arrays['rotation_observable'][anchor, body])))
                coverage[episode.source_key] += 1
    if any(count == 0 for count in coverage.values()):
        missing = [key for key, count in coverage.items() if count == 0]
        raise ValueError('sources without usable causal windows: ' + repr(missing[:10]))
    return result, coverage
