"""Only unchanged float32 features/uint8 labels of fit and selection are cached."""
from copy import deepcopy
from pathlib import Path
import hashlib
import json

import numpy as np
from torch.utils.data import Dataset

from ..near_goal_depth_refiner_20261103.data import PARTITION_IDS, SHOT_IDS, sha256

COUNTS = dict(fit=192, selection=32, calib=32)


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _array_sha(array):
    header = dict(shape=list(array.shape), dtype=array.dtype.str, order='C')
    digest = hashlib.sha256(json.dumps(header, sort_keys=True, separators=(',', ':')).encode())
    digest.update(array.tobytes(order='C'))
    return digest.hexdigest()


def _ordered_records(source):
    expected = []
    for i, parent in enumerate(PARTITION_IDS):
        role = 'fit' if i < 24 else ('selection' if i < 28 else 'calib')
        expected.extend((parent, parent+'/'+shot, role) for shot in SHOT_IDS)
    actual = [(r['parent_id'], r['scene_id'], r['role']) for r in source.records]
    if source.scope != 'full32' or len(source) != 256 or actual != expected:
        raise ValueError('Only full32 fixed ordered256 source records, never a role subset or pilot')
    return actual


class CachedRoleDataset(Dataset):
    """Read-only tensors plus an external ordering/hash ledger; no GT objects."""
    @classmethod
    def from_dataset(cls, source, role):
        if role not in ('fit', 'selection'):
            raise ValueError('Calibration cannot enter the training cache')
        ordered = _ordered_records(source)
        source.revalidate()
        source_binding = canonical_sha(source.bound_pointers)
        manifest_sha = sha256(source.path)
        features, semantics, identifiers = [], [], []
        for index, record in enumerate(ordered):
            if record[2] != role:
                continue
            sample = source[index]
            if (sample.get('parent_id'), sample.get('scene_id'), sample.get('role')) != record:
                raise ValueError('Dataset sample identity differs from the fixed ordered ledger')
            feature, semantic = sample['features'], sample['semantic']
            if (not isinstance(feature, np.ndarray) or feature.dtype != np.float32
                    or feature.shape != (4, 9, 64, 64) or not np.isfinite(feature).all()
                    or not isinstance(semantic, np.ndarray) or semantic.dtype != np.uint8
                    or semantic.shape != (4, 64, 64) or not set(np.unique(semantic)) <= {0, 1, 255}):
                raise ValueError('Cache copies original exact float32 features/uint8 labels only')
            features.append(feature.copy(order='C'))
            semantics.append(semantic.copy(order='C'))
            identifiers.append(record)
        if len(features) != COUNTS[role]:
            raise ValueError('No missing or filtered role samples')
        source.revalidate()
        if canonical_sha(source.bound_pointers) != source_binding or sha256(source.path) != manifest_sha:
            raise ValueError('Source bindings changed during cache materialization')
        result = cls()
        result.role, result._ordered = role, tuple(identifiers)
        result._features, result._semantic = np.stack(features), np.stack(semantics)
        result._features.setflags(write=False)
        result._semantic.setflags(write=False)
        result._source_binding, result._manifest_sha = source_binding, manifest_sha
        result._original = result._calculate()
        return result

    def _calculate(self):
        ordered_content = hashlib.sha256()
        for identity, features, labels in zip(self._ordered, self._features, self._semantic):
            ordered_content.update(json.dumps(identity, separators=(',', ':')).encode())
            ordered_content.update(features.tobytes(order='C'))
            ordered_content.update(labels.tobytes(order='C'))
        return dict(schema='full32_exact_current_feature_label_role_cache_v2', role=self.role,
            samples=len(self._features), fields_only=['features', 'semantic'],
            features=dict(shape=list(self._features.shape), dtype=self._features.dtype.str,
                          sha256=_array_sha(self._features)),
            semantic=dict(shape=list(self._semantic.shape), dtype=self._semantic.dtype.str,
                          sha256=_array_sha(self._semantic)),
            ordered_records=[list(r) for r in self._ordered],
            ordered_records_sha256=canonical_sha(self._ordered),
            ordered_content_sha256=ordered_content.hexdigest(),
            bytes=int(self._features.nbytes+self._semantic.nbytes),
            source_binding_sha256=self._source_binding, source_manifest_sha256=self._manifest_sha,
            calibration_cached=False, privileged_metadata_or_robot_or_CF_cached_as_features=False)

    def __len__(self):
        return len(self._features)

    def __getitem__(self, index):
        return dict(features=self._features[index], semantic=self._semantic[index])

    def attestation(self):
        return deepcopy(self._original)

    def revalidate(self):
        if self._calculate() != self._original:
            raise ValueError('Cached role order/content/shape/dtype changed')
