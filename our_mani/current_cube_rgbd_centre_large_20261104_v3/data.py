"""FIT/selection CF supervision, separated from public model tensors.

No reserved role is exposed. Constructor inherits the original hash/proof
metadata checks; only the requested FIT/selection getter decodes target NPZ.
Depth influence is neither a pure face nor a world-existence label.
"""
import numpy as np

from ..current_cube_rgbd_centre_20261103_v1.data import (
    DualCropCubeDataset, derive_targets, load_npz)
from ..current_cube_rgbd_centre_20261103_v1.features import make_dual_goal_crops
from ..near_goal_depth_refiner_20261103.data import binary_depth_labels

PUBLIC_KEYS = {'features', 'wide_features', 'K', 'camera_rotations',
               'initial_sphere', 'source_pixels_uv'}


def public_model_arrays(inputs):
    """Same real pixels/calibration, plus exact source UV for deduplication."""
    crops = make_dual_goal_crops(inputs)
    yy, xx = np.indices((64, 64))
    origin = crops['near']['origins_xy']
    near_uv = np.stack((origin[:, None, None, 0]+xx,
                        origin[:, None, None, 1]+yy), -1)
    wide_uv = crops['wide']['source_pixels_uv'].copy()
    # Behind-camera projection has no source pixel. Use an all-outside regular
    # lattice, rather than treating the old unused zero origin as real pixels.
    for camera in range(4):
        if not crops['near']['projection_available'][camera]:
            near_uv[camera] = np.stack((xx-256, yy-256), -1)
        if not crops['wide']['projection_available'][camera]:
            wide_uv[camera] = np.stack((2*xx-256, 2*yy-256), -1)
    return dict(features=crops['near']['features'],
        wide_features=crops['wide']['features'],
        K=np.asarray(inputs['K'], np.float32).copy(),
        camera_rotations=np.asarray(inputs['T'], np.float32)[:, :3, :3].copy(),
        initial_sphere=np.asarray(inputs['public_sphere_center'], np.float32).copy(),
        source_pixels_uv=np.stack((near_uv, wide_uv)).astype(np.int64))


def pixel_targets(inputs, labels, source_pixels_uv):
    """Sample original 0/1/255 by exact UV; padding always remains UNKNOWN.

    A scope-negative shot can have Cube foreground: do not derive these
    targets from shot-level availability or from privileged Cube projection.
    """
    uv = np.asarray(source_pixels_uv)
    if uv.shape != (2, 4, 64, 64, 2) or uv.dtype != np.int64:
        raise ValueError('Exact dual-scale integer source UV required')
    full = binary_depth_labels(inputs, labels)
    target = np.full((2, 4, 64, 64), 255, np.uint8)
    for scale in range(2):
        for camera in range(4):
            u, v = uv[scale, camera, ..., 0], uv[scale, camera, ..., 1]
            valid = (u >= 0) & (u < 256) & (v >= 0) & (v < 256)
            target[scale, camera, valid] = full[camera, v[valid], u[valid]]
    return target


class ForegroundProposalDataset:
    def __init__(self, manifest, role='fit'):
        if role not in ('fit', 'selection'):
            raise ValueError('Prototype excludes all reserved getters')
        self.base = DualCropCubeDataset(manifest, role)
        self.records, self.role = self.base.records, role

    def __len__(self):
        return len(self.base)

    def public_sample(self, index):
        sample = self.base.public_sample(index)
        sample['model_inputs'] = public_model_arrays(sample['inputs'])
        if set(sample['model_inputs']) != PUBLIC_KEYS:
            raise ValueError('Public tensor contract changed')
        return sample

    def __getitem__(self, index):
        sample = self.public_sample(index)
        record = sample['record']
        labels = load_npz(record['paths']['labels'])
        sample['targets'] = derive_targets(sample['inputs'], labels,
                                          record['metadata'], record['shot_id'])
        sample['pixel_labels'] = pixel_targets(sample['inputs'], labels,
                                               sample['model_inputs']['source_pixels_uv'])
        self.base.access.append(dict(scene_id=record['scene_id'], role=self.role,
                                     kind='label_supervision_v3'))
        return sample

    def revalidate(self):
        return self.base.revalidate()
