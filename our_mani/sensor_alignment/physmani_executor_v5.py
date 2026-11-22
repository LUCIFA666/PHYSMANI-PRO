"""v4 arm execution with shared author-domain native RGB preprocessing.

The temporary tensor binding applies only inside this adapter's reset,
observe and predict calls in its owned baseline process, and is restored on
exit. No frozen source file or other process is patched.
"""
from contextlib import contextmanager
import threading

import numpy as np
import torch

from ..aligned import physmani as inherited_physmani
from ..ablations.physmani_executor_v4 import (
    PhysManiAdapter as V4Adapter, ProtocolInvariantError,
    InfrastructureExecutionError, raise_protected_error,
)
from .author_rgbd import PROTOCOL, SensorProtocolInvariantError


_original_tensors = inherited_physmani.tensors
_tensor_binding_lock = threading.RLock()


def tensors(observation, robot, device='cuda'):
    raw = observation.get('observation/raw_rgb_255')
    if raw is None:
        raise SensorProtocolInvariantError('shared native RGB observation is missing')
    rgb = []
    for name in inherited_physmani.CAMERAS:
        image = np.asarray(raw[name], dtype=np.float32)
        if image.ndim != 3 or image.shape[-1] != 3 or not np.isfinite(image).all():
            raise SensorProtocolInvariantError('invalid shared native RGB observation')
        # Match released transform operation order before Actioner maps this
        # from [-1,1] back to [0,1]. No resize/augmentation/channel changes.
        rgb.append(2.*(image.transpose(2, 0, 1)/255.-.5))
    _, pointcloud, depth, gripper = _original_tensors(observation, robot, device=device)
    return torch.as_tensor(np.asarray(rgb)[None], device=device), pointcloud, depth, gripper


@contextmanager
def author_tensor_scope():
    with _tensor_binding_lock:
        previous = inherited_physmani.tensors
        inherited_physmani.tensors = tensors
        try:
            yield
        finally:
            inherited_physmani.tensors = previous


class PhysManiAdapter(V4Adapter):
    def reset(self, *args, **kwargs):
        with author_tensor_scope():
            return super().reset(*args, **kwargs)

    def observe(self, *args, **kwargs):
        with author_tensor_scope():
            return super().observe(*args, **kwargs)

    def predict(self, *args, **kwargs):
        with author_tensor_scope():
            return super().predict(*args, **kwargs)

    def execute(self, *args, **kwargs):
        receipt = super().execute(*args, **kwargs)
        receipt['executor'] = 'released_waypoints_counted_actuator_author_rgbd_v5'
        receipt['sensor_protocol'] = PROTOCOL
        receipt['inherited_arm_executor'] = 'released_waypoints_no_hidden_physics_v4'
        return receipt
