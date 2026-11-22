"""New typed Forecast provider. Never advertises the frozen ebec authority."""
from pathlib import Path
import hashlib
import threading

import numpy as np
import torch

from . import VERSION, CHECKPOINT_SCHEMA
from .model import CubeMotionResidualGRU, model_from_spec, public_inputs
from ..contracts import Forecast
from ..multitask.context_se3 import MultiBodyContextDynamics, SE3ContextNetwork

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_map():
    names = ('our_mani/cube_motion_residual_20261104_v1/__init__.py',
             'our_mani/cube_motion_residual_20261104_v1/model.py',
             'our_mani/cube_motion_residual_20261104_v1/losses.py',
             'our_mani/cube_motion_residual_20261104_v1/data.py',
             'our_mani/cube_motion_residual_20261104_v1/runtime.py',
             'our_mani/multitask/context_se3.py','our_mani/context_dynamics.py','our_mani/contracts.py')
    return {name:sha(ROOT/name) for name in names}


def state_digest(model):
    h = hashlib.sha256()
    for name,value in sorted(model.state_dict().items()):
        h.update(name.encode()); h.update(str(value.dtype).encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


class CubeResidualMotionDynamics(MultiBodyContextDynamics):
    """Same observe/reset/inputs/context_snapshot/predict API, explicit new type.

    Old PublicMotionBridge/cache exact-type and checkpoint guards must be
    replaced by a separately versioned adapter. This class never patches them.
    """
    name = 'Cube_public_real50ms_CV_GRU_residual_v1'

    def __init__(self, checkpoint, *, expected_sha256):
        self.path = Path(checkpoint).resolve(strict=True)
        if type(expected_sha256) is not str or len(expected_sha256)!=64 or sha(self.path)!=expected_sha256:
            raise ValueError('Exact newly fitted physical-clock Cube checkpoint required')
        payload = torch.load(self.path,map_location='cpu',weights_only=False)
        required = {'schema','model_spec','model_state','source_sha256','metadata','train_parents','selection_parents','heldout_parents'}
        if not isinstance(payload,dict) or not required <= set(payload) or payload['schema']!=CHECKPOINT_SCHEMA:
            raise ValueError('New trained residual family payload required; ebec/student cannot substitute')
        m = payload['metadata']
        if (m.get('training_complete') is not True or m.get('time_basis')!='actual_physics_and_task_50ms'
                or m.get('TEST_used_for_fitting') is not False or m.get('heldout_used_for_fitting') is not False
                or m.get('maximum_training_horizon_seconds') != .8):
            raise ValueError('Completed actual-clock TRAIN-only fit, heldout untouched, required')
        groups = [payload[key] for key in ('train_parents','selection_parents','heldout_parents')]
        if any(not isinstance(v,list) or not v or len(set(v))!=len(v) or any(type(p)is not str for p in v) for v in groups):
            raise ValueError('Complete distinct parent role identities required')
        if any(set(groups[i])&set(groups[j]) for i in range(3) for j in range(i)):
            raise ValueError('Whole-parent fit/selection/heldout overlap')
        self.sources = payload['source_sha256']
        if self.sources != source_map():
            raise ValueError('Exact newly frozen residual source family required')
        self.model = model_from_spec(payload['model_spec'])
        self.model.load_state_dict(payload['model_state'],strict=True)
        if any(not bool(torch.isfinite(v).all()) for v in self.model.state_dict().values()):
            raise ValueError('finite trained residual state required')
        self.model.cpu().eval().requires_grad_(False)
        self.metadata = m
        self.checkpoint_sha256 = expected_sha256
        self.encoder_sha256 = self.model.encoder_sha256()
        self.parameter_sha256 = state_digest(self.model)
        self.maximum_validated_horizon = .8  # inherited API name; no accuracy qualification is asserted.
        self.history, self.lock = {}, threading.RLock()

    def _assert_live(self):
        if (sha(self.path)!=self.checkpoint_sha256 or source_map()!=self.sources
                or type(self.model)is not CubeMotionResidualGRU or self.model.training
                or any(p.requires_grad or p.device.type!='cpu' for p in self.model.parameters())
                or state_digest(self.model)!=self.parameter_sha256):
            raise ValueError('Frozen residual model/source/checkpoint identity changed')
        if (getattr(self.model.forward,'__func__',None) is not CubeMotionResidualGRU.forward
                or getattr(self.model.encode_context,'__func__',None) is not SE3ContextNetwork.encode_context
                or getattr(self.model.normalized,'__func__',None) is not SE3ContextNetwork.normalized
                or any(m._forward_hooks or m._forward_pre_hooks for m in self.model.modules())):
            raise ValueError('No replaced model methods or injected forward hooks')

    def observe(self, objects, modes=None, orientation_observability=None):
        self._assert_live()
        modes = modes or {}
        orientation_observability = orientation_observability or {}
        if any(modes.get(o.object_id,'external')!='external' or orientation_observability.get(o.object_id) is not False
               or not np.allclose(o.pose.rotation,np.eye(3),atol=1e-8,rtol=0) for o in objects):
            raise ValueError('This Cube family uses public external motion and unobserved identity only')
        # Original storage/reset semantics remain; caller-provided velocities are
        # never used by inputs(), which recomputes them from retained public p/t.
        return super().observe(objects,modes=modes,orientation_observability=orientation_observability)

    def inputs(self, object_id):
        with self.lock:
            rows = list(self.history.get(object_id,()))
        if len(rows)<5:
            raise ValueError('five actual public Cube poses required')
        n=len(rows)
        p=torch.tensor(np.stack([r['position'] for r in rows]),dtype=torch.float32)[None]
        t=torch.tensor([[r['timestamp'] for r in rows]],dtype=torch.float64)
        c=torch.tensor([[r['confidence'] for r in rows]],dtype=torch.float32)
        history,lengths,state,cutoff=public_inputs(self.model,positions=p,timestamps=t,confidence=c,lengths=torch.tensor([n]))
        return history,lengths,state,float(cutoff[0])

    @torch.inference_mode()
    def context_snapshot(self, object_id):
        self._assert_live()
        result=super().context_snapshot(object_id)
        result.update(forecast_family=VERSION,physical_clock_training=True)
        return result

    @torch.inference_mode()
    def predict(self, objects, horizon, dt):
        self._assert_live()
        old=super().predict(objects,horizon,dt)
        metadata=dict(old.metadata,forecast_family=VERSION,physical_clock_training=True,
            source_sha256=dict(self.sources),velocity_source='causal_OLS_last_five_retained_public_positions',
            first_history_interval_is_unmeasured_zero=True,
            accuracy_qualified=False,learned_contact_or_attachment_dynamics=False,
            legacy_ebec_authority_used=False)
        return Forecast(old.object_ids,old.times,old.positions,old.rotations,
            old.linear_velocities,old.angular_velocities,metadata=metadata)
