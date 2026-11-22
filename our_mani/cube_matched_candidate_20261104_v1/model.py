"""Same 3.19M candidate topology; independently declared frame and time targets.

Architecture-compatible weights are an explicit initialization only. They do
not prove semantic compatibility with a different public GRU/context/frame.
"""
from pathlib import Path
import hashlib
import json
import torch

from . import VERSION,CHECKPOINT_SCHEMA
from ..candidate_transformer_cube_20261104_v1.model import (
    CandidateConfig,CubeFutureCandidateTransformer)
from ..candidate_transformer_cube_20261104_v1 import model as previous
from ..local_mpc_p4_20261103 import model as physical
from ..cube_candidate_matched_20261104_v2 import FRAME


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class CubeMatchedCandidateTransformer(CubeFutureCandidateTransformer):
    def spec(self):
        value=super().spec()
        value.update(schema=CHECKPOINT_SCHEMA,version=VERSION,
            checkpoint_family=CHECKPOINT_SCHEMA,target_frame=FRAME,
            output_pose_frame='causal public forecast reference coordinates at actual contact; not Native body-fixed coordinates',
            output_rotation_frame='TCP orientation relative to the same public forecast orientation basis',
            original_loss='Sup matched public-reference positive pose plus actual relative time masked outside [.05,.8]; no ranking score',
            actual_time_supervision='retain actual contact tau; pose-only loss for out-of-horizon tau; never clamp target time',
            required_forecast_identity='explicit newly trained forecast checkpoint SHA bound by data and runtime',
            old_weights_are_initialization_only=True,
            score_head='none; proposal_logits exactly zero and untrained',
            semantic_checkpoint_compatible_with_old=False)
        value['explicit_prior_semantics'].update(
            approach_direction='public reference-frame backoff from contact toward pregrasp; not a measured Native body axis',
            pregrasp_reference='predicted public centre + public reference rotation @ (contact offset + backoff)')
        # New family intentionally fixes the inherited tuple/list spec issue;
        # no frozen old model is changed or declared JSON-compatible retroactively.
        return json.loads(json.dumps(value,allow_nan=False))

    @classmethod
    def from_spec(cls,spec):return model_from_spec(spec)

    @classmethod
    def from_legacy_checkpoint(cls,checkpoint,*,expected_sha256):
        path=Path(checkpoint).resolve(strict=True)
        if type(expected_sha256)is not str or len(expected_sha256)!=64 or sha(path)!=expected_sha256:
            raise ValueError('Explicit original trained Cube checkpoint SHA required for warm initialization')
        payload=torch.load(path,map_location='cpu',weights_only=False)
        if (payload.get('schema')!='Cube_candidate_transformer_20261104_v1'
                or payload.get('task_classifier_trained') is not False):
            raise ValueError('Only explicitly declared old Cube architecture can initialize this new family')
        config=CandidateConfig.from_dict(payload['model_config'])
        old=CubeFutureCandidateTransformer(config)
        if payload['model_spec']!=old.spec():
            raise ValueError('Original torch specification/config/source differs')
        root=Path(__file__).resolve().parents[2]
        for name,expected in payload['source_sha256'].items():
            source=Path(name);source=source if source.is_absolute() else root/source
            if sha(source)!=expected:raise ValueError('Original warm-start source changed: '+str(source))
        model=cls(config)
        state=payload['model_state'];reference=model.state_dict()
        if set(state)!=set(reference):raise ValueError('Exact architecture-compatible parameter names required')
        for name,value in state.items():
            if (not isinstance(value,torch.Tensor) or value.device.type!='cpu' or value.dtype!=reference[name].dtype
                    or value.shape!=reference[name].shape or not bool(torch.isfinite(value).all())):
                raise ValueError('Invalid warm-start parameter: '+name)
        model.load_state_dict(state,strict=True)
        return model,dict(path=str(path),sha256=expected_sha256,
            initialized_from_schema=payload['schema'],same_state_shapes=True,
            Native_body_targets_not_reused_as_new_labels=True,
            new_forecast_or_frame_compatibility_proven=False,
            fitting_required_before_new_runtime=True,old_checkpoint_or_source_modified=False)


def model_from_spec(spec):
    if type(spec)is not dict or spec.get('schema')!=CHECKPOINT_SCHEMA or spec.get('version')!=VERSION:
        raise ValueError('Exact new public-reference matched-candidate family required')
    model=CubeMatchedCandidateTransformer(CandidateConfig.from_dict(spec.get('model_config')))
    if model.spec()!=spec:raise ValueError('Full canonical matched public-frame specification differs')
    return model


def parameter_count(config=CandidateConfig()):
    with torch.device('meta'):model=CubeMatchedCandidateTransformer(config)
    return sum(p.numel()for p in model.parameters())


def model_source_map():
    from ..cube_candidate_matched_20261104_v2.data import source_map as matched_sources
    result=matched_sources()
    paths=[Path(__file__),Path(__file__).with_name('__init__.py'),Path(__file__).with_name('losses.py'),
        Path(previous.__file__),Path(previous.__file__).with_name('__init__.py'),
        Path(previous.__file__).with_name('losses.py'),Path(physical.__file__)]
    result.update({str(p.resolve()):sha(p) for p in paths})
    return result
