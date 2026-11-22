"""Configurable Cube-only P4-contract future candidate Transformer.

Current public GRU/context and robot inputs preserve the original physical
contract. Labels are offline observed contact pose/time, never forward inputs.
No task classifier or candidate-success score is trained by this model.
"""
from dataclasses import asdict,dataclass
from copy import deepcopy
from pathlib import Path
import hashlib
import math
import torch

from ..local_mpc_p4_20261103 import model as original
from . import VERSION


@dataclass(frozen=True)
class CandidateConfig:
    d_model:int=256
    num_layers:int=4
    num_heads:int=8
    num_candidates:int=8
    dropout:float=.1

    def __post_init__(self):
        if (any(type(v)is not int or v<1 for v in
                (self.d_model,self.num_layers,self.num_heads,self.num_candidates))
                or self.d_model<16 or self.d_model%self.num_heads or self.num_candidates!=8
                or isinstance(self.dropout,bool)or not isinstance(self.dropout,(float,int))
                or not math.isfinite(self.dropout)or not 0<=self.dropout<1):
            raise ValueError('Explicit divisible future Transformer configuration and K8 required')

    def to_dict(self):
        value=asdict(self);value['dropout']=float(value['dropout']);return value

    @classmethod
    def from_dict(cls,value):
        if type(value)is not dict or set(value)!={'d_model','num_layers','num_heads','num_candidates','dropout'}:
            raise ValueError('Exact serialized Cube future candidate configuration required')
        return cls(**value)


class CubeFutureCandidateTransformer(original.CandidateTransformerPilot):
    def __init__(self,config=CandidateConfig()):
        if type(config)is not CandidateConfig:raise TypeError('Explicit new-family CandidateConfig required')
        super().__init__(**config.to_dict());self.config=config

    def forward(self,future_tokens,robot,geometry,context_c,task_e):
        if (not isinstance(task_e,torch.Tensor)or task_e.shape!=(len(future_tokens),16)
                or not torch.equal(task_e,torch.full_like(task_e,1./8))):
            raise ValueError('Cube-only model keeps independent fixed uniform SimNorm task embedding')
        return super().forward(future_tokens,robot,geometry,context_c,task_e)

    def spec(self):
        value=super().spec()
        value.update(schema='cube_future_candidate_transformer_20261104_v1',version=VERSION,
            checkpoint_family='new_trained_Cube_future_candidate_v1',model_config=self.config.to_dict(),
            actual_parameter_count=sum(p.numel()for p in self.parameters()),
            inherited_physical_math_sha256=hashlib.sha256(Path(original.__file__).read_bytes()).hexdigest(),
            task_embedding_source='fixed_uniform_single_Cube_operation_not_cross_task_identification',
            fixed_task_e=[1./8]*16,task_classifier_trained=False,
            ordinary_and_highspeed_are_one_Cube_operation=True,
            speed_training_coverage_is_dataset_evidence_not_architecture=True,
            original_loss='original body-local parallel-gripper pose and actual observed relative time matching',
            supported_training_precision='FP32_reference; no BF16 training claim',
            success_probability_trained=False,automatic_task_classification_learned=False,
            old_pilot_checkpoint_compatible=False,formal_pose_or_execution_qualified=False)
        return value

    @classmethod
    def from_spec(cls,spec):
        return model_from_spec(spec)


def model_from_spec(spec):
    if type(spec)is not dict or spec.get('schema')!='cube_future_candidate_transformer_20261104_v1' or spec.get('version')!=VERSION:
        raise ValueError('Exact new Cube future model specification required')
    model=CubeFutureCandidateTransformer(CandidateConfig.from_dict(spec.get('model_config')))
    if model.spec()!=spec:raise ValueError('Full model units/priors/config/source specification differs')
    return model


def parameter_count(config=CandidateConfig()):
    with torch.device('meta'):model=CubeFutureCandidateTransformer(config)
    return sum(p.numel()for p in model.parameters())


def model_source_map():
    paths=[Path(__file__).resolve(),Path(__file__).with_name('__init__.py'),Path(__file__).with_name('losses.py'),Path(original.__file__).resolve()]
    return {str(p):hashlib.sha256(p.read_bytes()).hexdigest()for p in paths}
