"""Explicit immutable demo-bank bridge; never pass raw model dictionaries."""
from dataclasses import dataclass

import numpy as np
import torch


def _array(value,shape):
    if isinstance(value,torch.Tensor):value=value.detach().float().cpu().numpy()
    value=np.asarray(value,dtype=np.float64)
    if value.shape!=shape or not np.isfinite(value).all():
        raise ValueError('Exact finite unbatched demo-bank shape '+str(shape)+' required')
    value=value.copy();value.setflags(write=False);return value


@dataclass(frozen=True)
class DemoCommandBankV2:
    world_tcp_positions:object
    world_tcp_rotations:object
    opening_intent:object
    opening_logits:object
    demo_logits:object

    def __post_init__(self):
        for name,shape in (('world_tcp_positions',(8,3)),('world_tcp_rotations',(8,3,3)),
                           ('opening_intent',(8,)),('opening_logits',(8,)),('demo_logits',(8,))):
            object.__setattr__(self,name,_array(getattr(self,name),shape))
        R=self.world_tcp_rotations
        if (not np.allclose(R.transpose(0,2,1)@R,np.eye(3),atol=1e-5,rtol=0)
                or not np.allclose(np.linalg.det(R),1.,atol=1e-5,rtol=0)
                or np.any((self.opening_intent<0)|(self.opening_intent>1))):
            raise ValueError('Proper world rotations and future opening intentions required')


def bank_from_model_output(raw,*,batch_index=0):
    expected={'world_tcp_positions','world_tcp_rotations','opening_intent','opening_logits',
        'proposal_logits','proposal_score_semantics','calibrated_success_probability',
        'output_semantics','handoff_tau_seconds','physical_handoff_qualified'}
    if set(raw)!=expected or type(batch_index)is not int or batch_index<0:
        raise ValueError('Exact raw v2 command schema and batch index required; no private extras')
    if (raw['proposal_score_semantics']!='expert_command_demonstration_mixture_not_task_success'
            or raw['output_semantics']!='expert_command_intent_not_observed_state'
            or raw['handoff_tau_seconds']is not None or raw['physical_handoff_qualified']is not False
            or raw['calibrated_success_probability']is not False):
        raise ValueError('No observed-state, handoff-time or success claim allowed')
    values={key:raw[key][batch_index]for key in ('world_tcp_positions','world_tcp_rotations','opening_intent','opening_logits')}
    values['demo_logits']=raw['proposal_logits'][batch_index]
    return DemoCommandBankV2(**values)


def generic_v1_proposals(bank):
    if type(bank)is not DemoCommandBankV2:
        raise ValueError('Exact typed DemoCommandBankV2 required, not arbitrary raw dictionaries')
    # scipy Rotation in the frozen sink needs a writable temporary buffer.
    # Copy only this public whitelist; the immutable bank stays unmodified.
    return dict(world_tcp_positions=bank.world_tcp_positions.copy(),world_tcp_rotations=bank.world_tcp_rotations.copy(),
        opening_intent=bank.opening_intent.copy(),opening_logits=bank.opening_logits.copy(),
        proposal_logits=bank.demo_logits.copy(),proposal_score_semantics='expert_command_demonstration_mixture_not_task_success',
        calibrated_success_probability=False,output_semantics='expert_command_intent_not_observed_state',
        handoff_tau_seconds=None,physical_handoff_qualified=False)


def select_generic_reference(bank,*,timestamp,policy_sha256,e_sha256,selection_rule,selected_index=None):
    from ..generic_reference_mpc_20261104_v1.contracts import select_reference
    # The sink's old field spelling is deliberately mapped here. Its
    # WorldTCPReference keeps opening intent pending/unissued, never held.
    return select_reference(generic_v1_proposals(bank),timestamp=timestamp,
        policy_sha256=policy_sha256,e_sha256=e_sha256,selection_rule=selection_rule,selected_index=selected_index)
