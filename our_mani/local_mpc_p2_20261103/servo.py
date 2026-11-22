"""Bounded actual-joint servo residual, plus exact validated PoE TCP FK."""
from __future__ import annotations

import torch
from torch import nn

from .dataset import DOF,DT,STATE_DIM,REFERENCE_DIM,FEATURE_DIM
from .torch_fk import TorchPoEKinematics


class JointServoResidual(nn.Module):
    def __init__(self,fk_snapshot,*,residual_q_bound,residual_dq_bound,
                 feature_mean=None,feature_std=None,aperture_step_bound=.5,hidden=128):
        super().__init__()
        for name,value,shape in (("residual_q_bound",residual_q_bound,(DOF,)),
                                 ("residual_dq_bound",residual_dq_bound,(DOF,)),
                                 ("feature_mean",[0.]*FEATURE_DIM if feature_mean is None else feature_mean,(FEATURE_DIM,)),
                                 ("feature_std",[1.]*FEATURE_DIM if feature_std is None else feature_std,(FEATURE_DIM,))):
            value=torch.as_tensor(value,dtype=torch.float32)
            if tuple(value.shape)!=shape or not bool(torch.isfinite(value).all()):raise ValueError(name+" shape/finite error")
            if name!="feature_mean" and not bool((value>0).all()):raise ValueError(name+" must be positive")
            self.register_buffer(name,value.clone())
        if not 0<aperture_step_bound<=1:raise ValueError("aperture per-step bound must be in (0,1]")
        self.aperture_step_bound=float(aperture_step_bound)
        self.network=nn.Sequential(nn.Linear(FEATURE_DIM,hidden),nn.SiLU(),nn.LayerNorm(hidden),
                                   nn.Linear(hidden,hidden),nn.SiLU(),nn.Linear(hidden,2*DOF+4))
        nn.init.normal_(self.network[-1].weight,std=.001)
        nn.init.zeros_(self.network[-1].bias)
        with torch.no_grad():self.network[-1].bias[2*DOF:].fill_(-6.)
        self.fk=TorchPoEKinematics(fk_snapshot)
        self.metadata=dict(schema="P2_bounded_joint_servo_residual_v1",DT=DT,
            residual_base="q_plus_actual_dq_times_dt_and_constant_actual_dq",
            actual_state="q_dq_native_finger_openings_held",input_action="known_dispatched_qref_vref_aref_gripper_target",
            reference_is_measured_state=False,TCP="exact_PoE_FK_of_predicted_actual_q_not_requested_TCP",
            held_rollout="fixed_anchor_mode_no_future_held_label_injected_contact_transition_not_qualified",
            model="memoryless_first_servo_response_baseline_not_complete_Native_contact_dynamics",
            formal_TEST_used=False,old_expert_advice_used=False,trained=False)

    @staticmethod
    def features(state,reference):
        return torch.cat((state[...,:DOF],state[...,DOF:2*DOF],state[...,2*DOF:],
                           reference[...,:DOF]-state[...,:DOF],reference[...,DOF:]),dim=-1)

    def forward(self,state,reference):
        if (state.shape[-1]!=STATE_DIM or reference.shape[:-1]!=state.shape[:-1]
                or reference.shape[-1]!=REFERENCE_DIM or not state.is_floating_point()
                or not reference.is_floating_point() or not bool(torch.isfinite(state).all())
                or not bool(torch.isfinite(reference).all())):raise ValueError("aligned finite actual state and known reference required")
        raw=self.network((self.features(state,reference)-self.feature_mean)/self.feature_std)
        rq=raw[...,:DOF].tanh()*self.residual_q_bound
        rv=raw[...,DOF:2*DOF].tanh()*self.residual_dq_bound
        q=state[...,:DOF]+DT*state[...,DOF:2*DOF]+rq
        dq=state[...,DOF:2*DOF]+rv
        opening=state[...,2*DOF:2*DOF+2]
        if not bool(((opening>=-1e-5)&(opening<=1+1e-5)).all()):raise ValueError("actual finger openings outside [0,1]")
        rates=raw[...,2*DOF:].reshape(*raw.shape[:-1],2,2).sigmoid()
        cap=opening.new_tensor(self.aperture_step_bound)
        # Separate close/open rates avoid dead gradients at fully open/closed.
        opened=opening-torch.minimum(opening,cap)*rates[...,0]+torch.minimum(1-opening,cap)*rates[...,1]
        next_state=torch.cat((q,dq,opened,state[...,-1:]),dim=-1)
        return dict(state=next_state,q=q,dq=dq,opening=opened,residual_q=rq,residual_dq=rv,tcp_matrix=self.fk(q))


def rollout_recorded_references(model,initial_state,references):
    """Free actual-state rollout under recorded references; no future truth input.

    This intentionally does NOT claim to regenerate the closed-loop executor's
    future references. That separate deterministic-chain integration is pending.
    """
    state,outputs=initial_state,[]
    for i in range(references.shape[1]):
        prediction=model(state,references[:,i]);outputs.append(prediction);state=prediction["state"]
    return {key:torch.stack([p[key] for p in outputs],dim=1) for key in outputs[0]}


def baseline_rollout(model,initial_state,references,kind="constant_velocity"):
    if kind not in ("hold","constant_velocity","instant_reference"):raise ValueError("explicit baseline required")
    steps=references.shape[1]
    state=initial_state[:,None].expand(-1,steps,-1).clone()
    if kind=="constant_velocity":
        time=(torch.arange(steps,device=state.device,dtype=state.dtype)+1)*DT
        state[...,:DOF]=state[...,:DOF]+time[None,:,None]*initial_state[:,None,DOF:2*DOF]
    elif kind=="instant_reference":
        state[...,:DOF]=references[...,:DOF]
        state[...,DOF:2*DOF]=references[...,DOF:2*DOF]
        state[...,2*DOF:2*DOF+2]=references[...,-1:,].expand(-1,-1,2)
    return dict(state=state,q=state[...,:DOF],dq=state[...,DOF:2*DOF],
                opening=state[...,2*DOF:2*DOF+2],tcp_matrix=model.fk(state[...,:DOF]))
