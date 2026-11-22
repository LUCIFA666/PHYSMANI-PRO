"""Pure frozen-generation adapter from P4 proposals to the existing MPC schema.

No checkpoint, perception, Native handle, task selector, or geometry fallback
is loaded here. The caller supplies a current causal model output at each NEW
generation and must honor availability masks before fresh GRU/collision checks.
"""
from copy import deepcopy
import math

import numpy as np
from scipy.spatial.transform import Rotation
import torch

from .model import EXPLICIT_PRIORS, POSITION_BOUND_M, SCORE_SEMANTICS


_MODEL_KEYS=frozenset(('positions','rotation_matrices','rotation_6d','rotation_6d_raw',
                       'times','proposal_logits','proposal_score_semantics'))
_EPS=1e-9


def _array(value,shape,name):
    if isinstance(value,torch.Tensor):value=value.detach().cpu().numpy()
    value=np.asarray(value,dtype=np.float64)
    # A model batch of one is accepted; multiple anchors are never flattened.
    if value.shape==(1,*shape):value=value[0]
    if value.shape!=shape or not np.isfinite(value).all():
        raise ValueError('finite '+name+' with shape '+str(shape)+' required')
    return value.copy()


def _proposal_arrays(output):
    if not isinstance(output,dict) or set(output)-_MODEL_KEYS or not {'positions','rotation_matrices','times'}<=set(output):
        raise ValueError('only declared single-anchor P4 model outputs are accepted')
    p=_array(output['positions'],(8,3),'object-relative positions')
    r=_array(output['rotation_matrices'],(8,3,3),'object-relative rotations')
    tau=_array(output['times'],(8,),'relative seconds')
    # Float32 endpoints may differ by one representation ulp. Accepted values
    # remain verbatim; fresh-Forecast coverage below still masks an overshoot.
    if np.any(np.abs(p)>POSITION_BOUND_M+1e-7) or np.any(tau<.05-1e-7) or np.any(tau>.8+1e-7):
        raise ValueError('P4 positions +/-0.12m and relative time0.05..0.8s required')
    if (not np.allclose(r.transpose(0,2,1)@r,np.eye(3),atol=1e-5,rtol=0)
            or not np.allclose(np.linalg.det(r),1.,atol=1e-5,rtol=0)):
        raise ValueError('proper SO(3) proposal rotations required')
    for key in ('rotation_6d','rotation_6d_raw'):
        if key in output:_array(output[key],(8,6),key)
    if 'proposal_logits' in output and np.any(_array(output['proposal_logits'],(8,),'untrained logits')!=0.):
        raise ValueError('pilot logits must stay exactly zero, not learned success scores')
    if output.get('proposal_score_semantics',SCORE_SEMANTICS)!=SCORE_SEMANTICS:
        raise ValueError('explicit untrained pilot score semantics required')
    return p,r,tau


def _xyzw(rotation):
    q=Rotation.from_matrix(rotation).as_quat()
    first=next((x for x in q[:3] if abs(x)>1e-12),0.)
    if q[3]<0 or (abs(q[3])<=1e-12 and first<0):q=-q
    return q.tolist()


class CandidateLedger:
    """Fixed object-local poses and absolute deadlines until all slots expire.

    A snapshot mask only states domain/caller-qualification availability; it is
    not a reachability or success certificate. Every downstream consumer must
    honor ``available_mask`` before evaluating a slot at its unchanged time.
    World targets come from a fresh GRU at that fixed absolute deadline.
    """
    def __init__(self):self.reset()
    def reset(self):
        self.generation=-1;self._candidates=[];self._events=[];self.last_time=None
        self._requested_generation=None
    @property
    def candidates(self):return deepcopy(self._candidates)
    @property
    def events(self):return deepcopy(self._events)
    def _clock(self,now):
        if isinstance(now,(bool,np.bool_)) or not np.isscalar(now) or not math.isfinite(now) or now<0:
            raise ValueError('finite nonnegative actual anchor/control time required')
        now=float(now)
        if self.last_time is not None and now<self.last_time-_EPS:
            raise ValueError('candidate ledger time cannot move backwards')
        self.last_time=now;return now
    def _expired(self,now):
        return bool(self._candidates and now>max(c['deadline'] for c in self._candidates)+_EPS)
    def new_generation(self,t0,model_output):
        now=self._clock(t0)
        if self._candidates and not self._expired(now):
            raise ValueError('active generation poses/times cannot be replaced or re-windowed')
        p,r,tau=_proposal_arrays(model_output)
        retired=[dict(id=c['id'],deadline=c['deadline']) for c in self._candidates]
        self.generation+=1;self._requested_generation=None;rows=[]
        for i in range(8):
            rows.append(dict(id=f'p4_g{self.generation}_slot{i}',object_tcp_position=p[i].tolist(),
                object_tcp_quaternion=_xyzw(r[i]),deadline=now+float(tau[i]),relative_time_seconds=float(tau[i]),
                generation=self.generation,prediction_anchor_time=now,
                approach_direction=list(EXPLICIT_PRIORS['approach_direction']),
                pregrasp_offset=EXPLICIT_PRIORS['pregrasp_offset_m'],opening=EXPLICIT_PRIORS['opening'],
                required_opening=EXPLICIT_PRIORS['required_opening'],required_width_m=EXPLICIT_PRIORS['required_width_m'],
                allowed_relative_speed=EXPLICIT_PRIORS['allowed_relative_speed_m_s'],
                proposal_logit=0.,proposal_weight=1./8.,proposal_score_semantics=SCORE_SEMANTICS,
                native_pose_qualified=False,physical_execution_qualified=False,
                provenance='P4_single_task_unranked_imitation_proposal_not_success_probability',
                object_pose_source='fresh_GRU_at_this_unchanged_absolute_deadline_supplied_downstream',
                geometry_fallback_used=False))
        self._candidates=rows
        self._events.append(dict(time=now,generation=self.generation,
            reason='initial_model_generation' if not retired else 'all_fixed_deadlines_expired_new_model_generation',
            retired=retired,old_deadlines_modified=False,warm_start_clear_required=True,geometry_fallback_used=False))
        return self._snapshot(now)
    def update_expiry(self,now,model_output=None,*,forecast_end=None,feasible_mask=None):
        now=self._clock(now)
        required=not self._candidates or self._expired(now)
        if model_output is not None:
            if not required:raise ValueError('new model output cannot roll an active generation forward')
            self.new_generation(now,model_output)
        elif required and self._requested_generation!=self.generation:
            self._events.append(dict(time=now,generation=self.generation,reason='current_model_output_required_for_new_generation',
                old_deadlines_modified=False,all_proposals_masked=True,geometry_fallback_used=False))
            self._requested_generation=self.generation
        return self._snapshot(now,forecast_end=forecast_end,feasible_mask=feasible_mask)
    def _snapshot(self,now,*,forecast_end=None,feasible_mask=None):
        if forecast_end is not None:
            if isinstance(forecast_end,(bool,np.bool_)) or not math.isfinite(forecast_end) or forecast_end<now-_EPS:
                raise ValueError('finite current Forecast domain ending at/after now required')
            forecast_end=float(forecast_end)
        n=len(self._candidates)
        if feasible_mask is None:feasible=np.ones(n,dtype=bool)
        else:
            feasible=np.asarray(feasible_mask)
            if feasible.shape!=(n,) or feasible.dtype!=np.bool_:raise ValueError('explicit bool[K] caller qualification mask required')
        rows=deepcopy(self._candidates);mask=np.ones(n,dtype=bool);reasons=[]
        for i,c in enumerate(rows):
            why=[]
            if c['deadline']<now-_EPS:why.append('fixed_deadline_expired')
            if forecast_end is not None and c['deadline']>forecast_end+_EPS:why.append('outside_current_forecast_domain')
            if not feasible[i]:why.append('caller_unreachable_or_unqualified')
            mask[i]=not why;c['proposal_available']=bool(mask[i]);c['availability_reasons']=why;reasons.append(why)
        return dict(generation=self.generation,candidates=rows,available_mask=mask,mask_reasons=reasons,
            new_generation_required=not rows or self._expired(now),
            current_time=now,current_forecast_end=forecast_end,forecast_domain_checked=forecast_end is not None,
            caller_qualification_checked=feasible_mask is not None,physical_execution_qualified=False,
            model_checkpoint_or_feature_provenance_authenticated=False,
            caller_must_supply_current_causal_model_output_at_each_new_generation=True,
            geometry_fallback_used=False,events=deepcopy(self._events))
