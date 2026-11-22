"""One predeclared43-feature ridge fit; fit-parent statistics only.

No optimization runs on import. ``fit_parameters`` is invoked explicitly only
after new TRAIN sources pass admission. Internal TRAIN validation is reporting
only; it does not choose lambda, features, a checkpoint, or a normalization.
"""
from collections import Counter,defaultdict
from copy import deepcopy

import numpy as np

from ..local_mpc_control_20261103.coupled_response import (
    FrozenCoupledResponse,FEATURE_ORDER,SCHEMA as BASE_SCHEMA)

SCHEMA='onpolicy_TRAIN_fixed43_parent_phase_balanced_ridge_20261103_v1'
DT=.05


def features(q,dq,refs):
    q=np.asarray(q,float);dq=np.asarray(dq,float);refs=np.asarray(refs,float)
    if q.ndim!=2 or q.shape[1]!=7 or dq.shape!=q.shape or refs.shape!=(len(q),21):
        raise ValueError('actual q/dq[N,7] and dispatched reference[N,21] required')
    if not all(np.isfinite(v).all() for v in (q,dq,refs)):raise ValueError('finite hardware/reference features required')
    return np.c_[refs[:,:7]-q,dq*DT,refs[:,7:14]*DT,refs[:,14:]*DT*DT,np.sin(q),np.cos(q)]


def balanced_rows(episodes,fit_parents):
    """Parent uniform; probe/nonprobe halves when both really exist."""
    fit=[e for e in episodes if e['role']=='fit'];expected=set(fit_parents)
    if len(expected)!=6 or {e['parent_id'] for e in fit}!=expected:
        raise ValueError('exact six preassigned fit parents required')
    counts=Counter()
    for e in fit:
        mask=np.asarray(e['probe_mask'])
        if mask.shape!=(len(e['refs']),) or mask.dtype!=np.bool_:raise ValueError('actual public phase mask required')
        if not len(mask):raise ValueError('empty parent cannot acquire synthetic training mass')
        for phase in (False,True):counts[e['parent_id'],phase]+=int((mask==phase).sum())
    weights=[];ledger=[]
    for e in fit:
        parent=e['parent_id'];available=sum(counts[parent,p]>0 for p in (False,True))
        value=np.asarray([1./(len(expected)*available*counts[parent,bool(p)]) for p in e['probe_mask']])
        weights.append(value)
        ledger.append(dict(episode_id=e['episode_id'],parent_id=parent,ticks=len(value),
            total_weight=float(value.sum()),probe_weight=float(value[e['probe_mask']].sum()),
            nonprobe_weight=float(value[~e['probe_mask']].sum())))
    w=np.concatenate(weights)
    if not np.isclose(w.sum(),1.,rtol=0,atol=1e-12):raise AssertionError('parent/phase probability mass must total1')
    return fit,w,ledger


def fit_parameters(episodes,protocol,*,source_identity):
    """Single explicit least-squares fit; never reads internal-val arrays."""
    if protocol.get('ridge')!=1e-3 or protocol.get('native_dt')!=DT:
        raise ValueError('predeclared ridge1e-3 and physical50ms required')
    expected=protocol['fit_parent_ids'];internal=set(protocol['internal_val_parent_ids'])
    if set(expected)&internal:raise ValueError('fit/internal parent leakage')
    if any(e['parent_id'] in internal for e in episodes if e['role']=='fit'):
        raise ValueError('internal validation parent cannot be fitted')
    fit,w,ledger=balanced_rows(episodes,expected)
    x=np.concatenate([features(e['q'][:-1],e['dq'][:-1],e['refs']) for e in fit])
    y=np.concatenate([np.diff(e['q'],axis=0) for e in fit])
    if y.shape!=(len(x),7) or not np.isfinite(y).all():raise ValueError('actual measured delta-q labels required')
    mean=np.sum(w[:,None]*x,axis=0)
    scale=np.maximum(np.sqrt(np.sum(w[:,None]*(x-mean)**2,axis=0)),1e-6)
    design=np.c_[(x-mean)/scale,np.ones(len(x))]
    penalty=np.eye(43)*1e-3;penalty[-1,-1]=0.
    matrix=design.T@(w[:,None]*design)+penalty
    coefficient=np.linalg.solve(matrix,design.T@(w[:,None]*y))
    return dict(schema=BASE_SCHEMA,onpolicy_schema=SCHEMA,
        feature_order=deepcopy(FEATURE_ORDER),coefficient_scaled=coefficient.tolist(),
        normalization=dict(fit_split='train',fit_role='fit_parents_only',mean=mean.tolist(),scale=scale.tolist(),std_floor=1e-6),
        hyperparameters=dict(ridge=1e-3,dt=DT,input_dimension=43,output_dimension=7,intercept_penalty=0.),
        fit_source=dict(parameter_fit_split='train',fit_parent_ids=sorted(expected),
            internal_val_parent_ids=sorted(internal),internal_val_used_for_fit_or_selection=False,
            transitions=len(x),parents=len(expected),weighting='equal parent; probe/nonprobe equal halves when both exist',
            parent_phase_weight_ledger=ledger,source_identity=deepcopy(source_identity)),
        parameter_selection='single predeclared fit; internal TRAIN validation reporting only',
        condition_number=float(np.linalg.cond(matrix)),dq_contract='(q_next-q_current)/.05',
        formal_TEST_used=False,official_or_development_VAL_used_for_fitting=False,
        old_frozen_B_or_NN_changed=False,contact_or_gripper_model_fitted=False,
        source_torque_or_physical_parameter_claim=False,globally_qualified=False)


class OnPolicyCoupledResponse(FrozenCoupledResponse):
    """Numerically identical43-feature inference, with a new parameter identity."""
    def __init__(self,parameters,*,parameter_sha256=None):
        if (parameters.get('onpolicy_schema')!=SCHEMA
                or parameters.get('official_or_development_VAL_used_for_fitting') is not False
                or parameters.get('parameter_selection')!='single predeclared fit; internal TRAIN validation reporting only'):
            raise ValueError('explicit new TRAIN-only single-fit on-policy identity required')
        super().__init__(parameters,parameter_sha256=parameter_sha256)
        self.evidence.update(onpolicy_schema=SCHEMA,internal_TRAIN_validation_reporting_only=True,
            learned_finger_response_included=False,actual_dynamic_TCP_FK_residual_qualified=False)
