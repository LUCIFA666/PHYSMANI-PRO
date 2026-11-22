"""Offline P4 sensitivity to forecast/context changes, never admission.

No public rows, tracker, checkpoint loader or live controller are constructed.
The caller supplies the already frozen CPU P4 model and genuine feature
bundles. Missing bundles are explicitly not_evaluated; no candidates are
invented. Successful diagnostics do not certify a checkpoint or input source.
"""
import hashlib
from pathlib import Path

import numpy as np
import torch
from torch.nn.modules import module as nn_module

from ..local_mpc_p4_20261103.model import (
    CandidateTransformerPilot, POSITION_BOUND_M, TIME_MIN_SECONDS,
    TIME_MAX_SECONDS, SCORE_SEMANTICS)

VERSION='offline_frozen_P4_context_gap_feature_sensitivity_20261103_v1'
FEATURE_SHAPES={'future_tokens':(17,16),'robot':(25,),'geometry':(4,),
                'context_c':(16,),'task_e':(16,)}
IMMUTABLE_FEATURES=('robot','geometry','task_e')
MODEL_SOURCE=Path(__file__).resolve().parents[1]/'local_mpc_p4_20261103/model.py'
MODEL_SOURCE_SHA256='a41b98ae09505ca6892e0edd0e317e1a1eb27179ad5cda3d30d4089162f81eb4'
_FORWARD=CandidateTransformerPilot.forward
_SPEC=CandidateTransformerPilot.spec
OUTPUT_KEYS=frozenset(('positions','rotation_matrices','rotation_6d',
    'rotation_6d_raw','times','proposal_logits','proposal_score_semantics'))


def _array(value,name):
    if isinstance(value,torch.Tensor):
        if value.device.type!='cpu' or value.requires_grad or value.dtype!=torch.float32:
            raise ValueError(name+' requires detached CPU float32, no device conversion')
        value=value.detach().numpy()
    if not isinstance(value,np.ndarray) or value.dtype!=np.float32 or not np.isfinite(value).all():
        raise ValueError(name+' requires finite float32 feature/output arrays')
    return value.copy()


def _digest_arrays(values):
    h=hashlib.sha256()
    for name in sorted(values):
        a=values[name];h.update((name+str(a.shape)+str(a.dtype)).encode());h.update(a.tobytes())
    return h.hexdigest()


def _checked_pair(teacher,student):
    pair=[];batched=None;batch=None
    for label,features in (('teacher',teacher),('student',student)):
        if not isinstance(features,dict) or set(features)!=set(FEATURE_SHAPES):
            raise ValueError('Exact five P4 feature keys required; no rows/labels/state metadata')
        arrays={k:_array(v,label+'/'+k)for k,v in features.items()}
        future=arrays['future_tokens'];is_batch=future.ndim==3
        n=future.shape[0]if is_batch else 1
        if n<1 or future.ndim not in (2,3):raise ValueError('Nonempty original P4 feature shape required')
        for name,shape in FEATURE_SHAPES.items():
            expected=(n,)+shape if is_batch else shape
            if arrays[name].shape!=expected:raise ValueError('Exact P4 feature shape required: '+name)
        if batched is not None and (batched!=is_batch or batch!=n):
            raise ValueError('Teacher/student batch and shapes must match exactly')
        batched,batch=is_batch,n
        view=arrays if is_batch else {k:v[None]for k,v in arrays.items()}
        if (not np.allclose(view['future_tokens'][:,:,0],np.arange(17)[None]/16.,rtol=0,atol=1e-5)
            or not np.allclose(view['future_tokens'][:,0,1:4],0.,rtol=0,atol=1e-5)):
            raise ValueError('Original relative clock and current position anchor required')
        if (np.any(view['geometry'][:,:3]<=0)or np.any((view['geometry'][:,3]<0)|(view['geometry'][:,3]>1))
            or np.any((view['robot'][:,-2:]<0)|(view['robot'][:,-2:]>1))):
            raise ValueError('Original geometry and normalized finger features required')
        for key in ('context_c','task_e'):
            if np.any(view[key]<0)or not np.allclose(view[key].reshape(n,2,8).sum(-1),1.,rtol=0,atol=1e-4):
                raise ValueError('Original two SimNorm groups required: '+key)
        pair.append(arrays)
    for key in IMMUTABLE_FEATURES:
        if not np.array_equal(pair[0][key],pair[1][key]):
            raise ValueError(key+' must remain value-identical; only future_tokens/context_c may differ')
    return pair[0],pair[1],batch,batched


def _model_digest(model):
    h=hashlib.sha256()
    for name,value in model.state_dict().items():
        if value.device.type!='cpu' or value.dtype!=torch.float32 or not torch.isfinite(value).all():
            raise ValueError('Finite frozen CPU float32 P4 tensors required')
        a=value.detach().numpy();h.update((name+str(a.shape)+str(a.dtype)).encode());h.update(a.tobytes())
    return h.hexdigest()


def _validate_model(model):
    if type(model)is not CandidateTransformerPilot:
        raise TypeError('Exact frozen CandidateTransformerPilot required; no subclass/testdouble')
    if (getattr(model.forward,'__func__',None)is not _FORWARD
        or CandidateTransformerPilot.forward is not _FORWARD
        or getattr(model.spec,'__func__',None)is not _SPEC):
        raise ValueError('Original P4 forward/spec methods required')
    if hashlib.sha256(MODEL_SOURCE.read_bytes()).hexdigest()!=MODEL_SOURCE_SHA256:
        raise ValueError('Frozen P4 model implementation source changed')
    if nn_module._global_forward_hooks or nn_module._global_forward_pre_hooks:
        raise ValueError('No injected global P4 hooks')
    modules=list(model.modules());parameters=list(model.parameters())
    if (not parameters or any(p.requires_grad or p.device.type!='cpu' for p in parameters)
        or any(m.training or m._forward_hooks or m._forward_pre_hooks for m in modules)):
        raise ValueError('Frozen eval CPU P4 without injected hooks required')
    expected=dict(d_model=128,num_layers=2,num_heads=4,num_candidates=8,dropout=0.,
                  memory_tokens=21,query_tokens=8,batch_first=True)
    if model.spec()['architecture']!=expected:raise ValueError('Original pilot architecture required')
    return _model_digest(model)


def _forward_pair(model,teacher,student,batched):
    def tensors(values):
        return {k:torch.from_numpy(v if batched else v[None])for k,v in values.items()}
    with torch.inference_mode():
        return model(**tensors(teacher)),model(**tensors(student))


def _outputs(value):
    if not isinstance(value,dict)or set(value)!=OUTPUT_KEYS or value['proposal_score_semantics']!=SCORE_SEMANTICS:
        raise ValueError('Exact original unranked P4 outputs required')
    arrays={k:_array(v,'candidate/'+k)for k,v in value.items()if k!='proposal_score_semantics'}
    n=arrays['positions'].shape[0]if arrays['positions'].ndim==3 else 0
    shapes=dict(positions=(n,8,3),rotation_matrices=(n,8,3,3),rotation_6d=(n,8,6),
                rotation_6d_raw=(n,8,6),times=(n,8),proposal_logits=(n,8))
    if n<1 or any(arrays[k].shape!=shape for k,shape in shapes.items()):
        raise ValueError('Exact nonempty K8 candidate shapes required')
    R=arrays['rotation_matrices'].astype(np.float64)
    if (not np.allclose(np.swapaxes(R,-2,-1)@R,np.eye(3),rtol=0,atol=1e-4)
        or np.any(np.linalg.det(R)<1.-1e-4)
        or np.any(np.abs(arrays['positions'])>POSITION_BOUND_M+1e-6)
        or np.any((arrays['times']<TIME_MIN_SECONDS)|(arrays['times']>TIME_MAX_SECONDS))
        or np.any(arrays['proposal_logits']!=0)):
        raise ValueError('Original bounded pose/time and constant untrained logits required')
    return arrays


def summarize_candidate_outputs(teacher_output,student_output):
    """Pure same-query-index comparison; no candidate matching or probability."""
    a,b=_outputs(teacher_output),_outputs(student_output)
    if any(a[k].shape!=b[k].shape for k in a):raise ValueError('Candidate batch/shape mismatch')
    dp=b['positions'].astype(np.float64)-a['positions']
    relative=np.swapaxes(a['rotation_matrices'].astype(np.float64),-2,-1)@b['rotation_matrices']
    angle=np.arccos(np.clip((np.trace(relative,axis1=-2,axis2=-1)-1.)/2.,-1.,1.))
    dt=b['times'].astype(np.float64)-a['times']
    distance=np.linalg.norm(dp,axis=-1)
    return dict(batch_size=len(dp),candidate_query_count=8,same_query_index_comparison=True,
        object_local_position_delta_m=dp.tolist(),position_distance_m=distance.tolist(),
        rotation_geodesic_rad=angle.tolist(),relative_tau_delta_seconds=dt.tolist(),
        max_position_distance_m=float(distance.max()),max_rotation_geodesic_rad=float(angle.max()),
        max_abs_tau_delta_seconds=float(np.abs(dt).max()),proposal_logits_constant_zero=True,
        calibrated_success_probability=False,world_pose_or_action_equivalence_claimed=False)


def not_evaluated_report(reason):
    return dict(version=VERSION,status='not_evaluated',reason=str(reason),
        model_forward_calls=0,compatibility_qualified=False,production_admission=False,
        real_selection_compatibility_claimed=False,Native_started=False,TEST_read=False)


def compare_candidate_features(model,teacher_features,student_features):
    """Evaluate exactly one supplied A/B pair, without choosing a passing gate.

    Bundles may be unbatched original public_features arrays or a common batch.
    The returned tensor digest identifies the caller's model; it does not
    authenticate checkpoint training/source or TRAIN selection provenance.
    """
    if teacher_features is None or student_features is None:
        return not_evaluated_report('real_teacher_or_student_feature_bundle_missing')
    teacher,student,batch,batched=_checked_pair(teacher_features,student_features)
    if model is None:return not_evaluated_report('frozen_P4_model_not_supplied')
    model_before=_validate_model(model)
    teacher_before=_digest_arrays(teacher);student_before=_digest_arrays(student)
    out_a,out_b=_forward_pair(model,teacher,student,batched)
    comparison=summarize_candidate_outputs(out_a,out_b)
    if (comparison['batch_size']!=batch or _validate_model(model)!=model_before
        or _digest_arrays(teacher)!=teacher_before or _digest_arrays(student)!=student_before):
        raise ValueError('P4 state or copied feature arrays changed during diagnostic')
    return dict(version=VERSION,status='evaluated_offline',model_forward_calls=2,
        model_state_sha256=model_before,model_source_sha256=MODEL_SOURCE_SHA256,
        teacher_feature_sha256=teacher_before,student_feature_sha256=student_before,
        identical_feature_keys_and_shapes=True,unchanged_features=list(IMMUTABLE_FEATURES),
        variable_features=['future_tokens','context_c'],comparison=comparison,
        compatibility_qualified=False,production_admission=False,
        checkpoint_and_input_provenance_certified=False,Native_started=False,TEST_read=False)
