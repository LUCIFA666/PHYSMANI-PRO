"""Independent frozen P4 feature/candidate sensitivity, never policy admission.

Only public robot/geometry and already frozen forecasts/contexts are inputs.
No observed future pose, evaluation label, Native state, or model fitting is
used. Same query indices are compared; they are not matched to a best target.
"""
from pathlib import Path
from collections import Counter
import json

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from .contracts import Forecast
from .local_mpc_p4_20261103.data import public_features, rot6
from .local_mpc_p4_20261103.model import CandidateTransformerPilot
from .context_gap_dropout_20261103_v1 import compatibility as frozen_helper
from .context_gap_dropout_20261103_v1.data import ROOT, sha
from .context_physical_eval_20261103_v1.evaluation import validate_freeze, source_map as physical_sources, VARIANTS

VERSION = 'TRAIN16_frozen_P4_context_student_candidate_AB_20261103_v1'
PHYSICAL = ROOT/'artifacts/framework/local_mpc_20261102/P4_TRAIN16_context_student_physical_eval_v1'
TF_CHECKPOINT = ROOT/'artifacts/framework/local_mpc_20261102/P4_candidate_training_v1/final.pt'
TF_SHA = '61e5827b4838649b8e477048bea56fcace18697f906fb610a3f963bba2437f60'
BATCH_SIZE = 32


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def sources():
    names = ('our_mani/context_candidate_compatibility_20261103_v1.py',
             'scripts/evaluate_context_candidate_compatibility_20261103_v1.py',
             'our_mani/local_mpc_p4_20261103/data.py')
    return {**physical_sources(), **{str(ROOT/name):sha(ROOT/name) for name in names}}


def predicted_features(reference, current_public_position, predictions, anchor_index, variant_index):
    """Exact public_features units: world relative translation, column rot6."""
    current = np.asarray(current_public_position, dtype=float)
    p = predictions['positions'][anchor_index, variant_index].astype(float)
    r = predictions['rotations'][anchor_index, variant_index].astype(float)
    v = predictions['velocities'][anchor_index, variant_index].astype(float)
    w = predictions['angular_velocities'][anchor_index, variant_index].astype(float)
    if current.shape != (3,) or not np.allclose(p[0], current, atol=1e-6, rtol=0):
        raise ValueError('Frozen forecast must retain this exact current public position anchor')
    result = {key:value.copy() for key,value in reference.items()}
    # Reuse the original relative time feature exactly; never infer clock from
    # HDF indexing. The prediction source already binds actual public clocks.
    result['future_tokens'] = np.concatenate((reference['future_tokens'][:, :1].astype(float),
        p-current, rot6(r), v, w), axis=1).astype(np.float32)
    result['context_c'] = predictions['contexts'][anchor_index, variant_index].astype(np.float32).copy()
    frozen_helper._checked_pair(reference, result)
    return result


def reconstruction_difference(reference, reconstructed):
    return {key:dict(exact=bool(np.array_equal(reference[key], reconstructed[key])),
        max_abs=float(np.max(np.abs(reference[key].astype(float)-reconstructed[key])))) for key in reference}


def prepare(output):
    freeze, windows, cases, predictions = validate_freeze(PHYSICAL)
    method_sources = sources()
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    shapes = dict(future_tokens=(969,6,17,16), context_c=(969,6,16), robot=(969,25), geometry=(969,4), task_e=(969,16),
                  reference_future_tokens=(969,17,16), reference_context_c=(969,16))
    features = {key:np.zeros(shape,np.float32) for key,shape in shapes.items()}
    available = np.zeros(969,bool)
    ledger = []
    for index, window in enumerate(windows):
        parent = window.parent[0]
        history = cases[parent]['public_rows'][:window.anchor+1]
        row = history[-1]
        record = dict(index=index,parent_id=parent,scene=cases[parent]['scene'],anchor_row=window.anchor,
                      anchor_time=float(window.times[-1]),bundle_available=False)
        try:
            reference, provenance = public_features(row, history)
            if not np.allclose(provenance['real_history_timestamps'], window.times, atol=1e-8, rtol=0):
                raise ValueError('Frozen physical and original P4 dense history clocks differ')
            variants = [predicted_features(reference,row['result']['object_observation']['pose']['position'],predictions,index,vi)
                        for vi in range(6)]
            for key in ('robot','geometry','task_e'):
                features[key][index] = reference[key]
                if any(not np.array_equal(value[key],reference[key]) for value in variants):
                    raise ValueError('Paired immutable P4 features changed')
            features['reference_future_tokens'][index] = reference['future_tokens']
            features['reference_context_c'][index] = reference['context_c']
            for vi,value in enumerate(variants):
                features['future_tokens'][index,vi] = value['future_tokens']
                features['context_c'][index,vi] = value['context_c']
            available[index] = True
            record.update(bundle_available=True,teacher_dense_reconstruction=reconstruction_difference(reference,variants[0]),
                          original_P4_feature_provenance=provenance)
        except (ValueError,TypeError,KeyError) as exc:
            # Preserve the complete969 ledger; no favourable subset or success
            # filtering. Missing bundles never become zero-error comparisons.
            record['reason'] = type(exc).__name__+': '+str(exc)
        ledger.append(record)
    features['available_mask'] = available
    np.savez_compressed(output/'frozen_feature_bundles.npz', **features)
    write(output/'all_anchor_ledger.json', ledger)
    if method_sources != sources():
        raise ValueError('Candidate diagnostic sources changed during preparation')
    fields = ('future_tokens','robot','geometry','context_c','task_e')
    reconstruction = {key:dict(available_anchors=int(available.sum()),
        exact_anchors=sum(record['teacher_dense_reconstruction'][key]['exact'] for record in ledger if record['bundle_available']),
        max_abs=max((record['teacher_dense_reconstruction'][key]['max_abs'] for record in ledger if record['bundle_available']),default=None)) for key in fields}
    request = dict(version=VERSION,complete=True,source_sha256=method_sources,
        physical_prediction_freeze=dict(path=str(PHYSICAL/'prediction_freeze.json'),sha256=sha(PHYSICAL/'prediction_freeze.json')),
        TF_checkpoint=dict(path=str(TF_CHECKPOINT),sha256=TF_SHA),variants=list(VARIANTS),batch_size=BATCH_SIZE,
        all_parent_ids=freeze['all_parent_ids'],all_anchors=969,available_feature_bundles=int(available.sum()),
        missing_feature_bundles=int((~available).sum()),missing_reasons=dict(Counter(record.get('reason') for record in ledger if not record['bundle_available'])),
        teacher_dense_vs_recorded_public_features=reconstruction,
        files={name:dict(path=str(output/name),sha256=sha(output/name)) for name in ('frozen_feature_bundles.npz','all_anchor_ledger.json')},
        actual_model_forward_calls=0,observed_future_or_GT_used_as_forward_input=False,
        generic_or_runtime_qualification=False,Native_started=False,TEST_read=False)
    write(output/'feature_freeze_before_TF.json',request)
    return request


def validate_prepared(output):
    output=Path(output).resolve(strict=True)
    request=json.loads((output/'feature_freeze_before_TF.json').read_text())
    if (request.get('version')!=VERSION or request.get('complete')is not True or request.get('all_anchors')!=969
            or request.get('variants')!=list(VARIANTS) or request.get('batch_size')!=32
            or request.get('source_sha256')!=sources() or request.get('TF_checkpoint')!=dict(path=str(TF_CHECKPOINT),sha256=TF_SHA)
            or set(request.get('files',{}))!={'frozen_feature_bundles.npz','all_anchor_ledger.json'}):
        raise ValueError('Exact fixed full-cohort prepared feature freeze required')
    for name,pointer in request['files'].items():
        if Path(pointer['path']).resolve()!=output/name or sha(pointer['path'])!=pointer['sha256']:
            raise ValueError('Feature bundle changed or borrowed from a different output')
    if sha(request['physical_prediction_freeze']['path'])!=request['physical_prediction_freeze']['sha256']:
        raise ValueError('Frozen forecast source changed')
    freeze,windows,cases,predictions=validate_freeze(PHYSICAL)
    ledger=json.loads((output/'all_anchor_ledger.json').read_text())
    if len(ledger)!=969 or request['all_parent_ids']!=freeze['all_parent_ids']:
        raise ValueError('All969 anchors/all16 parents required')
    for index,(record,window) in enumerate(zip(ledger,windows)):
        if (record['index']!=index or record['parent_id']!=window.parent[0] or record['anchor_row']!=window.anchor
                or record['anchor_time']!=float(window.times[-1]) or type(record['bundle_available'])is not bool):
            raise ValueError('Exact ordered frozen anchor identity required')
    with np.load(output/'frozen_feature_bundles.npz',allow_pickle=False)as stored:
        arrays={name:np.array(stored[name])for name in stored.files}
    if arrays['available_mask'].dtype!=np.bool_ or arrays['available_mask'].shape!=(969,) or not np.array_equal(arrays['available_mask'],[r['bundle_available']for r in ledger]):
        raise ValueError('Explicit full-cohort feature availability mask required')
    return request,ledger,arrays,windows,predictions


def checked_pair_outputs(model,teacher,student):
    """Original frozen helper guards/math, additionally return both outputs.

    No hooks/monkeypatching and no repeated forward for world-pose extraction.
    The only difference from compare_candidate_features is retaining outA/B.
    """
    a,b,batch,batched=frozen_helper._checked_pair(teacher,student)
    before=frozen_helper._validate_model(model)
    a_sha,b_sha=frozen_helper._digest_arrays(a),frozen_helper._digest_arrays(b)
    out_a,out_b=frozen_helper._forward_pair(model,a,b,batched)
    summary=frozen_helper.summarize_candidate_outputs(out_a,out_b)
    if (summary['batch_size']!=batch or frozen_helper._validate_model(model)!=before
            or frozen_helper._digest_arrays(a)!=a_sha or frozen_helper._digest_arrays(b)!=b_sha):
        raise ValueError('Frozen TF state/features mutated')
    return out_a,out_b,summary


def world_candidates(forecast_arrays,index,variant,anchor_time,local_positions,local_rotations,tau):
    f=Forecast(('target',),anchor_time+np.arange(17)*.05,
        forecast_arrays['positions'][index,variant,:,None,:],forecast_arrays['rotations'][index,variant,:,None,:,:],
        forecast_arrays['velocities'][index,variant,:,None,:],forecast_arrays['angular_velocities'][index,variant,:,None,:])
    world_p,world_R=[],[]
    for p,R,t in zip(local_positions,local_rotations,tau):
        object_pose,_,_=f.at('target',anchor_time+float(t))
        world_p.append(object_pose.apply(p));world_R.append(object_pose.rotation@R)
    return np.asarray(world_p),np.asarray(world_R)


def stats(values):
    a=np.asarray(values,float)
    return dict(count=int(a.size),mean=float(a.mean()),p95=float(np.quantile(a,.95)),max=float(a.max())) if a.size else dict(count=0)


def run(output):
    request,ledger,features,windows,predictions=validate_prepared(output)
    output=Path(output).resolve()
    write(output/'TF_run_request.json',dict(feature_freeze_sha256=sha(output/'feature_freeze_before_TF.json'),
        source_sha256=sources(),batch_size=32,CPU_threads=2,TF_checkpoint=request['TF_checkpoint'],qualification=False))
    if sha(TF_CHECKPOINT)!=TF_SHA:
        raise ValueError('Exact original frozen TF checkpoint required')
    torch.set_num_threads(2)
    payload=torch.load(TF_CHECKPOINT,map_location='cpu',weights_only=False)
    model=CandidateTransformerPilot().eval().requires_grad_(False)
    if payload['model_spec']!=model.spec():raise ValueError('Original P4 architecture specification required')
    model.load_state_dict(payload['model_state'],strict=True)
    model_sha=frozen_helper._validate_model(model)
    valid=np.flatnonzero(features['available_mask'])
    values=dict(local_positions=np.zeros((969,6,8,3),np.float32),local_rotations=np.zeros((969,6,8,3,3),np.float32),
        relative_times=np.zeros((969,6,8),np.float32),world_positions=np.zeros((969,6,8,3)),world_rotations=np.zeros((969,6,8,3,3)))
    forward_calls=0
    for begin in range(0,len(valid),32):
        indices=valid[begin:begin+32]
        for ai,bi in ((0,1),(2,3),(4,5)):
            def bundle(vi):
                return {key:(features[key][indices,vi]if key in ('future_tokens','context_c')else features[key][indices])for key in frozen_helper.FEATURE_SHAPES}
            out_a,out_b,_=checked_pair_outputs(model,bundle(ai),bundle(bi));forward_calls+=2
            for vi,out in ((ai,out_a),(bi,out_b)):
                values['local_positions'][indices,vi]=out['positions'].numpy()
                values['local_rotations'][indices,vi]=out['rotation_matrices'].numpy()
                values['relative_times'][indices,vi]=out['times'].numpy()
    for index in valid:
        for vi in range(6):
            p,R=world_candidates(predictions,index,vi,float(windows[index].times[-1]),
                values['local_positions'][index,vi],values['local_rotations'][index,vi],values['relative_times'][index,vi])
            values['world_positions'][index,vi]=p;values['world_rotations'][index,vi]=R
    if frozen_helper._validate_model(model)!=model_sha or sources()!=request['source_sha256']:
        raise ValueError('Model/source changed during offline TF evaluation')
    np.savez_compressed(output/'frozen_candidate_outputs.npz',available_mask=features['available_mask'],**values)
    comparison={}
    for ai,bi,name in ((0,1,'dense'),(2,3,'isolated'),(4,5,'burst')):
        p=values['local_positions'][valid,bi].astype(float)-values['local_positions'][valid,ai]
        R=np.swapaxes(values['local_rotations'][valid,ai].astype(float),-2,-1)@values['local_rotations'][valid,bi]
        tau=values['relative_times'][valid,bi].astype(float)-values['relative_times'][valid,ai]
        wp=values['world_positions'][valid,bi]-values['world_positions'][valid,ai]
        wr=np.swapaxes(values['world_rotations'][valid,ai],-2,-1)@values['world_rotations'][valid,bi]
        local_angle=Rotation.from_matrix(R.reshape(-1,3,3)).magnitude().reshape(len(valid),8)if len(valid)else np.empty((0,8))
        world_angle=Rotation.from_matrix(wr.reshape(-1,3,3)).magnitude().reshape(len(valid),8)if len(valid)else np.empty((0,8))
        comparison[name]=dict(local_position_change_m=stats(np.linalg.norm(p,axis=-1)),local_rotation_change_rad=stats(local_angle),
            relative_time_change_seconds=stats(np.abs(tau)),world_handoff_position_change_m=stats(np.linalg.norm(wp,axis=-1)),
            world_handoff_rotation_change_rad=stats(world_angle),per_query=[dict(query=q,local_position_change_m=stats(np.linalg.norm(p[:,q],axis=-1)),
                tau_abs_change_seconds=stats(np.abs(tau[:,q])),world_handoff_position_change_m=stats(np.linalg.norm(wp[:,q],axis=-1)))for q in range(8)])
    per_parent={}
    for parent in request['all_parent_ids']:
        own=np.array([i for i in valid if ledger[i]['parent_id']==parent],dtype=int)
        per_parent[parent]=dict(total_anchors=sum(r['parent_id']==parent for r in ledger),available_bundles=len(own),
            world_handoff_changes_m={name:stats(np.linalg.norm(values['world_positions'][own,bi]-values['world_positions'][own,ai],axis=-1))
                for ai,bi,name in ((0,1,'dense'),(2,3,'isolated'),(4,5,'burst'))})
    report=dict(version=VERSION,parents=16,all_anchors=969,available_bundles=len(valid),missing_bundles=969-len(valid),
        missing_reasons=request['missing_reasons'],teacher_dense_reconstruction=request['teacher_dense_vs_recorded_public_features'],
        comparisons=comparison,per_parent=per_parent,actual_TF_forward_batch_calls=forward_calls,
        evaluated_feature_rows=int(len(valid)*6),batch_size=32,CPU_threads=2,GPU_used=False,
        GRU_or_vision_forward_calls=0,source_sha256=sources(),model_state_sha256=model_sha,TF_checkpoint_sha256=TF_SHA,
        input_forecasts='Previously frozen model predictions, never observed future poses or GT',
        paired_features_unchanged=['robot','geometry','task_e'],same_query_indices_not_best_match=True,
        world_handoff_semantics='Each variant own Forecast.at(t0+its_tau), then objectPose.apply(local_point), R_obj@R_local; includes forecast and local candidate changes',
        constant_untrained_logits_not_success_probabilities=True,old_TRAIN16_P4_exposure=True,
        compatibility_qualified=False,live_runtime_admitted=False,Native_started=False,training_started=False,TEST_read=False,
        artifact_sha256={name:sha(output/name)for name in ('feature_freeze_before_TF.json','TF_run_request.json','frozen_candidate_outputs.npz')})
    write(output/'report.json',report)
    return report
