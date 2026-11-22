"""One reserved-head development evaluation: public freeze before target score.

No certificate/runtime is produced. A fixed .5 availability diagnostic and
uncalibrated uncertainty scores cannot certify physical or statistical bounds.
"""
from pathlib import Path
from collections import Counter
import hashlib
import json
import numpy as np
from .data import DualCropCubeDataset,load_npz,sha256,checked_pointer
from .training import VERSION,fixed_config,training_source_map,tensor_batch,pointer,write_new,SELECTED_ENCODER_SHA


def stats(values):
    a=np.asarray(values,float)
    return dict(count=len(a),mean_m=float(a.mean()),p95_m=float(np.quantile(a,.95)),max_m=float(a.max()),within3cm=int((a<=.03).sum()))if len(a)else dict(count=0)


def summarize_rows(rows):
    positive=[r for r in rows if r['presence_target']==1];negative=[r for r in rows if r['presence_target']==0];unknown=[r for r in rows if r['presence_target']==-1]
    available=[r for r in rows if r['candidate_available']]
    return dict(total=len(rows),positive=len(positive),negative=len(negative),unknown=len(unknown),
        positive_candidate_available=sum(r['candidate_available']for r in positive),negative_false_available=sum(r['candidate_available']for r in negative),
        unknown_predicted_available=sum(r['candidate_available']for r in unknown),
        known_positive_coverage=None if not positive else sum(r['candidate_available']for r in positive)/len(positive),
        known_availability_precision=None if not any(r['presence_target']>=0 for r in available)else sum(r['presence_target']==1 for r in available)/sum(r['presence_target']>=0 for r in available),
        availability_positive_unknown_count=sum(r['presence_target']==-1 for r in available),
        all_positive_centre_error=stats([r['centre_error_m']for r in positive]),
        available_positive_centre_error=stats([r['centre_error_m']for r in positive if r['candidate_available']]),
        supervision_reasons=dict(Counter(r['target_reason']for r in rows)),uncertainty_is_not_CI=True,qualified=False)


def predict(selected,expected_sha256,manifest,output,encoder_checkpoint):
    c=fixed_config()
    if sha256(selected)!=expected_sha256:raise ValueError('Explicit selected whole-SHA required')
    data=DualCropCubeDataset(manifest,'reserved_fresh_centre_VAL',allow_reserved=True)
    if len(data)!=104:raise ValueError('All8 reserved parents and104shots retained')
    sources=training_source_map();before=data.revalidate()
    out=Path(output).resolve()
    if out.exists():raise FileExistsError('Fresh evaluation output required')
    import torch
    from .model import DualCropCubeCentreHeadV1
    payload=torch.load(selected,map_location='cpu',weights_only=True);meta=payload.get('metadata',{})
    if meta.get('version')!=VERSION or meta.get('training_complete')is not True or meta.get('epochs_completed')!=40 or meta.get('actual_optimizer_steps')!=520 or meta.get('model_qualified')is not False or meta.get('source_sha256')!=sources or meta.get('manifest')!=pointer(manifest):
        raise ValueError('Completed same-source selected pilot40 model required; no old certificate transfer')
    side=Path(selected).resolve().parent;fit_status=json.loads((side/'fit_status.json').read_text())
    if fit_status.get('training_complete')is not True or fit_status.get('actual_optimizer_steps')!=520 or fit_status.get('selected_checkpoint')!=pointer(selected):raise ValueError('Actual completed same-selected fit status required')
    if set(meta.get('evidence',{}))!={'identity','cache_before','cache_after','history','sample_access','source_verification'}:raise ValueError('Complete original fit evidence required')
    for name,ref in meta['evidence'].items():
        if checked_pointer(side,ref)!=side/(name+'.json'):raise ValueError('Original selected evidence sibling identity required')
    if sha256(encoder_checkpoint)!=SELECTED_ENCODER_SHA:raise ValueError('Original encoder initializer changed')
    request=dict(version=VERSION,selected=pointer(selected),manifest=pointer(manifest),method_source_sha256=sources,
        role='reserved_fresh_centre_VAL',parents=8,shots=104,one_attempt=True,failures_consume=True,no_threshold_or_model_selection=True)
    claim=data.path.parent/'reserved_centre_evaluation_consumption.request.json'
    write_new(claim,request)  # Never remove/replace this on infrastructure or quality failure.
    out.mkdir(parents=True,exist_ok=False);write_new(out/'request.json',request)
    torch.set_num_threads(2)
    model=DualCropCubeCentreHeadV1(encoder_trainable=True)
    model.load_selected_v5_encoder(encoder_checkpoint,SELECTED_ENCODER_SHA)
    model.load_state_dict(payload['model_state_dict'],strict=True);model.eval()
    for p in model.parameters():p.requires_grad_(False)
    ledger=[];outputs=[];calls=0
    with torch.no_grad():
        for start in range(0,len(data),c['batch_size']):
            samples=[data.public_sample(i)for i in range(start,min(start+c['batch_size'],len(data)))]
            arrays={k:np.stack([s['model_inputs'][k]for s in samples])for k in samples[0]['model_inputs']}
            batch=tensor_batch(arrays,np.arange(len(samples)),'cpu')
            value=model(**batch);calls+=1
            outputs.append({k:value[k].detach().cpu().numpy()for k in ('centre_world','measurement_availability_logit','uncertainty_score_m','uncertainty_score_valid')})
            ledger.extend(dict(scene_id=s['scene_id'],parent_id=s['parent_id'],role=s['role'])for s in samples)
    arrays={k:np.concatenate([o[k]for o in outputs])for k in outputs[0]}
    if any(not np.isfinite(v).all()for v in arrays.values()):raise ValueError('Finite predictions for all104 required')
    np.savez_compressed(out/'predictions.npz',**arrays);write_new(out/'ordered_scenes.json',ledger)
    data.revalidate()
    if sources!=training_source_map():raise ValueError('Implementation changed during public prediction')
    freeze=dict(version=VERSION,complete=True,selected=pointer(selected),manifest=pointer(manifest),consumption=pointer(claim),
        files=dict(predictions=pointer(out/'predictions.npz'),ordered_scenes=pointer(out/'ordered_scenes.json'),request=pointer(out/'request.json')),
        method_source_sha256=sources,bound_data_sha256=data.bound_pointers,data_before=before,data_after=data.revalidate(),
        parents=8,samples=104,model_forward_batch_calls=calls,model_forward_rows=104,device='cpu',torch_num_threads=2,
        environment=dict(torch=torch.__version__,numpy=np.__version__),reserved_label_array_decodes_before_freeze=0,
        label_proof_metadata_validated_before_prediction=True,label_proof_metadata_not_model_input_or_prediction_selection=True,
        pure_public_current_RGBD_only=True,model_qualified=False,position_measurement_qualified=False,runtime_or_control_admitted=False,
        uncertainty_qualified=False,TEST_read=False,Native_started=False,training_started=False)
    write_new(out/'prediction_freeze.json',freeze)
    return dict(prediction_freeze=pointer(out/'prediction_freeze.json'),model_forward_batch_calls=calls,samples=104)


def validate_freeze(output):
    out=Path(output).resolve(strict=True);freeze=json.loads((out/'prediction_freeze.json').read_text())
    if freeze.get('version')!=VERSION or freeze.get('complete')is not True or freeze.get('samples')!=104 or freeze.get('parents')!=8 or freeze.get('reserved_label_array_decodes_before_freeze')!=0 or freeze.get('device')!='cpu'or freeze.get('torch_num_threads')!=2:
        raise ValueError('Complete public prediction freeze required before labels')
    for ref in [freeze['selected'],freeze['manifest'],freeze['consumption'],*freeze['files'].values()]:checked_pointer(out,ref)
    if freeze['method_source_sha256']!=training_source_map()or any(sha256(p)!=digest for p,digest in freeze['bound_data_sha256'].items()):raise ValueError('Frozen method/data changed')
    arrays=load_npz(freeze['files']['predictions']['path']);ledger=json.loads(Path(freeze['files']['ordered_scenes']['path']).read_text())
    shapes=dict(centre_world=(104,3),measurement_availability_logit=(104,),uncertainty_score_m=(104,3),uncertainty_score_valid=(104,))
    if set(arrays)!=set(shapes)or any(arrays[k].shape!=shape or not np.isfinite(arrays[k]).all()for k,shape in shapes.items())or len(ledger)!=104 or len({r['scene_id']for r in ledger})!=104:raise ValueError('All104 finite ordered frozen outputs required')
    return freeze,arrays,ledger


def score(output):
    freeze,arrays,ledger=validate_freeze(output)  # No label arrays opened before this line.
    write_new(Path(output)/'reserved_score_consumption.request.json',dict(prediction_freeze=pointer(Path(output)/'prediction_freeze.json'),one_attempt=True,failures_consume=True))
    data=DualCropCubeDataset(freeze['manifest']['path'],'reserved_fresh_centre_VAL',allow_reserved=True)
    if [r['scene_id']for r in data.records]!=[r['scene_id']for r in ledger]:raise ValueError('Ordered reserved identities changed')
    rows=[]
    for i in range(len(data)):
        sample=data[i];target=sample['targets'];record=sample['record']
        logit=float(arrays['measurement_availability_logit'][i]);prob=float(np.exp(-np.logaddexp(0.,-logit)))
        valid_score=bool(arrays['uncertainty_score_valid'][i]);available=prob>=.5 and valid_score
        rows.append(dict(scene_id=sample['scene_id'],parent_id=sample['parent_id'],task=record['source_task'],shot_id=record['shot_id'],
            condition=record['condition'],condition_detail=record['condition_detail'],presence_target=target['presence_target'],target_reason=target['reason'],
            centre_valid=target['centre_valid'],availability_probability=prob,availability_classifier_positive=prob>=.5,
            uncertainty_score_valid=valid_score,candidate_available=available,uncertainty_score_m=arrays['uncertainty_score_m'][i].tolist(),
            centre_error_m=None if not target['centre_valid']else float(np.linalg.norm(arrays['centre_world'][i]-target['centre_target_world'])),
            near_owner_pixels=target['near_owner_pixels'],wide_owner_pixels=target['wide_owner_pixels']))
    groups={field:{name:summarize_rows([r for r in rows if r[field]==name])for name in sorted({r[field]for r in rows})}for field in ('task','parent_id','shot_id','condition')}
    report=dict(schema='dual_crop_current_cube_centre_reserved8_development_report_v1',prediction_freeze=pointer(Path(output)/'prediction_freeze.json'),
        rows=rows,aggregate=summarize_rows(rows),groups=groups,all104_retained=True,
        teacher_preexposed_TRAIN_development=True,not_teacher_unseen_TEST=True,uncertainty_is_uncalibrated_score_not_CI=True,
        model_qualified=False,position_measurement_qualified=False,runtime_or_control_admitted=False,scope_qualified=False,
        new_model_forward_calls=0,Native_started=False,training_started=False,target_NPZ_pose_scoring_only_after_public_freeze=True,
        label_proof_GT_metadata_validated_before_prediction=True,label_proof_metadata_not_model_input_or_prediction_selection=True)
    data.revalidate();write_new(Path(output)/'report.json',report)
    return dict(report=pointer(Path(output)/'report.json'),aggregate=report['aggregate'])
