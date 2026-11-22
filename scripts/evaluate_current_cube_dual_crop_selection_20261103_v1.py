"""Independent selected-weight selection208 development report, never reserved.

No optimizer/threshold choice. Public predictions freeze before target-array
scoring, although the fixed selection set already chose the training epoch.
This script is deliberately outside the immutable13 training-source closure.
"""
from pathlib import Path
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from our_mani.current_cube_rgbd_centre_20261103_v1.data import DualCropCubeDataset,sha256,checked_pointer
from our_mani.current_cube_rgbd_centre_20261103_v1.training import VERSION,SELECTED_ENCODER_SHA,check,fixed_config,training_source_map,tensor_batch,pointer,write_new
from our_mani.current_cube_rgbd_centre_20261103_v1.evaluation import summarize_rows


def evaluate(selected,expected_sha256,manifest,encoder_checkpoint,output):
    if sha256(selected)!=expected_sha256 or sha256(encoder_checkpoint)!=SELECTED_ENCODER_SHA:raise ValueError('Exact selected and dbf whole-SHA required')
    config=fixed_config();sources=training_source_map();script_before=sha256(__file__)
    data=DualCropCubeDataset(manifest,'selection')
    if len(data)!=208:raise ValueError('All fixed16selection parents/208shots required')
    out=Path(output).resolve()
    if out.exists():raise FileExistsError('Fresh selected selection report required')
    import torch
    from our_mani.current_cube_rgbd_centre_20261103_v1.model import DualCropCubeCentreHeadV1
    payload=torch.load(selected,map_location='cpu',weights_only=True);meta=payload.get('metadata',{})
    if meta.get('version')!=VERSION or meta.get('training_complete')is not True or meta.get('epochs_completed')!=40 or meta.get('actual_optimizer_steps')!=520 or meta.get('model_qualified')is not False or meta.get('source_sha256')!=sources or meta.get('manifest')!=pointer(manifest):raise ValueError('Completed same-source selected head required')
    side=Path(selected).resolve().parent;status=json.loads((side/'fit_status.json').read_text())
    if status.get('training_complete')is not True or status.get('actual_optimizer_steps')!=520 or status.get('selected_checkpoint')!=pointer(selected):raise ValueError('Actual completed same-selected fit status required')
    expected={'identity','cache_before','cache_after','history','sample_access','source_verification'}
    if set(meta.get('evidence',{}))!=expected:raise ValueError('Complete original training evidence required')
    for name,ref in meta['evidence'].items():
        if checked_pointer(side,ref)!=side/(name+'.json'):raise ValueError('Original selected evidence sibling required')
    out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
    model=DualCropCubeCentreHeadV1(encoder_trainable=True)
    model.load_selected_v5_encoder(encoder_checkpoint,SELECTED_ENCODER_SHA);model.load_state_dict(payload['model_state_dict'],strict=True);model.eval()
    for parameter in model.parameters():parameter.requires_grad_(False)
    chunks=[];ledger=[];calls=0
    with torch.no_grad():
        for start in range(0,len(data),config['batch_size']):
            samples=[data.public_sample(i)for i in range(start,min(start+config['batch_size'],len(data)))]
            arrays={k:np.stack([s['model_inputs'][k]for s in samples])for k in samples[0]['model_inputs']}
            result=model(**tensor_batch(arrays,np.arange(len(samples)),'cpu'));calls+=1
            chunks.append({k:result[k].cpu().numpy()for k in ('centre_world','measurement_availability_logit','uncertainty_score_m','uncertainty_score_valid')})
            ledger.extend(dict(scene_id=s['scene_id'],parent_id=s['parent_id'])for s in samples)
    predictions={k:np.concatenate([c[k]for c in chunks])for k in chunks[0]}
    if any(not np.isfinite(a).all()for a in predictions.values()):raise ValueError('All208 predictions must remain finite')
    np.savez_compressed(out/'predictions.npz',**predictions);write_new(out/'ordered_scenes.json',ledger)
    data.revalidate()
    freeze=dict(schema='selected_dual_crop_selection208_public_freeze_v1',selected=pointer(selected),manifest=pointer(manifest),
        method_source_sha256=sources,selection_evaluator=pointer(__file__),files=dict(predictions=pointer(out/'predictions.npz'),ordered_scenes=pointer(out/'ordered_scenes.json')),
        samples=208,parents=16,role='selection',actual_model_forward_batch_calls=calls,actual_model_forward_rows=208,device='cpu',torch_num_threads=2,
        target_NPZ_array_decodes_before_public_freeze=0,label_proof_GT_metadata_prevalidated_not_model_input=True,
        reserved_getters=0,selected_epoch_already_chosen_on_this_set=True,not_fresh_or_unseen_validation=True,
        Native_started=False,training_started=False,model_qualified=False,runtime_or_control_admitted=False)
    write_new(out/'prediction_freeze.json',freeze)
    # Targets are opened only after every public prediction/file is frozen.
    rows=[]
    for i in range(len(data)):
        sample=data[i];target=sample['targets'];record=sample['record'];logit=float(predictions['measurement_availability_logit'][i])
        probability=float(np.exp(-np.logaddexp(0.,-logit)));score_valid=bool(predictions['uncertainty_score_valid'][i])
        rows.append(dict(scene_id=sample['scene_id'],parent_id=sample['parent_id'],task=record['source_task'],shot_id=record['shot_id'],condition=record['condition'],
            presence_target=target['presence_target'],target_reason=target['reason'],centre_valid=target['centre_valid'],availability_probability=probability,
            candidate_available=probability>=config['availability_evaluation_threshold']and score_valid,
            uncertainty_score_valid=score_valid,uncertainty_score_m=predictions['uncertainty_score_m'][i].tolist(),
            centre_error_m=None if not target['centre_valid']else float(np.linalg.norm(predictions['centre_world'][i]-target['centre_target_world'])),
            near_owner_pixels=target['near_owner_pixels'],wide_owner_pixels=target['wide_owner_pixels']))
    data.revalidate()
    if sources!=training_source_map()or sha256(__file__)!=script_before or sha256(selected)!=expected_sha256:raise ValueError('Frozen selected/source changed')
    report=dict(schema='selected_dual_crop_selection208_development_report_v1',prediction_freeze=pointer(out/'prediction_freeze.json'),
        aggregate=summarize_rows(rows),rows=rows,
        groups={key:{value:summarize_rows([r for r in rows if r[key]==value])for value in sorted({r[key]for r in rows})}for key in ('task','parent_id','shot_id','condition')},
        original_selection_loss_chose_epoch=meta['selected_epoch'],evaluation_threshold=config['availability_evaluation_threshold'],
        threshold_or_hyperparameter_selection=False,all208_retained=True,reserved_parents_read=0,TEST_read=False,
        target_NPZ_pose_scoring_only_after_public_freeze=True,label_proof_GT_metadata_prevalidated_not_model_input=True,
        actual_model_forward_batch_calls=calls,actual_model_forward_rows=208,Native_started=False,training_started=False,
        uncertainty_is_uncalibrated_score_not_CI=True,model_qualified=False,position_measurement_qualified=False,runtime_or_control_admitted=False)
    write_new(out/'report.json',report)
    return dict(report=pointer(out/'report.json'),prediction_freeze=pointer(out/'prediction_freeze.json'),aggregate=report['aggregate'])


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--selected');p.add_argument('--expected-sha256');p.add_argument('--manifest');p.add_argument('--encoder-checkpoint');p.add_argument('--output')
    mode=p.add_mutually_exclusive_group();mode.add_argument('--check',action='store_true');mode.add_argument('--evaluate',action='store_true');a=p.parse_args()
    if a.evaluate:
        if not all((a.selected,a.expected_sha256,a.manifest,a.encoder_checkpoint,a.output)):p.error('--evaluate requires selected/SHA/manifest/encoder/fresh output')
        value=evaluate(a.selected,a.expected_sha256,a.manifest,a.encoder_checkpoint,a.output)
    else:value=check(a.manifest)
    print(json.dumps(value,indent=2));return 0
if __name__=='__main__':raise SystemExit(main())
