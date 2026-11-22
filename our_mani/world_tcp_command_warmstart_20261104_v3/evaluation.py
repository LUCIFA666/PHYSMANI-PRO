"""Fixed5168 selection command prediction freeze then label-side BC metrics.

Expert command intent is not current object pose, arm arrival or task success.
The cache already joined training labels after a complete public feature freeze;
targets may be decoded by the cache reader but never enter model inputs or
demo-mixture reference selection. No TEST or Native evaluation is performed.
"""
from pathlib import Path
import json
import numpy as np
from . import VERSION


def load_selected(selected,expected_sha256,cache_manifest,*,device='cpu'):
    import torch
    from .common import source_map,verify_sources,pointer,checked_pointer,read,fixed_config
    from .model import WorldTCPCommandTransformerV3,CandidateConfig,parameter_count
    if pointer(selected)['sha256']!=expected_sha256:raise ValueError('Exact whole selected checkpoint SHA required')
    sources=source_map();verify_sources(sources);payload=torch.load(selected,map_location='cpu',weights_only=True)
    meta=payload.get('metadata',{});side=Path(selected).resolve().parent
    if (meta.get('version')!=VERSION or meta.get('source_sha256')!=sources
            or meta.get('training_complete')is not True or meta.get('epochs_completed')!=5
            or meta.get('actual_optimizer_steps')!=7685 or meta.get('config')!=fixed_config()
            or meta.get('cache_manifest')!=pointer(cache_manifest)
            or meta.get('actual_parameter_count')!=parameter_count()
            or meta.get('model_qualified')is not False or meta.get('runtime_or_control_admitted')is not False):
        raise ValueError('Complete same-source new-v3 five-epoch BC artifact required')
    status=read(side/'fit_status.json')
    if (status.get('training_complete')is not True or status.get('actual_optimizer_steps')!=7685
            or status.get('selected_checkpoint')!=pointer(selected)):
        raise ValueError('Actual complete fit_status and selected identity required')
    if set(meta.get('evidence',{}))!={'identity','history','sample_access'}:raise ValueError('Complete BC fit evidence required')
    for ref in meta['evidence'].values():checked_pointer(ref)
    checked_pointer(meta['actual_preflight']);checked_pointer(meta['encoder_checkpoint'])
    torch.set_num_threads(2);model=WorldTCPCommandTransformerV3(CandidateConfig(**meta['config']['model']))
    if meta['model_spec']!=model.spec()or sum(p.numel()for p in model.parameters())!=86772491:
        raise ValueError('Exact v3 specification and actual86772491parameter model required')
    model.load_state_dict(payload['model_state_dict'],strict=True);model.to(device).eval()
    for p in model.parameters():p.requires_grad_(False)
    return model,meta


def predict(selected,expected_sha256,cache_manifest,output,*,device='cpu'):
    import torch
    from .cache import CachedCommandDataset
    from .common import source_map,verify_sources,pointer,write_new
    from .resources import resources
    from .training import tensor_batch
    data=CachedCommandDataset(cache_manifest,'selection')
    if len(data)!=5168:raise ValueError('All fixed5168 selection command windows required')
    data.verify();sources=source_map();verify_sources(sources)
    out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False)
    request=dict(version=VERSION,selected=pointer(selected),cache_manifest=pointer(cache_manifest),source_sha256=sources,
        role='selection',rows=5168,device=device,threshold_search=False,targets_used_as_inputs=False,
        targets_are_prejoined_after_public_cache_freeze=True,physical_handoff_qualified=False)
    write_new(out/'prediction_request.json',request)
    arrays={k:[]for k in ('world_tcp_positions','world_tcp_rotations','opening_intent','opening_logits','proposal_logits')}
    identities=[];record_order=[];calls=0
    with resources():
        model,meta=load_selected(selected,expected_sha256,cache_manifest,device=device)
        micro=meta['microbatch']
        for group in data.parent_batches:
            for start in range(0,len(group),micro):
                with torch.inference_mode(False):
                    items=[data[int(i)]for i in group[start:start+micro]];batch=tensor_batch(items,device)
                with torch.inference_mode():
                    with torch.autocast(device_type='cuda',dtype=torch.bfloat16,enabled=str(device).startswith('cuda')):
                        prediction=model(**batch['inputs'])
                    for k in arrays:arrays[k].append(prediction[k].float().cpu().numpy().copy())
                    identities.extend([item['identity']for item in items]);record_order.extend(int(i)for i in group[start:start+micro]);calls+=1
        data.verify();verify_sources(sources)
    arrays={k:np.concatenate(v,axis=0)for k,v in arrays.items()}
    expected_shapes=dict(world_tcp_positions=(5168,8,3),world_tcp_rotations=(5168,8,3,3),
        opening_intent=(5168,8),opening_logits=(5168,8),proposal_logits=(5168,8))
    if (sorted(record_order)!=list(range(5168))
            or any(v.shape!=expected_shapes[k]or not np.isfinite(v).all()for k,v in arrays.items())):
        raise ValueError('All fixed current BC outputs must be finite and complete')
    np.savez_compressed(out/'predictions.npz',**arrays);write_new(out/'identities.json',identities)
    freeze=dict(**request,complete=True,predictions=pointer(out/'predictions.npz'),identities=pointer(out/'identities.json'),
        model_forward_calls=calls,model_forward_rows=5168,demo_rank_uses_labels=False,
        selection_local_indices=record_order,
        opening_intent_pending=True,Native=False,task_success_evaluated=False)
    write_new(out/'prediction_freeze.json',freeze)
    return dict(complete=True,version=VERSION,prediction_freeze=pointer(out/'prediction_freeze.json'),rows=5168,model_forward_calls=calls)


def _summary(values):
    a=np.asarray(values,float)
    return dict(count=len(a),mean=float(a.mean()),P50=float(np.quantile(a,.5)),P95=float(np.quantile(a,.95)),max=float(a.max()))


def score(directory):
    import torch
    from .cache import CachedCommandDataset
    from .common import read,source_map,verify_sources,pointer,checked_pointer,write_new
    from .model import command_imitation_loss
    out=Path(directory).resolve();freeze=read(out/'prediction_freeze.json')
    sources=source_map();verify_sources(sources)
    if (freeze.get('complete')is not True or freeze.get('version')!=VERSION or freeze.get('rows')!=5168
            or freeze.get('source_sha256')!=sources):raise ValueError('Complete same-source5168 freeze required before score')
    path=checked_pointer(freeze['predictions']);ids=read(checked_pointer(freeze['identities']))
    checked_pointer(freeze['selected']);cache=checked_pointer(freeze['cache_manifest'])
    with np.load(path,allow_pickle=False)as z:arrays={k:z[k].copy()for k in z.files}
    shapes=dict(world_tcp_positions=(5168,8,3),world_tcp_rotations=(5168,8,3,3),
        opening_intent=(5168,8),opening_logits=(5168,8),proposal_logits=(5168,8))
    order=freeze.get('selection_local_indices',[])
    if (set(arrays)!=set(shapes)or len(ids)!=5168 or sorted(order)!=list(range(5168))
            or any(arrays[k].shape!=s or not np.isfinite(arrays[k]).all()for k,s in shapes.items())):
        raise ValueError('Entire fixed frozen prediction bank required before target score')
    data=CachedCommandDataset(cache,'selection');data.verify()
    write_new(out/'score_consumption.request.json',dict(version=VERSION,prediction_freeze=pointer(out/'prediction_freeze.json'),rows=5168))
    records=[];mixture=0.
    for index in range(5168):
        item=data[order[index]]
        if item['identity']!=ids[index]:raise ValueError('Fixed selection prediction order differs from target-cache identity')
        target={k:v[None].float()for k,v in item['target'].items()}
        pred={k:torch.from_numpy(v[index:index+1])for k,v in arrays.items()}
        loss=command_imitation_loss(pred,**target);mixture+=float(loss['loss'])
        selected=int(np.argmax(arrays['proposal_logits'][index]))
        position=arrays['world_tcp_positions'][index,selected];R=arrays['world_tcp_rotations'][index,selected]
        target_position=target['position'][0].numpy();target_R=target['rotation'][0].numpy()
        cosine=np.clip((np.trace(R@target_R.T)-1)/2,-1,1)
        opening=float(arrays['opening_intent'][index,selected]);truth=float(target['opening'][0])
        records.append(dict(identity=ids[index],demo_selected_index=selected,
            world_TCP_command_error_m=float(np.linalg.norm(position-target_position)),
            world_TCP_orientation_error_rad=float(np.arccos(cosine)),
            future_opening_intent_absolute_error=abs(opening-truth),opening_intent_binary_correct=bool((opening>=.5)==(truth>=.5)),
            expert_stage=item['expert_stage']))
    data.verify();verify_sources(sources)
    write_new(out/'scored_rows.json',records)
    report=dict(version=VERSION,rows=5168,prediction_freeze=pointer(out/'prediction_freeze.json'),
        mixture_NLL=mixture/5168,reference_selection='highest_demo_likelihood_stable_argmax_no_target_selection',
        world_TCP_command_error_m=_summary([r['world_TCP_command_error_m']for r in records]),
        world_TCP_orientation_error_rad=_summary([r['world_TCP_orientation_error_rad']for r in records]),
        future_opening_intent_absolute_error=_summary([r['future_opening_intent_absolute_error']for r in records]),
        future_opening_intent_binary_accuracy=sum(r['opening_intent_binary_correct']for r in records)/5168,
        metric_semantics='recorded expert desired-command error, not observed TCP/body prediction or Native success',
        Native=False,TEST=False,physical_handoff_qualified=False,pose_precision_guaranteed=False,
        runtime_or_control_admitted=False,model_qualified=False,opening_timing_supervised=False)
    report['per_task']={task:dict(rows=len(rows),world_TCP_command_error_m=_summary([r['world_TCP_command_error_m']for r in rows]),
        world_TCP_orientation_error_rad=_summary([r['world_TCP_orientation_error_rad']for r in rows]))
        for task in sorted({r['identity']['task']for r in records})
        for rows in [[r for r in records if r['identity']['task']==task]]}
    write_new(out/'report.json',report);write_new(out/'completion.json',dict(complete=True,version=VERSION,report=pointer(out/'report.json'),rows=5168))
    return report
