"""Development operation-classification evaluation, never Native task success."""
from collections import Counter
import math
from pathlib import Path

import numpy as np
import torch

from . import VERSION
from .model import EncoderConfig, LargeSemanticTaskEncoderV1, OPERATIONS
from .common import (CONFIG, COMPONENT, PublicOperationWindows, checked_pointer,
    collate_public, device_inputs, fixed_config, pointer, preprocess_gate, read,
    root_authorization, sha, source_map, verify_sources, write_new)


def classification_metrics(predictions):
    if not predictions:
        raise ValueError('Complete nonempty development predictions required')
    confusion=np.zeros((8,8),dtype=np.int64);ce=0.;by_task={};correct=0
    for row in predictions:
        task=row['task'];family=task.removesuffix('_high_speed')
        if family not in OPERATIONS or task not in (family,family+'_high_speed'):
            raise ValueError('Exact normal/high task version required')
        truth=OPERATIONS.index(family);prob=np.asarray(row['operation_probabilities'],dtype=float)
        if (prob.shape!=(8,) or not np.isfinite(prob).all() or (prob<0).any()
                or not math.isclose(float(prob.sum()),1.,abs_tol=1e-5,rel_tol=0)):
            raise ValueError('Finite normalized eight-class public predictions required')
        guess=int(np.argmax(prob));hit=int(guess==truth);correct+=hit
        ce-=math.log(max(float(prob[truth]),1e-30));confusion[truth,guess]+=1
        counts=by_task.setdefault(task,dict(windows=0,correct=0))
        counts['windows']+=1;counts['correct']+=hit
    for counts in by_task.values():counts['accuracy']=counts['correct']/counts['windows']
    totals=confusion.sum(1)
    if (totals==0).any():raise ValueError('All eight operations must remain in evaluation denominator')
    return dict(windows=len(predictions),correct=correct,accuracy=correct/len(predictions),
        cross_entropy=ce/len(predictions),macro_operation_accuracy=float(np.mean(confusion.diagonal()/totals)),
        confusion_matrix=confusion.tolist(),operation_order=list(OPERATIONS),per_task=by_task,
        metric='development semantic operation classification; not Native task success',
        independent_fresh_holdout=False,task_instance_identity_observed=False,
        current_network_end_to_end_ready=False)


def evaluate_model(model,dataset,device,microbatch,output,sources,model_identity,*,split_manifest):
    output=Path(output)
    if output.exists():raise FileExistsError('Fresh evaluation outputs only')
    if type(microbatch) is not int or microbatch not in (1,2,4):
        raise ValueError('Preflight-bound microbatch1/2/4 required')
    if dataset.role!='selection':raise ValueError('Only the declared development selection cohort is scored here')
    output.mkdir(parents=True,exist_ok=False);model.eval();predictions=[]
    try:
        with torch.inference_mode():
            for start in range(0,len(dataset),microbatch):
                samples=[dataset[index] for index in range(start,min(start+microbatch,len(dataset)))]
                batch=collate_public(samples)
                with torch.autocast('cuda',dtype=torch.bfloat16,enabled=torch.device(device).type=='cuda'):
                    result=model(**device_inputs(batch,device))
                probabilities=result['operation_probabilities'].float().cpu().numpy()
                e=result['task_e'].float().cpu().numpy()
                if (e.shape!=(len(samples),16) or not np.isfinite(e).all()
                        or not np.allclose(e.reshape(-1,2,8).sum(-1),1.,rtol=0,atol=1e-5)):
                    raise ValueError('SimNorm e16 public prediction contract failed')
                for identity,p,latent in zip(batch['identities'],probabilities,e):
                    predictions.append(dict(identity,operation_probabilities=p.tolist(),task_e=latent.tolist()))
        dataset.verify();verify_sources(sources)
        if len(predictions)!=len(dataset):raise ValueError('Full fixed selection window denominator required')
        # Labels derive only from frozen task metadata. Prediction/source bytes
        # are frozen before any accuracy/confusion computation.
        write_new(output/'predictions.json',dict(complete=True,predictions=predictions,
            model_identity=model_identity,split_manifest=split_manifest,source_sha256=sources,
            public_input_only=True,HDF_expert_or_future_targets_read=False))
        write_new(output/'prediction_freeze.json',dict(complete=True,
            predictions=pointer(output/'predictions.json'),source_sha256=sources,
            model_identity=model_identity,split_manifest=split_manifest,
            frozen_before_metric_scoring=True))
        metrics=classification_metrics(predictions)
        write_new(output/'metrics.json',metrics)
        return metrics
    except BaseException as error:
        write_new(output/'failure.json',dict(complete=False,error_type=type(error).__name__,error=str(error),
            Native_started=False,model_or_threshold_retuned=False))
        raise


def load_selected(path,expected_sha,manifest,sources):
    if sha(path)!=expected_sha:raise ValueError('External exact selected checkpoint SHA required')
    payload=torch.load(path,map_location='cpu',weights_only=False)
    metadata=payload.get('metadata',{})
    config=fixed_config()
    if (payload.get('version')!=VERSION or metadata.get('training_complete') is not True
            or metadata.get('fitted_component')!=COMPONENT
            or metadata.get('current_network_end_to_end_ready') is not False
            or metadata.get('config')!=pointer(CONFIG)
            or metadata.get('split_manifest')!=pointer(manifest)
            or metadata.get('source_sha256')!=sources
            or metadata.get('operation_order')!=list(OPERATIONS)
            or metadata.get('real_fit_epochs')!=5 or metadata.get('TEST_used') is not False):
        raise ValueError('Actual complete five-epoch component-only selected checkpoint required')
    checked_pointer(metadata['history'])
    model=LargeSemanticTaskEncoderV1(EncoderConfig(**config['model']))
    if metadata.get('model_spec')!=model.spec():raise ValueError('Exact large semantic encoder spec required')
    model.load_state_dict(payload['model_state_dict'],strict=True)
    return model,metadata


def evaluate_selected(manifest,data_root,checkpoint,checkpoint_sha,output,authorization,
                      authorization_sha,*,pipeline_owned=False):
    from .training import resources,require_cuda,stopped
    output=Path(output).resolve()
    if output.exists():raise FileExistsError('Fresh selection evaluation only')
    with resources(pipeline_owned):
        stopped(output);sources=source_map();verify_sources(sources);fixed_config()
        auth=root_authorization(authorization,authorization_sha,manifest,sources)
        processed=preprocess_gate(manifest,data_root)
        dataset=PublicOperationWindows(manifest,data_root,'selection');dataset.verify()
        device=require_cuda();model,metadata=load_selected(checkpoint,checkpoint_sha,manifest,sources)
        if metadata.get('preprocess_completion')!=processed:
            raise ValueError('Selected model and current preprocess differ')
        try:
            model=model.to(device)
            metrics=evaluate_model(model,dataset,device,metadata['microbatch'],output,sources,
                pointer(checkpoint),split_manifest=pointer(manifest))
            verify_sources(sources);stopped(output)
            completion=dict(complete=True,method='current_network',stage='selection_evaluation',
                fitted_component=COMPONENT,current_network_end_to_end_ready=False,
                source_unchanged=True,selected_model=pointer(checkpoint),
                prediction_freeze=pointer(output/'prediction_freeze.json'),metrics=pointer(output/'metrics.json'),
                Root_authorization=auth,Native_started=False,raw_TEST_payload_reads=0)
            write_new(output/'completion.json',completion)
            return completion
        finally:
            del model
            torch.cuda.empty_cache()
