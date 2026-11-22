"""Root-only GPU preflight and fixed ten-epoch component fitting.

Default preparation has no model forward, optimizer or accelerator context.
No Native evaluation is launched. A semantic classifier checkpoint cannot
qualify the complete candidate/controller network.
"""
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import math
from pathlib import Path
import random
import time

import numpy as np
import torch

from . import VERSION
from .model import EncoderConfig, LargeSemanticTaskEncoderV2
from .common import (ROOT, CONFIG, COMPONENT, PublicOperationWindows, collate_public,
    datasets, device_inputs, fixed_config, pointer, preprocess_gate, read,
    root_authorization, sha, source_map, verify_sources, write_new, checked_pointer)


@contextmanager
def resources(pipeline_owned=False):
    if type(pipeline_owned) is not bool:
        raise ValueError('Explicit pipeline lock ownership required')
    if pipeline_owned:
        yield
        return
    handles=[]
    try:
        directory=ROOT/'runtime/framework/local_mpc_20261102'
        directory.mkdir(parents=True,exist_ok=True)
        for name in ('fixed_TRAIN104_geometry_native.lock','TRAIN_postcomplete_diagnostics_v1106.lock'):
            stream=(directory/name).open('a+')
            try: fcntl.flock(stream.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BaseException:
                stream.close()
                raise RuntimeError('Native/training resource busy; no GPU work started')
            handles.append(stream)
        yield
    finally:
        for stream in reversed(handles):
            fcntl.flock(stream.fileno(),fcntl.LOCK_UN);stream.close()


def stopped(output):
    paths=(Path(output)/'STOP',Path(output).parent/'STOP',ROOT/'runtime/framework/STOP',
           ROOT/'runtime/framework/local_mpc_20261102/STOP')
    if any(path.exists() for path in paths):
        raise RuntimeError('Administrative STOP; preserve output, no automatic retry')


def seed_all(value):
    random.seed(value);np.random.seed(value);torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)


def require_cuda():
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('Actual CUDA with BF16 support required; no CPU/precision fallback')
    return torch.device('cuda:0')


def microbatch_contract(size, predecessor, predecessor_sha, manifest, sources, processed):
    if type(size) is not int or size not in (1,2,4):
        raise ValueError('Only Root-directed OOM reductions4→2→1 are allowed')
    if size==4:
        if predecessor is not None or predecessor_sha is not None:
            raise ValueError('Initial microbatch4 has no OOM predecessor')
    else:
        if predecessor is None or sha(predecessor)!=predecessor_sha:
            raise ValueError('A preserved actual OOM predecessor and external SHA are required')
        value=read(predecessor)
        if (value.get('cuda_oom') is not True or value.get('stage')!='gpu_preflight'
                or value.get('microbatch')!=size*2 or value.get('config')!=pointer(CONFIG)
                or value.get('split_manifest')!=pointer(manifest)
                or value.get('source_sha256')!=sources
                or value.get('preprocess_completion')!=processed):
            raise ValueError('Only same-model/data/source actual CUDA OOM allows halving microbatch')
    return 16//size


def admitted_inputs(manifest,data_root,authorization,authorization_sha):
    config=fixed_config();sources=source_map();verify_sources(sources)
    auth=root_authorization(authorization,authorization_sha,manifest,sources)
    processed=preprocess_gate(manifest,data_root)
    fit,selection=datasets(manifest,data_root)
    fit.verify();selection.verify()
    return config,sources,auth,processed,fit,selection


def make_optimizer(model, config):
    return torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],
                             weight_decay=config['weight_decay'])


def logical_step_learning_rate(step,total_steps,config):
    """One-based optimizer step; gradient accumulation does not advance it."""
    if type(step)is not int or type(total_steps)is not int or not 1<=step<=total_steps:
        raise ValueError('Actual one-based logical optimizer step required')
    warmup=config['warmup_logical_steps'];base=config['learning_rate'];floor=config['minimum_learning_rate']
    if step<=warmup:return base*step/warmup
    progress=(step-warmup)/max(1,total_steps-warmup)
    return floor+.5*(base-floor)*(1+math.cos(math.pi*progress))


def combined_operation_loss(output,labels,config):
    raw=output['raw_group_logits']
    if raw.shape!=(len(labels),2,8)or not bool(torch.isfinite(raw).all()):
        raise ValueError('Actual raw pre-SimNorm two8-group logits required')
    primary=torch.nn.functional.cross_entropy(output['operation_logits'],labels)
    auxiliary=.5*(torch.nn.functional.cross_entropy(raw[:,0],labels)
                    +torch.nn.functional.cross_entropy(raw[:,1],labels))
    return primary+config['raw_group_auxiliary_weight']*auxiliary


def loss_for(model,batch,device,config):
    # Only these three explicit public tensors enter the encoder.
    output=model(**device_inputs(batch,device))
    labels=batch['operation_label'].to(device)
    loss=combined_operation_loss(output,labels,config)
    if not bool(torch.isfinite(loss)):
        raise ValueError('Nonfinite operation classification loss')
    return loss,output


def failure(output,error,common):
    value=dict(common,complete=False,error_type=type(error).__name__,error=str(error),
        cuda_oom=isinstance(error,torch.cuda.OutOfMemoryError),
        Native_started=False,current_network_end_to_end_ready=False)
    if Path(output).is_dir() and not (Path(output)/'failure.json').exists():
        write_new(Path(output)/'failure.json',value)


def full_history_indices(dataset,microbatch):
    """Metadata-only worst-history prefix; never label/quality selection."""
    if type(microbatch)is not int or microbatch not in (1,2,4):
        raise ValueError('Predeclared microbatch1/2/4 required')
    candidates=[index for index,record in enumerate(dataset.base.records)
                if record['current_frame_index']>=4]
    if len(candidates)<4:
        raise ValueError('At least four fixed full-history public windows required for peak preflight')
    return candidates[:microbatch],candidates[:4]


def gpu_preflight(manifest,data_root,output,authorization,authorization_sha,*,
                  microbatch=4,oom_predecessor=None,oom_sha=None,pipeline_owned=False):
    output=Path(output).resolve()
    if output.exists(): raise FileExistsError('Fresh preflight output only')
    with resources(pipeline_owned):
        stopped(output)
        config,sources,auth,processed,fit,selection=admitted_inputs(
            manifest,data_root,authorization,authorization_sha)
        accumulation=microbatch_contract(microbatch,oom_predecessor,oom_sha,manifest,sources,processed)
        indices,canonical_indices=full_history_indices(fit,microbatch)
        device=require_cuda()
        output.mkdir(parents=True,exist_ok=False)
        common=dict(version=VERSION,stage='gpu_preflight',method='current_network',
            fitted_component=COMPONENT,config=pointer(CONFIG),split_manifest=pointer(manifest),
            preprocess_completion=processed,source_sha256=sources,Root_authorization=auth,
            microbatch=microbatch,gradient_accumulation=accumulation,effective_batch_size=16,
            preflight_record_indices=indices,canonical_full_history_record_indices=canonical_indices,
            full_history_window_required=True,
            OOM_predecessor=None if oom_predecessor is None else pointer(oom_predecessor))
        model=optimizer=None
        try:
            seed_all(config['seed'])
            samples=[fit[index] for index in indices]
            batch=collate_public(samples)
            if not bool((batch['lengths']==5).all()):
                raise ValueError('GPU memory canary requires every public window to contain five real frames')
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats(device)
            start=time.perf_counter()
            model=LargeSemanticTaskEncoderV2(EncoderConfig(**config['model'])).to(device).train()
            if sum(value.numel() for value in model.parameters())!=config['exact_parameters']:
                raise ValueError('Exact parameter count changed')
            optimizer=make_optimizer(model,config)
            preflight_lr=logical_step_learning_rate(1,len(training_groups(len(fit)))*config['epochs'],config)
            for group in optimizer.param_groups:group['lr']=preflight_lr
            optimizer.zero_grad(set_to_none=True)
            begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
            # Tensor transfer is measured in full-step wall time, excluded from
            # CUDA forward-event latency. This is batch latency, not B1 RT.
            gpu={key:batch[key].to(device) for key in ('rgbd','robot_history','lengths','operation_label')}
            begin.record()
            with torch.autocast('cuda',dtype=torch.bfloat16):
                result=model(**{key:gpu[key] for key in ('rgbd','robot_history','lengths')})
                loss=combined_operation_loss(result,gpu['operation_label'],config)
            end.record()
            if not bool(torch.isfinite(loss)): raise ValueError('Nonfinite BF16 preflight loss')
            loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),config['gradient_clip_norm'],error_if_nonfinite=True)
            optimizer.step();torch.cuda.synchronize()
            fit.verify();selection.verify();verify_sources(sources);stopped(output)
            report=dict(common,complete=True,passed=True,source_unchanged=True,
                exact_parameters=config['exact_parameters'],image_size=256,history_frames=5,
                precision=config['precision'],device_name=torch.cuda.get_device_name(device),
                preflight_learning_rate=preflight_lr,
                actual_CUDA_BF16_supported=True,
                loss=float(loss.detach()),gradient_norm_before_clip=float(norm),
                peak_allocated_bytes=int(torch.cuda.max_memory_allocated(device)),
                peak_reserved_bytes=int(torch.cuda.max_memory_reserved(device)),
                forward_batch_cuda_ms=float(begin.elapsed_time(end)),
                latency_definition='batch microbatch, forward+CE only; not B1 or end-to-end realtime',
                full_initialization_and_one_step_wall_seconds=time.perf_counter()-start,
                optimizer_steps=1,optimizer_step_is_ephemeral_and_discarded=True,
                preflight_weights_reused_for_training=False,real_RGBD_model_forward_batches=1,
                sample_identities=batch['identities'],TEST_payloads_read=0,Native_started=False,
                actual_valid_frames=int(batch['lengths'].sum()),
                current_network_end_to_end_ready=False)
            write_new(output/'preflight.json',report)
            return report
        except BaseException as error:
            failure(output,error,common)
            raise
        finally:
            del model,optimizer
            torch.cuda.empty_cache()


def checked_preflight(path,checksum,manifest,sources,processed):
    if path is None or sha(path)!=checksum:
        raise ValueError('Actual GPU preflight report and its external SHA required')
    value=read(path)
    if (any(value.get(k) is not True for k in ('complete','passed','source_unchanged'))
            or value.get('stage')!='gpu_preflight' or value.get('version')!=VERSION
            or value.get('config')!=pointer(CONFIG) or value.get('split_manifest')!=pointer(manifest)
            or value.get('source_sha256')!=sources or value.get('preprocess_completion')!=processed
            or type(value.get('exact_parameters'))is not int or value['exact_parameters']!=207855640
            or type(value.get('image_size'))is not int or value['image_size']!=256
            or type(value.get('history_frames'))is not int or value['history_frames']!=5
            or type(value.get('optimizer_steps'))is not int or value['optimizer_steps']!=1
            or value.get('optimizer_step_is_ephemeral_and_discarded') is not True
            or value.get('preflight_weights_reused_for_training') is not False
            or type(value.get('microbatch'))is not int or value['microbatch'] not in (1,2,4)
            or type(value.get('gradient_accumulation'))is not int
            or value['gradient_accumulation']*value['microbatch']!=16
            or type(value.get('effective_batch_size'))is not int or value['effective_batch_size']!=16):
        raise ValueError('Actual same-source/data/207M/R256/BF16 preflight must pass before fit')
    if (value.get('actual_CUDA_BF16_supported')is not True
            or not isinstance(value.get('device_name'),str) or not value['device_name']
            or value.get('real_RGBD_model_forward_batches')!=1
            or not isinstance(value.get('sample_identities'),list)
            or len(value['sample_identities'])!=value['microbatch']
            or value.get('full_history_window_required')is not True
            or value.get('actual_valid_frames')!=value['microbatch']*5
            or any(item.get('length')!=5 for item in value['sample_identities'])
            or not isinstance(value.get('canonical_full_history_record_indices'),list)
            or len(value['canonical_full_history_record_indices'])!=4
            or value.get('preflight_record_indices')!=value['canonical_full_history_record_indices'][:value['microbatch']]
            or type(value.get('peak_allocated_bytes'))is not int or value['peak_allocated_bytes']<=0
            or type(value.get('peak_reserved_bytes'))is not int
            or value['peak_reserved_bytes']<value['peak_allocated_bytes']
            or any(isinstance(value.get(k),bool) or not isinstance(value.get(k),(int,float))
                   or not math.isfinite(value[k]) for k in ('loss','gradient_norm_before_clip','forward_batch_cuda_ms'))
            or value['forward_batch_cuda_ms']<=0):
        raise ValueError('Actual finite CUDA memory/forward/backward/step measurements required')
    return value


def training_groups(count,effective=16):
    if type(count) is not int or count<1 or effective!=16:
        raise ValueError('All nonempty windows and fixed effective batch16 required')
    return [(start,min(start+effective,count)) for start in range(0,count,effective)]


def weighted_micro_loss(loss,count,group_count):
    if type(count) is not int or type(group_count) is not int or not 1<=count<=group_count<=16:
        raise ValueError('Actual micro/partial-group counts required')
    return loss*(count/group_count)


def selection_key(metrics,epoch):
    if type(epoch) is not int or epoch<0:
        raise ValueError('Actual epoch index required')
    return (float(metrics['accuracy']),-float(metrics['cross_entropy']),-epoch)


def train_selected(manifest,data_root,output,authorization,authorization_sha,*,
                   preflight_report,preflight_sha,pipeline_owned=False):
    from .evaluation import evaluate_model
    output=Path(output).resolve()
    if output.exists(): raise FileExistsError('Fresh fit output only; no silent resume')
    with resources(pipeline_owned):
        stopped(output)
        config,sources,auth,processed,fit,selection=admitted_inputs(
            manifest,data_root,authorization,authorization_sha)
        preflight=checked_preflight(preflight_report,preflight_sha,manifest,sources,processed)
        device=require_cuda();microbatch=preflight['microbatch']
        output.mkdir(parents=True,exist_ok=False)
        common=dict(version=VERSION,method='current_network',stage='fit',
            fitted_component=COMPONENT,current_network_end_to_end_ready=False,
            config=pointer(CONFIG),split_manifest=pointer(manifest),preprocess_completion=processed,
            source_sha256=sources,Root_authorization=auth,
            GPU_preflight=dict(path=str(Path(preflight_report).resolve()),sha256=preflight_sha),
            microbatch=microbatch,gradient_accumulation=16//microbatch,effective_batch_size=16)
        model=optimizer=None
        try:
            seed_all(config['seed'])
            # Always new seed initialization. Ephemeral preflight weights are
            # never restored and cannot become a selected checkpoint.
            model=LargeSemanticTaskEncoderV2(EncoderConfig(**config['model'])).to(device)
            optimizer=make_optimizer(model,config)
            generator=torch.Generator().manual_seed(config['seed'])
            history=[];best=None;best_state=None;steps=0;forward_batches=0
            total_steps=len(training_groups(len(fit)))*config['epochs']
            for epoch in range(config['epochs']):
                model.train();order=torch.randperm(len(fit),generator=generator).tolist()
                total_loss=0.;seen=0
                with (output/f'epoch{epoch:02d}_public_access.jsonl').open('x') as access:
                    for start,end in training_groups(len(order)):
                        lr=logical_step_learning_rate(steps+1,total_steps,config)
                        for group_state in optimizer.param_groups:group_state['lr']=lr
                        optimizer.zero_grad(set_to_none=True);group=order[start:end]
                        for offset in range(0,len(group),microbatch):
                            stopped(output)
                            indices=group[offset:offset+microbatch]
                            batch=collate_public([fit[index] for index in indices])
                            with torch.autocast('cuda',dtype=torch.bfloat16):
                                loss,_=loss_for(model,batch,device,config)
                            weighted_micro_loss(loss,len(indices),len(group)).backward()
                            total_loss+=float(loss.detach())*len(indices);seen+=len(indices);forward_batches+=1
                            import json
                            access.write(json.dumps(dict(epoch=epoch,indices=indices,
                                public_windows=batch['identities'],expert_or_debug_targets_read=False))+'\n')
                        torch.nn.utils.clip_grad_norm_(model.parameters(),config['gradient_clip_norm'],error_if_nonfinite=True)
                        optimizer.step();steps+=1
                if seen!=len(fit): raise ValueError('Every fixed fit window must be retained')
                metrics=evaluate_model(model,selection,device,microbatch,
                    output/f'epoch{epoch:02d}_selection',sources,
                    dict(kind='in_memory_training_epoch',epoch=epoch,optimizer_steps=steps),
                    split_manifest=pointer(manifest))
                row=dict(epoch=epoch,fit_windows=seen,fit_loss=total_loss/seen,
                    fit_loss_semantics='operation_CE+.5mean(raw_group_CE)',
                    optimizer_steps=steps,learning_rate_last_logical_step=lr,selection_metrics=metrics)
                history.append(row)
                if best is None or selection_key(metrics,epoch)>selection_key(best['selection_metrics'],best['epoch']):
                    best=row
                    best_state={key:value.detach().cpu().clone() for key,value in model.state_dict().items()}
                fit.verify();selection.verify();verify_sources(sources)
                write_new(output/f'epoch{epoch:02d}.json',row)
            write_new(output/'history.json',dict(complete=True,epochs=history,fit=fit.summary(),selection=selection.summary()))
            metadata=dict(common,training_complete=True,semantic_task_e_classifier_trained=True,
                operation_order=list(model.spec()['operations']),model_spec=model.spec(),
                selected_epoch=best['epoch'],selection=best['selection_metrics'],history=pointer(output/'history.json'),
                real_fit_epochs=10,optimizer_steps=steps,real_fit_forward_batches=forward_batches,
                loss='operation_CE+.5*mean(two_raw_group_CE)',
                scheduler='50logical_warmup_then_cosine_to3e-6',total_planned_optimizer_steps=total_steps,
                task_instance_identity_or_goal_index_observed=False,
                Native_contact_or_debug_or_future_as_input=False,TEST_used=False,
                physical_clock_verified=False,component_only=True,Native_started=False)
            torch.save(dict(version=VERSION,model_state_dict=best_state,metadata=metadata),output/'selected.pt')
            fit.verify();selection.verify();verify_sources(sources);stopped(output)
            completion=dict(common,complete=True,source_unchanged=True,
                selected_model=pointer(output/'selected.pt'),history=pointer(output/'history.json'),
                fitted_epochs=10,selected_epoch=best['epoch'],optimizer_steps=steps,
                Native_started=False,Native_evaluation_started=False,raw_TEST_payload_reads=0)
            write_new(output/'completion.json',completion)
            return completion
        except BaseException as error:
            failure(output,error,common)
            raise
        finally:
            del model,optimizer
            torch.cuda.empty_cache()
