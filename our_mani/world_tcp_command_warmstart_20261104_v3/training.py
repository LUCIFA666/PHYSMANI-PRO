"""Fixed five-epoch cached-public BC; Root alone executes actual GPU work."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import random
import numpy as np

from . import VERSION


def check(cache_manifest=None):
    from .common import fixed_config,source_map,verify_sources,pointer
    from .model import parameter_count
    config=fixed_config();sources=source_map();verify_sources(sources)
    report=dict(version=VERSION,config=config,source_sha256=sources,exact_parameters=parameter_count(),
        cache_ready=False,model_constructed=False,optimizer_constructed=False,model_forward_calls=0,
        GPU=False,Native=False,training=False,TEST_reads=0,physical_handoff_qualified=False)
    if cache_manifest is not None:
        fit,selection=_datasets(cache_manifest)
        report.update(cache_ready=True,cache_manifest=pointer(cache_manifest),fit_samples=len(fit),selection_samples=len(selection))
    return report


def _datasets(manifest):
    from .cache import CachedCommandDataset
    fit=CachedCommandDataset(manifest,'fit');selection=CachedCommandDataset(manifest,'selection')
    if len(fit)!=24592 or len(selection)!=5168:raise ValueError('Full fixed24592fit/5168selection required')
    for data,count in ((fit,1537),(selection,323)):
        groups=data.parent_batches
        flat=[int(i)for group in groups for i in group]
        if len(groups)!=count or any(len(g)!=16 for g in groups)or sorted(flat)!=list(range(len(data))):
            raise ValueError('Every fixed parent contributes exactly16 unique stage-preserving windows')
    fit.verify();selection.verify()
    return fit,selection


def tensor_batch(items,device,*,require_ordinary=False):
    import torch
    from .cache import collate_cached
    value=collate_cached(items)
    for tree in ('inputs','target'):
        value[tree]={k:v.to(device=device,dtype=torch.float32 if v.is_floating_point()else v.dtype)
            for k,v in value[tree].items()}
        if require_ordinary and any(torch.is_inference(v)for v in value[tree].values()):
            raise ValueError('Trainable BC requires ordinary materialized cache tensors, not inference tensors')
    return value


def logical_backward(model,data,indices,device,microbatch):
    import torch
    from .model import command_imitation_loss
    if len(indices)!=16 or microbatch not in (2,4):raise ValueError('One actual effective16 logical batch required')
    total=0.;calls=0
    for start in range(0,16,microbatch):
        batch=tensor_batch([data[int(i)]for i in indices[start:start+microbatch]],device,require_ordinary=True)
        with torch.autocast(device_type='cuda',dtype=torch.bfloat16,enabled=str(device).startswith('cuda')):
            output=model(**batch['inputs'])
        with torch.autocast(device_type=torch.device(device).type,enabled=False):
            loss=command_imitation_loss(output,**batch['target'])['loss'].float()
            weighted=loss*(len(indices[start:start+microbatch])/16.)
        if not bool(torch.isfinite(weighted)):raise ValueError('Finite FP32 demonstration-mixture loss required')
        weighted.backward();total+=float(weighted.detach());calls+=1
    return dict(loss=total,model_forward_calls=calls,rows=16)


def selection_loss(model,data,device,microbatch):
    import torch
    from .model import command_imitation_loss
    model.eval();total=0.;count=calls=0
    for group in data.parent_batches:
        for start in range(0,len(group),microbatch):
            indices=group[start:start+microbatch]
            with torch.inference_mode(False):
                batch=tensor_batch([data[int(i)]for i in indices],device)
            with torch.inference_mode():
                with torch.autocast(device_type='cuda',dtype=torch.bfloat16,enabled=str(device).startswith('cuda')):
                    output=model(**batch['inputs'])
                loss=command_imitation_loss(output,**batch['target'])['loss'].float()
                if not bool(torch.isfinite(loss)):raise ValueError('Finite all-selection command loss required')
                total+=float(loss)*len(indices);count+=len(indices);calls+=1
    if count!=5168:raise ValueError('Every selection window must be evaluated')
    return dict(loss=total/count,rows=count,model_forward_calls=calls)


def _new_model(config,device):
    import torch
    from .model import WorldTCPCommandTransformerV3,CandidateConfig,parameter_count
    random.seed(config['seed']);np.random.seed(config['seed']);torch.manual_seed(config['seed'])
    torch.set_num_threads(2)
    if str(device)!='cuda:0' or not torch.cuda.is_available()or not torch.cuda.is_bf16_supported():
        raise ValueError('Actual Root-approved CUDA0 BF16 device required for BC training')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    torch.cuda.manual_seed_all(config['seed']);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    model=WorldTCPCommandTransformerV3(CandidateConfig(**config['model'])).to(device)
    if sum(p.numel()for p in model.parameters())!=parameter_count():raise ValueError('Fixed actual v3 parameter count changed')
    optimizer=torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
    return model,optimizer


def fit(cache_manifest,output,preflight_report,preflight_sha256,authorization,authorization_sha256,*,device='cuda:0'):
    from .common import fixed_config,source_map,verify_sources,pointer,write_new,root_authorization,checked_canary
    from .resources import resources
    config=fixed_config();sources=source_map();verify_sources(sources)
    if (config['epochs']!=5 or config['effective_batch_size']!=16 or config['learning_rate']!=3e-5
            or config['weight_decay']!=.01 or config['gradient_clip_norm']!=1. or config['seed']!=20261104):
        raise ValueError('Root-fixed five-epoch BC budget required')
    fit_data,selection=_datasets(cache_manifest);meta=fit_data.metadata
    admitted=checked_canary(preflight_report,preflight_sha256,sources=sources,subset_pointer=meta['subset'],
        encoder_pointer=meta['encoder_checkpoint'],cache_dtypes=meta['cache_dtypes'])
    micro=admitted['microbatch']
    if micro not in (2,4):raise ValueError('Actual admitted micro4 or explicit-fresh-OOM-successor micro2 required')
    auth=root_authorization(authorization,authorization_sha256,sources=sources,
        subset_pointer=meta['subset'],encoder_pointer=meta['encoder_checkpoint'])
    out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False)
    identity=dict(version=VERSION,config=config,source_sha256=sources,cache_manifest=pointer(cache_manifest),
        subset=meta['subset'],encoder_checkpoint=meta['encoder_checkpoint'],encoder_spec_sha256=meta['encoder_spec_sha256'],
        cache_dtypes=meta['cache_dtypes'],actual_preflight=pointer(preflight_report),root_authorization=auth,microbatch=micro,
        fitted_component='world_TCP_expert_command_BC_only',current_network_end_to_end_ready=False,
        opening_intent_pending=True,physical_handoff_qualified=False,Native_success_supervised=False)
    write_new(out/'identity.json',identity)
    status=dict(version=VERSION,training_complete=False,actual_optimizer_steps=0,epochs_completed=0,
        model_qualified=False,runtime_or_control_admitted=False)
    history=[];access=[];best_state=None;best=None;fit_calls=selection_calls=0
    try:
        import torch
        with resources():
            model,optimizer=_new_model(config,device);rng=np.random.default_rng(config['seed'])
            environment=dict(torch=str(torch.__version__),numpy=np.__version__,device=str(device),
                GPU_name=torch.cuda.get_device_name(0),BF16_autocast=True,FP32_parameters_and_loss=True,
                deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),torch_num_threads=torch.get_num_threads())
            for epoch in range(5):
                model.train();losses=[];order=rng.permutation(len(fit_data.parent_batches))
                # Parent chunks are the fixed16-window logical batches. Shuffle
                # only their order: every original stage/window remains once.
                for parent_index in order:
                    indices=fit_data.parent_batches[int(parent_index)]
                    optimizer.zero_grad(set_to_none=True)
                    value=logical_backward(model,fit_data,indices,device,micro)
                    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),config['gradient_clip_norm'])
                    if not bool(torch.isfinite(norm)):raise ValueError('Finite BC gradients required')
                    optimizer.step();status['actual_optimizer_steps']+=1;fit_calls+=value['model_forward_calls'];losses.append(value['loss'])
                selected=selection_loss(model,selection,device,micro);selection_calls+=selected['model_forward_calls']
                row=dict(epoch=epoch,optimizer_steps=status['actual_optimizer_steps'],fit_loss=float(np.mean(losses)),
                    selection_loss=selected['loss'],fit_rows=24592,selection_rows=5168)
                history.append(row)
                access.append(dict(epoch=epoch,fit_parent_order=[int(i)for i in order],
                    fit_window_count=24592,selection_fixed_window_count=5168,fit_unique_windows=24592))
                if best is None or selected['loss']<best['selection_loss']:
                    best=deepcopy(row);best_state={k:v.detach().cpu().clone()for k,v in model.state_dict().items()}
                status['epochs_completed']=epoch+1
                with (out/'history.jsonl').open('a')as stream:stream.write(json.dumps(row,allow_nan=False)+'\n')
                print(json.dumps(row,allow_nan=False),flush=True)
            if status['actual_optimizer_steps']!=7685:raise ValueError('Complete five epochs x1537 logical updates required')
            fit_data.verify();selection.verify();verify_sources(sources)
            write_new(out/'history.json',history);write_new(out/'sample_access.json',dict(epochs=access,
                fit_rows=122960,selection_rows=25840,fit_model_forward_calls=fit_calls,selection_model_forward_calls=selection_calls))
            metadata=dict(**identity,training_complete=True,epochs_completed=5,actual_optimizer_steps=7685,
                selected_epoch=best['epoch'],selection_loss=best['selection_loss'],selection_rule='minimum_full5168_mixture_NLL_strict_improvement_earliest_tie',
                model_spec=model.spec(),actual_parameter_count=sum(p.numel()for p in model.parameters()),
                training_environment=environment,
                evidence={name:pointer(out/(name+'.json'))for name in ('identity','history','sample_access')},
                model_qualified=False,position_measurement_qualified=False,physical_handoff_supervised=False,
                opening_is_future_intent_not_immediate_actuator_command=True,runtime_or_control_admitted=False)
            torch.save(dict(model_state_dict=best_state,metadata=metadata),out/'selected.pt')
            status.update(training_complete=True,selected_checkpoint=pointer(out/'selected.pt'),selected_epoch=best['epoch'],
                selection_loss=best['selection_loss'],source_unchanged=True,fit_forward_calls=fit_calls,selection_forward_calls=selection_calls)
        write_new(out/'completion.json',dict(method='current_network',stage='fit',complete=True,source_unchanged=True,
            selected_model=pointer(out/'selected.pt'),fitted_component='world_TCP_expert_command_BC_only',current_network_end_to_end_ready=False))
        return status
    except BaseException as error:
        status['error']=type(error).__name__+': '+str(error);raise
    finally:write_new(out/'fit_status.json',status)
