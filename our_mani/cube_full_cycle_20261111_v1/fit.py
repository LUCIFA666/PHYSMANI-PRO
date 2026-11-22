"""Warm fine-tune the current models on new ordinary/high-speed TRAIN parents."""
import time
from pathlib import Path
import numpy as np
import torch
from .common import ROOT,OUT,PLAN,sha,read,write,append,pointer,verify,stopped,parents,heldout_parents,training_sources,status


def configure(seed):
    if not torch.cuda.is_available():raise RuntimeError('Actual CUDA required; no CPU training fallback')
    torch.set_num_threads(2);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True)
    torch.cuda.reset_peak_memory_stats()


def fit(p,kind,forecast_checkpoint=None):
    if kind not in ('motion','candidate'):raise ValueError(kind)
    from ..current_cube_rgbd_centre_large_20261104_v2.training import exclusive_resources
    out=OUT/(kind+'_fit');out.mkdir(exist_ok=False)
    cfg=p[kind+'_training'];sources=training_sources();write(out/'sources.json',sources)
    key='corrected_forecast_checkpoint'if kind=='motion'else'candidate_transformer_checkpoint'
    initial=ROOT/p['model_initialization'][key]
    if sha(initial)!=p['model_initialization'][key+'_sha256']:raise ValueError('Warm start changed')
    role_parents={r:parents(p,r)for r in('fit','selection')};role_parents['heldout']=heldout_parents(p)
    inputs={str(initial):sha(initial)}
    if forecast_checkpoint is not None:inputs[str(forecast_checkpoint)]=sha(forecast_checkpoint)
    folder=OUT/(kind+'_data')
    for role in('fit','selection'):
        inputs.update({str(folder/role/name):sha(folder/role/name)for name in('dataset.npz','report.json')})
        verify(read(folder/role/'report.json')['source_inputs_sha256'])
    write(out/'inputs.json',inputs)
    write(out/'request.json',dict(configuration=cfg,partition=pointer(PLAN),roles=role_parents,initial_checkpoint=pointer(initial),
        initialization='exact current trained matched-family state, no head reset',TEST_read=False,heldout_payloads_read=False))
    started=time.perf_counter();steps=0
    try:
        with exclusive_resources()as resources:
            stopped(p);verify(sources);verify(inputs);configure(cfg['seed'])
            warm=torch.load(initial,map_location='cpu',weights_only=False)
            if kind=='motion':
                from ..cube_motion_residual_20261104_v1.model import model_from_spec
                from ..cube_motion_residual_20261104_v1.losses import physical_motion_loss
                from scripts.train_cube_motion_residual_20261104_v1 import load_role,to_device,evaluate,forward,INPUTS,TARGETS
                model=model_from_spec(warm['model_spec'])
                train_np,tr=load_role(folder/'fit',expected_role='fit',expected_parents=role_parents['fit'])
                sel_np,sr=load_role(folder/'selection',expected_role='selection',expected_parents=role_parents['selection'])
                def objective(model,data,ids):return physical_motion_loss(forward(model,data,ids),**{k:data[k][ids]for k in TARGETS})['per_example'].mean()
            else:
                from ..cube_matched_candidate_20261104_v1.model import model_from_spec
                from scripts.train_cube_matched_candidate_20261104_v1 import load_role,device_data as to_device,evaluate,features,loss_for
                model=model_from_spec(warm['model_spec'])
                train_np,tr=load_role(folder/'fit',role='fit',parents=role_parents['fit'],forecast_sha256=sha(forecast_checkpoint))
                sel_np,sr=load_role(folder/'selection',role='selection',parents=role_parents['selection'],forecast_sha256=sha(forecast_checkpoint))
                def objective(model,data,ids):return loss_for(model(**features(data,ids)),data,ids)['per_example'].mean()
            model.load_state_dict(warm['model_state'],strict=True)
            model.requires_grad_(True);model.cuda()
            train=to_device(train_np,'cuda:0');selection=to_device(sel_np,'cuda:0')
            initial_metrics=evaluate(model,selection,role_parents['selection']);write(out/'initial_selection_report.json',initial_metrics)
            optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
            rng=np.random.default_rng(cfg['seed'])
            groups={i:np.flatnonzero(train_np['parent_index']==i)for i in range(len(role_parents['fit']))}
            groups={i:v for i,v in groups.items()if len(v)}
            if not groups:raise ValueError('No real supervised fit parents')
            best=float('inf');best_state=None;best_epoch=None
            for epoch in range(1,cfg['epochs']+1):
                stopped(p);model.train();total=0.
                for _ in range(cfg['steps_per_epoch']):
                    stopped(p)
                    ids=np.concatenate([rng.choice(v,cfg['windows_per_parent_per_step'],replace=True)for v in groups.values()]);rng.shuffle(ids)
                    ids=torch.as_tensor(ids,device='cuda:0')
                    optimizer.zero_grad(set_to_none=True);loss=objective(model,train,ids)
                    if not bool(torch.isfinite(loss)):raise ValueError('Nonfinite '+kind+' objective')
                    loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                    optimizer.step();steps+=1;total+=float(loss.detach())
                entry=dict(epoch=epoch,optimizer_steps=steps,train_loss=total/cfg['steps_per_epoch'],gradient_norm=float(norm),elapsed_seconds=time.perf_counter()-started)
                if epoch%cfg['selection_interval']==0 or epoch==cfg['epochs']:
                    metrics=evaluate(model,selection,role_parents['selection']);entry['selection_loss']=metrics['parent_balanced_loss']
                    if not np.isfinite(entry['selection_loss']):raise ValueError('No finite real selection metric')
                    if entry['selection_loss']<best:
                        best=entry['selection_loss'];best_epoch=epoch;best_state={k:v.detach().cpu().clone()for k,v in model.state_dict().items()}
                    entry['selected_epoch']=best_epoch
                    print(__import__('json').dumps(dict(model=kind,**entry)),flush=True)
                    status(kind+'_training',epoch=epoch,epochs=cfg['epochs'],actual_optimizer_steps=steps,selection_loss=entry['selection_loss'],selected_epoch=best_epoch)
                append(out/'history.jsonl',entry)
            if best_state is None:raise RuntimeError('No actual trained selection available')
            if not any(not torch.equal(best_state[k],warm['model_state'][k])for k in best_state):raise RuntimeError('Selected weights never changed')
            shared=dict(train_parents=role_parents['fit'],selection_parents=role_parents['selection'],heldout_parents=role_parents['heldout'],
                training_updates=steps,epochs_completed=cfg['epochs'],selected_epoch=best_epoch,selected_epoch_indexing='one_based',seed=cfg['seed'],
                TEST_used_for_fitting=False,heldout_used_for_fitting=False,initial_checkpoint=pointer(initial),
                training_source_sha256=sources,training_input_sha256=inputs)
            if kind=='motion':
                from ..cube_motion_residual_20261104_v1 import CHECKPOINT_SCHEMA
                from ..cube_motion_residual_20261104_v1.runtime import source_map
                payload=dict(shared,schema=CHECKPOINT_SCHEMA,model_spec=model.model_spec(),source_sha256=source_map(),
                    metadata=dict(training_complete=True,time_basis='actual_physics_and_task_50ms',TEST_used_for_fitting=False,heldout_used_for_fitting=False,
                        maximum_training_horizon_seconds=.8,initialization='exact current physical-clock motion checkpoint; no output reset',
                        normalization_fit_split='unchanged original normalization; no heldout refit',selected_by='minimum parent-balanced new selection loss at predeclared 5-epoch intervals',
                        seed=cfg['seed'],actual_optimizer_steps=steps,role_partition=pointer(PLAN)))
            else:
                from ..cube_matched_candidate_20261104_v1 import CHECKPOINT_SCHEMA
                from ..cube_matched_candidate_20261104_v1.model import model_source_map
                from ..cube_candidate_matched_20261104_v2 import FRAME
                payload=dict(shared,schema=CHECKPOINT_SCHEMA,model_spec=model.spec(),model_config=model.config.to_dict(),source_sha256=model_source_map(),
                    forecast_checkpoint_sha256=sha(forecast_checkpoint),target_frame=FRAME,train_dataset_sha256=tr['dataset_sha256'],selection_dataset_sha256=sr['dataset_sha256'],
                    supervised_train_parents=tr['supervised_parents'],supervised_selection_parents=sr['supervised_parents'],task_classifier_trained=False,
                    selection='minimum parent-balanced new TRAIN-selection masked positive pose/time loss; validation/test unused',
                    time_supervision='actual_tau_masked_outside_0p05_0p8_no_clamp',proposal_score_semantics=model.spec()['proposal_score_semantics'],
                    supervision_scope='actual original-Native-success contact segments only; unmatched alternatives not negative labels')
            final_state={k:v.detach().cpu().clone()for k,v in model.state_dict().items()}
            torch.save(dict(payload,model_state=best_state),out/'selected.pt')
            torch.save(dict(payload,model_state=final_state,selected_epoch=cfg['epochs']),out/'final.pt')
            model.load_state_dict(best_state,strict=True)
            write(out/'train_report.json',evaluate(model,train,role_parents['fit']))
            write(out/'selection_report.json',evaluate(model,selection,role_parents['selection']))
            verify(sources);verify(inputs);stopped(p)
            result=dict(complete=True,model=kind,epochs_completed=cfg['epochs'],actual_optimizer_steps=steps,selected_epoch=best_epoch,
                initial_selection_loss=initial_metrics['parent_balanced_loss'],selected_selection_loss=best,checkpoint=pointer(out/'selected.pt'),
                parameters=sum(v.numel()for v in model.parameters()),wall_seconds=time.perf_counter()-started,
                GPU=torch.cuda.get_device_name(),peak_allocated_bytes=torch.cuda.max_memory_allocated(),resources=resources,
                TEST_read=False,heldout_payloads_read=False,source_unchanged=True,input_unchanged=True)
            write(out/'fit_status.json',result)
        # Native will only start after this function/process releases the locks.
        return result
    except BaseException as error:
        write(out/'failure.json',dict(complete=False,error_type=type(error).__name__,error=str(error),actual_optimizer_steps=steps,automatic_retry=False))
        raise
