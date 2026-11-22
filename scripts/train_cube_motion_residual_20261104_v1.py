"""Fit real-clock Cube GRU/context motion residuals; heldout is separate.

Uses the fixed whole-episode motion datasets. No RGBD, Native state or future
label is a recurrent model input. The original GRU is explicitly transferred
and its acceleration output reset to zero before supervised fine-tuning.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch

INPUTS = ('positions','timestamps','confidence','lengths')
TARGETS = ('target_positions','target_valid','target_times')


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text())
def pointer(path): return dict(path=str(Path(path).resolve(strict=True)),sha256=sha(path))


def write(path,value):
    with Path(path).open('x')as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')


def verify(mapping):
    for path,digest in mapping.items():
        if sha(path)!=digest:raise ValueError('Frozen input or source changed: '+path)


def stopped(output):
    if any(p.exists()for p in (output/'STOP',output.parent/'STOP',ROOT/'runtime/framework/STOP',
                              ROOT/'runtime/framework/local_mpc_20261102/STOP')):
        raise InterruptedError('Administrative STOP; preserve partial run')


def load_role(directory, *, expected_role, expected_parents):
    directory=Path(directory);report=read(directory/'report.json')
    if (expected_role not in ('fit','selection','heldout') or report.get('role')!=expected_role
            or report.get('parents')!=expected_parents):
        raise ValueError('Exact whole-parent role metadata required before payload access')
    expected=report.get('dataset_sha256',report.get('dataset',{}).get('sha256'))
    if expected!=sha(directory/'dataset.npz'):raise ValueError('Exact compiled motion role required')
    with np.load(directory/'dataset.npz',allow_pickle=False)as stored:
        arrays={k:np.array(stored[k],copy=True)for k in stored.files}
    required=set(INPUTS+TARGETS+('parent_index','sample_weight'))
    if not required<=set(arrays):raise ValueError('Complete physical motion tensors required')
    if len(arrays['positions'])<1:raise ValueError('Nonempty fit/selection role required')
    return arrays,report


def to_device(arrays,device):
    return {k:torch.from_numpy(v).to(device)for k,v in arrays.items()}


def quantiles(values):
    a=np.asarray(values,float)
    return dict(count=len(a),mean=float(a.mean()),p50=float(np.median(a)),
                p95=float(np.percentile(a,95)),maximum=float(a.max()))if len(a)else dict(count=0)


def forward(model,data,ids):
    return model.forward_public(**{k:data[k][ids]for k in INPUTS})


@torch.no_grad()
def evaluate(model,data,parents,batch_size=128):
    from our_mani.cube_motion_residual_20261104_v1.losses import physical_motion_loss
    model.eval();losses=[];predicted=[];cv=[];valid=[]
    for start in range(0,len(data['positions']),batch_size):
        ids=slice(start,start+batch_size);out=forward(model,data,ids)
        terms=physical_motion_loss(out,**{k:data[k][ids]for k in TARGETS})
        losses.append(terms['per_example'].cpu().numpy())
        predicted.append(out['positions'].cpu().numpy());cv.append(out['cv_positions'].cpu().numpy())
        valid.append(terms['valid_mask'].cpu().numpy())
    pred=np.concatenate(predicted);cv=np.concatenate(cv);mask=np.concatenate(valid)
    target=data['target_positions'].cpu().numpy();weights=data['sample_weight'].cpu().numpy()
    indices=data['parent_index'].cpu().numpy();loss=np.concatenate(losses)
    errors=np.linalg.norm(pred-target,axis=-1);cv_errors=np.linalg.norm(cv-target,axis=-1)
    horizons={}
    for h in (2,8,16):
        keep=mask[:,h]
        entry=dict(valid_windows=int(keep.sum()),trained_position_error_m=quantiles(errors[keep,h]),
                   CV_position_error_m=quantiles(cv_errors[keep,h]))
        # The optional recorded predictions are an evaluation baseline only.
        old_key = ('old_forecast_positions' if 'old_forecast_positions' in data else 'recorded_old_positions')
        if old_key in data and ('old_forecast_available' in data or 'recorded_old_valid' in data):
            old=data[old_key].cpu().numpy()
            if 'old_forecast_available' in data:
                available=data['old_forecast_available'].cpu().numpy()
                ov=np.broadcast_to(available[:,None],mask.shape)&np.isfinite(old).all(-1)
                if 'old_forecast_times' in data:
                    ov=ov&(np.abs(data['old_forecast_times'].cpu().numpy()-data['target_times'].cpu().numpy())<=1e-6)
            else:ov=data['recorded_old_valid'].cpu().numpy()
            common=keep&ov[:,h]
            old_error=np.linalg.norm(old-target,axis=-1)
            entry['same_recorded_old_anchor_subset']=dict(windows=int(common.sum()),
                trained_error_m=quantiles(errors[common,h]),CV_error_m=quantiles(cv_errors[common,h]),
                recorded_original_GRU_error_m=quantiles(old_error[common,h]))
        horizons[str(h*.05)]=entry
    eligible=sorted(set(indices.tolist()))
    return dict(parent_balanced_loss=float(np.sum(loss*weights)/weights.sum()),
        windows=len(loss),eligible_parent_count=len(eligible),horizons_seconds=horizons,
        per_parent={parents[i]:dict(windows=int(np.sum(indices==i)),mean_loss=float(loss[indices==i].mean()),
            horizon08_error_m=quantiles(errors[(indices==i)&mask[:,16],16]))for i in eligible},
        scope='same measured timestamps; current public inputs, private future positions as labels only',
        task_success_evaluated=False,position_hard_gate_used=False,NN_forward_calls=math.ceil(len(loss)/batch_size))


def payload_common(model,plan,roles,steps,epoch):
    from our_mani.cube_motion_residual_20261104_v1 import CHECKPOINT_SCHEMA
    from our_mani.cube_motion_residual_20261104_v1.runtime import source_map
    return dict(schema=CHECKPOINT_SCHEMA,model_spec=model.model_spec(),source_sha256=source_map(),
        train_parents=roles['fit'],selection_parents=roles['selection'],heldout_parents=roles['heldout'],
        training_updates=steps,selected_epoch=epoch,training_source_sha256=plan['source_sha256'],
        metadata=dict(training_complete=True,time_basis='actual_physics_and_task_50ms',
            TEST_used_for_fitting=False,heldout_used_for_fitting=False,maximum_training_horizon_seconds=.8,
            normalization_fit_split='original train normalization transferred without refitting',
            initialization='exact ebec parameter transfer, acceleration head explicitly zeroed',
            selected_by='minimum parent-balanced physical selection loss at epochs5,10,...',
            seed=plan['seed'],actual_optimizer_steps=steps,role_partition=plan['partition']))


def fit(plan_path,digest):
    if sha(plan_path)!=digest:raise ValueError('External training plan SHA required')
    plan=read(plan_path);verify(plan['source_sha256']);verify(plan['input_sha256'])
    output=Path(plan['output']);stopped(output);output.mkdir(parents=True,exist_ok=False)
    write(output/'request.json',dict(plan=pointer(plan_path),configuration=plan,Native_started=False,heldout_payloads_read=False))
    start=time.perf_counter();steps=0
    try:
        from our_mani.current_cube_rgbd_centre_large_20261104_v2.training import exclusive_resources
        from our_mani.cube_motion_residual_20261104_v1.model import CubeMotionResidualGRU
        from our_mani.cube_motion_residual_20261104_v1.losses import physical_motion_loss
        partition=read(plan['partition']['path'])
        roles={role:[p['parent_id']for p in partition['rows']if p['role']==role]for role in ('fit','selection','heldout')}
        train_np,tr=load_role(plan['fit_dataset'],expected_role='fit',expected_parents=roles['fit'])
        selection_np,sr=load_role(plan['selection_dataset'],expected_role='selection',expected_parents=roles['selection'])
        train_parents=tr['parents'];selection_parents=sr['parents']
        if set(train_parents)&set(selection_parents)or train_parents!=roles['fit']or selection_parents!=roles['selection']:
            raise ValueError('Whole episode training/selection identity mismatch')
        with exclusive_resources()as resources:
            if not torch.cuda.is_available():raise RuntimeError('Actual GPU required, no silent CPU fallback')
            os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
            torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
            torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
            torch.set_num_threads(2);torch.manual_seed(plan['seed']);torch.cuda.manual_seed_all(plan['seed'])
            torch.cuda.reset_peak_memory_stats()
            model,transfer=CubeMotionResidualGRU.from_legacy_checkpoint(plan['legacy_checkpoint'],
                expected_sha256=plan['legacy_sha256'],maximum_acceleration=plan['maximum_acceleration'])
            write(output/'weight_transfer.json',transfer)
            model.to('cuda:0');train=to_device(train_np,'cuda:0');selection=to_device(selection_np,'cuda:0')
            initial=evaluate(model,selection,selection_parents);write(output/'CV_initial_selection_report.json',initial)
            optimizer=torch.optim.AdamW(model.parameters(),lr=plan['learning_rate'],weight_decay=plan['weight_decay'])
            rng=np.random.default_rng(plan['seed'])
            groups={p:np.flatnonzero(train_np['parent_index']==i)for i,p in enumerate(train_parents)}
            groups={p:v for p,v in groups.items()if len(v)}
            if not groups:raise ValueError('No actual supervised TRAIN parents')
            best_loss=float('inf');best_state=None;best_epoch=None
            with (output/'history.jsonl').open('x')as log:
                for epoch in range(1,plan['epochs']+1):
                    stopped(output);model.train();total=0.
                    for _ in range(plan['steps_per_epoch']):
                        ids=np.concatenate([rng.choice(v,plan['windows_per_parent_per_step'],replace=True)for v in groups.values()])
                        rng.shuffle(ids);index=torch.as_tensor(ids,device='cuda:0')
                        optimizer.zero_grad(set_to_none=True)
                        terms=physical_motion_loss(forward(model,train,index),**{k:train[k][index]for k in TARGETS})
                        loss=terms['per_example'].mean()
                        if not bool(torch.isfinite(loss)):raise ValueError('Nonfinite real-position motion loss')
                        loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                        optimizer.step();steps+=1;total+=float(loss.detach())
                    row=dict(epoch=epoch,optimizer_steps=steps,train_loss=total/plan['steps_per_epoch'],
                             last_gradient_norm=float(norm),elapsed_seconds=time.perf_counter()-start)
                    if epoch%plan['selection_interval']==0 or epoch==plan['epochs']:
                        m=evaluate(model,selection,selection_parents);row['selection_loss']=m['parent_balanced_loss']
                        if m['parent_balanced_loss']<best_loss:
                            best_loss=m['parent_balanced_loss'];best_epoch=epoch
                            best_state={k:v.detach().cpu().clone()for k,v in model.state_dict().items()}
                        row['selected_epoch']=best_epoch
                        print(json.dumps(row),flush=True)
                    log.write(json.dumps(row,allow_nan=False)+'\n');log.flush()
            if best_state is None:raise RuntimeError('No trained selection checkpoint evaluated')
            final_state={k:v.detach().cpu().clone()for k,v in model.state_dict().items()}
            torch.save(dict(payload_common(model,plan,roles,steps,plan['epochs']),model_state=final_state),output/'final.pt')
            torch.save(dict(payload_common(model,plan,roles,steps,best_epoch),model_state=best_state),output/'selected.pt')
            model.load_state_dict(best_state,strict=True)
            write(output/'train_report.json',evaluate(model,train,train_parents))
            write(output/'selection_report.json',evaluate(model,selection,selection_parents))
            verify(plan['source_sha256']);verify(plan['input_sha256'])
            status=dict(complete=True,training_complete=True,epochs_completed=plan['epochs'],actual_optimizer_steps=steps,
                selected_epoch=best_epoch,selected_checkpoint=pointer(output/'selected.pt'),model_spec=model.model_spec(),
                train_report=pointer(output/'train_report.json'),selection_report=pointer(output/'selection_report.json'),
                initialization_report=pointer(output/'CV_initial_selection_report.json'),parameters=sum(p.numel()for p in model.parameters()),
                wall_seconds=time.perf_counter()-start,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(),device='cuda:0',precision='float32',resources=resources,
                source_unchanged=True,inputs_unchanged=True,heldout_payloads_read=False,TEST_read=False,Native_started=False)
            write(output/'fit_status.json',status)
            print(json.dumps({k:status[k]for k in ('complete','epochs_completed','actual_optimizer_steps','selected_epoch','parameters','wall_seconds')}),flush=True)
            return status
    except BaseException as error:
        write(output/'failure.json',dict(complete=False,error_type=type(error).__name__,error=str(error),
            actual_optimizer_steps=steps,Native_started=False,automatic_retry=False));raise


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--plan-sha256',required=True);parser.add_argument('--execute',action='store_true')
    args=parser.parse_args()
    if args.execute:fit(args.plan,args.plan_sha256)
    else:
        if sha(args.plan)!=args.plan_sha256:raise ValueError('Exact training plan required')
        p=read(args.plan);verify(p['source_sha256']);verify(p['input_sha256'])
        print(json.dumps(dict(complete=True,preparation_only=True,Native=False,GPU=False,model_constructed=False)))


if __name__=='__main__':main()
