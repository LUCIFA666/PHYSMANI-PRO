"""Train public-reference contact proposals on corrected-motion teacher data.

Real contact tau is never clipped. Far approach states receive pose supervision
only. Candidate coverage metrics remain separate from actual MPC selection and
closed-loop task outcomes. No task-classification or success score is trained.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch

FEATURES=('future_tokens','robot','geometry','context_c','task_e')
TARGETS=('target_position','target_rotation','target_dt','target_time_valid')


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def pointer(p):return dict(path=str(Path(p).resolve(strict=True)),sha256=sha(p))


def write(p,value):
    with Path(p).open('x')as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')


def verify(mapping):
    for p,h in mapping.items():
        if sha(p)!=h:raise ValueError('Frozen training source/input changed: '+p)


def load_role(path,*,role,parents,forecast_sha256):
    from our_mani.cube_candidate_matched_20261104_v2 import FRAME
    path=Path(path);report=read(path/'report.json')
    if (report.get('role')!=role or report.get('parents')!=parents
            or report.get('forecast_checkpoint_sha256')!=forecast_sha256 or report.get('target_frame')!=FRAME):
        raise ValueError('Exact whole-role, frame and corrected-forecast metadata required before payload access')
    if sha(path/'dataset.npz')!=report['dataset_sha256']:raise ValueError('Compiled candidate bytes changed')
    with np.load(path/'dataset.npz',allow_pickle=False)as f:arrays={k:np.array(f[k],copy=True)for k in f.files}
    required=set(FEATURES+TARGETS+('parent_index','sample_weight'))
    if set(arrays)!=required:raise ValueError('Exact eleven public-feature/label/weight tensors required')
    if any(not np.isfinite(v).all()for v in arrays.values()):raise ValueError('Finite candidate data required')
    if arrays['target_time_valid'].dtype!=np.bool_:raise ValueError('Actual bounded-horizon time mask must be Boolean')
    n=len(arrays['robot']);idx=arrays['parent_index'];weight=arrays['sample_weight']
    if (idx.dtype!=np.int64 or idx.shape!=(n,) or weight.shape!=(n,)
            or np.any(idx<0) or np.any(idx>=len(parents)) or np.any(weight<=0)):
        raise ValueError('Full-role int64 parent indices and positive aligned weights required')
    for parent in np.unique(idx):
        mask=idx==parent
        if not np.allclose(weight[mask],1./int(mask.sum()),atol=1e-8,rtol=1e-6):
            raise ValueError('Each supervised parent must have total sample weight one')
    if report.get('supervised_parents')!=[parents[i]for i in sorted(set(idx.tolist()))]:
        raise ValueError('Supervised subset must agree with full-role parent indices')
    return arrays,report


def device_data(arrays,device):return {k:torch.from_numpy(v).to(device)for k,v in arrays.items()}
def features(data,ids):return {k:data[k][ids]for k in FEATURES}


def loss_for(output,data,ids):
    from our_mani.cube_candidate_matched_20261104_v2.losses import candidate_loss
    return candidate_loss(output,**{k:data[k][ids]for k in TARGETS},ranking_weight=0.)


def quantiles(values):
    x=np.asarray(values,float)
    return dict(count=len(x),mean=float(x.mean()),p50=float(np.median(x)),p95=float(np.percentile(x,95)),maximum=float(x.max()))if len(x)else dict(count=0)


@torch.no_grad()
def evaluate(model,data,parents,batch_size=64):
    model.eval();n=len(data['robot'])
    if n==0:return dict(evaluated=False,reason='no_observed_local_grasp_positive_in_this_role',windows=0,parent_balanced_loss=None)
    values={k:[]for k in ('loss','p','r','tau','slot')}
    for start in range(0,n,batch_size):
        ids=slice(start,start+batch_size);out=model(**features(data,ids));terms=loss_for(out,data,ids)
        slot=terms['matching_index'];values['slot'].append(slot.cpu().numpy());values['loss'].append(terms['per_example'].cpu().numpy())
        for key,source in (('p','position_error_m'),('r','rotation_error_rad'),('tau','time_error_seconds')):
            values[key].append(terms[source].gather(1,slot[:,None]).squeeze(1).cpu().numpy())
    values={k:np.concatenate(v)for k,v in values.items()}
    idx=data['parent_index'].cpu().numpy();w=data['sample_weight'].cpu().numpy();near=data['target_time_valid'].cpu().numpy()
    return dict(evaluated=True,windows=n,supervised_parents=[parents[i]for i in sorted(set(idx.tolist()))],
        parent_balanced_loss=float(np.sum(values['loss']*w)/w.sum()),
        oracle_joint_matched_position_error_m=quantiles(values['p']),
        oracle_joint_matched_rotation_error_rad=quantiles(values['r']),
        oracle_joint_matched_time_error_seconds_valid_near_only=quantiles(values['tau'][near]),
        near_time_supervised_windows=int(near.sum()),far_pose_only_windows=int((~near).sum()),
        matched_slot_counts=np.bincount(values['slot'],minlength=8).tolist(),
        per_parent={parents[i]:dict(windows=int(np.sum(idx==i)),position_error_m=quantiles(values['p'][idx==i]),
            time_error_seconds_valid_near_only=quantiles(values['tau'][(idx==i)&near]))for i in sorted(set(idx.tolist()))},
        metric_scope='GT joint pose/time matching coverage only; live MPC choice/arrival remains separately evaluated',
        MPC_selected_candidate_metric='not_evaluated_by_offline_oracle',actual_arrival_metric='not_evaluated_by_offline_oracle',
        calibrated_success_probability=False,position_hard_gate_used=False)


def stopped(out):
    if any(p.exists()for p in (out/'STOP',out.parent/'STOP',ROOT/'runtime/framework/STOP',ROOT/'runtime/framework/local_mpc_20261102/STOP')):
        raise InterruptedError('Administrative STOP; retain all partial data')


def fit(plan_path,digest):
    if sha(plan_path)!=digest:raise ValueError('External frozen candidate fit plan SHA required')
    plan=read(plan_path)
    if plan.get('schema')!='Cube_matched_candidate_fit_20261104_v1':
        raise ValueError('Explicit matched public-reference candidate fit plan required')
    verify(plan['source_sha256']);verify(plan['input_sha256'])
    if sha(plan['partition']['path'])!=plan['partition']['sha256']:
        raise ValueError('Exact whole-parent role partition required')
    partition=read(plan['partition']['path']);roles={k:[r['parent_id']for r in partition['rows']if r['role']==k]for k in ('fit','selection','heldout')}
    train_np,tr=load_role(plan['fit_dataset'],role='fit',parents=roles['fit'],forecast_sha256=plan['forecast_checkpoint_sha256'])
    sel_np,sr=load_role(plan['selection_dataset'],role='selection',parents=roles['selection'],forecast_sha256=plan['forecast_checkpoint_sha256'])
    if len(train_np['robot'])==0:raise ValueError('No observed TRAIN local grasp contacts; do not fabricate supervision')
    out=Path(plan['output']);stopped(out);out.mkdir(parents=True,exist_ok=False)
    write(out/'request.json',dict(plan=pointer(plan_path),configuration=plan,heldout_read=False,Native_started=False))
    start=time.perf_counter();steps=0
    try:
        from our_mani.current_cube_rgbd_centre_large_20261104_v2.training import exclusive_resources
        from our_mani.cube_matched_candidate_20261104_v1.model import CubeMatchedCandidateTransformer
        from our_mani.cube_candidate_matched_20261104_v2 import FRAME
        from our_mani.cube_matched_candidate_20261104_v1 import CHECKPOINT_SCHEMA
        with exclusive_resources()as resource:
            if not torch.cuda.is_available():raise RuntimeError('Actual GPU required; no silent CPU fit')
            os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
            torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
            torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
            torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True)
            torch.set_num_threads(2);torch.manual_seed(plan['seed']);torch.cuda.manual_seed_all(plan['seed']);torch.cuda.reset_peak_memory_stats()
            model,transfer=CubeMatchedCandidateTransformer.from_legacy_checkpoint(plan['initial_checkpoint'],expected_sha256=plan['initial_checkpoint_sha256'])
            write(out/'weight_initialization.json',transfer);model.to('cuda:0')
            train=device_data(train_np,'cuda:0');selection=device_data(sel_np,'cuda:0')
            initial=evaluate(model,selection,roles['selection']);write(out/'initial_selection_report.json',initial)
            optimizer=torch.optim.AdamW(model.parameters(),lr=plan['learning_rate'],weight_decay=plan['weight_decay'])
            rng=np.random.default_rng(plan['seed']);groups={i:np.flatnonzero(train_np['parent_index']==i)for i in range(len(roles['fit']))}
            groups={i:v for i,v in groups.items()if len(v)}
            best=float('inf');best_state=None;best_epoch=None
            with(out/'history.jsonl').open('x')as log:
                for epoch in range(1,plan['epochs']+1):
                    stopped(out);model.train();total=0.
                    for _ in range(plan['steps_per_epoch']):
                        ids=np.concatenate([rng.choice(v,plan['windows_per_parent_per_step'],replace=True)for v in groups.values()]);rng.shuffle(ids)
                        ids=torch.as_tensor(ids,device='cuda:0');optimizer.zero_grad(set_to_none=True)
                        terms=loss_for(model(**features(train,ids)),train,ids);loss=terms['per_example'].mean()
                        if not bool(torch.isfinite(loss)):raise ValueError('Nonfinite matched candidate objective')
                        loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                        optimizer.step();steps+=1;total+=float(loss.detach())
                    entry=dict(epoch=epoch,optimizer_steps=steps,train_loss=total/plan['steps_per_epoch'],gradient_norm=float(norm),elapsed_seconds=time.perf_counter()-start)
                    if epoch%plan['selection_interval']==0 or epoch==plan['epochs']:
                        m=evaluate(model,selection,roles['selection']);entry['selection_evaluated']=m['evaluated']
                        if m['evaluated']:
                            entry['selection_loss']=m['parent_balanced_loss']
                            if m['parent_balanced_loss']<best:
                                best=m['parent_balanced_loss'];best_epoch=epoch;best_state={k:v.detach().cpu().clone()for k,v in model.state_dict().items()}
                        elif epoch==plan['epochs']:
                            best_epoch=epoch;best_state={k:v.detach().cpu().clone()for k,v in model.state_dict().items()}
                        entry['selected_epoch']=best_epoch;print(json.dumps(entry),flush=True)
                    log.write(json.dumps(entry,allow_nan=False)+'\n');log.flush()
            if best_state is None:raise RuntimeError('No completed trained checkpoint available')
            rule=('minimum existing development parent-balanced masked pose/time loss'if initial['evaluated']else'predeclared final update; no selection-positive labels; live development evaluation required')
            common=dict(schema=CHECKPOINT_SCHEMA,model_spec=model.spec(),model_config=model.config.to_dict(),source_sha256=plan['source_sha256'],
                forecast_checkpoint_sha256=plan['forecast_checkpoint_sha256'],target_frame=FRAME,
                train_parents=roles['fit'],selection_parents=roles['selection'],heldout_parents=roles['heldout'],
                supervised_train_parents=tr['supervised_parents'],supervised_selection_parents=sr['supervised_parents'],
                training_coverage='Only listed supervised parents contribute gradients; zero-positive assigned parents are retained in the role denominator',
                train_dataset_sha256=tr['dataset_sha256'],selection_dataset_sha256=sr['dataset_sha256'],
                training_updates=steps,epochs_completed=plan['epochs'],selected_epoch_indexing='one_based',seed=plan['seed'],
                task_classifier_trained=False,selection=rule,time_supervision='actual_tau_masked_outside_0p05_0p8_no_clamp',
                proposal_score_semantics=model.spec()['proposal_score_semantics'],local_grasp_not_full_task_supervision=True,
                heldout_used_for_fitting=False,TEST_used_for_fitting=False)
            torch.save(dict(common,model_state={k:v.detach().cpu().clone()for k,v in model.state_dict().items()},selected_epoch=plan['epochs']),out/'final.pt')
            torch.save(dict(common,model_state=best_state,selected_epoch=best_epoch),out/'selected.pt')
            model.load_state_dict(best_state,strict=True)
            write(out/'train_report.json',evaluate(model,train,roles['fit']));write(out/'selection_report.json',evaluate(model,selection,roles['selection']))
            verify(plan['source_sha256']);verify(plan['input_sha256'])
            status=dict(complete=True,training_complete=True,epochs_completed=plan['epochs'],actual_optimizer_steps=steps,
                selected_epoch=best_epoch,selected_epoch_indexing='one_based',selected_checkpoint=pointer(out/'selected.pt'),
                model_spec=model.spec(),parameters=sum(p.numel()for p in model.parameters()),
                train_report=pointer(out/'train_report.json'),selection_report=pointer(out/'selection_report.json'),
                supervised_fit_parents=tr['supervised_parents'],supervised_selection_parents=sr['supervised_parents'],
                full_assigned_parent_roles=roles,heldout_payloads_read=False,TEST_read=False,Native_started=False,
                source_unchanged=True,input_unchanged=True,wall_seconds=time.perf_counter()-start,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),resources=resource)
            write(out/'fit_status.json',status)
            print(json.dumps({k:status[k]for k in ('complete','epochs_completed','actual_optimizer_steps','selected_epoch','parameters','wall_seconds')}),flush=True)
            return status
    except BaseException as error:
        write(out/'failure.json',dict(complete=False,error_type=type(error).__name__,error=str(error),actual_optimizer_steps=steps,Native_started=False,automatic_retry=False));raise


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',type=Path,required=True);p.add_argument('--plan-sha256',required=True);p.add_argument('--execute',action='store_true')
    a=p.parse_args()
    if a.execute:fit(a.plan,a.plan_sha256)
    else:
        if sha(a.plan)!=a.plan_sha256:raise ValueError('Exact plan required')
        plan=read(a.plan)
        if plan.get('schema')!='Cube_matched_candidate_fit_20261104_v1':raise ValueError('Explicit matched candidate fit plan required')
        verify(plan['source_sha256']);verify(plan['input_sha256']);print(json.dumps(dict(preparation_only=True,NN=False,GPU=False,Native=False)))


if __name__=='__main__':main()
