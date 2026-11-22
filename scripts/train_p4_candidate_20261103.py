"""Single-task candidate pilot: TRAIN final weights, then separate held-out VAL.

Coverage errors below use the supervised best matching proposal. They are not
the error of an executed candidate, calibrated success probability or task rate.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from our_mani.local_mpc_p4_20261103.model import CandidateTransformerPilot, matching_loss

FEATURES=('future_tokens','robot','geometry','context_c','task_e')


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path,value):
    with Path(path).open('x') as stream: json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False)


def load_dataset(path,split):
    path=Path(path);report=json.loads((path/'report.json').read_text())
    if report.get('split')!=split or sha(path/'dataset.npz')!=report['dataset_sha256']:
        raise ValueError('exact preassigned '+split+' dataset and identity required')
    if (report['feature_source_sha256']!=sha(ROOT/'our_mani/local_mpc_p4_20261103/data.py')
            or report['compiler_sha256']!=sha(ROOT/'scripts/compile_p4_public_dataset_20261103.py')):
        raise ValueError('exact causal feature and dataset compiler contract required')
    with np.load(path/'dataset.npz',allow_pickle=False) as data:
        arrays={key:np.array(data[key],copy=True) for key in data.files}
    if not all(np.isfinite(value).all() for value in arrays.values()):raise ValueError('finite dataset required')
    return arrays,report


def tensor_features(arrays,index):
    return {key:torch.from_numpy(arrays[key][index]) for key in FEATURES}


def loss_output(output,arrays,index):
    return matching_loss(output,torch.from_numpy(arrays['target_position'][index]),
        torch.from_numpy(arrays['target_rotation'][index]),torch.from_numpy(arrays['target_dt'][index]))


def quantiles(values):
    values=np.asarray(values,float)
    return dict(mean=float(values.mean()),p50=float(np.percentile(values,50)),
        p95=float(np.percentile(values,95)),maximum=float(values.max()))


@torch.no_grad()
def evaluate(model,arrays,parents,batch_size=64):
    model.eval();outputs=[]
    for begin in range(0,len(arrays['robot']),batch_size):
        index=slice(begin,begin+batch_size)
        out=model(**tensor_features(arrays,index));loss=loss_output(out,arrays,index)
        # Independently derive metrics at exactly the joint SE3/time match used
        # by supervision; never choose a different favorable slot per metric.
        chosen=loss['matching_index'];batch=torch.arange(len(chosen))
        p=torch.from_numpy(arrays['target_position'][index]);tau=torch.from_numpy(arrays['target_dt'][index])
        position=torch.linalg.vector_norm(out['positions'][batch,chosen]-p,dim=-1)
        timing=torch.abs(out['times'][batch,chosen]-tau)
        outputs.append((position.numpy(),timing.numpy(),loss['per_example'].numpy(),loss['matched_rotation_error_rad'].numpy()))
    position=np.concatenate([row[0] for row in outputs]);timing=np.concatenate([row[1] for row in outputs])
    losses=np.concatenate([row[2] for row in outputs]);weights=arrays['sample_weight']
    return dict(oracle_matched_relative_position_error_m=quantiles(position),
        oracle_matched_parallel_gripper_rotation_error_rad=quantiles(np.concatenate([row[3] for row in outputs])),
        oracle_matched_time_error_seconds=quantiles(timing),
        parent_balanced_loss=float(np.sum(losses*weights)/weights.sum()),
        relative_label_coverage_position_gate_only=bool(position.mean()<=.03 and position.max()<=.05),
        per_parent={parent:dict(samples=int(np.sum(arrays['parent_index']==i)),
            relative_position_error_m=quantiles(position[arrays['parent_index']==i]),
            time_error_seconds=quantiles(timing[arrays['parent_index']==i])) for i,parent in enumerate(parents)},
        scope='oracle supervised matching coverage only; not selected/executed candidate accuracy or task success',
        world_handoff_precision_qualified=False,model_multimodal_coverage_qualified=False,
        task_classifier_trained=False,proposal_success_probabilities_calibrated=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=('train','val'),required=True)
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--epochs',type=int,default=100)
    parser.add_argument('--seed',type=int,default=261103)
    args=parser.parse_args()
    torch.set_num_threads(2);torch.manual_seed(args.seed);np.random.seed(args.seed)
    arrays,report=load_dataset(args.dataset,args.mode)
    if args.epochs<=0:raise ValueError('positive predeclared update budget required')
    if args.mode=='train' and report['positive_independent_parents']<8:
        raise ValueError('pilot gate requires at least8 positive TRAIN parents; no generalization claim implied')
    args.output.mkdir(parents=True,exist_ok=False)
    source_files=[Path(__file__),ROOT/'scripts/compile_p4_public_dataset_20261103.py',
                  *(ROOT/'our_mani/local_mpc_p4_20261103'/name for name in ('__init__.py','model.py','data.py'))]
    identity={str(path):sha(path) for path in source_files}
    model=CandidateTransformerPilot();started=time.perf_counter()
    if args.mode=='train':
        if args.checkpoint is not None:raise ValueError('fresh pilot training does not resume a previous owner')
        optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
        n=len(arrays['robot']);normalizer=float(arrays['sample_weight'].sum());history=[]
        for epoch in range(args.epochs):
            if (args.output/'STOP').exists():
                write_new(args.output/'stopped.json',dict(user_stop=True,completed_updates=len(history)));return 2
            model.train();optimizer.zero_grad(set_to_none=True);loss_total=0.
            # Gradient accumulation gives each parent exactly equal total
            # weight, even when visibility yields different window counts.
            for begin in range(0,n,64):
                index=slice(begin,begin+64);out=model(**tensor_features(arrays,index))
                loss=loss_output(out,arrays,index)['per_example']
                value=(loss*torch.from_numpy(arrays['sample_weight'][index])).sum()/normalizer
                if not torch.isfinite(value):raise ValueError('nonfinite pilot training loss')
                value.backward();loss_total+=float(value.detach())
            grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            optimizer.step();history.append(dict(epoch=epoch+1,parent_balanced_loss=loss_total,gradient_norm=float(grad)))
            if (epoch+1)%10==0:print(json.dumps(history[-1]),flush=True)
        checkpoint=dict(schema='P4_single_task_candidate_pilot_final_v1',model_state=model.state_dict(),
            model_spec=model.spec(),train_parents=report['parents'],train_dataset_sha256=report['dataset_sha256'],
            training_updates=args.epochs,seed=args.seed,source_sha256=identity,
            selection='predeclared final update; no VAL/TEST checkpoint selection',task_classifier_trained=False)
        torch.save(checkpoint,args.output/'final.pt')
        write_new(args.output/'training_history.json',history)
        checkpoint_path=args.output/'final.pt'
    else:
        if args.checkpoint is None:raise ValueError('frozen TRAIN checkpoint required for VAL')
        checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
        if set(report['parents'])&set(checkpoint['train_parents']):raise ValueError('TRAIN/VAL parent leakage')
        if checkpoint['model_spec']!=model.spec():raise ValueError('exact model specification required')
        for name,expected in checkpoint['source_sha256'].items():
            if sha(name)!=expected:raise ValueError('checkpoint training/model/data source changed: '+name)
        model.load_state_dict(checkpoint['model_state'],strict=True);checkpoint_path=args.checkpoint
    metrics=evaluate(model,arrays,report['parents'])
    if any(sha(path)!=expected for path,expected in identity.items()):raise ValueError('pilot source changed during execution')
    result=dict(mode=args.mode,complete=True,metrics=metrics,elapsed_seconds=time.perf_counter()-started,
        checkpoint=str(checkpoint_path),checkpoint_sha256=sha(checkpoint_path),
        dataset=str(args.dataset),dataset_sha256=report['dataset_sha256'],source_sha256=identity,
        real_robot_or_Native_started=False,formal_TEST_read=False,old_training_resumed=False,
        contact_response_trained=False,task_classifier_trained=False,formal_qualification=False)
    result['data_scope']=dict(total_teacher_cases=len(report['cases']),
        parents_with_accepted_contact_labels=report['positive_independent_parents'],
        cases_without_accepted_positive_label=sum(case['accepted_contact_labels']==0 for case in report['cases']),
        supervised_metrics_conditioned_on_later_successful_teacher_demonstrations=True,
        failed_teacher_episodes_retained_in_data_report=True,
        independent_sample_count_is_parent_count_not_window_count=True)
    write_new(args.output/'report.json',result)
    print(json.dumps(dict(mode=args.mode,complete=True,metrics=metrics,elapsed_seconds=result['elapsed_seconds'])),flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
