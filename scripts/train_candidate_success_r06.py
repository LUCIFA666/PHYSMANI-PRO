"""Fit actual candidate-attempt outcomes; TEST files cannot enter this loader."""
import argparse
from collections import Counter
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from scipy.special import expit,logit
from scipy.spatial import cKDTree
import torch
from our_mani.local_learning import MLP,probability_metrics
from our_mani.risk_v2.selector import attempt_controller_hash
from our_mani.risk_v2.model import TASKS,fit_calibration,calibrated_probabilities
from our_mani.benchmark_protocol import file_sha256


def read(predictor,split):
    base=ROOT/'artifacts/framework/aligned_r06'
    expected={(r['task'],r['raw_episode_id']) for r in json.loads((ROOT/f'artifacts/framework/protocol/{split}_manifest.json').read_text())['episodes'] if r['task'] in TASKS}
    episodes=[];samples=[];files={};keys=set()
    config=json.loads((ROOT/'config/framework/aligned_protocol_r06.json').read_text())
    expected_controller=attempt_controller_hash(ROOT,config,predictor)
    for directory in sorted(base.glob(f'collection_{predictor}_{split}_*')):
        report=json.loads((directory/'report.json').read_text())
        if not report['complete']:raise ValueError('candidate collection is incomplete')
        for p in sorted(directory.glob('pick*__episode*__*.json')):
            r=json.loads(p.read_text());key=(r['task'],r['source_episode'],r['replica'])
            if key in keys:raise ValueError('duplicate candidate rollout')
            if r['split']!=split or r['formal_test_used']:raise ValueError('TEST or incorrect split in probability fitting')
            if r.get('attempt_controller_sha256')!=expected_controller:raise ValueError('mixed full-candidate controller versions')
            if (r['task'],r['source_episode']) not in expected:raise ValueError('unrecognized training source episode')
            keys.add(key);episodes.append(r);files[str(p.relative_to(ROOT))]=file_sha256(p)
            for attempt in r['attempts']:
                if not attempt['complete']:raise ValueError('censored candidate attempt cannot be treated as failure')
                samples.append(dict(task=r['task'],source_episode=r['source_episode'],**attempt))
    if {(r['task'],r['source_episode']) for r in episodes}!=expected:raise ValueError('all source episodes must be executed')
    if len(episodes)!=3*len(expected):raise ValueError('three candidate sampling rollouts per source episode required')
    return samples,episodes,files


def source_weights(rows):
    counts=Counter((r['task'],r['source_episode']) for r in rows)
    tasks=Counter(t for t,e in counts)
    return np.array([1./(tasks[r['task']]*counts[r['task'],r['source_episode']]) for r in rows])


def metrics_by_task(probability,rows):
    y=np.array([r['official_task_success'] for r in rows],float)
    return {task:probability_metrics(np.asarray(probability)[mask:=[r['task']==task for r in rows]],y[mask]) for task in TASKS}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--predictor',choices=('cv','context'),required=True)
    p.add_argument('--updates',type=int,default=2000)
    args=p.parse_args();torch.set_num_threads(2)
    config=json.loads((ROOT/'config/framework/aligned_protocol_r06.json').read_text())
    train,train_episodes,train_files=read(args.predictor,'train')
    val,val_episodes,val_files=read(args.predictor,'validation')
    train_groups={(r['task'],r['source_episode']) for r in train_episodes}
    val_groups={(r['task'],r['source_episode']) for r in val_episodes}
    if train_groups&val_groups:raise ValueError('TRAIN/VAL episode overlap')
    rng=np.random.default_rng(810);selection=set();calibration_groups=set()
    for task in TASKS:
        groups=sorted(g for g in val_groups if g[0]==task)
        groups=[groups[i] for i in rng.permutation(len(groups))]
        selection.update(groups[:10]);calibration_groups.update(groups[10:])
    select=[r for r in val if (r['task'],r['source_episode']) in selection]
    calibration=[r for r in val if (r['task'],r['source_episode']) in calibration_groups]
    if not train or not select or not calibration:raise ValueError('empty candidate split')
    x=np.asarray([r['features'] for r in train],np.float32);y=np.asarray([r['official_task_success'] for r in train],np.float32)
    xs=np.asarray([r['features'] for r in select],np.float32);ys=np.array([r['official_task_success'] for r in select],np.float32)
    xv=np.asarray([r['features'] for r in val],np.float32);yv=np.array([r['official_task_success'] for r in val],np.float32)
    mean=x.mean(0);std=np.maximum(x.std(0),.05)
    tensor=lambda a:torch.tensor((a-mean)/std,dtype=torch.float32)
    xt,xst,xvt=tensor(x),tensor(xs),tensor(xv)
    yt=torch.tensor(y[:,None]);yst=torch.tensor(ys[:,None])
    groups=sorted({(r['task'],r['source_episode']) for r in train})
    group_rows={g:np.array([i for i,r in enumerate(train) if (r['task'],r['source_episode'])==g]) for g in groups}
    weights=source_weights(select);weights/=weights.sum()
    saved=[];histories=[];selected_updates=[];members=[]
    for member in range(5):
        seed=810+member;torch.manual_seed(seed);random=np.random.default_rng(seed)
        bootstrap=[groups[i] for i in random.integers(len(groups),size=len(groups))]
        model=MLP(x.shape[1],1,64);optimizer=torch.optim.AdamW(model.parameters(),lr=4e-4,weight_decay=.01)
        best=np.inf;best_state=None;history=[];selected_update=None
        for update in range(args.updates):
            if (ROOT/'runtime/framework/success_r06_STOP').exists():raise InterruptedError('r06 success fitting stopped')
            chosen=[bootstrap[i] for i in random.integers(len(bootstrap),size=128)]
            batch=np.array([random.choice(group_rows[g]) for g in chosen])
            loss=torch.nn.functional.binary_cross_entropy_with_logits(model(xt[batch]),yt[batch])
            optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),2.);optimizer.step()
            if update%100==0 or update==args.updates-1:
                with torch.inference_mode():
                    losses=torch.nn.functional.binary_cross_entropy_with_logits(model(xst),yst,reduction='none').numpy()[:,0]
                score=float(weights@losses);history.append(dict(update=update+1,selection_nll=score))
                if score<best:
                    best=score;selected_update=update+1;best_state={k:v.detach().clone() for k,v in model.state_dict().items()}
        model.load_state_dict(best_state);model.eval()
        with torch.inference_mode():members.append(torch.sigmoid(model(xvt)).numpy()[:,0])
        saved.append(best_state);histories.append(history);selected_updates.append(selected_update)
    raw=np.stack(members).mean(0)
    cmask=np.array([(r['task'],r['source_episode']) in calibration_groups for r in val])
    cal=fit_calibration(logit(np.clip(raw[cmask],1e-6,1-1e-6)),yv[cmask],
        [r['task'] for r in calibration],source_weights(calibration))
    probabilities=calibrated_probabilities(raw,[r['task'] for r in val],cal)
    # Leave-one-source-out calibration diagnostics: held source has no influence
    # on the affine correction, and is never part of neural checkpoint selection.
    crossfit=np.zeros(cmask.sum());calraw=raw[cmask];caly=yv[cmask]
    for group in sorted(calibration_groups):
        hold=np.array([(r['task'],r['source_episode'])==group for r in calibration])
        if not hold.any():continue
        fitrows=[r for r,h in zip(calibration,hold) if not h]
        fit=fit_calibration(logit(np.clip(calraw[~hold],1e-6,1-1e-6)),caly[~hold],[r['task'] for r in fitrows],source_weights(fitrows))
        crossfit[hold]=calibrated_probabilities(calraw[hold],[r['task'] for r,h in zip(calibration,hold) if h],fit)
    out=ROOT/f'artifacts/framework/aligned_r06/{args.predictor}_success';out.mkdir(parents=True,exist_ok=True)
    if (out/'value.pt').exists():raise FileExistsError('success weights are immutable; choose a new version')
    metadata=dict(created_utc=datetime.now(timezone.utc).isoformat(),kind='candidate_attempt_success_v2',
        probability_scope='success before candidate replacement, including actual approach/capture/transport and remaining episode time',
        controller_sha256=attempt_controller_hash(ROOT,config,args.predictor),predictor=args.predictor,
        context_dim=16 if args.predictor=='context' else 0,input_dim=x.shape[1],
        formal_test_used=False,all_200_TRAIN_source_episodes_executed=True,train_source_episodes=len(train_groups),
        train_source_episodes_with_candidate_features=len(groups),validation_source_episodes=len(val_groups),
        selection_groups=sorted(selection),calibration_groups=sorted(calibration_groups),selected_updates=selected_updates,
        train_data_sha256=hashlib.sha256(json.dumps(train_files,sort_keys=True).encode()).hexdigest(),
        validation_data_sha256=hashlib.sha256(json.dumps(val_files,sort_keys=True).encode()).hexdigest(),
        context_encoder_sha256=next(iter({r['context_encoder_sha256'] for r in train if r.get('context_encoder_sha256')}),None),
        limitations=['HDF5 development resets, not official TEST replay','probability is candidate-attempt specific, not eventual success across retries',
            'collection geometry/softmax/uniform policies differ from learned selection; policy distribution shift remains',
            'VAL selection/calibration metrics are development diagnostics; only calibration crossfit leaves held sources out of correction fitting'])
    torch.save(dict(state_dicts=saved,feature_mean=mean,feature_std=std,hidden=64,calibration=cal,metadata=metadata),out/'value.pt')
    support_scale=np.maximum(x.std(0),.05);z=(x-mean)/support_scale;tree=cKDTree(z)
    threshold=float(np.quantile(tree.query((xs-mean)/support_scale)[0],.99))
    slack=np.maximum(.1*x.std(0),.02)
    support=dict(value_checkpoint_sha256=file_sha256(out/'value.pt'),mean=mean.tolist(),scale=support_scale.tolist(),
        lower=(x.min(0)-slack).tolist(),upper=(x.max(0)+slack).tolist(),normalized_fit_features=z.tolist(),
        maximum_nearest_fit_distance=threshold,formal_test_used=False,rule='TRAIN bounds; q99 nearest TRAIN distance on VAL-selection sources')
    runner_write=lambda p,j:p.write_text(json.dumps(j,ensure_ascii=False,indent=2)+'\n')
    runner_write(out/'value_support.json',support)
    prior=np.array([np.mean([r['official_task_success'] for r in train if r['task']==v['task']]) for v in val])
    report=dict(metadata=metadata,train_episodes=len(train_episodes),train_attempts=len(train),validation_attempts=len(val),
        calibration=cal,validation_raw=probability_metrics(raw,yv),validation_calibrated=probability_metrics(probabilities,yv),
        per_task=metrics_by_task(probabilities,val),task_prevalence_baseline=metrics_by_task(prior,val),
        calibration_source_crossfit=metrics_by_task(crossfit,calibration),
        calibration_source_raw=metrics_by_task(calraw,calibration),training_history=histories)
    runner_write(out/'report.json',report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('metadata','training_history')},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
