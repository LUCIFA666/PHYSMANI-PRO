"""CPU admission of joint16 causal windows, without optimization or TEST."""
from pathlib import Path
import json
import sys
from collections import Counter
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from our_mani.multitask.evaluation import atomic_json,file_sha
from our_mani.multitask.context_se3 import SE3ContextNetwork
from our_mani.multitask.joint16_tracks_v31 import read_joint16_tracks
from our_mani.multitask.track_dataset import train_normalization,examples
from our_mani.multitask.fitting import balanced_parent_indices


def main():
    out=ROOT/'artifacts/framework/multitask_r10/joint16_window_admission_v38'
    if out.exists():raise FileExistsError('preserve prior CPU data admission')
    manifest=ROOT/'artifacts/framework/multitask_r10/joint16_body_track_adapter_v31/manifest.json'
    torch.set_num_threads(2);metadata,episodes=read_joint16_tracks(ROOT,manifest)
    normalization,normalization_coverage=train_normalization(episodes);model=SE3ContextNetwork(**normalization)
    reports={}
    for split,count in [('train',1600),('validation',315)]:
        samples,coverage=examples([row for row in episodes if row.entry['split']==split],model,22)
        if len(coverage)!=count:raise ValueError('missing original joint16 parents in causal windows')
        tasks={row['source_key'][1] for row in samples};full_tasks={row['source_key'][1] for row in samples if row['mask'][-1]}
        if len(tasks)!=16 or tasks!=full_tasks:raise ValueError('tasks missing original full-horizon targets')
        if split=='train':
            if set(coverage)!=set(normalization_coverage):raise ValueError('normalization and optimizer source groups differ')
            indices=balanced_parent_indices(samples,64,np.random.default_rng(1018));balanced=Counter(samples[int(i)]['source_key'] for i in indices)
            if len(balanced)!=1600 or set(balanced.values())!={64}:raise ValueError('joint sampler must use every parent equally')
        reports[split]=dict(parents=len(coverage),windows=len(samples),full_horizon_windows=sum(bool(row['mask'][-1]) for row in samples),
            per_task={task:dict(parents=sum(key[1]==task for key in coverage),windows=sum(row['source_key'][1]==task for row in samples),
                full_horizon_windows=sum(row['source_key'][1]==task and bool(row['mask'][-1]) for row in samples)) for task in sorted(tasks)})
    out.mkdir();report=dict(complete=True,manifest=dict(path=str(manifest.relative_to(ROOT)),sha256=file_sha(manifest)),
        source=dict(path=str(Path(__file__).relative_to(ROOT)),sha256=file_sha(__file__)),normalization=normalization,
        reports=reports,balanced_TRAIN_windows_per_parent=64,model_parameters=sum(p.numel() for p in model.parameters()),
        all16_tasks_each_split_have_full_1_1_second_targets=True,normalization_fit_split='train',
        optimizer_steps=0,GPU_used=False,TEST_read=False,executor_admission=False,formal_VAL_TEST_ready=False)
    atomic_json(out/'report.json',report);print(json.dumps({k:v for k,v in report.items() if k not in ('reports','normalization')}),flush=True)


if __name__=='__main__':main()
