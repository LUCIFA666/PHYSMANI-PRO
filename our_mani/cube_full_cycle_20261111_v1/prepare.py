"""Freeze every official source identity before collection; no TEST payload read."""
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import h5py
from .common import ROOT,OUT,PLAN,sha,read,write,pointer,identity

DATA=Path('/home/ubuntu/桌面/phsimani/PhysMani/data/rmt/physmani_bench')
TASKS=('pick_moving_target_on_the_table','pick_moving_target_on_the_table_high_speed')


def number(name):return int(name.removeprefix('episode'))
def derived_seed(split,task,raw):return int(hashlib.sha256(f'CubeFull20261111:{split}:{task}:{raw}'.encode()).hexdigest()[:8],16)


def main():
    if PLAN.exists()or OUT.exists():raise FileExistsError('Preserve previous full-cycle attempt')
    PLAN.parent.mkdir(parents=True,exist_ok=False);OUT.mkdir(parents=True,exist_ok=False)
    old=read(ROOT/'config/framework/cube_training_cycle_20261111_v1/plan.json')
    prior=read(ROOT/'artifacts/framework/cube_training_cycle_20261111_v1/report.json')
    initialization=deepcopy(old['model_initialization'])
    for model,key in(('motion_training','corrected_forecast_checkpoint'),('candidate_training','candidate_transformer_checkpoint')):
        cp=prior[model]['checkpoint'];initialization[key]=str(Path(cp['path']).relative_to(ROOT));initialization[key+'_sha256']=cp['sha256']
    cases=[]
    for split,package in(('train','train_package_compressed'),('validation','val_package_compressed')):
        by_task={}
        for task in TASKS:
            values=[]
            for path in sorted((DATA/package/task/'all_variations/episodes').glob('*/data.h5'),key=lambda q:number(q.parent.name)):
                with h5py.File(path,'r')as f:
                    variation=int(f.attrs['variation']);raw='episode'+str(int(f.attrs['episode']))
                    if str(f.attrs['task'])!=task or raw!=path.parent.name:raise ValueError('Original HDF identity differs')
                values.append(dict(source_split=split,task=task,raw_episode_id=raw,source_path=str(path.resolve()),
                    source_sha256=sha(path),variation=variation,seed=derived_seed(split,task,raw)))
            if len(values)!=(100 if split=='train'else 20):raise ValueError('Complete task HDF source coverage required')
            selected=set()
            if split=='train':
                for variation in(0,1):
                    group=[v for v in values if v['variation']==variation]
                    if len(group)!=50:raise ValueError('Expected original TRAIN50 per variation')
                    rank=sorted(group,key=lambda v:hashlib.sha256(('TRAINselection20261111:'+task+':'+v['raw_episode_id']).encode()).hexdigest())
                    selected.update(v['raw_episode_id']for v in rank[:10])
            for v in values:v['role']=('selection'if v['raw_episode_id']in selected else'fit')if split=='train'else'validation'
            by_task[task]=values
        for pair in zip(*(by_task[t]for t in TASKS)):cases.extend(pair)
    # Metadata/aliases only; archived target states, demos and RGBD stay unopened.
    test_metadata=ROOT/'artifacts/framework/pipeline_v1/official_test_manifest.json'
    original=read(test_metadata)
    for row in original['episodes']:
        if row['task']not in TASKS:continue
        alias=DATA/'test'/row['task']/f'variation{row["variation"]}'/'episodes'/f'episode{row["official_episode_index"]}'
        if not alias.is_symlink()or alias.resolve()!=Path(row['path']).resolve():raise ValueError('Original author TEST mapping differs')
        cases.append(dict(source_split='test',role='test',task=row['task'],raw_episode_id=row['raw_episode_id'],
            source_path=row['path'],variation=int(row['variation']),official_episode_index=row['official_episode_index'],
            official_alias_path=str(alias),seed=int(row.get('seed',0))))
    for c in cases:
        parent=f'{c["source_split"]}__{c["task"]}__{c["raw_episode_id"]}'
        c.update(parent_id=parent,id=parent,episode_id=parent+'__actual_capture_v1',kind='pipeline',calibration_episode=c['source_path'])
        cfg=deepcopy(next(r['config']for r in old['cases']if r['task']==c['task']))
        cfg.update(initialization);cfg.update(task=c['task'],seed=c['seed'],simulation_seconds=60.,task_time_limit_seconds=60.,budget_ticks=1200,
            save_rgbd_frames=True,raw_rgbd_stride=20,per_frame_pixel_hash=True)
        c['config']=cfg
    p=dict(schema='Cube_full_train_validation_test_cycle_20261111_v1',training_authorized=True,
        user_authorization='全部TRAIN重新训练，再跑完整VAL/TEST；PhysMani普通/高速方块；任务60秒',
        tasks=list(TASKS),historical_global_stop=identity(ROOT/'runtime/framework/STOP'),cases=cases,
        model_initialization=initialization,motion_training=dict(epochs=200,steps_per_epoch=8,windows_per_parent_per_step=2,
            selection_interval=5,learning_rate=3e-5,weight_decay=1e-4,seed=2611119200),
        candidate_training=dict(epochs=150,steps_per_epoch=8,windows_per_parent_per_step=2,
            selection_interval=5,learning_rate=5e-5,weight_decay=1e-3,seed=2611119201),
        minimum_supervised_parents_per_speed=dict(fit=2,selection=1),
        expected_source_counts=dict(train=200,validation=40,test=80),training_selection_partition=dict(fit=160,selection=40),
        validation_selection_rule='TRAIN-internal selection only; newly fitted pair frozen before complete VAL40 and unchanged for TEST80',
        source_interpretation=dict(train='one original-native generated episode per original HDF TRAIN source; source-specific camera/home/variation; not HDF motion replay',
            validation='one original-native generated episode per original HDF VAL source; no model fitting; not full archived VAL scene replay',
            test='all80 released official archived resets, including both variations; model inputs current/past RGBD and robot only'),
        official_test_identity_manifest=pointer(test_metadata),
        official_test_hash_manifest=pointer(ROOT/'artifacts/framework/multitask_r10/formal_TEST_after_environment_resume_v826_setup/official_test_manifest16.json'),
        success_authority='Original PhysMani Task.success first true, checked every actual50ms; stop immediately',
        task_time_limit_seconds=60.,original_paper_10s_protocol_comparable=False,Native_GPU_training_overlap=False,
        automatic_case_retry=False,old_queues_resumed=False,raw_rgbd_policy='Every actual frame hashed; raw RGBD saved every20 ticks to bound disk use')
    write(PLAN,p)
    counts=Counter((c['source_split'],c['task'],c['variation'])for c in cases)
    write(OUT/'source_inventory.json',dict(complete=True,cases=len(cases),counts=[dict(split=a,task=b,variation=c,sources=n)for(a,b,c),n in sorted(counts.items())],
        partition=pointer(PLAN),TEST_payloads_read=False,TRAIN_expert_trajectory_or_WM_predictions_read=False))
    print(json.dumps(dict(complete=True,cases=len(cases),source_counts=p['expected_source_counts'],partition=pointer(PLAN))))

if __name__=='__main__':main()
