"""Prepare all block histories for later joint16 training; does not train."""
from pathlib import Path
import hashlib
import json
import sys
from collections import Counter
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import h5py
import numpy as np
from our_mani.multitask.evaluation import atomic_json,file_sha
from our_mani.multitask.joint16_tracks_v31 import SCHEMA,BLOCKS,block_arrays,precontact_end,read_joint16_tracks
from our_mani.multitask.track_dataset import read_tracks

BASE=ROOT/'artifacts/framework/multitask_r10'
OUT=BASE/'joint16_body_track_adapter_v31'


def pointer(path):return dict(path=str(path.relative_to(ROOT)),sha256=file_sha(path))


def main():
    if OUT.exists():raise FileExistsError('preserve previous data preparation')
    source=ROOT/'artifacts/framework/pipeline_v1/context_tracks/tracks_manifest.json';old=json.loads(source.read_text())
    if old['formal_test_used'] is not False or any(e['split'] not in ('train','validation') or e['task'] not in BLOCKS for e in old['entries']):raise ValueError('TRAIN/VAL block tracks only')
    OUT.mkdir();(OUT/'tracks').mkdir();entries=[];inputs={str(source.relative_to(ROOT)):file_sha(source)};frames_removed=0
    for e in old['entries']:
        path=ROOT/e['cache']
        if file_sha(path)!=e['cache_sha256']:raise ValueError('original RGBD cache changed')
        inputs[e['cache']]=e['cache_sha256']
        with np.load(path,allow_pickle=False) as a:original={k:np.array(a[k]) for k in ('positions','velocities','confidence','valid','times')}
        with h5py.File(e['path']) as f:
            stages=f['low_dim_obs/expert_trajectory_stage'][:].reshape(-1);opening=f['low_dim_obs/gripper_open'][:].reshape(-1)
        end=precontact_end(stages,opening,len(original['times']));arrays=block_arrays(original,end)
        destination=OUT/'tracks'/(e['split']+'_'+e['task']+'_'+e['raw_episode_id']+'.npz')
        np.savez_compressed(destination,**arrays)
        label_digest=hashlib.sha256(json.dumps([str(v) for v in stages],ensure_ascii=False).encode()+np.ascontiguousarray(opening).tobytes()).hexdigest()
        entries.append(dict(split=e['split'],task=e['task'],raw_episode_id=e['raw_episode_id'],
            parent_raw_episode_id=e['raw_episode_id'],parent_source_group=e['split']+':'+e['task']+':'+e['raw_episode_id'],
            cache=str(destination.relative_to(ROOT)),cache_sha256=file_sha(destination),
            observation_source='causal_RGBD_and_proprioception',target_source='tracked_future_observations',
            dynamics_target_label_schema=None,original_causal_cache=dict(path=e['cache'],sha256=e['cache_sha256']),
            recorded_transition_labels=dict(path=e['path'],stage_and_opening_arrays_sha256=label_digest,offline_mask_only=True),
            precontact_frames=end,original_cache_frames=len(original['times']),
            original_truth_metric_only_array_copied=False,rotation_observed=False,task_outcome_used=False))
        frames_removed+=len(original['times'])-end
    shard=OUT/'block_manifest.json';atomic_json(shard,dict(version='r10_causal_body_tracks_v1',entries=entries,dynamics_target_label_schema=None,TEST_read=False,
        adapter_source=pointer(Path(__file__)),original_track_manifest=pointer(source),input_cache_sha256=inputs,
        observation_orientation='identity gauge with false rotation mask',label_scope='tracked future observations before first recorded closed gripper or lift; offline masking only'))
    _,blocks=read_tracks(ROOT,shard)
    r10=BASE/'context_training_view_full_v6/manifest.json'
    joint=dict(version=SCHEMA,shards=[dict(**pointer(r10),dynamics_target_label_schema=json.loads(r10.read_text())['dynamics_target_label_schema']),
        dict(**pointer(shard),dynamics_target_label_schema=None)],
        original_sources={split:pointer(ROOT/'artifacts/framework/protocol'/name) for split,name in [('train','train_manifest.json'),('validation','validation_manifest.json')]},
        TEST_read=False,model_training_started=False,executor_qualification_required_before_training=True,
        source_code=pointer(ROOT/'our_mani/multitask/joint16_tracks_v31.py'))
    atomic_json(OUT/'manifest.json',joint)
    admitted,episodes=read_joint16_tracks(ROOT,OUT/'manifest.json')
    counts=Counter(row.entry['split'] for row in episodes);parents=Counter(k[0] for k in {row.source_key for row in episodes})
    report=dict(complete=True,manifest=pointer(OUT/'manifest.json'),parents=dict(parents),acquisitions=dict(counts),
        tasks=16,block_cache_episodes=len(blocks),extra_post_contact_frames_removed=frames_removed,
        original_R10_data_unchanged=True,original_block_truth_labels_not_read=True,TEST_read=False,model_training_started=False,
        label_schema_validation='unchanged original read_tracks independently validates each explicit schema shard')
    atomic_json(OUT/'report.json',report);print(json.dumps(report),flush=True)


if __name__=='__main__':main()
