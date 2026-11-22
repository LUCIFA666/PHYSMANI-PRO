"""Joint16 data admission through independently validated label-schema shards.

The original R10 GT backward-pose twist validator is retained. R09 block
histories use measured RGB-D futures, with no native labels or observed SO3
claim. Both pass the existing body-track loader without schema impersonation.
"""
from collections import Counter
from pathlib import Path
import json
import numpy as np
from .evaluation import file_sha
from .track_dataset import read_tracks

SCHEMA='r10_joint16_explicit_label_shards_v31'
BLOCKS=('pick_moving_target_on_the_table','pick_moving_target_on_the_table_high_speed')


def block_arrays(old,end):
    n=int(end)
    if n<5 or n>len(old['times']):raise ValueError('block needs at least five pre-contact frames')
    required=('positions','velocities','confidence','valid','times')
    if any(k not in old for k in required):raise ValueError('original causal block cache is incomplete')
    return dict(times=np.array(old['times'][:n]),body_ids=np.asarray(['target']),
        positions=np.asarray(old['positions'][:n])[:,None,:],linear_velocities=np.asarray(old['velocities'][:n])[:,None,:],
        rotations=np.broadcast_to(np.eye(3),(n,1,3,3)).copy(),angular_velocities=np.zeros((n,1,3)),
        confidence=np.asarray(old['confidence'][:n])[:,None],valid=np.asarray(old['valid'][:n],bool)[:,None],
        modes=np.zeros((n,1),np.int64),rotation_observable=np.zeros((n,1),bool))


def precontact_end(stages,opening,cache_frames):
    stages=[v.decode() if isinstance(v,bytes) else str(v) for v in stages]
    opening=np.asarray(opening).reshape(-1)
    if len(stages)!=len(opening) or not np.isfinite(opening).all():raise ValueError('invalid original recorded transition labels')
    boundaries=[i for i,(stage,value) in enumerate(zip(stages,opening)) if stage=='lift' or value<=.5]
    return min([int(cache_frames)]+boundaries)


def read_joint16_tracks(root,manifest_path):
    root=Path(root).resolve();p=Path(manifest_path)
    if not p.is_absolute():p=root/p
    manifest=json.loads(p.read_text())
    if manifest.get('version')!=SCHEMA or manifest.get('TEST_read') is not False:raise ValueError('explicit joint16 source declaration required')
    episodes=[];seen=set();schemas={}
    for shard in manifest['shards']:
        path=(root/shard['path']).resolve()
        if not path.is_relative_to(root) or file_sha(path)!=shard['sha256']:raise ValueError('joint data shard changed')
        part,rows=read_tracks(root,path)
        if part.get('dynamics_target_label_schema')!=shard['dynamics_target_label_schema']:raise ValueError('joint shard label schema changed')
        for row in rows:
            key=(row.entry['split'],row.entry['task'],row.entry['raw_episode_id'])
            if key in seen:raise ValueError('duplicate joint source acquisition')
            seen.add(key)
        episodes.extend(rows);schemas[shard['path']]=shard['dynamics_target_label_schema']
    parents={row.source_key for row in episodes};counts=Counter((split,task) for split,task,_ in parents)
    for split,source_name in (('train','train_manifest.json'),('validation','validation_manifest.json')):
        source=root/'artifacts/framework/protocol'/source_name;pointer=manifest['original_sources'][split]
        if pointer!={'path':str(source.relative_to(root)),'sha256':file_sha(source)}:raise ValueError('joint original roster changed')
        original=json.loads(source.read_text());expected={(split,row['task'],row['raw_episode_id']) for row in original['episodes']}
        actual={key for key in parents if key[0]==split}
        if expected!=actual:raise ValueError('joint data must retain every original parent exactly, including failures')
        if len({task for _,task,_ in actual})!=16:raise ValueError('joint data must contain all 16 tasks')
    return dict(manifest,admitted_shard_schemas=schemas,parent_counts={':'.join(k):v for k,v in sorted(counts.items())}),episodes
