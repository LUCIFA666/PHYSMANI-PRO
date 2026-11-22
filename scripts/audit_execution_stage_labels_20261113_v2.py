#!/usr/bin/env python3
"""Map existing TRAIN stage labels to the frozen public feature row identities."""
import hashlib,json
from collections import Counter
from pathlib import Path
import h5py
import numpy as np

ROOT=Path(__file__).resolve().parents[1]

def main():
    plan=json.loads((ROOT/'config/framework/multitask_executor_cycle_20261112_v1/plan.json').read_text())
    freeze=json.loads((ROOT/'artifacts/framework/multitask_executor_cycle_20261112_v1/head_fit/public_feature_freeze.json').read_text())
    labels={};parents=[];by_parent={};missing=[]
    for parent in plan['cached_TRAIN']:
        assert parent['source_split']=='train'and parent['source']['split']=='train'
        path=Path(parent['source']['source_path']);before=path.stat();by_parent[parent['source_parent_group']]=parent
        with h5py.File(path,'r')as h:
            if 'expert_trajectory_stage'not in h['low_dim_obs']:
                missing.append(str(path));continue
            values=h['low_dim_obs/expert_trajectory_stage'][...].reshape(-1)
            stages=[x.decode('utf8')if isinstance(x,bytes)else str(x)for x in values]
            labels[parent['source_parent_group']]=stages
            parents.append(dict(parent=parent['source_parent_group'],task=parent['task'],role=parent['role'],
                source_path=str(path),frames=len(stages),stage_distribution=dict(Counter(stages)),
                physical_clock_verified=parent['source']['physical_clock_verified']))
        after=path.stat()
        assert before.st_size==after.st_size and before.st_mtime_ns==after.st_mtime_ns
    rows=[];role_index=Counter()
    for identity in freeze['identities']:
        parent=identity['source_parent_group'];frame=identity['current_frame_index']
        # The identity's legacy role describes the original pretraining cache.
        # New fit/selection assignment is pinned by parent in the prospective plan.
        role=by_parent[parent]['role'];assert role in ('fit','selection')
        rows.append(dict(task=identity['task'],parent=parent,role=role,source_cache_role=identity['role'],
            current_source_frame_index=frame,role_feature_row=role_index[role],stage=labels[parent][frame]))
        role_index[role]+=1
    features=Path(freeze['features']['path']);actual=hashlib.sha256(features.read_bytes()).hexdigest()
    assert actual==freeze['features']['sha256']
    with np.load(features)as data:
        shapes={k:list(data[k].shape)for k in data.files}
        for role,total in role_index.items():assert len(data[role+'_hidden'])==total
    report=dict(complete=True,TRAIN_only=True,parent_count=len(parents),missing_stage_fields=missing,
        window_label_count=len(rows),role_counts=dict(role_index),parents=parents,window_stage_labels=rows,
        frozen_public_feature_sha256=actual,feature_shapes=shapes,raw_RGBD_pixels_decoded=0,
        VAL_TEST_payloads_read=0,optimizer_steps=0,physical_dt_invented=False,
        role_assignment_source='prospective_plan_parent_role_not_legacy_cache_identity_role',
        target_time_offset_labels_not_yet_derived=True,
        source_stage_labels_are_expert_supervision_not_network_predictions=True)
    out=ROOT/'artifacts/framework/execution_protocol_20261113_v2';out.mkdir(exist_ok=True)
    (out/'TRAIN_stage_label_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    (out/'compatibility_decision.json').write_text(json.dumps(dict(
        user_choice='扩展协议，并保留旧 8 维兼容',legacy_pose8_values_preserved=True,
        metadata_fields_separate=True),ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:report[k]for k in('parent_count','window_label_count','role_counts','feature_shapes',
        'missing_stage_fields','physical_dt_invented','VAL_TEST_payloads_read')},ensure_ascii=False))

if __name__=='__main__':main()
