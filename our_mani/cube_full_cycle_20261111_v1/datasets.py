"""New whole-parent compilers; archived official heldout payloads stay unopened."""
from collections import Counter
from copy import deepcopy
from pathlib import Path
import numpy as np
from .common import ROOT,OUT,sha,read,rows,write,append,pointer,parents,stopped,verify

FEATURES=('future_tokens','robot','geometry','context_c','task_e')
TARGETS=('target_position','target_rotation','target_dt','target_time_valid')

def capture(p,c):
    if c['role'] not in ('fit','selection'): raise ValueError('No validation/test payload access during fitting')
    folder=OUT/'collection_v1'/c['id']/'episode'
    r=read(folder/'result.json');exit=read(folder/'worker_exit.json')
    if not r['complete'] or exit['native_closed']is not True or exit['error']is not None:
        raise ValueError('Complete, closed actual capture required')
    return folder

def assert_collection_closed():
    receipt=read(OUT/'collection_v1/collection_receipt.json')
    if receipt.get('complete')is not True or receipt.get('sources')!=200 or receipt.get('all_case_owners_actual_waited')is not True:
        raise ValueError('Complete full200 collection with actual Native-owner waits required')


def save_role(out,arrays,report):
    out.mkdir(parents=True,exist_ok=False)
    with (out/'dataset.npz').open('xb')as f: np.savez_compressed(f,**arrays)
    report['dataset_sha256']=sha(out/'dataset.npz');write(out/'report.json',report)
    return report

def compile_motion(p):
    from ..cube_motion_residual_20261104_v1.data import compile_episode
    assert_collection_closed()
    destination=OUT/'motion_data';destination.mkdir(exist_ok=False)
    for role in ('fit','selection'):
        data=[];reports=[];inputs={}
        role_parents=parents(p,role)
        for c in [v for v in p['cases']if v['role']==role]:
            stopped(p);folder=capture(p,c)
            public_path=folder/'public_observer/public_outputs.jsonl';label_path=folder/'private_TRAIN_labels.jsonl'
            public=rows(public_path);labels=rows(label_path)
            if len(public)!=len(labels): raise ValueError('Complete same-clock public/private ledger required')
            projected=[];private=[];segment=0;last_held=False
            for r,l in zip(public,labels):
                now=float(r['result']['timestamp']);o=r['result'].get('object_observation')
                if abs(now-l['time'])>1e-8 or l['episode_id']!=c['episode_id']: raise ValueError('No cross-clock/reset joins')
                available=bool(o is not None and o.get('visible',True) and r['result']['mode']=='FREE')
                projected.append(dict(timestamp=now,available=available,position_world=o['pose']['position']if available else None,confidence=float(o['confidence'])if available else None))
                if l['held']!=last_held: segment+=1
                last_held=l['held']
                private.append(dict(timestamp=l['time'],target_position_world=l['object_position'],free_motion=not l['held'],segment_id=segment))
            windows,report=compile_episode(projected,private,parent_id=c['parent_id'])
            # Zero-observation parents remain in the full role denominator.
            idx=role_parents.index(c['parent_id'])
            data.extend([{**w['inputs'],**w['targets'],'parent_index':np.int64(idx),'sample_weight':np.float32(1/len(windows))}for w in windows])
            reports.append(report);inputs.update({str(path):sha(path)for path in(public_path,label_path)})
        if not data:raise ValueError('No real motion windows in entire '+role+' role')
        arrays={k:np.stack([r[k]for r in data])for k in data[0]}
        # NaN slots are never labels; retain the explicit mask and use finite padding.
        arrays['target_positions']=np.where(arrays['target_valid'][...,None],arrays['target_positions'],0.).astype(np.float32)
        save_role(destination/role,arrays,dict(role=role,parents=role_parents,windows=len(data),episodes=reports,
            assigned_parents=len(role_parents),zero_window_parents=[r['parent_id']for r in reports if not r['windows']],source_inputs_sha256=inputs,TEST_read=False,heldout_read=False,private_values_are_inputs=False))
    return destination

def replay_public(p,checkpoint):
    """Freeze every numerical feature before any candidate private-label join."""
    from ..contracts import ObjectState,Pose
    from ..cube_motion_residual_20261104_v1.runtime import CubeResidualMotionDynamics
    from ..cube_candidate_matched_20261104_v2.data import public_features
    from ..cube_corrected_live_20261104_v1.candidate_runtime import _project_row
    forecast_sha=sha(checkpoint);dyn=CubeResidualMotionDynamics(checkpoint,expected_sha256=forecast_sha)
    destination=OUT/'public_replay';destination.mkdir(exist_ok=False)
    pointers={};reports=[]
    for c in [v for v in p['cases']if v['role']in('fit','selection')]:
        stopped(p);folder=capture(p,c);source=folder/'public_observer/public_outputs.jsonl'
        public=rows(source);provenance=rows(folder/'public_observer/frame_provenance.jsonl');out=destination/c['id'];out.mkdir()
        if len(provenance)!=len(public):raise ValueError('Every actual public RGBD frame needs its capture hash ledger')
        dyn.reset();replayed=[];features=[];indices=[];rejected=Counter()
        for i,row in enumerate(public):
            stopped(p);r=deepcopy(row['result']);o=r.get('object_observation');f=None
            if o is None or r['mode']!='FREE': dyn.reset()
            else:
                obj=ObjectState(o['object_id'],o['timestamp'],Pose(np.asarray(o['pose']['position']),np.asarray(o['pose']['rotation'])),np.asarray(o['points_world']),np.asarray(o['linear_velocity']),np.asarray(o['angular_velocity']),float(o['confidence']),bool(o['visible']))
                dyn.observe([obj],modes={'target':'external'},orientation_observability={'target':False})
                if len(dyn.history.get('target',()))>=5: f=dyn.predict([obj],.8,.05)
            r.update(forecast=f,available=f is not None,free_forecast_available=f is not None,frozen_checkpoint_sha256=forecast_sha)
            from ..local_mpc_20261102.records import plain
            replaced=plain(dict(index=row['index'],result=r,public_robot=row['public_robot'],close_requested=row['close_requested'],rgbd=row['rgbd']))
            replaced=_project_row(replaced,expected_forecast_sha256=forecast_sha)
            # Target labels beyond .8 need the explicit unobserved identity basis.
            if replaced['result']['forecast']is not None:
                replaced['result']['forecast']['metadata']['full_orientation_observable']={'target':False}
            replayed.append(replaced);append(out/'public_rows.jsonl',replaced)
            try: value,_=public_features(replaced,replayed,expected_forecast_sha256=forecast_sha)
            except (ValueError,KeyError)as error: rejected[str(error)]+=1;continue
            evidence=provenance[i]
            if (evidence['index']!=i or abs(float(evidence['timestamp'])-r['timestamp'])>1e-8 or evidence.get('actual_RGBD_captured')is not True or len(evidence['rgbd_sha256'])!=64):
                raise ValueError('Actual same-clock captured RGBD/hash evidence required')
            if evidence['raw_rgbd_saved']:
                image=folder/'public_observer'/replaced['rgbd']
                with np.load(image,allow_pickle=False)as rgbd:
                    if abs(float(rgbd['timestamp'])-r['timestamp'])>1e-8:raise ValueError('Saved public RGBD timestamp mismatch')
                    from .frame_recording import capture_evidence
                    observed={'timestamp':float(rgbd['timestamp'])}
                    for group in ('images','depths'):
                        observed['observation/'+group]={name.removeprefix(group+'_'):rgbd[name]for name in rgbd.files if name.startswith(group+'_')}
                    if capture_evidence(observed,i,evidence['camera_calibration_sha256'])['rgbd_sha256']!=evidence['rgbd_sha256']:
                        raise ValueError('Saved actual RGBD bytes differ from capture hash')
            features.append(value);indices.append(i)
        # Empty parents retain complete frame/rejection ledgers and finite zero-length feature arrays.
        shapes={'future_tokens':(17,16),'robot':(25,),'geometry':(4,),'context_c':(16,),'task_e':(16,)}
        arrays={k:np.stack([v[k]for v in features])if features else np.empty((0,*shapes[k]),np.float32)for k in FEATURES};arrays['public_index']=np.asarray(indices,dtype=np.int64)
        with(out/'features.npz').open('xb')as f: np.savez_compressed(f,**arrays)
        report=dict(parent_id=c['parent_id'],role=c['role'],rows=len(public),feature_rows=len(features),rejected=dict(rejected),
            public_source=pointer(source),frame_provenance=pointer(folder/'public_observer/frame_provenance.jsonl'),raw_RGBD_stride=20,private_labels_opened=False,forecast_checkpoint_sha256=forecast_sha)
        write(out/'report.json',report);reports.append(report)
        for name in ('public_rows.jsonl','features.npz','report.json'): pointers[str(out/name)]=sha(out/name)
    write(destination/'public_features_frozen.json',dict(complete=True,forecast_checkpoint=pointer(checkpoint),
        private_candidate_labels_read=False,files=pointers,episodes=reports,heldout_read=False,TEST_read=False))
    return destination

def compile_candidate(p,checkpoint):
    from ..cube_candidate_teacher_20261104_v5.labels import actual_local_grasp_contacts
    from ..cube_candidate_matched_20261104_v2.data import public_reference_target
    from ..cube_candidate_matched_20261104_v2 import FRAME
    destination=OUT/'candidate_data';destination.mkdir(exist_ok=False)
    frozen=read(OUT/'public_replay/public_features_frozen.json');verify(frozen['files'])
    if frozen['forecast_checkpoint']['sha256']!=sha(checkpoint):raise ValueError('Frozen public forecast differs')
    for role in ('fit','selection'):
        stopped(p);data=[];reports=[];inputs=dict(frozen['files']);inputs[str(OUT/'public_replay/public_features_frozen.json')]=sha(OUT/'public_replay/public_features_frozen.json');role_parents=parents(p,role);coverage=Counter()
        for c in [v for v in p['cases']if v['role']==role]:
            folder=capture(p,c);feature_folder=OUT/'public_replay'/c['id']
            public=rows(feature_folder/'public_rows.jsonl')
            with np.load(feature_folder/'features.npz',allow_pickle=False)as file: feat={k:np.array(file[k],copy=True)for k in file.files}
            # Only now open private contact supervision, after all public forward inputs froze.
            label_path=folder/'private_TRAIN_labels.jsonl';labels=rows(label_path)
            if len(labels)!=len(public): raise ValueError('Actual aligned full ledgers required')
            contacts=[x for x in actual_local_grasp_contacts(labels[0],labels[1:],episode_id=c['episode_id'])if x['original_Native_task_success_observed']]
            first_close=next((r['time']for r in labels[1:]if r['requested_opening']<.5),float('inf'))
            values=[];ledger=[]
            for j,index in enumerate(feat['public_index']):
                row=public[int(index)];now=float(row['result']['timestamp'])
                if abs(now-labels[int(index)]['time'])>1e-8:raise ValueError('No nearest-frame/cross-clock joins')
                if now>=first_close-1e-8:continue
                for contact in contacts[:1]:
                    if contact['time']<=now+1e-8:continue
                    target=public_reference_target(row['result']['forecast'],contact,anchor_time=now)
                    values.append({**{k:feat[k][j]for k in FEATURES},**{k:target[k]for k in TARGETS}})
                    ledger.append(dict(index=int(index),anchor_time=now,contact_time=contact['time'],time_valid=bool(target['target_time_valid'])))
            count=len(values);idx=role_parents.index(c['parent_id'])
            data.extend([{**v,'parent_index':np.int64(idx),'sample_weight':np.float32(1/count)}for v in values])
            if count:coverage['high_speed'if c['task'].endswith('high_speed')else'ordinary']+=1
            reports.append(dict(parent_id=c['parent_id'],windows=count,Native_success_contacts=len(contacts),ledger=ledger))
            inputs[str(label_path)]=sha(label_path)
        needed=p['minimum_supervised_parents_per_speed'][role]
        report=dict(role=role,parents=role_parents,assigned_parent_denominator=len(role_parents),episodes=reports,
            supervised_parents=[r['parent_id']for r in reports if r['windows']],zero_positive_parents=[r['parent_id']for r in reports if not r['windows']],
            supervised_parents_per_speed=dict(coverage),minimum_per_speed=needed,target_frame=FRAME,source_inputs_sha256=inputs,
            forecast_checkpoint_sha256=sha(checkpoint),failed_candidates_are_negatives=False,TEST_read=False,heldout_read=False)
        if any(coverage[k]<needed for k in ('ordinary','high_speed')):
            write(destination/f'{role}_insufficient_coverage.json',report)
            raise ValueError('Insufficient real Native-success candidate supervision in '+role+' '+str(dict(coverage)))
        arrays={k:np.stack([r[k]for r in data])for k in data[0]}
        save_role(destination/role,arrays,report)
    verify(frozen['files']);return destination
