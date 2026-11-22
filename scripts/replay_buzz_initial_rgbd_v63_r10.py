"""Replay already captured public RGBD; this is not native executor admission."""
from pathlib import Path
import json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from our_mani.contracts import Pose
from our_mani.perception import CameraCalibration
from our_mani.multitask.evaluation import atomic_json,file_sha
from our_mani.multitask.perception import MultiBodySE3Tracker
from our_mani.multitask.training_geometry import body_templates_from_asset
from our_mani.multitask.two_center_pose import load_two_center_pose_prior
from our_mani.multitask.buzz_branch_v63 import PriorConditionedRegistration
BASE=ROOT/'artifacts/framework/multitask_r10';OUT=BASE/'buzz_initial_rgbd_replay_v63/report.json'

def pointer(p):return dict(path=str(p.relative_to(ROOT)),sha256=file_sha(p))

def main():
    if OUT.exists():raise RuntimeError('preserve existing replay')
    audit=BASE/'buzz_initial_branch_source_audit_v61/report.json';assert json.loads(audit.read_text())['all_initials_passed']
    tasks={};inputs={str(Path(__file__).relative_to(ROOT)):file_sha(__file__),str(audit.relative_to(ROOT)):file_sha(audit)}
    for task,raw in [('beat_the_rotating_buzz','episode10'),('beat_the_rotating_buzz_high_speed','episode102')]:
        path=BASE/'buzz_visual_capture_candidates'/f'native_v39_{task}_{raw}_cv_geometry/manifest.json'
        manifest=json.loads(path.read_text());asset=json.loads((ROOT/manifest['canonical_CAD']['path']).read_text())
        bindings={role:{part['name']:tuple(manifest['component_instance_ids'][part['name']]) for part in body['components']}
            for role,body in asset['bodies'].items()}
        templates=body_templates_from_asset(asset,bindings,orientation_observability={'frame':'full','wand':'full'})
        tracker=MultiBodySE3Tracker(templates)
        model=BASE/'buzz_two_center_TRAIN_models_and_teacher_scope_v3/models'/task/'model.json'
        prior=load_two_center_pose_prior(ROOT,pointer(model),canonical_asset_pointer=manifest['canonical_CAD'])
        adapter=PriorConditionedRegistration(tracker,prior);tracker._registration=adapter.register
        inputs[str(path.relative_to(ROOT))]=file_sha(path);rows=[]
        for item in manifest['frames']:
            frame_path=ROOT/item['path'];assert file_sha(frame_path)==item['sha256'];inputs[item['path']]=item['sha256']
            frame=json.loads(frame_path.read_text());pt=frame['arrays'];assert file_sha(ROOT/pt['path'])==pt['sha256'];inputs[pt['path']]=pt['sha256']
            with np.load(ROOT/pt['path'],allow_pickle=False) as z:
                now=float(z['observation_timestamp']);calibrations={}
                for camera in ('front','8','16','36'):
                    t=z['camera_to_world_'+camera]
                    calibrations[camera]=CameraCalibration(z['K_'+camera],Pose(t[:3,3],t[:3,:3]),
                        depth_scale=float(z['depth_scale_'+camera]),timestamp_offset=float(z['timestamp_offset_'+camera]))
                observation=dict(timestamp=now,**{'observation/depths':{c:z['depth_'+c] for c in calibrations},
                    'observation/instance_masks':{c:z['mask_'+c] for c in calibrations},'observation/calibrations':calibrations,
                    'observation/camera_timestamps':{c:float(z['timestamp_'+c]) for c in calibrations}})
                adapter.set_public_context(timestamp=now,gripper_open=bool(z['robot_gripper_open']),contact=bool(z['robot_public_contact']))
                states,metadata=tracker.observe(observation,modes={'frame':'external','wand':'external'})
                rows.append(dict(time=now,roles={role:dict(valid=d['registration_valid'],orientation_observable=d['orientation_observable'])
                    for role,d in metadata['bodies'].items()}))
        dense=rows[:5];tasks[task]=dict(rows=rows,events=adapter.events,
            first_five_dense_observations_qualified=all(all(d['orientation_observable'] for d in r['roles'].values()) for r in dense),
            sparse_tail_expected_stale_rejection=True,native_executor_qualified=False)
    report=dict(schema='r10_buzz_candidate_RGBD_branch_integration_replay_v63',tasks=tasks,inputs_sha256=inputs,
        first_five_dense_observations_qualified=all(t['first_five_dense_observations_qualified'] for t in tasks.values()),
        old_orientation_none_models_not_modified=True,formal_permission=False,TEST_used=False,
        source=pointer(ROOT/'our_mani/multitask/buzz_branch_v63.py'))
    atomic_json(OUT,report);print(json.dumps(dict(report=str(OUT.relative_to(ROOT)),first_five_dense_observations_qualified=report['first_five_dense_observations_qualified'],
        tasks={task:dict(frames=len(t['rows']),dense_pass=t['first_five_dense_observations_qualified']) for task,t in tasks.items()})))

if __name__=='__main__':main()
