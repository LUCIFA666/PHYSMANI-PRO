"""Fixed two-case saved TRAIN RGBD replay; GT only after all predictions.

Uses filtered V6 TRAIN visual CAD and historical same-process instance IDs.
No native simulator, teacher, WM, GPU, VAL/TEST or parameter search.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from our_mani.contracts import Pose
from our_mani.perception import CameraCalibration
from our_mani.multitask.perception import (BodyTemplate,GeometryComponent,PriorProvenance,
    RegistrationConfig,MultiBodySE3Tracker)
from our_mani.multitask.bin_swept_geometry import partial_collision_axis_ready

BASE=ROOT/'artifacts/framework/multitask_r10'
VERSION='r10_TRAIN_saved_bin_directed_axis_replay_v1'
TASKS=('put_rubbish_in_moving_bin','put_rubbish_in_moving_bin_high_speed')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def require(value,message):
    if not value:raise ValueError(message)
def write(path,value):
    path=Path(path).resolve();require(path.is_relative_to(BASE.resolve()) and not path.exists(),'new TRAIN output only')
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def prepare(path):
    input_paths=[Path(__file__),ROOT/'our_mani/multitask/perception.py',ROOT/'our_mani/multitask/axis_evidence.py',
        ROOT/'our_mani/multitask/depth_pose_consistency.py',ROOT/'our_mani/multitask/bin_swept_geometry.py',
        ROOT/'our_mani/perception.py',ROOT/'our_mani/contracts.py']
    prototype_path=BASE/'bin_swept_geometry_prototype_v1/report.json';prototype=read(prototype_path)
    require(prototype['complete'] is True and prototype['TRAIN_only'] is True
        and prototype['runtime_installed'] is False,'TRAIN CAD diagnostic identity changed')
    queue_path=BASE/'orientation_pilot_v2/queue_report.json';queue=read(queue_path)
    require(queue['complete'] is True,'historical TRAIN pilot was incomplete')
    input_paths.extend((prototype_path,queue_path));cases=[]
    for task in TASKS:
        report_path=BASE/'orientation_pilot_v2'/task/'report.json';report=read(report_path)
        receipt=next(row for row in queue['results'] if row['task']==task)
        require(receipt['completed'] is True and sha(report_path)==receipt['report_sha256']
            and report['complete'] is True and report['TRAIN_only'] is True and report['TEST_used'] is False
            and report['teacher_used'] is False and report['actual_frames']==16,'historical TRAIN case differs')
        sensor_path=ROOT/report['causal_sensor_observations']['path']
        require(report['causal_sensor_observations']['native_GT_or_task_outcome_included'] is False
            and sha(sensor_path)==report['causal_sensor_observations']['sha256'],'saved causal sensor stream differs')
        old_geometry=report_path.parent/'TRAIN_geometry.json';old_asset=read(old_geometry)
        new_geometry=ROOT/prototype['tasks'][task]['source_asset'];new_asset=read(new_geometry)
        require(sha(new_geometry)==prototype['tasks'][task]['source_asset_sha256']
            and all(asset['task']==task and asset['split']=='train' and asset['shared_with_baseline'] is True
                and asset['world_pose_or_motion_schedule_in_asset'] is False for asset in (old_asset,new_asset)),
            'same-task immutable shared TRAIN CAD required')
        old_components={row['name']:row for row in old_asset['bodies']['bin']['components']}
        visual=new_asset['bodies']['bin']['visual_component_names']
        require(visual==['bin_visual'] and all(name in old_components for name in visual),
            'native-audited filtered bin visual component must exist in old cache inventory')
        # This is the only binding exposed: historical same-process sensor IDs,
        # restricted by pre-existing frozen TRAIN renderable component names.
        binding={name:old_components[name]['TRAIN_instance_ids'] for name in visual}
        cases.append(dict(task=task,report=str(report_path.relative_to(ROOT)),sensor=str(sensor_path.relative_to(ROOT)),
            geometry=str(new_geometry.relative_to(ROOT)),original_sensor_geometry=str(old_geometry.relative_to(ROOT)),
            historical_sensor_instance_bindings=binding,
            public_cameras=report['initial_conditions']['public_initial_conditions']['cameras'],frames=16,
            parent_source_group='train:'+task+':'+report['source']['raw_episode_id']))
        input_paths.extend((report_path,sensor_path,old_geometry,new_geometry))
    result=dict(version=VERSION,TRAIN_only=True,no_native_or_GPU=True,VAL_TEST_used=False,teacher_used=False,
        source_kind='previous TRAIN native-generated hold observations; not original HDF5 replay',
        cases=cases,expected_cases=2,configuration=asdict(RegistrationConfig()),
        axis_local=[1.,0.,0.],axis_polarity='directed',free_space_score_weight=0.,
        no_parameter_search=True,automatic_annotation_promotion=False,
        GT_access='native GT frame arrays accessed only after both complete prediction streams have been saved',
        inputs_sha256={str(p.resolve()):sha(p) for p in input_paths})
    result['plan_sha256']=digest(result);write(path,result);return result


def verify(path):
    plan=read(path);require(plan['version']==VERSION and plan['plan_sha256']==digest({k:v for k,v in plan.items() if k!='plan_sha256'}),
        'replay plan identity changed')
    for name,expected in plan['inputs_sha256'].items():require(sha(name)==expected,'replay source/input changed: '+name)
    require(plan['configuration']['free_space_weight']==0. and plan['expected_cases']==2
        and tuple(c['task'] for c in plan['cases'])==TASKS,'fixed replay inventory or algorithm changed')
    return plan


def predict(case,configuration):
    """No report/GT object is accepted by this causal prediction function."""
    asset=read(ROOT/case['geometry']);body=asset['bodies']['bin']
    components=tuple(GeometryComponent(row['name'],tuple(case['historical_sensor_instance_bindings'][row['name']]),
        np.asarray(row['points_body'])) for row in body['components'])
    provenance=PriorProvenance('declared_cad','filtered immutable TRAIN CAD:'+asset['asset_sha256'],
        body['frame_definition'],True,'train',(asset['source_group'],),asset['asset_sha256'])
    template=BodyTemplate('bin',components,provenance,orientation_observability='axis',
        symmetry_axis=np.asarray([1.,0.,0.]),axis_polarity='directed')
    tracker=MultiBodySE3Tracker((template,),config=RegistrationConfig(**configuration))
    cameras={name:CameraCalibration(np.asarray(row['intrinsics']),Pose.from_xyzw(row['camera_to_world']))
        for name,row in case['public_cameras'].items()};predictions=[]
    with np.load(ROOT/case['sensor'],allow_pickle=False) as stored:
        require(len(stored['times'])==16 and np.allclose(np.diff(stored['times']),.05),'saved frame clock differs')
        for frame,now in enumerate(stored['times']):
            observation={'timestamp':float(now),'observation/calibrations':cameras,
                'observation/camera_timestamps':{name:float(now) for name in cameras}}
            for channel,key in (('observation/images','rgb'),('observation/depths','depth'),('observation/instance_masks','mask')):
                observation[channel]={name:stored[key+'_'+name][frame] for name in cameras}
            states,metadata=tracker.observe(observation,modes={'bin':'external'})
            state=next((row for row in states if row.object_id=='bin'),None);diagnostic=metadata['bodies']['bin']
            predictions.append(dict(frame=frame,time=float(now),pose_xyzw=state.pose.xyzw().tolist() if state else None,
                visible=bool(state and state.visible),causal_registration=diagnostic,
                proposed_partial_geometry_readiness=bool(diagnostic['registration_valid'] and state and state.visible
                    and partial_collision_axis_ready(diagnostic.get('axis_evidence'),axis_local=[1.,0.,0.])),GT_used=False))
    return predictions


def metric(case,predictions):
    # Only called after every case's causal predictions have been saved.
    report=read(ROOT/case['report']);frames=[]
    for prediction,native in zip(predictions,report['frames']):
        index=native['metric_only']['body_ids'].index('bin');truth=native['metric_only']['native_current_ground_truth']
        row=dict(frame=prediction['frame'],axis_observable=prediction['causal_registration']['axis_observable'],
            proposed_partial_geometry_readiness=prediction['proposed_partial_geometry_readiness'])
        if prediction['pose_xyzw'] is not None:
            pose=Pose.from_xyzw(prediction['pose_xyzw']);axis=pose.rotation[:,0];actual=np.asarray(truth['rotations'][index])[:,0]
            row.update(position_error_m=float(np.linalg.norm(pose.position-np.asarray(truth['positions'][index]))),
                directed_axis_error_degrees=float(np.rad2deg(np.arccos(np.clip(axis@actual,-1.,1.)))),
                undirected_axis_error_degrees=float(np.rad2deg(np.arccos(np.clip(abs(axis@actual),-1.,1.)))))
        frames.append(row)
    values=[r['directed_axis_error_degrees'] for r in frames if 'directed_axis_error_degrees' in r]
    eligible=[r['directed_axis_error_degrees'] for r in frames if r['proposed_partial_geometry_readiness'] and 'directed_axis_error_degrees' in r]
    stats=lambda x:dict(n=len(x),median=float(np.median(x)) if x else None,
        p90=float(np.quantile(x,.9)) if x else None,max=float(max(x)) if x else None)
    return dict(frames=frames,axis_observable_rows=sum(r['axis_observable'] for r in frames),
        proposed_partial_geometry_ready_rows=sum(r['proposed_partial_geometry_readiness'] for r in frames),
        directed_axis_error_degrees=stats(values),ready_directed_axis_error_degrees=stats(eligible),
        polarity_error_over90deg_rows=sum(r.get('directed_axis_error_degrees',0)>90 for r in frames))


def run(plan_path,output):
    plan=verify(plan_path);output=Path(output).resolve()
    require(output.is_relative_to(BASE.resolve()) and not output.exists(),'fresh TRAIN replay directory only');output.mkdir(parents=True)
    predictions={}
    for case in plan['cases']:
        predictions[case['task']]=predict(case,plan['configuration'])
        write(output/(case['task']+'__predictions.json'),predictions[case['task']])
        print('predicted '+case['task'],flush=True)
    verify(plan_path)
    metrics={case['task']:metric(case,predictions[case['task']]) for case in plan['cases']}
    result=dict(version=VERSION,complete=True,TRAIN_only=True,no_native_or_GPU=True,VAL_TEST_used=False,
        plan=str(Path(plan_path).resolve().relative_to(ROOT)),plan_sha256=plan['plan_sha256'],
        all_case_predictions_saved_before_GT_metric_access=True,metrics=metrics,
        outputs_sha256={str((output/(task+'__predictions.json')).relative_to(ROOT)):sha(output/(task+'__predictions.json')) for task in TASKS},
        runtime_installed=False,annotation_promoted=False,thresholds_tuned=False,
        limitations=['Two correlated 16-frame TRAIN hold streams, not across-source or contact validation.',
            'Near-optimal directed-axis agreement is not an accuracy certificate; posterior native GT is metric-only.',
            'No full orientation claim, model fitting, native control, collision-engine contact check or policy SR evaluation.'])
    write(output/'report.json',result);return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--prepare',action='store_true');group.add_argument('--run',action='store_true')
    parser.add_argument('--plan',required=True);parser.add_argument('--out');args=parser.parse_args()
    value=prepare(args.plan) if args.prepare else run(args.plan,args.out)
    print(json.dumps(dict(complete=value.get('complete'),plan_sha256=value['plan_sha256'])))
