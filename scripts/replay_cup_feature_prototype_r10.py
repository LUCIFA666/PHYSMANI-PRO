"""Freeze and execute one current-RGBD cup-feature replay, TRAIN caches only."""
import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from our_mani.contracts import Pose
from our_mani.perception import CameraCalibration
from our_mani.multitask.perception import decode_instance_mask
from scripts.experimental_cup_features_r10 import (VERSION,FeatureConfig,
    features_from_frozen_cad,estimate_cup_feature_pose)

BASE=ROOT/'artifacts/framework/multitask_r10'
ROLES=('cup0','cup1','cup2')
CASES=tuple([('orientation_pilot_v2',task) for task in ('place_cups_on_rotating_frame',
    'place_cups_on_rotating_frame_high_speed','remove_cups_from_rotating_frame',
    'remove_cups_from_rotating_frame_high_speed')]+[('rigid_orientation_pilot_v3',task) for task in (
    'place_cups_on_rotating_frame','place_cups_on_rotating_frame_high_speed','remove_cups_from_rotating_frame')])


def utc():return datetime.now(timezone.utc).isoformat()
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def canonical(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def require(condition,message):
    if not condition:raise ValueError(message)
def read(path):return json.loads(Path(path).read_text())
def write(path,value):
    path=Path(path);temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');temporary.replace(path)
def pointer(path):
    path=Path(path).resolve();require(path.is_relative_to(ROOT),'all artifacts must stay inside our-mani')
    return dict(path=str(path.relative_to(ROOT)),sha256=sha(path))


def prepare(plan_path):
    plan_path=Path(plan_path).resolve();require(plan_path.is_relative_to(BASE) and not plan_path.exists(),'fresh R10 plan required')
    inputs={};cases=[]
    for directory,task in CASES:
        folder=BASE/directory/task;report_path=folder/'report.json';report=read(report_path)
        geometry=folder/'TRAIN_geometry.json';sensor=ROOT/report['causal_sensor_observations']['path'];asset=read(geometry)
        require(report['complete'] is True and report['TRAIN_only'] is True and report['TEST_used'] is False
            and report['teacher_used'] is False and report['actual_frames']==16,'complete fixed16 TRAIN sensor pilot required')
        require(report['causal_sensor_observations']['native_GT_or_task_outcome_included'] is False
            and sha(sensor)==report['causal_sensor_observations']['sha256'],'raw sensor cache boundary/hash differs')
        require(asset['split']=='train' and asset['task']==task and asset['shared_with_baseline'] is True
            and asset['world_pose_or_motion_schedule_in_asset'] is False
            and asset['asset_sha256']==canonical({k:v for k,v in asset.items() if k!='asset_sha256'}),'immutable local TRAIN CAD required')
        group='train:'+task+':'+report['source']['raw_episode_id']
        require(asset['source_group']==group,'CAD/sensor TRAIN parent differs')
        cases.append(dict(case_id=directory+'__'+task,task=task,parent_source_group=group,
            geometry=pointer(geometry),sensor=pointer(sensor),metric_report=pointer(report_path),
            public_cameras=report['initial_conditions']['public_initial_conditions']['cameras'],
            body_ids=list(ROLES),frames=16))
        for path in (geometry,sensor,report_path):inputs[str(path.relative_to(ROOT))]=sha(path)
    for name in ('scripts/experimental_cup_features_r10.py','scripts/replay_cup_feature_prototype_r10.py',
        'tests/test_cup_feature_prototype_r10.py','our_mani/contracts.py','our_mani/perception.py',
        'our_mani/multitask/perception.py','our_mani/multitask/depth_pose_consistency.py'):
        inputs[name]=sha(ROOT/name)
    parents=sorted({case['parent_source_group'] for case in cases})
    require(len(parents)==4,'fixed archives must represent exactly four distinct TRAIN parents')
    result=dict(version=VERSION,created_utc=utc(),configuration=asdict(FeatureConfig()),cases=cases,
        canonical_CAD=cases[0]['geometry'],inputs_sha256=inputs,
        expected_cases=7,expected_body_sequences=21,expected_body_frames=336,
        independent_TRAIN_parent_count=4,independent_TRAIN_parents=parents,
        case_selection='all seven previously saved place/remove TRAIN RGBD caches, every cup, every one of16 frames',
        observations='current four calibrated RGBD views and explicitly shared current component masks plus immutable TRAIN local CAD',
        native_GT_access='all336 sensor/CAD-only predictions finish and are hashed before metric reports are opened by the runner',
        static_world_axis_or_known_native_pose_used=False,history_or_future_observations_used=False,
        generic_ICP_used=False,parameter_search=False,TRAIN_only=True,VAL_used=False,TEST_used=False,
        native_simulator_started=False,GPU_used=False,production_policy_authorized=False,
        source_kind='previous TRAIN native-generated hold diagnostics; not original source HDF5 motion replay',
        limitations=['Repeated caches and adjacent frames are not independent data.',
            'The feature annotation uses a simulation-assisted TRAIN CAD prior shared with the baseline.',
            'Feature readiness is an engineering diagnostic and never authorizes a full-orientation policy.',
            'Timing is measured while the separate fixed native TRAIN collection is running.'])
    result['plan_sha256']=canonical(result);write(plan_path,result);return result


def verify(plan_path):
    plan=read(plan_path);require(plan['version']==VERSION and plan['plan_sha256']==canonical({k:v for k,v in plan.items() if k!='plan_sha256'}),'frozen plan identity differs')
    for path,expected in plan['inputs_sha256'].items():require(sha(ROOT/path)==expected,'frozen replay input changed: '+path)
    require(plan['configuration']==asdict(FeatureConfig()),'fixed primitive feature parameters changed')
    return plan


def predict_case(case,canonical_asset,output):
    """This function receives no metric report object or GT fields."""
    started=time.perf_counter();asset=read(ROOT/case['geometry']['path'])
    models={role:features_from_frozen_cad(canonical_asset,asset,role) for role in ROLES}
    components={role:next(component for component in asset['bodies'][role]['components']
        if component['name']=='mug_visual'+role[-1]) for role in ROLES}
    calibrations={name:CameraCalibration(np.asarray(row['intrinsics']),Pose.from_xyzw(row['camera_to_world']))
        for name,row in case['public_cameras'].items()}
    with np.load(ROOT/case['sensor']['path'],allow_pickle=False) as stored:
        require(len(stored['times'])==16,'fixed sensor frame count differs')
        # Decompress each image array only once; caches have no GT namespace.
        arrays={name:stored[name] for name in stored.files if name.startswith(('depth_','mask_'))}
        times=stored['times'].copy()
    filename=output/(case['case_id']+'.predictions.jsonl');count=0
    with filename.open('w') as stream:
        for frame,now in enumerate(times):
            views={camera:(calibration,arrays['depth_'+camera][frame]) for camera,calibration in calibrations.items()}
            decoded={camera:decode_instance_mask(arrays['mask_'+camera][frame]) for camera in calibrations}
            for role in ROLES:
                tick=time.perf_counter();component=components[role]
                points=np.concatenate([calibration.backproject(arrays['depth_'+camera][frame],
                    np.isin(decoded[camera],component['TRAIN_instance_ids'])) for camera,calibration in calibrations.items()])
                try:
                    prediction=estimate_cup_feature_pose(points,models[role],views,reference_time=float(now))
                except (ValueError,np.linalg.LinAlgError,FloatingPointError) as error:
                    prediction=dict(version=VERSION,pose_xyzw=None,status='prediction_failure',ambiguous=True,
                        candidate_count=0,error_type=type(error).__name__,error=str(error),
                        GT_or_native_pose_used=False,policy_authorized=False,full_orientation_promoted=False)
                row=dict(case_id=case['case_id'],task=case['task'],parent_source_group=case['parent_source_group'],
                    frame=frame,body_id=role,time=float(now),raw_current_points=len(points),
                    prediction_seconds=time.perf_counter()-tick,prediction=prediction,GT_used=False)
                stream.write(json.dumps(row,ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n');count+=1
    return dict(case_id=case['case_id'],body_frames=count,wall_seconds=time.perf_counter()-started,
        predictions=pointer(filename),GT_used=False,CAD_feature_receipts={role:model.receipt for role,model in models.items()})


def statistics(values):
    values=np.asarray(values,float)
    if not len(values):return dict(n=0,p50=None,p90=None,p95=None,max=None)
    return dict(n=len(values),p50=float(np.median(values)),p90=float(np.percentile(values,90)),
        p95=float(np.percentile(values,95)),max=float(np.max(values)))


def metrics_after_predictions(plan,completed):
    """First GT access during execution occurs here, after all files are saved."""
    rows=[];sequences={};outputs={case['case_id']:case for case in completed}
    for case in plan['cases']:
        report=read(ROOT/case['metric_report']['path'])
        predictions=[json.loads(line) for line in (ROOT/outputs[case['case_id']]['predictions']['path']).read_text().splitlines()]
        require(len(predictions)==48,'every fixed cup/frame prediction must be retained')
        for row in predictions:
            frame,role=row['frame'],row['body_id'];native=report['frames'][frame]['metric_only']
            j=native['body_ids'].index(role);gt=native['native_current_ground_truth'];prediction=row['prediction']
            metric=dict(case_id=case['case_id'],task=case['task'],parent_source_group=case['parent_source_group'],
                frame=frame,body_id=role,pose_present=prediction['pose_xyzw'] is not None,
                ambiguous=prediction['ambiguous'],status=prediction['status'],
                prediction_seconds=row['prediction_seconds'],position_error_m=None,SO3_error_degrees=None,
                GT_used_as_prediction_input=False)
            if metric['pose_present']:
                pose=Pose.from_xyzw(prediction['pose_xyzw']);target=np.asarray(gt['positions'][j]);rotation=np.asarray(gt['rotations'][j])
                metric['position_error_m']=float(np.linalg.norm(pose.position-target))
                metric['SO3_error_degrees']=float(np.degrees(Rotation.from_matrix(pose.rotation@rotation.T).magnitude()))
            rows.append(metric)
    def summarize(selected):
        present=[r for r in selected if r['pose_present']];angles=[r['SO3_error_degrees'] for r in present];positions=[r['position_error_m'] for r in present]
        return dict(body_frames=len(selected),pose_present=len(present),pose_missing=len(selected)-len(present),
            ambiguous=sum(r['ambiguous'] for r in selected),unambiguous_proposals=sum(not r['ambiguous'] and r['pose_present'] for r in selected),
            full_SO3_error_degrees=statistics(angles),position_error_m=statistics(positions),
            SO3_below_10_degrees=sum(a<10 for a in angles),SO3_below_30_degrees=sum(a<30 for a in angles),
            SO3_above_150_degrees=sum(a>150 for a in angles),position_below_5mm=sum(p<.005 for p in positions),
            jointly_SO3_below_10deg_position_below_5mm=sum(r['SO3_error_degrees']<10 and r['position_error_m']<.005 for r in present),
            unambiguous_but_SO3_above_30deg=sum(not r['ambiguous'] and r['SO3_error_degrees']>30 for r in present),
            prediction_seconds=statistics([r['prediction_seconds'] for r in selected]),
            status_counts=dict(Counter(r['status'] for r in selected)))
    for case in plan['cases']:
        for role in ROLES:
            key=case['case_id']+'__'+role;sequences[key]=summarize([r for r in rows if r['case_id']==case['case_id'] and r['body_id']==role])
    return dict(overall=summarize(rows),by_sequence=sequences,per_frame=rows,
        by_independent_parent={parent:summarize([r for r in rows if r['parent_source_group']==parent]) for parent in plan['independent_TRAIN_parents']})


def run(plan_path,output):
    plan=verify(plan_path);output=Path(output).resolve()
    require(output.is_relative_to(BASE) and not output.exists(),'fresh R10 output directory required')
    output.mkdir(parents=True);started=time.perf_counter();completed=[]
    canonical_asset=read(ROOT/plan['canonical_CAD']['path'])
    for case in plan['cases']:
        result=predict_case(case,canonical_asset,output);completed.append(result)
        write(output/'prediction_progress.json',dict(plan_sha256=plan['plan_sha256'],completed_cases=len(completed),
            expected_cases=7,completed_body_frames=sum(row['body_frames'] for row in completed),GT_metrics_read=False,updated_utc=utc()))
        print(json.dumps(dict(case=case['case_id'],body_frames=result['body_frames'],wall_seconds=result['wall_seconds'],GT_used=False)),flush=True)
    require(sum(row['body_frames'] for row in completed)==336,'complete fixed prediction coverage required')
    completed_at=utc();prediction_receipt=dict(version=VERSION,complete=True,completed_utc=completed_at,
        plan_sha256=plan['plan_sha256'],expected_body_frames=336,actual_body_frames=336,cases=completed,
        all_predictions_completed_before_GT_metrics=True,GT_used=False,parameter_search=False)
    write(output/'PREDICTIONS_COMPLETE.json',prediction_receipt)
    # Reverify all source/image hashes after prediction and before GT access.
    verify(plan_path);metrics_started=utc();metrics=metrics_after_predictions(plan,completed)
    result=dict(version=VERSION,complete=True,plan=pointer(plan_path),plan_sha256=plan['plan_sha256'],
        predictions_complete=pointer(output/'PREDICTIONS_COMPLETE.json'),predictions_completed_utc=completed_at,
        metrics_started_utc=metrics_started,total_wall_seconds=time.perf_counter()-started,
        TRAIN_only=True,VAL_used=False,TEST_used=False,native_simulator_started=False,GPU_used=False,
        fixed_parameter_search_used=False,core_files_modified=False,production_policy_authorized=False,
        independent_TRAIN_parent_count=4,repeated_caches_are_not_independent=True,**metrics,
        limitations=plan['limitations'])
    write(output/'report.json',result)
    o=result['overall'];lines=[VERSION,'complete: true; TRAIN-only; no native/GPU; no policy authorization',
        f"Predictions {o['body_frames']}/336; pose present {o['pose_present']}; ambiguous {o['ambiguous']}",
        f"Independent TRAIN parents: 4; repeated cache/body frames are correlated",
        f"SO3<10deg: {o['SO3_below_10_degrees']}/336; SO3<30deg: {o['SO3_below_30_degrees']}/336",
        'Full SO3 error degrees: '+json.dumps(o['full_SO3_error_degrees']),
        'Position error m: '+json.dumps(o['position_error_m']),
        'Prediction seconds: '+json.dumps(o['prediction_seconds']),
        f"Unambiguous proposals with SO3>30deg: {o['unambiguous_but_SO3_above_30deg']}",
        f"Wall seconds: {result['total_wall_seconds']:.3f}",
        'No failure/ambiguous frame was removed; all336 predictions preceded GT metrics.']
    (output/'report.txt').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(complete=True,overall=o,report_sha256=sha(output/'report.json'))),flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare');p.add_argument('--plan',required=True)
    p=sub.add_parser('run');p.add_argument('--plan',required=True);p.add_argument('--out',required=True)
    args=parser.parse_args()
    if args.command=='prepare':
        plan=prepare(args.plan);print(json.dumps(dict(plan_sha256=plan['plan_sha256'],body_frames=336,independent_parents=4)))
    else:run(args.plan,args.out)
