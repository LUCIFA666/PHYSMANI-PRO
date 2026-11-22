"""Fixed TRAIN first-frame or saved-sequence joint-depth refinement diagnostic.

The original tracker is observed read-only using a local Python trace at its
registration return. This captures its exact retained post-ICP candidates;
it neither injects GT nor changes the core candidate generator. All native
metric fields are accessed after the complete prediction inventory.
"""
import argparse
from dataclasses import asdict
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from our_mani.contracts import Pose
from our_mani.perception import CameraCalibration
from our_mani.multitask.perception import MultiBodySE3Tracker,RegistrationConfig
from our_mani.multitask.specs import get_task_spec
from our_mani.multitask.depth_pose_refinement import JointRefinementConfig,refine_pose_candidates
from scripts import replay_depth_pose_consistency_r10 as original
from scripts import analyze_multitask_smoke_r10 as common
from scripts import collect_multitask_training_r10 as c

VERSION='r10_fixed_TRAIN_joint_depth_refinement_replay_v1'


def clean(value):
    if isinstance(value,Pose):return dict(position=value.position.tolist(),rotation=value.rotation.tolist(),xyzw=value.xyzw().tolist())
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,dict):return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [clean(v) for v in value]
    return value


def prepare(path,*,full_sequences=False):
    old=common.read(c.BASE/'TRAIN_current_depth_pose_replay_plan_v1.json')
    cases=old['cases'] if full_sequences else old['cases'][:4]
    common.require(len(cases)==(7 if full_sequences else 4),'fixed parent inventory differs')
    if not full_sequences:common.require(len({case['parent_source_group'] for case in cases})==4,'four independent TRAIN parents required')
    inputs={}
    for case in cases:
        for key in ('report','sensor','geometry'):
            p=(ROOT/case[key]).resolve();expected=old['inputs_sha256'][str(p)]
            common.require(c.sha(p)==expected,'saved TRAIN input changed: '+str(p));inputs[str(p)]=expected
    sources=(Path(__file__).resolve(),Path(original.__file__).resolve(),Path(common.__file__).resolve(),Path(c.__file__).resolve(),
        ROOT/'our_mani/multitask/perception.py',ROOT/'our_mani/multitask/depth_pose_refinement.py',
        ROOT/'our_mani/multitask/depth_pose_consistency.py',ROOT/'our_mani/multitask/axis_evidence.py',
        ROOT/'our_mani/multitask/specs.py',ROOT/'our_mani/contracts.py',ROOT/'our_mani/perception.py')
    inputs.update({str(p):c.sha(p) for p in sources})
    plan=dict(version=VERSION,created_utc=c.utc(),cases=cases,inputs_sha256=inputs,
        frames_per_case=16 if full_sequences else 1,expected_cases=len(cases),
        expected_body_frames=len(cases)*3*(16 if full_sequences else 1),
        registration_config=asdict(RegistrationConfig(free_space_weight=2.,free_space_tolerance_m=.004,free_space_maximum_gap_m=.03)),
        refinement_config=asdict(JointRefinementConfig()),retained_candidates_per_registration=4,
        retained_history_candidate='original core may fit an extra forced-history candidate; refine the lowest-scored original four only, record original count and omitted history candidate; no GT candidates',
        first_frame_scope=not full_sequences,exact_original_candidate_capture='read-only _registration return locals.fitted; no source mutation',
        GT_access='only final metrics after every prediction; never seed, bounds, optimization or selection',
        TRAIN_only=True,no_VAL_TEST=True,no_native_GPU=True,no_threshold_search=True,
        no_core_changes=True,annotation_promotion=False)
    plan['plan_sha256']=c.canonical_sha(plan);c.write_json(path,plan);return plan


def verify(path):
    plan=common.read(path);common.require(plan['version']==VERSION and plan['plan_sha256']==c.canonical_sha(
        {k:v for k,v in plan.items() if k!='plan_sha256'}),'joint replay plan changed')
    for name,digest in plan['inputs_sha256'].items():common.require(c.sha(name)==digest,'joint replay source/input changed: '+name)
    return plan


def observe_and_capture(tracker,observation,modes):
    captures={};code=MultiBodySE3Tracker._registration.__code__;previous_trace=sys.gettrace()
    def trace(frame,event,arg):
        if frame.f_code is not code:return None
        if event=='return' and isinstance(arg,dict) and 'fitted' in frame.f_locals:
            local=frame.f_locals;members=local['members']
            common.require(len(members)==1,'this cup prototype never assumes permanent body binding')
            role=next(iter(members));surface={}
            for member,transform in members.items():
                for component in tracker.templates[member].components:
                    surface[member+':'+component.name]=transform.apply(component.points_body)
            observed={role+':'+name:points.copy() for member,name,_,points,_ in local['pieces'] for role in (member,)}
            captures[role]=dict(candidate_poses=[pose for _,pose in local['fitted'][:4]],
                original_fitted_candidate_count=len(local['fitted']),surface=surface,
                observed=observed,original_selection=arg['pose'])
        return trace
    try:
        sys.settrace(trace);states,metadata=tracker.observe(observation,modes=modes)
    finally:sys.settrace(previous_trace)
    return states,metadata,captures


def predict_case(case,plan):
    asset=common.read(ROOT/case['geometry']);templates=original.templates_from_original_train_asset(asset,case['roles'])
    tracker=MultiBodySE3Tracker(templates,task_spec=get_task_spec(case['task']),config=RegistrationConfig(**plan['registration_config']))
    calibration={name:CameraCalibration(np.asarray(row['intrinsics']),Pose.from_xyzw(row['camera_to_world']))
        for name,row in case['public_cameras'].items()}
    rows=[]
    with np.load(ROOT/case['sensor'],allow_pickle=False) as a:
        for i in range(plan['frames_per_case']):
            common.require(not a['robot_public_contact'][i],'hold-only public no-contact diagnostic expected')
            now=float(a['times'][i]);observation={'timestamp':now,'observation/calibrations':calibration,
                'observation/camera_timestamps':{name:now for name in calibration}}
            for channel,prefix in (('observation/depths','depth'),('observation/instance_masks','mask')):
                observation[channel]={name:a[prefix+'_'+name][i] for name in calibration}
            states,metadata,captures=observe_and_capture(tracker,observation,{role:'external' for role in case['roles']})
            views={name:(calibration[name],observation['observation/depths'][name]) for name in calibration}
            bodies={}
            for role in case['roles']:
                capture=captures.get(role)
                if capture is None:
                    bodies[role]=dict(status='no_original_registration_candidates',original_pose=None,joint_pose=None)
                    continue
                count=len(capture['candidate_poses'])
                common.require(count==4 and capture['original_fitted_candidate_count'] in (4,5),
                    'original retained candidate budget differs')
                result=refine_pose_candidates(capture['surface'],capture['observed'],capture['candidate_poses'],views)
                bodies[role]=dict(status='refined_fixed_original_candidates',original_pose=capture['original_selection'],
                    joint_pose=result['pose'],original_causal_registration=metadata['bodies'][role],refinement=result,
                    original_fitted_candidate_count=capture['original_fitted_candidate_count'],
                    original_extra_candidate_not_optimized=max(0,capture['original_fitted_candidate_count']-4),
                    refined_pose_observability_claimed=False)
            rows.append(clean(dict(frame=i,time=now,bodies=bodies,GT_used=False)))
    return rows


def metrics(case,predictions):
    # No native GT fields are accessed during predict_case/candidate capture.
    report=common.read(ROOT/case['report']);rows=[]
    for frame in predictions:
        truth=report['frames'][frame['frame']]['metric_only'];gt=truth['native_current_ground_truth']
        for role,prediction in frame['bodies'].items():
            j=truth['body_ids'].index(role);metric=dict(frame=frame['frame'],role=role,status=prediction['status'])
            for key in ('original_pose','joint_pose'):
                pose=prediction[key]
                if pose is None:continue
                p=Pose(np.asarray(pose['position']),np.asarray(pose['rotation']))
                metric[key]=dict(position_error_m=float(np.linalg.norm(p.position-np.asarray(gt['positions'][j]))),
                    full_SO3_error_deg=float(np.rad2deg(Rotation.from_matrix(p.rotation@np.asarray(gt['rotations'][j]).T).magnitude())))
            if prediction['status']=='refined_fixed_original_candidates':
                ref=prediction['refinement'];metric.update(candidate_count=ref['candidate_count'],
                    selected_candidate=ref['selected_index'],runtime_seconds=ref['total_runtime_seconds'],
                    max_nfev_per_candidate=max(c['nfev'] for c in ref['candidates']),
                    joint_objective_before=min(c['initial']['joint_cost_m2'] for c in ref['candidates']),
                    joint_objective_after=ref['selection_cost_m2'])
            rows.append(metric)
    return rows


def run(plan_path,output):
    plan=verify(plan_path);output=c.destination(output);common.require(not output.exists(),'fresh output required');output.mkdir(parents=True)
    predictions={}
    for case in plan['cases']:
        predictions[case['case_id']]=predict_case(case,plan)
        c.write_json(output/(case['case_id']+'.json'),predictions[case['case_id']])
        print('predicted '+case['case_id'],flush=True)
    verify(plan_path)
    rows=[dict(case_id=case['case_id'],parent_source_group=case['parent_source_group'],metrics=metrics(case,predictions[case['case_id']])) for case in plan['cases']]
    result=dict(version=VERSION,complete=True,plan=common.pointer(plan_path),plan_internal_sha256=plan['plan_sha256'],
        cases=rows,expected_body_frames=plan['expected_body_frames'],all_predictions_completed_before_GT_metrics=True,
        no_native_GPU=True,no_core_changes=True,no_parameter_search=True,TRAIN_only=True,VAL_TEST_used=False,
        annotation_promotion=False,limitations=['joint local least-squares is not global registration or observability proof',
            'bilinear within cells, nearest CAD/model-ray ownership and visibility boundaries remain nonsmooth',
            'matching uses current observed points and raw local CAD; true foreground occlusion supplies no full-pose evidence',
            'full-sequence variant keeps original scorer history for candidate generation and is a refinement diagnostic, not an integrated joint-history tracker'])
    c.write_json(output/'report.json',result);return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);action=parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--prepare',action='store_true');action.add_argument('--run',action='store_true')
    parser.add_argument('--full-sequences',action='store_true');parser.add_argument('--plan',type=Path,required=True);parser.add_argument('--out',type=Path)
    args=parser.parse_args()
    if args.prepare:r=prepare(args.plan,full_sequences=args.full_sequences);print({'plan_sha256':r['plan_sha256'],'expected_body_frames':r['expected_body_frames']})
    else:
        common.require(args.out is not None,'fresh output required');r=run(args.plan,args.out);print({'complete':r['complete'],'expected_body_frames':r['expected_body_frames']})


if __name__=='__main__':main()
