"""TRAIN-only orientation-registration pilot, without teacher or model fitting.

Preparation is CPU-only. Native single-case execution is separately invoked
after the new collection/runtime inputs are frozen. Full rotation is a test
hypothesis, never an automatically accepted observability annotation.
"""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from our_mani.contracts import Pose
from our_mani.multitask.perception import MultiBodySE3Tracker,RegistrationConfig
from our_mani.multitask.specs import TASK_SPECS

BASE=ROOT/'artifacts/framework/multitask_r10'
VERSION='r10_TRAIN_orientation_registration_pilot_v1'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def clean(value):
    if isinstance(value,dict):return {k:clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [clean(v) for v in value]
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    return value


def digest(value):
    return hashlib.sha256(json.dumps(clean(value),sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def require(value,message):
    if not value:raise ValueError(message)


def write(path,value):
    path=Path(path).resolve()
    require(path.is_relative_to(BASE.resolve()) and not path.exists(),'new R10 output only; preserve existing evidence')
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(clean(value),ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


def rotation_metrics(estimated,truth):
    """Raw physical SO3 error and three directed/undirected CAD-axis errors."""
    estimated,truth=np.asarray(estimated,float),np.asarray(truth,float)
    require(estimated.shape==truth.shape==(3,3) and np.isfinite(estimated).all() and np.isfinite(truth).all(),
            'finite rotation matrices required for metric-only comparison')
    require(all(np.allclose(r.T@r,np.eye(3),atol=1e-5) and abs(np.linalg.det(r)-1)<1e-5
                for r in (estimated,truth)),'rotation metrics require physical SO3 matrices')
    result=dict(raw_SO3_error_rad=float(Rotation.from_matrix(estimated@truth.T).magnitude()),axes={})
    for index,name in enumerate(('local_x','local_y','local_z')):
        dot=float(np.clip(estimated[:,index]@truth[:,index],-1.,1.))
        result['axes'][name]=dict(directed_error_rad=float(np.arccos(dot)),
                                 undirected_error_rad=float(np.arccos(abs(dot))))
    return result


def symmetry_residuals(asset,*,maximum_points=4096):
    """Raw finite-sample CAD self-Chamfer curves, never a symmetry certificate."""
    result={}
    for role,body in asset['bodies'].items():
        points=np.asarray(body['collision_vertices_body'],float)
        if len(points)>maximum_points:points=points[np.linspace(0,len(points)-1,maximum_points,dtype=int)]
        center=points.mean(0);points=points-center
        tree=cKDTree(points);diameter=float(np.linalg.norm(np.ptp(points,axis=0)))
        curves={}
        for index,name in enumerate(('local_x','local_y','local_z')):
            axis=np.eye(3)[index];rows=[]
            for angle in np.arange(1,12)*np.pi/6:
                moved=points@Rotation.from_rotvec(axis*angle).as_matrix().T
                forward=tree.query(moved)[0];reverse=cKDTree(moved).query(points)[0]
                rms=float(np.sqrt((np.mean(forward**2)+np.mean(reverse**2))/2))
                rows.append(dict(angle_rad=float(angle),symmetric_nn_rms_m=rms,
                                 rms_over_body_diameter=rms/diameter if diameter>0 else None))
            curves[name]=rows
        result[role]=dict(points=len(points),body_diameter_m=diameter,curves=curves,
            rotation_center='sampled CAD vertex centroid; free translation cannot disambiguate an off-origin symmetry',
            rotation_center_body=center.tolist(),
            interpretation='raw finite-sample geometry only; component labels/view visibility and continuous/discrete symmetry need review',
            absence_of_detected_symmetry_is_not_proof_of_full_observability=True)
    return result


class PilotTracker(MultiBodySE3Tracker):
    """Same tracker math; retain raw fits before any symmetry-gauge projection."""
    def _registration(self,members,clouds,previous=None):
        fitted=super()._registration(members,clouds,previous)
        row=dict(members=list(members),fit_returned=fitted is not None)
        if fitted is not None:
            raw={}
            for role,transform in members.items():
                raw[role]=Pose(fitted['pose'].apply(transform.position),
                               fitted['pose'].rotation@transform.rotation).xyzw().tolist()
            row.update(raw_body_poses_xyzw=raw,rms_m=fitted['rms_m'],per_role=clean(fitted['per_role']),
                valid=fitted['valid'],rotation_ambiguous=fitted['rotation_ambiguous'])
        self.raw_calls.append(row)
        return fitted

    def pilot_observe(self,observation,*,modes):
        self.raw_calls=[]
        states,metadata=super().observe(observation,modes=modes)
        return states,metadata,self.raw_calls


def build_plan(collection_plan_path,output,*,frames=16,rigid_layout_map=None):
    from collect_multitask_training_r10 import verify_plan,load_rigid_constraints
    from audit_train_resets_r10 import native_inputs
    require(type(frames) is int and 10<=frames<=20,'pilot uses 10..20 current frames including reset')
    collection_plan_path=Path(collection_plan_path).resolve();collection=verify_plan(collection_plan_path)
    source_files=dict(collection['inputs']);source_files[str(collection_plan_path)]=sha(collection_plan_path)
    source_files.update(native_inputs(collection['protocol']))
    native_repo=Path(collection['protocol']['baseline_repo'])
    source_files.update({str(native_repo/name):sha(native_repo/name) for name in
        ('third_party/PyRep/pyrep/backend/sim.py','third_party/PyRep/pyrep/objects/shape.py','third_party/PyRep/cffi_build/sim.h')})
    own=[Path(__file__),ROOT/'scripts/audit_train_resets_r10.py',
        *[ROOT/'our_mani/multitask'/name for name in ('perception.py','mode_observer.py','safe_native_mesh.py',
          'training_geometry.py','maps.py','simulation.py','reset_parity.py','native_tasks.py','specs.py','protocol.py')]]
    source_files.update({str(p):sha(p) for p in own})
    layouts={}
    if rigid_layout_map is not None:
        pointer=Path(rigid_layout_map).resolve();layouts=json.loads(pointer.read_text());source_files[str(pointer)]=sha(pointer)
    require(not set(layouts)-set(TASK_SPECS),'rigid layout map contains unknown task')
    cases=[]
    for task in TASK_SPECS:
        rows=[row for row in collection['cases'] if row['task']==task and row['split']=='train' and row['replica']==0]
        require(rows,'missing fixed TRAIN case '+task)
        row=rows[0];rigid=None
        source=Path(row['source']['path']).resolve();cache=ROOT/row['offline_demonstration']['path']
        require(source.name=='data.h5' and 'train_package_compressed' in source.parts,
                'orientation fitting cannot use VAL/TEST sources')
        require(sha(cache)==row['offline_demonstration']['sha256'],'original small cache changed')
        source_files[str(source)]=sha(source);source_files[str(cache.resolve())]=sha(cache)
        entry=layouts.get(task)
        if entry is not None:
            require(isinstance(entry,dict) and entry.get('validated_on_TRAIN') is True,
                    'rigid-group pilot requires explicitly validated TRAIN evidence')
            p=(ROOT/entry['path']).resolve();require(sha(p)==entry['sha256'],'rigid layout pointer changed')
            constraints=load_rigid_constraints(p,task);require(constraints,'empty accepted TRAIN rigid layout')
            source_files[str(p)]=sha(p);rigid=dict(path=str(p),sha256=sha(p),accepted_constraints=len(constraints))
        cases.append(dict(task=task,source=row['source'],replica=0,
            parent_source_group=row['parent_source_group'],offline_demonstration=row['offline_demonstration'],
            fixed_collection_case_id=row['case_id'],rigid_layout=rigid))
    payload=dict(version=VERSION,created_utc=datetime.now(timezone.utc).isoformat(),split='train',
        collection_plan_path=str(collection_plan_path),collection_plan_sha256=sha(collection_plan_path),
        inputs_sha256=source_files,protocol=collection['protocol'],cases=cases,frames=frames,hold_ticks=frames-1,
        registration_config=asdict(RegistrationConfig()),teacher_used=False,GPU_used=False,
        policy_or_dynamics_fitted=False,TEST_used=False,source_selection='first fixed TRAIN source/replica0 per task',
        hypotheses=['none_baseline','candidate_full_except_ball_none'],
        observation_variants='direct; same-frame rigid-group only when a separately validated frozen TRAIN artifact is supplied',
        axis_evaluation='raw fitted local x/y/z directed and undirected axis errors, metric-only; no claimed gauge certificate',
        sphere_orientation='ball always none; no physical rotational error label for ball',
        metric_boundary='native current GT read only after all causal trackers; independent metric namespace; never tracker initialization/history',
        full_hypothesis_is_not_verified_annotation=True,automatic_annotation_promotion=False,
        limitations=['one TRAIN source/seed and short unmanipulated hold sequence per task; not across-source observability validation',
            'finite CAD symmetry curves are diagnostic, not proof that no continuous/discrete symmetry exists',
            'native fatal stops immediately; shortened coverage is explicit and never padded'],
        native_run_performed=False,execution_gate='accepted R09 formal400 and canonical ten videos; new collection/runtime lock must remain unchanged')
    payload['plan_sha256']=digest(payload);write(output,payload);return payload


def verify_plan(path):
    from collect_multitask_training_r10 import verify_plan as verify_collection
    from audit_train_resets_r10 import native_inputs
    plan=json.loads(Path(path).read_text())
    require(plan.get('version')==VERSION and plan.get('split')=='train' and plan.get('TEST_used') is False
        and plan.get('plan_sha256')==digest({k:v for k,v in plan.items() if k!='plan_sha256'}),'orientation plan changed')
    require(len(plan['cases'])==14 and {r['task'] for r in plan['cases']}==set(TASK_SPECS),'all fourteen fixed TRAIN cases required')
    collection=verify_collection(plan['collection_plan_path'])
    require(sha(plan['collection_plan_path'])==plan['collection_plan_sha256'] and collection['protocol']==plan['protocol'],
            'collection/runtime identity changed')
    for name,expected in plan['inputs_sha256'].items():require(sha(name)==expected,'pilot source changed: '+name)
    needed={str(Path(__file__).resolve()),str(ROOT/'scripts/audit_train_resets_r10.py'),
        *[str(ROOT/'our_mani/multitask'/name) for name in ('perception.py','mode_observer.py','safe_native_mesh.py',
          'training_geometry.py','maps.py','simulation.py','reset_parity.py','native_tasks.py','specs.py','protocol.py')]}
    needed.update(native_inputs(plan['protocol']))
    native_repo=Path(plan['protocol']['baseline_repo'])
    needed.update(str(native_repo/name) for name in
        ('third_party/PyRep/pyrep/backend/sim.py','third_party/PyRep/pyrep/objects/shape.py','third_party/PyRep/cffi_build/sim.h'))
    require(needed<=set(plan['inputs_sha256']) and all(plan['inputs_sha256'].get(k)==v for k,v in collection['inputs'].items()),
            'pilot source hash table omitted or changed a frozen collection/runtime input')
    require(10<=plan['frames']<=20 and plan['hold_ticks']==plan['frames']-1
        and plan['registration_config']==clean(asdict(RegistrationConfig())),'pilot budget or registration parameters changed')
    for row in plan['cases']:
        first=next(c for c in collection['cases'] if c['task']==row['task'] and c['split']=='train' and c['replica']==0)
        require(row['source']==first['source'] and row['fixed_collection_case_id']==first['case_id'],
                'pilot source differs from fixed first TRAIN source')
        require(str(Path(row['source']['path']).resolve()) in plan['inputs_sha256']
            and str((ROOT/row['offline_demonstration']['path']).resolve()) in plan['inputs_sha256'],
            'pilot must bind selected HDF5 and original small cache content SHA')
    return plan


def metrics_for_variant(states,metadata,raw_calls,gt,body_ids):
    output={};by_id={state.object_id:state for state in states}
    # The last call for each role is the actual direct fallback when a group fails.
    raw={}
    for call in raw_calls:
        for role,pose in call.get('raw_body_poses_xyzw',{}).items():raw[role]=(pose,call)
    for index,role in enumerate(body_ids):
        diagnostic=metadata['bodies'][role];state=by_id.get(role);truth_r=gt['rotations'][index]
        row=dict(causal_registration=clean(diagnostic),candidate_state_present=state is not None,
            full_orientation_verified=False,sphere_rotation_label_forbidden=role=='ball')
        if state is not None:
            row['emitted_position_error_m']=float(np.linalg.norm(state.pose.position-gt['positions'][index]))
            row['emitted_rotation_gauge_xyzw']=state.pose.xyzw().tolist()
        if role in raw:
            pose,call=raw[role];estimated=Pose.from_xyzw(pose)
            row['raw_fit_metric_only']=dict(raw_pose_xyzw=pose,
                position_error_m=float(np.linalg.norm(estimated.position-gt['positions'][index])),
                rotation_errors=None if role=='ball' else rotation_metrics(estimated.rotation,truth_r),
                rms_m=call['rms_m'],fit_valid=call['valid'],rotation_ambiguous=call['rotation_ambiguous'],
                physical_orientation_claimed=False)
        else:row['raw_fit_metric_only']=None
        output[role]=row
    return output


def run_case(plan_path,task,output):
    from audit_train_resets_r10 import require_completed_delivery
    from collect_multitask_training_r10 import read_native_targets,load_rigid_constraints
    from our_mani.multitask.mode_observer import PublicModeObserver
    plan=verify_plan(plan_path);gate=require_completed_delivery()
    rows=[r for r in plan['cases'] if r['task']==task];require(len(rows)==1,'unknown task')
    row=rows[0];output=Path(output).resolve()
    require(output.is_relative_to(BASE.resolve()) and not output.exists(),'fresh pilot output required')
    output.mkdir(parents=True);env=None;frames=[];sensor_frames=[]
    try:
        from our_mani.multitask.simulation import AlignedDevelopmentSimulation
        from our_mani.multitask.training_geometry import export_train_geometry,declared_instance_bindings,body_templates_from_asset
        from our_mani.multitask.reset_parity import capture_recorded_initial_conditions
        env=AlignedDevelopmentSimulation(task,row['source']['path'],source=row['source'],split='train',
            protocol=plan['protocol'],seed=plan['protocol'].get('seed',0),replica=0,
            servo_lead=plan['protocol'].get('local_servo_lead',1.4))
        env.reset_development(variation=row['source']['variation'])
        observation,robot=env.observe();initial=capture_recorded_initial_conditions(env,observation,robot)
        asset=export_train_geometry(env,dict(row['source'],split='train'));write(output/'TRAIN_geometry.json',asset)
        bindings=declared_instance_bindings(env,asset,shared_permission=True);ids=tuple(asset['bodies'])
        constraints=load_rigid_constraints(row['rigid_layout']['path'],task) if row['rigid_layout'] else ()
        trackers={}
        for hypothesis in plan['hypotheses']:
            annotations={role:'none' if hypothesis=='none_baseline' or role=='ball' else 'full' for role in ids}
            templates=body_templates_from_asset(asset,bindings,orientation_observability=annotations,
                sphere_radii=plan['protocol'].get('body_sphere_radii',{}).get(task))
            trackers[hypothesis+'__direct']=PilotTracker(templates,task_spec=env.spec)
            if constraints:trackers[hypothesis+'__rigid_group']=PilotTracker(templates,task_spec=env.spec,rigid_children=constraints)
        observer=PublicModeObserver(ids,env.spec.graspable_roles,plan['protocol']['attachment_association_distance_m'])
        previous=[];journal=output/'frames.jsonl'
        for frame in range(plan['frames']):
            if frame:observation,robot=env.observe()
            before=env.physics_time(),float(env.task.t),env.ledger.ticks
            sensors=dict(times=robot.timestamp,robot_pose=robot.pose.xyzw().copy(),robot_joints=robot.joints.copy(),
                robot_joint_velocities=robot.joint_velocities.copy(),robot_gripper_open=robot.gripper_open,
                robot_public_contact=robot.contact)
            for camera in ('front','8','16','36'):
                for key,channel in (('rgb','observation/images'),('depth','observation/depths'),('mask','observation/instance_masks')):
                    sensors[key+'_'+camera]=np.asarray(observation[channel][camera]).copy()
            sensor_frames.append(sensors)
            modes,invalid=observer.observe(previous,robot);estimates={}
            for name,tracker in trackers.items():estimates[name]=tracker.pilot_observe(observation,modes=modes)
            # This native read occurs strictly after every tracker, never before fitting.
            truth=read_native_targets(env,ids)
            metric={name:metrics_for_variant(*values,truth,ids) for name,values in estimates.items()}
            record=dict(frame=frame,time=robot.timestamp,counted_tick=env.ledger.ticks,
                causal_tracker_inputs='current/past four RGBD/current masks/explicit shared CAD and optional frozen TRAIN rigid layout',
                tracker_diagnostics={name:clean(values[1]) for name,values in estimates.items()},
                raw_registration_calls={name:values[2] for name,values in estimates.items()},
                public_mode_identity_invalid=sorted(invalid),
                metric_only=dict(native_current_ground_truth=clean(truth),body_ids=ids,registration_errors=metric),
                GT_used_as_tracker_input=False)
            require(before==(env.physics_time(),float(env.task.t),env.ledger.ticks),'current fit/metric read advanced physics')
            frames.append(clean(record))
            with journal.open('a') as stream:stream.write(json.dumps(clean(record),allow_nan=False)+'\n');stream.flush()
            previous=estimates['none_baseline__direct'][0]
            if frame==plan['frames']-1 or env._native_terminal_latched:break
            env.hold_tick()
        alignment=env.alignment_receipt();env.close();env=None
        verify_plan(plan_path);require_completed_delivery()
        sensor_path=output/'sensor_observations.npz'
        with sensor_path.open('wb') as stream:
            np.savez_compressed(stream,**{k:np.asarray([r[k] for r in sensor_frames]) for k in sensor_frames[0]})
        report=dict(version=VERSION,complete=True,TRAIN_only=True,TEST_used=False,task=task,source=row['source'],
            metric_only=True,formal_metrics_included=False,teacher_used=False,GPU_used=False,
            planned_frames=plan['frames'],actual_frames=len(frames),counted_hold_ticks=len(frames)-1,
            shortened_coverage=len(frames)!=plan['frames'],initial_conditions=initial,alignment_receipt=alignment,
            plan_sha256=plan['plan_sha256'],release_gate=gate,variants=list(trackers),frames=frames,
            causal_sensor_observations=dict(path=str(sensor_path.relative_to(ROOT)),sha256=sha(sensor_path),
                native_GT_or_task_outcome_included=False,calibrations='recorded_initial_conditions.public_initial_conditions.cameras'),
            CAD_symmetry_diagnostic=symmetry_residuals(asset),annotation_changes_written=False,
            automatic_annotation_promotion=False,orientation_verified_for_policy=False)
        write(output/'report.json',report);return report
    finally:
        if env is not None:env.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__);mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare',action='store_true');mode.add_argument('--case',choices=tuple(TASK_SPECS))
    parser.add_argument('--collection-plan');parser.add_argument('--plan');parser.add_argument('--frames',type=int,default=16)
    parser.add_argument('--rigid-layout-map');parser.add_argument('--out',required=True);args=parser.parse_args()
    if args.prepare:
        require(args.collection_plan,'preparation requires the newly frozen collection plan')
        value=build_plan(ROOT/args.collection_plan,ROOT/args.out,frames=args.frames,
            rigid_layout_map=None if args.rigid_layout_map is None else ROOT/args.rigid_layout_map)
    else:
        require(args.plan,'single native pilot requires frozen pilot plan')
        value=run_case(ROOT/args.plan,args.case,ROOT/args.out)
    print(json.dumps(dict(complete=value.get('complete'),native_run_performed=value.get('native_run_performed'),
        automatic_annotation_promotion=False),ensure_ascii=False))


if __name__=='__main__':main()
