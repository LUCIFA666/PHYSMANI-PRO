"""No-FASTER candidate interception: CV/context x geometry/local success Value."""
import argparse
from dataclasses import asdict
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import numpy as np
import torch
from PIL import Image,ImageDraw
from our_mani.benchmark_perception import RedBoxTracker,box_surface
from our_mani.benchmark_protocol import file_sha256,validate_manifest,wilson
from our_mani.context_control import controller_sha256
from our_mani.context_dynamics import ContextLatentDynamics
from our_mani.contracts import ObjectState,Pose
from our_mani.dynamics import ConstantVelocityDynamics
from our_mani.forecast_collision import ForecastCollisionScene
from our_mani.goal_perception import RedGoalMarker
from our_mani.intercept_planner import CandidateInterception
from our_mani.local_capture import LocalConfig
from our_mani.local_learning import CandidateSuccessValue
from our_mani.official_simulation import OfficialTestSimulation
from our_mani.pyrep_env import LocalSimulation
from our_mani.video import write_video
from collect_local_rollouts import initialization


def clean(value):
    if isinstance(value,np.ndarray):return clean(value.tolist())
    if isinstance(value,(np.floating,float)):return float(value) if np.isfinite(value) else None
    if isinstance(value,np.integer):return int(value)
    if isinstance(value,dict):return {k:clean(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [clean(v) for v in value]
    if isinstance(value,Pose):return value.xyzw().tolist()
    return value


def write(path,value):
    q=path.with_suffix('.tmp');q.write_text(json.dumps(clean(value),ensure_ascii=False,indent=2,allow_nan=False)+'\n');q.replace(path)


def local_controller_hash(config,predictor):
    local=LocalConfig(max_speed=config['local_max_speed'],max_acceleration=config['local_max_acceleration'])
    if predictor=='context':return controller_sha256(ROOT/config['dynamics_checkpoint'],local,config['local_servo_lead'])
    names=('local_capture.py','control.py','benchmark_perception.py','goal_perception.py','pyrep_env.py')
    return hashlib.sha256(b''.join((ROOT/'our_mani'/name).read_bytes() for name in names)
        +json.dumps({'controller_config':asdict(local),'servo_lead':config['local_servo_lead']},sort_keys=True).encode()).hexdigest()


def run_episode(env,row,config,*,predictor,ranking,official,video):
    dynamics=ContextLatentDynamics(ROOT/config['dynamics_checkpoint']) if predictor=='context' else ConstantVelocityDynamics()
    tracker=RedBoxTracker()
    marker=RedGoalMarker(json.loads((ROOT/config['goal_calibration']).read_text())['offset_train_calibrated_m'])
    local_hash=local_controller_hash(config,predictor)
    value_path=config['context_local_value' if predictor=='context' else 'cv_local_value']
    value=CandidateSuccessValue(ROOT/value_path,controller_sha256=local_hash) if ranking=='value' else None
    env.robot_model.acceleration_limits[:]=config['global_joint_acceleration']
    started=time.perf_counter();frames=[];steps=[]
    record={'created_utc':datetime.now(timezone.utc).isoformat(),'task':row['task'],'raw_episode_id':row['raw_episode_id'],
        'predictor':predictor,'ranking':ranking,'formal_test_used':official,'official_task_success':False,
        'controller_sha256':local_hash,'FASTER_used':False,'NGFF_used':False,'handovers':0,
        'value_scope':config['value_scope'] if value else 'geometry_only_no_probability_claim',
        'value_checkpoint_sha256':file_sha256(ROOT/value_path) if value else None,
        'value_controller_match_verified':bool(value),'synchronous_simulation':True,
        'reset_kind':'released_official_TEST' if official else 'HDF5_kinematics_home_development_reset'}
    if official:
        env.reset_official(row);record['reset_receipt']=env.reset_receipt
    else:
        x,v,a,goal=initialization(row['path'])
        if not env.reset(target_position=x,target_velocity=v,target_acceleration=a,goal_position=goal):
            return {**record,'failure':'development_reset_ik_failed','steps':[],'wall_seconds':time.perf_counter()-started},[]
        env.arm.set_joint_positions(env.home_joints.tolist(),disable_dynamics=True)
        env.arm.set_joint_target_positions(env.home_joints.tolist())
        for _ in range(3):env.pyrep.step()
    collision=ForecastCollisionScene(env.arm,{'target':env.task.target_block})
    workflow=None;goal_pose=None;carrying=None;previous_velocity=None
    max_acceleration=0.;global_acceleration=0.;sim_start=env.time
    for tick in range(round(config['simulation_seconds']/config['dt'])):
        if (ROOT/'runtime/framework/pipeline_v2_STOP').exists():raise InterruptedError('v2 STOP')
        observation,robot=env.observe()
        if robot.task_success:record['official_task_success']=True;break
        if previous_velocity is not None:
            measured=float(np.max(np.abs(robot.joint_velocities-previous_velocity)/config['dt']))
            max_acceleration=max(max_acceleration,measured)
            if workflow is None or workflow.stage!='local':global_acceleration=max(global_acceleration,measured)
        previous_velocity=robot.joint_velocities.copy()
        if goal_pose is None:
            try:goal_pose,record['goal_perception']=marker.observe(observation)
            except ValueError:pass
        try:
            obj,perception=tracker.observe(observation)
            if robot.contact:carrying=(robot.pose.inverse_apply(obj.pose.position),robot.pose.rotation.T@obj.pose.rotation)
        except ValueError as error:
            obj,perception=None,{'error':str(error)}
            if robot.contact and carrying is not None:
                offset,rotation=carrying;pose=Pose(robot.pose.apply(offset),robot.pose.rotation@rotation)
                obj=ObjectState('target',env.time,pose,box_surface(pose,tracker.prior.dimensions),robot.linear_velocity,robot.angular_velocity,confidence=.8)
                perception={'method':'last_visual_contact_and_current_proprioception'}
        if obj is not None and hasattr(dynamics,'observe'):dynamics.observe([obj])
        if workflow is None and goal_pose is not None and tick>=config['warmup_ticks']:
            workflow=CandidateInterception(dynamics,env.robot_model,collision,goal_pose,config,predictor=predictor,ranking=ranking,value=value)
        decision=(workflow.step(robot,obj,config['simulation_seconds']-(env.time-sim_start)) if workflow else
                  {'command':'hold','reason':'history_and_goal_warmup'})
        perception_error=float(np.linalg.norm(obj.pose.position-env.metric_target_position())) if obj else None
        if decision['command']=='joint_trajectory':
            env.arm.set_joint_target_positions(decision['joints'].tolist());env.gripper.release()
            env.gripper.actuate(1.,velocity=.2);env.scene.step();env.time+=config['dt']
        elif decision['command']=='local_capture':
            env.advance(decision['pose'],opening=decision['opening'])
        else:
            env.arm.set_joint_target_positions(robot.joints.tolist());env.scene.step();env.time+=config['dt']
        success,_=env.task.success()
        step={'tick':tick,'time':robot.timestamp-sim_start,'stage':workflow.stage if workflow else 'warmup',
            'decision':decision,'robot_pose':robot.pose.xyzw(),'robot_velocity':robot.linear_velocity,
            'robot_joint_velocities':robot.joint_velocities,'contact':robot.contact,'official_success_after_step':bool(success),
            'perception':perception,'perception_error_metric_only_m':perception_error,
            'diagnostics':dict(workflow.diagnostics) if workflow else {},
            'candidate_goal':{'pose':workflow.goal.pose.xyzw(),'arrival':workflow.goal.arrival_time,'score':workflow.goal.score,
                'local_success_probability':workflow.goal.region.value if ranking=='value' else None} if workflow and workflow.goal else None}
        steps.append(clean(step))
        if video:
            images=observation['observation/images']
            image=Image.fromarray(np.concatenate([np.concatenate([images[n] for n in ('front','8')],1),np.concatenate([images[n] for n in ('16','36')],1)],0))
            ImageDraw.Draw(image).text((8,8),f"NO FASTER | {predictor} + {ranking} | {step['stage']}\nt={step['time']:.2f}s",fill='white',stroke_width=1,stroke_fill='black')
            frames.append(np.asarray(image))
        if tick%20==0:print(json.dumps(clean({'task':row['task'],'episode':row['raw_episode_id'],'tick':tick,'stage':step['stage'],
            'command':decision['command'],'diagnostics':step['diagnostics']}),ensure_ascii=False),flush=True)
        if success:record['official_task_success']=True;record['success_time']=env.time-sim_start;break
        if workflow and workflow.stage=='failed':break
    record.update(handovers=workflow.handovers if workflow else 0,events=workflow.events if workflow else [],steps=steps,
        wall_seconds=time.perf_counter()-started,simulation_seconds=env.time-sim_start,
        maximum_measured_joint_acceleration_rad_s2=max_acceleration,
        maximum_measured_global_joint_acceleration_rad_s2=global_acceleration,
        local_joint_acceleration_hard_constraint_verified=False,
        maximum_servo_reference_error_rad=max([s['decision'].get('servo_reference_error_rad',0.) for s in steps],default=0.),
        captured=any(s['contact'] for s in steps),failure=None if record['official_task_success'] else 'episode_budget_or_policy_failure')
    return clean(record),frames


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictor',choices=('cv','context'),required=True)
    parser.add_argument('--ranking',choices=('geometry','value'),required=True)
    parser.add_argument('--split',choices=('train','validation','test'),default='train')
    parser.add_argument('--episodes-per-task',type=int,default=1)
    parser.add_argument('--config',default='config/framework/pipeline_v2_intercept.json')
    parser.add_argument('--out',required=True)
    parser.add_argument('--videos',type=int,default=0)
    args=parser.parse_args();torch.set_num_threads(4)
    config=json.loads((ROOT/args.config).read_text())
    official=args.split=='test'
    manifest=ROOT/('artifacts/framework/pipeline_v1/official_test_manifest.json' if official else f'artifacts/framework/protocol/{args.split}_manifest.json')
    entries=json.loads(manifest.read_text())['episodes']
    tasks=('pick_moving_target_on_the_table','pick_moving_target_on_the_table_high_speed')
    if official:validate_manifest(entries)
    rows=[r for task in tasks for r in [a for a in entries if a['task']==task][:args.episodes_per_task or None]]
    files=[ROOT/args.config,Path(__file__).resolve(),ROOT/'scripts/collect_local_rollouts.py',*sorted((ROOT/'our_mani').glob('*.py'))]
    identity={'config':config,'predictor':args.predictor,'ranking':args.ranking,'split':args.split,
        'files':{str(p.relative_to(ROOT)):file_sha256(p) for p in files},'manifest_sha256':file_sha256(manifest),
        'dynamics_sha256':file_sha256(ROOT/config['dynamics_checkpoint']) if args.predictor=='context' else None,
        'value_sha256':file_sha256(ROOT/config['context_local_value' if args.predictor=='context' else 'cv_local_value']) if args.ranking=='value' else None}
    identity_sha=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    out=ROOT/args.out;out.mkdir(parents=True,exist_ok=True)
    if (out/'inputs.json').exists() and json.loads((out/'inputs.json').read_text())['sha256']!=identity_sha:
        raise ValueError('use a new run directory when implementation or inputs change')
    write(out/'inputs.json',{'sha256':identity_sha,**identity})
    records=[];recorded_tasks=set()
    for row in rows:
        name=row['task']+'__'+row['raw_episode_id'];path=out/(name+'.json')
        if path.exists():
            record=json.loads(path.read_text())
            if record['inputs_sha256']!=identity_sha:raise ValueError('resumed episode inputs differ')
        else:
            env=None
            try:
                env=(OfficialTestSimulation(row['task'],row['path'],seed=config['seed'],servo_lead=config['local_servo_lead']) if official else
                     LocalSimulation(row['task'],row['path'],seed=config['seed'],servo_lead=config['local_servo_lead']))
                make_video=len(recorded_tasks)<args.videos and row['task'] not in recorded_tasks
                record,frames=run_episode(env,row,config,predictor=args.predictor,ranking=args.ranking,official=official,video=make_video)
                record.update(inputs_sha256=identity_sha,split=args.split,version=config['version'])
                write(path,record)
                if frames:write_video(out/(name+'.mp4'),frames);recorded_tasks.add(row['task'])
            except Exception as error:
                write(out/(name+'.infrastructure_error.json'),{'error':repr(error),'counted_as_policy_failure':False});raise
            finally:
                if env is not None:env.close()
        records.append(record)
        per_task={task:{'completed':sum(r['task']==task for r in records),
            'successes':sum(r['task']==task and r['official_task_success'] for r in records),
            'handovers':sum(r['task']==task and r['handovers']>0 for r in records)} for task in tasks}
        report={'created_utc':datetime.now(timezone.utc).isoformat(),'formal_test_used':official,'split':args.split,
            'kind':'official_TEST' if official else 'development_home_resets_not_paper_benchmark',
            'predictor':args.predictor,'ranking':args.ranking,'completed':len(records),'expected':len(rows),
            'complete':len(records)==len(rows),'inputs_sha256':identity_sha,'per_task':per_task,
            'results':[{k:v for k,v in r.items() if k!='steps'} for r in records]}
        write(out/'report.json',report)
        print(json.dumps({'completed':len(records),'expected':len(rows),'success':record['official_task_success'],'handovers':record['handovers']}),flush=True)


if __name__=='__main__':main()
