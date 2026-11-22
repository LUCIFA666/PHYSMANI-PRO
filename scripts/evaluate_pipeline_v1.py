"""Causal context/SimNorm pipeline on TRAIN home resets or frozen official TEST."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts'),str(ROOT/'third_party/FASTER/packages/openpi-client/src')]
import numpy as np
import torch
from PIL import Image,ImageDraw
from openpi_client.websocket_client_policy import WebsocketClientPolicy
from our_mani.benchmark_protocol import file_sha256,validate_manifest,verify_lock,summarize
from our_mani.benchmark_perception import RedBoxTracker,box_surface
from our_mani.capture_workflow import CaptureWorkflow
from our_mani.context_control import ContextCaptureController,controller_sha256
from our_mani.context_dynamics import ContextLatentDynamics
from our_mani.context_value import ContextCandidateSuccessValue
from our_mani.contracts import Pose,ObjectState,HandoverRegion
from our_mani.execution import FeasibleExecutor
from our_mani.faster import ActionBuffer,FasterApproach
from our_mani.forecast_collision import ForecastCollisionScene
from our_mani.goal_perception import RedGoalMarker
from our_mani.local_capture import LocalConfig
from our_mani.official_simulation import OfficialTestSimulation
from our_mani.pipeline import Stage
from our_mani.planning import RegionSelector
from our_mani.pyrep_env import LocalSimulation
from our_mani.reachability import TimedReachability
from our_mani.video import write_video
from collect_local_rollouts import initialization


def write(path,data):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


def wait_inference(workflow,approach):
    # Benchmark simulation stops while inference runs, as in the released
    # synchronous evaluator. Wall latency remains separately recorded.
    if workflow.future is not None:
        try:workflow.future.result(timeout=120)
        except (ValueError,IndexError):pass  # workflow reports unavailable history
    if approach.future is not None:
        approach.future.result(timeout=120)


def run_episode(env,row,config,port,checkpoint_hash,*,official,video):
    dynamics=ContextLatentDynamics(ROOT/config['dynamics'])
    local=LocalConfig(max_speed=config['local_max_speed'],max_acceleration=config['local_max_acceleration'])
    controller_hash=controller_sha256(ROOT/config['dynamics'],local,config['servo_lead'])
    tracker=RedBoxTracker()
    marker=RedGoalMarker(json.loads((ROOT/config['goal_calibration']).read_text())['offset_train_calibrated_m'])
    client,workflow=None,None
    started=time.perf_counter()
    record={'task':row['task'],'raw_episode_id':row['raw_episode_id'],'split':'test' if official else 'train',
        'formal_test_used':official,'version':config['version'],'official_task_success':False,
        'stable_workflow_success':False,'handovers':0,'controller_sha256':controller_hash,
        'frozen_inputs_sha256':checkpoint_hash,'primary_success_definition':config['primary_success'],
        'synchronous_simulation':True,'physical_budget_seconds':config['simulation_seconds']}
    logs,frames=[],[]
    try:
        if official:
            env.reset_official(row)
            record['reset_receipt']=env.reset_receipt
        else:
            x,v,a,goal=initialization(row['path'])
            # Development home start, not a local handover reset.
            if not env.reset(target_position=x,target_velocity=v,target_acceleration=a,goal_position=goal):
                return {**record,'failure':'development_reset_ik_failed','steps':[],'wall_seconds':time.perf_counter()-started},[]
            env.arm.set_joint_positions(env.home_joints.tolist(),disable_dynamics=True)
            env.arm.set_joint_target_positions(env.home_joints.tolist())
            for _ in range(3):env.pyrep.step()
        client=WebsocketClientPolicy('127.0.0.1',port)
        server=client.get_server_metadata()
        if server['checkpoint_sha256']!=config['faster_sha256']:
            raise ValueError('neural server does not match frozen FASTER checkpoint')
        record['neural_server']=server
        buffer=ActionBuffer()
        approach=FasterApproach(client,buffer,clock=lambda:env.time)
        value=ContextCandidateSuccessValue(ROOT/config['success'],dynamics,controller_sha256=controller_hash,clock=lambda:env.time)
        record['success_model_outcome']=value.value.metadata['outcome']
        collision=ForecastCollisionScene(env.arm,{'target':env.task.target_block})
        env.robot_model.collision_query=collision.check
        env.robot_model.collision_scene='nominal_forecast_target_geometry_and_current_static_scene'
        reachability=TimedReachability(env.robot_model)
        regions=[HandoverRegion(f'candidate_{i}','target',Pose.from_xyzw([*offset,0,1,0,0]),[.04]*3,
            max_relative_speed=.14,source='fixed_TRAIN_templates_context_controller_Value') for i,offset in enumerate(config['regions'])]
        selector=RegionSelector(reachability,success_value=value,minimum_success_probability=config['minimum_success_probability'],
            object_position_bounds={'target':(np.array(config['object_workspace_lower']),np.array(config['object_workspace_upper']))})
        validator=FeasibleExecutor(reachability)
        goal_pose,carrying=None,None
        rejection_counts={}
        sim_start=env.time
        for tick in range(round(config['simulation_seconds']/config['dt'])):
            if (ROOT/'runtime/framework/pipeline_v1_STOP').exists():
                raise InterruptedError('v1 benchmark stop marker')
            observation,robot=env.observe()
            if robot.task_success:
                record['official_task_success']=True
                record['success_simulation_time']=env.time-sim_start
                break
            observation['prompt']='pick up the red block and lift it up to the target'
            if goal_pose is None:
                try:goal_pose,record['goal_perception']=marker.observe(observation)
                except ValueError:pass
            try:
                obj,perception=tracker.observe(observation)
                dynamics.observe([obj])
                if robot.contact:
                    carrying=(robot.pose.inverse_apply(obj.pose.position),robot.pose.rotation.T@obj.pose.rotation)
            except ValueError as error:
                obj,perception=None,{'error':str(error)}
                if robot.contact and carrying is not None:
                    offset,rotation=carrying
                    pose=Pose(robot.pose.apply(offset),robot.pose.rotation@rotation)
                    obj=ObjectState('target',env.time,pose,box_surface(pose,tracker.prior.dimensions),robot.linear_velocity,robot.angular_velocity,confidence=.8)
                    dynamics.observe([obj])
                    perception={'method':'last_visual_contact_and_current_proprioception'}
            if workflow is None and goal_pose is not None and tick>=config['context_warmup_ticks']:
                workflow=CaptureWorkflow(dynamics,selector,regions,approach,buffer,goal_pose,local,
                    horizon=config['forecast_seconds'],collision_scene=collision,
                    local_controller_factory=lambda goal,cfg:ContextCaptureController(goal,cfg,dynamics))
            decision=workflow.step(observation,robot,obj,tick=tick) if workflow is not None else None
            if workflow is not None:wait_inference(workflow,approach)
            reason,applied='context_or_goal_warmup',False
            if decision is not None and decision.command=='faster_action':
                command,reason=validator.prepare_joint_limited(decision.details['action'],robot,now=env.time)
                if command is not None:
                    env.arm.set_joint_target_positions(command.joints.tolist())
                    env.gripper.actuate(1.,velocity=.2)
                    env.scene.step();env.time+=config['dt'];applied=True
                else:
                    rejection_counts[reason]=rejection_counts.get(reason,0)+1
                    env.hold_tick() if official else env.advance(robot.pose)
            elif decision is not None and decision.command=='local_capture':
                applied=env.advance(decision.pose,opening=decision.details['gripper_open'])
                reason='context_MPC_local_executor'
            else:
                env.hold_tick() if official else env.advance(robot.pose,opening=float(not robot.contact))
                reason=(decision.details or {}).get('reason','hold') if decision else reason
            step={'tick':tick,'simulation_time':robot.timestamp-sim_start,'stage':decision.stage.value if decision else 'warmup',
                'command':decision.command if decision else 'hold','applied':applied,'reason':reason,
                'contact':robot.contact,'perception':perception,'selector':dict(selector.last_diagnostics),
                'robot_pose':robot.pose.xyzw().tolist(),'target_position_metric_only':env.metric_target_position().tolist()}
            logs.append(step)
            if video:
                pictures=observation['observation/images']
                mosaic=np.concatenate([np.concatenate([pictures[n] for n in ('front','8')],1),np.concatenate([pictures[n] for n in ('16','36')],1)],0)
                image=Image.fromarray(mosaic)
                ImageDraw.Draw(image).text((8,8),f"Context + SimNorm v1 | {step['stage']}\nt={step['simulation_time']:.2f}s | {reason}",fill='white',stroke_width=1,stroke_fill='black')
                frames.append(np.asarray(image))
            # Check after the final physics step too; no added settling budget.
            success,_=env.task.success()
            if success:
                record['official_task_success']=True
                record['success_simulation_time']=env.time-sim_start
                break
            if tick%20==0:
                print(json.dumps({'task':row['task'],'episode':row['raw_episode_id'],**step},ensure_ascii=False),flush=True)
            if decision is not None and decision.stage in (Stage.SUCCESS,Stage.FAILED):break
        strict_success=workflow is not None and workflow.stage==Stage.SUCCESS
        record.update({'stable_workflow_success':True if strict_success else (None if record['official_task_success'] else False),
            'strict_secondary_not_assessed_after_primary_early_stop':record['official_task_success'] and not strict_success,
            'handovers':sum(e['to']==Stage.LOCAL.value for e in workflow.events) if workflow else 0,
            'events':workflow.events if workflow else [],'steps':logs,'rejected_neural_actions':rejection_counts,
            'approach_timings':approach.timings,'wall_seconds':time.perf_counter()-started,'simulation_seconds':env.time-sim_start,
            'collision_checks':'nominal moving target and current static scene; no uncertainty margin',
            'failure':None if record['official_task_success'] else ('episode_budget_or_policy_failure' if workflow else 'goal_perception_unavailable')})
        return record,frames
    finally:
        if workflow is not None:workflow.close()
        elif client is not None and 'approach' in locals():approach.close()
        if client is not None:client._ws.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=('development','test'),required=True)
    parser.add_argument('--lock',required=True)
    parser.add_argument('--port',type=int,default=18473)
    parser.add_argument('--out',required=True)
    args=parser.parse_args()
    torch.set_num_threads(4)
    lock_path=ROOT/args.lock
    lock=json.loads(lock_path.read_text());verify_lock(ROOT,lock)
    config=lock['config'];lock_hash=file_sha256(lock_path)
    official=args.mode=='test'
    if official:
        rows=json.loads((ROOT/config['test_manifest']).read_text())['episodes'];validate_manifest(rows)
    else:
        all_rows=json.loads((ROOT/'artifacts/framework/protocol/train_manifest.json').read_text())['episodes']
        rows=[next(r for r in all_rows if r['task']==task) for task in ('pick_moving_target_on_the_table','pick_moving_target_on_the_table_high_speed')]
    out=ROOT/args.out;out.mkdir(parents=True,exist_ok=True)
    records=[]
    paper=json.loads((ROOT/'config/framework/paper_reference_comparison.json').read_text())
    for row in rows:
        name=row['task']+'__'+row['raw_episode_id']
        path=out/(name+'.json')
        if path.exists():
            record=json.loads(path.read_text())
            if record['frozen_inputs_sha256']!=lock_hash or record['formal_test_used']!=official:
                raise ValueError('cannot resume results from a different frozen protocol')
        else:
            env=None
            try:
                env=(OfficialTestSimulation(row['task'],row['path'],seed=config['seed'],servo_lead=config['servo_lead']) if official else
                     LocalSimulation(row['task'],row['path'],seed=config['seed'],servo_lead=config['servo_lead']))
                # Only the first episode of each task is recorded, independent
                # of success. No outcome-based cherry-picking of videos.
                make_video=not any(r['task']==row['task'] for r in records)
                record,frames=run_episode(env,row,config,args.port,lock_hash,official=official,video=make_video)
                write(path,record)
                if frames:write_video(out/(name+'.mp4'),frames)
            except Exception as error:
                write(out/(name+'.infrastructure_error.json'),{'task':row['task'],'raw_episode_id':row['raw_episode_id'],
                    'error':repr(error),'counted_as_policy_failure':False,'frozen_inputs_sha256':lock_hash})
                raise
            finally:
                if env is not None:env.close()
        records.append(record)
        report=summarize(records,rows,paper) if official else {'formal_test_used':False,'kind':'TRAIN_home_start_development',
            'completed':len(records),'expected':len(rows),'handovers':sum(r['handovers'] for r in records),
            'successes':sum(r['official_task_success'] for r in records)}
        report.update({'created_utc':datetime.now(timezone.utc).isoformat(),'version':config['version'],
            'frozen_inputs_sha256':lock_hash,'results':[{k:v for k,v in r.items() if k not in ('steps','approach_timings')} for r in records]})
        write(out/'report.json',report)
        print(json.dumps({'completed':len(records),'expected':len(rows),'episode':name,'success':record['official_task_success'],'wall_seconds':record['wall_seconds']}),flush=True)


if __name__=='__main__':main()
