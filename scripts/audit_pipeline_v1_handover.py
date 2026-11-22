"""TRAIN-only instrumentation of frozen v1; does not modify controller behavior."""
import json
from pathlib import Path
import sys
from collections import Counter

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import numpy as np
import torch
import evaluate_pipeline_v1 as benchmark
from our_mani.capture_workflow import CaptureWorkflow
from our_mani.execution import FeasibleExecutor
from our_mani.benchmark_protocol import file_sha256,verify_lock
from our_mani.pyrep_env import LocalSimulation

GATES=[]
ACTIONS=[]


class AuditWorkflow(CaptureWorkflow):
    def step(self,observation,robot,obj,*,tick):
        previous_goal=self.goal
        decision=super().step(observation,robot,obj,tick=tick)
        goal=self.goal if self.goal is not None else previous_goal
        if goal is not None and obj is not None:
            gate=self.gate.check(goal,obj,robot,robot.timestamp)
            current_pose,velocity=goal.region.in_world(obj.pose,obj.linear_velocity,obj.angular_velocity)
            def finite(value):return float(value) if np.isfinite(value) else None
            GATES.append({'tick':tick,'time':robot.timestamp,'goal_id':goal.goal_id,
                'stage':decision.stage.value,'gate_reason':gate.reason,'allowed':gate.allowed,
                'arrival_time':goal.arrival_time,'position_metric':finite(gate.position_metric),
                'relative_speed':finite(gate.relative_speed),'orientation_error':finite(gate.orientation_error),
                'current_target_region_pose':current_pose.xyzw().tolist(),
                'selected_future_region_pose':goal.pose.xyzw().tolist(),
                'current_region_velocity':velocity.tolist(),
                'robot_velocity':robot.linear_velocity.tolist(),'robot_pose':robot.pose.xyzw().tolist(),
                'region_radii':goal.region.radii.tolist(),'maximum_relative_speed':goal.region.max_relative_speed,
                'maximum_orientation_error':goal.region.max_orientation_error})
        return decision


class AuditExecutor(FeasibleExecutor):
    def prepare_joint_limited(self,action,robot,**kwargs):
        result=super().prepare_joint_limited(action,robot,**kwargs)
        command,reason=result
        ACTIONS.append({'time':robot.timestamp,'neural_action':np.asarray(action).tolist(),
            'robot_pose':robot.pose.xyzw().tolist(),'robot_joints':robot.joints.tolist(),
            'measured_joint_velocities':robot.joint_velocities.tolist(),
            'commanded_joints':command.joints.tolist() if command else None,'reason':reason})
        return result


def main():
    torch.set_num_threads(4)
    lock_path=ROOT/'artifacts/framework/pipeline_v1/frozen_test_inputs.json'
    lock=json.loads(lock_path.read_text());verify_lock(ROOT,lock)
    config=lock['config']
    benchmark.CaptureWorkflow=AuditWorkflow
    benchmark.FeasibleExecutor=AuditExecutor
    out=ROOT/'artifacts/framework/pipeline_v1/TRAIN_handover_audit'
    out.mkdir(parents=True,exist_ok=True)
    rows=json.loads((ROOT/'artifacts/framework/protocol/train_manifest.json').read_text())['episodes']
    summaries=[]
    for task in ('pick_moving_target_on_the_table','pick_moving_target_on_the_table_high_speed'):
        row=next(r for r in rows if r['task']==task)
        GATES.clear();ACTIONS.clear()
        env=LocalSimulation(task,row['path'],seed=config['seed'],servo_lead=config['servo_lead'])
        try:
            record,_=benchmark.run_episode(env,row,config,18473,file_sha256(lock_path),official=False,video=False)
            window=[g for g in GATES if g['position_metric'] is not None]
            near=min(window,key=lambda g:g['position_metric']) if window else None
            summary={'task':task,'raw_episode_id':row['raw_episode_id'],'split':'train',
                'diagnostic_only':True,'controller_behavior_changed':False,'formal_test_used':False,
                'success':record['official_task_success'],'handovers':record['handovers'],
                'gate_reasons':dict(Counter(g['gate_reason'] for g in GATES)),
                'within_time_window':len(window),'nearest_in_time_window':near,
                'position_passes':sum(g['position_metric']<=1 for g in window),
                'velocity_passes':sum(g['relative_speed']<=g['maximum_relative_speed'] for g in window),
                'orientation_passes':sum(g['orientation_error']<=g['maximum_orientation_error'] for g in window)}
            benchmark.write(out/(task+'.json'),{'summary':summary,'gates':GATES,'actions':ACTIONS,'rollout':record})
            summaries.append(summary)
            print(json.dumps(summary,ensure_ascii=False),flush=True)
        finally:env.close()
    benchmark.write(out/'report.json',{'formal_test_used':False,'scope':'TRAIN only, frozen v1 diagnostic','summaries':summaries})


if __name__=='__main__':main()
