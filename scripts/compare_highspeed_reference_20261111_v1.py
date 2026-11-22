"""Real P2 static/dynamic comparison at four fixed TRAIN-only oracle anchors."""
from copy import deepcopy
from pathlib import Path
import argparse
import json
import sys
import time
import numpy as np
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from our_mani.contracts import Forecast
from our_mani.local_mpc_20261102.executor import JointLimits
from our_mani.local_mpc_20261102.records import write_new
from our_mani.local_mpc_control_20261103.response_factory import build_planner
from our_mani.local_mpc_control_20261103.forecast_control import ForecastJointGoalCEMPlanner,BoundForecastExecutionModel
from our_mani.cube_dynamic_reference_20261104_v1.planner import DynamicReferenceForecastJointGoalCEMPlannerV1

FIXED=((2611042100,36),(2611042101,41),(2611042102,67),(2611042103,35))
FAMILY=ROOT/'runtime/framework/cube_exact_future_teacher_TRAIN_20261104_v2'


def metrics(result,candidates,forecast):
    if result['action'] is None:return dict(action_available=False,diagnostics=result['diagnostics'])
    candidate=next(c for c in candidates if c['id']==result['selected_candidate_id'])
    values=[]
    for s in result['predicted_states'][1:]:
        pose,v,w=forecast.at('target',s['time'])
        desired=pose.apply(np.asarray(candidate['object_tcp_position']))
        desired_R=pose.rotation@Rotation.from_quat(candidate['object_tcp_quaternion']).as_matrix()
        distance=float(np.linalg.norm(s['tcp_position']-desired))
        angle=float(np.linalg.norm(Rotation.from_matrix(desired_R@Rotation.from_quat(s['tcp_quaternion']).as_matrix().T).as_rotvec()))
        speed=float(np.linalg.norm(s['tcp_velocity']-v))
        values.append(dict(time=s['time'],position_m=distance,angle_rad=angle,relative_speed_mps=speed,
            all_pose_speed_gates=distance<=.025 and angle<=.2 and speed<=candidate['allowed_relative_speed']))
    return dict(action_available=True,cost=result['cost'],selected_candidate=candidate['id'],
        first_action=result['action'],minimum_position=min(values,key=lambda x:x['position_m']),
        any_joint_pose_speed_gates=any(v['all_pose_speed_gates'] for v in values),endpoints=values,
        first_predicted_close=next((i for i,a in enumerate(result['sequence']) if a['opening']==0.),None),
        wall_seconds=result['diagnostics']['wall_seconds'])


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    results=[]
    for seed,index in FIXED:
        case=FAMILY/f'Cube_native_generated_high_speed_seed{seed}'
        tr=next(v for v in map(json.loads,(case/'transitions.jsonl').read_text().splitlines()) if v['index']==index)
        state=deepcopy(tr['before']);candidates=deepcopy(tr['candidates'])
        native=json.loads((case/'initial_snapshot_PRIVATE.json').read_text())['joint_limits']
        limits=JointLimits(native['lower'],native['upper'],native['velocity'])
        core,helper,provenance=build_planner(ROOT,limits,controller='coupled',action_space='joint',joint_goal_config={'lower':(-.1,-.65,.755)})
        trace=tr['private_oracle']['trajectory_trace']
        forecast=Forecast(('target',),state['time']+np.arange(17)*.05,
            np.asarray([state['object_position']]+[r['position'] for r in trace])[:,None],
            np.tile(np.eye(3),(17,1,1,1)),
            np.asarray([state['object_linear_velocity']]+[r['nominal50ms_difference_velocity'] for r in trace])[:,None],np.zeros((17,1,3)))
        row=dict(seed=seed,transition_index=index,time=state['time'])
        for name,cls in [('static',ForecastJointGoalCEMPlanner),('dynamic',DynamicReferenceForecastJointGoalCEMPlannerV1)]:
            bound=BoundForecastExecutionModel(core.execution_model,forecast,anchor_time=state['time'])
            planner=cls(core.config,execution_model=bound,goal_helper=helper,constraint_cost=core.planner.constraint_cost,terminal_config=core.terminal_config)
            row[name]=metrics(planner.plan(state,candidates),candidates,forecast)
        results.append(row)
        print(json.dumps({k:row[k] for k in ('seed','time')}|{k:{n:row[k].get(n) for n in ('action_available','minimum_position','any_joint_pose_speed_gates','first_predicted_close','wall_seconds')} for k in ('static','dynamic')},ensure_ascii=False),flush=True)
    write_new(args.output/'report.json',dict(complete=True,results=results,Native_starts=0,
        input_scope='fixed_TRAIN_private_oracle_diagnostic_not_public_policy_score',formal_VAL_TEST=False,
        success_claimed=False,limits_and_gates_unchanged=True,real_P2_factory_used=True))
    return 0


if __name__=='__main__':raise SystemExit(main())
