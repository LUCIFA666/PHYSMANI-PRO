"""Model-only fixed TRAIN tracking response check, no physical success claim."""
from copy import deepcopy
from pathlib import Path
import argparse,json,sys
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.compare_highspeed_reference_20261111_v1 import FIXED,FAMILY
from our_mani.contracts import Forecast
from our_mani.local_mpc_20261102.executor import JointLimits
from our_mani.local_mpc_20261102.records import write_new
from our_mani.local_mpc_control_20261103.response_factory import build_planner
from our_mani.local_mpc_control_20261103.free_phase_dwell_v72 import update_phase
from our_mani.local_mpc_20261102.mpc import validate_candidate
from our_mani.highspeed_control_20261111_v1.tracking import TrackingReferencePlanner


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    results=[]
    for seed,index in FIXED:
        case=FAMILY/f'Cube_native_generated_high_speed_seed{seed}'
        tr=next(r for r in map(json.loads,(case/'transitions.jsonl').read_text().splitlines()) if r['index']==index)
        state=deepcopy(tr['before']);state={k:np.asarray(v) if isinstance(v,list) and k not in ('phase_state','actuator_state') else v for k,v in state.items()}
        native=json.loads((case/'initial_snapshot_PRIVATE.json').read_text())['joint_limits']
        limits=JointLimits(native['lower'],native['upper'],native['velocity'])
        core,helper,_=build_planner(ROOT,limits,controller='coupled',action_space='joint',joint_goal_config={'lower':(-.1,-.65,.755)})
        planner=TrackingReferencePlanner(helper.kinematics,core.execution_model.transition_batch.joint_response,limits)
        candidate=validate_candidate(next(c for c in tr['candidates'] if c['id']==tr['plan']['selected_candidate_id']))
        trace=tr['private_oracle']['trajectory_trace']
        forecast=Forecast(('target',),state['time']+np.arange(17)*.05,np.asarray([state['object_position']]+[r['position'] for r in trace])[:,None],np.tile(np.eye(3),(17,1,1,1)),np.asarray([state['object_linear_velocity']]+[r['nominal50ms_difference_velocity'] for r in trace])[:,None],np.zeros((17,1,3)))
        rows=[];phase=state.get('phase_state')
        for k in range(16):
            phase=update_phase(state,candidate,phase,core.config)
            result=planner.plan(state,candidate,forecast,pregrasp=phase['phase']=='approach')
            if result['action'] is None:break
            row=dict(time=state['time'],phase=deepcopy(phase),predicted_linear_velocity=result['predicted_linear_velocity'],target_velocity=result['target_velocity'])
            rows.append(row)
            previous_R=Rotation.from_quat(state['tcp_quaternion']).as_matrix()
            state.update(q=result['predicted_q'],dq=result['predicted_dq'],tcp_position=result['predicted_TCP'][:3,3],tcp_quaternion=Rotation.from_matrix(result['predicted_TCP'][:3,:3]).as_quat(),tcp_velocity=result['predicted_linear_velocity'],tcp_angular_velocity=Rotation.from_matrix(result['predicted_TCP'][:3,:3]@previous_R.T).as_rotvec()/.05,time=state['time']+.05)
            pose,v,w=forecast.at('target',state['time'])
            state.update(object_position=pose.position,object_quaternion=pose.xyzw()[3:],object_linear_velocity=v,object_angular_velocity=w)
            planner.last_applied=dict(time=state['time'],**result['reference'])
        values=[r['phase']['metrics'] for r in rows]
        result=dict(seed=seed,rows=rows,minimum_position=min(v['distance'] for v in values),any_joint_gates=any(v['distance']<=.025 and v['angle']<=.2 and v['relative_speed']<=.14 for v in values),first_close=next((r['time'] for r in rows if r['phase']['phase']=='close'),None))
        results.append(result);print(json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)
    write_new(a.output/'report.json',dict(complete=True,results=results,Native_starts=0,input_scope='private_TRAIN_model_diagnostic_only',task_success_claimed=False,formal_VAL_TEST=False))

if __name__=='__main__':main()
