"""TRAIN kinematic parity with the author's waypoint extraction, no policy tuning."""
import json
from pathlib import Path
import random
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import numpy as np
from collect_local_rollouts import initialization
from our_mani.aligned.simulation import AlignedLocalSimulation,physics_time
from our_mani.ablations.physmani_executor_v3 import path_poses_without_physics,released_waypoint_indices


def main():
    from physmani_bench.env import get_action_traj
    from pyrep.robots.configuration_paths.arm_configuration_path import ArmConfigurationPath
    config=json.loads((ROOT/'config/framework/aligned_protocol_r05_baseline_v3.json').read_text())
    row=next(r for r in json.loads((ROOT/'artifacts/framework/protocol/train_manifest.json').read_text())['episodes'] if r['task']=='pick_moving_target_on_the_table')
    np.random.seed(0);random.seed(0)
    env=AlignedLocalSimulation(row['task'],row['path'],protocol=config,seed=0)
    results=[]
    try:
        for joint_index,offset in ((0,.3),(1,.2),(2,-.3),(3,.2)):
            x,v,a,goal=initialization(row['path']);assert env.reset(target_position=x,target_velocity=v,target_acceleration=a,goal_position=goal)
            env.finish_home_reset()
            initial=np.asarray(env.arm.get_joint_positions())
            final=initial.copy();final[joint_index]+=offset
            path=ArmConfigurationPath(env.arm,np.linspace(initial,final,100).reshape(-1))
            before=physics_time()
            poses,joints=path_poses_without_physics(path)
            after=physics_time()
            author_poses,author_joints=get_action_traj(path)
            new_indices=released_waypoint_indices(poses)
            original_indices=released_waypoint_indices(author_poses)
            position_delta=np.max(np.linalg.norm(poses[:,:3]-author_poses[:,:3],axis=1))
            item=dict(joint_index=joint_index,offset=offset,waypoints=len(poses),no_physics_advance=before==after,
                joints_match=bool(np.allclose(joints,author_joints,rtol=0,atol=1e-7)),
                max_cartesian_position_delta_m=float(position_delta),selection_identical=new_indices==original_indices,
                selected_waypoints=len(new_indices))
            item['passed']=bool(item['no_physics_advance'] and item['joints_match'] and item['selection_identical'] and position_delta<1e-5)
            results.append(item);print(json.dumps(item),flush=True)
        out=ROOT/'artifacts/framework/aligned_r05/baseline_executor_v3_probe';out.mkdir(parents=True,exist_ok=True)
        report=dict(passed=all(r['passed'] for r in results),formal_test_used=False,
            source='released get_action_traj + modify_path rule on TRAIN kinematic probes',
            preserved=dict(ignore_collisions=True,initial_discard=10,cartesian_spacing_m=.01,arm_ticks=9,gripper_ticks=1),results=results)
        (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        if not report['passed']:raise RuntimeError('released waypoint parity failed')
    finally:env.close()

if __name__=='__main__':main()
