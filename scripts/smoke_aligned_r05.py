"""Native TRAIN-only checks: clock parity, moving target, grasp calls and probes."""
from datetime import datetime, timezone
import argparse
import json
from pathlib import Path
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT),str(ROOT/'scripts')]
import numpy as np
from our_mani.aligned.simulation import AlignedLocalSimulation, physics_time
from our_mani.benchmark_perception import RedBoxTracker
from collect_local_rollouts import initialization,TASKS

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--task',choices=TASKS);args=parser.parse_args()
    base=ROOT/'artifacts/framework/aligned_r05/smoke_protocol_v2'
    if not args.task:
        for task in TASKS:
            subprocess.run([sys.executable,*sys.argv,'--task',task],check=True)
        reports=[json.loads((base/(task+'.json')).read_text()) for task in TASKS]
        (base/'report.json').write_text(json.dumps(dict(passed=all(r['passed'] for r in reports),formal_test_used=False,results=reports),ensure_ascii=False,indent=2)+'\n')
        return
    protocol=json.loads((ROOT/'config/framework/aligned_protocol_r05.json').read_text())
    rows=json.loads((ROOT/'artifacts/framework/protocol/train_manifest.json').read_text())['episodes']
    reports=[]
    out=base;out.mkdir(parents=True,exist_ok=True)
    for task in (args.task,):
        row=next(row for row in rows if row['task']==task)
        env=AlignedLocalSimulation(task,row['path'],protocol=protocol,seed=421)
        try:
            x,v,a,goal=initialization(row['path'])
            assert env.reset(target_position=x,target_velocity=v,target_acceleration=a,goal_position=goal)
            assert env.arm.get_handle() not in {shape.get_handle() for shape in env.map_service.static}
            assert env.gripper.get_handle() not in {shape.get_handle() for shape in env.map_service.static}
            from scipy.spatial import cKDTree
            for source,ghost in zip(env.map_service.static,env.map_service.ghosts):
                assert not ghost.get_objects_in_tree(exclude_base=True)
                assert not ghost.is_dynamic() and not ghost.is_respondable() and not ghost.is_collidable()
                source_vertices=source.get_mesh_data()[0]
                ghost_vertices=ghost.get_mesh_data()[0]
                source_transform,ghost_transform=source.get_matrix(),ghost.get_matrix()
                source_world=np.asarray(source_vertices)@source_transform[:3,:3].T+source_transform[:3,3]
                ghost_world=np.asarray(ghost_vertices)@ghost_transform[:3,:3].T+ghost_transform[:3,3]
                assert np.max(cKDTree(source_world).query(ghost_world)[0])<1e-5
            q=np.asarray(env.arm.get_joint_positions()); tracker=RedBoxTracker()
            observation,_=env.observe();obj,_=tracker.observe(observation)
            from pyrep.objects.vision_sensor import VisionSensor
            assert set(observation['observation/instance_masks'])==set(env.cameras)
            for name,camera in env.cameras.items():
                depth=observation['observation/depths'][name]
                expected=VisionSensor.pointcloud_from_depth_and_camera_params(depth,camera.get_matrix(),camera.get_intrinsic_matrix())
                actual=observation['observation/calibrations'][name].backproject(depth,np.ones(depth.shape,dtype=bool)).reshape(*depth.shape,3)
                assert np.allclose(actual,expected,atol=1e-6)
            actual=np.asarray(env.task.target_block.get_pose());before=physics_time()
            for _ in range(4):
                env.map_service.check(obj.pose)
                env.map_service.check()
            assert np.allclose(env.task.target_block.get_pose(),actual,atol=1e-7)
            assert np.allclose(env.arm.get_joint_positions(),q,atol=1e-7)
            assert abs(physics_time()-before)<1e-7
            answer=env.map_service.check(obj.pose)
            env.task.target_block.set_position([.45,.45,.775],reset_dynamics=False)
            assert env.map_service.check(obj.pose)==answer
            env.task.target_block.set_pose(actual.tolist(),reset_dynamics=False)
            positions=[]
            for tick in range(8):
                env.command_joints(q,opening=0. if 2<=tick<6 else 1.,phase='gripper_protocol_check')
                positions.append(env.metric_target_position().tolist())
            receipt=env.alignment_receipt()
            assert receipt['clock']['grasp_requests']==1
            assert receipt['clock']['release_requests']==1
            assert abs(receipt['actual_physics_seconds']-.4)<2e-5
            assert abs(receipt['actual_task_seconds']-.4)<2e-5
            assert np.linalg.norm(np.asarray(positions[5])-positions[2])>1e-4
            reports.append(dict(task=task,source=row['raw_episode_id'],passed=True,
                                target_positions_metric_only=positions,receipt=receipt))
            print(json.dumps(reports[-1]),flush=True)
        finally:
            env.close()
    report=dict(created_utc=datetime.now(timezone.utc).isoformat(),passed=True,
                formal_test_used=False,split='train',results=reports)
    (out/(args.task+'.json')).write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')

if __name__=='__main__':main()
