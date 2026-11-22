"""TRAIN-only diagnostic of occupancy/self filtering, no outcome-based tuning."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import numpy as np
from collect_local_rollouts import initialization
from our_mani.aligned.observed_map import ObservedMapLocalSimulation

def main():
    from pyrep.const import ObjectType
    config=json.loads((ROOT/'config/framework/aligned_protocol_r05.json').read_text())
    row=next(r for r in json.loads((ROOT/'artifacts/framework/protocol/train_manifest.json').read_text())['episodes'] if r['task']=='pick_moving_target_on_the_table')
    env=ObservedMapLocalSimulation(row['task'],row['path'],protocol=config,seed=0)
    try:
        x,v,a,goal=initialization(row['path']);assert env.reset(target_position=x,target_velocity=v,target_acceleration=a,goal_position=goal)
        env.finish_home_reset();observation,robot=env.observe()
        shapes=env.arm.get_objects_in_tree(object_type=ObjectType.SHAPE)+env.gripper.get_objects_in_tree(object_type=ObjectType.SHAPE)
        ids={shape.get_name():shape.get_handle() for shape in shapes}
        counts={}
        for name,mask in observation['observation/instance_masks'].items():
            rgb=np.rint(mask*255).astype(np.int64);code=rgb[...,0]+rgb[...,1]*256+rgb[...,2]*65536
            values,n=np.unique(code,return_counts=True);counts[name]=dict(sorted(zip(values.tolist(),n.tolist()),key=lambda r:-r[1])[:30])
        native=bool(env.arm.check_arm_collision())
        observed=env.map_service.check()
        print(json.dumps(dict(robot_ids=ids,mask_ids=counts,native_current_collision=native,
            observed_map_current_collision=observed,map=env.map_service.receipt()),indent=2),flush=True)
    finally:env.close()

if __name__=='__main__':main()
