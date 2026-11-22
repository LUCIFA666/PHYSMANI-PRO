"""Record a TRAIN initial frame to diagnose render/reset parity, no policy rollout."""
import argparse
import json
from pathlib import Path
import random
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import numpy as np
from collect_local_rollouts import initialization
from our_mani.aligned.simulation import AlignedLocalSimulation
from our_mani.aligned.audit import public_initial_conditions

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--name',required=True);args=parser.parse_args()
    config=json.loads((ROOT/'config/framework/aligned_protocol_r05.json').read_text())
    row=next(r for r in json.loads((ROOT/'artifacts/framework/protocol/train_manifest.json').read_text())['episodes'] if r['task']=='pick_moving_target_on_the_table')
    np.random.seed(0);random.seed(0)
    env=AlignedLocalSimulation(row['task'],row['path'],protocol=config,seed=0)
    try:
        x,v,a,goal=initialization(row['path']);assert env.reset(target_position=x,target_velocity=v,target_acceleration=a,goal_position=goal)
        env.finish_home_reset();observation,robot=env.observe()
        out=ROOT/'artifacts/framework/aligned_r05/initial_sensor_probe';out.mkdir(parents=True,exist_ok=True)
        arrays={name+'_'+cam:array for name,key in [('rgb','observation/images'),('depth','observation/depths')] for cam,array in observation[key].items()}
        np.savez_compressed(out/(args.name+'.npz'),**arrays)
        record=public_initial_conditions(env,observation,robot)
        record['target_pose_metric_only']=np.asarray(env.task.target_block.get_pose()).tolist()
        (out/(args.name+'.json')).write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
    finally:env.close()

if __name__=='__main__':main()
