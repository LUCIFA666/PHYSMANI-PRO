"""Collect actual global-approach candidate outcomes on TRAIN/VAL only."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import random
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import numpy as np
import torch
import evaluate_aligned_r05_v2 as runner
from our_mani.risk_v2.selector import AttemptInterception,attempt_controller_hash
from our_mani.sensor_alignment.author_rgbd import AlignedLocalSimulation,public_initial_conditions
from our_mani.benchmark_protocol import file_sha256,verify_lock
TASKS=('pick_moving_target_on_the_table','pick_moving_target_on_the_table_high_speed')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--predictor',choices=('cv','context'),required=True)
    p.add_argument('--split',choices=('train','validation'),required=True)
    p.add_argument('--task',choices=TASKS,required=True)
    p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1)
    p.add_argument('--replicas',type=int,default=3)
    p.add_argument('--episodes',type=int,default=0)
    p.add_argument('--out',required=True);args=p.parse_args()
    if not 0<=args.shard<args.shards:raise ValueError('invalid shard')
    torch.set_num_threads(2)
    config=json.loads((ROOT/'config/framework/aligned_protocol_r07.json').read_text())
    rows=[r for r in json.loads((ROOT/f'artifacts/framework/protocol/{args.split}_manifest.json').read_text())['episodes'] if r['task']==args.task]
    if args.episodes:rows=rows[:args.episodes]
    rows=rows[args.shard::args.shards]
    out=ROOT/args.out;out.mkdir(parents=True,exist_ok=True)
    source_files=[Path(__file__).resolve(),ROOT/'our_mani/risk_v2/selector.py',ROOT/'scripts/evaluate_aligned_r05_v2.py',ROOT/'config/framework/aligned_protocol_r07.json',ROOT/'our_mani/sensor_alignment/author_rgbd.py']
    files={str(f.relative_to(ROOT)):file_sha256(f) for f in source_files}
    identity=hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()
    runner.CandidateInterception=AttemptInterception
    runner.AlignedLocalSimulation=AlignedLocalSimulation
    runner.public_initial_conditions=public_initial_conditions
    records=[];env=None
    try:
        for row in rows:
            for replica in range(args.replicas):
                name=row['task']+'__'+row['raw_episode_id']+f'__{replica}'
                path=out/(name+'.json')
                if path.exists():
                    record=json.loads(path.read_text())
                    if record['inputs_sha256']!=identity:raise ValueError('collection inputs changed')
                else:
                    if (ROOT/'runtime/framework/success_r07_STOP').exists():raise InterruptedError('r07 collection stopped')
                    seed=int.from_bytes(hashlib.sha256(f"r07:{args.split}:{name}".encode()).digest()[:4],'little')
                    np.random.seed(seed);random.seed(seed);torch.manual_seed(seed)
                    if env is None:env=runner.AlignedLocalSimulation(row['task'],row['path'],protocol=config,seed=config['seed'],servo_lead=config['local_servo_lead'])
                    sink=[];settings=dict(config,_workflow_sink=sink,_episode_task=row['task'],_collection_seed=seed,
                        _exploration=('geometry','softmax','uniform')[replica%3])
                    record,_=runner.run_episode(env,row,settings,predictor=args.predictor,ranking='geometry',official=False,video=False)
                    for workflow in sink:workflow.close_attempt(record['official_task_success'],'native_task_success' if record['official_task_success'] else 'rollout_ended_without_task_success',env.time)
                    attempts=[a for w in sink for a in w.attempts]
                    record.pop('steps',None)
                    record.update(version='candidate_success_r07_actual_approach',split=args.split,source_episode=row['raw_episode_id'],
                        attempt_controller_sha256=attempt_controller_hash(ROOT,config,args.predictor),
                        sensor_domain='author_OPENGL',replica=replica,exploration=settings['_exploration'],seed=seed,formal_test_used=False,
                        attempts=attempts,inputs_sha256=identity,collection_sources=files,
                        probability_scope='native task success on the selected attempt before candidate replacement, with actual global approach and remaining episode budget')
                    runner.write(path,record)
                records.append(record)
                all_attempts=[a for r in records for a in r['attempts']]
                runner.write(out/'report.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),completed=len(records),
                    expected=len(rows)*args.replicas,complete=len(records)==len(rows)*args.replicas,
                    attempts=len(all_attempts),positive=sum(a['official_task_success'] for a in all_attempts),
                    source_episodes=len({r['source_episode'] for r in records}),split=args.split,task=args.task,predictor=args.predictor,
                    formal_test_used=False,inputs_sha256=identity))
                print(json.dumps(dict(case=name,success=record['official_task_success'],attempts=len(record['attempts']),completed=len(records))),flush=True)
    finally:
        if env is not None:env.close()

if __name__=='__main__':main()
