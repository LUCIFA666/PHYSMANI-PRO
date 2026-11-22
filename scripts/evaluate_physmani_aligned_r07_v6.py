"""Local released PhysMani weights, r07 common author-domain RGB-D conditions, no expert actions."""
import argparse
import fcntl
from datetime import datetime,timezone
import gc
import hashlib
import json
from pathlib import Path
import random
import sys
import subprocess
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import numpy as np
import torch
from collect_local_rollouts import initialization,TASKS
from our_mani.sensor_alignment.author_rgbd import (
    AlignedOfficialSimulation, AlignedLocalSimulation, SensorProtocolInvariantError)
from our_mani.sensor_alignment.author_rgbd import public_initial_conditions
from our_mani.aligned.physmani import build_policy
from our_mani.sensor_alignment.physmani_executor_v6 import (
    PhysManiAdapter, ProtocolInvariantError, InfrastructureExecutionError, raise_protected_error)
from our_mani.benchmark_protocol import file_sha256

def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');tmp.replace(path)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='config/framework/aligned_protocol_r07_baseline_v6.json')
    parser.add_argument('--split',choices=('train','validation','test'),default='train')
    parser.add_argument('--episodes-per-task',type=int,default=1)
    parser.add_argument('--out',required=True)
    parser.add_argument('--init-only',action='store_true')
    parser.add_argument('--one-decision',action='store_true')
    parser.add_argument('--task',choices=TASKS)
    args=parser.parse_args()
    if not args.task:
        for task in TASKS:
            subprocess.run([sys.executable,*sys.argv,'--task',task],check=True)
        out=ROOT/args.out
        records=[json.loads(p.read_text()) for p in sorted(out.glob('pick*__episode*.json')) if not p.name.endswith('infrastructure_error.json')]
        expected=2*(args.episodes_per_task or (40 if args.split=='test' else 20 if args.split=='validation' else 100))
        write(out/'report.json',dict(completed=len(records),expected=expected,complete=len(records)==expected and not args.one_decision,formal_test_used=args.split=='test',per_task={t:dict(completed=sum(r['task']==t for r in records),successes=sum(r['task']==t and r['official_task_success'] for r in records)) for t in TASKS}))
        return
    config=json.loads((ROOT/args.config).read_text())
    out=ROOT/args.out;out.mkdir(parents=True,exist_ok=True)
    run_lock=(out/(args.task+'.lock')).open('a')
    fcntl.flock(run_lock,fcntl.LOCK_EX)
    files=[ROOT/args.config,ROOT/'our_mani/ablations/physmani_executor_v4.py',ROOT/'our_mani/sensor_alignment/physmani_executor_v5.py',ROOT/'our_mani/sensor_alignment/physmani_executor_v6.py',ROOT/'config/framework/aligned_protocol_r07.json',ROOT/'our_mani/sensor_alignment/author_rgbd.py',ROOT/'our_mani/sensor_alignment/__init__.py',Path(__file__).resolve(),*[ROOT/'our_mani/aligned'/name for name in ('protocol.py','simulation.py','physmani.py','audit.py')],ROOT/'our_mani/official_simulation.py',ROOT/'our_mani/pyrep_env.py']
    identity=dict(config=config,files={str(p.relative_to(ROOT)):file_sha256(p) for p in files})
    identity['sha256']=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    identity_path=out/(args.task+'_inputs.json')
    if identity_path.exists() and json.loads(identity_path.read_text())['sha256']!=identity['sha256']:
        raise ValueError('baseline protocol/source changed; choose a new output directory')
    write(identity_path,identity)
    manifest=ROOT/('artifacts/framework/pipeline_v1/official_test_manifest.json' if args.split=='test' else f'artifacts/framework/protocol/{args.split}_manifest.json')
    rows=json.loads(manifest.read_text())['episodes']
    tasks=(args.task,) if args.task else TASKS
    records=[]
    torch.set_num_threads(4)
    checkpoint_hash=file_sha256(Path(config['baseline_checkpoint']))
    for task in tasks:
        selected=[r for r in rows if r['task']==task][:args.episodes_per_task or None]
        torch.manual_seed(config['seed']);np.random.seed(config['seed']);random.seed(config['seed'])
        print('Loading released policy',task,flush=True)
        model=build_policy(ROOT,config,task)
        print('Released policy strict weights loaded',flush=True)
        if args.init_only:
            write(out/'report.json',dict(passed=True,stage='strict_policy_weights_loaded',checkpoint_sha256=checkpoint_hash,formal_test_used=False))
            return
        env=None
        for row in selected:
            name=task+'__'+row['raw_episode_id'];path=out/(name+'.json')
            if path.exists():
                record=json.loads(path.read_text())
                if record['checkpoint_sha256']!=checkpoint_hash:raise ValueError('resume checkpoint differs')
                records.append(record);continue
            adapter=None;start=time.perf_counter()
            try:
                torch.manual_seed(config['seed']);np.random.seed(config['seed']);random.seed(config['seed'])
                official=args.split=='test'
                cls=AlignedOfficialSimulation if official else AlignedLocalSimulation
                if env is None:env=cls(task,row['path'],protocol=config,seed=config['seed'],servo_lead=1.)
                if official:
                    env.reset_official(row)
                else:
                    x,v,a,goal=initialization(row['path'])
                    if not env.reset(target_position=x,target_velocity=v,target_acceleration=a,goal_position=goal):
                        raise RuntimeError('development reset IK failed')
                    env.finish_home_reset()
                observation,robot=env.observe()
                initial_conditions=public_initial_conditions(env,observation,robot)
                variation=int(row.get('variation',0))
                adapter=PhysManiAdapter(ROOT,config,task,model)
                print('Initializing online world model',task,row['raw_episode_id'],flush=True)
                adapter.reset(env,task,variation,observation,robot)
                print('Online world model initialized',flush=True)
                steps=[];policy_failure=None
                while env.ledger.remaining>=env.ledger.dt-1e-8 and not env._native_success_latched:
                    if (ROOT/'runtime/framework/aligned_r07_STOP').exists() or (ROOT/'runtime/framework/success_r07_STOP').exists():raise InterruptedError('r07 stop requested')
                    observation,robot=env.observe()
                    try:
                        action=adapter.predict(observation,robot)
                        details=adapter.execute(env,action)
                    except (ProtocolInvariantError,InfrastructureExecutionError,SensorProtocolInvariantError):
                        raise
                    except (ValueError,RuntimeError,AssertionError) as exc:
                        raise_protected_error(exc)
                        if any(token in str(exc).lower() for token in ('out of memory','illegal memory','device-side assert','cuda error','compilation')):
                            raise
                        policy_failure=repr(exc)
                        print('Policy inference/execution failure counted as task failure:',policy_failure,flush=True)
                        break
                    steps.append(dict(time=robot.timestamp,action=action.tolist(),execution=details))
                    print(json.dumps(dict(task=task,episode=row['raw_episode_id'],time=env.time,decisions=adapter.decisions,execution=details)),flush=True)
                    if args.one_decision:break
                record=dict(created_utc=datetime.now(timezone.utc).isoformat(),task=task,raw_episode_id=row['raw_episode_id'],
                    split=args.split,formal_test_used=official,official_task_success=bool(env._native_success_latched),
                    method='PhysMani_released_weights_released_waypoints_counted_actuator_author_rgbd_v6',executor_revision=config['baseline_executor'],checkpoint_sha256=checkpoint_hash,
                    public_initial_conditions=initial_conditions,alignment_receipt=env.alignment_receipt(),reset_receipt=env.reset_receipt if official else None,
                    compute_seconds=adapter.compute_seconds,wall_seconds=time.perf_counter()-start,
                    one_decision_diagnostic=args.one_decision,policy_failure=policy_failure,steps=steps)
                write(path,record);records.append(record)
                write(out/'report.json',dict(completed=len(records),expected=len(tasks)*(args.episodes_per_task or (40 if args.split=='test' else 20 if args.split=='validation' else 100)),
                    complete=not args.one_decision and len(records)==sum(len([r for r in rows if r['task']==t][:args.episodes_per_task or None]) for t in tasks),
                    formal_test_used=official,per_task={t:dict(completed=sum(r['task']==t for r in records),successes=sum(r['task']==t and r['official_task_success'] for r in records)) for t in tasks}))
            except Exception as exc:
                write(out/(name+'.infrastructure_error.json'),dict(error=repr(exc),formal_test_used=args.split=='test'))
                raise
            finally:
                if env and (row==selected[-1] or sys.exc_info()[0] is not None):env.close();env=None
                del adapter;gc.collect();torch.cuda.empty_cache()
        del model;gc.collect();torch.cuda.empty_cache()

if __name__=='__main__':main()
