"""Check existing dynamics on causal RGB-D histories in the aligned VAL environment."""
import json
import argparse
import subprocess
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import numpy as np
import torch
from collect_local_rollouts import initialization,TASKS
from our_mani.aligned.simulation import AlignedLocalSimulation
from our_mani.benchmark_perception import RedBoxTracker
from our_mani.context_dynamics import ContextLatentDynamics
from our_mani.dynamics import ConstantVelocityDynamics

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--task',choices=TASKS);args=parser.parse_args()
    if not args.task:
        base=ROOT/'artifacts/framework/aligned_r05/dynamics_validation'
        for task in TASKS:
            subprocess.run([sys.executable,*sys.argv,'--task',task],check=True)
        cases=[case for task in TASKS for case in json.loads((base/(task+'.json')).read_text())['cases']]
        report=dict(completed=len(cases),expected=40,complete=len(cases)==40,formal_test_used=False,split='validation',unchanged_history_dt=.05,cases=cases,
            metrics={name:{str(h):dict(n=len(values:=[e['error_m'] for case in cases for e in case['errors'] if e['predictor']==name and e['horizon']==h]),endpoint_mean_mm=float(np.mean(values)*1000) if values else None) for h in (.25,.5,1.1)} for name in ('cv','context')})
        (base/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');return
    torch.set_num_threads(2)
    config=json.loads((ROOT/'config/framework/aligned_protocol_r05.json').read_text())
    rows=[r for r in json.loads((ROOT/'artifacts/framework/protocol/validation_manifest.json').read_text())['episodes'] if r['task'] in TASKS]
    out=ROOT/'artifacts/framework/aligned_r05/dynamics_validation';out.mkdir(parents=True,exist_ok=True)
    cases=[]
    for task in (args.task,):
        selected=[r for r in rows if r['task']==task]
        env=AlignedLocalSimulation(task,selected[0]['path'],protocol=config,seed=421)
        try:
            for row in selected:
                x,v,a,goal=initialization(row['path'])
                if not env.reset(target_position=x,target_velocity=v,target_acceleration=a,goal_position=goal):
                    raise RuntimeError('VAL source reset failed')
                env.finish_home_reset()
                tracker=RedBoxTracker();context=ContextLatentDynamics(ROOT/config['dynamics_checkpoint']);cv=ConstantVelocityDynamics()
                truth=[];predictions=[];tracking_failures=0
                for tick in range(65):
                    observation,robot=env.observe()
                    truth.append(env.metric_target_position().copy())
                    try:
                        obj,_=tracker.observe(observation);context.observe([obj])
                        if tick in (20,40):
                            forecasts={name:model.predict([obj],1.1,.05) for name,model in (('context',context),('cv',cv))}
                            predictions.append((tick,{name:[forecast.at('target',obj.timestamp+horizon)[0].position.tolist() for horizon in (.25,.5,1.1)] for name,forecast in forecasts.items()}))
                    except ValueError:
                        tracking_failures+=1
                    env.hold_tick()
                errors=[]
                for tick,predictions_at_tick in predictions:
                    for name,values in predictions_at_tick.items():
                        for horizon,position in zip((.25,.5,1.1),values):
                            endpoint=truth[tick+round(horizon/.05)]
                            errors.append(dict(predictor=name,horizon=horizon,error_m=float(np.linalg.norm(np.asarray(position)-endpoint))))
                cases.append(dict(task=task,source_episode=row['raw_episode_id'],errors=errors,
                                  tracking_failures=tracking_failures,alignment_receipt=env.alignment_receipt()))
                report=dict(completed=len(cases),expected=40,complete=len(cases)==40,formal_test_used=False,
                    split='validation',checkpoint=config['dynamics_checkpoint'],unchanged_history_dt=.05,
                    metrics={name:{str(h):dict(n=len(values:=[e['error_m'] for case in cases for e in case['errors'] if e['predictor']==name and e['horizon']==h]),
                        endpoint_mean_mm=float(np.mean(values)*1000) if values else None) for h in (.25,.5,1.1)} for name in ('cv','context')},
                    cases=cases)
                tmp=out/'report.tmp';tmp.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');tmp.replace(out/(task+'.json'))
                print(json.dumps(dict(task=task,source=row['raw_episode_id'],completed=len(cases))),flush=True)
        finally:env.close()

if __name__=='__main__':main()
