"""Collect controller-matched CV outcomes and fit the no-context Value baseline."""
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/framework/pipeline_v2'


def status(stage,**extra):
    OUT.mkdir(parents=True,exist_ok=True)
    p=OUT/'value_training_status.json'
    q=p.with_suffix('.tmp')
    q.write_text(json.dumps({'created_utc':datetime.now(timezone.utc).isoformat(),'stage':stage,'formal_test_used':False,**extra},indent=2)+'\n');q.replace(p)


def run(stage,command):
    status(stage)
    with (ROOT/f'logs/framework/{stage}.log').open('a') as log:
        subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)


def main():
    try:
        common=['bash','scripts/run_local_sim.sh','scripts/collect_local_rollouts.py','--replicas','4','--full-task']
        run('pipeline_v2_CV_local_TRAIN',[*common,'--split','train','--episodes-per-task','24','--name','pipeline_v2_cv_train_smoke'])
        run('pipeline_v2_CV_local_VAL',[*common,'--split','validation','--episodes-per-task','20','--name','pipeline_v2_cv_validation_smoke'])
        run('pipeline_v2_CV_success_fit',[sys.executable,'scripts/train_success_predictor.py',
            '--data','artifacts/framework/local_policy/pipeline_v2_cv_train_smoke',
            '--validation','artifacts/framework/local_policy/pipeline_v2_cv_validation_smoke',
            '--out','artifacts/framework/pipeline_v2/cv_success','--context-dim','0','--outcome','official_task_success_immediate'])
        status('ready',checkpoint='artifacts/framework/pipeline_v2/cv_success/value.pt')
    except Exception as error:
        status('failed',error=repr(error));raise


if __name__=='__main__':main()
