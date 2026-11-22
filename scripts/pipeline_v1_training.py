"""Complete controller-specific context outcome collection and success fitting."""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/framework/pipeline_v1'
DYNAMICS=OUT/'dynamics_simnorm_v1/dynamics.pt'


def status(stage,**fields):
    data={'created_utc':datetime.now(timezone.utc).isoformat(),'stage':stage,'formal_test_used':False,**fields}
    temporary=OUT/'training_status.tmp'
    temporary.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    temporary.replace(OUT/'training_status.json')


def run(stage,command):
    status(stage,command=command)
    with (ROOT/f'logs/framework/{stage}.log').open('a') as log:
        subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)


def main():
    try:
        while not DYNAMICS.exists() or not DYNAMICS.with_name('report.json').exists():
            status('waiting_for_context_simnorm_training')
            time.sleep(5)
        common=['bash','scripts/run_local_sim.sh','scripts/collect_local_rollouts.py','--replicas','4',
                '--full-task','--latent-checkpoint',str(DYNAMICS.relative_to(ROOT))]
        run('pipeline_v1_context_MPC_TRAIN',[*common,'--split','train','--episodes-per-task','24',
            '--name','pipeline_v1_context_train_v1_smoke'])
        run('pipeline_v1_context_MPC_VAL',[*common,'--split','validation','--episodes-per-task','20',
            '--name','pipeline_v1_context_validation_v1_smoke'])
        run('pipeline_v1_context_success_training',[sys.executable,'scripts/train_success_predictor.py',
            '--data','artifacts/framework/local_policy/pipeline_v1_context_train_v1_smoke',
            '--validation','artifacts/framework/local_policy/pipeline_v1_context_validation_v1_smoke',
            '--out','artifacts/framework/pipeline_v1/context_success','--context-dim','16',
            '--outcome','official_task_success_immediate'])
        status('context_dynamics_MPC_and_success_ready_for_frozen_TEST',
               dynamics=str(DYNAMICS.relative_to(ROOT)),success='artifacts/framework/pipeline_v1/context_success/value.pt')
    except Exception as error:
        status('failed',error=repr(error))
        raise


if __name__=='__main__':main()
