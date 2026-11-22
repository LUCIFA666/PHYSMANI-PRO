"""Continue local model fitting and independent evaluation when collection ends."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
STOP = ROOT/'runtime/framework/local_state/STOP'
STATUS = ROOT/'artifacts/framework/local_policy/pipeline_status.json'


def write(stage, **extra):
    record = {'stage': stage, 'created_utc': datetime.now(timezone.utc).isoformat(), 'formal_test_used': False, **extra}
    temp = STATUS.with_suffix('.tmp')
    temp.write_text(json.dumps(record, ensure_ascii=False, indent=2)+'\n')
    temp.replace(STATUS)


def run(stage, command):
    write(stage, command=command)
    log_path = ROOT/f'logs/framework/{stage}.log'
    with log_path.open('a') as log:
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        while process.poll() is None:
            if STOP.exists():
                # Rollout/training loops inspect STOP and close/save themselves.
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGTERM)
                    process.wait(timeout=20)
                raise InterruptedError('deadline STOP')
            time.sleep(5)
    if process.returncode:
        raise RuntimeError(f'{stage} failed; see {log_path}')


def main():
    try:
        while subprocess.run(['systemctl', '--user', 'is-active', 'our-mani-local-collect.service'],
                             capture_output=True, text=True).stdout.strip() == 'active':
            if STOP.exists():
                raise InterruptedError('deadline STOP')
            write('waiting_for_frozen_controller_training_rollouts')
            time.sleep(10)
        report_path = ROOT/'artifacts/framework/local_policy/mpc_train_v2/report.json'
        report = json.loads(report_path.read_text())
        if report['completed'] != 48*6*2:
            raise RuntimeError('training rollout collection incomplete; inspect before fitting')
        if not (ROOT/'artifacts/framework/local_policy/models_v1/report.json').exists():
            run('local_model_fitting_v1', [sys.executable, 'scripts/train_local_models.py'])
        common = ['--split', 'validation', '--episodes-per-task', '20', '--replicas', '4', '--full-task']
        run('local_mpc_validation_v1', ['bash', 'scripts/run_local_sim.sh', 'scripts/collect_local_rollouts.py',
                                      *common, '--name', 'mpc_validation_v1'])
        run('local_servo_validation_v1', ['bash', 'scripts/run_local_sim.sh', 'scripts/collect_local_rollouts.py',
                    *common, '--name', 'servo_validation_v1', '--learned-checkpoint',
                    str(ROOT/'artifacts/framework/local_policy/models_v1/servo.pt')])
        run('local_value_and_paired_servo_analysis_v1', [sys.executable, 'scripts/validate_local_models.py'])
        write('local_training_and_independent_validation_complete')
    except InterruptedError as error:
        write('stopped', reason=str(error))
    except Exception as error:
        write('failed', reason=str(error))
        raise


if __name__ == '__main__':
    main()
