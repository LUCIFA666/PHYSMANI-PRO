"""Continue real FASTER training and inference validation within the deadline."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
STOP = ROOT/'runtime/framework/state/STOP'
STATUS = ROOT/'artifacts/framework/faster_pipeline_status.json'
EXPERIMENT = 'physmani_all16_goal_v1'
CHECKPOINTS = ROOT/'artifacts/framework/faster_checkpoints/pi05_faster_our_mani_lora'/EXPERIMENT
sys.path.insert(0, str(ROOT))
from our_mani.checkpoint_snapshot import snapshot_checkpoint


def retain_best(validation_path):
    validation = json.loads((ROOT/validation_path).read_text())
    best_path = ROOT/'artifacts/framework/faster_best_validation.json'
    if best_path.exists():
        previous = json.loads(best_path.read_text())
        if previous['mean_position_action_error_m'] <= validation['mean_position_action_error_m']:
            return
    source = Path(validation['checkpoint'])
    destination = ROOT/'artifacts/framework/faster_pinned'/('step_'+str(validation['optimizer_updates']).zfill(8))
    receipt = snapshot_checkpoint(source, destination)
    record = {'validation_report': str(validation_path), 'checkpoint': str(destination),
              'optimizer_updates': validation['optimizer_updates'],
              'checkpoint_sha256': receipt['checkpoint_sha256'],
              'mean_position_action_error_m': validation['mean_position_action_error_m'],
              'selection_rule': 'minimum mean XYZ action error on fixed eight grasp2 VAL observations',
              'formal_test_used': False, 'closed_loop_success_verified': False}
    temporary = best_path.with_suffix('.tmp')
    temporary.write_text(json.dumps(record, indent=2)+'\n')
    temporary.replace(best_path)


def write(stage, **extra):
    record = {'stage': stage, 'created_utc': datetime.now(timezone.utc).isoformat(), 'formal_test_used': False, **extra}
    temp = STATUS.with_suffix('.tmp')
    temp.write_text(json.dumps(record, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    temp.replace(STATUS)


def run(stage, command, *, update_limit=0):
    env = dict(os.environ)
    env.update(PYTHONPATH=':'.join([str(ROOT), str(ROOT/'third_party/FASTER/src'),
                                   str(ROOT/'third_party/FASTER/packages/openpi-client/src')]),
               HF_HOME=str(ROOT/'runtime/framework/huggingface'), HF_LEROBOT_HOME=str(ROOT/'artifacts/framework/lerobot'),
               OPENPI_DATA_HOME=str(ROOT/'artifacts/framework/openpi_cache'), XDG_CACHE_HOME=str(ROOT/'runtime/framework/cache'),
               TMPDIR=str(ROOT/'runtime/framework/tmp'), JAX_COMPILATION_CACHE_DIR=str(ROOT/'runtime/framework/jax-cache'),
               XLA_PYTHON_CLIENT_PREALLOCATE='false', XLA_PYTHON_CLIENT_MEM_FRACTION='.45', WANDB_MODE='offline',
               OUR_MANI_RUN_UPDATE_LIMIT=str(update_limit), OUR_MANI_STOP_FILE=str(STOP), PYTHONUNBUFFERED='1',
               OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
    write(stage, command=command, maximum_updates_this_run=update_limit)
    log_path = ROOT/f'logs/framework/{stage}.log'
    with log_path.open('a') as log:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        while process.poll() is None:
            if STOP.exists():
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGTERM)
                    process.wait(timeout=20)
                raise InterruptedError('deadline STOP')
            time.sleep(5)
    if process.returncode:
        raise RuntimeError(f'{stage} failed; see {log_path}')


def latest():
    return max((p for p in CHECKPOINTS.iterdir() if p.is_dir() and p.name.isdigit()), key=lambda p:int(p.name))


def validate_checkpoint(path):
    updates = int(path.name)+1
    directory = ROOT/'artifacts/framework/faster_validation'
    directory.mkdir(parents=True, exist_ok=True)
    output = f'artifacts/framework/faster_validation/checkpoint_{updates:08d}.json'
    run(f'faster_validation_at_{updates:08d}', [sys.executable, 'scripts/validate_faster_model.py',
        '--checkpoint', str(path), '--episodes-per-task', '4', '--out', output])
    retain_best(output)


def closed_loop_milestone():
    updates = int(latest().name)+1
    milestones = [value for value in (5000, 10000, 20000, 30000) if updates >= value]
    if not milestones:
        return
    milestone = max(milestones)
    receipt_path = ROOT/f'artifacts/framework/full_workflow/milestone_{milestone:08d}.json'
    if receipt_path.exists():
        return
    best = json.loads((ROOT/'artifacts/framework/faster_best_validation.json').read_text())
    write('closed_loop_development_milestone', optimizer_updates=updates, milestone=milestone,
          selected_checkpoint=best['checkpoint'])
    log_path = ROOT/f'logs/framework/faster_server_milestone_{milestone:08d}.log'
    server, dynamics_server = None, None
    failures, outputs = [], []
    try:
        with log_path.open('a') as log:
            server = subprocess.Popen([sys.executable, 'scripts/serve_faster.py', '--checkpoint', best['checkpoint'], '--port', '18471'],
                cwd=ROOT, env={**os.environ, 'XLA_PYTHON_CLIENT_PREALLOCATE':'false',
                               'OMP_NUM_THREADS':'4', 'OPENBLAS_NUM_THREADS':'4', 'MKL_NUM_THREADS':'4'},
                stdout=log, stderr=subprocess.STDOUT)
            for _ in range(90):
                if server.poll() is not None:
                    raise RuntimeError(f'neural server exited {server.returncode}; see {log_path}')
                if STOP.exists():
                    raise InterruptedError('STOP during milestone inference setup')
                try:
                    with urllib.request.urlopen('http://127.0.0.1:18471/healthz', timeout=2) as response:
                        if response.status==200:
                            break
                except OSError:
                    time.sleep(2)
            else:
                raise TimeoutError('neural server did not become ready')
            run(f'neural_transport_milestone_{milestone:08d}',
                ['/home/ubuntu/physmani-conda/envs/physmani5090/bin/python', 'scripts/smoke_faster_neural_transport.py'])
            shutil.copy2(ROOT/'artifacts/framework/smoke/faster_neural_transport.json',
                         ROOT/f'artifacts/framework/full_workflow/neural_transport_{milestone:08d}.json')
            ngff_socket = ROOT/'runtime/framework/ngff_dynamics.sock'
            if not ngff_socket.exists():
                dynamics_server = subprocess.Popen([str(ROOT/'.conda-env/bin/python'), 'scripts/serve_ngff_dynamics.py'],
                    cwd=ROOT, env={**os.environ, 'CUDA_VISIBLE_DEVICES':''}, stdout=log, stderr=subprocess.STDOUT)
            for _ in range(30):
                if dynamics_server is not None and dynamics_server.poll() is not None:
                    raise RuntimeError('frozen NGFF inference worker failed to initialize')
                if (ROOT/'runtime/framework/ngff_dynamics.sock').exists():
                    break
                time.sleep(1)
            else:
                raise TimeoutError('frozen NGFF worker did not become ready')
            for predictor in ('cv', 'driver', 'ngff_driver'):
                output = f'artifacts/framework/full_workflow/milestone_{milestone:08d}_{predictor}'
                run(f'closed_loop_{milestone:08d}_{predictor}', ['bash', 'scripts/run_local_sim.sh',
                    'scripts/smoke_full_workflow.py', '--dynamics', predictor, '--seconds', '8', '--learned-local', '--out', output])
                outputs.append(output+'/report.json')
    except InterruptedError:
        raise
    except Exception as error:
        # A diagnostic failure is evidence to retain, not a reason to lose a
        # saved training state. Formal TEST remains locked in every case.
        failures.append(str(error))
    finally:
        if dynamics_server is not None and dynamics_server.poll() is None:
            dynamics_server.terminate()
            dynamics_server.wait(timeout=20)
        if server is not None and server.poll() is None:
            server.terminate()
            try:
                server.wait(timeout=20)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps({'optimizer_updates':updates, 'milestone':milestone,
        'checkpoint':best, 'development_reports':outputs, 'infrastructure_failures':failures,
        'formal_test_used':False, 'kind':'development_full_workflow_checks_not_paper_success_rate'}, indent=2)+'\n')


def main():
    try:
        while True:
            if STOP.exists():
                raise InterruptedError('deadline STOP')
            report = json.loads((ROOT/'artifacts/framework/bootstrap_status.json').read_text())
            if report['stage'] == 'training_smoke_complete':
                break
            write('waiting_for_successful_ten_update_faster_smoke', bootstrap_stage=report['stage'])
            time.sleep(10)
        common = ['--repo-id', 'hdf5/physmani_train_all16', '--exp-name', EXPERIMENT, '--batch-size', '2', '--updates', '30000']
        existing = CHECKPOINTS.exists() and any(p.is_dir() and p.name.isdigit() for p in CHECKPOINTS.iterdir())
        if not existing or int(latest().name)+1 < 100:
            run('faster_all16_first_100_updates', [sys.executable, 'scripts/faster_training.py', 'train', *common,
                                                *(['--resume'] if existing else [])], update_limit=100)
        path = latest()
        validate_checkpoint(path)
        closed_loop_milestone()
        if STOP.exists():
            raise InterruptedError('deadline STOP')
        # Keep the 30k-step schedule and exact sampling resume. Intermediate
        # training runs reserve opportunities for real VAL inference checks.
        first_tranche = max(1, (int(latest().name)+1-100)//500+1)
        for tranche in range(first_tranche, 62):
            current = ROOT/'config/framework/current_run.json'
            run_config = json.loads((current if current.exists() else ROOT/'config/framework/overnight.json').read_text())
            raw_deadline = run_config.get('deadline_utc')
            deadline = datetime.fromisoformat(raw_deadline) if raw_deadline else None
            if deadline is not None and (deadline-datetime.now(timezone.utc)).total_seconds() < 20*60:
                break
            if int(latest().name)+1 >= 30000:
                break
            run(f'faster_all16_tranche_{tranche}', [sys.executable, 'scripts/faster_training.py', 'train', *common, '--resume'],
                update_limit=500)
            validate_checkpoint(latest())
            closed_loop_milestone()
        write('bounded_faster_training_and_val_checks_complete', latest_checkpoint=str(latest()))
    except InterruptedError as error:
        write('stopped', reason=str(error))
    except Exception as error:
        write('failed', reason=str(error))
        raise


if __name__ == '__main__':
    main()
