"""Finish the fixed visual-centre pilot after its existing capture queue closes.

Default is a metadata-only check. --execute waits for the exact forty-parent
capture, compiles its data, trains, and evaluates the selected weights. It never
starts Native, retries a failed stage, changes a split, or admits a runtime.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from our_mani.current_cube_rgbd_centre_20261103_v1.training import (
    check, training_source_map, SELECTED_ENCODER_SHA)

PLAN = ROOT/'config/framework/current_cube_dual_crop_CF_pilot40_20261103_v1/plan.json'
PLAN_SHA = '3c56dbf61708c369e1a80bc5fb97a20a1f9b5d5c4fa718bbc0eab755561ff2cf'
CAPTURE = ROOT/'runtime/framework/near_goal_perception_20261103/current_cube_dual_crop_CF_pilot40_v2'
BASE = ROOT/'artifacts/framework/local_mpc_20261102'
DATA = BASE/'current_cube_dual_crop_CF_pilot40_dataset_v1'
FIT = BASE/'current_cube_dual_crop_CF_pilot40_fit_v1'
VAL = BASE/'current_cube_dual_crop_CF_pilot40_selection_VAL_v1'
RESERVED = BASE/'current_cube_dual_crop_CF_pilot40_reserved8_eval_v1'
ENCODER = ROOT/'artifacts/framework/near_goal_segmentation_20261103/full32_geometryF_unknownKL_fit_v5/selected.pt'
COMPILER = ROOT/'scripts/prepare_current_cube_dual_crop_TRAIN_manifest_20261103_v1.py'
TRAINER = ROOT/'scripts/train_current_cube_dual_crop_centre_20261103_v1.py'
EVALUATOR = ROOT/'scripts/evaluate_current_cube_dual_crop_centre_20261103_v1.py'
SELECTION = ROOT/'scripts/evaluate_current_cube_dual_crop_selection_20261103_v1.py'


def now(): return datetime.now(timezone.utc).isoformat()
def read(path): return json.loads(Path(path).read_text())
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def pointer(path): return dict(path=str(Path(path).resolve()), sha256=sha(path))


def write_new(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def sources():
    result = training_source_map()
    for path in (Path(__file__).resolve(), COMPILER, SELECTION, PLAN, ENCODER):
        result[str(path)] = sha(path)
    if result[str(PLAN)] != PLAN_SHA or result[str(ENCODER)] != SELECTED_ENCODER_SHA:
        raise ValueError('Fixed final capture plan and original encoder required')
    return result


def unchanged(frozen):
    if any(sha(path) != digest for path, digest in frozen.items()):
        raise ValueError('Frozen pilot pipeline implementation changed')


def capture_progress():
    ledger = CAPTURE/'serial_queue_v1.jsonl'
    if not ledger.is_file(): return dict(complete=False, closed=1)
    content = ledger.read_text()
    lines = content.splitlines()
    if content and not content.endswith('\n'): lines = lines[:-1]
    rows = [json.loads(line) for line in lines]
    if any(row['event'] == 'infrastructure_stop' for row in rows):
        raise ValueError('Existing capture queue stopped; preserve failure, never retry')
    starts = [r for r in rows if r['event'] == 'start']
    if len(starts) != 1 or starts[0]['plan_sha256'] != PLAN_SHA or starts[0]['closed_prefix_count'] != 1:
        raise ValueError('Exact frozen remaining39 queue required')
    complete = [r for r in rows if r['event'] == 'complete']
    closed = 1 + sum(row['event'] == 'actual_closed' for row in rows)
    if complete and (len(complete) != 1 or closed != 40 or complete[0].get('all40_actual_closed') is not True):
        raise ValueError('Whole forty-parent capture completion required')
    return dict(complete=bool(complete), closed=closed)


def execute(output, frozen, wait_seconds):
    output.mkdir(parents=True, exist_ok=False)
    write_new(output/'request.json', dict(utc=now(), sources=frozen, plan=pointer(PLAN),
        capture_root=str(CAPTURE), training_epochs=40, expected_optimizer_steps=520,
        no_Native_launches=True, no_retries=True, formal_TEST_read=False,
        reserved_evaluation_is_visual_TRAIN_development=True))
    stops = (output/'STOP', CAPTURE/'STOP', CAPTURE.parent/'STOP',
             ROOT/'runtime/framework/local_mpc_20261102/STOP')
    interrupted = dict(signal=None)
    old_handlers = {}
    for signum in (signal.SIGINT, signal.SIGTERM):
        old_handlers[signum] = signal.signal(signum, lambda n, f: interrupted.update(signal=n))
    status = dict(complete=False, stage='waiting_for_capture', training_started=False,
                  root_actual_returncode=None, pipeline_source_frozen=True)
    def update(**values):
        status.update(values, updated_utc=now())
        temporary = output/'status.tmp.json'
        temporary.write_text(json.dumps(status, ensure_ascii=False, indent=2)+'\n')
        temporary.replace(output/'status.json')
    def guard():
        if interrupted['signal'] or any(p.exists() for p in stops):
            raise InterruptedError('Administrative stop/signal; no further stages')
        unchanged(frozen)
    def run(stage, command, *, gpu=False):
        guard(); update(stage=stage, training_started=status['training_started'] or stage == 'train')
        environment = os.environ.copy()
        environment.update(CUDA_VISIBLE_DEVICES='0' if gpu else '',
            OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
            CUBLAS_WORKSPACE_CONFIG=':4096:8', PYTHONDONTWRITEBYTECODE='1')
        process = None
        started = now()
        try:
            with (output/(stage+'.log')).open('x') as log:
                process = subprocess.Popen(command, cwd=ROOT, env=environment,
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                while process.poll() is None:
                    guard(); time.sleep(2)
                code = process.wait()
            source_unchanged = all(sha(path) == digest for path, digest in frozen.items())
            write_new(output/(stage+'.actual_exit.json'), dict(stage=stage,
                started_utc=started, finished_utc=now(), argv=[str(x) for x in command],
                actual_returncode=code, actual_wait_completed=True, source_unchanged=source_unchanged))
            guard()
            if code: raise RuntimeError('Stage failed without retry: '+stage)
            print(json.dumps(dict(event='stage_complete', stage=stage, utc=now())), flush=True)
        finally:
            if process is not None and process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try: process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL); process.wait()
                write_new(output/(stage+'.interrupted_exit.json'), dict(stage=stage,
                    actual_returncode=process.returncode, actual_wait_completed=True,
                    interrupted=True, further_stages_not_started=True))
    locks = []
    try:
        begin = time.monotonic()
        while True:
            guard(); progress = capture_progress(); update(capture=progress)
            if progress['complete']: break
            if time.monotonic()-begin > wait_seconds:
                raise TimeoutError('Capture wait limit; no training on incomplete data')
            time.sleep(5)
        # Same lock order as Native. No Native/GPU overlap, including another owner.
        update(stage='waiting_for_exclusive_resources')
        for name in ('fixed_TRAIN104_geometry_native.lock', 'TRAIN_postcomplete_diagnostics_v1106.lock'):
            handle = (ROOT/'runtime/framework'/name).open('a')
            locks.append(handle)
            while True:
                guard()
                try: fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB); break
                except BlockingIOError: time.sleep(2)
        py = sys.executable
        common = [py, '-B', str(COMPILER), '--capture-plan', str(PLAN), '--capture-root', str(CAPTURE)]
        run('closed40_audit', common+['--audit-closed'])
        run('compile', common+['--compile', '--output', str(DATA)])
        run('train_check', [py, '-B', str(TRAINER), '--check', '--manifest', str(DATA/'manifest.json')])
        run('train', [py, '-B', str(TRAINER), '--train', '--manifest', str(DATA/'manifest.json'),
            '--encoder-checkpoint', str(ENCODER), '--expected-encoder-sha256', SELECTED_ENCODER_SHA,
            '--output', str(FIT), '--device', 'cuda'], gpu=True)
        selected = pointer(FIT/'selected.pt')
        fit_status = read(FIT/'fit_status.json')
        if fit_status.get('training_complete') is not True or fit_status.get('actual_optimizer_steps') != 520:
            raise ValueError('Actual complete forty-epoch fit required')
        write_new(output/'selected_freeze.json', dict(utc=now(), selected=selected,
            sources=frozen, fit_status=pointer(FIT/'fit_status.json'), runtime_admitted=False))
        run('selection_VAL', [py, '-B', str(SELECTION), '--evaluate', '--manifest', str(DATA/'manifest.json'),
            '--selected', selected['path'], '--expected-sha256', selected['sha256'],
            '--encoder-checkpoint', str(ENCODER), '--output', str(VAL)])
        evaluation = [py, '-B', str(EVALUATOR), '--manifest', str(DATA/'manifest.json'),
            '--selected', selected['path'], '--expected-sha256', selected['sha256'],
            '--encoder-checkpoint', str(ENCODER), '--output', str(RESERVED)]
        run('reserved_predict', evaluation+['--predict'])
        run('reserved_score', [py, '-B', str(EVALUATOR), '--score', '--output', str(RESERVED)])
        update(complete=True, stage='complete', root_actual_returncode=0,
               training_complete=True, runtime_admitted=False,
               selected=selected, selection_report=pointer(VAL/'report.json'),
               reserved_report=pointer(RESERVED/'report.json'))
        return 0
    except BaseException as error:
        update(stage='stopped', root_actual_returncode=1,
               error_type=type(error).__name__, error=str(error), no_retry=True)
        raise
    finally:
        for handle in reversed(locks): handle.close()
        for signum, handler in old_handlers.items(): signal.signal(signum, handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--wait-seconds', type=float, default=5400)
    args = parser.parse_args()
    checked = check(); frozen = sources()
    if not args.execute:
        print(json.dumps(dict(complete=True, Native_started=False, training_started=False,
            model_forward_calls=0, stages=['wait_capture40', 'closed40_audit', 'compile', 'train_check',
                'train40', 'selected_freeze', 'selection_VAL', 'reserved_predict', 'reserved_score'],
            sources=frozen, training_config=checked['config']), ensure_ascii=False))
        return 0
    if args.output is None or not 0 < args.wait_seconds <= 14400:
        parser.error('Execution needs fresh output and bounded capture wait')
    output = args.output.resolve()
    if BASE not in output.parents or any(p.exists() for p in (DATA, FIT, VAL, RESERVED)):
        parser.error('Fresh artifact namespace and no previous dataset/fit/evaluation required')
    return execute(output, frozen, args.wait_seconds)


if __name__ == '__main__': raise SystemExit(main())
