"""Fixed first four ordinary-bin TRAIN sources, two geometry methods, no retries."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from our_mani.multitask.evaluation import atomic_json, file_sha, verify_inputs

TASK = 'put_rubbish_in_moving_bin'
SOURCES = ('episode105', 'episode106', 'episode119', 'episode123')
METHODS = ('cv_geometry', 'context_geometry')
SETUP = ROOT / 'artifacts/framework/multitask_r10/bin_executor_v16_setup'
OUTPUT = ROOT / 'artifacts/framework/multitask_r10/bin_executor_v16'
WALL_TIMEOUT = 285.


def utc():
    return datetime.now(timezone.utc).isoformat()


def command(method, raw):
    if method not in METHODS or raw not in SOURCES:
        raise ValueError('Only preregistered normal-bin TRAIN cases are allowed')
    return ['bash', 'scripts/run_local_sim.sh', 'scripts/run_geometry_bin_v16_r10.py',
        '--config', str((SETUP / 'configuration.json').relative_to(ROOT)),
        '--manifest', str((SETUP / 'manifest.json').relative_to(ROOT)),
        '--frozen-inputs', str((SETUP / 'frozen_inputs.json').relative_to(ROOT)),
        '--task', TASK, '--raw-episode-id', raw, '--split', 'train', '--replica', '0',
        '--development-demo', '--method', method,
        '--out', str((OUTPUT / 'development_demo_only' / raw / method).relative_to(ROOT))]


def run_one(method, raw, files):
    verify_inputs(ROOT, files)
    if (ROOT / 'runtime/framework/multitask_r10_STOP').exists():
        raise RuntimeError('Existing controlled STOP prevents launch')
    logs = OUTPUT / 'parent_raw' / raw / method
    logs.mkdir(parents=True, exist_ok=False)
    argv = command(method, raw)
    started = time.monotonic()
    with (logs / 'native.log').open('wb') as log:
        process = subprocess.Popen(argv, cwd=ROOT,
            env=dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1'),
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        ticks = int(Path(f'/proc/{process.pid}/stat').read_text().rsplit(')', 1)[1].split()[19])
        launch = dict(launched_utc=utc(), parent_pid=os.getpid(), launcher_pid=process.pid,
            launcher_start_ticks=ticks, command=argv, automatic_retry=False)
        atomic_json(logs / 'launch.json', launch)
        timeout = False
        try:
            code = process.wait(timeout=WALL_TIMEOUT)
        except subprocess.TimeoutExpired:
            timeout = True
            os.killpg(process.pid, signal.SIGTERM)
            try:
                code = process.wait(timeout=5.)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                code = process.wait(timeout=5.)
    row = dict(launch, finished_utc=utc(), method=method, source_raw_episode_id=raw,
        returncode=code, timed_out=timeout, wall_seconds=time.monotonic()-started,
        case=str((OUTPUT/'development_demo_only'/raw/method/'case.json').relative_to(ROOT)),
        raw_log=dict(path=str((logs/'native.log').relative_to(ROOT)), sha256=file_sha(logs/'native.log')))
    atomic_json(logs / 'parent_process_receipt.json', row)
    return row


def main():
    if OUTPUT.exists():
        raise RuntimeError('Preserve previous attempt; no same-namespace rerun')
    files = json.loads((SETUP/'frozen_inputs.json').read_text())['files']
    verify_inputs(ROOT, files)
    if files.get(str(Path(__file__).resolve())) != file_sha(__file__):
        raise RuntimeError('Actual supervisor is not frozen')
    OUTPUT.mkdir()
    jobs = [(m, raw) for raw in SOURCES for m in METHODS]
    atomic_json(OUTPUT/'launch_review.json', dict(created_utc=utc(), task=TASK,
        sources=list(SOURCES), methods=list(METHODS), expected_cases=len(jobs),
        maximum_native_workers=2, GPU_workers=0, formal_metric=False,
        outcome_independent_sources=True, models_retrained=False, automatic_retry=False))
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(run_one, m, raw, files) for m, raw in jobs]
        rows = [future.result() for future in futures]
    verify_inputs(ROOT, files)
    atomic_json(OUTPUT/'parent_summary.json', dict(created_utc=utc(), children=rows,
        all_children_terminal=True, frozen_inputs_unchanged=True, formal_metric=False))
    print(json.dumps(dict(terminal=True, cases=len(rows), returncodes=[r['returncode'] for r in rows])))


if __name__ == '__main__':
    from scripts.run_native_actuator_diagnostic_pair_v5_2_r10 import exclusive_pair, process_visibility
    visibility = process_visibility()
    if not visibility['host_visibility_available'] or visibility['live_project_native_processes'] or visibility['same_uid_inventory_incomplete']:
        raise RuntimeError('Host process inventory rejects duplicate or incomplete view')
    with exclusive_pair():
        main()
