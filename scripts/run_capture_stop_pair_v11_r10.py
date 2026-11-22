"""Exactly two fixed nonfitting TRAIN104 children, CPU-only, no retries."""
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
sys.path.insert(0,str(ROOT))
from our_mani.multitask.evaluation import atomic_json, file_sha, verify_inputs

SETUP = ROOT/'artifacts/framework/multitask_r10/geometry_capture_stop_v11_setup'
OUTPUT = ROOT/'artifacts/framework/multitask_r10/geometry_capture_stop_v11'
MANIFEST = ROOT/'artifacts/framework/multitask_r10/basketball_Context_geometry_TRAIN_development_demo_v1_setup/manifest.json'
WALL_TIMEOUT = 285.  # two bounded five-second termination waits remain below300s


def utc(): return datetime.now(timezone.utc).isoformat()


def command(method):
    leaf='context_normal' if method=='context_geometry' else 'cv_normal'
    return ['bash','scripts/run_local_sim.sh','scripts/run_geometry_capture_stop_v11_r10.py',
        '--config',str((SETUP/'configuration.json').relative_to(ROOT)),
        '--manifest',str(MANIFEST.relative_to(ROOT)),
        '--frozen-inputs',str((SETUP/'frozen_inputs.json').relative_to(ROOT)),
        '--task','moving_basketball_in_hoop','--raw-episode-id','episode104',
        '--split','train','--replica','0','--development-demo','--method',method,
        '--out',str((OUTPUT/'development_demo_only'/leaf).relative_to(ROOT))]


def run_one(method, files):
    verify_inputs(ROOT,files)
    if (ROOT/'runtime/framework/multitask_r10_STOP').exists():
        raise RuntimeError('Existing controlled STOP prevents a new native child')
    leaf='context_normal' if method=='context_geometry' else 'cv_normal'
    raw=OUTPUT/'parent_raw'/leaf
    raw.mkdir(parents=True,exist_ok=False)
    argv=command(method)
    launched=utc();started=time.monotonic()
    with (raw/'native.log').open('wb') as log:
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1')
        process=subprocess.Popen(argv,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        stat=Path(f'/proc/{process.pid}/stat').read_text().rsplit(')',1)[1].split()
        atomic_json(raw/'launch.json',dict(schema='r10_owned_fixed_TRAIN104_pair_child_launch_v1',
            launched_utc=launched,parent_process_id=os.getpid(),launcher_process_id=process.pid,
            launcher_start_ticks=int(stat[19]),command=argv,CUDA_VISIBLE_DEVICES='',
            timeout_seconds=WALL_TIMEOUT,automatic_retry=False,policy_outcome_selection=False))
        timed_out=False
        try:
            returncode=process.wait(timeout=WALL_TIMEOUT)
        except subprocess.TimeoutExpired:
            timed_out=True
            os.killpg(process.pid,signal.SIGTERM)
            try: returncode=process.wait(timeout=5.)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGKILL)
                returncode=process.wait(timeout=5.)
    row=dict(method=method,launched_utc=launched,finished_utc=utc(),command=argv,
        launcher_process_id=process.pid,returncode=returncode,wall_seconds=time.monotonic()-started,
        timed_out=timed_out,timeout_seconds=WALL_TIMEOUT,automatic_retry=False,
        raw_log=dict(path=str((raw/'native.log').relative_to(ROOT)),sha256=file_sha(raw/'native.log')),
        case=str((OUTPUT/'development_demo_only'/leaf/'case.json').relative_to(ROOT)))
    atomic_json(raw/'parent_process_receipt.json',row)
    return row


def main():
    if OUTPUT.exists(): raise RuntimeError('Preserve previous attempt; no automatic rerun')
    lock=json.loads((SETUP/'frozen_inputs.json').read_text())
    files=lock['files']
    verify_inputs(ROOT,files)
    if files.get(str(Path(__file__).resolve()))!=file_sha(__file__):
        raise RuntimeError('Owned fixed-pair supervisor source is not frozen')
    for method in ('cv_geometry','context_geometry'):
        child = (ROOT/command(method)[2]).resolve()
        if not child.is_file() or files.get(str(child)) != file_sha(child):
            raise RuntimeError('Actual parent worker source is missing or not frozen')
    OUTPUT.mkdir()
    atomic_json(OUTPUT/'launch_review.json',dict(schema='r10_native_actuator_readonly_TRAIN104_fixed_pair_v5',
        created_utc=utc(),configuration=dict(path=str((SETUP/'configuration.json').relative_to(ROOT)),
            sha256=file_sha(SETUP/'configuration.json')),
        frozen_inputs=dict(path=str((SETUP/'frozen_inputs.json').relative_to(ROOT)),
            sha256=file_sha(SETUP/'frozen_inputs.json')),
        parent_process_id=os.getpid(),methods=['cv_geometry','context_geometry'],
        task='moving_basketball_in_hoop',raw_episode_id='episode104',split='train',
        commands=[command(m) for m in ('cv_geometry','context_geometry')],
        CPU_native_workers=2,CUDA_VISIBLE_DEVICES='',wall_timeout_per_case=WALL_TIMEOUT,
        no_extra_videos=True,nonfitting=True,formal_metric=False,automatic_retry=False,
        actual_window_acceleration_limit=6.,motor_or_native_scalar_acceleration_change=False))
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(run_one,m,files) for m in ('cv_geometry','context_geometry')]
        rows=[future.result() for future in futures]
    verify_inputs(ROOT,files)
    atomic_json(OUTPUT/'parent_summary.json',dict(created_utc=utc(),children=rows,
        all_children_terminal=True,original_frozen_inputs_verified_after=True,formal_metric=False))
    print(json.dumps(rows,ensure_ascii=False))


if __name__=='__main__':
    from scripts.run_native_actuator_diagnostic_pair_v5_2_r10 import exclusive_pair, process_visibility
    p=process_visibility()
    if not p['host_visibility_available'] or p['live_project_native_processes'] or p['same_uid_inventory_incomplete']:
        raise RuntimeError('Native process preflight rejected a duplicate or unavailable host view')
    with exclusive_pair(): main()
