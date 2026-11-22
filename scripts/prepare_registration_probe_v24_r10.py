"""Source-bound metric-only registration probe of the frozen v22 actor."""
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from our_mani.multitask.evaluation import atomic_json,file_sha,verify_inputs
from our_mani.multitask.task_native_backend_v24 import diagnostic_binding
from scripts.run_task_executor_smoke_v24_r10 import parser,preflight

BASE=ROOT/'artifacts/framework/multitask_r10'
SETUP=BASE/'registration_probe_v24_setup'
OUTPUT=BASE/'registration_probe_v24'


def command():
    return ['bash','scripts/run_local_sim.sh','scripts/run_task_executor_smoke_v24_r10.py',
        '--config',str((SETUP/'configuration.json').relative_to(ROOT)),
        '--manifest','artifacts/framework/protocol/train_manifest.json',
        '--frozen-inputs',str((SETUP/'frozen_inputs.json').relative_to(ROOT)),
        '--task','put_rubbish_in_moving_bin','--raw-episode-id','episode105','--split','train',
        '--method','context_geometry','--replica','0','--development-demo',
        '--out',str((OUTPUT/'development_demo_only'/'bin_context').relative_to(ROOT))]


def main():
    if SETUP.exists():raise RuntimeError('Preserve completed preparation')
    old=json.loads((BASE/'task_executor_smoke_v22_setup/frozen_inputs.json').read_text());verify_inputs(ROOT,old['files'])
    for launch in ([sys.executable],['bash','scripts/run_local_sim.sh']):
        subprocess.run(launch+['-m','unittest','discover','-s','tests','-p','test_registration_metric_observer_v24.py'],cwd=ROOT,check=True)
    c=json.loads((BASE/'task_executor_smoke_v22_setup/configuration.json').read_text())
    c['native_metric_diagnostics']=diagnostic_binding(ROOT,c['baseline_repo'])
    SETUP.mkdir();atomic_json(SETUP/'configuration.json',c)
    files=dict(old['files'])
    for p in [Path(__file__),ROOT/'scripts/run_task_executor_smoke_v24_r10.py',
        ROOT/'our_mani/multitask/task_native_backend_v24.py',ROOT/'our_mani/multitask/registration_metric_observer_v24.py',
        ROOT/'tests/test_registration_metric_observer_v24.py',SETUP/'configuration.json']:
        files[str(p.resolve())]=file_sha(p)
    atomic_json(SETUP/'frozen_inputs.json',dict(schema='r10_registration_metric_only_v24_TRAIN_lock',
        files=files,approved_for_execution=True,native_started=False,models_and_policy_from_v22_unchanged=True,
        native_labels_not_actor_inputs=True,formal_release=False))
    preflight(parser().parse_args(command()[3:]))
    result=subprocess.run(command()+['--check'],cwd=ROOT,capture_output=True,text=True,check=True)
    atomic_json(SETUP/'native_CPU_preflight.json',json.loads(result.stdout))
    verify_inputs(ROOT,files);print(json.dumps(dict(prepared=True,frozen_inputs=len(files),metric_only=True)))


if __name__=='__main__':main()
