"""Freeze metric-only all-phase collision instrumentation of the v16 TRAIN policy."""
from copy import deepcopy
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from our_mani.multitask.evaluation import atomic_json,file_sha,verify_inputs
from our_mani.multitask.bin_native_backend_v17 import diagnostic_binding
from scripts.run_bin_TRAIN_v17_r10 import SETUP,command
from scripts.run_geometry_bin_v17_r10 import parser,preflight


def main():
    if SETUP.exists():raise RuntimeError('Preserve prepared namespace')
    base=ROOT/'artifacts/framework/multitask_r10/bin_executor_v16_setup'
    old=json.loads((base/'frozen_inputs.json').read_text());verify_inputs(ROOT,old['files'])
    for launch in ([sys.executable],['bash','scripts/run_local_sim.sh']):
        subprocess.run(launch+['-m','unittest','discover','-s','tests','-p','test_all_phase_collision_diagnostic_v17.py'],cwd=ROOT,check=True)
    c=json.loads((base/'configuration.json').read_text());original=deepcopy(c)
    c['native_metric_diagnostics']=diagnostic_binding(ROOT,c['baseline_repo'])
    assert {k:v for k,v in c.items() if k!='native_metric_diagnostics'}=={k:v for k,v in original.items() if k!='native_metric_diagnostics'}
    SETUP.mkdir();atomic_json(SETUP/'configuration.json',c)
    atomic_json(SETUP/'manifest.json',json.loads((base/'manifest.json').read_text()))
    files=dict(old['files'])
    paths=[Path(__file__),ROOT/'scripts/run_bin_TRAIN_v17_r10.py',ROOT/'scripts/run_geometry_bin_v17_r10.py',
        ROOT/'our_mani/multitask/bin_native_backend_v17.py',ROOT/'our_mani/multitask/all_phase_collision_diagnostic_v17.py',
        ROOT/'tests/test_all_phase_collision_diagnostic_v17.py',SETUP/'configuration.json',SETUP/'manifest.json']
    for pointer in (c['native_metric_diagnostics']['public_collision_observer']['source'],c['native_metric_diagnostics']['public_collision_observer']['arm_API']):paths.append(ROOT/pointer['path'])
    for p in paths:
        p=p.resolve();sha=file_sha(p);assert str(p) not in files or files[str(p)]==sha;files[str(p)]=sha
    atomic_json(SETUP/'frozen_inputs.json',dict(schema='r10_bin_metric_only_v17_TRAIN_lock',files=files,
        approved_for_execution=True,native_started=False,policy_sources_and_models_unchanged_from_v16=True,TEST_read=False))
    for method in ('cv_geometry','context_geometry'):
        preflight(parser().parse_args(command(method,'episode105')[3:]))
        result=subprocess.run(command(method,'episode105')+['--check'],cwd=ROOT,capture_output=True,text=True,check=True)
        atomic_json(SETUP/(method+'_native_CPU_preflight.json'),json.loads(result.stdout))
    atomic_json(SETUP/'CPU_acceptance.json',dict(complete=True,tests_per_environment=5,
        actual_argv_preflight=True,native_started=False,policy_unchanged=True,model_fitting_performed=False))
    verify_inputs(ROOT,files)
    print(json.dumps(dict(prepared=True,frozen_inputs=len(files),metric_only=True)))


if __name__=='__main__':main()
