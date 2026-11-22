"""Freeze one read-only collision diagnostic of the unchanged RML high-bin policy."""
from pathlib import Path
import json
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from our_mani.multitask.evaluation import atomic_json,file_sha
from our_mani.multitask.input_stat_cache_v79 import verify_inputs
from scripts.run_task_executor_smoke_public_replay_v578_r10 import EXTRA_SOURCES


def main():
    base=ROOT/'artifacts/framework/multitask_r10';old=base/'physmani_skills_v536_setup';setup=base/'public_replay_v579_setup'
    if setup.exists():raise FileExistsError('preserve every diagnostic attempt')
    rows=json.loads((old/'CPU_acceptance.json').read_text())['actual_argv_preflights']
    row=next(r for r in rows if r['task']=='beat_the_rotating_buzz' and r['method']=='context_geometry')
    config=json.loads((ROOT/row['argv'][row['argv'].index('--config')+1]).read_text())
    config['public_frontend_trace_v566']=dict(policy_decisions_modified=False,extra_camera_or_physics_calls=0)
    files=json.loads((old/'frozen_inputs.json').read_text())['files']
    for n in (*EXTRA_SOURCES,'scripts/prepare_public_replay_v579_r10.py'):
        files[str(ROOT/n)]=file_sha(ROOT/n)
    verify_inputs(ROOT,files);setup.mkdir();atomic_json(setup/'configuration.json',config)
    files[str(setup/'configuration.json')]=file_sha(setup/'configuration.json')
    atomic_json(setup/'frozen_inputs.json',dict(files=files,approved_for_execution=True,formal_VAL_TEST=False))
    argv=list(row['argv']);argv[2]='scripts/run_task_executor_smoke_public_replay_v578_r10.py'
    argv[argv.index('--config')+1]=str((setup/'configuration.json').relative_to(ROOT))
    argv[argv.index('--frozen-inputs')+1]=str((setup/'frozen_inputs.json').relative_to(ROOT))
    argv[argv.index('--out')+1]=str((base/'public_replay_v579'/'development_demo_only'/row['task']/row['raw']/row['method']).relative_to(ROOT))
    r=subprocess.run(argv+['--check'],cwd=ROOT,text=True,capture_output=True)
    if r.returncode:
        atomic_json(setup/'CPU_failed.json',dict(stdout=r.stdout,stderr=r.stderr,automatic_retry=False,native_started=False));r.check_returncode()
    proof=json.loads(r.stdout)
    if not proof['complete'] or proof['native_started'] or proof['GPU_started']:raise ValueError('actual CPU-only argv proof required')
    atomic_json(setup/'CPU_preflight.json',proof);verify_inputs(ROOT,files)
    atomic_json(setup/'CPU_acceptance.json',dict(complete=True,actual_argv_preflights=[dict(row,argv=argv)],
        policy_commands_unchanged=True,native_started=False,formal_VAL_TEST=False))
    print(json.dumps(dict(complete=True,checks=1,first_original_source_retained=True)),flush=True)


if __name__=='__main__':main()
