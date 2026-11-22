"""Freeze one read-only collision diagnostic of the unchanged RML high-bin policy."""
from pathlib import Path
import json
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from our_mani.multitask.evaluation import atomic_json,file_sha
from our_mani.multitask.input_stat_cache_v79 import verify_inputs
from our_mani.multitask.all_phase_collision_diagnostic_v17 import binding


def main():
    base=ROOT/'artifacts/framework/multitask_r10';old=base/'physmani_skills_v536_setup';setup=base/'remove_collision_v600_setup'
    if setup.exists():raise FileExistsError('preserve every diagnostic attempt')
    rows=json.loads((old/'CPU_acceptance.json').read_text())['actual_argv_preflights']
    row=next(r for r in rows if r['task']=='remove_cups_from_rotating_frame' and r['method']=='context_geometry')
    config=json.loads((ROOT/row['argv'][row['argv'].index('--config')+1]).read_text())
    config['RML_public_collision_diagnostics_v598']=binding(ROOT,config['baseline_repo'])
    files=json.loads((old/'frozen_inputs.json').read_text())['files']
    for n in ('scripts/run_task_executor_smoke_remove_collision_v599_r10.py','scripts/prepare_remove_collision_v600_r10.py',
              'our_mani/multitask/all_phase_collision_diagnostic_v17.py'):
        files[str(ROOT/n)]=file_sha(ROOT/n)
    verify_inputs(ROOT,files);setup.mkdir();atomic_json(setup/'configuration.json',config)
    files[str(setup/'configuration.json')]=file_sha(setup/'configuration.json')
    atomic_json(setup/'frozen_inputs.json',dict(files=files,approved_for_execution=True,formal_VAL_TEST=False))
    argv=list(row['argv']);argv[2]='scripts/run_task_executor_smoke_remove_collision_v599_r10.py'
    argv[argv.index('--config')+1]=str((setup/'configuration.json').relative_to(ROOT))
    argv[argv.index('--frozen-inputs')+1]=str((setup/'frozen_inputs.json').relative_to(ROOT))
    argv[argv.index('--out')+1]=str((base/'remove_collision_v600'/'development_demo_only'/row['task']/row['raw']/row['method']).relative_to(ROOT))
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
