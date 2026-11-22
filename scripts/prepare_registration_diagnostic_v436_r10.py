"""One unchanged fixed TRAIN case; only existing public diagnostic fields added."""
from pathlib import Path
import json
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from scripts.run_task_executor_smoke_registration_diagnostic_r10 import SOURCES,binding
from our_mani.multitask.evaluation import atomic_json,file_sha
from our_mani.multitask.input_stat_cache_v79 import verify_inputs
from scripts.run_native_resource_owner_v19_r10 import validate_request


def main():
    base=ROOT/'artifacts/framework/multitask_r10';old=base/'remove_continuous_v431_setup';setup=base/'registration_diagnostic_v436_setup'
    if setup.exists():raise FileExistsError('preserve previous preparation')
    task='remove_cups_from_rotating_frame_high_speed'
    c=json.loads((old/(task+'_configuration.json')).read_text());c['public_registration_diagnostic']=binding()
    files=json.loads((old/'frozen_inputs.json').read_text())['files']
    for p in (*SOURCES,'scripts/prepare_registration_diagnostic_v436_r10.py','tests/test_public_registration_diagnostic.py'):
        files[str(ROOT/p)]=file_sha(ROOT/p)
    verify_inputs(ROOT,files);setup.mkdir();atomic_json(setup/'configuration.json',c)
    files[str(setup/'configuration.json')]=file_sha(setup/'configuration.json')
    atomic_json(setup/'frozen_inputs.json',dict(files=files,approved_for_execution=True,formal_VAL_TEST=False))
    rows=[r for r in json.loads((old/'CPU_acceptance.json').read_text())['actual_argv_preflights']
        if r['task']==task and r['method']=='cv_geometry']
    if len(rows)!=1:raise ValueError('one same original fixed source')
    row=rows[0];argv=list(row['argv']);argv[2]='scripts/run_task_executor_smoke_registration_diagnostic_r10.py'
    argv[argv.index('--config')+1]=str((setup/'configuration.json').relative_to(ROOT))
    argv[argv.index('--frozen-inputs')+1]=str((setup/'frozen_inputs.json').relative_to(ROOT))
    argv[argv.index('--out')+1]=str((base/'registration_diagnostic_v436'/'development_demo_only'/task/row['raw']/row['method']).relative_to(ROOT))
    validate_request(dict(schema='r10_owned_native_TRAIN_request_v19',kind='executor_smoke',job_id='registration_diagnostic_v436',argv=argv,use_gpu=False,
        timeout_seconds=1800,source_sha256=file_sha(ROOT/argv[2])))
    r=subprocess.run(argv+['--check'],cwd=ROOT,text=True,capture_output=True)
    if r.returncode:atomic_json(setup/'CPU_failed.json',dict(stdout=r.stdout,stderr=r.stderr,native_started=False))
    r.check_returncode();proof=json.loads(r.stdout)
    assert proof['complete'] and not proof['native_started'] and not proof['GPU_started']
    atomic_json(setup/'CPU_preflight.json',proof);verify_inputs(ROOT,files)
    atomic_json(setup/'CPU_acceptance.json',dict(complete=True,actual_argv_preflights=[dict(row,argv=argv)],formal_VAL_TEST=False))
    print(json.dumps(dict(prepared=True,checks=1,policy_changed=False)),flush=True)


if __name__=='__main__':main()
