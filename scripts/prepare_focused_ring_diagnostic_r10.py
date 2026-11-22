"""Freeze one necessary metric-only replay of the inserted-phase collision."""
from pathlib import Path
import json
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.run_task_executor_smoke_focused_collision_r10 import binding,SOURCES
from scripts.run_native_resource_owner_v19_r10 import validate_request
from our_mani.multitask.evaluation import atomic_json,file_sha
from our_mani.multitask.input_stat_cache_v79 import verify_inputs


def main():
    base=ROOT/'artifacts/framework/multitask_r10';old=base/'registered_tasks_v408_setup';setup=base/'focused_ring_v415_setup'
    if setup.exists():raise FileExistsError('preserve diagnostic replay')
    row=next(r for r in json.loads((old/'CPU_acceptance.json').read_text())['actual_argv_preflights'] if
        r['task']=='insert_onto_rotating_peg' and r['method']=='context_geometry')
    argv=list(row['argv']);c=json.loads((ROOT/argv[argv.index('--config')+1]).read_text())
    c['focused_collision_diagnostic']=binding(c,row['task'])
    files=json.loads((old/'frozen_inputs.json').read_text())['files']
    for p in (*SOURCES,'scripts/prepare_focused_ring_diagnostic_r10.py'):files[str(ROOT/p)]=file_sha(ROOT/p)
    verify_inputs(ROOT,files);setup.mkdir();atomic_json(setup/'configuration.json',c)
    files[str(setup/'configuration.json')]=file_sha(setup/'configuration.json')
    atomic_json(setup/'frozen_inputs.json',dict(files=files,approved_for_execution=True,
        schema='r10_focused415_lock',formal_VAL_TEST=False))
    argv[2]='scripts/run_task_executor_smoke_focused_collision_r10.py'
    argv[argv.index('--config')+1]=str((setup/'configuration.json').relative_to(ROOT))
    argv[argv.index('--frozen-inputs')+1]=str((setup/'frozen_inputs.json').relative_to(ROOT))
    argv[argv.index('--out')+1]=str((base/'focused_ring_v415'/'development_demo_only'/row['task']/row['raw']/row['method']).relative_to(ROOT))
    request=dict(schema='r10_owned_native_TRAIN_request_v19',kind='executor_smoke',
        job_id='focused_ring_v415_'+row['task']+'_'+row['raw']+'_context',argv=argv,use_gpu=False,
        timeout_seconds=1800,source_sha256=file_sha(ROOT/argv[2]))
    validate_request(request)
    result=subprocess.run(argv+['--check'],cwd=ROOT,text=True,capture_output=True)
    if result.returncode:atomic_json(setup/'CPU_failed.json',dict(stdout=result.stdout,stderr=result.stderr,native_started=False))
    result.check_returncode();proof=json.loads(result.stdout)
    assert proof['complete'] and not proof['native_started'] and not proof['GPU_started']
    atomic_json(setup/'CPU_preflight.json',proof)
    atomic_json(setup/'CPU_acceptance.json',dict(complete=True,actual_argv_preflights=[dict(row,argv=argv)],
        original_fixed_TRAIN_sources=True,native_started=False,formal_VAL_TEST=False))
    print(json.dumps(dict(prepared=True,checks=1,frozen_inputs=len(files))),flush=True)


if __name__=='__main__':main()
