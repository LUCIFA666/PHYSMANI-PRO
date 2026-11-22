"""Freeze exact original failed sources for the constraint authority regression."""
from pathlib import Path
import json
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from our_mani.multitask.task_constraint_executor_native import binding, SOURCES
from our_mani.multitask.evaluation import atomic_json, file_sha
from our_mani.multitask.input_stat_cache_v79 import verify_inputs
from scripts.run_native_resource_owner_v19_r10 import validate_request


def main():
    base = ROOT/'artifacts/framework/multitask_r10'
    old, setup = base/'task_geometry_v407_setup', base/'task_constraint_v409_setup'
    if setup.exists():
        raise FileExistsError('preserve previous constraint pilot')
    files = json.loads((old/'frozen_inputs.json').read_text())['files']
    checks = json.loads((old/'CPU_acceptance.json').read_text())['actual_argv_preflights']
    configs = {}
    for row in checks:
        old_path = row['argv'][row['argv'].index('--config')+1]
        if old_path not in configs:
            c = json.loads((ROOT/old_path).read_text())
            tasks = sorted(c['task_geometry_executor'])
            c['task_constraint_executor'] = {t:binding(ROOT,c,t) for t in tasks}
            c['execution_scope'] = 'TRAIN_only_consistent_constraint_authority_v409'
            configs[old_path] = c
    for p in (*SOURCES,'scripts/run_task_executor_smoke_constraint_r10.py',
              'scripts/prepare_task_constraint_pilot_r10.py','tests/test_reference_constraint_authority.py'):
        files[str(ROOT/p)] = file_sha(ROOT/p)
    verify_inputs(ROOT,files); setup.mkdir()
    new_paths = {}
    for old_path,c in configs.items():
        path = setup/Path(old_path).name
        atomic_json(path,c); files[str(path)] = file_sha(path); new_paths[old_path] = path
    atomic_json(setup/'frozen_inputs.json',dict(files=files,approved_for_execution=True,
        schema='r10_task_constraint409_lock',formal_VAL_TEST=False))
    accepted = []
    for row in checks:
        argv = list(row['argv']); old_path = argv[argv.index('--config')+1]
        argv[2] = 'scripts/run_task_executor_smoke_constraint_r10.py'
        argv[argv.index('--config')+1] = str(new_paths[old_path].relative_to(ROOT))
        argv[argv.index('--frozen-inputs')+1] = str((setup/'frozen_inputs.json').relative_to(ROOT))
        argv[argv.index('--out')+1] = str((base/'task_constraint_v409'/'development_demo_only'/
            row['task']/row['raw']/row['method']).relative_to(ROOT))
        job = 'task_constraint_v409_'+row['task']+'_'+row['raw']+'_'+row['method'].split('_')[0]
        request = dict(schema='r10_owned_native_TRAIN_request_v19',job_id=job,
            kind='executor_smoke',argv=argv,use_gpu=False,timeout_seconds=1800,
            source_sha256=file_sha(ROOT/argv[2]))
        validate_request(request)
        result = subprocess.run(argv+['--check'],cwd=ROOT,text=True,capture_output=True)
        if result.returncode:
            atomic_json(setup/'preparation_failed.json',dict(task=row['task'],method=row['method'],
                stdout=result.stdout,stderr=result.stderr,native_started=False))
        result.check_returncode(); proof = json.loads(result.stdout)
        assert proof['complete'] and not proof['native_started'] and not proof['GPU_started']
        atomic_json(setup/(job+'_CPU_preflight.json'),proof)
        accepted.append(dict(task=row['task'],raw=row['raw'],method=row['method'],argv=argv))
        print(json.dumps(dict(CPU_pass=len(accepted),total=len(checks),task=row['task'],method=row['method'])),flush=True)
    verify_inputs(ROOT,files)
    atomic_json(setup/'CPU_acceptance.json',dict(complete=True,actual_argv_preflights=accepted,
        original_fixed_TRAIN_sources=True,native_started=False,formal_VAL_TEST=False))
    print(json.dumps(dict(prepared=True,checks=len(accepted),frozen_inputs=len(files))),flush=True)


if __name__ == '__main__':
    main()
