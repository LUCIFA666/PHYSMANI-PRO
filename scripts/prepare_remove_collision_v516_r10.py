"""One fixed first-source TRAIN Context diagnostic; unchanged remove policy."""
from pathlib import Path
import json
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT),str(ROOT/'scripts')]
from our_mani.multitask.remove_approach_collision_v514 import binding, SOURCES
from our_mani.multitask.evaluation import atomic_json, file_sha
from our_mani.multitask.input_stat_cache_v79 import verify_inputs
from scripts.run_native_resource_owner_v19_r10 import validate_request


def main():
    base = ROOT/'artifacts/framework/multitask_r10'
    old, setup = base/'remove_terminal_v472_setup', base/'remove_collision_v516_setup'
    if setup.exists():
        raise FileExistsError('preserve prior diagnostic setup')
    task, method = 'remove_cups_from_rotating_frame','context_geometry'
    rows = [r for r in json.loads((old/'CPU_acceptance.json').read_text())['actual_argv_preflights']
            if r['task']==task and r['method']==method]
    if len(rows)!=1:
        raise ValueError('same original first TRAIN source, independent of outcomes')
    row = rows[0]
    config = json.loads((old/(task+'_configuration.json')).read_text())
    config['remove_approach_collision_v514'] = binding(ROOT,config,task)
    config.update(execution_scope='TRAIN_only_read_only_public_collision_diagnostic_v516',
                  formal_comparison_ready=False)
    files = json.loads((old/'frozen_inputs.json').read_text())['files']
    for name in (*SOURCES,'scripts/run_task_executor_smoke_remove_collision_v515_r10.py',
                 'scripts/prepare_remove_collision_v516_r10.py'):
        files[str(ROOT/name)] = file_sha(ROOT/name)
    api = config['remove_approach_collision_v514']['native_arm_static_API']
    files[api['path']] = api['sha256']
    verify_inputs(ROOT,files)
    setup.mkdir()
    path = setup/(task+'_configuration.json')
    atomic_json(path,config)
    files[str(path)] = file_sha(path)
    atomic_json(setup/'frozen_inputs.json',dict(files=files,approved_for_execution=True,
                                              formal_VAL_TEST=False))
    argv = list(row['argv'])
    argv[2] = 'scripts/run_task_executor_smoke_remove_collision_v515_r10.py'
    argv[argv.index('--config')+1] = str(path.relative_to(ROOT))
    argv[argv.index('--frozen-inputs')+1] = str((setup/'frozen_inputs.json').relative_to(ROOT))
    argv[argv.index('--out')+1] = str((base/'remove_collision_v516'/'development_demo_only'/
                                    task/row['raw']/method).relative_to(ROOT))
    validate_request(dict(schema='r10_owned_native_TRAIN_request_v19',kind='executor_smoke',
        job_id='remove_collision_v516_'+task+'_'+row['raw']+'_context',argv=argv,use_gpu=False,
        timeout_seconds=1800,source_sha256=file_sha(ROOT/argv[2])))
    result = subprocess.run(argv+['--check'],cwd=ROOT,text=True,capture_output=True)
    if result.returncode:
        atomic_json(setup/'CPU_failed.json',dict(stdout=result.stdout,stderr=result.stderr,
                                               native_started=False,automatic_retry=False))
    result.check_returncode()
    proof = json.loads(result.stdout)
    if not proof['complete'] or proof['native_started'] or proof['GPU_started']:
        raise ValueError('CPU-only actual native entrypoint preflight required')
    atomic_json(setup/'CPU_preflight.json',proof)
    verify_inputs(ROOT,files)
    atomic_json(setup/'CPU_acceptance.json',dict(complete=True,
        actual_argv_preflights=[dict(row,argv=argv)],original_fixed_TRAIN_source=True,
        policy_and_models_unchanged=True,native_started=False,formal_VAL_TEST=False))
    print(json.dumps(dict(prepared=True,checks=1,policy_commands_unchanged=True)))


if __name__ == '__main__':
    main()
