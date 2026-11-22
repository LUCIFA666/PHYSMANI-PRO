"""Freeze, check actual argv, and queue the moving-frame button ablation once."""
from pathlib import Path
import json
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from our_mani.multitask.moving_button_native import binding
from our_mani.multitask.evaluation import atomic_json, file_sha
from our_mani.multitask.input_stat_cache_v79 import verify_inputs
from our_mani.multitask.case_output_isolation_v238 import require_development_output
from scripts.run_native_resource_owner_v19_r10 import validate_request


def main():
    base = ROOT/'artifacts/framework/multitask_r10'
    old, setup = base/'continuous_buttons_v400_setup', base/'moving_button_v404_setup'
    if setup.exists():
        raise FileExistsError('preserve prior preparation')
    config = json.loads((old/'configuration.json').read_text())
    files = json.loads((old/'frozen_inputs.json').read_text())['files']
    old_checks = json.loads((old/'CPU_acceptance.json').read_text())['actual_argv_preflights']
    tasks = sorted({row['task'] for row in old_checks})
    config['moving_button'] = {task: binding(ROOT, config, task) for task in tasks}
    config.update(execution_scope='TRAIN_only_common_moving_button_v404', formal_comparison_ready=False)
    for name in ('our_mani/multitask/moving_button_native.py',
                 'our_mani/multitask/moving_reference.py', 'our_mani/multitask/executor_trace.py',
                 'scripts/run_task_executor_smoke_moving_button_r10.py',
                 'scripts/prepare_moving_button_pilot_r10.py', 'tests/test_moving_reference.py'):
        files[str(ROOT/name)] = file_sha(ROOT/name)
    verify_inputs(ROOT, files)
    setup.mkdir()
    atomic_json(setup/'configuration.json', config)
    files[str(setup/'configuration.json')] = file_sha(setup/'configuration.json')
    atomic_json(setup/'frozen_inputs.json', dict(files=files, approved_for_execution=True,
        schema='r10_moving_button_current_reference_v404_lock', formal_VAL_TEST=False))
    accepted = []
    for index, row in enumerate(old_checks):
        argv = list(row['argv'])
        argv[2] = 'scripts/run_task_executor_smoke_moving_button_r10.py'
        argv[argv.index('--config')+1] = str((setup/'configuration.json').relative_to(ROOT))
        argv[argv.index('--frozen-inputs')+1] = str((setup/'frozen_inputs.json').relative_to(ROOT))
        argv[argv.index('--out')+1] = str((base/'moving_button_v404'/'development_demo_only'/
            row['task']/row['raw']/row['method']).relative_to(ROOT))
        request = dict(schema='r10_owned_native_TRAIN_request_v19',
            job_id='moving_button_v404_'+row['task']+'_'+row['raw']+'_'+row['method'].split('_')[0],
            kind='executor_smoke', argv=argv, use_gpu=False, timeout_seconds=1800,
            source_sha256=file_sha(ROOT/argv[2]))
        validate_request(request)
        result = subprocess.run(argv+['--check'], cwd=ROOT, text=True, capture_output=True)
        if result.returncode:
            atomic_json(setup/'preparation_failed.json', dict(index=index, stderr=result.stderr,
                stdout=result.stdout, native_started=False))
        result.check_returncode()
        proof = json.loads(result.stdout)
        assert proof['complete'] and not proof['native_started'] and not proof['GPU_started']
        atomic_json(setup/(request['job_id']+'_CPU_preflight.json'), proof)
        accepted.append(request)
        print(json.dumps(dict(CPU_pass=index+1, total=len(old_checks),
            task=row['task'], method=row['method'])), flush=True)
    verify_inputs(ROOT, files)
    owner = ROOT/'runtime/framework/native_resource_owner_v401'
    launch, status = (json.loads((owner/name).read_text()) for name in ('launch.json', 'status.json'))
    if (not Path(f'/proc/{launch["pid"]}/stat').exists()
            or int(Path(f'/proc/{launch["pid"]}/stat').read_text().rsplit(')', 1)[1].split()[19]) != launch['start_ticks']
            or status['stop_requested']):
        raise ValueError('source-bound resource owner unavailable')
    existing = [json.loads(p.read_text()) for p in (owner/'requests').glob('*.json')]
    for row, request in zip(old_checks, accepted):
        output = request['argv'][request['argv'].index('--out')+1]
        path = require_development_output(ROOT, output, task=row['task'], raw=row['raw'], method=row['method'])
        if path.exists() or any(r.get('job_id') == request['job_id'] for r in existing):
            raise ValueError('pilot output or job already used')
    atomic_json(setup/'CPU_acceptance.json', dict(complete=True, requests=accepted,
        original_fixed_TRAIN_sources=True, formal_VAL_TEST=False))
    for index, request in enumerate(accepted):
        atomic_json(owner/'requests'/f'0020_{index}_{request["job_id"]}.json', request)
    atomic_json(setup/'queued_once.json', dict(queued_once=True,
        jobs=[r['job_id'] for r in accepted], automatic_retry=False, formal_VAL_TEST=False))
    print(json.dumps(dict(queued=len(accepted), frozen_inputs=len(files))), flush=True)


if __name__ == '__main__':
    main()
