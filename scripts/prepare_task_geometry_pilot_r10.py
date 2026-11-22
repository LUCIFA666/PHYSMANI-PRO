"""Freeze a common executor pilot with preserved bin geometry and guarded press."""
from pathlib import Path
import json
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from our_mani.multitask.task_geometry_executor_native import binding, SOURCES
from our_mani.multitask.evaluation import atomic_json, file_sha
from our_mani.multitask.input_stat_cache_v79 import verify_inputs
from scripts.run_native_resource_owner_v19_r10 import validate_request


def main():
    base = ROOT/'artifacts/framework/multitask_r10'
    setup = base/'task_geometry_v407_setup'
    if setup.exists():
        raise FileExistsError('preserve prior preparation')
    files, cohorts = {}, []
    for label in ('continuous_buttons_v400', 'continuous_bin_v400'):
        old = base/(label+'_setup')
        config = json.loads((old/'configuration.json').read_text())
        config.pop('continuous_executor', None)
        checks = json.loads((old/'CPU_acceptance.json').read_text())['actual_argv_preflights']
        tasks = sorted({r['task'] for r in checks})
        config['task_geometry_executor'] = {t:binding(ROOT, config, t) for t in tasks}
        config.update(execution_scope='TRAIN_only_original_task_geometry_and_guarded_press_v407', formal_comparison_ready=False)
        cohorts.append((label, config, checks))
        for name,sha in json.loads((old/'frozen_inputs.json').read_text())['files'].items():
            if name in files and files[name] != sha:
                raise ValueError('inherited input bytes differ')
            files[name] = sha
    for name in (*SOURCES, 'scripts/run_task_executor_smoke_task_geometry_r10.py',
                 'scripts/prepare_task_geometry_pilot_r10.py',
                 'tests/test_receding_task_transit.py', 'tests/test_button_standoff.py'):
        files[str(ROOT/name)] = file_sha(ROOT/name)
    verify_inputs(ROOT, files)
    setup.mkdir()
    for label, config, _ in cohorts:
        path = setup/(label+'_configuration.json')
        atomic_json(path, config)
        files[str(path)] = file_sha(path)
    atomic_json(setup/'frozen_inputs.json', dict(files=files, approved_for_execution=True,
        schema='r10_task_geometry407_lock', formal_VAL_TEST=False))
    requests, checks = [], []
    for label, config, old_checks in cohorts:
        for row in old_checks:
            argv = list(row['argv'])
            argv[2] = 'scripts/run_task_executor_smoke_task_geometry_r10.py'
            argv[argv.index('--config')+1] = str((setup/(label+'_configuration.json')).relative_to(ROOT))
            argv[argv.index('--frozen-inputs')+1] = str((setup/'frozen_inputs.json').relative_to(ROOT))
            argv[argv.index('--out')+1] = str((base/'task_geometry_v407'/'development_demo_only'/
                row['task']/row['raw']/row['method']).relative_to(ROOT))
            request = dict(schema='r10_owned_native_TRAIN_request_v19',
                job_id='task_geometry_v407_'+row['task']+'_'+row['raw']+'_'+row['method'].split('_')[0],
                kind='executor_smoke', argv=argv, use_gpu=False, timeout_seconds=1800,
                source_sha256=file_sha(ROOT/argv[2]))
            validate_request(request)
            result = subprocess.run(argv+['--check'], cwd=ROOT, text=True, capture_output=True)
            if result.returncode:
                atomic_json(setup/'preparation_failed.json', dict(task=row['task'], method=row['method'],
                    stdout=result.stdout, stderr=result.stderr, native_started=False))
            result.check_returncode()
            proof = json.loads(result.stdout)
            assert proof['complete'] and not proof['native_started'] and not proof['GPU_started']
            atomic_json(setup/(request['job_id']+'_CPU_preflight.json'), proof)
            requests.append(request)
            checks.append(dict(task=row['task'],raw=row['raw'],method=row['method'],argv=argv))
            print(json.dumps(dict(CPU_pass=len(checks), total=6, task=row['task'], method=row['method'])), flush=True)
    verify_inputs(ROOT, files)
    atomic_json(setup/'CPU_acceptance.json', dict(complete=True, actual_argv_preflights=checks,
        original_fixed_TRAIN_sources=True, native_started=False, formal_VAL_TEST=False))
    print(json.dumps(dict(prepared=True, checks=len(checks), frozen_inputs=len(files))), flush=True)


if __name__ == '__main__':
    main()
