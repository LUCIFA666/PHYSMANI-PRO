"""Freeze one source-controlled pilot without copying task execution loops."""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import json
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from our_mani.multitask.continuous_executor_native import binding, BUTTONS, BINS
from our_mani.multitask.evaluation import atomic_json, file_sha
from our_mani.multitask.input_stat_cache_v79 import verify_inputs

BASE = ROOT/'artifacts/framework/multitask_r10'
SOURCES = ('our_mani/multitask/continuous_transit.py',
    'our_mani/multitask/continuous_executor_native.py',
    'our_mani/multitask/bounded_task_transit.py',
    'scripts/run_continuous_executor_smoke_r10.py',
    'scripts/prepare_continuous_executor_smoke_r10.py',
    'scripts/queue_continuous_executor_smoke_r10.py',
    'tests/test_continuous_transit.py')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--label', required=True)
    parser.add_argument('--pilot', choices=('buttons', 'bin'), required=True)
    args = parser.parse_args()
    if not args.label.replace('_', '').isalnum():
        raise ValueError('simple artifact label required')
    setup, out = BASE/(args.label+'_setup'), BASE/args.label
    if setup.exists() or out.exists():
        raise FileExistsError('preserve previous pilot and input lock')
    old = BASE/('button_reference_v389_setup' if args.pilot == 'buttons' else 'bin_physical_smoke_v240_setup')
    config = json.loads((old/'configuration.json').read_text())
    files = json.loads((old/'frozen_inputs.json').read_text())['files']
    verify_inputs(ROOT, files)
    tasks = BUTTONS if args.pilot == 'buttons' else BINS
    config['continuous_executor'] = {task: binding(ROOT, config, task) for task in tasks}
    config.update(execution_scope='TRAIN_only_common_continuous_executor_'+args.label,
        formal_comparison_ready=False)
    for name in SOURCES:
        path = ROOT/name
        if str(path) in files and files[str(path)] != file_sha(path):
            raise ValueError('new policy must not mutate a prior frozen input')
        files[str(path)] = file_sha(path)
    setup.mkdir()
    atomic_json(setup/'configuration.json', config)
    files[str(setup/'configuration.json')] = file_sha(setup/'configuration.json')
    atomic_json(setup/'frozen_inputs.json', dict(schema='r10_common_continuous_executor_pilot_lock',
        files=files, approved_for_execution=True, formal_VAL_TEST=False))
    plan = json.loads((BASE/'all16_executor_joint_train_v1/plan.json').read_text())['tasks']
    checks = []
    for task in tasks:
        raw = next(row['fixed_smoke_source'] for row in plan if row['task'] == task)
        for method in ('cv_geometry', 'context_geometry'):
            argv = ['bash', 'scripts/run_local_sim.sh', 'scripts/run_continuous_executor_smoke_r10.py',
                '--config', str((setup/'configuration.json').relative_to(ROOT)),
                '--manifest', 'artifacts/framework/protocol/train_manifest.json',
                '--frozen-inputs', str((setup/'frozen_inputs.json').relative_to(ROOT)),
                '--task', task, '--raw-episode-id', raw, '--split', 'train',
                '--method', method, '--replica', '0', '--development-demo',
                '--out', str((out/'development_demo_only'/task/raw/method).relative_to(ROOT))]
            result = subprocess.run(argv+['--check'], cwd=ROOT, text=True, capture_output=True)
            if result.returncode:
                atomic_json(setup/'preparation_failed.json', dict(task=task, method=method,
                    returncode=result.returncode, stdout=result.stdout, stderr=result.stderr,
                    native_started=False))
            result.check_returncode()
            proof = json.loads(result.stdout)
            assert proof['complete'] and not proof['native_started'] and not proof['GPU_started']
            proof_name = task+'_'+raw+'_'+method+'_CPU_preflight.json'
            atomic_json(setup/proof_name, proof)
            checks.append(dict(task=task, raw=raw, method=method, argv=argv, proof=proof_name))
    verify_inputs(ROOT, files)
    atomic_json(setup/'CPU_acceptance.json', dict(complete=True,
        created_utc=datetime.now(timezone.utc).isoformat(), actual_argv_preflights=checks,
        pilot=args.pilot, label=args.label, tasks=list(tasks),
        fixed_original_TRAIN_sources=True, native_started=False, formal_VAL_TEST=False))
    print(json.dumps(dict(prepared=True, label=args.label, checks=len(checks), frozen_inputs=len(files))), flush=True)


if __name__ == '__main__':
    main()
