"""Freeze fresh current-target cup tracking, preserving the complete v271 scope."""
from datetime import datetime, timezone
from pathlib import Path
import json
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from our_mani.multitask.evaluation import atomic_json, file_sha
from our_mani.multitask.input_stat_cache_v79 import verify_inputs
from our_mani.multitask import cup_current_tracking_v377 as policy
from our_mani.multitask import reference_stop_guard_v274 as stop

B = ROOT/'artifacts/framework/multitask_r10'
SETUP = B/'cup_tracking_v380_setup'
OUT = B/'cup_tracking_v380'


def command(task, raw, method):
    return ['bash', 'scripts/run_local_sim.sh', 'scripts/run_task_executor_smoke_cup_tracking_v379_r10.py',
        '--config', str((SETUP/'configuration.json').relative_to(ROOT)),
        '--manifest', 'artifacts/framework/protocol/train_manifest.json',
        '--frozen-inputs', str((SETUP/'frozen_inputs.json').relative_to(ROOT)),
        '--task', task, '--raw-episode-id', raw, '--split', 'train', '--method', method,
        '--replica', '0', '--development-demo', '--out', str((OUT/'development_demo_only'/task/raw/method).relative_to(ROOT))]


def main():
    if SETUP.exists():
        raise FileExistsError('preserve previous preparation')
    previous = B/'cup_horizon_v271_setup'
    c = json.loads((previous/'configuration.json').read_text())
    files = dict(json.loads((previous/'frozen_inputs.json').read_text())['files'])
    verify_inputs(ROOT, files)
    for key, module in [('cup_current_tracking_v377', policy), ('reference_stop_guard_v274', stop)]:
        c[key] = {task: module.binding(ROOT, c, task) for task in policy.TASKS}
        for proof in c[key].values():
            files.update({str((ROOT/name).resolve()): sha for name, sha in proof['sources_sha256'].items()})
    c.update(execution_scope='cup_current_external_tracking_checked_stop_TRAIN_v380', formal_comparison_ready=False)
    for path in [Path(__file__), ROOT/'scripts/run_task_executor_smoke_cup_tracking_v379_r10.py',
                 ROOT/'our_mani/multitask/cup_tracking_native_v378.py', ROOT/'tests/test_cup_current_tracking_v377.py',
                 ROOT/'tests/test_reference_stop_guard_v274.py']:
        files[str(path.resolve())] = file_sha(path)
    SETUP.mkdir()
    atomic_json(SETUP/'configuration.json', c)
    files[str(SETUP/'configuration.json')] = file_sha(SETUP/'configuration.json')
    atomic_json(SETUP/'frozen_inputs.json', dict(schema='r10_cup_current_tracking_checked_stop_TRAIN_v380',
        files=files, approved_for_execution=True, native_started=False, GPU_started=False, formal_VAL_TEST=False))
    roster = json.loads((ROOT/'artifacts/framework/protocol/train_manifest.json').read_text())['episodes']
    checks = []
    try:
        for task in policy.TASKS:
            raw = next(r['raw_episode_id'] for r in roster if r['task'] == task)
            for method in ('cv_geometry', 'context_geometry'):
                argv = command(task, raw, method)
                result = subprocess.run(argv+['--check'], cwd=ROOT, capture_output=True, text=True)
                label = task+'_'+raw+'_'+method
                (SETUP/(label+'_CPU_stdout.log')).write_text(result.stdout)
                (SETUP/(label+'_CPU_stderr.log')).write_text(result.stderr)
                result.check_returncode()
                proof = json.loads(result.stdout)
                assert proof['complete'] and not proof['native_started'] and not proof['GPU_started']
                atomic_json(SETUP/(label+'_CPU_preflight.json'), proof)
                checks.append(dict(task=task, raw=raw, method=method, argv=argv))
        verify_inputs(ROOT, files)
        atomic_json(SETUP/'CPU_acceptance.json', dict(complete=True, created_utc=datetime.now(timezone.utc).isoformat(),
            actual_argv_preflights=checks, current_SE3_stream_tests_each_environment=4,
            original_support_fit_acquire_author_guard_slack_horizon_and10s_unchanged=True,
            actual_window_and_next_reference_stop_checked=True,
            native_started=False, GPU_started=False, formal_VAL_TEST=False))
    except BaseException as error:
        atomic_json(SETUP/'preparation_failed.json', dict(type=type(error).__name__, message=str(error),
            native_started=False, GPU_started=False))
        raise
    print(json.dumps(dict(prepared=True, frozen_inputs=len(files), CPU_checks=len(checks), native_started=False)), flush=True)


if __name__ == '__main__':
    main()
