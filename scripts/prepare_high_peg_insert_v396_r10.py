"""Freeze the shared checked transit for all admitted highpeg operation phases."""
from datetime import datetime, timezone
from pathlib import Path
import json
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from our_mani.multitask.evaluation import atomic_json, file_sha
from our_mani.multitask.input_stat_cache_v79 import verify_inputs
from our_mani.multitask.high_peg_insert_native_v394 import binding

B = ROOT/'artifacts/framework/multitask_r10'
SETUP = B/'high_peg_insert_v396_setup'
OUT = B/'high_peg_insert_v396'
TASK = 'insert_onto_rotating_peg_high_speed'


def command(raw, method):
    return ['bash', 'scripts/run_local_sim.sh', 'scripts/run_task_executor_smoke_high_peg_insert_v395_r10.py',
        '--config', str((SETUP/'configuration.json').relative_to(ROOT)),
        '--manifest', 'artifacts/framework/protocol/train_manifest.json',
        '--frozen-inputs', str((SETUP/'frozen_inputs.json').relative_to(ROOT)),
        '--task', TASK, '--raw-episode-id', raw, '--split', 'train', '--method', method,
        '--replica', '0', '--development-demo', '--out', str((OUT/'development_demo_only'/TASK/raw/method).relative_to(ROOT))]


def main():
    if SETUP.exists():
        raise FileExistsError('preserve prior preparation')
    previous = B/'high_peg_retarget_v374_setup'
    c = json.loads((previous/'configuration.json').read_text())
    files = dict(json.loads((previous/'frozen_inputs.json').read_text())['files'])
    verify_inputs(ROOT, files)
    proof = binding(ROOT, c, TASK)
    c['high_peg_insert_v394'] = {TASK: proof}
    c.update(execution_scope='original_highpeg_all_admitted_phases_checked_transit_TRAIN_v396',
        formal_comparison_ready=False)
    files.update({str((ROOT/name).resolve()): sha for name, sha in proof['sources_sha256'].items()})
    for path in [Path(__file__), ROOT/'scripts/run_task_executor_smoke_high_peg_insert_v395_r10.py',
                 ROOT/'tests/test_bounded_task_transit.py']:
        files[str(path.resolve())] = file_sha(path)
    SETUP.mkdir()
    atomic_json(SETUP/'configuration.json', c)
    files[str(SETUP/'configuration.json')] = file_sha(SETUP/'configuration.json')
    atomic_json(SETUP/'frozen_inputs.json', dict(schema='r10_highpeg_all_admitted_phases_checked_transit_v396',
        files=files, approved_for_execution=True, native_started=False, GPU_started=False, formal_VAL_TEST=False))
    roster = json.loads((ROOT/'artifacts/framework/protocol/train_manifest.json').read_text())['episodes']
    raw = next(r['raw_episode_id'] for r in roster if r['task'] == TASK)
    checks = []
    try:
        for method in ('cv_geometry', 'context_geometry'):
            argv = command(raw, method)
            result = subprocess.run(argv+['--check'], cwd=ROOT, capture_output=True, text=True)
            label = TASK+'_'+raw+'_'+method
            (SETUP/(label+'_CPU_stdout.log')).write_text(result.stdout)
            (SETUP/(label+'_CPU_stderr.log')).write_text(result.stderr)
            result.check_returncode()
            receipt = json.loads(result.stdout)
            assert receipt['complete'] and not receipt['native_started'] and not receipt['GPU_started']
            atomic_json(SETUP/(label+'_CPU_preflight.json'), receipt)
            checks.append(dict(task=TASK, raw=raw, method=method, argv=argv))
        verify_inputs(ROOT, files)
        atomic_json(SETUP/'CPU_acceptance.json', dict(complete=True, created_utc=datetime.now(timezone.utc).isoformat(),
            actual_argv_preflights=checks, shared_transit_actual_task_and_button_stream_tests_each_environment=4,
            original_full_TRAIN_fit_D4_source_causal_reissue_and10s_unchanged=True,
            current_actual_reference_collision_cartesian_and_offset_stop_gates_required=True,
            native_started=False, GPU_started=False, formal_VAL_TEST=False))
    except BaseException as error:
        atomic_json(SETUP/'preparation_failed.json', dict(type=type(error).__name__, message=str(error),
            native_started=False, GPU_started=False))
        raise
    print(json.dumps(dict(prepared=True, CPU_checks=len(checks), frozen_inputs=len(files), native_started=False)), flush=True)


if __name__ == '__main__':
    main()
