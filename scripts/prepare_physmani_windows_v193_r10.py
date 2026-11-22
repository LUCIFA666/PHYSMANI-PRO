"""Original ordinary-bin baseline, with required read-only all-stage motion metrics."""
from pathlib import Path
from datetime import datetime, timezone
import json
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from our_mani.multitask.evaluation import atomic_json, file_sha
from our_mani.multitask.input_stat_cache_v79 import verify_inputs
from our_mani.multitask.native_window_observer_v191 import binding

B = ROOT/'artifacts/framework/multitask_r10'
SETUP = B/'physmani_windows_v193_setup'
OUT = B/'physmani_windows_v193'
TASK = 'put_rubbish_in_moving_bin'


def command(raw):
    return ['bash', 'scripts/run_aligned_gpu.sh', 'scripts/run_task_executor_smoke_physmani_v192_r10.py',
        '--config', str((SETUP/'configuration.json').relative_to(ROOT)),
        '--manifest', 'artifacts/framework/protocol/train_manifest.json',
        '--frozen-inputs', str((SETUP/'frozen_inputs.json').relative_to(ROOT)),
        '--task', TASK, '--raw-episode-id', raw, '--split', 'train', '--method', 'physmani',
        '--replica', '0', '--development-demo',
        '--out', str((OUT/'development_demo_only'/TASK/raw/'physmani').relative_to(ROOT))]


def main():
    if SETUP.exists(): raise FileExistsError('preserve previous preparation')
    c = json.loads((B/'task_executor_smoke_v69_setup/configuration.json').read_text())
    files = dict(json.loads((B/'joint16_block_reference_v189_setup/frozen_inputs.json').read_text())['files'])
    c.update(baseline_actual_window_observer_v191=binding(ROOT),
        execution_scope='original_PhysMani_fixed_TRAIN_bin_readonly_actual_windows_v193', formal_comparison_ready=False)
    for p in [Path(__file__), ROOT/'scripts/run_task_executor_smoke_physmani_v192_r10.py',
            ROOT/'our_mani/multitask/native_window_observer_v191.py', ROOT/'tests/test_native_window_observer_v191.py']:
        files[str(p)] = file_sha(p)
    verify_inputs(ROOT, files)
    SETUP.mkdir()
    atomic_json(SETUP/'configuration.json', c)
    files[str(SETUP/'configuration.json')] = file_sha(SETUP/'configuration.json')
    atomic_json(SETUP/'frozen_inputs.json', dict(schema='r10_original_PhysMani_readonly_actual_windows_TRAIN_v193_lock',
        files=files, approved_for_execution=True, native_started=False, GPU_started=False, formal_VAL_TEST=False))
    roster = json.loads((ROOT/'artifacts/framework/protocol/train_manifest.json').read_text())['episodes']
    raw = next(r['raw_episode_id'] for r in roster if r['task'] == TASK)
    argv = command(raw)
    try:
        r = subprocess.run(argv+['--check'], cwd=ROOT, capture_output=True, text=True)
        (SETUP/'CPU_stdout.log').write_text(r.stdout)
        (SETUP/'CPU_stderr.log').write_text(r.stderr)
        r.check_returncode()
        proof = json.loads(r.stdout)
        assert proof['complete'] and not proof['native_started'] and not proof['GPU_started']
        atomic_json(SETUP/'CPU_preflight.json', proof)
        verify_inputs(ROOT, files)
        atomic_json(SETUP/'CPU_acceptance.json', dict(complete=True, created_utc=datetime.now(timezone.utc).isoformat(),
            actual_argv_preflights=[dict(task=TASK, raw=raw, method='physmani', argv=argv)],
            read_only_actual_window_tests_each_environment=4,
            original_checkpoint_neural_policy_RML_actuator_unchanged=True,
            reason_for_new_original_source_smoke='previous baseline record has no per-counted-window actual joint acceleration evidence',
            native_started=False, GPU_started=False, formal_VAL_TEST=False))
    except BaseException as error:
        atomic_json(SETUP/'preparation_failed.json', dict(type=type(error).__name__, message=str(error), native_started=False, GPU_started=False))
        raise
    print(json.dumps(dict(prepared=True, frozen_inputs=len(files), CPU_checks=1, raw=raw, native_started=False)), flush=True)


if __name__ == '__main__': main()
