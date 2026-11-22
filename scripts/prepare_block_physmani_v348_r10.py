"""Freeze the first original TRAIN source of both missing block baseline routes."""
from pathlib import Path
from datetime import datetime, timezone
import json
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from our_mani.multitask.evaluation import atomic_json, file_sha
from our_mani.multitask.input_stat_cache_v79 import verify_inputs
from our_mani.multitask.block_physmani_v346 import binding
from our_mani.multitask.joint16_tracks_v31 import BLOCKS

B = ROOT / 'artifacts/framework/multitask_r10'
SETUP = B / 'block_physmani_v348_setup'
OUT = B / 'block_physmani_v348'


def main():
    if SETUP.exists():
        raise FileExistsError('preserve previous preparation')
    original = ROOT / 'config/framework/aligned_protocol_r09_baseline_v6.json'
    c = json.loads(original.read_text())
    c.update(cv_local_value='', context_local_value='', value_input_context=False,
        uncertainty_model_present=False, formal_comparison_ready=False,
        execution_scope='first original TRAIN blocks, original PhysMani V6 plus read-only windows',
        baseline_output=str(OUT / 'world_model_disabled_logs'))
    c['block_physmani_v346'] = binding(ROOT, c)
    files = dict(json.loads((B / 'physmani_windows_v194_setup/frozen_inputs.json').read_text())['files'])
    roster = ROOT / 'artifacts/framework/protocol/train_manifest.json'
    rows = json.loads(roster.read_text())['episodes']
    sources = [next(r for r in rows if r['task'] == task) for task in sorted(BLOCKS)]
    paths = [original, roster, Path(__file__), ROOT / 'scripts/run_task_executor_smoke_block_physmani_v347_r10.py',
        ROOT / 'tests/test_block_physmani_v346.py']
    paths.extend(ROOT / name for name in c['block_physmani_v346']['sources_sha256'])
    paths.extend(Path(r['path']) for r in sources)
    for p in paths:
        p = p.resolve()
        sha = file_sha(p)
        if str(p) in files and files[str(p)] != sha:
            raise ValueError('original frozen baseline dependency changed: ' + str(p))
        files[str(p)] = sha
    verify_inputs(ROOT, files)
    SETUP.mkdir()
    atomic_json(SETUP / 'configuration.json', c)
    files[str(SETUP / 'configuration.json')] = file_sha(SETUP / 'configuration.json')
    atomic_json(SETUP / 'frozen_inputs.json', dict(schema='r10_two_block_baseline_TRAIN_v348_lock',
        files=files, approved_for_execution=True, native_started=False, GPU_started=False, formal_VAL_TEST=False))
    checks = []
    try:
        for row in sources:
            task, raw = row['task'], row['raw_episode_id']
            argv = ['bash', 'scripts/run_aligned_gpu.sh', 'scripts/run_task_executor_smoke_block_physmani_v347_r10.py',
                '--config', str((SETUP / 'configuration.json').relative_to(ROOT)),
                '--manifest', 'artifacts/framework/protocol/train_manifest.json',
                '--frozen-inputs', str((SETUP / 'frozen_inputs.json').relative_to(ROOT)),
                '--task', task, '--raw-episode-id', raw, '--split', 'train', '--method', 'physmani',
                '--development-demo', '--out', str((OUT / 'development_demo_only' / task / raw / 'physmani').relative_to(ROOT))]
            r = subprocess.run(argv + ['--check'], cwd=ROOT, capture_output=True, text=True)
            name = task + '_' + raw + '_physmani'
            (SETUP / (name + '_CPU_stdout.log')).write_text(r.stdout)
            (SETUP / (name + '_CPU_stderr.log')).write_text(r.stderr)
            r.check_returncode()
            proof = json.loads(r.stdout)
            assert proof['complete'] and not proof['native_started'] and not proof['GPU_started']
            atomic_json(SETUP / (name + '_CPU_preflight.json'), proof)
            checks.append(dict(task=task, raw=raw, method='physmani', argv=argv))
        verify_inputs(ROOT, files)
        atomic_json(SETUP / 'CPU_acceptance.json', dict(complete=True,
            created_utc=datetime.now(timezone.utc).isoformat(), actual_argv_preflights=checks,
            helper_tests_each_environment=3, original_author_weights_WM_RML_commands_unchanged=True,
            original_legacy_block_reset_shared_with_geometry=True, actual_motion_is_measured=True,
            first_fixed_TRAIN_sources_only=True, outcome_filtering=False,
            native_started=False, GPU_started=False, formal_VAL_TEST=False))
    except BaseException as error:
        atomic_json(SETUP / 'preparation_failed.json', dict(type=type(error).__name__, message=str(error),
            native_started=False, GPU_started=False))
        raise
    print(json.dumps(dict(prepared=True, frozen_inputs=len(files), CPU_checks=len(checks), native_started=False)), flush=True)


if __name__ == '__main__':
    main()
