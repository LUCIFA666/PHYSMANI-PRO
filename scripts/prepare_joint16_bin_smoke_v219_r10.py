"""One fixed Context-only smoke per buzz task with the actual shared16 model."""
from pathlib import Path
from datetime import datetime, timezone
import json
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from our_mani.multitask.evaluation import atomic_json, file_sha
from our_mani.multitask.input_stat_cache_v79 import verify_inputs
from our_mani.multitask.task_actor_backend_v69 import persistent_reference_continuation_identity
from our_mani.multitask.interaction import InteractionLibrary, CandidateConfig, MPCConfig

B = ROOT/'artifacts/framework/multitask_r10'
SETUP = B/'joint16_bin_smoke_v219_setup'
OUT = B/'joint16_bin_smoke_v219'
TASKS = ('put_rubbish_in_moving_bin',)


def command(task, raw):
    return ['bash', 'scripts/run_local_sim.sh', 'scripts/run_task_executor_smoke_joint16_v174_r10.py',
        '--config', str((SETUP/'configuration.json').relative_to(ROOT)),
        '--manifest', 'artifacts/framework/protocol/train_manifest.json',
        '--frozen-inputs', str((SETUP/'frozen_inputs.json').relative_to(ROOT)),
        '--task', task, '--raw-episode-id', raw, '--split', 'train', '--method', 'context_geometry',
        '--replica', '0', '--development-demo',
        '--out', str((OUT/'development_demo_only'/task/raw/'context_geometry').relative_to(ROOT))]


def main():
    if SETUP.exists(): raise FileExistsError('preserve previous preparation')
    previous = B/'task_executor_smoke_v69_setup'
    c = json.loads((previous/'configuration.json').read_text())
    files = dict(json.loads((previous/'frozen_inputs.json').read_text())['files'])
    shared = B/'cup_transport_v163_setup'
    for name, checksum in json.loads((shared/'frozen_inputs.json').read_text())['files'].items():
        if name in files and files[name] != checksum: raise ValueError('frozen source hashes differ')
        files[name] = checksum
    verify_inputs(ROOT, files)
    original = json.loads((shared/'configuration.json').read_text())
    c['context_checkpoint'] = original['context_checkpoint']
    c['joint16_development_checkpoint_v157'] = original['joint16_development_checkpoint_v157']
    c.update(execution_scope='shared16_bin_fixed_first_four_TRAIN_v219', formal_comparison_ready=False,
        optional_native_force_contact_diagnostics_enabled=False)
    for task in TASKS:
        library = InteractionLibrary.load(ROOT/c['learned_artifacts'][task]['interaction']['path'])
        identity = persistent_reference_continuation_identity(ROOT, c, task, 'context', library,
            CandidateConfig(**c['candidate_config']), MPCConfig(**c['mpc_config']))
        c['continuation_identities'].setdefault(task, {})['context'] = identity['sha256']
    paths = [Path(__file__), ROOT/'scripts/run_task_executor_smoke_joint16_v174_r10.py',
        ROOT/'our_mani/multitask/joint16_checkpoint_v172.py', ROOT/'our_mani/multitask/joint16_native_v173.py',
        ROOT/'tests/test_joint16_checkpoint_v172.py']
    for p in paths: files[str(p)] = file_sha(p)
    SETUP.mkdir()
    atomic_json(SETUP/'configuration.json', c)
    files[str(SETUP/'configuration.json')] = file_sha(SETUP/'configuration.json')
    atomic_json(SETUP/'frozen_inputs.json', dict(schema='r10_shared16_bin_fixed_first_four_TRAIN_v219_lock', files=files,
        approved_for_execution=True, native_started=False, GPU_started=False, formal_VAL_TEST=False))
    roster = json.loads((ROOT/'artifacts/framework/protocol/train_manifest.json').read_text())['episodes']
    checks = []
    try:
        for task in TASKS:
            first = [r for r in roster if r['task'] == task][:4]
            assert len(first) == 4
            for row in first:
                raw = row['raw_episode_id']
                argv = command(task, raw)
                r = subprocess.run(argv+['--check'], cwd=ROOT, capture_output=True, text=True)
                name = task+'_'+raw
                (SETUP/(name+'_CPU_stdout.log')).write_text(r.stdout)
                (SETUP/(name+'_CPU_stderr.log')).write_text(r.stderr)
                r.check_returncode()
                proof = json.loads(r.stdout)
                assert proof['complete'] and not proof['native_started'] and not proof['GPU_started']
                atomic_json(SETUP/(name+'_CPU_preflight.json'), proof)
                checks.append(dict(task=task, raw=raw, method='context_geometry', argv=argv))
        verify_inputs(ROOT, files)
        atomic_json(SETUP/'CPU_acceptance.json', dict(complete=True, created_utc=datetime.now(timezone.utc).isoformat(),
            actual_argv_preflights=checks, shared_checkpoint_tests_each_environment=5,
            original_policy_dispatch_motion_gates_unchanged=True,
            same_original_bin_executor_with_new_shared16_checkpoint=True,
            replacement_gate='actual completed shared16, exact1600 originalTRAIN; all firstfour parents independently selected without outcome filtering',
            optional_native_force_contact_debug_omitted=True, actual_window_metrics_retained=True,
            native_started=False, GPU_started=False, formal_VAL_TEST=False))
    except BaseException as error:
        atomic_json(SETUP/'preparation_failed.json', dict(type=type(error).__name__, message=str(error),
            native_started=False, GPU_started=False))
        raise
    print(json.dumps(dict(prepared=True, frozen_inputs=len(files), CPU_checks=len(checks), native_started=False)), flush=True)


if __name__ == '__main__': main()
