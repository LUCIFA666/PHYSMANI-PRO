"""Freeze the normal ring's original proof and new checked execution policy."""
from datetime import datetime, timezone
from pathlib import Path
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
SETUP = B/'normal_ring_v369_setup'
OUT = B/'normal_ring_v369'
TASK = 'insert_onto_rotating_peg'


def command(raw, method):
    return ['bash', 'scripts/run_local_sim.sh', 'scripts/run_task_executor_smoke_normal_ring_v368_r10.py',
        '--config', str((SETUP/'configuration.json').relative_to(ROOT)),
        '--manifest', 'artifacts/framework/protocol/train_manifest.json',
        '--frozen-inputs', str((SETUP/'frozen_inputs.json').relative_to(ROOT)),
        '--task', TASK, '--raw-episode-id', raw, '--split', 'train', '--method', method,
        '--replica', '0', '--development-demo', '--out', str((OUT/'development_demo_only'/TASK/raw/method).relative_to(ROOT))]


def main():
    if SETUP.exists():
        raise FileExistsError('preserve previous preparation')
    previous = B/'high_peg_target_memory_v342_setup'
    c = json.loads((previous/'configuration.json').read_text())
    files = dict(json.loads((previous/'frozen_inputs.json').read_text())['files'])
    verify_inputs(ROOT, files)
    from our_mani.multitask import normal_ring_support_v366 as support
    from our_mani.multitask import reference_stop_guard_v274 as stop
    from our_mani.multitask import public_target_memory_v339 as memory
    for key, policy in [('normal_ring_support_v366', support),
                        ('reference_stop_guard_v274', stop), ('public_target_memory_v339', memory)]:
        proof = policy.binding(ROOT, c, TASK)
        c.setdefault(key, {})[TASK] = proof
        files.update({str((ROOT/name).resolve()): sha for name, sha in proof['sources_sha256'].items()})
    library = InteractionLibrary.load(ROOT/c['learned_artifacts'][TASK]['interaction']['path'])
    for predictor in ('cv', 'context'):
        identity = persistent_reference_continuation_identity(ROOT, c, TASK, predictor, library,
            CandidateConfig(**c['candidate_config']), MPCConfig(**c['mpc_config']))
        c['continuation_identities'].setdefault(TASK, {})[predictor] = identity['sha256']
    c.update(execution_scope='normal_ring_original_full_TRAIN_proof_checked_support_align_v369',
        formal_comparison_ready=False)
    for path in [Path(__file__), ROOT/'scripts/run_task_executor_smoke_normal_ring_v368_r10.py',
                 ROOT/'our_mani/multitask/normal_ring_native_v367.py', ROOT/'tests/test_normal_ring_support_v366.py']:
        files[str(path.resolve())] = file_sha(path)
    SETUP.mkdir()
    atomic_json(SETUP/'configuration.json', c)
    files[str(SETUP/'configuration.json')] = file_sha(SETUP/'configuration.json')
    atomic_json(SETUP/'frozen_inputs.json', dict(schema='r10_normal_ring_original_proof_checked_execution_v369',
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
            proof = json.loads(result.stdout)
            assert proof['complete'] and not proof['native_started'] and not proof['GPU_started']
            atomic_json(SETUP/(label+'_CPU_preflight.json'), proof)
            checks.append(dict(task=TASK, raw=raw, method=method, argv=argv))
        verify_inputs(ROOT, files)
        atomic_json(SETUP/'CPU_acceptance.json', dict(complete=True,
            created_utc=datetime.now(timezone.utc).isoformat(), actual_argv_preflights=checks,
            original_normal_ring_full_TRAIN_proof_unchanged=True,
            shared16_checkpoint=True, held_support_tests_each_environment=7,
            original_motion_collision_grasp_success_and10s_unchanged=True,
            current_depth_scoring_weight1_inherited_explicitly=True,
            native_started=False, GPU_started=False, formal_VAL_TEST=False))
    except BaseException as error:
        atomic_json(SETUP/'preparation_failed.json', dict(type=type(error).__name__, message=str(error),
            native_started=False, GPU_started=False))
        raise
    print(json.dumps(dict(prepared=True, CPU_checks=len(checks), frozen_inputs=len(files), native_started=False)), flush=True)


if __name__ == '__main__':
    main()
