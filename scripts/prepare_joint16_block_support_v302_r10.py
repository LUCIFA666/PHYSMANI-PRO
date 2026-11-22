"""Freeze four original first-source block development cases; no formal release."""
from pathlib import Path
from datetime import datetime, timezone
import json
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from our_mani.multitask.evaluation import atomic_json, file_sha
from our_mani.multitask.input_stat_cache_v79 import verify_inputs
from our_mani.multitask.joint16_tracks_v31 import BLOCKS

B = ROOT/'artifacts/framework/multitask_r10'
SETUP = B/'joint16_block_reference_v302_setup'
OUT = B/'joint16_block_reference_v302'


def command(task, raw, method):
    return ['bash', 'scripts/run_local_sim.sh', 'scripts/run_task_executor_smoke_block_v301_r10.py',
        '--config', str((SETUP/'configuration.json').relative_to(ROOT)),
        '--manifest', 'artifacts/framework/protocol/train_manifest.json',
        '--frozen-inputs', str((SETUP/'frozen_inputs.json').relative_to(ROOT)),
        '--task', task, '--raw-episode-id', raw, '--split', 'train', '--method', method,
        '--development-demo', '--out', str((OUT/'development_demo_only'/task/raw/method).relative_to(ROOT))]


def main():
    if SETUP.exists(): raise FileExistsError('preserve previous preparation')
    original_config = ROOT/'config/framework/aligned_protocol_r09.json'
    c = json.loads(original_config.read_text())
    shared = B/'joint16_block_reference_v200_setup'
    previous = json.loads((shared/'configuration.json').read_text())
    files = dict(json.loads((shared/'frozen_inputs.json').read_text())['files'])
    c['context_checkpoint'] = previous['context_checkpoint']
    c['joint16_development_checkpoint_v157'] = previous['joint16_development_checkpoint_v157']
    c['joint16_context_checkpoint'] = c['context_checkpoint']
    c['dynamics_checkpoint'] = c['context_checkpoint']['path']
    c['cv_local_value'] = c['context_local_value'] = ''
    c.update(value_input_context=False, uncertainty_model_present=False,
        version='r10_shared16_block_held_support_contact_TRAIN_v302',
        execution_scope='fixed first-source geometry development, original201 references plus298 held support/contact and counted failure holds',
        formal_comparison_ready=False)
    from our_mani.multitask.block_held_support_v298 import binding as held_binding
    c['block_held_support_v298']=held_binding(ROOT)
    roster_path = ROOT/'artifacts/framework/protocol/train_manifest.json'
    roster = json.loads(roster_path.read_text())['episodes']
    sources = {task: next(r for r in roster if r['task'] == task) for task in sorted(BLOCKS)}
    paths = [ROOT/'our_mani/multitask/block_held_support_v298.py',ROOT/'tests/test_block_held_support_v298.py',ROOT/'our_mani/multitask/joint16_block_support_episode_v300.py',original_config, Path(__file__), ROOT/'scripts/run_task_executor_smoke_block_v301_r10.py',
        ROOT/'scripts/evaluate_aligned_r09.py', ROOT/'scripts/collect_local_rollouts.py',
        ROOT/c['goal_calibration'], roster_path,
        ROOT/'our_mani/multitask/joint16_block_episode_v98.py', ROOT/'our_mani/multitask/joint16_block_context_v97.py',
        ROOT/'our_mani/multitask/block_exact_reference_v201.py',ROOT/'our_mani/multitask/joint16_block_reference_episode_v202.py',ROOT/'tests/test_block_exact_reference_v201.py',ROOT/'our_mani/multitask/block_reference_v197.py',ROOT/'our_mani/multitask/joint16_block_reference_episode_v198.py',ROOT/'tests/test_block_reference_v197.py',ROOT/'our_mani/multitask/block_reference_v186.py', ROOT/'our_mani/multitask/joint16_block_reference_episode_v187.py',ROOT/'tests/test_block_reference_v186.py',ROOT/'our_mani/multitask/bounded_transport_v159.py',ROOT/'our_mani/multitask/block_full_budget_v176.py', ROOT/'our_mani/multitask/joint16_block_full_episode_v177.py',
        ROOT/'tests/test_block_full_budget_v176.py', ROOT/'tests/test_joint16_block_episode_v98.py',
        ROOT/'tests/test_joint16_block_context_v97.py']
    for folder in ('our_mani', 'our_mani/aligned', 'our_mani/risk_v2', 'our_mani/risk_v3',
            'our_mani/risk_v4', 'our_mani/sensor_alignment'):
        paths.extend(sorted((ROOT/folder).glob('*.py')))
    paths.extend(Path(row['path']) for row in sources.values())
    paths.extend(ROOT/name for name in c['block_held_support_v298']['sources_sha256'])
    for p in paths:
        p = p.resolve()
        checksum = file_sha(p)
        if str(p) in files and files[str(p)] != checksum: raise ValueError('frozen dependency changed')
        files[str(p)] = checksum
    verify_inputs(ROOT, files)
    SETUP.mkdir()
    atomic_json(SETUP/'configuration.json', c)
    files[str(SETUP/'configuration.json')] = file_sha(SETUP/'configuration.json')
    atomic_json(SETUP/'frozen_inputs.json', dict(schema='r10_shared16_block_exact_next_reference_TRAIN_v302_lock', files=files,
        approved_for_execution=True, native_started=False, GPU_started=False, formal_VAL_TEST=False))
    checks = []
    try:
        for task, row in sources.items():
            for method in ('cv_geometry', 'context_geometry'):
                argv = command(task, row['raw_episode_id'], method)
                r = subprocess.run(argv+['--check'], cwd=ROOT, capture_output=True, text=True)
                name = task+'_'+row['raw_episode_id']+'_'+method
                (SETUP/(name+'_CPU_stdout.log')).write_text(r.stdout)
                (SETUP/(name+'_CPU_stderr.log')).write_text(r.stderr)
                r.check_returncode()
                proof = json.loads(r.stdout)
                assert proof['complete'] and not proof['native_started'] and not proof['GPU_started']
                atomic_json(SETUP/(name+'_CPU_preflight.json'), proof)
                checks.append(dict(task=task, raw=row['raw_episode_id'], method=method, argv=argv))
        verify_inputs(ROOT, files)
        atomic_json(SETUP/'CPU_acceptance.json', dict(complete=True, created_utc=datetime.now(timezone.utc).isoformat(),
            actual_argv_preflights=checks, actual_shared1600_adapter_loads=True,
            held_support_contact_tests_each_environment=6,full_budget_tail_tests_each_environment=4,global_local_reference_tests_each_environment=9,exact_next_reference_tests_each_environment=10,original_exact_next_tick_collision_gate_retained=True,original_geometry_kernel_file_unchanged=True,global_and_local_joint_actuator_changed=True,local_velocity_delta_projection_per_joint=True,reference_acceleration_cap=4.,native_acceleration_cap=6.,
            original_early_policy_failure_retained=True, counted_failure_holds_are_explicit_new_policy=True,
            actual_window_metrics_all_stages=True, policy_retry=False,
            native_started=False, GPU_started=False, formal_VAL_TEST=False))
    except BaseException as error:
        atomic_json(SETUP/'preparation_failed.json', dict(type=type(error).__name__, message=str(error),
            native_started=False, GPU_started=False))
        raise
    print(json.dumps(dict(prepared=True, frozen_inputs=len(files), CPU_checks=len(checks), native_started=False)), flush=True)


if __name__ == '__main__': main()
