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
SETUP = B/'high_bin_depth_stop_v290_setup'
OUT = B/'high_bin_depth_stop_v290'
TASKS = ('put_rubbish_in_moving_bin_high_speed',)


def command(task, raw, method):
    return ['bash', 'scripts/run_local_sim.sh', 'scripts/run_task_executor_smoke_bin_stop_v276_r10.py',
        '--config', str((SETUP/'configuration.json').relative_to(ROOT)),
        '--manifest', 'artifacts/framework/protocol/train_manifest.json',
        '--frozen-inputs', str((SETUP/'frozen_inputs.json').relative_to(ROOT)),
        '--task', task, '--raw-episode-id', raw, '--split', 'train', '--method', method,
        '--replica', '0', '--development-demo',
        '--out', str((OUT/'development_demo_only'/task/raw/method).relative_to(ROOT))]


def main():
    if SETUP.exists(): raise FileExistsError('preserve previous preparation')
    previous = B/'joint16_bin_smoke_v219_setup'
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
    c.update(execution_scope='bin_physical_acquire_fixed_first_four_TRAIN_v290', formal_comparison_ready=False,
        optional_native_force_contact_diagnostics_enabled=False)
    from our_mani.multitask.bin_physical_acquire_v234 import binding
    c['bin_physical_acquire_v234'] = {task: binding(ROOT, c, task) for task in TASKS}
    from our_mani.multitask.reference_stop_guard_v274 import binding as stop_binding
    c['reference_stop_guard_v274']={task:stop_binding(ROOT,c,task) for task in TASKS}
    from dataclasses import asdict
    from our_mani.multitask.perception import RegistrationConfig
    before=RegistrationConfig(**c.get('registration_config',{}));assert before.free_space_weight==0.
    c['registration_config']=dict(c.get('registration_config',{}),free_space_weight=1.)
    after=RegistrationConfig(**c['registration_config'])
    assert all(getattr(before,k)==getattr(after,k) for k in asdict(before) if k!='free_space_weight')
    c['high_bin_current_depth_scoring_v290']=dict(previous_weight=0.,actual_weight=1.,
        current_observed_surface_and_depth_free_space_cost_equal_unit_weight=True,
        original_numeric_RMS_inlier_axis_readiness_and_fallback_rules_unchanged=True,
        every_retained_current_hypothesis_still_supplied=True,
        additional_native_camera_physics_GT_or_future_depth_used=False,formal_VAL_TEST=False)
    for task in TASKS:
        for pointer in c['learned_artifacts'][task].values():
            if pointer is None:continue
            path=(ROOT/pointer['path']).resolve();assert file_sha(path)==pointer['sha256']
            if str(path) in files:assert files[str(path)]==pointer['sha256']
            files[str(path)]=pointer['sha256']
    for task in TASKS:
        library = InteractionLibrary.load(ROOT/c['learned_artifacts'][task]['interaction']['path'])
        for predictor in ('cv', 'context'):
            identity = persistent_reference_continuation_identity(ROOT, c, task, predictor, library,
                CandidateConfig(**c['candidate_config']), MPCConfig(**c['mpc_config']))
            c['continuation_identities'].setdefault(task, {})[predictor] = identity['sha256']
    paths = [Path(__file__), ROOT/'scripts/run_task_executor_smoke_bin_stop_v276_r10.py',
        ROOT/'our_mani/multitask/joint16_checkpoint_v172.py', ROOT/'our_mani/multitask/joint16_native_v173.py',
        ROOT/'tests/test_joint16_checkpoint_v172.py',
        ROOT/'our_mani/multitask/bin_physical_acquire_v234.py', ROOT/'our_mani/multitask/bin_physical_native_v235.py',
        ROOT/'our_mani/multitask/cup_physical_acquire_v167.py', ROOT/'tests/test_bin_physical_acquire_v234.py', ROOT/'our_mani/multitask/case_output_isolation_v238.py', ROOT/'tests/test_case_output_isolation_v238.py', ROOT/'our_mani/multitask/reference_stop_guard_v274.py', ROOT/'our_mani/multitask/bin_stop_native_v275.py', ROOT/'tests/test_reference_stop_guard_v274.py', ROOT/'our_mani/multitask/cpu_topology_revalidation_v258.py', ROOT/'tests/test_cpu_topology_revalidation_v258.py',ROOT/'tests/test_multitask_depth_pose_consistency.py']
    for p in paths: files[str(p)] = file_sha(p)
    SETUP.mkdir()
    atomic_json(SETUP/'configuration.json', c)
    files[str(SETUP/'configuration.json')] = file_sha(SETUP/'configuration.json')
    atomic_json(SETUP/'frozen_inputs.json', dict(schema='r10_bin_physical_acquire_fixed_first_four_TRAIN_v290_lock', files=files,
        approved_for_execution=True, native_started=False, GPU_started=False, formal_VAL_TEST=False))
    roster = json.loads((ROOT/'artifacts/framework/protocol/train_manifest.json').read_text())['episodes']
    checks = []
    try:
        for task in TASKS:
            first = [r for r in roster if r['task'] == task][:1]
            assert len(first) == 1
            for row in first:
                for method in ('cv_geometry', 'context_geometry'):
                    raw = row['raw_episode_id']
                    argv = command(task, raw, method)
                    r = subprocess.run(argv+['--check'], cwd=ROOT, capture_output=True, text=True)
                    name = task+'_'+raw+'_'+method
                    (SETUP/(name+'_CPU_stdout.log')).write_text(r.stdout)
                    (SETUP/(name+'_CPU_stderr.log')).write_text(r.stderr)
                    r.check_returncode()
                    proof = json.loads(r.stdout)
                    assert proof['complete'] and not proof['native_started'] and not proof['GPU_started']
                    atomic_json(SETUP/(name+'_CPU_preflight.json'), proof)
                    checks.append(dict(task=task, raw=raw, method=method, argv=argv))
        verify_inputs(ROOT, files)
        atomic_json(SETUP/'CPU_acceptance.json', dict(complete=True, created_utc=datetime.now(timezone.utc).isoformat(),
            actual_argv_preflights=checks, shared_checkpoint_tests_each_environment=5,
            original_dispatch_motion_collision_and_success_unchanged=True,
            explicit_physical_acquire_control_change=True, original_fit_caps_unchanged=True,
            replacement_gate='actual completed shared16, exact1600 originalTRAIN; all firstfour parents independently selected without outcome filtering',
            optional_native_force_contact_debug_omitted=True, actual_window_metrics_retained=True,
            native_started=False, GPU_started=False, formal_VAL_TEST=False))
    except BaseException as error:
        atomic_json(SETUP/'preparation_failed.json', dict(type=type(error).__name__, message=str(error),
            native_started=False, GPU_started=False))
        raise
    print(json.dumps(dict(prepared=True, frozen_inputs=len(files), CPU_checks=len(checks), native_started=False)), flush=True)


if __name__ == '__main__': main()
