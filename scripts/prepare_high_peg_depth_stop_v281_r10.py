"""First-source highpeg initial operation with full100 TRAIN D4 evidence."""
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
SETUP = B/'high_peg_depth_stop_v281_setup'
OUT = B/'high_peg_depth_stop_v281'
TASKS = ('insert_onto_rotating_peg_high_speed',)


def command(task, raw, method):
    return ['bash', 'scripts/run_local_sim.sh', 'scripts/run_task_executor_smoke_high_peg_stop_v280_r10.py',
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
    c.update(execution_scope='high_peg_initial_full100_fixed_first_TRAIN_v281', formal_comparison_ready=False,
        optional_native_force_contact_diagnostics_enabled=False)
    from our_mani.multitask.high_peg_native_v245 import binding
    from our_mani.multitask.high_peg_ring_runtime_v243 import binding as ring_binding
    from our_mani.multitask.high_peg_operational_gauge_v244 import binding as gauge_binding
    from our_mani.multitask.high_peg_primary_fitting_v231 import PrimarySequenceLibrary
    task=TASKS[0]
    from dataclasses import asdict
    from our_mani.multitask.perception import RegistrationConfig
    before=RegistrationConfig(**c.get('registration_config',{}))
    assert before.free_space_weight==0.
    c['registration_config']=dict(c.get('registration_config',{}),free_space_weight=1.)
    after=RegistrationConfig(**c['registration_config'])
    assert all(getattr(before,k)==getattr(after,k) for k in asdict(before) if k!='free_space_weight')
    c['high_peg_current_depth_scoring_v263']=dict(previous_free_space_weight=0.,actual_free_space_weight=1.,
        current_observed_residual_and_free_space_cost_equal_unit_weight=True,
        original_registration_RMS_inlier_and_D4_quotient_thresholds_unchanged=True,
        every_retained_current_ICP_hypothesis_still_supplied=True,
        native_waypoints_GT_future_depth_or_extra_camera_calls_used=False,
        geometry_fallback_and_phase_contact_permissions_unchanged=True,formal_VAL_TEST=False)
    fit_report=B/'high_peg_primary_minimax_v232/report.json'
    report=json.loads(fit_report.read_text())
    c['learned_artifacts'][task]['interaction']=report['model']
    c['high_peg_primary_report_v232']=dict(path=str(fit_report.relative_to(ROOT)),sha256=file_sha(fit_report))
    proof_report=B/'high_peg_initial_affordance_v242/report.json'
    proof=json.loads(proof_report.read_text());row=proof['entries'][0]
    assert row['task']==task and row['policy_skill_authorized']
    c['ring_affordances'][task]=row['permission']
    replay=dict(path=str(proof_report.relative_to(ROOT)),sha256=file_sha(proof_report))
    c['high_peg_ring_runtime_v243']={task:ring_binding(ROOT,row['permission'],replay)}
    c['high_peg_operational_gauge_v244']={task:gauge_binding(ROOT,c,task)}
    c['high_peg_primary_v245']=binding(ROOT,c)
    from our_mani.multitask.reference_stop_guard_v274 import binding as stop_binding
    c['reference_stop_guard_v274']={task:stop_binding(ROOT,c,task)}
    for name,checksum in report['verified_files_sha256'].items():
        name=str((ROOT/name).resolve())
        if name in files:assert files[name]==checksum
        files[name]=checksum
    for name,checksum in c['high_peg_ring_runtime_v243'][task]['inputs_sha256'].items():
        name=str((ROOT/name).resolve())
        if name in files:assert files[name]==checksum
        files[name]=checksum
    for name,checksum in c['high_peg_primary_v245']['sources_sha256'].items():
        files[str((ROOT/name).resolve())]=checksum
    files[str(fit_report)]=file_sha(fit_report)
    for pointer in c['learned_artifacts'][task].values():
        if pointer is None:continue
        path=(ROOT/pointer['path']).resolve();assert file_sha(path)==pointer['sha256']
        if str(path) in files:assert files[str(path)]==pointer['sha256']
        files[str(path)]=pointer['sha256']
    for task in TASKS:
        library = PrimarySequenceLibrary.load(ROOT/c['learned_artifacts'][task]['interaction']['path'])
        for predictor in ('cv', 'context'):
            identity = persistent_reference_continuation_identity(ROOT, c, task, predictor, library,
                CandidateConfig(**c['candidate_config']), MPCConfig(**c['mpc_config']))
            c['continuation_identities'].setdefault(task, {})[predictor] = identity['sha256']
    paths = [Path(__file__), ROOT/'scripts/run_task_executor_smoke_high_peg_stop_v280_r10.py',
        ROOT/'our_mani/multitask/joint16_checkpoint_v172.py', ROOT/'our_mani/multitask/joint16_native_v173.py',
        ROOT/'tests/test_joint16_checkpoint_v172.py',
        ROOT/'our_mani/multitask/bin_physical_acquire_v234.py', ROOT/'our_mani/multitask/bin_physical_native_v235.py',
        ROOT/'our_mani/multitask/cup_physical_acquire_v167.py', ROOT/'tests/test_bin_physical_acquire_v234.py', ROOT/'our_mani/multitask/case_output_isolation_v238.py', ROOT/'tests/test_case_output_isolation_v238.py', ROOT/'our_mani/multitask/cpu_topology_revalidation_v258.py', ROOT/'tests/test_cpu_topology_revalidation_v258.py', ROOT/'tests/test_multitask_depth_pose_consistency.py', ROOT/'our_mani/multitask/reference_stop_guard_v274.py', ROOT/'our_mani/multitask/high_peg_stop_native_v279.py', ROOT/'tests/test_reference_stop_guard_v274.py']
    for p in paths: files[str(p)] = file_sha(p)
    SETUP.mkdir()
    atomic_json(SETUP/'configuration.json', c)
    files[str(SETUP/'configuration.json')] = file_sha(SETUP/'configuration.json')
    atomic_json(SETUP/'frozen_inputs.json', dict(schema='r10_high_peg_initial_full100_fixed_first_TRAIN_v281_lock', files=files,
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
            explicit_initial_highpeg_consumer_and_full100_D4_proof=True, original_fit_caps_unchanged=True,
            replacement_gate='actual completed shared16, exact1600 originalTRAIN; fixed first original highpeg TRAIN parent; no outcome filtering',
            optional_native_force_contact_debug_omitted=True, actual_window_metrics_retained=True,
            native_started=False, GPU_started=False, formal_VAL_TEST=False))
    except BaseException as error:
        atomic_json(SETUP/'preparation_failed.json', dict(type=type(error).__name__, message=str(error),
            native_started=False, GPU_started=False))
        raise
    print(json.dumps(dict(prepared=True, frozen_inputs=len(files), CPU_checks=len(checks), native_started=False)), flush=True)


if __name__ == '__main__': main()
