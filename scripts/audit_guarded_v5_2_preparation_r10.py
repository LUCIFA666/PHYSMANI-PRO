"""Audit guarded CPU readiness and failed offscreen RGBD, preserving prior runs."""
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from our_mani.multitask.evaluation import atomic_json, file_sha, verify_inputs

BASE = ROOT / 'artifacts/framework/multitask_r10'
SETUP = BASE / 'geometry_native_actuator_diagnostic_v5_2_setup'
OUT = BASE / 'geometry_guarded_v5_2_preparation_audit'


def read(path):
    return json.loads(path.read_text())


def pointer(path):
    return dict(path=str(path.relative_to(ROOT)), sha256=file_sha(path))


def main():
    if OUT.exists():
        raise RuntimeError('Preserve this preparation audit; never overwrite')
    previous_path = ROOT / 'runtime/framework/handoff_runtime_latest.json'
    previous = read(previous_path)
    assert previous['schema'] == 'our_mani_acquire_lift_v5_handoff_status_v1'
    for key in ('report', 'previous_v4_status', 'configuration', 'frozen_inputs', 'CPU_acceptance', 'parent_summary'):
        assert file_sha(ROOT / previous[key]['path']) == previous[key]['sha256']
    lock = read(SETUP / 'frozen_inputs.json')
    verify_inputs(ROOT, lock['files'])
    old = read(ROOT / previous['frozen_inputs']['path'])
    assert all(lock['files'].get(k) == v for k, v in old['files'].items())
    assert file_sha(SETUP / 'configuration.json') == previous['configuration']['sha256']
    assert not (BASE / 'geometry_native_actuator_diagnostic_v5_2').exists()
    tests = []
    interfaces = []
    for environment in ('offline', 'native_environment'):
        path = ROOT / f'runtime/framework/test_guarded_native_v5_2_{environment}.log'
        text = path.read_text()
        assert 'Ran 10 tests' in text and text.rstrip().endswith('OK')
        tests.append(dict(environment=environment, test_count=10, log=pointer(path)))
        path = ROOT / f'runtime/framework/guarded_v5_2_{environment}_preflight.json'
        value = read(path)
        assert [x['method'] for x in value['methods']] == ['cv_geometry', 'context_geometry']
        assert value['original_v5_policy_unchanged'] and value['native_started'] is False
        assert value['process_visibility']['host_visibility_available'] is False
        assert value['frozen_input_count'] == len(lock['files'])
        interfaces.append(dict(environment=environment, report=pointer(path),
            process_visibility=value['process_visibility']))
    probes = []
    for name in ('native_offscreen_bootstrap_probe_v1', 'native_offscreen_bootstrap_probe_v2',
            'native_offscreen_RGBD_probe_v1'):
        directory = BASE / name
        report = read(directory / 'report.json')
        assert file_sha(directory / 'probe.py') == report['probe_source_sha256']
        assert file_sha(directory / 'native.log') == report['log_sha256']
        probes.append(dict(report=pointer(directory / 'report.json'), log=pointer(directory / 'native.log'),
            returncode=report['returncode'], timed_out=report['timed_out'],
            launch_ok=report['launch_ok'], shutdown_ok=report['shutdown_ok'],
            benchmark_policy_run=False, owned_process_terminal=True))
    assert 'UnicodeEncodeError' in (BASE / 'native_offscreen_bootstrap_probe_v1/native.log').read_text()
    assert read(BASE / 'native_offscreen_bootstrap_probe_v2/report.json')['launch_ok'] is True
    rgbd = read(BASE / 'native_offscreen_RGBD_probe_v1/report.json')
    assert rgbd['rgbd_captured'] is False and rgbd['mask_captured'] is False
    assert 'does not support createPlatformOpenGLContext' in (BASE / 'native_offscreen_RGBD_probe_v1/native.log').read_text()
    OUT.mkdir()
    shutil.copyfile(previous_path, OUT / 'previous_v5_runtime_status.json')
    shutil.copyfile(Path(__file__), OUT / 'audit_source.py')
    atomic_json(SETUP / 'CPU_acceptance.json', dict(
        schema='r10_guarded_v5_2_CPU_acceptance_v1', created_utc=datetime.now(timezone.utc).isoformat(),
        new_launch_tests_each_environment=10, tests=tests, parent_derived_preflights=interfaces,
        unchanged_v5_controller_CPU_acceptance=previous['CPU_acceptance'],
        v5_controller_tests_each_environment=26, frozen_input_count=len(lock['files']),
        configuration_identity_unchanged=True, all_previous_sources_preserved=True,
        native_launch_ready=False, native_started=False, formal_metric=False))
    report = dict(schema='r10_guarded_v5_2_preparation_audit_v1',
        created_utc=datetime.now(timezone.utc).isoformat(), audit_source=pointer(OUT / 'audit_source.py'),
        previous_status=pointer(OUT / 'previous_v5_runtime_status.json'),
        allowed_comparison=['cv_geometry', 'context_geometry', 'physmani'],
        prohibited_restart=['Value', 'NGFF', 'FASTER'],
        configuration=pointer(SETUP / 'configuration.json'), frozen_inputs=pointer(SETUP / 'frozen_inputs.json'),
        CPU_acceptance=pointer(SETUP / 'CPU_acceptance.json'),
        frozen_input_count=len(lock['files']), all225_inputs_verified=True,
        prior219_inputs_preserved_exact=True, policy_and_model_identity_unchanged=True,
        new_launch_sources=[pointer(ROOT / name) for name in (
            'scripts/run_native_actuator_diagnostic_pair_v5_2_r10.py',
            'scripts/native_display_guard_r10.py', 'scripts/run_local_sim_display_checked_r10.sh')],
        preflights=interfaces, launch_tests=tests, diagnostic_probes=probes,
        offscreen_conclusion=dict(bootstrap_with_ASCII_base_scene_passed=True,
            first_probe_unicode_path_error_not_a_graphics_failure=True,
            OPENGL_RGBD_context_supported=False, RGB_or_depth_or_mask_available=False,
            offscreen_admitted_for_policy=False),
        next_native_output_namespace_exists=False, next_native_pair_launched=False,
        original_task_seconds=10., dt=.05, extra_policy_physics_ticks=0,
        formal_VAL_or_TEST_started=False, auto_retry=False,
        next_required_steps=[
            'Restore execution with the original X11/OPENGL RGBD path and usable host process visibility.',
            'Use the already frozen v5_2 supervisor once; its process inventory and shared lock reject duplicate native jobs.',
            'The display guard must connect to X11 before the original v5 worker can import Qt/PyRep; retain any display failure receipt as infrastructure-pending.',
            'Audit the same fixed TRAIN104 pair for actual motion limits, gripper edges, phase/deadline, cleanup and native task outcome separately.',
            'Do not alter old v4/v5 namespaces, widen gates, extend the 10s budget or start formal VAL/TEST before executor acceptance.'
        ],
        limitations=[
            'CPU interface success is not a native policy result.',
            'Offscreen simulator bootstrap succeeded but the original RGBD OPENGL rendering crashed; the platform cannot replace X11 for this experiment.',
            'Current isolated process inventories cannot prove absence of host jobs; no v5_2 policy worker was launched.'
        ])
    atomic_json(OUT / 'report.json', report)
    verify_inputs(ROOT, lock['files'])
    atomic_json(previous_path, dict(
        schema='our_mani_guarded_v5_2_handoff_status_v1',
        status='guarded_v5_2_frozen_CPU_passed_original_RGBD_environment_blocked',
        updated_utc=datetime.now(timezone.utc).isoformat(), report=pointer(OUT / 'report.json'),
        previous_status=report['previous_status'], configuration=report['configuration'],
        frozen_inputs=report['frozen_inputs'], CPU_acceptance=report['CPU_acceptance'],
        scope='three-method comparison; same fixed normal basketball TRAIN104 CV/Context development pair',
        prohibited_restart=report['prohibited_restart'],
        v4_terminal=True, previous_v5_owned_launchers_terminal=True,
        v5_2_pair_launched=False, native_policy_result_available=False,
        offscreen_RGBD_supported=False, host_process_visibility_available=False,
        prepared_entrypoint='scripts/run_native_actuator_diagnostic_pair_v5_2_r10.py',
        next_launch_command=None, automatic_retry=False, executor_fully_solved=False,
        next_required_steps=report['next_required_steps']), overwrite=True)
    print(json.dumps(dict(report=pointer(OUT / 'report.json'), frozen_inputs=len(lock['files']),
        new_launch_tests_each_environment=10, v5_2_native_launched=False)))


if __name__ == '__main__':
    main()
