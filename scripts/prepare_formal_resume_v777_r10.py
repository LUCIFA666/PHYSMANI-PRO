"""Freeze an explicitly authorized continuation without starting any jobs.

Only a fresh resume setup is written. The old STOP, run namespaces, interrupted
outputs, physical routes, model weights and original source locks are preserved.
This program admits validation only; it never selects or opens TEST sources.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from our_mani.multitask.evaluation import atomic_json, digest, file_sha, validate_completed_case
from our_mani.multitask.input_stat_cache_v79 import verify_inputs
from our_mani.multitask.formal_suite_admission_v728 import read_formal_source
from our_mani.multitask.common_rml_protocol_v530 import fingerprint, require_attestation
from our_mani.multitask.three_method_release16 import measured_motion
from our_mani.multitask.evaluation_roster16_v283 import METHODS
from scripts import run_formal_suite16_owner_v753_r10 as old_owner
from scripts import audit_formal_suite16_results_v754_r10 as old_auditor
from scripts.run_formal_case_author_outcome_v751_r10 import BINDING_KEYS, option, require_native_protocol

SETUP = 'artifacts/framework/multitask_r10/formal_VAL_resume_v772_setup'
PREVIOUS = 'runtime/framework/formal_VAL_v753'
SUITE = 'artifacts/framework/multitask_r10/formal_suite16_v752_setup/suite.json'
SUITE_LOCK = 'artifacts/framework/multitask_r10/formal_suite16_v752_setup/frozen_inputs.json'
SOURCE_INPUTS = 'artifacts/framework/multitask_r10/formal_suite16_v752_setup/validation_source_inputs.json'
GLOBAL_STOP = 'runtime/framework/multitask_r10_STOP'
STOP_RECORD = 'runtime/framework/user_stop_v771.json'
RESUME_RECORD = 'runtime/framework/user_resume_v772.json'
OWNER = 'scripts/run_resumed_formal_VAL_owner_v772_r10.py'
AUDITOR = 'scripts/audit_resumed_formal_VAL_v773_r10.py'
CONTROL_SCHEMA = 'r10_user_authorized_formal_VAL_resume_control_v772'
LOCK_SCHEMA = 'r10_user_authorized_formal_VAL_resume_inputs_v772'
REQUIRED_SOURCES = (
    OWNER, AUDITOR, 'scripts/prepare_formal_TEST_after_resume_v774_r10.py',
    'scripts/continue_resumed_VAL_TEST_v775_r10.py',
    'scripts/prepare_formal_resume_v777_r10.py',
    'tests/test_resumed_formal_VAL_owner_v772.py',
    'tests/test_resumed_formal_VAL_audit_v773.py',
    'tests/test_formal_TEST_resume_gate_v774.py',
    'tests/test_resumed_VAL_TEST_controller_v775.py',
    'tests/test_prepare_formal_resume_v777.py',
    'scripts/report_formal_VAL_halfhour_v750_r10.py',
)
INTERRUPTED = frozenset((
    'beat_the_rotating_buzz_high_speed_episode30_cv_geometry',
    'beat_the_rotating_buzz_high_speed_episode234_physmani',
))


def require(condition, message):
    if not condition:
        raise ValueError('formal resume preparation: ' + message)


def pointer(root, path):
    root = Path(root).resolve()
    path = (root / path).resolve(strict=True)
    require(path.is_relative_to(root) and path.is_file(), 'project evidence file required')
    return dict(path=str(path.relative_to(root)), sha256=file_sha(path))


def freeze_file(root, files, path, *, expected=None):
    """Add a project supplement without changing a previously declared SHA."""
    item = pointer(root, path)
    if expected is not None:
        require(item['sha256'] == expected, 'evidence bytes differ: ' + item['path'])
    canonical = str((Path(root) / item['path']).resolve())
    for key in (canonical, item['path']):
        if key in files:
            require(files[key] == item['sha256'], 'conflicting declared evidence SHA: ' + key)
    files[canonical] = item['sha256']
    return item


def read_evidence(root, files, path):
    item = freeze_file(root, files, path)
    value = json.loads((Path(root) / item['path']).read_text())
    require(file_sha(Path(root) / item['path']) == item['sha256'], 'JSON changed while being read')
    return value, item


def require_new_sources(root):
    for name in REQUIRED_SOURCES:
        require((Path(root) / name).is_file(), 'new control source/test is not ready: ' + name)


def partition(plan, status, terminal):
    require(plan.get('split') == status.get('split') == terminal.get('split') == 'validation',
            'validation-only continuation required')
    require(plan.get('expected_cases') == status.get('expected') == terminal.get('expected') == 945,
            'the original945-case cohort is required')
    rows = plan.get('jobs')
    require(isinstance(rows, list) and len(rows) == 945, 'all original plan rows required')
    jobs = {row['job_id']: row for row in rows}
    require(len(jobs) == 945, 'duplicate original plan key')
    completed, interrupted = status.get('completed'), status.get('failed')
    require(isinstance(completed, dict) and len(completed) == 95, 'exactly95 completed records required')
    require(isinstance(interrupted, dict) and set(interrupted) == INTERRUPTED,
            'the two recorded user interruptions must be preserved')
    require(set(completed) <= set(jobs) and set(interrupted) <= set(jobs)
            and not (set(completed) & set(interrupted)), 'foreign or overlapping previous result keys')
    require(terminal.get('complete') is False and terminal.get('stop_requested') is True
            and status.get('stop_requested') is True and terminal.get('completed') == 95
            and terminal.get('failed') == 2, 'explicit stopped95/2 terminal required')
    require(not status.get('active') and not status.get('checking')
            and status.get('native_slots') == 0 and status.get('GPU_slots') == 0,
            'previous native/CPU work must have exited')
    require(plan.get('source_outcome_filtering') is False and plan.get('automatic_retry') is False,
            'the original unfiltered one-time plan is required')
    for result in interrupted.values():
        require(result.get('stage') == 'native' and result.get('complete') is False
                and result.get('interrupted') is True and result.get('timed_out') is False
                and result.get('returncode') == -2 and result.get('automatic_retry') is False,
                'administrative interruptions cannot become native task failures')
    policy = plan.get('policy_sha256')
    require(isinstance(policy, str) and len(policy) == 64 and terminal.get('policy_sha256') == policy,
            'the original frozen policy must be identical')
    done = sorted(completed)
    remaining = sorted(set(jobs) - set(done))
    require(len(remaining) == 850, 'exactly850 original uncompleted keys required')
    return jobs, done, sorted(interrupted), remaining, policy


def process_exited(pid, expected, *, proc=Path('/proc')):
    require(type(pid) is int and type(expected) is int, 'explicit previous process identity required')
    try:
        fields = (Path(proc) / str(pid) / 'stat').read_text().rsplit(')', 1)[1].split()
    except FileNotFoundError:
        return True
    actual, state = int(fields[19]), fields[0]
    require(actual != expected or state in ('Z', 'X'),
            'previous process still owns this identity: ' + str(pid))
    return True


def require_previous_exited(root, runtime, launch):
    process_exited(launch['pid'], launch['start_ticks'])
    controller = json.loads((Path(root) / 'runtime/framework/ordered_author_outcome_v758/launch.json').read_text())
    process_exited(controller['pid'], controller['start_ticks'])
    count = 0
    for stage in ('CPU', 'native'):
        for path in (runtime / stage).glob('*/launch.json'):
            item = json.loads(path.read_text())
            process_exited(item['launcher_pid'], item['launcher_start_ticks'])
            count += 1
    return dict(previous_owner_and_controller_exited=True, recorded_child_launches_checked=count)


def admit_validation(root, files):
    verify_inputs(root, files)
    suite = json.loads((Path(root) / SUITE).read_text())
    old_auditor.driver_bindings(suite, files)
    manifest, sources, proof = read_formal_source(root, suite, files, 'validation')
    source_lock = json.loads((Path(root) / SOURCE_INPUTS).read_text())
    require(source_lock.get('schema') == 'r10_full16_post_admission_split_source_lock_v1'
            and source_lock.get('split') == 'validation' and source_lock.get('source_count') == 315
            and len(sources) == 315 and source_lock.get('policy_sha256') == proof['policy_sha256']
            and source_lock.get('TEST_used_for_fitting') is False
            and source_lock.get('source_manifest_sha256') == suite['source_manifests']['validation']['sha256'],
            'the complete original admitted validation source lock is required')
    verify_inputs(root, source_lock['files'])
    return suite, sources, proof


def check_plan(plan, suite, sources, proof):
    routes = {(r['task'], r['method']): r for r in proof['routes']['routes']}
    source_rows = {(r['task'], r['raw_episode_id']): r for r in sources}
    expected_keys = {task + '_' + raw + '_' + method for task, raw in source_rows for method in METHODS}
    require({r['job_id'] for r in plan['jobs']} == expected_keys, 'plan differs from the exact original source grid')
    for row in plan['jobs']:
        route = routes[row['task'], row['method']]
        expected = dict(admission_driver=suite['formal_case_driver'], admission_guard=suite['formal_admission_guard'],
            case_worker=route['worker'], native_runtime={k: route['consumer'][k] for k in ('path', 'sha256')},
            original_native_runtime=suite['author_outcome_protocol']['original_runtime'],
            author_outcome_protocol=suite['author_outcome_protocol'])
        require(all(row.get(k) == v for k, v in expected.items())
                and row.get('worker_sha256') == expected['admission_driver']['sha256']
                and row.get('configuration_sha256') == route['configuration']['sha256']
                and row.get('input_lock_sha256') == route['frozen_inputs']['sha256']
                and row.get('source_row_sha256') == digest(source_rows[row['task'], row['raw']]),
                'original plan source or physical bindings changed')
        argv = ['bash', 'scripts/run_aligned_gpu.sh' if row['method'] == 'physmani' else 'scripts/run_local_sim.sh',
            suite['formal_case_driver']['path'], '--config', route['configuration']['path'],
            '--manifest', suite['source_manifests']['validation']['path'], '--frozen-inputs', route['frozen_inputs']['path'],
            '--suite', SUITE, '--suite-lock', SUITE_LOCK, '--source-inputs', SOURCE_INPUTS,
            '--task', row['task'], '--raw-episode-id', row['raw'], '--method', row['method'], '--split', 'validation',
            '--out', 'artifacts/framework/multitask_r10/formal_suite16_v752/validation/'
                + '/'.join((row['task'], row['raw'], row['method']))]
        require(row['argv'] == argv, 'original actual argv changed')
    return routes, source_rows


def check_CPU(root, files, runtime, row, source_row):
    leaf = runtime / 'CPU' / row['job_id']
    CPU, _ = read_evidence(root, files, leaf / 'CPU_preflight.json')
    stdout, _ = read_evidence(root, files, leaf / 'stdout.log')
    terminal, _ = read_evidence(root, files, leaf / 'terminal.json')
    child_launch, _ = read_evidence(root, files, leaf / 'launch.json')
    freeze_file(root, files, leaf / 'stderr.log')
    old_owner.require_CPU_driver(CPU, row)
    identity = CPU['identity']
    require(stdout == CPU and CPU.get('complete') is True and CPU.get('native_started') is False
            and CPU.get('GPU_started') is False and CPU.get('actual_split') == 'validation'
            and CPU.get('component_domain', {}).get('purpose') == 'evaluation'
            and identity.get('source_row_sha256') == digest(source_row)
            and identity.get('config_sha256') == row['configuration_sha256']
            and identity.get('frozen_inputs_sha256') == row['input_lock_sha256']
            and identity.get('source_input_lock_sha256') == file_sha(Path(root) / SOURCE_INPUTS),
            'original actual-argv CPU proof changed')
    require(terminal.get('complete') is True and terminal.get('returncode') == 0
            and terminal.get('interrupted') is False and terminal.get('timed_out') is False,
            'CPU proof must have completed before the stop')
    expected = {k: row[k] for k in BINDING_KEYS}
    old_auditor.require_actual_launch(child_launch, row, expected, check=True, identity=identity)
    return identity


def check_completed(root, files, runtime, row, status_result, identity, profile, policy):
    leaf = runtime / 'native' / row['job_id']
    result, _ = read_evidence(root, files, leaf / 'terminal.json')
    child_launch, _ = read_evidence(root, files, leaf / 'launch.json')
    for name in ('stdout.log', 'stderr.log'):
        freeze_file(root, files, leaf / name)
    require(result.get('complete') is True and result.get('returncode') == 0
            and result.get('interrupted') is False and result.get('timed_out') is False,
            'a completed import cannot be interrupted or censored')
    old_auditor.require_actual_launch(child_launch, row, {k: row[k] for k in BINDING_KEYS}, check=False, identity=identity)
    out = Path(root) / option(row['argv'], '--out')
    case, case_pointer = read_evidence(root, files, out / 'case.json')
    driver, driver_pointer = read_evidence(root, files, out / 'admission_driver_receipt.json')
    old_owner.require_native_driver(driver, row, out / 'case.json', 'validation')
    require_native_protocol(case, row['author_outcome_protocol'])
    validate_completed_case(case, identity)
    require_attestation(case.get('common_author_RML_parameters', {}), profile)
    require(case.get('cleanup_completed') is True and case.get('component_domain', {}).get('purpose') == 'evaluation'
            and case['component_domain'].get('split') == 'validation'
            and case.get('full16_frozen_execution', {}).get('policy_sha256') == policy
            and type(measured_motion(case)) is bool
            and len(case.get('all_stage_counted_joint_motion', {}).get('events', [])) == case['counted_clock']['ticks'],
            'completed original native domain/motion/cleanup contract changed')
    require(result.get('case') == status_result.get('case') == case_pointer
            and result.get('admission_driver_receipt') == status_result.get('admission_driver_receipt') == driver_pointer
            and status_result.get('complete') is True and status_result.get('returncode') == 0
            and status_result.get('interrupted') is False and status_result.get('timed_out') is False
            and status_result.get('native_success') is case['official_task_success']
            and status_result.get('motion_pass') is measured_motion(case), 'completed status pointers/outcomes changed')
    return case['official_task_success']


def freeze_interrupted(root, files, runtime, row, status_result, identity):
    leaf = runtime / 'native' / row['job_id']
    terminal, _ = read_evidence(root, files, leaf / 'terminal.json')
    launch, _ = read_evidence(root, files, leaf / 'launch.json')
    require(terminal == {k: v for k, v in status_result.items() if k != 'stage'}, 'interrupted terminal changed')
    old_auditor.require_actual_launch(launch, row, {k: row[k] for k in BINDING_KEYS}, check=False, identity=identity)
    for name in ('stdout.log', 'stderr.log'):
        freeze_file(root, files, leaf / name)
    out = Path(root) / option(row['argv'], '--out')
    require(not (out / 'case.json').exists() and not (out / 'admission_driver_receipt.json').exists(),
            'interrupted output cannot silently become a completed case')
    for path in sorted(out.rglob('*')):
        if path.is_file():
            freeze_file(root, files, path)


def prepare(root=ROOT):
    root = Path(root).resolve()
    setup = root / SETUP
    require(not setup.exists(), 'preserve the one-time setup; no implicit restart')
    require_new_sources(root)
    files = dict(json.loads((root / SUITE_LOCK).read_text())['files'])
    suite, sources, proof = admit_validation(root, files)
    pointers = {}
    for key, name in (('suite', SUITE), ('suite_lock', SUITE_LOCK), ('source_inputs', SOURCE_INPUTS)):
        pointers[key] = freeze_file(root, files, name)
    runtime = root / PREVIOUS
    previous = {}
    for name in ('launch', 'plan', 'status', 'terminal'):
        previous[name], pointers['previous_' + name] = read_evidence(root, files, runtime / (name + '.json'))
    jobs, completed, interrupted, remaining, policy = partition(previous['plan'], previous['status'], previous['terminal'])
    require(policy == proof['policy_sha256'], 'admitted policy differs from the stopped run')
    launch = previous['launch']
    require(launch.get('source_sha256') == file_sha(root / 'scripts/run_formal_suite16_owner_v753_r10.py')
            and launch.get('split') == 'validation' and launch.get('policy_sha256') == policy
            and launch.get('source_manifest') == suite['source_manifests']['validation']
            and launch.get('suite_lock_sha256') == pointers['suite_lock']['sha256']
            and launch.get('split_source_lock_sha256') == pointers['source_inputs']['sha256']
            and launch.get('admission_driver') == suite['formal_case_driver']
            and launch.get('admission_guard') == suite['formal_admission_guard']
            and launch.get('execution_protocol_changed') is True
            and launch.get('native_runtime') == suite['author_outcome_protocol']['runtime_consumer']
            and launch.get('original_native_runtime') == suite['author_outcome_protocol']['original_runtime']
            and launch.get('author_outcome_protocol') == suite['author_outcome_protocol']
            and launch.get('max_native_slots') == 2 and launch.get('max_GPU_slots') == 1
            and launch.get('max_CPU_preflight_slots') == 2 and launch.get('automatic_retry') is False
            and previous['terminal'].get('author_outcome_protocol') == suite['author_outcome_protocol']
            and previous['terminal'].get('execution_protocol_changed') is True
            and previous['terminal'].get('automatic_retry') is False,
            'original stopped owner binding differs from the frozen suite')
    exited = require_previous_exited(root, runtime, launch)
    routes, source_rows = check_plan(previous['plan'], suite, sources, proof)
    profile = fingerprint(proof['routes']['common_author_RML_profile'])
    successes = 0
    for key in completed + interrupted:
        row = jobs[key]
        identity = check_CPU(root, files, runtime, row, source_rows[row['task'], row['raw']])
        if key in previous['status']['completed']:
            successes += int(check_completed(root, files, runtime, row, previous['status']['completed'][key], identity, profile, policy))
        else:
            freeze_interrupted(root, files, runtime, row, previous['status']['failed'][key], identity)
    require(previous['terminal'].get('native_successes') == successes, 'completed outcome total changed')
    stop_record, pointers['user_stop_record'] = read_evidence(root, files, STOP_RECORD)
    resume_record, pointers['user_resume_record'] = read_evidence(root, files, RESUME_RECORD)
    stop_bytes = (root / GLOBAL_STOP).read_bytes()
    stop_sha = hashlib.sha256(stop_bytes).hexdigest()
    global_stop = json.loads(stop_bytes)
    require(stop_record.get('user_stop_received') is True and stop_record.get('user_message') == '先停下'
            and global_stop.get('created_utc') == stop_record.get('created_utc')
            and global_stop.get('user_stop_received') is True
            and resume_record.get('schema') == 'r10_explicit_user_resume_v772'
            and resume_record.get('user_authorized_resume') is True and resume_record.get('revokes_previous_stop') is True
            and resume_record.get('previous_user_stop') == pointers['user_stop_record']
            and resume_record.get('previous_global_stop_sha256') == stop_sha
            and resume_record.get('preserve_completed95') is True
            and resume_record.get('resume_remaining850_in_new_namespace') is True
            and resume_record.get('training_parameters_unchanged') is True
            and resume_record.get('TEST_requires_full_VAL_release') is True
            and datetime.fromisoformat(resume_record['created_utc']) > datetime.fromisoformat(stop_record['created_utc']),
            'a later explicit user-authorized resume of this exact stop is required')
    for name in REQUIRED_SOURCES:
        freeze_file(root, files, name)
    # The mutable STOP path is deliberately excluded; only its byte-identical
    # archive is locked, so authorized removal cannot invalidate the new graph.
    require(str((root / GLOBAL_STOP).resolve()) not in files and GLOBAL_STOP not in files,
            'the mutable global STOP must not be an immutable input')
    verify_inputs(root, files)
    require((root / GLOBAL_STOP).read_bytes() == stop_bytes, 'STOP changed during preparation')
    setup.mkdir(parents=True, exist_ok=False)
    archive = setup / 'previous_user_STOP.json'
    with archive.open('xb') as handle:
        handle.write(stop_bytes)
    archive_pointer = freeze_file(root, files, archive, expected=stop_sha)
    control = dict(schema=CONTROL_SCHEMA, created_utc=datetime.now(timezone.utc).isoformat(),
        previous_runtime=PREVIOUS, completed_job_ids=completed, interrupted_job_ids=interrupted,
        remaining_job_ids=remaining, expected_cases=945, policy_sha256=policy, split='validation',
        resume_owner=pointer(root, OWNER), resume_auditor=pointer(root, AUDITOR),
        user_authorized_resume=True, previous_user_STOP=archive_pointer,
        completed_key_sha256=digest(completed), remaining_key_sha256=digest(remaining),
        author_outcome_protocol=suite['author_outcome_protocol'], **pointers,
        previous_process_exit_proof=exited, native_started=False, GPU_started=False, TEST_read=False,
        completed_results_selected_by_success=False, user_interruptions_are_task_failures=False,
        physical_policy_changed=False, models_retrained=False, old_evidence_modified=False)
    atomic_json(setup / 'resume_control.json', control)
    control_pointer = freeze_file(root, files, setup / 'resume_control.json')
    atomic_json(setup / 'resume_inputs.json', dict(schema=LOCK_SCHEMA, approved_for_execution=True,
        files=files, control=control_pointer, split='validation', policy_sha256=policy,
        native_started=False, GPU_started=False, TEST_read=False))
    return dict(complete=True, completed=95, remaining=850, interrupted=2, policy_sha256=policy,
        resume_control=control_pointer, resume_control_lock=pointer(root, setup / 'resume_inputs.json'),
        native_started=False, GPU_started=False, TEST_read=False, STOP_modified=False)


if __name__ == '__main__':
    print(json.dumps(prepare(), ensure_ascii=False), flush=True)
