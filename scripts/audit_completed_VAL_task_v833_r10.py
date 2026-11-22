"""Read-only evidence audit of one complete original three-method VAL task.

This is a task-subset report, never a945-case VAL release. It reads recorded
753/772/823 evidence without creating a simulator, process launches, admission,
input-stat caches, or live environment capture. Only main writes one fresh
report after every original task key passes the existing per-case checks.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import audit_formal_suite16_results_v754_r10 as base
from scripts import audit_environment_resumed_formal_VAL_v824_r10 as resumed
from scripts.run_formal_case_author_outcome_v751_r10 import option
from our_mani.multitask.common_rml_protocol_v530 import fingerprint, require_attestation
from our_mani.multitask.evaluation import atomic_json, digest, file_sha, validate_completed_case
from our_mani.multitask.evaluation_roster16_v283 import METHODS, TASKS
from our_mani.multitask.three_method_release16 import measured_motion

SUITE = 'artifacts/framework/multitask_r10/formal_suite16_v752_setup/suite.json'
CONTROL = 'artifacts/framework/multitask_r10/formal_VAL_env_resume_v823_setup/resume_control.json'
LOCK = 'artifacts/framework/multitask_r10/formal_VAL_env_resume_v823_setup/resume_inputs.json'
OWNER_SOURCES = {
    'runtime/framework/formal_VAL_v753': 'scripts/run_formal_suite16_owner_v753_r10.py',
    'runtime/framework/formal_VAL_resume_v772': 'scripts/run_resumed_formal_VAL_owner_v772_r10.py',
    resumed.RUNTIME: resumed.OWNER,
}
COLLISION_REASON = 'public_forecast_collision_on_RML_target_sweep'
SHA = re.compile(r'^[0-9a-f]{64}$')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def require_pointer(value, *, filename=None):
    require(isinstance(value, dict) and set(value) == {'path', 'sha256'}, 'exact evidence pointer required')
    path, checksum = value['path'], value['sha256']
    require(isinstance(path, str) and bool(path) and not Path(path).is_absolute()
            and '..' not in Path(path).parts and path != '.', 'relative project evidence path required')
    require(isinstance(checksum, str) and SHA.fullmatch(checksum), 'explicit lowercase SHA256 required')
    if filename is not None:
        require(Path(path).name == filename, 'unexpected evidence filename')
    return value


def select_complete_task(plan, status, task, source_rows):
    """Select the exact original source x3 grid; other tasks may remain pending."""
    require(task in TASKS, 'original full16 task required')
    require(plan.get('split') == status.get('split') == 'validation'
            and plan.get('expected_cases') == status.get('expected') == 945, 'VAL-only original945 plan required')
    jobs = plan.get('jobs')
    require(isinstance(jobs, list) and len(jobs) == 945, 'complete original plan required')
    require(all(isinstance(r, dict) and isinstance(r.get('job_id'), str) for r in jobs)
            and len({r['job_id'] for r in jobs}) == len(jobs), 'unique original plan keys required')
    sources = [r for r in source_rows if r.get('task') == task]
    raws = [r.get('raw_episode_id') for r in sources]
    require(raws and all(isinstance(v, str) and v for v in raws) and len(set(raws)) == len(raws),
            'unique original task source IDs required')
    selected = [r for r in jobs if r.get('task') == task]
    expected = {(raw, method) for raw in raws for method in METHODS}
    require(len(selected) == len(expected)
            and {(r.get('raw'), r.get('method')) for r in selected} == expected,
            'task plan must retain the same exact original source set for all three methods')
    completed, failed = status.get('completed'), status.get('failed')
    require(isinstance(completed, dict) and isinstance(failed, dict), 'actual completed/failed status maps required')
    for row in selected:
        name = row['job_id']
        require(name == task + '_' + row['raw'] + '_' + row['method'], 'original task/raw/method key changed')
        require(name in completed and name not in failed, 'task is pending or contains infrastructure failure: ' + name)
        result = completed[name]
        resumed.require_clean_terminal(result)
        require_pointer(result.get('case'), filename='case.json')
        require_pointer(result.get('admission_driver_receipt'), filename='admission_driver_receipt.json')
    return sorted(selected, key=lambda r: (r['raw'], list(METHODS).index(r['method'])))


def classify_case(case):
    success, fatal = case.get('official_task_success'), case.get('native_terminal_failure')
    require(type(success) is bool and type(fatal) is bool and not (success and fatal),
            'independent mutually exclusive native boolean outcomes required')
    if success:
        return 'native_success'
    if fatal:
        return 'native_terminal_failure'
    failure = case.get('policy_failure')
    if isinstance(failure, str) and failure.strip():
        return 'explicit_policy_failure'
    clock = case.get('counted_clock', {})
    ticks, dt, seconds = clock.get('ticks'), clock.get('dt'), clock.get('seconds')
    if (type(ticks) is int and ticks == 200 and type(dt) in (int, float) and dt == .05
            and type(seconds) in (int, float) and math.isfinite(seconds) and abs(seconds - 10.) <= 1e-6):
        return 'budget_exhausted_no_native_terminal'
    return 'unknown'


def geometry_diagnostics(case, method):
    """Count recorded rejection events, never physical collisions or pairs."""
    require(method in METHODS, 'one of the original three methods required')
    actor = case.get('actor_metrics')
    actor = actor if isinstance(actor, dict) else {}
    intents = actor.get('geometry_intents')
    phase = intents.get('current_phase') if isinstance(intents, dict) else None
    phase = phase if isinstance(phase, str) and phase else None
    executor = actor.get('physmani_geometry_executor')
    events = executor.get('generated_target_admissions') if isinstance(executor, dict) else None
    if not isinstance(events, list):
        return dict(final_geometry_phase=phase, public_forecast_collision_rejections=None,
                    recorded_rejection_reasons=None, admission_events_recorded=False)
    reasons = Counter()
    valid = True
    for event in events:
        if not isinstance(event, dict) or type(event.get('accepted')) is not bool:
            valid = False
            continue
        if event['accepted'] is not False:
            continue
        proof = event.get('proof')
        reason = proof.get('reason') if isinstance(proof, dict) else None
        if not isinstance(reason, str) or not reason:
            valid = False
            continue
        reasons[reason] += 1
    return dict(final_geometry_phase=phase,
                public_forecast_collision_rejections=reasons[COLLISION_REASON] if valid else None,
                recorded_rejection_reasons=dict(reasons) if valid else None,
                admission_events_recorded=valid)


def require_case_owner(actual, stage_owner, row, control, *, expected_environment):
    require(row.get('evidence_runtime') in OWNER_SOURCES, 'unknown actual evidence owner tier')
    pid = stage_owner.get('pid')
    require(type(pid) is int and pid > 0 and type(actual.get('owner_pid')) is int
            and actual['owner_pid'] == pid, 'actual child must retain its real tier owner PID')
    if row['evidence_runtime'] == resumed.RUNTIME:
        require(row.get('imported_completed') is False, 'new823 case cannot impersonate a historical import')
        resumed.require_inherited_environment(actual, stage_owner, control, expected_environment)
    else:
        require(row.get('imported_completed') is True, 'historical case must remain an explicit import')


def require_historical_owner_pointer(tier, pointer, files):
    require(tier in OWNER_SOURCES, 'unknown actual evidence owner tier')
    require_pointer(pointer, filename='launch.json')
    require(pointer['path'] == str(Path(tier) / 'launch.json'), 'actual owner launch path changed')
    if tier != resumed.RUNTIME:
        require(files.get(str((ROOT / pointer['path']).resolve())) == pointer['sha256'],
                'historical owner launch differs from its frozen accepted bytes')


def require_CPU_evidence(CPU, stdout, row, expected, source_row, source_sha, route, *, policy_sha256=None):
    """Bind actual CPU output to this row, including its real method identity."""
    identity = CPU.get('identity')
    require(isinstance(identity, dict), 'actual CPU identity required')
    domain = CPU.get('component_domain', {})
    require(CPU == stdout and CPU.get('complete') is True and CPU.get('native_started') is False
            and CPU.get('GPU_started') is False and CPU.get('execution_protocol_changed') is True
            and CPU.get('actual_split') == 'validation' and domain.get('purpose') == 'evaluation'
            and domain.get('split') == 'validation' and domain.get('task') == row['task']
            and domain.get('method') == row['method'] and domain.get('raw_episode_id') == row['raw']
            and (policy_sha256 is None or domain.get('policy_sha256') == policy_sha256)
            and all(CPU.get(k) == v for k, v in expected.items())
            and identity.get('source_row_sha256') == digest(source_row)
            and identity.get('config_sha256') == route['configuration']['sha256']
            and identity.get('frozen_inputs_sha256') == route['frozen_inputs']['sha256']
            and identity.get('source_input_lock_sha256') == source_sha
            and identity.get('task') == row['task'] and identity.get('raw_episode_id') == row['raw'] + '__policy_aug_r0'
            and identity.get('parent_raw_episode_id') == identity.get('source_raw_episode_id') == row['raw']
            and type(identity.get('replica')) is int and identity['replica'] == 0
            and identity.get('method') == row['method'] and identity.get('split') == 'validation'
            and identity.get('exploration_seed') is None
            and identity.get('sha256') == digest({k: v for k, v in identity.items() if k != 'sha256'})
            and identity.get('demo_only') is False, 'actual CPU identity/source/domain proof changed')
    return identity


class Evidence:
    """Read stable project bytes once and retain their exact SHA and pointer."""
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.inputs = {}

    def path(self, path):
        path = (self.root / path).resolve()
        require(path.is_relative_to(self.root), 'project-only task evidence required')
        return path

    def read(self, path, *, expected=None, json_value=True):
        path = self.path(path)
        before = path.stat()
        data = path.read_bytes()
        after = path.stat()
        signature = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
        require(signature(before) == signature(after), 'evidence changed while being read: ' + str(path))
        checksum = hashlib.sha256(data).hexdigest()
        require(expected is None or checksum == expected, 'frozen evidence SHA changed: ' + str(path))
        require(str(path) not in self.inputs or self.inputs[str(path)] == checksum, 'conflicting evidence bytes')
        self.inputs[str(path)] = checksum
        pointer = dict(path=str(path.relative_to(self.root)), sha256=checksum)
        return (json.loads(data) if json_value else data), pointer

    def pointer(self, pointer, files=None, *, json_value=True):
        require_pointer(pointer)
        path = self.path(pointer['path'])
        if files is not None:
            require(files.get(str(path)) == pointer['sha256'], 'pointer absent from immutable input lock')
        return self.read(path, expected=pointer['sha256'], json_value=json_value)[0]

    def snapshot(self, path):
        """Capture one atomic status version; do not freeze its mutable pathname."""
        path = self.path(path)
        data = path.read_bytes()
        text = data.decode('utf-8')
        value = json.loads(text)
        pointer = dict(path=str(path.relative_to(self.root)), sha256=hashlib.sha256(data).hexdigest())
        # atomic_json replaces this pathname, so the opened bytes are either
        # complete old or complete new JSON. The report retains these exact
        # bytes as UTF8 text, independently of all later owner status writes.
        return value, pointer, text


def audit_task(task, *, runtime=resumed.RUNTIME, resume_control=CONTROL, resume_lock=LOCK):
    """Return one complete task report; never write or require global945 completion."""
    require(task in TASKS, 'only an original VAL task is supported')
    evidence = Evidence(ROOT)
    require(evidence.path(runtime) == ROOT / resumed.RUNTIME, 'exact current823 VAL runtime required')
    control, control_pointer = evidence.read(resume_control)
    lock, lock_pointer = evidence.read(resume_lock)
    require(lock.get('schema') == resumed.LOCK_SCHEMA and lock.get('approved_for_execution') is True,
            'frozen823 control input lock required')
    files = lock['files']
    evidence.pointer(control_pointer, files)
    suite = evidence.pointer(control['suite'], files)
    require(control['suite']['path'] == SUITE, 'unchanged original752 suite required')
    suite_lock = evidence.pointer(control['suite_lock'], files)
    source_lock = evidence.pointer(control['source_inputs'], files)
    require(all(files.get(name) == checksum for name, checksum in suite_lock['files'].items()),
            'environment lock must retain the entire immutable752 input graph')
    previous_control = evidence.pointer(control['previous_resume_control'], files)
    previous_lock = evidence.pointer(control['previous_resume_control_lock'], files)
    require(previous_control.get('schema') == resumed.legacy.CONTROL_SCHEMA
            and previous_lock.get('schema') == resumed.legacy.LOCK_SCHEMA
            and previous_lock.get('approved_for_execution') is True
            and all(files.get(name) == checksum for name, checksum in previous_lock['files'].items()),
            'environment lock must retain the entire original772 input graph')
    require(suite.get('methods') == list(METHODS) and suite.get('tasks') == list(TASKS),
            'original full16 three-method suite required')
    require(source_lock.get('split') == 'validation' and source_lock.get('source_count') == 315
            and source_lock.get('policy_sha256') == control['policy_sha256']
            and source_lock.get('source_manifest_sha256') == suite['source_manifests']['validation']['sha256']
            and source_lock.get('TEST_used_for_fitting') is False, 'original315-source VAL scope required')
    manifest = evidence.pointer(suite['source_manifests']['validation'], files)
    require(manifest.get('split') == 'validation', 'actual VAL manifest required')
    routes = evidence.pointer(suite['routes'], files)
    protocol = suite['author_outcome_protocol']
    require(protocol == control.get('author_outcome_protocol') == routes.get('author_outcome_protocol'),
            'unchanged author-outcome protocol required')
    for key in ('implementation', 'runtime_consumer', 'original_runtime'):
        evidence.pointer(protocol[key], files, json_value=False)
    for key in ('formal_case_driver', 'formal_admission_guard', 'formal_result_auditor'):
        evidence.pointer(suite[key], files, json_value=False)
    for key, name in (('resume_owner', resumed.OWNER), ('resume_auditor', 'scripts/audit_environment_resumed_formal_VAL_v824_r10.py')):
        source_bytes, source_pointer = evidence.read(name, json_value=False)
        require(control.get(key) == source_pointer, 'actual frozen resume implementation required: ' + key)
        require(files.get(str(ROOT / name)) == source_pointer['sha256'], 'resume implementation absent from control lock')
    profile = fingerprint(routes['common_author_RML_profile'])
    prior = {}
    for field in ('previous_plan', 'previous_status', 'previous_terminal'):
        prior[field] = evidence.pointer(control[field], files)
    plan, plan_pointer = evidence.read(Path(runtime) / 'plan.json')
    status, status_pointer, status_text = evidence.snapshot(Path(runtime) / 'status.json')
    resumed.require_resume_plan(control, prior['previous_plan'], prior['previous_status'],
                                prior['previous_terminal'], plan, runtime=ROOT / runtime)
    selected = select_complete_task(plan, status, task, manifest['episodes'])
    selected_ids = {r['job_id'] for r in selected}
    previous_rows = {r['job_id']: r for r in prior['previous_plan']['jobs']}
    old_environment_failures = []
    for name in control['failed_job_ids']:
        if name not in selected_ids:
            continue
        leaf = Path(control['previous_runtime']) / 'native' / name
        terminal, terminal_pointer = evidence.read(leaf / 'terminal.json')
        stderr, stderr_pointer = evidence.read(leaf / 'stderr.log', json_value=False)
        for pointer in (terminal_pointer, stderr_pointer):
            require(files.get(str(ROOT / pointer['path'])) == pointer['sha256'], 'old environment failure evidence changed')
        require(terminal == {k: v for k, v in prior['previous_status']['failed'][name].items() if k != 'stage'},
                'preserved old failure must be its actual old terminal')
        lines = [line.strip() for line in stderr.decode('utf-8').splitlines() if line.strip()]
        prefix = 'our_mani.multitask.protocol.MultitaskProtocolError: frozen input SHA changed: '
        require(lines and lines[-1].startswith(prefix) and lines[-1][len(prefix):] in resumed.environment.RESTORED_TARGETS,
                'only the recorded frozen OS mismatch is an excluded old environment attempt')
        old_out = evidence.path(option(previous_rows[name]['argv'], '--out'))
        require(not (old_out / 'case.json').exists() and not (old_out / 'admission_driver_receipt.json').exists(),
                'old completed case cannot be replaced as an environment failure')
        old_environment_failures.append(dict(job_id=name, evidence_runtime=control['previous_runtime'],
            terminal=terminal_pointer, stderr=stderr_pointer, classification='preserved_infrastructure_failure',
            included_in_task_native_outcomes=False))
    source_rows = {(r['task'], r['raw_episode_id']): r for r in manifest['episodes']}
    source_sha = control['source_inputs']['sha256']
    owners, owner_pointers = {}, {}
    for tier, source in OWNER_SOURCES.items():
        source_bytes, _ = evidence.read(source, json_value=False)
        require(files.get(str(ROOT / source)) == hashlib.sha256(source_bytes).hexdigest(), 'actual frozen tier owner source changed')
        owner, owner_pointer = evidence.read(Path(tier) / 'launch.json')
        require_historical_owner_pointer(tier, owner_pointer, files)
        require(owner.get('source_sha256') == hashlib.sha256(source_bytes).hexdigest()
                and owner.get('split') == 'validation' and owner.get('policy_sha256') == control['policy_sha256']
                and owner.get('author_outcome_protocol') == protocol
                and owner.get('admission_driver') == suite['formal_case_driver']
                and owner.get('admission_guard') == suite['formal_admission_guard']
                and owner.get('suite_lock_sha256') == control['suite_lock']['sha256']
                and owner.get('split_source_lock_sha256') == source_sha, 'real tier owner source/domain/policy binding changed')
        owners[tier], owner_pointers[tier] = owner, owner_pointer
    current = owners[resumed.RUNTIME]
    require(current.get('resume_control') == control_pointer and current.get('resume_control_lock') == lock_pointer,
            'current owner must bind the actual frozen control supplement')
    current_environment = resumed.require_environment_receipt(current.get('frozen_environment'), control, owner_pid=current['pid'])
    for key in resumed.ENVIRONMENT_FIELDS:
        evidence.pointer(control[key], files, json_value=key != 'environment_wrapper')
    records = []
    for row in selected:
        key, tier = row['job_id'], row['evidence_runtime']
        stage, stage_owner = Path(tier), owners[tier]
        route_rows = [r for r in routes['routes'] if (r['task'], r['method']) == (task, row['method'])]
        require(len(route_rows) == 1, 'one original executable route per method required')
        route = route_rows[0]
        for field in ('configuration', 'worker', 'consumer', 'frozen_inputs'):
            evidence.pointer(route[field], files, json_value=False)
        expected = dict(admission_driver=suite['formal_case_driver'], admission_guard=suite['formal_admission_guard'],
                        case_worker=route['worker'], native_runtime={k: route['consumer'][k] for k in ('path', 'sha256')},
                        original_native_runtime=protocol['original_runtime'], author_outcome_protocol=protocol)
        require(all(row.get(k) == v for k, v in expected.items())
                and row.get('worker_sha256') == expected['admission_driver']['sha256']
                and row.get('configuration_sha256') == route['configuration']['sha256']
                and row.get('input_lock_sha256') == route['frozen_inputs']['sha256']
                and row.get('source_row_sha256') == digest(source_rows[(task, row['raw'])]), 'frozen original case route/source changed')
        require(row['argv'][:3] == ['bash', 'scripts/run_aligned_gpu.sh' if row['method'] == 'physmani' else 'scripts/run_local_sim.sh', suite['formal_case_driver']['path']],
                'original formal launcher route required')
        for flag, value in (('--split', 'validation'), ('--task', task), ('--raw-episode-id', row['raw']),
                            ('--method', row['method']), ('--suite', control['suite']['path']),
                            ('--suite-lock', control['suite_lock']['path']), ('--source-inputs', control['source_inputs']['path']),
                            ('--manifest', suite['source_manifests']['validation']['path']),
                            ('--config', route['configuration']['path']), ('--frozen-inputs', route['frozen_inputs']['path'])):
            require(option(row['argv'], flag) == value, 'actual unchanged VAL argv required: ' + flag)
        CPU_leaf, native_leaf = stage / 'CPU' / key, stage / 'native' / key
        CPU, CPU_pointer = evidence.read(CPU_leaf / 'CPU_preflight.json')
        stdout, _ = evidence.read(CPU_leaf / 'stdout.log')
        identity = require_CPU_evidence(CPU, stdout, row, expected, source_rows[(task, row['raw'])], source_sha, route,
                                        policy_sha256=control['policy_sha256'])
        launches = {}
        for kind, leaf in (('CPU', CPU_leaf), ('native', native_leaf)):
            terminal, terminal_pointer = evidence.read(leaf / 'terminal.json')
            resumed.require_clean_terminal(terminal)
            actual, actual_pointer = evidence.read(leaf / 'launch.json')
            base.require_actual_launch(actual, row, expected, check=kind == 'CPU', identity=identity)
            require_case_owner(actual, stage_owner, row, control, expected_environment=current_environment)
            for filename in ('stdout.log', 'stderr.log'):
                _, log_pointer = evidence.read(leaf / filename, json_value=False)
                if row['imported_completed']:
                    require(files.get(str(ROOT / log_pointer['path'])) == log_pointer['sha256'],
                            'historical log no longer matches frozen accepted evidence')
            if row['imported_completed']:
                for pointer in (actual_pointer, terminal_pointer):
                    require(files.get(str(ROOT / pointer['path'])) == pointer['sha256'],
                            'historical CPU/native provenance differs from frozen evidence')
            launches[kind] = (actual_pointer, terminal, terminal_pointer)
        out = evidence.path(option(row['argv'], '--out'))
        case, case_pointer = evidence.read(out / 'case.json')
        driver, driver_pointer = evidence.read(out / 'admission_driver_receipt.json')
        base.require_driver_evidence(driver, suite['formal_case_driver'], suite['formal_admission_guard'], route, out / 'case.json', 'validation')
        base.require_native_protocol(case, protocol)
        validate_completed_case(case, identity)
        require_attestation(case.get('common_author_RML_parameters', {}), profile)
        require(case.get('cleanup_completed') is True and case.get('component_domain', {}).get('purpose') == 'evaluation'
                and case['component_domain'].get('split') == 'validation'
                and case['component_domain'].get('task') == task
                and case['component_domain'].get('method') == row['method']
                and case['component_domain'].get('raw_episode_id') == row['raw']
                and case['component_domain'].get('policy_sha256') == control['policy_sha256']
                and case.get('full16_frozen_execution', {}).get('policy_sha256') == control['policy_sha256']
                and len(case.get('all_stage_counted_joint_motion', {}).get('events', [])) == case['counted_clock']['ticks'],
                'full physical domain/policy/cleanup/counting evidence required')
        result = status['completed'][key]
        native = launches['native'][1]
        require(native.get('case') == result['case'] == case_pointer
                and native.get('admission_driver_receipt') == result['admission_driver_receipt'] == driver_pointer
                and result.get('native_success') is case['official_task_success']
                and result.get('motion_pass') is measured_motion(case), 'status and actual native evidence differ')
        if row['imported_completed']:
            previous = prior['previous_status']['completed'][key]
            require(previous.get('case') == case_pointer and previous.get('admission_driver_receipt') == driver_pointer,
                    'historical accepted case must retain its exact evidence bytes')
            for pointer in (case_pointer, driver_pointer, CPU_pointer, launches['CPU'][0], launches['native'][0]):
                require(files.get(str(ROOT / pointer['path'])) == pointer['sha256'], 'imported evidence not frozen in control lock')
        records.append(dict(job_id=key, source_raw_episode_id=row['raw'], source_row_sha256=identity['source_row_sha256'],
                            method=row['method'], case=case_pointer, evidence_runtime=tier,
                            imported_completed=row['imported_completed'], previous_imported_completed=row['previous_imported_completed'],
                            native_terminal=launches['native'][2], external_driver_receipt=driver_pointer,
                            CPU_receipt=CPU_pointer, CPU_launch=launches['CPU'][0], native_launch=launches['native'][0],
                            actual_owner_pid=stage_owner['pid'], actual_owner_launch=owner_pointers[tier], actual_argv=row['argv'],
                            official_task_success=case['official_task_success'], native_terminal_failure=case['native_terminal_failure'],
                            counted_ticks=case['counted_clock']['ticks'], classification=classify_case(case),
                            cleanup_completed=True, infrastructure_error=None, **geometry_diagnostics(case, row['method'])))
    statistics = {}
    for method in METHODS:
        group = [r for r in records if r['method'] == method]
        counts = Counter(r['classification'] for r in group)
        instrumented = [r for r in group if r['admission_events_recorded']]
        reasons = Counter()
        for row in instrumented:
            reasons.update(row['recorded_rejection_reasons'])
        statistics[method] = dict(cases=len(group), **{k: counts[k] for k in ('native_success', 'native_terminal_failure', 'budget_exhausted_no_native_terminal', 'explicit_policy_failure', 'unknown')},
            infrastructure_errors=0, cleanup_incomplete=0,
            final_geometry_phase=dict(Counter(r['final_geometry_phase'] or 'unknown' for r in group)),
            public_forecast_collision_rejections=sum(r['public_forecast_collision_rejections'] for r in instrumented) if len(instrumented) == len(group) else None,
            public_forecast_collision_cases=sum(r['public_forecast_collision_rejections'] > 0 for r in instrumented) if len(instrumented) == len(group) else None,
            target_admission_metrics_cases=len(instrumented), target_admission_metrics_missing_cases=len(group) - len(instrumented),
            all_recorded_rejection_reasons=dict(reasons) if len(instrumented) == len(group) else None)
    evidence.read(__file__, json_value=False)
    return dict(schema='r10_complete_task_actual_VAL_readonly_results_v833', created_utc=datetime.now(timezone.utc).isoformat(),
        complete=True, read_only_analysis=True, task=task, split='validation', source_cases=len(records) // 3,
        method_cases=len(records), methods=list(METHODS), source_raw_episode_ids=sorted({r['source_raw_episode_id'] for r in records}),
        same_exact_original_sources_for_all_methods=True, unique_job_ids=len(records), statistics=statistics, case_evidence=records,
        missing_or_invalid_fields=[],
        unknown_diagnostic_fields=[dict(job_id=r['job_id'], fields=[field for field in ('final_geometry_phase', 'public_forecast_collision_rejections', 'recorded_rejection_reasons') if r[field] is None])
                                   for r in records if any(r[field] is None for field in ('final_geometry_phase', 'public_forecast_collision_rejections', 'recorded_rejection_reasons'))],
        status_snapshot_utc=status.get('updated_utc'), status_snapshot_sha256=status_pointer['sha256'],
        status_snapshot_source_path=status_pointer['path'], status_snapshot_json_utf8=status_text,
        status_snapshot_mutable_source_is_frozen_input=False,
        status_snapshot_total_completed=len(status['completed']), status_snapshot_target_completed_records={r['job_id']: status['completed'][r['job_id']] for r in selected},
        resume_control=control_pointer, resume_control_lock=lock_pointer, actual_stage_launch=owner_pointers[resumed.RUNTIME],
        actual_stage_launches=owner_pointers, actual_plan=plan_pointer, policy_sha256=control['policy_sha256'],
        evidence_runtime_provenance_counts=dict(Counter(r['evidence_runtime'] for r in records)),
        preserved_old_environment_failed_attempts=[item['job_id'] for item in old_environment_failures],
        preserved_old_environment_failed_evidence=old_environment_failures,
        old_environment_failures_are_native_outcomes=False,
        classification_rule='native success,then native terminal failure,then explicit policy failure,then200 counted ticks with no native terminal;otherwise unknown;native failure wins over budget',
        collision_count_definition=dict(field='actor_metrics.physmani_geometry_executor.generated_target_admissions', accepted=False,
            reason_field='proof.reason', exact_reason=COLLISION_REASON, unit='recorded admission rejection events,not physical collisions or pairs',
            baseline_uninstrumented_is_unknown_not_zero=True, case_wide_rejection_counts_not_attributed_to_final_phase=True),
        named_collision_pair='not recorded;no pair or root cause inferred', geometry_phase_field='actor_metrics.geometry_intents.current_phase',
        baseline_geometry_phase='unknown when not recorded', full945_VAL_release_granted=False,
        scope='one complete original task VAL subset only,not complete945 release', TEST_data_read=False,
        execution_or_configuration_modified=False, live_environment_probe_performed=False, source_sha256=file_sha(__file__), inputs_sha256=evidence.inputs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=TASKS, required=True)
    parser.add_argument('--runtime', default=resumed.RUNTIME)
    parser.add_argument('--resume-control', default=CONTROL)
    parser.add_argument('--resume-lock', default=LOCK)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    out = (ROOT / args.out).resolve()
    require(out.is_relative_to(ROOT / 'runtime/framework') and not out.exists(), 'fresh runtime task-report output required')
    report = audit_task(args.task, runtime=args.runtime, resume_control=args.resume_control, resume_lock=args.resume_lock)
    require(not out.exists(), 'preserve existing task report')
    atomic_json(out, report)
    print(json.dumps(dict(complete=True, task=args.task, cases=report['method_cases'], out=str(out.relative_to(ROOT)), full945_VAL_release_granted=False)))


if __name__ == '__main__':
    main()
