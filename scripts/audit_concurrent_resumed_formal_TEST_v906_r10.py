"""Independent, inactive composite audit of the original753/new902 TEST grid.

The frozen original scientific admission and full input verification remain.
Frozen pointer lookups are one scoped transaction; runtime evidence is read
from explicit plan paths and checked again at close, never by directory glob.
This auditor does not launch work, retry cases, train, or require task success.
"""
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import argparse
import hashlib
import json
import math
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from our_mani.multitask.evaluation import atomic_json, digest, file_sha, validate_completed_case
from our_mani.multitask.evaluation_roster16_v283 import TASKS, METHODS
from our_mani.multitask.input_stat_cache_v79 import verify_inputs, checked_sha, stamp
from our_mani.multitask.frozen_input_edges_v463 import edge_stamp
from our_mani.multitask.frozen_lookup_batch_v900 import LookupBatch
from our_mani.multitask.formal_suite_admission_v728 import read_formal_source
from our_mani.multitask.common_rml_protocol_v530 import fingerprint, require_attestation
from our_mani.multitask.three_method_release16 import measured_motion
from our_mani.multitask.three_method_release_v531 import audit_grid
from our_mani.multitask.rml_case_preflight_v742 import outcome_protocol_binding
from scripts import run_concurrent_resumed_formal_TEST_owner_v902_r10 as owner
from scripts import run_formal_suite16_owner_v753_r10 as original_owner
from scripts import audit_formal_suite16_results_v754_r10 as original_audit
from scripts import audit_environment_resumed_formal_VAL_v824_r10 as environment_audit
from scripts.run_formal_case_author_outcome_v751_r10 import BINDING_KEYS, require_native_protocol

SOURCE = 'scripts/audit_concurrent_resumed_formal_TEST_v906_r10.py'
SCHEMA = 'r10_composite_concurrent_resumed_formal_TEST_results_v906'
OUTPUT = 'artifacts/framework/multitask_r10/formal_TEST_concurrent_resume_v902/TEST_results.json'
ORIGINAL_OUTPUT = 'artifacts/framework/multitask_r10/formal_VAL_env_resume_v823/test'
ENVIRONMENT_FIELDS = ('environment_wrapper', 'environment_manifest', 'environment_namespace_proof')
CONTROL_POINTERS = ('resume_owner', 'resume_auditor', 'suite', 'suite_lock', 'source_inputs', 'validation_release',
    'resource_qualification', 'resource_qualifier', 'standing_continuation_record', 'concurrency_authorization_record',
    'previous_launch', 'previous_plan', 'previous_status', 'previous_terminal', 'previous_controller_launch',
    'previous_controller_status', 'previous_controller_failure', 'previous_DRAIN', *ENVIRONMENT_FIELDS)
SOURCES = (SOURCE, 'scripts/run_concurrent_resumed_formal_TEST_owner_v902_r10.py',
    'scripts/audit_formal_suite16_results_v754_r10.py', 'scripts/audit_environment_resumed_formal_VAL_v824_r10.py',
    'our_mani/multitask/frozen_lookup_batch_v900.py', 'our_mani/multitask/author_outcome_receipt_v749.py',
    'our_mani/multitask/evaluation.py', 'our_mani/multitask/evaluation_roster16_v283.py',
    'our_mani/multitask/input_stat_cache_v79.py', 'our_mani/multitask/frozen_input_edges_v463.py',
    'our_mani/multitask/common_rml_protocol_v530.py', 'our_mani/multitask/three_method_release16.py',
    'our_mani/multitask/three_method_release_v531.py', 'our_mani/multitask/rml_case_preflight_v742.py',
    'scripts/run_formal_case_author_outcome_v751_r10.py', 'scripts/run_formal_suite16_owner_v753_r10.py')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def require_not_stopped():
    names = ('runtime/framework/multitask_r10_STOP', owner.RUNTIME+'/STOP',
             owner.PREVIOUS+'/STOP', owner.PREVIOUS_CONTROLLER+'/STOP')
    require(not any((ROOT/name).exists() for name in names), 'STOP has priority over the independent audit')


def require_clean_terminal(value):
    environment_audit.require_clean_terminal(value)
    require(value.get('error') is None and value.get('leaf_future_exception') is not True,
            'ordinary infrastructure errors cannot be clean native evidence')


def require_resume_plan(control, old_plan, old_status, old_terminal, new_plan):
    """Pure, full1920 provenance check; False outcomes are not filtered."""
    old_by_id = owner.require_partition(control, old_plan, old_status, old_terminal)
    expected = owner.resumed_jobs(old_plan, control, control['execution_entry'])
    require(new_plan.get('jobs') == expected and new_plan.get('expected_cases') == 1920
        and new_plan.get('imported_completed_cases') == len(control['completed_job_ids'])
        and new_plan.get('source_outcome_filtering') is False,
        'new plan must preserve every exact old row beneath explicit902 provenance')
    return old_by_id


def require_mixed_launch(actual, row, bindings, *, check, identity, stage_owner, control):
    """Validate the actual tier, without relabelling a901 entry as direct751."""
    original_audit.require_actual_launch(actual, row, bindings, check=check, identity=identity)
    require(type(actual.get('launcher_pid')) is int and actual['launcher_pid'] > 0
        and type(actual.get('launcher_start_ticks')) is int and actual['launcher_start_ticks'] > 0
        and type(stage_owner.get('pid')) is int and actual.get('owner_pid') == stage_owner['pid'],
        'child PID/start and real evidence-tier owner must be explicit')
    if row['imported_completed']:
        require(row['evidence_runtime'] == owner.PREVIOUS and row['execution_entry']['kind'] == 'original751'
            and row['physical_argv'] == row['argv'], 'old753 evidence cannot acquire a new entry or runtime')
        # These fields were never written by original753; do not fabricate them.
        require(all(key not in actual for key in ('execution_entry', 'physical_argv', 'inherited_frozen_environment',
            'current_mount_namespace')), 'old753 launch must remain its original recorded format')
    else:
        first = environment_audit.require_environment_receipt(stage_owner.get('frozen_environment'), control,
            owner_pid=stage_owner['pid'])
        require(row['evidence_runtime'] == owner.RUNTIME
            and actual.get('execution_entry') == row['execution_entry']
            and actual.get('physical_argv') == row['physical_argv'] + (['--check'] if check else [])
            and actual.get('inherited_frozen_environment') == first
            and actual.get('current_mount_namespace') == first['current_mount_namespace'],
            'new902 child must disclose its actual entry, forwarded argv and inherited owner namespace')


def require_case_outcome(case, identity, protocol, policy, profile):
    """Original741/749, physical clock and RML checks, without a success gate."""
    require(identity.get('sha256') == digest({key: value for key, value in identity.items() if key != 'sha256'}),
            'actual CPU/case identity digest must match its exact fields')
    validate_completed_case(case, identity)
    require_native_protocol(case, protocol)
    require_attestation(case.get('common_author_RML_parameters', {}), profile)
    domain = case.get('component_domain', {})
    witness = case.get('full16_frozen_execution', {})
    motion = case.get('all_stage_counted_joint_motion', {})
    ticks = case['counted_clock']['ticks']
    require(case.get('cleanup_completed') is True and domain.get('purpose') == 'evaluation'
        and domain.get('split') == 'test' and identity.get('split') == 'test'
        and witness.get('policy_sha256') == policy and witness.get('source_row_sha256') == identity['source_row_sha256']
        and isinstance(motion.get('events'), list) and len(motion['events']) == ticks
        and type(measured_motion(case)) is bool,
        'native cleanup, evaluation scope, policy, source and each counted motion window must agree')
    return dict(native_success=case['official_task_success'], motion_pass=measured_motion(case), ticks=ticks)


def _peak(selected):
    values = [c.get('all_stage_counted_joint_motion', {}).get('peak_actual_window_acceleration') for c in selected]
    known = [v for v in values if type(v) in (int, float) and math.isfinite(v) and v >= 0]
    return (max(known) if known else None), len(values) - len(known)


def summarize(cases):
    """Per-task native outcomes and unchanged diagnostic motion, case weighted."""
    result = []
    for task in TASKS:
        methods = {}
        for method in METHODS:
            selected = [c for c in cases if c['task'] == task and c['method'] == method]
            successes = sum(c['official_task_success'] for c in selected)
            peak, unknown = _peak(selected)
            methods[method] = dict(successes=successes, episodes=len(selected),
                success_rate=successes / len(selected) if selected else None,
                native_terminal_failures=sum(c['native_terminal_failure'] for c in selected),
                budget_exhausted=sum(not c['official_task_success'] and not c['native_terminal_failure']
                    and c['counted_clock']['ticks'] == 200 for c in selected),
                other_nonterminal_stops=sum(not c['official_task_success'] and not c['native_terminal_failure']
                    and c['counted_clock']['ticks'] < 200 for c in selected),
                zero_tick_cases=sum(c['counted_clock']['ticks'] == 0 for c in selected),
                actual_window_acceleration_peak=peak, actual_window_acceleration_peak_unknown=unknown,
                measured_motion_violation_cases=sum(measured_motion(c) is False for c in selected))
        result.append(dict(task=task, methods=methods))
    return result


class EvidenceLedger:
    """Exact used files with strong identity and ancestor checks at both ends.

    Dynamic native outputs are not silently inserted into a frozen data lock.
    Their actual byte SHAs and paths form independent result evidence. Original
    locks are handled by LookupBatch and unchanged full verify_inputs calls.
    """
    def __init__(self, root, *, cache=None):
        self.root = Path(root).resolve()
        self.cache = cache
        self.used = {}
        self.closed = False

    def signature(self, path):
        requested = self.root / path
        canonical = requested.resolve(strict=True)
        require(canonical.is_relative_to(self.root) and canonical == requested,
                'canonical project evidence paths required')
        edges = tuple((str(p), edge_stamp(p)) for p in (requested, *requested.parents))
        return str(canonical), stamp(canonical), edges

    def read(self, path, *, expected=None, parse=False):
        require(not self.closed, 'evidence ledger already closed')
        path = self.root / path
        first = self.signature(path)
        payload = path.read_bytes()
        actual = hashlib.sha256(payload).hexdigest()
        require(self.signature(path) == first and (expected is None or actual == expected),
                'evidence bytes/path changed: ' + str(path))
        old = self.used.get(str(path))
        record = dict(pointer=dict(path=str(path.relative_to(self.root)), sha256=actual), signature=first)
        require(old is None or old == record, 'previously read evidence changed: ' + str(path))
        self.used[str(path)] = record
        return json.loads(payload) if parse else record['pointer']

    def json(self, path, *, expected=None):
        return self.read(path, expected=expected, parse=True)

    def pointer(self, path):
        return self.read(path)

    def frozen(self, batch, files, pointer, *, parse=True):
        batch.locked_pointer(self.root, files, pointer)
        return self.read(pointer['path'], expected=pointer['sha256'], parse=parse)

    def close(self):
        snapshot = deepcopy(self.used)
        for name, value in snapshot.items():
            require(self.signature(name) == value['signature'], 'used evidence path changed before audit end')
            if self.cache is None:
                actual, _ = checked_sha(name, value['pointer']['sha256'])
            else:
                actual, _ = checked_sha(name, value['pointer']['sha256'], cache=self.cache)
            require(actual == value['pointer']['sha256'] and self.signature(name) == value['signature'],
                    'used evidence changed during final byte/stat verification')
        require(self.used == snapshot, 'evidence key/value set changed at audit close')
        self.closed = True
        return {name: value['pointer']['sha256'] for name, value in snapshot.items()}


def require_original_plan(old_plan, control, suite, sources, proof, driver, guard):
    rows = {(r['task'], r['raw_episode_id']): r for r in sources}
    routes = {(r['task'], r['method']): r for r in proof['routes']['routes']}
    for row in old_plan['jobs']:
        route = routes[row['task'], row['method']]
        source = rows[row['task'], row['raw']]
        bindings = dict(admission_driver=driver, admission_guard=guard, case_worker=route['worker'],
            native_runtime={k: route['consumer'][k] for k in ('path', 'sha256')},
            original_native_runtime=control['author_outcome_protocol']['original_runtime'],
            author_outcome_protocol=control['author_outcome_protocol'])
        require(all(row.get(k) == v for k, v in bindings.items())
            and route['worker']['path'] == 'scripts/run_task_executor_smoke_author_outcome_v743_r10.py'
            and route['consumer']['path'] == 'scripts/run_task_executor_smoke_author_suite16_v744_r10.py'
            and row.get('worker_sha256') == driver['sha256']
            and row.get('source_row_sha256') == digest(source)
            and row.get('configuration_sha256') == route['configuration']['sha256']
            and row.get('input_lock_sha256') == route['frozen_inputs']['sha256'],
            'original753 row differs from admitted immutable route/source')
        argv = ['bash', 'scripts/run_aligned_gpu.sh' if row['method'] == 'physmani' else 'scripts/run_local_sim.sh',
            driver['path'], '--config', route['configuration']['path'], '--manifest', suite['source_manifests']['test']['path'],
            '--frozen-inputs', route['frozen_inputs']['path'], '--suite', control['suite']['path'],
            '--suite-lock', control['suite_lock']['path'], '--source-inputs', control['source_inputs']['path'],
            '--task', row['task'], '--raw-episode-id', row['raw'], '--method', row['method'], '--split', 'test',
            '--out', str(Path(ORIGINAL_OUTPUT) / row['task'] / row['raw'] / row['method']),
            '--validation-release', control['validation_release']['path']]
        require(row['argv'] == argv, 'original753 actual argv/output scope changed')


def require_owner_records(control, old_launch, launch, plan, status, terminal, *, source_sha, suite_lock_sha,
                          driver, guard, resume_control, resume_lock, expected_environment=None):
    """Pure full owner provenance; only902 owns the new physical children."""
    protocol = control['author_outcome_protocol']
    bindings = dict(split='test', policy_sha256=control['policy_sha256'], admission_driver=driver,
        admission_guard=guard, execution_protocol_changed=True, native_runtime=protocol['runtime_consumer'],
        original_native_runtime=protocol['original_runtime'], author_outcome_protocol=protocol, automatic_retry=False)
    for key, value in bindings.items():
        require(old_launch.get(key) == value, 'original753 owner binding changed: ' + key)
        require(all(v.get(key) == value for v in (launch, plan, status, terminal)),
                'new902 owner records changed binding: ' + key)
    for value, source in ((old_launch, 'scripts/run_formal_suite16_owner_v753_r10.py'),
                           (launch, 'scripts/run_concurrent_resumed_formal_TEST_owner_v902_r10.py')):
        require(value.get('source_sha256') == file_sha(ROOT / source)
            and value.get('suite_lock_sha256') == suite_lock_sha and value.get('split_source_lock_sha256') == source_sha
            and type(value.get('pid')) is int and value['pid'] > 0
            and type(value.get('start_ticks')) is int and value['start_ticks'] > 0,
            'actual stage-owner PID/start/source/input locks required')
    common = dict(resume_control=resume_control, resume_control_lock=resume_lock,
        resource_profile=control['resource_profile'], resource_qualification=control['resource_qualification'],
        execution_implementation=control['execution_implementation'])
    for key, value in common.items():
        require(all(v.get(key) == value for v in (launch, plan, status, terminal)),
                'new902 immutable continuation/resource binding differs: ' + key)
    require(launch.get('execution_entry') == control['execution_entry']
        and launch.get('base_stage_owner') == owner.pointer(ROOT / 'scripts/run_formal_suite16_owner_v753_r10.py'),
        'new902 actual entry and original owner source required')
    profile = owner.validate_resource_profile(control['resource_profile'])
    require(all(launch.get(key) == profile[key] for key in ('max_native_slots', 'max_GPU_slots', 'max_CPU_preflight_slots'))
        and old_launch.get('max_native_slots') == 2 and old_launch.get('max_GPU_slots') == 1
        and old_launch.get('max_CPU_preflight_slots') == 2,
        'each actual stage owner must disclose its real, separately qualified resource quotas')
    argv = launch.get('argv')
    expected_options = {'--suite': control['suite']['path'], '--suite-lock': control['suite_lock']['path'],
        '--source-inputs': control['source_inputs']['path'], '--validation-release': control['validation_release']['path'],
        '--runtime': owner.RUNTIME, '--out': owner.OUTPUT, '--resume-control': resume_control['path'],
        '--resume-lock': resume_lock['path'], '--resume-lock-sha256': resume_lock['sha256']}
    require(isinstance(argv, list) and len(argv) == 1+2*len(expected_options)
        and (ROOT/argv[0]).resolve() == (ROOT/owner.__file__).resolve()
        and all(owner.option(argv, flag) == value for flag, value in expected_options.items()),
        'new902 real owner invocation must bind every scope and externally pinned resume-lock SHA')
    count = len(control['completed_job_ids'])
    require(launch.get('imported_completed_cases') == count and plan.get('imported_completed_cases') == count
        and status.get('imported_completed') == count and terminal.get('imported_completed_cases') == count
        and terminal.get('resumed_new_cases') == 1920-count
        and terminal.get('complete') is True and terminal.get('expected') == terminal.get('completed') == 1920
        and terminal.get('failed') == 0 and terminal.get('owner_error') is None
        and terminal.get('stop_requested') is False and terminal.get('drain_requested') is False
        and terminal.get('drained') is False and terminal.get('CPU_not_launched') == 0
        and terminal.get('CPU_ready_unexecuted') == []
        and terminal.get('formal_release_requires_independent_audit') is True
        and status.get('expected') == 1920 and len(status.get('completed', {})) == 1920
        and status.get('failed') == {} and status.get('active') == status.get('checking') == status.get('ready_jobs') == []
        and status.get('CPU_ready') == status.get('native_slots') == status.get('GPU_slots') == 0
        and status.get('CPU_launched') == 1920-count and status.get('owner_error') is None
        and status.get('stop_requested') is False and status.get('drain_requested') is False,
        'complete unique1920 clean terminal/status required; drain or STOP is not full results')
    first = environment_audit.require_environment_receipt(launch.get('frozen_environment'), control,
        owner_pid=launch['pid'], expected=expected_environment)
    require(first['observer_start_ticks'] == launch['start_ticks'], 'real902 owner namespace start differs')
    require(all(value.get('frozen_environment') == first for value in (plan, status, terminal)),
            'new902 environment receipt must remain its real owner capture')
    return first


def require_control(control, files, batch, ledger, control_pointer, lock_pointer):
    require(control.get('schema') == owner.CONTROL_SCHEMA and control.get('split') == 'test'
        and control.get('expected_cases') == 1920 and control.get('source_count') == 640
        and control.get('previous_runtime') == owner.PREVIOUS and control.get('previous_controller_runtime') == owner.PREVIOUS_CONTROLLER
        and control.get('user_authorized_resume') is True and control.get('standing_authorization_reused') is True
        and control.get('new_user_event_invented') is False and control.get('physical_policy_changed') is False,
        'frozen authorized unchanged-policy concurrent TEST control required')
    owner.validate_resource_profile(control['resource_profile'])
    entry = owner.validate_entry(control['execution_entry'])
    for key in CONTROL_POINTERS:
        require(owner.valid_pointer(control.get(key)), 'exact immutable control pointer required: ' + key)
        ledger.frozen(batch, files, control[key], parse=False)
    for name in SOURCES:
        ledger.frozen(batch, files, owner.pointer(ROOT/name), parse=False)
    require(control['resume_owner'] == owner.pointer(ROOT / owner.__file__)
        and control['resume_auditor'] == owner.pointer(ROOT / SOURCE), 'exact902 owner and independent906 source required')
    require(control['resource_qualifier']['path'] == 'scripts/audit_concurrent_TRAIN_resource_canary_v904_r10.py',
            'resource qualification must belong to independent904, never the903 runner itself')
    for key, runtime, name in (
        ('previous_launch', owner.PREVIOUS, 'launch.json'), ('previous_plan', owner.PREVIOUS, 'plan.json'),
        ('previous_status', owner.PREVIOUS, 'status.json'), ('previous_terminal', owner.PREVIOUS, 'terminal.json'),
        ('previous_DRAIN', owner.PREVIOUS, 'DRAIN'), ('previous_controller_launch', owner.PREVIOUS_CONTROLLER, 'launch.json'),
        ('previous_controller_status', owner.PREVIOUS_CONTROLLER, 'status.json'),
        ('previous_controller_failure', owner.PREVIOUS_CONTROLLER, 'failure.json')):
        require(control[key]['path'] == str(Path(runtime)/name), 'actual historical drain evidence path differs')
    standing = ledger.json(control['standing_continuation_record']['path'])
    authorization = ledger.json(control['concurrency_authorization_record']['path'])
    require(standing.get('continue_until_user_stop') is True and standing.get('user_stop_received') is False
        and authorization.get('user_authorized_concurrency_change') is True, 'real standing/user concurrency authorization required')
    ctl_status = ledger.json(control['previous_controller_status']['path'])
    ctl_failure = ledger.json(control['previous_controller_failure']['path'])
    require(ctl_status.get('stage') == 'requires_concrete_repair' and ctl_status.get('automatic_retry') is False
        and ctl_failure.get('complete') is False and ctl_failure.get('user_stop') is False
        and ctl_failure.get('automatic_retry') is False, 'genuine827 planned-drain failure/status required, never a made-up terminal')
    sources = {name: owner.pointer(ROOT/name) for name in owner.COMMON_SOURCES}
    if entry['kind'] == 'batched901':
        from scripts import run_batched_formal_driver_v901_r10 as bootstrap
        for key in ('batch_control', 'batch_lock'):
            ledger.frozen(batch, files, entry[key], parse=False)
        small = ledger.json(entry['batch_lock']['path'])
        require(small.get('schema') == bootstrap.LOCK_SCHEMA and small.get('execution_enabled') is True,
                'real enabled901 source/control supplement required')
        sources = {name: owner.pointer(ROOT/name) for name in bootstrap.SOURCES}
        for pointer in sources.values():
            ledger.frozen(batch, small['files'], pointer, parse=False)
        # Source binding is read-only. The original901 gate already verified its
        # complete small control; independently repeat it on one real row below.
    for pointer in sources.values():
        ledger.frozen(batch, files, pointer, parse=False)
    require(control.get('execution_implementation') == owner.execution_implementation(entry, sources),
            'resource qualification and actual TEST implementation differ')
    owner.require_resource_qualification(ledger.json(control['resource_qualification']['path']), control)
    return sources if entry['kind'] == 'batched901' else {}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime', default=owner.RUNTIME)
    parser.add_argument('--out', default=OUTPUT)
    parser.add_argument('--resume-control', required=True)
    parser.add_argument('--resume-lock', required=True)
    parser.add_argument('--resume-lock-sha256', required=True)
    args = parser.parse_args()
    require(args.runtime == owner.RUNTIME and args.out == OUTPUT, 'exact independent906 full TEST audit namespace required')
    out = ROOT/args.out
    require(not out.exists(), 'preserve prior independent TEST results; no overwrite/retry')
    require_not_stopped()
    for name in (args.resume_control, args.resume_lock):
        require((ROOT/name).resolve(strict=True).is_relative_to(ROOT/'artifacts/framework/multitask_r10'),
                'separately frozen project artifact control/lock required')
    ledger = EvidenceLedger(ROOT)
    lock_pointer = ledger.pointer(args.resume_lock)
    require(owner.valid_pointer(lock_pointer) and lock_pointer['sha256'] == args.resume_lock_sha256,
            'externally pinned actual resume lock SHA required')
    lock = ledger.json(args.resume_lock)
    require(lock.get('schema') == owner.LOCK_SCHEMA and lock.get('approved_for_execution') is True
        and isinstance(lock.get('files'), dict), 'frozen902 execution supplement required')
    files = lock['files']
    control_pointer = ledger.pointer(args.resume_control)
    names = (str(ROOT/control_pointer['path']), control_pointer['path'])
    pinned = [files[name] for name in names if name in files]
    require(len(pinned) == 1 and pinned[0] == control_pointer['sha256'],
            'exact control bytes must be pinned before following any release/data pointer')
    control = ledger.json(args.resume_control)
    # Complete genuine VAL is verified before a full input graph can touch TEST.
    release = owner.require_complete_VAL_release(control, control['validation_release']['path'])
    verify_inputs(ROOT, files)
    audit_environment = environment_audit.capture_environment(control)
    with LookupBatch(ROOT) as batch:
        ledger.frozen(batch, files, control_pointer, parse=False)
        batch_sources = require_control(control, files, batch, ledger, control_pointer, lock_pointer)
        suite_lock = ledger.json(control['suite_lock']['path'])
        suite_files = suite_lock['files']
        verify_inputs(ROOT, suite_files)
        suite = ledger.frozen(batch, suite_files, control['suite'])
        expected = {'formal_case_driver': owner.ORIGINAL_ENTRY,
            'formal_admission_guard': 'our_mani/multitask/formal_suite_admission_v728.py',
            'formal_result_auditor': 'scripts/audit_formal_suite16_results_v754_r10.py'}
        for key, name in expected.items():
            require(suite.get(key) == owner.pointer(ROOT/name), 'original frozen formal suite binding differs: '+key)
            ledger.frozen(batch, suite_files, suite[key], parse=False)
        protocol = outcome_protocol_binding(ROOT)
        require(suite.get('author_outcome_protocol') == control['author_outcome_protocol'] == protocol,
                'original741 protocol must remain identical')
        for key in ('implementation', 'runtime_consumer', 'original_runtime'):
            ledger.frozen(batch, suite_files, protocol[key], parse=False)
        for name in ('our_mani/multitask/rml_case_preflight_v742.py', 'scripts/run_formal_suite16_owner_v753_r10.py'):
            ledger.frozen(batch, suite_files, owner.pointer(ROOT/name), parse=False)
        driver, guard = suite['formal_case_driver'], suite['formal_admission_guard']
        manifest, sources, proof = read_formal_source(ROOT, suite, suite_files, 'test', validation_release=release)
        policy = proof['policy_sha256']
        require(policy == control['policy_sha256'] and len(sources) == 640, 'same640 original TEST policy/source scope required')
        source_meta = ledger.json(control['source_inputs']['path'])
        source_sha = control['source_inputs']['sha256']
        require(source_meta.get('schema') == 'r10_full16_post_admission_split_source_lock_v1'
            and source_meta.get('split') == 'test' and source_meta.get('source_count') == 640
            and source_meta.get('policy_sha256') == policy and source_meta.get('TEST_used_for_fitting') is False
            and source_meta.get('source_manifest_sha256') == suite['source_manifests']['test']['sha256'],
            'same original admitted TEST split source lock required')
        verify_inputs(ROOT, source_meta['files'])
        old_launch = ledger.json(control['previous_launch']['path'])
        old_plan = ledger.json(control['previous_plan']['path'])
        old_status = ledger.json(control['previous_status']['path'])
        old_terminal = ledger.json(control['previous_terminal']['path'])
        runtime = ROOT/args.runtime
        launch, plan, status, terminal = (ledger.json(runtime/name) for name in ('launch.json', 'plan.json', 'status.json', 'terminal.json'))
        require_resume_plan(control, old_plan, old_status, old_terminal, plan)
        require_original_plan(old_plan, control, suite, sources, proof, driver, guard)
        require_owner_records(control, old_launch, launch, plan, status, terminal, source_sha=source_sha,
            suite_lock_sha=control['suite_lock']['sha256'], driver=driver, guard=guard,
            resume_control=control_pointer, resume_lock=lock_pointer, expected_environment=audit_environment)
        require(all(value.get('source_manifest') == suite['source_manifests']['test'] for value in (old_launch, launch))
            and launch.get('batch_sources') == batch_sources,
            'real owner source-manifest and optional901-source declarations must agree with the frozen suite')
        ctl_launch = ledger.json(control['previous_controller_launch']['path'])
        require(not owner.alive(old_launch) and not owner.alive(ctl_launch) and not owner.alive(launch),
                'all real physical stage owners must have exited before final composite results')
        old_ids = set(control['completed_job_ids'])
        require(set(status['completed']) == {row['job_id'] for row in plan['jobs']}, 'complete unique1920 completed keys required')
        if batch_sources:
            from scripts import run_batched_formal_driver_v901_r10 as bootstrap
            entry = control['execution_entry']
            example = next(row for row in plan['jobs'] if not row['imported_completed'])
            repeated = bootstrap.load_control(entry['batch_control']['path'], entry['batch_lock']['path'],
                entry['batch_lock']['sha256'], 'formal', example['physical_argv'][3:])
            require(repeated['sources'] == batch_sources, 'independently repeated901 source/control gate differs')
        profile = fingerprint(proof['routes']['common_author_RML_profile'])
        cases, case_pointers, driver_pointers, batch_pointers, tier_rows = [], [], [], [], []
        seen_processes = set()
        for row in plan['jobs']:
            require_not_stopped()
            name = row['job_id']
            stage_runtime = ROOT/row['evidence_runtime']
            stage_owner = old_launch if row['imported_completed'] else launch
            bindings = {key: row[key] for key in BINDING_KEYS}
            cpu_leaf, native_leaf = stage_runtime/'CPU'/name, stage_runtime/'native'/name
            CPU = ledger.json(cpu_leaf/'CPU_preflight.json')
            require(CPU == ledger.json(cpu_leaf/'stdout.log'), 'CPU stdout must be the actual frozen preflight JSON')
            identity = owner.accept_CPU(CPU, row, source_sha, {} if row['imported_completed'] else batch_sources)
            require(identity.get('task') == row['task'] and identity.get('source_raw_episode_id') == row['raw']
                and identity.get('split') == 'test' and identity.get('demo_only') is False,
                'CPU identity differs from actual original source/task/formal split')
            for leaf, check in ((cpu_leaf, True), (native_leaf, False)):
                actual = ledger.json(leaf/'launch.json')
                require_mixed_launch(actual, row, bindings, check=check, identity=identity, stage_owner=stage_owner, control=control)
                key = actual['launcher_pid'], actual['launcher_start_ticks']
                require(key not in seen_processes, 'each CPU/native request must have a unique actual PID/start')
                seen_processes.add(key)
                require_clean_terminal(ledger.json(leaf/'terminal.json'))
                ledger.pointer(leaf/'stdout.log'); ledger.pointer(leaf/'stderr.log')
                if row['imported_completed']:
                    # These old leaves were frozen before the new owner started.
                    for filename in ('launch.json', 'terminal.json', 'stdout.log', 'stderr.log'):
                        batch.locked_pointer(ROOT, files, ledger.pointer(leaf/filename))
            if row['imported_completed']:
                batch.locked_pointer(ROOT, files, ledger.pointer(cpu_leaf/'CPU_preflight.json'))
            native_result = ledger.json(native_leaf/'terminal.json')
            record = status['completed'][name]
            require_clean_terminal(record)
            require(all(record.get(k) == v for k, v in native_result.items()), 'completed status must preserve the actual native terminal')
            if name in old_ids:
                require(record == old_status['completed'][name], 'historical completed outcome may not be rewritten')
            case_path = ROOT/owner.option(row['physical_argv'], '--out')/'case.json'
            driver_path = case_path.parent/'admission_driver_receipt.json'
            case_pointer = ledger.pointer(case_path)
            driver_pointer = ledger.pointer(driver_path)
            require(native_result.get('case') == case_pointer and native_result.get('admission_driver_receipt') == driver_pointer
                and record.get('case') == case_pointer and record.get('admission_driver_receipt') == driver_pointer,
                'actual native terminal/status must bind its true case and original751 receipt')
            if row['imported_completed']:
                batch.locked_pointer(ROOT, files, case_pointer); batch.locked_pointer(ROOT, files, driver_pointer)
            case = ledger.json(case_path)
            original_audit.require_driver_evidence(ledger.json(driver_path), driver, guard,
                dict(worker=row['case_worker'], consumer=row['native_runtime']), case_path, 'test')
            metrics = require_case_outcome(case, identity, protocol, policy, profile)
            require(record.get('native_success') is metrics['native_success'] and record.get('motion_pass') is metrics['motion_pass'],
                    'status native and diagnostic outcomes differ from original case')
            if row['execution_entry']['kind'] == 'batched901':
                batch_path = case_path.parent/'batch_driver_receipt.json'
                bp = ledger.pointer(batch_path)
                owner.require_batch_execution(ledger.json(batch_path), row, check=False, expected_sources=batch_sources,
                    case=case_pointer, original_receipt=driver_pointer)
                require(native_result.get('batch_driver_receipt') == record.get('batch_driver_receipt') == bp,
                        'new901 native sidecar must be bound independently by actual terminal/status')
                batch_pointers.append(bp)
            else:
                require('batch_driver_receipt' not in native_result and 'batch_driver_receipt' not in record,
                        'original751 evidence cannot invent a901 sidecar')
            cases.append(case); case_pointers.append(case_pointer); driver_pointers.append(driver_pointer)
            tier_rows.append(dict(job_id=name, evidence_runtime=row['evidence_runtime'], imported_completed=row['imported_completed'],
                actual_stage_owner_pid=stage_owner['pid'], actual_stage_owner_start_ticks=stage_owner['start_ticks'],
                previous_row_sha256=row['previous_row_sha256'], execution_entry=row['execution_entry'],
                case=case_pointer, admission_driver_receipt=driver_pointer))
        result = audit_grid(cases, manifest, 'test', proof['routes'], completed_VAL_audit=True)
        require(result.get('complete') is True and result.get('source_cases') == 640
            and result.get('method_cases') == result.get('completed_method_cases') == 1920,
            'independent full16/640source/1920 composite TEST cohort required')
        require(terminal.get('native_successes') == sum(c['official_task_success'] for c in cases),
                'owner terminal native-success count differs from independently audited cases')
        # Original full declared-file gates are retained at both boundaries.
        for frozen in (files, suite_files, source_meta['files']):
            verify_inputs(ROOT, frozen)
        inputs = ledger.close()
    result.update(schema=SCHEMA, created_utc=datetime.now(timezone.utc).isoformat(), per_task=summarize(cases),
        cases=case_pointers, admission_driver_receipts=driver_pointers, batch_driver_receipts=batch_pointers,
        evidence_tiers=tier_rows, inputs_sha256=inputs, resume_control=control_pointer, resume_control_lock=lock_pointer,
        base_result_auditor=owner.pointer(ROOT/'scripts/audit_formal_suite16_results_v754_r10.py'),
        actual_stage_owners=[dict(runtime=owner.PREVIOUS, launch=control['previous_launch'], terminal=control['previous_terminal'],
            pid=old_launch['pid'], start_ticks=old_launch['start_ticks'], completed=len(old_ids),
            resource_limits={key:old_launch[key] for key in ('max_native_slots','max_GPU_slots','max_CPU_preflight_slots')}),
            dict(runtime=owner.RUNTIME, launch=ledger.used[str(runtime/'launch.json')]['pointer'],
            terminal=ledger.used[str(runtime/'terminal.json')]['pointer'], pid=launch['pid'], start_ticks=launch['start_ticks'],
            completed=1920-len(old_ids), resource_profile=control['resource_profile'])],
        source_input_lock=control['source_inputs'], validation_release=control['validation_release'],
        resource_profile=control['resource_profile'], resource_qualification=control['resource_qualification'],
        execution_implementation=control['execution_implementation'], frozen_environment=audit_environment,
        auditor_invocation=dict(argv=list(sys.argv), pid=os.getpid(), start_ticks=owner.ticks(os.getpid()),
            source=control['resume_auditor'], inherited_physical_stage_identity_claimed=False),
        admission_driver=driver, admission_guard=guard, native_runtime=protocol['runtime_consumer'],
        original_native_runtime=protocol['original_runtime'], author_outcome_protocol=protocol,
        execution_protocol_changed=True, source_sha256=inputs[str(ROOT/SOURCE)],
        frozen_lookup_batch_metrics=dict(batch.metrics), used_evidence_rechecked_at_close=True,
        one_fixed_original_checkpoint_not_top3_selected=True, source_outcome_filtering=False,
        native_failures_retained=True, Value_NGFF_FASTER_used=False, automatic_retry=False,
        old753_results_not_rerun=True, independent_composite_audit=True, original754_single_owner_result_claimed=False)
    require_not_stopped()
    atomic_json(out, result)
    print(json.dumps(dict(complete=True, schema=SCHEMA, split='test', cases=1920,
        native_successes_by_method=result['native_successes_by_method'], motion_is_diagnostic_only=True,
        out=OUTPUT)), flush=True)


if __name__ == '__main__':
    main()
