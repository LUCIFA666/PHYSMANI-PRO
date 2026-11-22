"""Pure contracts for a fresh TEST continuation with two historical tiers.

No files, processes or resources are inspected here.  An infrastructure attempt
is never a native task outcome.  Historical rows keep every original provenance
field; this continuation adds one explicitly named lineage layer.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

CONTROL_SCHEMA = 'r10_composite_formal_TEST_resume_control_v955'
LOCK_SCHEMA = 'r10_composite_formal_TEST_resume_inputs_v955'
LINEAGE_SCHEMA = 'r10_composite_TEST_parent_row_lineage_v955'
SOURCE = 'our_mani/multitask/composite_TEST_resume_v955.py'
PREVIOUS = 'runtime/framework/formal_TEST_concurrent_resume_v902'
PREVIOUS_CONTROLLER = 'runtime/framework/ordered_concurrent_TEST_v907'
ORIGINAL = 'runtime/framework/formal_TEST_after_environment_resume_v753'
RUNTIME = 'runtime/framework/formal_TEST_composite_resume_v956'
OUTPUT = 'artifacts/framework/multitask_r10/formal_TEST_composite_resume_v956/test'
SETUP = 'artifacts/framework/multitask_r10/formal_TEST_composite_resume_v956_setup'
PREVIOUS_SETUP = 'artifacts/framework/multitask_r10/formal_TEST_concurrent_resume_v902_setup'
PREVIOUS_CONTROL = PREVIOUS_SETUP + '/resume_control.json'
PREVIOUS_LOCK = PREVIOUS_SETUP + '/resume_inputs.json'
CONTROL = SETUP + '/resume_control.json'
LOCK = SETUP + '/resume_inputs.json'
OWNER = 'scripts/run_composite_TEST_owner_v956_r10.py'
AUDITOR = 'scripts/audit_composite_TEST_results_v957_r10.py'
CONTROLLER = 'scripts/continue_composite_TEST_v958_r10.py'
ORIGINAL_ENTRY = 'scripts/run_formal_case_author_outcome_v751_r10.py'
BATCH_ENTRY = 'scripts/run_batched_formal_driver_v901_r10.py'
METHODS = ('cv_geometry', 'context_geometry', 'physmani')
TASKS = tuple(sorted(name + suffix for name in (
    'beat_the_rotating_buzz', 'insert_onto_rotating_peg', 'moving_basketball_in_hoop',
    'pick_moving_target_on_the_table', 'place_cups_on_rotating_frame',
    'push_moving_button', 'put_rubbish_in_moving_bin', 'remove_cups_from_rotating_frame')
    for suffix in ('', '_high_speed')))
PARENT_FIELDS = frozenset(('evidence_runtime', 'imported_completed', 'previous_row_sha256',
    'execution_entry', 'physical_argv', 'previous_worker_sha256'))


def require(value, message):
    if not value:
        raise ValueError('composite TEST continuation: ' + message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def valid_sha(value):
    return isinstance(value, str) and len(value) == 64 and not set(value) - set('0123456789abcdef')


def valid_pointer(value):
    return (isinstance(value, dict) and set(value) == {'path', 'sha256'}
        and isinstance(value['path'], str) and bool(value['path'])
        and not Path(value['path']).is_absolute() and '..' not in Path(value['path']).parts
        and valid_sha(value['sha256']))


def option(argv, name):
    require(isinstance(argv, list) and argv.count(name) == 1, 'one explicit ' + name + ' required')
    index = argv.index(name)
    require(index + 1 < len(argv), 'missing value for ' + name)
    return argv[index + 1]


def validate_entry(entry):
    require(isinstance(entry, dict) and entry.get('kind') in ('original751', 'batched901'),
        'real original751 or batched901 entry required')
    keys = {'kind', 'pointer'} | ({'batch_control', 'batch_lock'} if entry['kind'] == 'batched901' else set())
    require(set(entry) == keys and all(valid_pointer(entry[key]) for key in keys - {'kind'}),
        'exact immutable entry pointers required')
    require(entry['pointer']['path'] == (BATCH_ENTRY if entry['kind'] == 'batched901' else ORIGINAL_ENTRY),
        'entry path does not match its declared implementation')
    return deepcopy(entry)


def outer_argv(inner, entry):
    entry = validate_entry(entry)
    require(isinstance(inner, list) and len(inner) >= 3 and inner[2] == ORIGINAL_ENTRY,
        'one original751 physical argv required; never double-wrap901')
    if entry['kind'] == 'original751':
        return deepcopy(inner)
    return [*inner[:2], entry['pointer']['path'], '--batch-control', entry['batch_control']['path'],
        '--batch-lock', entry['batch_lock']['path'], '--batch-lock-sha256', entry['batch_lock']['sha256'],
        '--entry', 'formal', '--', *inner[3:]]


def require_parent_row(row):
    require(isinstance(row, dict) and PARENT_FIELDS <= set(row) and 'resume_lineage' not in row,
        'actual unmodified902 parent row required')
    require(type(row['imported_completed']) is bool and valid_sha(row['previous_row_sha256'])
        and valid_sha(row['previous_worker_sha256']), 'original902 provenance must remain explicit')
    entry = validate_entry(row['execution_entry'])
    require(row['evidence_runtime'] == (ORIGINAL if row['imported_completed'] else PREVIOUS),
        'historical753/902 evidence tier changed')
    require(entry['kind'] == ('original751' if row['imported_completed'] else 'batched901'),
        'historical tier must retain its actual entry')
    require(row['argv'] == outer_argv(row['physical_argv'], entry)
        and row['worker_sha256'] == entry['pointer']['sha256'], 'historical outer/physical argv or source changed')
    return row


def require_partition(control, plan, status, terminal):
    require(control.get('schema') == CONTROL_SCHEMA and control.get('split') == 'test'
        and control.get('expected_cases') == 1920 and control.get('source_count') == 640
        and control.get('user_authorized_resume') is True and control.get('standing_authorization_reused') is True
        and control.get('new_user_event_invented') is False and control.get('physical_policy_changed') is False,
        'explicit same-policy authorized full TEST continuation required')
    rows = plan.get('jobs', [])
    require(isinstance(rows, list) and len(rows) == 1920, 'original complete1920 parent plan required')
    by_id, coordinates, groups = {}, set(), {}
    for row in rows:
        require_parent_row(row)
        require(isinstance(row.get('job_id'), str) and row['job_id'] not in by_id,
            'unique original job IDs required')
        task, raw, method = row.get('task'), row.get('raw'), row.get('method')
        require(task in TASKS and method in METHODS and isinstance(raw, str) and raw,
            'exact16 tasks/three methods and original raw IDs required')
        coordinate = task, raw, method
        require(coordinate not in coordinates, 'duplicate task/raw/method coordinate')
        coordinates.add(coordinate); by_id[row['job_id']] = row
        groups.setdefault((task, method), set()).add(raw)
    require(len(groups) == 48 and all(len(groups.get((t, m), ())) == 40 for t in TASKS for m in METHODS)
        and all(groups[t, m] == groups[t, METHODS[0]] for t in TASKS for m in METHODS),
        'same original40 sources per task/method; no padding or filtering')
    parts = {}
    for key in ('completed', 'failed', 'remaining'):
        values = control.get(key + '_job_ids')
        require(isinstance(values, list) and all(isinstance(x, str) for x in values)
            and len(values) == len(set(values)) and control.get(key + '_key_sha256') == digest(sorted(values)),
            'exact unique ' + key + ' keys and digest required')
        parts[key] = set(values)
    done, failed, remaining = (parts[key] for key in ('completed', 'failed', 'remaining'))
    require(done and remaining and not done & remaining and done | remaining == set(by_id)
        and failed <= remaining and not failed & done
        and isinstance(status.get('completed'), dict) and set(status['completed']) == done
        and isinstance(status.get('failed'), dict) and set(status['failed']) == failed,
        'adopt all completed outcomes and retain all failed keys in the remaining partition')
    require(plan.get('split') == status.get('split') == 'test' and plan.get('expected_cases') == 1920
        and plan.get('source_outcome_filtering') is False and plan.get('automatic_retry') is False
        and status.get('expected') == terminal.get('expected') == 1920
        and status.get('active') == status.get('checking') == []
        and status.get('native_slots') == status.get('GPU_slots') == 0
        and status.get('drain_requested') is True and status.get('stop_requested') is False
        and status.get('automatic_retry') is False and terminal.get('complete') is False
        and terminal.get('drained') is True and terminal.get('drain_requested') is True
        and terminal.get('stop_requested') is False and terminal.get('automatic_retry') is False
        and type(terminal.get('failed')) is int and terminal['failed'] == len(failed)
        and type(terminal.get('completed')) is int and terminal['completed'] == len(done),
        'real failed/noninterrupting902 DRAIN with all owned children exited required')
    for record in status['completed'].values():
        require(isinstance(record, dict) and record.get('complete') is True
            and type(record.get('native_success')) is bool and 'motion_pass' in record
            and (record['motion_pass'] is None or type(record['motion_pass']) is bool)
            and type(record.get('returncode')) is int and record['returncode'] == 0
            and record.get('timed_out') is False and record.get('interrupted') is False
            and record.get('automatic_retry') is False, 'only clean outcomes, including nativeFalse/motion unknown, are adopted')
    for record in status['failed'].values():
        require(isinstance(record, dict) and record.get('complete') is not True
            and record.get('automatic_retry') is False, 'failed infrastructure attempt cannot become a completed native outcome')
    require(all(value.get('policy_sha256') == control.get('policy_sha256') for value in (plan, terminal))
        and valid_sha(control.get('policy_sha256')), 'original policy must remain unchanged')
    return by_id


def resumed_jobs(old_plan, control, entry):
    entry = validate_entry(entry)
    require(entry['kind'] == 'batched901', 'unchanged qualified901 entry required for new attempts')
    require(entry == control.get('execution_entry'), 'new entry must match the frozen continuation descriptor')
    done = set(control['completed_job_ids']); result = []
    for parent in old_plan['jobs']:
        require_parent_row(parent)
        row = deepcopy(parent); imported = row['job_id'] in done
        row['resume_lineage'] = dict(schema=LINEAGE_SCHEMA, parent_row_sha256=digest(parent),
            parent_imported_completed=parent['imported_completed'], parent_evidence_runtime=parent['evidence_runtime'],
            parent_runtime=PREVIOUS)
        row['imported_completed'] = imported
        if not imported:
            inner = deepcopy(parent['physical_argv'])
            option(inner, '--out')
            inner[inner.index('--out') + 1] = str(Path(OUTPUT)/row['task']/row['raw']/row['method'])
            row.update(evidence_runtime=RUNTIME, physical_argv=inner, argv=outer_argv(inner, entry),
                worker_sha256=entry['pointer']['sha256'], execution_entry=deepcopy(entry))
        result.append(row)
    return result


def require_resume_plan(control, old_plan, old_status, old_terminal, new_plan):
    by_id = require_partition(control, old_plan, old_status, old_terminal)
    require(new_plan.get('jobs') == resumed_jobs(old_plan, control, control['execution_entry'])
        and new_plan.get('expected_cases') == 1920
        and new_plan.get('imported_completed_cases') == len(control['completed_job_ids'])
        and new_plan.get('source_outcome_filtering') is False,
        'exact new plan with literal parent provenance and one added lineage layer required')
    return by_id


def parent_from_imported(row):
    require(row.get('imported_completed') is True and isinstance(row.get('resume_lineage'), dict),
        'adopted row and explicit parent lineage required')
    parent = deepcopy(row); lineage = parent.pop('resume_lineage')
    require(lineage.get('schema') == LINEAGE_SCHEMA and lineage.get('parent_runtime') == PREVIOUS
        and type(lineage.get('parent_imported_completed')) is bool,
        'actual902 parent lineage required')
    parent['imported_completed'] = lineage['parent_imported_completed']
    require(parent.get('evidence_runtime') == lineage.get('parent_evidence_runtime')
        and digest(parent) == lineage.get('parent_row_sha256'), 'imported row bytes differ from the exact parent row')
    return require_parent_row(parent)


def require_environment_recheck(proof, control):
    """A recorded CPU point observation, never a scientific/resource release."""
    require(isinstance(proof, dict)
        and proof.get('schema') == 'r10_fresh_frozen_TEST_environment_CPU_observation_v954'
        and proof.get('complete') is True and valid_pointer(proof.get('source'))
        and proof['source']['path'] == 'scripts/check_frozen_TEST_environment_v954_r10.py'
        and proof.get('Native_started') is False and proof.get('GPU_started') is False
        and proof.get('admission_or_TEST_started') is False
        and proof.get('declared_model_case_raw_bytes_read') is False
        and proof.get('scientific_inputs_or_old_locks_modified') is False
        and proof.get('original_v900_topology_constructor_passed') is True
        and proof.get('original_full_topology_final_check_passed') is True
        and proof.get('parsed_suite_lock_is_exact_hashed_payload') is True
        and proof.get('topology_is_point_observation_not_continuous') is True
        and proof.get('suite_lock') == control.get('suite_lock')
        and type(proof.get('declared_files')) is int and proof['declared_files'] > 0
        and proof.get('unique_canonical_paths') == proof['declared_files']
        and type(proof.get('watched_directories')) is int and proof['watched_directories'] > 0,
        'genuine same-byte parsed954 topology/OS point observation required')
    actual = proof.get('actual_environment', {})
    require(actual.get('schema') == 'r10_actual_frozen_OS_namespace_receipt_v820'
        and actual.get('complete') is True and actual.get('checked_OS_files') == 6418
        and actual.get('all_current_OS_targets_full_byte_SHA_verified') is True
        and actual.get('manifest') == control.get('environment_manifest')
        and actual.get('wrapper') == control.get('environment_wrapper')
        and actual.get('namespace_qualification') == control.get('environment_namespace_proof'),
        '954 must observe the same original820 environment; a fresh owner captures its own namespace')
    return proof
