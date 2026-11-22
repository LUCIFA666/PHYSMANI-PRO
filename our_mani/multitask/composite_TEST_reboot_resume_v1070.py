"""Pure contracts for a fresh TEST continuation with six historical tiers after a real host reboot.

No files, processes or resources are inspected here.  An infrastructure attempt
is never a native task outcome.  Historical rows keep every original provenance
field; this continuation adds one explicitly named lineage layer.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

CONTROL_SCHEMA = 'r10_seven_tier_reboot_formal_TEST_resume_control_v1070'
LOCK_SCHEMA = 'r10_seven_tier_reboot_formal_TEST_resume_inputs_v1070'
LINEAGE_SCHEMA = 'r10_seven_tier_TEST_parent_row_lineage_v1070'
LINEAGE_FIELD = 'resume_lineage_v1070'
SOURCE = 'our_mani/multitask/composite_TEST_reboot_resume_v1070.py'
PREVIOUS = 'runtime/framework/formal_TEST_reboot_resume_v1056'
FIFTH = 'runtime/framework/formal_TEST_composite_resume_v1041'
PARENT_OLDER = 'runtime/framework/formal_TEST_composite_resume_v1011'
OLDER = 'runtime/framework/formal_TEST_composite_resume_v956'
INTERMEDIATE = 'runtime/framework/formal_TEST_concurrent_resume_v902'
PREVIOUS_CONTROLLER = 'runtime/framework/ordered_reboot_TEST_v1058'
ORIGINAL = 'runtime/framework/formal_TEST_after_environment_resume_v753'
RUNTIME = 'runtime/framework/formal_TEST_reboot_resume_v1071'
OUTPUT = 'artifacts/framework/multitask_r10/formal_TEST_reboot_resume_v1071/test'
SETUP = 'artifacts/framework/multitask_r10/formal_TEST_reboot_resume_v1071_setup'
PREVIOUS_SETUP = 'artifacts/framework/multitask_r10/formal_TEST_reboot_resume_v1056_setup'
PREVIOUS_CONTROL = PREVIOUS_SETUP + '/resume_control.json'
PREVIOUS_LOCK = PREVIOUS_SETUP + '/resume_inputs.json'
CONTROL = SETUP + '/resume_control.json'
LOCK = SETUP + '/resume_inputs.json'
OWNER = 'scripts/run_second_reboot_TEST_owner_v1071_r10.py'
AUDITOR = 'scripts/audit_second_reboot_TEST_results_v1072_r10.py'
CONTROLLER = 'scripts/continue_second_reboot_TEST_v1073_r10.py'
PREVIOUS_OWNER = 'scripts/run_reboot_TEST_owner_v1056_r10.py'
PREVIOUS_CONTROLLER_SOURCE = 'scripts/continue_reboot_TEST_v1058_r10.py'
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
    require(isinstance(row, dict) and PARENT_FIELDS <= set(row)
        and isinstance(row.get('resume_lineage'), dict)
        and isinstance(row.get('resume_lineage_v1010'), dict)
        and isinstance(row.get('resume_lineage_v1040'), dict)
        and isinstance(row.get('resume_lineage_v1055'), dict) and LINEAGE_FIELD not in row,
        'actual unmodified1056 parent row with all four historical lineages required')
    require(type(row['imported_completed']) is bool and valid_sha(row['previous_row_sha256'])
        and valid_sha(row['previous_worker_sha256']), 'all original902 provenance must remain explicit')
    first, second = row['resume_lineage'], row['resume_lineage_v1010']
    require(first.get('schema') == 'r10_composite_TEST_parent_row_lineage_v955'
        and first.get('parent_runtime') == INTERMEDIATE and valid_sha(first.get('parent_row_sha256'))
        and type(first.get('parent_imported_completed')) is bool
        and first.get('parent_evidence_runtime') in (ORIGINAL, INTERMEDIATE),
        'unchanged original955 lineage must remain nested, never overwritten')
    require(second.get('schema') == 'r10_four_tier_TEST_parent_row_lineage_v1010'
        and second.get('parent_runtime') == OLDER and valid_sha(second.get('parent_row_sha256'))
        and type(second.get('parent_imported_completed')) is bool
        and second.get('parent_evidence_runtime') in (ORIGINAL, INTERMEDIATE, OLDER),
        'unchanged original1010 whole-row lineage must remain nested, never overwritten')
    third = row['resume_lineage_v1040']
    require(third.get('schema') == 'r10_five_tier_TEST_parent_row_lineage_v1040'
        and third.get('parent_runtime') == PARENT_OLDER and valid_sha(third.get('parent_row_sha256'))
        and type(third.get('parent_imported_completed')) is bool
        and third.get('parent_evidence_runtime') in (ORIGINAL, INTERMEDIATE, OLDER, PARENT_OLDER),
        'unchanged original1040 whole-row lineage must remain nested')
    fourth = row['resume_lineage_v1055']
    require(fourth.get('schema') == 'r10_six_tier_TEST_parent_row_lineage_v1055'
        and fourth.get('parent_runtime') == FIFTH and valid_sha(fourth.get('parent_row_sha256'))
        and type(fourth.get('parent_imported_completed')) is bool
        and fourth.get('parent_evidence_runtime') in (ORIGINAL, INTERMEDIATE, OLDER, PARENT_OLDER, FIFTH),
        'unchanged original1055 whole-row lineage must remain nested')
    entry = validate_entry(row['execution_entry'])
    require(row['evidence_runtime'] in ((ORIGINAL, INTERMEDIATE, OLDER, PARENT_OLDER, FIFTH) if row['imported_completed'] else (PREVIOUS,)),
        'historical753/902/956/1011/1041 evidence tier changed')
    require(entry['kind'] == ('original751' if row['evidence_runtime'] == ORIGINAL else 'batched901'),
        'historical tier must retain its actual entry')
    require(row['argv'] == outer_argv(row['physical_argv'], entry)
        and row['worker_sha256'] == entry['pointer']['sha256'], 'historical outer/physical argv or source changed')
    return row


def audit_partition(audit):
    """Pure exhaustive recovery roster; unavailable outcomes remain unknown."""
    require(isinstance(audit, dict)
        and audit.get('schema') == 'r10_readonly_host_second_reboot_TEST_evidence_observation_v1067'
        and audit.get('audit_complete') is True and audit.get('formal_full_TEST_complete') is False
        and audit.get('inputs_unchanged_through_audit') is True
        and audit.get('old_outputs_modified') is False and audit.get('missing_evidence_reconstructed') is False
        and audit.get('unknown_counted_as_NativeFalse') is False
        and audit.get('success_based_case_selection') is False
        and audit.get('new_terminal_claimed_for_interrupted_owner') is False
        and audit.get('Native_started') is False and audit.get('scientific_inputs_read') is False
        and audit.get('raw_data_read') is False and audit.get('model_read') is False
        and audit.get('full_source_graph_read') is False
        and audit.get('old_owner_process', {}).get('same_process_alive') is False,
        'genuine read-only1067 second reboot recovery audit required')
    records = audit.get('completed_evidence')
    require(isinstance(records, list) and records and all(isinstance(r, dict) for r in records),
        'exhaustive saved completion evidence required')
    indexed = {r.get('job_id'): r for r in records}
    require(len(indexed) == len(records) and all(isinstance(k, str) and k for k in indexed),
        'one evidence classification per saved completion')
    accepted, rejected = set(), set()
    for key, record in indexed.items():
        require(type(record.get('recoverable_complete')) is bool, 'explicit evidence completeness required')
        if record['recoverable_complete']:
            require(type(record.get('native_success')) is bool and record.get('errors') == [],
                'both clean NativeFalse and NativeTrue outcomes must be retained')
            accepted.add(key)
        else:
            require(record.get('native_success') is None and isinstance(record.get('errors'), list)
                and record['errors'], 'damaged evidence is unknown, never NativeFalse')
            rejected.add(key)
    sets = dict(accepted_completed_job_ids=accepted, rejected_saved_completed_job_ids=rejected,
        saved_completed_job_ids=set(indexed))
    for name, values in sets.items():
        require(audit.get(name) == sorted(values) and audit.get(name + '_sha256') == digest(sorted(values)),
            'audit recovery roster/digest changed: ' + name)
    require(audit.get('saved_status_completed_count') == len(indexed)
        and audit.get('recoverable_complete_count') == len(accepted)
        and audit.get('damaged_registered_count') == len(rejected)
        and audit.get('unregistered_clean_native_terminals') == [],
        'exact exhaustive saved count and no unregistered clean outcomes required')
    boot_id = audit.get('current_boot_id')
    require(isinstance(boot_id, str) and len(boot_id) == 36, 'actual current boot identity required')
    return accepted, rejected, set(indexed)


def require_partition(control, plan, status, reboot_audit):
    require(control.get('schema') == CONTROL_SCHEMA and control.get('split') == 'test'
        and control.get('expected_cases') == 1920 and control.get('source_count') == 640
        and control.get('user_authorized_resume') is True and control.get('standing_authorization_reused') is True
        and control.get('new_user_event_invented') is False and control.get('physical_policy_changed') is False
        and control.get('explicit_host_reboot_recovery') is True,
        'explicit same-policy authorized full TEST reboot continuation required')
    rows = plan.get('jobs', [])
    require(isinstance(rows, list) and len(rows) == 1920, 'original full1920 six-tier parent plan required')
    by_id, coordinates, groups = {}, set(), {}
    for row in rows:
        require_parent_row(row)
        require(isinstance(row.get('job_id'), str) and row['job_id'] not in by_id, 'unique original job IDs required')
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
    accepted, rejected, saved = audit_partition(reboot_audit)
    require(isinstance(status.get('completed'), dict) and set(status['completed']) == saved
        and isinstance(status.get('failed'), dict), 'literal saved1056 state must remain unchanged')
    remaining, failed = set(by_id) - accepted, set(status['failed'])
    require(accepted and remaining and not saved - set(by_id) and not failed & saved
        and failed <= remaining, 'all complete and damaged outcomes need an exhaustive partition')
    parts = dict(completed=accepted, failed=failed, remaining=remaining,
        saved_completed=saved, rejected_saved_completed=rejected)
    for name, values in parts.items():
        require(control.get(name + '_job_ids') == sorted(values)
            and control.get(name + '_key_sha256') == digest(sorted(values)),
            'exact immutable reboot partition required: ' + name)
    require(reboot_audit.get('remaining_job_ids') == sorted(remaining)
        and reboot_audit.get('remaining_job_ids_sha256') == digest(sorted(remaining))
        and reboot_audit.get('still_unclosed_count') == len(remaining), 'all remaining rows must be retried explicitly')
    require(plan.get('split') == status.get('split') == 'test' and plan.get('expected_cases') == status.get('expected') == 1920
        and plan.get('source_outcome_filtering') is False and plan.get('automatic_retry') is False
        and status.get('stop_requested') is False and status.get('automatic_retry') is False,
        'literal interrupted status and unchanged full TEST scope required')
    evidence = {r['job_id']: r for r in reboot_audit['completed_evidence']}
    for key in accepted:
        record = status['completed'][key]
        require(evidence[key]['native_success'] is record.get('native_success'), 'audited Native outcome must remain literal')
        require(isinstance(record, dict) and record.get('complete') is True
            and type(record.get('native_success')) is bool and 'motion_pass' in record
            and (record['motion_pass'] is None or type(record['motion_pass']) is bool)
            and type(record.get('returncode')) is int and record['returncode'] == 0
            and record.get('timed_out') is False and record.get('interrupted') is False
            and record.get('automatic_retry') is False, 'only fully evidenced clean outcomes are adopted')
    require(plan.get('policy_sha256') == control.get('policy_sha256') == status.get('policy_sha256')
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
        row[LINEAGE_FIELD] = dict(schema=LINEAGE_SCHEMA, parent_row_sha256=digest(parent),
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


def require_resume_plan(control, old_plan, old_status, reboot_audit, new_plan):
    by_id = require_partition(control, old_plan, old_status, reboot_audit)
    require(new_plan.get('jobs') == resumed_jobs(old_plan, control, control['execution_entry'])
        and new_plan.get('expected_cases') == 1920
        and new_plan.get('imported_completed_cases') == len(control['completed_job_ids'])
        and new_plan.get('source_outcome_filtering') is False,
        'exact new plan with literal parent provenance and one added lineage layer required')
    return by_id


def parent_from_imported(row):
    require(row.get('imported_completed') is True and isinstance(row.get(LINEAGE_FIELD), dict),
        'adopted row and explicit parent lineage required')
    parent = deepcopy(row); lineage = parent.pop(LINEAGE_FIELD)
    require(lineage.get('schema') == LINEAGE_SCHEMA and lineage.get('parent_runtime') == PREVIOUS
        and type(lineage.get('parent_imported_completed')) is bool,
        'actual1056 complete parent lineage required')
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
