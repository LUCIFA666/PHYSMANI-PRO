"""Inactive preparation of one new, same-policy, three-tier TEST continuation.

Only --prepare writes anything. Old915/901 controls, completed nativeFalse
outcomes and all failed infrastructure attempts keep their original bytes.
No worker or native task is started by this preparer.
"""
from contextlib import ExitStack, contextmanager
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import argparse
import fcntl
import json
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from our_mani.multitask.evaluation import atomic_json, digest, file_sha
from our_mani.multitask.input_stat_cache_v79 import verify_inputs
from our_mani.multitask.frozen_input_edges_v463 import FrozenInputIndex
from our_mani.multitask.formal_suite_admission_v728 import read_formal_source
from our_mani.multitask.common_rml_protocol_v530 import fingerprint
from our_mani.multitask.rml_case_preflight_v742 import outcome_protocol_binding
from our_mani.multitask import composite_TEST_resume_v955 as core
from scripts import run_composite_TEST_owner_v956_r10 as owner
from scripts import prepare_concurrent_TEST_resume_v915_r10 as base
from scripts import audit_concurrent_resumed_formal_TEST_v906_r10 as old_audit
from scripts import run_batched_formal_driver_v901_r10 as bootstrap

SOURCE = 'scripts/prepare_composite_TEST_resume_v955_r10.py'
SETUP, CONTROL, LOCK = core.SETUP, core.CONTROL, core.LOCK
PREVIOUS, PREVIOUS_CONTROLLER = core.PREVIOUS, core.PREVIOUS_CONTROLLER
SUITE, SUITE_LOCK, SOURCE_INPUTS, VAL_RELEASE = base.SUITE, base.SUITE_LOCK, base.SOURCE_INPUTS, base.VAL_RELEASE
PREVIOUS_CONTROL, PREVIOUS_LOCK = base.CONTROL, base.LOCK
ENVIRONMENT_RECHECK = 'runtime/framework/frozen_TEST_environment_recheck_v954.json'
RECHECK_SOURCE = 'scripts/check_frozen_TEST_environment_v954_r10.py'
PROFILE = deepcopy(base.PROFILE)
TESTS = ('tests/test_composite_TEST_resume_v955.py', 'tests/test_prepare_composite_TEST_resume_v955.py',
    'tests/test_composite_TEST_owner_v956.py', 'tests/test_composite_TEST_results_v957.py',
    'tests/test_composite_TEST_controller_v958.py')
REQUIRED_SOURCES = tuple(sorted({SOURCE, core.SOURCE, core.OWNER, core.AUDITOR, core.CONTROLLER,
    RECHECK_SOURCE, *TESTS, *base.REQUIRED_SOURCES}))
STOP_PATHS = tuple(dict.fromkeys((*base.STOP_PATHS, core.RUNTIME+'/STOP',
    'runtime/framework/ordered_composite_TEST_v958/STOP')))
merge_maps = base.merge_maps
add_frozen_file = base.add_frozen_file
require_canonical_consistency = base.require_canonical_consistency
require_resource_proof = base.require_resource_proof


def require(value, message):
    if not value:
        raise ValueError('composite TEST preparation: '+message)


def stopped(root=ROOT):
    return any((Path(root)/name).exists() for name in STOP_PATHS)


def partition(plan, status, terminal):
    done = status.get('completed'); failed = status.get('failed')
    require(isinstance(done, dict) and isinstance(failed, dict), 'real902 completed and infrastructure maps required')
    completed, failures = sorted(done), sorted(failed)
    remaining = sorted({row['job_id'] for row in plan.get('jobs', [])}-set(completed))
    control = dict(schema=core.CONTROL_SCHEMA, split='test', expected_cases=1920, source_count=640,
        completed_job_ids=completed, failed_job_ids=failures, remaining_job_ids=remaining,
        completed_key_sha256=digest(completed), failed_key_sha256=digest(failures),
        remaining_key_sha256=digest(remaining), policy_sha256=plan.get('policy_sha256'),
        user_authorized_resume=True, standing_authorization_reused=True,
        new_user_event_invented=False, physical_policy_changed=False)
    jobs = core.require_partition(control, plan, status, terminal)
    return dict(jobs=jobs, completed=completed, failed=failures, remaining=remaining,
        policy_sha256=plan['policy_sha256'], partition_control=control)


def old_evidence_paths(row):
    """Real recorded tier, including the additional native901 sidecar."""
    core.require_parent_row(row)
    key = row['job_id']; runtime = Path(row['evidence_runtime'])
    names = [str(runtime/'CPU'/key/name) for name in
        ('launch.json', 'stdout.log', 'stderr.log', 'terminal.json', 'CPU_preflight.json')]
    names += [str(runtime/'native'/key/name) for name in ('launch.json', 'stdout.log', 'stderr.log', 'terminal.json')]
    out = Path(core.option(row['physical_argv'], '--out'))
    names += [str(out/'case.json'), str(out/'admission_driver_receipt.json')]
    if row['execution_entry']['kind'] == 'batched901':
        names.append(str(out/'batch_driver_receipt.json'))
    return names


def require_previous_exit(previous, is_alive):
    for key in ('launch', 'controller_launch'):
        value = previous[key]
        require(type(value.get('pid')) is int and value['pid'] > 0
            and type(value.get('start_ticks')) is int and value['start_ticks'] > 0
            and not is_alive(value), 'actual902 and907 PID/start must have exited')
    state, failure = previous['controller_status'], previous['controller_failure']
    require(state.get('stage') == 'requires_concrete_repair' and state.get('automatic_retry') is False
        and failure.get('complete') is False and failure.get('user_stop') is False
        and failure.get('automatic_retry') is False, 'real907 failure/status required; no successful terminal is invented')
    part = partition(previous['plan'], previous['status'], previous['terminal'])
    return dict(previous_owner_and_controller_exited=True,
        identities=[dict(pid=previous[k]['pid'], start_ticks=previous[k]['start_ticks'])
            for k in ('launch', 'controller_launch')], completed=len(part['completed']),
        infrastructure_failed=len(part['failed']), remaining=len(part['remaining']))


@contextmanager
def idle_resources(env, previous, root=ROOT):
    require(not stopped(root), 'STOP before resource checks')
    require_previous_exit(previous, owner.alive)
    with ExitStack() as stack:
        for name in ('runtime/framework/ordered_concurrent_TEST_v907.lock', *base.canary.LOCKS):
            stream = stack.enter_context((Path(root)/name).open('a+'))
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        require_previous_exit(previous, owner.alive)
        inventory = owner.process_visibility()
        require(inventory.get('host_visibility_available') is True
            and inventory.get('same_uid_inventory_incomplete') is False
            and inventory.get('live_project_native_processes') == []
            and inventory.get('ancestor_chain_complete') is True,
            'complete empty host-visible native/controller inventory required')
        actual = owner.environment_owner.require_frozen_environment(env)
        require(not stopped(root), 'STOP after resource checks')
        yield dict(inventory=inventory, frozen_environment=actual)


def failed_attempt_inventory(root, row, ledger, failure=None):
    """Bounded to one known failed output and its two recorded log leaves.

    There is no input/model/data discovery. Absence is explicit; every existing
    partial file stays in its old directory and is frozen without modification.
    """
    root = Path(root).resolve(); key = row['job_id']; records = []
    if failure is not None:
        stage = failure.get('stage')
        require(stage in ('CPU_preflight', 'CPU_receipt', 'native', 'native_case_contract'),
            'explicit failed infrastructure phase required')
        failed_leaf = root/PREVIOUS/('CPU' if stage.startswith('CPU') else 'native')/key
        require(all((failed_leaf/name).is_file() for name in ('launch.json', 'terminal.json', 'stdout.log', 'stderr.log')),
            'actual failed phase launch, terminal and two logs must be retained')
    for kind in ('CPU', 'native'):
        leaf = root/PREVIOUS/kind/key
        for name in ('launch.json', 'terminal.json', 'stdout.log', 'stderr.log', 'CPU_preflight.json'):
            path = leaf/name
            if path.exists():
                records.append(ledger.read(path))
    out = root/core.option(row['physical_argv'], '--out')
    require(out.is_relative_to(root/'artifacts/framework/multitask_r10') and not out.is_symlink(),
        'failed partial output must stay in its original project attempt')
    directories, partial = [], []
    exists = out.exists()
    if exists:
        require(out.is_dir() and out.resolve(strict=True) == out, 'canonical failed partial directory required')
        for current, dirs, files in os.walk(out, followlinks=False):
            current = Path(current)
            dirs.sort()
            require(current.resolve(strict=True) == current, 'failed partial directory alias refused')
            directories.append(str(current.relative_to(root)))
            for name in sorted([*dirs, *files]):
                path = current/name
                require(not path.is_symlink(), 'failed partial symlink refused')
            for name in sorted(files):
                path = current/name
                require(path.is_file(), 'failed partial special file refused')
                partial.append(ledger.read(path))
    return dict(job_id=key, infrastructure_attempt_not_native_outcome=True,
        output_path=str(out.relative_to(root)), output_existed=exists,
        directories=sorted(directories), log_files=records, partial_files=partial)


def freeze_completed(row, record, ledger, control, stage_owner, sources, profile):
    from scripts import audit_composite_TEST_results_v957_r10 as audit
    for name in old_evidence_paths(row):
        ledger.read(name)
    key = row['job_id']; runtime = Path(row['evidence_runtime'])
    cpu_leaf, native_leaf = (runtime/kind/key for kind in ('CPU', 'native'))
    cpu = ledger.json(cpu_leaf/'CPU_preflight.json')
    require(cpu == ledger.json(cpu_leaf/'stdout.log'), 'actual CPU stdout/preflight bytes describe different payloads')
    batch_sources = sources if row['execution_entry']['kind'] == 'batched901' else {}
    identity = owner.accept_CPU(cpu, row, control['source_inputs']['sha256'], batch_sources)
    bindings = {name:row[name] for name in owner.BINDING_KEYS}
    for leaf, check in ((cpu_leaf, True), (native_leaf, False)):
        launch = ledger.json(leaf/'launch.json')
        audit.require_tier_launch(launch, row, bindings, check=check, identity=identity,
            stage_owner=stage_owner, control=control)
        audit.require_clean_terminal(ledger.json(leaf/'terminal.json'))
        require(not owner.alive(dict(pid=launch['launcher_pid'], start_ticks=launch['launcher_start_ticks'])),
            'every imported physical child PID/start must have exited')
    native = ledger.json(native_leaf/'terminal.json'); audit.require_clean_terminal(record)
    require(all(record.get(name) == value for name,value in native.items()), 'status must retain its actual native terminal')
    out = Path(core.option(row['physical_argv'], '--out'))
    case = ledger.json(out/'case.json'); driver = ledger.json(out/'admission_driver_receipt.json')
    cp, dp = ledger.pointer(out/'case.json'), ledger.pointer(out/'admission_driver_receipt.json')
    require(native.get('case') == record.get('case') == cp
        and native.get('admission_driver_receipt') == record.get('admission_driver_receipt') == dp,
        'actual case/751 receipt pointers and status SHA must agree')
    audit.original_audit.require_driver_evidence(driver, row['admission_driver'], row['admission_guard'],
        dict(worker=row['case_worker'], consumer=row['native_runtime']), ROOT/out/'case.json', 'test')
    metrics = audit.require_case_outcome(case, identity, control['author_outcome_protocol'], control['policy_sha256'], profile)
    require(record['native_success'] is metrics['native_success'] and record['motion_pass'] is metrics['motion_pass'],
        'nativeFalse and unknown motion outcomes must remain literal')
    if row['execution_entry']['kind'] == 'batched901':
        bp = ledger.pointer(out/'batch_driver_receipt.json')
        owner.require_batch_execution(ledger.json(out/'batch_driver_receipt.json'), row, check=False,
            expected_sources=sources, case=cp, original_receipt=dp)
        require(native.get('batch_driver_receipt') == record.get('batch_driver_receipt') == bp,
            'old901 actual native sidecar must retain its exact pointer')
    return metrics


def prepare(*, environment_recheck_sha256, root=ROOT):
    root = Path(root).resolve()
    require(core.valid_sha(environment_recheck_sha256), 'external actual954 proof SHA required')
    require(root == ROOT and not stopped(root), 'actual project and continuing authorization required')
    require(not any((root/name).exists() for name in (SETUP, core.RUNTIME, core.OUTPUT,
        'runtime/framework/ordered_composite_TEST_v958')), 'all new one-time namespaces must be fresh')
    for name in REQUIRED_SOURCES:
        require((root/name).is_file(), 'new source/test is not ready: '+name)
    ledger = old_audit.EvidenceLedger(root)
    pointer, read = ledger.pointer, ledger.json
    release_pointer = pointer(VAL_RELEASE)
    require(release_pointer['sha256'] == base.VAL_RELEASE_SHA, 'exact real complete824 VAL release required first')
    release = read(VAL_RELEASE)
    owner.require_complete_VAL_release(dict(validation_release=release_pointer,
        policy_sha256=release.get('policy_sha256'), author_outcome_protocol=release.get('author_outcome_protocol')), VAL_RELEASE)
    prior_pointer, prior_lock_pointer = pointer(PREVIOUS_CONTROL), pointer(PREVIOUS_LOCK)
    prior, prior_lock = read(PREVIOUS_CONTROL), read(PREVIOUS_LOCK)
    require(prior.get('schema') == old_audit.owner.CONTROL_SCHEMA
        and prior_lock.get('schema') == old_audit.owner.LOCK_SCHEMA
        and prior_lock.get('approved_for_execution') is True, 'actual original915 control/lock required')
    for name in ('suite', 'suite_lock', 'source_inputs', 'validation_release'):
        require(prior.get(name) == pointer(dict(suite=SUITE, suite_lock=SUITE_LOCK,
            source_inputs=SOURCE_INPUTS, validation_release=VAL_RELEASE)[name]), 'original915 source scope changed: '+name)
    original = {name:ledger.json(prior['previous_'+name]['path'], expected=prior['previous_'+name]['sha256'])
        for name in ('launch', 'plan', 'status', 'terminal')}
    previous = {name:read(PREVIOUS+'/'+name+'.json') for name in ('launch', 'plan', 'status', 'terminal')}
    previous.update({name:read(PREVIOUS_CONTROLLER+'/'+name.removeprefix('controller_')+'.json')
        for name in ('controller_launch', 'controller_status', 'controller_failure')})
    previous_proof = require_previous_exit(previous, owner.alive)
    part = partition(previous['plan'], previous['status'], previous['terminal'])
    old_audit.require_resume_plan(prior, original['plan'], original['status'], original['terminal'], previous['plan'])
    require(all(previous['status']['completed'].get(key) == value for key,value in original['status']['completed'].items()),
        'all original753 outcomes must still exist unchanged in902')
    env = {key:pointer(name) for key,name in base.ENVIRONMENT_PATHS.items()}
    require(all(prior.get(key) == value for key,value in env.items()), 'original820 material pointers unchanged')
    seed = dict(prior, **part['partition_control'], previous_runtime=PREVIOUS,
        previous_controller_runtime=PREVIOUS_CONTROLLER, suite=pointer(SUITE), suite_lock=pointer(SUITE_LOCK),
        source_inputs=pointer(SOURCE_INPUTS), validation_release=release_pointer, **env)
    recheck = ledger.json(ENVIRONMENT_RECHECK, expected=environment_recheck_sha256)
    core.require_environment_recheck(recheck, seed)
    ledger.read(recheck['source']['path'], expected=recheck['source']['sha256'])
    for value, source in ((previous['launch'], 'scripts/run_concurrent_resumed_formal_TEST_owner_v902_r10.py'),
            (previous['controller_launch'], 'scripts/continue_concurrent_TEST_v907_r10.py')):
        require(value.get('source_sha256') == pointer(source)['sha256']
            and value.get('resume_control') == prior_pointer and value.get('resume_control_lock') == prior_lock_pointer,
            'historical902/907 source/control bytes differ')
    with idle_resources(env, previous):
        pass
    verify_inputs(root, prior_lock['files'])
    suite, suite_lock, source_meta = read(SUITE), read(SUITE_LOCK), read(SOURCE_INPUTS)
    suite_files, source_files = suite_lock['files'], source_meta['files']
    verify_inputs(root, suite_files)
    manifest, sources, proof = read_formal_source(root, suite, suite_files, 'test', validation_release=release)
    require(proof['policy_sha256'] == seed['policy_sha256']
        and suite['author_outcome_protocol'] == seed['author_outcome_protocol'] == outcome_protocol_binding(root),
        'same original physical policy and author outcome protocol required')
    require(source_meta.get('schema') == 'r10_full16_post_admission_split_source_lock_v1'
        and source_meta.get('split') == 'test' and source_meta.get('source_count') == len(sources) == 640
        and source_meta.get('policy_sha256') == seed['policy_sha256']
        and source_meta.get('TEST_used_for_fitting') is False
        and source_meta.get('source_manifest_sha256') == suite['source_manifests']['test']['sha256'],
        'unchanged original826 verified640 source map required')
    verify_inputs(root, source_files)
    old_audit.require_original_plan(original['plan'], seed, suite, sources, proof,
        suite['formal_case_driver'], suite['formal_admission_guard'])
    shared = {name:pointer(name) for name in bootstrap.SOURCES}
    require(seed['execution_entry'] == prior['execution_entry'] and seed['resource_profile'] == PROFILE
        and seed['execution_implementation'] == owner.execution_implementation(seed['execution_entry'], shared),
        'same qualified4/2/2/8 implementation and old915 formal controls required')
    batch_control = read(seed['execution_entry']['batch_control']['path'])
    batch_lock = read(seed['execution_entry']['batch_lock']['path'])
    require(pointer(seed['execution_entry']['batch_control']['path']) == seed['execution_entry']['batch_control']
        and pointer(seed['execution_entry']['batch_lock']['path']) == seed['execution_entry']['batch_lock']
        and batch_control == base.formal_bootstrap_control(suite)
        and batch_lock.get('schema') == bootstrap.LOCK_SCHEMA and batch_lock.get('execution_enabled') is True
        and all(batch_lock['files'].get(str(root/name), batch_lock['files'].get(name)) == p['sha256']
            for name,p in shared.items()), 'original915 batch control/lock and exact seven implementation sources must remain unchanged')
    qualification = ledger.json(seed['resource_qualification']['path'], expected=seed['resource_qualification']['sha256'])
    require(owner.valid_pointer(qualification.get('actual_independent_audit'))
        and qualification['actual_independent_audit']['path'] == base.QUALIFICATION_LAUNCH,
        'exact actual independent904 launch pointer required before opening proof dependencies')
    audit_launch = ledger.json(qualification['actual_independent_audit']['path'],
        expected=qualification['actual_independent_audit']['sha256'])
    require_resource_proof(qualification, seed, audit_launch=audit_launch)
    for name,checksum in qualification['inputs'].items():
        require(base.qualification_path_allowed(name), 'noncanary qualification payload refused')
        ledger.read(name, expected=checksum)
    for key in ('control', 'lock'):
        ledger.read(qualification[key]['path'], expected=qualification[key]['sha256'])
    files = merge_maps(prior_lock['files'], suite_files, source_files)
    profile = fingerprint(proof['routes']['common_author_RML_profile']); successes = 0
    stage_owners = {core.ORIGINAL:original['launch'], PREVIOUS:previous['launch']}
    for key in part['completed']:
        require(not stopped(root), 'STOP during historical evidence freeze')
        row = part['jobs'][key]
        metrics = freeze_completed(row, previous['status']['completed'][key], ledger, seed,
            stage_owners[row['evidence_runtime']], shared, profile)
        successes += int(metrics['native_success'])
    require(previous['terminal'].get('native_successes') == successes, 'historical native total differs')
    failed_inventory = {key:failed_attempt_inventory(root, part['jobs'][key], ledger, previous['status']['failed'][key])
        for key in part['failed']}
    standing = read(base.STANDING); authorization = read(base.AUTHORIZATION)
    require(standing.get('continue_until_user_stop') is True and standing.get('user_stop_received') is False
        and authorization.get('user_authorized_concurrency_change') is True, 'real continuing authorization required')
    for name in (*REQUIRED_SOURCES, *base.ENVIRONMENT_EXTRA, PREVIOUS+'/DRAIN'):
        ledger.read(name)
    for name,value in ledger.used.items():
        add_frozen_file(root, files, name, value['pointer']['sha256'])
    require(all(files.get(k) == v for k,v in prior_lock['files'].items()), 'every original915 key/SHA must remain literal')
    verify_inputs(root, files)
    graph_files = dict(files); graph = FrozenInputIndex(root, graph_files)
    require_canonical_consistency(graph, graph_files)
    ledger.close()
    # The directory remains untouched until every conflict, proof and old case passes.
    with idle_resources(env, previous) as actual:
        require(not any((root/name).exists() for name in (SETUP, core.RUNTIME, core.OUTPUT)), 'fresh namespaces required')
        require(graph.valid() and not stopped(root), 'STOP or changed topology before committing setup')
        # Re-read the bounded failed-output inventory without adding evidence.
        class ClosedLedger:
            def read(self, path):
                absolute = str(root/path)
                require(absolute in ledger.used and file_sha(absolute) == ledger.used[absolute]['pointer']['sha256'],
                    'failed partial evidence changed before commit')
                return deepcopy(ledger.used[absolute]['pointer'])
        require({key:failed_attempt_inventory(root, part['jobs'][key], ClosedLedger(), previous['status']['failed'][key])
            for key in part['failed']} == failed_inventory,
            'failed attempt directory inventory changed before commit')
        def frozen_pointer(name):
            value = ledger.used.get(str(root/name))
            require(value is not None, 'unobserved pointer after evidence close')
            return deepcopy(value['pointer'])
        control = dict(seed, created_utc=datetime.now(timezone.utc).isoformat(),
            resume_owner=frozen_pointer(core.OWNER), resume_auditor=frozen_pointer(core.AUDITOR),
            result_auditor=frozen_pointer(core.AUDITOR), resume_controller=frozen_pointer(core.CONTROLLER),
            previous_resume_control=prior_pointer, previous_resume_control_lock=prior_lock_pointer,
            environment_recheck=frozen_pointer(ENVIRONMENT_RECHECK),
            failed_attempt_evidence=failed_inventory, original_input_maps_preserved=True,
            prepared_frozen_environment=actual['frozen_environment'], preparation_inventory=actual['inventory'],
            previous_process_exit_proof=previous_proof, native_started=False, GPU_started=False,
            models_retrained=False, old_evidence_modified=False, TEST_used_for_fitting=False,
            completed_results_selected_by_success=False, automatic_retry=False)
        for name in ('launch', 'plan', 'status', 'terminal'):
            control['previous_'+name] = frozen_pointer(PREVIOUS+'/'+name+'.json')
            control['original_'+name] = deepcopy(prior['previous_'+name])
        for name in ('launch', 'status', 'failure'):
            control['previous_controller_'+name] = frozen_pointer(PREVIOUS_CONTROLLER+'/'+name+'.json')
        control['previous_DRAIN'] = frozen_pointer(PREVIOUS+'/DRAIN')
        core.require_partition(control, previous['plan'], previous['status'], previous['terminal'])
        owner.require_resource_qualification(qualification, control)
        (root/SETUP).mkdir(parents=True, exist_ok=False)
        atomic_json(root/CONTROL, control)
        cp = add_frozen_file(root, files, CONTROL)
        final_graph = FrozenInputIndex(root, files); require_canonical_consistency(final_graph, files)
        verify_inputs(root, files)
        require(final_graph.valid() and not stopped(root), 'STOP/topology change before final lock publication')
        atomic_json(root/LOCK, dict(schema=core.LOCK_SCHEMA, approved_for_execution=True, control=cp, files=files,
            split='test', policy_sha256=seed['policy_sha256'], original_suite_map_preserved=True,
            original_source_map_preserved=True, native_started=False, GPU_started=False, automatic_retry=False))
        return dict(complete=True, expected=1920, imported_completed=len(part['completed']),
            historical_infrastructure_failed=len(part['failed']), remaining=len(part['remaining']),
            resume_control=cp, resume_control_lock=add_frozen_file(root, {}, LOCK),
            native_started=False, GPU_started=False, jobs_started=False, old_evidence_modified=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--environment-recheck-sha256')
    args = parser.parse_args(argv)
    if not args.prepare:
        print(json.dumps(dict(execution_enabled=False, native_started=False, GPU_started=False,
            message='inactive: no inputs inspected and no execution materials created')))
        return
    print(json.dumps(prepare(environment_recheck_sha256=args.environment_recheck_sha256), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
