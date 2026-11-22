"""Inactive, one-time seven-tier recovery after the observed host reboot.

Only --prepare writes a new control. No worker, model or simulator is started.
The damaged saved completions and unknown interrupted attempts retain their old
bytes; every intact NativeFalse/unknown-motion completion is adopted literally.
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
from our_mani.multitask import composite_TEST_reboot_resume_v1070 as core
from our_mani.multitask import composite_TEST_reboot_resume_v1055 as parent_core
from our_mani.multitask import composite_TEST_resume_v1040 as fifth_core
from scripts import run_composite_TEST_owner_v1041_r10 as owner
from scripts import prepare_composite_TEST_resume_v1040_r10 as parent_preparer
from scripts import prepare_concurrent_TEST_resume_v915_r10 as base
from scripts import audit_concurrent_resumed_formal_TEST_v906_r10 as old_audit
from scripts import run_batched_formal_driver_v901_r10 as bootstrap

SOURCE = 'scripts/prepare_composite_TEST_reboot_resume_v1070_r10.py'
SETUP, CONTROL, LOCK = core.SETUP, core.CONTROL, core.LOCK
PREVIOUS, PREVIOUS_CONTROLLER = core.PREVIOUS, core.PREVIOUS_CONTROLLER
SUITE, SUITE_LOCK, SOURCE_INPUTS, VAL_RELEASE = base.SUITE, base.SUITE_LOCK, base.SOURCE_INPUTS, base.VAL_RELEASE
PREVIOUS_CONTROL, PREVIOUS_LOCK = core.PREVIOUS_CONTROL, core.PREVIOUS_LOCK
ENVIRONMENT_RECHECK = 'runtime/framework/frozen_TEST_environment_recheck_after_second_reboot_v1070.json'
RECHECK_SOURCE = 'scripts/check_frozen_TEST_environment_v954_r10.py'
REBOOT_AUDIT = 'runtime/framework/second_reboot_TEST_evidence_v1067.json'
REBOOT_AUDITOR = 'scripts/audit_second_reboot_TEST_evidence_v1067_r10.py'
HOST_REBOOT_OBSERVATION = 'runtime/framework/reboot_hardware_boot_observation_v1068.json'
HOST_REBOOT_OBSERVATION_SHA = 'e904de8eddfccf2858d70e575540e754900e7f82225459fd96183bcc4981c4ef'
CONTROLLER_RUNTIME = 'runtime/framework/ordered_second_reboot_TEST_v1073'
PROFILE = deepcopy(parent_preparer.PROFILE)
QUALIFIER, QUALIFICATION = parent_preparer.QUALIFIER, parent_preparer.QUALIFICATION
QUALIFICATION_LAUNCH = parent_preparer.QUALIFICATION_LAUNCH
AUTHORIZATION = parent_preparer.AUTHORIZATION
TESTS = ('tests/test_composite_TEST_reboot_resume_v1070.py', 'tests/test_prepare_composite_TEST_reboot_resume_v1070.py',
    'tests/test_second_reboot_TEST_owner_v1071.py','tests/test_second_reboot_TEST_results_v1072.py',
    'tests/test_second_reboot_TEST_controller_v1073.py')
REQUIRED_SOURCES = tuple(sorted({SOURCE, core.SOURCE, core.OWNER, core.AUDITOR, core.CONTROLLER,
    RECHECK_SOURCE, REBOOT_AUDITOR, *TESTS,
    'scripts/prepare_composite_TEST_reboot_resume_v1055_r10.py',
    'our_mani/multitask/composite_TEST_reboot_resume_v1055.py',
    'scripts/run_reboot_TEST_owner_v1056_r10.py',
    'scripts/audit_reboot_TEST_results_v1057_r10.py',
    'scripts/continue_reboot_TEST_v1058_r10.py', *parent_preparer.REQUIRED_SOURCES}))
STOP_PATHS = tuple(dict.fromkeys((*parent_preparer.STOP_PATHS, core.RUNTIME+'/STOP',
    PREVIOUS+'/STOP', PREVIOUS_CONTROLLER+'/STOP', CONTROLLER_RUNTIME+'/STOP')))
merge_maps = base.merge_maps
add_frozen_file = base.add_frozen_file
require_canonical_consistency = base.require_canonical_consistency
qualification_path_allowed = parent_preparer.qualification_path_allowed
require_resource_proof = parent_preparer.require_resource_proof
capacity_selection = parent_preparer.capacity_selection


def require(value, message):
    if not value:
        raise ValueError('reboot TEST preparation: '+message)


def stopped(root=ROOT):
    return any((Path(root)/name).exists() for name in STOP_PATHS)


def partition(plan, status, reboot_audit):
    accepted, rejected, saved = core.audit_partition(reboot_audit)
    failed = set(status.get('failed', {})); remaining = {r['job_id'] for r in plan.get('jobs', [])} - accepted
    control = dict(schema=core.CONTROL_SCHEMA, split='test', expected_cases=1920, source_count=640,
        policy_sha256=plan.get('policy_sha256'), user_authorized_resume=True,
        standing_authorization_reused=True, new_user_event_invented=False,
        physical_policy_changed=False, explicit_host_reboot_recovery=True)
    for name, values in dict(completed=accepted, failed=failed, remaining=remaining,
            saved_completed=saved, rejected_saved_completed=rejected).items():
        values = sorted(values); control[name+'_job_ids'] = values; control[name+'_key_sha256'] = digest(values)
    jobs = core.require_partition(control, plan, status, reboot_audit)
    return dict(jobs=jobs, completed=sorted(accepted), failed=sorted(failed), remaining=sorted(remaining),
        rejected=sorted(rejected), saved=sorted(saved), policy_sha256=plan['policy_sha256'], partition_control=control)


def old_evidence_paths(row):
    core.require_parent_row(row)
    key = row['job_id']; runtime = Path(row['evidence_runtime'])
    names = [str(runtime/'CPU'/key/name) for name in ('launch.json','stdout.log','stderr.log','terminal.json','CPU_preflight.json')]
    names += [str(runtime/'native'/key/name) for name in ('launch.json','stdout.log','stderr.log','terminal.json')]
    out = Path(core.option(row['physical_argv'], '--out'))
    names += [str(out/'case.json'), str(out/'admission_driver_receipt.json')]
    if row['execution_entry']['kind'] == 'batched901': names.append(str(out/'batch_driver_receipt.json'))
    return names


def require_absent_evidence(audit, root=ROOT):
    expected = {runtime+'/'+leaf for runtime in (PREVIOUS, PREVIOUS_CONTROLLER)
        for leaf in ('terminal.json','failure.json','DRAIN','STOP')}
    observations = audit.get('absent_evidence')
    require(isinstance(observations, dict) and set(observations) == expected, 'exact interrupted owner/controller absences required')
    for name, record in observations.items():
        require(record == dict(path=name, exists=False, is_file=False)
            and not os.path.lexists(Path(root)/name), 'old absent evidence appeared or was invented: '+name)
    return deepcopy(observations)


def require_previous_exit(previous, audit, is_alive, root=ROOT):
    require_absent_evidence(audit, root)
    require(audit.get('current_boot_id') == (Path('/proc/sys/kernel/random/boot_id').read_text().strip()),
        'recovery observation belongs to another boot')
    for key in ('launch','controller_launch'):
        value = previous[key]
        require(type(value.get('pid')) is int and value['pid'] > 0
            and type(value.get('start_ticks')) is int and value['start_ticks'] > 0 and not is_alive(value),
            'actual old1056/1058 PID/start must have exited')
    observed = audit['old_owner_process']
    require(observed.get('pid') == previous['launch']['pid']
        and observed.get('expected_start_ticks') == previous['launch']['start_ticks']
        and observed.get('same_process_alive') is False, 'actual1067 old owner identity differs')
    part = partition(previous['plan'], previous['status'], audit)
    return dict(previous_owner_and_controller_exited=True, host_reboot_observed=True,
        current_boot_id=audit['current_boot_id'], old_terminal_reconstructed=False,
        identities=[dict(pid=previous[k]['pid'],start_ticks=previous[k]['start_ticks']) for k in ('launch','controller_launch')],
        saved_completed=len(part['saved']), adopted_completed=len(part['completed']),
        damaged_saved_completed=len(part['rejected']), remaining=len(part['remaining']))


def require_host_reboot_observation(value, previous, audit):
    launch = previous['launch']
    require(value.get('schema') == 'reboot_hardware_boot_point_observation_v1068'
        and value.get('boot_id') == audit.get('current_boot_id')
        and value.get('old_owner_pid') == launch.get('pid')
        and value.get('old_owner_start_ticks') == launch.get('start_ticks')
        and value.get('old_owner_created_utc') == launch.get('created_utc')
        and value.get('old_owner_created_before_current_boot') is True
        and value.get('old_start_ticks_impossible_in_current_boot') is True
        and value.get('old_owner_PID_exists') is False
        and value.get('historical_1031_capacity_reused') is True
        and value.get('capacity_canary_rerun_after_reboot') is False
        and value.get('GPU_exclusive_claimed') is False
        and value.get('other_project_processes_modified') is False and value.get('Native_started') is False,
        'actual same-boot1068 owner-loss and historical-capacity observation required')
    btime, uptime, hz = (value.get(k) for k in ('btime_unix','uptime_seconds','clock_ticks_per_second'))
    require(type(btime) is int and btime > 0 and type(uptime) in (int,float) and uptime > 0
        and type(hz) is int and hz > 0
        and launch['start_ticks'] > uptime * hz
        and datetime.fromisoformat(launch['created_utc']).timestamp() < btime,
        'old owner must predate this boot and cannot have its saved ticks within this boot')
    return value


@contextmanager
def idle_resources(env, previous, audit, root=ROOT):
    require(not stopped(root), 'STOP before resource checks')
    require_previous_exit(previous, audit, owner.alive, root)
    with ExitStack() as stack:
        locks = (PREVIOUS_CONTROLLER+'.lock','runtime/framework/ordered_composite_TEST_v1043.lock','runtime/framework/ordered_composite_TEST_v1013.lock',
            'runtime/framework/ordered_composite_TEST_v958.lock','runtime/framework/ordered_concurrent_TEST_v907.lock',*base.canary.LOCKS)
        for name in dict.fromkeys(locks):
            stream = stack.enter_context((Path(root)/name).open('a+')); fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        require_previous_exit(previous, audit, owner.alive, root)
        inventory = owner.process_visibility()
        require(inventory.get('host_visibility_available') is True and inventory.get('same_uid_inventory_incomplete') is False
            and inventory.get('live_project_native_processes') == [] and inventory.get('ancestor_chain_complete') is True,
            'complete empty host-visible Native/controller inventory required')
        actual = owner.environment_owner.require_frozen_environment(env)
        require(not stopped(root), 'STOP after resource checks')
        yield dict(inventory=inventory, frozen_environment=actual)


def interrupted_attempt_inventory(root, row, ledger):
    """Hash existing files in this exact old output and its two stage leaves."""
    root = Path(root).resolve(); key = row['job_id']; records = []
    for kind in ('CPU','native'):
        leaf = root/PREVIOUS/kind/key
        for name in ('launch.json','terminal.json','stdout.log','stderr.log','CPU_preflight.json'):
            path = leaf/name
            require(not path.is_symlink(), 'old interrupted stage alias refused')
            if path.exists(): records.append(ledger.read(path))
    out = root/core.option(row['physical_argv'], '--out')
    require(out.is_relative_to(root/'artifacts/framework/multitask_r10') and not out.is_symlink(), 'bounded old attempt output required')
    directories, partial = [], []; exists = out.exists()
    if exists:
        require(out.is_dir() and out.resolve(strict=True) == out, 'canonical old partial output directory required')
        for current, dirs, files in os.walk(out, followlinks=False):
            current = Path(current); dirs.sort()
            require(current.resolve(strict=True) == current, 'old partial directory alias refused')
            directories.append(str(current.relative_to(root)))
            require(all(not (current/name).is_symlink() for name in [*dirs,*files]), 'old partial symlink refused')
            for name in sorted(files):
                path = current/name; require(path.is_file(), 'old partial special file refused'); partial.append(ledger.read(path))
    return dict(job_id=key, infrastructure_attempt_not_native_outcome=True, outcome_unknown=True,
        output_path=str(out.relative_to(root)), output_existed=exists, directories=sorted(directories),
        log_files=records, partial_files=partial)


def freeze_completed(row, record, ledger, control, stage_owner, sources, profile):
    from scripts import audit_reboot_TEST_results_v1057_r10 as audit
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



def require_historical_chain(prior, files, ledger, current_parent_plan):
    """Close frozen transformations without invoking a historical current-boot gate.

    Every byte map remains verified. This reads only historical metadata; the
    old1411 case semantics are inherited from the frozen1055 preparation.
    """
    p = prior['previous_resume_control']; lp = prior['previous_resume_control_lock']
    require(p.get('path') == fifth_core.CONTROL and lp.get('path') == fifth_core.LOCK,
        'exact1040 ancestor control and lock paths required')
    fifth_control = ledger.json(p['path'], expected=p['sha256'])
    fifth_lock = ledger.json(lp['path'], expected=lp['sha256'])
    require(fifth_control.get('schema') == fifth_core.CONTROL_SCHEMA
        and fifth_lock.get('schema') == fifth_core.LOCK_SCHEMA
        and fifth_lock.get('approved_for_execution') is True,
        'actual immutable1040 ancestor control/lock required')
    verify_inputs(ROOT, fifth_lock['files'])
    require(all(files.get(k) == v for k,v in fifth_lock['files'].items()),
        'all original1040 keys remain literal in1055 map')
    for key in ('suite','suite_lock','source_inputs','validation_release','policy_sha256','author_outcome_protocol',
            'execution_entry','execution_implementation','environment_manifest','environment_wrapper',
            'environment_namespace_proof','standing_continuation_record'):
        require(prior.get(key) == fifth_control.get(key), 'unchanged historical scientific binding required: '+key)
    _, original, older = owner.require_history(fifth_control, fifth_lock['files'])
    fifth = {name:ledger.json(prior['previous_'+name]['path'], expected=prior['previous_'+name]['sha256'])
        for name in ('launch','plan','status')}
    fifth_core.require_resume_plan(fifth_control, older['plan'], older['status'], older['terminal'], fifth['plan'])
    old_boot = ledger.json(prior['reboot_audit']['path'], expected=prior['reboot_audit']['sha256'])
    parent_core.require_resume_plan(prior, fifth['plan'], fifth['status'], old_boot, current_parent_plan)
    require(all(prior.get('original_'+k) == fifth_control.get('original_'+k) for k in ('launch','plan','status','terminal')),
        'literal original753 pointers must survive both host interruptions')
    return original, older, fifth


def fsync_path(path):
    descriptor = os.open(path, os.O_RDONLY | (os.O_DIRECTORY if Path(path).is_dir() else 0))
    try: os.fsync(descriptor)
    finally: os.close(descriptor)


def prepare(*, environment_recheck_sha256, resource_qualification_sha256, reboot_audit_sha256,
            concurrency_authorization_sha256=None, resource_profile=None, root=ROOT):
    root = Path(root).resolve()
    require(all(core.valid_sha(v) for v in (environment_recheck_sha256,resource_qualification_sha256,reboot_audit_sha256)),
        'external actual fresh954, historical1031 and reboot1067 SHA required')
    require(concurrency_authorization_sha256 is None or core.valid_sha(concurrency_authorization_sha256), 'invalid optional original authorization SHA')
    require(root == ROOT and not stopped(root), 'actual project and continuing authorization required')
    require(not any((root/name).exists() for name in (SETUP,core.RUNTIME,core.OUTPUT,CONTROLLER_RUNTIME)), 'all new one-time namespaces must be fresh')
    for name in REQUIRED_SOURCES: require((root/name).is_file(), 'new source/test is not ready: '+name)
    ledger = old_audit.EvidenceLedger(root); pointer, read = ledger.pointer, ledger.json
    release_pointer = pointer(VAL_RELEASE)
    require(release_pointer['sha256'] == base.VAL_RELEASE_SHA, 'exact real complete824 VAL release required')
    release = read(VAL_RELEASE)
    owner.require_complete_VAL_release(dict(validation_release=release_pointer,policy_sha256=release.get('policy_sha256'),
        author_outcome_protocol=release.get('author_outcome_protocol')), VAL_RELEASE)
    prior_pointer, prior_lock_pointer = pointer(PREVIOUS_CONTROL), pointer(PREVIOUS_LOCK)
    prior, prior_lock = read(PREVIOUS_CONTROL), read(PREVIOUS_LOCK)
    require(prior.get('schema') == parent_core.CONTROL_SCHEMA and prior_lock.get('schema') == parent_core.LOCK_SCHEMA
        and prior_lock.get('approved_for_execution') is True, 'actual immutable1055 control/lock required')
    verify_inputs(root, prior_lock['files'])
    previous = {name:read(PREVIOUS+'/'+name+'.json') for name in ('launch','plan','status')}
    previous.update({name:read(PREVIOUS_CONTROLLER+'/'+name.removeprefix('controller_')+'.json') for name in ('controller_launch','controller_status')})
    audit = ledger.json(REBOOT_AUDIT, expected=reboot_audit_sha256)
    require(audit.get('source') == pointer(REBOOT_AUDITOR), 'actual frozen1067 source required')
    ledger.read(REBOOT_AUDITOR, expected=audit['source']['sha256'])
    for name in ('launch','plan','status'):
        observation = audit.get('inputs', {}).get('previous_owner_'+name, {})
        require(observation.get('path') == PREVIOUS+'/'+name+'.json' and observation.get('SHA_matches') is True
            and observation.get('sha256') == pointer(PREVIOUS+'/'+name+'.json')['sha256'], '1067 old input changed: '+name)
    host_reboot_observation = ledger.json(HOST_REBOOT_OBSERVATION,expected=HOST_REBOOT_OBSERVATION_SHA)
    require_host_reboot_observation(host_reboot_observation,previous,audit)
    previous_proof = require_previous_exit(previous,audit,owner.alive,root)
    part = partition(previous['plan'],previous['status'],audit)
    original, older, fifth = require_historical_chain(prior,prior_lock['files'],ledger,previous['plan'])
    inherited = set(prior['completed_job_ids'])
    require(inherited <= set(part['completed'])
        and all(previous['status']['completed'].get(k)==fifth['status']['completed'].get(k) for k in inherited),
        'all frozen intact1411 historical outcomes must be adopted literally')
    fresh_completed = set(part['completed']) - inherited
    require(all(part['jobs'][k]['evidence_runtime'] == PREVIOUS for k in fresh_completed),
        'only actual1056 completions may extend the previously adopted cohort')
    env = {key:pointer(name) for key,name in base.ENVIRONMENT_PATHS.items()}
    require(all(prior.get(key)==value for key,value in env.items()), 'original820 material pointers changed')
    authorization_pointer = prior['concurrency_authorization_record']
    require(authorization_pointer['path'] == AUTHORIZATION and (concurrency_authorization_sha256 is None
        or authorization_pointer['sha256'] == concurrency_authorization_sha256), 'reuse the exact real original concurrency authorization')
    authorization = ledger.json(authorization_pointer['path'], expected=authorization_pointer['sha256'])
    require(authorization.get('user_authorized_concurrency_change') is True, 'standing concurrency authorization required')
    selected = capacity_selection('qualified1031', resource_profile)
    seed = dict(prior, **part['partition_control'], previous_runtime=PREVIOUS, previous_controller_runtime=PREVIOUS_CONTROLLER,
        suite=pointer(SUITE), suite_lock=pointer(SUITE_LOCK), source_inputs=pointer(SOURCE_INPUTS), validation_release=release_pointer,
        resource_profile=selected['resource_profile'],resource_capacity_envelope=selected['resource_capacity_envelope'],
        resource_capacity_source='qualified1031',resource_profile_within_capacity=True,
        resource_qualification=ledger.read(QUALIFICATION,expected=resource_qualification_sha256),resource_qualifier=pointer(QUALIFIER),
        previous_resource_profile=deepcopy(prior['resource_profile']),previous_resource_qualification=deepcopy(prior['resource_qualification']),
        resource_profile_changed=selected['resource_profile']!=prior['resource_profile'],
        concurrency_authorization_record=authorization_pointer,reboot_audit=pointer(REBOOT_AUDIT),
        host_reboot_observation=pointer(HOST_REBOOT_OBSERVATION),
        historical_capacity_qualification_reused=True,resource_canary_rerun_after_reboot=False,
        absent_evidence=deepcopy(audit['absent_evidence']),**env)
    for key in ('previous_terminal','previous_controller_failure','previous_DRAIN'):
        seed.pop(key,None)
    recheck = ledger.json(ENVIRONMENT_RECHECK,expected=environment_recheck_sha256)
    core.require_environment_recheck(recheck,seed); ledger.read(recheck['source']['path'],expected=recheck['source']['sha256'])
    for value,source in ((previous['launch'],core.PREVIOUS_OWNER),(previous['controller_launch'],core.PREVIOUS_CONTROLLER_SOURCE)):
        require(value.get('source_sha256') == pointer(source)['sha256'] and value.get('resume_control') == prior_pointer
            and value.get('resume_control_lock') == prior_lock_pointer, 'old1056/1058 source/control changed')
    with idle_resources(env,previous,audit,root): pass
    suite, suite_lock, source_meta = read(SUITE),read(SUITE_LOCK),read(SOURCE_INPUTS)
    suite_files, source_files = suite_lock['files'],source_meta['files']; verify_inputs(root,suite_files)
    manifest,sources,proof = read_formal_source(root,suite,suite_files,'test',validation_release=release)
    require(proof['policy_sha256'] == seed['policy_sha256'] and suite['author_outcome_protocol'] == seed['author_outcome_protocol'] == outcome_protocol_binding(root),
        'same original physical policy/outcome protocol required')
    require(source_meta.get('schema') == 'r10_full16_post_admission_split_source_lock_v1' and source_meta.get('split') == 'test'
        and source_meta.get('source_count') == len(sources) == 640 and source_meta.get('policy_sha256') == seed['policy_sha256']
        and source_meta.get('TEST_used_for_fitting') is False and source_meta.get('source_manifest_sha256') == suite['source_manifests']['test']['sha256'],
        'unchanged original826 admitted640 source map required')
    verify_inputs(root,source_files)
    old_audit.require_original_plan(original['plan'],seed,suite,sources,proof,suite['formal_case_driver'],suite['formal_admission_guard'])
    shared = {name:pointer(name) for name in bootstrap.SOURCES}
    require(seed['execution_entry'] == prior['execution_entry'] and seed['execution_implementation'] == owner.execution_implementation(seed['execution_entry'],shared),
        'original physical901 implementation cannot change')
    batch_control = read(seed['execution_entry']['batch_control']['path']); batch_lock = read(seed['execution_entry']['batch_lock']['path'])
    require(pointer(seed['execution_entry']['batch_control']['path']) == seed['execution_entry']['batch_control']
        and pointer(seed['execution_entry']['batch_lock']['path']) == seed['execution_entry']['batch_lock']
        and batch_control == base.formal_bootstrap_control(suite) and batch_lock.get('schema') == bootstrap.LOCK_SCHEMA
        and batch_lock.get('execution_enabled') is True and all(batch_lock['files'].get(str(root/n),batch_lock['files'].get(n)) == p['sha256'] for n,p in shared.items()),
        'original915 batch control/lock and seven implementation sources must remain unchanged')
    qualification = ledger.json(QUALIFICATION,expected=resource_qualification_sha256)
    require(qualification.get('actual_independent_audit',{}).get('path') == QUALIFICATION_LAUNCH, 'actual historical1031 independent audit required')
    audit_launch = ledger.json(QUALIFICATION_LAUNCH,expected=qualification['actual_independent_audit']['sha256'])
    require_resource_proof(qualification,seed,audit_launch=audit_launch)
    for name,checksum in qualification['inputs'].items():
        require(qualification_path_allowed(name), 'noncanary qualification input refused'); ledger.read(name,expected=checksum)
    for key in ('control','lock'): ledger.read(qualification[key]['path'],expected=qualification[key]['sha256'])
    files = merge_maps(prior_lock['files'],suite_files,source_files)
    profile = fingerprint(proof['routes']['common_author_RML_profile'])
    # Every adopted row identifies its real physical owner; no flattened ownership.
    stage_owners = {runtime:read(runtime+'/launch.json') for runtime in {row['evidence_runtime'] for row in previous['plan']['jobs']}}
    # Older1411 full byte/SHA maps and literal records were closed above.
    # Reparse semantic case evidence only for new1056 completions; no success filtering.
    for key in sorted(fresh_completed):
        require(not stopped(root), 'STOP during complete evidence freeze')
        row = part['jobs'][key]
        freeze_completed(row,previous['status']['completed'][key],ledger,seed,stage_owners[row['evidence_runtime']],shared,profile)
    attempts = {key:interrupted_attempt_inventory(root,part['jobs'][key],ledger) for key in part['remaining']}
    standing = read(base.STANDING)
    require(standing.get('continue_until_user_stop') is True and standing.get('user_stop_received') is False, 'real standing continuation required')
    for name in (*REQUIRED_SOURCES,*base.ENVIRONMENT_EXTRA): ledger.read(name)
    for name,value in ledger.used.items(): add_frozen_file(root,files,name,value['pointer']['sha256'])
    require(all(files.get(k)==v for k,v in prior_lock['files'].items()), 'every original1055 key/SHA must remain literal')
    verify_inputs(root,files); graph = FrozenInputIndex(root,dict(files)); require_canonical_consistency(graph,files); ledger.close()
    with idle_resources(env,previous,audit,root) as actual:
        require(not any((root/n).exists() for n in (SETUP,core.RUNTIME,core.OUTPUT,CONTROLLER_RUNTIME)), 'fresh namespaces required')
        require(graph.valid() and not stopped(root), 'STOP or changed topology before setup commit')
        class ClosedLedger:
            def read(self,path):
                absolute = str(root/path)
                require(absolute in ledger.used and file_sha(absolute)==ledger.used[absolute]['pointer']['sha256'], 'old interrupted evidence changed')
                return deepcopy(ledger.used[absolute]['pointer'])
        require({k:interrupted_attempt_inventory(root,part['jobs'][k],ClosedLedger()) for k in part['remaining']} == attempts,
            'old interrupted directory inventories changed before commit')
        def frozen_pointer(name):
            value = ledger.used.get(str(root/name)); require(value is not None,'unobserved evidence pointer after close'); return deepcopy(value['pointer'])
        control = dict(seed,created_utc=datetime.now(timezone.utc).isoformat(),resume_owner=frozen_pointer(core.OWNER),
            resume_auditor=frozen_pointer(core.AUDITOR),result_auditor=frozen_pointer(core.AUDITOR),resume_controller=frozen_pointer(core.CONTROLLER),
            previous_resume_control=prior_pointer,previous_resume_control_lock=prior_lock_pointer,environment_recheck=frozen_pointer(ENVIRONMENT_RECHECK),
            reboot_audit=frozen_pointer(REBOOT_AUDIT),interrupted_attempt_evidence=attempts,failed_attempt_evidence={},
            original_input_maps_preserved=True,prepared_frozen_environment=actual['frozen_environment'],preparation_inventory=actual['inventory'],
            previous_process_exit_proof=previous_proof,
            inherited_completed_job_ids=sorted(inherited), inherited_completed_key_sha256=digest(sorted(inherited)),
            fresh_completed_job_ids=sorted(fresh_completed), fresh_completed_key_sha256=digest(sorted(fresh_completed)),
            old_semantic_case_evidence_inherited=True, full_original_input_maps_reverified=True, native_started=False,GPU_started=False,models_retrained=False,
            old_evidence_modified=False,TEST_used_for_fitting=False,completed_results_selected_by_success=False,automatic_retry=False)
        for name in ('launch','plan','status'):
            control['previous_'+name] = frozen_pointer(PREVIOUS+'/'+name+'.json')
        for name in ('launch','plan','status','terminal'): control['original_'+name] = deepcopy(prior['original_'+name])
        for name in ('launch','status'): control['previous_controller_'+name] = frozen_pointer(PREVIOUS_CONTROLLER+'/'+name+'.json')
        core.require_partition(control,previous['plan'],previous['status'],audit); owner.require_resource_qualification(qualification,control)
        (root/SETUP).mkdir(parents=True,exist_ok=False)
        atomic_json(root/CONTROL,control); fsync_path(root/CONTROL); fsync_path(root/SETUP); fsync_path((root/SETUP).parent)
        cp = add_frozen_file(root,files,CONTROL)
        final_graph = FrozenInputIndex(root,files); require_canonical_consistency(final_graph,files); verify_inputs(root,files)
        require(final_graph.valid() and not stopped(root), 'STOP/topology change before lock publication')
        atomic_json(root/LOCK,dict(schema=core.LOCK_SCHEMA,approved_for_execution=True,control=cp,files=files,
            split='test',policy_sha256=seed['policy_sha256'],original_suite_map_preserved=True,original_source_map_preserved=True,
            native_started=False,GPU_started=False,automatic_retry=False))
        fsync_path(root/LOCK); fsync_path(root/SETUP); fsync_path((root/SETUP).parent)
        return dict(complete=True,expected=1920,saved_completed=len(part['saved']),imported_completed=len(part['completed']),
            damaged_saved_completed=len(part['rejected']),remaining=len(part['remaining']),resume_control=cp,
            resume_control_lock=add_frozen_file(root,{},LOCK),native_started=False,GPU_started=False,jobs_started=False,old_evidence_modified=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--prepare',action='store_true')
    for name in ('environment-recheck-sha256','resource-qualification-sha256','reboot-audit-sha256','concurrency-authorization-sha256'):
        parser.add_argument('--'+name)
    parser.add_argument('--resource-profile-json', help='explicit frozen actual quotas within historical1031 capacity')
    args = parser.parse_args(argv)
    if not args.prepare:
        print(json.dumps(dict(execution_enabled=False,prepared=False,Native_started=False))); return
    print(json.dumps(prepare(environment_recheck_sha256=args.environment_recheck_sha256,
        resource_qualification_sha256=args.resource_qualification_sha256,reboot_audit_sha256=args.reboot_audit_sha256,
        concurrency_authorization_sha256=args.concurrency_authorization_sha256,
        resource_profile=None if args.resource_profile_json is None else json.loads(args.resource_profile_json)),ensure_ascii=False),flush=True)


if __name__ == '__main__': main()
