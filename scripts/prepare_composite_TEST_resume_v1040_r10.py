"""Inactive preparation of one new, same-policy, five-tier TEST continuation.

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
from our_mani.multitask import composite_TEST_resume_v1040 as core
from our_mani.multitask import composite_TEST_resume_v1010 as parent_core
from our_mani.multitask import composite_TEST_resume_v955 as grandparent_core
from scripts import run_composite_TEST_owner_v1041_r10 as owner
from scripts import prepare_composite_TEST_resume_v1010_r10 as parent_preparer
from scripts import prepare_concurrent_TEST_resume_v915_r10 as base
from scripts import audit_concurrent_resumed_formal_TEST_v906_r10 as old_audit
from scripts import run_batched_formal_driver_v901_r10 as bootstrap

SOURCE = 'scripts/prepare_composite_TEST_resume_v1040_r10.py'
SETUP, CONTROL, LOCK = core.SETUP, core.CONTROL, core.LOCK
PREVIOUS, PREVIOUS_CONTROLLER = core.PREVIOUS, core.PREVIOUS_CONTROLLER
SUITE, SUITE_LOCK, SOURCE_INPUTS, VAL_RELEASE = base.SUITE, base.SUITE_LOCK, base.SOURCE_INPUTS, base.VAL_RELEASE
PREVIOUS_CONTROL, PREVIOUS_LOCK = core.PREVIOUS_CONTROL, core.PREVIOUS_LOCK
ENVIRONMENT_RECHECK = 'runtime/framework/frozen_TEST_environment_recheck_v954.json'
RECHECK_SOURCE = 'scripts/check_frozen_TEST_environment_v954_r10.py'
PROFILE = dict(schema='r10_frozen_concurrent_native_resource_profile_v1031',max_native_slots=12,max_GPU_slots=5,max_CPU_preflight_slots=6,ready_limit=24)
FALLBACK_PROFILE=deepcopy(parent_preparer.PROFILE)
QUALIFIER='scripts/audit_concurrent_TRAIN_resource_canary_v1031_r10.py'
QUALIFICATION='runtime/framework/concurrent_TRAIN_resource_qualification_v1031.json'
QUALIFICATION_LAUNCH='runtime/framework/concurrent_TRAIN_resource_audit_v1031/launch.json'
CANARY_RUNTIME='runtime/framework/concurrent_TRAIN_resource_canary_v1031'
CANARY_OUTPUT='artifacts/framework/multitask_r10/concurrent_TRAIN_resource_canary_v1031/development_demo_only'
AUTHORIZATION='runtime/framework/TEST_speed_authorization_for_v1040.json'
QUALIFICATION_SOURCES=('scripts/prepare_concurrent_TRAIN_resource_canary_v1031_r10.py',
    'scripts/run_concurrent_TRAIN_resource_canary_v1031_r10.py',QUALIFIER)
TESTS = ('tests/test_composite_TEST_resume_v1040.py', 'tests/test_prepare_composite_TEST_resume_v1040.py',
    'tests/test_composite_TEST_owner_v1041.py', 'tests/test_composite_TEST_results_v1042.py',
    'tests/test_composite_TEST_controller_v1043.py')
REQUIRED_SOURCES = tuple(sorted({SOURCE, core.SOURCE, core.OWNER, core.AUDITOR, core.CONTROLLER,
    RECHECK_SOURCE, *TESTS, *QUALIFICATION_SOURCES,
    'tests/test_concurrent_TRAIN_resource_canary_v1031.py','tests/test_concurrent_TRAIN_resource_audit_v1031.py',
    'tests/test_prepare_concurrent_TRAIN_resource_canary_v1031.py',*parent_preparer.REQUIRED_SOURCES}))
STOP_PATHS = tuple(dict.fromkeys((*base.STOP_PATHS, core.RUNTIME+'/STOP',
    'runtime/framework/formal_TEST_composite_resume_v956/STOP', 'runtime/framework/formal_TEST_composite_resume_v1011/STOP',
    'runtime/framework/ordered_composite_TEST_v958/STOP','runtime/framework/ordered_composite_TEST_v1013/STOP', 'runtime/framework/ordered_composite_TEST_v1043/STOP')))
merge_maps = base.merge_maps
add_frozen_file = base.add_frozen_file
require_canonical_consistency = base.require_canonical_consistency



def require(value, message):
    if not value:
        raise ValueError('composite TEST preparation: '+message)


def stopped(root=ROOT):
    return any((Path(root)/name).exists() for name in STOP_PATHS)


def partition(plan, status, terminal):
    done = status.get('completed'); failed = status.get('failed')
    require(isinstance(done, dict) and isinstance(failed, dict), 'real1011 completed and infrastructure maps required')
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
            and not is_alive(value), 'actual1011 and1013 PID/start must have exited')
    state, failure = previous['controller_status'], previous['controller_failure']
    require(state.get('stage') == 'requires_concrete_repair' and state.get('automatic_retry') is False
        and failure.get('complete') is False and failure.get('user_stop') is False
        and failure.get('automatic_retry') is False, 'real1013 partial failure/status required; no successful terminal is invented')
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
        for name in ('runtime/framework/ordered_composite_TEST_v1013.lock','runtime/framework/ordered_composite_TEST_v958.lock','runtime/framework/ordered_concurrent_TEST_v907.lock', *base.canary.LOCKS):
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
    from scripts import audit_composite_TEST_results_v1042_r10 as audit
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


def qualification_path_allowed(name, *, capacity_source='qualified1031'):
    if capacity_source=='qualified1005_fallback':return parent_preparer.qualification_path_allowed(name)
    if capacity_source!='qualified1031' or not isinstance(name,str) or Path(name).is_absolute() or '..' in Path(name).parts:return False
    path=Path(name)
    if path.is_relative_to(CANARY_RUNTIME):
        suffix=path.relative_to(CANARY_RUNTIME).parts
        return (len(suffix)==1 and suffix[0] in ('plan.json','CPU_owner_launch.json','CPU_owner_terminal.json','native_owner_launch.json',
            'native_owner_terminal.json','run_summary.json','native_resource_events.json','native_resource_samples.jsonl')) or (
            len(suffix)==3 and suffix[0] in ('CPU','native') and suffix[2] in ('launch.json','terminal.json','stdout.log','stderr.log','CPU_preflight.json'))
    if path.is_relative_to(CANARY_OUTPUT):
        suffix=path.relative_to(CANARY_OUTPUT).parts
        selection={('pick_moving_target_on_the_table',r) for r in ('episode0','episode1','episode105','episode106')}
        selection.update({('pick_moving_target_on_the_table_high_speed','episode100'),('pick_moving_target_on_the_table_high_speed','episode106')})
        return len(suffix)==4 and (suffix[0],suffix[1]) in selection and suffix[2] in core.METHODS and suffix[3] in ('case.json','batch_driver_receipt.json')
    return False


def capacity_selection(capacity_source, resource_profile=None):
    if capacity_source not in ('qualified1031','qualified1005_fallback'):
        raise ValueError('explicit qualified1031 capacity or genuine qualified1005 fallback required')
    high=capacity_source=='qualified1031';envelope=deepcopy(PROFILE if high else FALLBACK_PROFILE)
    actual=deepcopy(envelope if resource_profile is None else resource_profile)
    owner.require_capacity(dict(resource_profile=actual,resource_capacity_envelope=envelope,
        resource_capacity_source=capacity_source,resource_profile_within_capacity=True))
    return dict(resource_profile=actual,resource_capacity_envelope=envelope,resource_capacity_source=capacity_source,
        resource_profile_within_capacity=True,qualifier=QUALIFIER if high else parent_preparer.QUALIFIER,
        qualification=QUALIFICATION if high else parent_preparer.QUALIFICATION,
        qualification_launch=QUALIFICATION_LAUNCH if high else parent_preparer.QUALIFICATION_LAUNCH)


def require_resource_proof(value,control,*,audit_launch):
    owner.require_resource_qualification(value,control)
    if control['resource_capacity_source']=='qualified1005_fallback':
        return parent_preparer.require_resource_proof(value,control,audit_launch=audit_launch)
    require(control['resource_qualifier']['path']==QUALIFIER and owner.valid_pointer(value.get('actual_independent_audit'))
        and value['actual_independent_audit']['path']==QUALIFICATION_LAUNCH
        and audit_launch.get('source_sha256')==control['resource_qualifier']['sha256']
        and type(audit_launch.get('pid')) is int and audit_launch['pid']>0
        and type(audit_launch.get('start_ticks')) is int and audit_launch['start_ticks']>0
        and not owner.alive(audit_launch),'actual independent exited1031 source/PID proof required')
    for key in ('control','lock'):
        require(owner.valid_pointer(value.get(key)) and audit_launch.get(key)==value[key],'actual1031 control/lock differs')
    argv=audit_launch.get('argv',[])
    require(isinstance(argv,list) and argv and (ROOT/argv[0]).resolve()==(ROOT/QUALIFIER).resolve()
        and (ROOT/owner.option(argv,'--control')).resolve()==(ROOT/value['control']['path']).resolve()
        and (ROOT/owner.option(argv,'--lock')).resolve()==(ROOT/value['lock']['path']).resolve()
        and owner.option(argv,'--lock-sha256')==value['lock']['sha256']
        and ('--out' not in argv or owner.option(argv,'--out')==QUALIFICATION),'actual independent1031 argv required')
    env=base.environment_auditor.require_environment_receipt(audit_launch.get('frozen_environment'),control,owner_pid=audit_launch['pid'])
    require(env['observer_start_ticks']==audit_launch['start_ticks'],'actual1031 namespace start differs')
    overlap=value.get('launcher_overlap',{});cpu=value.get('CPU_preflight_overlap',{})
    for record,native,gpu in ((overlap,12,5),):
        require(record.get('independently_swept_all_child_intervals') is True and record.get('overlap_verified') is True
            and record.get('observed_native_peak')==native and record.get('observed_GPU_peak')==gpu
            and isinstance(record.get('intervals'),dict) and len(record['intervals'])==18,'actual eighteen native intervals and12/5 overlap required')
    require(cpu.get('independently_swept_all_child_intervals') is True and cpu.get('overlap_verified') is True
        and cpu.get('observed_CPU_preflight_peak')==6 and isinstance(cpu.get('intervals'),dict) and len(cpu['intervals'])==18,
        'actual eighteen CPU intervals and6-preflight overlap required')
    joint=value.get('joint_supply_overlap',{});jw=joint.get('witness',{});when=jw.get('monotonic_ns')
    require(joint.get('complete') is True and joint.get('independently_swept_all_child_intervals') is True
        and joint.get('native_slots')==12 and joint.get('GPU_slots')==5 and joint.get('CPU_preflight_slots')==6
        and joint.get('each_job_CPU_checks')==1 and joint.get('each_job_Native_attempts')==1
        and joint.get('simultaneous_robot_ticks_or_hardware_saturation_claimed') is False
        and type(when) is int and when>=0,'independently verified actual joint12/5/6 supply required')
    native_ids,CPU_ids=jw.get('native_job_ids'),jw.get('CPU_job_ids')
    require(isinstance(native_ids,list) and isinstance(CPU_ids,list) and len(native_ids)==len(set(native_ids))==12
        and len(CPU_ids)==len(set(CPU_ids))==6 and not set(native_ids)&set(CPU_ids)
        and set(native_ids)|set(CPU_ids)==set(overlap['intervals'])==set(cpu['intervals'])
        and sum(k.endswith('_physmani') for k in native_ids)==5
        and all(jw.get(k)==v for k,v in dict(native_slots=12,GPU_slots=5,CPU_preflight_slots=6).items()),
        'twelve distinct Native and six distinct different CPU jobs at the same time required')
    for ids,intervals in ((native_ids,overlap['intervals']),(CPU_ids,cpu['intervals'])):
        require(all(isinstance(intervals[k],(list,tuple)) and len(intervals[k])==4
            and type(intervals[k][0]) is int and type(intervals[k][1]) is int
            and intervals[k][0]<=when<intervals[k][1] for k in ids),'joint witness must lie inside all eighteen actual intervals')
    hardware=value.get('hardware_observation',{});w=hardware.get('witness',{})
    require(hardware.get('actual_five_compute_contexts_observed') is True and type(hardware.get('valid_samples')) is int
        and hardware.get('actual_five_compute_contexts_during_joint12_5_6_observed') is True
        and hardware['valid_samples']>0 and type(hardware.get('unknown_samples')) is int and hardware['unknown_samples']==0 and hardware.get('required_headroom_mib')==2048
        and type(hardware.get('minimum_sampled_headroom_mib')) is int and hardware['minimum_sampled_headroom_mib']>=2048
        and hardware.get('continuous_hardware_bound_claimed') is False
        and hardware.get('GPU_queries_atomic_snapshot_claimed') is False,'real5-context GPU samples with2GiB headroom required')
    require(isinstance(w.get('jobs'),dict) and len(w['jobs'])==5 and set(w['jobs'])<=set(overlap['intervals'])
        and all(k.endswith('_physmani') for k in w['jobs'])
        and all(type(r.get('pid')) is int and r['pid']>0 and type(r.get('start_ticks')) is int and r['start_ticks']>0 for r in w['jobs'].values())
        and len({(r['pid'],r['start_ticks']) for r in w['jobs'].values()})==5
        and type(w.get('headroom_mib')) is int and w['headroom_mib']>=2048,'five distinct actual PhysMani main-process witnesses required')
    until=w.get('sampled_until_monotonic_ns');now=w.get('monotonic_ns')
    require(type(now) is int and type(until) is int and now<=until
        and w.get('GPU_queries_atomic_snapshot_claimed') is False,
        'actual non-atomic GPU query bracket must be explicit')
    live_native=[k for k,v in overlap['intervals'].items() if v[0]<=now<=until<v[1]]
    live_CPU=[k for k,v in cpu['intervals'].items() if v[0]<=now<=until<v[1]]
    require(len(live_native)==12 and len(live_CPU)==6 and not set(live_native)&set(live_CPU)
        and sum(k.endswith('_physmani') for k in live_native)==5 and set(w['jobs'])<=set(live_native),
        'entire actual five-context GPU query interval must fit the joint12/5/6 witness')
    require(isinstance(value.get('inputs'),dict) and value['inputs'] and all(owner.valid_pointer(dict(path=n,sha256=h))
        and qualification_path_allowed(n) for n,h in value['inputs'].items()) and value.get('native_failures_retained') is True
        and value.get('raw_manifest_model_contents_read') is False and value.get('wall_clock_contention_invariance_claimed') is False,
        'honest frozen1031 metadata/physical evidence only required')
    return value


def prepare(*, environment_recheck_sha256, resource_qualification_sha256, concurrency_authorization_sha256,
            concurrency_authorization=AUTHORIZATION, resource_capacity_source='qualified1031', resource_profile=None, root=ROOT):
    root = Path(root).resolve()
    selected=capacity_selection(resource_capacity_source,resource_profile)
    require(all(core.valid_sha(v) for v in (environment_recheck_sha256,resource_qualification_sha256,concurrency_authorization_sha256)),
        'external actual954,selected qualification and current authorization SHA required')
    require(root == ROOT and not stopped(root), 'actual project and continuing authorization required')
    require(not Path(concurrency_authorization).is_absolute() and '..' not in Path(concurrency_authorization).parts
        and Path(concurrency_authorization).is_relative_to('runtime/framework'), 'isolated actual authorization metadata path required')
    require(not any((root/name).exists() for name in (SETUP, core.RUNTIME, core.OUTPUT,
        'runtime/framework/ordered_composite_TEST_v1043')), 'all new one-time namespaces must be fresh')
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
    require(prior.get('schema') == parent_core.CONTROL_SCHEMA
        and prior_lock.get('schema') == parent_core.LOCK_SCHEMA
        and prior_lock.get('approved_for_execution') is True, 'actual original1010 control/lock required')
    for name in ('suite', 'suite_lock', 'source_inputs', 'validation_release'):
        require(prior.get(name) == pointer(dict(suite=SUITE, suite_lock=SUITE_LOCK,
            source_inputs=SOURCE_INPUTS, validation_release=VAL_RELEASE)[name]), 'original915 source scope changed: '+name)
    original = {name:ledger.json(prior['original_'+name]['path'], expected=prior['original_'+name]['sha256'])
        for name in ('launch', 'plan', 'status', 'terminal')}
    older = {name:ledger.json(prior['previous_'+name]['path'],expected=prior['previous_'+name]['sha256'])
        for name in ('launch','plan','status','terminal')}
    grandprior=ledger.json(prior['previous_resume_control']['path'],expected=prior['previous_resume_control']['sha256'])
    grandlock=ledger.json(prior['previous_resume_control_lock']['path'],expected=prior['previous_resume_control_lock']['sha256'])
    require(grandprior.get('schema')==grandparent_core.CONTROL_SCHEMA
        and grandlock.get('schema')==grandparent_core.LOCK_SCHEMA and grandlock.get('approved_for_execution') is True,
        'actual nested955 control and frozen lock required')
    require(all(prior.get('original_'+k)==grandprior.get('original_'+k) for k in ('launch','plan','status','terminal')),
        'all original753 metadata pointers must remain exact through1010')
    middle = {name:ledger.json(grandprior['previous_'+name]['path'],expected=grandprior['previous_'+name]['sha256'])
        for name in ('launch','plan','status','terminal')}
    ancestor_control=ledger.json(grandprior['previous_resume_control']['path'],expected=grandprior['previous_resume_control']['sha256'])
    ancestor_lock=ledger.json(grandprior['previous_resume_control_lock']['path'],expected=grandprior['previous_resume_control_lock']['sha256'])
    require(ancestor_control.get('schema')==old_audit.owner.CONTROL_SCHEMA
        and ancestor_lock.get('approved_for_execution') is True and ancestor_lock.get('schema')==old_audit.owner.LOCK_SCHEMA,
        'actual nested915 parent control and lock required')
    old_audit.require_resume_plan(ancestor_control,original['plan'],original['status'],original['terminal'],middle['plan'])
    grandparent_core.require_resume_plan(grandprior,middle['plan'],middle['status'],middle['terminal'],older['plan'])
    previous = {name:read(PREVIOUS+'/'+name+'.json') for name in ('launch','plan','status','terminal')}
    previous.update({name:read(PREVIOUS_CONTROLLER+'/'+name.removeprefix('controller_')+'.json')
        for name in ('controller_launch','controller_status','controller_failure')})
    previous_proof = require_previous_exit(previous,owner.alive)
    part = partition(previous['plan'],previous['status'],previous['terminal'])
    parent_core.require_resume_plan(prior,older['plan'],older['status'],older['terminal'],previous['plan'])
    require(all(previous['status']['completed'].get(k)==v for k,v in older['status']['completed'].items())
        and all(older['status']['completed'].get(k)==v for k,v in middle['status']['completed'].items())
        and all(middle['status']['completed'].get(k)==v for k,v in original['status']['completed'].items()),
        'all historical753/902/956 outcomes must exist literally in1011')
    env = {key:pointer(name) for key,name in base.ENVIRONMENT_PATHS.items()}
    require(all(prior.get(key) == value for key,value in env.items()), 'original820 material pointers unchanged')
    authorization=ledger.json(concurrency_authorization,expected=concurrency_authorization_sha256)
    require(authorization.get('user_authorized_concurrency_change') is True,'actual new concurrency authorization required')
    seed = dict(prior, **part['partition_control'],previous_runtime=PREVIOUS,previous_controller_runtime=PREVIOUS_CONTROLLER,
        suite=pointer(SUITE),suite_lock=pointer(SUITE_LOCK),source_inputs=pointer(SOURCE_INPUTS),validation_release=release_pointer,
        resource_profile=selected['resource_profile'],resource_capacity_envelope=selected['resource_capacity_envelope'],
        resource_capacity_source=selected['resource_capacity_source'],resource_profile_within_capacity=True,
        resource_qualification=ledger.read(selected['qualification'],expected=resource_qualification_sha256),
        resource_qualifier=pointer(selected['qualifier']),previous_resource_profile=deepcopy(prior['resource_profile']),
        previous_resource_qualification=deepcopy(prior['resource_qualification']),
        resource_profile_changed=selected['resource_profile']!=prior['resource_profile'],
        concurrency_authorization_record=pointer(concurrency_authorization),**env)
    recheck = ledger.json(ENVIRONMENT_RECHECK, expected=environment_recheck_sha256)
    core.require_environment_recheck(recheck, seed)
    ledger.read(recheck['source']['path'], expected=recheck['source']['sha256'])
    for value, source in ((previous['launch'], 'scripts/run_composite_TEST_owner_v1011_r10.py'),
            (previous['controller_launch'], 'scripts/continue_composite_TEST_v1013_r10.py')):
        require(value.get('source_sha256') == pointer(source)['sha256']
            and value.get('resume_control') == prior_pointer and value.get('resume_control_lock') == prior_lock_pointer,
            'historical1011/1013 source/control bytes differ')
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
    require(seed['execution_entry'] == prior['execution_entry'] and seed['resource_profile'] == selected['resource_profile']
        and seed['execution_implementation'] == owner.execution_implementation(seed['execution_entry'], shared),
        'same physical901 implementation and explicitly qualified capacity-bounded actual resource tier and old915 formal controls required')
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
        and qualification['actual_independent_audit']['path'] == selected['qualification_launch'],
        'exact actual independent904 launch pointer required before opening proof dependencies')
    audit_launch = ledger.json(qualification['actual_independent_audit']['path'],
        expected=qualification['actual_independent_audit']['sha256'])
    require_resource_proof(qualification, seed, audit_launch=audit_launch)
    for name,checksum in qualification['inputs'].items():
        require(qualification_path_allowed(name,capacity_source=resource_capacity_source), 'noncanary qualification payload refused')
        ledger.read(name, expected=checksum)
    for key in ('control', 'lock'):
        ledger.read(qualification[key]['path'], expected=qualification[key]['sha256'])
    files = merge_maps(prior_lock['files'], suite_files, source_files)
    profile = fingerprint(proof['routes']['common_author_RML_profile']); successes = 0
    stage_owners = {core.ORIGINAL:original['launch'],core.INTERMEDIATE:middle['launch'],core.OLDER:older['launch'],PREVIOUS:previous['launch']}
    for key in part['completed']:
        require(not stopped(root), 'STOP during historical evidence freeze')
        row = part['jobs'][key]
        metrics = freeze_completed(row, previous['status']['completed'][key], ledger, seed,
            stage_owners[row['evidence_runtime']], shared, profile)
        successes += int(metrics['native_success'])
    require(previous['terminal'].get('native_successes') == successes, 'historical native total differs')
    failed_inventory = {key:failed_attempt_inventory(root, part['jobs'][key], ledger, previous['status']['failed'][key])
        for key in part['failed']}
    standing = read(base.STANDING); authorization = ledger.json(concurrency_authorization,expected=concurrency_authorization_sha256)
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
            control['original_'+name] = deepcopy(prior['original_'+name])
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
    parser.add_argument('--resource-qualification-sha256');parser.add_argument('--concurrency-authorization-sha256')
    parser.add_argument('--concurrency-authorization',default=AUTHORIZATION)
    parser.add_argument('--resource-capacity-source',choices=('qualified1031','qualified1005_fallback'),default='qualified1031')
    parser.add_argument('--max-native-slots',type=int);parser.add_argument('--max-GPU-slots',type=int)
    parser.add_argument('--max-CPU-preflight-slots',type=int);parser.add_argument('--ready-limit',type=int)
    args = parser.parse_args(argv)
    if not args.prepare:
        print(json.dumps(dict(execution_enabled=False, native_started=False, GPU_started=False,
            message='inactive: no inputs inspected and no execution materials created')))
        return
    quotas=(args.max_native_slots,args.max_GPU_slots,args.max_CPU_preflight_slots,args.ready_limit)
    require(all(v is None for v in quotas) or all(v is not None for v in quotas),'all four actual quota overrides must be explicit together')
    actual=None if all(v is None for v in quotas) else dict(schema=PROFILE['schema'] if args.resource_capacity_source=='qualified1031' else FALLBACK_PROFILE['schema'],
        **dict(zip(('max_native_slots','max_GPU_slots','max_CPU_preflight_slots','ready_limit'),quotas)))
    print(json.dumps(prepare(environment_recheck_sha256=args.environment_recheck_sha256,resource_qualification_sha256=args.resource_qualification_sha256,
        concurrency_authorization_sha256=args.concurrency_authorization_sha256,concurrency_authorization=args.concurrency_authorization,
        resource_capacity_source=args.resource_capacity_source,resource_profile=actual), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
