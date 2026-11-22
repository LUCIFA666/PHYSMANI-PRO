"""Inactive metadata-only observer for explicit907/902/906 TEST continuation.

It never schedules work, reads case/raw/model content, verifies the large input
graph, restores an OS namespace, sends a chat message, or acknowledges delivery.
An externally pinned small source lock selects one immutable continuation.
"""
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import report_formal_TEST_halfhour_v830_r10 as legacy
SOURCE = 'scripts/report_concurrent_TEST_halfhour_v909_r10.py'
STATE = 'runtime/framework/concurrent_TEST_report_v909'
OUT = 'TEST_结果更新_并发版.txt'
SCHEMA = 'r10_readonly_concurrent_TEST_report_source_v909'
LOCK_SCHEMA = 'r10_readonly_concurrent_TEST_report_inputs_v909'
RUNTIME = 'runtime/framework/formal_TEST_concurrent_resume_v902'
CONTROLLER = 'runtime/framework/ordered_concurrent_TEST_v907'
OLD_RUNTIME = 'runtime/framework/formal_TEST_after_environment_resume_v753'
OLD_CONTROLLER = 'runtime/framework/ordered_env_resume_VAL_TEST_v827'
OLD_REPORT = 'runtime/framework/formal_TEST_report_v830'
SETUP = 'artifacts/framework/multitask_r10/formal_TEST_concurrent_resume_v902_setup'
CONTROL, CONTROL_LOCK = SETUP + '/resume_control.json', SETUP + '/resume_inputs.json'
RESULTS = 'artifacts/framework/multitask_r10/formal_TEST_concurrent_resume_v902/TEST_results.json'
VAL_RELEASE = 'artifacts/framework/multitask_r10/formal_VAL_env_resume_v823/VAL_release.json'
SOURCE_INPUTS = 'artifacts/framework/multitask_r10/formal_TEST_after_environment_resume_v826_setup/test_source_inputs.json'
QUALIFICATION = 'runtime/framework/concurrent_TRAIN_resource_qualification_v904.json'
OWNER_SOURCE = 'scripts/run_concurrent_resumed_formal_TEST_owner_v902_r10.py'
AUDITOR_SOURCE = 'scripts/audit_concurrent_resumed_formal_TEST_v906_r10.py'
CONTROLLER_SOURCE = 'scripts/continue_concurrent_TEST_v907_r10.py'
QUALIFIER_SOURCE = 'scripts/audit_concurrent_TRAIN_resource_canary_v904_r10.py'
FIXED_METADATA = dict(environment_wrapper='scripts/run_frozen_system_environment_v820_r10.py',
    environment_manifest='artifacts/framework/multitask_r10/frozen_environment_restore_v820_setup/hybrid_environment_manifest_v820.json',
    environment_namespace_proof='artifacts/framework/multitask_r10/frozen_environment_restore_v820_setup/cpu_namespace_proof_v820.json',
    standing_continuation_record='runtime/framework/latest_executor_continuation_v817.json',
    concurrency_authorization_record='runtime/framework/user_concurrency_authorization_v905.json')
SOURCES = (SOURCE, 'scripts/report_formal_TEST_halfhour_v830_r10.py', OWNER_SOURCE, AUDITOR_SOURCE,
    CONTROLLER_SOURCE, QUALIFIER_SOURCE, 'scripts/audit_environment_resumed_formal_VAL_v824_r10.py',
    'scripts/prepare_formal_split_source_lock_v756_r10.py', 'our_mani/multitask/native_outcome_sampling_v741.py',
    'scripts/run_task_executor_smoke_author_suite16_v744_r10.py', 'scripts/run_task_executor_smoke_rml_suite16_v664_r10.py',
    'scripts/run_formal_suite16_owner_v753_r10.py')
METHODS, LABELS, TASK_LABELS = legacy.METHODS, legacy.LABELS, legacy.TASK_LABELS
PROFILE = dict(schema='r10_frozen_concurrent_native_resource_profile_v902',
    max_native_slots=4, max_GPU_slots=2, max_CPU_preflight_slots=2, ready_limit=8)
ROW_FIELDS = {'evidence_runtime', 'imported_completed', 'previous_row_sha256', 'execution_entry', 'physical_argv', 'previous_worker_sha256'}
STOP_PATHS = ('runtime/framework/multitask_r10_STOP', STATE + '/STOP', CONTROLLER + '/STOP', RUNTIME + '/STOP',
    OLD_RUNTIME + '/STOP', OLD_CONTROLLER + '/STOP', 'runtime/framework/ordered_resume_VAL_TEST_v775/STOP',
    'runtime/framework/formal_VAL_resume_v772/STOP', 'runtime/framework/formal_VAL_env_resume_v823/STOP')


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def digest(value):
    return sha(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode())


def valid_pointer(value, name=None):
    return (isinstance(value, dict) and set(value) == {'path', 'sha256'} and isinstance(value.get('path'), str)
        and bool(value['path']) and not Path(value['path']).is_absolute() and '..' not in Path(value['path']).parts
        and isinstance(value.get('sha256'), str) and len(value['sha256']) == 64
        and not set(value['sha256']) - set('0123456789abcdef') and (name is None or value['path'] == name))


def base_allowed():
    names = {*(SOURCES), CONTROL, CONTROL_LOCK, VAL_RELEASE, SOURCE_INPUTS, QUALIFICATION,
        RESULTS, STATE + '/source.json', STATE + '/source_inputs.json',
        *(OLD_REPORT + '/' + n for n in ('status.json', 'chat_ready.json', 'history.jsonl')),
        *(OLD_RUNTIME + '/' + n for n in ('launch.json', 'plan.json', 'status.json', 'terminal.json', 'DRAIN')),
        *(OLD_CONTROLLER + '/' + n for n in ('launch.json', 'status.json', 'failure.json')),
        *(RUNTIME + '/' + n for n in ('launch.json', 'plan.json', 'status.json', 'terminal.json')),
        *(CONTROLLER + '/' + n for n in ('launch.json', 'status.json', 'terminal.json', 'failure.json'))}
    for stage in ('formal_TEST_1920', 'independent_TEST_results'):
        names.update(CONTROLLER + '/stages/' + stage + '/' + n for n in ('launch.json', 'terminal.json', 'stdout.log'))
    return names


class Metadata:
    def __init__(self, *, stop=lambda: False, read_bytes=None):
        self.stop, self.callback, self.allowed = stop, read_bytes, base_allowed()
        self.raw, self.checksums = {}, {}

    def permit_pointer(self, pointer):
        require(valid_pointer(pointer), 'explicit project metadata pointer required')
        name = pointer['path']
        require(name in FIXED_METADATA.values(), 'only exact fixed control/environment metadata permitted')
        self.allowed.add(name)

    def bytes(self, name, expected=None, *, missing=False):
        if self.stop():
            raise legacy.ObservationStopped('STOP before metadata observation')
        require(isinstance(name, str) and name in self.allowed, 'metadata read outside reporter allowlist: ' + str(name))
        if name not in self.raw:
            try:
                if self.callback is not None:
                    raw = self.callback(name)
                else:
                    path = ROOT / name
                    require(path.resolve(strict=True) == path.absolute(), 'metadata aliases are refused')
                    raw = path.read_bytes()
            except FileNotFoundError:
                if missing:
                    return None
                raise
            require(type(raw) is bytes, 'metadata reader must return exact bytes')
            self.raw[name], self.checksums[name] = raw, sha(raw)
        require(expected is None or self.checksums[name] == expected, 'frozen metadata SHA changed: ' + name)
        return self.raw[name]

    def json(self, name, expected=None, *, missing=False):
        raw = self.bytes(name, expected, missing=missing)
        if raw is None:
            return {}
        value = json.loads(raw)
        require(isinstance(value, dict), 'metadata JSON object required')
        return value

    def pointer(self, value, name=None):
        require(valid_pointer(value, name), 'exact frozen metadata pointer required')
        return self.json(value['path'], value['sha256'])


def source_bundle(config, frozen, *, config_sha, lock_sha, reader):
    require(config.get('schema') == SCHEMA and config.get('protocol_status') == 'formal_TEST_concurrent', 'explicit909 source required')
    for key, value in (('runtime', RUNTIME), ('controller', CONTROLLER), ('test_results', RESULTS)):
        require(config.get(key) == value, 'reporter cannot switch owner/controller/output: ' + key)
    exact = dict(resume_control=CONTROL, resume_control_lock=CONTROL_LOCK, validation_release=VAL_RELEASE,
        test_source_inputs=SOURCE_INPUTS, resource_qualification=QUALIFICATION, resource_qualifier=QUALIFIER_SOURCE)
    for key, name in exact.items():
        require(valid_pointer(config.get(key), name), 'exact source binding required: ' + key)
    require(frozen.get('schema') == LOCK_SCHEMA and frozen.get('approved_for_observation') is True, 'externally pinned observer input lock required')
    files, sources = frozen.get('files'), config.get('source_bindings')
    require(isinstance(files, dict) and isinstance(sources, dict) and set(sources) == set(SOURCES), 'complete small source/control binding map required')
    for name, pointer in sources.items():
        require(valid_pointer(pointer, name), 'exact reporter dependency pointer required')
        reader.bytes(name, pointer['sha256'])
    for key in ('environment_wrapper', 'environment_manifest', 'environment_namespace_proof', 'standing_continuation_record', 'concurrency_authorization_record'):
        require(valid_pointer(config.get(key),FIXED_METADATA[key]),'wrong fixed metadata source: '+key)
        reader.permit_pointer(config.get(key))
    legacy_pointers = config.get('legacy_schedule')
    require(isinstance(legacy_pointers, dict) and set(legacy_pointers) == {'status', 'chat_ready', 'history'}, 'frozen830 schedule/history provenance required')
    for key, filename in (('status','status.json'),('chat_ready','chat_ready.json'),('history','history.jsonl')):
        require(valid_pointer(legacy_pointers[key], OLD_REPORT + '/' + filename), 'only original830 deadline/history may seed this observer')
    canonical = {}
    immutable={*SOURCES,CONTROL,CONTROL_LOCK,VAL_RELEASE,SOURCE_INPUTS,QUALIFICATION,STATE+'/source.json',
        *FIXED_METADATA.values(),*(p['path'] for p in legacy_pointers.values())}
    for name, checksum in files.items():
        path = Path(name)
        if path.is_absolute():
            require(path.is_relative_to(ROOT), 'observer lock cannot scan external inputs')
            name = str(path.relative_to(ROOT))
        require(name in immutable and name not in canonical and valid_pointer(dict(path=name,sha256=checksum)),
                'unknown/duplicate/non-SHA or dynamic observer input cannot be prefetched')
        canonical[name] = checksum
    require(canonical.get(STATE + '/source.json') == config_sha, 'observer config must be frozen by its small lock')
    # This is an observer configuration check, not verify_inputs. TEST source
    # lock/status/result bytes are consumed only after the actual VAL gate.
    for name in (STATE+'/source.json',*SOURCES):
        reader.bytes(name,canonical[name])
    required_pointers = [*sources.values(), *legacy_pointers.values(), *(config[k] for k in exact),
        *(config[k] for k in ('environment_wrapper','environment_manifest','environment_namespace_proof','standing_continuation_record','concurrency_authorization_record'))]
    require(all(canonical.get(p['path']) == p['sha256'] for p in required_pointers), 'observer lock omits a required explicit source/control pointer')
    require(valid_pointer(dict(path=CONTROL,sha256=config.get('policy_sha256')))
        and isinstance(config.get('author_outcome_protocol'), dict), 'fixed policy/protocol required')
    protocol = config['author_outcome_protocol']
    require(protocol == dict(schema='r10_author_action_outcome_protocol_v741', implementation=sources[SOURCES[8]],
        runtime_consumer=sources[SOURCES[9]], original_runtime=sources[SOURCES[10]],
        failure_sampling='current_native_result_at_explicit_action_end', success_sampling='original_per_tick_success_latch',
        pose_action_max_ticks=10, public_hold_max_ticks=1, budget_partial_action_consumed=True,
        native_RML_parameters_unchanged=True, block_native_fail_conditions=0, block_extra_task_success_calls=0), 'unchanged author-outcome protocol required')
    return dict(config=config, config_sha256=config_sha, lock_sha256=lock_sha, frozen=canonical)


def load_source(expected_lock_sha, *, stop=lambda: False, read_bytes=None):
    reader = Metadata(stop=stop, read_bytes=read_bytes)
    raw = reader.bytes(STATE + '/source_inputs.json', expected_lock_sha)
    config_raw = reader.bytes(STATE + '/source.json')
    return source_bundle(json.loads(config_raw), json.loads(raw), config_sha=sha(config_raw), lock_sha=sha(raw), reader=reader)


def schedule_seed(config, reader):
    pointers = config['legacy_schedule']
    for key in ('chat_ready', 'status'):
        value = reader.pointer(pointers[key]).get('next_update_utc')
        if value:
            return legacy.parse_utc(value)
    raw = reader.bytes(pointers['history']['path'], pointers['history']['sha256'])
    for line in reversed(raw.decode().splitlines()):
        try:
            value = json.loads(line).get('next_update_utc')
            if value:
                return legacy.parse_utc(value)
        except (ValueError, AttributeError):
            continue
    raise ValueError('original830 has no valid recorded UTC deadline; never reset to now+1800')


def plan_grid(plan):
    require(plan.get('split') == 'test' and plan.get('expected_cases') == 1920, 'complete TEST plan required')
    rows = plan.get('jobs')
    require(isinstance(rows, list) and len(rows) == 1920, 'original640 sources times3 plan required')
    by_id, groups = {}, {}
    for row in rows:
        require(isinstance(row, dict) and row.get('task') in TASK_LABELS and row.get('method') in METHODS
            and isinstance(row.get('raw'), str) and row['raw'] and isinstance(row.get('job_id'), str)
            and row['job_id'] and row['job_id'] not in by_id, 'unknown/duplicate TEST plan row')
        by_id[row['job_id']] = row
        groups.setdefault((row['task'],row['method']), []).append(row['raw'])
    require(len(groups) == 48 and all(len(raws) == len(set(raws)) == 40 for raws in groups.values()), 'each task/method needs40 unique original sources')
    for task in TASK_LABELS:
        require(all(set(groups[task,m]) == set(groups[task,METHODS[0]]) for m in METHODS), 'three methods need exactly the same40 source IDs')
    return by_id


def plan_provenance(plan, old_plan, control):
    jobs, old = plan_grid(plan), plan_grid(old_plan)
    require(set(jobs) == set(old), 'no key padding/skipping across continuation')
    imported = control.get('completed_job_ids'); remaining = control.get('remaining_job_ids')
    require(isinstance(imported,list) and isinstance(remaining,list) and len(imported)==len(set(imported))
        and len(remaining)==len(set(remaining)) and set(imported).isdisjoint(remaining)
        and set(imported)|set(remaining)==set(jobs), 'exact old/new key partition required')
    for key,row in jobs.items():
        previous=old[key];adopted=key in imported
        require(row.get('previous_row_sha256')==digest(previous) and row.get('previous_worker_sha256')==previous['worker_sha256']
            and row.get('imported_completed') is adopted and row.get('evidence_runtime')==(OLD_RUNTIME if adopted else RUNTIME), 'real old/new provenance differs')
        expected=deepcopy(previous)
        expected.update(evidence_runtime=OLD_RUNTIME if adopted else RUNTIME,imported_completed=adopted,
            previous_row_sha256=digest(previous),previous_worker_sha256=previous['worker_sha256'])
        if adopted:
            expected.update(physical_argv=deepcopy(previous['argv']),execution_entry=dict(kind='original751',pointer=previous['admission_driver']))
        else:
            inner=deepcopy(previous['argv']);index=inner.index('--out')+1
            inner[index]=str(Path('artifacts/framework/multitask_r10/formal_TEST_concurrent_resume_v902/test')/row['task']/row['raw']/row['method'])
            entry=control['execution_entry'];argv=deepcopy(inner)
            if entry['kind']=='batched901':
                argv=[*inner[:2],entry['pointer']['path'],'--batch-control',entry['batch_control']['path'],
                    '--batch-lock',entry['batch_lock']['path'],'--batch-lock-sha256',entry['batch_lock']['sha256'],'--entry','formal','--',*inner[3:]]
            expected.update(physical_argv=inner,argv=argv,execution_entry=entry,worker_sha256=entry['pointer']['sha256'])
        require(row==expected, 'full plan row changed beyond explicit continuation metadata/output')
    return jobs,set(imported)


def count_status(status,jobs,old_status,imported):
    require(status.get('split')=='test' and status.get('expected')==1920 and isinstance(status.get('completed'),dict)
        and isinstance(status.get('failed'),dict), 'actual1920 TEST status required')
    records=status['completed'];old=old_status.get('completed',{})
    require(set(old)==imported and imported<=set(records) and all(records[k]==old[k] for k in imported), 'all imported outcomes must remain literal old records')
    complete,success=Counter(),Counter()
    for key,value in records.items():
        require(key in jobs and isinstance(value,dict) and value.get('complete') is True
            and type(value.get('native_success')) is bool and value.get('interrupted') is False
            and value.get('timed_out') is False and type(value.get('returncode')) is int and value['returncode']==0
            and value.get('automatic_retry') is False and value.get('error') is None
            and value.get('leaf_future_exception') is None, 'only clean explicit native outcomes may be counted')
        inner=jobs[key]['physical_argv'];out=ROOT/inner[inner.index('--out')+1]
        for field,filename in (('case','case.json'),('admission_driver_receipt','admission_driver_receipt.json')):
            require(valid_pointer(value.get(field),str((out/filename).resolve().relative_to(ROOT))), 'completed record evidence path differs from its exact plan output')
        group=jobs[key]['task'],jobs[key]['method'];complete[group]+=1;success[group]+=int(value['native_success'])
    failed=status['failed'];active=status.get('active',[]);checking=status.get('checking',[])
    require(set(failed)<=set(jobs) and set(failed).isdisjoint(records), 'unknown/overlapping infrastructure failure keys')
    require(all(isinstance(x,list) and len(x)==len(set(x)) and set(x)<=set(jobs) for x in (active,checking))
        and set(active).isdisjoint(records) and set(checking).isdisjoint(records) and set(active).isdisjoint(checking), 'active/checking keys overlap or are unknown')
    require(len(active)<=4 and sum(jobs[k]['method']=='physmani' for k in active)<=2 and len(checking)<=2, 'status exceeds the qualified4/2/2 profile')
    return dict(completed=complete,success=success,failed=len(failed),imported_completed=len(imported),new_completed=len(records)-len(imported),status=status)


def common(value,config):
    require(value.get('policy_sha256')==config['policy_sha256'] and value.get('author_outcome_protocol')==config['author_outcome_protocol'], 'policy/protocol binding changed')


def stage_inner(launch,source):
    argv=launch.get('argv')
    require(isinstance(argv,list) and argv[:4]==[str(ROOT/'.conda-env/bin/python'),'-B','-u',source],
            'actual907 child must use the frozen Python/-B/-u/source argv prefix')
    return argv[3:]


def qualification_gate(value,control):
    require(value.get('schema')=='r10_concurrent_TRAIN_resource_qualification_v902' and value.get('complete') is True
        and value.get('split')=='train' and value.get('resource_profile')==PROFILE
        and value.get('execution_entry')==control['execution_implementation'] and value.get('source_sha256')==control['resource_qualifier']['sha256']
        and type(value.get('case_count')) is int and value['case_count']==6
        and type(value.get('infrastructure_failed')) is int and value['infrastructure_failed']==0
        and all(value.get(k) is True for k in ('cleanup_completed','resource_bounds_verified','overlap_verified','independent_audit_complete'))
        and value.get('observed_native_peak')==4 and type(value.get('observed_native_peak')) is int
        and value.get('observed_GPU_peak')==2 and type(value.get('observed_GPU_peak')) is int
        and value.get('formal_score_contribution') is False and value.get('TEST_read') is False, 'real independent six-TRAIN-case resource qualification required')
    common(value,control)


def collect(bundle,*,stop=lambda:False,read_bytes=None):
    reader=Metadata(stop=stop,read_bytes=read_bytes);config=bundle['config']
    for key in ('environment_wrapper','environment_manifest','environment_namespace_proof','standing_continuation_record','concurrency_authorization_record'):
        reader.permit_pointer(config[key])
    reader.bytes(STATE+'/source_inputs.json',bundle['lock_sha256'])
    for name in (STATE+'/source.json',*SOURCES):reader.bytes(name,bundle['frozen'][name])
    release=reader.pointer(config['validation_release'],VAL_RELEASE)
    legacy._require_shape(release,split='validation',sources=315,cases=945,policy=config['policy_sha256'])
    require(release.get('formal_test_permitted') is True and release.get('TEST_used') is False
        and release.get('schema')=='r10_full16_common_author_RML_validation_release_v531'
        and len(release.get('cases',[]))==len(release.get('admission_driver_receipts',[]))==945
        and release.get('source_sha256')==config['source_bindings'][SOURCES[6]]['sha256'], 'actual complete824 VAL release required')
    common(release,config)
    control=reader.pointer(config['resume_control'],CONTROL);lock=reader.pointer(config['resume_control_lock'],CONTROL_LOCK)
    require(control.get('schema')=='r10_concurrent_resumed_formal_TEST_control_v902' and control.get('expected_cases')==1920
        and control.get('source_count')==640 and control.get('split')=='test' and control.get('resource_profile')==PROFILE
        and control.get('physical_policy_changed') is False and control.get('user_authorized_resume') is True
        and control.get('standing_authorization_reused') is True and control.get('new_user_event_invented') is False,
        'one explicitly authorized unchanged TEST continuation required')
    common(control,config)
    require(lock.get('schema')=='r10_concurrent_resumed_formal_TEST_inputs_v902' and lock.get('approved_for_execution') is True
        and isinstance(lock.get('files'),dict), 'actual frozen902 input-lock metadata required')
    for key in ('resource_qualification','resource_qualifier','standing_continuation_record','concurrency_authorization_record',
                'environment_wrapper','environment_manifest','environment_namespace_proof'):
        require(control.get(key)==config[key], 'source config/control binding differs: '+key)
    require(control.get('resume_controller')==config['source_bindings'][CONTROLLER_SOURCE]
        and control.get('resume_owner')==config['source_bindings'][OWNER_SOURCE]
        and control.get('resume_auditor')==config['source_bindings'][AUDITOR_SOURCE]
        and control.get('validation_release')==config['validation_release'] and control.get('source_inputs')==config['test_source_inputs'], 'actual907/902/906 source chain required')
    for pointer in (config['resume_control'],*(config['source_bindings'][p] for p in (CONTROLLER_SOURCE,OWNER_SOURCE,AUDITOR_SOURCE))):
        files=lock['files'];require(files.get(str(ROOT/pointer['path']),files.get(pointer['path']))==pointer['sha256'], 'execution lock omits actual source/control')
    standing=reader.pointer(config['standing_continuation_record']);authorization=reader.pointer(config['concurrency_authorization_record'])
    require(standing.get('continue_until_user_stop') is True and standing.get('user_stop_received') is False
        and authorization.get('user_authorized_concurrency_change') is True,'standing continuation or concurrency authorization revoked')
    qualification_gate(reader.pointer(config['resource_qualification'],QUALIFICATION),control)
    launch=reader.json(CONTROLLER+'/launch.json');controller=reader.json(CONTROLLER+'/status.json')
    require(launch.get('source_sha256')==config['source_bindings'][CONTROLLER_SOURCE]['sha256']
        and launch.get('resume_control')==config['resume_control'] and launch.get('resume_control_lock')==config['resume_control_lock'], 'actual907 controller binding changed')
    environment=legacy._recorded_environment(launch.get('frozen_environment'),config,owner_pid=launch.get('pid'))
    require(environment['observer_start_ticks']==launch.get('start_ticks'), '907 recorded environment PID/start differs from its actual launch')
    require(controller.get('automatic_retry') is False and controller.get('frozen_environment')==environment
        and controller.get('stage') in ('formal_TEST_1920','independent_TEST_results','complete','draining_for_concrete_repair','requires_concrete_repair','user_stopped'), 'actual907 stage/environment required')
    source_lock=reader.pointer(config['test_source_inputs'],SOURCE_INPUTS)
    require(source_lock.get('schema')=='r10_full16_post_admission_split_source_lock_v1' and source_lock.get('split')=='test'
        and type(source_lock.get('source_count')) is int and source_lock['source_count']==640
        and source_lock.get('policy_sha256')==config['policy_sha256'] and source_lock.get('TEST_used_for_fitting') is False
        and source_lock.get('source_content_used_for_policy_fitting') is False and source_lock.get('complete_VAL_release_observed') is True
        and source_lock.get('source_lock_preparer_sha256')==config['source_bindings'][SOURCES[7]]['sha256'], 'original756640-source TEST metadata required')
    owner_launch=reader.json(RUNTIME+'/launch.json');plan=reader.json(RUNTIME+'/plan.json');status=reader.json(RUNTIME+'/status.json')
    common(owner_launch,config);common(plan,config);common(status,config)
    require(owner_launch.get('source_sha256')==config['source_bindings'][OWNER_SOURCE]['sha256']
        and owner_launch.get('split_source_lock_sha256')==config['test_source_inputs']['sha256']
        and owner_launch.get('source_manifest',{}).get('sha256')==source_lock.get('source_manifest_sha256')
        and owner_launch.get('resource_profile')==PROFILE and owner_launch.get('resume_control')==config['resume_control']
        and owner_launch.get('resume_control_lock')==config['resume_control_lock'], 'actual902 launch required')
    legacy._recorded_environment(owner_launch.get('frozen_environment'),config,owner_pid=owner_launch.get('pid'),expected=environment)
    require(owner_launch['frozen_environment']['current_mount_namespace']==environment['current_mount_namespace']
        and owner_launch['frozen_environment']['observer_start_ticks']==owner_launch.get('start_ticks'), '902 must inherit the actual907 mount namespace with its actual PID/start')
    owner_stage=reader.json(CONTROLLER+'/stages/formal_TEST_1920/launch.json')
    require(owner_stage.get('pid')==owner_launch.get('pid') and owner_stage.get('start_ticks')==owner_launch.get('start_ticks')
        and owner_stage.get('source_sha256')==config['source_bindings'][OWNER_SOURCE]['sha256']
        and owner_stage.get('inherited_frozen_environment')==environment
        and owner_stage.get('current_mount_namespace')==environment['current_mount_namespace']
        and stage_inner(owner_stage,OWNER_SOURCE)==owner_launch.get('argv'), '902 owner must be the actual907 launched child')
    old_plan=reader.pointer(control['previous_plan'],OLD_RUNTIME+'/plan.json');old_status=reader.pointer(control['previous_status'],OLD_RUNTIME+'/status.json')
    common(old_plan,config)
    jobs,imported=plan_provenance(plan,old_plan,control);data=count_status(status,jobs,old_status,imported)
    data.update(admitted=True,controller=controller,plan_valid=True,release_complete=False,errors=[],reader=reader,config=config)
    if status.get('owner_error') or status.get('stop_requested') or status.get('drain_requested') or data['failed']:
        data['errors'].append('新owner存在停止／排空／基础设施异常，仅报告部分进度')
    if controller['stage'] in ('draining_for_concrete_repair','requires_concrete_repair','user_stopped'):
        data['errors'].append('新907控制器正在停止／排空或等待修复，仅报告部分进度')
    failure=reader.json(CONTROLLER+'/failure.json',missing=True)
    if failure:data['errors'].append('新907控制器记录失败，未完成最终比较')
    terminal=reader.json(RUNTIME+'/terminal.json',missing=True)
    if len(status['completed'])==1920 and not data['errors'] and controller['stage']=='complete':
        data['release_complete']=final_gate(data,jobs,terminal,owner_launch,control,reader)
    return data


def final_gate(data,jobs,terminal,owner_launch,control,reader):
    config=data['config'];common(terminal,config)
    require(terminal.get('complete') is True and terminal.get('expected')==terminal.get('completed')==1920
        and terminal.get('failed')==0 and terminal.get('stop_requested') is False and terminal.get('automatic_retry') is False
        and terminal.get('CPU_ready_unexecuted')==[] and terminal.get('CPU_not_launched')==0
        and terminal.get('owner_error') is None and terminal.get('drain_requested') is False and terminal.get('drained') is False
        and terminal.get('imported_completed_cases')==data['imported_completed'] and terminal.get('resumed_new_cases')==data['new_completed']
        and data['status'].get('active')==[] and data['status'].get('checking')==[] and data['status'].get('ready_jobs')==[]
        and data['status'].get('CPU_ready')==0 and data['status'].get('native_slots')==data['status'].get('GPU_slots')==0,
        'full clean902 terminal required before final reporting')
    ctl_terminal=reader.json(CONTROLLER+'/terminal.json',missing=True)
    if not ctl_terminal:return False
    require(ctl_terminal.get('complete') is True and ctl_terminal.get('automatic_retry') is False
        and valid_pointer(ctl_terminal.get('TEST_results'),RESULTS), 'actual907 final result pointer required')
    leaf=CONTROLLER+'/stages/independent_TEST_results'
    launch=reader.json(leaf+'/launch.json');ended=reader.json(leaf+'/terminal.json')
    require(ended.get('complete') is True and type(ended.get('returncode')) is int and ended['returncode']==0
        and ended.get('automatic_retry') is False and launch.get('source_sha256')==config['source_bindings'][AUDITOR_SOURCE]['sha256']
        and launch.get('inherited_frozen_environment')==data['controller']['frozen_environment']
        and launch.get('current_mount_namespace')==data['controller']['frozen_environment']['current_mount_namespace'], 'actual clean906 audit stage required')
    stdout=reader.json(leaf+'/stdout.log')
    require(stdout.get('complete') is True and stdout.get('split')=='test' and stdout.get('cases')==1920 and stdout.get('out')==RESULTS,'actual906 full audit stdout required')
    result=reader.pointer(ctl_terminal['TEST_results'],RESULTS)
    legacy._require_shape(result,split='test',sources=640,cases=1920,policy=config['policy_sha256']);common(result,config)
    require(result.get('schema')=='r10_composite_concurrent_resumed_formal_TEST_results_v906'
        and result.get('source_sha256')==config['source_bindings'][AUDITOR_SOURCE]['sha256']
        and result.get('resume_control')==config['resume_control'] and result.get('resume_control_lock')==config['resume_control_lock']
        and result.get('source_input_lock')==config['test_source_inputs'] and result.get('validation_release')==config['validation_release']
        and result.get('resource_profile')==PROFILE and result.get('resource_qualification')==config['resource_qualification']
        and result.get('source_outcome_filtering') is False and result.get('native_failures_retained') is True
        and result.get('Value_NGFF_FASTER_used') is False and result.get('independent_composite_audit') is True
        and len(result.get('cases',[]))==len(result.get('admission_driver_receipts',[]))==1920, 'actual complete906 composite result required')
    invocation=result.get('auditor_invocation',{})
    require(invocation.get('pid')==launch.get('pid') and invocation.get('start_ticks')==launch.get('start_ticks')
        and invocation.get('source')==config['source_bindings'][AUDITOR_SOURCE] and invocation.get('argv')==stage_inner(launch,AUDITOR_SOURCE), '906 invocation must match the real907 child')
    legacy._recorded_environment(result.get('frozen_environment'),config,owner_pid=invocation['pid'],expected=data['controller']['frozen_environment'])
    require(result['frozen_environment']['current_mount_namespace']==data['controller']['frozen_environment']['current_mount_namespace']
        and result['frozen_environment']['observer_start_ticks']==invocation['start_ticks'], '906 must inherit the actual907 mount namespace with its actual PID/start')
    required=(CONTROL,CONTROL_LOCK,VAL_RELEASE,SOURCE_INPUTS,QUALIFICATION,OWNER_SOURCE,AUDITOR_SOURCE,
        *(RUNTIME+'/'+n for n in ('launch.json','plan.json','status.json','terminal.json')),
        *(OLD_RUNTIME+'/'+n for n in ('launch.json','plan.json','status.json','terminal.json')))
    # Read only explicit, already generated metadata; never dereference case pointers.
    for key,name in (('previous_launch','launch.json'),('previous_terminal','terminal.json')):
        reader.pointer(control[key],OLD_RUNTIME+'/'+name)
    inputs=result.get('inputs_sha256',{})
    require(isinstance(inputs,dict) and all(inputs.get(str(ROOT/name),inputs.get(name))==reader.checksums.get(name)
        and name in reader.checksums for name in required), '906 audit must bind these exact old/new metadata bytes')
    old_launch=reader.json(OLD_RUNTIME+'/launch.json');owners=result.get('actual_stage_owners',[])
    require(isinstance(owners,list) and len(owners)==2 and {v.get('runtime') for v in owners}=={OLD_RUNTIME,RUNTIME}, 'exact two real execution owners required')
    for record in owners:
        old=record['runtime']==OLD_RUNTIME;actual=old_launch if old else owner_launch
        expected_launch=control['previous_launch'] if old else dict(path=RUNTIME+'/launch.json',sha256=reader.checksums[RUNTIME+'/launch.json'])
        expected_terminal=control['previous_terminal'] if old else dict(path=RUNTIME+'/terminal.json',sha256=reader.checksums[RUNTIME+'/terminal.json'])
        require(record.get('launch')==expected_launch and record.get('terminal')==expected_terminal
            and record.get('pid')==actual['pid'] and record.get('start_ticks')==actual['start_ticks']
            and record.get('completed')==(data['imported_completed'] if old else data['new_completed']), '906 actual old/new owner identity/count differs')
    tiers=result.get('evidence_tiers',[]);require(len(tiers)==1920 and len({t.get('job_id') for t in tiers})==1920, 'complete exact composite evidence tiers required')
    for index,tier in enumerate(tiers):
        key=tier.get('job_id');require(key in jobs and all(tier.get(k)==jobs[key].get(k) for k in ('evidence_runtime','imported_completed','previous_row_sha256','execution_entry')), 'composite result tier differs from actual plan')
        record=data['status']['completed'][key]
        require(tier.get('case')==record['case']==result['cases'][index]
            and tier.get('admission_driver_receipt')==record['admission_driver_receipt']==result['admission_driver_receipts'][index], '906 case/driver metadata must match the actual counted status rows')
    totals={m:sum(data['success'][t,m] for t in TASK_LABELS) for m in METHODS}
    require(result.get('native_successes_by_method')==totals, '906 totals differ from clean status snapshot')
    per_task=result.get('per_task',[]);require(len(per_task)==16 and {r.get('task') for r in per_task}==set(TASK_LABELS), 'complete16-task906 summary required')
    for row in per_task:
        require(set(row.get('methods',{}))==set(METHODS),'exact three-method906 summary required')
        for method,value in row['methods'].items():
            require(type(value.get('episodes')) is int and value['episodes']==40 and type(value.get('successes')) is int
                and value['successes']==data['success'][row['task'],method],'906 per-task outcomes differ from status')
    return True


def empty_data(error=None):
    return dict(completed=Counter(),success=Counter(),failed=0,imported_completed=0,new_completed=0,
        status={},controller={},admitted=False,plan_valid=False,release_complete=False,errors=[] if error is None else [str(error)])


def save_report(data,bundle,history,*,next_due,reason,stopping=False,now=None):
    now=time.time() if now is None else now;done=stopping or data['release_complete']
    require(done or next_due is not None,'preserve an explicit reporting deadline')
    state=ROOT/STATE;state.mkdir(parents=True,exist_ok=True)
    stamp=datetime.fromtimestamp(now,timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    methods={m:dict(completed=sum(data['completed'][t,m] for t in TASK_LABELS),successes=sum(data['success'][t,m] for t in TASK_LABELS)) for m in METHODS}
    total=sum(v['completed'] for v in methods.values())
    phase='完整1920条TEST已由实际906独立审计及907终态确认' if data['release_complete'] else '部分进度，尚未通过完整906独立审计；非最终比较结果'
    if not data['admitted']:phase='绑定读取门未通过，不推算当前TEST完成量'
    summary=legacy.clock_text(now)+f'｜并发TEST {total}/1920｜infra {data["failed"]}｜'+'；'.join(f'{LABELS[m]} {v["successes"]}/{v["completed"]}' for m,v in methods.items())+'｜'+reason
    lines=['正式TEST结果更新（并发续跑版）',phase,'更新时间：'+legacy.clock_text(now),
        '实际902状态时间（UTC）：'+str(data['status'].get('updated_utc','尚未读取')),
        '下一次更新：'+(legacy.clock_text(next_due) if not done else '本观察器结束'),
        f'完成{total}/1920；原753导入{data["imported_completed"]}；新902完成{data["new_completed"]}；基础设施失败{data["failed"]}',
        '导入与新完成记录按唯一job_id合并，只计一次；nativeFalse保留为任务结果。',
        '三组共同RML参数及原任务成功口径不变；运动6诊断不作成功过滤。',
        '只读元数据和已记录环境证明；不读取case、raw、model或OS输入大图。',
        '当前控制器阶段：'+str(data['controller'].get('stage','未绑定')),'',
        '任务 | CV成功/完成 | Context成功/完成 | PhysMani成功/完成 | 每组应评']
    for task,label in TASK_LABELS.items():lines.append(label+' | '+' | '.join(f'{data["success"][task,m]}/{data["completed"][task,m]}' for m in METHODS)+' | 40')
    lines.extend(['','读取提示：',*data['errors'],'','历史摘要（分别标记830原记录与909并发记录）：',
        *((('[830原记录] ' if h.get('origin')=='original830' else '[909并发记录] ')+h['summary'])
          for h in history if isinstance(h.get('summary'),str)),summary])
    evidence={}
    reader=data.get('reader')
    if reader:
        for name in (RUNTIME+'/status.json',CONTROLLER+'/status.json'):
            if name in reader.raw:
                path=state/'snapshots'/stamp/(('owner' if name.startswith(RUNTIME) else 'controller')+'_status.json')
                path.parent.mkdir(parents=True,exist_ok=True)
                with path.open('xb') as handle:handle.write(reader.raw[name])
                evidence[name]=dict(snapshot=str(path.relative_to(ROOT)),sha256=reader.checksums[name])
    item=dict(schema='r10_readonly_concurrent_TEST_snapshot_v909',timestamp=now,summary=summary,completed=total,expected=1920,
        source_cases=640,methods=methods,infrastructure_failed=data['failed'],imported_completed=data['imported_completed'],
        resumed_new_completed=data['new_completed'],independently_released=data['release_complete'],admitted=data['admitted'],
        source_config_sha256=None if bundle is None else bundle['config_sha256'],
        source_lock_sha256=None if bundle is None else bundle['lock_sha256'],input_snapshots=evidence,
        source_status_updated_utc=data['status'].get('updated_utc'),
        next_update_utc=None if done else datetime.fromtimestamp(next_due,timezone.utc).isoformat(),reason=reason,
        TEST_raw_data_read=False,TEST_case_files_read=False,TEST_manifest_read=False,model_contents_read=False,
        live_OS_graph_verified=False,native_started=False,GPU_started=False,recorded_environment_only=True)
    text='\n'.join(lines)+'\n';legacy.atomic_text(ROOT/OUT,text);legacy.atomic_text(state/'snapshots'/(stamp+'.txt'),text)
    with (state/'history.jsonl').open('a',encoding='utf-8') as handle:handle.write(json.dumps(item,ensure_ascii=False)+'\n')
    legacy.atomic_json(state/'status.json',item)
    legacy.atomic_json(state/'chat_ready.json',dict(snapshot_id=stamp,created_utc=datetime.fromtimestamp(now,timezone.utc).isoformat(),
        next_update_utc=item['next_update_utc'],text=summary+'\n'+phase+'\nTXT：'+str(ROOT/OUT)))
    # No acknowledgement or delivery claim is created here.
    history.append(item);return item


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-lock-sha256',required=True)
    args=parser.parse_args();event=threading.Event()
    for number in (signal.SIGINT,signal.SIGTERM):signal.signal(number,lambda *_:event.set())
    stop=lambda:event.is_set() or any((ROOT/p).exists() for p in STOP_PATHS)
    state=ROOT/STATE
    with legacy.instance_lock(state):
        if stop():return
        bundle=load_source(args.source_lock_sha256,stop=stop)
        previous=legacy.read_json(state/'launch.json',missing=True)
        if previous:
            require(previous.get('source_config_sha256')==bundle['config_sha256']
                and previous.get('source_lock_sha256')==bundle['lock_sha256'], 'preserve this observer state; source/control binding cannot change on restart')
        seed_reader=Metadata(stop=stop)
        inherited=schedule_seed(bundle['config'],seed_reader)
        own=legacy.read_history(state/'history.jsonl')
        if own and own[-1].get('next_update_utc'):inherited=legacy.parse_utc(own[-1]['next_update_utc'])
        legacy_bytes=seed_reader.bytes(bundle['config']['legacy_schedule']['history']['path'],bundle['config']['legacy_schedule']['history']['sha256'])
        inherited_history=[]
        for line in legacy_bytes.decode().splitlines():
            try:
                value=json.loads(line)
                if isinstance(value,dict) and isinstance(value.get('summary'),str):inherited_history.append(dict(value,origin='original830'))
            except ValueError:pass
        history=[*inherited_history,*own];due=inherited;last=empty_data();first=True;old_signature=None
        legacy.atomic_json(state/'launch.json',dict(pid=os.getpid(),start_ticks=legacy.start_ticks(os.getpid()),
            source_sha256=sha((ROOT/SOURCE).read_bytes()),source_config_sha256=bundle['config_sha256'],source_lock_sha256=bundle['lock_sha256'],
            inherited_deadline_utc=datetime.fromtimestamp(inherited,timezone.utc).isoformat(),native_started=False,GPU_started=False))
        while True:
            stopping=stop()
            if stopping:data=deepcopy(last)
            else:
                try:data=collect(bundle,stop=stop)
                except legacy.ObservationStopped:stopping=True;data=deepcopy(last)
                except (OSError,ValueError,TypeError,KeyError,AttributeError,IndexError) as error:
                    data=deepcopy(last);data['errors']=[str(error)];data['release_complete']=False
            signature=digest(dict(errors=data['errors'],failed=data['failed'],admitted=data['admitted']))
            changed=old_signature is not None and signature!=old_signature;old_signature=signature
            if first or stopping or data['release_complete'] or changed or time.time()>=due:
                due=legacy.advance_deadline(due,time.time(),1800)
                save_report(data,bundle,history,next_due=due,reason='STOP仅停止观察' if stopping else
                    '实际906审计及907终态确认' if data['release_complete'] else '启动快照' if first else
                    '读取／基础设施状态变化' if changed else '每30分钟定时更新',stopping=stopping)
                first=False
            last=data
            if stopping or data['release_complete']:
                legacy.atomic_json(state/'terminal.json',dict(complete=True,observer_only=True,
                    final_TEST_independently_released=data['release_complete'],user_or_signal_stop=stopping,native_started=False,GPU_started=False))
                return
            event.wait(min(10,max(.1,due-time.time())))


if __name__=='__main__':main()
