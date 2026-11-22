"""Inactive, metadata-only reporter for the explicit five-tier TEST continuation.

Only --execute activates observation with an externally pinned small source lock.
No task is launched, stopped or retried; case/raw/model/source-manifest bytes and
the scientific input graph are never read. Chat text is prepared, never sent or
acknowledged. The original :08/:38 phase is independent of activation time.
"""
from collections import Counter
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import threading
import time
import types

ROOT = Path(__file__).resolve().parents[1]
SOURCE = 'scripts/report_composite_TEST_halfhour_v1044_r10.py'
CORE = 'our_mani/multitask/composite_TEST_resume_v1040.py'
OWNER_SOURCE = 'scripts/run_composite_TEST_owner_v1041_r10.py'
AUDITOR_SOURCE = 'scripts/audit_composite_TEST_results_v1042_r10.py'
CONTROLLER_SOURCE = 'scripts/continue_composite_TEST_v1043_r10.py'
ORIGINAL_SOURCE = 'scripts/run_formal_suite16_owner_v753_r10.py'
PARENT_SOURCE = 'scripts/run_composite_TEST_owner_v956_r10.py'
MIDDLE_SOURCE = 'scripts/run_concurrent_resumed_formal_TEST_owner_v902_r10.py'
PARENT_CORE = 'our_mani/multitask/composite_TEST_resume_v955.py'
PREVIOUS_CORE = 'our_mani/multitask/composite_TEST_resume_v1010.py'
PREVIOUS_SOURCE = 'scripts/run_composite_TEST_owner_v1011_r10.py'
LEGACY_SOURCE = 'scripts/report_composite_TEST_halfhour_v1014_r10.py'
CHECK_SOURCE = 'scripts/check_frozen_TEST_environment_v954_r10.py'
SOURCES = (SOURCE, CORE, OWNER_SOURCE, AUDITOR_SOURCE, CONTROLLER_SOURCE,
           ORIGINAL_SOURCE, PARENT_SOURCE, MIDDLE_SOURCE, PARENT_CORE, PREVIOUS_CORE, PREVIOUS_SOURCE, LEGACY_SOURCE, CHECK_SOURCE)
STATE = 'runtime/framework/concurrent_TEST_report_v1044'
OUT = 'TEST_结果更新_容量续跑.txt'
SCHEMA = 'r10_readonly_composite_TEST_report_source_v1044'
LOCK_SCHEMA = 'r10_readonly_composite_TEST_report_inputs_v1044'
RETIREMENT_SCHEMA = 'r10_private_legacy1014_retirement_for_reporter1044'
RETIREMENT = STATE + '/legacy1014_retirement.json'
ORIGINAL = 'runtime/framework/formal_TEST_after_environment_resume_v753'
PARENT = 'runtime/framework/formal_TEST_composite_resume_v956'
PREVIOUS = 'runtime/framework/formal_TEST_composite_resume_v1011'
MIDDLE = 'runtime/framework/formal_TEST_concurrent_resume_v902'
RUNTIME = 'runtime/framework/formal_TEST_composite_resume_v1041'
CONTROLLER = 'runtime/framework/ordered_composite_TEST_v1043'
LEGACY = 'runtime/framework/concurrent_TEST_report_v1014'
SETUP = 'artifacts/framework/multitask_r10/formal_TEST_composite_resume_v1041_setup'
CONTROL, CONTROL_LOCK = SETUP + '/resume_control.json', SETUP + '/resume_inputs.json'
PREVIOUS_SETUP = 'artifacts/framework/multitask_r10/formal_TEST_composite_resume_v1011_setup'
PREVIOUS_CONTROL, PREVIOUS_LOCK = PREVIOUS_SETUP+'/resume_control.json', PREVIOUS_SETUP+'/resume_inputs.json'
OLD_SETUP = 'artifacts/framework/multitask_r10/formal_TEST_composite_resume_v956_setup'
MIDDLE_SETUP = 'artifacts/framework/multitask_r10/formal_TEST_concurrent_resume_v902_setup'
MIDDLE_CONTROL, MIDDLE_LOCK = MIDDLE_SETUP+'/resume_control.json', MIDDLE_SETUP+'/resume_inputs.json'
OLD_CONTROL, OLD_LOCK = OLD_SETUP + '/resume_control.json', OLD_SETUP + '/resume_inputs.json'
RESULTS = 'artifacts/framework/multitask_r10/formal_TEST_composite_resume_v1041/TEST_results.json'
RECHECK = 'runtime/framework/frozen_TEST_environment_recheck_v954.json'
METHODS = ('cv_geometry', 'context_geometry', 'physmani')
LABELS = dict(cv_geometry='CV+几何', context_geometry='Context+几何', physmani='PhysMani')
FAMILIES = dict(beat_the_rotating_buzz='旋转电线穿环', insert_onto_rotating_peg='向旋转插杆套环',
    moving_basketball_in_hoop='向移动篮筐投球', pick_moving_target_on_the_table='抓取桌上移动方块',
    push_moving_button='按压移动按钮', put_rubbish_in_moving_bin='向移动垃圾桶投放垃圾',
    place_cups_on_rotating_frame='向旋转杯架挂杯子', remove_cups_from_rotating_frame='从旋转杯架取下杯子')
TASK_LABELS = {task + suffix: label + ('（高速）' if suffix else '（普通）')
    for task, label in FAMILIES.items() for suffix in ('', '_high_speed')}
PROFILE = dict(schema='r10_frozen_concurrent_native_resource_profile_v1005',
    max_native_slots=8, max_GPU_slots=3, max_CPU_preflight_slots=4, ready_limit=16)
CAPACITY = dict(schema='r10_frozen_concurrent_native_resource_profile_v1031',
    max_native_slots=12, max_GPU_slots=5, max_CPU_preflight_slots=6, ready_limit=24)
PHASE_US = 501384754  # 00:08:21.384754 UTC modulo 1800 seconds.
PERIOD_US = 1800000000
STOP_PATHS = ('runtime/framework/multitask_r10_STOP', STATE + '/STOP',
    CONTROLLER + '/STOP', RUNTIME + '/STOP')
RETIREMENT_FILES = dict(launch=LEGACY+'/launch.json', source=LEGACY_SOURCE,
    source_config=LEGACY+'/source.json', source_lock=LEGACY+'/source_inputs.json',
    history=LEGACY+'/history.jsonl', status=LEGACY+'/status.json',
    chat_ready=LEGACY+'/chat_ready.json', private_STOP=LEGACY+'/STOP')


class ObservationStopped(InterruptedError):
    pass


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def digest(value):
    return sha(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode())


def valid_sha(value):
    return isinstance(value, str) and len(value) == 64 and not set(value) - set('0123456789abcdef')


def valid_pointer(value, name=None):
    if not isinstance(value, dict) or set(value) != {'path', 'sha256'}:
        return False
    path = value.get('path')
    return (isinstance(path, str) and bool(path) and not Path(path).is_absolute()
        and str(Path(path)) == path and '..' not in Path(path).parts and valid_sha(value['sha256'])
        and (name is None or path == name))


def allowed_names():
    names = {*SOURCES, CONTROL, CONTROL_LOCK, OLD_CONTROL, OLD_LOCK, MIDDLE_CONTROL, MIDDLE_LOCK, PREVIOUS_CONTROL, PREVIOUS_LOCK, RECHECK, RETIREMENT,
        RESULTS, STATE+'/source.json', STATE+'/source_inputs.json', *RETIREMENT_FILES.values(),
        LEGACY+'/terminal.json'}
    for runtime in (ORIGINAL, MIDDLE, PARENT, PREVIOUS, RUNTIME, CONTROLLER):
        names.update(runtime+'/'+name for name in ('launch.json', 'plan.json', 'status.json', 'terminal.json', 'failure.json'))
    for stage in ('formal_TEST_1920', 'independent_TEST_results'):
        names.update(CONTROLLER+'/stages/'+stage+'/'+name for name in ('launch.json', 'terminal.json', 'stdout.log'))
    return names


class Metadata:
    """A single immutable byte snapshot, with no pointer-following read expansion."""
    def __init__(self, *, root=None, stop=lambda: False, read_bytes=None):
        self.root = ROOT if root is None else Path(root)
        self.stop, self.callback = stop, read_bytes
        self.allowed, self.raw, self.checksums = allowed_names(), {}, {}

    def bytes(self, name, expected=None, *, missing=False):
        if self.stop():
            raise ObservationStopped('STOP before metadata read')
        require(isinstance(name, str) and name in self.allowed and str(Path(name)) == name,
            'metadata read outside fixed reporter allowlist: '+str(name))
        if name not in self.raw:
            try:
                if self.callback is not None:
                    raw = self.callback(name)
                else:
                    path = self.root/name
                    require(path.resolve(strict=True) == path.absolute(), 'metadata symlink/alias refused')
                    raw = path.read_bytes()
            except FileNotFoundError:
                if missing:
                    return None
                raise
            require(type(raw) is bytes, 'exact metadata bytes required')
            self.raw[name], self.checksums[name] = raw, sha(raw)
        require(expected is None or self.checksums[name] == expected, 'frozen metadata SHA changed: '+name)
        return self.raw[name]

    def json(self, name, expected=None, *, missing=False):
        raw = self.bytes(name, expected, missing=missing)
        if raw is None:
            return {}
        value = json.loads(raw)
        require(isinstance(value, dict), 'metadata JSON object required: '+name)
        return value

    def pointer(self, pointer, name=None):
        require(valid_pointer(pointer, name), 'exact explicit metadata pointer required')
        return self.json(pointer['path'], pointer['sha256'])

    def point(self, name):
        require(name in self.checksums, 'pointer needs actually consumed bytes')
        return dict(path=name, sha256=self.checksums[name])


def declared(files, name, checksum, root):
    require(isinstance(files, dict), 'frozen input map required')
    keys = [key for key in (name, str(Path(root)/name)) if key in files]
    require(len(keys) == 1 and files[keys[0]] == checksum, 'missing/conflicting small metadata declaration: '+name)


def source_bundle(config, lock, *, config_sha, lock_sha, reader):
    require(config.get('schema') == SCHEMA and config.get('protocol_status') == 'formal_TEST_composite'
        and config.get('runtime') == RUNTIME and config.get('controller') == CONTROLLER
        and config.get('test_results') == RESULTS, 'one explicit five-tier1044 source required')
    require(lock.get('schema') == LOCK_SCHEMA and lock.get('approved_for_observation') is True,
        'externally pinned small observation lock required')
    bindings = config.get('source_bindings')
    require(isinstance(bindings, dict) and set(bindings) == set(SOURCES), 'exact reporting source map required')
    required = [dict(path=STATE+'/source.json', sha256=config_sha)]
    for name, pointer in bindings.items():
        require(valid_pointer(pointer, name), 'exact reporting source pointer required')
        required.append(pointer)
    for key, name in (('resume_control', CONTROL), ('resume_control_lock', CONTROL_LOCK),
                      ('legacy_retirement', RETIREMENT), ('environment_recheck', RECHECK)):
        require(valid_pointer(config.get(key), name), 'exact reporter metadata selection required: '+key)
        required.append(config[key])
    files = lock.get('files')
    require(isinstance(files, dict) and set(files) == {p['path'] for p in required},
        'observer lock is a small exact relative-path map, never a transitive input graph')
    for pointer in required:
        require(files[pointer['path']] == pointer['sha256'], 'observer small lock pointer differs')
    require(valid_sha(config.get('policy_sha256')) and isinstance(config.get('author_outcome_protocol'), dict),
        'frozen policy/protocol required')
    protocol = config['author_outcome_protocol']
    require(protocol.get('schema') == 'r10_author_action_outcome_protocol_v741'
        and protocol.get('native_RML_parameters_unchanged') is True
        and protocol.get('pose_action_max_ticks') == 10 and protocol.get('public_hold_max_ticks') == 1,
        'unchanged shared native author-action protocol required')
    for pointer in required:
        if pointer['path'] in SOURCES or pointer['path'] == STATE+'/source.json':
            reader.bytes(pointer['path'], pointer['sha256'])
    return dict(config=config, frozen=files, config_sha256=config_sha, lock_sha256=lock_sha)


def load_source(expected_lock_sha, *, root=None, stop=lambda: False, read_bytes=None):
    require(valid_sha(expected_lock_sha), 'external small source-lock SHA required')
    reader = Metadata(root=root, stop=stop, read_bytes=read_bytes)
    lock_raw = reader.bytes(STATE+'/source_inputs.json', expected_lock_sha)
    config_raw = reader.bytes(STATE+'/source.json')
    return source_bundle(json.loads(config_raw), json.loads(lock_raw), config_sha=sha(config_raw),
        lock_sha=sha(lock_raw), reader=reader)


def require_retirement(config, reader):
    value = reader.pointer(config['legacy_retirement'], RETIREMENT)
    require(value.get('schema') == RETIREMENT_SCHEMA
        and all(value.get(k) is True for k in ('complete', 'observer_only', 'private_stop_only', 'exit_observed'))
        and value.get('experiment_STOP_written') is False
        and type(value.get('session_id')) is int and value['session_id'] > 0
        and type(value.get('exit_returncode')) is int and bool(value.get('observed_exit_utc')),
        'root-recorded real private1014 retirement is required; reporter cannot retire it')
    for key, name in RETIREMENT_FILES.items():
        require(valid_pointer(value.get(key), name), 'exact frozen1014 retirement evidence required: '+key)
        reader.bytes(name, value[key]['sha256'])
    launch = reader.pointer(value['launch'])
    require(type(value.get('pid')) is int and value['pid'] > 0
        and type(value.get('start_ticks')) is int and value['start_ticks'] > 0
        and launch.get('pid') == value['pid'] and ('start_ticks' not in launch or launch['start_ticks'] == value['start_ticks'])
        and launch.get('source_sha256') == value['source']['sha256']
        and launch.get('source_config_sha256') == value['source_config']['sha256']
        and launch.get('source_lock_sha256') == value['source_lock']['sha256']
        and value['source'] == config['source_bindings'][LEGACY_SOURCE], 'actual retired1014 identity/source differs')
    # A terminal is optional. The root receipt records the actual session exit;
    # this observer neither probes a process nor invents a legacy self-terminal.
    if 'terminal' in value:
        reader.pointer(value['terminal'], LEGACY+'/terminal.json')
    history = []
    for line in reader.bytes(value['history']['path'], value['history']['sha256']).decode().splitlines():
        record = json.loads(line)
        require(isinstance(record, dict), 'frozen1014 JSONL records must be objects')
        history.append(dict(origin='actual1014', record=record))
    return value, history


def load_core(reader, config, name=CORE):
    # This reviewed module contains only pure contracts and stdlib imports. It
    # is read from the same verified snapshot, never imported via mutable cache.
    raw = reader.bytes(name, config['source_bindings'][name]['sha256'])
    module = types.ModuleType('report1044_verified_pure1040')
    exec(compile(raw, name, 'exec'), module.__dict__)
    return module


def common(value, config):
    require(value.get('policy_sha256') == config['policy_sha256']
        and value.get('author_outcome_protocol') == config['author_outcome_protocol'], 'frozen policy/protocol differs')


def recorded_environment(value, control, *, pid, start, expected=None):
    require(type(pid) is int and pid > 0 and type(start) is int and start > 0
        and isinstance(value, dict) and value.get('schema') == 'r10_actual_frozen_OS_namespace_receipt_v820'
        and value.get('complete') is True and value.get('checked_OS_files') == 6418
        and value.get('all_current_OS_targets_full_byte_SHA_verified') is True
        and value.get('observer_pid') == pid and value.get('observer_start_ticks') == start
        and isinstance(value.get('current_mount_namespace'), str), 'actual recorded820 observer identity required')
    for key, field in (('environment_manifest', 'manifest'), ('environment_wrapper', 'wrapper'),
                       ('environment_namespace_proof', 'namespace_qualification')):
        require(value.get(field) == control.get(key), 'recorded820 environment pointer differs')
    if expected is not None:
        require(value['current_mount_namespace'] == expected['current_mount_namespace'], 'recorded child mount namespace differs')
    return value


def grid(plan):
    require(plan.get('split') == 'test' and plan.get('expected_cases') == 1920
        and plan.get('source_outcome_filtering') is False, 'original640-source three-method plan required')
    rows = plan.get('jobs'); require(isinstance(rows, list) and len(rows) == 1920, 'exact1920 plan rows required')
    result, groups, coords = {}, {}, set()
    for row in rows:
        require(isinstance(row, dict) and row.get('task') in TASK_LABELS and row.get('method') in METHODS
            and isinstance(row.get('raw'), str) and bool(row['raw']) and isinstance(row.get('job_id'), str)
            and row['job_id'] not in result, 'unknown or duplicate plan key')
        coordinate = row['task'], row['raw'], row['method']
        require(coordinate not in coords, 'duplicate original source/method coordinate')
        coords.add(coordinate); result[row['job_id']] = row
        groups.setdefault((row['task'], row['method']), set()).add(row['raw'])
    require(len(groups) == 48 and all(len(v) == 40 for v in groups.values())
        and all(groups[t, m] == groups[t, METHODS[0]] for t in TASK_LABELS for m in METHODS),
        'three methods must share all40 original sources per task')
    return result


def require_parent_plan(parent, original, old_control):
    """The actual902 plan is reconstructed from753 without adding old NS fields."""
    jobs, old = grid(parent), grid(original)
    adopted = old_control.get('completed_job_ids'); remaining = old_control.get('remaining_job_ids')
    require(isinstance(adopted, list) and isinstance(remaining, list)
        and len(adopted) == len(set(adopted)) and len(remaining) == len(set(remaining))
        and not set(adopted) & set(remaining) and set(adopted) | set(remaining) == set(old)
        and set(jobs) == set(old), 'original753 to902 partition differs')
    for key, row in jobs.items():
        previous, imported = old[key], key in adopted
        expected = deepcopy(previous)
        expected.update(evidence_runtime=ORIGINAL if imported else MIDDLE, imported_completed=imported,
            previous_row_sha256=digest(previous), previous_worker_sha256=previous['worker_sha256'])
        if imported:
            expected.update(physical_argv=deepcopy(previous['argv']), execution_entry=dict(kind='original751', pointer=previous['admission_driver']))
        else:
            inner = deepcopy(previous['argv']); require(inner.count('--out') == 1, 'one physical output required')
            inner[inner.index('--out')+1] = str(Path('artifacts/framework/multitask_r10/formal_TEST_concurrent_resume_v902/test')/row['task']/row['raw']/row['method'])
            entry = old_control['execution_entry']; outer = [*inner[:2], entry['pointer']['path'],
                '--batch-control', entry['batch_control']['path'], '--batch-lock', entry['batch_lock']['path'],
                '--batch-lock-sha256', entry['batch_lock']['sha256'], '--entry', 'formal', '--', *inner[3:]]
            expected.update(physical_argv=inner, argv=outer, execution_entry=entry, worker_sha256=entry['pointer']['sha256'])
        require(row == expected, 'historical902 row differs beyond its declared old continuation')
    return set(adopted)


def count_status(status, jobs, parent_status, imported, profile):
    require(status.get('split') == 'test' and status.get('expected') == 1920
        and status.get('automatic_retry') is False and isinstance(status.get('completed'), dict)
        and isinstance(status.get('failed'), dict), 'actual1041 partial status required')
    records = status['completed']
    require(set(parent_status.get('completed', {})) == imported and imported <= set(records)
        and all(records[k] == parent_status['completed'][k] for k in imported), 'adopt all literal historical outcomes exactly once')
    completed, success, unknown, motion_false, motion_unknown, tiers = (Counter() for _ in range(6))
    for key, value in records.items():
        require(key in jobs and isinstance(value, dict) and value.get('complete') is True
            and 'native_success' in value and (value['native_success'] is None or type(value['native_success']) is bool)
            and 'motion_pass' in value and (value['motion_pass'] is None or type(value['motion_pass']) is bool)
            and type(value.get('returncode')) is int and value['returncode'] == 0
            and all(value.get(k) is False for k in ('timed_out', 'interrupted', 'automatic_retry'))
            and value.get('error') is None and value.get('leaf_future_exception') is None,
            'only clean completed outcomes are counted; native/motion unknown remain explicit')
        row = jobs[key]; inner = row['physical_argv']
        require(inner.count('--out') == 1, 'one physical output required')
        out = inner[inner.index('--out')+1]
        for field, leaf in (('case', 'case.json'), ('admission_driver_receipt', 'admission_driver_receipt.json')):
            require(valid_pointer(value.get(field), str(Path(out)/leaf)), 'completed pointer differs from its real plan output')
        group = row['task'], row['method']; completed[group] += 1
        success[group] += value['native_success'] is True
        unknown[group] += value['native_success'] is None
        motion_false[group] += value['motion_pass'] is False
        motion_unknown[group] += value['motion_pass'] is None
        tiers[row['evidence_runtime']] += 1
    failed = status['failed']; active, checking, ready = (status.get(k) for k in ('active', 'checking', 'ready_jobs'))
    require(set(failed) <= set(jobs)-imported and not set(failed) & set(records), 'currentinfra cannot overlap completed/imported keys')
    for value in failed.values():
        require(isinstance(value, dict) and value.get('complete') is not True and value.get('automatic_retry') is False,
            'current infrastructure failure is an attempt, not nativeFalse')
    for keys in (active, checking, ready):
        require(isinstance(keys, list) and len(keys) == len(set(keys)) and set(keys) <= set(jobs)
            and not set(keys) & (set(records)|set(failed)), 'unknown/overlapping active, checking or ready job')
    require(not set(active) & set(checking) and not set(active) & set(ready) and not set(checking) & set(ready)
        and len(active) <= profile['max_native_slots'] and sum(jobs[k]['method'] == 'physmani' for k in active) <= profile['max_GPU_slots']
        and len(checking) <= profile['max_CPU_preflight_slots'] and len(ready) <= profile['ready_limit']
        and all(type(status.get(k)) is int for k in ('CPU_ready', 'native_slots', 'GPU_slots'))
        and status.get('CPU_ready') == len(ready) and status.get('native_slots') == len(active)
        and status.get('GPU_slots') == sum(jobs[k]['method'] == 'physmani' for k in active),
        'recorded1041 resource inventory differs from its frozen actual quota')
    return dict(completed=completed, success=success, unknown_native=unknown, motion_false=motion_false,
        unknown_motion=motion_unknown, evidence_tiers=tiers, failed=len(failed),
        imported_completed=len(imported), new_completed=len(records)-len(imported), status=status)


def stage_inner(value, source, root):
    argv = value.get('argv')
    require(isinstance(argv, list) and argv[:4] == [str(Path(root)/'.conda-env/bin/python'), '-B', '-u', source],
        'actual1043 child Python/-B/-u/source prefix required')
    return argv[3:]


def recorded_profiles(control):
    """Recorded metadata only; the reporter never signs a resource release."""
    profile, capacity = control.get('resource_profile'), control.get('resource_capacity_envelope')
    fields = set(PROFILE)
    require(isinstance(profile, dict) and isinstance(capacity, dict)
        and set(profile) == set(capacity) == fields
        and profile.get('schema') == capacity.get('schema'), 'explicit actual quota and recorded capacity required')
    source = control.get('resource_capacity_source')
    require(source in ('qualified1031', 'qualified1005_fallback')
        and capacity == (CAPACITY if source == 'qualified1031' else PROFILE)
        and control.get('resource_profile_within_capacity') is True,
        'one separately recorded qualified capacity and actual quota required')
    for name in fields-{'schema'}:
        require(type(profile[name]) is int and 1 <= profile[name] <= capacity[name],
            'actual quota must be a strict integer within its recorded capacity')
    require(profile['max_GPU_slots'] <= profile['max_native_slots'], 'GPU quota exceeds Native quota')
    if source == 'qualified1005_fallback':
        require(profile == PROFILE, '1005 fallback retains its original qualified full profile')
    version = '1031' if source == 'qualified1031' else '1005'
    require(valid_pointer(control.get('resource_qualification'),
        'runtime/framework/concurrent_TRAIN_resource_qualification_v'+version+'.json')
        and valid_pointer(control.get('resource_qualifier'),
        'scripts/audit_concurrent_TRAIN_resource_canary_v'+version+'_r10.py'),
        'selected capacity must retain its actual qualifier and proof pointers')
    return deepcopy(profile)


def collect(bundle, *, root=None, stop=lambda: False, read_bytes=None):
    reader = Metadata(root=root, stop=stop, read_bytes=read_bytes); config = bundle['config']
    reader.bytes(STATE+'/source_inputs.json', bundle['lock_sha256'])
    for name in (STATE+'/source.json', *SOURCES): reader.bytes(name, bundle['frozen'][name])
    retirement, history = require_retirement(config, reader)
    core = load_core(reader, config)
    control = reader.pointer(config['resume_control'], CONTROL)
    locked = reader.pointer(config['resume_control_lock'], CONTROL_LOCK)
    common(control, config)
    profile=recorded_profiles(control)
    require(locked.get('schema') == core.LOCK_SCHEMA
        and locked.get('approved_for_execution') is True and locked.get('control')==config['resume_control'],
        'frozen1040 small continuation metadata required')
    for key, name in (('resume_owner', OWNER_SOURCE), ('resume_auditor', AUDITOR_SOURCE), ('resume_controller', CONTROLLER_SOURCE)):
        require(control.get(key) == config['source_bindings'][name], 'actual1040/1041/1042/1043 source binding differs')
    for pointer in (config['resume_control'], *(config['source_bindings'][name] for name in (CORE, OWNER_SOURCE, AUDITOR_SOURCE, CONTROLLER_SOURCE))):
        declared(locked.get('files'), pointer['path'], pointer['sha256'], reader.root)
    require(control.get('environment_recheck') == config['environment_recheck'], 'selected954 point observation differs')
    core.require_environment_recheck(reader.pointer(config['environment_recheck'], RECHECK), control)
    parent_core=load_core(reader,config,PARENT_CORE)
    previous_core=load_core(reader,config,PREVIOUS_CORE)
    previous_control=reader.pointer(control['previous_resume_control'],PREVIOUS_CONTROL)
    previous_lock=reader.pointer(control['previous_resume_control_lock'],PREVIOUS_LOCK)
    require(previous_control.get('schema')==previous_core.CONTROL_SCHEMA and previous_lock.get('schema')==previous_core.LOCK_SCHEMA
        and previous_lock.get('approved_for_execution') is True,'actual immutable1010 parent control/lock required')
    old_control=reader.pointer(previous_control['previous_resume_control'],OLD_CONTROL)
    old_lock=reader.pointer(previous_control['previous_resume_control_lock'],OLD_LOCK)
    middle_control=reader.pointer(old_control['previous_resume_control'],MIDDLE_CONTROL)
    middle_lock=reader.pointer(old_control['previous_resume_control_lock'],MIDDLE_LOCK)
    require(old_control.get('schema')==parent_core.CONTROL_SCHEMA and old_lock.get('schema')==parent_core.LOCK_SCHEMA
        and old_lock.get('approved_for_execution') is True
        and middle_control.get('schema')=='r10_concurrent_resumed_formal_TEST_control_v902'
        and middle_lock.get('schema')=='r10_concurrent_resumed_formal_TEST_inputs_v902'
        and middle_lock.get('approved_for_execution') is True,'actual955 and915 historical controls/locks required')
    common(previous_control,config);common(old_control,config);common(middle_control,config)
    original={key:reader.pointer(control['original_'+key],ORIGINAL+'/'+key+'.json') for key in ('launch','plan','status','terminal')}
    middle={key:reader.pointer(old_control['previous_'+key],MIDDLE+'/'+key+'.json') for key in original}
    parent={key:reader.pointer(previous_control['previous_'+key],PARENT+'/'+key+'.json') for key in original}
    previous={key:reader.pointer(control['previous_'+key],PREVIOUS+'/'+key+'.json') for key in original}
    for key in original:
        require(control['original_'+key]==previous_control['original_'+key]==old_control['original_'+key]==middle_control['previous_'+key],
            'all three controls retain exact original753 pointers')
    for pointer in (old_control['previous_resume_control'],old_control['previous_resume_control_lock'],
                    *(old_control['original_'+key] for key in original),*(old_control['previous_'+key] for key in middle)):
        declared(old_lock.get('files'),pointer['path'],pointer['sha256'],reader.root)
    for pointer in (old_control['previous_resume_control'],*(middle_control['previous_'+key] for key in original)):
        declared(middle_lock.get('files'),pointer['path'],pointer['sha256'],reader.root)
    for pointer in (previous_control['previous_resume_control'],previous_control['previous_resume_control_lock'],
                    *(previous_control['previous_'+key] for key in parent),*(previous_control['original_'+key] for key in original)):
        declared(previous_lock.get('files'),pointer['path'],pointer['sha256'],reader.root)
    for pointer in (control['previous_resume_control'],control['previous_resume_control_lock'],
                    *(control['previous_'+key] for key in previous),*(control['original_'+key] for key in original)):
        declared(locked.get('files'),pointer['path'],pointer['sha256'],reader.root)
    for group,name in ((original,ORIGINAL_SOURCE),(middle,MIDDLE_SOURCE),(parent,PARENT_SOURCE),(previous,PREVIOUS_SOURCE)):
        common(group['launch'],config)
        require(group['launch'].get('source_sha256')==config['source_bindings'][name]['sha256']
            and group['plan'].get('policy_sha256')==config['policy_sha256'],'actual historical owner source/policy differs')
    original_imported=require_parent_plan(middle['plan'],original['plan'],middle_control)
    require(len(original_imported)==134 and set(original['status'].get('completed',{}))==original_imported
        and all(middle['status']['completed'].get(k)==original['status']['completed'][k] for k in original_imported),
        'all original134 outcomes remain literal real753 records')
    parent_core.require_resume_plan(old_control,middle['plan'],middle['status'],middle['terminal'],parent['plan'])
    require(len(old_control['completed_job_ids'])==584 and len(old_control['failed_job_ids'])==6
        and all(parent['status']['completed'].get(k)==v for k,v in middle['status']['completed'].items()),
        'original753134 and902450 outcomes and six historical infra attempts remain separate')
    previous_core.require_resume_plan(previous_control,parent['plan'],parent['status'],parent['terminal'],previous['plan'])
    require(len(previous_control['completed_job_ids'])==1266
        and all(previous['status']['completed'].get(k)==v for k,v in parent['status']['completed'].items()),
        'every original753/902/956 completion remains literal in1011')
    core.require_partition(control,previous['plan'],previous['status'],previous['terminal'])
    require(control.get('resource_profile_changed') is (profile!=previous_control.get('resource_profile'))
        and control.get('previous_resource_profile')==previous_control.get('resource_profile')
        and control.get('previous_resource_qualification')==previous_control.get('resource_qualification'),
        'selected actual quota must retain its genuine previous profile and qualification')
    launch = reader.json(CONTROLLER+'/launch.json'); controller = reader.json(CONTROLLER+'/status.json')
    require(launch.get('source_sha256') == config['source_bindings'][CONTROLLER_SOURCE]['sha256']
        and launch.get('resume_control') == config['resume_control'] and launch.get('resume_control_lock') == config['resume_control_lock']
        and launch.get('automatic_retry') is False, 'actual1043 controller launch differs')
    environment = recorded_environment(launch.get('frozen_environment'), control, pid=launch.get('pid'), start=launch.get('start_ticks'))
    require(controller.get('frozen_environment') == environment and controller.get('automatic_retry') is False
        and controller.get('stage') in ('formal_TEST_1920', 'independent_TEST_results', 'complete',
            'draining_for_concrete_repair', 'requires_concrete_repair', 'user_stopped'), 'actual1043 status required')
    owner = reader.json(RUNTIME+'/launch.json'); plan = reader.json(RUNTIME+'/plan.json'); status = reader.json(RUNTIME+'/status.json')
    for value in (owner, plan, status):
        common(value, config)
        require(value.get('resume_control') == config['resume_control'] and value.get('resume_control_lock') == config['resume_control_lock']
            and value.get('resource_profile') == profile, 'actual1041 owner/control/profile differs')
        for key in ('previous_resume_control', 'previous_resume_control_lock', 'environment_recheck',
                    'resource_profile_changed','previous_resource_profile','previous_resource_qualification',
                    'resource_capacity_envelope','resource_profile_within_capacity','resource_capacity_source'):
            require(value.get(key) == control.get(key), 'actual1041 parent/recheck binding differs: '+key)
    require(owner.get('base_stage_owner') == config['source_bindings'][ORIGINAL_SOURCE]
        and owner.get('parent_stage_owner') == config['source_bindings'][PREVIOUS_SOURCE],
        'actual1041 historical owner source pointers differ from their real entrypoints')
    require(owner.get('source_sha256') == config['source_bindings'][OWNER_SOURCE]['sha256'], 'actual1041 source differs')
    recorded_environment(owner.get('frozen_environment'), control, pid=owner.get('pid'), start=owner.get('start_ticks'), expected=environment)
    for value in (plan, status): require(value.get('frozen_environment') == owner['frozen_environment'], '1041 recorded environment changed')
    owner_stage = reader.json(CONTROLLER+'/stages/formal_TEST_1920/launch.json')
    require(owner_stage.get('pid') == owner.get('pid') and owner_stage.get('start_ticks') == owner.get('start_ticks')
        and owner_stage.get('source_sha256') == config['source_bindings'][OWNER_SOURCE]['sha256']
        and owner_stage.get('inherited_frozen_environment') == environment
        and owner_stage.get('current_mount_namespace') == environment['current_mount_namespace']
        and stage_inner(owner_stage, OWNER_SOURCE, reader.root) == owner.get('argv'), '1041 must be the actual1043 child')
    core.require_resume_plan(control, previous['plan'], previous['status'], previous['terminal'], plan)
    jobs = grid(plan); imported = set(control['completed_job_ids'])
    require(owner.get('imported_completed_cases')==status.get('imported_completed')==len(imported)
        and all(owner.get(key)==profile[key] for key in ('max_native_slots','max_GPU_slots','max_CPU_preflight_slots')),
        'actual1041 imported count and selected launch quotas must match')
    data = count_status(status, jobs, previous['status'], imported, profile)
    require(data['evidence_tiers'][ORIGINAL] == 134 and data['evidence_tiers'][MIDDLE] == 450
        and data['evidence_tiers'][PARENT]==682 and data['evidence_tiers'][PREVIOUS]==len(imported)-1266,
        'preserved753134 +902450 must be counted once under their real tiers')
    data.update(admitted=True, plan_valid=True, controller=controller, errors=[], release_complete=False,
        historical_infrastructure_failed=6, historical_failure_job_ids=list(old_control['failed_job_ids']),
        historical956_infrastructure_failed=len(previous_control['failed_job_ids']),
        historical1011_infrastructure_failed=len(control['failed_job_ids']),
        historical956_failure_job_ids=list(previous_control['failed_job_ids']),
        historical1011_failure_job_ids=list(control['failed_job_ids']),old_control=old_control,previous_control=previous_control,profile=profile,
        resource_capacity_envelope=deepcopy(control['resource_capacity_envelope']),
        resource_capacity_source=control['resource_capacity_source'],resource_profile_within_capacity=True,
        retirement=retirement, legacy_history=history, reader=reader, config=config)
    if status.get('owner_error') or status.get('stop_requested') or status.get('drain_requested') or data['failed']:
        data['errors'].append('1041当前基础设施／停止／排空状态，仅报告部分进度')
    if controller['stage'] in ('draining_for_concrete_repair', 'requires_concrete_repair', 'user_stopped') or controller.get('infrastructure_failure_detected') or controller.get('owner_error'):
        data['errors'].append('1043正在排空／停止或等待具体修复，非完整结果')
    if reader.json(CONTROLLER+'/failure.json', missing=True): data['errors'].append('1043已记录真实失败')
    if len(status['completed']) == 1920 and not data['errors'] and controller['stage'] == 'complete':
        data['release_complete'] = final_gate(data, jobs, owner, original, middle, parent, previous, control, reader)
    return data


def final_gate(data, jobs, owner, original, middle, parent, previous, control, reader):
    config = data['config']; status = data['status']; profile=data['profile']
    terminal = reader.json(RUNTIME+'/terminal.json'); common(terminal, config)
    require(terminal.get('resume_control') == config['resume_control'] and terminal.get('resume_control_lock') == config['resume_control_lock']
        and terminal.get('resource_profile') == profile and terminal.get('frozen_environment') == owner['frozen_environment']
        and all(terminal.get(k) == control.get(k) for k in ('previous_resume_control', 'previous_resume_control_lock', 'environment_recheck',
            'resource_profile_changed','previous_resource_profile','previous_resource_qualification',
            'resource_capacity_envelope','resource_profile_within_capacity','resource_capacity_source')),
        'actual complete1041 terminal must bind the same control/environment/parent')
    require(terminal.get('complete') is True and terminal.get('expected') == terminal.get('completed') == 1920
        and terminal.get('failed') == 0 and terminal.get('owner_error') is None
        and all(terminal.get(k) is False for k in ('stop_requested', 'drain_requested', 'drained', 'automatic_retry'))
        and terminal.get('CPU_ready_unexecuted') == [] and terminal.get('CPU_not_launched') == 0
        and terminal.get('imported_completed_cases') == len(control['completed_job_ids'])
        and terminal.get('resumed_new_cases') == 1920-len(control['completed_job_ids'])
        and status.get('active') == status.get('checking') == status.get('ready_jobs') == []
        and status.get('CPU_ready') == status.get('native_slots') == status.get('GPU_slots') == 0,
        'complete clean10411920 terminal required before opening1042 result metadata')
    ctl_terminal = reader.json(CONTROLLER+'/terminal.json')
    require(ctl_terminal.get('complete') is True and ctl_terminal.get('automatic_retry') is False
        and valid_pointer(ctl_terminal.get('TEST_results'), RESULTS), 'actual1043 complete result pointer required')
    stages = {}
    for stage, source in (('formal_TEST_1920', OWNER_SOURCE), ('independent_TEST_results', AUDITOR_SOURCE)):
        leaf = CONTROLLER+'/stages/'+stage
        launched, ended = reader.json(leaf+'/launch.json'), reader.json(leaf+'/terminal.json')
        require(ended.get('complete') is True and type(ended.get('returncode')) is int and ended['returncode'] == 0
            and ended.get('automatic_retry') is False and launched.get('automatic_retry') is False
            and launched.get('source_sha256') == config['source_bindings'][source]['sha256']
            and launched.get('inherited_frozen_environment') == data['controller']['frozen_environment']
            and launched.get('current_mount_namespace') == data['controller']['frozen_environment']['current_mount_namespace'],
            'real clean1043 owner/auditor stages required before result read')
        stage_inner(launched, source, reader.root); stages[stage] = launched
    stdout = reader.json(CONTROLLER+'/stages/independent_TEST_results/stdout.log')
    require(stdout.get('complete') is True and stdout.get('schema') == 'r10_five_tier_composite_formal_TEST_results_v1042'
        and stdout.get('split') == 'test' and stdout.get('cases') == 1920 and stdout.get('out') == RESULTS,
        'real full1042 stdout required; it has no TEST_permitted field')
    result = reader.pointer(ctl_terminal['TEST_results'], RESULTS); common(result, config)
    require(result.get('schema') == 'r10_five_tier_composite_formal_TEST_results_v1042'
        and result.get('complete') is True and result.get('split') == 'test' and result.get('source_cases') == 640
        and result.get('method_cases') == result.get('completed_method_cases') == 1920
        and result.get('methods') == list(METHODS) and isinstance(result.get('tasks'), list)
        and len(result['tasks']) == 16 and set(result['tasks']) == set(TASK_LABELS)
        and result.get('source_sha256') == config['source_bindings'][AUDITOR_SOURCE]['sha256']
        and result.get('resume_control') == config['resume_control'] and result.get('resume_control_lock') == config['resume_control_lock']
        and result.get('source_input_lock') == control['source_inputs'] and result.get('validation_release') == control['validation_release']
        and result.get('resource_profile') == profile and result.get('resource_qualification') == control['resource_qualification']
        and result.get('resource_profile_changed') is control['resource_profile_changed']
        and result.get('previous_resource_profile')==control['previous_resource_profile']
        and result.get('previous_resource_qualification')==control['previous_resource_qualification']
        and result.get('resource_capacity_envelope')==control['resource_capacity_envelope']
        and result.get('resource_profile_within_capacity') is True
        and result.get('resource_capacity_source')==control['resource_capacity_source']
        and result.get('execution_implementation') == control['execution_implementation']
        and result.get('environment_recheck') == control['environment_recheck']
        and all(result.get(k) is True for k in ('independent_composite_audit', 'used_evidence_rechecked_at_close',
            'native_failures_retained', 'motion_unknown_retained', 'zero_tick_cases_retained', 'motion_is_diagnostic_only', 'old_completed_results_not_rerun'))
        and all(result.get(k) is False for k in ('source_outcome_filtering', 'Value_NGFF_FASTER_used', 'automatic_retry',
            'old907_or958_or1013_successful_terminal_claimed', 'original754_906_957_or1012_complete_result_claimed')),
        'actual complete independent1042 five-tier result required')
    audit_launch = stages['independent_TEST_results']; invocation = result.get('auditor_invocation', {})
    require(invocation.get('pid') == audit_launch.get('pid') and invocation.get('start_ticks') == audit_launch.get('start_ticks')
        and invocation.get('source') == config['source_bindings'][AUDITOR_SOURCE]
        and invocation.get('argv') == stage_inner(audit_launch, AUDITOR_SOURCE, reader.root), 'real1042 auditor invocation differs')
    recorded_environment(result.get('frozen_environment'), control, pid=invocation['pid'], start=invocation['start_ticks'], expected=data['controller']['frozen_environment'])
    owner_data = {ORIGINAL: original, MIDDLE: middle, PARENT: parent, PREVIOUS: previous, RUNTIME: dict(launch=owner, terminal=terminal)}
    owners = result.get('actual_stage_owners'); require(isinstance(owners, list) and len(owners) == 5, 'five real owners required')
    require({v.get('runtime') for v in owners} == set(owner_data), 'actual owner tiers differ')
    for value in owners:
        runtime = value['runtime']; actual = owner_data[runtime]['launch']
        require(value.get('launch') == reader.point(runtime+'/launch.json') and value.get('terminal') == reader.point(runtime+'/terminal.json')
            and value.get('pid') == actual.get('pid') and value.get('start_ticks') == actual.get('start_ticks')
            and value.get('completed') == data['evidence_tiers'][runtime], 'actual1042 owner provenance/count differs')
    cases, drivers, tiers = (result.get(k) for k in ('cases', 'admission_driver_receipts', 'evidence_tiers'))
    require(all(isinstance(v, list) and len(v) == 1920 for v in (cases, drivers, tiers))
        and len({v.get('job_id') for v in tiers}) == 1920, 'complete unique1920 result pointer/tier grid required')
    for index, row in enumerate(jobs.values()):
        tier, record = tiers[index], status['completed'][row['job_id']]
        require(tier.get('job_id') == row['job_id'] and all(tier.get(k) == row.get(k)
            for k in ('evidence_runtime', 'imported_completed', 'previous_row_sha256', 'execution_entry', 'resume_lineage', 'resume_lineage_v1010', 'resume_lineage_v1040'))
            and tier.get('case') == cases[index] == record['case']
            and tier.get('admission_driver_receipt') == drivers[index] == record['admission_driver_receipt'],
            'result pointers must match the same ordered1920 metadata grid; no cases are opened')
        actual = owner_data[row['evidence_runtime']]['launch']
        require(tier.get('actual_stage_owner_pid') == actual['pid'] and tier.get('actual_stage_owner_start_ticks') == actual['start_ticks'],
            'case actual historical/current owner identity differs')
    require(not sum(data['unknown_native'].values()), 'a final audited native outcome must be explicit, never unknown')
    totals = {m: sum(data['success'][t, m] for t in TASK_LABELS) for m in METHODS}
    require(result.get('native_successes_by_method') == stdout.get('native_successes_by_method') == totals, '1042 totals differ from actual status metadata')
    tasks = result.get('per_task')
    require(isinstance(tasks, list) and len(tasks) == 16 and {v.get('task') for v in tasks} == set(TASK_LABELS), 'complete16-task1042 summary required')
    for task in tasks:
        require(set(task.get('methods', {})) == set(METHODS), 'three-method1042 per-task summary required')
        for method, value in task['methods'].items():
            require(type(value.get('episodes')) is int and value['episodes'] == 40
                and type(value.get('successes')) is int and value['successes'] == data['success'][task['task'], method],
                '1042 per-task results differ from real completed status')
    historical=result.get('historical_infrastructure_attempts')
    expected_history=[dict(runtime=runtime,job_ids=ctl['failed_job_ids'],previous_status=ctl['previous_status'],
        previous_terminal=ctl['previous_terminal'],counted_as_native_task_failures=False,
        all_keys_reexecuted_once_in_new_namespace=True) for runtime,ctl in ((MIDDLE,data['old_control']),(PARENT,data['previous_control']),(PREVIOUS,control))]
    require(historical==expected_history,'all three historical infrastructure attempt cohorts remain separate from native outcomes')
    inputs = result.get('inputs_sha256')
    required = (CONTROL, CONTROL_LOCK, OLD_CONTROL, OLD_LOCK, MIDDLE_CONTROL, MIDDLE_LOCK, PREVIOUS_CONTROL, PREVIOUS_LOCK, RECHECK, OWNER_SOURCE, AUDITOR_SOURCE,
        *(runtime+'/'+name for runtime in (ORIGINAL, MIDDLE, PARENT, PREVIOUS, RUNTIME) for name in ('launch.json', 'plan.json', 'status.json', 'terminal.json')))
    for name in required:
        declared(inputs, name, reader.checksums[name], reader.root)
    return True


def empty_data(error=None):
    # None means unadmitted/unknown, never a fabricated zero completion count.
    return dict(admitted=False, plan_valid=False, release_complete=False, completed=None, success=None,
        unknown_native=None, motion_false=None, unknown_motion=None, evidence_tiers=None, failed=None,
        historical_infrastructure_failed=None, historical956_infrastructure_failed=None, historical1011_infrastructure_failed=None, imported_completed=None, new_completed=None,
        status={}, controller={}, errors=[] if error is None else [str(error)], legacy_history=[])


def observation_error(previous, error):
    """Current counts become unknown; previously frozen history stays visible."""
    value = empty_data(error)
    value['legacy_history'] = deepcopy(previous.get('legacy_history', []))
    value['historical_infrastructure_failed'] = previous.get('historical_infrastructure_failed')
    value['historical956_infrastructure_failed']=previous.get('historical956_infrastructure_failed')
    value['historical1011_infrastructure_failed']=previous.get('historical1011_infrastructure_failed')
    return value


def phase_deadline(now):
    """Next strictly future point on the original :08:21.384754/:38 phase."""
    current = int(round(now*1000000))
    return (PHASE_US + ((current-PHASE_US)//PERIOD_US+1)*PERIOD_US)/1000000


def is_phase(value):
    return (int(round(value*1000000))-PHASE_US) % PERIOD_US == 0


def clock_text(value):
    return datetime.fromtimestamp(value, timezone(timedelta(hours=8))).isoformat(timespec='seconds')


def snapshot(data, *, now, next_due, reason):
    require(is_phase(next_due) and next_due > now, 'fixed original half-hour phase required')
    methods, total = {}, None
    if data['admitted']:
        for m in METHODS:
            methods[m] = {field: sum(data[key][t, m] for t in TASK_LABELS) for field, key in (
                ('completed', 'completed'), ('successes', 'success'), ('unknown_native', 'unknown_native'),
                ('motion_false', 'motion_false'), ('unknown_motion', 'unknown_motion'))}
        total = sum(v['completed'] for v in methods.values())
    else:
        methods = {m: dict(completed=None, successes=None, unknown_native=None, motion_false=None, unknown_motion=None) for m in METHODS}
    return dict(schema='r10_composite_TEST_halfhour_snapshot_v1044', created_utc=datetime.fromtimestamp(now, timezone.utc).isoformat(),
        next_update_utc=datetime.fromtimestamp(next_due, timezone.utc).isoformat(), admitted=data['admitted'],
        release_complete=data['release_complete'], total_completed=total, expected=1920, methods=methods,
        imported_completed=data['imported_completed'], new_completed=data['new_completed'],
        evidence_tiers=None if data['evidence_tiers'] is None else dict(data['evidence_tiers']),
        current1041_infrastructure_failed=data['failed'], historical902_infrastructure_failed=data['historical_infrastructure_failed'],
        historical956_infrastructure_failed=data['historical956_infrastructure_failed'],
        historical1011_infrastructure_failed=data['historical1011_infrastructure_failed'],
        resource_profile=data.get('profile'),resource_capacity_envelope=data.get('resource_capacity_envelope'),
        resource_capacity_source=data.get('resource_capacity_source'),resource_profile_within_capacity=data.get('resource_profile_within_capacity'),
        owner_updated_utc=data['status'].get('updated_utc'), controller_stage=data['controller'].get('stage'),
        reason=reason, errors=list(data['errors']), Native_started=False, GPU_started=False,
        chat_sent=False, delivery_acknowledged=False, scientific_graph_checked=False)


def render(data, item, history):
    number = lambda x: '未知／未获准读取' if x is None else str(x)
    phase = ('完整1920已由实际1042独立审计及1043终态确认' if item['release_complete'] else
        '部分进度，尚未通过完整1042独立审计；非最终比较结果') if item['admitted'] else '元数据准入未通过；不推算当前完成量'
    summary = clock_text(datetime.fromisoformat(item['created_utc']).timestamp())+'｜TEST '+number(item['total_completed'])+'/1920'
    summary += '｜本轮1041 infra '+number(item['current1041_infrastructure_failed'])+'｜历史902失败尝试 '+number(item['historical902_infrastructure_failed'])+'｜历史956失败尝试 '+number(item['historical956_infrastructure_failed'])+'｜历史1011失败尝试 '+number(item['historical1011_infrastructure_failed'])
    summary += '｜'+'；'.join(LABELS[m]+' '+number(v['successes'])+'/'+number(v['completed'])+'（未知'+number(v['unknown_native'])+'）' for m, v in item['methods'].items())
    lines = ['正式TEST结果更新（五层提速续跑版）', phase, '更新时间：'+item['created_utc'],
        '下一次固定相位更新：'+item['next_update_utc'], '实际1041状态时间：'+str(item['owner_updated_utc']), summary,
        '753原134 +902原450 +956原682 +1011已完成结果只导入一次；当前1041新增单列。902/956/1011历史infra尝试各自保留，不加到完成数或当前infra。',
        'nativeFalse是已保留任务结果；None为未知。motionFalse/None只作诊断，不筛成功率。',
        '954为已记录环境CPU点观察，非科学／资源放行；本报告未读case/raw/model/manifest或验证大图。',
        '对话文本仅已准备；实际对话发送后由root另行ack，报告器不会发送或自ack。']
    def quota(value):
        return ('未知／未获准读取' if value is None else
            '/'.join(str(value[k]) for k in ('max_native_slots','max_GPU_slots','max_CPU_preflight_slots','ready_limit')))
    lines.append('本轮实际冻结配额 Native/GPU/CPU/ready：'+quota(item.get('resource_profile'))+
        '；资格容量：'+quota(item.get('resource_capacity_envelope'))+'；资格来源：'+number(item.get('resource_capacity_source'))+
        '。容量为独立资格证据上限，不代表瞬时占用、正式配额或机器极限。')
    if item['errors']: lines += ['读取／运行提示：'+'；'.join(item['errors'])]
    lines += ['', '任务｜CV+几何 成功/完成/未知｜Context+几何 成功/完成/未知｜PhysMani 成功/完成/未知']
    for task, label in TASK_LABELS.items():
        cells = []
        for m in METHODS:
            cells.append('未知／未获准读取' if not data['admitted'] else f"{data['success'][task,m]}/{data['completed'][task,m]}/{data['unknown_native'][task,m]}")
        lines.append(label+'｜'+'｜'.join(cells))
    lines += ['', '冻结旧1014真实历史（来自已核SHA的退休快照；不代表当前读取成功；原文件保留）']
    for entry in data.get('legacy_history', []):
        value = entry['record']
        if isinstance(value.get('summary'),str):
            message=value['summary']
        else:
            message=str(value.get('created_utc'))+' 完成'+number(value.get('total_completed'))+'/1920'
            message+='｜当时1011 infra '+number(value.get('current1011_infrastructure_failed'))
            message+='｜历史956失败尝试 '+number(value.get('historical956_infrastructure_failed'))
            message+='｜历史902失败尝试 '+number(value.get('historical902_infrastructure_failed'))
            for method in METHODS:
                old=value.get('methods',{}).get(method,{})
                message+='｜'+LABELS[method]+' '+number(old.get('successes'))+'/'+number(old.get('completed'))
        lines.append('[1014真实历史] '+message)
    lines += ['', '1044观察历史']
    lines += ['[1044观察] '+str(h.get('created_utc'))+' '+str(h.get('reason'))+' 完成'+number(h.get('total_completed')) for h in history]
    return '\n'.join(lines)+'\n', summary+'\n'+phase


def atomic(path, payload):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name('.'+path.name+'.'+str(os.getpid())+'.tmp')
    with temp.open('xb') as stream:
        stream.write(payload); stream.flush(); os.fsync(stream.fileno())
    os.replace(temp, path)


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)+'\n').encode()


def require_fresh_observer(root):
    # Frozen source/lock/retirement may already exist in STATE. Runtime outputs
    # belong to one actual activation and are never replaced by a restart.
    for name in ('launch.json', 'status.json', 'history.jsonl', 'terminal.json', 'chat_ready.json', 'snapshots'):
        require(not (Path(root)/STATE/name).exists(), 'preserve an earlier1044 observation session: '+name)
    require(not (Path(root)/OUT).exists(), 'preserve an earlier1044 TXT')


def save_report(data, bundle, history, *, next_due, reason, now=None, root=None):
    now = time.time() if now is None else now; root = ROOT if root is None else Path(root)
    item = snapshot(data, now=now, next_due=next_due, reason=reason)
    item.update(source_config_sha256=bundle['config_sha256'], source_lock_sha256=bundle['lock_sha256'])
    item['observed_metadata_sha256'] = dict(data['reader'].checksums) if data.get('reader') else {}
    item['input_snapshots'] = {}
    if data.get('reader'):
        stamp = datetime.fromtimestamp(now, timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
        for original, leaf in ((RUNTIME+'/status.json', 'owner_status.json'), (CONTROLLER+'/status.json', 'controller_status.json')):
            if original in data['reader'].raw:
                name = STATE+'/snapshots/'+stamp+'/'+leaf
                require(not (root/name).exists(), 'preserve existing1044 metadata snapshot')
                atomic(root/name, data['reader'].raw[original])
                item['input_snapshots'][original] = dict(path=name, sha256=data['reader'].checksums[original])
    history = [*history, item]; text, chat = render(data, item, history)
    atomic(root/OUT, text.encode())
    atomic(root/STATE/'status.json', encoded(item))
    atomic(root/STATE/'history.jsonl', b''.join(encoded(row) for row in history))
    atomic(root/STATE/'chat_ready.json', encoded(dict(schema='r10_chat_ready_only_v1044',
        snapshot_id=item['created_utc'], created_utc=item['created_utc'], next_update_utc=item['next_update_utc'],
        text=chat, chat_sent=False, delivery_acknowledged=False)))
    # No ack file, delivery record, STOP, case, or legacy file is written.
    return history


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--source-lock-sha256')
    args = parser.parse_args(argv)
    if not args.execute:
        print(json.dumps(dict(active=False, reason='explicit --execute and separately frozen source material required', Native_started=False, GPU_started=False)))
        return 0
    require(valid_sha(args.source_lock_sha256), 'external observer small lock SHA required')
    import fcntl
    event = threading.Event()
    for number in (signal.SIGINT, signal.SIGTERM): signal.signal(number, lambda *_: event.set())
    stop = lambda: event.is_set() or any((ROOT/name).exists() for name in STOP_PATHS)
    if stop(): return 0
    # Admission/retirement are read before creating the reporter runtime/lock.
    bundle = load_source(args.source_lock_sha256, stop=stop)
    data = collect(bundle, stop=stop)
    state = ROOT/STATE; state.mkdir(parents=True, exist_ok=True)
    with (state/'observer.lock').open('a+') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        require_fresh_observer(ROOT)
        atomic(state/'launch.json', encoded(dict(pid=os.getpid(), created_utc=datetime.now(timezone.utc).isoformat(),
            source_sha256=bundle['config']['source_bindings'][SOURCE]['sha256'],
            source_config_sha256=bundle['config_sha256'], source_lock_sha256=bundle['lock_sha256'],
            observer_only=True, Native_started=False, GPU_started=False)))
        history, first, signature = [], True, None
        due = phase_deadline(time.time())
        while not stop():
            if not first:
                try: data = collect(bundle, stop=stop)
                except ObservationStopped: break
                except (OSError, ValueError, TypeError, KeyError, AttributeError, IndexError) as error: data = observation_error(data, error)
            marker = digest(dict(admitted=data['admitted'], errors=data['errors'], failed=data['failed'], release=data['release_complete']))
            now = time.time(); changed = signature is not None and signature != marker
            if first or changed or now >= due or data['release_complete']:
                due = phase_deadline(now)
                reason = '完整1042及1043终态确认' if data['release_complete'] else '启动快照' if first else '状态／准入变化' if changed else '固定30分钟更新'
                history = save_report(data, bundle, history, next_due=due, reason=reason, now=now)
                first = False
            signature = marker
            if data['release_complete']: break
            event.wait(min(10, max(.1, due-time.time())))
        atomic(state/'terminal.json', encoded(dict(complete=True, observer_only=True,
            final_TEST_independently_released=data['release_complete'], user_or_signal_stop=stop(),
            Native_started=False, GPU_started=False)))
    return 0


if __name__ == '__main__':
    sys.exit(main())
