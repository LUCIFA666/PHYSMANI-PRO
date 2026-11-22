"""Unactivated six-TRAIN-case resource canary for a separately frozen profile.

Default execution is CPU-only, but even that requires an enabled, externally
pinned control and exclusive idle resources after the old TEST naturally drains.
--execute-native consumes all six clean CPU receipts without retry/overwrite.
This runner never signs an independent resource qualification or selects by SR.
Import performs no native import, resource probe, lock acquisition or data read.
"""
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from contextlib import ExitStack, contextmanager
from datetime import datetime, timezone
import argparse
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SOURCE = 'scripts/run_concurrent_TRAIN_resource_canary_v903_r10.py'
RUNTIME = 'runtime/framework/concurrent_TRAIN_resource_canary_v903'
OUTPUT = 'artifacts/framework/multitask_r10/concurrent_TRAIN_resource_canary_v903/development_demo_only'
INVENTORY = 'runtime/framework/train_executor_failure_inventory_v790.json'
TASK = 'pick_moving_target_on_the_table'
RAWS = ('episode0', 'episode1')
METHODS = ('cv_geometry', 'context_geometry', 'physmani')
WORKER = 'scripts/run_task_executor_smoke_author_outcome_v743_r10.py'
BOOTSTRAP = 'scripts/run_batched_formal_driver_v901_r10.py'
BASE_LOCK = 'artifacts/framework/multitask_r10/author_outcome_qualification_v745_setup/frozen_inputs.json'
MANIFEST = 'artifacts/framework/protocol/train_manifest.json'
CONTROL_SCHEMA = 'r10_concurrent_TRAIN_resource_canary_control_v903'
LOCK_SCHEMA = 'r10_concurrent_TRAIN_resource_canary_inputs_v903'
PLAN_SCHEMA = 'r10_concurrent_TRAIN_resource_plan_v903'
SUMMARY_SCHEMA = 'r10_concurrent_TRAIN_resource_run_summary_v903'
AUDITOR = 'scripts/audit_concurrent_TRAIN_resource_canary_v904_r10.py'
PROFILE = dict(schema='r10_frozen_concurrent_native_resource_profile_v902',
               max_native_slots=4, max_GPU_slots=2, max_CPU_preflight_slots=2, ready_limit=8)
LOCKS = ('runtime/framework/ordered_resume_VAL_TEST_v775.lock',
         'runtime/framework/ordered_env_resume_VAL_TEST_v827.lock',
         'runtime/framework/fixed_TRAIN104_geometry_native.lock')
PREVIOUS = 'runtime/framework/formal_TEST_after_environment_resume_v753'
CONTROLLER = 'runtime/framework/ordered_env_resume_VAL_TEST_v827'
PREVIOUS_PATHS = dict(previous_launch=PREVIOUS + '/launch.json', previous_status=PREVIOUS + '/status.json',
    previous_terminal=PREVIOUS + '/terminal.json', previous_DRAIN=PREVIOUS + '/DRAIN',
    previous_controller_launch=CONTROLLER + '/launch.json', previous_controller_status=CONTROLLER + '/status.json',
    previous_controller_failure=CONTROLLER + '/failure.json')
STOP_PATHS = ('runtime/framework/multitask_r10_STOP', PREVIOUS + '/STOP', CONTROLLER + '/STOP',
    'runtime/framework/ordered_resume_VAL_TEST_v775/STOP', 'runtime/framework/formal_VAL_resume_v772/STOP',
    'runtime/framework/formal_VAL_env_resume_v823/STOP', RUNTIME + '/STOP')
SOURCES = (SOURCE, BOOTSTRAP, 'our_mani/multitask/frozen_lookup_batch_v900.py', WORKER,
    'scripts/run_formal_case_author_outcome_v751_r10.py', 'our_mani/multitask/formal_suite_admission_v728.py',
    'our_mani/multitask/rml_case_preflight_v742.py', 'scripts/run_task_executor_smoke_author_suite16_v744_r10.py',
    'scripts/run_local_sim.sh', 'scripts/run_aligned_gpu.sh', 'scripts/run_frozen_system_environment_v820_r10.py',
    'scripts/run_environment_resumed_formal_VAL_owner_v823_r10.py', 'scripts/run_native_resource_owner_v19_r10.py',
    'scripts/audit_concurrent_TRAIN_resource_canary_v904_r10.py', 'our_mani/multitask/evaluation.py',
    'our_mani/multitask/author_outcome_receipt_v749.py', 'our_mani/multitask/common_rml_protocol_v530.py')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(payload):
    return hashlib.sha256(payload).hexdigest()


def valid_pointer(value):
    return (isinstance(value, dict) and set(value) == {'path', 'sha256'}
        and isinstance(value.get('path'), str) and bool(value['path'])
        and not Path(value['path']).is_absolute() and '..' not in Path(value['path']).parts
        and isinstance(value.get('sha256'), str) and len(value['sha256']) == 64
        and not set(value['sha256']) - set('0123456789abcdef'))


def point(path):
    path = (ROOT / path).resolve(strict=True)
    require(path.is_relative_to(ROOT), 'project-only canary evidence required')
    return dict(path=str(path.relative_to(ROOT)), sha256=sha(path.read_bytes()))


def read_pointer(pointer):
    require(valid_pointer(pointer), 'explicit immutable project evidence required')
    path = (ROOT / pointer['path']).resolve(strict=True)
    require(path.is_relative_to(ROOT) and str(path.relative_to(ROOT)) == pointer['path'], 'canonical evidence path required')
    payload = path.read_bytes()
    require(sha(payload) == pointer['sha256'], 'canary evidence SHA changed: ' + pointer['path'])
    return json.loads(payload)


def atomic_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # These are immutable one-time evidence leaves; no previous bytes replaced.
    with path.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def option(argv, flag):
    require(isinstance(argv, list) and argv.count(flag) == 1 and argv.index(flag) + 1 < len(argv),
            'one original TRAIN option required: ' + flag)
    return argv[argv.index(flag) + 1]


def ticks(pid):
    return int(Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[19])


def alive(value):
    require(type(value.get('pid')) is int and type(value.get('start_ticks')) is int,
            'actual previous PID/start required')
    try:
        fields = Path(f"/proc/{value['pid']}/stat").read_text().rsplit(')', 1)[1].split()
        return fields[0] not in ('Z', 'X') and int(fields[19]) == value['start_ticks']
    except FileNotFoundError:
        return False


def stopped(event):
    return event.is_set() or any((ROOT / name).exists() for name in STOP_PATHS)


def require_previous_exit(control, metadata, is_alive):
    """Pure metadata contract, deliberately does not require an827 terminal."""
    for key, path in PREVIOUS_PATHS.items():
        require(valid_pointer(control.get(key)) and control[key]['path'] == path,
                'real immutable historical drain evidence required: ' + key)
    previous, controller = metadata['previous_launch'], metadata['previous_controller_launch']
    require(not is_alive(previous) and not is_alive(controller), 'old753/827 must both have exited')
    status, terminal = metadata['previous_status'], metadata['previous_terminal']
    require(status.get('split') == 'test' and status.get('expected') == 1920 and status.get('failed') == {}
        and status.get('active') == [] and status.get('checking') == []
        and status.get('drain_requested') is True and status.get('stop_requested') is False,
        'old753 must naturally drain without interrupted or failed cases')
    count = len(status.get('completed', {}))
    require(0 < count < 1920 and terminal.get('complete') is False and terminal.get('drained') is True
        and terminal.get('drain_requested') is True and terminal.get('stop_requested') is False
        and terminal.get('expected') == 1920 and terminal.get('completed') == count
        and terminal.get('failed') == 0 and terminal.get('automatic_retry') is False,
        'preserve the genuine partial drain terminal, not a fabricated completion')
    cs, failure = metadata['previous_controller_status'], metadata['previous_controller_failure']
    require(cs.get('stage') == 'requires_concrete_repair' and cs.get('automatic_retry') is False
        and failure.get('complete') is False and failure.get('user_stop') is False
        and failure.get('automatic_retry') is False,
        'preserve actual827 failure/status after planned drain, not a user stop or final terminal')
    require(previous.get('split') == 'test' and previous.get('policy_sha256') == control['policy_sha256']
        and previous.get('author_outcome_protocol') == control['author_outcome_protocol']
        and terminal.get('policy_sha256') == control['policy_sha256'], 'historical policy/protocol changed')
    return dict(completed=count, launches=[dict(pid=v['pid'], start_ticks=v['start_ticks'])
                for v in (previous, controller)], pointers={key: control[key] for key in PREVIOUS_PATHS})


def implementation_from_sources(sources):
    names = (BOOTSTRAP, 'our_mani/multitask/frozen_lookup_batch_v900.py',
        'scripts/run_formal_case_author_outcome_v751_r10.py', 'our_mani/multitask/formal_suite_admission_v728.py',
        'our_mani/multitask/rml_case_preflight_v742.py', WORKER,
        'scripts/run_task_executor_smoke_author_suite16_v744_r10.py')
    require(set(sources) == set(names) and all(valid_pointer(sources[name]) and sources[name]['path'] == name for name in names),
            'exact seven-source901 implementation required')
    return dict(schema='r10_resource_execution_implementation_v902', kind='batched901',
        entrypoint=sources[BOOTSTRAP], lookup_batch=sources[names[1]], original_formal_driver=sources[names[2]],
        original_case_worker=sources[WORKER], original_native_runtime=sources[names[6]], sources=sources)


def load_control(control_path, lock_path, expected_lock_sha):
    """Small explicitly scoped metadata/source lock, never the107k data graph."""
    control_path, lock_path = (ROOT / control_path, ROOT / lock_path)
    require(all(p.resolve(strict=True).is_relative_to(ROOT / 'artifacts/framework/multitask_r10') for p in (control_path, lock_path)),
            'separately frozen artifact canary control required')
    payload = lock_path.read_bytes()
    require(isinstance(expected_lock_sha, str) and sha(payload) == expected_lock_sha, 'externally pinned903 lock SHA required')
    lock = json.loads(payload)
    require(lock.get('schema') == LOCK_SCHEMA and lock.get('execution_enabled') is True, '903 is unactivated without an enabled lock')
    files = lock.get('files', {})
    require(isinstance(files, dict) and files, 'explicit small903 source/metadata input map required')
    canonical = {}
    for name, expected in files.items():
        path = (ROOT / name).resolve(strict=True)
        require(path.is_relative_to(ROOT) and str(path) not in canonical, 'unique project input mapping required')
        canonical[str(path)] = expected
    expected = canonical.get(str(control_path.resolve()))
    require(isinstance(expected, str), '903 control must be frozen by its input lock')
    control = read_pointer(dict(path=str(control_path.resolve().relative_to(ROOT)), sha256=expected))
    require(control.get('schema') == CONTROL_SCHEMA and control.get('execution_enabled') is True
        and control.get('split') == 'train' and control.get('task') == TASK and control.get('raw_episode_ids') == list(RAWS)
        and control.get('resource_profile') == PROFILE and control.get('training_weights_updated') is False
        and control.get('formal_score_contribution') is False and control.get('TEST_read') is False,
        'explicit unchanged fixed six-TRAIN-case resource scope required')
    checksum = control.get('common_RML_profile_sha256')
    require(isinstance(checksum, str) and len(checksum) == 64 and not set(checksum) - set('0123456789abcdef'),
            'explicit frozen common-RML profile SHA required')
    require(valid_pointer(control.get('resource_qualifier')) and control['resource_qualifier']['path'] == AUDITOR,
            'independent904 qualifier must be frozen before execution')
    require(valid_pointer(control.get('inventory')) and control['inventory']['path'] == INVENTORY, 'fixed790 TRAIN metadata required')
    train = control.get('training_bootstrap', {})
    require(set(train) == {'control', 'lock'} and all(valid_pointer(v) for v in train.values()), 'separate immutable901 TRAIN control/lock required')
    implementation = control.get('execution_implementation', {})
    require(implementation == implementation_from_sources(implementation.get('sources', {})), 'shared902 implementation descriptor changed')
    allowed = {str(control_path.resolve()), *(str((ROOT / name).resolve()) for name in SOURCES), str((ROOT / INVENTORY).resolve())}
    pointers = [*train.values(), control['resource_qualifier'], *(control.get(k) for k in PREVIOUS_PATHS),
        *(control.get(k) for k in ('environment_manifest', 'environment_wrapper', 'environment_namespace_proof'))]
    for pointer in pointers:
        require(valid_pointer(pointer), 'complete historical/environment/bootstrap bindings required')
        allowed.add(str((ROOT / pointer['path']).resolve(strict=True)))
    # Only already recorded six TRAIN CPU receipts may supplement this map.
    cpu_pointers = control.get('previous_CPU_receipts', [])
    require(isinstance(cpu_pointers, list) and len(cpu_pointers) == 6, 'six original TRAIN CPU metadata pointers required')
    for pointer in cpu_pointers:
        require(valid_pointer(pointer) and pointer['path'].startswith('artifacts/framework/multitask_r10/author_outcome_qualification_v745_setup/')
                and pointer['path'].endswith('_CPU_preflight.json'), 'original745 CPU metadata only')
        allowed.add(str((ROOT / pointer['path']).resolve(strict=True)))
    require(len({p['path'] for p in cpu_pointers}) == 6, 'six distinct original CPU metadata records required')
    require(set(canonical) == allowed, '903 lock must contain exactly declared small source/metadata inputs; no raw/model/TEST source scan')
    for name, expected in canonical.items():
        require(isinstance(expected, str) and sha(Path(name).read_bytes()) == expected, '903 frozen source/metadata changed: ' + name)
    for pointer in [*pointers, *cpu_pointers, control['inventory'], *implementation['sources'].values()]:
        require(canonical.get(str((ROOT / pointer['path']).resolve())) == pointer['sha256'], 'pointer differs from pinned903 map')
    return control, dict(control=point(control_path), lock=point(lock_path), files=canonical)


def selected_rows(inventory, cpu_receipts, control):
    """Pure selection of two sources regardless of any earlier native outcome."""
    rows = [r for r in inventory.get('cases', []) if r.get('task') == TASK and r.get('raw_episode_id') in RAWS and r.get('method') in METHODS]
    require(len(rows) == 6 and {(r['raw_episode_id'], r['method']) for r in rows} == {(raw, method) for raw in RAWS for method in METHODS},
            'exact two original TRAIN sources times three methods required')
    result = []
    for raw in RAWS:
        for method in METHODS:
            old = next(r for r in rows if (r['raw_episode_id'], r['method']) == (raw, method))
            prefix = ['bash', 'scripts/run_aligned_gpu.sh' if method == 'physmani' else 'scripts/run_local_sim.sh', WORKER]
            argv = list(old['provenance']['actual_argv'])
            require(argv[:3] == prefix and argv.count('--development-demo') == 1 and '--check' not in argv, 'real original743 TRAIN route required')
            for key, expected in (('--task', TASK), ('--raw-episode-id', raw), ('--split', 'train'), ('--method', method),
                    ('--manifest', MANIFEST), ('--frozen-inputs', BASE_LOCK)):
                require(option(argv, key) == expected, 'frozen original TRAIN route differs: ' + key)
            previous = old['provenance']['CPU_receipt']
            require(previous in control['previous_CPU_receipts'], 'six original CPU receipts must be pinned')
            proof = cpu_receipts[previous['path']]
            identity = proof.get('identity', {})
            require(proof.get('complete') is True and proof.get('native_started') is False and proof.get('GPU_started') is False
                and proof.get('actual_split') == 'train' and identity.get('split') == 'train'
                and identity.get('method') == method and identity.get('task') == TASK and identity.get('source_raw_episode_id') == raw
                and identity.get('source_row_sha256') == old['source_row_sha256']
                and proof.get('author_outcome_protocol') == control['author_outcome_protocol'], 'original TRAIN CPU identity/protocol required')
            out = str(Path(OUTPUT) / TASK / raw / method)
            argv[argv.index('--out') + 1] = out
            outer = [*argv[:2], BOOTSTRAP, '--batch-control', control['training_bootstrap']['control']['path'],
                '--batch-lock', control['training_bootstrap']['lock']['path'], '--batch-lock-sha256', control['training_bootstrap']['lock']['sha256'],
                '--entry', 'train', '--', *argv[3:]]
            result.append(dict(job_id='concurrent_TRAIN_v903_' + raw + '_' + method, raw=raw, method=method,
                argv=outer, physical_argv=argv, out=out, expected_identity=identity, source_row_sha256=old['source_row_sha256'],
                previous_CPU_receipt=previous, previous_inventory_case=old['case']))
    return result


def require_CPU_receipt(row, receipt, control):
    domain = receipt.get('component_domain', {})
    require(receipt.get('complete') is True and receipt.get('native_started') is False and receipt.get('GPU_started') is False
        and receipt.get('actual_split') == 'train' and receipt.get('identity') == row['expected_identity']
        and receipt.get('author_outcome_protocol') == control['author_outcome_protocol']
        and domain.get('purpose') == 'train_qualification' and domain.get('split') == 'train'
        and domain.get('method') == row['method'] and domain.get('raw_episode_id') == row['raw'] and domain.get('task') == TASK,
        'genuine unchanged CPU-only TRAIN identity/domain required')
    batch = receipt.get('batch_execution', {})
    require(batch.get('schema') == 'r10_scoped_frozen_lookup_bootstrap_receipt_v901' and batch.get('complete') is True
        and batch.get('entry') == 'train' and batch.get('check_only') is True
        and batch.get('actual_outer_argv') == row['argv'][2:] + ['--check']
        and batch.get('forwarded_original_argv') == row['physical_argv'][2:] + ['--check']
        and batch.get('control') == control['training_bootstrap']['control'] and batch.get('lock') == control['training_bootstrap']['lock']
        and batch.get('sources') == control['execution_implementation']['sources']
        and batch.get('scientific_policy_changed') is False and batch.get('full_verify_inputs_calls_retained') is True,
        'real TRAIN901 outer/forwarded/source/control CPU evidence required')
    require([p.get('phase') for p in batch.get('phases', [])] == ['original742_prepare']
        and all(p.get('boundary_recheck_complete') is True and p.get('metrics', {}).get('complete') is True
                and p['metrics'].get('closed') is True for p in batch['phases']), 'closed successful TRAIN preparation batch required')


def ordered_native_rows(rows):
    by_key = {(r['raw'], r['method']): r for r in rows}
    order = [('episode0', 'physmani'), ('episode1', 'physmani'), ('episode0', 'cv_geometry'),
             ('episode0', 'context_geometry'), ('episode1', 'cv_geometry'), ('episode1', 'context_geometry')]
    require(len(rows) == len(by_key) == 6 and set(by_key) == set(order), 'fixed six-case scheduling grid required')
    return [by_key[key] for key in order]


def schedule_native(rows, *, launch, poll, finish, identify, interrupt, pause, should_stop, clock, observe=None):
    """CPU-testable stable4/2 scheduler; failures drain active jobs, no retries."""
    pending, active, results, events = ordered_native_rows(rows), {}, {}, []
    failed = False
    def sample(kind, row=None):
        living = [dict(job_id=r['job_id'], method=r['method'], **identify(handle))
                  for r, handle in active.values() if poll(handle) is None]
        native, gpu = len(living), sum(v['method'] == 'physmani' for v in living)
        require(native <= 4 and gpu <= 2, 'live native/GPU requests exceeded frozen4/2 profile')
        events.append(dict(event=kind, job_id=None if row is None else row['job_id'], monotonic_ns=clock(),
            active=living, native_slots=native, GPU_slots=gpu))
        if observe is not None:
            observe(living)
    try:
        while pending or active:
            if should_stop():
                for _, handle in active.values():
                    interrupt(handle)
                pending.clear()
                failed = True
            while pending and not failed and len(active) < 4:
                row = pending.pop(0)
                require(sum(r['method'] == 'physmani' for r, _ in active.values()) + int(row['method'] == 'physmani') <= 2,
                        'GPU quota exceeded before launch')
                handle = launch(row)
                active[row['job_id']] = (row, handle)
                sample('launched', row)
            for key, (row, handle) in list(active.items()):
                if poll(handle) is None:
                    continue
                result = finish(row, handle)
                results[key] = result
                del active[key]
                failed = failed or result.get('complete') is not True
                sample('finished', row)
            if failed:
                pending.clear()
            if active:
                pause()
        return dict(complete=not failed and len(results) == 6, results=results, events=events,
            observed_native_peak=max((v['native_slots'] for v in events), default=0),
            observed_GPU_peak=max((v['GPU_slots'] for v in events), default=0))
    except BaseException:
        for _, handle in active.values():
            interrupt(handle)
        raise


def process_stat(pid):
    fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
    return dict(pid=pid, state=fields[0], ppid=int(fields[1]), start_ticks=int(fields[19]))


def attribute_compute_pid(pid, jobs, stat_reader=process_stat):
    """Only a stable, actually read ancestry chain may associate a compute PID."""
    result = dict(pid=pid, pid_start_ticks=None, job_id=None, launcher_pid=None, launcher_start_ticks=None)
    try:
        first = dict(stat_reader(pid))
        result['pid_start_ticks'] = first['start_ticks']
        chain, seen, current = [], set(), first
        while current['pid'] > 0 and current['pid'] not in seen:
            seen.add(current['pid'])
            chain.append(dict(pid=current['pid'], start_ticks=current['start_ticks'], ppid=current['ppid'], state=current['state']))
            matches = [j for j in jobs if j['pid'] == current['pid'] and j['start_ticks'] == current['start_ticks']]
            if matches:
                require(len(matches) == 1 and matches[0]['method'] == 'physmani', 'compute PID ancestry must uniquely match a PhysMani request')
                last = stat_reader(pid)
                require(all(last[key] == first[key] for key in ('pid', 'ppid', 'start_ticks'))
                    and first['state'] not in ('Z', 'X') and last['state'] not in ('Z', 'X'),
                    'compute PID changed while sampling')
                job = matches[0]
                result.update(job_id=job['job_id'], launcher_pid=job['pid'], launcher_start_ticks=job['start_ticks'])
                break
            if current['ppid'] <= 0:
                break
            current = stat_reader(current['ppid'])
        result['ancestor_chain'] = chain
    except (OSError, ValueError, IndexError) as error:
        result['ancestry_error'] = repr(error)
    return result


def sample_resources(jobs, *, query=None, stat_reader=process_stat, clock=time.monotonic_ns, cpu_reader=None):
    """Future read-only hardware observation; errors are unknown, never zero."""
    if query is None:
        def query(arguments):
            return subprocess.check_output(['nvidia-smi', *arguments], text=True, timeout=5, stderr=subprocess.STDOUT)
    record = dict(monotonic_ns=clock(), gpu=None, gpus=[], compute_apps=[], GPU_observation_complete=False)
    try:
        raw = query(['--query-gpu=uuid,memory.used,memory.total,utilization.gpu', '--format=csv,noheader,nounits'])
        for columns in csv.reader(raw.splitlines()):
            if not columns:
                continue
            require(len(columns) == 4, 'four actual GPU inventory columns required')
            uuid, used, total, utilization = (v.strip() for v in columns)
            gpu = dict(uuid=uuid, memory_used_mib=int(used), memory_total_mib=int(total), utilization_gpu_percent=int(utilization))
            require(0 <= gpu['memory_used_mib'] <= gpu['memory_total_mib'] and 0 <= gpu['utilization_gpu_percent'] <= 100, 'finite actual GPU counters required')
            record['gpus'].append(gpu)
        require(len(record['gpus']) == 1, 'single explicitly observed GPU required for this qualification')
        record['gpu'] = record['gpus'][0]
        raw_apps = query(['--query-compute-apps=pid,used_gpu_memory,gpu_uuid', '--format=csv,noheader,nounits'])
        for columns in csv.reader(raw_apps.splitlines()):
            if not columns:
                continue
            require(len(columns) == 3, 'three actual compute application columns required')
            pid, memory, gpu_uuid = (v.strip() for v in columns)
            app = attribute_compute_pid(int(pid), jobs, stat_reader)
            app.update(used_memory_mib=int(memory), gpu_uuid=gpu_uuid)
            require(app['used_memory_mib'] >= 0 and gpu_uuid == record['gpu']['uuid'], 'actual GPU application device/counter required')
            record['compute_apps'].append(app)
        record['GPU_observation_complete'] = True
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        record['GPU_observation_error'] = repr(error)
    if cpu_reader is not None:
        try:
            record['host_CPU_RAM'] = cpu_reader()
        except (OSError, ValueError, IndexError) as error:
            record['host_CPU_RAM_error'] = repr(error)
    return record


def read_host_CPU_RAM():
    cpu = [int(value) for value in Path('/proc/stat').read_text().splitlines()[0].split()[1:]]
    memory = {}
    for line in Path('/proc/meminfo').read_text().splitlines():
        key, value = line.split(':', 1)
        if key in ('MemTotal', 'MemAvailable'):
            memory[key + '_kib'] = int(value.split()[0])
    return dict(cpu_aggregate_jiffies=cpu, cpu_idle_jiffies=cpu[3] + cpu[4], **memory)


class ResourceSampler:
    """Only constructed/started by the future native opt-in, never on import."""
    def __init__(self, path):
        self.path, self.jobs = Path(path), []
        self.lock, self.stop_event, self.thread = threading.Lock(), threading.Event(), None
        self.failure = None

    def observe(self, jobs):
        with self.lock:
            self.jobs = [dict(job) for job in jobs]

    def __enter__(self):
        self.stream = self.path.open('x')
        def loop():
            try:
                previous_cpu, number = None, 0
                while True:
                    with self.lock:
                        jobs = [dict(job) for job in self.jobs]
                    value = sample_resources(jobs, cpu_reader=read_host_CPU_RAM if number % 3 == 0 else None)
                    cpu = value.get('host_CPU_RAM')
                    if cpu is not None:
                        if previous_cpu is not None:
                            # Linux guest jiffies already occur in user/nice counters.
                            total = sum(cpu['cpu_aggregate_jiffies'][:8]) - sum(previous_cpu['cpu_aggregate_jiffies'][:8])
                            idle = cpu['cpu_idle_jiffies'] - previous_cpu['cpu_idle_jiffies']
                            if total > 0 and 0 <= idle <= total:
                                cpu['aggregate_percent_since_previous_CPU_sample'] = 100 * (total - idle) / total
                        previous_cpu = cpu
                    self.stream.write(json.dumps(value, allow_nan=False) + '\n')
                    self.stream.flush()
                    number += 1
                    if self.stop_event.wait(1.):
                        break
            except Exception as error:
                self.failure = repr(error)
        self.thread = threading.Thread(target=loop, name='TRAIN_resource_observer_v903', daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop_event.set()
        self.thread.join(timeout=12)
        require(not self.thread.is_alive(), 'read-only resource observer did not exit cleanly')
        self.stream.close()
        require(self.failure is None, 'resource observer failed: ' + str(self.failure))
        return False


def resource_inventory():
    from unittest.mock import patch
    from scripts import run_environment_resumed_formal_VAL_owner_v823_r10 as owner
    with patch.object(owner, 'NATIVE_PREFIXES', (*owner.NATIVE_PREFIXES, 'run_batched_formal_driver_', 'run_concurrent_', 'continue_concurrent_TEST_')):
        return owner.process_visibility()


def require_empty_inventory(value):
    require(value.get('host_visibility_available') is True and value.get('same_uid_inventory_incomplete') is False
        and value.get('live_project_native_processes') == [] and value.get('ancestor_chain_complete') is True,
        'complete host inventory and empty project native/controller resources required')


@contextmanager
def resource_guard(control, event):
    require(not stopped(event), 'STOP before any903 resource acquisition')
    metadata = {key: read_pointer(control[key]) for key in PREVIOUS_PATHS}
    previous = require_previous_exit(control, metadata, alive)
    with ExitStack() as stack:
        for name in LOCKS:
            stream = stack.enter_context((ROOT / name).open('a+'))
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # Recheck identities after acquisition; no stale metadata may grant a run.
        require_previous_exit(control, metadata, alive)
        inventory = resource_inventory()
        require_empty_inventory(inventory)
        require(not stopped(event), 'STOP after exclusive resource inventory')
        from scripts.run_environment_resumed_formal_VAL_owner_v823_r10 import require_frozen_environment
        environment = require_frozen_environment(control)
        yield dict(previous_exit_evidence=previous, resource_inventory=inventory, frozen_environment=environment)


def launch_child(row, check, event, environment):
    require(not stopped(event), 'STOP immediately before child launch')
    require(os.readlink('/proc/self/ns/mnt') == environment['current_mount_namespace'], 'actual820 mount namespace changed')
    leaf = ROOT / RUNTIME / ('CPU' if check else 'native') / row['job_id']
    leaf.mkdir(parents=True, exist_ok=False)
    argv = row['argv'] + (['--check'] if check else [])
    stdout, stderr = (leaf / 'stdout.log').open('xb'), (leaf / 'stderr.log').open('xb')
    process = None
    try:
        require(not stopped(event), 'STOP immediately before Popen')
        began = time.monotonic_ns()
        process = subprocess.Popen(argv, cwd=ROOT, env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'),
            stdout=stdout, stderr=stderr, start_new_session=True)
        start = ticks(process.pid)
        atomic_new(leaf / 'launch.json', dict(created_utc=utc(), argv=argv,
            physical_argv=row['physical_argv'] + (['--check'] if check else []), launcher_pid=process.pid,
            launcher_start_ticks=start, owner_pid=os.getpid(), owner_start_ticks=ticks(os.getpid()),
            check_only=check, native_slots=0 if check else 1, GPU_slots=0 if check else int(row['method'] == 'physmani'),
            started_monotonic_ns=began, inherited_frozen_environment=environment,
            current_mount_namespace=os.readlink('/proc/self/ns/mnt')))
        return dict(process=process, stdout=stdout, stderr=stderr, leaf=leaf, row=row, check=check,
                    started=began, start_ticks=start, timed_out=False, interrupted=False, environment=environment)
    except BaseException:
        if process is not None and process.poll() is None:
            from scripts.run_native_resource_owner_v19_r10 import interrupt_owned_tree
            interrupt_owned_tree(process)
        stdout.close()
        stderr.close()
        raise


def interrupt_child(handle):
    from scripts.run_native_resource_owner_v19_r10 import interrupt_owned_tree
    if handle['process'].poll() is None:
        interrupt_owned_tree(handle['process'])


def poll_child(handle, event):
    if handle['process'].poll() is None:
        handle['interrupted'] = stopped(event)
        handle['timed_out'] = time.monotonic_ns() - handle['started'] > 1800 * 10**9
        if handle['interrupted'] or handle['timed_out']:
            interrupt_child(handle)
    return handle['process'].poll()


def finish_child(row, handle, control, event):
    code = handle['process'].wait()
    handle['stdout'].close()
    handle['stderr'].close()
    result = dict(complete=code == 0 and not handle['timed_out'] and not handle['interrupted'] and not stopped(event),
        returncode=code, timed_out=handle['timed_out'], interrupted=handle['interrupted'] or stopped(event),
        automatic_retry=False, finished_utc=utc(), started_monotonic_ns=handle['started'], finished_monotonic_ns=time.monotonic_ns(),
        launcher_pid=handle['process'].pid, launcher_start_ticks=handle['start_ticks'], launch=point(handle['leaf'] / 'launch.json'))
    try:
        if result['complete'] and handle['check']:
            proof = json.loads((handle['leaf'] / 'stdout.log').read_bytes())
            require_CPU_receipt(row, proof, control)
            atomic_new(handle['leaf'] / 'CPU_preflight.json', proof)
            result['CPU_preflight'] = point(handle['leaf'] / 'CPU_preflight.json')
        elif result['complete']:
            case_path = ROOT / row['out'] / 'case.json'
            case = json.loads(case_path.read_bytes())
            require(case.get('complete') is True and case.get('identity') == row['expected_identity']
                and case.get('cleanup_completed') is True and type(case.get('official_task_success')) is bool
                and case.get('split') == 'train' and case.get('component_domain', {}).get('purpose') == 'train_qualification'
                and case.get('author_outcome_protocol') == control['author_outcome_protocol'], 'clean same-identity native TRAIN case required')
            result.update(case=point(case_path), batch_driver_receipt=point(ROOT / row['out'] / 'batch_driver_receipt.json'),
                native_success=case['official_task_success'], cleanup_completed=True)
    except Exception as error:
        result.update(complete=False, error=repr(error))
    atomic_new(handle['leaf'] / 'terminal.json', result)
    return result


def run_CPU(rows, control, event, environment):
    results, pending, active = {}, list(rows), {}
    def one(row):
        handle = launch_child(row, True, event, environment)
        try:
            while poll_child(handle, event) is None:
                event.wait(.2)
            return finish_child(row, handle, control, event)
        finally:
            interrupt_child(handle)
    with ThreadPoolExecutor(max_workers=2) as pool:
        failed = False
        while pending or active:
            while pending and len(active) < 2 and not failed and not stopped(event):
                row = pending.pop(0)
                active[pool.submit(one, row)] = row
            if not active:
                break
            done, _ = wait(active, timeout=.2, return_when=FIRST_COMPLETED)
            for future in done:
                row = active.pop(future)
                result = future.result()
                results[row['job_id']] = result
                failed = failed or result.get('complete') is not True
            if failed or stopped(event):
                pending.clear()
    require(len(results) == 6 and all(v.get('complete') is True for v in results.values()), 'all six real CPU checks must pass before native opt-in')
    return results


def require_saved_CPU(rows, control):
    for row in rows:
        leaf = ROOT / RUNTIME / 'CPU' / row['job_id']
        launch, terminal = (json.loads((leaf / name).read_bytes()) for name in ('launch.json', 'terminal.json'))
        require(terminal.get('complete') is True and terminal.get('returncode') == 0 and terminal.get('timed_out') is False
            and terminal.get('interrupted') is False and terminal.get('launch') == point(leaf / 'launch.json')
            and terminal.get('CPU_preflight') == point(leaf / 'CPU_preflight.json')
            and launch.get('argv') == row['argv'] + ['--check'] and launch.get('check_only') is True
            and launch.get('native_slots') == launch.get('GPU_slots') == 0, 'six clean original actual CPU launch/terminal proofs required')
        require_CPU_receipt(row, json.loads((leaf / 'CPU_preflight.json').read_bytes()), control)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--control', required=True)
    parser.add_argument('--lock', required=True)
    parser.add_argument('--lock-sha256', required=True)
    parser.add_argument('--execute-native', action='store_true')
    args = parser.parse_args()
    event = threading.Event()
    for number in (signal.SIGINT, signal.SIGTERM):
        signal.signal(number, lambda *_: event.set())
    require(not stopped(event), 'STOP before903 admission')
    control, binding = load_control(args.control, args.lock, args.lock_sha256)
    with resource_guard(control, event) as resource:
        inventory = read_pointer(control['inventory'])
        cpu = {p['path']: read_pointer(p) for p in control['previous_CPU_receipts']}
        rows = selected_rows(inventory, cpu, control)
        plan = dict(schema=PLAN_SCHEMA, task=TASK, rows=rows, resource_profile=PROFILE,
            execution_implementation=control['execution_implementation'], policy_sha256=control['policy_sha256'],
            author_outcome_protocol=control['author_outcome_protocol'], common_RML_profile_sha256=control['common_RML_profile_sha256'],
            training_bootstrap=control['training_bootstrap'], inputs=binding)
        runtime = ROOT / RUNTIME
        if args.execute_native:
            require(json.loads((runtime / 'plan.json').read_bytes()) == plan, 'unchanged original CPU plan required')
            require_saved_CPU(rows, control)
            require(not (runtime / 'native').exists() and not (runtime / 'run_summary.json').exists()
                and all(not (ROOT / r['out']).exists() for r in rows), 'never reuse or overwrite a native attempt')
        else:
            runtime.mkdir(parents=True, exist_ok=False)
            atomic_new(runtime / 'plan.json', plan)
        phase = 'native' if args.execute_native else 'CPU'
        owner_path = runtime / (phase + '_owner_launch.json')
        atomic_new(owner_path, dict(created_utc=utc(), argv=list(sys.argv), pid=os.getpid(), start_ticks=ticks(os.getpid()),
            source_sha256=point(SOURCE)['sha256'], control=binding['control'], lock=binding['lock'], plan=point(runtime / 'plan.json'), **resource))
        if not args.execute_native:
            results = run_CPU(rows, control, event, resource['frozen_environment'])
            atomic_new(runtime / 'CPU_owner_terminal.json', dict(complete=True, results=results, native_started=False,
                GPU_started=False, automatic_retry=False, finished_utc=utc(), independent_audit_complete=False))
            print(json.dumps(dict(complete=True, phase='CPU', case_count=6, native_started=False, GPU_started=False)))
            return
        samples = runtime / 'native_resource_samples.jsonl'
        with ResourceSampler(samples) as sampler:
            run = schedule_native(rows, launch=lambda row: launch_child(row, False, event, resource['frozen_environment']),
                poll=lambda handle: poll_child(handle, event), finish=lambda row, handle: finish_child(row, handle, control, event),
                identify=lambda h: dict(pid=h['process'].pid, start_ticks=h['start_ticks']),
                interrupt=interrupt_child, pause=lambda: event.wait(.2), should_stop=lambda: stopped(event),
                clock=time.monotonic_ns, observe=sampler.observe)
        atomic_new(runtime / 'native_resource_events.json', dict(schema='r10_live_launcher_resource_events_v903', events=run.pop('events'),
            scope='actual live native launcher requests, not sampled hardware utilization or simultaneous robot ticks'))
        final_inventory = resource_inventory()
        require_empty_inventory(final_inventory)
        # Revalidate both the small frozen inputs and current820 environment.
        require(load_control(args.control, args.lock, args.lock_sha256)[1] == binding, '903 source/control changed during canary')
        from scripts.run_environment_resumed_formal_VAL_owner_v823_r10 import require_frozen_environment
        final_environment = require_frozen_environment(control)
        summary = dict(schema=SUMMARY_SCHEMA, split='train', task=TASK, case_count=len(run['results']),
            plan=point(runtime / 'plan.json'), owner=point(owner_path), control=binding['control'], lock=binding['lock'],
            resource_events=point(runtime / 'native_resource_events.json'), resource_samples=point(samples), resource_profile=PROFILE,
            execution_implementation=control['execution_implementation'], execution_entry=control['execution_implementation'],
            policy_sha256=control['policy_sha256'], author_outcome_protocol=control['author_outcome_protocol'],
            common_RML_profile_sha256=control['common_RML_profile_sha256'],
            cases={row['job_id']: point(runtime / 'native' / row['job_id'] / 'terminal.json') for row in rows if row['job_id'] in run['results']},
            cleanup_completed=True, final_resource_inventory=final_inventory, frozen_environment=final_environment,
            infrastructure_failed=sum(v.get('complete') is not True for v in run['results'].values()),
            not_launched_job_ids=[r['job_id'] for r in rows if r['job_id'] not in run['results']],
            resource_bounds_verified=False, scheduler_resource_bounds_verified=True,
            independent_audit_complete=False, formal_score_contribution=False,
            TEST_read=False, training_weights_updated=False, automatic_retry=False, finished_utc=utc(), **run)
        atomic_new(runtime / 'run_summary.json', summary)
        atomic_new(runtime / 'native_owner_terminal.json', dict(complete=run['complete'] and not stopped(event),
            pid=os.getpid(), start_ticks=ticks(os.getpid()), source_sha256=point(SOURCE)['sha256'],
            launch=point(owner_path), run_summary=point(runtime / 'run_summary.json'),
            resource_samples=point(samples), resource_events=point(runtime / 'native_resource_events.json'),
            automatic_retry=False, independent_audit_complete=False, finished_utc=utc()))
        require(run['complete'] and not stopped(event), 'resource canary incomplete; preserve all evidence, no implicit retry')
        print(json.dumps(dict(complete=True, phase='native', case_count=6, independent_audit_complete=False,
            out=str((runtime / 'run_summary.json').relative_to(ROOT)))))


if __name__ == '__main__':
    main()
