"""Independently qualify twelve original TRAIN resource cases; never run a simulator.

Reads explicitly named CPU/native evidence and sampled hardware observations.
Native task failures remain valid resource witnesses. A runner summary cannot
sign its own qualification. Sampling proves observed headroom, not a universal
hardware ceiling or invariance of wall-clock-bounded planners under contention.
"""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys
import threading

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SOURCE = 'scripts/audit_concurrent_TRAIN_resource_canary_v1005_r10.py'
OUTPUT = 'runtime/framework/concurrent_TRAIN_resource_qualification_v1005.json'
AUDIT_RUNTIME = 'runtime/framework/concurrent_TRAIN_resource_audit_v1005'


def require(value, message):
    if not value:
        raise ValueError(message)


def integer(value, minimum=0):
    return type(value) is int and value >= minimum


def valid_pointer(value):
    return (isinstance(value, dict) and set(value) == {'path', 'sha256'}
            and isinstance(value.get('path'), str) and bool(value['path'])
            and not Path(value['path']).is_absolute() and '..' not in Path(value['path']).parts
            and isinstance(value.get('sha256'), str) and len(value['sha256']) == 64
            and not set(value['sha256']) - set('0123456789abcdef'))


def clean_terminal(value):
    require(value.get('complete') is True and type(value.get('returncode')) is int
            and value['returncode'] == 0 and value.get('timed_out') is False
            and value.get('interrupted') is False and value.get('automatic_retry') is False,
            'clean actual one-time process terminal required')


def require_batch_metrics(value):
    names = ('maps', 'declared_files', 'watched_directories', 'pointer_checks', 'full_graph_boundary_checks',
             'used_path_end_checks', 'used_external', 'actual_SHA_checks', 'strong_stat_reuses', 'exact_JSON_payloads')
    require(isinstance(value, dict) and value.get('schema') == 'r10_scoped_frozen_lookup_metrics_v900'
            and value.get('closed') is True and value.get('complete') is True
            and all(integer(value.get(k)) for k in names), 'complete actual lookup-batch counters required')
    require(value['maps'] >= 1 and value['declared_files'] >= value['maps']
            and value['watched_directories'] >= 1 and value['pointer_checks'] >= 1
            and 1 <= value['used_path_end_checks'] <= value['pointer_checks']
            and value['full_graph_boundary_checks'] == 2 * value['maps']
            and value['actual_SHA_checks'] == value['pointer_checks'] + value['used_path_end_checks']
            and value['strong_stat_reuses'] <= value['actual_SHA_checks']
            and value['used_external'] <= value['pointer_checks'], 'lookup counters do not cover all graphs/used inputs')


def interval_audit(rows, launches, terminals, events, profile):
    """Independent bounds and simultaneous-live witness from actual child IDs."""
    keys = {r['job_id'] for r in rows}
    require(len(rows) == len(keys) == 12 and set(launches) == set(terminals) == keys,
            'exactly twelve unique case launch/terminal intervals required')
    methods = {r['job_id']: r['method'] for r in rows}
    intervals, identities = {}, set()
    for key in keys:
        launch, terminal = launches[key], terminals[key]
        clean_terminal(terminal)
        pid, start = launch.get('launcher_pid'), launch.get('launcher_start_ticks')
        begin, end = launch.get('started_monotonic_ns'), terminal.get('finished_monotonic_ns')
        require(integer(pid, 1) and integer(start, 1) and (pid, start) not in identities,
                'twelve distinct actual launcher PID/start identities required')
        require(integer(begin, 1) and integer(end, 1) and begin < end
                and terminal.get('started_monotonic_ns') == begin
                and terminal.get('launcher_pid') == pid and terminal.get('launcher_start_ticks') == start,
                'terminal must cover its exact actual launcher interval')
        identities.add((pid, start)); intervals[key] = (begin, end, pid, start)
    # A filtered event list cannot conceal over-allocation. Independently
    # sweep every actual child lifetime, ending equal-time intervals first.
    endpoints = [(begin, 1, key) for key, (begin, end, pid, start) in intervals.items()]
    endpoints += [(end, -1, key) for key, (begin, end, pid, start) in intervals.items()]
    active_intervals = set(); interval_native_peak = interval_gpu_peak = 0; interval_overlap = False
    for now, change, key in sorted(endpoints):
        if change < 0:
            active_intervals.remove(key)
        else:
            active_intervals.add(key)
        count = len(active_intervals); gpu_count = sum(methods[k] == 'physmani' for k in active_intervals)
        require(count <= profile['max_native_slots'] and gpu_count <= profile['max_GPU_slots'],
                'actual child lifetime intervals exceed the frozen resource quota')
        interval_native_peak = max(interval_native_peak, count)
        interval_gpu_peak = max(interval_gpu_peak, gpu_count)
        interval_overlap |= count == profile['max_native_slots'] and gpu_count == profile['max_GPU_slots']
    require(interval_overlap, 'actual child intervals do not contain a eight-launcher/three-PhysMani overlap')
    require(isinstance(events, list) and events, 'actual immutable live-launcher events required')
    previous = -1; peak_native = peak_gpu = 0; simultaneous = False
    for event in events:
        now = event.get('monotonic_ns'); living = event.get('active')
        require(integer(now, 1) and now >= previous and isinstance(living, list), 'ordered actual event clock required')
        previous = now
        active_keys = [v.get('job_id') for v in living]
        require(len(active_keys) == len(set(active_keys)) and set(active_keys) <= keys, 'event contains foreign/duplicate active request')
        for value in living:
            key = value['job_id']; begin, end, pid, start = intervals[key]
            require(value.get('method') == methods[key] and value.get('pid') == pid
                    and value.get('start_ticks') == start and begin <= now <= end,
                    'live event must bind an actual child PID and lifetime')
        native = len(living); gpu = sum(methods[k] == 'physmani' for k in active_keys)
        require(type(event.get('native_slots')) is int and event['native_slots'] == native
                and type(event.get('GPU_slots')) is int and event['GPU_slots'] == gpu,
                'event slot counts differ from actual unique live requests')
        require(native <= profile['max_native_slots'] and gpu <= profile['max_GPU_slots'], 'resource request quota exceeded')
        peak_native, peak_gpu = max(peak_native, native), max(peak_gpu, gpu)
        simultaneous |= native == profile['max_native_slots'] and gpu == profile['max_GPU_slots']
    require(simultaneous, 'must observe all eight live launchers including three PhysMani requests together')
    return dict(intervals=intervals, observed_native_peak=interval_native_peak, observed_GPU_peak=interval_gpu_peak,
                event_observed_native_peak=peak_native, event_observed_GPU_peak=peak_gpu,
                independently_swept_all_child_intervals=True, overlap_verified=True)


def CPU_interval_audit(rows, launches, terminals, profile):
    """Sweep all actual CPU child lifetimes, not executor submission counts."""
    keys = {row['job_id'] for row in rows}
    require(len(rows) == len(keys) == 12 and set(launches) == set(terminals) == keys,
            'all twelve actual CPU intervals required')
    intervals, identities, endpoints = {}, set(), []
    for key in keys:
        launch, terminal = launches[key], terminals[key]
        clean_terminal(terminal)
        pid, start = launch.get('launcher_pid'), launch.get('launcher_start_ticks')
        begin, end = launch.get('started_monotonic_ns'), terminal.get('finished_monotonic_ns')
        require(integer(pid, 1) and integer(start, 1) and (pid, start) not in identities
            and integer(begin, 1) and integer(end, 1) and begin < end
            and terminal.get('started_monotonic_ns') == begin
            and terminal.get('launcher_pid') == pid and terminal.get('launcher_start_ticks') == start
            and launch.get('check_only') is True and type(launch.get('native_slots')) is int
            and launch['native_slots'] == 0 and type(launch.get('GPU_slots')) is int and launch['GPU_slots'] == 0,
            'CPU interval must bind real distinct PID/start and zero Native/GPU reservation')
        identities.add((pid, start)); intervals[key] = (begin, end, pid, start)
        endpoints.extend(((begin, 1, key), (end, -1, key)))
    live, peak = set(), 0
    for _, change, key in sorted(endpoints):
        live.add(key) if change > 0 else live.remove(key)
        require(len(live) <= profile['max_CPU_preflight_slots'], 'actual CPU intervals exceed frozen quota')
        peak = max(peak, len(live))
    require(peak == profile['max_CPU_preflight_slots'] == 4, 'actual four-CPU overlap required')
    return dict(intervals=intervals, observed_CPU_preflight_peak=peak, overlap_verified=True,
                independently_swept_all_child_intervals=True,
                scope='actual live CPU-only preflight processes; not simultaneous Native supply or host utilization')


def joint_interval_audit(rows, native, CPU, first_wave):
    """An actual shared interval, not separately achieved resource peaks."""
    methods = {r['job_id']: r['method'] for r in rows}
    first = {r['job_id'] for r in first_wave}
    late = set(methods) - first
    ni, ci = native['intervals'], CPU['intervals']
    require(len(first) == 8 and len(late) == 4 and set(ni) == set(ci) == set(methods),
            'exact twelve disjoint CPU-once/Native-once lifetimes required')
    require(max(ci[k][1] for k in first) <= min(ni[k][0] for k in first)
        and max(ni[k][0] for k in first) <= min(ci[k][0] for k in late)
        and all(ci[k][1] <= ni[k][0] for k in late),
        'first8 CPU before Native; late4 CPU under that load before their own Native')
    endpoints = [(v[0], 1, 'native', k) for k, v in ni.items()]
    endpoints += [(v[1], -1, 'native', k) for k, v in ni.items()]
    endpoints += [(v[0], 1, 'CPU', k) for k, v in ci.items()]
    endpoints += [(v[1], -1, 'CPU', k) for k, v in ci.items()]
    live = {'native': set(), 'CPU': set()}; witness = None
    for now, change, phase, key in sorted(endpoints):
        live[phase].add(key) if change > 0 else live[phase].remove(key)
        require(not live['native'] & live['CPU'], 'a job may not execute CPU and Native at once')
        counts = (len(live['native']), sum(methods[k] == 'physmani' for k in live['native']), len(live['CPU']))
        require(counts[0] <= 8 and counts[1] <= 3 and counts[2] <= 4, 'joint resource quota exceeded')
        if counts == (8, 3, 4) and witness is None:
            witness = dict(monotonic_ns=now, native_job_ids=sorted(live['native']),
                           CPU_job_ids=sorted(live['CPU']), native_slots=8, GPU_slots=3, CPU_preflight_slots=4)
    require(witness is not None, 'separate resource peaks cannot qualify: actual joint8/3/4 interval required')
    return dict(complete=True, independently_swept_all_child_intervals=True,
                native_slots=8, GPU_slots=3, CPU_preflight_slots=4, witness=witness,
                first_wave_CPU_job_ids=sorted(first), late_CPU_job_ids=sorted(late),
                each_job_CPU_checks=1, each_job_Native_attempts=1,
                simultaneous_robot_ticks_or_hardware_saturation_claimed=False)


def hardware_audit(samples, rows, launches, terminals, cases, *, minimum_headroom_mib=2048, joint_intervals=None):
    """Require real simultaneous compute contexts; reserved GPU slots do not suffice."""
    methods = {r['job_id']: r['method'] for r in rows}
    require(isinstance(samples, list) and samples, 'actual hardware samples required')
    valid, unknown, minimum, witness = 0, 0, None, None
    previous = -1
    for sample in samples:
        now = sample.get('monotonic_ns')
        until = sample.get('sampled_until_monotonic_ns')
        require(integer(now, 1) and now >= previous and integer(until, 1) and until >= now
            and sample.get('GPU_queries_atomic_snapshot_claimed') is False,
            'ordered bounded non-atomic hardware observation clock required')
        previous = now
        if sample.get('GPU_observation_complete') is not True:
            unknown += 1
            continue
        gpu = sample.get('gpu', {})
        used, total = gpu.get('memory_used_mib'), gpu.get('memory_total_mib')
        require(integer(used) and integer(total, 1) and used <= total and isinstance(gpu.get('uuid'), str),
                'real GPU memory inventory required')
        utilization = gpu.get('utilization_gpu_percent')
        require(integer(utilization) and utilization <= 100, 'actual bounded GPU utilization counter required')
        headroom = total - used
        require(headroom >= minimum_headroom_mib, 'sampled GPU headroom below frozen resource margin')
        minimum = headroom if minimum is None else min(minimum, headroom)
        valid += 1
        apps = sample.get('compute_apps')
        require(isinstance(apps, list), 'actual compute application observation required')
        associated = {}; primary = {}; seen_pids = set()
        for app in apps:
            pid, start = app.get('pid'), app.get('pid_start_ticks')
            require(integer(pid, 1) and pid not in seen_pids, 'unique actual compute PID required')
            seen_pids.add(pid)
            key = app.get('job_id')
            if key is None:
                continue  # Explicitly unknown; never supplies a qualifying context.
            require(key in methods and methods[key] == 'physmani' and integer(start, 1), 'compute process must belong to an exact PhysMani case')
            launch, terminal = launches[key], terminals[key]
            require(launch['started_monotonic_ns'] <= until and now < terminal['finished_monotonic_ns']
                    and app.get('launcher_pid') == launch['launcher_pid']
                    and app.get('launcher_start_ticks') == launch['launcher_start_ticks']
                    and app.get('gpu_uuid') == gpu['uuid']
                    and integer(app.get('used_memory_mib'), 1), 'actual compute process/case/device lifetime differs')
            chain = app.get('ancestor_chain', [])
            require(isinstance(chain, list) and chain
                    and all(isinstance(node, dict) and integer(node.get('pid'), 1)
                            and integer(node.get('start_ticks'), 1) and integer(node.get('ppid'))
                            and node.get('state') not in (None, 'Z', 'X') for node in chain)
                    and len({node['pid'] for node in chain}) == len(chain)
                    and chain[0]['pid'] == pid and chain[0]['start_ticks'] == start
                    and chain[-1]['pid'] == launch['launcher_pid']
                    and chain[-1]['start_ticks'] == launch['launcher_start_ticks']
                    and all(a['ppid'] == b['pid'] for a, b in zip(chain, chain[1:])),
                    'real compute process ancestry must reach its exact live launcher')
            associated.setdefault(key, []).append(dict(pid=pid, start_ticks=start))
            # A process may start/exit during the two-query transaction. It
            # remains visible diagnostic evidence, but only a main process
            # whose entire interval contains the query bracket can qualify.
            if (pid == cases[key].get('process_id')
                and launch['started_monotonic_ns'] <= now <= until < terminal['finished_monotonic_ns']):
                primary[key] = dict(pid=pid, start_ticks=start)
        # Do not interpret a non-atomic list spanning a job turnover as an
        # instantaneous slot count. All request intervals are swept separately.
        under_joint_load = True
        if joint_intervals is not None:
            native_intervals, CPU_intervals = joint_intervals
            nl = [key for key, v in native_intervals.items() if v[0] <= now <= until < v[1]]
            cl = [key for key, v in CPU_intervals.items() if v[0] <= now <= until < v[1]]
            under_joint_load = len(nl) == 8 and sum(methods[key] == 'physmani' for key in nl) == 3 and len(cl) == 4
        if len(primary) == 3 and witness is None and under_joint_load:
            witness = dict(monotonic_ns=now, sampled_until_monotonic_ns=until,
                           GPU_queries_atomic_snapshot_claimed=False,
                           jobs=primary, all_owned_compute_processes=associated,
                           gpu=gpu, headroom_mib=headroom)
    require(valid > 0 and witness is not None and unknown == 0, 'no actual three-case GPU compute overlap observed; unknown is not zero')
    return dict(actual_three_compute_contexts_observed=True, witness=witness, valid_samples=valid,
                unknown_samples=unknown, minimum_sampled_headroom_mib=minimum,
                required_headroom_mib=minimum_headroom_mib,
                actual_three_compute_contexts_during_joint8_3_4_observed=joint_intervals is not None,
                GPU_queries_atomic_snapshot_claimed=False,
                continuous_hardware_bound_claimed=False)


class Evidence:
    def __init__(self, root):
        self.root = Path(root).resolve(); self.inputs = {}

    def payload(self, path, expected=None):
        path = self.root / path
        canonical = path.resolve(strict=True)
        require(canonical.is_relative_to(self.root) and canonical == path.absolute(), 'canonical explicit project evidence required')
        payload = canonical.read_bytes(); actual = hashlib.sha256(payload).hexdigest()
        require(expected is None or actual == expected, 'immutable canary evidence SHA changed')
        name = str(canonical.relative_to(self.root))
        if name in self.inputs:
            require(self.inputs[name] == actual, 'consumed evidence changed during independent audit')
        self.inputs[name] = actual
        return payload

    def json(self, path, expected=None):
        return json.loads(self.payload(path, expected))

    def pointer(self, value, expected_path=None):
        require(valid_pointer(value), 'explicit evidence pointer with exact lowercase SHA required')
        require(expected_path is None or value['path'] == expected_path, 'evidence path is outside exact canary allowlist')
        return self.json(value['path'], value['sha256'])

    def recheck(self):
        for path, expected in list(self.inputs.items()):
            self.payload(path, expected)


def audit(control, binding, runner):
    """Consume twelve explicit fresh TRAIN cases; no manifest/raw/model file reads."""
    from our_mani.multitask.evaluation import validate_completed_case
    from our_mani.multitask.author_outcome_receipt_v749 import require_sampling_case
    from our_mani.multitask.common_rml_protocol_v530 import require_attestation
    from scripts.audit_environment_resumed_formal_VAL_v824_r10 import require_environment_receipt
    evidence = Evidence(ROOT); runtime = runner.RUNTIME
    plan = evidence.json(runtime + '/plan.json')
    expected_rows = runner.selected_rows(runner.read_pointer(control['inventory']),
        {p['path']: runner.read_pointer(p) for p in control['previous_CPU_receipts']}, control,
        runner.read_pointer(control['original_TRAIN_plan']))
    require(plan.get('schema') == runner.PLAN_SCHEMA and plan.get('rows') == expected_rows
            and plan.get('inputs') == binding and plan.get('resource_profile') == runner.PROFILE
            and plan.get('execution_implementation') == control['execution_implementation'], 'actual original twelve-case frozen plan changed')
    for key in ('policy_sha256', 'author_outcome_protocol', 'common_RML_profile_sha256', 'training_bootstrap'):
        require(plan.get(key) == control[key], 'plan/control changed: ' + key)
    rows = plan['rows']; row_keys = {r['job_id'] for r in rows}
    first_wave = runner.ordered_native_rows(rows)[:8]
    first_keys = {r['job_id'] for r in first_wave}; late_keys = row_keys - first_keys
    owners = {}
    for phase in ('CPU', 'native'):
        owner = evidence.json(runtime + '/' + phase + '_owner_launch.json'); owners[phase] = owner
        require(not runner.alive(owner), 'the actual canary owner must exit before independent audit')
        argv = owner.get('argv', [])
        require(argv and (ROOT / argv[0]).resolve() == ROOT / runner.SOURCE
                and (ROOT / runner.option(argv, '--control')).resolve() == ROOT / binding['control']['path']
                and (ROOT / runner.option(argv, '--lock')).resolve() == ROOT / binding['lock']['path']
                and runner.option(argv, '--lock-sha256') == binding['lock']['sha256']
                and ('--execute-native' in argv) is (phase == 'native')
                and owner.get('source_sha256') == runner.point(runner.SOURCE)['sha256']
                and owner.get('control') == binding['control'] and owner.get('lock') == binding['lock'], 'actual separate CPU/native owner provenance required')
        evidence.pointer(owner['plan'], runtime + '/plan.json')
        runner.require_empty_inventory(owner.get('resource_inventory', {}))
        environment = require_environment_receipt(owner.get('frozen_environment'), control, owner_pid=owner['pid'])
        require(environment['observer_start_ticks'] == owner['start_ticks'],
                'owner must capture its own actual820 PID/start environment')
    cpu_owner_terminal = evidence.json(runtime + '/CPU_owner_terminal.json')
    require(cpu_owner_terminal.get('complete') is True and cpu_owner_terminal.get('native_started') is False
            and cpu_owner_terminal.get('GPU_started') is False
            and set(cpu_owner_terminal.get('results', {})) == first_keys
            and cpu_owner_terminal.get('pid') == owners['CPU']['pid']
            and cpu_owner_terminal.get('start_ticks') == owners['CPU']['start_ticks']
            and cpu_owner_terminal.get('source_sha256') == runner.point(runner.SOURCE)['sha256']
            and cpu_owner_terminal.get('control') == binding['control'] and cpu_owner_terminal.get('lock') == binding['lock'],
            'eight first-wave actual CPU checks required')
    evidence.pointer(cpu_owner_terminal['launch'], runtime + '/CPU_owner_launch.json')
    summary = evidence.json(runtime + '/run_summary.json')
    require(summary.get('schema') == runner.SUMMARY_SCHEMA and summary.get('complete') is True
            and summary.get('independent_audit_complete') is False and summary.get('resource_bounds_verified') is False
            and summary.get('TEST_read') is False and summary.get('formal_score_contribution') is False
            and summary.get('training_weights_updated') is False
            and summary.get('infrastructure_failed') == 0 and type(summary.get('infrastructure_failed')) is int
            and summary.get('not_launched_job_ids') == [] and summary.get('case_count') == 12
            and set(summary.get('late_CPU_results', {})) == late_keys
            and set(summary.get('cases', {})) == row_keys and set(summary.get('results', {})) == row_keys,
            'runner must retain twelve clean outcomes without signing independent qualification')
    for key in ('resource_profile', 'execution_implementation', 'policy_sha256', 'author_outcome_protocol', 'common_RML_profile_sha256'):
        require(summary.get(key) == plan[key], 'runner summary differs from its real plan: ' + key)
    evidence.pointer(summary['plan'], runtime + '/plan.json')
    evidence.pointer(summary['owner'], runtime + '/native_owner_launch.json')
    native_owner_terminal = evidence.json(runtime + '/native_owner_terminal.json')
    require(native_owner_terminal.get('complete') is True and native_owner_terminal.get('independent_audit_complete') is False
            and native_owner_terminal.get('pid') == owners['native']['pid']
            and native_owner_terminal.get('start_ticks') == owners['native']['start_ticks']
            and native_owner_terminal.get('source_sha256') == runner.point(runner.SOURCE)['sha256']
            and native_owner_terminal.get('automatic_retry') is False,
            'actual native owner terminal required')
    evidence.pointer(native_owner_terminal['launch'], runtime + '/native_owner_launch.json')
    evidence.pointer(native_owner_terminal['run_summary'], runtime + '/run_summary.json')
    launches, terminals, cases, CPU_launches, CPU_terminals = {}, {}, {}, {}, {}
    for row in rows:
        key = row['job_id']
        for phase in ('CPU', 'native'):
            leaf = runtime + '/' + phase + '/' + key
            launch = evidence.json(leaf + '/launch.json'); terminal = evidence.json(leaf + '/terminal.json')
            clean_terminal(terminal); evidence.pointer(terminal['launch'], leaf + '/launch.json')
            check = phase == 'CPU'; suffix = ['--check'] if check else []
            actual_owner = owners['native' if not check or key in late_keys else 'CPU']
            require(launch.get('argv') == row['argv'] + suffix and launch.get('physical_argv') == row['physical_argv'] + suffix
                    and launch.get('owner_pid') == actual_owner['pid'] and launch.get('owner_start_ticks') == actual_owner['start_ticks']
                    and launch.get('check_only') is check and launch.get('native_slots') == (0 if check else 1)
                    and launch.get('GPU_slots') == (int(row['method'] == 'physmani') if not check else 0)
                    and launch.get('inherited_frozen_environment') == actual_owner['frozen_environment']
                    and launch.get('current_mount_namespace') == actual_owner['frozen_environment']['current_mount_namespace'],
                    'actual child argv/owner/namespace differs from frozen original TRAIN route')
            evidence.payload(leaf + '/stdout.log'); evidence.payload(leaf + '/stderr.log')
            if check:
                receipt = evidence.pointer(terminal['CPU_preflight'], leaf + '/CPU_preflight.json')
                runner.require_CPU_receipt(row, receipt, control)
                recorded_CPU = summary['late_CPU_results'][key] if key in late_keys else cpu_owner_terminal['results'][key]
                require(json.loads(evidence.payload(leaf + '/stdout.log')) == receipt
                        and recorded_CPU == terminal, 'actual CPU stdout/owner terminal differs from receipt')
                batch = receipt['batch_execution']
                CPU_launches[key], CPU_terminals[key] = launch, terminal
            else:
                require(summary['results'][key] == terminal, 'runner result differs from actual immutable terminal')
                evidence.pointer(summary['cases'][key], leaf + '/terminal.json')
                case = evidence.pointer(terminal['case'], row['out'] + '/case.json')
                require(case.get('split') == 'train' and case.get('cleanup_completed') is True
                        and case.get('component_domain', {}).get('purpose') == 'train_qualification', 'clean original TRAIN-only case required')
                validate_completed_case(case, row['expected_identity'])
                require_sampling_case(case, control['author_outcome_protocol'], block=True)
                require_attestation(case.get('common_author_RML_parameters', {}), control['common_RML_profile_sha256'])
                require(case.get('Value_NGFF_FASTER_used') is False and case.get('TEST_used_for_fitting') is False
                    and 'policy_sha256' in case.get('full16_frozen_execution', {})
                    and case['full16_frozen_execution']['policy_sha256'] is None
                    and case['full16_frozen_execution'].get('source_row_sha256') == row['source_row_sha256']
                    and case.get('author_outcome_protocol') == control['author_outcome_protocol'],
                    'original TRAIN writer policy=None, fixed745 source/protocol; forbidden methods and TEST fitting excluded')
                require(terminal.get('native_success') is case['official_task_success'], 'nativeFalse outcome must be retained exactly')
                batch = evidence.pointer(terminal['batch_driver_receipt'], row['out'] + '/batch_driver_receipt.json')
                require(batch.get('case') == terminal['case'], 'bootstrap must bind its actual case')
                launches[key], terminals[key], cases[key] = launch, terminal, case
            require(batch.get('schema') == 'r10_scoped_frozen_lookup_bootstrap_receipt_v901'
                    and batch.get('complete') is True and batch.get('entry') == 'train' and batch.get('check_only') is check
                    and batch.get('actual_outer_argv') == row['argv'][2:] + suffix
                    and batch.get('forwarded_original_argv') == row['physical_argv'][2:] + suffix
                    and batch.get('control') == control['training_bootstrap']['control']
                    and batch.get('lock') == control['training_bootstrap']['lock']
                    and batch.get('sources') == control['execution_implementation']['sources']
                    and batch.get('scientific_policy_changed') is False
                    and batch.get('full_verify_inputs_calls_retained') is True,
                    'actual explicit901 TRAIN scope/source/argv proof required')
            require([p.get('phase') for p in batch.get('phases', [])] == ['original742_prepare'], 'actual single TRAIN prepare scope required')
            for phase_record in batch['phases']:
                require(phase_record.get('boundary_recheck_complete') is True, 'batch boundary check missing')
                require_batch_metrics(phase_record.get('metrics'))
    event_record = evidence.pointer(summary['resource_events'], runtime + '/native_resource_events.json')
    evidence.pointer(native_owner_terminal['resource_events'], runtime + '/native_resource_events.json')
    intervals = interval_audit(rows, launches, terminals, event_record.get('events'), runner.PROFILE)
    cpu_intervals = CPU_interval_audit(rows, CPU_launches, CPU_terminals, runner.PROFILE)
    joint = joint_interval_audit(rows, intervals, cpu_intervals, first_wave)
    sample_pointer = summary['resource_samples']
    require(valid_pointer(sample_pointer) and sample_pointer['path'] == runtime + '/native_resource_samples.jsonl', 'explicit hardware sample log required')
    samples = [json.loads(line) for line in evidence.payload(sample_pointer['path'], sample_pointer['sha256']).splitlines() if line]
    require(native_owner_terminal['resource_samples'] == sample_pointer, 'actual native owner must bind the hardware log')
    ready_events = [event for event in event_record.get('events', []) if event.get('event') == 'contexts_ready']
    require(len(ready_events) == 1 and summary.get('context_readiness_observed') is True
        and summary.get('qualification_blocked_reason') is None, 'one actual sampler-ready release event required')
    ready = ready_events[0].get('context_readiness', {})
    ready_sample = ready.get('hardware_sample')
    first_three = {row['job_id'] for row in first_wave[:3]}
    require(ready.get('complete') is True and set(ready.get('job_ids', [])) == first_three
        and ready_sample in samples and ready.get('monotonic_ns') == ready_sample.get('monotonic_ns')
        and ready_sample['sampled_until_monotonic_ns'] <= ready_events[0]['monotonic_ns']
        <= min(launches[row['job_id']]['started_monotonic_ns'] for row in first_wave[3:]),
        'real recorded three-context sample must precede geometry and late CPU release')
    hardware_audit([ready_sample], rows, launches, terminals, cases)
    hardware = hardware_audit(samples, rows, launches, terminals, cases,
                             joint_intervals=(intervals['intervals'], cpu_intervals['intervals']))
    runner.require_empty_inventory(summary.get('final_resource_inventory', {}))
    evidence.recheck()
    return dict(schema='r10_independent_concurrent_TRAIN_resource_qualification_v1005', complete=True, split='train',
        source_sha256=runner.point(SOURCE)['sha256'], resource_profile=runner.PROFILE,
        execution_entry=control['execution_implementation'], policy_sha256=control['policy_sha256'],
        author_outcome_protocol=control['author_outcome_protocol'], case_count=12, infrastructure_failed=0,
        cleanup_completed=True, resource_bounds_verified=True, overlap_verified=True,
        independent_audit_complete=True, formal_score_contribution=False, TEST_read=False,
        observed_native_peak=intervals['observed_native_peak'], observed_GPU_peak=intervals['observed_GPU_peak'],
        actual_CPU_preflight_peak=cpu_intervals['observed_CPU_preflight_peak'], CPU_preflight_overlap=cpu_intervals,
        joint_supply_overlap=joint,
        native_successes_diagnostic=sum(c['official_task_success'] for c in cases.values()),
        native_failures_retained=True, hardware_observation=hardware, launcher_overlap=intervals,
        inputs=evidence.inputs, control=binding['control'], lock=binding['lock'],
        raw_manifest_model_contents_read=False, wall_clock_contention_invariance_claimed=False,
        CPU_and_Native_simultaneous_supply_observed=True, automatic_retry=False,
        Value_NGFF_FASTER_used=False, training_weights_updated=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--control', required=True); parser.add_argument('--lock', required=True)
    parser.add_argument('--lock-sha256', required=True); parser.add_argument('--out', default=OUTPUT)
    args = parser.parse_args()
    require(args.out == OUTPUT, 'one-time exact independent resource qualification namespace required')
    from scripts import run_concurrent_TRAIN_resource_canary_v1005_r10 as runner
    control, binding = runner.load_control(args.control, args.lock, args.lock_sha256)
    require(control.get('resource_qualifier') == runner.point(SOURCE), 'actual independent auditor must have been frozen before canary')
    event = threading.Event()
    with runner.resource_guard(control, event) as actual:
        leaf = ROOT / AUDIT_RUNTIME; leaf.mkdir(parents=True, exist_ok=False)
        runner.atomic_new(leaf / 'launch.json', dict(created_utc=datetime.now(timezone.utc).isoformat(),
            pid=os.getpid(), start_ticks=runner.ticks(os.getpid()), argv=list(sys.argv),
            source_sha256=runner.point(SOURCE)['sha256'], control=binding['control'], lock=binding['lock'], **actual))
        result = audit(control, binding, runner)
        require(not runner.stopped(event), 'user STOP before publishing qualification')
        result.update(created_utc=datetime.now(timezone.utc).isoformat(), actual_independent_audit=runner.point(leaf / 'launch.json'))
        runner.atomic_new(ROOT / args.out, result)
        print(json.dumps(dict(complete=True, resource_profile=result['resource_profile'], case_count=12,
                              out=args.out, observed_native_peak=result['observed_native_peak'], observed_GPU_peak=result['observed_GPU_peak'])))


if __name__ == '__main__':
    main()
