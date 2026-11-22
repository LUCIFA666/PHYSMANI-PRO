"""One fixed TRAIN passive ring readiness diagnosis after diagnoses naturally complete.

Default invocation is inert. No preparation, queue continuation or Native work
is implicit. Exact pinned plan, candidate CPU evidence, actual predecessor
exit and a root technical-review record are required. CPU then Native, serial,
once per source; every outcome and infrastructure failure is retained.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
SOURCE = 'scripts/run_TRAIN_ring_readiness_canary_v1136_r10.py'
OWNER_TEST = 'tests/test_TRAIN_ring_readiness_canary_owner_v1136.py'
WORKER = 'scripts/diagnose_TRAIN_ring_readiness_v1135_r10.py'
WRAPPER = 'scripts/run_frozen_system_environment_v820_r10.py'
SIM = 'scripts/run_local_sim.sh'
NAME = 'TRAIN_ring_readiness_v1135'
SETUP = 'artifacts/framework/multitask_r10/' + NAME + '_setup/'
ART = 'artifacts/framework/multitask_r10/' + NAME + '/development_demo_only/observed_v1136'
CASE_RUNTIME = 'runtime/framework/' + NAME + '/cases/observed_v1136'
RUNTIME = 'runtime/framework/' + NAME + '/owner_v1136'
PLAN = SETUP + 'bounded_plan_v1136.json'
DIAGNOSTIC_LOCK = SETUP + 'source_overlay_lock.json'
ROOT_REVIEW = SETUP + 'root_review_v1136.json'
ENV_SHA = '1c33a2e35c1134b90aebe896da6176a997242e8aa10dd6c9b08c9e4d5730df30'
PLAN_SCHEMA = 'r10_bounded_one_TRAIN_ring_readiness_diagnostic_plan_v1136'
STATUS_SCHEMA = 'r10_bounded_one_TRAIN_ring_readiness_diagnostic_status_v1136'
REVIEW_SCHEMA = 'r10_fixed_one_TRAIN_ring_readiness_diagnostic_release_review_v1136'
EXIT_SCHEMA = 'r10_actual_natural_public_phase_canary_owner_exit_v1133'
PREDECESSOR = 'runtime/framework/TRAIN_public_phase_full16_v1132/owner_v1133'
PREDECESSOR_PLAN = 'artifacts/framework/multitask_r10/TRAIN_public_phase_full16_v1132_setup/bounded_plan_v1133.json'
PREDECESSOR_PLAN_SHA = '88f8f1a70d5ecee4d489ee812290c4ffb21be5f833f4c4446af7c219dad32e91'
PREDECESSOR_SOURCE = 'scripts/run_TRAIN_public_phase_canary_v1133_r10.py'
PREDECESSOR_SOURCE_SHA = '9b1c2fa245ba6b32138a59c9cfbefea54e7412cb54348f7af5c05b1d89de3ad5'
PREDECESSOR_COMPLETION = PREDECESSOR + '/exec_completion.json'
PREDECESSOR_CANARIES = (
    ('insert_onto_rotating_peg', 'episode101', 'cv_geometry'),
    ('beat_the_rotating_buzz', 'episode10', 'cv_geometry'),
)
TASKS = {'ring': ('insert_onto_rotating_peg',)}
METHODS = ('cv_geometry',)
CANARIES = PREDECESSOR_CANARIES[:1]
FIXED_SOURCES = [dict(task=task, raw_episode_id=raw, method=method) for task,raw,method in CANARIES]
DIAGNOSTIC_SCHEMA = 'r10_fixed_one_TRAIN_ring_readiness_diagnostic_v1135'
DIAGNOSTIC_PINS = {
    'our_mani/multitask/train_public_execution_trace_v1126.py': 'ffa610f07d45fafc1ef43785e66b86f111af20408ab650b9a042e4da122921bc',
    'tests/test_train_public_execution_trace_v1126.py': '7d64bc3d5eb519a56f0113c934f745a23086d9c12a9d9bb7f92371de0145d743',
}
PARENT_LOCK = 'artifacts/framework/multitask_r10/TRAIN_public_phase_full16_v1132_setup/source_overlay_lock.json'
PARENT_LOCK_SHA = '99a04a6434f90d588656a9b5b8b0f708e5c54848a80868703349f5cc3509d290'
PARENT_FACTOR = 'nominal_public_ring_seating_guard'



def require(value, message):
    if not value:
        raise ValueError('TRAIN1136: ' + message)


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),
        ensure_ascii=False,allow_nan=False).encode()).hexdigest()


def fresh(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def atomic(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp.' + str(os.getpid()))
    fresh(temporary, value)
    os.replace(temporary, path)


def pointer(name):
    return dict(path=name, sha256=sha(ROOT / name))


def overlay_module():
    for name, expected in DIAGNOSTIC_PINS.items():
        require(sha(ROOT/name) == expected,'frozen candidate source changed: '+name)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from our_mani.multitask import train_ring_readiness_overlay_v1135
    return train_ring_readiness_overlay_v1135


def build_jobs(roster):
    require(roster.get('split') == 'train', 'actual TRAIN manifest required')
    rows = []
    kinds = {task:kind for kind,tasks in TASKS.items() for task in tasks}
    for task, raw, method in CANARIES:
        parents = [r['raw_episode_id'] for r in roster['episodes'] if r['task'] == task][:4]
        require(len(parents) == 4 and len(set(parents)) == 4,
                'original first four unique TRAIN parents required')
        require(raw in parents and method in METHODS,
                'same fixed predecessor source must belong to original first-four geometry grid')
        kind = kinds[task]
        job = '_'.join((kind, task, raw, method))
        rows.append(dict(kind=kind, task=task, raw=raw, method=method,
            config='artifacts/framework/multitask_r10/post_training_qualification_v680_setup/' + task + '_ours_configuration.json',
            job_id=job, out=f'{ART}/{task}/{raw}/{method}', runtime=f'{CASE_RUNTIME}/{job}'))
    require(len(rows) == len({r['job_id'] for r in rows}) == 1, 'exactly one fixed diagnostic case required')
    return rows


def checked_reference(ref):
    require(type(ref) is dict and set(ref) == {'path','sha256'}
        and type(ref['path']) is str and not Path(ref['path']).is_absolute()
        and '..' not in Path(ref['path']).parts, 'canonical project evidence pointer required')
    require(sha(ROOT/ref['path']) == ref['sha256'], 'completed evidence changed: '+ref['path'])
    return ref


def process_not_live(pid, ticks):
    require(type(pid) is int and pid > 0 and type(ticks) is int and ticks > 0,
            'actual predecessor PID/start identity required')
    try:
        fields=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
    except (FileNotFoundError,ProcessLookupError):
        return True
    return int(fields[19]) != ticks or fields[0] in ('Z','X')


def predecessor_complete():
    """Require the actual naturally exited1133 owner and complete evidence."""
    require(sha(ROOT/PREDECESSOR_PLAN) == PREDECESSOR_PLAN_SHA
        and sha(ROOT/PREDECESSOR_SOURCE) == PREDECESSOR_SOURCE_SHA, 'frozen actual1133 plan/source required')
    names={key:PREDECESSOR+'/'+key+'.json' for key in ('launch','status','terminal')}
    launch=json.loads((ROOT/names['launch']).read_text())
    status=json.loads((ROOT/names['status']).read_text())
    require(launch.get('plan') == dict(path=PREDECESSOR_PLAN,sha256=PREDECESSOR_PLAN_SHA)
        and status.get('stage') == 'complete' and status.get('active') is None
        and status.get('failed') == [] and len(status.get('completed',[])) == 2
        and not status.get('error'), '1133 must naturally complete with no active child or infrastructure error')
    require([(r['task'],r['raw'],r['method']) for r in status['completed']] == list(PREDECESSOR_CANARIES),
            'exact original two diagnostic sources/order required')
    terminal=json.loads((ROOT/names['terminal']).read_text())
    require(terminal.get('complete') is True and terminal.get('completed') == 2
        and terminal.get('infrastructure_failed') == 0 and terminal.get('automatic_retry') is False,
        'actual natural1133 terminal required')
    pid,ticks=launch.get('pid'),launch.get('start_ticks')
    require(process_not_live(pid,ticks), '1133 owner still alive; candidate owner cannot coexist')
    completion=json.loads((ROOT/PREDECESSOR_COMPLETION).read_text())
    require(completion.get('schema') == EXIT_SCHEMA and completion.get('actual_tool_exit_observed') is True
        and completion.get('unified_exec_session') == 16893
        and type(completion.get('returncode')) is int and completion['returncode'] == 0
        and completion.get('owner_pid') == pid and completion.get('owner_start_ticks') == ticks
        and completion.get('owner_alive') is False and completion.get('natural_completion') is True
        and completion.get('native_cases_complete') == 2
        and completion.get('completed_case_infrastructure_failures') == 0
        and completion.get('plan') == launch['plan'] and completion.get('terminal') == pointer(names['terminal']),
        'root actual tool exit0/PID-death/natural completion receipt required')
    for row in status['completed']:
        checked_reference(row['case'])
        for key in ('CPU_check','native'):
            for field in ('launch','terminal','sidecar'): checked_reference(row[key][field])
        checked_reference(row['owned_hooks'])
    return dict(launch=pointer(names['launch']),status=pointer(names['status']),terminal=pointer(names['terminal']),
        actual_exec_completion=pointer(PREDECESSOR_COMPLETION),original_pid=pid,original_start_ticks=ticks,
        process_identity_not_live=True,natural_completion=True,prior_native_outcomes_relabelled=False)


def root_review(lock_reference, predecessor):
    value=json.loads((ROOT/ROOT_REVIEW).read_text())
    require(value.get('schema') == REVIEW_SCHEMA and value.get('review_complete') is True
        and value.get('release_bounded_one') is True and value.get('reviewer') == 'root'
        and value.get('fixed_TRAIN_sources') == FIXED_SOURCES
        and value.get('exec_completion') == predecessor['actual_exec_completion']
        and value.get('diagnostic_lock') == lock_reference and value.get('diagnostic_only') is True,
        'root technical review must release only the fixed one passive ring readiness diagnosis')
    return pointer(ROOT_REVIEW)

def _stop_requested():
    return any((ROOT / name).exists() for name in (
        RUNTIME + '/STOP', 'runtime/framework/' + NAME + '/STOP',
        'runtime/framework/multitask_r10_STOP'))


def source_names(overlay, locked, jobs):
    from our_mani.multitask.public_task_phase_integration_v1131 import SUBSTRATE_INPUTS
    parent = overlay.parent
    parent_lock, _ = parent.read_overlay_lock(ROOT, PARENT_LOCK)
    names={SOURCE,OWNER_TEST,WRAPPER,SIM,WORKER,overlay.SOURCE,overlay.TEST_SOURCE,
        parent.MANIFEST,parent.TRAIN_INPUTS,parent.ROUTES,DIAGNOSTIC_LOCK,PARENT_LOCK,
        PREDECESSOR_PLAN,PREDECESSOR_SOURCE,PREDECESSOR_COMPLETION,ROOT_REVIEW,
        PREDECESSOR+'/launch.json',PREDECESSOR+'/status.json',PREDECESSOR+'/terminal.json',
        *DIAGNOSTIC_PINS,*overlay.OVERLAY_SOURCES,*overlay.PASSIVE_PINS,*overlay.PARENT_PINS,
        *parent.OVERLAY_SOURCES,*parent.FROZEN_PINS,*SUBSTRATE_INPUTS,
        locked['integration_cpu_evidence']['path'],parent_lock['integration_cpu_evidence']['path'],
        *(row['config'] for row in jobs)}
    worker=__import__('scripts.run_TRAIN_public_phase_full16_overlay_v1132_r10',fromlist=['ENVIRONMENT'])
    names.update(entry['path'] for entry in worker.ENVIRONMENT.values())
    return names


def validate(plan):
    require(plan.get('schema') == PLAN_SCHEMA and plan.get('expected_cases') == 1
        and plan.get('source_parent_count') == 1 and plan.get('fixed_TRAIN_sources') == FIXED_SOURCES
        and plan.get('automatic_retry') is False and plan.get('max_native_slots') == 1
        and plan.get('split') == 'train' and plan.get('formal_VAL_TEST') is False
        and plan.get('source_outcome_filtering') is False and plan.get('remaining_queue') == []
        and plan.get('CPU_check_before_every_native_case') is True
        and plan.get('predecessor_plan') == dict(path=PREDECESSOR_PLAN,sha256=PREDECESSOR_PLAN_SHA)
        and plan.get('source_overlay_lock',{}).get('path') == DIAGNOSTIC_LOCK
        and plan.get('diagnostic_only') is True and plan.get('parent_factor') == PARENT_FACTOR,
        'fixed bounded one TRAIN ring readiness diagnostic plan required')
    for name,checksum in plan['inputs_sha256'].items(): require(sha(ROOT/name) == checksum,'pinned input changed: '+name)
    overlay=overlay_module();locked,reference=overlay.read_diagnostic_lock(ROOT,DIAGNOSTIC_LOCK)
    require(locked['integration_complete'] is True and not locked['unpinned_sources'], 'complete CPU-qualified diagnostic lock required')
    require(reference == plan['source_overlay_lock'],'diagnostic source lock pointer changed')
    require(locked['parent_source_overlay_lock'] == dict(path=PARENT_LOCK,sha256=PARENT_LOCK_SHA)
        and overlay.PARENT_LOCK == PARENT_LOCK and overlay.PARENT_LOCK_SHA == PARENT_LOCK_SHA,
        'exact frozen1132 parent candidate lock required')
    roster=json.loads(overlay.parent.raw(ROOT,overlay.parent.MANIFEST,overlay.parent.MANIFEST_SHA))
    require(plan['jobs'] == build_jobs(roster),'exact one source/output paths changed')
    require(tuple(CANARIES) == tuple(overlay.SELECTIONS),'diagnostic worker selection differs from bounded owner')
    for row in plan['jobs']:
        overlay.selection(ROOT,argparse.Namespace(task=row['task'],method=row['method'],raw_episode_id=row['raw'],config=row['config']))
    predecessor=predecessor_complete()
    require(plan.get('predecessor') == predecessor and plan.get('root_review') == root_review(reference,predecessor),
            'natural predecessor completion/root review changed')
    require(source_names(overlay,locked,plan['jobs']) <= set(plan['inputs_sha256']),'all source/config/CPU/predecessor/review pins required')
    return locked


def prepare():
    require(not (ROOT/PLAN).exists() and not (ROOT/RUNTIME).exists(), 'fresh owner namespace required; no repeat preparation')
    predecessor=predecessor_complete();overlay=overlay_module()
    locked,reference=overlay.read_diagnostic_lock(ROOT,DIAGNOSTIC_LOCK)
    require(locked['integration_complete'] is True and not locked['unpinned_sources'],'complete candidate CPU evidence required')
    review=root_review(reference,predecessor)
    roster=json.loads(overlay.parent.raw(ROOT,overlay.parent.MANIFEST,overlay.parent.MANIFEST_SHA));jobs=build_jobs(roster)
    require(tuple(CANARIES) == tuple(overlay.SELECTIONS),'diagnostic worker selection differs from bounded owner')
    for row in jobs:
        overlay.selection(ROOT,argparse.Namespace(task=row['task'],method=row['method'],raw_episode_id=row['raw'],config=row['config']))
        require(not (ROOT/row['out']).exists() and not (ROOT/row['runtime']).exists(),'fresh candidate case namespaces required')
    pins={name:sha(ROOT/name) for name in sorted(source_names(overlay,locked,jobs))}
    plan=dict(schema=PLAN_SCHEMA,created_utc=utc(),jobs=jobs,expected_cases=1,source_parent_count=1,
        fixed_TRAIN_sources=FIXED_SOURCES,methods=list(METHODS),split='train',formal_VAL_TEST=False,
        diagnostic_only=True,parent_factor=PARENT_FACTOR,source_outcome_filtering=False,automatic_retry=False,max_native_slots=1,
        CPU_check_before_every_native_case=True,source_overlay_lock=reference,namespace_manifest_sha256=ENV_SHA,
        remaining_queue=[],predecessor_plan=dict(path=PREDECESSOR_PLAN,sha256=PREDECESSOR_PLAN_SHA),
        predecessor=predecessor,root_review=review,inputs_sha256=pins,new_protocol_formal_qualification=False)
    validate(plan);fresh(ROOT/PLAN,plan)
    result=dict(plan=PLAN,sha256=sha(ROOT/PLAN),cases=1,Native_started=False)
    print(json.dumps(result,ensure_ascii=False),flush=True);return result

def command(row, check=False):
    argv = [str(ROOT / '.conda-env/bin/python'), '-B', WRAPPER,
        '--manifest-sha256', ENV_SHA, '--', 'bash', SIM, WORKER,
        '--config', row['config'], '--task', row['task'], '--raw-episode-id', row['raw'],
        '--method', row['method'], '--out', row['out'], '--runtime', row['runtime'],
        '--diagnostic-lock', DIAGNOSTIC_LOCK]
    return argv + (['--check'] if check else [])


def status_write(status):
    atomic(ROOT / RUNTIME / 'status.json', status)
    lines = ['套环被动readiness诊断：固定单来源TRAIN（独立试验，不计正式VAL/TEST）',
        '更新时间：' + status['updated_utc'],
        f"完成：{len(status['completed'])}/1；基础设施错误：{len(status['failed'])}；Native并发：1",
        '阶段：' + status['stage'], '当前：' + str(status.get('active') or '无'),
        '每例先CPU检查再Native，无自动重试，所有成功和失败保留。']
    if status.get('error'):
        lines.append('错误：' + json.dumps(status['error'], ensure_ascii=False))
    path = ROOT / RUNTIME / 'progress.txt'
    temporary = path.with_name(path.name + '.tmp.' + str(os.getpid()))
    with temporary.open('x', encoding='utf-8') as stream:
        stream.write('\n'.join(lines) + '\n')
    os.replace(temporary, path)


def child_phase(row, check, status):
    phase = 'CPU_check' if check else 'native'
    leaf = ROOT / RUNTIME / 'launches' / row['job_id'] / phase
    leaf.mkdir(parents=True, exist_ok=False)
    argv = command(row, check)
    status.update(active=dict(job_id=row['job_id'], stage=phase), updated_utc=utc())
    status_write(status)
    started, child, ticks = time.monotonic(), None, None
    with (leaf / 'stdout.log').open('x') as out, (leaf / 'stderr.log').open('x') as err:
        try:
            child = subprocess.Popen(argv, cwd=ROOT, env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'),
                stdin=subprocess.DEVNULL, stdout=out, stderr=err)
            ticks = int(Path(f'/proc/{child.pid}/stat').read_text().rsplit(')', 1)[1].split()[19])
            launch = dict(created_utc=utc(), pid=child.pid, start_ticks=ticks,
                argv=argv, check_only=check, parent_pid=os.getpid())
            fresh(leaf / 'launch.json', launch)
            code = child.wait()
        except BaseException as error:
            observed = None if child is None else child.poll()
            fresh(leaf / 'owner_interruption.json', dict(observed_utc=utc(),
                child_pid=None if child is None else child.pid, child_start_ticks=ticks,
                observed_returncode=observed, child_exit_unknown=child is not None and observed is None,
                exception_type=type(error).__name__, normal_child_terminal_claimed=False, automatic_retry=False))
            raise
    terminal = dict(finished_utc=utc(), returncode=code, wall_seconds=time.monotonic()-started,
        pid=child.pid, start_ticks=ticks, automatic_retry=False)
    fresh(leaf / 'terminal.json', terminal)
    require(type(code) is int and code == 0, phase + ' infrastructure failure; no retry: ' + row['job_id'])
    name = row['runtime'] + '/' + phase + '/worker_sidecar.json'
    sidecar = json.loads((ROOT / name).read_text())
    execution = sidecar.get('execution', {})
    process = sidecar.get('process', {})
    namespace = sidecar.get('frozen_OS_environment', {})
    overlay=overlay_module()
    require(sidecar.get('schema') == overlay.SCHEMA and sidecar.get('complete') is True
        and sidecar.get('source_overlay_lock') == status['source_overlay_lock']
        and sidecar.get('parent_source_overlay_lock') == dict(path=PARENT_LOCK,sha256=PARENT_LOCK_SHA)
        and sidecar.get('diagnostic_only') is True
        and sidecar.get('formal_score_contribution') is False
        and sidecar.get('selection', {}).get('task') == row['task']
        and sidecar['selection'].get('method') == row['method']
        and sidecar['selection'].get('raw_episode_id') == row['raw']
        and execution.get('complete') is True
        and namespace.get('outer_wrapper_pid') == child.pid
        and namespace.get('outer_wrapper_start_ticks') == ticks
        and namespace.get('observer_pid') == process.get('pid')
        and namespace.get('observer_start_ticks') == process.get('start_ticks')
        and type(process.get('pid')) is int and process['pid'] > 0
        and type(process.get('start_ticks')) is int and process['start_ticks'] > 0,
        'actual completed worker sidecar/process/source/selection binding required')
    if check:
        require(execution.get('mode') == 'CPU_check' and execution.get('native_started') is False
            and execution.get('GPU_started') is False and execution.get('actual_split') == 'train'
            and execution.get('native_execution_ready') is True,
            'actual CPU-only successful complete-integration preflight required')
        validate_protocol(row,execution.get('execution_protocol',{}),status['source_overlay_lock'])
    else:
        require(execution.get('mode') == 'passive_ring_readiness_native_TRAIN'
            and execution.get('native_started') is True and execution.get('native_false_preserved') is True,
            'actual new-protocol Native sidecar required')
    return dict(launch=pointer(str((leaf / 'launch.json').relative_to(ROOT))),
        terminal=pointer(str((leaf / 'terminal.json').relative_to(ROOT))),
        sidecar=pointer(name), process=process, execution=execution)




def validate_protocol(row, protocol, lock):
    require(protocol.get('schema') == DIAGNOSTIC_SCHEMA and protocol.get('diagnostic_source_overlay_lock') == lock
        and protocol.get('parent_source_overlay_lock') == dict(path=PARENT_LOCK,sha256=PARENT_LOCK_SHA)
        and protocol.get('actual_worker') == pointer(WORKER)
        and protocol.get('actual_preparer') == pointer('our_mani/multitask/train_ring_readiness_overlay_v1135.py')
        and protocol.get('diagnostic_only') is True
        and protocol.get('formal_score_contribution') is False
        and protocol.get('formal_VAL_TEST_qualification') is False,
        'actual1135 diagnostic protocol/source/nested lock required')
    inherited=protocol.get('inherited_1132_execution_protocol',{})
    require(inherited.get('schema') == 'r10_full16_fixed_TRAIN_public_phase_overlay_v1132'
        and inherited.get('source_overlay_lock') == dict(path=PARENT_LOCK,sha256=PARENT_LOCK_SHA)
        and inherited.get('execution_branch') == 'core_selected_integration'
        and inherited.get('changed_factor') == PARENT_FACTOR
        and inherited.get('core_factory',{}).get('module') == 'our_mani.multitask.public_task_phase_integration_v1131',
        'unchanged original1132 ring candidate protocol required')
    component=inherited.get('public_phase_component_binding',{})
    rule=component.get('task_rule',{})
    require(rule.get('task') == row['task']
        and rule.get('source_sha256') == '00116825ce901934bcf6ea72fbe2b4727e0a02fa4783a100aaf3504c5914318e'
        and component.get('integration',{}).get('schema') ==
            'r10_estimated_public_ring_seating_and_buzz_approach_integration_v1131',
        'actual source-bound task rule and1131 integration required')


def validate_clock(case):
    clock=case.get('counted_clock',{});ticks=clock.get('ticks')
    require(type(ticks) is int and 0<=ticks<=200 and clock.get('dt') == .05
        and type(clock.get('seconds')) in (int,float)
        and abs(clock['seconds']-ticks*.05)<=1e-8,
        'original finite200tick/.05s counted clock required')
    alignment=case.get('alignment_receipt',{});owned=alignment.get('clock',{})
    sampled=case.get('native_outcome_sampling',{})
    require(owned.get('ticks') == ticks and owned.get('dt') == .05
        and owned.get('simulation_seconds') == clock['seconds']
        and alignment.get('outcomes_are_policy_observations') is False
        and sampled.get('last_tick') == ticks and sampled.get('active_action') is None
        and sampled.get('aborted') is False,
        'unchanged original clock and Native sampling completion required')
    actions=sampled.get('completed_actions')
    require(type(actions) is list and all(type(a.get('ticks')) is int and a['ticks']>0
        and type(a.get('start_tick')) is int and type(a.get('end_tick')) is int
        and type(a.get('max_ticks')) is int and 1<=a['ticks']<=a['max_ticks']<=10
        and a['end_tick']-a['start_tick']==a['ticks']
        and a.get('complete') is True for a in actions)
        and sum(a['ticks'] for a in actions) == ticks
        and (not actions and ticks==0 or actions and actions[0]['start_tick']==0
             and actions[-1]['end_tick']==ticks
             and all(b['start_tick']==a['end_tick'] for a,b in zip(actions,actions[1:]))),
        'original contiguous completed action scopes must account for every counted tick')
    return dict(counted_clock=clock,completed_actions=len(actions),Native_clock_preserved=True)


def diagnostic_quality(row, events):
    """Missing collection is incomplete infrastructure, never a task failure.

    False readiness, empty original candidates and an original public-helper
    exception are recorded policy inputs/results. They are not observer errors.
    """
    require(type(events) is dict and events.get('schema') == DIAGNOSTIC_SCHEMA+'_observers',
            'diagnostic collection incomplete: actual observer schema required')
    require(all(events.get(key) is True for key in
        ('restored','factory_restored','writer_restored','parent_annotation_restored'))
        and events.get('observer_finalization_errors') == [],
        'diagnostic collection incomplete: observer restoration/finalization')
    sink=events.get('readiness_probe');public=events.get('public_execution_trace')
    require(type(sink) is dict and sink.get('restored') is True
        and type(public) is dict and public.get('restored') is True,
        'diagnostic collection incomplete: both passive streams required')
    probe=sink.get('readiness_probe')
    require(type(probe) is dict
        and probe.get('schema') == 'r10_evaluation_existing_public_readiness_probe_v1134'
        and probe.get('restored') is True and probe.get('errors') == []
        and probe.get('restoration_error') is None and probe.get('scope_exception_type') is None
        and probe.get('sysprofile_installed') is False
        and probe.get('original_return_or_exception_rewritten') is False
        and type(probe.get('extra_readiness_reference_forecast_or_native_calls')) is int
        and probe['extra_readiness_reference_forecast_or_native_calls'] == 0
        and type(probe.get('restoration_checks')) is list
        and len(probe['restoration_checks'])>0
        and all(item.get('restored') is True for item in probe['restoration_checks']),
        'diagnostic collection incomplete: actual1134 probe/restoration required')
    for key in ('recording_errors','dropped_events'):
        require(type(probe.get(key)) is int and probe[key] == 0,
                'diagnostic readiness quality counter: '+key)
    rows=probe.get('events');counts=probe.get('method_call_counts')
    require(type(rows) is list and len(rows)>0 and type(counts) is dict,
            'diagnostic collection incomplete: no actual readiness events')
    methods=('frontend.ready','collision_port.readiness','source_policy.candidates','generator.observed_reference')
    for method in methods:
        count=counts.get(method,{})
        require(type(count.get('calls')) is int and count['calls']>0
            and type(count.get('returns')) is int and count['returns']>=0
            and type(count.get('exceptions')) is int and count['exceptions']>=0
            and count['calls'] == count['returns']+count['exceptions'],
            'diagnostic collection incomplete: unobserved or unpaired original '+method)
        recorded=[item for item in rows if item.get('method') == method]
        require(len(recorded) == count['calls']
            and sum(item.get('original_returned') is True for item in recorded) == count['returns']
            and sum(item.get('original_returned') is False for item in recorded) == count['exceptions']
            and all('original_return' in item if item.get('original_returned') is True
                else type(item.get('original_exception')) is dict
                     and type(item['original_exception'].get('type')) is str for item in recorded),
            'diagnostic collection incomplete: method counters differ from actual recorded '+method)
    require(all(item.get('method') in methods and type(item.get('call_sequence')) is int
        and item['call_sequence']>0 for item in rows)
        and len({item['call_sequence'] for item in rows}) == len(rows),
        'diagnostic collection incomplete: actual unique readiness call records required')
    require(type(probe.get('geometry_calls')) is int and probe['geometry_calls']>0
        and type(probe.get('geometry_returns')) is int and probe['geometry_returns']>0
        and type(probe.get('geometry_exceptions')) is int and probe['geometry_exceptions']>=0
        and probe['geometry_calls'] == probe['geometry_returns']+probe['geometry_exceptions'],
        'diagnostic collection incomplete: original geometry construction accounting')
    trace=public.get('trace_receipt')
    require(type(trace) is dict and trace.get('schema') == 'r10_evaluation_existing_public_execution_trace_v1126'
        and trace.get('restored') is True and trace.get('restoration_error') is None
        and type(trace.get('restoration_checks')) is list and len(trace['restoration_checks'])>0
        and all(item.get('restored') is True for item in trace['restoration_checks']),
        'diagnostic collection incomplete: trace restoration required')
    for key in ('recording_errors','frame_dropped','plan_dropped','plan_value_truncations',
                'extra_env_observe_calls','extra_native_getter_or_condition_calls','extra_prediction_calls',
                'extra_path_step_or_FK_calls'):
        require(type(trace.get(key)) is int and trace[key] == 0,'diagnostic trace quality counter: '+key)
    require(trace.get('errors') == [] and trace.get('scope_exception_type') is None,
            'diagnostic collection incomplete: trace finalization/error')
    for key in ('frame_calls','frame_returns','plan_calls'):
        require(type(trace.get(key)) is int and trace[key]>0,
                'diagnostic collection incomplete: no actual '+key)
    require(type(trace.get('frame_exceptions')) is int and trace['frame_exceptions']>=0
        and type(trace.get('plan_exceptions')) is int and trace['plan_exceptions']>=0
        and type(trace.get('plan_returns')) is int and trace['plan_returns']>=0
        and trace['frame_calls'] == trace['frame_returns']+trace['frame_exceptions']
        and trace['plan_calls'] == trace['plan_returns']+trace['plan_exceptions'],
        'diagnostic collection incomplete: trace call/return accounting')
    require(type(trace.get('frames')) is list and len(trace['frames']) == trace['frame_returns']
        and type(trace.get('plans')) is list and len(trace['plans']) == trace['plan_calls']
        and len(trace['frames'])>0 and len(trace['plans'])>0,
        'diagnostic collection incomplete: nonempty actual public frame and plan records required')
    return dict(complete=True,original_method_counts=counts,readiness_events=len(rows),
        original_geometry_calls=probe['geometry_calls'],
        original_method_exceptions=sum(counts[m]['exceptions'] for m in methods),
        public_frames=len(trace['frames']),original_plans=len(trace['plans']),
        original_plan_exceptions=trace['plan_exceptions'],restored=True,
        dropped=0,recording_errors=0,readiness_false_or_empty_candidates_are_policy_results=True,
        native_failure_inferred_from_missing_diagnostics=False)


def completed_case(row, cpu, native, lock):
    name = row['out'] + '/case.json'
    case = json.loads((ROOT / name).read_text())
    case_pointer = pointer(name)
    overlay = overlay_module()
    require(case.get('version') == overlay.SCHEMA + '_case' and case.get('complete') is True
        and case.get('cleanup_completed') is True and case.get('split') == 'train'
        and case.get('task') == row['task'] and case.get('method') == row['method']
        and case.get('source_raw_episode_id') == row['raw']
        and case.get('process_id') == native['process']['pid']
        and type(case.get('official_task_success')) is bool
        and case.get('source_overlay_lock') == lock
        and case.get('parent_source_overlay_lock') == dict(path=PARENT_LOCK,sha256=PARENT_LOCK_SHA)
        and case.get('diagnostic_only') is True
        and case.get('infrastructure_error') is None
        and case.get('formal_score_contribution') is False
        and case.get('original_745_execution_acceptance_granted') is False
        and native['execution'].get('case') == case_pointer
        and native['execution'].get('native_success') is case['official_task_success']
        and type(case.get('native_terminal_failure')) is bool
        and native['execution'].get('native_failure') is case['native_terminal_failure']
        and cpu['execution']['identity'] == native['execution']['identity'] == case.get('identity')
        and case.get('execution_protocol') == cpu['execution'].get('execution_protocol'),
        'complete clean actual new-protocol TRAIN case/result binding required')
    require(cpu['execution']['identity'] == native['execution']['identity'],
            'CPU/native prepared case identity must match')
    validate_protocol(row,case['execution_protocol'],lock)
    inherited=cpu['execution'].get('inherited_1132_identity',{})
    identity=case['identity'];protocol=case['execution_protocol']
    require(type(inherited.get('sha256')) is str and len(inherited['sha256']) == 64
        and inherited['sha256'] == protocol.get('inherited_1132_identity_sha256')
        == identity.get('inherited_1132_identity_sha256')
        and inherited.get('execution_protocol_schema') == 'r10_full16_fixed_TRAIN_public_phase_overlay_v1132'
        and inherited.get('execution_protocol_sha256') == digest(protocol['inherited_1132_execution_protocol'])
        and inherited.get('source_overlay_lock_sha256') == PARENT_LOCK_SHA
        and identity.get('execution_protocol_schema') == DIAGNOSTIC_SCHEMA
        and identity.get('execution_protocol_sha256') == digest(protocol)
        and identity.get('source_overlay_lock_sha256') == lock['sha256'],
        'actual new diagnostic identity must retain the prepared frozen1132 parent identity')
    clock_quality=validate_clock(case)
    quality=diagnostic_quality(row,native['execution'].get('diagnostic_events'))
    hook = row['out'] + '/owned_hook_receipt.json'
    hooks = json.loads((ROOT / hook).read_text())
    require(hooks.get('actual_split') == 'train' and len(hooks.get('reset', [])) == 1
        and hooks['reset'][0].get('complete') is True
        and all(h.get('close_complete') is True for h in hooks.get('paths', [])),
        'actual original reset/RML cleanup receipt required')
    return dict(job_id=row['job_id'], task=row['task'], raw=row['raw'], method=row['method'],
        diagnostic_quality=quality,parent_factor=PARENT_FACTOR,clock_quality=clock_quality,
        native_success=case['official_task_success'], case=case_pointer, owned_hooks=pointer(hook),
        CPU_check=cpu, native=native)


def execute(checksum):
    require(bool(checksum) and sha(ROOT / PLAN) == checksum, 'explicit actual frozen plan SHA required')
    plan = json.loads((ROOT / PLAN).read_text())
    validate(plan)
    predecessor = predecessor_complete()
    base = ROOT / RUNTIME
    base.mkdir(parents=True, exist_ok=True)
    with (base / 'owner.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        require(not (base / 'launch.json').exists(), 'already launched; no automatic resume/retry')
        for row in plan['jobs']:
            require(not (ROOT / row['out']).exists() and not (ROOT / row['runtime']).exists(),
                    'both candidate case namespaces must still be fresh before launch')
        fresh(base / 'launch.json', dict(created_utc=utc(), pid=os.getpid(),
            start_ticks=int(Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19]),
            argv=list(sys.argv), plan=dict(path=PLAN, sha256=checksum),
            source_overlay_lock=plan['source_overlay_lock'], predecessor=predecessor,
            automatic_retry=False, max_native_slots=1))
        status = dict(schema=STATUS_SCHEMA, expected=1, completed=[], failed=[], active=None,
            stage='running', updated_utc=utc(), automatic_retry=False, formal_VAL_TEST=False,
            source_overlay_lock=plan['source_overlay_lock'])
        status_write(status)
        try:
            for row in plan['jobs']:
                if _stop_requested():
                    raise InterruptedError('actual STOP observed before case')
                validate(plan)
                predecessor_complete()
                cpu = child_phase(row, True, status)
                if _stop_requested():
                    raise InterruptedError('actual STOP observed after CPU check')
                validate(plan)
                predecessor_complete()
                native = child_phase(row, False, status)
                status['active']['stage'] = 'result_review'
                result = completed_case(row, cpu, native, plan['source_overlay_lock'])
                status['completed'].append(result)
                status.update(active=None, updated_utc=utc())
                status_write(status)
                print(json.dumps(dict(completed=len(status['completed']), expected=1,
                    last=row['job_id'], native_success=result['native_success']), ensure_ascii=False), flush=True)
            validate(plan)
            status.update(stage='complete', active=None, updated_utc=utc())
            fresh(base / 'terminal.json', dict(complete=True, completed=1, infrastructure_failed=0,
                finished_utc=utc(), automatic_retry=False, formal_VAL_TEST=False))
        except BaseException as error:
            active = status.get('active')
            if active:
                status['failed'].append(dict(**active, exception_type=type(error).__name__,
                    native_failure_inferred=False, automatic_retry=False))
            status.update(stage='failed', updated_utc=utc(), error=dict(type=type(error).__name__, message=str(error)))
            fresh(base / 'failure.json', status)
            raise
        finally:
            status_write(status)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--prepare', action='store_true')
    group.add_argument('--execute', action='store_true')
    parser.add_argument('--plan-sha256')
    args = parser.parse_args(argv)
    if args.prepare:
        return prepare()
    if args.execute:
        return execute(args.plan_sha256)
    print(json.dumps(dict(enabled=False, Native_started=False, GPU_started=False)))


if __name__ == '__main__':
    main()
