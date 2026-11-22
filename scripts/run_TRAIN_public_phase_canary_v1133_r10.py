"""Two fixed TRAIN public-phase canaries after diagnoses naturally complete.

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
SOURCE = 'scripts/run_TRAIN_public_phase_canary_v1133_r10.py'
OWNER_TEST = 'tests/test_TRAIN_public_phase_canary_owner_v1133.py'
WORKER = 'scripts/run_TRAIN_public_phase_full16_overlay_v1132_r10.py'
WRAPPER = 'scripts/run_frozen_system_environment_v820_r10.py'
SIM = 'scripts/run_local_sim.sh'
NAME = 'TRAIN_public_phase_full16_v1132'
SETUP = 'artifacts/framework/multitask_r10/' + NAME + '_setup/'
ART = 'artifacts/framework/multitask_r10/' + NAME + '/development_demo_only/canary_v1133'
CASE_RUNTIME = 'runtime/framework/' + NAME + '/cases/canary_v1133'
RUNTIME = 'runtime/framework/' + NAME + '/owner_v1133'
PLAN = SETUP + 'bounded_plan_v1133.json'
SOURCE_LOCK = SETUP + 'source_overlay_lock.json'
ROOT_REVIEW = SETUP + 'root_review_v1133.json'
ENV_SHA = '1c33a2e35c1134b90aebe896da6176a997242e8aa10dd6c9b08c9e4d5730df30'
PLAN_SCHEMA = 'r10_bounded_two_TRAIN_public_phase_canary_plan_v1133'
STATUS_SCHEMA = 'r10_bounded_two_TRAIN_public_phase_canary_status_v1133'
REVIEW_SCHEMA = 'r10_fixed_two_public_phase_candidate_release_review_v1133'
EXIT_SCHEMA = 'r10_actual_natural_remaining_failure_diagnostic_owner_exit_v1128'
PREDECESSOR = 'runtime/framework/TRAIN_remaining_failures_v1127/owner_v1128'
PREDECESSOR_PLAN = 'artifacts/framework/multitask_r10/TRAIN_remaining_failures_v1127_setup/bounded_plan_v1128.json'
PREDECESSOR_PLAN_SHA = 'aa757c110d7595976f6195ed80b3b4d9c1935552c86ca9e6d7ab66c1bd86611e'
PREDECESSOR_SOURCE = 'scripts/run_TRAIN_remaining_failures_v1128_r10.py'
PREDECESSOR_SOURCE_SHA = 'e91dcb7dce3cfe3915a2e6c034c0ec68a8a9a98c93fe00b9fac570d9f6beab66'
PREDECESSOR_COMPLETION = PREDECESSOR + '/exec_completion.json'
PREDECESSOR_CANARIES = (
    ('insert_onto_rotating_peg', 'episode101', 'cv_geometry'),
    ('beat_the_rotating_buzz', 'episode10', 'cv_geometry'),
)
TASKS = {'ring': ('insert_onto_rotating_peg',), 'buzz': ('beat_the_rotating_buzz',)}
METHODS = ('cv_geometry',)
CANARIES = PREDECESSOR_CANARIES
FIXED_SOURCES = [dict(task=task, raw_episode_id=raw, method=method) for task,raw,method in CANARIES]
CANDIDATE_SCHEMA = 'r10_full16_fixed_TRAIN_public_phase_overlay_v1132'
CANDIDATE_PINS = {
    'our_mani/multitask/train_public_phase_full16_overlay_v1132.py': 'be0f763aba7e23caddacd6fb6b6cdfbdb38ed273522f19004ee5ba61905e850a',
    WORKER: 'ce5253dc95e0653155476e3f7dd6ef7608a8a7475c6d553769365edc49edcbb0',
    'tests/test_train_public_phase_full16_overlay_v1132.py': 'd04e762b51a2beff661adc9f8b3729c34b7086392f3aa9cbd429eb78714536ae',
    'our_mani/multitask/public_task_phase_integration_v1131.py': '1dba11eab1cc55f59cd7a83ca8bec1d65eca855d43ce15fd860abb020eea0bce',
    'our_mani/multitask/public_ring_seating_policy_v1129.py': '00116825ce901934bcf6ea72fbe2b4727e0a02fa4783a100aaf3504c5914318e',
    'our_mani/multitask/public_buzz_approach_curve_v1130.py': '2f71d16fd045e03fedf9633cd52960290b60e2981ae1cd05d0f620247ffcdbc4',
}
FACTORS = dict(ring='nominal_public_ring_seating_guard', buzz='frozen_TRAIN_approach_curve_continuation')


def require(value, message):
    if not value:
        raise ValueError('TRAIN1133: ' + message)


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


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
    for name, expected in CANDIDATE_PINS.items():
        require(sha(ROOT/name) == expected,'frozen candidate source changed: '+name)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from our_mani.multitask import train_public_phase_full16_overlay_v1132
    return train_public_phase_full16_overlay_v1132


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
    require(len(rows) == len({r['job_id'] for r in rows}) == 2, 'exactly two distinct fixed candidate cases required')
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
    """Require the actual naturally exited1128 owner and complete evidence."""
    require(sha(ROOT/PREDECESSOR_PLAN) == PREDECESSOR_PLAN_SHA
        and sha(ROOT/PREDECESSOR_SOURCE) == PREDECESSOR_SOURCE_SHA, 'frozen actual1128 plan/source required')
    names={key:PREDECESSOR+'/'+key+'.json' for key in ('launch','status','terminal')}
    launch=json.loads((ROOT/names['launch']).read_text())
    status=json.loads((ROOT/names['status']).read_text())
    require(launch.get('plan') == dict(path=PREDECESSOR_PLAN,sha256=PREDECESSOR_PLAN_SHA)
        and status.get('stage') == 'complete' and status.get('active') is None
        and status.get('failed') == [] and len(status.get('completed',[])) == 2
        and not status.get('error'), '1128 must naturally complete with no active child or infrastructure error')
    require([(r['task'],r['raw'],r['method']) for r in status['completed']] == list(PREDECESSOR_CANARIES),
            'exact original two diagnostic sources/order required')
    terminal=json.loads((ROOT/names['terminal']).read_text())
    require(terminal.get('complete') is True and terminal.get('completed') == 2
        and terminal.get('infrastructure_failed') == 0 and terminal.get('automatic_retry') is False,
        'actual natural1128 terminal required')
    pid,ticks=launch.get('pid'),launch.get('start_ticks')
    require(process_not_live(pid,ticks), '1128 owner still alive; candidate owner cannot coexist')
    completion=json.loads((ROOT/PREDECESSOR_COMPLETION).read_text())
    require(completion.get('schema') == EXIT_SCHEMA and completion.get('actual_tool_exit_observed') is True
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
        and value.get('release_bounded_two') is True and value.get('reviewer') == 'root'
        and value.get('fixed_TRAIN_sources') == FIXED_SOURCES
        and value.get('exec_completion') == predecessor['actual_exec_completion']
        and value.get('source_overlay_lock') == lock_reference and value.get('candidate_only') is True,
        'root technical review must release only the fixed two public-phase canaries')
    return pointer(ROOT_REVIEW)

def _stop_requested():
    return any((ROOT / name).exists() for name in (
        RUNTIME + '/STOP', 'runtime/framework/' + NAME + '/STOP',
        'runtime/framework/multitask_r10_STOP'))


def source_names(overlay, locked, jobs):
    from our_mani.multitask.public_task_phase_integration_v1131 import SUBSTRATE_INPUTS
    names={SOURCE,OWNER_TEST,WRAPPER,SIM,WORKER,overlay.SOURCE,overlay.TEST_SOURCE,
        overlay.MANIFEST,overlay.TRAIN_INPUTS,overlay.ROUTES,SOURCE_LOCK,
        PREDECESSOR_PLAN,PREDECESSOR_SOURCE,PREDECESSOR_COMPLETION,ROOT_REVIEW,
        PREDECESSOR+'/launch.json',PREDECESSOR+'/status.json',PREDECESSOR+'/terminal.json',
        *CANDIDATE_PINS,*overlay.OVERLAY_SOURCES,*overlay.FROZEN_PINS,*SUBSTRATE_INPUTS,
        locked['integration_cpu_evidence']['path'],
        *(row['config'] for row in jobs)}
    worker=__import__('scripts.run_TRAIN_public_phase_full16_overlay_v1132_r10',fromlist=['ENVIRONMENT'])
    names.update(entry['path'] for entry in worker.ENVIRONMENT.values())
    return names


def validate(plan):
    require(plan.get('schema') == PLAN_SCHEMA and plan.get('expected_cases') == 2
        and plan.get('source_parent_count') == 2 and plan.get('fixed_TRAIN_sources') == FIXED_SOURCES
        and plan.get('automatic_retry') is False and plan.get('max_native_slots') == 1
        and plan.get('split') == 'train' and plan.get('formal_VAL_TEST') is False
        and plan.get('source_outcome_filtering') is False and plan.get('remaining_queue') == []
        and plan.get('CPU_check_before_every_native_case') is True
        and plan.get('predecessor_plan') == dict(path=PREDECESSOR_PLAN,sha256=PREDECESSOR_PLAN_SHA)
        and plan.get('source_overlay_lock',{}).get('path') == SOURCE_LOCK
        and plan.get('candidate_only') is True and plan.get('factors') == FACTORS,
        'fixed bounded two TRAIN public-phase plan required')
    for name,checksum in plan['inputs_sha256'].items(): require(sha(ROOT/name) == checksum,'pinned input changed: '+name)
    overlay=overlay_module();locked,reference=overlay.read_overlay_lock(ROOT,SOURCE_LOCK)
    require(locked['integration_complete'] is True and not locked['unpinned_sources'], 'complete CPU-qualified candidate lock required')
    require(reference == plan['source_overlay_lock'],'candidate source lock pointer changed')
    require(locked['core_factory']['module'] == 'our_mani.multitask.public_task_phase_integration_v1131',
            'explicit1131 candidate core required')
    roster=json.loads(overlay.raw(ROOT,overlay.MANIFEST,overlay.MANIFEST_SHA))
    require(plan['jobs'] == build_jobs(roster),'exact two source/order/output paths changed')
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
    locked,reference=overlay.read_overlay_lock(ROOT,SOURCE_LOCK)
    require(locked['integration_complete'] is True and not locked['unpinned_sources'],'complete candidate CPU evidence required')
    review=root_review(reference,predecessor)
    roster=json.loads(overlay.raw(ROOT,overlay.MANIFEST,overlay.MANIFEST_SHA));jobs=build_jobs(roster)
    for row in jobs:
        overlay.selection(ROOT,argparse.Namespace(task=row['task'],method=row['method'],raw_episode_id=row['raw'],config=row['config']))
        require(not (ROOT/row['out']).exists() and not (ROOT/row['runtime']).exists(),'fresh candidate case namespaces required')
    pins={name:sha(ROOT/name) for name in sorted(source_names(overlay,locked,jobs))}
    plan=dict(schema=PLAN_SCHEMA,created_utc=utc(),jobs=jobs,expected_cases=2,source_parent_count=2,
        fixed_TRAIN_sources=FIXED_SOURCES,methods=list(METHODS),split='train',formal_VAL_TEST=False,
        candidate_only=True,factors=FACTORS,source_outcome_filtering=False,automatic_retry=False,max_native_slots=1,
        CPU_check_before_every_native_case=True,source_overlay_lock=reference,namespace_manifest_sha256=ENV_SHA,
        remaining_queue=[],predecessor_plan=dict(path=PREDECESSOR_PLAN,sha256=PREDECESSOR_PLAN_SHA),
        predecessor=predecessor,root_review=review,inputs_sha256=pins,new_protocol_formal_qualification=False)
    validate(plan);fresh(ROOT/PLAN,plan)
    result=dict(plan=PLAN,sha256=sha(ROOT/PLAN),cases=2,Native_started=False)
    print(json.dumps(result,ensure_ascii=False),flush=True);return result

def command(row, check=False):
    argv = [str(ROOT / '.conda-env/bin/python'), '-B', WRAPPER,
        '--manifest-sha256', ENV_SHA, '--', 'bash', SIM, WORKER,
        '--config', row['config'], '--task', row['task'], '--raw-episode-id', row['raw'],
        '--method', row['method'], '--out', row['out'], '--runtime', row['runtime'],
        '--overlay-lock', SOURCE_LOCK]
    return argv + (['--check'] if check else [])


def status_write(status):
    atomic(ROOT / RUNTIME / 'status.json', status)
    lines = ['公共阶段策略：固定两来源TRAIN候选（独立试验，不计正式VAL/TEST）',
        '更新时间：' + status['updated_utc'],
        f"完成：{len(status['completed'])}/2；基础设施错误：{len(status['failed'])}；Native并发：1",
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
        require(execution.get('mode') == 'new_protocol_native_TRAIN'
            and execution.get('native_started') is True and execution.get('native_false_preserved') is True,
            'actual new-protocol Native sidecar required')
    return dict(launch=pointer(str((leaf / 'launch.json').relative_to(ROOT))),
        terminal=pointer(str((leaf / 'terminal.json').relative_to(ROOT))),
        sidecar=pointer(name), process=process, execution=execution)




def validate_protocol(row, protocol, lock):
    require(protocol.get('schema') == CANDIDATE_SCHEMA and protocol.get('source_overlay_lock') == lock
        and protocol.get('actual_worker') == pointer(WORKER)
        and protocol.get('actual_preparer') == pointer('our_mani/multitask/train_public_phase_full16_overlay_v1132.py')
        and protocol.get('execution_branch') == 'core_selected_integration'
        and protocol.get('core_factory',{}).get('module') == 'our_mani.multitask.public_task_phase_integration_v1131'
        and protocol.get('changed_factor') == FACTORS[row['kind']]
        and protocol.get('formal_score_contribution') is False
        and protocol.get('formal_VAL_TEST_qualification') is False,
        'actual1132 candidate protocol and correct per-task factor required')
    component=protocol.get('public_phase_component_binding',{})
    rule=component.get('task_rule',{})
    module='public_ring_seating_policy_v1129' if row['kind']=='ring' else 'public_buzz_approach_curve_v1130'
    require(rule.get('task') == row['task']
        and rule.get('source_sha256') == CANDIDATE_PINS['our_mani/multitask/'+module+'.py']
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
    clock_quality=validate_clock(case)
    hook = row['out'] + '/owned_hook_receipt.json'
    hooks = json.loads((ROOT / hook).read_text())
    require(hooks.get('actual_split') == 'train' and len(hooks.get('reset', [])) == 1
        and hooks['reset'][0].get('complete') is True
        and all(h.get('close_complete') is True for h in hooks.get('paths', [])),
        'actual original reset/RML cleanup receipt required')
    return dict(job_id=row['job_id'], task=row['task'], raw=row['raw'], method=row['method'],
        changed_factor=FACTORS[row['kind']],clock_quality=clock_quality,
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
        status = dict(schema=STATUS_SCHEMA, expected=2, completed=[], failed=[], active=None,
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
                print(json.dumps(dict(completed=len(status['completed']), expected=2,
                    last=row['job_id'], native_success=result['native_success']), ensure_ascii=False), flush=True)
            validate(plan)
            status.update(stage='complete', active=None, updated_utc=utc())
            fresh(base / 'terminal.json', dict(complete=True, completed=2, infrastructure_failed=0,
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
