"""Four matched TRAIN sources; restore only the first-preclose qualification.

Default invocation is inert. Preparation only freezes the complete new source
overlay and fixed source grid. Execution requires that exact plan SHA and the
prior1119 owner to be quiescent. Every outcome is retained without retry. Four
cases end this owner naturally; it has no continuation queue or review bypass.
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
SOURCE = 'scripts/run_TRAIN_preclose_comparison_v1124_r10.py'
OWNER_TEST = 'tests/test_TRAIN_preclose_comparison_owner_v1124.py'
WORKER = 'scripts/run_TRAIN_author_aligned_full16_overlay_v1122_r10.py'
WRAPPER = 'scripts/run_frozen_system_environment_v820_r10.py'
SIM = 'scripts/run_local_sim.sh'
NAME = 'TRAIN_author_aligned_full16_v1122'
SETUP = 'artifacts/framework/multitask_r10/' + NAME + '_setup/'
ART = 'artifacts/framework/multitask_r10/' + NAME + '/development_demo_only/matched_preclose_v1124'
CASE_RUNTIME = 'runtime/framework/' + NAME + '/cases/matched_preclose_v1124'
RUNTIME = 'runtime/framework/' + NAME + '/owner_v1124'
PLAN = SETUP + 'matched_preclose_plan_v1124.json'
OVERLAY_LOCK = SETUP + 'source_overlay_lock.json'
ENV_SHA = '1c33a2e35c1134b90aebe896da6176a997242e8aa10dd6c9b08c9e4d5730df30'
PLAN_SCHEMA = 'r10_matched4_TRAIN_preclose_comparison_plan_v1124'
STATUS_SCHEMA = 'r10_matched4_TRAIN_preclose_comparison_status_v1124'
PREDECESSOR = 'runtime/framework/TRAIN_author_aligned_repair_v1117/owner_v1119'
PREDECESSOR_PLAN = 'artifacts/framework/multitask_r10/TRAIN_author_aligned_repair_v1117_setup/bounded_plan_v1119.json'
PREDECESSOR_PLAN_SHA = '45a8395c6f659d56fd6db5159f8aae759c594bbab525b6023e89626329b6bca5'
PREDECESSOR_SOURCE = 'scripts/run_TRAIN_author_aligned_repair_v1119_r10.py'
PREDECESSOR_SOURCE_SHA = '19904670631395b45bd9b4a66ce5302cd94bb82e7452e51d1ba1b229c21e4f0e'
PREDECESSOR_CONTROL = PREDECESSOR + '/controlled_exec_completion_v1121.json'
PREDECESSOR_DECISION = PREDECESSOR + '/control_end_requested_v1121.json'
TASKS = {
    'remove': ('remove_cups_from_rotating_frame', 'remove_cups_from_rotating_frame_high_speed'),
    'ring': ('insert_onto_rotating_peg', 'insert_onto_rotating_peg_high_speed'),
    'buzz': ('beat_the_rotating_buzz', 'beat_the_rotating_buzz_high_speed'),
}
METHODS = ('cv_geometry', 'context_geometry')
CANARIES = (
    ('remove_cups_from_rotating_frame', 'episode104', 'cv_geometry'),
    ('remove_cups_from_rotating_frame_high_speed', 'episode1', 'context_geometry'),
    ('insert_onto_rotating_peg', 'episode101', 'cv_geometry'),
    ('beat_the_rotating_buzz', 'episode10', 'cv_geometry'),
)


def require(value, message):
    if not value:
        raise ValueError('TRAIN1124: ' + message)


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
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from our_mani.multitask import train_author_aligned_full16_overlay_v1122
    return train_author_aligned_full16_overlay_v1122


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
    require(len(rows) == len({r['job_id'] for r in rows}) == 4, 'exactly four distinct matched cases required')
    return rows


def predecessor_quiescent():
    """Read actual predecessor identity; do not infer or manufacture its exit."""
    require(sha(ROOT / PREDECESSOR_PLAN) == PREDECESSOR_PLAN_SHA
        and sha(ROOT / PREDECESSOR_SOURCE) == PREDECESSOR_SOURCE_SHA,
        'actual frozen1119 plan/source required')
    launch_name, status_name = PREDECESSOR + '/launch.json', PREDECESSOR + '/status.json'
    launch = json.loads((ROOT / launch_name).read_text())
    status = json.loads((ROOT / status_name).read_text())
    control = json.loads((ROOT / PREDECESSOR_CONTROL).read_text())
    decision = json.loads((ROOT / PREDECESSOR_DECISION).read_text())
    require(launch.get('plan') == dict(path=PREDECESSOR_PLAN, sha256=PREDECESSOR_PLAN_SHA)
        and len(status.get('completed', [])) == 4 and status.get('active') is None
        and status.get('failed') == [] and status.get('stage') == 'failed'
        and status.get('error', {}).get('type') == 'InterruptedError',
        'exactly four completed predecessor cases and no active child required')
    require([(r['task'], r['raw'], r['method']) for r in status['completed']] == list(CANARIES),
            'same four matched predecessor sources/order required')
    pid, ticks = launch.get('pid'), launch.get('start_ticks')
    require(type(pid) is int and pid > 0 and type(ticks) is int and ticks > 0,
            'actual predecessor PID/start identity required')
    require(control.get('schema') == 'r10_actual_controlled_pilot_owner_exit_v1121'
        and control.get('actual_tool_exit_observed') is True and control.get('returncode') == 1
        and control.get('owner_pid') == pid and control.get('owner_start_ticks') == ticks
        and control.get('owner_alive') is False and control.get('controlled_end') is True
        and control.get('native_cases_complete') == 4
        and control.get('completed_case_infrastructure_failures') == 0
        and control.get('remaining44_not_launched') is True
        and control.get('normal_full48_completion_claimed') is False
        and decision.get('owner_process') == dict(pid=pid, start_ticks=ticks)
        and decision.get('plan') == launch['plan'] and decision.get('remaining44_released') is False,
        'actual controlled1119 exit receipt and unreleased remainder required')
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
    except (FileNotFoundError, ProcessLookupError):
        fields = None
    require(fields is None or int(fields[19]) != ticks or fields[0] in ('Z', 'X'),
            '1119 owner still alive; matched owner cannot coexist')
    for row in status['completed']:
        for ref in (row['case'], row['native']['sidecar']):
            require(sha(ROOT / ref['path']) == ref['sha256'], 'predecessor completed evidence changed')
    return dict(launch=pointer(launch_name), status=pointer(status_name),
        actual_controlled_exec_completion=pointer(PREDECESSOR_CONTROL), decision=pointer(PREDECESSOR_DECISION),
        original_pid=pid, original_start_ticks=ticks, process_identity_not_live=True,
        natural_exit_claimed=False, prior_native_outcomes_relabelled=False)


def _stop_requested():
    return any((ROOT / name).exists() for name in (
        RUNTIME + '/STOP', 'runtime/framework/' + NAME + '/STOP',
        'runtime/framework/multitask_r10_STOP'))


def validate(plan):
    require(plan.get('schema') == PLAN_SCHEMA and plan.get('expected_cases') == 4
        and plan.get('source_parent_count') == 4 and plan.get('matched_source_count') == 4
        and plan.get('automatic_retry') is False and plan.get('max_native_slots') == 1
        and plan.get('split') == 'train' and plan.get('formal_VAL_TEST') is False
        and plan.get('source_outcome_filtering') is False
        and plan.get('remaining_queue') == []
        and plan.get('predecessor_plan') == dict(path=PREDECESSOR_PLAN, sha256=PREDECESSOR_PLAN_SHA)
        and plan.get('source_overlay_lock', {}).get('path') == OVERLAY_LOCK,
        'fixed TRAIN plan bounds and explicit new overlay required')
    for name, checksum in plan['inputs_sha256'].items():
        require(sha(ROOT / name) == checksum, 'pinned input changed: ' + name)
    overlay = overlay_module()
    locked, reference = overlay.read_overlay_lock(ROOT, OVERLAY_LOCK)
    overlay.require_native_ready(dict(source_overlay=locked))
    require(reference == plan['source_overlay_lock'], 'complete source overlay pointer changed')
    roster = json.loads(overlay.raw(ROOT, overlay.MANIFEST, overlay.MANIFEST_SHA))
    require(plan['jobs'] == build_jobs(roster), 'exact four matched sources/order changed')
    require(locked['core_factory']['module'] == 'our_mani.multitask.retained_preclose_policy_integration_v1121'
        and locked['core_factory']['matched_policy_variant']['changed_factor'] ==
            'first_acquire_close_uses_original_policy_qualification', 'only locked1121 preclose candidate permitted')
    expected = {SOURCE, OWNER_TEST, WRAPPER, SIM, overlay.MANIFEST, overlay.TRAIN_INPUTS, overlay.ROUTES, OVERLAY_LOCK,
                PREDECESSOR_PLAN, PREDECESSOR_SOURCE, PREDECESSOR_CONTROL, PREDECESSOR_DECISION,
                PREDECESSOR + '/launch.json', PREDECESSOR + '/status.json',
                *overlay.OVERLAY_SOURCES, *overlay.FROZEN_PINS,
                locked['integration_cpu_evidence']['path'],
                *(r['config'] for r in plan['jobs'])}
    require(expected <= set(plan['inputs_sha256']), 'all source/CPU/config pins required')
    return locked


def prepare():
    require(not (ROOT / PLAN).exists() and not (ROOT / RUNTIME).exists(),
            'preserve existing owner namespace; no repeat preparation/resume')
    overlay = overlay_module()
    predecessor_quiescent()
    locked, reference = overlay.read_overlay_lock(ROOT, OVERLAY_LOCK)
    overlay.require_native_ready(dict(source_overlay=locked))
    roster = json.loads(overlay.raw(ROOT, overlay.MANIFEST, overlay.MANIFEST_SHA))
    jobs = build_jobs(roster)
    for row in jobs:
        overlay.selection(ROOT, argparse.Namespace(task=row['task'], method=row['method'],
            raw_episode_id=row['raw'], config=row['config']))
        require(not (ROOT / row['out']).exists() and not (ROOT / row['runtime']).exists(),
                'all four matched output/runtime cases must be fresh')
    names = {SOURCE, OWNER_TEST, WRAPPER, SIM, overlay.MANIFEST, overlay.TRAIN_INPUTS, overlay.ROUTES, OVERLAY_LOCK,
             PREDECESSOR_PLAN, PREDECESSOR_SOURCE, PREDECESSOR_CONTROL, PREDECESSOR_DECISION,
             PREDECESSOR + '/launch.json', PREDECESSOR + '/status.json',
             *overlay.OVERLAY_SOURCES, *overlay.FROZEN_PINS,
             locked['integration_cpu_evidence']['path'], *(r['config'] for r in jobs)}
    # The worker independently validates its complete same-process820 evidence.
    worker = __import__('scripts.run_TRAIN_author_aligned_full16_overlay_v1122_r10', fromlist=['ENVIRONMENT'])
    names.update(entry['path'] for entry in worker.ENVIRONMENT.values())
    pins = {name: sha(ROOT / name) for name in sorted(names)}
    plan = dict(schema=PLAN_SCHEMA, created_utc=utc(), jobs=jobs, expected_cases=4,
        source_parent_count=4, matched_source_count=4, methods=list(METHODS), split='train',
        formal_VAL_TEST=False, source_outcome_filtering=False, automatic_retry=False,
        max_native_slots=1, CPU_check_before_every_native_case=True,
        source_overlay_lock=reference, namespace_manifest_sha256=ENV_SHA,
        remaining_queue=[], predecessor_plan=dict(path=PREDECESSOR_PLAN, sha256=PREDECESSOR_PLAN_SHA),
        changed_factor='first_acquire_close_uses_original_policy_qualification', inputs_sha256=pins,
        old_sources_models_results_modified=False, new_protocol_formal_qualification=False)
    validate(plan)
    fresh(ROOT / PLAN, plan)
    result = dict(plan=PLAN, sha256=sha(ROOT / PLAN), cases=4, Native_started=False)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return result


def command(row, check=False):
    argv = [str(ROOT / '.conda-env/bin/python'), '-B', WRAPPER,
        '--manifest-sha256', ENV_SHA, '--', 'bash', SIM, WORKER,
        '--config', row['config'], '--task', row['task'], '--raw-episode-id', row['raw'],
        '--method', row['method'], '--out', row['out'], '--runtime', row['runtime'],
        '--overlay-lock', OVERLAY_LOCK]
    return argv + (['--check'] if check else [])


def status_write(status):
    atomic(ROOT / RUNTIME / 'status.json', status)
    lines = ['首次闭爪资格：同四来源TRAIN比较（独立新协议，不计正式VAL/TEST）',
        '更新时间：' + status['updated_utc'],
        f"完成：{len(status['completed'])}/4；基础设施错误：{len(status['failed'])}；Native并发：1",
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
    require(code == 0, phase + ' infrastructure failure; no retry: ' + row['job_id'])
    name = row['runtime'] + '/' + phase + '/worker_sidecar.json'
    sidecar = json.loads((ROOT / name).read_text())
    execution = sidecar.get('execution', {})
    process = sidecar.get('process', {})
    namespace = sidecar.get('frozen_OS_environment', {})
    require(sidecar.get('schema') == overlay_module().SCHEMA and sidecar.get('complete') is True
        and sidecar.get('source_overlay_lock') == status['source_overlay_lock']
        and sidecar.get('selection', {}).get('task') == row['task']
        and sidecar['selection'].get('method') == row['method']
        and sidecar['selection'].get('raw_episode_id') == row['raw']
        and execution.get('complete') is True
        and namespace.get('outer_wrapper_pid') == child.pid
        and namespace.get('outer_wrapper_start_ticks') == ticks
        and namespace.get('observer_pid') == process.get('pid')
        and namespace.get('observer_start_ticks') == process.get('start_ticks')
        and type(process.get('pid')) is int and type(process.get('start_ticks')) is int,
        'actual completed worker sidecar/process/source/selection binding required')
    if check:
        require(execution.get('mode') == 'CPU_check' and execution.get('native_started') is False
            and execution.get('GPU_started') is False and execution.get('actual_split') == 'train'
            and execution.get('native_execution_ready') is True,
            'actual CPU-only successful complete-integration preflight required')
    else:
        require(execution.get('mode') == 'new_protocol_native_TRAIN'
            and execution.get('native_started') is True, 'actual new-protocol Native sidecar required')
    return dict(launch=pointer(str((leaf / 'launch.json').relative_to(ROOT))),
        terminal=pointer(str((leaf / 'terminal.json').relative_to(ROOT))),
        sidecar=pointer(name), process=process, execution=execution)


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
        and case.get('formal_score_contribution') is False
        and case.get('original_745_execution_acceptance_granted') is False
        and native['execution'].get('case') == case_pointer
        and native['execution'].get('native_success') is case['official_task_success'],
        'complete clean actual new-protocol TRAIN case/result binding required')
    require(cpu['execution']['identity'] == native['execution']['identity'],
            'CPU/native prepared case identity must match')
    hook = row['out'] + '/owned_hook_receipt.json'
    hooks = json.loads((ROOT / hook).read_text())
    require(hooks.get('actual_split') == 'train' and len(hooks.get('reset', [])) == 1
        and hooks['reset'][0].get('complete') is True
        and all(h.get('close_complete') is True for h in hooks.get('paths', [])),
        'actual original reset/RML cleanup receipt required')
    return dict(job_id=row['job_id'], task=row['task'], raw=row['raw'], method=row['method'],
        native_success=case['official_task_success'], case=case_pointer, owned_hooks=pointer(hook),
        CPU_check=cpu, native=native)


def execute(checksum):
    require(bool(checksum) and sha(ROOT / PLAN) == checksum, 'explicit actual frozen plan SHA required')
    plan = json.loads((ROOT / PLAN).read_text())
    validate(plan)
    predecessor = predecessor_quiescent()
    base = ROOT / RUNTIME
    base.mkdir(parents=True, exist_ok=True)
    with (base / 'owner.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        require(not (base / 'launch.json').exists(), 'already launched; no automatic resume/retry')
        for row in plan['jobs']:
            require(not (ROOT / row['out']).exists() and not (ROOT / row['runtime']).exists(),
                    'all four matched case namespaces must still be fresh before launch')
        fresh(base / 'launch.json', dict(created_utc=utc(), pid=os.getpid(),
            start_ticks=int(Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19]),
            argv=list(sys.argv), plan=dict(path=PLAN, sha256=checksum),
            source_overlay_lock=plan['source_overlay_lock'], predecessor=predecessor,
            automatic_retry=False, max_native_slots=1))
        status = dict(schema=STATUS_SCHEMA, expected=4, completed=[], failed=[], active=None,
            stage='running', updated_utc=utc(), automatic_retry=False, formal_VAL_TEST=False,
            source_overlay_lock=plan['source_overlay_lock'])
        status_write(status)
        try:
            for row in plan['jobs']:
                if _stop_requested():
                    raise InterruptedError('actual STOP observed before case')
                validate(plan)
                predecessor_quiescent()
                cpu = child_phase(row, True, status)
                if _stop_requested():
                    raise InterruptedError('actual STOP observed after CPU check')
                validate(plan)
                predecessor_quiescent()
                native = child_phase(row, False, status)
                status['active']['stage'] = 'result_review'
                result = completed_case(row, cpu, native, plan['source_overlay_lock'])
                status['completed'].append(result)
                status.update(active=None, updated_utc=utc())
                status_write(status)
                print(json.dumps(dict(completed=len(status['completed']), expected=4,
                    last=row['job_id'], native_success=result['native_success']), ensure_ascii=False), flush=True)
            validate(plan)
            status.update(stage='complete', active=None, updated_utc=utc())
            fresh(base / 'terminal.json', dict(complete=True, completed=4, infrastructure_failed=0,
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
