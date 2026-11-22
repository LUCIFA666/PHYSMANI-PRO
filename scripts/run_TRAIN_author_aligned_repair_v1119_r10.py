"""Bounded new-protocol TRAIN owner: fixed48, serial check/Native, four canaries.

Default invocation is inert. Preparation only freezes the complete new source
overlay and fixed source grid. Execution requires that exact plan SHA. Every
success, native failure and infrastructure failure is retained without retry.
Root writes the actual four-canary review before the remaining44 may start.
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
SOURCE = 'scripts/run_TRAIN_author_aligned_repair_v1119_r10.py'
WORKER = 'scripts/run_TRAIN_author_aligned_overlay_v1117_r10.py'
WRAPPER = 'scripts/run_frozen_system_environment_v820_r10.py'
SIM = 'scripts/run_local_sim.sh'
NAME = 'TRAIN_author_aligned_repair_v1117'
SETUP = 'artifacts/framework/multitask_r10/' + NAME + '_setup/'
ART = 'artifacts/framework/multitask_r10/' + NAME + '/development_demo_only'
CASE_RUNTIME = 'runtime/framework/' + NAME + '/cases'
RUNTIME = 'runtime/framework/' + NAME + '/owner_v1119'
PLAN = SETUP + 'bounded_plan_v1119.json'
OVERLAY_LOCK = SETUP + 'source_overlay_lock.json'
ENV_SHA = '1c33a2e35c1134b90aebe896da6176a997242e8aa10dd6c9b08c9e4d5730df30'
PLAN_SCHEMA = 'r10_fixed48_TRAIN_author_aligned_repair_plan_v1119'
STATUS_SCHEMA = 'r10_fixed48_TRAIN_author_aligned_repair_status_v1119'
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
        raise ValueError('TRAIN1119: ' + message)


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
    from our_mani.multitask import train_author_aligned_overlay_v1117
    return train_author_aligned_overlay_v1117


def build_jobs(roster):
    require(roster.get('split') == 'train', 'actual TRAIN manifest required')
    rows = []
    for kind, tasks in TASKS.items():
        for task in tasks:
            parents = [r['raw_episode_id'] for r in roster['episodes'] if r['task'] == task][:4]
            require(len(parents) == 4 and len(set(parents)) == 4, 'original first four unique TRAIN parents required')
            for raw in parents:
                for method in METHODS:
                    job = '_'.join((kind, task, raw, method))
                    rows.append(dict(kind=kind, task=task, raw=raw, method=method,
                        config='artifacts/framework/multitask_r10/post_training_qualification_v680_setup/' + task + '_ours_configuration.json',
                        job_id=job, out=f'{ART}/{task}/{raw}/{method}', runtime=f'{CASE_RUNTIME}/{job}'))
    by_key = {(r['task'], r['raw'], r['method']): r for r in rows}
    require(len(rows) == len(by_key) == 48, 'exactly48 distinct cases required')
    require(all(key in by_key for key in CANARIES), 'fixed canaries must belong to original first-four grid')
    first = [by_key[key] for key in CANARIES]
    ids = {r['job_id'] for r in first}
    return first + [r for r in rows if r['job_id'] not in ids]


def _stop_requested():
    return any((ROOT / name).exists() for name in (
        RUNTIME + '/STOP', 'runtime/framework/' + NAME + '/STOP',
        'runtime/framework/multitask_r10_STOP'))


def validate(plan):
    require(plan.get('schema') == PLAN_SCHEMA and plan.get('expected_cases') == 48
        and plan.get('source_parent_count') == 24 and plan.get('canary_count') == 4
        and plan.get('automatic_retry') is False and plan.get('max_native_slots') == 1
        and plan.get('split') == 'train' and plan.get('formal_VAL_TEST') is False
        and plan.get('source_outcome_filtering') is False
        and plan.get('source_overlay_lock', {}).get('path') == OVERLAY_LOCK,
        'fixed TRAIN plan bounds and explicit new overlay required')
    for name, checksum in plan['inputs_sha256'].items():
        require(sha(ROOT / name) == checksum, 'pinned input changed: ' + name)
    overlay = overlay_module()
    locked, reference = overlay.read_overlay_lock(ROOT, OVERLAY_LOCK)
    overlay.require_native_ready(dict(source_overlay=locked))
    require(reference == plan['source_overlay_lock'], 'complete source overlay pointer changed')
    roster = json.loads(overlay.raw(ROOT, overlay.MANIFEST, overlay.MANIFEST_SHA))
    require(plan['jobs'] == build_jobs(roster), 'exact fixed48 grid/canary order changed')
    expected = {SOURCE, WRAPPER, SIM, overlay.MANIFEST, overlay.TRAIN_INPUTS, OVERLAY_LOCK,
                *overlay.OVERLAY_SOURCES, *overlay.FROZEN_PINS,
                locked['integration_cpu_evidence']['path'],
                *(r['config'] for r in plan['jobs'])}
    require(expected <= set(plan['inputs_sha256']), 'all source/CPU/config pins required')
    return locked


def prepare():
    require(not (ROOT / PLAN).exists() and not (ROOT / RUNTIME).exists(),
            'preserve existing owner namespace; no repeat preparation/resume')
    overlay = overlay_module()
    locked, reference = overlay.read_overlay_lock(ROOT, OVERLAY_LOCK)
    overlay.require_native_ready(dict(source_overlay=locked))
    roster = json.loads(overlay.raw(ROOT, overlay.MANIFEST, overlay.MANIFEST_SHA))
    jobs = build_jobs(roster)
    for row in jobs:
        overlay.selection(ROOT, argparse.Namespace(task=row['task'], method=row['method'],
            raw_episode_id=row['raw'], config=row['config']))
        require(not (ROOT / row['out']).exists() and not (ROOT / row['runtime']).exists(),
                'all48 output/runtime cases must be fresh')
    names = {SOURCE, WRAPPER, SIM, overlay.MANIFEST, overlay.TRAIN_INPUTS, OVERLAY_LOCK,
             *overlay.OVERLAY_SOURCES, *overlay.FROZEN_PINS,
             locked['integration_cpu_evidence']['path'], *(r['config'] for r in jobs)}
    # The worker independently validates its complete same-process820 evidence.
    worker = __import__('scripts.run_TRAIN_author_aligned_overlay_v1117_r10', fromlist=['ENVIRONMENT'])
    names.update(entry['path'] for entry in worker.ENVIRONMENT.values())
    pins = {name: sha(ROOT / name) for name in sorted(names)}
    plan = dict(schema=PLAN_SCHEMA, created_utc=utc(), jobs=jobs, expected_cases=48,
        source_parent_count=24, canary_count=4, methods=list(METHODS), split='train',
        formal_VAL_TEST=False, source_outcome_filtering=False, automatic_retry=False,
        max_native_slots=1, CPU_check_before_every_native_case=True,
        source_overlay_lock=reference, namespace_manifest_sha256=ENV_SHA,
        canary_review_path=RUNTIME + '/root_review.json', inputs_sha256=pins,
        old_sources_models_results_modified=False, new_protocol_formal_qualification=False)
    validate(plan)
    fresh(ROOT / PLAN, plan)
    result = dict(plan=PLAN, sha256=sha(ROOT / PLAN), cases=48, canaries=4, Native_started=False)
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
    lines = ['作者动作对齐修复：固定48 TRAIN（独立新协议，不计正式VAL/TEST）',
        '更新时间：' + status['updated_utc'],
        f"完成：{len(status['completed'])}/48；基础设施错误：{len(status['failed'])}；Native并发：1",
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


def await_review(plan, checksum, status):
    status.update(stage='await_canary_review', updated_utc=utc())
    status_write(status)
    path = ROOT / plan['canary_review_path']
    while not path.exists():
        if _stop_requested():
            raise InterruptedError('actual STOP while waiting for root four-canary review')
        time.sleep(2)
    review = json.loads(path.read_text())
    require(review.get('plan_sha256') == checksum
        and review.get('canary_cases') == status['completed']
        and review.get('source_overlay_lock') == plan['source_overlay_lock']
        and review.get('canary_sidecars_reviewed') is True
        and review.get('continue_remaining44') is True,
        'actual root review must bind this plan/source overlay/all four canaries')
    status['canary_review'] = pointer(plan['canary_review_path'])
    status.update(stage='running', updated_utc=utc())
    status_write(status)


def execute(checksum):
    require(bool(checksum) and sha(ROOT / PLAN) == checksum, 'explicit actual frozen plan SHA required')
    plan = json.loads((ROOT / PLAN).read_text())
    validate(plan)
    base = ROOT / RUNTIME
    base.mkdir(parents=True, exist_ok=True)
    with (base / 'owner.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        require(not (base / 'launch.json').exists(), 'already launched; no automatic resume/retry')
        require(not (base / 'root_review.json').exists(), 'canary review must be freshly written after actual four cases')
        for row in plan['jobs']:
            require(not (ROOT / row['out']).exists() and not (ROOT / row['runtime']).exists(),
                    'all48 new case namespaces must still be fresh before launch')
        fresh(base / 'launch.json', dict(created_utc=utc(), pid=os.getpid(),
            start_ticks=int(Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19]),
            argv=list(sys.argv), plan=dict(path=PLAN, sha256=checksum),
            source_overlay_lock=plan['source_overlay_lock'], automatic_retry=False, max_native_slots=1))
        status = dict(schema=STATUS_SCHEMA, expected=48, completed=[], failed=[], active=None,
            stage='running', updated_utc=utc(), automatic_retry=False, formal_VAL_TEST=False,
            source_overlay_lock=plan['source_overlay_lock'])
        status_write(status)
        try:
            for row in plan['jobs']:
                if _stop_requested():
                    raise InterruptedError('actual STOP observed before case')
                validate(plan)
                cpu = child_phase(row, True, status)
                if _stop_requested():
                    raise InterruptedError('actual STOP observed after CPU check')
                validate(plan)
                native = child_phase(row, False, status)
                status['active']['stage'] = 'result_review'
                result = completed_case(row, cpu, native, plan['source_overlay_lock'])
                status['completed'].append(result)
                status.update(active=None, updated_utc=utc())
                status_write(status)
                print(json.dumps(dict(completed=len(status['completed']), expected=48,
                    last=row['job_id'], native_success=result['native_success']), ensure_ascii=False), flush=True)
                if len(status['completed']) == 4:
                    await_review(plan, checksum, status)
            validate(plan)
            status.update(stage='complete', active=None, updated_utc=utc())
            fresh(base / 'terminal.json', dict(complete=True, completed=48, infrastructure_failed=0,
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
