"""Bounded post-completion TRAIN diagnostics; never starts formal VAL/TEST.

The plan preserves the original first four TRAIN parents, both geometry methods,
all outcomes, original worker/source locks and the qualified v820 OS namespace.
Only new diagnostic sidecars and outputs are written. Default invocation is inert.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
SOURCE = 'scripts/run_TRAIN_zero_success_diagnosis_v1107_r10.py'
RUNTIME = 'runtime/framework/TRAIN_zero_success_diagnosis_v1107'
SETUP = 'artifacts/framework/multitask_r10/TRAIN_zero_success_diagnosis_v1107_setup'
ART = 'artifacts/framework/multitask_r10/TRAIN_zero_success_diagnosis_v1107/development_demo_only'
PLAN = SETUP + '/plan.json'
REPORT = 'TRAIN_零成功任务诊断_20261023.txt'
WORKER = 'scripts/diagnose_TRAIN_ring_postcomplete_v1103_r10.py'
WRAPPER = 'scripts/run_frozen_system_environment_v820_r10.py'
MANIFEST_SHA = '1c33a2e35c1134b90aebe896da6176a997242e8aa10dd6c9b08c9e4d5730df30'
TASKS = {
    'ring': ('insert_onto_rotating_peg', 'insert_onto_rotating_peg_high_speed'),
    'remove': ('remove_cups_from_rotating_frame', 'remove_cups_from_rotating_frame_high_speed'),
    'buzz': ('beat_the_rotating_buzz', 'beat_the_rotating_buzz_high_speed'),
}
METHODS = ('cv_geometry', 'context_geometry')
NEW_SOURCES = (
    SOURCE, WORKER,
    'our_mani/multitask/post_completed_TEST_resource_guard_v1106.py',
    'scripts/diagnose_TRAIN_remove_postcomplete_v1104_r10.py',
    'our_mani/multitask/train_buzz_failure_tap_v1105.py',
)


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
    tmp = path.with_name(path.name + '.tmp.' + str(os.getpid()))
    fresh(tmp, value)
    os.replace(tmp, path)


def prepare():
    if (ROOT / PLAN).exists() or (ROOT / RUNTIME / 'launch.json').exists():
        raise FileExistsError('Existing diagnostic namespace: inspect it, do not re-prepare')
    roster = json.loads((ROOT / 'artifacts/framework/protocol/train_manifest.json').read_text())
    routes = json.loads((ROOT / 'artifacts/framework/multitask_r10/author_outcome_qualification_v745_setup/routes.json').read_text())
    route_map = {(r['task'], r['method']): r for r in routes['routes']}
    rows = []
    configs = set()
    for kind, tasks in TASKS.items():
        for task in tasks:
            parents = [r['raw_episode_id'] for r in roster['episodes'] if r['task'] == task][:4]
            if len(parents) != 4 or len(set(parents)) != 4:
                raise ValueError('Exactly original first four TRAIN parents required')
            for parent in parents:
                for method in METHODS:
                    route = route_map[task, method]
                    config = route['configuration']
                    if sha(ROOT / config['path']) != config['sha256']:
                        raise ValueError('Original configuration changed')
                    configs.add(config['path'])
                    job_id = '_'.join((kind, task, parent, method))
                    rows.append(dict(kind=kind, task=task, raw=parent, method=method,
                        config=config['path'], job_id=job_id,
                        out=f'{ART}/{task}/{parent}/{method}', runtime=f'{RUNTIME}/cases/{job_id}'))
    if len(rows) != 48:
        raise ValueError('Fixed 48-case grid required')
    # One fixed, outcome-independent canary of each kind precedes the remaining grid.
    first = [next(r for r in rows if r['kind'] == kind) for kind in TASKS]
    first_ids = {r['job_id'] for r in first}
    rows = first + [r for r in rows if r['job_id'] not in first_ids]
    names = (*NEW_SOURCES, WRAPPER,
        'scripts/run_task_executor_smoke_author_outcome_v743_r10.py',
        'scripts/run_task_executor_smoke_author_suite16_v744_r10.py',
        'our_mani/multitask/native_outcome_sampling_v741.py',
        'artifacts/framework/protocol/train_manifest.json',
        'artifacts/framework/multitask_r10/author_outcome_qualification_v745_setup/routes.json',
        'artifacts/framework/multitask_r10/author_outcome_qualification_v745_setup/frozen_inputs.json',
        'runtime/framework/latest_executor_continuation_v1102.json',
        *sorted(configs))
    source_map = {name: sha(ROOT / name) for name in names}
    if any((ROOT / name).stat().st_size == 0 for name in names):
        raise ValueError('Empty diagnostic or source file')
    plan = dict(schema='r10_fixed_TRAIN_zero_success_diagnostic_plan_v1107', created_utc=utc(),
        jobs=rows, expected_cases=48, source_parent_count=24, methods=list(METHODS),
        split='train', formal_VAL_TEST=False, source_outcome_filtering=False,
        automatic_retry=False, max_native_slots=1, model_or_policy_changes=False,
        namespace_manifest_sha256=MANIFEST_SHA, inputs_sha256=source_map)
    fresh(ROOT / PLAN, plan)
    print(json.dumps(dict(plan=PLAN, sha256=sha(ROOT / PLAN), cases=len(rows)), ensure_ascii=False), flush=True)
    return plan


def command(row, check=False):
    argv = [str(ROOT / '.conda-env/bin/python'), '-B', WRAPPER,
        '--manifest-sha256', MANIFEST_SHA, '--', 'bash', 'scripts/run_local_sim.sh', WORKER,
        '--diagnostic', row['kind'], '--config', row['config'], '--task', row['task'],
        '--raw-episode-id', row['raw'], '--method', row['method'],
        '--out', row['out'], '--runtime', row['runtime']]
    return argv + (['--check'] if check else [])


def validate(plan):
    for name, expected in plan['inputs_sha256'].items():
        if sha(ROOT / name) != expected:
            raise ValueError('Pinned input changed: ' + name)


def report(status):
    lines = ['TRAIN零成功任务诊断（独立开发诊断，不计入正式VAL/TEST）',
        '更新时间：' + status['updated_utc'],
        f"完成：{len(status['completed'])}/48；基础设施错误：{len(status['failed'])}；并发：1 Native",
        '批次状态：' + status['stage'],
        '当前：' + str(status.get('active') or '无'),
        '模型、执行策略、旧证据及封存TEST保持不变；所有成功和失败完整保留。',
        '本文件为本次诊断状态，不代表已经在对话发送。']
    if status.get('error'):
        lines.append('错误：' + json.dumps(status['error'], ensure_ascii=False))
    path = ROOT / REPORT
    tmp = path.with_name(path.name + '.tmp.' + str(os.getpid()))
    with tmp.open('x', encoding='utf-8') as stream:
        stream.write('\n'.join(lines) + '\n')
    os.replace(tmp, path)


def execute(expected):
    path = ROOT / PLAN
    if not expected or sha(path) != expected:
        raise ValueError('Explicit actual plan SHA required')
    plan = json.loads(path.read_text())
    validate(plan)
    runtime = ROOT / RUNTIME
    runtime.mkdir(parents=True, exist_ok=True)
    with (runtime / 'owner.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (runtime / 'launch.json').exists():
            raise FileExistsError('Already launched: inspect state; no automatic resume or retry')
        fresh(runtime / 'launch.json', dict(created_utc=utc(), pid=os.getpid(),
            start_ticks=int(Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19]),
            plan=dict(path=PLAN, sha256=expected), argv=list(__import__('sys').argv),
            max_native_slots=1, automatic_retry=False))
        status = dict(schema='r10_TRAIN_zero_success_diagnostic_status_v1107',
            updated_utc=utc(), expected=48, completed=[], failed=[], active=None,
            stage='running', automatic_retry=False, formal_VAL_TEST=False)
        try:
            checked = set()
            for row in plan['jobs']:
                if (runtime / 'STOP').exists() or (ROOT / 'runtime/framework/multitask_r10_STOP').exists():
                    raise InterruptedError('Actual STOP file observed')
                validate(plan)
                modes = ([True] if row['kind'] not in checked else []) + [False]
                for check in modes:
                    stage = 'CPU_check' if check else 'native'
                    leaf = runtime / 'launches' / row['job_id'] / stage
                    leaf.mkdir(parents=True, exist_ok=False)
                    argv = command(row, check)
                    status.update(active=dict(job_id=row['job_id'], stage=stage), updated_utc=utc())
                    atomic(runtime / 'status.json', status)
                    report(status)
                    started = time.monotonic()
                    with (leaf / 'stdout.log').open('x') as out, (leaf / 'stderr.log').open('x') as err:
                        child = subprocess.Popen(argv, cwd=ROOT, env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'),
                            stdin=subprocess.DEVNULL, stdout=out, stderr=err)
                        child_start = int(Path(f'/proc/{child.pid}/stat').read_text().rsplit(')', 1)[1].split()[19])
                        fresh(leaf / 'launch.json', dict(created_utc=utc(), pid=child.pid, start_ticks=child_start,
                            argv=argv, check_only=check, parent_pid=os.getpid()))
                        try:
                            code = child.wait()
                        except BaseException:
                            observed_code = child.poll()
                            fresh(leaf / 'interrupted_owner_observation.json', dict(observed_utc=utc(),
                                child_pid=child.pid, child_start_ticks=child_start,
                                observed_returncode=observed_code, child_exit_unknown=observed_code is None,
                                normal_child_terminal_claimed=False, automatic_retry=False))
                            raise
                    fresh(leaf / 'terminal.json', dict(finished_utc=utc(), returncode=code,
                        wall_seconds=time.monotonic()-started, automatic_retry=False))
                    if code != 0:
                        status['failed'].append(dict(job_id=row['job_id'], stage=stage, returncode=code))
                        raise RuntimeError('Diagnostic infrastructure failure; no retry: '+row['job_id'])
                    if check:
                        checked.add(row['kind'])
                case_path = ROOT / row['out'] / 'case.json'
                case = json.loads(case_path.read_text())
                if (case.get('split') != 'train' or case.get('task') != row['task']
                        or case.get('method') != row['method'] or case.get('source_raw_episode_id') != row['raw']
                        or case.get('cleanup_completed') is not True or case.get('complete') is not True
                        or type(case.get('official_task_success')) is not bool):
                    raise ValueError('Actual diagnostic case identity mismatch')
                sidecar_path = ROOT / row['runtime'] / 'native/diagnostic_sidecar.json'
                sidecar = json.loads(sidecar_path.read_text())
                if (sidecar.get('complete') is not True
                        or sidecar['execution']['diagnostic_events'].get('restored') is not True
                        or sidecar['execution']['case']['sha256'] != sha(case_path)):
                    raise ValueError('Actual completed diagnostic/restoration sidecar required')
                status['completed'].append(dict(job_id=row['job_id'], task=row['task'],
                    method=row['method'], raw=row['raw'], native_success=case['official_task_success'],
                    case=dict(path=str(case_path.relative_to(ROOT)), sha256=sha(case_path)),
                    sidecar=dict(path=str(sidecar_path.relative_to(ROOT)), sha256=sha(sidecar_path))))
                status.update(active=None, updated_utc=utc())
                atomic(runtime / 'status.json', status)
                report(status)
                print(json.dumps(dict(completed=len(status['completed']), expected=48,
                    last=row['job_id'], native_success=case['official_task_success']), ensure_ascii=False), flush=True)
                if len(status['completed']) == 3:
                    status.update(stage='await_canary_review', updated_utc=utc())
                    atomic(runtime / 'status.json', status)
                    report(status)
                    review_path = runtime / 'canary_review.json'
                    while not review_path.exists():
                        if (runtime / 'STOP').exists():
                            raise InterruptedError('Actual STOP while awaiting canary inspection')
                        time.sleep(2)
                    review = json.loads(review_path.read_text())
                    if (review.get('plan_sha256') != expected or review.get('canary_cases') != status['completed']
                            or review.get('diagnostic_observers_reviewed') is not True):
                        raise ValueError('Actual three-canary sidecar review required')
                    status.update(stage='running', updated_utc=utc())
            validate(plan)
            status.update(stage='complete', active=None, updated_utc=utc())
            fresh(runtime / 'terminal.json', dict(complete=True, finished_utc=utc(),
                completed=48, infrastructure_failed=0, automatic_retry=False, formal_VAL_TEST=False))
        except BaseException as error:
            active = status.get('active')
            if active and not any(r['job_id'] == active['job_id'] for r in status['failed']):
                status['failed'].append(dict(job_id=active['job_id'], stage=active['stage'],
                    error_type=type(error).__name__, native_failure_inferred=False))
            status.update(stage='failed', updated_utc=utc(), error=dict(type=type(error).__name__, message=str(error)))
            fresh(runtime / 'failure.json', status)
            raise
        finally:
            atomic(runtime / 'status.json', status)
            report(status)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument('--prepare', action='store_true')
    action.add_argument('--execute', action='store_true')
    parser.add_argument('--plan-sha256')
    args = parser.parse_args()
    if args.prepare:
        prepare()
    elif args.execute:
        execute(args.plan_sha256)
    else:
        print(json.dumps(dict(enabled=False, Native_started=False)))


if __name__ == '__main__':
    main()
