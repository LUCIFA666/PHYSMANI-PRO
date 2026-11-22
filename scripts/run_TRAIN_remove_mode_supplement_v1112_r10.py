"""Two preselected TRAIN remove mode traces after original48 and Buzz4.

Only explicit prepare/execute writes fresh1112 artifacts. Never changes any
earlier source, model, plan, output or control. No automatic retry.
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
SOURCE = 'scripts/run_TRAIN_remove_mode_supplement_v1112_r10.py'
ENTRY = 'scripts/diagnose_TRAIN_remove_mode_supplement_v1111_r10.py'
MODE = 'our_mani/multitask/train_remove_mode_tap_v1111.py'
BASE = 'runtime/framework/TRAIN_zero_success_diagnosis_v1107'
RUNTIME = BASE + '/remove_mode_supplement_v1111'
PLAN = 'artifacts/framework/multitask_r10/TRAIN_zero_success_diagnosis_v1107_setup/remove_mode_supplement_plan_v1112.json'
ART = 'artifacts/framework/multitask_r10/TRAIN_zero_success_diagnosis_v1107/development_demo_only/supplement_remove_mode_v1111'
ORIGINAL_PLAN = 'artifacts/framework/multitask_r10/TRAIN_zero_success_diagnosis_v1107_setup/plan.json'
ORIGINAL_PLAN_SHA = '7af96771e16f33506d8770fd75c010cbdc900e1e86d09b27c9118fb7b8948515'
BUZZ_PLAN = 'artifacts/framework/multitask_r10/TRAIN_zero_success_diagnosis_v1107_setup/buzz_collision_supplement_plan_v1110.json'
BUZZ_PLAN_SHA = '61f26dd6238727cdeb5e7f1aa6dfc179848ebbcdfb24ffee31ec701b70ca5ba9'
BUZZ_RUNTIME = BASE + '/buzz_collision_supplement_v1109'
ENV_SHA = '1c33a2e35c1134b90aebe896da6176a997242e8aa10dd6c9b08c9e4d5730df30'
REPRESENTATIVES = (('remove_cups_from_rotating_frame', 'episode104'),
                   ('remove_cups_from_rotating_frame_high_speed', 'episode1'))
ENTRY_SHA = '9ecfdfaab74c2e373f9aaa8c288c042d320cfbb25b04c9f069eab0fd13eff2d6'
MODE_SHA = 'de567a5c5d139ed9ef0a13904ced9e715b470e17a9d37c64c6877ddc6579cc15'


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


def status_write(status):
    path = ROOT / RUNTIME / 'status.json'
    temporary = path.with_suffix('.tmp.' + str(os.getpid()))
    fresh(temporary, status)
    os.replace(temporary, path)
    report = ROOT / 'TRAIN_取杯身份补充诊断_20261023.txt'
    temporary = report.with_name(report.name + '.tmp.' + str(os.getpid()))
    with temporary.open('x', encoding='utf-8') as stream:
        stream.write('取杯公共持有身份补充诊断（独立TRAIN代表，不计入正式VAL/TEST）\n')
        stream.write('更新时间：' + status['updated_utc'] + '\n')
        stream.write(f"完成：{len(status['completed'])}/2；基础设施错误：{len(status['failed'])}；状态：{status['stage']}\n")
        stream.write('当前：' + str(status['active']) + '\n')
        stream.write('固定普通episode104 CV、高速episode1 CV；记录原关联候选/模式返回，不改变模型、策略或原结果。\n')
        if status.get('error'):
            stream.write('错误：' + json.dumps(status['error'], ensure_ascii=False) + '\n')
    os.replace(temporary, report)


def prepare():
    if (ROOT / PLAN).exists() or (ROOT / RUNTIME / 'launch.json').exists():
        raise FileExistsError('Preserve existing1112 materials; inspect rather than repeat prepare')
    if sha(ROOT / ORIGINAL_PLAN) != ORIGINAL_PLAN_SHA or sha(ROOT / BUZZ_PLAN) != BUZZ_PLAN_SHA:
        raise ValueError('Actual unchanged48/Buzz4 plans required')
    names = (SOURCE, ENTRY, MODE, 'tests/test_TRAIN_remove_mode_supplement_v1111.py',
        'our_mani/multitask/mode_observer.py',
        'scripts/diagnose_TRAIN_ring_postcomplete_v1103_r10.py',
        'scripts/diagnose_TRAIN_remove_postcomplete_v1104_r10.py',
        'our_mani/multitask/train_remove_capture_ast_v892.py',
        'our_mani/multitask/post_completed_TEST_resource_guard_v1106.py',
        'scripts/diagnose_TRAIN_buzz_collision_supplement_v1109_r10.py',
        'scripts/run_TRAIN_buzz_collision_supplement_v1110_r10.py',
        'scripts/run_frozen_system_environment_v820_r10.py', ORIGINAL_PLAN, BUZZ_PLAN)
    pins = {name: sha(ROOT / name) for name in names}
    if pins[ENTRY] != ENTRY_SHA or pins[MODE] != MODE_SHA:
        raise ValueError('Reviewed fixed1111 entry and mode observer required')
    jobs = []
    for task, raw in REPRESENTATIVES:
        config = 'artifacts/framework/multitask_r10/post_training_qualification_v680_setup/' + task + '_ours_configuration.json'
        pins[config] = sha(ROOT / config)
        method = 'cv_geometry'
        jobs.append(dict(task=task, raw=raw, method=method, config=config,
            job_id='_'.join((task, raw, method)), out=f'{ART}/{task}/{raw}/{method}',
            runtime=f'{BASE}/cases/supplement_remove_mode_v1111/{task}/{raw}/{method}'))
    plan = dict(schema='r10_fixed_two_TRAIN_remove_mode_supplement_plan_v1112', created_utc=utc(),
        jobs=jobs, expected=2, unique_parents=2, input_sha256=pins,
        source_selection='fixed normal104CV/high1CV original TRAIN close-evidence representatives; no TEST/outcome filtering',
        original48_plan=dict(path=ORIGINAL_PLAN, sha256=ORIGINAL_PLAN_SHA),
        predecessor_buzz4_plan=dict(path=BUZZ_PLAN, sha256=BUZZ_PLAN_SHA),
        automatic_retry=False, max_native_slots=1, formal_VAL_TEST=False,
        diagnostic_only=True, policy_or_model_modified=False)
    fresh(ROOT / PLAN, plan)
    print(json.dumps(dict(plan=PLAN, sha256=sha(ROOT / PLAN), cases=2)), flush=True)


def argv(row, check=False):
    result = [str(ROOT / '.conda-env/bin/python'), '-B', 'scripts/run_frozen_system_environment_v820_r10.py',
        '--manifest-sha256', ENV_SHA, '--', 'bash', 'scripts/run_local_sim.sh', ENTRY, '--execute',
        '--diagnostic', 'remove', '--config', row['config'], '--task', row['task'], '--raw-episode-id', row['raw'],
        '--method', row['method'], '--out', row['out'], '--runtime', row['runtime']]
    return result + (['--check'] if check else [])


def validate(plan):
    if plan.get('expected') != 2 or len(plan.get('jobs', [])) != 2 or plan.get('automatic_retry') is not False:
        raise ValueError('Exactly two frozen representatives without retry required')
    for name, checksum in plan['input_sha256'].items():
        if sha(ROOT / name) != checksum:
            raise ValueError('Pinned remove-mode supplement input changed: ' + name)


def require_predecessors_finished():
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from scripts import diagnose_TRAIN_buzz_collision_supplement_v1109_r10 as completed
    original = completed.require_main48_complete(ROOT)
    if sha(ROOT / BUZZ_PLAN) != BUZZ_PLAN_SHA:
        raise ValueError('Actual frozen Buzz4 plan required')
    state = json.loads((ROOT / BUZZ_RUNTIME / 'status.json').read_text())
    terminal = json.loads((ROOT / BUZZ_RUNTIME / 'terminal.json').read_text())
    launch = json.loads((ROOT / BUZZ_RUNTIME / 'launch.json').read_text())
    if (state.get('stage') != 'complete' or state.get('expected') != 4
            or len(state.get('completed', [])) != 4 or state.get('failed') != [] or state.get('active') is not None
            or terminal.get('complete') is not True or terminal.get('completed') != 4
            or terminal.get('infrastructure_failed') != 0 or terminal.get('automatic_retry') is not False
            or launch.get('plan') != dict(path=BUZZ_PLAN, sha256=BUZZ_PLAN_SHA)):
        raise ValueError('Buzz4 must complete naturally with no active/failed cases before remove mode supplement')
    try:
        fields = (Path('/proc') / str(launch['pid']) / 'stat').read_text().rsplit(')', 1)[1].split()
    except (FileNotFoundError, ProcessLookupError):
        fields = None
    if fields is not None and int(fields[19]) == launch['start_ticks'] and fields[0] not in ('Z', 'X'):
        raise ValueError('Buzz4 owner is still alive')
    return dict(original48=original, buzz4=dict(complete=True, plan=dict(path=BUZZ_PLAN, sha256=BUZZ_PLAN_SHA),
        status=dict(path=BUZZ_RUNTIME + '/status.json', sha256=sha(ROOT / BUZZ_RUNTIME / 'status.json')),
        terminal=dict(path=BUZZ_RUNTIME + '/terminal.json', sha256=sha(ROOT / BUZZ_RUNTIME / 'terminal.json')),
        launch=dict(path=BUZZ_RUNTIME + '/launch.json', sha256=sha(ROOT / BUZZ_RUNTIME / 'launch.json')),
        original_owner_identity_not_live=True))


def execute(checksum):
    if not checksum or sha(ROOT / PLAN) != checksum:
        raise ValueError('Actual1112 plan SHA required')
    plan = json.loads((ROOT / PLAN).read_text())
    validate(plan)
    prerequisites = require_predecessors_finished()
    base = ROOT / RUNTIME
    base.mkdir(parents=True, exist_ok=True)
    with (base / 'owner.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fresh(base / 'launch.json', dict(created_utc=utc(), pid=os.getpid(),
            start_ticks=int(Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19]),
            plan=dict(path=PLAN, sha256=checksum), prerequisites=prerequisites,
            automatic_retry=False, max_native_slots=1))
        state = dict(schema='r10_TRAIN_two_remove_mode_supplement_status_v1112', expected=2,
            completed=[], failed=[], stage='running', active=None, updated_utc=utc(), formal_VAL_TEST=False)
        try:
            for row in plan['jobs']:
                validate(plan)
                require_predecessors_finished()
                if (ROOT / BASE / 'STOP').exists() or (base / 'STOP').exists():
                    raise InterruptedError('Actual diagnostic STOP observed')
                for check in (True, False):
                    phase = 'CPU_check' if check else 'native'
                    leaf = base / 'launches' / row['job_id'] / phase
                    leaf.mkdir(parents=True, exist_ok=False)
                    state.update(active=dict(job_id=row['job_id'], stage=phase), updated_utc=utc())
                    status_write(state)
                    command = argv(row, check)
                    started = time.monotonic()
                    with (leaf / 'stdout.log').open('x') as out, (leaf / 'stderr.log').open('x') as err:
                        child = subprocess.Popen(command, cwd=ROOT,
                            env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'), stdin=subprocess.DEVNULL,
                            stdout=out, stderr=err)
                        ticks = int(Path(f'/proc/{child.pid}/stat').read_text().rsplit(')', 1)[1].split()[19])
                        fresh(leaf / 'launch.json', dict(created_utc=utc(), pid=child.pid, start_ticks=ticks, argv=command))
                        try:
                            code = child.wait()
                        except BaseException:
                            fresh(leaf / 'owner_interruption.json', dict(child_pid=child.pid, start_ticks=ticks,
                                observed_returncode=child.poll(), normal_terminal_claimed=False))
                            raise
                    fresh(leaf / 'terminal.json', dict(returncode=code, finished_utc=utc(),
                        wall_seconds=time.monotonic() - started, automatic_retry=False))
                    if code:
                        state['failed'].append(dict(job_id=row['job_id'], stage=phase, returncode=code))
                        raise RuntimeError('Remove mode supplement infrastructure failure; no retry')
                case = ROOT / row['out'] / 'case.json'
                sidecar = ROOT / row['runtime'] / 'native/diagnostic_sidecar.json'
                saved = json.loads(sidecar.read_text())
                if (saved.get('complete') is not True or saved['execution']['diagnostic_events'].get('restored') is not True
                        or saved['execution']['case']['sha256'] != sha(case)):
                    raise ValueError('Original completed unchanged case and restored capture/mode observers required')
                state['completed'].append(dict(job_id=row['job_id'], task=row['task'], raw=row['raw'], method=row['method'],
                    case=dict(path=str(case.relative_to(ROOT)), sha256=sha(case)),
                    sidecar=dict(path=str(sidecar.relative_to(ROOT)), sha256=sha(sidecar))))
                state.update(active=None, updated_utc=utc())
                status_write(state)
                print(json.dumps(dict(completed=len(state['completed']), expected=2, last=row['job_id'])), flush=True)
            state.update(stage='complete', updated_utc=utc())
            fresh(base / 'terminal.json', dict(complete=True, completed=2, infrastructure_failed=0,
                finished_utc=utc(), automatic_retry=False, formal_score_contribution=False))
        except BaseException as error:
            if state.get('active') and not state['failed']:
                state['failed'].append(dict(**state['active'], exception_type=type(error).__name__, native_failure_inferred=False))
            state.update(stage='failed', updated_utc=utc(), error=dict(type=type(error).__name__, message=str(error)))
            fresh(base / 'failure.json', state)
            raise
        finally:
            status_write(state)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--prepare', action='store_true')
    group.add_argument('--execute', action='store_true')
    parser.add_argument('--plan-sha256')
    arguments = parser.parse_args()
    if arguments.prepare:
        prepare()
    elif arguments.execute:
        execute(arguments.plan_sha256)
    else:
        print(json.dumps(dict(enabled=False, Native_started=False)))
