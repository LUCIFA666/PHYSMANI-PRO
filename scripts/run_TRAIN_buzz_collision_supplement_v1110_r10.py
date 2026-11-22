"""Four preselected TRAIN collision traces after the original 48 diagnostics.

No original source, model, plan, result or case is modified. Default is inert.
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
SOURCE = 'scripts/run_TRAIN_buzz_collision_supplement_v1110_r10.py'
ENTRY = 'scripts/diagnose_TRAIN_buzz_collision_supplement_v1109_r10.py'
BASE = 'runtime/framework/TRAIN_zero_success_diagnosis_v1107'
RUNTIME = BASE + '/buzz_collision_supplement_v1109'
PLAN = 'artifacts/framework/multitask_r10/TRAIN_zero_success_diagnosis_v1107_setup/buzz_collision_supplement_plan_v1110.json'
ART = 'artifacts/framework/multitask_r10/TRAIN_zero_success_diagnosis_v1107/development_demo_only/supplement_buzz_collision_v1109'
ORIGINAL_PLAN = 'artifacts/framework/multitask_r10/TRAIN_zero_success_diagnosis_v1107_setup/plan.json'
ORIGINAL_PLAN_SHA = '7af96771e16f33506d8770fd75c010cbdc900e1e86d09b27c9118fb7b8948515'
ENV_SHA = '1c33a2e35c1134b90aebe896da6176a997242e8aa10dd6c9b08c9e4d5730df30'


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fresh(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def status_write(status):
    path = ROOT / RUNTIME / 'status.json'
    temp = path.with_suffix('.tmp.' + str(os.getpid()))
    fresh(temp, status)
    os.replace(temp, path)
    report = ROOT / 'TRAIN_穿环补充诊断_20261023.txt'
    draft = report.with_name(report.name + '.tmp.' + str(os.getpid()))
    with draft.open('x', encoding='utf-8') as f:
        f.write('穿环补充调用树诊断（原48例已完成；本补充不计入正式VAL/TEST）\n')
        f.write('更新时间：' + status['updated_utc'] + '\n')
        f.write(f"完成：{len(status['completed'])}/4；基础设施错误：{len(status['failed'])}；状态：{status['stage']}\n")
        f.write('当前：' + str(status['active']) + '\n')
        f.write('固定普通episode10、高速episode102，各CV/Context；模型、策略及原结果保持不变。\n')
        if status.get('error'):
            f.write('错误：' + json.dumps(status['error'], ensure_ascii=False) + '\n')
    os.replace(draft, report)


def prepare():
    if sha(ROOT / ORIGINAL_PLAN) != ORIGINAL_PLAN_SHA:
        raise ValueError('Actual unchanged original 48-case plan required')
    names = (SOURCE, ENTRY, 'scripts/diagnose_TRAIN_ring_postcomplete_v1103_r10.py',
        'our_mani/multitask/train_public_collision_tap_v934.py',
        'our_mani/multitask/train_buzz_failure_tap_v1105.py',
        'our_mani/multitask/post_completed_TEST_resource_guard_v1106.py',
        'scripts/run_frozen_system_environment_v820_r10.py', ORIGINAL_PLAN)
    pins = {name: sha(ROOT / name) for name in names}
    if pins[ENTRY] != '41720a55b74daceb855815b7eb5dd0ad20fb158375d9d7ff2f6e0aefffc64994':
        raise ValueError('Reviewed fixed supplement entry required')
    jobs = []
    for task, raw in (('beat_the_rotating_buzz', 'episode10'),
                      ('beat_the_rotating_buzz_high_speed', 'episode102')):
        config = 'artifacts/framework/multitask_r10/post_training_qualification_v680_setup/' + task + '_ours_configuration.json'
        pins[config] = sha(ROOT / config)
        for method in ('cv_geometry', 'context_geometry'):
            jobs.append(dict(task=task, raw=raw, method=method, config=config,
                job_id='_'.join((task, raw, method)), out=f'{ART}/{task}/{raw}/{method}',
                runtime=f'{BASE}/cases/supplement_buzz_collision_v1109/{task}/{raw}/{method}'))
    p = dict(schema='r10_fixed_four_TRAIN_buzz_collision_supplement_plan_v1110', created_utc=utc(),
        jobs=jobs, expected=4, unique_parents=2, input_sha256=pins,
        source_selection='first original TRAIN parent of each Buzz speed, both geometry methods; chosen before new Buzz16 completion',
        original48_plan=dict(path=ORIGINAL_PLAN, sha256=ORIGINAL_PLAN_SHA),
        automatic_retry=False, max_native_slots=1, formal_VAL_TEST=False,
        diagnostic_only=True, policy_or_model_modified=False)
    fresh(ROOT / PLAN, p)
    print(json.dumps(dict(plan=PLAN, sha256=sha(ROOT / PLAN), cases=4)), flush=True)


def argv(row, check=False):
    a = [str(ROOT / '.conda-env/bin/python'), '-B', 'scripts/run_frozen_system_environment_v820_r10.py',
        '--manifest-sha256', ENV_SHA, '--', 'bash', 'scripts/run_local_sim.sh', ENTRY, '--execute',
        '--diagnostic', 'buzz', '--config', row['config'], '--task', row['task'], '--raw-episode-id', row['raw'],
        '--method', row['method'], '--out', row['out'], '--runtime', row['runtime']]
    return a + (['--check'] if check else [])


def validate(plan):
    for name, checksum in plan['input_sha256'].items():
        if sha(ROOT / name) != checksum:
            raise ValueError('Pinned supplement input changed: ' + name)


def require_original_finished():
    state = json.loads((ROOT / BASE / 'status.json').read_text())
    terminal = json.loads((ROOT / BASE / 'terminal.json').read_text())
    launch = json.loads((ROOT / BASE / 'launch.json').read_text())
    if (state['stage'] != 'complete' or len(state['completed']) != 48 or state['failed'] or state['active'] is not None
            or terminal.get('complete') is not True or terminal.get('completed') != 48 or terminal.get('infrastructure_failed') != 0):
        raise ValueError('Original fixed 48 must complete naturally before any supplement')
    p = Path('/proc') / str(launch['pid']) / 'stat'
    if p.exists():
        fields = p.read_text().rsplit(')', 1)[1].split()
        if int(fields[19]) == launch['start_ticks'] and fields[0] != 'Z':
            raise ValueError('Original owner still alive')


def execute(checksum):
    if not checksum or sha(ROOT / PLAN) != checksum:
        raise ValueError('Actual supplement plan SHA required')
    plan = json.loads((ROOT / PLAN).read_text())
    validate(plan)
    require_original_finished()
    base = ROOT / RUNTIME
    base.mkdir(parents=True, exist_ok=True)
    with (base / 'owner.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fresh(base / 'launch.json', dict(created_utc=utc(), pid=os.getpid(),
            start_ticks=int(Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19]),
            plan=dict(path=PLAN, sha256=checksum), original48_naturally_complete=True,
            automatic_retry=False, max_native_slots=1))
        state = dict(schema='r10_TRAIN_four_buzz_collision_supplement_status_v1110', expected=4,
            completed=[], failed=[], stage='running', active=None, updated_utc=utc(), formal_VAL_TEST=False)
        try:
            for index, row in enumerate(plan['jobs']):
                validate(plan)
                if (ROOT / BASE / 'STOP').exists():
                    raise InterruptedError('Actual diagnostic STOP observed')
                for check in ([True, False] if index == 0 else [False]):
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
                        wall_seconds=time.monotonic()-started, automatic_retry=False))
                    if code:
                        state['failed'].append(dict(job_id=row['job_id'], stage=phase, returncode=code))
                        raise RuntimeError('Supplement infrastructure failure; no retry')
                case = ROOT / row['out'] / 'case.json'
                sidecar = ROOT / row['runtime'] / 'native/diagnostic_sidecar.json'
                saved = json.loads(sidecar.read_text())
                if (not saved['complete'] or not saved['execution']['diagnostic_events']['restored']
                        or saved['execution']['case']['sha256'] != sha(case)):
                    raise ValueError('Completed unchanged case/restored observers required')
                state['completed'].append(dict(job_id=row['job_id'], task=row['task'], raw=row['raw'], method=row['method'],
                    case=dict(path=str(case.relative_to(ROOT)), sha256=sha(case)),
                    sidecar=dict(path=str(sidecar.relative_to(ROOT)), sha256=sha(sidecar))))
                state.update(active=None, updated_utc=utc())
                status_write(state)
                print(json.dumps(dict(completed=len(state['completed']), expected=4, last=row['job_id'])), flush=True)
            state.update(stage='complete', updated_utc=utc())
            fresh(base / 'terminal.json', dict(complete=True, completed=4, infrastructure_failed=0,
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
    p = argparse.ArgumentParser(description=__doc__)
    group = p.add_mutually_exclusive_group()
    group.add_argument('--prepare', action='store_true')
    group.add_argument('--execute', action='store_true')
    p.add_argument('--plan-sha256')
    a = p.parse_args()
    if a.prepare:
        prepare()
    elif a.execute:
        execute(a.plan_sha256)
    else:
        print(json.dumps(dict(enabled=False, Native_started=False)))
