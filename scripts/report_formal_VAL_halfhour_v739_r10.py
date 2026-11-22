"""Write a read-only Chinese progress report for genuine formal VAL runs.

This stdlib-only process never launches cases or selects TEST data. The source
configuration can point to a repaired VAL namespace without restarting it.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import threading
import time
import unicodedata
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
ZONE = ZoneInfo('Asia/Shanghai')
METHODS = ('cv_geometry', 'context_geometry', 'physmani')
LABELS = dict(cv_geometry='CV+几何', context_geometry='Context+几何', physmani='PhysMani')
FAMILIES = (
    ('beat_the_rotating_buzz', '旋转电线穿环'),
    ('insert_onto_rotating_peg', '向旋转插杆套环'),
    ('moving_basketball_in_hoop', '向移动篮筐投球'),
    ('pick_moving_target_on_the_table', '抓取桌上移动方块'),
    ('push_moving_button', '按压移动按钮'),
    ('put_rubbish_in_moving_bin', '向移动垃圾桶投放垃圾'),
    ('place_cups_on_rotating_frame', '向旋转杯架挂杯子'),
    ('remove_cups_from_rotating_frame', '从旋转杯架取下杯子'),
)
TASK_LABELS = {name + suffix: label + '（' + speed + '）'
               for name, label in FAMILIES
               for suffix, speed in (('', '普通'), ('_high_speed', '高速'))}


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def clock_text(timestamp):
    return datetime.fromtimestamp(timestamp, ZONE).strftime('%Y-%m-%d %H:%M:%S（北京时间）')


def atomic_text(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp.' + str(os.getpid()))
    with temporary.open('w', encoding='utf-8') as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_json(path, value):
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def start_ticks(pid):
    try:
        return int(Path('/proc', str(pid), 'stat').read_text().rsplit(')', 1)[1].split()[19])
    except (OSError, ValueError, IndexError):
        return None


def read_json(path, errors, *, missing_ok=True):
    try:
        value = json.loads(path.read_text(encoding='utf-8'))
        if not isinstance(value, dict):
            raise ValueError('JSON 顶层必须为对象')
        return value
    except FileNotFoundError:
        if not missing_ok:
            errors.append(str(path) + '：文件尚不可用')
    except (OSError, ValueError) as error:
        errors.append(str(path) + '：' + str(error))
    return {}


def project_path(value, field, prefix):
    if not isinstance(value, str) or not value:
        raise ValueError('source.json 缺少 ' + field)
    path = (ROOT / value).resolve()
    if not path.is_relative_to(ROOT / prefix):
        raise ValueError(field + ' 必须位于项目的 ' + prefix + ' 内')
    return path


def load_source(path):
    raw = path.read_bytes()
    config = json.loads(raw)
    if not isinstance(config, dict):
        raise ValueError('source.json 顶层必须为对象')
    runtime = project_path(config.get('runtime'), 'runtime', 'runtime/framework')
    controller = project_path(config.get('controller'), 'controller', 'runtime/framework')
    release = project_path(config.get('validation_release'), 'validation_release',
                           'artifacts/framework/multitask_r10')
    if 'formal_VAL_' not in runtime.name or release.name != 'VAL_release.json':
        raise ValueError('仅允许正式 VAL runtime 与 VAL_release.json')
    return dict(runtime=runtime, controller=controller, validation_release=release), sha_bytes(raw)


def independent_release_complete(release):
    return (release.get('schema') == 'r10_full16_common_author_RML_validation_release_v531'
            and release.get('complete') is True and release.get('split') == 'validation'
            and release.get('source_cases') == 315 and release.get('method_cases') == 945
            and release.get('completed_method_cases') == 945
            and release.get('formal_test_permitted') is True
            and release.get('TEST_used') is False and release.get('TEST_used_for_fitting') is False
            and release.get('methods') == list(METHODS)
            and isinstance(release.get('tasks'), list)
            and len(release['tasks']) == 16 and set(release['tasks']) == set(TASK_LABELS))


def collect(source):
    errors = []
    runtime, controller = source['runtime'], source['controller']
    paths = dict(status=runtime / 'status.json', plan=runtime / 'plan.json',
                 terminal=runtime / 'terminal.json', controller=controller / 'status.json',
                 failure=controller / 'failure.json', release=source['validation_release'])
    documents = {key: read_json(path, errors) for key, path in paths.items()}
    plan, status = documents['plan'], documents['status']
    jobs = plan.get('jobs', [])
    by_id, expected = {}, Counter()
    plan_valid = bool(jobs)
    if not isinstance(jobs, list):
        errors.append('VAL plan.jobs 不是列表')
        jobs, plan_valid = [], False
    for job in jobs:
        if (not isinstance(job, dict) or job.get('task') not in TASK_LABELS
                or job.get('method') not in METHODS or not isinstance(job.get('job_id'), str)
                or not job['job_id'] or job['job_id'] in by_id):
            errors.append('VAL plan 含未知任务、方法或重复 job_id')
            plan_valid = False
            continue
        if '--split' not in job.get('argv', []):
            errors.append('VAL plan 缺少显式 split')
            plan_valid = False
        else:
            argv = job['argv']
            offset = argv.index('--split')
            if offset + 1 >= len(argv) or argv[offset + 1] != 'validation':
                errors.append('拒绝将非 VAL 作业纳入报告')
                plan_valid = False
                continue
        by_id[job['job_id']] = job
        expected[job['task'], job['method']] += 1
    if jobs and (len(by_id) != 945 or len(expected) != 48
                 or any(expected[t, m] != expected[t, METHODS[0]]
                        for t in TASK_LABELS for m in METHODS)):
        errors.append('VAL plan 不是完整 16任务×3方法、945条正式评测计划')
        plan_valid = False
    completed, success = Counter(), Counter()
    records = status.get('completed', {})
    if not isinstance(records, dict):
        errors.append('VAL status.completed 不是对象')
        records = {}
    if plan_valid:
        for name, result in records.items():
            if name not in by_id:
                errors.append('completed 出现 plan 之外的作业：' + str(name))
                continue
            if (not isinstance(result, dict) or result.get('complete') is not True
                    or type(result.get('native_success')) is not bool):
                errors.append('completed 缺少干净完成标记或明确 native_success：' + str(name))
                continue
            job = by_id[name]
            group = job['task'], job['method']
            completed[group] += 1
            success[group] += int(result['native_success'])
    failed = status.get('failed', {})
    failed_count = len(failed) if isinstance(failed, (dict, list)) else 0
    if documents['terminal'].get('complete') is False:
        errors.append('VAL owner terminal.complete=False；等待主执行器修复或切换来源')
    if documents['failure']:
        errors.append('控制器记录基础设施错误：' + json.dumps(documents['failure'], ensure_ascii=False))
    release_complete = independent_release_complete(documents['release'])
    if release_complete and (not plan_valid or sum(completed.values()) != 945):
        errors.append('独立 VAL release 已通过，但当前 plan/status 尚未完整对应945条；等待数据源同步')
        release_complete = False
    data_time = None
    if paths['status'].exists():
        try:
            data_time = paths['status'].stat().st_mtime
        except OSError:
            pass
    if data_time is None:
        for key in ('terminal', 'controller', 'plan'):
            try:
                data_time = paths[key].stat().st_mtime
                break
            except OSError:
                pass
    return dict(expected=expected if plan_valid else Counter(), completed=completed, success=success,
                plan_valid=plan_valid, failed=failed_count, status=status,
                controller=documents['controller'], data_time=data_time,
                release_complete=release_complete, errors=sorted(set(errors)))


def pad(value, width):
    size = sum(2 if unicodedata.east_asian_width(c) in ('W', 'F') else 1 for c in value)
    return value + ' ' * max(0, width - size)


def percentage(successes, count):
    return f'{100.0 * successes / count:.2f}%' if count else '待完成'


def render(data, source, now, next_update, reason, history):
    final = data['release_complete']
    phase = '最终结果：完整945条VAL已通过独立审计' if final else '部分进度：尚未通过完整独立VAL审计，非最终结果'
    total = sum(data['completed'].values())
    status = data['status']
    active = status.get('active', [])
    active_count = len(active) if isinstance(active, (list, dict)) else status.get('native_slots', 0)
    rows = [
        '正式 VAL 结果更新', phase,
        '最近更新时间：' + clock_text(now),
        '数据源更新时间：' + (clock_text(data['data_time']) if data['data_time'] else '等待正式VAL数据'),
        '下一次更新时间：' + (clock_text(next_update) if next_update else '本报告进程已结束'),
        '本次更新原因：' + reason,
        '正式VAL来源：' + str(source['runtime'].relative_to(ROOT)) if source else '正式VAL来源：等待 source.json',
        f'总体完成：{total}/945；基础设施 failed：{data["failed"]}',
        f'资源：active={active_count}；GPU={status.get("GPU_slots", 0)}/1',
        '控制器阶段：' + str(data['controller'].get('stage', data['controller'].get('status', '等待状态'))),
        '', '三种方法总计（成功 / 已完成 / 应评；成功率仅以已完成样本为分母）',
    ]
    for method in METHODS:
        done = sum(data['completed'][task, method] for task in TASK_LABELS)
        good = sum(data['success'][task, method] for task in TASK_LABELS)
        expected = str(sum(data['expected'][task, method] for task in TASK_LABELS)) if data['plan_valid'] else '等待计划'
        rows.append(f'{LABELS[method]}：{good} / {done} / {expected}；已完成样本成功率 {percentage(good, done)}')
    rows.extend(['', '16任务明细（各格为成功/已完成；0/0表示待完成，不计为失败）'])
    headers, widths = ('任务', 'CV 成功/完成', 'Context 成功/完成', 'PhysMani 成功/完成', '每组应评'), (34, 18, 21, 22, 10)
    rows.append(' | '.join(pad(x, width) for x, width in zip(headers, widths)))
    rows.append('-' * 116)
    for task, label in TASK_LABELS.items():
        values = [label]
        for method in METHODS:
            done, good = data['completed'][task, method], data['success'][task, method]
            values.append(f'{good}/{done}' if done else '0/0（待完成）')
        values.append(str(data['expected'][task, METHODS[0]]) if data['plan_valid'] else '等待计划')
        rows.append(' | '.join(pad(x, width) for x, width in zip(values, widths)))
    if not final:
        rows.extend(['', '当前仅反映已完成的正式VAL样本；原任务失败保留，完整独立审计前不作为最终比较结果。'])
    if data['errors']:
        rows.extend(['', '当前读取 / 基础设施状态（报告器继续等待修复）：'])
        rows.extend('- ' + error for error in data['errors'])
    rows.extend(['', '历次更新时间摘要（仅正式VAL，不混入TRAIN）'])
    rows.extend(item['summary'] for item in history)
    return '\n'.join(rows) + '\n'


def read_history(path):
    if not path.exists():
        return []
    values = []
    for line in path.read_text(encoding='utf-8').splitlines():
        try:
            value = json.loads(line)
            if isinstance(value, dict) and isinstance(value.get('summary'), str):
                values.append(value)
        except ValueError:
            continue
    return values


def save_report(state, out, data, source, source_sha, interval, reason, history, *, stopping=False):
    now = time.time()
    finished = stopping or data['release_complete']
    next_update = None if finished else now + interval
    methods = {}
    for method in METHODS:
        methods[method] = dict(successes=sum(data['success'][task, method] for task in TASK_LABELS),
                               completed=sum(data['completed'][task, method] for task in TASK_LABELS))
    summary = (clock_text(now) + f'｜完成 {sum(data["completed"].values())}/945｜failed {data["failed"]}｜'
               + '；'.join(f'{LABELS[m]} {v["successes"]}/{v["completed"]}' for m, v in methods.items())
               + '｜' + reason)
    stamp = datetime.fromtimestamp(now, ZONE).strftime('%Y%m%dT%H%M%S.%f')
    snapshot = state / 'snapshots' / (stamp + '.txt')
    item = dict(updated_at=clock_text(now), timestamp=now, summary=summary,
                source_config_sha256=source_sha, runtime=str(source['runtime']) if source else None,
                completed=sum(data['completed'].values()), expected=945,
                infrastructure_failed=data['failed'], methods=methods,
                independently_released=data['release_complete'], reason=reason,
                snapshot=str(snapshot), next_update_at=clock_text(next_update) if next_update else None)
    history.append(item)
    report = render(data, source, now, next_update, reason, history)
    atomic_text(snapshot, report)
    with (state / 'history.jsonl').open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(item, ensure_ascii=False) + '\n')
        handle.flush()
        os.fsync(handle.fileno())
    atomic_text(out, report)
    atomic_json(state / 'status.json', item)
    chat_text = (f'VAL更新（{clock_text(now)}）：完成 {sum(data["completed"].values())}/945，'
                 f'基础设施 failed {data["failed"]}。\n'
                 + '；'.join(f'{LABELS[m]} 成功/已完成 {v["successes"]}/{v["completed"]}'
                            for m, v in methods.items())
                 + ('。\n完整945条VAL已通过独立审计。' if data['release_complete']
                    else '。\n当前为部分进度，非最终结果。')
                 + '\n' + reason + '\nTXT：' + str(out))
    atomic_json(state / 'chat_ready.json', dict(
        snapshot_id=stamp, created_utc=datetime.fromtimestamp(now, timezone.utc).isoformat(),
        next_update_utc=datetime.fromtimestamp(next_update, timezone.utc).isoformat() if next_update else None,
        text=chat_text))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source-config', 'state', 'out'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--interval-seconds', type=float, default=1800)
    args = parser.parse_args()
    if args.interval_seconds <= 0:
        parser.error('--interval-seconds 必须为正数')
    state, out, config_path = ((ROOT / x).resolve() for x in (args.state, args.out, args.source_config))
    if not state.is_relative_to(ROOT / 'runtime/framework'):
        parser.error('--state 必须位于项目 runtime/framework 内')
    if out == config_path or out == Path(__file__).resolve() or out.suffix.lower() != '.txt':
        parser.error('--out 必须为独立 TXT 输出文件')
    state.mkdir(parents=True, exist_ok=True)
    with (state / 'single_instance.lock').open('a') as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit('该 state 的VAL报告器已运行，拒绝重复启动')
        old = read_json(state / 'launch.json', [])
        if (old.get('pid') != os.getpid() and old.get('start_ticks') is not None
                and start_ticks(old.get('pid')) == old.get('start_ticks')):
            raise SystemExit('旧VAL报告器 PID/start_ticks 仍存活，拒绝重复启动')
        event = threading.Event()
        for signum in (signal.SIGINT, signal.SIGTERM):
            signal.signal(signum, lambda *_: event.set())
        atomic_json(state / 'launch.json', dict(pid=os.getpid(), start_ticks=start_ticks(os.getpid()),
                    source_sha256=sha_bytes(Path(__file__).read_bytes()), source_config=str(config_path),
                    out=str(out), interval_seconds=args.interval_seconds, started_at=clock_text(time.time()),
                    native_started=False, GPU_started=False, TEST_read=False))
        history = read_history(state / 'history.jsonl')
        source, source_sha, last_error_signature = None, None, None
        next_due, first = 0., True
        while True:
            source_error = None
            changed = False
            try:
                candidate, candidate_sha = load_source(config_path)
                changed = candidate_sha != source_sha
                source, source_sha = candidate, candidate_sha
            except (OSError, ValueError) as error:
                source_error = 'source.json 暂不可用：' + str(error)
            if source:
                data = collect(source)
            else:
                data = dict(expected=Counter(), completed=Counter(), success=Counter(), plan_valid=False,
                            failed=0, status={}, controller={}, data_time=None, release_complete=False, errors=[])
            if source_error:
                data['errors'].append(source_error)
            stop = (ROOT / 'runtime/framework/multitask_r10_STOP').exists() or event.is_set()
            signature = sha_bytes(json.dumps(dict(errors=data['errors'], failed=data['failed']),
                                             ensure_ascii=False, sort_keys=True).encode())
            infrastructure_changed = last_error_signature is not None and signature != last_error_signature
            last_error_signature = signature
            if first or changed or stop or data['release_complete'] or infrastructure_changed or time.monotonic() >= next_due:
                reason = ('完整独立VAL审计通过' if data['release_complete'] else
                          '收到全局STOP或报告器停止信号，保存部分结果' if stop else
                          '启动立即更新' if first else '正式VAL数据源已更新' if changed else
                          '基础设施或读取状态变化' if infrastructure_changed else '每30分钟定时更新')
                save_report(state, out, data, source, source_sha, args.interval_seconds, reason, history, stopping=stop)
                first = False
                next_due = time.monotonic() + args.interval_seconds
            if stop or data['release_complete']:
                atomic_json(state / 'terminal.json', dict(complete=True, finished_at=clock_text(time.time()),
                            final_VAL_independently_released=data['release_complete'], user_or_signal_stop=stop,
                            native_started=False, GPU_started=False, TEST_read=False))
                return
            event.wait(min(10., max(.1, next_due - time.monotonic())))


if __name__ == '__main__':
    main()
