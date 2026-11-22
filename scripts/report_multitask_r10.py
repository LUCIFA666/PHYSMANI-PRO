"""Read-only evidence snapshot across the completed R09 and ongoing R10."""
from datetime import datetime, timezone
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'artifacts/framework/multitask_r10'


def read(path):
    try:
        value = json.loads(path.read_text())
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def write(path, body):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.report.tmp')
    temporary.write_text(body)
    temporary.replace(path)


def full_TRAIN_progress(status, *, root=ROOT):
    """Count canonical committed receipts, with cheap provenance checks.

    Raw attempt directories and a partial queue manifest are never the
    denominator. This is progress reporting; authoritative queue acceptance
    separately hashes all track/model inputs. Do not rehash them every 20 min.
    """
    root = Path(root).resolve()
    result = dict(expected=2800, expected_teacher=1400, expected_hold=1400,
        committed=0, teacher=0, hold=0, parent_sources=0, complete=False,
        configured=False, invalid_receipts=[], accepted_case_ids=[],
        infrastructure_attempt_count=0, infrastructure_attempts=[])
    directory = status.get('active_full_TRAIN_directory')
    pointer = status.get('active_full_TRAIN_plan')
    if not directory or not pointer:
        return result
    try:
        folder = (root / directory).resolve()
        plan_path = (root / (pointer['path'] if isinstance(pointer, dict) else pointer)).resolve()
        if not folder.is_relative_to(root) or not plan_path.is_relative_to(root):
            raise ValueError('full TRAIN progress paths must stay in our-mani')
        raw_plan = plan_path.read_bytes()
        if isinstance(pointer, dict) and hashlib.sha256(raw_plan).hexdigest() != pointer['sha256']:
            raise ValueError('full TRAIN plan file SHA differs')
        plan = json.loads(raw_plan)
        if not isinstance(plan, dict):
            raise ValueError('full TRAIN plan must be a JSON object')
        identity = {k: v for k, v in plan.items() if k != 'plan_sha256'}
        digest = hashlib.sha256(json.dumps(identity, sort_keys=True, ensure_ascii=False,
            allow_nan=False, separators=(',', ':')).encode()).hexdigest()
        cases = {row['case_id']: row for row in plan['cases']}
        parents = {row['parent_source_group'] for row in cases.values()}
        behaviors = Counter(row['behavior'] for row in cases.values())
        if (plan.get('split') != 'train' or plan.get('plan_sha256') != digest
                or len(cases) != len(plan['cases']) or len(cases) != 2800 or len(parents) != 1400
                or behaviors != {'TRAIN_teacher_committed_waypoint': 1400,
                                 'TRAIN_hold_motion_identification': 1400}):
            raise ValueError('full TRAIN fixed 2800/1400 plan identity differs')
        result['configured'] = True
        result['plan_sha256'] = digest
        for path in sorted(folder.glob('infrastructure_attempts/*/*/parent_receipt.json')):
            attempt = read(path)
            if attempt.get('status') == 'infrastructure_error':
                result['infrastructure_attempts'].append(dict(
                    path=str(path.relative_to(root)), case_id=attempt.get('case_id'),
                    process_id=attempt.get('process_id'), returncode=attempt.get('returncode'),
                    policy_failure_label_assigned=attempt.get('policy_failure_label_assigned', False)))
        result['infrastructure_attempt_count'] = len(result['infrastructure_attempts'])
        accepted = []
        for path in sorted(folder.glob('cases/train/*/*/case.json')):
            try:
                record = read(path)
                case = cases.get(record.get('case_id'))
                if case is None or path != folder/'cases/train'/case['task']/case['case_id']/'case.json':
                    raise ValueError('unknown or noncanonical case identity')
                if record.get('status') != 'completed' or record.get('plan_sha256') != digest:
                    raise ValueError('case not committed under this full plan')
                for key in ('case_id', 'split', 'task', 'raw_episode_id', 'parent_source_group'):
                    if record.get(key) != case[key]:
                        raise ValueError('case header differs: ' + key)
                entry = record.get('entry', {})
                if not isinstance(entry, dict):
                    raise ValueError('case entry must be a JSON object')
                for key in ('split', 'task', 'raw_episode_id', 'parent_raw_episode_id',
                            'parent_source_group', 'behavior', 'behavior_replica', 'teacher_used',
                            'context_primary_view', 'interaction_eligible', 'maximum_collection_ticks'):
                    if entry.get(key) != case[key]:
                        raise ValueError('case entry differs: ' + key)
                parent = record.get('parent_process_receipt', {})
                if not isinstance(parent, dict):
                    raise ValueError('parent process receipt must be a JSON object')
                parent_pid = parent.get('process_id')
                if type(parent_pid) is not int or parent_pid <= 0:
                    raise ValueError('missing raw parent process identity')
                child = (root / parent.get('child_case_path', '')).resolve()
                relative = child.relative_to(folder).parts
                if (len(relative) != 9 or relative[0] != 'infrastructure_attempts'
                        or relative[1] != case['case_id'] or not relative[2].isdigit()
                        or relative[3:] != ('output', 'cases', 'train', case['task'], case['case_id'], 'case.json')):
                    raise ValueError('raw child receipt is outside the matching attempt')
                child_bytes = child.read_bytes()
                if hashlib.sha256(child_bytes).hexdigest() != parent.get('child_case_sha256'):
                    raise ValueError('raw child receipt SHA differs')
                child_record = json.loads(child_bytes)
                if {k: v for k, v in record.items() if k != 'parent_process_receipt'} != child_record:
                    raise ValueError('canonical receipt differs from its raw child')
                parent_record = read(folder/'infrastructure_attempts'/case['case_id']/relative[2]/'parent_receipt.json')
                if (parent_record.get('case_id') != case['case_id']
                        or parent_record.get('status') != 'completed'
                        or parent_record.get('returncode') != 0
                        or parent_record.get('process_id') != parent_pid
                        or parent_record.get('command') != parent.get('command')):
                    raise ValueError('raw parent completion/process identity differs or is not yet committed')
                command = parent['command']
                if (not isinstance(command, list) or not all(isinstance(item, str) for item in command)
                        or command[command.index('--case-id') + 1] != case['case_id']
                        or Path(command[command.index('--plan') + 1]).resolve() != plan_path):
                    raise ValueError('parent command differs from case or plan')
                accepted.append(case)
            except (OSError, ValueError, TypeError, KeyError, IndexError) as error:
                result['invalid_receipts'].append(dict(path=str(path.relative_to(root)), reason=str(error)))
        result.update(committed=len(accepted),
            teacher=sum(row['teacher_used'] is True for row in accepted),
            hold=sum(row['context_primary_view'] is True for row in accepted),
            parent_sources=len({row['parent_source_group'] for row in accepted}),
            accepted_case_ids=[row['case_id'] for row in accepted],
            complete=len(accepted) == 2800)
    except (OSError, ValueError, TypeError, KeyError) as error:
        result['configuration_error'] = str(error)
    return result


def full_TRAIN_runtime(status, *, root=ROOT, proc_root=Path('/proc')):
    """Observe only direct collection Python workers for the active plan."""
    root = Path(root).resolve()
    pointer = status.get('active_full_TRAIN_plan')
    pids = []
    if pointer:
        try:
            plan = (root / (pointer['path'] if isinstance(pointer, dict) else pointer)).resolve()
            collector = str(root/'scripts/collect_multitask_training_r10.py')
            for proc in Path(proc_root).iterdir():
                if not proc.name.isdigit():
                    continue
                try:
                    args = (proc/'cmdline').read_bytes().decode().split('\0')
                    if (not Path(args[0]).name.startswith('python') or collector not in args
                            or '--collect-case' not in args or '--plan' not in args):
                        continue
                    worker_plan = (root / args[args.index('--plan') + 1]).resolve()
                    if worker_plan == plan:
                        pids.append(int(proc.name))
                except (OSError, UnicodeError, ValueError, IndexError):
                    continue
        except (OSError, TypeError, KeyError):
            pass
    return dict(active_worker_count=len(pids), active_worker_pids=sorted(pids),
        sampled_utc=datetime.now(timezone.utc).isoformat(),
        source='live /proc cmdline: direct Python --collect-case and exact active plan',
        zero_workers_does_not_imply_completed=True)


def main():
    now = datetime.now(timezone.utc)
    final = read(ROOT / 'artifacts/framework/aligned_r09/final_report.json')
    demos = read(ROOT / 'artifacts/framework/aligned_r09/demo_index.json')
    status = read(BASE / 'status.json')
    full_train = full_TRAIN_progress(status)
    full_runtime = full_TRAIN_runtime(status)
    smoke_root = ROOT / status.get('active_smoke_directory',
        'artifacts/framework/multitask_r10/TRAIN_smoke_collection_v3')
    smoke = read(smoke_root / 'manifest.json')
    smoke_queue = read(smoke_root / 'queue_report.json')
    smoke_infra = [row for row in smoke_queue.get('receipts', [])
                   if row.get('status') == 'infrastructure_error']
    # Count committed parent receipts only, never raw child attempts twice.
    smoke_cases = [read(p) for p in smoke_root.glob('cases/train/*/*/case.json')]
    smoke_count = sum(r.get('status') == 'completed' for r in smoke_cases)
    smoke_expected = status.get('TRAIN_smoke_attempts', len(smoke.get('entries', [])))
    reset_root = ROOT / status.get('active_reset_directory',
        'artifacts/framework/multitask_r10/TRAIN_reset_repeat_v1')
    reset = read(reset_root / 'pending_report.json')
    reset_records = [read(p) for p in reset_root.glob('cases/*/case.json')]
    reset_count = sum(r.get('completed') is True for r in reset_records)
    errors = [str(p.relative_to(ROOT)) for folder in (smoke_root, reset_root)
              for p in folder.rglob('infrastructure_error.json')]
    try:
        gpu = subprocess.check_output(['nvidia-smi',
            '--query-gpu=utilization.gpu,memory.used,memory.free', '--format=csv,noheader'],
            text=True, timeout=10).strip()
    except (OSError, subprocess.SubprocessError):
        gpu = 'unavailable'
    lines = [f'北京时间：{now.astimezone(ZoneInfo("Asia/Shanghai")):%Y-%m-%d %H:%M:%S}',
             '当前两个任务：正式 TEST ' + str(final.get('total_test_runs', 0)) + '/400；硬协议验收：'
             + ('通过' if final.get('protocol_acceptance_passed') is True else '尚未通过'),
             f'两个任务 demo：{demos.get("completed", 0)}/10；每组每任务固定一个案例。']
    for method, tasks in final.get('variants', {}).items():
        lines.append(method + '：' + '；'.join(
            f'{task} {row["successes"]}/{row["episodes"]}' for task, row in tasks.items()))
    lines.extend(['R09 开发性能筛选未通过；观察 VAL 后按用户指令完成固定候选评估，未重调模型。',
        '正式统计与补录 demo 结果分开；属于共享 CAD/语义绑定/静态地图条件下的适配对比。',
        '剩余 14 个任务：1400 TRAIN 源、275 个实际 VAL 源、560 个官方 TEST 源。',
        'R10 阶段：' + str(status.get('stage', '等待启动')),
        f'当前 TRAIN smoke：{smoke_count}/{smoke_expected} 份已提交父进程记录。',
        f'全源 TRAIN 采集开始：{status.get("TRAIN_v5_behavior_design", {}).get("actual_full_collection_started", False)}；'
        f'教师 smoke 原生成功：{status.get("TRAIN_smoke_teacher_native_successes", 0)}/14（不是已训练策略成功率）。',
        f'全量 TRAIN 已提交记录：{full_train["committed"]}/2800；'
        f'teacher {full_train["teacher"]}/1400；hold {full_train["hold"]}/1400；'
        f'涉及原始父源 {full_train["parent_sources"]}/1400。',
        f'全量 TRAIN 基础设施失败尝试：{full_train["infrastructure_attempt_count"]}；'
        '保留失败 raw attempts，不计策略失败。',
        f'当前全量采集 worker 进程：{full_runtime["active_worker_count"]}（实时进程核查；0不代表已完成）。',
        f'全量 canonical 记录核查异常/尚未提交父进程记录：{len(full_train["invalid_receipts"])}；'
        '按 canonical case + plan 身份 + 原始父/子进程记录计数；采集完成不等于模型已训练或策略成功。',
        f'TRAIN 重复 reset：{reset_count}/56 份已提交记录；正式冻结：'
            + ('是' if reset.get('automatically_frozen') is True else '否'),
        f'Context 真训练完成：{status.get("real_context_training_completed", False)}；'
        f'交互库真拟合完成：{status.get("real_interaction_library_fit_completed", False)}；'
        f'成功率模型真拟合完成：{status.get("real_success_model_fit_completed", False)}。',
        f'剩余任务正式 TEST：{status.get("remaining_official_TEST_cases_completed", 0)}/2800。',
        '原始 TRAIN 动作与阶段作为离线先验；新原生轨迹是源分组的合成增强，不是 HDF5 运动重放。',
        f'Smoke 基础设施失败尝试：{len(smoke_infra)}；其他错误记录：{len(errors)}；不计为策略失败。',
        f'GPU：{gpu}',
        'FASTER 排除；原 NGFF 训练保持停止。',
        '对话进度由主代理每 20 分钟发送；本文件为本地快照。'])
    snapshot = dict(created_utc=now.isoformat(), status=status,
        R09_formal_TEST=final.get('total_test_runs', 0),
        R09_protocol_passed=final.get('protocol_acceptance_passed'),
        R09_demos=demos.get('completed', 0), TRAIN_smoke_committed=smoke_count,
        TRAIN_smoke_manifest=smoke, TRAIN_reset_repeat_committed=reset_count,
        TRAIN_smoke_infrastructure_attempts=smoke_infra,
        full_TRAIN_progress=full_train,
        full_TRAIN_runtime=full_runtime,
        TRAIN_reset_report=reset, infrastructure_error_records=errors, gpu=gpu)
    name = now.strftime('%Y%m%dT%H%M%SZ')
    body = '\n'.join(lines) + '\n'
    write(BASE / 'progress' / (name + '.json'), json.dumps(snapshot, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    write(BASE / 'progress' / (name + '.txt'), body)
    for filename in ('最新项目状态.txt', '二十分钟进度报告.txt', '十分钟进度报告.txt'):
        write(ROOT / filename, body)
    print(body, end='', flush=True)


if __name__ == '__main__':
    main()
