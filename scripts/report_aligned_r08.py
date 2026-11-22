"""Report r08 nested-calibration repair using the preserved r07 collection.

The r08 pretraining data audit is a frozen snapshot. This reporter only reruns
the r07 data auditor and reads its live audit; it never rewrites the r08 audit.
"""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'artifacts/framework/aligned_r08'
DATA_BASE = ROOT / 'artifacts/framework/aligned_r07'
TASKS = ('pick_moving_target_on_the_table', 'pick_moving_target_on_the_table_high_speed')


def read(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.report.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def compact_metrics(metrics):
    return {key: metrics.get(key) for key in ('n', 'positive', 'brier', 'log_loss', 'auroc', 'ece_5_bins',
                                             'mean_probability', 'prevalence')} if metrics else None


def model_summary(report):
    metadata = report.get('metadata', {})
    per_task = {}
    for task in TASKS:
        selection = report.get('model_selection', {}).get(task, {})
        chosen = selection.get('selected')
        candidate = next((c for c in selection.get('candidates', []) if c.get('candidate') == chosen), {})
        crossfit = report.get('calibration_crossfit', {}).get(task, {})
        per_task[task] = dict(selected_algorithm=chosen,
            ranking_enabled=selection.get('ranking_enabled', metadata.get('per_task', {}).get(task, {}).get('ranking_enabled')),
            grouped_TRAIN_CV=compact_metrics(candidate.get('out_of_source_TRAIN')),
            grouped_TRAIN_prior=compact_metrics(selection.get('grouped_TRAIN_task_constant_prior')),
            calibration_crossfit_valid=crossfit.get('valid'), calibration_crossfit_reasons=crossfit.get('reasons'),
            calibration_crossfit=compact_metrics(report.get('calibration_source_crossfit', {}).get(task)),
            calibration_crossfit_prior=compact_metrics(report.get('calibration_source_crossfit_prior', {}).get(task)),
            fitted_CAL_development_only=compact_metrics(report.get('per_task', {}).get(task)),
            source_coverage=report.get('source_coverage', {}).get(task))
        nested = report.get('nested_calibration_selection', {}).get(task, {})
        if nested:
            per_task[task]['nested_CAL_selection'] = {key: nested.get(key) for key in
                ('selected', 'selected_correction', 'source_coverage', 'fallback_reason')}
            # Each outer fold selects its own calibrator without its held
            # sources. The final full-CAL mode cannot substitute for this.
            per_task[task]['outer_crossfit_calibrators'] = [dict(
                held_fold=fold.get('held_fold'), correction=fold.get('correction'),
                selected=fold.get('inner_selection', {}).get('selected'),
                fallback_reason=fold.get('inner_selection', {}).get('fallback_reason'),
                fit_source_count=len(fold.get('fit_sources', [])),
                held_source_count=len(fold.get('held_sources', [])))
                for fold in crossfit.get('folds', [])]
    return dict(train_rollouts=report.get('train_episodes'), train_attempts=report.get('train_attempts'),
                validation_rollouts=report.get('validation_episodes'), validation_attempts=report.get('validation_attempts'),
                context_dim=metadata.get('context_dim'), calibration_crossfit_valid=report.get('calibration_crossfit_valid'),
                formal_test_used=metadata.get('formal_test_used'),
                classifier_retrained=metadata.get('classifier_retrained'),
                classifier_selection_used_CAL=metadata.get('classifier_selection_used_CAL'),
                calibration_selector_used_CAL=metadata.get('calibration_selector_used_CAL'), per_task=per_task)


def main():
    now = datetime.now(timezone.utc)
    status = read(BASE / 'status.json') or dict(stage='preparing_nested_calibration',
                                               description='准备嵌套校准修复')
    if (DATA_BASE / 'collection_inputs.json').exists():
        try:
            audit_process = subprocess.run([sys.executable, 'scripts/audit_candidate_data_r07.py'], cwd=ROOT,
                                           capture_output=True, text=True, timeout=50)
            audit_error = audit_process.stderr[-1500:] if audit_process.returncode else None
        except (OSError, subprocess.TimeoutExpired) as error:
            audit_error = str(error)
        if audit_error is not None:
            stop = ROOT / 'runtime/framework/success_r08_STOP'
            stop.parent.mkdir(parents=True, exist_ok=True)
            stop.write_text('r08 source data audit failed; inspect artifacts/framework/aligned_r07/data_quality_audit.json\n')
            status.update(stage='data_quality_needs_attention',
                description='R07数据源审计未通过，已写R08停止标志；修复前阻止继续拟合与评估。',
                audit_error=audit_error)
            write(BASE / 'status.json', status)
    audit = read(DATA_BASE / 'data_quality_audit.json')
    lines = ['r08 成功率模块嵌套校准修复进度',
        f'北京时间：{now.astimezone(ZoneInfo("Asia/Shanghai")):%Y-%m-%d %H:%M:%S}',
        f"阶段：{status.get('stage', '尚未启动')}", status.get('description', ''),
        '复用已完成的R07作者OPENGL域640次采集与R06 TRAIN；保留原始预测头，修复校准选择。',
        '校准候选包含identity/shrink；按来源嵌套交叉验证，避免校准损害已有预测效果。',
        'FASTER 排除，NGFF 训练保持停止。']
    collections = {}
    for predictor in ('cv', 'context'):
        for split in ('train', 'validation'):
            reports = [read(p) for p in DATA_BASE.glob(f'collection_{predictor}_{split}_*/report.json')]
            row = {key: sum(r.get(key, 0) for r in reports) for key in
                   ('completed', 'attempts', 'positive', 'source_episodes')}
            row['expected'] = 200 if split == 'train' else 120
            verified = audit.get('counts', {}).get(predictor + '_' + split, {})
            # Audit reads committed records; shard reports can lag by one write.
            row['audited_completed'] = verified.get('completed_rollouts')
            row['audited_per_task'] = verified.get('per_task', {})
            collections[predictor + '_' + split] = row
            lines.append(f"{predictor} {split}：{row['completed']}/{row['expected']} 完整执行，"
                         f"候选尝试 {row['attempts']}，原生任务成功标签 {row['positive']}。")
    lines.append(f"保留R07采集合计：{sum(r['completed'] for r in collections.values())}/640。"
                 f"数据审计：{'通过' if audit.get('passed') else '未通过或尚未生成'}。")
    models = {}
    for predictor in ('cv', 'context'):
        report = read(BASE / f'{predictor}_success/report.json')
        if report:
            models[predictor] = model_summary(report)
            lines.append(predictor + '概率模型：' + json.dumps(models[predictor], ensure_ascii=False))
    evaluation = {}
    directories = sorted({*BASE.glob('validation_*'), *BASE.glob('test_*')})
    for directory in directories:
        if not directory.is_dir():
            continue
        rows = [read(path) for path in directory.glob('pick*__episode*.json')
                if not path.name.endswith('infrastructure_error.json')]
        rows = [r for r in rows if type(r.get('official_task_success')) is bool and r.get('task') in TASKS]
        if not rows:
            continue
        evaluation[directory.name] = {}
        for task in TASKS:
            task_rows = [r for r in rows if r['task'] == task]
            n, successes = len(task_rows), sum(r['official_task_success'] for r in task_rows)
            evaluation[directory.name][task] = dict(n=n, native_successes=successes,
                                                    success_rate=successes/n if n else None)
        provenance = '（继承R07几何组缓存，原始元数据保留）' if directory.name.startswith('validation_') and directory.name.endswith('_geometry') else ''
        lines.append(directory.name + provenance + '：' + json.dumps(evaluation[directory.name], ensure_ascii=False))
    gate = read(BASE / 'success_acceptance.json')
    if gate:
        lines.append('成功率模块验收：' + json.dumps(gate, ensure_ascii=False))
    try:
        gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.free',
                                       '--format=csv,noheader'], text=True, timeout=10).strip()
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        gpu = 'unavailable'
    lines.extend([f'GPU：{gpu}', 'TRAIN/VAL用于开发与校准；完整共同TEST完成前不宣称超过基线。',
                  '20分钟对话汇报由运行中的主代理发送；本文件是本地进度快照。'])
    directory = BASE / 'progress'
    directory.mkdir(parents=True, exist_ok=True)
    name = now.strftime('%Y%m%dT%H%M%SZ')
    snapshot = dict(created_utc=now.isoformat(), status=status, collections=collections, models=models,
                    evaluation=evaluation, gate=gate, gpu=gpu, data_audit_passed=audit.get('passed'),
                    collection_data_source=str(DATA_BASE.relative_to(ROOT)),
                    live_data_audit_source=str((DATA_BASE / 'data_quality_audit.json').relative_to(ROOT)),
                    geometry_validation_provenance='inherited r07 caches with original record metadata')
    write(directory / (name + '.json'), snapshot)
    body = '\n'.join(line for line in lines if line) + '\n'
    (directory / (name + '.txt')).write_text(body)
    for path in (ROOT / '最新项目状态.txt', ROOT / '二十分钟进度报告.txt', ROOT / '十分钟进度报告.txt'):
        temporary = path.with_suffix('.r08.tmp')
        temporary.write_text(body)
        temporary.replace(path)
    print(body, flush=True)


if __name__ == '__main__':
    main()
