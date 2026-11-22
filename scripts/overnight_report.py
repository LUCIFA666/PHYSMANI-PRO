"""Write timestamped, evidence-based overnight progress/final reports."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]


def read(relative):
    path = ROOT/relative
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (ValueError, OSError):
        return {'status': 'report_temporarily_unavailable'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--final', action='store_true')
    args = parser.parse_args()
    now = datetime.now(timezone.utc)
    local = now.astimezone(ZoneInfo('Asia/Shanghai'))
    evidence = {key: read(path) for key, path in {
        'faster': 'artifacts/framework/bootstrap_status.json',
        'faster_download': 'artifacts/framework/faster_download_status.json',
        'faster_pipeline': 'artifacts/framework/faster_pipeline_status.json',
        'faster_training': 'artifacts/framework/faster_checkpoints/pi05_faster_our_mani_lora/physmani_all16_goal_v1/our_mani_progress.json',
        'framework_smoke': 'artifacts/framework/smoke/report.json',
        'domain_study': 'artifacts/framework/domain_study/status.json',
        'domain_results': 'artifacts/framework/domain_study/report.json',
        'domain_adaptation': 'artifacts/framework/domain_adaptation/report.json',
        'local_collection': 'artifacts/framework/local_policy/collection_status.json',
        'local_training': 'artifacts/framework/local_policy/training_status.json',
        'local_validation': 'artifacts/framework/local_policy/validation_report.json',
        'formal_comparison': 'artifacts/framework/comparison.json',
        'current_unit_checks': 'artifacts/framework/smoke/unit_checks_current.json',
        'faster_best': 'artifacts/framework/faster_best_validation.json',
        'actor_value': 'artifacts/framework/local_policy/servo_value_validation.json',
        'local_reset_audit': 'artifacts/framework/local_policy/validation_reset_audited.json',
        'local_task_success': 'artifacts/framework/local_policy/success_models_smoke_v1/report.json',
        'local_task_success_comparison': 'artifacts/framework/local_policy/success_models_smoke_v1/comparison_with_existing_value.json',
        'paper_baseline_reference': 'config/framework/paper_reference_comparison.json',
        'pipeline_v1_training': 'artifacts/framework/pipeline_v1/training_status.json',
        'pipeline_v1_benchmark': 'artifacts/framework/pipeline_v1/benchmark_status.json',
        'pipeline_v1_dynamics': 'artifacts/framework/pipeline_v1/dynamics_simnorm_v1/report.json',
        'pipeline_v1_success': 'artifacts/framework/pipeline_v1/context_success/report.json',
        'pipeline_v1_test': 'artifacts/framework/pipeline_v1/official_test/report.json',
        'pipeline_v2_value_training': 'artifacts/framework/pipeline_v2/value_training_status.json',
        'pipeline_v2_comparison': 'artifacts/framework/pipeline_v2/comparison_status.json',
        'pipeline_v2_full_training': 'artifacts/framework/pipeline_v2/full_train_status.json',
        'pipeline_v2_full_protocol': 'artifacts/framework/pipeline_v2/full_train_r04/protocol.json',
    }.items()}
    validations = sorted((ROOT/'artifacts/framework/faster_validation').glob('checkpoint_*.json'))
    validations = [p for p in validations if not p.stem.endswith('_partial')]
    evidence['latest_faster_validation'] = read(str(validations[-1].relative_to(ROOT))) if validations else None
    milestones = sorted((ROOT/'artifacts/framework/full_workflow').glob('milestone_*.json'))
    evidence['full_workflow_milestone'] = read(str(milestones[-1].relative_to(ROOT))) if milestones else None
    evidence['full_workflow_results'] = []
    for path in (evidence['full_workflow_milestone'] or {}).get('development_reports', []):
        report = read(path) or {}
        evidence['full_workflow_results'].extend(report.get('results', []))
    evidence['created_utc'] = now.isoformat()
    # Keep progress reports readable; the complete per-window rows remain in
    # the domain study artifact rather than repeated every ten minutes.
    for key in ('domain_results', 'domain_adaptation', 'local_validation', 'local_reset_audit', 'pipeline_v1_test', 'pipeline_v1_dynamics'):
        if isinstance(evidence[key], dict):
            evidence[key] = {k:v for k,v in evidence[key].items() if k not in ('rows', 'results', 'per_window', 'paired_rows', 'history')}
    evidence['created_beijing'] = local.isoformat()
    evidence['final'] = args.final
    try:
        evidence['gpu'] = subprocess.check_output(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.free',
                                                    '--format=csv,noheader'], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        evidence['gpu'] = 'unavailable'
    stage = (evidence['faster'] or {}).get('stage', 'not_started')
    title = '早十点结果汇总' if args.final else '十分钟进度报告'
    session_path = ROOT/'config/framework/current_run.json'
    session = json.loads(session_path.read_text()) if session_path.exists() else {}
    if not args.final and session.get('report_interval_minutes') == 3:
        title = '三分钟进度报告'
    execution_window = ('执行截止：按最新继续指令推进，未指定新的停止时间。'
                        if session.get('deadline_superseded_by_user_continue') else '执行截止：2026-10-17 10:00 北京时间')
    lines = [title, f"北京时间：{local:%Y-%m-%d %H:%M:%S}",
             execution_window,
             'NGFF 原训练保持停止；使用保存的最佳权重。', f'FASTER 当前步骤：{stage}',
             f"逻辑检查通过：{(evidence['framework_smoke'] or {}).get('unit_tests', '未知')} 项",
             f"GPU 当前状态（利用率、已用显存、空闲显存）：{evidence['gpu']}"]
    for label, key in [('FASTER 权重下载', 'faster_download'), ('迁移实验', 'domain_study'), ('迁移定量结果', 'domain_results'),
                       ('FASTER 完整数据训练与验证', 'faster_pipeline'), ('FASTER 优化进度', 'faster_training'),
                       ('驱动/支撑适配与无 NGFF 消融', 'domain_adaptation'),
                       ('局部控制数据采集', 'local_collection'), ('局部策略/Value 训练', 'local_training'),
                       ('局部验证', 'local_validation'), ('正式基线对比', 'formal_comparison')]:
        value = evidence[key]
        lines += [f'\n{label}：', '尚无结果' if value is None else json.dumps(value, ensure_ascii=False, indent=2)]
    lines += ['\n说明：smoke 检查和离线预测误差不计入官方闭环抓取成功率。']
    # Keep the human report short. Full structured evidence remains alongside
    # it, with links to the original per-episode artifacts.
    training, best = evidence['faster_training'] or {}, evidence['faster_best'] or {}
    validation, actor_value = evidence['latest_faster_validation'] or {}, evidence['actor_value'] or {}
    progress, target = training.get('optimizer_updates',0), training.get('target_updates',30000)
    text_lines = [title, f'北京时间：{local:%Y-%m-%d %H:%M:%S}', execution_window,
                  '当前 v1=context + SimNorm + 成功率模块 + FASTER + MPC；不使用 NGFF。原始 NGFF 训练保持停止。',
                  f"FASTER 阶段：{(evidence['faster_pipeline'] or {}).get('stage','未知')}",
                  f'FASTER 已更新 {progress}/{target}（{100*progress/max(1,target):.1f}%），batch={training.get("batch_size","未知")}。',
                  f'最近训练耗时：{training.get("seconds_per_update","未知")} 秒/次；验证和闭环检查另计。']
    paper = evidence['paper_baseline_reference'] or {}
    primary = paper.get('task_success_reference', {}).get('methods', {}).get('PhysMani', {})
    if primary:
        normal = primary['pick_moving_target_on_the_table']
        fast = primary['pick_moving_target_on_the_table_high_speed']
        text_lines[4:4] = [
            f'主基线：PhysMani 论文Table 1报告，普通移动抓取 {normal["mean_percent"]:.1f}±{normal["std_percent"]:.1f}%，高速 {fast["mean_percent"]:.1f}±{fast["std_percent"]:.1f}%。',
            'v1 的正式 TEST 进度见下方；开发案例和局部 Value 指标单独报告。',
        ]
    if validation:
        text_lines.append(f'最新检查点 {validation.get("optimizer_updates")} 次：VAL 平均动作位置误差 {100*validation["mean_position_action_error_m"]:.2f} cm，姿态误差 {validation["mean_orientation_action_error_rad"]:.3f} rad。')
        text_lines.append(f'目标位置三个方向响应均通过：{validation.get("all_three_position_axes_follow_goal_shift")}；流式动作一致性：{validation.get("streaming_partial_final_parity_passed")}。')
    if best:
        text_lines.append(f'按固定 VAL 位置误差选中的最佳检查点：{best.get("optimizer_updates")} 次，误差 {100*best["mean_position_action_error_m"]:.2f} cm。')
    metrics = actor_value.get('metrics',{})
    if metrics:
        text_lines.append(f'学习局部策略的 Value：独立局部 VAL n={metrics["n"]}，Brier={metrics["brier"]:.3f}，AUROC={metrics["auroc"]:.3f}。')
    success_report = evidence['local_task_success'] or {}
    success_metrics = success_report.get('validation', {})
    if success_metrics:
        text_lines.append(f'新局部任务成功率基线（稳定提起且完成任务）：TRAIN={success_report["train_records"]}，VAL={success_metrics["n"]}，Brier={success_metrics["brier"]:.3f}，AUROC={success_metrics["auroc"]:.3f}。')
        text_lines.append('以上是旧的无 context 成功率基线；v1 的 context 成功率模块另行训练，不估计 FASTER 接近阶段成功率。')
    audit = (evidence['local_reset_audit'] or {}).get('paired_servo',{})
    if audit:
        text_lines.append(f'局部策略与 MPC 可匹配重置的开发子集：{audit["n"]} 条；另外 {len(audit["excluded_mismatched_reset_keys"])} 条重置状态偏差较大，保留但不计入配对推断。')
    v1_training=evidence['pipeline_v1_training'] or {}
    v1_benchmark=evidence['pipeline_v1_benchmark'] or {}
    text_lines += [f'\nv1 训练阶段：{v1_training.get("stage","未启动")}',
                   f'v1 对比阶段：{v1_benchmark.get("stage","等待训练") }']
    v1_dynamics=evidence['pipeline_v1_dynamics'] or {}
    forecast=v1_dynamics.get('validation',{}).get('context_model',{}).get('0.5',{})
    if forecast:
        text_lines.append(f'v1 动力学：完成10,000更新，0.5秒端点预测误差 {1000*forecast["macro_endpoint_error_m"]:.1f}mm（两任务VAL，不等同论文全16任务TEST）。')
    for split,folder,total in [('TRAIN','pipeline_v1_context_train_v1_smoke',192),('VAL','pipeline_v1_context_validation_v1_smoke',160)]:
        collection=read(f'artifacts/framework/local_policy/{folder}/report.json') or {}
        text_lines.append(f'v1 局部 {split} 采集：{collection.get("completed",0)}/{total}。')
    v1_success=(evidence['pipeline_v1_success'] or {}).get('validation',{})
    if v1_success:
        text_lines.append(f'v1 context 成功率模块 VAL：Brier={v1_success["brier"]:.3f}，AUROC={v1_success["auroc"]:.3f}。')
    v1_test=evidence['pipeline_v1_test'] or {}
    text_lines.append(f'v1 官方TEST：{v1_test.get("completed",0)}/80。')
    for task,result in v1_test.get('per_task',{}).items():
        text_lines.append(f'  {task}：成功 {result["successes"]}/{result["completed"]}。')
    text_lines += ['以下是旧 NGFF/恒速框架的开发记录，不作为 context+SimNorm v1 的效果：']
    for row in evidence['full_workflow_results']:
        text_lines.append(f'  {row["dynamics"]} / {row["task"]}：成功={row["success"]}，接管={row["handovers"]}，真实神经动作执行={row["faster_actions_applied"]}。')
    text_lines += [f'GPU：{evidence["gpu"]}',
                   f'当前逻辑检查通过：{(evidence["current_unit_checks"] or {}).get("unit_tests","未知")} 项。',
                   'v1 将在 TRAIN/VAL 完成并冻结权重后运行80条官方测试，保留全部失败案例。',
                   '说明：上述训练误差、局部 VAL 和开发闭环检查不能当作论文 benchmark 成功率。']
    if session.get('current_framework')=='pipeline_v2_candidate_interception_without_FASTER':
        text_lines=[title,f'北京时间：{local:%Y-%m-%d %H:%M:%S}',execution_window,
            '当前框架：历史context + SimNorm预测 → 候选截获 → 可达性筛选 → 评分 → 显式关节轨迹/MPC → 跟随抓取。',
            'FASTER按用户要求排除当前全部候选；原NGFF训练保持停止。',
            f'四组对比阶段：{(evidence["pipeline_v2_comparison"] or {}).get("stage","未启动")}',
            f'恒速控制器成功率模块阶段：{(evidence["pipeline_v2_value_training"] or {}).get("stage","未启动")}']
        for split,folder,total in [('TRAIN','pipeline_v2_cv_train_smoke',192),('VAL','pipeline_v2_cv_validation_smoke',160)]:
            report=read(f'artifacts/framework/local_policy/{folder}/report.json') or {}
            text_lines.append(f'恒速局部 {split} 数据：{report.get("completed",0)}/{total}。')
        text_lines.append('四组：恒速/Context预测 × 几何/成功率排序；成功率头当前均不输入context。')
        for label,folder in [('context+成功率','smoke_r02_context_value'),('恒速+几何','smoke_r01_cv_geometry'),('context+几何','smoke_r00_context_geometry')]:
            report=read(f'artifacts/framework/pipeline_v2/{folder}/report.json') or {}
            k=sum(v.get('successes',0) for v in report.get('per_task',{}).values())
            text_lines.append(f'初始TRAIN smoke {label}：成功{k}/{report.get("completed",0)}。')
        text_lines.append('上述三种配置复用相同两条TRAIN案例，不能算作六条独立测试，也不是官方成功率。')
        for split,total in [('train',8),('validation',40),('test',80)]:
            for predictor in ('cv','context'):
                for ranking in ('geometry','value'):
                    report=read(f'artifacts/framework/pipeline_v2/comparison_r03_{split}_{predictor}_{ranking}/report.json') or {}
                    if report:
                        k=sum(v.get('successes',0) for v in report.get('per_task',{}).values())
                        text_lines.append(f'{split} {predictor}+{ranking}: 完成{report.get("completed",0)}/{total}，成功{k}。')
        text_lines += ['当前Value预测接管后的局部任务成功；接近阶段使用名义可达性硬筛，尚未声明全链路概率校准。',
            '显式轨迹已记录实际伺服误差；局部控制的关节加速度硬约束仍需补齐。',
            f'逻辑检查：{(evidence["current_unit_checks"] or {}).get("unit_tests","未知")}项。',
            f'GPU：{evidence["gpu"]}；当前主要耗时在CPU仿真与评测。',
            '正式对比：按固定TRAIN检查、VAL评测、冻结四组TEST的流程推进；论文基线使用PhysMani Table1作者报告值。',
            '思路总结：思路总结_无FASTER截获.txt']
    if session.get('active_experiment') == 'pipeline_v2_full_task_TRAIN_r04':
        full = read('artifacts/framework/pipeline_v2/full_train_status.json') or {}
        dyn = read('artifacts/framework/pipeline_v2/full_train_r04/dynamics/status.json') or {}
        text_lines = [title, f'北京时间：{local:%Y-%m-%d %H:%M:%S}', execution_window,
            '当前：r04 两个目标任务完整TRAIN训练；FASTER排除，原NGFF训练停止。',
            '训练范围：普通100+高速100条TRAIN，共200条；其余14个任务未纳入。',
            '本轮要求200条均参与参数更新；VAL用于模型选择与成功率校准，不进入梯度训练。',
            f'阶段：{full.get("stage", "准备")}',
            f'动力学更新：{dyn.get("update", dyn.get("metadata", {}).get("completed_updates", 0))}/10000。']
        for predictor in ('cv', 'context'):
            for split, shards, total in [('train', 3, 800), ('validation', 1, 160)]:
                records = [read(f'artifacts/framework/local_policy/full_train_r04_{predictor}_{split}_shard{i}/report.json') or {} for i in range(shards)]
                n = sum(r.get('completed', 0) for r in records)
                valid = sum(r.get('valid_initializations', 0) for r in records)
                groups = sum(r.get('valid_source_episodes', 0) for r in records)
                text_lines.append(f'{predictor} 局部{split}: {n}/{total}次尝试，有效{valid}，覆盖{groups}条源episode。')
            head = read(f'artifacts/framework/pipeline_v2/full_train_r04/{predictor}_success/report.json') or {}
            if head:
                metadata = head.get('metadata', {})
                text_lines.append(f'{predictor} 成功率头已训练：fit {head.get("partitions", {}).get("fit", {}).get("episodes", 0)}条TRAIN源episode，完成{metadata.get("completed_updates")}次更新，选中第{metadata.get("selected_update")}次。')
                for task, result in head.get('per_task', {}).items():
                    text_lines.append(f'  {task} 开发VAL：n={result.get("n")}，AUROC={result.get("auroc")}，Brier={result.get("brier")}。')
                evidence[f'full_train_r04_{predictor}_success'] = {k:head.get(k) for k in ('metadata', 'partitions', 'validation', 'per_task', 'checkpoint_sha256')}
        if full.get('error'):
            text_lines.append(f'执行错误：{full["error"]}')
        text_lines += ['r03已停止并保留为初步实验，不作为完整训练后的最终对比。',
            '成功率头VAL用于选模与校准，以上指标是开发诊断；AUROC不等于抓取成功率。',
            '训练与校准完成后冻结r04，再运行四组各80条官方TEST；不会拼接旧r03结果。',
            'TEST已在旧版本看过；重测使用固定公开benchmark，不宣称全新未见测试集。', f'GPU：{evidence["gpu"]}']
        for split, total in [('validation', 40), ('test', 80)]:
            for predictor in ('cv', 'context'):
                for ranking in ('geometry', 'value'):
                    report = read(f'artifacts/framework/pipeline_v2/comparison_r04_{split}_{predictor}_{ranking}/report.json') or {}
                    if report:
                        successes = sum(v.get('successes', 0) for v in report.get('per_task', {}).values())
                        text_lines.append(f'r04 {split} {predictor}+{ranking}: {report.get("completed", 0)}/{total}，成功{successes}。')
    text = '\n'.join(text_lines)+'\n'
    directory = ROOT/'artifacts/framework/overnight/reports'
    directory.mkdir(parents=True, exist_ok=True)
    basename = local.strftime('%Y%m%d-%H%M%S')+('-final' if args.final else '')
    (directory/f'{basename}.json').write_text(json.dumps(evidence, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    (directory/f'{basename}.txt').write_text(text, encoding='utf-8')
    (ROOT/f'{title}.txt').write_text(text, encoding='utf-8')
    (ROOT/'最新项目状态.txt').write_text(text, encoding='utf-8')
    print(f'{title} 已保存：{directory/basename}')


if __name__ == '__main__':
    main()
