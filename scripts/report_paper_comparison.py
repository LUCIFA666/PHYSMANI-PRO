"""Report paper references alongside development evidence without mixing them."""
from datetime import datetime, timezone
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    reference = json.loads((ROOT/'config/framework/paper_reference_comparison.json').read_text())
    milestones = sorted((ROOT/'artifacts/framework/full_workflow').glob('milestone_*.json'))
    milestone = json.loads(milestones[-1].read_text()) if milestones else {}
    rows = []
    for path in milestone.get('development_reports', []):
        report = json.loads((ROOT/path).read_text())
        for row in report.get('results', []):
            rows.append({key: row.get(key) for key in (
                'task', 'source_episode', 'split', 'kind', 'dynamics', 'success', 'handovers',
                'faster_actions_applied', 'maximum_control_lag_seconds')})
    record = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'kind': 'paper_reported_baseline_reference_with_separate_development_status',
        'reference': reference,
        'ours_formal_test': {'status': 'not_run', 'success_rate_percent': None},
        'ours_latest_development': {
            'milestone_report': str(milestones[-1].relative_to(ROOT)) if milestones else None,
            'checkpoint': milestone.get('checkpoint'), 'cases': rows,
            'successes': sum(bool(r['success']) for r in rows),
            'handovers': sum(r['handovers'] or 0 for r in rows),
            'case_count': len(rows),
            'independent_trial_count_inferred': False,
            'limitation': 'predictor variants repeat development resets; not original TEST or independent repetitions',
        },
        'direct_paired_comparison_valid': False, 'formal_test_used_by_this_report': False,
        'new_baseline_inference_launched': False,
    }
    out = ROOT/'artifacts/framework/paper_reference'
    out.mkdir(parents=True, exist_ok=True)
    (out/'comparison_latest.json').write_text(json.dumps(record, ensure_ascii=False, indent=2)+'\n')
    task_a, task_b = 'pick_moving_target_on_the_table', 'pick_moving_target_on_the_table_high_speed'
    methods = reference['task_success_reference']['methods']
    lines = ['PhysMani 论文宣称结果与 our-mani 当前状态',
             f"来源：{reference['paper']}，Table 1，PDF 第10页。",
             '论文统计口径：每任务40条TEST，最后5个检查点中最好的3个模型的均值±标准差；不是置信区间。',
             '', '方法 | 普通速度移动抓取 | 高速移动抓取']
    for method in ('PhysMani', '3DFA'):
        a, b = methods[method][task_a], methods[method][task_b]
        lines.append(f'{method}（论文报告） | {a["mean_percent"]:.1f}±{a["std_percent"]:.1f}% | {b["mean_percent"]:.1f}±{b["std_percent"]:.1f}%')
    lines += ['our-mani 完整框架（同协议TEST） | 尚未评测 | 尚未评测', '',
              f'最近开发检查：{len(rows)}个案例，成功{record["ours_latest_development"]["successes"]}，接管{record["ours_latest_development"]["handovers"]}。',
              '这些案例使用开发重置，且不同动力学版本重复相同场景；不能与论文TEST直接计算提升百分点或显著性。',
              '成功率预测器的AUROC、局部接管后的成功率和离线筛选子集成功率均不作为完整方法成功率。',
              'context+SimNorm 方案尚未训练/闭环评测，当前不能填入其成功率。', '',
              '阶段判断：当前接近/接管仍未跑通，尚无证据达到论文完整方法表现。',
              '论文中高速任务的3DFA均值高于PhysMani；高速抓取应同时参照18.3%和21.7%，而不能假设引入动力学一定提高该任务成功率。', '',
              '动力学参考（Table 2，全部16任务TEST）：50ms/250ms/500ms目标中心误差为8/39/74mm。',
              '现有our-mani动力学结果来自两任务VAL、预测时长和聚合口径不同，尚不计算直接误差差值。', '',
              '用户已允许使用论文报告作为基线参照，本阶段无需为此新增本机PhysMani推理任务。']
    (ROOT/'与PhysMani论文对比.txt').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps({'report': str((out/'comparison_latest.json').relative_to(ROOT)),
                      'development_cases': len(rows), 'formal_ours_test': 'not_run',
                      'baseline_source': 'paper_reported'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
