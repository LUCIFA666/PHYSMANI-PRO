"""Audit completed r04 development VAL; keep it separate from official TEST."""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'scripts')]
from our_mani.benchmark_protocol import verify_lock
from report_interception_v2_full import metrics

BASE = ROOT/'artifacts/framework/pipeline_v2'
TASKS = ('pick_moving_target_on_the_table', 'pick_moving_target_on_the_table_high_speed')
VARIANTS = ('cv_geometry', 'cv_value', 'context_geometry', 'context_value')


def main():
    lock = json.loads((BASE/'comparison_r04_frozen_inputs.json').read_text())
    verify_lock(ROOT, lock)
    manifest = json.loads((ROOT/'artifacts/framework/protocol/validation_manifest.json').read_text())['episodes']
    expected = {(r['task'], r['raw_episode_id']) for r in manifest if r['task'] in TASKS}
    variants = {}
    for name in VARIANTS:
        directory = BASE/f'comparison_r04_validation_{name}'
        report = json.loads((directory/'report.json').read_text())
        if report['complete'] is not True or report['completed'] != 40:
            raise ValueError(f'incomplete VAL: {name}')
        rows = []
        for path in sorted(directory.glob('pick*__episode*.json')):
            if path.name.endswith('.infrastructure_error.json'):
                if not path.with_name(path.name.replace('.infrastructure_error.json', '.json')).exists():
                    raise ValueError('unresolved infrastructure failure')
                continue
            row = json.loads(path.read_text())
            if row['split'] != 'validation' or row['formal_test_used'] is not False or type(row['official_task_success']) is not bool:
                raise ValueError('incompatible VAL outcome')
            if f"{row['predictor']}_{row['ranking']}" != name:
                raise ValueError('variant mismatch')
            rows.append(row)
        if len(rows) != 40 or {(r['task'], r['raw_episode_id']) for r in rows} != expected:
            raise ValueError('VAL source coverage mismatch')
        if len({r['inputs_sha256'] for r in rows}) != 1:
            raise ValueError('mixed frozen VAL inputs')
        variants[name] = {task: metrics([r for r in rows if r['task'] == task]) for task in TASKS}
    heads = {}
    for name in ('cv', 'context'):
        r = json.loads((BASE/f'full_train_r04/{name}_success/report.json').read_text())
        heads[name] = {k:r[k] for k in ('partitions', 'validation', 'per_task', 'temperature', 'checkpoint_sha256')}
    result = {'created_utc': datetime.now(timezone.utc).isoformat(), 'complete': True,
        'kind': 'development_VAL_kinematic_home_resets_not_official_TEST', 'closed_loop_cases': 160,
        'frozen_inputs_verified': True, 'head_training': heads, 'variants': variants,
        'limits': ['VAL is used for model selection and probability calibration; diagnostics are not independent TEST',
            '20 development episodes per task per variant; do not combine with official TEST success rates',
            'local probability is conditional on handover; global approach probability is not calibrated']}
    path = BASE/'full_train_r04/closed_loop_validation_report.json'
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    lines = ['r04 对应任务完整TRAIN训练与开发验证', f'UTC：{result["created_utc"]}',
        '动力学：全部200条TRAIN源episode，2860窗口，10000次更新。',
        '两个成功率头均覆盖全部200条TRAIN源episode；各2000次更新，VAL用于选模和校准。']
    for name, head in heads.items():
        lines.append(f'{name}：有效TRAIN样本{head["partitions"]["fit"]["records"]}。')
        for task, label in zip(TASKS, ('普通', '高速')):
            m = head['per_task'][task]
            lines.append(f'  成功率头{label}开发VAL：n={m["n"]}，AUROC={m["auroc"]:.3f}，Brier={m["brier"]:.3f}。')
    lines.append('四组闭环开发VAL（每组普通20+高速20，不是论文官方TEST）：')
    for name, tasks in variants.items():
        for task, label in zip(TASKS, ('普通', '高速')):
            m = tasks[task]
            lines.append(f'{name} / {label}：成功{m["successes"]}/{m["episodes"]}，接管{m["episodes_with_handover"]}，抓住{m["episodes_with_capture"]}；失败分类{m["failure_stages"]}。')
    lines += ['上述开发VAL不作为独立测试指标；AUROC不是抓取成功率。',
        '四组官方TEST按既定冻结模型继续执行，单独汇总，不拼接VAL或旧r03结果。']
    (ROOT/'训练与验证结果_r04.txt').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'complete': True, 'cases': 160, 'variants': {n:{t:m['successes'] for t,m in tasks.items()} for n,tasks in variants.items()}}, ensure_ascii=False))


if __name__ == '__main__':
    main()
