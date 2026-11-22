"""Read the independent 48-case TRAIN diagnostic batch, preserving unknowns."""
from collections import Counter
from datetime import datetime, timezone
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = 'runtime/framework/TRAIN_zero_success_diagnosis_v1107'
PLAN = 'artifacts/framework/multitask_r10/TRAIN_zero_success_diagnosis_v1107_setup/plan.json'
OUTPUT = BASE + '/diagnostic_summary_v1108.json'
TXT = 'TRAIN_零成功任务诊断结论_20261023.txt'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_pointer(pointer):
    path = ROOT / pointer['path']
    if sha(path) != pointer['sha256']:
        raise ValueError('Saved diagnostic evidence changed: ' + pointer['path'])
    return json.loads(path.read_text())


def fresh(path, text):
    with path.open('x', encoding='utf-8') as stream:
        stream.write(text)


def phase(case):
    intentions = case['actor_metrics'].get('geometry_intents', {})
    return intentions.get('current_phase', intentions.get('phase', 'unknown'))


def follow(case):
    curve = case['actor_metrics'].get('geometry_intents', {}).get('public_trained_follow_curve', {})
    rows = curve.get('events', [])
    return dict(recorded_events=len(rows), last_record=rows[-1] if rows else None)


def summarize(write=False):
    status = json.loads((ROOT / BASE / 'status.json').read_text())
    terminal = json.loads((ROOT / BASE / 'terminal.json').read_text())
    plan = json.loads((ROOT / PLAN).read_text())
    if (status['stage'] != 'complete' or terminal.get('complete') is not True
            or len(status['completed']) != 48 or status['failed'] or len(plan['jobs']) != 48):
        raise ValueError('Actual complete fixed 48-case diagnostic batch required')
    jobs = {r['job_id']: r for r in plan['jobs']}
    if len(jobs) != 48 or {r['job_id'] for r in status['completed']} != set(jobs):
        raise ValueError('Exact original fixed diagnostic grid required')
    cases = []
    for done in status['completed']:
        row = jobs[done['job_id']]
        case, sidecar = read_pointer(done['case']), read_pointer(done['sidecar'])
        events = sidecar['execution']['diagnostic_events']
        if (not sidecar['complete'] or not events['restored'] or case['split'] != 'train'
                or case['task'] != row['task'] or case['method'] != row['method']
                or case['source_raw_episode_id'] != row['raw'] or not case['cleanup_completed']):
            raise ValueError('Incomplete original TRAIN evidence')
        item = dict(job_id=row['job_id'], kind=row['kind'], task=row['task'], method=row['method'],
            raw=row['raw'], native_success=case['official_task_success'],
            native_terminal_failure=case['native_terminal_failure'],
            counted_ticks=case['counted_clock']['ticks'], final_phase=phase(case),
            case=done['case'], sidecar=done['sidecar'], restored=True)
        item['policy_failure'] = case.get('policy_failure')
        item['terminal_class'] = ('success' if case['official_task_success'] else
            'native_terminal_failure' if case['native_terminal_failure'] else
            'explicit_policy_failure' if case.get('policy_failure') is not None else
            'budget_exhausted' if case['counted_clock']['ticks'] == 200 else 'unknown_nonterminal_stop')
        actions = case['actor_metrics'].get('physmani_geometry_executor', {}).get('actions', [])
        paths = [a.get('original_executor_receipt', {}) for a in actions]
        item['planning'] = dict(pose_actions=len(actions),
            path_found_true=sum(p.get('path_found') is True for p in paths),
            path_found_false=sum(p.get('path_found') is False for p in paths),
            path_found_unknown=sum(type(p.get('path_found')) is not bool for p in paths),
            recorded_errors=dict(Counter(str(p['error']) for p in paths if p.get('error'))))
        item['observer_quality'] = [{k: r[k] for k in
            ('dropped_events', 'error_count', 'copy_truncations', 'errors', 'observer_errors', 'dropped_calls',
             'dropped_errors', 'non_owner_thread_calls_unobserved') if k in r}
            for r in events.get('observer_receipts', [])]
        item['diagnostic_quality_complete'] = bool(item['observer_quality']) and all(
            not q.get(k, 0) for q in item['observer_quality'] for k in
            ('dropped_events', 'error_count', 'copy_truncations', 'errors', 'observer_errors', 'dropped_calls', 'dropped_errors'))
        old = ROOT / 'artifacts/framework/multitask_r10/author_outcome_qualification_v745/development_demo_only' / row['task'] / row['raw'] / row['method'] / 'case.json'
        if old.exists():
            previous = json.loads(old.read_text())
            item['previous_uninstrumented_TRAIN'] = dict(path=str(old.relative_to(ROOT)), sha256=sha(old),
                native_success=previous['official_task_success'], counted_ticks=previous['counted_clock']['ticks'],
                final_phase=phase(previous), same_native_outcome=previous['official_task_success'] == case['official_task_success'],
                same_final_phase=phase(previous) == phase(case),
                physical_equivalence_claimed=False)
        summary = events['summary']
        if row['kind'] == 'ring':
            rejected = [r for r in summary['admissions'] if r.get('accepted') is False]
            lift = [r for r in rejected if r.get('policy_context', {}).get('selected', {}).get('phase') == 'lift']
            static_map = case.get('recorded_initial_conditions', {}).get('public_initial_conditions', {}).get('static_map', [])
            static_sha = hashlib.sha256(json.dumps(static_map, sort_keys=True).encode()).hexdigest()
            map_binding = case.get('alignment_receipt', {}).get('map', {}).get('map_sha256')
            registered_roles = set()
            for admission in lift:
                for pair in admission.get('registered_pairs', []):
                    objects = [pair.get('receiver', {})] + pair.get('arguments', {}).get('args', [])
                    for obj in objects:
                        if isinstance(obj, dict):
                            registered_roles.update(r['role'] for r in obj.get('registered_roles', []) if r.get('role', '').startswith('static:'))
            mapping = {}
            if static_sha == map_binding:
                for role in sorted(registered_roles):
                    index = int(role.split(':')[1])
                    if index < len(static_map):
                        mapping[role] = static_map[index]
            item['diagnosis'] = dict(admission_calls=summary['admission_call_count'],
                branch_counts=summary['branch_counts'], rejected_calls=len(rejected), lift_rejected_calls=len(lift),
                lift_branch_counts=dict(Counter(r['classification'] for r in lift)),
                lift_fraction_zero=sum(r.get('original_proof', {}).get('failed_fraction') == 0 for r in lift),
                first_lift_rejection=lift[0] if lift else None,
                all_rejected_call_ids=[r['call_id'] for r in rejected],
                static_map_snapshot_sha256=static_sha, static_map_matches_alignment=static_sha == map_binding,
                rejected_static_roles_from_saved_snapshot=mapping,
                physical_collision_or_false_positive_established=False)
        elif row['kind'] == 'remove':
            if len(summary) != 1:
                raise ValueError('One remove observer required')
            s = summary[0]
            calls = events['observer_receipts'][0]['calls']
            stage_done = [c for c in calls if c.get('source', {}).get('method') == '_done'
                and c.get('source', {}).get('class_name') != 'SkillGeometryIntents']
            item['diagnosis'] = dict(acquire_done_blocker_counts=s['acquire_done_blocker_counts'],
                public_capture_blocker_counts=s['public_capture_blocker_counts'],
                last_acquire_done=s['acquire_done'][-1] if s['acquire_done'] else None,
                last_public_capture=s['public_capture'][-1] if s['public_capture'] else None,
                readiness_false_calls=sum(r['result'] is False for r in s['readiness']),
                last_stage_done=stage_done[-1] if stage_done else None,
                native_grasp_identity_not_inferred=True,
                last_called_predicate_is_not_necessarily_terminal_state=True)
            if not item['diagnostic_quality_complete']:
                for name in ('last_acquire_done', 'last_public_capture', 'last_stage_done'):
                    item['diagnosis']['last_recorded_only_' + name] = item['diagnosis'][name]
                    item['diagnosis'][name] = None
                item['diagnosis']['actual_last_condition_unknown_due_to_incomplete_telemetry'] = True
        else:
            sampling = case.get('native_outcome_sampling', {})
            completed_actions = sampling.get('completed_actions', [])
            item['diagnosis'] = dict(summary, saved_follow_curve=follow(case),
                actual_action_end_record=completed_actions[-1] if completed_actions else None,
                recorded_intermediate_failure_samples=sampling.get('intermediate_failure_samples'))
        cases.append(item)
    by_kind = {}
    for kind in ('ring', 'remove', 'buzz'):
        selected = [c for c in cases if c['kind'] == kind]
        d = dict(cases=len(selected), native_success=sum(c['native_success'] for c in selected),
            native_terminal_failure=sum(c['native_terminal_failure'] for c in selected),
            final_phase_counts=dict(Counter(c['final_phase'] for c in selected)),
            prior_native_outcome_changes=sum(c.get('previous_uninstrumented_TRAIN', {}).get('same_native_outcome') is False for c in selected))
        d['terminal_class_counts'] = dict(Counter(c['terminal_class'] for c in selected))
        d['prior_cases_compared'] = sum('previous_uninstrumented_TRAIN' in c for c in selected)
        d['prior_evidence_missing'] = len(selected) - d['prior_cases_compared']
        d['prior_final_phase_changes'] = sum(c.get('previous_uninstrumented_TRAIN', {}).get('same_final_phase') is False for c in selected)
        d['diagnostic_quality_complete_cases'] = sum(c['diagnostic_quality_complete'] for c in selected)
        d['path_found_false_calls'] = sum(c['planning']['path_found_false'] for c in selected)
        d['cases_with_path_found_false'] = sum(c['planning']['path_found_false'] > 0 for c in selected)
        if kind == 'ring':
            d['cases_with_lift_rejections'] = sum(c['diagnosis']['lift_rejected_calls'] > 0 for c in selected)
            d['lift_rejection_call_counts'] = dict(sum((Counter(c['diagnosis']['lift_branch_counts']) for c in selected), Counter()))
        elif kind == 'remove':
            d['last_acquire_blocker_case_counts'] = dict(Counter(
                ((c['diagnosis']['last_acquire_done'] or {}).get('first_supported_blocker', 'no_acquire_done_call')
                 if c['diagnostic_quality_complete'] else 'unknown_incomplete_telemetry') for c in selected))
            d['last_capture_blocker_case_counts'] = dict(Counter(
                ((c['diagnosis']['last_public_capture'] or {}).get('first_supported_blocker', 'no_public_capture_call')
                 if c['diagnostic_quality_complete'] else 'unknown_incomplete_telemetry') for c in selected))
        else:
            d['cases_with_observed_failure_condition'] = sum(c['diagnosis']['first_observed_failure_condition'] is not None for c in selected)
        by_kind[kind] = d
    result = dict(schema='r10_fixed_TRAIN_zero_success_diagnostic_summary_v1108',
        created_utc=datetime.now(timezone.utc).isoformat(), complete=True, split='train',
        case_count=48, parent_sources=24, by_kind=by_kind, cases=cases,
        plan=dict(path=PLAN, sha256=sha(ROOT / PLAN)),
        status=dict(path=BASE+'/status.json', sha256=sha(ROOT / BASE / 'status.json')),
        terminal=dict(path=BASE+'/terminal.json', sha256=sha(ROOT / BASE / 'terminal.json')),
        instrumented_diagnostic_only=True, formal_score_contribution=False,
        model_or_policy_changed=False, TEST_used_for_fitting=False,
        physical_equivalence_claimed=False, repeated_calls_are_not_independent_trials=True,
        unknown_values_not_filled=True,
        caveats=['ring static:N is mapped only through the hash-bound saved public static_map; this is not native physical collision proof.',
            'A successful close request does not establish native grasp of the intended cup.',
            'The latest evaluated capture/done call may precede a close; it is not automatically the terminal state.',
            'Intermediate native failure does not replace action-end outcome sampling.',
            'Diagnostic reruns do not replace uninstrumented formal TRAIN/VAL/TEST evidence.'])
    if write:
        fresh(ROOT / OUTPUT, json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
        lines = ['TRAIN零成功任务诊断结论（独立旁路观测，不计入正式VAL/TEST）',
            '完成48/48：6任务版本×固定原TRAIN前4来源×CV/Context；24个父来源。',
            '模型、策略、原证据及封存TEST保持不变。',
            '重复调用次数不是独立案例数；缺失条件保持未知。', '']
        for kind, label in (('ring', '套环'), ('remove', '取杯'), ('buzz', '穿环')):
            d = by_kind[kind]
            lines.extend([f"{label}：{d['cases']}例，原生成功{d['native_success']}例，原生终止失败{d['native_terminal_failure']}例。",
                '最终阶段分布：'+json.dumps(d['final_phase_counts'], ensure_ascii=False),
                f"未生成路径的调用{d['path_found_false_calls']}次，涉及{d['cases_with_path_found_false']}例。",
                f"诊断记录完整{d['diagnostic_quality_complete_cases']}/{d['cases']}例；旧资格证据可对照{d['prior_cases_compared']}例，缺失{d['prior_evidence_missing']}例。",
                f"相对旧资格运行，原生结果变化{d['prior_native_outcome_changes']}例，最终阶段变化{d['prior_final_phase_changes']}例。"])
            if kind == 'ring':
                lines.append('抬升拒绝分支（原调用次数）：'+json.dumps(d['lift_rejection_call_counts'], ensure_ascii=False))
                roles = Counter()
                for c in cases:
                    if c['kind'] == kind:
                        roles.update(x.get('name', 'unknown') for x in c['diagnosis']['rejected_static_roles_from_saved_snapshot'].values())
                lines.append('拒绝涉及的静态公共对象（每案例每角色计一次）：'+json.dumps(roles, ensure_ascii=False))
                lines.append('当前状态若在fraction0已被公共几何判交，同一状态下候选目标都先被拒，后续离开起点的样本无法进入检查。此为准入逻辑机制，不等于已证明真实物理碰撞或误检。')
            elif kind == 'remove':
                lines.append('最后抓取完成检查（按案例，未调用与记录不全分别保留）：'+json.dumps(d['last_acquire_blocker_case_counts'], ensure_ascii=False))
                lines.append('最后公共抓取确认检查（按案例）：'+json.dumps(d['last_capture_blocker_case_counts'], ensure_ascii=False))
                lines.append('闭爪请求不证明抓住指定杯；未进入抓取阶段的案例不归因为抓取确认失败。')
                lines.append('最后一次实际检查可能早于闭爪；若后续ready不成立而不再调用，不能把早先叶条件冒充终止时状态。')
            else:
                lines.append(f"原失败条件被实际观测到的案例{d['cases_with_observed_failure_condition']}例。")
                lines.append('原失败条件为wand被middle_sensor检测；中间tick触发与动作结束当前判定分别保留，不能相互替代。')
            lines.append('')
        lines.extend(['不同运行可能有不同建路/执行轨迹；本轮不宣称与无观测运行严格物理等价，也不替换旧资格或正式评测成绩。',
            '详细逐案例证据、调用叶子和未知值：'+OUTPUT])
        fresh(ROOT / TXT, '\n'.join(lines)+'\n')
    print(json.dumps(dict(complete=True, by_kind=by_kind, written=write), ensure_ascii=False))
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--write', action='store_true')
    summarize(p.parse_args().write)
