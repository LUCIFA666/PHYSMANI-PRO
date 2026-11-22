#!/usr/bin/env python3
"""Merge the authorized release-strategy comparison into the executor report."""
import json
from pathlib import Path
from summarize_executor_continuation_20261113_v1 import ROOT,read,sha,pointer,cohort

OUT=ROOT/'artifacts/framework/executor_release_order_proposal_20261113_v19'

def main():
    plan=read(OUT/'predeclared_plan.json');decision=read(OUT/'user_control_strategy_decision.json')
    assert decision['authorized']
    previous=ROOT/'artifacts/framework/executor_diagnostics_20261113_v2/repair_report.json'
    report=read(previous);group=cohort('gripper_release_control_strategy',
        'artifacts/framework/executor_release_order_proposal_20261113_v19/native_validation/*/worker/result.json',
        'TRAIN_adapted_source_oracle_NN0_diagnostic_not_model_rate')
    if len(group['rows'])!=len(plan['cases']):raise RuntimeError('Incomplete paired control strategy cohort')
    by_case={Path(row['result']['path']).parents[1].name:row for row in group['rows']}
    pairs=[];strategy={};qualified=True
    for name in plan['cases']:
        path=OUT/'native_validation'/Path(name).stem;result=read(path/'worker/result.json');life=read(path/'actual_exit.json')
        if not(life['complete']and life['actual_wait_completed']and not life['unreaped_descendants']
               and result['native_closed']and result['source_unchanged']):raise RuntimeError('Incomplete Native lifecycle')
        assert result['NN_forward_calls']==0 and result['optimizer_steps']==0 and not result['formal_VAL_TEST']
        mode=result['gripper_release_order'];events=result['release_order_events']
        if mode=='open_fingers_then_native_release':
            if not all(e['physical_opening_completed']and min(e['physical_finger_open_amounts'])>=.95 for e in events):
                raise RuntimeError('Detached before the measured fingers opened')
        row=by_case[path.name];row.update(control_strategy=mode,profile=result['profile'],
            ever_held=result['ever_held'],rejected_pose_chunks=result['rejected_pose_chunks'],
            release_events=events,model_policy_success_claimed=False)
        audit=result['issued_reference_audit'];safe=(audit['all_reference_differences_within_limits']
            and audit['all_position_commands_within_native_bounds']
            and result['hardware_contact_audit']['prohibited_robot_table_contact_frames']==0)
        qualified=qualified and safe
        record=strategy.setdefault(mode,dict(successes=0,total=0,all_mechanical_checks_passed=True,seeds=[]))
        record['total']+=1;record['successes']+=int(result['original_native_success'])
        record['all_mechanical_checks_passed']=record['all_mechanical_checks_passed']and safe
        record['seeds'].append(result['seed'])
    for a,b in zip(plan['cases'][::2],plan['cases'][1::2]):
        ar=by_case[Path(a).stem];br=by_case[Path(b).stem]
        assert ar['seed']==br['seed']and ar['task']==br['task']
        pairs.append(dict(task=ar['task'],seed=ar['seed'],native_order=ar,open_then_detach=br))
    for row in report['frozen_baseline_audit']:
        if sha(row['path'])!=row['expected']:raise RuntimeError('Frozen model or baseline changed')
    tests=read(OUT/'unit_test_report.json');assert tests['passed']
    report.update(state='executor_control_strategy_evaluated_repairs_in_progress',previous_report=pointer(previous),
        cohorts=report['cohorts']+[group],new_native_case_count=report['new_native_case_count']+len(group['rows']),
        Native_cases_this_control_strategy_cycle=len(group['rows']),remaining_user_decision=None,
        user_control_strategy_decision=pointer(OUT/'user_control_strategy_decision.json'),
        reporting_policy='unified_executor_report_with_explicit_control_strategy',
        control_strategy_comparison=dict(by_strategy=strategy,paired_rows=pairs,all_mechanical_checks_passed=qualified),
        protocol_and_executor_unit_tests=pointer(OUT/'unit_test_report.json'),release_order_proposal_Native_started=True,
        prepared_release_order_cases=pointer(OUT/'predeclared_plan.json'),active_service=None,
        model_weights_changed=False,optimizer_steps=0,new_formal_VAL_TEST_started=False,
        all_tasks_fixed=False,default_runtime_changed=False)
    report['new_predeclared_plans'].append(pointer(OUT/'predeclared_plan.json'))
    report['new_process_exit_receipts'] += [row['process_lifecycle']for row in group['rows']]
    (OUT/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    lines=['执行器统一报告：释放顺序控制策略（TRAIN 专家目标诊断）','',
        '按用户要求统一报告，逐行注明控制策略；任务场景、时间步、动力学、成功判定和模型权重保留。',
        '全部案例 NN=0；这些结果不是模型成功率，也不是新的正式 VAL／TEST。','']
    for mode,value in strategy.items():lines.append(f"{mode}: {value['successes']}/{value['total']}；mechanical_checks={value['all_mechanical_checks_passed']}")
    lines+=['','种子 | 先解除再张开 | 先张开再解除']
    for pair in pairs:
        a=pair['native_order'];b=pair['open_then_detach']
        lines.append(f"{pair['seed']} | success={a['success']} {a['physical_seconds']:.2f}s {a['stop_reason']} | success={b['success']} {b['physical_seconds']:.2f}s {b['stop_reason']}")
    lines+=['',f"单元测试 {tests['tests_run']} 项通过。",'旧方块完整 TEST 仍为普通 39/40、高速 27/40；当前默认运行入口尚未切换。']
    (OUT/'统一结果.txt').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(report=str(OUT/'report.json'),by_strategy=strategy,
        all_mechanical_checks_passed=qualified,unit_tests=tests['tests_run']),ensure_ascii=False))

if __name__=='__main__':main()
