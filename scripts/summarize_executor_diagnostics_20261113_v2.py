#!/usr/bin/env python3
"""Index every predeclared diagnostic; never pool prototype trials into a policy rate."""
import json
from pathlib import Path
from summarize_executor_continuation_20261113_v1 import ROOT,read,sha,pointer,cohort

OUT=ROOT/'artifacts/framework/executor_diagnostics_20261113_v2'
GROUPS=[
    ('capture_precision_factorial','executor_capture_precision_20261113_v2','factorial','predeclared_plan.json'),
    ('contact_frame_and_new_seed_checks','executor_contact_frame_20261113_v3','native_validation','predeclared_plan.json'),
    ('live_reference_unload','executor_reference_unload_20261113_v4','native_validation','predeclared_plan.json'),
    ('held_placement_alignment','executor_placement_alignment_20261113_v5','native_validation','predeclared_plan.json'),
    ('phase_ACK_and_goal_continuity','executor_goal_continuity_20261113_v6','native_validation','predeclared_plan.json'),
    ('held_only_continuity_and_joint_avoidance','executor_held_target_20261113_v7','native_validation','predeclared_plan.json'),
    ('Cartesian_feedback_axis_ablation','executor_held_target_20261113_v7','cartesian_feedback','predeclared_cartesian_feedback.json'),
    ('protocol_bridge','executor_protocol_bridge_20261113_v8','native_validation','predeclared_plan.json'),
    ('protocol_bridge_buzz_fresh_seeds','executor_protocol_bridge_20261113_v8','fresh_seeds','predeclared_fresh_seeds.json'),
    ('protocol_bridge_cup_peg_fresh_seeds','executor_protocol_bridge_20261113_v8','remaining_task_seeds','predeclared_remaining_task_seeds.json'),
    ('control_lead_ablation','executor_control_lead_20261113_v9','native_validation','predeclared_plan.json'),
    ('TRAIN_robot_posture_bank','executor_robot_posture_20261113_v10','native_validation','predeclared_plan.json'),
    ('explicit_release_contract','executor_release_contract_20261113_v11','native_validation','predeclared_plan.json'),
    ('strict_release_control','executor_release_contract_20261113_v11','controls','predeclared_controls.json'),
    ('release_contract_fresh_seeds','executor_release_contract_20261113_v11','fresh_seeds','predeclared_fresh_seeds.json'),
    ('posture_planning_before_grasp','executor_pregrasp_posture_20261113_v12','native_validation','predeclared_plan.json'),
    ('pregrasp_posture_additional_seeds','executor_pregrasp_posture_20261113_v12','additional_seeds','predeclared_additional_seeds.json'),
    ('contact_vs_transport_frame','executor_dual_grasp_frame_20261113_v13','native_validation','predeclared_plan.json'),
    ('aligned_drop_and_opening_motion_factorial','executor_drop_alignment_20261113_v14','native_validation_r2','predeclared_retry_plan.json'),
    ('posture_branch_control_lead','executor_posture_lead_20261113_v15','native_validation','predeclared_plan.json'),
    ('posture_control_lead_with_aligned_drop_and_public_views','executor_drop_visual_20261113_v16','native_validation','predeclared_plan.json'),
    ('release_retreat_after_physical_detach','executor_release_retreat_20261113_v17','native_validation','predeclared_plan.json'),
    ('vertical_settle_before_physical_release','executor_release_settle_20261113_v18','native_validation','predeclared_plan.json')]

def main():
    previous=ROOT/'artifacts/framework/executor_continuation_20261113_v1/repair_report.json'
    report=read(previous);new=[];plans=[];exits=[]
    for name,namespace,folder,plan_name in GROUPS:
        base=ROOT/'artifacts/framework'/namespace;plan=read(base/plan_name)
        group=cohort(name,f'artifacts/framework/{namespace}/{folder}/*/worker/result.json',
            'TRAIN_adapted_source_oracle_NN0_diagnostic_not_model_rate')
        for row in group['rows']:
            value=read(row['result']['path'])
            row.update(profile=value['profile'],ever_held=value.get('ever_held'),
                rejected_pose_chunks=value.get('rejected_pose_chunks'),
                model_policy_success_claimed=False)
        if len(group['rows'])!=len(plan['cases']):raise RuntimeError('Incomplete predeclared cohort: '+name)
        for case in plan['cases']:
            path=base/folder/Path(case).stem;exit_receipt=read(path/'actual_exit.json');result=read(path/'worker/result.json')
            if not(exit_receipt['complete']and exit_receipt['actual_wait_completed']
                   and not exit_receipt['unreaped_descendants']and result['native_closed']and result['source_unchanged']):
                raise RuntimeError('Unqualified Native process lifecycle: '+str(path))
            if result['NN_forward_calls']!=0 or result['optimizer_steps']!=0:raise RuntimeError('Oracle/policy evidence mixed')
            exits.append(pointer(path/'actual_exit.json'))
        new.append(group);plans.append(pointer(base/plan_name))
    for row in report['frozen_baseline_audit']:
        if sha(row['path'])!=row['expected']:raise RuntimeError('Frozen baseline changed: '+row['path'])
    report.update(state='executor_semantic_and_contact_repairs_in_progress',previous_report=pointer(previous),
        cohorts=report['cohorts']+new,new_diagnostic_cohorts=[g['name']for g in new],
        new_native_case_count=sum(len(g['rows'])for g in new),new_predeclared_plans=plans,
        new_process_exit_receipts=exits,active_service=None,all_tasks_fixed=False,
        model_weights_changed=False,optimizer_steps=0,new_formal_VAL_TEST_started=False,
        default_runtime_changed=False,experimental_profiles_not_globally_adopted=True,
        NN_forward_calls_this_continuation=0,phase_identifiers_are_explicit_oracle_metadata_not_neural_predictions=True,
        controller_does_not_read_TRAIN_object_rigidity_labels=True,
        compatibility_decision=pointer(ROOT/'artifacts/framework/execution_protocol_20261113_v2/compatibility_decision.json'),
        execution_protocol_version=2,legacy_pose8_compatibility_preserved=True,
        execution_protocol_code=str(ROOT/'our_mani/execution_protocol_20261113_v2'),
        source_snapshot=pointer(OUT/'source_snapshot_final/manifest.json'),
        prepared_release_order_proposal=pointer(ROOT/'our_mani/executor_release_order_proposal_20261113_v19/README.md'),
        prepared_release_order_cases=pointer(ROOT/'artifacts/framework/executor_release_order_proposal_20261113_v19/predeclared_plan.json'),
        release_order_proposal_Native_started=False,
        remaining_user_decision='whether_to_allow_separate_open_fingers_then_detach_experiment',
        TRAIN_stage_label_audit=pointer(ROOT/'artifacts/framework/execution_protocol_20261113_v2/TRAIN_stage_label_audit.json'),
        protocol_and_executor_unit_tests=pointer(OUT/'unit_test_report.json'),
        remaining_compatibility_constraint=None)
    report['task_types_with_any_qualified_oracle_success']=sorted({row['task']for group in report['cohorts']
        if group['kind'].startswith('TRAIN_')and'not_adopted'not in group['name']
        for row in group['rows']if row['success']and row['reference_differences_within_limits']
        and row['hardware_table_contact_frames']==0})
    report['unresolved_mechanical_faults']=[dict(task=row['task'],seed=row['seed'],result=row['result'])
        for group in report['cohorts']for row in group['rows']if row['stop_reason']=='hardware_model_diverged']
    report['diagnostic_findings']=[
        dict(problem='contact_pose_and_nominal_grasp_reference_were_mixed',
             correction='capture_quality_uses_contact_pose_transport_delta_uses_nominal_reference',
             deliberate_reference_offset_m=.02),
        dict(problem='coarse_capture_and_uncompensated_grasp_in_fast_buzz',
             factorial_native_success=[False,False,False,True],development_success_seconds=8.2,
             fresh_TRAIN_seeds_successes=2,fresh_TRAIN_seeds_total=3,
             fresh_seed_profile='protocol_bridge_buzz_fresh_seeds'),
        dict(problem='stalled_reference_kept_a_stale_recovery_snapshot',
             correction='bounded_reference_velocity_toward_current_robot_sensing',
             actual_measurements_clipped=False,reference_state_reseeded=False),
        dict(problem='held_attachment_flag_does_not_guarantee_rigid_object_transform',
             TRAIN_label_audit_only=True,peg_relative_translation_before_m=.0493986194,
             peg_relative_rotation_before_rad=.7530025575,
             peg_relative_translation_after_alignment_m=.0009904176,
             peg_relative_rotation_after_alignment_rad=.0561417642,
             same_seed=True,alignment_alone_did_not_solve_task=True),
        dict(problem='same_phase_cup_target_can_switch_to_another_spoke',
             evidence='17.6s_same_wp5_transition_has_about_12cm_position_discontinuity',
             correction='phase_aware_target_continuity_only_after_grasp',
             early_unheld_filter_regression_preserved=True),
        dict(problem='static_lift_source_phase_advanced_after_one_100ms_query',
             correction='explicit_static_segment_completion_ACK',
             source_phase_labels_not_available_from_current_frozen_neural_head=True),
        dict(problem='vertical_insertion_prior_is_not_a_general_cup_placement_path',
             evidence='cup_final_segment_has_horizontal_and_vertical_motion',
             diagnostic='separate_actual_bounded_bias_and_vertical_compliant_feedback'),
        dict(problem='fast_peg_local_controller_reaches_multiple_native_joint_bounds',
             joint_limits_not_relaxed=True,robot_only_TRAIN_posture_bank_added=True,
             endpoint_IK_is_not_global_path_certificate=True),
        dict(problem='generic_terminal_release_gate_vetoed_source_opening',
             blocked_release_decisions=1004,previous_native_success=False,
             strict_release_control_native_success=False,
             correction='explicit_caller_release_contract_and_counted_gripper_event',
             high_peg_development_success_seconds=8.9,normal_cup_development_success_seconds=32.45,
             release_delay_10_ticks_high_peg_success=False,
             source_threshold_m=.03,old_release_position_tolerance_m=.005,
             body_goal_and_native_success_predicate_unchanged=True),
        dict(problem='same_contact_goal_received_transport_reference_offset_while_fingers_closed',
             correction='retain_contact_delta_until_phase_changes_then_use_nominal_transport_delta',
             deliberate_cup_contact_stroke_m=.02,
             additional_fix='capture_receipt_checks_terminal_contact_not_raised_hover_waypoint',
             Native_body_pose_used=False),
        dict(problem='configured_control_lead_was_omitted_from_primary_robot_posture_planning',
             correction='worker_passes_control_lead_and_posture_IK_uses_total_declared_horizon',
             calibration='TRAIN_robot_goal_projection_only_not_task_body_pose',
             control_lead_seconds=.035,release_capture_and_completion_evaluated_at_current_time=True)]
    report['known_unresolved']=[
        'seed_diversity_and_mechanical_stability_not_yet_qualified_for_all_tasks',
        'high_speed_peg_fresh_seed_failures',
        'frozen_neural_pose_quality_and_missing_trained_stage_and_timing_heads',
        'language_adapter_not_yet_trained',
        'original_boundary_precision_failure_mode_not_patched']
    report['common_profile_checks']=[]
    for group in new:
        profiles={(row['profile'],row['task'])for row in group['rows']}
        for profile,task in sorted(profiles):
            selected=[row for row in group['rows']if row['profile']==profile and row['task']==task]
            if len(selected)>=3 and len({row['seed']for row in selected})==len(selected):
                report['common_profile_checks'].append(dict(cohort=group['name'],profile=profile,task=task,
                    TRAIN_oracle_only=True,formal_VAL_TEST=False,successes=sum(bool(row['success'])for row in selected),
                    total=len(selected),seeds=[row['seed']for row in selected],
                    physical_seconds=[row['physical_seconds']for row in selected]))
    report['physical_phase_diagnostics']=[pointer(OUT/name)for name in (
        'release_command_blockade_audit.json','placement_alignment_blockade_audit.json',
        'peg_release_rigidity_audit.json','peg_release_landing_audit.json',
        'posture_branch_tracking_lag_audit.json','posture_control_lead_causal_audit.json')]
    report['physical_phase_diagnostics'].append(pointer(OUT/'absolute_ring_alignment_at_release_audit.json'))
    report['incomplete_native_cases'].append(dict(
        failure=pointer(ROOT/'artifacts/framework/executor_drop_alignment_20261113_v14/native_validation/00_peg_aligned_only_seed2611133001/worker/failure.json'),
        process_lifecycle=pointer(ROOT/'artifacts/framework/executor_drop_alignment_20261113_v14/native_validation/00_peg_aligned_only_seed2611133001/actual_exit.json'),
        detail='Launcher selected old v13 driver; aligned_drop was rejected before action execution. Failure retained; corrected owner reran all six unchanged cases in native_validation_r2.'))
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'repair_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    lines=['执行器诊断与续修记录（2026-11-13，第二轮）','',
        '全部新实验为 TRAIN 真值目标输入，NN 调用和优化器步数为 0；不是模型成功率，也不是正式 VAL／TEST。',
        '原生任务、物理、关节限位和成功判定保留；所有预登记案例与失败均列出，进程已真实 wait 并回收。','',
        '按用户选择扩展执行协议 v2，并保留旧 pose8。时间、阶段、完成回执和释放规则为独立元数据；旧输入经原终点执行器，旧／新驱动共享持续的执行器状态。',
        '已修复或验证：接触与搬运坐标分离、终端接触抓取门控、受限参考卸载、静态抬升完成回执、已抓取阶段目标连续性、物理开合完成锁存。',
        '高速穿环四组对照只有“精确闭合＋抓取补偿”成功，开发种子 8.20 秒；后续协议版本三个新种子 2/3 成功，仍不稳定。',
        '高速插环抓取漂移同种子曾从约 49 mm／43° 降到 1 mm 内／3.2°，但单独改善漂移不足以完成任务。',
        '目标连续性在抓取前误拦合法重选，曾使普通取杯回归；限定到已抓取阶段后恢复 17.65 秒原生成功。',
        '确认一处释放接口错误：原生目标在 3 cm 内请求开夹，旧通用 5 mm／角度门控反复拒绝了 1004 次；显式释放规则修复后，高速插环开发种子 8.90 秒成功，同环境严格门控对照仍 60 秒失败。',
        '普通放杯显式释放后 32.45 秒成功；加入抓取前姿态规划，已有 4.05／4.25／5.55 秒成功案例，困难种子配合悬停对准为 5.15 秒。它们分属不同原型，不能混成统一成功率。',
        '旧 TRAIN 来源已确认有阶段标签：56 个来源覆盖 896 个缓存窗口（672 拟合、224 TRAIN 内选型），尚未训练新的阶段或时间输出，物理采样间隔仍待核验。',
        '14 类非方块任务现均有专项原型原生成功案例；换种子失败和硬件模型偏差仍保留，不能认定全部修好。','',
        '同任务、同参数的跨种子检查（仍是 TRAIN 专家目标，不是模型成功率）：']
    for profile in report['common_profile_checks']:
        lines.append(f"  {profile['task']} | {profile['profile']} | {profile['successes']}/{profile['total']} | physical_seconds={profile['physical_seconds']}")
    lines+=['','本轮所有对照结果：']
    for group in new:
        lines.append(group['name'])
        for row in group['rows']:
            lines.append(f"  {row['task']} seed={row['seed']} success={row['success']} time={row['physical_seconds']:.2f}s stop={row['canonical_stop_reason']}")
    lines+=['','旧方块完整 TEST 保持普通 39/40、高速 27/40；冻结模型和旧报告哈希不变。',
        '新执行器仍是独立原型，尚未全量切换。现有模型只有 pose8 输出；新增阶段与时间元数据目前来自明确标注的 TRAIN 诊断调用方。',
        '单元验证和原生积分验证分开记录；本轮没有正式模型 VAL／TEST 结果。',
        '高速套环固定 35 ms 原型 2/4；加严格释放对准 1/4；再加退让 1/4；松夹前垂直稳定 0/4，均未作为默认。',
        '原生“先解除挂接、再张开”后的单步翻转仍未解决；“先张开再解除挂接”版本与 8 个成对案例已准备，尚未执行，等待用户选择是否允许改变抓取语义作独立实验。']
    (OUT/'诊断与修复进展.txt').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(report=str(OUT/'repair_report.json'),Native_cases=report['new_native_case_count'],
        task_types_with_any_oracle_success=len(report['task_types_with_any_qualified_oracle_success']),all_tasks_fixed=False),ensure_ascii=False))

if __name__=='__main__':main()
