#!/usr/bin/env python3
"""Build an evidence index; oracle checks and learned-policy outcomes stay separate."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/framework/executor_continuation_20261113_v1'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def read(path):return json.loads(Path(path).read_text())
def pointer(path):return dict(path=str(Path(path).resolve()),sha256=sha(path))

def cohort(name,pattern,kind):
    rows=[]
    for path in sorted(ROOT.glob(pattern)):
        value=read(path);audit=value.get('issued_reference_audit',{})
        terminal=value.get('native_task_judge',{}).get('original_native_metric',{}).get('native_first_terminal')or{}
        row=dict(task=value['task'],seed=value['seed'],success=value.get('episode_success',value.get('original_native_success')),
            physical_seconds=value['actual_seconds'],physical_ticks=value['actual_ticks'],stop_reason=value['stop_reason'],
            native_closed=value['native_closed'],source_unchanged=value['source_unchanged'],
            hardware_table_contact_frames=value.get('hardware_contact_audit',{}).get('prohibited_robot_table_contact_frames'),
            reference_differences_within_limits=audit.get('all_reference_differences_within_limits'),result=pointer(path))
        row['Native_terminal_failure']=bool(terminal.get('terminal_failure'))
        row['canonical_stop_reason']='original_Native_terminal_failure'if row['Native_terminal_failure']else row['stop_reason']
        exit_path=path.parents[1]/'actual_exit.json'
        row['process_lifecycle']=pointer(exit_path)if exit_path.exists()else None
        rows.append(row)
    by_task={}
    for row in rows:
        result=by_task.setdefault(row['task'],dict(successes=0,total=0))
        result['total']+=1;result['successes']+=int(bool(row['success']))
    return dict(name=name,kind=kind,formal_VAL_TEST=False,rows=rows,by_task=by_task)

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    groups=[
        cohort('button_frozen_policy','artifacts/framework/executor_contract_repair_20261112_v6/frozen_policy_60s_r2/*/worker/result.json','frozen_policy_TRAIN_diagnostic'),
        cohort('button_frozen_policy_new_seeds','artifacts/framework/executor_contract_repair_20261112_v6/frozen_policy_new_seeds/*/worker/result.json','frozen_policy_TRAIN_diagnostic'),
        cohort('button_oracle','artifacts/framework/executor_contract_repair_20261112_v6/oracle_validation_r2/*/worker/result.json','TRAIN_oracle_inputs_NN0'),
        cohort('basketball_frozen_policy','artifacts/framework/executor_cartesian_consumer_20261112_v1/frozen_policy_60s/*/worker/result.json','frozen_policy_TRAIN_diagnostic'),
        cohort('basketball_oracle','artifacts/framework/executor_cartesian_consumer_20261112_v1/oracle_qualification/*/worker/result.json','TRAIN_oracle_inputs_NN0'),
        cohort('peg_compliant_phase','artifacts/framework/executor_cartesian_tracking_20261112_v1/compliant_insert_phase/*/worker/result.json','TRAIN_oracle_inputs_NN0_single_case_each_not_rate'),
        cohort('peg_native_limit_experiment_not_adopted','artifacts/framework/executor_differential_tracking_20261112_v1/native_limits/*/worker/result.json','TRAIN_oracle_experiment_not_default'),
        cohort('other_eight_timed_source','artifacts/framework/executor_timed_tasks_20261112_v1/native_qualification/*/worker/result.json','TRAIN_oracle_inputs_NN0_single_case_each_not_rate'),
        cohort('grasp_frame_taskspace_trial','artifacts/framework/executor_grasp_frame_20261113_v1/native_qualification/*/worker/result.json','TRAIN_oracle_inputs_NN0_trial'),
        cohort('nominal_grasp_frame','artifacts/framework/executor_grasp_frame_20261113_v1/nominal_frame/*/worker/result.json','TRAIN_adapted_source_oracle_NN0_single_case_each_not_rate'),
        cohort('contact_semantics','artifacts/framework/executor_grasp_frame_20261113_v1/contact_semantics/*/worker/result.json','TRAIN_adapted_source_oracle_NN0_single_case_each_not_rate'),
        cohort('forecast_call_trace_and_joint_recovery','artifacts/framework/executor_grasp_frame_20261113_v1/timing_recovery_r2/*/worker/result.json','TRAIN_adapted_source_oracle_NN0_single_case_each_not_rate'),
        cohort('aligned_approach_and_tracking_recovery','artifacts/framework/executor_grasp_frame_20261113_v1/approach_recovery/*/worker/result.json','TRAIN_adapted_source_oracle_NN0_ablation_not_rate'),
        cohort('aligned_hover_new_seeds_and_transfer','artifacts/framework/executor_grasp_frame_20261113_v1/hover_validation/*/worker/result.json','TRAIN_adapted_source_oracle_NN0_new_seed_checks_not_formal_VAL')]
    expected={
        'artifacts/framework/cube_full_reuse_20261112_v1/report.json':'e1b240fe224615e6a47fb83b0b79f7bd2f2b7c6df86fd8761924b6b49d894c09',
        'artifacts/framework/multitask_executor_cycle_20261112_v1/head_fit/selected.pt':'65a906fc96ffe8d7b453fa22b47199bf153d91ff712235f001c368682e1c36af',
        'runtime/framework/STOP':'e2bb2d8aff8697cf3f0c18a7a0707b2bd7e8e817d18b563513c365bf0e5a1850'}
    baseline=[dict(path=str(ROOT/path),expected=target,actual=sha(ROOT/path))for path,target in expected.items()]
    if any(r['expected']!=r['actual']for r in baseline):raise RuntimeError('Frozen baseline changed')
    report=dict(state='executor_repairs_in_progress_model_success_still_low',all_tasks_fixed=False,
        model_weights_changed=False,optimizer_steps=0,new_formal_VAL_TEST_started=False,default_runtime_changed=False,
        original_native_physics_and_success_preserved=True,task_time_limit_seconds=60,
        cohorts=groups,frozen_baseline_audit=baseline,
        executor_priors=['robot_FK_and_static_table_geometry','original_gripper_proximity_and_held_outside_NN',
            'task_specific_press_retract','explicit_target_sample_lookahead_and_goal_motion_history',
            'gripper_release_event_completion','optional_rigid_grasp_frame_and_task_space_control'],
        source_oracle_adaptations=['shadow_scene_for_read_only_original_expert_queries',
            'optional_measured_grasp_frame_corrected_robot_pose_for_source_waypoint_distance_guards',
            'actual_forecast_helper_call_trace_including_function_local_imports'],
        known_unresolved=['high_speed_peg','normal_and_high_speed_cup_place','high_speed_cup_remove','high_speed_buzz',
            'frozen_policy_pose_and_timing_quality',
            'original_boundary_precision_failure_mode_not_patched'],
        not_a_PhysMani_policy_comparison=True,
        development_failures_retained=['executor_contract_repair_20261112_v6/oracle_validation/00_oracle_push_moving_button_seed2611138000',
            'executor_capture_repair_20261112_v1/precise_release/00_moving_basketball_in_hoop_f6.0',
            'executor_cartesian_tracking_20261112_v1/native_qualification/02_insert_onto_rotating_peg',
            'executor_grasp_frame_20261113_v1/native_qualification/01_place_cups_on_rotating_frame_high_speed',
            'executor_grasp_frame_20261113_v1/timing_recovery/launcher_failure.json'])
    report['incomplete_native_cases']=[dict(failure=pointer(path),detail=read(path))for path in
        sorted(ROOT.glob('artifacts/framework/executor_grasp_frame_20261113_v1/*/*/worker/failure.json'))]
    report['predeclared_grasp_frame_cohorts']=[pointer(path)for path in
        sorted(ROOT.glob('artifacts/framework/executor_grasp_frame_20261113_v1/predeclared_*.json'))]
    report['latest_grasp_frame_safety_qualification']='experimental_profiles_not_production_qualified'
    report['unresolved_mechanical_faults']=[dict(task=row['task'],seed=row['seed'],result=row['result'])
        for group in groups for row in group['rows']if row['stop_reason']=='hardware_model_diverged']
    report['task_types_with_any_qualified_oracle_success']=sorted({row['task']for group in groups
        if group['kind'].startswith('TRAIN_')and'not_adopted'not in group['name']
        for row in group['rows']if row['success']and row['reference_differences_within_limits']
        and row['hardware_table_contact_frames']==0})
    report['any_success_is_not_all_tasks_fixed_or_model_success_rate']=True
    (OUT/'repair_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    text='''执行器续修记录（2026-11-13）

当前结论：接口与物理执行已有改善，尚未全部修好。新结果全部是 TRAIN 诊断，不是正式 VAL／TEST。

冻结模型输入：普通按钮 1/4，高速按钮 1/4；普通／高速篮球各 0/1。
专家目标输入、NN 调用为 0：按钮普通 3/4、高速 4/4；篮球普通 3/4、高速 4/4。
普通插环已有 7.15 秒原生成功单例，高速仍失败；后续关节限位实验未改善成功率，未采用为默认。
普通／高速投垃圾分别在 7.20／6.15 秒原生成功，普通 buzz 穿环 4.30 秒原生成功（各一个专家目标诊断）。
普通取杯在抓取坐标补偿原型中 12.45 秒原生成功；后续接触语义、时序修复试验同一种子分别 18.95、17.65 秒成功，尚未成为默认入口。
高速取杯在上方对齐再下降的开发种子中首次原生成功，用时 18.15 秒；单加停滞恢复失败，组合恢复仍在同一时刻成功。3 个新 TRAIN 种子为 1/3：14.45 秒成功、60 秒未抓住、30.50 秒硬件模型偏差保护。不能把开发种子当成泛化成功率。
同一接近动作在高速放杯中 12.30 秒原生成功，但关节跟踪误差 L2 最大 0.888 rad，累计 3.30 秒超过 0.5 rad，不能据此认定执行稳定。
普通取杯在新接近动作下回归为 60 秒失败，普通放杯在 42.60 秒触发硬件模型偏差保护，因此不全量替换既有执行器。
14 类非方块任务中有 QUALIFIED_TYPE_COUNT 类至少在一个专项执行器原型中出现过原生成功；这不是这些类型全部修好，也不是模型成功率。
高速插环、普通放杯和高速 buzz 穿环仍未通过；高速取杯／放杯尚未稳定，最新复测全部逐条列在 repair_report.json 中。

已修内容：位姿与轨迹接口区分、连续且受限的关节参考、静态桌面避碰、实际末端反馈、目标时间提前量、夹爪空夹、短暂释放意图被取消、实际开合完成与故障执行记录。
新试验：按实际抓取偏差补偿后续放置位姿；直接在任务空间分配受限关节动作；捕获任务函数局部导入的真实预测时长；实际关节越限后显式下发向内退让参考，保留原始测量值。
最新高速插环退让试验在 15.80 秒触发硬件运动学模型偏差保护，不能认定为机械合格，失败保留在报告中。
抓取前先在上方对齐、关节跟踪停滞后受限退让的消融实验独立列出，未切换为默认。
高速 buzz 在新方案下 7.00 秒抓住，7.20 秒触发原生终止失败；部分旧 worker 将其写成 budget_exhausted，报告另列 canonical_stop_reason 按原生判定纠正，原始记录保留。

所有成功只来自原始 Task.success()，每个实际 50 ms 仿真步采样，首次成功即停止。未改原生任务物理、场景或判定，未把专家目标诊断当成模型成功率。
本轮未更新神经网络权重，未启动新的正式 VAL／TEST，旧运行默认入口未切换。
方块既有完整 TEST：普通 39/40（97.5%）、高速 27/40（67.5%），报告哈希已核对不变。

关键限制：执行器使用了明确的操作与时间先验及原生抓取传感器反馈，不能据此声称模型优于 PhysMani；当前模型仍有大量目标被几何约束拒绝，历史训练帧间隔与新接口的匹配也尚未完成核验。
'''
    text=text.replace('QUALIFIED_TYPE_COUNT',str(len(report['task_types_with_any_qualified_oracle_success'])))
    (OUT/'修复进展.txt').write_text(text)
    print(json.dumps(dict(report=str(OUT/'repair_report.json'),cohort_sizes={g['name']:len(g['rows'])for g in groups}),ensure_ascii=False))

if __name__=='__main__':main()
