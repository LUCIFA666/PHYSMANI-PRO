from .common import ROOT,OUT,PLAN,read,write,pointer,stopped

def prepare():
    stopped()
    previous=ROOT/'config/framework/multitask_executor_cycle_20261112_v6/plan.json'
    plan=read(previous)
    plan.update(schema='fourteen_task_author_arm_then_gripper_small_cycle_20261112_v7',
                previous_plan_v6=pointer(previous),
                previous_retirement_v6=pointer(ROOT/'artifacts/framework/multitask_executor_cycle_20261112_v6/retirement_receipt.json'),
                gripper_sequence='original_MoveArmThenGripper_arm_sequence_then_gripper',
                added_one_cm_measured_arrival_gate_removed=True,
                measured_arm_arrival_is_diagnostic_not_success_or_gripper_gate=True,
                same_predeclared_84_cases_and420_seeds=True,all_v7_cases_same_executor=True,
                no_VAL_TEST_outcomes_observed=True)
    write(PLAN,plan)
    write(OUT/'prospective_plan.json',dict(plan=pointer(PLAN),no_VAL_TEST_outcomes_read=True,
                                          same_frozen_head_checkpoint=True,Native_sources_modified=False))

if __name__=='__main__':prepare()
