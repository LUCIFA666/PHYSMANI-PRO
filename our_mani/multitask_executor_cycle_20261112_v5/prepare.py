from .common import ROOT,OUT,PLAN,read,write,pointer,stopped

def prepare():
    stopped()
    previous=ROOT/'config/framework/multitask_executor_cycle_20261112_v4/plan.json'
    plan=read(previous)
    plan.update(schema='fourteen_task_bounded_native_path_small_cycle_20261112_v5',
                previous_plan_v4=pointer(previous),
                previous_retirement_v4=pointer(ROOT/'artifacts/framework/multitask_executor_cycle_20261112_v4/retirement_receipt.json'),
                RML_scalar_endpoint_extrapolation_removed=True,
                invalid_planned_joint_references_explicitly_rejected_before_execution=True,
                same_predeclared_84_cases_and420_seeds=True,all_v5_cases_same_executor=True,
                no_VAL_TEST_outcomes_observed=True)
    write(PLAN,plan)
    write(OUT/'prospective_plan.json',dict(plan=pointer(PLAN),no_VAL_TEST_outcomes_read=True,
                                          same_frozen_head_checkpoint=True,Native_sources_modified=False))

if __name__=='__main__':prepare()
