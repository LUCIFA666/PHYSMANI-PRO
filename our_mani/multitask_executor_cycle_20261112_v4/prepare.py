from .common import ROOT, OUT, PLAN, read, write, pointer, stopped


def prepare():
    stopped()
    previous = ROOT/'config/framework/multitask_executor_cycle_20261112_v3/plan.json'
    plan = read(previous)
    plan.update(schema='fourteen_task_atomic_query_fixed_cold_reset_small_cycle_20261112_v4',
                robot_query_rollback_added=True, previous_plan_v3=pointer(previous),
                previous_retirement_v3=pointer(ROOT/'artifacts/framework/multitask_executor_cycle_20261112_v3/retirement_receipt.json'),
                same_predeclared_84_cases_and420_seeds=True, all_v4_cases_same_executor=True,
                no_VAL_TEST_outcomes_observed=True)
    write(PLAN, plan)
    write(OUT/'prospective_plan.json',dict(plan=pointer(PLAN), no_VAL_TEST_outcomes_read=True,
                                          same_frozen_head_checkpoint=True, Native_sources_modified=False))


if __name__ == '__main__':
    prepare()
