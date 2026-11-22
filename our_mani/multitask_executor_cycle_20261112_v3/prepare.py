from .common import ROOT, OUT, PLAN, read, write, pointer, stopped


def prepare():
    stopped()
    previous = ROOT/'config/framework/multitask_executor_cycle_20261112_v2/plan.json'
    plan = read(previous)
    plan.update(schema='fourteen_task_fixed_cold_reset_small_cycle_20261112_v3',
                RML_velocity_preview_corrected=True, previous_cold_reset_plan=pointer(previous),
                previous_cycle_retirement=pointer(ROOT/'artifacts/framework/multitask_executor_cycle_20261112_v2/retirement_receipt.json'),
                same_predeclared_84_cases_and420_seeds=True, all_v3_cases_rerun_with_same_fixed_executor=True,
                no_VAL_TEST_outcomes_observed=True)
    write(PLAN, plan)
    write(OUT/'prospective_plan.json', dict(plan=pointer(PLAN), no_VAL_TEST_outcomes_read=True,
                                           same_frozen_head_checkpoint=True, Native_task_sources_modified=False))
    print(dict(cases=84, corrected='RML_independent_velocity_preview_initial_state'), flush=True)


if __name__ == '__main__':
    prepare()
