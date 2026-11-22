from .common import ROOT,OUT,PLAN,read,write,pointer,stopped

def prepare():
    stopped()
    previous=ROOT/'config/framework/multitask_executor_cycle_20261112_v5/plan.json'
    plan=read(previous)
    plan.update(schema='fourteen_task_contiguous_public_history_small_cycle_20261112_v6',
                previous_plan_v5=pointer(previous),
                previous_retirement_v5=pointer(ROOT/'artifacts/framework/multitask_executor_cycle_20261112_v5/retirement_receipt.json'),
                public_observation_cadence_seconds=.05,
                history_source='every_actual_physics_step_not_sparse_action_endpoints',
                inference='same_frozen_weights_GPU_BF16_same_as_original_cached_feature_extraction',
                GPU_fitting_during_Native=False,same_predeclared_84_cases_and420_seeds=True,
                all_v6_cases_same_executor=True,no_VAL_TEST_outcomes_observed=True)
    write(PLAN,plan)
    write(OUT/'prospective_plan.json',dict(plan=pointer(PLAN),no_VAL_TEST_outcomes_read=True,
                                          same_frozen_head_checkpoint=True,Native_sources_modified=False))

if __name__=='__main__':prepare()
