import hashlib
from .common import ROOT, OUT, PLAN, PREVIOUS_PLAN, PREVIOUS_OUT, read, write, pointer, stopped


def prepare():
    stopped()
    previous = read(PREVIOUS_PLAN)
    cases = []
    for case in previous['cases']:
        seeds = [case['seed']]
        for attempt in range(1, 5):
            material = f'cold-reset-only-v2:{case["id"]}:{attempt}'
            seeds.append(int.from_bytes(hashlib.sha256(material.encode()).digest()[:4], 'big'))
        cases.append(dict(case, cold_reset_seeds=seeds))
    all_seeds = [s for c in cases for s in c['cold_reset_seeds']]
    if len(set(all_seeds)) != len(all_seeds):
        raise ValueError('Every potential generated reset needs a unique identity')
    plan = dict(schema='fourteen_task_fixed_cold_reset_small_cycle_20261112_v2',
                previous_plan=pointer(PREVIOUS_PLAN), previous_fit_status=pointer(PREVIOUS_OUT/'head_fit/fit_status.json'),
                previous_retirement=pointer(PREVIOUS_OUT/'retirement_receipt.json'),
                checkpoint=pointer(PREVIOUS_OUT/'head_fit/selected.pt'),
                tasks=previous['tasks'], cases=cases, expected_cases=84,
                preparation_uses_only_TRAIN_reset_failure=True,
                VAL_TEST_outcomes_observed_before_retry_rule=False,
                cold_reset_max_processes=5, retry_allowed_only_before_initial_json=True,
                retry_error_type='TaskEnvironmentError',
                valid_policy_failure_or_NativeFalse_retry_allowed=False,
                invalid_reset_attempts_saved_and_not_counted_as_NativeFalse=True,
                no_action_outcome_based_seed_selection=True,
                task_native_constructor_reset_validation_preserved=True,
                generated_reset_distribution='native init_task sampler, conditioned on original reset validation completing',
                official_archived_VAL_TEST=False, small_generated_development_evaluation=True,
                default_entry_or_weights_changed=False,
                import_first_failed_reset=dict(case_id=cases[0]['id'], attempt=0,
                    owner=str(PREVIOUS_OUT/'native'/cases[0]['id'])))
    write(PLAN, plan)
    write(OUT/'prospective_plan.json', dict(plan=pointer(PLAN), no_VAL_TEST_outcomes_read=True,
                                           previous_training_reused_without_mutation=True))
    print(dict(cases=84, max_reset_attempts=5, model=plan['checkpoint']), flush=True)


if __name__ == '__main__':
    prepare()
