"""Count the remaining block budget after a legacy policy failure, without retry.

Original early policy failure is retained. Additional commands are explicitly
stationary holds through the same counted native actuator. This is a new policy
completion rule; its motion compliance is measured, never assumed.
"""
import numpy as np


def finish_counted_budget(boundary, record):
    if record.get('failure') == 'development_reset_ik_failed':
        raise RuntimeError('original block development reset failed; infrastructure failure')
    before = int(boundary.ledger.ticks)
    if before < 0 or before > 200 or abs(boundary.ledger.dt-.05) > 1e-8:
        raise ValueError('original counted 200x50ms block clock required')
    original_success = bool(record['official_task_success'])
    original_failure = record.get('failure')
    outcomes = []
    terminal_failure = False
    while boundary.ledger.ticks < 200 and not record['official_task_success'] and not terminal_failure:
        boundary.hold_tick()
        success, terminal = boundary.task.success()
        if not isinstance(success, (bool, np.bool_)) or not isinstance(terminal, (bool, np.bool_)):
            raise ValueError('original native task outcome must be explicit')
        success, terminal = bool(success), bool(terminal)
        terminal_failure = terminal and not success
        record['official_task_success'] = success
        outcomes.append(dict(tick=int(boundary.ledger.ticks), native_success=success,
            native_terminal_failure=terminal_failure))
    record['block_full_budget_v176'] = dict(
        original_component_native_success=original_success,
        original_component_failure=original_failure,
        original_component_end_tick=before, final_tick=int(boundary.ledger.ticks),
        tail_counted_holds=len(outcomes), native_outcomes=outcomes,
        policy_failed_retries=0, stationary_hold_is_not_an_actual_motion_guarantee=True,
        original_early_failure_retained=True, native_outcomes_are_not_policy_inputs=True,
        additional_camera_queries=0, extra_unaccounted_physics_ticks=0)
    record['native_terminal_failure'] = terminal_failure
    if record['official_task_success']:
        record['failure'] = None
    return record
