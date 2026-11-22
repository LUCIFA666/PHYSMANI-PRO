"""Install the new budget after an unchanged legacy reset, before any action."""
from copy import deepcopy
from . import TaskBudget


def install_legacy_budget(env,budget):
    if not isinstance(budget,TaskBudget):raise TypeError('Shared TaskBudget required')
    if env.ledger.ticks!=0 or abs(env.time)>1e-9:raise ValueError('A task budget must be declared before its first controlled tick')
    if env.ledger.dt!=.05:raise ValueError('Original50ms Native physics required')
    before=env.time;old=env.ledger.budget
    env.ledger.budget=budget.seconds
    env.protocol=deepcopy(env.protocol)
    env.protocol['simulation_seconds']=budget.seconds
    receipt=dict(budget.receipt(),legacy_budget_seconds=old,declared_before_first_controlled_tick=True,
        source_protocol_preserved=True,Native_reset_and_success_methods_changed=False,
        old_trained_policy_protocol_requalified=False)
    env.task_budget_override=receipt
    if env.time!=before or env.ledger.ticks!=0:raise RuntimeError('Budget installation advanced physics')
    return receipt
