"""Parent-only legacy block outcome receipt for the unified native validator.

The two source tasks register no failure conditions. Verify that actual native
configuration explicitly; do not invent a missing failure measurement or call
task.success() again. The original counted tick remains its only evaluator.
"""
from .physmani_block_policy_v538 import TerminalBlockView


class ReceiptBlockView(TerminalBlockView):
    def __init__(self,env):
        if not isinstance(env.task._fail_conditions,list) or env.task._fail_conditions:
            raise ValueError('legacy block adapter requires the actual original empty native failure-condition list')
        original=env.alignment_receipt
        def receipt():
            if env.task._fail_conditions:raise ValueError('native block failure conditions changed during execution')
            result=original()
            return dict(result,native_terminal_failure_latched=False,
                legacy_block_native_outcome_contract=dict(native_failure_conditions=0,
                    observed_native_condition_registration=True,extra_task_success_calls=0,
                    original_native_success_latch_preserved=True))
        env.alignment_receipt=receipt
        super().__init__(env)
