from types import SimpleNamespace as NS
import unittest
import numpy as np
from our_mani.aligned.protocol import TickLedger
from our_mani.multitask.physmani_block_receipt_v541 import ReceiptBlockView
from our_mani.multitask.baseline_measured_windows_v345 import CountedWindowObserver


class Tests(unittest.TestCase):
    def test_original_native_failure_conditions_must_actually_be_absent(self):
        with self.assertRaisesRegex(ValueError,'empty native failure'):
            ReceiptBlockView(NS(task=NS(_fail_conditions=[object()])))

    def test_receipt_composes_with_original_counted_window_observer_without_recursion_or_extra_calls(self):
        raw=NS(task=NS(_fail_conditions=[]),ledger=TickLedger(),_native_success_latched=False,time=0.,
            arm=NS(get_joint_positions=lambda:np.zeros(7)),
            robot_model=NS(acceleration_limits=np.full(7,12.),lower=np.full(7,-3.),
                           upper=np.full(7,3.),velocity_limits=np.full(7,3.)))
        def tick(phase):raw.ledger.ticks+=1;raw.time=raw.ledger.time
        raw._tick=tick
        raw.alignment_receipt=lambda:dict(clock=raw.ledger.receipt(),native_success_latched=raw._native_success_latched)
        env=ReceiptBlockView(raw);observer=CountedWindowObserver(env)
        env._tick('author_RML');receipt=env.alignment_receipt()
        self.assertFalse(receipt['native_terminal_failure_latched'])
        self.assertEqual(receipt['legacy_block_native_outcome_contract']['extra_task_success_calls'],0)
        self.assertEqual(len(receipt['all_stage_counted_joint_motion_v191']['events']),1)
        self.assertEqual(raw.ledger.ticks,1)
        raw._native_success_latched=True
        self.assertTrue(env.alignment_receipt()['native_success_latched'])


if __name__=='__main__':unittest.main()
