from types import SimpleNamespace as NS
import unittest
import numpy as np
from our_mani.multitask.native_window_observer_v191 import CountedWindowObserver


class Native:
    def __init__(self):
        self.q = np.zeros(7)
        self.delta = np.full(7,.001)
        self.calls = 0
        self.ledger=NS(ticks=0,dt=.05)
        self.arm=NS(get_joint_positions=lambda:self.q.copy())
        self.robot_model=NS(lower=np.full(7,-10.),upper=np.full(7,10.),velocity_limits=np.full(7,2.),acceleration_limits=np.full(7,6.))
    def _tick(self,phase):
        self.q += self.delta
        self.ledger.ticks += 1
        self.calls += 1
    def alignment_receipt(self):return dict(original_receipt=True)


class Tests(unittest.TestCase):
    def test_original_tick_forwarded_once_and_receipt_adds_no_physics(self):
        env=Native();o=CountedWindowObserver(env)
        env._tick('baseline_RML');env._tick('baseline_gripper')
        r=env.alignment_receipt()
        self.assertTrue(r['original_receipt'])
        self.assertEqual((env.calls,len(o.events)),(2,2))
        self.assertTrue(r['all_stage_counted_joint_motion_v191']['all_stage_actual_window_checks_passed'])
        self.assertIsNone(o.events[0]['acceleration_passed'])

    def test_actual_violation_preserved_without_changing_original_motion(self):
        env=Native();o=CountedWindowObserver(env)
        env._tick('baseline_RML');env.delta=np.full(7,.05);env._tick('baseline_RML')
        np.testing.assert_allclose(env.q,np.full(7,.051))
        self.assertEqual(o.receipt()['violations'],[2])
        self.assertAlmostEqual(o.receipt()['peak_actual_window_acceleration'],19.6)

    def test_unobserved_or_extra_tick_cannot_satisfy_complete_observer(self):
        env=Native();CountedWindowObserver(env);env.ledger.ticks=1
        with self.assertRaises(ValueError):env.alignment_receipt()
        env=Native();CountedWindowObserver(env);env.ledger.dt=.1
        with self.assertRaises(ValueError):env._tick('bad-clock')
        self.assertEqual(env.calls,1)

    def test_observer_cannot_attach_after_counted_rollout_started(self):
        env=Native();env._tick('already-started')
        with self.assertRaises(ValueError):CountedWindowObserver(env)


if __name__ == '__main__':unittest.main()
