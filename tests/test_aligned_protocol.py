import unittest
from our_mani.aligned.protocol import TickLedger


class AlignedProtocolTests(unittest.TestCase):
    def test_gripper_steps_consume_budget(self):
        clock = TickLedger(budget=.15)
        for phase in ('arm','close','open'):
            clock.before(); clock.after(.05,.05,phase)
        self.assertAlmostEqual(clock.time,.15)
        self.assertEqual(clock.phases,{'arm':1,'close':1,'open':1})
        with self.assertRaises(TimeoutError):
            clock.before()

    def test_target_cannot_freeze_during_gripper(self):
        clock = TickLedger()
        with self.assertRaises(RuntimeError):
            clock.after(.05,0.,'close')
        self.assertEqual(clock.ticks,0)

    def test_hidden_physics_step_rejected(self):
        clock = TickLedger()
        with self.assertRaises(RuntimeError):
            clock.after(.1,.05,'path')
        self.assertEqual(clock.ticks,0)


if __name__ == '__main__':
    unittest.main()
