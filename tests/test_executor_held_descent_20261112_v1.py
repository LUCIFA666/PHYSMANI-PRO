import unittest
from our_mani.executor_cartesian_tracking_20261112_v1.phases import HeldDescent

class HeldDescentTests(unittest.TestCase):
    def test_lift_then_lower_activates_independently_of_table_height_threshold(self):
        phase=HeldDescent()
        self.assertFalse(phase.observe(held=False,goal_z=.759))
        self.assertFalse(phase.observe(held=True,goal_z=.759))
        self.assertFalse(phase.observe(held=True,goal_z=.997))
        self.assertFalse(phase.observe(held=True,goal_z=.969))
        self.assertTrue(phase.observe(held=True,goal_z=.881))
        self.assertFalse(phase.observe(held=False,goal_z=.881))

if __name__=='__main__':unittest.main()
