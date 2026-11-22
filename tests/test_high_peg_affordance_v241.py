from types import SimpleNamespace as NS
import unittest
from our_mani.multitask.high_peg_affordance_v241 import primary_events


class Tests(unittest.TestCase):
    def test_released_recovery_does_not_replace_initial_operation(self):
        first=NS(phase='acquire',attempt_start_time=1.,attempt_end_time=2.,reference_mode='external')
        retry=NS(phase='acquire',attempt_start_time=5.,attempt_end_time=6.,reference_mode='released')
        self.assertIs(primary_events([retry,first])['acquire'],first)

    def test_missing_initial_completion_cannot_be_fabricated(self):
        event=NS(phase='lift',attempt_start_time=2.,attempt_end_time=3.)
        self.assertEqual(primary_events([event]),{})


if __name__=='__main__':unittest.main()
