from types import SimpleNamespace
import unittest
from our_mani.multitask.button_success_metrics_v100 import snapshot


def env(position=.004,history=None):
    def forbidden():raise AssertionError('native success callback/setter must not be called')
    task=SimpleNamespace(t=1.,joint=SimpleNamespace(get_joint_position=lambda:position),
        goal_condition=SimpleNamespace(_original_pos=0.,_pos=.003,condition_met=forbidden),
        descent_condition=SimpleNamespace(_window=10,_min_descent=.03,condition_met=forbidden),
        _tip_z_history=history if history is not None else [.04-i*.04/9 for i in range(10)],success=forbidden)
    return SimpleNamespace(ledger=SimpleNamespace(ticks=20),task=task,time=1.,physics_time=lambda:1.75)


class Tests(unittest.TestCase):
    def test_both_original_components_are_read_without_success_calls_or_setters(self):
        e=env();d=snapshot(e)
        self.assertTrue(d['joint_component_met']);self.assertTrue(d['descent_component_met'])
        self.assertTrue(d['simultaneous_components_met']);self.assertTrue(d['clock_unchanged'])
        self.assertFalse(d['diagnostic_values_supplied_to_policy'])

    def test_original_strict_thresholds_are_not_loosened(self):
        d=snapshot(env(position=.003,history=[.03]*9+[0.]))
        self.assertFalse(d['joint_component_met']);self.assertFalse(d['descent_component_met'])
        self.assertFalse(d['simultaneous_components_met'])

    def test_short_native_history_is_not_invented_or_padded(self):
        d=snapshot(env(history=[.9,.8]));self.assertIsNone(d['descent_over_native_window'])
        self.assertFalse(d['descent_component_met']);self.assertTrue(d['joint_component_met'])

    def test_clock_change_is_an_infrastructure_error(self):
        e=env()
        def moved():e.task.t+=.05;return .004
        e.task.joint.get_joint_position=moved
        with self.assertRaisesRegex(ValueError,'changed native clock'):snapshot(e)


if __name__=='__main__':unittest.main()
