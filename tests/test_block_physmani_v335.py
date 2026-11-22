from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
import json
import tempfile
import unittest
import numpy as np

from our_mani.multitask import block_physmani_v335 as baseline

ROOT = Path(__file__).resolve().parents[1]


class Ledger:
    dt = .05
    ticks = 0
    @property
    def remaining(self):
        return 10. - self.ticks * self.dt


class Environment:
    def __init__(self):
        self.ledger = Ledger()
        self.q = np.zeros(7)
        self.time = 0.
        self.closed = False
        self.observations = 0
        self._native_success_latched = False
        self.arm = NS(get_joint_positions=lambda: self.q.copy())
        self.robot_model = NS(lower=np.full(7, -3.), upper=np.full(7, 3.),
            velocity_limits=np.full(7, 2.), acceleration_limits=np.full(7, 6.))
    def reset(self, **kwargs):
        return True
    def finish_home_reset(self):
        pass
    def observe(self):
        self.observations += 1
        return {}, NS(timestamp=self.time)
    def _tick(self, phase):
        self.q += .001
        self.ledger.ticks += 1
        self.time = self.ledger.ticks * .05
        self._native_success_latched = self.ledger.ticks == 2
    def alignment_receipt(self):
        return dict(clock=dict(ticks=self.ledger.ticks))
    def close(self):
        self.closed = True


class Adapter:
    compute_seconds = .01
    def __init__(self, *args):
        pass
    def reset(self, *args):
        pass
    def predict(self, *args):
        return np.array([0., 0., 1., 0., 0., 0., 1., 1.])
    def execute(self, env, action):
        for tick in range(10):
            if env._native_success_latched:
                break
            env._tick('original_baseline')
        return dict(ticks=env.ledger.ticks)


class Tests(unittest.TestCase):
    def fixture(self, checkpoint):
        config = json.loads((ROOT / 'config/framework/aligned_protocol_r09_baseline_v6.json').read_text())
        config['baseline_checkpoint'] = checkpoint
        config['block_physmani_v335'] = baseline.binding(ROOT, config)
        row = dict(task='pick_moving_target_on_the_table', raw_episode_id='episode0', path='synthetic-only-test')
        return config, row

    def test_original_decision_stops_at_native_success_and_every_real_tick_is_observed(self):
        with tempfile.NamedTemporaryFile() as checkpoint:
            config, row = self.fixture(checkpoint.name)
            env = Environment()
            with patch.object(baseline, 'build_policy', return_value=object()) as build, \
                    patch.object(baseline, 'AlignedLocalSimulation', return_value=env), \
                    patch.object(baseline, 'PhysManiAdapter', Adapter), \
                    patch.object(baseline, 'public_initial_conditions', return_value={}), \
                    patch('scripts.collect_local_rollouts.initialization', return_value=(0, 0, 0, 0)):
                record = baseline.run_TRAIN_baseline(ROOT, row, config)
            build.assert_called_once()
            self.assertTrue(env.closed)
            self.assertEqual(env.observations, 2)
            self.assertTrue(record['official_task_success'])
            self.assertEqual(record['counted_clock']['ticks'], 2)
            self.assertEqual(len(record['steps']), 1)
            motion = record['all_stage_counted_joint_motion']
            self.assertEqual(len(motion['events']), 2)
            self.assertTrue(motion['all_stage_actual_window_checks_passed'])
            self.assertEqual(motion['extra_camera_queries'], 0)

    def test_original_policy_failure_is_not_retried_or_fabricated_as_full_budget(self):
        class Failed(Adapter):
            def predict(self, *args):
                raise ValueError('invalid policy action')
        with tempfile.NamedTemporaryFile() as checkpoint:
            config, row = self.fixture(checkpoint.name)
            env = Environment()
            with patch.object(baseline, 'build_policy', return_value=object()), \
                    patch.object(baseline, 'AlignedLocalSimulation', return_value=env), \
                    patch.object(baseline, 'PhysManiAdapter', Failed), \
                    patch.object(baseline, 'public_initial_conditions', return_value={}), \
                    patch('scripts.collect_local_rollouts.initialization', return_value=(0, 0, 0, 0)):
                record = baseline.run_TRAIN_baseline(ROOT, row, config)
            self.assertTrue(env.closed)
            self.assertFalse(record['official_task_success'])
            self.assertFalse(record['budget_fully_consumed_or_native_success'])
            self.assertIn('invalid policy action', record['policy_failure'])
            self.assertEqual(record['counted_clock']['ticks'], 0)


if __name__ == '__main__':
    unittest.main()
