import json
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from our_mani.multitask import joint16_block_support_episode_v316 as consumer
from our_mani.multitask.block_clip_brake_v315 import binding
from our_mani.multitask.evaluation import file_sha

ROOT = Path(__file__).resolve().parents[1]


class Tests(unittest.TestCase):
    def fixture(self):
        from scripts import evaluate_aligned_r09 as evaluator
        roster = json.loads((ROOT / 'artifacts/framework/protocol/train_manifest.json').read_text())['episodes']
        row = next(r for r in roster if r['task'] == 'pick_moving_target_on_the_table')
        config = dict(dt=.05, simulation_seconds=10., global_joint_acceleration=6.,
            warmup_ticks=5, block_clip_brake_v315=binding(ROOT))
        return evaluator, row, config

    def test_actual_consumer_finishes_receipt_and_restores_real_scopes(self):
        evaluator, row, config = self.fixture()
        tracker, factory, context = evaluator.RedBoxTracker, evaluator.CandidateInterception, evaluator.ContextLatentDynamics
        def original_run(boundary, actual_row, actual_config, **kwargs):
            self.assertIs(actual_row, row)
            self.assertEqual(kwargs, dict(predictor='cv', ranking='geometry', official=False, video=False))
            self.assertIsNot(evaluator.RedBoxTracker, tracker)
            self.assertIsNot(evaluator.CandidateInterception, factory)
            return dict(official_task_success=False, controller_sha256='a' * 64), []
        with patch.object(evaluator, 'run_episode', original_run), patch.object(consumer, 'finish_counted_budget') as finish:
            record, frames = consumer.run_TRAIN_development_episode(
                NS(time=10.), row, config, method='cv_geometry', root=ROOT)
            finish.assert_called_once()
        self.assertIs(evaluator.RedBoxTracker, tracker)
        self.assertIs(evaluator.CandidateInterception, factory)
        self.assertIs(evaluator.ContextLatentDynamics, context)
        self.assertFalse(frames)
        self.assertTrue(record['task_budget_fully_consumed_or_native_success'])
        self.assertFalse(record['formal_validation_metric'])
        sources = record['joint16_block_execution']['sources_sha256']
        own = 'our_mani/multitask/joint16_block_support_episode_v316.py'
        self.assertEqual(sources[own], file_sha(ROOT / own))
        self.assertTrue(all((ROOT / name).is_file() for name in sources))

    def test_wrong_policy_binding_rejected_before_original_episode(self):
        evaluator, row, config = self.fixture()
        config['block_clip_brake_v315'] = None
        with patch.object(evaluator, 'run_episode') as run:
            with self.assertRaisesRegex(ValueError, 'freeze actual held'):
                consumer.run_TRAIN_development_episode(NS(time=0.), row, config, method='cv_geometry', root=ROOT)
            run.assert_not_called()


if __name__ == '__main__':
    unittest.main()
