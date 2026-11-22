"""Real-inventory CPU boundaries for task-local formal evaluation; no simulator."""
import ast
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts import run_task_formal_case_r10 as case
from scripts import run_task_formal_queue_r10 as queue
from our_mani.multitask.protocol import MultitaskProtocolError

ROOT = Path(__file__).resolve().parents[1]
TASK = 'moving_basketball_in_hoop'


class FormalTaskBoundaries(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.val = json.loads((ROOT / 'artifacts/framework/protocol/validation_manifest.json').read_text())
        cls.test = json.loads((ROOT / 'artifacts/framework/multitask_r10/official_test_manifest.json').read_text())

    def test_real_VAL_inventory_selects_all_twenty_original_parents(self):
        rows = queue.rows_for(self.val, 'validation', TASK)
        self.assertEqual(len(rows), 20)
        self.assertEqual(rows, [row for row in self.val['episodes'] if row['task'] == TASK])
        self.assertEqual(sum(row['task'] in queue.TASK_SPECS for row in self.val['episodes']), 275)
        self.assertEqual(len(self.val['episodes']), 315)

    def test_real_TEST_inventory_remains_full560_while_task_selects40(self):
        before = deepcopy(self.test)
        rows = queue.rows_for(self.test, 'test', TASK)
        self.assertEqual(len(rows), 40)
        self.assertEqual(self.test, before)
        self.assertEqual(len(self.test['episodes']), 560)

    def test_missing_unrelated_TEST_parent_rejects_even_when_this_task_has40(self):
        broken = deepcopy(self.test)
        other = next(index for index, row in enumerate(broken['episodes']) if row['task'] != TASK)
        broken['episodes'].pop(other)
        with self.assertRaises(MultitaskProtocolError):
            queue.rows_for(broken, 'test', TASK)

    def test_duplicate_TEST_source_rejects_correct_total560(self):
        broken = deepcopy(self.test)
        broken['episodes'][-1] = deepcopy(broken['episodes'][-2])
        with self.assertRaises(MultitaskProtocolError):
            queue.rows_for(broken, 'test', TASK)

    def test_each_method_records_only_fixed_first_TEST_source(self):
        rows = queue.rows_for(self.test, 'test', TASK)
        jobs = queue.build_jobs(rows, 'test', queue.OURS, 'a' * 64, 'b' * 64,
            replicas=1, explore_first=False, record_demos=True, seed=0)
        self.assertEqual(len(jobs), 160)
        demos = [job for job in jobs if job['record_demo']]
        self.assertEqual(len(demos), 4)
        self.assertEqual({job['identity']['method'] for job in demos}, set(queue.OURS))
        self.assertEqual({job['source_raw_episode_id'] for job in demos}, {rows[0]['raw_episode_id']})
        self.assertTrue(all(job['exploration_seed'] is None and job['replica'] == 0 for job in jobs))

    def test_formal_VAL_has_no_demos_or_TRAIN_exploration(self):
        rows = queue.rows_for(self.val, 'validation', TASK)
        jobs = queue.build_jobs(rows, 'validation', queue.OURS, 'a' * 64, 'b' * 64,
            replicas=1, explore_first=False, record_demos=False, seed=0)
        self.assertEqual(len(jobs), 80)
        self.assertTrue(all(not job['record_demo'] and job['exploration_seed'] is None for job in jobs))

    def test_formal_wrapper_preserves_actual_TRAIN_actor_preflight(self):
        def functions(path):
            tree = ast.parse(path.read_text())
            return {node.name:ast.dump(node, include_attributes=False) for node in tree.body
                if isinstance(node, ast.FunctionDef)}
        trained = functions(ROOT / 'scripts/run_trained_task_case_v2_r10.py')
        formal = functions(ROOT / 'scripts/run_task_formal_case_r10.py')
        for name in ('actor_model_preflight', 'verify_value_query_heads'):
            self.assertEqual(trained[name], formal[name])

    def test_formal_entrypoint_rejects_TRAIN_before_loading_or_launching(self):
        args = SimpleNamespace(method='context_geometry', split='train', demo_only=False,
            development_demo=False, record_demo=False)
        with patch.object(case.evaluator, 'preflight', side_effect=AssertionError('must not run')):
            with self.assertRaises(MultitaskProtocolError):
                case.preflight(args)

    def test_formal_entrypoint_rejects_demo_only_and_VAL_video(self):
        for values in ({'demo_only':True, 'record_demo':False}, {'demo_only':False, 'record_demo':True}):
            args = SimpleNamespace(method='cv_geometry', split='validation', development_demo=False, **values)
            with patch.object(case.evaluator, 'preflight', side_effect=AssertionError('must not run')):
                with self.assertRaises(MultitaskProtocolError):
                    case.preflight(args)


if __name__ == '__main__':
    unittest.main()
