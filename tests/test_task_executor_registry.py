from pathlib import Path
import inspect
import json
import unittest
from our_mani.multitask.task_executor_registry import ADAPTERS, SharedTaskBackend, driver, binding
from our_mani.multitask.specs import TASK_SPECS


class Tests(unittest.TestCase):
    def test_actual_adapters_cover_exactly_the_remaining_eleven_tasks(self):
        expected = set(TASK_SPECS)-{'push_moving_button','push_moving_button_high_speed','put_rubbish_in_moving_bin'}
        self.assertEqual(set(ADAPTERS), expected)
        root = Path(__file__).resolve().parents[1]
        checkpoint = '6447ac755032b022d18884936e02a90d8ec2f2b219b5d42fc140cdd1618cd017'
        for task, route in ADAPTERS.items():
            with self.subTest(task=task):
                module = driver(task)
                self.assertIn('backend', inspect.signature(module.run_prepared).parameters)
                self.assertTrue(callable(module.preflight))
                self.assertTrue(callable(SharedTaskBackend(task).original.actor))
                config = json.loads((root/'artifacts/framework/multitask_r10'/route.setup/'configuration.json').read_text())
                self.assertEqual(config['context_checkpoint']['sha256'], checkpoint)
                self.assertEqual(binding(root, config, task)['task'], task)


if __name__ == '__main__':
    unittest.main()
