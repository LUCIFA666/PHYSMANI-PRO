"""Three-method formal task boundaries; no simulator or cancelled method work."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts import run_geometry_task_formal_case_r10 as case
from scripts import run_geometry_task_formal_queue_r10 as queue
from our_mani.multitask.protocol import MultitaskProtocolError

ROOT = Path(__file__).resolve().parents[1]
TASK = 'moving_basketball_in_hoop'


class GeometryFormalBoundaries(unittest.TestCase):
    def test_original_VAL_inventory_gives_twenty_sources_and_forty_ours_cases(self):
        value = json.loads((ROOT / 'artifacts/framework/protocol/validation_manifest.json').read_text())
        rows = queue.rows_for(value, 'validation', TASK)
        jobs = queue.build_jobs(rows, 'validation', queue.OURS, 'a' * 64, 'b' * 64)
        self.assertEqual(len(rows), 20)
        self.assertEqual(len(jobs), 40)
        self.assertEqual({job['identity']['method'] for job in jobs}, {'cv_geometry', 'context_geometry'})
        self.assertTrue(all(not job['record_demo'] and job['exploration_seed'] is None for job in jobs))

    def test_original_TEST560_remains_intact_and_only_first_source_saves_two_demos(self):
        value = json.loads((ROOT / 'artifacts/framework/multitask_r10/official_test_manifest.json').read_text())
        original = deepcopy(value)
        rows = queue.rows_for(value, 'test', TASK)
        jobs = queue.build_jobs(rows, 'test', queue.OURS, 'a' * 64, 'b' * 64, record_demos=True)
        self.assertEqual(value, original)
        self.assertEqual(len(jobs), 80)
        demos = [job for job in jobs if job['record_demo']]
        self.assertEqual(len(demos), 2)
        self.assertEqual({job['source_raw_episode_id'] for job in demos}, {rows[0]['raw_episode_id']})
        broken = deepcopy(value); broken['episodes'].pop()
        with self.assertRaises(MultitaskProtocolError):
            queue.rows_for(broken, 'test', TASK)

    def test_disallowed_method_rejected_before_actor_loading(self):
        args = SimpleNamespace(method='not_an_enabled_method', split='validation',
            demo_only=False, development_demo=False, record_demo=False)
        with patch.object(case.evaluator, 'preflight', side_effect=AssertionError('must not load')):
            with self.assertRaises(MultitaskProtocolError):
                case.preflight(args)

    def test_stop_after_slow_preflight_prevents_Popen_and_does_not_create_attempt(self):
        with tempfile.TemporaryDirectory(dir=queue.BASE, prefix='geometry_queue_stop_CPU_') as tmp:
            base = Path(tmp); output = base / 'out'; stop = base / 'stop'
            job = dict(job_id='example', source_raw_episode_id='episode104',
                identity=dict(method='cv_geometry'), record_demo=False)
            plan = dict(output=str(output.relative_to(ROOT)), split='validation', task=TASK,
                expected_cases=1, sha256='a' * 64, jobs=[job],
                config=dict(path='config.json'), manifest=dict(path='manifest.json'),
                frozen_inputs=dict(path='lock.json'), child_source=dict(path='child.py'),
                task_execution_contract=dict(launchers=dict(ours=dict(path='launcher.sh'))))
            calls = []
            def verified(_):
                calls.append(True)
                if len(calls) > 1:
                    stop.write_text('stop during preflight')
                return plan, {}
            with patch.object(queue, 'verify_plan', side_effect=verified), patch.object(queue, 'STOP', stop), \
                    patch.object(queue.subprocess, 'Popen') as popen:
                with self.assertRaisesRegex(ValueError, 'incomplete'):
                    queue.run(base / 'plan.json', 1)
                popen.assert_not_called()
            self.assertFalse((output / 'cases' / 'example').exists())
            self.assertFalse(json.loads((output / 'status.json').read_text())['complete'])


if __name__ == '__main__':
    unittest.main()
