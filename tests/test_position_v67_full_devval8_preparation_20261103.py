"""Preparation-only V67 full8 identity and no-execution contracts."""
import ast
from copy import deepcopy
import importlib.util
import inspect
import io
import json
from pathlib import Path
from contextlib import redirect_stdout, redirect_stderr
import textwrap
import unittest
from unittest.mock import patch

from our_mani.local_mpc_20261103 import position_v67_devval8_worker as subject
from our_mani.local_mpc_20261103 import position_v67_worker as frozen

ROOT = Path(__file__).resolve().parents[1]


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, ROOT/path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def tree(function):
    return ast.dump(ast.parse(textwrap.dedent(inspect.getsource(function))), include_attributes=False)


class Full8Preparation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = json.loads(subject.CANARY_PLAN.read_text())
        cls.host = module('scripts/launch_local_mpc_position_v67_devval8_20261103.py', 'v67_full8_new_host')
        cls.queue = module('scripts/run_position_v67_full_devval8_20261103.py', 'v67_full8_new_queue')

    def test_all_eight_exact_old_development_assignments_and_reference_numbers(self):
        self.assertEqual(tuple(j['id'] for j in self.plan['jobs']), subject.CASE_IDS)
        self.assertEqual([j['seed'] for j in self.plan['jobs']], list(range(261103100, 261103108)))
        self.assertTrue(self.plan['preparation_only'])
        self.assertFalse(self.plan['execution_authorized'])
        for job in self.plan['jobs']:
            config = json.loads((ROOT/job['config']).read_text())
            original, _ = subject._reference(config)
            self.assertEqual((job['id'], job['scene'], job['seed']),
                             (original['id'], original['scene'], original['seed']))
            report = subject.check_config(ROOT/job['config'], job['scene'])
            self.assertFalse(report['ready'])
            self.assertEqual(report['model_forward_calls'], 0)
            self.assertEqual(set(report['missing_admission']),
                             {'final_source_freeze_and_external_SHA256', 'final_case_config'})
            changed = deepcopy(config); changed['seed'] += 1
            with self.assertRaises(ValueError):
                subject._reference(changed)

    def test_execute_and_physical_recording_exact_current_frozen_v67(self):
        self.assertEqual(tree(subject.execute), tree(frozen.execute))
        self.assertEqual(subject.original.make_backend, frozen.original.make_backend)
        self.assertEqual(subject.original.make_executor, frozen.original.make_executor)
        self.assertEqual(subject.original.append_ticks, frozen.original.append_ticks)
        self.assertEqual(len(subject.OWN_FILES), 5)
        self.assertTrue(all('devval8' in name for name in subject.OWN_FILES))

    def test_default_host_queue_have_no_mkdir_child_or_Native(self):
        job = self.plan['jobs'][0]
        with patch.object(Path, 'mkdir', side_effect=AssertionError('no mkdir in preparation')), \
             patch.object(self.host.subprocess, 'Popen', side_effect=AssertionError('no child')), \
             patch.object(self.queue.subprocess, 'run', side_effect=AssertionError('no child')), \
             patch.object(subject.original, 'make_backend', side_effect=AssertionError('no Native')), \
             redirect_stdout(io.StringIO()):
            self.assertEqual(self.host.main(['--config', str(ROOT/job['config']), '--scene', job['scene']]), 0)
            self.assertEqual(self.queue.main([]), 0)

    def test_nonadmitted_execute_and_source_collection_block_before_side_effects(self):
        job = self.plan['jobs'][0]
        with patch.object(Path, 'mkdir', side_effect=AssertionError('no mkdir')), \
             patch.object(self.queue.subprocess, 'run', side_effect=AssertionError('no child')), \
             patch.object(self.host.subprocess, 'Popen', side_effect=AssertionError('no child')), \
             patch.object(subject, '_source_closure', side_effect=AssertionError('no Runtime/source collection')), \
             redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                self.queue.main(['--execute', '--acknowledge-evaluation-only'])
            with self.assertRaises(ValueError):
                self.host.main(['--config', str(ROOT/job['config']), '--scene', job['scene'],
                                '--output', str(ROOT/job['path']), '--execute'])
            with self.assertRaises(ValueError):
                subject.check_config(ROOT/job['config'], job['scene'], collect_sources=True)


if __name__ == '__main__':
    unittest.main()
