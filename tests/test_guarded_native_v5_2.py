"""Actual failed-display and duplicate-launch boundaries; no native simulator."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch, Mock

from scripts import native_display_guard_r10 as guard
from scripts import run_native_actuator_diagnostic_pair_v5_2_r10 as parent


class DisplayTests(unittest.TestCase):
    def test_failed_display_is_not_ready_and_does_not_claim_sensor_validation(self):
        opener = Mock(return_value=False)
        value = guard.readiness({'CUDA_VISIBLE_DEVICES': '', 'DISPLAY': ':99'}, opener)
        opener.assert_called_once_with()
        self.assertFalse(value['ready'])
        self.assertEqual(value['reason'], 'X11_connection_failed')
        self.assertFalse(value['policy_started'])
        self.assertFalse(value['rendered_sensor_validated'])

    def test_offscreen_CPU_and_missing_display_reject_before_connection(self):
        cases = [dict(CUDA_VISIBLE_DEVICES='', DISPLAY=':99', QT_QPA_PLATFORM='offscreen'),
            dict(DISPLAY=':99'), dict(CUDA_VISIBLE_DEVICES='')]
        for environment in cases:
            opener = Mock()
            self.assertFalse(guard.readiness(environment, opener)['ready'])
            opener.assert_not_called()

    def test_X11_connection_success_still_does_not_claim_RGBD_or_physics(self):
        value = guard.readiness({'CUDA_VISIBLE_DEVICES': '', 'DISPLAY': ':99'}, lambda: True)
        self.assertTrue(value['ready'])
        self.assertFalse(value['physics_started'])
        self.assertFalse(value['rendered_sensor_validated'])

    def test_missing_library_or_query_error_is_unavailable_not_zero_success(self):
        def failed():
            raise RuntimeError('no libX11')
        value = guard.readiness({'CUDA_VISIBLE_DEVICES': '', 'DISPLAY': ':99'}, failed)
        self.assertFalse(value['ready'])
        self.assertEqual(value['error_type'], 'RuntimeError')

    def test_failed_display_records_receipt_before_worker_can_execute(self):
        with TemporaryDirectory(dir=parent.ROOT / 'runtime/framework/tmp') as temporary:
            raw = Path(temporary)
            value = dict(ready=False, reason='X11_connection_failed', policy_started=False)
            with patch.object(guard, 'paths', return_value=(Path('/fake/worker.py'), raw)), \
                    patch.object(guard, 'readiness', return_value=value), patch.object(guard.os, 'execv') as execute:
                self.assertEqual(guard.main(['worker.py', '--out', 'out']), 78)
                execute.assert_not_called()
            self.assertEqual(json.loads((raw / 'display_readiness.json').read_text()), value)

    def test_ready_display_preserves_the_exact_worker_arguments(self):
        with TemporaryDirectory(dir=parent.ROOT / 'runtime/framework/tmp') as temporary:
            with patch.object(guard, 'paths', return_value=(Path('/fake/worker.py'), Path(temporary))), \
                    patch.object(guard, 'readiness', return_value={'ready': True}), \
                    patch.object(guard.os, 'execv') as execute:
                argv = ['worker.py', '--split', 'train', '--method', 'context_geometry', '--out', 'out']
                self.assertEqual(guard.main(argv), 0)
                execute.assert_called_once_with(guard.sys.executable,
                    [guard.sys.executable, '/fake/worker.py', *argv[1:]])

    def test_foreign_worker_or_existing_namespace_is_rejected(self):
        for argv in [['foreign.py'], ['scripts/run_geometry_native_actuator_diagnostic_v5_r10.py',
                '--out', 'artifacts/framework/multitask_r10/geometry_native_actuator_diagnostic_v5/development_demo_only/cv_normal']]:
            with self.assertRaises(ValueError):
                guard.paths(argv)

    def test_shared_kernel_lock_prevents_a_second_owner_and_releases_after_exit(self):
        with TemporaryDirectory(dir=parent.ROOT / 'runtime/framework/tmp') as temporary:
            path = Path(temporary) / 'run.lock'
            with parent.exclusive_pair(path):
                with self.assertRaises(RuntimeError):
                    with parent.exclusive_pair(path):
                        self.fail('second owner entered')
            with parent.exclusive_pair(path):
                self.assertTrue(path.exists())

    def test_parent_keeps_train_pair_and_actual_new_wrapper(self):
        with patch.object(parent.original, 'SETUP', parent.SETUP), patch.object(parent.original, 'OUTPUT', parent.OUTPUT):
            for method in ('cv_geometry', 'context_geometry'):
                argv = parent.command(method)
                self.assertEqual(argv[1], parent.WRAPPER)
                self.assertEqual(argv[2], 'scripts/run_geometry_native_actuator_diagnostic_v5_r10.py')
                self.assertEqual(argv[argv.index('--split') + 1], 'train')
                self.assertEqual(argv[argv.index('--raw-episode-id') + 1], 'episode104')
                self.assertIn('geometry_native_actuator_diagnostic_v5_2/', argv[-1])
            with self.assertRaises(ValueError):
                parent.command('cv_value')

    def test_isolated_process_view_blocks_launch_before_lock_or_original_parent(self):
        with patch.object(parent, 'preflight', return_value={'process_visibility': {'host_visibility_available': False}}), \
                patch.object(parent, 'exclusive_pair') as lock, patch.object(parent.original, 'main') as launch, \
                patch.object(parent.sys, 'argv', ['parent.py']):
            with self.assertRaisesRegex(RuntimeError, 'Host process visibility'):
                parent.main()
            lock.assert_not_called()
            launch.assert_not_called()


if __name__ == '__main__':
    unittest.main()
