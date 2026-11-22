"""Narrow CPU fakes for 829 lifecycle and existing-CPU identity boundaries."""
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import types
import unittest
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[1]/'scripts/run_frozen_environment_TRAIN_canary_v829_r10.py'
spec = importlib.util.spec_from_file_location('canary_v829_review_subject', SOURCE)
canary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(canary)


class FakeProcess:
    pid = os.getpid()
    def __init__(self, *, running, on_wait=None):
        self.running = running
        self.on_wait = on_wait
    def poll(self):
        return None if self.running else 0
    def wait(self):
        self.running = False
        if self.on_wait:
            self.on_wait()
        return 0


class CanaryReviewTests(unittest.TestCase):
    def fake_interrupt_module(self, calls):
        module = types.ModuleType('scripts.run_native_resource_owner_v19_r10')
        def interrupt(process):
            calls.append(process)
            process.running = False
        module.interrupt_owned_tree = interrupt
        return module

    def test_exception_after_Popen_interrupts_owned_fake_child(self):
        calls = []
        fake = FakeProcess(running=True)
        event = threading.Event()
        row = dict(method='cv_geometry', argv=['never-execute-this-CPU-fake'])
        original_atomic = canary.atomic
        def atomic(path, value):
            if Path(path).name == 'launch.json':
                raise OSError('injected launch write failure')
            original_atomic(path, value)
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            runtime = root/'runtime'
            with patch.multiple(canary, ROOT=root, RUNTIME=runtime), \
                    patch.object(canary.subprocess, 'Popen', return_value=fake) as launch, \
                    patch.object(canary, 'atomic', side_effect=atomic), \
                    patch.dict(sys.modules, {'scripts.run_native_resource_owner_v19_r10':self.fake_interrupt_module(calls)}):
                with self.assertRaisesRegex(OSError, 'injected launch write failure'):
                    canary.run_child(row, True, event)
            terminal = json.loads((runtime/'CPU/cv_geometry/terminal.json').read_text())
        self.assertEqual(calls, [fake])
        self.assertFalse(fake.running)
        self.assertFalse(terminal['complete'])
        self.assertFalse(terminal['automatic_retry'])
        launch.assert_called_once()

    def test_STOP_after_already_exited_fake_child_cannot_complete(self):
        calls = []
        event = threading.Event()
        fake = FakeProcess(running=False, on_wait=event.set)
        row = dict(method='cv_geometry', argv=['never-execute-this-CPU-fake'])
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            runtime = root/'runtime'
            with patch.multiple(canary, ROOT=root, RUNTIME=runtime), \
                    patch.object(canary.subprocess, 'Popen', return_value=fake), \
                    patch.dict(sys.modules, {'scripts.run_native_resource_owner_v19_r10':self.fake_interrupt_module(calls)}):
                with self.assertRaisesRegex(RuntimeError, 'canary failed'):
                    canary.run_child(row, True, event)
            terminal = json.loads((runtime/'CPU/cv_geometry/terminal.json').read_text())
            self.assertFalse((runtime/'CPU/cv_geometry/CPU_preflight.json').exists())
        self.assertFalse(terminal['complete'])
        self.assertTrue(terminal['interrupted'])
        self.assertEqual(calls, [])

    def receipt_fixture(self, root):
        (root/'config.json').write_text('{}')
        (root/'lock.json').write_text('{"tiny":"fixture"}')
        identity = dict(method='cv_geometry', source_row_sha256='same-source',
            config_sha256=canary.sha(root/'config.json'), frozen_inputs_sha256=canary.sha(root/'lock.json'))
        row = dict(method='cv_geometry', expected_identity=identity,
            argv=['fake-only', '--config','config.json','--frozen-inputs','lock.json'])
        receipt = dict(complete=True, actual_split='train', native_started=False, GPU_started=False,
            identity=deepcopy(identity), component_domain=dict(purpose='train_qualification', split='train',
                method='cv_geometry', raw_episode_id='episode0', task='pick_moving_target_on_the_table'))
        return row,receipt

    def stage_fixture(self, root, runtime):
        row,receipt = self.receipt_fixture(root)
        leaf = runtime/'CPU/cv_geometry'
        canary.atomic(leaf/'launch.json',dict(argv=row['argv']+['--check'],check_only=True,native_slots=0,GPU_slots=0))
        canary.atomic(leaf/'CPU_preflight.json',receipt)
        canary.atomic(leaf/'terminal.json',dict(complete=True,returncode=0,timed_out=False,interrupted=False,
            launch=canary.pointer(leaf/'launch.json'),CPU_preflight=canary.pointer(leaf/'CPU_preflight.json')))
        return row,receipt,leaf

    def test_existing_matching_CPU_stage_passes_without_launch(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);runtime=root/'runtime'
            with patch.multiple(canary,ROOT=root,RUNTIME=runtime), patch.object(canary.subprocess,'Popen') as launch:
                row,_,_=self.stage_fixture(root,runtime)
                canary.require_CPU_stage(row)
                launch.assert_not_called()

    def test_swapped_same_source_method_receipt_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            with patch.object(canary,'ROOT',root):
                row,receipt=self.receipt_fixture(root)
                receipt['identity']['method']='context_geometry'
                with self.assertRaisesRegex(ValueError,'same-method'):
                    canary.require_CPU_receipt(row,receipt)

    def test_changed_launch_argv_is_rejected_even_with_updated_SHA_pointer(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);runtime=root/'runtime'
            with patch.multiple(canary,ROOT=root,RUNTIME=runtime):
                row,_,leaf=self.stage_fixture(root,runtime)
                launch=json.loads((leaf/'launch.json').read_text())
                launch['argv']=['different-command','--check']
                canary.atomic(leaf/'launch.json',launch)
                terminal=json.loads((leaf/'terminal.json').read_text())
                terminal['launch']=canary.pointer(leaf/'launch.json')
                canary.atomic(leaf/'terminal.json',terminal)
                with self.assertRaisesRegex(ValueError,'immutable CPU launch'):
                    canary.require_CPU_stage(row)


if __name__=='__main__':
    unittest.main()
