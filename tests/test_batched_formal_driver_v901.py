"""CPU-only fake controls and lookup scopes; no original worker import/run."""
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[1] / 'scripts/run_batched_formal_driver_v901_r10.py'
spec = importlib.util.spec_from_file_location('batch_bootstrap_v901_test', SOURCE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class ControlTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / 'fake_source.py'
        self.source.write_text('# immutable CPU fixture\n')
        self.setup = self.root / 'artifacts/framework/multitask_r10/fake_setup'
        self.setup.mkdir(parents=True)
        self.control = self.setup / 'control.json'
        self.lock = self.setup / 'lock.json'
        self.argv = ['--split', 'test', '--manifest', 'manifest.json', '--suite', 'suite.json',
                     '--suite-lock', 'suite_lock.json', '--source-inputs', 'sources.json',
                     '--validation-release', 'release.json', '--out', 'fresh']
        self.data = dict(schema=module.SCHEMA, execution_enabled=True, entry='formal',
                         scientific_policy_changed=False, models_or_training_changed=False,
                         argv_bindings={self.argv[i]: self.argv[i + 1] for i in range(0, 12, 2)})
        self.root_patch = patch.object(module, 'ROOT', self.root)
        self.sources_patch = patch.object(module, 'SOURCES', ('fake_source.py',))
        self.root_patch.start(); self.sources_patch.start()
        self.addCleanup(self.sources_patch.stop); self.addCleanup(self.root_patch.stop)

    def freeze(self, extra=None):
        self.control.write_text(json.dumps(self.data))
        files = {str(self.control): sha(self.control), str(self.source): sha(self.source)}
        files.update(extra or {})
        self.lock.write_text(json.dumps(dict(schema=module.LOCK_SCHEMA, execution_enabled=True, files=files)))
        return sha(self.lock)

    def load(self, pinned=None, entry='formal'):
        return module.load_control(str(self.control), str(self.lock), pinned or sha(self.lock), entry, self.argv)

    def test_explicit_control_binds_actual_source_and_entry(self):
        self.freeze()
        proof = self.load()
        self.assertEqual(proof['control']['sha256'], sha(self.control))
        self.assertEqual(proof['original_entry'], 'scripts/run_formal_case_author_outcome_v751_r10.py')

    def test_disabled_control_rejected(self):
        self.data['execution_enabled'] = False
        self.freeze()
        with self.assertRaises(ValueError): self.load()

    def test_missing_enabled_control_rejected(self):
        del self.data['execution_enabled']
        self.freeze()
        with self.assertRaises(ValueError): self.load()

    def test_unpinned_lock_rejected(self):
        self.freeze()
        with self.assertRaises(ValueError): self.load('0' * 64)

    def test_source_changed_after_freeze_rejected(self):
        self.freeze(); self.source.write_text('# replaced\n')
        with self.assertRaises(ValueError): self.load()

    def test_control_changed_after_freeze_rejected(self):
        self.freeze(); self.control.write_text('{}')
        with self.assertRaises(ValueError): self.load()

    def test_canonical_duplicate_rejected(self):
        alias = self.root / 'alias.py'; alias.symlink_to(self.source)
        self.freeze({str(alias): sha(self.source)})
        with self.assertRaises(ValueError): self.load()

    def test_unknown_or_unbound_scope_rejected(self):
        self.data['argv_bindings']['--foo'] = 'x'; self.freeze()
        with self.assertRaises(ValueError): self.load()

    def test_missing_required_scope_rejected(self):
        del self.data['argv_bindings']['--manifest']; self.freeze()
        with self.assertRaises(ValueError): self.load()

    def test_changed_forwarded_manifest_rejected(self):
        self.freeze(); self.argv[self.argv.index('--manifest') + 1] = 'other.json'
        with self.assertRaises(ValueError): self.load()

    def test_duplicate_forwarded_option_rejected(self):
        self.freeze(); self.argv += ['--split', 'test']
        with self.assertRaises(ValueError): self.load()

    def test_formal_train_or_development_flag_rejected(self):
        for addition in (['--development-demo'],):
            self.freeze(); self.argv += addition
            with self.assertRaises(ValueError): self.load()

    def test_scientific_or_model_change_rejected(self):
        for field in ('scientific_policy_changed', 'models_or_training_changed'):
            with self.subTest(field=field):
                self.data[field] = True; self.freeze()
                with self.assertRaises(ValueError): self.load()
                self.data[field] = False

    def test_train_requires_exact_frozen_case(self):
        self.argv = ['--split', 'train', '--manifest', 'train.json', '--frozen-inputs', 'base.json',
                     '--task', 'block', '--raw-episode-id', 'episode0', '--method', 'physmani',
                     '--development-demo', '--out', 'development_demo_only/fresh']
        self.data.update(entry='train', TRAIN_cases=[['block', 'episode0', 'physmani']],
                         argv_bindings={'--split': 'train', '--manifest': 'train.json', '--frozen-inputs': 'base.json'})
        self.freeze()
        self.assertEqual(self.load(entry='train')['entry'], 'train')
        self.argv[self.argv.index('--raw-episode-id') + 1] = 'episode1'
        with self.assertRaises(ValueError): self.load(entry='train')


class ScopeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.path = self.root / 'input.json'; self.path.write_text('{"a": 1}')
        self.files = {'input.json': sha(self.path)}
        self.guard = SimpleNamespace(locked_pointer=lambda *a: 'old pointer', locked_json=lambda *a: 'old JSON')
        self.contract = SimpleNamespace(_checked=lambda *a: 'old checked', locked_pointer=lambda *a: 'old pointer',
                                        locked_json=lambda *a: 'old JSON')
        self.driver = SimpleNamespace(bindings=lambda *a: self.guard.locked_json(
            self.root, self.files, dict(path='input.json', sha256=self.files['input.json'])))
        self.contract.prepare = lambda *a: self.contract._checked(self.root, self.files, self.path)
        from our_mani.multitask.frozen_lookup_batch_v900 import LookupBatch
        self.batch_type = lambda root: LookupBatch(root, cache=self.root / 'cache')
        self.root_patch = patch.object(module, 'ROOT', self.root); self.root_patch.start()
        self.addCleanup(self.root_patch.stop)

    def test_real_batch_scopes_finish_before_return_and_restore(self):
        old = (self.driver.bindings, self.contract.prepare, self.contract._checked, self.guard.locked_json)
        phases = []
        with module.scoped_calls(self.driver, self.contract, self.guard, self.batch_type, phases):
            self.assertEqual(self.driver.bindings(), {'a': 1})
            self.assertEqual(self.contract.prepare(), self.path)
        self.assertEqual(old, (self.driver.bindings, self.contract.prepare, self.contract._checked, self.guard.locked_json))
        self.assertEqual([p['phase'] for p in phases], ['original751_bindings', 'original742_prepare'])
        self.assertTrue(all(p['metrics']['closed'] and p['metrics']['complete'] for p in phases))

    def test_prepare_input_change_rejects_before_return_and_restores(self):
        def function():
            self.contract._checked(self.root, self.files, self.path)
            self.path.write_text('{"a": 2}')
            return 'must not escape'
        self.contract.prepare = function
        phases = []
        old_checked = self.contract._checked
        with self.assertRaises(Exception):
            with module.scoped_calls(self.driver, self.contract, self.guard, self.batch_type, phases):
                self.contract.prepare()
        self.assertIs(self.contract.prepare, function)
        self.assertIs(self.contract._checked, old_checked)
        self.assertEqual(phases, [])

    def test_original_error_is_not_success_and_scope_restores(self):
        def function(): raise LookupError('original rejected input')
        self.contract.prepare = function
        with self.assertRaisesRegex(LookupError, 'original rejected'):
            with module.scoped_calls(self.driver, self.contract, self.guard, self.batch_type, []):
                self.contract.prepare()
        self.assertIs(self.contract.prepare, function)

    def test_other_thread_prepare_and_bindings_follow_original_without_batch(self):
        phases, results = [], []
        old_prepare, old_bindings = self.contract.prepare, self.driver.bindings
        with module.scoped_calls(self.driver, self.contract, self.guard, self.batch_type, phases):
            thread = threading.Thread(target=lambda: results.extend([
                self.contract.prepare(), self.driver.bindings()]))
            thread.start(); thread.join(timeout=2)
            self.assertFalse(thread.is_alive())
        self.assertEqual(results, ['old checked', 'old JSON'])
        self.assertEqual(phases, [])
        self.assertIs(self.contract.prepare, old_prepare)
        self.assertIs(self.driver.bindings, old_bindings)

    def test_nested_scope_rejected_without_losing_outer_hooks(self):
        phases = []
        old = self.contract.prepare
        with module.scoped_calls(self.driver, self.contract, self.guard, self.batch_type, phases):
            outer = self.contract.prepare
            with self.assertRaisesRegex(RuntimeError, 'nested'):
                with module.scoped_calls(self.driver, self.contract, self.guard, self.batch_type, []):
                    self.fail('nested bootstrap scope must not open')
            self.assertIs(self.contract.prepare, outer)
            self.assertEqual(self.contract.prepare(), self.path)
        self.assertIs(self.contract.prepare, old)
        self.assertEqual(len(phases), 1)


class OutputTests(unittest.TestCase):
    @staticmethod
    def phases():
        return [dict(phase=name, boundary_recheck_complete=True,
                     metrics=dict(closed=True, complete=True, maps=1, pointer_checks=1))
                for name in ('original751_bindings', 'original742_prepare')]

    def test_missing_or_unclosed_preparation_cannot_publish_complete_receipt(self):
        for phases in ([], self.phases()[1:], list(reversed(self.phases()))):
            with self.assertRaises(ValueError): module.require_phases('formal', phases)
        phases = self.phases(); phases[0]['metrics']['closed'] = False
        with self.assertRaises(ValueError): module.require_phases('formal', phases)

    def test_CPU_metadata_augmentation_creates_no_case_output(self):
        fake = dict(complete=True, native_started=False, GPU_started=False, identity={'sha256': 'original'})
        proof = dict(original_entry='original751.py', entry='formal')
        argv = ['bootstrap.py', '--batch-control', 'control', '--batch-lock', 'lock',
                '--batch-lock-sha256', 'a' * 64, '--entry', 'formal', '--',
                '--split', 'test', '--out', 'never_created', '--check']
        def original(entry, forwarded, batch_type, phases):
            phases.extend(self.phases()); print(json.dumps(fake))
        capture = io.StringIO()
        with patch.object(sys, 'argv', argv), patch.object(module, 'load_control', return_value=proof), \
                patch.object(module, 'run_original', side_effect=original), contextlib.redirect_stdout(capture):
            module.main()
        output = json.loads(capture.getvalue())
        self.assertEqual(output['identity'], fake['identity'])
        self.assertTrue(output['batch_execution']['check_only'])
        self.assertEqual(output['batch_execution']['actual_outer_argv'], argv)
        self.assertEqual(output['batch_execution']['forwarded_original_argv'], ['original751.py', *argv[argv.index('--') + 1:]])
        self.assertFalse((module.ROOT / 'never_created').exists())

    def test_native_diagnostics_are_forwarded_without_JSON_parsing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); out = root / 'artifacts/framework/multitask_r10/fresh'; out.mkdir(parents=True)
            (out / 'case.json').write_text('{"complete": true}')
            (out / 'admission_driver_receipt.json').write_text('{"complete": true}')
            proof = dict(original_entry='original751.py', entry='formal')
            argv = ['bootstrap.py', '--batch-control', 'control', '--batch-lock', 'lock',
                    '--batch-lock-sha256', 'a' * 64, '--entry', 'formal', '--', '--out', str(out)]
            def original(entry, forwarded, batch_type, phases):
                phases.extend(self.phases()); print('RML diagnostic\n{"complete": true}\nCoppeliaSim done.')
            capture = io.StringIO()
            with patch.object(module, 'ROOT', root), patch.object(sys, 'argv', argv), \
                    patch.object(module, 'load_control', return_value=proof), \
                    patch.object(module, 'run_original', side_effect=original), contextlib.redirect_stdout(capture):
                module.main()
            self.assertIn('RML diagnostic', capture.getvalue())
            receipt = json.loads((out / 'batch_driver_receipt.json').read_text())
            self.assertEqual(receipt['case']['sha256'], sha(out / 'case.json'))
            self.assertFalse(receipt['check_only'])


if __name__ == '__main__':
    unittest.main()
