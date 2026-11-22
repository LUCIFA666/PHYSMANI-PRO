"""Frozen entrypoint and scoped guard tests, without native/GPU/data access."""
from contextlib import ExitStack, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts import run_formal_case_admission_v719_r10 as driver
from our_mani.multitask.evaluation import file_sha
from our_mani.multitask.protocol import MultitaskProtocolError


class DriverFixture:
    """Actual code bytes and synthetic suite files, all under one temporary root."""

    def __init__(self, root):
        self.root = root
        self.files = {}
        self.module_paths = {}
        for name, module in (
            ('driver', driver), ('guard', driver.guard),
            ('worker', driver.worker), ('runtime', driver.worker.runtime)):
            relative = ('our_mani/multitask/' if name == 'guard' else 'scripts/') + Path(module.__file__).name
            pointer = self.write(relative, Path(module.__file__).read_bytes())
            self.module_paths[name] = self.root / pointer['path']
        self.route = dict(task='move_block', method='context_geometry',
            worker=self.pointer(self.module_paths['worker']),
            consumer=self.pointer(self.module_paths['runtime']))
        self.routes = dict(routes=[self.route])
        self.suite = dict(
            formal_case_driver=self.pointer(self.module_paths['driver']),
            formal_admission_guard=self.pointer(self.module_paths['guard']))
        self.refresh()

    def pointer(self, path):
        return dict(path=str(path.relative_to(self.root)), sha256=file_sha(path))

    def write(self, relative, value):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value if isinstance(value, bytes) else json.dumps(value).encode())
        pointer = self.pointer(path)
        self.files[str(path)] = pointer['sha256']
        return pointer

    def refresh(self):
        self.suite['routes'] = self.write('routes.json', self.routes)
        self.write('suite.json', self.suite)
        (self.root / 'suite_lock.json').write_text(json.dumps(dict(files=self.files)))

    def argv(self, check=False):
        result = ['--split', 'validation', '--suite', 'suite.json',
            '--suite-lock', 'suite_lock.json', '--task', self.route['task'],
            '--method', self.route['method'], '--config', 'unused_config.json',
            '--manifest', 'unused_VAL_manifest.json', '--frozen-inputs', 'unused_inputs.json',
            '--raw-episode-id', 'fixed-parent-0', '--out', 'result',
            '--source-inputs', 'unused_source_lock.json']
        return result + ['--check'] if check else result

    def scoped_modules(self):
        stack = ExitStack()
        stack.enter_context(patch.object(driver, 'ROOT', self.root))
        stack.enter_context(patch.object(driver.worker, 'ROOT', self.root))
        for name, module in (
            ('driver', driver), ('guard', driver.guard),
            ('worker', driver.worker), ('runtime', driver.worker.runtime)):
            stack.enter_context(patch.object(module, '__file__', str(self.module_paths[name])))
        return stack


class FormalAdmissionDriverTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.fixture = DriverFixture(Path(temporary.name))
        scopes = self.fixture.scoped_modules()
        self.addCleanup(scopes.close)

    def run_driver(self, *, check=False):
        output = io.StringIO()
        with patch.object(driver.sys, 'argv', ['v719'] + self.fixture.argv(check)), redirect_stdout(output):
            driver.main()
        return output.getvalue()

    def assert_guard_restored(self, original_admit, original_read):
        self.assertIs(driver.original_guard.admit, original_admit)
        self.assertIs(driver.original_guard.read_formal_source, original_read)

    def test_real_bindings_require_explicit_frozen_driver_guard_worker_and_runtime(self):
        expected, split = driver.bindings(self.fixture.argv())
        self.assertEqual(split, 'validation')
        self.assertEqual(expected['admission_driver'], self.fixture.suite['formal_case_driver'])
        self.assertEqual(expected['admission_guard'], self.fixture.suite['formal_admission_guard'])
        self.assertEqual(expected['unchanged_case_worker'], self.fixture.route['worker'])
        self.assertEqual(expected['native_runtime'], self.fixture.route['consumer'])

    def test_replacement_driver_or_guard_is_rejected_before_worker(self):
        for field in ('formal_case_driver', 'formal_admission_guard'):
            with self.subTest(field=field):
                original = self.fixture.suite[field]
                self.fixture.suite[field] = self.fixture.write('replacement.py', b'# replacement\n')
                self.fixture.refresh()
                with patch.object(driver.worker, 'main') as worker:
                    with self.assertRaisesRegex(ValueError, 'explicit frozen admission binding'):
                        self.run_driver(check=True)
                    worker.assert_not_called()
                self.fixture.suite[field] = original
                self.fixture.refresh()

    def test_alternate_physical_worker_or_runtime_and_duplicate_route_are_rejected(self):
        for key in ('worker', 'consumer'):
            with self.subTest(key=key):
                original = self.fixture.route[key]
                self.fixture.route[key] = self.fixture.write('replacement.py', b'# replacement\n')
                self.fixture.refresh()
                with self.assertRaisesRegex(ValueError, 'cannot replace the physical'):
                    driver.bindings(self.fixture.argv())
                self.fixture.route[key] = original
        self.fixture.routes['routes'].append(dict(self.fixture.route))
        self.fixture.refresh()
        with self.assertRaisesRegex(ValueError, 'one unchanged physical case route'):
            driver.bindings(self.fixture.argv())

    def test_frozen_suite_and_route_bytes_cannot_change(self):
        for name in ('suite.json', 'routes.json'):
            with self.subTest(name=name):
                path = self.fixture.root / name
                original = path.read_bytes()
                path.write_bytes(original + b' ')
                try:
                    with self.assertRaises((ValueError, MultitaskProtocolError)):
                        driver.bindings(self.fixture.argv())
                finally:
                    path.write_bytes(original)

    def test_formal_split_and_exactly_one_explicit_argument_are_required(self):
        argv = self.fixture.argv()
        argv[argv.index('--split') + 1] = 'train'
        with self.assertRaisesRegex(ValueError, 'formal-only'):
            driver.bindings(argv)
        for name in ('--suite', '--suite-lock', '--task', '--method'):
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, 'one explicit'):
                    driver.bindings(self.fixture.argv() + [name, 'duplicate'])

    def test_actual_worker_CPU_entrypoint_remains_CPU_only_and_restores_scopes(self):
        original_admit = driver.original_guard.admit
        original_read = driver.original_guard.read_formal_source
        receipt = {'purpose': 'evaluation'}
        prepared = dict(identity={'case': 'synthetic'}, component_domain=SimpleNamespace(receipt=lambda: receipt),
            model_preflight={'CPU_only': True}, source_lock_sha256='a' * 64)

        def prepare(root, args, worker_file):
            self.assertIs(driver.original_guard.admit, driver.guard.admit)
            self.assertIs(driver.original_guard.read_formal_source, driver.guard.read_formal_source)
            self.assertEqual(root, self.fixture.root)
            self.assertEqual(worker_file, self.fixture.module_paths['worker'])
            self.assertTrue(args.check)
            self.assertEqual(args.split, 'validation')
            return prepared

        # Execute the original v709.main parser and check branch. Only its
        # expensive model/data preflight boundary is replaced by a CPU fixture.
        with patch.object(driver.worker.contract, 'prepare', side_effect=prepare) as preflight, \
             patch.object(driver.worker.runtime, 'run', side_effect=AssertionError('native/GPU started')) as native:
            proof = json.loads(self.run_driver(check=True))
        preflight.assert_called_once()
        native.assert_not_called()
        self.assertIs(proof['native_started'], False)
        self.assertIs(proof['GPU_started'], False)
        self.assertEqual(proof['admission_driver'], self.fixture.suite['formal_case_driver'])
        self.assertEqual(proof['admission_guard'], self.fixture.suite['formal_admission_guard'])
        self.assertEqual(proof['unchanged_case_worker'], self.fixture.route['worker'])
        self.assertFalse((self.fixture.root / 'result').exists())
        self.assert_guard_restored(original_admit, original_read)

    def test_CPU_proof_cannot_claim_native_or_GPU_started(self):
        original_admit = driver.original_guard.admit
        original_read = driver.original_guard.read_formal_source
        for native, gpu in ((True, False), (False, True), (None, False), (False, 0)):
            with self.subTest(native=native, gpu=gpu):
                with patch.object(driver.worker, 'main', side_effect=lambda: print(json.dumps(
                        dict(native_started=native, GPU_started=gpu)))):
                    with self.assertRaisesRegex(ValueError, 'CPU-only'):
                        self.run_driver(check=True)
                self.assert_guard_restored(original_admit, original_read)
        self.assertFalse((self.fixture.root / 'result').exists())

    def test_guard_scope_is_restored_on_worker_and_JSON_failure(self):
        original_admit = driver.original_guard.admit
        original_read = driver.original_guard.read_formal_source

        def failing_worker():
            self.assertIs(driver.original_guard.admit, driver.guard.admit)
            self.assertIs(driver.original_guard.read_formal_source, driver.guard.read_formal_source)
            raise RuntimeError('preflight stopped')

        with patch.object(driver.worker, 'main', side_effect=failing_worker):
            with self.assertRaisesRegex(RuntimeError, 'preflight stopped'):
                self.run_driver(check=True)
        self.assert_guard_restored(original_admit, original_read)
        with patch.object(driver.worker, 'main', side_effect=lambda: print('invalid proof')):
            with self.assertRaises(json.JSONDecodeError):
                self.run_driver(check=True)
        self.assert_guard_restored(original_admit, original_read)

    def test_native_incomplete_cleanup_or_wrong_split_cannot_receive_receipt(self):
        out = self.fixture.root / 'result'
        out.mkdir()
        case = out / 'case.json'
        for record in (
            {}, dict(complete=False, cleanup_completed=True, split='validation'),
            dict(complete=True, cleanup_completed=False, split='validation'),
            dict(complete=True, cleanup_completed=True, split='train'),
            dict(complete=1, cleanup_completed=True, split='validation')):
            with self.subTest(record=record):
                case.write_text(json.dumps(record))
                before = case.read_bytes()
                with patch.object(driver.worker, 'main') as worker:
                    with self.assertRaisesRegex(ValueError, 'original complete actual-split'):
                        self.run_driver()
                worker.assert_called_once()
                self.assertEqual(case.read_bytes(), before)
                self.assertFalse((out / 'admission_driver_receipt.json').exists())

    def test_native_receipt_binds_original_case_SHA_without_rewriting_record(self):
        original_admit = driver.original_guard.admit
        original_read = driver.original_guard.read_formal_source
        case = self.fixture.root / 'result/case.json'
        original_record = b'{"complete": true, "cleanup_completed": true, "split": "validation", "native_success": false}\n'

        def worker_fixture():
            self.assertIs(driver.original_guard.admit, driver.guard.admit)
            self.assertIs(driver.original_guard.read_formal_source, driver.guard.read_formal_source)
            case.parent.mkdir()
            case.write_bytes(original_record)

        with patch.object(driver.worker, 'main', side_effect=worker_fixture):
            self.run_driver()
        proof = json.loads((case.parent / 'admission_driver_receipt.json').read_text())
        self.assertEqual(case.read_bytes(), original_record)
        self.assertEqual(proof['case'], self.fixture.pointer(case))
        self.assertIs(proof['physical_policy_changed'], False)
        self.assertEqual(proof['unchanged_case_worker'], self.fixture.route['worker'])
        self.assertEqual(proof['native_runtime'], self.fixture.route['consumer'])
        self.assert_guard_restored(original_admit, original_read)
        # A receipt is append-only evidence and cannot silently be replaced.
        with patch.object(driver.worker, 'main'):
            with self.assertRaises(FileExistsError):
                self.run_driver()


if __name__ == '__main__':
    unittest.main()
