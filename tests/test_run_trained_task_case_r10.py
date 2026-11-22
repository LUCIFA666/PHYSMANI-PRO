"""Trained-task wrapper boundaries; original run_case executes CPU mocks."""
from argparse import Namespace
from contextlib import contextmanager
from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts import run_trained_task_case_r10 as w
from tests.test_evaluate_multitask_case_r10 import MockBackend
from our_mani.multitask.evaluation import case_identity, digest, file_sha, validate_completed_case
from our_mani.multitask.protocol import MultitaskProtocolError


class TrainedTaskWrapperTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=w.evaluator.BASE)
        self.root = Path(self.temp.name)

    def tearDown(self): self.temp.cleanup()

    def put(self, name, value):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
        return path

    def prepared(self, development=False, split='train'):
        row = dict(task='moving_basketball_in_hoop', raw_episode_id='episode104', path='/readonly/source')
        config = self.put('config.json', dict(seed=3))
        manifest = self.put('manifest.json', dict(split=split, episodes=[row]))
        files = {str(p):file_sha(p) for p in (config, manifest, Path(w.evaluator.__file__), Path(w.__file__))}
        lock = self.put('lock.json', dict(files=files))
        identity = case_identity(split=split, task=row['task'],
            raw_episode_id=row['raw_episode_id'] + '__policy_aug_r0', method='cv_geometry', source_row=row,
            config_sha256=file_sha(config), inputs_sha256=file_sha(lock),
            parent_raw_episode_id=row['raw_episode_id'], replica=0, demo_only=development)
        if development:
            identity.update(development_demo_only=True, fitting_source=False, formal_validation_metric=False)
            identity.pop('sha256'); identity['sha256'] = digest(identity)
        return dict(args=Namespace(method='cv_geometry', split=split, task=row['task'],
            raw_episode_id=row['raw_episode_id'], replica=0, exploration_seed=None,
            demo_only=development, record_demo=False, development_demo=development),
            config=read(config), config_path=config, manifest_path=manifest,
            manifest_sha256=file_sha(manifest), row=row, original_row=row,
            out=self.root/'case', identity=identity, files=files, lock_path=lock,
            lock_sha256=file_sha(lock), r09_release={}, r10_validation_release=None,
            record_demo=development, shared_hook_plan=dict(protocol={}, inputs=files),
            original_source_manifest=w.pointer(manifest))

    def hooks(self, backend, *, complete=True):
        @contextmanager
        def scope(plan, captures):
            backend.env.reset_recovery_receipt = dict(complete=complete, source_resampled=False)
            captures['simulations'].append(backend.env)
            yield
        return scope

    def test_original_run_case_preserved_with_actual_public_actor_boundary_and_hook_receipt(self):
        p = self.prepared(); backend = MockBackend()
        record = w.run_prepared(p, backend=backend, hooks=self.hooks(backend))
        base = read(p['out']/'evaluate_run_case_receipt.json')
        self.assertEqual(base, {k:v for k,v in record.items() if k != 'task_execution_witness'})
        self.assertEqual(record['entrypoint_sha256'], file_sha(w.evaluator.__file__))
        self.assertEqual(record['task_execution_witness']['actual_wrapper'], w.pointer(w.__file__))
        self.assertTrue(backend.actor_instance.assert_public)
        self.assertEqual(backend.env.ledger.ticks, 3)
        self.assertTrue(backend.env.closed)
        hook = read(p['out']/'trained_task_hook_receipt.json')
        self.assertTrue(hook['complete'])
        self.assertFalse(hook['automatic_retry'])
        self.assertTrue(validate_completed_case(record, p['identity'])['passed'])

    def test_development_video_copies_existing_frames_after_cleanup_and_cannot_enter_formal_metrics(self):
        p = self.prepared(development=True); backend = MockBackend(); frames = []
        def encode(path, images):
            self.assertTrue(backend.env.closed)
            frames.extend(int(image[0,0,0]) for image in images)
            return dict(path='CPU_mock_only.mp4', decoded_frame_count=len(images))
        with patch.object(w.evaluator, 'save_mp4', side_effect=encode):
            record = w.run_prepared(p, backend=backend, hooks=self.hooks(backend))
        self.assertEqual(frames, [0,1,2,3])
        self.assertEqual(backend.env.observe_calls, 4)
        self.assertEqual(backend.env.close_calls, 1)
        self.assertTrue(record['demo_only'])
        self.assertFalse(record['formal_test_used'])
        self.assertFalse(record['recorded_video_in_formal_metrics'])
        self.assertFalse(record['development_demo_execution']['fitting_source'])
        ordinary = dict(p['identity'], demo_only=False)
        with self.assertRaises(MultitaskProtocolError): validate_completed_case(record, ordinary)

    def test_native_error_keeps_failed_hook_and_censored_attempts_without_wrapper_commit(self):
        p = self.prepared(); backend = MockBackend(); backend.fail_dispatch = True
        with self.assertRaisesRegex(RuntimeError, 'physics backend failed'):
            w.run_prepared(p, backend=backend, hooks=self.hooks(backend))
        self.assertFalse((p['out']/'case.json').exists())
        self.assertFalse((p['out']/'evaluate_run_case_receipt.json').exists())
        hook = read(p['out']/'trained_task_hook_receipt.json')
        self.assertFalse(hook['complete']); self.assertFalse(hook['policy_failure_label_assigned'])
        raw = read(p['out']/'infrastructure_error.json')
        self.assertIsNone(raw['continuation_attempts'][0]['official_task_success'])
        self.assertTrue(raw['continuation_attempts'][0]['censored'])

    def test_missing_actual_reset_witness_never_creates_a_wrapper_commit(self):
        p = self.prepared(); backend = MockBackend()
        with self.assertRaisesRegex(MultitaskProtocolError, 'physical reset/RML hooks'):
            w.run_prepared(p, backend=backend, hooks=self.hooks(backend, complete=False))
        self.assertFalse((p['out']/'evaluate_run_case_receipt.json').exists())
        self.assertNotIn('task_execution_witness', read(p['out']/'case.json'))

    def test_development_video_and_unqualified_method_rejected_before_original_preflight(self):
        args = Namespace(method='context_geometry', split='validation', development_demo=True,
            record_demo=False, demo_only=False)
        with patch.object(w.evaluator, 'preflight', side_effect=AssertionError('not reached')):
            with self.assertRaisesRegex(MultitaskProtocolError, 'TRAIN-only'): w.preflight(args)
        args.method='physmani'; args.development_demo=False
        with patch.object(w.evaluator, 'preflight', side_effect=AssertionError('not reached')):
            with self.assertRaisesRegex(MultitaskProtocolError, 'four ours'): w.preflight(args)

    def test_development_source_row_must_match_the_real_original_manifest(self):
        p = self.prepared(); source = self.put('original.json', dict(split='train', episodes=[p['row']]))
        p['files'][str(source)] = file_sha(source)
        with patch.object(w, 'source_manifest_for', return_value=source):
            self.assertEqual(w.verify_development_source(p)[0], source)
            p['row'] = dict(p['row'], path='/a/different/source')
            with self.assertRaisesRegex(MultitaskProtocolError, 'differs from original'):
                w.verify_development_source(p)


read = lambda path: json.loads(Path(path).read_text())


if __name__ == '__main__': unittest.main()
