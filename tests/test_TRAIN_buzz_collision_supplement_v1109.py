"""Pure CPU combo ordering, exact original calls, cleanup and fixed selection."""
from contextlib import contextmanager
from types import SimpleNamespace
import os
from pathlib import Path
import unittest
from unittest.mock import patch

from scripts import diagnose_TRAIN_buzz_collision_supplement_v1109_r10 as subject


class CombinedScopeTests(unittest.TestCase):
    selection = dict(task='beat_the_rotating_buzz', raw_episode_id='episode10', method='cv_geometry')

    def scopes(self, runtime, order, *, native_enter_error=None):
        @contextmanager
        def collision(root, module, selection, sink, *, permit):
            order.append('collision_enter')
            original = runtime.operation
            def observed(*args):
                sink['original_calls'] = sink.get('original_calls', 0) + 1
                return original(*args)
            try:
                with patch.object(runtime, 'operation', observed):
                    yield
            finally:
                sink.update(restored=runtime.operation is original, observer_receipts=[{'CPU_collision': True}],
                            summary={'CPU_collision': True})
                order.append('collision_exit')
        @contextmanager
        def native(root, module, selection, sink, *, permit):
            order.append('native_enter')
            if native_enter_error:
                raise native_enter_error
            try:
                yield
            finally:
                sink.update(restored=True, observer_receipts=[{'CPU_native': True}], summary={'CPU_native': True})
                order.append('native_exit')
        return collision, native

    def test_each_original_operation_once_and_both_restore(self):
        calls = []
        token = object()
        original = lambda value: (calls.append(value), token)[1]
        runtime = SimpleNamespace(operation=original)
        order, events = [], {}
        collision, native = self.scopes(runtime, order)
        with subject.observer_context_factory('/unused', runtime, self.selection, events, permit=object(),
                                             collision_factory=collision, native_factory=native):
            self.assertIs(runtime.operation('one'), token)
        self.assertEqual(calls, ['one'])
        self.assertEqual(events['collision']['original_calls'], 1)
        self.assertEqual(order, ['collision_enter', 'native_enter', 'native_exit', 'collision_exit'])
        self.assertIs(runtime.operation, original)
        self.assertTrue(events['restored'])
        self.assertFalse(events['summary']['represents_all16_Buzz_cases'])

    def test_body_exception_same_instance_and_scopes_restore(self):
        runtime = SimpleNamespace(operation=lambda: None)
        original = runtime.operation
        order, events = [], {}
        collision, native = self.scopes(runtime, order)
        error = RuntimeError('original body')
        with self.assertRaises(RuntimeError) as caught:
            with subject.observer_context_factory('/unused', runtime, self.selection, events, permit=object(),
                                                 collision_factory=collision, native_factory=native):
                raise error
        self.assertIs(caught.exception, error)
        self.assertIs(runtime.operation, original)
        self.assertTrue(events['restored'])

    def test_native_entry_failure_still_restores_collision_and_no_false_success(self):
        runtime = SimpleNamespace(operation=lambda: None)
        original = runtime.operation
        order, events = [], {}
        error = RuntimeError('native observer refuses entry')
        collision, native = self.scopes(runtime, order, native_enter_error=error)
        with self.assertRaises(RuntimeError) as caught:
            with subject.observer_context_factory('/unused', runtime, self.selection, events, permit=object(),
                                                 collision_factory=collision, native_factory=native):
                self.fail('scope should not enter')
        self.assertIs(caught.exception, error)
        self.assertIs(runtime.operation, original)
        self.assertTrue(events['collision']['restored'])
        self.assertFalse(events['restored'])

    def test_exact_four_representatives(self):
        for task, raw in subject.REPRESENTATIVES:
            for method in subject.METHODS:
                subject.validate_selection(dict(task=task, raw_episode_id=raw, method=method))
        for change in (dict(raw_episode_id='episode104'), dict(method='physmani'), dict(task='insert_onto_rotating_peg')):
            selection = dict(self.selection)
            selection.update(change)
            with self.assertRaises(ValueError):
                subject.validate_selection(selection)


class Prior48CompletionTests(unittest.TestCase):
    def metadata(self):
        return dict(
            terminal=dict(complete=True, completed=48, infrastructure_failed=0, automatic_retry=False),
            status=dict(stage='complete', expected=48, completed=[dict(job_id=str(i)) for i in range(48)],
                        failed=[], active=None),
            launch=dict(plan=dict(path=subject.MAIN_PLAN, sha256=subject.MAIN_PLAN_SHA),
                        pid=999999999, start_ticks=1))

    def pointers(self, metadata):
        def read(root, name):
            kind = Path(name).stem
            return dict(path=name, sha256='a' * 64), metadata[kind]
        return read

    def test_complete_count_does_not_authorize_an_active_owner(self):
        metadata = self.metadata()
        fields = Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()
        metadata['launch'].update(pid=os.getpid(), start_ticks=int(fields[19]))
        with patch.object(subject, '_raw', return_value=b'CPU_only'), \
             patch.object(subject, '_pointer', side_effect=self.pointers(metadata)):
            with self.assertRaisesRegex(ValueError, 'naturally exited'):
                subject.require_main48_complete('/unused')

    def test_partial48_is_refused_and_exited_complete48_is_accepted(self):
        metadata = self.metadata()
        with patch.object(subject, '_raw', return_value=b'CPU_only'), \
             patch.object(subject, '_pointer', side_effect=self.pointers(metadata)):
            self.assertTrue(subject.require_main48_complete('/unused')['complete'])
            metadata['status']['active'] = dict(job_id='still-running')
            with self.assertRaises(ValueError):
                subject.require_main48_complete('/unused')


if __name__ == '__main__':
    unittest.main()
