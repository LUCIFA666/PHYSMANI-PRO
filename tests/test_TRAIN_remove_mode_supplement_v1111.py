"""Pure Python/NumPy mode transparency and delayed combo activation tests."""
from contextlib import contextmanager
from types import SimpleNamespace
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

from our_mani.multitask.mode_observer import PublicModeObserver
from our_mani.multitask.train_remove_mode_tap_v1111 import PassiveRemoveModeTap, summarize
from scripts import diagnose_TRAIN_remove_mode_supplement_v1111_r10 as entry
from scripts import run_TRAIN_remove_mode_supplement_v1112_r10 as owner


def body(identity, x, *, visible=True):
    return SimpleNamespace(object_id=identity, visible=visible, timestamp=0.,
                           pose=SimpleNamespace(position=np.array([x, 0., 0.])))


def robot(contact=True):
    return SimpleNamespace(contact=contact, timestamp=.05, pose=SimpleNamespace(position=np.zeros(3)))


class OriginalModeTransparency(unittest.TestCase):
    def test_unique_original_candidate_and_filter_called_once(self):
        observer = PublicModeObserver(('cup0', 'cup1'), ('cup0', 'cup1'), .15)
        states = [body('cup0', .04)]
        baseline = PublicModeObserver(('cup0', 'cup1'), ('cup0', 'cup1'), .15).observe(states, robot())
        original_norm = np.linalg.norm
        with patch.object(np.linalg, 'norm', wraps=original_norm) as norm:
            with PassiveRemoveModeTap(observer) as tap:
                result = observer.observe(states, robot())
        self.assertEqual(result, baseline)
        self.assertEqual(norm.call_count, 1)
        receipt = tap.receipt()
        self.assertEqual(receipt['original_observe_calls'], 1)
        self.assertEqual(receipt['observer_errors'], 0)
        self.assertTrue(receipt['restored'])
        event = receipt['events'][0]
        self.assertEqual(event['original_candidates_local'],
                         dict(status='recorded_original_local', count=1, object_ids=['cup0']))
        self.assertEqual(event['held_before'], None)
        self.assertEqual(event['held_after'], 'cup0')
        self.assertEqual(summarize(receipt)['branch_counts'], {'unique_candidate': 1})

    def test_zero_and_multiple_distinct_from_uncomputed(self):
        for states, expected in (([body('cup0', .4)], 'zero_candidates'),
                                 ([body('cup0', .04), body('cup1', .08)], 'multiple_candidates')):
            observer = PublicModeObserver(('cup0', 'cup1'), ('cup0', 'cup1'), .15)
            with PassiveRemoveModeTap(observer) as tap:
                modes, invalid = observer.observe(states, robot())
            self.assertEqual(invalid, {'cup0', 'cup1'})
            self.assertEqual(modes, {'cup0': 'external', 'cup1': 'external'})
            self.assertEqual(summarize(tap.receipt())['branch_counts'], {expected: 1})
        observer = PublicModeObserver(('cup0',), ('cup0',), .15)
        observer.held = 'cup0'
        with PassiveRemoveModeTap(observer) as tap:
            result = observer.observe([], robot())
        self.assertEqual(result, ({'cup0': 'attached'}, set()))
        self.assertEqual(tap.receipt()['events'][0]['original_candidates_local'], {'status': 'unknown_not_evaluated'})

    def test_contact_false_release_does_not_invent_candidate_list(self):
        observer = PublicModeObserver(('cup0',), ('cup0',), .15)
        observer.held = 'cup0'
        with PassiveRemoveModeTap(observer) as tap:
            result = observer.observe([], robot(False))
        self.assertEqual(result, ({'cup0': 'released'}, set()))
        event = tap.receipt()['events'][0]
        self.assertIsNone(event['held_after'])
        self.assertEqual(event['original_candidates_local'], {'status': 'unknown_not_evaluated'})

    def test_no_extra_property_getter_and_original_exception_identity(self):
        error = RuntimeError('original visible getter')
        class OriginalInput:
            def __init__(self):
                self.object_id = 'cup0'
                self.pose = SimpleNamespace(position=np.zeros(3))
                self.getter_calls = 0
            @property
            def visible(self):
                self.getter_calls += 1
                raise error
        state = OriginalInput()
        observer = PublicModeObserver(('cup0',), ('cup0',), .15)
        with self.assertRaises(RuntimeError) as caught:
            with PassiveRemoveModeTap(observer) as tap:
                observer.observe([state], robot())
        self.assertIs(caught.exception, error)
        self.assertEqual(state.getter_calls, 1)
        self.assertIsNone(sys.getprofile())
        self.assertEqual(tap.receipt()['events'][0]['original_candidates_local'], {'status': 'unknown_not_evaluated'})
        self.assertEqual(tap.receipt()['events'][0]['original_return_status'], 'unknown_nonstandard_or_exception_return')

    def test_existing_profiler_never_overwritten(self):
        def existing(frame, event, value):
            pass
        sys.setprofile(existing)
        try:
            with self.assertRaises(RuntimeError):
                with PassiveRemoveModeTap(PublicModeObserver(('cup0',), ('cup0',), .15)):
                    self.fail('must reject existing profiler')
            self.assertIs(sys.getprofile(), existing)
        finally:
            sys.setprofile(None)


class DelayedCombinedScope(unittest.TestCase):
    selection = dict(task='remove_cups_from_rotating_frame', raw_episode_id='episode104', method='cv_geometry')

    def test_profile_enters_after_exact_original_geometry_return_and_restores(self):
        order = []
        result = (SimpleNamespace(mode_observer=object()), None, None, None, None)
        def geometry(*args):
            order.append('original_geometry')
            return result
        runtime = SimpleNamespace(_geometry=geometry)
        @contextmanager
        def capture(root, module, selected, sink, *, permit):
            order.append('capture_enter')
            try:
                yield
            finally:
                sink.update(restored=True, observer_receipts=[], summary=[])
                order.append('capture_exit')
        class Tap:
            def __init__(self, observer):
                self.restored = False
                self.assert_original_returned = order[-1] == 'original_geometry'
            def __enter__(self):
                if not self.assert_original_returned:
                    raise AssertionError('profiler activated before original constructor completed')
                order.append('mode_enter')
                return self
            def __exit__(self, *exc):
                self.restored = True
                order.append('mode_exit')
            def receipt(self):
                return dict(restored=self.restored, events=[], original_observe_calls=0,
                            dropped_events=0, observer_errors=0)
        events = {}
        with entry.observer_context_factory('/unused', runtime, self.selection, events, permit=object(),
                                           capture_factory=capture, mode_tap_class=Tap):
            self.assertEqual(order, ['capture_enter'])
            self.assertIs(runtime._geometry(None, None, None), result)
            with self.assertRaises(ValueError):
                runtime._geometry(None, None, None)
        self.assertEqual(order, ['capture_enter', 'original_geometry', 'mode_enter', 'mode_exit', 'capture_exit'])
        self.assertIs(runtime._geometry, geometry)
        self.assertTrue(events['restored'])

    def test_only_two_preselected_cv_cases(self):
        for task, raw in entry.REPRESENTATIVES:
            entry.validate_selection(dict(task=task, raw_episode_id=raw, method='cv_geometry'))
        for change in (dict(raw_episode_id='episode102'), dict(method='context_geometry')):
            selected = dict(self.selection)
            selected.update(change)
            with self.assertRaises(ValueError):
                entry.validate_selection(selected)


class OrderedOwnerPrerequisites(unittest.TestCase):
    def test_buzz_owner_must_exit_even_after_complete4(self):
        from scripts import diagnose_TRAIN_buzz_collision_supplement_v1109_r10 as gate
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = root / owner.BUZZ_PLAN
            plan.parent.mkdir(parents=True)
            plan.write_text('{}')
            checksum = hashlib.sha256(plan.read_bytes()).hexdigest()
            runtime = root / owner.BUZZ_RUNTIME
            runtime.mkdir(parents=True)
            status = dict(stage='complete', expected=4, completed=[{}] * 4, failed=[], active=None)
            terminal = dict(complete=True, completed=4, infrastructure_failed=0, automatic_retry=False)
            fields = Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()
            launch = dict(pid=os.getpid(), start_ticks=int(fields[19]),
                          plan=dict(path=owner.BUZZ_PLAN, sha256=checksum))
            for name, value in (('status', status), ('terminal', terminal), ('launch', launch)):
                (runtime / (name + '.json')).write_text(json.dumps(value))
            with patch.object(owner, 'ROOT', root), patch.object(owner, 'BUZZ_PLAN_SHA', checksum), \
                 patch.object(gate, 'require_main48_complete', return_value={'CPU_fake_original48': True}):
                with self.assertRaisesRegex(ValueError, 'Buzz4 owner is still alive'):
                    owner.require_predecessors_finished()
                launch['pid'] = 999999999
                (runtime / 'launch.json').write_text(json.dumps(launch))
                self.assertTrue(owner.require_predecessors_finished()['buzz4']['complete'])
                status['stage'] = 'running'
                (runtime / 'status.json').write_text(json.dumps(status))
                with self.assertRaises(ValueError):
                    owner.require_predecessors_finished()


if __name__ == '__main__':
    unittest.main()
