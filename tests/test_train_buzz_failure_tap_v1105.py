"""CPU-only transparency checks; no RLBench, PyRep, model, or Native imports."""
import sys
import unittest
from types import SimpleNamespace

from our_mani.multitask.train_buzz_failure_tap_v1105 import PassiveBuzzFailureTap, summarize


def fixtures(*, fail=False, error=None):
    namespace = {'__name__': 'rlbench.backend.conditions'}
    exec(compile('''
class Condition:
    def __init__(self, result, error=None):
        self.result, self.error, self.calls = result, error, 0
    @property
    def forbidden_property(self):
        raise AssertionError('telemetry must not read properties')
    def condition_met(self):
        self.calls += 1
        if self.error is not None:
            raise self.error
        return self.result
''', '/cpu_fixture/rlbench/backend/conditions.py', 'exec'), namespace)
    condition = namespace['Condition']
    namespace = {'__name__': 'rlbench.tasks.beat_the_rotating_buzz'}
    exec(compile('''
class BeatTheRotatingBuzz:
    def success(self):
        fails = [x.condition_met()[0] for x in self._fail_conditions]
        if self._fail_conditions and all(fails):
            return self.failure_result
        good = [x.condition_met()[0] for x in self._success_conditions]
        return self.success_result if all(good) else self.pending_result
''', '/cpu_fixture/rlbench/backend/task.py', 'exec'), namespace)
    task = namespace['BeatTheRotatingBuzz']()
    task._fail_conditions = [condition((fail, False), error)]
    task._success_conditions = [condition((True, False)), condition((True, False))]
    task.failure_result, task.success_result, task.pending_result = (False, True), (True, True), (False, False)

    class Environment:
        def __init__(self):
            self.task = task
            self.ledger = SimpleNamespace(ticks=7)
            self._native_action_sampling = SimpleNamespace(active=dict(action_index=2, label='geometry_execute', start_tick=4))

        def _tick(self, phase):
            return self.task.success()

    return task, Environment()


class PassiveTapTransparencyTests(unittest.TestCase):
    def test_same_return_identity_calls_and_counted_tick(self):
        task, env = fixtures()
        with PassiveBuzzFailureTap('beat_the_rotating_buzz', 'cv_geometry') as tap:
            result = env._tick('arm')
        self.assertIs(result, task.success_result)
        self.assertEqual([x.calls for x in task._fail_conditions + task._success_conditions], [1, 1, 1])
        receipt = tap.receipt()
        self.assertEqual(receipt['original_success_calls_observed'], 1)
        self.assertEqual(receipt['original_condition_calls_observed'], 3)
        self.assertEqual(receipt['observer_errors'], 0)
        self.assertTrue(receipt['restored'])
        self.assertTrue(all(x['counted_tick'] == 7 and x['native_tick_phase'] == 'arm' for x in receipt['events']))

    def test_original_failure_short_circuit_is_preserved(self):
        task, env = fixtures(fail=True)
        with PassiveBuzzFailureTap('beat_the_rotating_buzz', 'context_geometry') as tap:
            result = env._tick('gripper')
        self.assertIs(result, task.failure_result)
        self.assertEqual([x.calls for x in task._fail_conditions + task._success_conditions], [1, 0, 0])
        summary = summarize(tap.receipt())
        self.assertEqual(summary['first_observed_failure_condition']['source_condition_label'], 'wand_detected_by_middle_sensor')
        self.assertEqual(summary['first_observed_raw_terminal']['result'], [False, True])

    def test_same_exception_instance_and_no_profiler_left(self):
        error = RuntimeError('original condition error')
        task, env = fixtures(error=error)
        tap = PassiveBuzzFailureTap('beat_the_rotating_buzz', 'cv_geometry')
        with self.assertRaises(RuntimeError) as caught:
            with tap:
                env._tick('arm')
        self.assertIs(caught.exception, error)
        self.assertIsNone(sys.getprofile())
        self.assertEqual(task._fail_conditions[0].calls, 1)
        self.assertGreater(tap.receipt()['exception_or_nonstandard_returns'], 0)
        self.assertEqual(tap.receipt()['scope_exception_type'], 'RuntimeError')

    def test_existing_profiler_is_never_overwritten(self):
        def existing(frame, event, arg):
            pass
        sys.setprofile(existing)
        try:
            with self.assertRaises(RuntimeError):
                with PassiveBuzzFailureTap('beat_the_rotating_buzz', 'cv_geometry'):
                    self.fail('must refuse before entering scope')
            self.assertIs(sys.getprofile(), existing)
        finally:
            sys.setprofile(None)

    def test_bounded_events_keep_dropped_count_and_original_calls(self):
        task, env = fixtures()
        with PassiveBuzzFailureTap('beat_the_rotating_buzz', 'cv_geometry', max_events=2) as tap:
            env._tick('arm')
        receipt = tap.receipt()
        self.assertEqual(len(receipt['events']), 2)
        self.assertEqual(receipt['dropped_events'], 2)
        self.assertEqual(receipt['original_condition_returns_observed'], 3)
        self.assertEqual([x.calls for x in task._fail_conditions + task._success_conditions], [1, 1, 1])
        self.assertTrue(summarize(receipt)['absence_with_dropped_events_is_unknown'])


if __name__ == '__main__':
    unittest.main()
