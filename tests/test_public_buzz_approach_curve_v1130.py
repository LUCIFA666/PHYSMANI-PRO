"""CPU-only tests: frozen real TRAIN curves, public DTOs, no simulator imports."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace as NS, MethodType
from unittest.mock import Mock, patch
import json
import unittest

import numpy as np

from our_mani.contracts import Pose, RobotState, ObjectState, Forecast
from our_mani.multitask.interaction import InteractionLibrary
from our_mani.multitask.physmani_terminal_opening_v620 import BuzzCurveIntents
from our_mani.multitask import public_buzz_approach_curve_v1130 as subject

ROOT = Path(__file__).resolve().parents[1]


class ApproachCurveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.configs = {}
        cls.rules = {}
        cls.libraries = {}
        for task in subject.TASKS:
            path = ROOT / ('artifacts/framework/multitask_r10/post_training_qualification_v680_setup/'
                           + task + '_ours_configuration.json')
            config = json.loads(path.read_text())
            cls.configs[task] = config
            cls.rules[task] = subject.binding(ROOT, config, task)
            cls.libraries[task] = InteractionLibrary.load(
                ROOT / config['learned_artifacts'][task]['interaction']['path'])

    def fixture(self, task=subject.TASKS[0], robot_position=None, timestamp=0.):
        library = self.libraries[task]
        t = library.for_phase('approach')[0]
        robot_position = t.positions[0] if robot_position is None else robot_position
        robot = RobotState(timestamp, Pose(robot_position), np.zeros(3), np.zeros(3),
                           np.zeros(7), np.zeros(7))
        reference = ObjectState('wand', timestamp, Pose(np.zeros(3)), np.zeros((3, 3)),
                                np.zeros(3), np.zeros(3))
        generator = NS(observed_reference=Mock(return_value=reference))
        controller = NS(library=library, generator=generator, committed_roles={})
        forecast = Forecast(('wand',), np.array([timestamp, timestamp+1.]),
            np.zeros((2, 1, 3)), np.broadcast_to(np.eye(3), (2, 1, 3, 3)),
            np.zeros((2, 1, 3)), np.zeros((2, 1, 3)), metadata={'observed_until': timestamp})
        frontend = NS(actor=NS(controller=controller, predictor='cv_geometry'), forecast=forecast,
            state=NS(robot=robot, role_bindings={'wand': ('wand',)},
                     perception_metadata={'bodies': {'wand': {'motion_mode': 'external'}}}))
        policy = BuzzCurveIntents(frontend, self.configs[task]['public_Buzz_curve_v602'][task])
        return policy, deepcopy(self.rules[task]), t

    def test_source_binding_both_tasks_keeps_original_train_cad_and_guards(self):
        for task, rule in self.rules.items():
            self.assertEqual(rule['current_public_progress_distance_m'], .1)
            self.assertEqual(rule['approach_lookahead_seconds'], 0.)
            self.assertEqual(rule['number_of_TRAIN_knots'], 21)
            self.assertEqual(rule['artifacts']['geometry']['sha256'], subject.PINS[task]['geometry'])
            self.assertFalse(rule['new_numeric_thresholds'])
            self.assertFalse(rule['global_budget_extended'])
            self.assertTrue(rule['candidate_not_validated_as_solution'])

    def test_current_geometry_can_pass_multiple_knots_in_one_call_without_time_gate(self):
        policy, rule, t = self.fixture()
        original = policy.candidates('approach')[0]
        self.assertTrue(np.allclose(original['pose'].position, t.positions[-1]))
        events = {}
        with subject.owned_buzz_approach_scope(policy, rule, events=events) as same:
            self.assertIs(same, policy)
            row = policy.candidates('approach')[0]
            self.assertGreater(row['curve_index'], 1)
            self.assertLess(row['curve_index'], 20)
            self.assertTrue(np.allclose(row['pose'].position, t.positions[row['curve_index']]))
            self.assertEqual(row['selected_at'], 0.)
            self.assertEqual(row['predicted_reference_time'], 0.)
            self.assertEqual(row['opening'], original['opening'])
            self.assertEqual(row['manipulated_ids'], original['manipulated_ids'])
            self.assertEqual(row['reference_id'], original['reference_id'])
            self.assertFalse(policy._done(row))
        self.assertTrue(events['restored'])

    def test_elapsed_time_and_forecast_cannot_advance_a_far_current_tcp(self):
        for timestamp in (0., 100.):
            policy, rule, t = self.fixture(robot_position=[10., 0., 0.], timestamp=timestamp)
            with patch.object(Forecast, 'at', side_effect=AssertionError('no future query')) as at, \
                    subject.owned_buzz_approach_scope(policy, rule):
                for _ in range(3):
                    row = policy.candidates('approach')[0]
                    self.assertEqual(row['curve_index'], 0)
                    self.assertTrue(np.allclose(row['pose'].position, t.positions[0]))
                at.assert_not_called()

    def test_stable_cursor_and_target_follow_current_public_cad_frame(self):
        policy, rule, t = self.fixture()
        with subject.owned_buzz_approach_scope(policy, rule):
            first = policy.candidates('approach')[0]
            robot = policy.frontend.state.robot
            policy.frontend.state.robot = RobotState(robot.timestamp, Pose([10., 0., 0.]),
                np.zeros(3), np.zeros(3), np.zeros(7), np.zeros(7))
            shift = np.array([.4, -.2, .1])
            reference = ObjectState('wand', 0., Pose(shift), np.zeros((3, 3)), np.zeros(3), np.zeros(3))
            policy.frontend.actor.controller.generator.observed_reference.return_value = reference
            second = policy.candidates('approach')[0]
            self.assertEqual(first['curve_index'], second['curve_index'])
            self.assertTrue(np.allclose(second['pose'].position-first['pose'].position, shift))

    def test_cursor_keys_isolate_different_reference_and_manipulated_ids(self):
        policy, rule, _ = self.fixture()
        row = policy.candidates('approach')[0]
        rows = [row, dict(row, reference_id='wand2', manipulated_ids=('wand2',))]
        policy.candidates = Mock(return_value=rows)
        events = {}
        with subject.owned_buzz_approach_scope(policy, rule, events=events):
            result = policy.candidates('approach')
            self.assertEqual(len(result), 2)
        cursors = events['approach_receipt']['cursors']
        self.assertEqual({c['reference_id'] for c in cursors}, {'wand', 'wand2'})
        policy.candidates.assert_called_once_with('approach')

    def test_last_knot_still_requires_exact_original_completion_result(self):
        policy, rule, t = self.fixture(robot_position=[10., 0., 0.])
        sentinel = object()
        policy._done = Mock(return_value=sentinel)
        with subject.owned_buzz_approach_scope(policy, rule):
            row = policy.candidates('approach')[0]
            self.assertFalse(policy._done(row))
            # Move the current public TCP successively; no clock changes.
            for position in t.positions:
                policy.frontend.state.robot = RobotState(0., Pose(position), np.zeros(3), np.zeros(3),
                                                        np.zeros(7), np.zeros(7))
                row = policy.candidates('approach')[0]
            self.assertEqual(row['curve_index'], 20)
            self.assertIs(policy._done(row), sentinel)
        policy._done.assert_called_once_with(row)

    def test_last_knot_retains_author_ten_cm_phase_guard(self):
        policy, rule, t = self.fixture()
        with subject.owned_buzz_approach_scope(policy, rule):
            for position in t.positions:
                policy.frontend.state.robot = RobotState(0., Pose(position), np.zeros(3), np.zeros(3),
                                                        np.zeros(7), np.zeros(7))
                row = policy.candidates('approach')[0]
            tcp = policy.frontend.state.robot.pose.position
            policy._completion_target = Mock(return_value=Pose(tcp + [.101, 0., 0.]))
            self.assertFalse(policy._done(row))
            policy._completion_target.return_value = Pose(tcp + [.099, 0., 0.])
            self.assertTrue(policy._done(row))

    def test_nonapproach_exact_delegate_preserves_object_and_exception(self):
        policy, rule, _ = self.fixture()
        rows, result = object(), object()
        policy.candidates = Mock(return_value=rows)
        policy._done = Mock(return_value=result)
        with subject.owned_buzz_approach_scope(policy, rule):
            for phase in ('acquire', 'follow', 'release'):
                self.assertIs(policy.candidates(phase), rows)
                self.assertIs(policy._done({'phase': phase}), result)
        self.assertEqual(policy.candidates.call_count, 3)
        self.assertEqual(policy._done.call_count, 3)
        error = RuntimeError('original candidates failed')
        policy.candidates.side_effect = error
        with self.assertRaises(RuntimeError) as caught:
            with subject.owned_buzz_approach_scope(policy, rule):
                policy.candidates('follow')
        self.assertIs(caught.exception, error)

    def test_restoration_nested_rejection_and_bounded_trace(self):
        policy, rule, _ = self.fixture()
        before = (policy.candidates, policy._done, policy.receipt,
                  policy.qualify_opening, policy.confirm_opening)
        events = {}
        error = RuntimeError('run failed')
        with self.assertRaises(RuntimeError) as caught:
            with subject.owned_buzz_approach_scope(policy, rule, events=events, max_events=1):
                with self.assertRaisesRegex(RuntimeError, 'cannot nest'):
                    with subject.owned_buzz_approach_scope(policy, rule):
                        self.fail('nested scope entered')
                policy.candidates('approach')
                policy.candidates('approach')
                receipt = policy.receipt()['public_buzz_approach_curve_v1130']
                self.assertEqual(len(receipt['events']), 1)
                self.assertEqual(receipt['dropped_events'], 1)
                raise error
        self.assertIs(caught.exception, error)
        self.assertEqual(before, (policy.candidates, policy._done, policy.receipt,
                                 policy.qualify_opening, policy.confirm_opening))
        self.assertTrue(events['restored'])

    def test_live_library_rule_or_cad_substitution_rejected(self):
        for change in ('task', 'cad', 'library', 'templates', 'guard', 'lookahead'):
            policy, rule, _ = self.fixture()
            if change == 'task':
                rule['task'] = subject.TASKS[1]
            elif change == 'cad':
                rule['artifacts']['geometry']['sha256'] = '0'*64
            elif change == 'library':
                rule['library_canonical_sha256'] = '0'*64
            elif change == 'templates':
                rule['templates_sha256'] = {}
            elif change == 'guard':
                rule['current_public_progress_distance_m'] = .03
            else:
                rule['approach_lookahead_seconds'] = .5
            with self.assertRaises(ValueError):
                with subject.owned_buzz_approach_scope(policy, rule):
                    self.fail('unbound rule entered')

    def test_configuration_artifact_substitution_rejected_before_scope(self):
        config = deepcopy(self.configs[subject.TASKS[0]])
        config['learned_artifacts'][subject.TASKS[0]]['interaction']['sha256'] = '0'*64
        with self.assertRaisesRegex(ValueError, 'frozen TRAIN artifact'):
            subject.binding(ROOT, config, subject.TASKS[0])

    def test_empty_original_candidates_do_not_invent_a_current_frame(self):
        policy, rule, _ = self.fixture()
        rows = []
        policy.candidates = Mock(return_value=rows)
        policy.frontend.state = None
        with subject.owned_buzz_approach_scope(policy, rule):
            self.assertIs(policy.candidates('approach'), rows)
        policy.candidates.assert_called_once_with('approach')

    def test_rule_snapshot_cannot_be_mutated_into_a_new_guard_inside_scope(self):
        policy, rule, _ = self.fixture(robot_position=[10., 0., 0.])
        with subject.owned_buzz_approach_scope(policy, rule):
            rule['current_public_progress_distance_m'] = 100.
            row = policy.candidates('approach')[0]
            self.assertEqual(row['curve_index'], 0)


if __name__ == '__main__':
    unittest.main()
