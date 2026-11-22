"""CPU-only saved-tap interpretation and adapter cleanup; never Native work."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts import diagnose_TRAIN_remove_postcomplete_v1104_r10 as subject


def call(method='_done', class_name='SkillGeometryIntents', result=False, values=()):
    expressions, evaluations = {}, {}
    for index, (expression, kind, value) in enumerate(values):
        key = str(index)
        expressions[key] = dict(expression=expression, kind=kind, instrumented=True)
        evaluations[key] = [dict(status='recorded', value=value)]
    return dict(call_id=1, parent_call_id=None, source=dict(class_name=class_name, method=method),
        return_value=dict(status='recorded', value=result), expressions=expressions, evaluations=evaluations,
        not_evaluated={'unvisited': dict(status='unknown_not_evaluated_on_original_path')},
        unobserved_compound_conditions={'compound': dict(status='unknown_not_instrumented_to_preserve_conditional_truth_semantics')})


class RemoveSummaryTests(unittest.TestCase):
    def test_unvisited_and_compound_never_become_false(self):
        summary = subject.summarize_capture_receipt(dict(calls=[call()], errors=[], dropped_calls=0))
        row = summary['acquire_done'][0]
        self.assertEqual(row['first_supported_blocker'], 'unknown_unobserved_or_unmaterialized_condition')
        self.assertEqual(row['unknown_not_evaluated'], ['unvisited'])
        self.assertEqual(row['compound_conditions_unknown'], ['compound'])
        self.assertTrue(summary['unknown_not_observed_is_not_false'])

    def test_original_done_short_circuit_order_is_respected(self):
        for expression, expected in (
            ('target is None', 'completion_target_unavailable'),
            ('s.robot.timestamp - self.phase_epoch < t.duration_seconds - 1e-08', 'minimum_acquire_duration_not_elapsed'),
        ):
            row = call(values=[(expression, 'comparison', True)])
            self.assertEqual(subject._blocker(row, []), expected)
        row = call(values=[('distance <= 0.02', 'comparison', False)])
        self.assertEqual(subject._blocker(row, []), 'current_position_outside_2cm')
        row = call(values=[('distance <= 0.02', 'comparison', True), ('angle <= 0.1', 'comparison', False)])
        self.assertEqual(subject._blocker(row, []), 'current_orientation_outside_0_1rad')

    def test_capture_false_precedes_unvisited_distance_gate(self):
        parent = call()
        capture = call('_public_capture', result=False)
        capture.update(call_id=2, parent_call_id=1)
        summary = subject.summarize_capture_receipt(dict(calls=[capture, parent]))
        self.assertEqual(summary['acquire_done_blocker_counts'], {'public_capture_returned_false': 1})
        self.assertEqual(summary['public_capture_blocker_counts'], {'unknown_unobserved_or_unmaterialized_condition': 1})

    def test_capture_contact_witness_and_attachment_are_distinct(self):
        rows = (
            ([('s.robot.contact', 'public_boolean_operand', False)], 'shared_held_presence_false'),
            ([('s.robot.contact', 'public_boolean_operand', True), ('w is None', 'comparison', True)], 'own_close_witness_absent'),
            ([('associated == {identity}', 'comparison', False)], 'public_attached_identity_not_unique_or_mismatch'),
            ([('identity in f.actor.controller.measured_attachments', 'comparison', False)], 'measured_attachment_transform_absent'),
        )
        for values, expected in rows:
            self.assertEqual(subject._capture_blocker(call('_public_capture', values=values)), expected)

    def test_selection_is_fixed_train_only(self):
        subject.validate_selection(dict(task='remove_cups_from_rotating_frame', method='cv_geometry', raw_episode_id='episode102'))
        for extra in (dict(method='physmani'), dict(raw_episode_id='episode999'), dict(split='test')):
            selection=dict(task='remove_cups_from_rotating_frame', method='cv_geometry', raw_episode_id='episode102')
            selection.update(extra)
            with self.assertRaises(ValueError): subject.validate_selection(selection)

    def test_pre_install_failure_releases_scope_without_changing_geometry(self):
        original = lambda *args: None
        runtime = SimpleNamespace(_geometry=original)
        selection=dict(task='remove_cups_from_rotating_frame', method='cv_geometry', raw_episode_id='episode102')
        for _ in range(2):
            events={}
            with patch.object(subject, '_load_observer', side_effect=RuntimeError('expected fake failure')):
                with self.assertRaisesRegex(RuntimeError, 'expected fake failure'):
                    with subject.observer_context_factory('/unused', runtime, selection, events): pass
            self.assertIs(runtime._geometry, original)
            self.assertTrue(events['geometry_hook_restored'])
            self.assertFalse(subject._SCOPE_LOCK.locked())


if __name__ == '__main__':
    unittest.main()
