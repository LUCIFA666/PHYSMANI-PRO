"""CPU public-DTO tests for delayed actual-close intent lifetime; no Native."""
from copy import deepcopy
import unittest
from unittest.mock import patch

import numpy as np

from our_mani.contracts import ObjectState, Pose, RobotState
from our_mani.multitask.mode_observer import PublicModeError, PublicModeObserver
from our_mani.multitask.action_conditioned_mode_observer_v1115b import ActionConditionedPublicModeObserver


IDS = ('cup0', 'cup1', 'cup2')


def body(identity, position, *, time=1., visible=True):
    return ObjectState(identity, time, Pose(position), np.array([[0., 0., 0.], [.01, 0., 0.], [0., .01, 0.]]),
                       np.zeros(3), np.zeros(3), visible=visible)


def states(*, time=1.):
    return [body('cup0', [.02, 0., 0.], time=time), body('cup1', [.06, 0., 0.], time=time),
            body('cup2', [.10, 0., 0.], time=time)]


def robot(*, time=1.05, contact=True, gripper_open=None):
    return RobotState(time, Pose([0., 0., 0.]), np.zeros(3), np.zeros(3), np.zeros(7), np.zeros(7),
                      gripper_open=not contact if gripper_open is None else gripper_open, contact=contact)


def publication(identity='cup1', *, issued=1., selected_at=.55, confirmed=1.05, previous=None):
    return dict(selected=dict(phase='acquire', manipulated_ids=(identity,), selected_at=selected_at),
        proof=dict(phase='acquire', time=issued, desired_opening=0., current_requested_opening=1.,
                   qualified_opening=0., native_truth_read=False),
        proposed=dict(phase_index=1, issued_at=issued, manipulated_ids=[identity]),
        confirmed_after=confirmed, previous_states=states(time=issued) if previous is None else previous)


class DelayedCloseAssociation(unittest.TestCase):
    def observer(self, **kwargs):
        return ActionConditionedPublicModeObserver(IDS, IDS, .15, **kwargs)

    def confirmed(self, mode):
        data = publication()
        witness = dict(data['proposed'], confirmed_after=data['confirmed_after'])
        self.assertTrue(mode.on_confirmed_opening(**data, opening=0., phase_index=1,
                                                confirmed_witness=witness))
        return data

    def test_delayed_held_false_preserves_pending_over_many_ticks(self):
        mode = self.observer()
        self.confirmed(mode)
        original = deepcopy(mode.pending_close)
        for now in (1.05, 1.10, 1.25, 1.50):
            with patch.object(np.linalg, 'norm', side_effect=AssertionError('no candidate query while held=False')):
                modes, invalid = mode.observe([], robot(time=now, contact=False, gripper_open=True))
            self.assertEqual(modes, dict.fromkeys(IDS, 'external'))
            self.assertEqual(invalid, set())
            self.assertEqual(mode.pending_close, original)
        modes, invalid = mode.observe(states(time=1.50), robot(time=1.55))
        self.assertEqual(modes['cup1'], 'attached')
        self.assertEqual(invalid, set())
        self.assertIsNone(mode.pending_close)
        event = mode.receipt()['events'][-1]
        self.assertEqual(len(event['candidates']), 3)
        self.assertEqual(event['nearest_candidate_evidence']['identity'], 'cup0')
        self.assertFalse(event['native_grasp_identity_verified'])

    def test_repeated_same_closed_commands_and_replanned_pose_keep_original_edge(self):
        mode = self.observer()
        data = self.confirmed(mode)
        original = deepcopy(mode.pending_close)
        for now in (1.10, 1.35, 1.70):
            selected = dict(data['selected'], selected_at=now-.05)
            mode.synchronize_intent_context(phase='acquire', phase_index=1, manipulated_ids=['cup1'],
                                            requested_opening=0., selected_at=now-.05)
            self.assertTrue(mode.on_confirmed_opening(selected=selected, opening=0., proof={}, proposed=None,
                confirmed_after=now, previous_states=(), confirmed_witness=None, phase_index=1))
            mode.observe([], robot(time=now, contact=False))
            self.assertEqual(mode.pending_close, original)
        self.assertEqual(mode.observe(states(time=1.7), robot(time=1.75))[0]['cup1'], 'attached')

    def test_one_tick_freshness_is_still_required_at_initial_close_publication(self):
        for data in (publication(previous=states(time=.9)), publication(confirmed=1.10),
                     publication(previous=states(time=1.01)), publication(selected_at=1.01)):
            mode = self.observer()
            self.assertFalse(mode.record_confirmed_close(**data))
            self.assertIsNone(mode.pending_close)

    def test_explicit_open_cancels_and_blocks_stale_held_bit_new_association(self):
        mode = self.observer()
        data = self.confirmed(mode)
        self.assertFalse(mode.on_confirmed_opening(selected=data['selected'], opening=1., proof={}, proposed=None,
            confirmed_after=1.10, previous_states=(), phase_index=1))
        self.assertIsNone(mode.pending_close)
        result = mode.observe([states()[1]], robot(time=1.15))
        self.assertEqual(result[1], set(IDS))
        self.assertIsNone(mode.held)
        self.assertEqual(mode.receipt()['events'][-1]['candidates_status'], 'not_evaluated')
        self.assertFalse(mode.record_confirmed_close(**data))
        self.assertIsNone(mode.pending_close)

    def test_phase_index_phase_target_or_explicit_empty_target_cancel(self):
        for change, reason in (
                (dict(phase='detach'), 'public_phase_changed'),
                (dict(phase_index=2), 'public_phase_index_changed'),
                (dict(manipulated_ids=('cup2',)), 'public_selected_manipulated_identity_changed'),
                (dict(manipulated_ids=()), 'selected_manipulated_identity_no_longer_single'),
                (dict(manipulated_ids=('cup1', 'cup2')), 'selected_manipulated_identity_no_longer_single')):
            mode = self.observer()
            self.confirmed(mode)
            self.assertEqual(mode.synchronize_intent_context(**change), reason)
            self.assertIsNone(mode.pending_close)
            self.assertEqual(mode.observe(states(), robot())[1], set(IDS))

    def test_unknown_context_does_not_turn_into_false_phase_or_target_change(self):
        mode = self.observer()
        self.confirmed(mode)
        original = deepcopy(mode.pending_close)
        self.assertIsNone(mode.synchronize_intent_context())
        self.assertEqual(mode.pending_close, original)
        self.assertEqual(mode.observe(states(time=3.), robot(time=3.05))[0]['cup1'], 'attached')

    def test_malformed_context_raises_before_mutating_pending_or_command(self):
        for args in (dict(manipulated_ids='cup1'), dict(manipulated_ids=42),
                     dict(manipulated_ids=[None]), dict(phase_index=True),
                     dict(requested_opening=float('nan'))):
            mode = self.observer()
            self.confirmed(mode)
            before = deepcopy(mode.receipt())
            with self.assertRaises(PublicModeError):
                mode.synchronize_intent_context(**args)
            self.assertEqual(mode.receipt(), before)

    def test_duplicate_edge_rejected_without_erasing_pending_and_cannot_replay_after_cancel(self):
        mode = self.observer()
        data = self.confirmed(mode)
        original = deepcopy(mode.pending_close)
        self.assertFalse(mode.record_confirmed_close(**data))
        self.assertEqual(mode.pending_close, original)
        mode.cancel_pending('caller_aborted_acquire')
        self.assertFalse(mode.record_confirmed_close(**data))
        self.assertIsNone(mode.pending_close)

    def test_after_association_phase_change_preserves_held_until_public_contact_false(self):
        mode = self.observer()
        self.confirmed(mode)
        mode.observe(states(), robot())
        mode.synchronize_intent_context(phase='detach', phase_index=2, manipulated_ids=('cup1',), requested_opening=0.)
        with patch.object(np.linalg, 'norm', side_effect=AssertionError('no repeated candidate pass')):
            self.assertEqual(mode.observe([], robot(time=1.10))[0]['cup1'], 'attached')
        result = mode.observe([], robot(time=1.15, contact=False))
        self.assertEqual(result[0]['cup1'], 'released')
        self.assertIsNone(mode.held)
        self.assertIsNone(mode.pending_close)
        self.assertEqual(mode.observe(states(time=1.15), robot(time=1.20))[1], set(IDS))

    def test_open_after_association_retains_existing_until_release_without_new_identity(self):
        mode = self.observer()
        self.confirmed(mode)
        mode.observe(states(), robot())
        mode.synchronize_intent_context(requested_opening=1.)
        self.assertEqual(mode.observe([], robot(time=1.10))[0]['cup1'], 'attached')
        self.assertEqual(mode.observe([], robot(time=1.15, contact=False))[0]['cup1'], 'released')
        self.assertEqual(mode.observe([states()[0]], robot(time=1.20))[1], set(IDS))

    def test_pending_must_still_be_in_original_visible_graspable_radius_candidate_set(self):
        for candidate in (body('cup1', [.06, 0., 0.], time=2., visible=False),
                          body('cup1', [.151, 0., 0.], time=2.)):
            mode = self.observer()
            self.confirmed(mode)
            current = [body('cup0', [.02, 0., 0.], time=2.), candidate,
                       body('cup2', [.10, 0., 0.], time=2.)]
            modes, invalid = mode.observe(current, robot(time=2.05))
            self.assertIsNone(mode.held)
            self.assertIsNotNone(mode.pending_close)
            self.assertEqual(invalid, set(IDS))
            self.assertEqual(mode.receipt()['events'][-1]['intent_status'],
                             'intended_body_not_in_current_original_candidate_set')
            self.assertEqual(mode.observe(states(time=2.05), robot(time=2.10))[0]['cup1'], 'attached')

    def test_unknown_non_graspable_and_multi_target_cannot_publish(self):
        mode = ActionConditionedPublicModeObserver(IDS, ('cup0', 'cup2'), .15)
        self.assertFalse(mode.record_confirmed_close(**publication()))
        self.assertFalse(self.observer().record_confirmed_close(**publication('native-id')))
        data = publication()
        data['selected']['manipulated_ids'] = ('cup0', 'cup1')
        self.assertFalse(self.observer().record_confirmed_close(**data))

    def test_no_intent_keeps_original_unique_fallback_and_ambiguity(self):
        for current in (states(), [states()[1]], []):
            self.assertEqual(self.observer().observe(current, robot()),
                             PublicModeObserver(IDS, IDS, .15).observe(current, robot()))

    def test_close_confirmation_requires_original_witness_and_public_proof(self):
        for key, value in (('native_truth_read', True), ('current_requested_opening', 0.),
                           ('desired_opening', 1.), ('qualified_opening', 1.), ('phase', 'detach')):
            data = publication()
            data['proof'][key] = value
            self.assertFalse(self.observer().record_confirmed_close(**data))
        data = publication()
        witness = dict(data['proposed'], confirmed_after=data['confirmed_after'], manipulated_ids=['cup0'])
        self.assertFalse(self.observer().record_confirmed_close(**data, confirmed_witness=witness))

    def test_actual_hook_phase_must_match_the_new_edge(self):
        mode = self.observer()
        self.assertFalse(mode.on_confirmed_opening(**publication(), opening=0., phase_index=2))
        self.assertIsNone(mode.pending_close)
        self.assertEqual(mode.receipt()['events'][-1]['reason'],
                         'actual_confirmed_public_phase_index_does_not_match_fresh_edge')

    def test_recorded_geometry_is_diagnostic_not_old_two_cm_point_one_radian_gate(self):
        mode = self.observer()
        data = publication()
        data['proposed'].update(distance_m=.12, angle_rad=2.)
        self.assertTrue(mode.record_confirmed_close(**data))
        self.assertEqual(mode.observe(states(time=2.), robot(time=2.05))[0]['cup1'], 'attached')

    def test_one_candidate_pass_and_original_tuple_contract(self):
        mode = self.observer()
        self.confirmed(mode)
        original = np.linalg.norm
        with patch.object(np.linalg, 'norm', wraps=original) as norm, \
             patch.object(PublicModeObserver, 'observe', side_effect=AssertionError('must not call second pass')):
            result = mode.observe(states(time=2.), robot(time=2.05))
        self.assertEqual(norm.call_count, 3)
        self.assertIs(type(result), tuple)
        self.assertIs(type(result[0]), dict)
        self.assertIs(type(result[1]), set)

    def test_exact_public_dtos_reject_duck_typed_getters(self):
        class Poison:
            def __getattribute__(self, name):
                raise AssertionError('forbidden getter')
        with self.assertRaises(PublicModeError):
            self.observer().observe([], Poison())
        with self.assertRaises(PublicModeError):
            self.observer().observe([Poison()], robot())
        data = publication(previous=[Poison()])
        with self.assertRaises(PublicModeError):
            self.observer().record_confirmed_close(**data)

    def test_reset_clears_pending_held_clock_and_release_but_preserves_audit(self):
        mode = self.observer()
        data = self.confirmed(mode)
        mode.observe(states(), robot())
        mode.observe([], robot(time=1.10, contact=False))
        mode.reset()
        self.assertEqual(mode.released, set())
        self.assertIsNone(mode.held)
        self.assertIsNone(mode.pending_close)
        self.assertIsNone(mode.action_context)
        self.assertFalse(mode.open_command_active)
        self.assertEqual(mode.receipt()['reset_epoch'], 1)
        self.assertTrue(mode.record_confirmed_close(**data))
        self.assertEqual(mode.observe(states(), robot())[0]['cup1'], 'attached')

    def test_clock_regression_and_future_candidate_raise_without_native_work(self):
        mode = self.observer()
        mode.observe([], robot(time=2., contact=False))
        with self.assertRaises(PublicModeError):
            mode.observe([], robot(time=1., contact=False))
        with self.assertRaises(PublicModeError):
            self.observer().observe(states(time=2.), robot(time=1.05))

    def test_bounded_diagnostics_do_not_change_pending_lifecycle(self):
        mode = self.observer(max_events=1)
        self.confirmed(mode)
        mode.observe([], robot(time=1.5, contact=False))
        self.assertEqual(mode.observe(states(time=2.), robot(time=2.05))[0]['cup1'], 'attached')
        receipt = mode.receipt()
        self.assertEqual(receipt['dropped_events'], 2)
        self.assertTrue(receipt['pending_close_has_no_age_expiry'])
        self.assertTrue(receipt['one_tick_limit_applies_only_to_close_publication_evidence'])
        self.assertTrue(receipt['public_association_is_not_verified_native_grasp_identity'])


if __name__ == '__main__':
    unittest.main()
