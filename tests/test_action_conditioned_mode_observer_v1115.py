"""CPU-only public DTO association tests. No simulator/model/job construction."""
from copy import deepcopy
import unittest
from unittest.mock import patch

import numpy as np

from our_mani.contracts import ObjectState, Pose, RobotState
from our_mani.multitask.mode_observer import PublicModeObserver, PublicModeError
from our_mani.multitask.action_conditioned_mode_observer_v1115 import ActionConditionedPublicModeObserver


IDS = ('cup0', 'cup1', 'cup2')


def body(identity, position, *, time=1., visible=True):
    return ObjectState(identity, time, Pose(position), np.array([[0.,0.,0.], [.01,0.,0.], [0.,.01,0.]]),
                       np.zeros(3), np.zeros(3), visible=visible)


def states(*, time=1.):
    return [body('cup0', [.02,0.,0.], time=time), body('cup1', [.06,0.,0.], time=time),
            body('cup2', [.10,0.,0.], time=time)]


def robot(*, time=1.05, contact=True):
    return RobotState(time, Pose([0.,0.,0.]), np.zeros(3), np.zeros(3), np.zeros(7), np.zeros(7),
                      gripper_open=not contact, contact=contact)


def publication(identity='cup1', *, issued=1., selected_at=.55, confirmed=1.05, previous=None):
    return dict(selected=dict(phase='acquire', manipulated_ids=(identity,), selected_at=selected_at),
        proof=dict(phase='acquire', time=issued, desired_opening=0., current_requested_opening=1.,
                   qualified_opening=0., native_truth_read=False),
        proposed=dict(phase_index=1, issued_at=issued, manipulated_ids=[identity]),
        confirmed_after=confirmed, previous_states=states(time=issued) if previous is None else previous)


class ActionConditionedAssociation(unittest.TestCase):
    def observer(self, **kwargs):
        return ActionConditionedPublicModeObserver(IDS, IDS, .15, **kwargs)

    def test_confirmed_intent_disambiguates_three_existing_candidates_not_nearest(self):
        mode = self.observer()
        self.assertTrue(mode.record_confirmed_close(**publication()))
        modes, invalid = mode.observe(states(), robot())
        self.assertEqual(modes, {'cup0':'external', 'cup1':'attached', 'cup2':'external'})
        self.assertEqual(invalid, set())
        event = mode.receipt()['events'][-1]
        self.assertEqual([x['identity'] for x in event['candidates']], list(IDS))
        self.assertEqual(event['nearest_candidate_evidence']['identity'], 'cup0')
        self.assertEqual(event['held_after'], 'cup1')
        self.assertEqual(event['decision'], 'associate_using_confirmed_public_action_intent')
        self.assertFalse(event['native_grasp_identity_verified'])

    def test_no_intent_keeps_original_ambiguity_and_unique_fallback(self):
        mode = self.observer()
        expected = PublicModeObserver(IDS, IDS, .15).observe(states(), robot())
        self.assertEqual(mode.observe(states(), robot()), expected)
        self.assertIsNone(mode.held)
        mode = self.observer()
        self.assertEqual(mode.observe([states()[1]], robot()),
                         ({'cup0':'external', 'cup1':'attached', 'cup2':'external'}, set()))
        self.assertEqual(mode.receipt()['events'][-1]['decision'], 'fallback_original_unique_candidate')

    def test_unknown_identity_or_multiple_manipulated_ids_cannot_publish(self):
        for identity in ('native-grasp-id', 'frame'):
            mode = self.observer()
            self.assertFalse(mode.record_confirmed_close(**publication(identity)))
            self.assertIsNone(mode.pending_close)
            self.assertEqual(mode.observe(states(), robot())[1], set(IDS))
        mode = self.observer()
        data = publication()
        data['selected']['manipulated_ids'] = ('cup0', 'cup1')
        self.assertFalse(mode.record_confirmed_close(**data))

    def test_publication_needs_fresh_actual_qualified_close_edge(self):
        for field, value in (('desired_opening', 1.), ('current_requested_opening', 0.),
                             ('qualified_opening', 1.), ('native_truth_read', True)):
            mode = self.observer()
            data = publication()
            data['proof'][field] = value
            self.assertFalse(mode.record_confirmed_close(**data), field)
        mode = self.observer()
        data = publication()
        data['proposed']['manipulated_ids'] = ['cup0']
        self.assertFalse(mode.record_confirmed_close(**data))

    def test_close_diagnostics_are_not_an_extra_two_cm_or_angle_gate(self):
        mode = self.observer()
        data = publication()
        data['proposed'].update(distance_m=.049, angle_rad=.8)
        self.assertTrue(mode.record_confirmed_close(**data))
        self.assertEqual(mode.observe(states(), robot())[0]['cup1'], 'attached')

    def test_noncausal_confirmation_and_future_or_stale_snapshot_rejected(self):
        for data in (publication(selected_at=1.1), publication(confirmed=1.), publication(confirmed=1.2),
                     publication(previous=states(time=1.05)), publication(previous=states(time=.8))):
            self.assertFalse(self.observer().record_confirmed_close(**data))
        mode = self.observer()
        mode.observe([], robot(time=2., contact=False))
        self.assertFalse(mode.record_confirmed_close(**publication()))

    def test_invisible_or_duplicate_intended_snapshot_rejected(self):
        data = publication(previous=[body('cup1', [.06,0.,0.], visible=False)])
        self.assertFalse(self.observer().record_confirmed_close(**data))
        data = publication(previous=[states()[1], states()[1]])
        self.assertFalse(self.observer().record_confirmed_close(**data))

    def test_confirmed_witness_cross_check_and_duplicate_close_idempotence(self):
        mode = self.observer()
        data = publication()
        witness = dict(data['proposed'], confirmed_after=data['confirmed_after'])
        self.assertTrue(mode.record_confirmed_close(**data, confirmed_witness=witness))
        saved = deepcopy(mode.pending_close)
        self.assertFalse(mode.record_confirmed_close(**data, confirmed_witness=witness))
        self.assertEqual(mode.pending_close, saved)
        mode = self.observer()
        witness['manipulated_ids'] = ['cup0']
        self.assertFalse(mode.record_confirmed_close(**data, confirmed_witness=witness))

    def test_selected_id_must_remain_in_original_visible_distance_candidate_set(self):
        for current in ([states()[0], body('cup1', [.06,0.,0.], visible=False), states()[2]],
                        [states()[0], body('cup1', [.4,0.,0.]), states()[2]]):
            mode = self.observer()
            self.assertTrue(mode.record_confirmed_close(**publication()))
            modes, invalid = mode.observe(current, robot())
            self.assertIsNone(mode.held)
            self.assertEqual(invalid, set(IDS))
            self.assertEqual(mode.receipt()['events'][-1]['intent_status'],
                             'intended_body_not_in_current_original_candidate_set')

    def test_expired_intent_not_used_and_contact_false_clears_it(self):
        mode = self.observer()
        mode.record_confirmed_close(**publication())
        self.assertEqual(mode.observe(states(time=1.2), robot(time=1.25))[1], set(IDS))
        self.assertIsNone(mode.pending_close)
        self.assertEqual(mode.receipt()['events'][-1]['intent_status'],
                         'confirmed_intent_not_observed_within_one_counted_tick')
        mode = self.observer()
        mode.record_confirmed_close(**publication())
        mode.observe([], robot(contact=False))
        self.assertIsNone(mode.pending_close)
        self.assertEqual(mode.receipt()['events'][-1]['candidates_status'], 'not_evaluated')

    def test_association_persists_without_requery_then_release_clears(self):
        mode = self.observer()
        mode.record_confirmed_close(**publication())
        mode.observe(states(), robot())
        with patch.object(np.linalg, 'norm', side_effect=AssertionError('no repeated candidate pass')):
            self.assertEqual(mode.observe([], robot(time=1.10))[0]['cup1'], 'attached')
        modes, invalid = mode.observe([], robot(time=1.15, contact=False))
        self.assertEqual(modes['cup1'], 'released')
        self.assertEqual(invalid, set())
        self.assertIsNone(mode.held)
        self.assertIsNone(mode.pending_close)

    def test_exact_public_dtos_refuse_native_or_duck_typed_getters(self):
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

    def test_one_original_distance_pass_and_plain_tuple_return(self):
        mode = self.observer()
        mode.record_confirmed_close(**publication())
        original_norm = np.linalg.norm
        with patch.object(np.linalg, 'norm', wraps=original_norm) as norm, \
             patch.object(PublicModeObserver, 'observe', side_effect=AssertionError('old pass must not run too')):
            result = mode.observe(states(), robot())
        self.assertEqual(norm.call_count, 3)
        self.assertIs(type(result), tuple)
        self.assertIs(type(result[0]), dict)
        self.assertIs(type(result[1]), set)

    def test_clock_regression_or_future_candidate_is_an_input_error(self):
        mode = self.observer()
        mode.observe([], robot(time=1.05, contact=False))
        with self.assertRaises(PublicModeError):
            mode.observe([], robot(time=1., contact=False))
        with self.assertRaises(PublicModeError):
            self.observer().observe(states(time=2.), robot(time=1.05))

    def test_bounded_diagnostics_do_not_change_decision(self):
        mode = self.observer(max_events=1)
        mode.record_confirmed_close(**publication())
        self.assertEqual(mode.observe(states(), robot())[0]['cup1'], 'attached')
        self.assertEqual(mode.receipt()['dropped_events'], 1)
        self.assertTrue(mode.receipt()['public_association_is_not_verified_native_grasp_identity'])


if __name__ == '__main__':
    unittest.main()
