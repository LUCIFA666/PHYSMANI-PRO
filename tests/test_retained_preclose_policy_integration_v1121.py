"""Matched-factor contracts: deferred close is not a confirmed close edge."""
import unittest
from unittest.mock import Mock

from our_mani.multitask import retained_preclose_policy_integration_v1121 as new
from tests import test_author_aligned_policy_integration_v1118 as integration


class Tests(unittest.TestCase):
    def setUp(self):
        integration.Tests.setUp(self)

    def scope(self, task=integration.TASK):
        return new.integrated_components(None, integration.fixture.CONFIG, task, self.frontend.actor)

    def gate(self, requested, current):
        now = float(self.frontend.state.robot.timestamp)
        actual = float(requested >= .5 or now < .95)
        proposed = None
        if actual < .5 <= current:
            proposed = dict(phase_index=self.policy.phase_index, issued_at=now,
                manipulated_ids=list(self.policy.selected['manipulated_ids']), distance_m=.01, angle_rad=.05)
        return actual, dict(time=now, phase=self.policy.selected['phase'], desired_opening=float(requested >= .5),
            current_requested_opening=current, qualified_opening=actual,
            reason='original_qualified_close' if actual < .5 else 'original_deferred_close',
            native_truth_read=False), proposed

    def test_retained_preclose_defers_then_publishes_only_the_real_later_edge(self):
        self.policy.qualify_opening = Mock(side_effect=self.gate)
        with self.scope() as (frontend, dispatched, executor):
            dispatched.select()
            self.fixture.run_action(self.env, executor)
            observer = frontend.actor.mode_observer
            self.assertIsNone(self.policy.close_witness)
            self.assertIsNone(observer.pending_close)
            self.assertIsNone(observer.held)
            self.assertEqual(self.env.ledger.grasp_requests, 0)
            dispatched.select()
            self.fixture.run_action(self.env, executor)
            self.assertEqual(self.env.ledger.grasp_requests, 1)
            self.assertAlmostEqual(self.policy.close_witness['issued_at'], .95)
            self.assertAlmostEqual(self.policy.close_witness['confirmed_after'], 1.)
            self.assertEqual(observer.held, 'cup1')
            self.assertEqual(self.policy.qualify_opening.call_count, 20)
            self.assertEqual(executor.collision_diagnostic_limit, 0)
            self.assertIsNone(executor.admit_joint_target)
            receipt = dispatched.receipt()
            self.assertIn('author_aligned_policy_integration_v1118', receipt)
            self.assertNotIn('author_requested_opening_witness_v1114', receipt)
            self.assertEqual(len(receipt['retained_preclose_dispatch_v1121']['events']), 20)

    def test_release_still_uses_parent_passthrough_even_if_original_qualifier_defers(self):
        self.policy.select()
        self.policy.selected['phase'] = 'place'
        self.policy.qualify_opening = Mock(return_value=(0., dict(time=0., phase='place',
            desired_opening=1., current_requested_opening=0., qualified_opening=0., native_truth_read=False), None))
        adapter = new.RetainedPrecloseWitness(self.policy)
        actual, proof, proposed = adapter.qualify_opening(1., 0.)
        self.assertEqual(actual, 1.)
        self.assertIsNone(proposed)
        self.assertTrue(proof['legacy_geometric_qualification_is_diagnostic_only'])
        self.policy.qualify_opening.assert_called_once()

    def test_already_commanded_closed_without_witness_does_not_invent_reopening(self):
        self.policy.select()
        self.policy.qualify_opening = Mock(return_value=(1., dict(time=0., phase='acquire',
            desired_opening=0., current_requested_opening=0., qualified_opening=1., native_truth_read=False), None))
        adapter = new.RetainedPrecloseWitness(self.policy)
        actual, proof, proposed = adapter.qualify_opening(0., 0.)
        self.assertEqual(actual, 0.)
        self.assertIsNone(proposed)
        self.assertTrue(proof['legacy_geometric_qualification_is_diagnostic_only'])
        self.policy.qualify_opening.assert_called_once()

    def test_failed_apply_cannot_publish_deferred_or_qualified_intent(self):
        self.policy.qualify_opening = Mock(side_effect=self.gate)
        self.env.apply_error = RuntimeError('physical application failed')
        with self.scope() as (frontend, dispatched, executor):
            dispatched.select()
            with self.assertRaises(Exception):
                self.fixture.run_action(self.env, executor)
            self.assertIsNone(self.policy.close_witness)
            self.assertIsNone(frontend.actor.mode_observer.pending_close)
            self.assertEqual(dispatched.opening_events, [])


if __name__ == '__main__':
    unittest.main()
