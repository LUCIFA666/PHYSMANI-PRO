"""CPU-only phase tests using the original public capture and original MRO."""
from contextlib import contextmanager
from types import MethodType, SimpleNamespace as NS
from unittest.mock import Mock, patch
import unittest

from our_mani.contracts import Pose
from our_mani.multitask import author_aligned_remove_phase_v1116 as subject
from our_mani.multitask.physmani_skill_policy_v548 import SkillGeometryIntents
from our_mani.multitask.physmani_remove_intercept_v667 import InterceptRemoveIntents


class Template:
    manipulated_role = 'selected_cup'

    @property
    def duration_seconds(self):
        raise AssertionError('new acquire completion must not query fitted duration')


def policy_fixture(cls=subject.AuthorAlignedRemoveIntents):
    # Skip only the library-loading constructor. All tested phase/capture and
    # advance methods are the actual source methods, without simulator imports.
    policy = object.__new__(cls)
    policy.task = 'remove_cups_from_rotating_frame_high_speed'
    policy.phase_order = ('approach', 'acquire', 'detach', 'transport', 'place')
    policy.phase_index = 1
    policy.phase_epoch = 10.
    policy.origin = Pose([0., 0., 0.])
    policy.selected = dict(phase='acquire', reference_id='spoke1', manipulated_ids=('cup1',),
        selected_at=9.55, pose=Pose([20., 0., 0.]), opening=0., template=Template())
    policy.committed = policy.selected
    policy.finished = False
    policy.events = []
    policy.close_witness = dict(phase_index=1, issued_at=9.95, confirmed_after=10.,
                                manipulated_ids=['cup1'])
    robot = NS(timestamp=10., pose=Pose([0., 0., 0.]), contact=True, gripper_open=False)
    metadata = dict(invalid_mode_ids=[], bodies={
        'cup1': dict(motion_mode='attached', mode_provenance='caller_controller_touch_action_proprioception'),
        'cup0': dict(motion_mode='external'), 'spoke1': dict(motion_mode='external')})
    controller = NS(phase_index=1, measured_attachments={'cup1': {'already_measured_public_transform': True}})
    policy.frontend = NS(state=NS(robot=robot, perception_metadata=metadata,
                                 role_bindings={'selected_cup': ('cup1',)}),
                         actor=NS(controller=controller))
    policy._completion_target = Mock(side_effect=AssertionError('new acquire must not query current spoke target'))
    policy.remove_rule = dict(phase_positions=dict(approach=.05, detach=.025, transport=.05, place=.03))
    if cls is subject.AuthorAlignedRemoveIntents:
        policy._remove_phase_trace_v1116 = subject._AcquisitionTrace(policy.task, 1000)
    else:
        policy.receipt = MethodType(lambda self: dict(schema='original_fixture_receipt'), policy)
    return policy


class RemoveCaptureTransitionTests(unittest.TestCase):
    def test_original_capture_true_ignores_far_spoke_pose_and_minimum_duration(self):
        policy = policy_fixture()
        original = SkillGeometryIntents._public_capture
        calls = []

        def once(goal):
            calls.append(goal)
            return original(policy, goal)

        policy._public_capture = once
        selected = policy.selected
        self.assertTrue(policy._done(selected))
        self.assertEqual(calls, [selected])
        self.assertIs(policy.selected, selected)
        self.assertEqual(policy.selected['selected_at'], 9.55)
        self.assertEqual(policy.selected['opening'], 0.)
        policy._completion_target.assert_not_called()

    def test_real_original_advance_enters_detach_with_public_capture(self):
        policy = policy_fixture()
        policy._advance()
        self.assertEqual(policy.phase_order[policy.phase_index], 'detach')
        self.assertEqual(policy.frontend.actor.controller.phase_index, 2)
        self.assertIsNone(policy.selected)
        self.assertEqual(policy.events[0]['phase'], 'acquire')
        self.assertFalse(policy.events[0]['native_success_claimed'])
        self.assertEqual(policy._remove_phase_trace_v1116.capture_calls, 1)
        policy._completion_target.assert_not_called()

    def test_contact_false_missing_witness_and_future_witness_cannot_complete(self):
        for change in ('contact_false', 'witness_missing', 'witness_future', 'wrong_phase', 'wrong_identity'):
            with self.subTest(change=change):
                policy = policy_fixture()
                if change == 'contact_false':
                    policy.frontend.state.robot.contact = False
                elif change == 'witness_missing':
                    policy.close_witness = None
                elif change == 'witness_future':
                    policy.close_witness['confirmed_after'] = 10.05
                elif change == 'wrong_phase':
                    policy.close_witness['phase_index'] = 0
                else:
                    policy.close_witness['manipulated_ids'] = ['cup0']
                policy._advance()
                self.assertEqual(policy.phase_index, 1)
                self.assertEqual(policy.events, [])
                policy._completion_target.assert_not_called()

    def test_public_identity_role_provenance_and_attachment_membership_are_retained(self):
        for change in ('invalid', 'ambiguous', 'role', 'provenance', 'attachment_missing'):
            with self.subTest(change=change):
                policy = policy_fixture()
                state = policy.frontend.state
                if change == 'invalid':
                    state.perception_metadata['invalid_mode_ids'] = ['cup1']
                elif change == 'ambiguous':
                    state.perception_metadata['bodies']['cup0']['motion_mode'] = 'attached'
                elif change == 'role':
                    state.role_bindings['selected_cup'] = ('cup0',)
                elif change == 'provenance':
                    state.perception_metadata['bodies']['cup1']['mode_provenance'] = 'untrusted_native_ID'
                else:
                    policy.frontend.actor.controller.measured_attachments.clear()
                self.assertFalse(policy._done(policy.selected))
                policy._completion_target.assert_not_called()

    def test_shared_any_held_alone_remains_insufficient(self):
        policy = policy_fixture()
        policy.frontend.state.perception_metadata['bodies']['cup0']['motion_mode'] = 'attached'
        self.assertTrue(policy.frontend.state.robot.contact)
        self.assertFalse(policy._done(policy.selected))

    def test_nonacquire_dispatches_exact_original_method_and_return(self):
        policy = policy_fixture()
        for phase in ('approach', 'detach', 'transport', 'place'):
            goal = dict(policy.selected, phase=phase)
            sentinel = object()
            with patch.object(InterceptRemoveIntents, '_done', return_value=sentinel) as original:
                self.assertIs(policy._done(goal), sentinel)
                original.assert_called_once_with(goal)
        self.assertEqual(policy._remove_phase_trace_v1116.capture_calls, 0)
        self.assertEqual(policy._remove_phase_trace_v1116.delegated_nonacquire_calls, 4)

    def test_original_nonacquire_remove_position_guard_unchanged(self):
        policy = policy_fixture()
        policy._completion_target = Mock(return_value=Pose([.03, 0., 0.]))
        goal = dict(policy.selected, phase='detach')
        self.assertEqual(policy._done(goal), InterceptRemoveIntents._done(policy, goal))
        self.assertFalse(policy._done(goal))  # original detach gate remains2.5cm
        self.assertEqual(policy._remove_phase_trace_v1116.capture_calls, 0)

    def test_capture_exception_same_instance_not_converted_to_false(self):
        policy = policy_fixture()
        error = RuntimeError('original public predicate failed')
        policy._public_capture = Mock(side_effect=error)
        with self.assertRaises(RuntimeError) as caught:
            policy._done(policy.selected)
        self.assertIs(caught.exception, error)
        policy._public_capture.assert_called_once_with(policy.selected)


class OwnedOriginalInstanceTests(unittest.TestCase):
    def test_scope_uses_same_instance_fields_and_restores_methods(self):
        policy = policy_fixture(InterceptRemoveIntents)
        original_done, original_receipt = policy._done, policy.receipt
        selected = policy.selected
        events = {}
        with subject.owned_remove_phase_scope(policy, events=events) as adapted:
            self.assertIs(adapted, policy)
            self.assertIs(adapted.selected, selected)
            self.assertTrue(adapted._done(selected))
            receipt = adapted.receipt()
            self.assertEqual(receipt['schema'], 'original_fixture_receipt')
            self.assertEqual(receipt['author_aligned_remove_phase_v1116']['capture_calls'], 1)
        self.assertEqual(policy._done, original_done)
        self.assertEqual(policy.receipt, original_receipt)
        self.assertTrue(events['restored'])

    def test_scope_restores_on_exception_and_refuses_nested_scope(self):
        policy = policy_fixture(InterceptRemoveIntents)
        before_done, before_receipt = policy._done, policy.receipt
        events = {}
        error = RuntimeError('body failed')
        with self.assertRaises(RuntimeError) as caught:
            with subject.owned_remove_phase_scope(policy, events=events):
                with self.assertRaisesRegex(RuntimeError, 'cannot nest'):
                    with subject.owned_remove_phase_scope(policy):
                        self.fail('must not enter nested scope')
                raise error
        self.assertIs(caught.exception, error)
        self.assertEqual(policy._done, before_done)
        self.assertEqual(policy.receipt, before_receipt)
        self.assertTrue(events['restored'])

    def test_scope_nonacquire_exact_delegate_and_bounded_trace(self):
        policy = policy_fixture(InterceptRemoveIntents)
        policy._done = Mock(return_value='original-nonacquire-result')
        with subject.owned_remove_phase_scope(policy, max_events=1):
            self.assertEqual(policy._done(dict(policy.selected, phase='transport')), 'original-nonacquire-result')
            self.assertTrue(policy._done(policy.selected))
            self.assertTrue(policy._done(policy.selected))
            trace = policy.receipt()['author_aligned_remove_phase_v1116']
            self.assertEqual(trace['capture_calls'], 2)
            self.assertEqual(trace['dropped_events'], 1)


if __name__ == '__main__':
    unittest.main()
