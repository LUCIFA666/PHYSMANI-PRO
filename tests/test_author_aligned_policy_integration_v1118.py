"""Exercise integration ordering through the original complete actuator MRO."""
from copy import deepcopy
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch
import numpy as np

from our_mani.contracts import ObjectState, Pose, RobotState
from our_mani.multitask.mode_observer import PublicModeObserver
from our_mani.multitask import author_aligned_policy_integration_v1118 as new
from tests import test_author_aligned_geometry_executor_v1114 as fixture

IDS = ('cup0', 'cup1', 'cup2')
TASK = 'beat_the_rotating_buzz'


def robot(time, *, contact=False, gripper_open=True):
    return RobotState(time, Pose([0., 0., 1.]), np.zeros(3), np.zeros(3),
                      np.zeros(7), np.zeros(7), contact=contact, gripper_open=gripper_open)


def bodies(time):
    return [ObjectState(identity, time, Pose([distance, 0., 1.]),
        np.array([[0.,0.,0.], [.01,0.,0.], [0.,.01,0.]]), np.zeros(3), np.zeros(3))
        for identity, distance in zip(IDS, (.02, .10, .09))]


class Frontend:
    def __init__(self, env):
        self.env = env
        self.actor = NS(mode_observer=PublicModeObserver(IDS, IDS, .15),
                        previous_states=bodies(env.time), requested_opening=1.)
        self.state = NS(robot=robot(env.time))
        self.observed = []

    def observe(self, observation, observed_robot):
        # Deliberately keep the physical aperture open even after the close
        # command: public command context must not be inferred from aperture.
        r = robot(observed_robot.timestamp, contact=observed_robot.timestamp >= .65)
        result = self.actor.mode_observer.observe(self.actor.previous_states, r)
        self.observed.append(result)
        self.actor.previous_states = bodies(r.timestamp)
        self.state = NS(robot=r)
        return result


class Policy:
    def __init__(self, frontend):
        self.frontend = frontend
        self.phase_index = 1
        self.phase_order = ('approach', 'acquire', 'transport')
        self.selected = None
        self.close_witness = None
        self.finished = False

    def select(self):
        self.selected = dict(pose=Pose.from_xyzw(fixture.ACTION[:7]), opening=0.,
            selected_at=float(self.frontend.state.robot.timestamp), phase=self.phase_order[self.phase_index],
            manipulated_ids=('cup1',))
        return fixture.ACTION.copy()

    def qualify_opening(self, requested, current):
        return 1., dict(time=float(self.frontend.state.robot.timestamp), phase=self.selected['phase'],
            desired_opening=requested, current_requested_opening=current, qualified_opening=1.,
            reason='legacy_SE3_defer', native_truth_read=False), None

    def confirm_opening(self, opening, proof, proposed, *, confirmed_after):
        self.frontend.actor.requested_opening = opening
        if proposed is not None:
            self.close_witness = dict(proposed, confirmed_after=confirmed_after)

    def receipt(self):
        return dict(own_close_witness=deepcopy(self.close_witness))


class Tests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixture.Tests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.env = fixture.Env()
        self.frontend = Frontend(self.env)
        self.policy = Policy(self.frontend)
        self.original_mode = self.frontend.actor.mode_observer
        self.original_observe = self.frontend.observe
        self.original_select = self.policy.select
        self.factory = patch.object(new.original_components, 'construct', return_value=(
            self.frontend, self.policy, NS()))
        self.factory.start()
        self.addCleanup(self.factory.stop)

    def scope(self, task=TASK):
        return new.integrated_components(None, fixture.CONFIG, task, self.frontend.actor)

    def test_actual_confirmation_precedes_observe_and_delayed_contact_uses_intent_not_nearest(self):
        with self.scope() as (frontend, policy, executor):
            self.assertEqual(executor.collision_diagnostic_limit, 0)
            self.assertIsNone(executor.admit_joint_target)
            policy.select()
            self.fixture.run_action(self.env, executor)
            observer = frontend.actor.mode_observer
            self.assertIsNone(observer.held)
            self.assertIsNotNone(observer.pending_close)
            self.assertEqual(observer.pending_close['identity'], 'cup1')
            self.assertAlmostEqual(observer.pending_close['issued_at'], .45)
            self.assertAlmostEqual(observer.pending_close['confirmed_after'], .5)
            policy.select()
            self.fixture.run_action(self.env, executor)
            self.assertEqual(observer.held, 'cup1')
            self.assertTrue(frontend.state.robot.gripper_open)
            receipt = policy.receipt()['author_aligned_policy_integration_v1118']
            self.assertEqual(receipt['maximum_attachment_candidate_distance_m'], .15)
            self.assertFalse(receipt['native_held_identity_verified'])
            events = receipt['public_mode']['events']
            association = next(e for e in events if e.get('decision') == 'associate_using_confirmed_public_action_intent')
            self.assertEqual(association['nearest_candidate_evidence']['identity'], 'cup0')
            self.assertEqual(association['held_after'], 'cup1')
            self.assertGreater(association['timestamp'], .5)
        self.assertIs(self.frontend.actor.mode_observer, self.original_mode)
        self.assertEqual(self.frontend.observe, self.original_observe)
        self.assertEqual(self.policy.select, self.original_select)

    def test_reselect_phase_change_cancels_pending_before_next_native_action(self):
        with self.scope() as (_, policy, executor):
            policy.select()
            self.fixture.run_action(self.env, executor)
            observer = self.frontend.actor.mode_observer
            self.assertIsNotNone(observer.pending_close)
            self.policy.phase_index = 2
            policy.select()
            self.assertIsNone(observer.pending_close)
            self.assertEqual(self.env.ledger.ticks, 10)

    def test_without_confirmed_edge_ambiguous_public_contact_remains_unknown(self):
        with self.scope() as (frontend, policy, _):
            self.frontend.actor.requested_opening = 0.
            policy.select()
            frontend.observe({}, NS(timestamp=.7))
            self.assertIsNone(frontend.actor.mode_observer.held)
            self.assertEqual(frontend.observed[-1][1], set(IDS))

    def test_owned_adapters_restore_on_exception(self):
        with self.assertRaisesRegex(RuntimeError, 'test failure'):
            with self.scope() as (_, policy, _):
                policy.select()
                raise RuntimeError('test failure')
        self.assertIs(self.frontend.actor.mode_observer, self.original_mode)
        self.assertEqual(self.frontend.observe, self.original_observe)
        self.assertEqual(self.policy.select, self.original_select)

    def test_remove_scope_is_active_for_entire_executor_lifetime(self):
        entered = []
        from contextlib import contextmanager
        @contextmanager
        def phase_scope(policy):
            entered.append(True)
            try:
                yield policy
            finally:
                entered.append(False)
        with patch.object(new, 'owned_remove_phase_scope', phase_scope):
            with self.scope('remove_cups_from_rotating_frame') as (_, policy, _):
                self.assertEqual(entered, [True])
                policy.receipt()
            self.assertEqual(entered, [True, False])

    def test_button_without_skill_callbacks_keeps_original_actuator(self):
        for name in ('qualify_opening', 'confirm_opening'):
            setattr(self.policy, name, None)
        with self.scope('push_moving_button') as (_, policy, executor):
            self.assertIs(policy, self.policy)
            self.assertIsNone(executor.skill_policy)
            self.assertEqual(executor.collision_diagnostic_limit, 0)
            self.assertFalse(policy.receipt()['author_aligned_policy_integration_v1118']['remove_phase_changed'])
            policy.select()
            self.fixture.run_action(self.env, executor)
            self.assertEqual(self.env.ledger.ticks, 10)
        self.assertNotIn('author_aligned_policy_integration_v1118', self.policy.receipt())


if __name__ == '__main__':
    unittest.main()
