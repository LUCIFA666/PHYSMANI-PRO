"""CPU integration: no legacy admission callback can veto a dispatched intent."""
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch
import numpy as np

from our_mani.contracts import Pose, RobotState
from our_mani.multitask import author_aligned_geometry_integration_v1114b as integration
from tests import test_author_aligned_geometry_executor_v1114 as original_fixture

ACTION, CONFIG, Env = original_fixture.ACTION, original_fixture.CONFIG, original_fixture.Env


def robot(timestamp):
    return RobotState(timestamp, Pose([0.,0.,0.]), np.zeros(3), np.zeros(3),
                      np.zeros(7), np.zeros(7))


class Policy:
    def __init__(self, env):
        self.env = env; self.phase_index = 0; self.close_witness = None
        self.selected = dict(pose=Pose.from_xyzw(ACTION[:7]), opening=float(ACTION[7]),
                             selected_at=float(env.time), phase='acquire', manipulated_ids=['cup'])
        self.frontend = NS(state=NS(robot=robot(env.time)), actor=NS(requested_opening=1.),
                           observe=self.observe)
        self.admit_pose = Mock(side_effect=AssertionError('legacy pose admission must not run'))
        self.admit_joint = Mock(side_effect=AssertionError('legacy collision predicate must not run'))
        self.qualifies = self.confirms = 0

    def observe(self, observation, observed_robot):
        self.frontend.state = NS(robot=robot(observed_robot.timestamp))

    def qualify_opening(self, requested, current):
        self.qualifies += 1
        return 1., dict(time=float(self.frontend.state.robot.timestamp), phase='acquire',
            desired_opening=requested, current_requested_opening=current, qualified_opening=1.,
            reason='legacy_geometry_defer', native_truth_read=False), None

    def confirm_opening(self, opening, proof, proposed, *, confirmed_after):
        self.confirms += 1
        if proposed is not None:
            self.close_witness = dict(proposed, confirmed_after=confirmed_after)
        self.frontend.actor.requested_opening = opening

    def receipt(self):
        return dict(own_close_witness=deepcopy(self.close_witness))


class Tests(unittest.TestCase):
    def setUp(self):
        # The helper installs only fake pyrep import modules; no Native launch.
        self.fixture = original_fixture.Tests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)

    def test_facade_uses_author_opening_and_original_native_calls_without_legacy_veto(self):
        expected_env, actual_env = Env(), Env()
        expected = self.fixture.run_action(expected_env)
        policy = Policy(actual_env); hook = Mock()
        frontend, dispatched, executor = integration.construct(CONFIG, policy.frontend, policy,
                                                              on_confirmed_opening=hook)
        result = self.fixture.run_action(actual_env, executor)
        self.fixture.equal_native_effects(expected_env, actual_env)
        self.assertEqual(result, expected)
        self.assertIs(frontend, policy.frontend); self.assertIs(dispatched.source_policy, policy)
        self.assertIsNone(executor.admit_joint_target)
        self.assertEqual(executor.collision_diagnostic_limit, 0)
        policy.admit_pose.assert_not_called(); policy.admit_joint.assert_not_called()
        self.assertEqual(policy.qualifies, 10); self.assertEqual(policy.confirms, 10)
        self.assertEqual(actual_env.ledger.grasp_requests, 1)
        self.assertEqual(hook.call_count, 10)
        close = [c for c in hook.call_args_list if c.args[3] is not None]
        self.assertEqual(len(close), 1)
        self.assertEqual(close[0].args[2]['qualified_opening'], 0.)
        self.assertEqual(close[0].args[2]['legacy_qualification']['proof']['reason'], 'legacy_geometry_defer')
        self.assertIsNotNone(policy.close_witness)

    def test_protocol_mismatch_is_not_a_task_failure_and_no_native_command_is_dispatched(self):
        for update in ({'selected_at':-.05}, {'opening':1.}, {'pose':Pose([.1,0.,1.])}):
            env=Env(); policy=Policy(env); policy.selected.update(update)
            _, _, executor = integration.construct(CONFIG, policy.frontend, policy)
            with self.assertRaises(integration.ProtocolInvariantError):
                self.fixture.run_action(env, executor)
            self.assertEqual(env.plan_calls, []); self.assertEqual(env.ledger.ticks, 0)
            policy.admit_pose.assert_not_called(); policy.admit_joint.assert_not_called()

    def test_admission_rejects_native_duck_getters_without_calling_them(self):
        env=Env(); policy=Policy(env)
        native_pose = NS(xyzw=Mock(side_effect=AssertionError('Native pose getter forbidden')))
        policy.selected['pose'] = native_pose
        accepted, proof = integration.protocol_pose_admission(policy)(ACTION)
        self.assertFalse(accepted); native_pose.xyzw.assert_not_called()
        self.assertEqual(proof['reason'], 'original_public_robot_and_pose_DTOs_required')

    def test_admission_depends_only_on_plain_selected_pose_opening_and_current_public_robot(self):
        env=Env(); policy=Policy(env)
        policy.frontend.actor = None  # Even collision/model/controller access is unnecessary.
        accepted, proof = integration.protocol_pose_admission(policy)(ACTION.copy())
        self.assertTrue(accepted)
        self.assertFalse(proof['collision_predicate_called']); self.assertTrue(proof['protocol_only'])
        policy.selected['selected_at'] = float('nan')
        self.assertFalse(integration.protocol_pose_admission(policy)(ACTION)[0])

    def test_constructor_refuses_different_frontend_and_binding_requires_frozen_actuator_sha(self):
        env=Env(); policy=Policy(env)
        with self.assertRaises(integration.ProtocolInvariantError):
            integration.construct(CONFIG, NS(observe=lambda *a:None), policy)
        with TemporaryDirectory() as temporary:
            root=Path(temporary)
            path=root/integration.FROZEN_EXECUTOR_SOURCE; path.parent.mkdir(parents=True)
            path.write_text('unexpected implementation')
            with patch.object(integration, 'executor_binding') as old_binding:
                with self.assertRaisesRegex(integration.ProtocolInvariantError, 'frozen v1114'):
                    integration.binding(root, root)
                old_binding.assert_not_called()


if __name__ == '__main__':
    unittest.main()
