from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace as NS
import json
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
import test_remove_retarget_v66 as ports
from our_mani.contracts import Pose, RobotState, ObjectState
from our_mani.dynamics import ConstantVelocityDynamics
from our_mani.reachability import TimedReachability
from our_mani.multitask.interaction import (InteractionLibrary, CandidateGenerator, CandidateConfig,
    InteractionController, InteractionCandidate, SE3InteractionMPC, MPCConfig, PolicyState, _world_target, GripperTimingConfig)
from our_mani.multitask.feedback_interaction import ObservedFeedbackController
from our_mani.multitask.causal_retarget_interaction import CausalRetargetManager
from our_mani.multitask.high_peg_primary_fitting_v231 import PrimarySequenceLibrary
from our_mani.multitask.candidate_success import FEATURE_NAMES
from our_mani.multitask.ring_causal_retarget_v371 import binding, install

ROOT = Path(__file__).resolve().parents[1]


class Tests(unittest.TestCase):
    def fixture(self, task='insert_onto_rotating_peg_high_speed'):
        c = json.loads((ROOT/'artifacts/framework/multitask_r10/high_peg_alignment_v355_setup/configuration.json').read_text())
        cls = PrimarySequenceLibrary if task.endswith('_high_speed') else InteractionLibrary
        library = cls.load(ROOT/c['learned_artifacts'][task]['interaction']['path'])
        template = library.for_phase('align')[0]
        model, scene = ports.Cartesian(), ports.Scene()
        limits = dict(c['candidate_config'], maximum_linear_speed=100., maximum_angular_speed=100.)
        gate = TimedReachability(model)
        generator = CandidateGenerator(library, gate, scene, CandidateConfig(**limits))
        controller = ObservedFeedbackController.from_controller(InteractionController(library, generator,
            SE3InteractionMPC(MPCConfig(**c['mpc_config'])), predictor='cv', ranking='geometry',
            continuation_sha256='CPU_ring_retarget_fixture',
            gripper_timing_config=GripperTimingConfig(**c['gripper_timing_config'])), {'phase_tracking_slack_seconds': 0.})
        controller.phase_index = library.phase_order.index('align')
        target = _world_target(Pose([0., 0., 0.]), np.zeros(3), np.zeros(3), template, 1.)[0]
        self.q = model.inverse(Pose(target.position+[.15, 0., 0.], target.rotation), np.zeros(7))
        state, forecast = self.observed(0.)
        checked = gate.check_joints(state.robot, model.inverse(target, self.q), np.zeros(7), .3)
        controller.goal = InteractionCandidate('CPU_ring_align', 'align', controller.phase_index,
            'peg0', ('ring',), template, 0., .3, target, np.zeros(3), np.zeros(3),
            Pose([0., 0., 0.]), np.zeros(3), np.zeros(3), state.bodies['peg0'], checked, 0.,
            template.duration_seconds, template.duration_budget_seconds, .85, True, {})
        controller.attempts.begin('align', 'CPU_ring_align', 0., np.zeros(len(FEATURE_NAMES)))
        manager = CausalRetargetManager(controller, dict(minimum_issue_interval_seconds=.1,
            scheduled_issue_interval_seconds=.1, reference_shift_threshold_m=.0001,
            arrival_delays_seconds=(.3, .15, .05)))
        actor = NS(continuous_reference_manager=NS(retarget=manager), metrics_report=lambda: {})
        c['ring_causal_retarget_v371'] = {task: binding(ROOT, c, task)}
        install(actor, c, ROOT, task)
        return controller, manager, model, scene

    def observed(self, time):
        robot = RobotState(time, Pose(self.q[:3], Rotation.from_rotvec(self.q[3:6]).as_matrix()),
            np.zeros(3), np.zeros(3), self.q.copy(), np.zeros(7), gripper_open=False, contact=True)
        body = ObjectState('peg0', time, Pose([.1*time, 0., 0.]), np.tile([.1*time, 0., 0.], (3, 1)),
            np.array([.1, 0., 0.]), np.zeros(3))
        state = PolicyState(robot, {'peg0': body}, {'bodies': {'peg0': dict(directly_visible=True,
            registration_valid=True, orientation_observable=True, motion_mode='external')}},
            {'selected_peg': ('peg0',), 'ring': ('ring',)})
        forecast = ConstantVelocityDynamics().predict([body], 1.1, .05)
        forecast.metadata['observed_until'] = float(time)
        return state, forecast

    def test_both_original_ring_libraries_reissue_without_resetting_attempt_or_deadline(self):
        for task in ('insert_onto_rotating_peg', 'insert_onto_rotating_peg_high_speed'):
            c, manager, model, scene = self.fixture(task)
            old, records = c.goal, deepcopy(c.attempts.to_report())
            state, forecast = self.observed(.15)
            event = manager.before_step(state, forecast, 9.85)
            self.assertEqual(event['action'], 'new_candidate_issue', event)
            self.assertIs(c.goal.template, old.template)
            self.assertEqual((c.goal.selected_at, c.goal.phase_budget, c.goal.suffix_budget),
                             (old.selected_at, old.phase_budget, old.suffix_budget))
            self.assertEqual(c.attempts.to_report(), records)
            self.assertGreater(model.queries, 0)
            self.assertTrue(scene.bindings)
            self.assertFalse(event['phase_completion_claimed'])

    def test_collision_rejection_retains_original_goal(self):
        c, manager, _, scene = self.fixture()
        old = c.goal
        scene.blocked = True
        state, forecast = self.observed(.15)
        event = manager.before_step(state, forecast, 9.85)
        self.assertEqual(event['action'], 'reissue_rejected_original_candidate_retained')
        self.assertIs(c.goal, old)

    def test_stale_or_changed_mode_cannot_create_another_commitment(self):
        for stale in (True, False):
            c, manager, _, _ = self.fixture()
            old = c.goal
            state, forecast = self.observed(.15)
            if stale:
                forecast.metadata['observed_until'] = -.2
            else:
                state.perception_metadata['bodies']['peg0']['motion_mode'] = 'attached'
            event = manager.before_step(state, forecast, 9.85)
            self.assertEqual(event['action'], 'reissue_rejected_original_candidate_retained')
            self.assertIs(c.goal, old)

    def test_expired_suffix_budget_cannot_be_renewed(self):
        c, manager, _, _ = self.fixture()
        old = c.goal
        state, forecast = self.observed(.15)
        self.assertIsNone(manager.before_step(state, forecast, .85))
        self.assertIs(c.goal, old)


if __name__ == '__main__':
    unittest.main()
