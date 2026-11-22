from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace as NS, MethodType
import json
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from our_mani.contracts import Pose, RobotState
from our_mani.multitask.interaction import InteractionController, GripperTimingConfig, _world_target, COMMAND_GOAL
from our_mani.multitask.normal_cup_primary_library_v109 import PrimarySequenceLibrary
from our_mani.multitask.cup_physical_acquire_v167 import install_confirmation, install_completion

ROOT = Path(__file__).resolve().parents[1]


class Tests(unittest.TestCase):
    def setUp(self):
        protocol = json.loads((ROOT/'artifacts/framework/multitask_r10/cup_transport_v163_setup/configuration.json').read_text())
        library = PrimarySequenceLibrary.load(ROOT/protocol['learned_artifacts']['place_cups_on_rotating_frame']['interaction']['path'])
        template = library.for_phase('acquire')[0]
        now = 2.+template.duration_seconds+.05
        reference = Pose([.2, .1, .8])
        target = _world_target(reference, np.zeros(3), np.zeros(3), template, 1.)[0]
        # Valid original physical grasp, intentionally outside fitted .03rad cap.
        actual = Pose(target.position+[.006, 0., 0.], Rotation.from_rotvec([0., 0., .06]).as_matrix() @ target.rotation)
        robot = RobotState(now, actual, np.zeros(3), np.zeros(3), np.zeros(7), np.zeros(7), gripper_open=True, contact=False)
        self.state = NS(robot=robot, bodies={}, role_bindings={'cup0': ('cup0',)},
            perception_metadata={'bodies': {'cup0': {'motion_mode': 'attached', 'mode_provenance': 'caller_controller_touch_action_proprioception'}}})
        goal = NS(phase='acquire', candidate_id='actual-capture', reference_id='cup0', manipulated_ids=('cup0',), selected_at=2., template=template)
        self.c = NS(goal=goal, phase_index=1, origin=Pose([0., 0., 0.]),
            gripper_timing_config=GripperTimingConfig(.02, .1), _close_edge_issued_at=now,
            generator=NS(observed_reference=lambda *args: NS(pose=reference)))
        self.c._public_acquire_contact = MethodType(InteractionController._public_acquire_contact, self.c)
        self.c._phase_done = MethodType(InteractionController._phase_done, self.c)
        self.m = NS(controller=self.c, current_state=self.state, last_confirmed={'opening': 1.},
            capture_stopped=lambda robot: True, qualification_failed=False,
            confirm_actual_issued=lambda **kwargs: None)
        self.clock = NS(epochs={})
        self.witnesses, self.events = {}, []
        self.target, self.library = target, library
        install_confirmation(self.m, self.witnesses, self.events)
        install_completion(self.c, self.clock, self.witnesses, self.events)

    def dispatch(self, motion_pass=True):
        t = self.state.robot.timestamp
        self.m.confirm_actual_issued(time=t+.05, command={'opening': 0.}, execution={
            'actual_issued_at': t, 'actual_window_velocity_within_limits': motion_pass,
            'actual_window_acceleration_within_limits': motion_pass})
        self.state.robot = replace(self.state.robot, timestamp=t+.05, contact=True)

    def test_physical_capture_requires_actual_acknowledgement_then_unique_contact(self):
        self.assertFalse(self.c._phase_done(self.state, self.target))
        self.dispatch()
        self.assertEqual(len(self.witnesses), 1)
        self.assertTrue(self.c._phase_done(self.state, self.target))
        self.state.robot = replace(self.state.robot, contact=False)
        self.assertFalse(self.c._phase_done(self.state, self.target))

    def test_original_regression_completion_remains_false_for_this_valid_physical_grasp(self):
        self.dispatch()
        self.assertFalse(InteractionController._phase_done(self.c, self.state, self.target))
        self.assertTrue(self.c._phase_done(self.state, self.target))
        self.assertEqual(self.c.goal.template.orientation_tolerance_rad, .03)

    def test_ambiguous_public_held_identity_cannot_complete(self):
        self.dispatch()
        self.state.perception_metadata['bodies']['cup1'] = {'motion_mode': 'attached'}
        self.assertFalse(self.c._phase_done(self.state, self.target))

    def test_unstopped_or_outside_original_grasp_never_creates_witness(self):
        self.m.capture_stopped = lambda robot: False
        self.dispatch()
        self.assertFalse(self.witnesses)
        self.state.robot = replace(self.state.robot, timestamp=self.c._close_edge_issued_at, contact=False,
            pose=Pose(self.target.position+[.03, 0., 0.], self.target.rotation))
        self.m.capture_stopped = lambda robot: True
        self.dispatch()
        self.assertFalse(self.witnesses)

    def test_original_dispatch_rejection_or_actual_limit_failure_never_creates_witness(self):
        self.dispatch(motion_pass=False)
        self.assertFalse(self.witnesses)
        self.state.robot = replace(self.state.robot, timestamp=self.c._close_edge_issued_at, contact=False)
        def reject(**kwargs): raise ValueError('actual backend rejected')
        self.m.confirm_actual_issued = reject
        install_confirmation(self.m, self.witnesses, self.events)
        with self.assertRaises(ValueError): self.dispatch()
        self.assertFalse(self.witnesses)

    def transport(self):
        template = self.library.for_phase('transport')[0]
        self.c.phase_index = 3
        self.c.goal = NS(template=template, phase='transport', candidate_id='endpoint-after-prefix', selected_at=7.55)
        self.state.robot = replace(self.state.robot, timestamp=8.55, pose=self.target)
        self.clock.epochs[3] = dict(issued_at=6.55, confirmed_after=6.6, deadline=8.85)

    def test_transport_prefix_elapsed_is_counted_without_rewriting_selected_time(self):
        self.transport()
        self.assertFalse(InteractionController._phase_done(self.c, self.state, self.target))
        self.assertTrue(self.c._phase_done(self.state, self.target))
        self.assertEqual(self.c.goal.selected_at, 7.55)

    def test_transport_geometry_and_original_deadline_still_apply(self):
        self.transport()
        wrong = Pose(self.target.position+[.1, 0., 0.], self.target.rotation)
        self.assertFalse(self.c._phase_done(self.state, wrong))
        self.state.robot = replace(self.state.robot, timestamp=8.9)
        self.assertFalse(self.c._phase_done(self.state, self.target))

    def test_transport_unissued_prefix_or_insufficient_elapsed_never_gets_credit(self):
        self.transport()
        self.clock.epochs.clear()
        self.assertFalse(self.c._phase_done(self.state, self.target))
        issued_at = self.state.robot.timestamp-self.c.goal.template.duration_seconds+.01
        self.clock.epochs[3] = dict(issued_at=issued_at, confirmed_after=issued_at+.05, deadline=8.85)
        self.assertFalse(self.c._phase_done(self.state, self.target))


if __name__ == '__main__': unittest.main()
