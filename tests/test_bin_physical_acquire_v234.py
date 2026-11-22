from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace as NS, MethodType
import json
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from our_mani.contracts import Pose, RobotState
from our_mani.multitask.interaction import InteractionController, InteractionLibrary, GripperTimingConfig, _world_target
from our_mani.multitask.cup_physical_acquire_v167 import install_confirmation
from our_mani.multitask.bin_physical_acquire_v234 import install_completion, binding

ROOT = Path(__file__).resolve().parents[1]


class Tests(unittest.TestCase):
    def setUp(self):
        protocol = json.loads((ROOT/'artifacts/framework/multitask_r10/joint16_bin_smoke_v219_setup/configuration.json').read_text())
        task = 'put_rubbish_in_moving_bin'
        self.binding = binding(ROOT, protocol, task)
        library = InteractionLibrary.load(ROOT/protocol['learned_artifacts'][task]['interaction']['path'])
        template = library.for_phase('acquire')[0]
        reference = Pose([.1, .2, .77])
        target = _world_target(reference, np.zeros(3), np.zeros(3), template, 1.)[0]
        actual = Pose(target.position+[.009, 0., 0.], Rotation.from_rotvec([0., 0., .06]).as_matrix()@target.rotation)
        now = 2.+template.duration_seconds+.05
        robot = RobotState(now, actual, np.zeros(3), np.zeros(3), np.zeros(7), np.zeros(7), gripper_open=True, contact=False)
        self.state = NS(robot=robot, bodies={}, role_bindings={template.manipulated_role: ('rubbish',)},
            perception_metadata={'bodies': {'rubbish': {'motion_mode': 'attached',
                'mode_provenance': 'caller_controller_touch_action_proprioception'}}})
        goal = NS(phase='acquire', candidate_id='actual-bin-close', reference_id='rubbish',
            manipulated_ids=('rubbish',), selected_at=2., template=template)
        self.c = NS(goal=goal, phase_index=1, origin=reference,
            gripper_timing_config=GripperTimingConfig(.02, .1), _close_edge_issued_at=now,
            generator=NS(observed_reference=lambda *args: NS(pose=reference)))
        self.c._public_acquire_contact = MethodType(InteractionController._public_acquire_contact, self.c)
        self.c._phase_done = MethodType(InteractionController._phase_done, self.c)
        self.m = NS(controller=self.c, current_state=self.state, last_confirmed={'opening': 1.},
            capture_stopped=lambda robot: True, qualification_failed=False, confirm_actual_issued=lambda **kwargs: None)
        self.witnesses, self.events = {}, []
        install_confirmation(self.m, self.witnesses, self.events)
        install_completion(self.c, self.witnesses, self.events)
        self.target = target

    def dispatch(self, passed=True):
        now = self.state.robot.timestamp
        self.m.confirm_actual_issued(time=now+.05, command={'opening': 0.}, execution={
            'actual_issued_at': now, 'actual_window_velocity_within_limits': passed,
            'actual_window_acceleration_within_limits': passed})
        self.state.robot = replace(self.state.robot, timestamp=now+.05, contact=True)

    def test_actual_physical_grasp_requires_ack_and_unique_identity(self):
        self.assertFalse(self.c._phase_done(self.state, self.target))
        self.dispatch()
        self.assertTrue(self.c._phase_done(self.state, self.target))
        self.assertFalse(InteractionController._phase_done(self.c, self.state, self.target))
        self.state.perception_metadata['bodies']['tomato0'] = {'motion_mode': 'attached'}
        self.assertFalse(self.c._phase_done(self.state, self.target))

    def test_actual_motion_failure_or_missing_contact_cannot_complete(self):
        self.dispatch(False)
        self.assertFalse(self.witnesses)
        self.assertFalse(self.c._phase_done(self.state, self.target))

    def test_original_physical_geometry_and_stop_still_required(self):
        self.m.capture_stopped = lambda robot: False
        self.dispatch()
        self.assertFalse(self.witnesses)
        self.m.capture_stopped = lambda robot: True
        self.state.robot = replace(self.state.robot, timestamp=self.c._close_edge_issued_at,
            contact=False, pose=Pose(self.target.position+[.021, 0., 0.], self.target.rotation))
        self.dispatch()
        self.assertFalse(self.witnesses)

    def test_binding_discloses_control_change_and_preserves_fit_caps(self):
        self.assertTrue(self.binding['extra_controller_intersection_relaxed'])
        self.assertTrue(self.binding['original_fitting_caps_unchanged'])
        self.assertEqual(self.c.goal.template.position_tolerance_m, .005)


if __name__ == '__main__': unittest.main()
