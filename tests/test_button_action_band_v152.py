from types import SimpleNamespace
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
import test_button_intent_actor_v74 as original_factory
from our_mani.contracts import Pose,RobotState,ObjectState
from our_mani.multitask.interaction import PolicyState,_world_target
from our_mani.multitask.button_action_band_v152 import install


class Tests(unittest.TestCase):
    def setUp(self):
        self.fixture=original_factory.Tests();self.fixture.setUp();self.actor=self.fixture.actor();install(self.actor)
    def tearDown(self):self.fixture.tearDown()
    def state(self,offset,angle=0.,phase='approach'):
        c=self.actor.controller;template=c.library.for_phase(phase)[0]
        body=ObjectState('button',1.,Pose([.2,0.,.8]),np.tile([.2,0.,.8],(3,1)),np.zeros(3),np.zeros(3))
        target=_world_target(body.pose,np.zeros(3),np.zeros(3),template,1.)[0]
        pose=Pose(target.position+np.array([offset,0.,0.]),Rotation.from_rotvec([0.,0.,angle]).as_matrix()@target.rotation)
        robot=RobotState(1.,pose,np.zeros(3),np.zeros(3),np.zeros(7),np.zeros(7),gripper_open=True,contact=False)
        state=PolicyState(robot,{'button':body},{'bodies':{'button':dict(directly_visible=True,registration_valid=True,motion_mode='external')}},{'button':('button',)})
        c.goal=SimpleNamespace(phase=phase,template=template,reference_id='button')
        return state,target
    def test_original_30cm_guard_alone_is_still_insufficient(self):
        state,target=self.state(.05)
        self.assertFalse(self.actor.controller._phase_done(state,target,public_geometry=True))
    def test_one_step_of_admitted_target_motion_is_part_of_the_action_guard(self):
        state,target=self.state(.04,.12)
        self.assertTrue(self.actor.controller._phase_done(state,target,public_geometry=True))
        event=self.actor.metrics_report()['button_approach_capture_v152']['events'][-1]
        self.assertAlmostEqual(event['position_cap_m'],.045);self.assertAlmostEqual(event['orientation_cap_rad'],.13)
        self.assertEqual(event['original_action_position_error_cap_m'],.005)
    def test_orientation_beyond_original_error_plus_one_step_is_rejected(self):
        state,target=self.state(.004,.14)
        self.assertFalse(self.actor.controller._phase_done(state,target,public_geometry=True))
    def test_press_geometry_cannot_replace_native_success(self):
        state,target=self.state(0.,0.,phase='press')
        self.assertFalse(self.actor.controller._phase_done(state,target,public_geometry=True))


if __name__=='__main__':unittest.main()
