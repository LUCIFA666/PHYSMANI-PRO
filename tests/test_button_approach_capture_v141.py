from types import SimpleNamespace
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
import test_button_intent_actor_v74 as original_factory
from our_mani.contracts import Pose,RobotState,ObjectState
from our_mani.multitask.interaction import PolicyState,_world_target
from our_mani.multitask.button_approach_capture_v141 import install


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
    def test_original_30cm_guard_alone_cannot_start_press(self):
        state,target=self.state(.05)
        self.assertFalse(self.actor.controller._phase_done(state,target,public_geometry=True))
        self.assertEqual(self.actor.controller.phase_index,0)
    def test_original_fitted_action_caps_allow_the_state_transition(self):
        state,target=self.state(.004,.02)
        self.assertTrue(self.actor.controller._phase_done(state,target,public_geometry=True))
        self.actor.controller._complete_phase(state);self.assertEqual(self.actor.controller.phase_index,1)
        self.assertFalse(self.actor.metrics_report()['button_author_intent_v74']['source_state_events'][0]['verified_completed_skill_claimed'])
    def test_approach_orientation_remains_part_of_the_guard(self):
        state,target=self.state(.004,.04)
        self.assertFalse(self.actor.controller._phase_done(state,target,public_geometry=True))
    def test_press_geometry_cannot_replace_native_success(self):
        state,target=self.state(0.,0.,phase='press')
        self.assertFalse(self.actor.controller._phase_done(state,target,public_geometry=True))


if __name__=='__main__':unittest.main()
