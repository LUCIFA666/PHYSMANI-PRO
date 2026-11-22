"""Non-symmetric held grasps retain the unchanged public contact/open bits."""
from dataclasses import replace
from types import SimpleNamespace
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from our_mani.contracts import Pose
from our_mani.multitask.interaction import (InteractionController, GripperTimingConfig,
    _observed_reference)
from tests.test_multitask_interaction import synthetic_library, state, candidates_config


def controller():
    c=InteractionController.__new__(InteractionController)
    c.gripper_timing_config=GripperTimingConfig(.02,.10)
    c.gripper_timing_events=[];c.last_diagnostics={}
    c._gripper_goal_token=c._close_edge_issued_at=None
    c.measured_attachments={};c.origin=Pose(np.zeros(3))
    c.attempts=SimpleNamespace(records=[])
    template=replace(synthetic_library('translation_only').templates[0],phase='acquire')
    c.goal=SimpleNamespace(candidate_id='selected-button',selected_at=0.,template=template,
        manipulated_ids=('button',),reference_id='button')
    c.generator=SimpleNamespace(observed_reference=lambda current,identity,t,origin:
        _observed_reference(current,identity,t,origin,candidates_config()))
    return c


def held(time=.6):
    current=state(time,robot_z=0.)
    current=replace(current,robot=replace(current.robot,gripper_open=True,contact=True))
    current.perception_metadata['bodies']['button'].update(motion_mode='attached',
        mode_provenance='caller_controller_touch_action_proprioception')
    return current


class PublicAcquireContactTests(unittest.TestCase):
    def closed_controller(self):
        c=controller()
        self.assertEqual(c._gripper_opening(state(.05,robot_z=0.),0.,Pose(np.zeros(3))),0.)
        return c

    def test_asymmetric_open_proxy_true_does_not_reopen_own_uniquely_associated_grasp(self):
        c=self.closed_controller();current=held()
        self.assertTrue(current.robot.gripper_open)
        self.assertTrue(c._public_acquire_contact(current))
        self.assertEqual(c._gripper_opening(current,1.,Pose(np.zeros(3))),0.)
        self.assertEqual(c.gripper_timing_events[-1]['reason'],'preserve_public_contact_while_verifying_acquire')
        self.assertTrue(c._phase_done(current,Pose(np.zeros(3))))

    def test_no_own_close_does_not_create_a_completion(self):
        c=controller();current=held()
        self.assertFalse(c._public_acquire_contact(current))
        self.assertFalse(c._phase_done(current,Pose(np.zeros(3))))

    def test_missing_contact_does_not_create_a_completion(self):
        c=self.closed_controller();current=held()
        current=replace(current,robot=replace(current.robot,contact=False))
        self.assertFalse(c._phase_done(current,Pose(np.zeros(3))))

    def test_wrong_or_ambiguous_body_association_cannot_complete(self):
        for mode in ('wrong_body','ambiguous_role','ambiguous_mode','invalid_mode'):
            with self.subTest(mode=mode):
                c=self.closed_controller();current=held()
                if mode=='wrong_body':
                    current.perception_metadata['bodies']['button']['motion_mode']='external'
                    current.perception_metadata['bodies']['other']={'motion_mode':'attached'}
                elif mode=='ambiguous_role':
                    current=replace(current,role_bindings={'button':('button','other')})
                elif mode=='ambiguous_mode':
                    current.perception_metadata['bodies']['other']={'motion_mode':'attached'}
                else:
                    current.perception_metadata['invalid_mode_ids']=['button']
                self.assertFalse(c._public_acquire_contact(current))
                self.assertFalse(c._phase_done(current,Pose(np.zeros(3))))

    def test_stale_or_unregistered_body_without_measured_attachment_fails(self):
        for key in ('directly_visible','registration_valid'):
            c=self.closed_controller();current=held()
            current.perception_metadata['bodies']['button'][key]=False
            self.assertFalse(c._public_acquire_contact(current))

    def test_contact_does_not_relax_pose_or_angle_caps(self):
        c=self.closed_controller();current=held()
        self.assertFalse(c._phase_done(current,Pose(np.array([.2,0.,0.]))))
        self.assertFalse(c._phase_done(current,Pose(np.zeros(3),Rotation.from_euler('z',.3).as_matrix())))


if __name__=='__main__':
    unittest.main()
