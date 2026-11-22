import test_physmani_skill_v533 as previous
from our_mani.multitask.physmani_skill_policy_v548 import SkillGeometryIntents
from types import SimpleNamespace as NS
from our_mani.contracts import Pose
from dataclasses import replace
import numpy as np
import unittest


class Tests(previous.Tests):
    def fixture(self,phase='acquire'):
        _,f=super().fixture(phase)
        f.actor.controller.gripper_timing_config=NS(close_position_tolerance_m=.005,close_orientation_tolerance_rad=.03)
        p=SkillGeometryIntents(f);p.phase_index=p.phase_order.index(phase);p.select()
        return p,f

    def test_inactive_controller_intersection_is_not_the_new_RML_policy_grasp_gate(self):
        p,f=self.fixture();target=p._completion_target(p.selected)
        self.move(f,Pose(target.position+[.015,0,0],target.rotation))
        command,proof,edge=p.qualify_opening(0.,1.)
        self.assertEqual(command,0.);self.assertIsNotNone(edge)
        self.assertEqual(f.actor.controller.gripper_timing_config.close_position_tolerance_m,.005)
        self.assertFalse(p.receipt()['inactive_old_controller_gripper_gate_used'])
        self.move(f,Pose(target.position+[.021,0,0],target.rotation))
        self.assertEqual(p.qualify_opening(0.,1.)[0],1.)


if __name__=='__main__':unittest.main()
