from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace as NS
import json
import unittest
import numpy as np
from our_mani.contracts import Pose,RobotState,ObjectState
from our_mani.multitask.physmani_block_policy_v538 import BlockPublicFrontend,BlockGeometryIntents,TerminalBlockView

ROOT=Path(__file__).resolve().parents[1]


class Tests(unittest.TestCase):
    def fixture(self):
        config=json.loads((ROOT/'artifacts/framework/multitask_r10/joint16_block_support_v318_setup/configuration.json').read_text())
        collision=NS(held_local=None,observe_contact=lambda *a:None,check=lambda *a:False,receipt=lambda:{})
        model=NS(lower=np.full(7,-3.),upper=np.full(7,3.))
        f=BlockPublicFrontend(ROOT,config,'pick_moving_target_on_the_table','cv_geometry',model,collision)
        f.goal_pose=Pose(np.array([0.,0.,1.]))
        robot=RobotState(.5,Pose.from_xyzw([.1,0.,.76,0.,1.,0.,0.]),np.zeros(3),np.zeros(3),np.zeros(7),np.zeros(7))
        body=ObjectState('target',.5,Pose(np.array([.1,0.,.775])),np.zeros((20,3)),np.array([.3,0,0]),np.zeros(3),1.,True)
        f.tracker.observe=lambda observation:(body,{})
        f.observe(dict(timestamp=.5),robot);f.frames=5
        f.forecast=f.dynamics.predict([body],1.1,.05)
        p=BlockGeometryIntents(f);p.phase='acquire';p.select()
        return p,f

    def test_forecast_arrival_cannot_issue_a_close_until_current_physical_geometry(self):
        p,f=self.fixture();f.robot=replace(f.robot,pose=p.selected['pose'])
        self.assertEqual(p.qualify_opening(0.,1.)[0],1.)
        f.robot=replace(f.robot,pose=p.current_grasp())
        command,proof,edge=p.qualify_opening(0.,1.)
        self.assertEqual(command,0.);self.assertIsNotNone(edge);self.assertIsNone(p.close_witness)
        p.confirm_opening(command,proof,edge,confirmed_after=.55)
        self.assertTrue(f.close_confirmed)
        self.assertEqual(p.qualify_opening(0.,0.)[0],0.)

    def test_contact_without_own_close_or_supported_transform_does_not_enter_lift(self):
        p,f=self.fixture();f.robot=replace(f.robot,contact=True,gripper_open=False)
        p.select();self.assertEqual(p.phase,'acquire')
        p.close_witness=dict(confirmed_after=.5);p.select();self.assertEqual(p.phase,'acquire')
        f.collision.held_local=Pose(np.array([0.,0.,.015]));p.select();self.assertEqual(p.phase,'lift')
        np.testing.assert_allclose(p.lift_goal.position,f.robot.pose.position+[0,0,.1])

    def test_stale_forecast_and_native_success_channel_are_rejected(self):
        p,f=self.fixture();f.robot=replace(f.robot,timestamp=.55)
        self.assertFalse(p.admit_joint(np.zeros(7))[0])
        with self.assertRaises(ValueError):f.observe(dict(timestamp=.6),replace(f.robot,timestamp=.6,task_success=True))

    def test_original_preposition_height_is_an_absolute_world_coordinate(self):
        p,f=self.fixture();p.phase='approach';action=p.select()
        self.assertAlmostEqual(action[2],1.02)
        self.assertEqual(p.phase,'approach')

    def test_terminal_view_forwards_actuator_writes_and_hides_legacy_outcome_from_policy(self):
        _,f=self.fixture();raw=NS(_requested_opening=1.,_native_success_latched=False,
            observe=lambda:({},replace(f.robot,task_success=True)))
        view=TerminalBlockView(raw);view._requested_opening=0.
        self.assertEqual(raw._requested_opening,0.);self.assertFalse(view.observe()[1].task_success)
        self.assertFalse(view._native_terminal_latched);raw._native_success_latched=True
        self.assertTrue(view._native_terminal_latched)


if __name__=='__main__':unittest.main()
