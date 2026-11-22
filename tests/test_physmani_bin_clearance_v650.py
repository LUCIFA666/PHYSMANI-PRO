from pathlib import Path
from dataclasses import replace
from types import SimpleNamespace as NS
import json
import unittest
import numpy as np
from our_mani.contracts import Pose
from our_mani.multitask.interaction import PolicyState
from our_mani.multitask.forecasting import ObservedAttachment
from our_mani.multitask.physmani_author_stages_v574 import binding as stage_binding
from our_mani.multitask.physmani_stage_lookahead_v594 import binding as delay_binding
from our_mani.multitask.physmani_bin_clearance_v650 import NormalBinClearanceIntents,binding
from our_mani.multitask.physmani_geometry_goals_v593 import pose_goals
import test_physmani_skill_v548 as previous
ROOT=Path(__file__).resolve().parents[1];NATIVE=ROOT.parent/'phsimani/PhysMani';TASK='put_rubbish_in_moving_bin'


class Tests(unittest.TestCase):
    def fixture(self):
        _,f=previous.Tests().fixture('deposit')
        c=json.loads((ROOT/'artifacts/framework/multitask_r10/bin_filter_progress_v634_setup'/f'{TASK}_configuration.json').read_text())
        f.state=PolicyState(replace(f.state.robot,contact=True,gripper_open=False),f.state.bodies,
            f.state.perception_metadata,f.state.role_bindings)
        f.state.perception_metadata['bodies']['rubbish']['motion_mode']='attached'
        f.actor.bound_attachments={'rubbish'};f.actor.attachments={'rubbish':ObservedAttachment('rubbish',Pose(np.array([0.,0.,-.02])),1.)}
        f.actor.collision_port=NS(_geometry_modes=lambda:({'bin':'primary','rubbish':'primary'},{}))
        p=NormalBinClearanceIntents(f,stage_binding(ROOT,NATIVE,TASK),delay_binding(ROOT,NATIVE,TASK),
            root=ROOT,config=c,clearance_rule=binding(ROOT,c,TASK))
        p.phase_index=p.phase_order.index('deposit');p.origin=f.state.robot.pose;p.phase_epoch=1.
        p.close_witness=dict(phase_index=1,manipulated_ids=['rubbish'],confirmed_after=1.)
        row=pose_goals(f,'deposit',origin=p.origin,lookahead=0.)[0]
        return p,f,row

    def move(self,f,pose):
        f.state=PolicyState(replace(f.state.robot,pose=pose),f.state.bodies,f.state.perception_metadata,f.state.role_bindings)

    def test_raise_at_current_XY_before_lateral_transport_then_keep_original_gripper_request_tick(self):
        p,f,row=self.fixture();goal=p._clearance_goal(row);z=goal['minimum_clearance_TCP_z']
        self.move(f,Pose(np.array([1.,0.,z-.03]),goal['pose'].rotation));action=p.select()
        self.assertEqual(p.selected['clearance_substage'],'raise_held_mesh_before_crossing_bin_rim')
        np.testing.assert_allclose(action[:2],f.state.robot.pose.position[:2]);self.assertEqual(action[-1],0.)
        self.move(f,goal['pose']);action=p.select()
        self.assertEqual(p.selected['clearance_substage'],'carry_above_complete_public_bin_mesh')
        self.assertEqual(p.qualify_opening(0.,0.)[0],0.)
        command,proof,edge=p.qualify_opening(1.,0.);self.assertEqual(command,1.)
        self.assertFalse(p._done(p.selected));p.confirm_opening(command,proof,edge,confirmed_after=1.05)
        self.move(f,Pose(goal['pose'].position+[.1,0.,0.],goal['pose'].rotation))
        self.assertEqual(p.qualify_opening(0.,1.)[0],1.)
        self.assertEqual(p.phase_epoch,1.)

    def test_unknown_or_nonunique_held_geometry_cannot_generate_clearance_transport(self):
        p,f,row=self.fixture();p.close_witness=None;self.assertEqual(p.candidates('deposit'),[])
        p.close_witness=dict(phase_index=1,confirmed_after=1.)
        f.state.perception_metadata['bodies']['bin']['motion_mode']='attached'
        self.assertIsNone(p._clearance_goal(row))


if __name__=='__main__':unittest.main()
