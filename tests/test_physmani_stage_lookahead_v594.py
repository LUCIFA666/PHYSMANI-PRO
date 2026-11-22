from dataclasses import replace
from pathlib import Path
import unittest
import numpy as np
from our_mani.contracts import Pose
from our_mani.multitask.interaction import PolicyState
from our_mani.multitask.physmani_author_stages_v574 import binding as stage_binding
from our_mani.multitask.physmani_stage_lookahead_v594 import AuthorLookaheadIntents,binding
from our_mani.multitask.physmani_geometry_goals_v593 import pose_goals
import test_physmani_skill_v548 as previous

ROOT=Path(__file__).resolve().parents[1];NATIVE=ROOT.parent/'phsimani/PhysMani';TASK='put_rubbish_in_moving_bin'


class Tests(unittest.TestCase):
    def fixture(self,phase):
        _,f=previous.Tests().fixture(phase)
        p=AuthorLookaheadIntents(f,stage_binding(ROOT,NATIVE,TASK),binding(ROOT,NATIVE,TASK))
        p.phase_index=p.phase_order.index(phase);p.select()
        return p,f

    def move(self,f,target,offset=0.,**kwargs):
        f.state=PolicyState(replace(f.state.robot,pose=Pose(target.position+[offset,0.,0.],target.rotation),**kwargs),
            f.state.bodies,f.state.perception_metadata,f.state.role_bindings)

    def test_zero_delay_uses_current_observation_and_still_rejects_stale_or_negative_queries(self):
        p,f=self.fixture('acquire');goal=p.selected
        self.assertEqual(goal['lookahead_seconds'],0.)
        np.testing.assert_allclose(goal['pose'].position,goal['present_pose'].position,atol=1e-12)
        future=pose_goals(f,'acquire',origin=p.origin,lookahead=.5)[0]
        self.assertGreater(np.linalg.norm(goal['pose'].position-future['pose'].position),.04)
        f.forecast.metadata['observed_until']=.9;self.assertEqual(p.candidates('acquire'),[])
        with self.assertRaises(ValueError):pose_goals(f,'acquire',lookahead=-.01)

    def test_bin_carries_to_current_target_then_requests_predicted_release_inside_source5cm(self):
        p,f=self.fixture('deposit');self.assertEqual(p.selected['opening'],0.)
        self.assertEqual(p.selected['lookahead_seconds'],0.)
        target=p._completion_target(p.selected);self.move(f,target,.04,contact=True,gripper_open=False)
        phase_epoch=p.phase_epoch;action=p.select();self.assertEqual(action[-1],1.)
        self.assertEqual(p.selected['lookahead_seconds'],.5);self.assertEqual(p.phase_epoch,phase_epoch)
        self.assertEqual(p.qualify_opening(1.,0.)[0],1.)
        self.assertFalse(p._done(p.selected))
        self.move(f,target,.051,contact=True,gripper_open=False)
        self.assertEqual(p.qualify_opening(1.,0.)[0],0.)

    def test_source_delay_bindings_cover_normal_and_high_speed_without_changing_physical_rules(self):
        for family in (TASK,'place_cups_on_rotating_frame'):
            for suffix in ('','_high_speed'):
                rule=binding(ROOT,NATIVE,family+suffix)
                self.assertTrue(rule['physical_close_and_collision_gates_unchanged'])
                self.assertFalse(rule['TRAIN_templates_refitted']);self.assertFalse(rule['native_success_changed'])
                self.assertEqual(rule['delays']['acquire'],0.)
                self.assertGreater(len(rule['source_branches']),5)


if __name__=='__main__':unittest.main()
