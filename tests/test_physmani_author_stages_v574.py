from pathlib import Path
from dataclasses import replace
import unittest
from our_mani.contracts import Pose
from our_mani.multitask.interaction import PolicyState
from our_mani.multitask.physmani_author_stages_v574 import AuthorStageIntents,binding
import test_physmani_skill_v548 as prior

ROOT=Path(__file__).resolve().parents[1];NATIVE=ROOT.parent/'phsimani/PhysMani'


class Tests(unittest.TestCase):
    def fixture(self,phase):
        _,f=prior.Tests().fixture(phase);task='put_rubbish_in_moving_bin'
        p=AuthorStageIntents(f,binding(ROOT,NATIVE,task));p.phase_index=p.phase_order.index(phase);p.select()
        return p,f

    def move(self,f,target,offset=0.,**kw):
        f.state=PolicyState(replace(f.state.robot,pose=Pose(target.position+[offset,0,0],target.rotation),**kw),
            f.state.bodies,f.state.perception_metadata,f.state.role_bindings)

    def test_original_bin1cm_close_is_nested_inside_the_unchanged2cm_physical_gate(self):
        p,f=self.fixture('acquire');target=p._completion_target(p.selected)
        self.move(f,target,.015);self.assertEqual(p.qualify_opening(0.,1.)[0],1.)
        self.move(f,target,.009);command,_,edge=p.qualify_opening(0.,1.)
        self.assertEqual(command,0.);self.assertIsNotNone(edge)

    def test_lift_and_release_use_author5cm_instead_of_fit_error_caps(self):
        p,f=self.fixture('lift');target=p._completion_target(p.selected)
        self.move(f,target,.04);self.assertTrue(p._done(p.selected))
        self.move(f,target,.051);self.assertFalse(p._done(p.selected))
        p,f=self.fixture('deposit');target=p._completion_target(p.selected)
        self.move(f,target,.04,contact=True,gripper_open=False)
        self.assertEqual(p.qualify_opening(1.,0.)[0],1.)
        self.assertFalse(p._done(p.selected))

    def test_all_four_original_native_sources_are_checked_without_modifying_them(self):
        for family in ('put_rubbish_in_moving_bin','place_cups_on_rotating_frame'):
            for suffix in ('','_high_speed'):
                r=binding(ROOT,NATIVE,family+suffix)
                self.assertTrue(r['original_model_fit_caps_and_native_success_unchanged'])
                self.assertEqual(r['physical_close_maximum_m'],.02)


if __name__=='__main__':unittest.main()
