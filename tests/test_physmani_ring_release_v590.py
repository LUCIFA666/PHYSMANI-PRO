from dataclasses import replace
from pathlib import Path
import unittest
from our_mani.contracts import Pose
from our_mani.multitask.interaction import PolicyState
from our_mani.multitask.physmani_ring_release_v590 import AuthorRingReleaseIntents,binding
import test_physmani_ring_compensation_v585 as previous
from test_physmani_skill_v533 import Forecast

ROOT=Path(__file__).resolve().parents[1];NATIVE=ROOT.parent/'phsimani/PhysMani'


class Tests(unittest.TestCase):
    def fixture(self):
        old,f=previous.Tests().fixture()
        p=AuthorRingReleaseIntents(f,old.comp_rule,old.prior,binding(ROOT,NATIVE,old.task))
        body=replace(f.state.bodies['ring'],object_id='peg0')
        bodies={**f.state.bodies,'peg0':body}
        metadata=dict(bodies={**f.state.perception_metadata['bodies'],'peg0':dict(
            directly_visible=True,registration_valid=True,orientation_observable=True,motion_mode='external')})
        f.state=PolicyState(f.state.robot,bodies,metadata,{'ring':('ring',),'selected_peg':('peg0',)})
        f.forecast=Forecast(f.state)
        p.phase_index=p.phase_order.index('insert');p.select()
        return p,f

    def move(self,f,target,offset=0.,**kwargs):
        f.state=PolicyState(replace(f.state.robot,pose=Pose(target.position+[offset,0.,0.],target.rotation),**kwargs),
            f.state.bodies,f.state.perception_metadata,f.state.role_bindings)

    def test_actual_insert_completion_appends_source_release_and_keeps_peg_identity(self):
        p,f=self.fixture();goal=p.selected;self.assertIsNotNone(goal)
        current=p._completion_target(goal);self.move(f,current)
        action=p.select()
        self.assertIsNotNone(action);self.assertEqual(p.selected['phase'],'release')
        self.assertEqual(p.selected['reference_id'],'peg0');self.assertEqual(action[-1],1.)
        self.assertEqual(p.events[-1]['event'],'source_author_terminal_release_pose_intent')
        self.assertFalse(p.events[-1]['trained_release_skill_claimed'])
        self.assertEqual(tuple(f.actor.controller.library.phase_order),('approach','acquire','lift','align','insert'))

    def test_release_uses_current3cm_guard_and_only_observed_open_counts_complete(self):
        p,f=self.fixture();self.move(f,p._completion_target(p.selected));p.select()
        target=p._completion_target(p.selected)
        self.move(f,target,.031);self.assertEqual(p.qualify_opening(1.,0.)[0],0.)
        self.move(f,target,.029);self.assertEqual(p.qualify_opening(1.,0.)[0],1.)
        self.assertFalse(p._done(p.selected))
        self.move(f,target,.029,gripper_open=True,contact=False);self.assertTrue(p._done(p.selected))
        for task in ('insert_onto_rotating_peg','insert_onto_rotating_peg_high_speed'):
            self.assertFalse(binding(ROOT,NATIVE,task)['native_success_rule_changed'])


if __name__=='__main__':unittest.main()
