from dataclasses import replace
from pathlib import Path
import json
import unittest
import numpy as np
from our_mani.contracts import Pose
from our_mani.multitask.interaction import InteractionLibrary,PolicyState,_world_target
from our_mani.multitask.physmani_buzz_curve_v602 import BuzzCurveIntents,binding
import test_physmani_skill_v548 as previous
from test_physmani_skill_v533 import Forecast

ROOT=Path(__file__).resolve().parents[1];NATIVE=ROOT.parent/'phsimani/PhysMani';TASK='beat_the_rotating_buzz'


class Tests(unittest.TestCase):
    def fixture(self,phase='follow'):
        _,f=previous.Tests().fixture('lift')
        c=json.loads((ROOT/'artifacts/framework/multitask_r10/buzz_frontend_v588_setup'/f'{TASK}_configuration.json').read_text())
        f.actor.controller.library=InteractionLibrary.load(ROOT/c['learned_artifacts'][TASK]['interaction']['path'])
        template=f.actor.controller.library.for_phase(phase)[0]
        frame=replace(f.state.bodies['bin'],object_id='frame');wand=replace(frame,object_id='wand')
        metadata=dict(bodies={'frame':dict(directly_visible=True,registration_valid=True,orientation_observable=True,motion_mode='external'),
                              'wand':dict(directly_visible=True,registration_valid=True,orientation_observable=True,motion_mode='attached')})
        current=_world_target(frame.pose,np.zeros(3),np.zeros(3),template,0.)[0]
        f.state=PolicyState(replace(f.state.robot,pose=current,gripper_open=False,contact=True),
            {'frame':frame,'wand':wand},metadata,{'frame':('frame',),'wand':('wand',)})
        f.forecast=Forecast(f.state)
        p=BuzzCurveIntents(f,binding(ROOT,NATIVE,TASK));p.phase_index=p.phase_order.index(phase);p.select()
        return p,f

    def test_first_follow_target_preserves_intermediate_TRAIN_curve_instead_of_jumping_to_end(self):
        p,f=self.fixture();goal=p.selected
        self.assertGreater(goal['curve_progress'],0.);self.assertLess(goal['curve_progress'],1.)
        self.assertFalse(p._done(goal));events=p.curve_events
        self.assertTrue(events[-1]['progress_from_current_observation_only'])
        first_progress=goal['curve_progress'];p.select()
        self.assertEqual(p.selected['curve_progress'],first_progress)
        event_count=len(p.curve_events)
        f.forecast.metadata['observed_until']=.9;self.assertIsNone(p.select())
        self.assertEqual(len(p.curve_events),event_count)

    def test_release_requires_current_author3cm_and_observed_release_not_future_arrival(self):
        p,f=self.fixture('release');goal=p.selected;target=p._completion_target(goal)
        f.state=PolicyState(replace(f.state.robot,pose=Pose(target.position+[.031,0.,0.],target.rotation)),
            f.state.bodies,f.state.perception_metadata,f.state.role_bindings)
        self.assertEqual(p.qualify_opening(1.,0.)[0],0.)
        f.state=PolicyState(replace(f.state.robot,pose=target),f.state.bodies,f.state.perception_metadata,f.state.role_bindings)
        self.assertEqual(p.qualify_opening(1.,0.)[0],1.);self.assertFalse(p._done(goal))
        for task in (TASK,TASK+'_high_speed'):
            self.assertTrue(binding(ROOT,NATIVE,task)['collision_checks_retained'])


if __name__=='__main__':unittest.main()
