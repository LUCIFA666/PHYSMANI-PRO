from dataclasses import replace
from pathlib import Path
import json
import unittest
import numpy as np
from our_mani.contracts import Pose
from our_mani.multitask.interaction import InteractionLibrary,PolicyState
from our_mani.multitask.physmani_basketball_release_v558 import AuthorBasketballReleaseIntents,binding
import test_physmani_skill_v533 as prior

ROOT=Path(__file__).resolve().parents[1];NATIVE=ROOT.parent/'phsimani/PhysMani'


class Tests(unittest.TestCase):
    def fixture(self,phase='deposit'):
        _,f=prior.Tests().fixture('deposit');task='moving_basketball_in_hoop'
        config=json.loads((ROOT/'artifacts/framework/multitask_r10/physmani_skills_v550_setup'/f'{task}_configuration.json').read_text())
        f.actor.controller.library=InteractionLibrary.load(ROOT/config['learned_artifacts'][task]['interaction']['path'])
        f.actor.controller.committed_roles={}
        mapping={'rubbish':'ball','bin':'hoop'}
        bodies={mapping[k]:replace(v,object_id=mapping[k]) for k,v in f.state.bodies.items()}
        metadata=dict(bodies={mapping[k]:dict(v) for k,v in f.state.perception_metadata['bodies'].items()})
        f.state=PolicyState(f.state.robot,bodies,metadata,{k:(k,) for k in bodies})
        f.forecast=prior.Forecast(f.state)
        p=AuthorBasketballReleaseIntents(f,binding(ROOT,NATIVE,task));p.phase_index=p.phase_order.index(phase);p.select()
        return p,f

    def test_original_position_only3cm_release_guard_does_not_use_fit_error_as_a_controller_gate(self):
        p,f=self.fixture();target=p._completion_target(p.selected)
        for offset,expected in ((.029,1.),(.031,0.)):
            f.state=PolicyState(replace(f.state.robot,pose=Pose(target.position+[offset,0,0],target.rotation),
                contact=True,gripper_open=False),f.state.bodies,f.state.perception_metadata,f.state.role_bindings)
            command,proof,edge=p.qualify_opening(1.,0.)
            self.assertEqual(command,expected);self.assertIsNone(edge)
            self.assertTrue(proof['fit_tolerance_is_not_the_release_guard'])

    def test3cm_release_guard_does_not_relax2cm_physical_acquisition(self):
        p,f=self.fixture('acquire');target=p._completion_target(p.selected)
        f.state=PolicyState(replace(f.state.robot,pose=Pose(target.position+[.021,0,0],target.rotation)),
            f.state.bodies,f.state.perception_metadata,f.state.role_bindings)
        self.assertEqual(p.qualify_opening(0.,1.)[0],1.)

    def test_rule_is_verified_against_both_unchanged_author_sources(self):
        for task in ('moving_basketball_in_hoop','moving_basketball_in_hoop_high_speed'):
            self.assertEqual(binding(ROOT,NATIVE,task)['release_distance_m'],.03)
        p,f=self.fixture();bad=dict(p.release_rule,release_distance_m=.05)
        with self.assertRaises(ValueError):AuthorBasketballReleaseIntents(f,bad)


if __name__=='__main__':unittest.main()
