from dataclasses import replace
from pathlib import Path
import json
import unittest
import numpy as np
import test_physmani_skill_v548 as prior_fixture
from our_mani.contracts import Pose
from our_mani.multitask.interaction import InteractionLibrary,PolicyState
from our_mani.multitask.grasp_frame_compensation_native_compat import correction,compose
from our_mani.multitask.physmani_ring_compensation_v580 import CorrectedRingIntents,binding

ROOT=Path(__file__).resolve().parents[1];NATIVE=ROOT.parent/'phsimani/PhysMani'


class Tests(unittest.TestCase):
    def fixture(self):
        _,f=prior_fixture.Tests().fixture('lift');task='insert_onto_rotating_peg'
        config=json.loads((ROOT/'artifacts/framework/multitask_r10/public_ring_frontend_v572_setup/configuration.json').read_text()) if (ROOT/'artifacts/framework/multitask_r10/public_ring_frontend_v572_setup/configuration.json').exists() else json.loads((ROOT/'artifacts/framework/multitask_r10/public_ring_frontend_v572_setup'/f'{task}_configuration.json').read_text())
        f.actor.controller.library=InteractionLibrary.load(ROOT/config['learned_artifacts'][task]['interaction']['path'])
        body=replace(f.state.bodies['rubbish'],object_id='ring')
        metadata=dict(bodies={'ring':dict(f.state.perception_metadata['bodies']['rubbish'],motion_mode='attached')})
        f.state=PolicyState(replace(f.state.robot,contact=True,gripper_open=False),{'ring':body},metadata,{'ring':('ring',)})
        rule=binding(ROOT,NATIVE,task);prior=json.loads((ROOT/rule['prior']['path']).read_text())
        return CorrectedRingIntents(f,rule,prior),f

    def test_contact_alone_or_missing_typed_evidence_cannot_modify_training_templates(self):
        p,f=self.fixture();original=f.actor.controller.library.templates
        p._measure_compensation();self.assertIs(f.actor.controller.library.templates,original)
        p.close_witness=dict(manipulated_ids=['ring'],confirmed_after=1.)
        f.actor.collision_port=type('Port',(),{'ring_affordance':None})()
        p._measure_compensation();self.assertIs(f.actor.controller.library.templates,original)
        self.assertIsNone(p.compensation)

    def test_existing_correction_preserves_the_intended_ring_pose_for_both_original_priors(self):
        for task in ('insert_onto_rotating_peg','insert_onto_rotating_peg_high_speed'):
            rule=binding(ROOT,NATIVE,task);prior=json.loads((ROOT/rule['prior']['path']).read_text())
            nominal=Pose.from_xyzw(prior['tool_to_ring_xyzw'])
            measured=Pose(nominal.position+[.005,-.003,.002],nominal.rotation)
            delta,actual=correction(prior,measured)
            original_TCP=Pose(np.array([.2,-.1,1.]))
            actual_ring=compose(compose(original_TCP,delta),actual)
            intended_ring=compose(original_TCP,nominal)
            np.testing.assert_allclose(actual_ring.position,intended_ring.position,atol=1e-10)
            np.testing.assert_allclose(actual_ring.rotation,intended_ring.rotation,atol=1e-10)
            self.assertFalse(rule['full_orientation_promoted'])


if __name__=='__main__':unittest.main()
