from pathlib import Path
from dataclasses import replace
import json
import unittest
import numpy as np
from our_mani.contracts import Pose
from our_mani.multitask.interaction import PolicyState
from our_mani.multitask.physmani_remove_wp0_v637 import AuthorPreApproachRemoveIntents,binding
from our_mani.multitask.physmani_remove_stages_v611 import binding as remove_binding
import test_physmani_remove_stages_v611 as previous
ROOT=Path(__file__).resolve().parents[1];NATIVE=ROOT.parent/'phsimani/PhysMani';TASK='remove_cups_from_rotating_frame'


class Tests(unittest.TestCase):
    def fixture(self):
        _,f=previous.Tests().fixture();rule=binding(ROOT,NATIVE,TASK)
        model=json.loads((ROOT/rule['model']['path']).read_text())
        p=AuthorPreApproachRemoveIntents(f,remove_binding(ROOT,NATIVE,TASK),rule,model);p.select()
        return p,f

    def move(self,f,target,offset=0.):
        f.state=PolicyState(replace(f.state.robot,pose=Pose(target.position+[offset,0.,0.],target.rotation)),
            f.state.bodies,f.state.perception_metadata,f.state.role_bindings)

    def test_wp0_source_guard_precedes_wp1_without_consuming_first1s_or_resetting_phase_epoch(self):
        p,f=self.fixture();self.assertEqual(p.selected['author_substage'],'wp0')
        self.assertEqual(p.selected['lookahead_seconds'],0.);epoch=p.phase_epoch
        current=p._completion_target(p.selected);self.move(f,current,.09);p.select()
        self.assertTrue(p.wp0_passed);self.assertEqual(p.selected['author_substage'],'wp1')
        self.assertEqual(p.selected['lookahead_seconds'],1.);self.assertEqual(p.phase_index,0)
        self.assertEqual(p.phase_epoch,epoch);self.assertIsNone(p.close_witness)
        self.assertAlmostEqual(np.linalg.norm(p.selected['pose'].position-p.selected['source_predicted_canonical'].position),.03)
        p.select();self.assertEqual(p.selected['lookahead_seconds'],.5)

    def test_wp1_standoff_does_not_replace_true_current5cm_acquire_transition(self):
        p,f=self.fixture();self.move(f,p._completion_target(p.selected));p.select()
        canonical=p._completion_target(p.selected);self.move(f,canonical,.051)
        p.select();self.assertEqual(p.selected['phase'],'approach')
        self.move(f,canonical,.04);p.select();self.assertEqual(p.selected['phase'],'acquire')
        self.assertEqual(p.selected['lookahead_seconds'],0.)
        self.assertEqual(p.remove_rule['physical_acquire_gate_position_m'],.02)

    def test_both_models_retain_all100_parents_and_never_claim_completed_wp0_skill(self):
        for task in (TASK,TASK+'_high_speed'):
            rule=binding(ROOT,NATIVE,task);model=json.loads((ROOT/rule['model']['path']).read_text())
            self.assertEqual(model['declared_parent_count'],100)
            self.assertFalse(model['native_success_or_completion_used_for_selection'])
            self.assertFalse(rule['completed_wp0_skill_claimed'])


if __name__=='__main__':unittest.main()
