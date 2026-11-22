from dataclasses import replace
from pathlib import Path
import json
import unittest
from our_mani.contracts import Pose
from our_mani.multitask.interaction import InteractionLibrary,PolicyState
from our_mani.multitask.physmani_remove_stages_v611 import RemoveAuthorIntents,binding
import test_physmani_skill_v548 as previous
from test_physmani_skill_v533 import Forecast
ROOT=Path(__file__).resolve().parents[1];NATIVE=ROOT.parent/'phsimani/PhysMani';TASK='remove_cups_from_rotating_frame'


class Tests(unittest.TestCase):
    def fixture(self):
        _,f=previous.Tests().fixture('lift')
        c=json.loads((ROOT/'artifacts/framework/multitask_r10/physmani_skills_v536_setup'/f'{TASK}_configuration.json').read_text())
        f.actor.controller.library=InteractionLibrary.load(ROOT/c['learned_artifacts'][TASK]['interaction']['path'])
        bodies={k:replace(f.state.bodies['bin'],object_id=k) for k in ('cup2','spoke2')}
        metadata=dict(bodies={k:dict(directly_visible=True,registration_valid=True,orientation_observable=True,motion_mode='external') for k in bodies})
        f.state=PolicyState(f.state.robot,bodies,metadata,{'selected_cup':('cup2',),'selected_spoke':('spoke2',)})
        f.forecast=Forecast(f.state)
        p=RemoveAuthorIntents(f,binding(ROOT,NATIVE,TASK));p.select()
        return p,f

    def test_current_geometry_advances_to_acquire_and_switches_to_author_zero_delay(self):
        p,f=self.fixture();self.assertEqual(p.selected['lookahead_seconds'],1.)
        self.assertEqual(p.selected['curve_progress'],0.)
        p.select();self.assertEqual(p.selected['lookahead_seconds'],.5)
        target=p._completion_target(p.selected)
        f.state=PolicyState(replace(f.state.robot,pose=Pose(target.position+[.04,0.,0.],target.rotation)),
            f.state.bodies,f.state.perception_metadata,f.state.role_bindings)
        p.select();self.assertEqual(p.selected['phase'],'acquire');self.assertEqual(p.selected['lookahead_seconds'],0.)
        self.assertIsNone(p.close_witness)

    def test_both_original_sources_keep_original_physical_grasp_and_task_success(self):
        for task in (TASK,TASK+'_high_speed'):
            rule=binding(ROOT,NATIVE,task)
            self.assertEqual(rule['physical_acquire_gate_position_m'],.02)
            self.assertEqual(rule['physical_acquire_gate_angle_rad'],.1)
            self.assertTrue(rule['original_TRAIN_templates_and_native_task_success_unchanged'])


if __name__=='__main__':unittest.main()
