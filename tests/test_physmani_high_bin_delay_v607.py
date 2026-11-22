from pathlib import Path
from dataclasses import replace
import json
import unittest
from our_mani.contracts import Pose
from our_mani.multitask.interaction import InteractionLibrary,PolicyState
from our_mani.multitask.physmani_author_stages_v574 import binding as stage_binding
from our_mani.multitask.physmani_stage_lookahead_v594 import binding as delay_binding
from our_mani.multitask.physmani_high_bin_delay_v601 import HighBinDelayIntents,binding,TASK
import test_physmani_skill_v548 as previous

ROOT=Path(__file__).resolve().parents[1];NATIVE=ROOT.parent/'phsimani/PhysMani'


class Tests(unittest.TestCase):
    def test_source_transition_current_then_closed1s_then_near_open_half_second(self):
        _,f=previous.Tests().fixture('deposit')
        c=json.loads((ROOT/'artifacts/framework/multitask_r10/stage_lookahead_v596_setup'/f'{TASK}_configuration.json').read_text())
        f.actor.controller.library=InteractionLibrary.load(ROOT/c['learned_artifacts'][TASK]['interaction']['path'])
        f.state=PolicyState(replace(f.state.robot,contact=True,gripper_open=False),f.state.bodies,
            f.state.perception_metadata,f.state.role_bindings)
        p=HighBinDelayIntents(f,stage_binding(ROOT,NATIVE,TASK),delay_binding(ROOT,NATIVE,TASK),binding(ROOT,NATIVE,TASK))
        p.phase_index=p.phase_order.index('deposit');p.select()
        self.assertEqual(p.selected['lookahead_seconds'],0.);self.assertEqual(p.selected['opening'],0.)
        p.select();self.assertEqual(p.selected['lookahead_seconds'],1.);self.assertEqual(p.selected['opening'],0.)
        target=p._completion_target(p.selected)
        f.state=PolicyState(replace(f.state.robot,pose=Pose(target.position+[.04,0,0],target.rotation)),
            f.state.bodies,f.state.perception_metadata,f.state.role_bindings)
        p.select();self.assertEqual(p.selected['lookahead_seconds'],.5);self.assertEqual(p.selected['opening'],1.)
        self.assertEqual(p.phase_epoch,1.)


if __name__=='__main__':unittest.main()
