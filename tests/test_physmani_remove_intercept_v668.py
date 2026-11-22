from pathlib import Path
from dataclasses import replace
from types import SimpleNamespace as NS
from unittest.mock import patch
import json,unittest
import numpy as np
from our_mani.contracts import Pose
from our_mani.multitask.interaction import PolicyState,COMMAND_GOAL
from our_mani.multitask import physmani_remove_intercept_v667 as m
import test_physmani_remove_wp0_v637 as previous
from test_physmani_skill_v533 import Forecast
from our_mani.multitask.physmani_remove_wp0_v637 import binding as wp0
from our_mani.multitask.physmani_remove_stages_v611 import binding as stages
from our_mani.multitask.physmani_remove_source_roles_v653 import binding as roles
ROOT=Path(__file__).resolve().parents[1];NATIVE=ROOT.parent/'phsimani/PhysMani';TASK='remove_cups_from_rotating_frame'


class Tests(unittest.TestCase):
    def fixture(self):
        _,f=previous.Tests().fixture();f.actor.controller.generator.config=NS(maximum_observation_age=.15)
        c=json.loads((ROOT/'artifacts/framework/multitask_r10/remove_source_roles_v655_setup'/f'{TASK}_configuration.json').read_text())
        rule=wp0(ROOT,NATIVE,TASK)
        p=m.InterceptRemoveIntents(f,stages(ROOT,NATIVE,TASK),rule,json.loads((ROOT/rule['model']['path']).read_text()),
            source_role_rule=roles(ROOT,c,TASK),intercept_rule=m.binding(ROOT,TASK))
        return p,f

    def test_future_public_canonical_is_used_only_as_intent_and_not_as_current_close_evidence(self):
        p,f=self.fixture();template=f.actor.controller.library.for_phase('acquire')[0]
        self.assertEqual(template.reference_time_basis,COMMAND_GOAL)
        now=float(f.state.robot.timestamp);future=Pose(f.state.robot.pose.position+[.1,0.,0.],f.state.robot.pose.rotation)
        row=dict(template=template,reference_id='cup2',manipulated_ids=('cup2',),predicted_reference_time=now+.5,
            lookahead_seconds=.5,selected_at=now)
        with patch.object(m,'pose_goals',return_value=[row]) as goals,patch.object(m,'_reference_at',return_value=(future,None,None)):
            candidates=p.candidates('acquire')
            self.assertEqual(goals.call_args.kwargs['lookahead'],.5)
            local=Pose.from_xyzw(template.completion_geometry_xyzw)
            np.testing.assert_allclose(candidates[0]['pose'].position,future.apply(local.position))
        p.phase_index=1;p.selected=dict(candidates[0],phase='acquire')
        # A perfectly selected future intent does not authorize a close when
        # the current canonical grasp is outside the original physical gate.
        with patch.object(p,'_completion_target',return_value=Pose(f.state.robot.pose.position+[.1,0.,0.],f.state.robot.pose.rotation)):
            opening,proof,witness=p.qualify_opening(0.,1.)
            self.assertEqual(opening,1.);self.assertIsNone(witness)
        self.assertEqual(p.physical_grasp_gate.close_position_tolerance_m,.02)
        self.assertEqual(p.physical_grasp_gate.close_orientation_tolerance_rad,.1)


if __name__=='__main__':unittest.main()
