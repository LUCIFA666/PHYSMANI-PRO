from pathlib import Path
from dataclasses import replace
from types import SimpleNamespace as NS
import json
import unittest
import numpy as np
from our_mani.contracts import Pose
from our_mani.multitask.interaction import PolicyState
from our_mani.multitask.physmani_remove_source_roles_v653 import AuthorLiveRoleRemoveIntents,binding
from our_mani.multitask.physmani_remove_wp0_v637 import binding as wp0_binding
from our_mani.multitask.physmani_remove_stages_v611 import binding as remove_binding
import test_physmani_remove_wp0_v637 as previous
from test_physmani_skill_v533 import Forecast
ROOT=Path(__file__).resolve().parents[1];NATIVE=ROOT.parent/'phsimani/PhysMani';TASK='remove_cups_from_rotating_frame'


class Tests(unittest.TestCase):
    def test_current_nearest_cup_can_change_before_close_and_is_immutable_after_own_close_dispatch(self):
        _,f=previous.Tests().fixture();f.actor.controller.generator.config=NS(maximum_observation_age=.15)
        c=json.loads((ROOT/'artifacts/framework/multitask_r10/remove_wp0_v639_setup'/f'{TASK}_configuration.json').read_text())
        rule=wp0_binding(ROOT,NATIVE,TASK);model=json.loads((ROOT/rule['model']['path']).read_text())
        bodies={**f.state.bodies,'cup1':replace(f.state.bodies['cup2'],object_id='cup1',pose=Pose(np.array([2.,0.,.8])))}
        metadata=dict(bodies={**f.state.perception_metadata['bodies'],'cup1':dict(f.state.perception_metadata['bodies']['cup2'])})
        f.state=PolicyState(f.state.robot,bodies,metadata,f.state.role_bindings);f.forecast=Forecast(f.state)
        p=AuthorLiveRoleRemoveIntents(f,remove_binding(ROOT,NATIVE,TASK),rule,model,source_role_rule=binding(ROOT,c,TASK));p.select()
        self.assertEqual(f.state.role_bindings['selected_cup'],('cup2',));epoch=p.phase_epoch
        bodies['cup1']=replace(bodies['cup1'],pose=Pose(f.state.robot.pose.position+[.02,0.,0.]))
        f.state=PolicyState(f.state.robot,bodies,metadata,f.state.role_bindings);f.forecast=Forecast(f.state)
        p.select();self.assertEqual(f.state.role_bindings['selected_cup'],('cup1',));self.assertEqual(p.phase_epoch,epoch)
        count=len(p.source_role_events);p.close_witness=dict(phase_index=1,manipulated_ids=['cup1'],confirmed_after=1.)
        bodies['cup2']=replace(bodies['cup2'],pose=Pose(f.state.robot.pose.position+[.001,0.,0.]))
        f.state=PolicyState(f.state.robot,bodies,metadata,f.state.role_bindings);f.forecast=Forecast(f.state)
        p.select();self.assertEqual(f.state.role_bindings['selected_cup'],('cup1',))
        self.assertEqual(len(p.source_role_events),count)


if __name__=='__main__':unittest.main()
