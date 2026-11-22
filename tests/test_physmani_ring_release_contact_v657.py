from pathlib import Path
from types import SimpleNamespace as NS
from dataclasses import replace
import unittest
import numpy as np
from our_mani.multitask.pipeline import PhaseCollisionPort
from our_mani.multitask.interaction import PolicyState
from our_mani.multitask.specs import TASK_SPECS
from our_mani.multitask.physmani_ring_release_contact_v657 import SourceContactRingIntents,binding
import test_physmani_ring_release_v590 as previous
from test_physmani_skill_v533 import Forecast
ROOT=Path(__file__).resolve().parents[1];NATIVE=ROOT.parent/'phsimani/PhysMani'


class Tests(unittest.TestCase):
    def test_terminal_release_keeps_only_selected_peg_contact_and_still_rejects_other_obstacles(self):
        old,f=previous.Tests().fixture();task=old.task
        p=SourceContactRingIntents(f,old.comp_rule,old.prior,old.release_rule,contact_rule=binding(ROOT,NATIVE,task))
        p.phase_index=p.phase_order.index('insert');p.select();previous.Tests().move(f,p._completion_target(p.selected));p.select()
        self.assertEqual(p.selected['phase'],'release')
        bodies={**f.state.bodies,'frame':replace(f.state.bodies['peg0'],object_id='frame')}
        f.state=PolicyState(f.state.robot,bodies,f.state.perception_metadata,dict(f.state.role_bindings,selected_peg=('peg0',)))
        f.forecast=Forecast(f.state);f.forecast.object_ids=tuple(bodies)
        seen=[];block={'frame':False}
        def check(poses,**kw):
            seen.append(tuple(kw['allowed_robot_contact_ids']))
            return bool(block['frame'] or 'peg0' not in kw['allowed_robot_contact_ids'])
        service=NS(required_dynamic_role_ids=('ring','peg0','frame'),check=check,estimated_bodies={})
        port=PhaseCollisionPort(TASK_SPECS[task],service,f.actor.robot_model,maximum_observation_age=.15)
        port.configure_frame(f.state,f.forecast,attachment_ids=('ring',))
        f.actor.controller.generator.bind_candidate_scene=lambda state,forecast,phase,ref,ids,arrival:port.bind_candidate(
            state=state,forecast=forecast,phase=phase,reference_id=ref,manipulated_ids=ids,arrival_time=arrival)
        f.actor.robot_model.collision=lambda q,offset:port.check_future(offset)
        selected=p.selected;accepted,proof=p.admit_joint(np.zeros(7));self.assertTrue(accepted)
        self.assertIs(p.selected,selected);self.assertEqual(p.selected['phase'],'release')
        self.assertEqual(set(seen[-1]),{'peg0'});self.assertTrue(proof['other_collision_checks_unchanged'])
        block['frame']=True;self.assertFalse(p.admit_joint(np.zeros(7))[0]);self.assertIs(p.selected,selected)
        def fail(*a):raise RuntimeError('geometry backend failed')
        f.actor.robot_model.collision=fail
        with self.assertRaisesRegex(RuntimeError,'geometry backend failed'):p.admit_joint(np.zeros(7))
        self.assertIs(p.selected,selected)


if __name__=='__main__':unittest.main()
