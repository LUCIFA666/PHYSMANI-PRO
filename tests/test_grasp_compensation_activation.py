from pathlib import Path
from types import SimpleNamespace as NS,MethodType
from unittest.mock import patch
import json
import unittest
import numpy as np
from our_mani.contracts import Pose,ObjectState
from our_mani.multitask.forecasting import ObservedAttachment
from our_mani.multitask import grasp_frame_compensation_native_compat as policy
import test_ring_causal_retarget_v371 as actual
ROOT=Path(__file__).resolve().parents[1]


class Evidence:pass


class Tests(unittest.TestCase):
    def test_actual_controller_step_reaches_compensation_activation_in_native_scipy(self):
        helper=actual.Tests();c,retarget,model,scene=helper.fixture('insert_onto_rotating_peg')
        c.goal=None
        c.step=MethodType(lambda self,*a:{'command':'hold'},c)
        config=json.loads((ROOT/'artifacts/framework/multitask_r10/registered_tasks_v408_setup/insert_onto_rotating_peg_configuration.json').read_text())
        config['grasp_frame_compensation']=policy.binding(ROOT,config,'insert_onto_rotating_peg')
        prior=json.loads((ROOT/config['grasp_frame_compensation']['prior']['path']).read_text())
        expected=Pose.from_xyzw(prior['tool_to_ring_xyzw'])
        measured=Pose(expected.position+[.002,0.,-.003],expected.rotation)
        actor=NS(controller=c,continuous_reference_manager=NS(confirmations=[]),metrics_report=lambda:{},
            attachments={'ring':ObservedAttachment('ring',measured,.15)},
            collision_port=NS(ring_affordance=NS(permits_geometry=lambda *a,**kw:True)))
        events=policy.install(actor,config,ROOT,'insert_onto_rotating_peg')
        state,forecast=helper.observed(.15)
        state.bodies['ring']=ObjectState('ring',.15,Pose([0.,0.,.9]),np.tile([0.,0.,.9],(3,1)),np.zeros(3),np.zeros(3))
        state.perception_metadata['bodies']['ring']=dict(motion_mode='attached',
            mode_provenance='caller_controller_touch_action_proprioception',rotation_equivalence_evidence=Evidence())
        with patch.object(policy,'CurrentRotationEquivalence',Evidence):c.step(state,forecast,9.85)
        admitted=[e for e in events if e['event']=='measured_grasp_frame_placement_compensation_admitted']
        self.assertEqual(len(admitted),1,events)
        self.assertIsInstance(c.library.for_phase('align')[0],policy.CorrectedTemplate)
        self.assertIsInstance(c.library.for_phase('insert')[0],policy.CorrectedTemplate)
        c.step(state,forecast,9.85)
        self.assertEqual(sum(e['event']=='measured_grasp_frame_placement_compensation_admitted' for e in events),1)


if __name__=='__main__':unittest.main()
