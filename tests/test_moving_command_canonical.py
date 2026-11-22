from dataclasses import replace
from types import SimpleNamespace as NS,MethodType
import unittest
import numpy as np
from our_mani.contracts import Pose
from our_mani.multitask.interaction import CURRENT
from our_mani.multitask.moving_command_canonical import install
import test_ring_causal_retarget_v371 as actual


class Tests(unittest.TestCase):
    def fixture(self):
        helper=actual.Tests();c,_,_,_=helper.fixture('insert_onto_rotating_peg');goal=c.goal;c.goal=None
        c.step=MethodType(lambda self,*a:{'command':'hold'},c)
        m=NS(confirmations=[],config=NS(dt=.05),check_stream=lambda *a:None)
        actor=NS(controller=c,continuous_reference_manager=m,metrics_report=lambda:{})
        events=install(actor)
        state,forecast=helper.observed(2.)
        c.step(state,forecast,8.)
        source=goal.template;live=c.library.for_phase('align')[0]
        c.goal=replace(goal,template=live)
        canonical=Pose.from_xyzw(source.completion_geometry_xyzw);ref=state.bodies['peg0'].pose
        target=Pose(ref.position+ref.rotation@canonical.position,ref.rotation@canonical.rotation)
        state=replace(state,robot=replace(state.robot,pose=target))
        return c,state,target,live,events

    def test_control_current_and_completion_still_require_the_original_fresh_canonical_pose(self):
        c,state,target,live,events=self.fixture()
        self.assertEqual(live.reference_time_basis,CURRENT)
        self.assertIsNone(live.completion_geometry_xyzw)
        self.assertTrue(c._phase_done(state,Pose([100.,100.,100.])))
        shifted=replace(state,robot=replace(state.robot,pose=Pose(target.position+[.05,0.,0.],target.rotation)))
        self.assertFalse(c._phase_done(shifted,target))
        self.assertEqual([e['phase'] for e in events],['align','insert'])
        self.assertTrue(all(e['original_position_tolerance_m']<=.03 for e in events))

    def test_stale_reference_cannot_be_called_physical_completion(self):
        c,state,target,_,_=self.fixture()
        state.bodies['peg0']=replace(state.bodies['peg0'],timestamp=1.5)
        self.assertFalse(c._phase_done(state,target))



class CombinedActivationTests(unittest.TestCase):
    def test_actual_backend_calibrates_before_converting_and_keeps_corrected_canonical_geometry(self):
        from pathlib import Path
        from unittest.mock import patch
        import json
        from our_mani.contracts import ObjectState
        from our_mani.multitask.forecasting import ObservedAttachment
        from our_mani.multitask import grasp_frame_compensation_native_compat as comp
        from our_mani.multitask.live_grasp_executor_native import LiveGraspBackend,binding
        from our_mani.multitask.task_executor_registry import SharedTaskBackend
        class Evidence:pass
        root=Path(__file__).resolve().parents[1]
        config=json.loads((root/'artifacts/framework/multitask_r10/grasp_compensated_v423_setup/configuration.json').read_text())
        task='insert_onto_rotating_peg';config['moving_command_executor']=binding(root,config,task)
        helper=actual.Tests();c,_,_,_=helper.fixture(task);c.goal=None
        c.step=MethodType(lambda self,*a:{'command':'hold'},c)
        prior=json.loads((root/config['grasp_frame_compensation']['prior']['path']).read_text())
        mean=Pose.from_xyzw(prior['tool_to_ring_xyzw']);measured=Pose(mean.position+[.002,0.,-.003],mean.rotation)
        m=NS(confirmations=[],config=NS(dt=.05),check_stream=lambda *a:None)
        actor=NS(controller=c,continuous_reference_manager=m,metrics_report=lambda:{},
            attachments={'ring':ObservedAttachment('ring',measured,.15)},
            collision_port=NS(ring_affordance=NS(permits_geometry=lambda *a,**kw:True)))
        prepared=dict(config=config,args=NS(task=task))
        with patch.object(SharedTaskBackend,'actor',return_value=(actor,{})):
            LiveGraspBackend(task).actor(prepared,None,None)
        state,forecast=helper.observed(.15)
        state.bodies['ring']=ObjectState('ring',.15,Pose([0.,0.,.9]),np.tile([0.,0.,.9],(3,1)),np.zeros(3),np.zeros(3))
        state.perception_metadata['bodies']['ring']=dict(motion_mode='attached',
            mode_provenance='caller_controller_touch_action_proprioception',rotation_equivalence_evidence=Evidence())
        with patch.object(comp,'CurrentRotationEquivalence',Evidence):c.step(state,forecast,9.85)
        metrics=actor.metrics_report()
        self.assertTrue(any(e['event']=='measured_grasp_frame_placement_compensation_admitted' for e in metrics['grasp_frame_compensation']['events']))
        converted=[e for e in metrics['moving_command_canonical']['events'] if e['event']=='live_control_with_original_canonical_completion']
        self.assertEqual([e['phase'] for e in converted],['align','insert'])
        self.assertIsInstance(c.library.for_phase('align')[0],comp.CorrectedTemplate)
        self.assertEqual(c.library.for_phase('align')[0].reference_time_basis,CURRENT)


if __name__=='__main__':unittest.main()
