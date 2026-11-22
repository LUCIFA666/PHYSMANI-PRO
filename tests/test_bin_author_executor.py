from pathlib import Path
from dataclasses import replace
from types import SimpleNamespace as NS
import json
import unittest
import numpy as np
from our_mani.contracts import Pose,ObjectState
from our_mani.multitask.interaction import InteractionLibrary,PolicyState
from our_mani.multitask.bin_author_executor_native import install_guard,TASK
import test_task_reference_v37 as actual
ROOT=Path(__file__).resolve().parents[1]


class Tests(unittest.TestCase):
    def test_actual_controller_uses_raw_author_wp0_and_current_public_body(self):
        actor,_,_=actual.installed_actor();c=actor.controller
        config=json.loads((ROOT/'artifacts/framework/multitask_r10/registered_tasks_v408_setup'/
            (TASK+'_configuration.json')).read_text())
        c.library=InteractionLibrary.load(ROOT/config['learned_artifacts'][TASK]['interaction']['path'])
        c.generator.library=c.library;c.phase_index=0
        model=json.loads((ROOT/'artifacts/framework/multitask_r10/bin_author_approach_v427'/TASK/'model.json').read_text())
        events=install_guard(actor,model)
        t=c.library.for_phase('approach')[0]
        c.goal=NS(phase='approach',template=t,reference_id='rubbish',selected_at=1.)
        body=ObjectState('rubbish',1.,Pose([.1,.2,.767]),np.tile([.1,.2,.767],(3,1)),np.zeros(3),np.zeros(3))
        point=body.pose.position+np.asarray(model['target_offset_world'])
        from tests.test_feedback_interaction import robot
        state=PolicyState(robot(np.zeros(7),np.zeros(7),1.),{'rubbish':body},{'bodies':{'rubbish':dict(
            directly_visible=True,registration_valid=True,motion_mode='external')}},{'rubbish':('rubbish',)})
        for offset,expected in ((.099,True),(.101,False)):
            r=replace(state.robot,pose=Pose(point+[0.,0.,offset]))
            self.assertEqual(c._phase_done(replace(state,robot=r),Pose([100.,100.,100.])),expected)
        state.bodies['rubbish']=replace(body,timestamp=.5)
        self.assertFalse(c._phase_done(state,Pose(point)))
        self.assertTrue(events)
        self.assertEqual(model['parent_count'],100)
        self.assertFalse(model['native_success_or_completion_used_for_source_selection'])


if __name__=='__main__':unittest.main()
