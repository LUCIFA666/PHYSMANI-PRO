from dataclasses import replace
from pathlib import Path
import json
import unittest
import numpy as np
from our_mani.contracts import Pose
from our_mani.multitask.author_approach_handoff import binding,install
import test_command_reference_service as actual
ROOT=Path(__file__).resolve().parents[1]


class Tests(unittest.TestCase):
    def test_actual_buzz_controller_uses_author_distance_without_waiting_for_fitted_duration(self):
        for task in ('beat_the_rotating_buzz','beat_the_rotating_buzz_high_speed'):
            helper=actual.Tests();actor,c,m,_,_=helper.fixture(task,'approach')
            config=json.loads((ROOT/'artifacts/framework/multitask_r10/buzz_command_v412_setup'/
                (task+'_configuration.json')).read_text())
            config['author_approach_handoff']={task:binding(ROOT,config,task)}
            # Bind before execution, then restore this synthetic active goal.
            goal=c.goal;c.goal=None
            actor.continuous_reference_manager.confirmations=[]
            install(actor,config,ROOT,task);c.goal=goal
            state,_=helper.observed(.15,goal.reference_id)
            target=Pose(np.zeros(3))
            for offset,expected in ((.099,True),(.101,False)):
                robot=replace(state.robot,pose=Pose([offset,0.,0.]))
                current=replace(state,robot=robot)
                self.assertEqual(c._phase_done(current,target,public_geometry=True),expected)
                self.assertFalse(c._phase_done(current,target,public_geometry=False))
            self.assertEqual(c.goal.selected_at,goal.selected_at)
            self.assertLess(state.robot.timestamp-goal.selected_at,goal.template.duration_seconds)


if __name__=='__main__':unittest.main()
