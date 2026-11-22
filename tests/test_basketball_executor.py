from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
import json
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
import test_button_segment_forecast_v155 as fixture
from our_mani.multitask.interaction import InteractionLibrary,GripperTimingConfig
from our_mani.multitask.basketball_executor_native import BasketballBackend,binding,TASKS
from our_mani.multitask.task_executor_registry import SharedTaskBackend
ROOT=Path(__file__).resolve().parents[1]


class Tests(unittest.TestCase):
    def test_actual_backend_installs_both_task_libraries_and_certifies_downward_approach(self):
        for task in TASKS:
            config=json.loads((ROOT/'artifacts/framework/multitask_r10/registered_tasks_v408_setup'/
                (task+'_configuration.json')).read_text())
            config['basketball_executor']=binding(ROOT,config,task)
            actor,model,robot=fixture.Tests().fixture();c,m=actor.controller,actor.continuous_reference_manager
            library=InteractionLibrary.load(ROOT/config['learned_artifacts'][task]['interaction']['path'])
            c.library=c.generator.library=library;c.goal=None;c.phase_index=0
            c.gripper_timing_config=GripperTimingConfig(**config['basketball_executor']['previous_extra_fitted_close_gate'])
            prepared=dict(config=config,args=NS(task=task))
            with patch.object(SharedTaskBackend,'actor',return_value=(actor,{})):
                result,_=BasketballBackend(task).actor(prepared,None,None)
            self.assertIs(result,actor)
            self.assertEqual(c.gripper_timing_config,GripperTimingConfig(.02,.1))
            q=robot.joints.copy();q[2]=1.47
            q[3:6]=Rotation.from_matrix(library.for_phase('approach')[0].at(1.)[0].rotation.copy()).as_rotvec()
            robot=replace(robot,joints=q,pose=model.forward(q))
            m.current_state=replace(m.current_state,robot=robot)
            m.last_confirmed.update(joints=q.tolist())
            c.generator.last_diagnostics={'feasible_candidates':0}
            self.assertTrue(m.approach_transit.try_approach(robot,9.5),m.approach_transit.events)
            self.assertLess(model.forward(m.plan.q1).position[2],robot.pose.position[2]-.001)
            self.assertTrue(m.pending['reference_gate_passed'])
            np.testing.assert_allclose(model.acceleration_limits,6.)


if __name__=='__main__':unittest.main()
