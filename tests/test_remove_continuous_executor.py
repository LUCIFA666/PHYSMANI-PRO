from pathlib import Path
from dataclasses import replace
from types import SimpleNamespace as NS
from unittest.mock import patch
import ast
import json
import unittest
import numpy as np
import test_button_segment_forecast_v155 as fixture
from our_mani.contracts import Pose
from our_mani.multitask.interaction import InteractionLibrary,COMMAND_GOAL
from our_mani.multitask.reference_constraint_authority import install as install_authority
from our_mani.multitask.remove_continuous_executor import RemoveContinuousBackend,binding,TASKS
from our_mani.multitask.task_executor_registry import SharedTaskBackend
ROOT=Path(__file__).resolve().parents[1]


class Tests(unittest.TestCase):
    def fixture(self,task):
        actor,model,robot=fixture.Tests().fixture();c,m=actor.controller,actor.continuous_reference_manager
        del m.reference_reserve_gate
        config=json.loads((ROOT/'artifacts/framework/multitask_r10/registered_tasks_v408_setup'/
            (task+'_configuration.json')).read_text())
        library=InteractionLibrary.load(ROOT/config['learned_artifacts'][task]['interaction']['path'])
        c.library=c.generator.library=library;c.goal=None;c.phase_index=0
        install_authority(actor);config.update(binding(ROOT,config,task))
        with patch.object(SharedTaskBackend,'actor',return_value=(actor,{})):
            RemoveContinuousBackend(task).actor(dict(config=config,args=NS(task=task)),None,None)
        return actor,model,robot

    def test_both_actual_task_libraries_keep_native6_and_same_stop_callbacks_with_reference4(self):
        for task in TASKS:
            with self.subTest(task=task):
                actor,model,robot=self.fixture(task);m=actor.continuous_reference_manager
                v=np.zeros(7);v[0]=.8;robot=replace(robot,joint_velocities=v)
                m.current_state=replace(m.current_state,robot=robot);m.last_confirmed['nominal_velocity']=v.tolist()
                self.assertTrue(m.make_brake(robot,9.5,'remove_reference_regression'))
                self.assertTrue(m.check_stream(robot,.05).feasible)
                np.testing.assert_allclose(model.acceleration_limits,6.)
                np.testing.assert_allclose(m.reference_reserve_gate.model.acceleration_limits,4.)
                self.assertEqual(m.retarget.config.maximum_reissues_per_phase,16)
                for phase in ('approach','acquire','detach'):
                    self.assertTrue(m.retarget._eligible(NS(phase=phase,template=actor.controller.library.for_phase(phase)[0])))

    def test_actual_dynamic_collision_is_still_required_for_braking(self):
        actor,model,robot=self.fixture(TASKS[0]);model.dynamic_block=True
        self.assertFalse(actor.continuous_reference_manager.make_brake(robot,9.5,'collision'))

    def test_author_approach_is_wp1_five_cm_and_requires_current_canonical_geometry(self):
        for task in TASKS:
            actor,_,robot=self.fixture(task);c=actor.controller
            t=c.library.for_phase('approach')[0]
            self.assertEqual(t.reference_time_basis,COMMAND_GOAL)
            c.goal=NS(phase='approach',template=t,selected_at=.5,phase_budget=2.)
            target=Pose(robot.pose.position+[.049,0.,0.])
            self.assertFalse(c._phase_done(NS(robot=robot),target,public_geometry=False))
            self.assertTrue(c._phase_done(NS(robot=robot),target,public_geometry=True))
            self.assertFalse(c._phase_done(NS(robot=robot),Pose(robot.pose.position+[.051,0.,0.]),public_geometry=True))
            spec=c.library.task_spec
            self.assertEqual(spec.phase_skills[0].TRAIN_expert_stages,('wp0','wp1'))
            source=Path('/home/ubuntu/桌面/phsimani/PhysMani/third_party/RLBench/rlbench/tasks')/(task+'.py')
            values={n.targets[0].id:n.value.value for n in ast.walk(ast.parse(source.read_text()))
                if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name)
                and isinstance(n.value,ast.Constant)}
            self.assertEqual(values['th_w1'],.05)


if __name__=='__main__':unittest.main()
