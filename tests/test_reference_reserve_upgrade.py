from pathlib import Path
from dataclasses import replace
import json
import unittest
import numpy as np
import test_button_segment_forecast_v155 as fixture
from our_mani.multitask.high_peg_primary_fitting_v231 import PrimarySequenceLibrary
from our_mani.multitask.reference_constraint_authority import install as install_authority
from our_mani.multitask.reference_reserve_upgrade import binding,install,TASK
ROOT=Path(__file__).resolve().parents[1]


class Tests(unittest.TestCase):
    def fixture(self):
        actor,model,robot=fixture.Tests().fixture();c,m=actor.controller,actor.continuous_reference_manager
        del m.reference_reserve_gate
        config=json.loads((ROOT/'artifacts/framework/multitask_r10/registered_tasks_v408_setup'/
            (TASK+'_configuration.json')).read_text())
        library=PrimarySequenceLibrary.load(ROOT/config['learned_artifacts'][TASK]['interaction']['path'])
        c.library=c.generator.library=library;c.goal=None;c.phase_index=0
        install_authority(actor)
        config['reference_reserve_upgrade']=binding(ROOT,config,TASK)
        guard=install(actor,config,ROOT,TASK)
        return actor,model,robot,guard

    def test_actual_old_manager_and_stop_callbacks_share_reference4_without_changing_native6(self):
        actor,model,robot,guard=self.fixture();m=actor.continuous_reference_manager
        velocity=np.zeros(7);velocity[0]=.8
        robot=replace(robot,joint_velocities=velocity)
        m.current_state=replace(m.current_state,robot=robot)
        m.last_confirmed['nominal_velocity']=velocity.tolist()
        self.assertTrue(m.make_brake(robot,9.5,'upgrade_regression'))
        self.assertTrue(m.check_stream(robot,.05).feasible)
        self.assertGreaterEqual(m.plan.end-m.plan.start,.2-1e-8)
        self.assertTrue(m.pending['reference_gate_passed'])
        np.testing.assert_allclose(model.acceleration_limits,6.)
        np.testing.assert_allclose(m.reference_reserve_gate.model.acceleration_limits,4.)
        np.testing.assert_allclose(m.approach_transit.reference_gate.model.acceleration_limits,4.)
        self.assertEqual(guard.constraint_contract.reference_acceleration,(4.,)*7)
        self.assertEqual(type(m).__module__,'our_mani.multitask.task_reference_v37')

    def test_reference_reserve_does_not_waive_original_collision_rejections(self):
        actor,model,robot,_=self.fixture();model.dynamic_block=True
        self.assertFalse(actor.continuous_reference_manager.make_brake(robot,9.5,'blocked'))


if __name__=='__main__':unittest.main()
