from dataclasses import replace
import unittest
import numpy as np
import test_cup_reference_reserve as actual
from our_mani.multitask.contact_speed_reserve import install


class Tests(unittest.TestCase):
    def fixture(self):
        actor,model,robot=actual.Tests().fixture();c,m=actor.controller,actor.continuous_reference_manager
        c.goal=None;m.plan=None;m.plan_kind=None;c.phase_index=c.library.phase_order.index('acquire')
        m.last_confirmed['nominal_velocity']=np.zeros(7).tolist();m.last_confirmed['joints']=robot.joints.tolist()
        events=install(actor)
        return actor,model,robot,events

    def test_full_actual_and_reference_candidate_speed_reserve_is_stricter_without_changing_native_limits(self):
        actor,model,robot,_=self.fixture();m=actor.continuous_reference_manager
        fast=robot.joints.copy();fast[0]+=.04
        self.assertTrue(m.dual_gate.original.check_joints(robot,fast,np.zeros(7),.3).feasible)
        self.assertFalse(m.dual_gate.check_joints(robot,fast,np.zeros(7),.3).feasible)
        slow=robot.joints.copy();slow[0]+=.01
        self.assertTrue(m.dual_gate.check_joints(robot,slow,np.zeros(7),.3).feasible)
        np.testing.assert_allclose(model.acceleration_limits,6.)
        np.testing.assert_allclose(m.approach_transit.reference_gate.model.velocity_limits,.15)
        actor.controller.phase_index=actor.controller.library.phase_order.index('approach')
        self.assertTrue(m.dual_gate.check_joints(robot,fast,np.zeros(7),.3).feasible)

    def test_existing_collision_and_high_speed_brake_are_retained(self):
        actor,model,robot,_=self.fixture();m=actor.continuous_reference_manager
        v=np.zeros(7);v[0]=.8;robot=replace(robot,joint_velocities=v)
        m.current_state=replace(m.current_state,robot=robot);m.last_confirmed['nominal_velocity']=v.tolist()
        self.assertTrue(m.make_brake(robot,9.5,'contact_reserve_brake'))
        self.assertTrue(m.check_stream(robot,.05).feasible)
        actor,model,robot,_=self.fixture();model.dynamic_block=True
        self.assertFalse(actor.continuous_reference_manager.dual_gate.check_joints(robot,robot.joints,np.zeros(7),.3).feasible)


if __name__=='__main__':unittest.main()
