"""Stricter reference planning must leave the actual six-unit monitor intact."""
from types import SimpleNamespace
from unittest.mock import patch
import unittest
import numpy as np

import test_button_intent_actor_v74 as original_test
from our_mani.multitask import button_intent_actor_v88
from our_mani.multitask.task_reference_v86 import ContinuousReferenceManager
from our_mani.contracts import Pose, RobotState
from our_mani.reachability import TimedReachability
from test_multitask_interaction import CartesianRobot


def robot(time=0., q=None, v=None):
    q=np.zeros(7) if q is None else np.asarray(q,float)
    v=np.zeros(7) if v is None else np.asarray(v,float)
    return RobotState(time,Pose(np.zeros(3)),np.zeros(3),np.zeros(3),q,v)


class FactoryTests(original_test.Tests):
    # Reuse the actual fake-port factory fixtures with the new source binding.
    def actor(self):
        with patch('test_button_intent_actor_v74.binding',button_intent_actor_v88.binding), \
             patch('test_button_intent_actor_v74.create_actor',button_intent_actor_v88.create_actor):
            return super().actor()

    def test_actual_factory_installs_final_controller_guard_and_press_prefix_with_original_executor(self):
        super().test_actual_factory_installs_final_controller_guard_and_press_prefix_with_original_executor()
        actor=self.actor();m=actor.continuous_reference_manager
        self.assertEqual(type(m).__module__,'our_mani.multitask.task_reference_v86')
        np.testing.assert_allclose(m.reference_reserve_gate.model.acceleration_limits,
            actor.robot_model.acceleration_limits*(2./3.))


class ReserveTests(unittest.TestCase):
    def manager(self):
        model=CartesianRobot();model.acceleration_limits=np.full(7,6.)
        gate=TimedReachability(model)
        c=SimpleNamespace(original_reachability=gate,phase_index=0,
            library=SimpleNamespace(phase_order=('approach',)),
            generator=SimpleNamespace(config=SimpleNamespace(dt=.05)),
            mpc=SimpleNamespace(config=SimpleNamespace(dt=.05)))
        m=ContinuousReferenceManager(c)
        m.last_confirmed=dict(time=0.,joints=np.zeros(7).tolist(),nominal_velocity=np.zeros(7).tolist(),command='joint_trajectory')
        c.original_reachability=m.dual_gate
        return model,m

    def test_original_feasible_five_unit_plan_is_rejected_by_stricter_reference(self):
        model,m=self.manager();r=robot();q=np.full(7,5./6.)
        self.assertTrue(TimedReachability.check_joints(m.native_gate,r,q,np.zeros(7),1.).feasible)
        checked=m.dual_gate.check_joints(r,q,np.zeros(7),1.)
        self.assertFalse(checked.feasible)
        self.assertEqual(checked.reason,'reference_continuity_trajectory_joint_acceleration_limit')
        np.testing.assert_array_equal(model.acceleration_limits,np.full(7,6.))

    def test_accepted_reference_retains_collision_check_and_physical_limits(self):
        model,m=self.manager();r=robot();q=np.full(7,.5)
        self.assertTrue(m.dual_gate.check_joints(r,q,np.zeros(7),1.).feasible)
        model.collision=lambda joints,offset: True
        self.assertFalse(m.dual_gate.check_joints(r,q,np.zeros(7),1.).feasible)
        np.testing.assert_array_equal(m.reference_reserve_gate.model.acceleration_limits,np.full(7,4.))
        np.testing.assert_array_equal(model.acceleration_limits,np.full(7,6.))

    def test_measured_five_unit_acceleration_is_checked_against_original_six(self):
        model,m=self.manager()
        m.observe(robot(0.))
        m.observe(robot(.05))
        event=m.observe(robot(.1,np.full(7,.0125)))
        np.testing.assert_allclose(event['backward_window_acceleration'],np.full(7,5.))
        self.assertFalse(event['violations'])
        self.assertFalse(m.qualification_failed)


if __name__=='__main__':unittest.main()
