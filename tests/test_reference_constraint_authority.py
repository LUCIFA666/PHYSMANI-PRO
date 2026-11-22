from dataclasses import replace
from types import MethodType
import unittest
import numpy as np
import test_button_segment_forecast_v155 as fixture
from our_mani.multitask.task_reference_v86 import ContinuousReferenceManager
from our_mani.multitask.reference_stop_guard_v274 import NextReferenceStopGuard
from our_mani.multitask.reference_constraint_authority import ConsistentStopGuard, install
from our_mani.multitask.continuous_transit import ConstraintContract


class Tests(unittest.TestCase):
    def fixture(self, cls):
        actor, model, robot = fixture.Tests().fixture(); m = actor.continuous_reference_manager
        m.check_stream = MethodType(ContinuousReferenceManager.check_stream,m)
        velocity = np.zeros(7); velocity[0] = .8
        robot = replace(robot,joint_velocities=velocity)
        m.current_state = replace(m.current_state,robot=robot)
        m.last_confirmed['nominal_velocity'] = velocity.tolist()
        guard = cls(m)
        guard.constraint_contract = ConstraintContract.from_manager(m)
        return actor, model, robot, guard

    def test_reproduces_six_bound_brake_rejected_by_actual_four_bound_stream(self):
        actor, _, robot, guard = self.fixture(NextReferenceStopGuard)
        m = actor.continuous_reference_manager
        self.assertTrue(guard.make_brake(robot,9.5,'regression'))
        result = m.check_stream(robot,.05)
        self.assertFalse(result.feasible)
        self.assertEqual(result.reason,'trajectory_joint_acceleration_limit')

    def test_consistent_full_brake_is_accepted_by_the_same_actual_stream(self):
        actor, model, robot, guard = self.fixture(ConsistentStopGuard)
        m = actor.continuous_reference_manager
        self.assertTrue(guard.make_brake(robot,9.5,'regression'))
        self.assertTrue(m.check_stream(robot,.05).feasible)
        self.assertTrue(m.pending['reference_gate_passed'])
        np.testing.assert_allclose(model.acceleration_limits,6.)
        np.testing.assert_allclose(m.reference_reserve_gate.model.acceleration_limits,4.)

    def test_future_stop_certificate_also_uses_the_same_four_bound(self):
        actor, _, robot, guard = self.fixture(ConsistentStopGuard)
        m = actor.continuous_reference_manager
        self.assertTrue(guard.make_brake(robot,9.5,'regression'))
        self.assertTrue(m.check_stream(robot,.05).feasible)
        plan, proof = guard.next_stop(robot,m.pending)
        self.assertIsNotNone(plan)
        self.assertEqual(proof['reference_acceleration_limits'],[4.]*7)
        self.assertLessEqual(proof['offset_reference_full_gate']['maximum_acceleration_fraction'],1.+1e-7)

    def test_collision_and_insufficient_global_budget_still_reject(self):
        for blocked, remaining in ((True,9.5),(False,.01)):
            actor, model, robot, guard = self.fixture(ConsistentStopGuard)
            model.dynamic_block = blocked
            self.assertFalse(guard.make_brake(robot,remaining,'regression'))
            self.assertIsNone(actor.continuous_reference_manager.plan)

    def test_install_reuses_the_service_already_captured_by_actual_stream(self):
        actor, _, robot, old = self.fixture(NextReferenceStopGuard)
        m = actor.continuous_reference_manager
        old.install_stream(); old.install_confirmation()
        m.make_brake = MethodType(lambda instance,*a,**kw:old.make_brake(*a,**kw),m)
        original_stream = m.check_stream
        installed = install(actor)
        self.assertIs(installed, old)
        self.assertIs(m.check_stream, original_stream)
        self.assertTrue(m.make_brake(robot,9.5,'actor-local-regression'))
        self.assertTrue(m.check_stream(robot,.05).feasible)
        m.plan_kind = 'normal'
        self.assertTrue(m.check_stream(robot,.05).feasible)
        self.assertEqual(old.queries,1)
        proof = m.pending['next_reference_stop_guard_v274']['proof']
        self.assertEqual(proof['reference_acceleration_limits'],[4.]*7)
        self.assertTrue(actor.metrics_report()['reference_constraint_authority']['existing_stopping_service_reused'])


if __name__ == '__main__':
    unittest.main()
