from dataclasses import replace
from types import MethodType
import unittest
import numpy as np
import test_button_segment_forecast_v155 as fixture
from our_mani.multitask.task_reference_v86 import ContinuousReferenceManager
from our_mani.multitask.button_reference_transit_v386 import BoundedApproach as ButtonTransit
from our_mani.multitask.receding_task_transit import RecedingTaskTransit
from our_mani.multitask.bounded_approach_v36 import transit_waypoint
from our_mani.contracts import Pose


class Tests(unittest.TestCase):
    def fixture(self):
        actor, model, robot = fixture.Tests().fixture()
        m = actor.continuous_reference_manager
        m.check_stream = MethodType(ContinuousReferenceManager.check_stream, m)
        old = ButtonTransit(m, {'maximum_TCP_step_m': .2})
        m.approach_transit = RecedingTaskTransit(m, old)
        return actor, model, robot, old

    def test_first_segment_preserves_original_task_generator_exactly(self):
        actor, _, robot, old = self.fixture(); m = actor.continuous_reference_manager
        self.assertTrue(m.approach_transit.try_approach(robot, 9.5))
        self.assertIs(m.approach_transit.original, old)
        self.assertEqual(m.approach_transit.metrics()['receding_task_transit']['original_implementation'],
            'our_mani.multitask.button_reference_transit_v386.BoundedApproach')
        # The bin adapter's horizontal-before-descent geometry is retained by
        # delegation; it must not be replaced with a generic 3-D waypoint.
        point = transit_waypoint(Pose([0., 0., 1.]), Pose([.5, 0., .8]), .08)
        self.assertEqual(point.position[2], 1.)

    def test_nonzero_confirmed_velocity_and_deadline_survive_renewal(self):
        actor, model, robot, old = self.fixture(); m = actor.continuous_reference_manager
        self.assertTrue(m.approach_transit.try_approach(robot, 9.5))
        deadline = old.active['deadline']; q, v = m.plan.at(.7)
        robot = replace(robot, timestamp=.7, joints=q, joint_velocities=v, pose=model.forward(q))
        m.last_confirmed.update(time=.7, joints=q.tolist(), nominal_velocity=v.tolist())
        m.current_state = replace(m.current_state, robot=robot,
            bodies={k: replace(b, timestamp=.7) for k,b in m.current_state.bodies.items()})
        m.current_forecast.metadata['observed_until'] = .7
        m.forecast_observed_until = .7
        self.assertTrue(m.approach_transit.try_approach(robot, 9.3))
        np.testing.assert_array_equal(m.plan.v0, v)
        self.assertEqual(old.active['deadline'], deadline)

    def test_original_collision_failure_still_rejects(self):
        actor, model, robot, _ = self.fixture(); model.dynamic_block = True
        self.assertFalse(actor.continuous_reference_manager.approach_transit.try_approach(robot, 9.5))


if __name__ == '__main__':
    unittest.main()
