from dataclasses import replace
from types import SimpleNamespace as NS
import unittest
import numpy as np
import test_bounded_transport_v159 as prior
from our_mani.contracts import Pose
from our_mani.multitask.bounded_transit_v180 import BoundedApproach, sampled_cartesian_motion


class Tests(prior.Tests):
    def fixture(self):
        actor, model, robot = super().fixture()
        actor.continuous_reference_manager.approach_transit = BoundedApproach(
            actor.continuous_reference_manager, {'maximum_TCP_step_m': .2})
        return actor, model, robot

    def test_larger_proposal_can_advance_further_with_original_dual_gates_and_Cartesian_limits(self):
        actor, model, robot = self.fixture()
        m = actor.continuous_reference_manager
        self.assertTrue(m.approach_transit.try_approach(robot, 9.5))
        event = next(e for e in m.approach_transit.events if e['event']=='partial_approach_staged')
        distance = np.linalg.norm(np.array(event['end_xyzw'][:3])-robot.pose.position)
        self.assertGreater(distance, .08)
        admitted = [r for r in event['full_gate_trials'] if r.get('backup_brake_reason') == 'feasible_sampled_trajectory'][-1]
        self.assertTrue(all(p['feasible'] for p in admitted['additional_sampled_Cartesian_gates']))
        self.assertTrue(event['actual_and_reference_full_gate_feasible'])
        self.assertFalse(m.transport_phase_clock.epochs)

    def test_excessive_reference_cartesian_motion_is_rejected_instead_of_relaxing_control_caps(self):
        actor, model, robot = self.fixture()
        gate = NS(model=model, minimum_samples=21)
        proof = sampled_cartesian_motion(gate, robot, robot.joints+np.array([1.,0.,0.,0.,0.,0.,0.]),
            np.zeros(7), .5, .8, 2.)
        self.assertFalse(proof['feasible'])
        self.assertGreater(proof['maximum_sampled_linear_speed_m_s'], .8)
        self.assertFalse(proof['actual_motion_guaranteed'])


if __name__ == '__main__': unittest.main()
