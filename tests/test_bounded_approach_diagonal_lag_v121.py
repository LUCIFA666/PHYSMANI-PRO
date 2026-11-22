from dataclasses import replace
import unittest
import numpy as np
import test_task_reference_v37 as original_test
from our_mani.multitask.bounded_approach_v121 import BoundedApproach,transit_waypoint
from our_mani.contracts import Pose


class Tests(unittest.TestCase):
    def fixture(self,lag=.05):
        actor,model,robot=original_test.PartialApproachTests().setup_scene()
        library=actor.controller.library
        library.templates=tuple(replace(t,lag_seconds=lag) if t.phase=='approach' else t for t in library.templates)
        actor.controller.generator.last_diagnostics={'feasible_candidates':0}
        m=actor.continuous_reference_manager;m.approach_transit=BoundedApproach(m)
        return actor,model,robot

    def test_horizontal_distance_does_not_prevent_descent(self):
        current=Pose(np.array([.28,0.,1.47]),np.eye(3))
        target=Pose(np.array([.28,.29,.8]),np.eye(3))
        waypoint=transit_waypoint(current,target,.08)
        delta=waypoint.position-current.position
        fraction=.08/np.linalg.norm(delta)
        first=current.position+fraction*delta
        self.assertLess(first[2],current.position[2]);self.assertGreater(first[1],current.position[1])
        self.assertLessEqual(np.linalg.norm(first-current.position),.08+1e-12)

    def test_positive_lag_does_not_extrapolate_and_requires_both_motion_proofs(self):
        actor,model,r=self.fixture();m=actor.continuous_reference_manager
        self.assertTrue(m.approach_transit.try_approach(r,9.5))
        e=next(e for e in m.approach_transit.events if e['event']=='partial_approach_staged')
        self.assertLessEqual(e['predicted_reference_time'],m.current_forecast.times[-1]+1e-8)
        self.assertTrue(e['actual_and_reference_full_gate_feasible']);self.assertGreater(len(model.collision_calls),0)

    def test_missing_history_cannot_be_extrapolated(self):
        actor,_,r=self.fixture(1.1);m=actor.continuous_reference_manager
        self.assertFalse(m.approach_transit.try_approach(r,9.5));self.assertIsNone(m.pending)

    def test_collision_still_blocks_the_diagonal_path(self):
        actor,model,r=self.fixture();model.dynamic_block=True;m=actor.continuous_reference_manager
        self.assertFalse(m.approach_transit.try_approach(r,9.5));self.assertIsNone(m.pending)

    def test_global_budget_is_not_extended_to_finish_approach(self):
        actor,_,r=self.fixture();m=actor.continuous_reference_manager
        self.assertFalse(m.approach_transit.try_approach(r,.05));self.assertIsNone(m.pending)


if __name__=='__main__':unittest.main()
