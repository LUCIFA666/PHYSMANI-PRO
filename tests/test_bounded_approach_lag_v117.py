from dataclasses import replace
import unittest
import numpy as np
import test_task_reference_v37 as original_test
from our_mani.multitask.bounded_approach_v117 import BoundedApproach


class Tests(unittest.TestCase):
    def fixture(self,lag):
        actor,model,robot=original_test.PartialApproachTests().setup_scene()
        library=actor.controller.library
        library.templates=tuple(replace(t,lag_seconds=lag) if t.phase=='approach' else t for t in library.templates)
        actor.controller.generator.last_diagnostics={'feasible_candidates':0}
        return actor,model,robot

    def test_positive_fitted_lag_keeps_query_inside_original_full_horizon(self):
        actor,model,robot=self.fixture(.05);m=actor.continuous_reference_manager
        self.assertFalse(m.approach_transit.try_approach(robot,9.5))
        m.approach_transit=BoundedApproach(m)
        self.assertTrue(m.approach_transit.try_approach(robot,9.5))
        e=next(e for e in m.approach_transit.events if e['event']=='partial_approach_staged')
        self.assertLessEqual(e['predicted_reference_time'],m.current_forecast.times[-1]+1e-8)
        self.assertTrue(e['actual_and_reference_full_gate_feasible']);self.assertGreater(len(model.collision_calls),0)

    def test_zero_lag_preserves_original_accepted_joint_segment(self):
        old,_,r0=self.fixture(0.);new,_,r1=self.fixture(0.)
        self.assertTrue(old.continuous_reference_manager.approach_transit.try_approach(r0,9.5))
        m=new.continuous_reference_manager;m.approach_transit=BoundedApproach(m)
        self.assertTrue(m.approach_transit.try_approach(r1,9.5))
        np.testing.assert_array_equal(old.continuous_reference_manager.plan.q1,m.plan.q1)
        np.testing.assert_array_equal(old.continuous_reference_manager.plan.v1,m.plan.v1)

    def test_no_history_beyond_lag_cannot_create_an_extrapolated_segment(self):
        actor,model,r=self.fixture(1.1);m=actor.continuous_reference_manager;m.approach_transit=BoundedApproach(m)
        self.assertFalse(m.approach_transit.try_approach(r,9.5));self.assertIsNone(m.pending)

    def test_dynamic_collision_gate_still_blocks_all_new_segments(self):
        actor,model,r=self.fixture(.05);model.dynamic_block=True
        m=actor.continuous_reference_manager;m.approach_transit=BoundedApproach(m)
        self.assertFalse(m.approach_transit.try_approach(r,9.5));self.assertIsNone(m.pending)


if __name__=='__main__':unittest.main()
