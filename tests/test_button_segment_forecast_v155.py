from dataclasses import replace
import unittest
import numpy as np
import test_task_reference_v37 as original_test
from our_mani.multitask.button_intent_prefix_v155 import BoundedApproach
from our_mani.multitask.task_reference_v86 import reserved_reference_gate


class Tests(unittest.TestCase):
    def fixture(self,lag=0.):
        actor,model,robot=original_test.PartialApproachTests().setup_scene()
        library=actor.controller.library
        library.templates=tuple(replace(t,lag_seconds=lag) if t.phase=='approach' else t for t in library.templates)
        actor.controller.generator.last_diagnostics={'feasible_candidates':0}
        m=actor.continuous_reference_manager;m.reference_reserve_gate=reserved_reference_gate(m.native_gate)
        m.approach_transit=BoundedApproach(m)
        return actor,model,robot
    def test_segment_targets_its_end_and_keeps_both_complete_motion_proofs(self):
        actor,model,r=self.fixture(.05);m=actor.continuous_reference_manager
        self.assertTrue(m.approach_transit.try_approach(r,9.5))
        e=next(e for e in m.approach_transit.events if e['event']=='partial_approach_staged')
        self.assertAlmostEqual(e['predicted_reference_time'],r.timestamp+.5+.05)
        self.assertLessEqual(e['predicted_reference_time'],m.current_forecast.times[-1]+1e-8)
        self.assertTrue(e['actual_and_reference_full_gate_feasible']);self.assertGreater(len(model.collision_calls),0)
        np.testing.assert_allclose(m.reference_reserve_gate.model.acceleration_limits,model.acceleration_limits*(2./3.))
    def test_original_dynamic_collision_gate_still_blocks_every_segment(self):
        actor,model,r=self.fixture();model.dynamic_block=True;m=actor.continuous_reference_manager
        self.assertFalse(m.approach_transit.try_approach(r,9.5));self.assertIsNone(m.pending)
    def test_unavailable_lag_horizon_is_not_extrapolated(self):
        actor,_,r=self.fixture(1.1);m=actor.continuous_reference_manager
        self.assertFalse(m.approach_transit.try_approach(r,9.5));self.assertIsNone(m.pending)
    def test_stale_forecast_is_not_used_for_the_nearer_target(self):
        actor,_,r=self.fixture();m=actor.continuous_reference_manager;m.current_forecast.metadata['observed_until']=.3
        self.assertFalse(m.approach_transit.try_approach(r,9.5));self.assertIsNone(m.pending)
    def test_global_budget_and_suffix_remain_reserved(self):
        actor,_,r=self.fixture();m=actor.continuous_reference_manager
        self.assertFalse(m.approach_transit.try_approach(r,.05));self.assertIsNone(m.pending)


if __name__=='__main__':unittest.main()
