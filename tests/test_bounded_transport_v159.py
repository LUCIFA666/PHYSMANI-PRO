from pathlib import Path
from dataclasses import replace
import json,unittest
import numpy as np
import test_task_reference_v37 as original_test
from our_mani.contracts import Pose,ObjectState
from our_mani.dynamics import ConstantVelocityDynamics
from our_mani.multitask.interaction import PolicyState,CandidateConfig
from our_mani.multitask.normal_cup_primary_library_v109 import PrimarySequenceLibrary
from our_mani.multitask.transport_phase_clock_v158 import TransportPhaseClock
from our_mani.multitask.bounded_transport_v159 import BoundedApproach

ROOT=Path(__file__).resolve().parents[1]


class Tests(unittest.TestCase):
    def fixture(self):
        actor,model,r=original_test.PartialApproachTests().setup_scene();r=replace(r,gripper_open=False)
        c=actor.controller;m=actor.continuous_reference_manager
        library=PrimarySequenceLibrary.load(ROOT/'artifacts/framework/multitask_r10/normal_cup_primary_policy_v110/interaction.json')
        c.library=c.generator.library=library;c.phase_index=library.phase_order.index('transport');c.goal=None
        c.generator.last_diagnostics={'feasible_candidates':0}
        body=ObjectState('spoke0',r.timestamp,Pose([1.5,0.,.7]),np.array([[1.5,0.,.7],[1.51,0.,.7],[1.5,.01,.7]]),np.zeros(3),np.zeros(3))
        state=PolicyState(r,{'spoke0':body},{'bodies':{'spoke0':dict(directly_visible=True,registration_valid=True,orientation_observable=True,motion_mode='external')}},
            {'selected_spoke':('spoke0',),'cup0':('cup0',)})
        forecast=ConstantVelocityDynamics().predict([body],1.1,.05);forecast.metadata['observed_until']=r.timestamp
        m.current_state,m.current_forecast=state,forecast;m.transport_phase_clock=TransportPhaseClock(library,c.generator.config)
        m.approach_transit=BoundedApproach(m);actor.requested_opening=0.;m.last_confirmed['opening']=0.
        return actor,model,r
    def test_actual_fitted_transport_can_stage_a_bounded_dual_proof_segment(self):
        actor,model,r=self.fixture();m=actor.continuous_reference_manager
        self.assertTrue(m.approach_transit.try_approach(r,9.5))
        self.assertEqual(m.approach_transit.active['phase'],'transport')
        self.assertAlmostEqual(m.approach_transit.active['deadline'],r.timestamp+2.3)
        event=next(e for e in m.approach_transit.events if e['event']=='partial_approach_staged')
        self.assertTrue(event['actual_and_reference_full_gate_feasible']);self.assertGreater(len(model.collision_calls),0)
        self.assertAlmostEqual(event['predicted_reference_time'],r.timestamp+.5)
        self.assertFalse(m.transport_phase_clock.epochs);self.assertEqual(actor.requested_opening,0.)
    def test_dynamic_collision_still_blocks_transport(self):
        actor,model,r=self.fixture();model.dynamic_block=True;m=actor.continuous_reference_manager
        self.assertFalse(m.approach_transit.try_approach(r,9.5));self.assertIsNone(m.pending)
        self.assertFalse(m.transport_phase_clock.epochs)
    def test_global_suffix_budget_cannot_be_spent_by_transport_prefix(self):
        actor,_,r=self.fixture();m=actor.continuous_reference_manager
        self.assertFalse(m.approach_transit.try_approach(r,.5));self.assertIsNone(m.pending)
    def test_expired_confirmed_transport_clock_is_not_renewed(self):
        actor,_,r=self.fixture();m=actor.continuous_reference_manager
        m.transport_phase_clock.confirm(actor.controller.phase_index,0.,.05,.4)
        self.assertFalse(m.approach_transit.try_approach(r,9.5));self.assertIsNone(m.pending)
        self.assertEqual(m.transport_phase_clock.epochs[actor.controller.phase_index]['deadline'],.4)


if __name__=='__main__':unittest.main()
