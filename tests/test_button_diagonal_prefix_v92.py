"""A moving horizontal target must not force an indefinitely level approach."""
from dataclasses import replace
from unittest.mock import patch
import unittest
import numpy as np

import test_button_intent_actor_v74 as original_test
from our_mani.multitask import button_intent_actor_v91
from our_mani.multitask.button_intent_prefix_v90 import BoundedApproach
from our_mani.multitask.task_reference_v86 import reserved_reference_gate
from our_mani.contracts import Pose,ObjectState
from our_mani.dynamics import ConstantVelocityDynamics
import test_task_reference_v37 as reference_test


class FactoryTests(original_test.Tests):
    def actor(self):
        with patch('test_button_intent_actor_v74.binding',button_intent_actor_v91.binding), \
             patch('test_button_intent_actor_v74.create_actor',button_intent_actor_v91.create_actor):
            return super().actor()


class DiagonalTests(unittest.TestCase):
    def setup_scene(self):
        fixture=reference_test.PartialApproachTests();actor,model,r=fixture.setup_scene()
        m=actor.continuous_reference_manager
        m.reference_reserve_gate=reserved_reference_gate(m.native_gate)
        obj=m.current_state.bodies['ball']
        obj=ObjectState(obj.object_id,obj.timestamp,Pose([1.5,0.,.05]),obj.points_world,
            np.array([.1,0.,0.]),obj.angular_velocity)
        m.current_state=replace(m.current_state,bodies={'ball':obj})
        forecast=ConstantVelocityDynamics().predict([obj],1.1,.05)
        forecast.metadata['observed_until']=r.timestamp;m.current_forecast=forecast
        actor.controller.generator.last_diagnostics={'feasible_candidates':0}
        m.approach_transit=BoundedApproach(m)
        return actor,model,r

    def test_certified_prefix_descends_while_horizontal_target_is_still_far(self):
        actor,model,r=self.setup_scene();m=actor.continuous_reference_manager
        self.assertTrue(m.approach_transit.try_approach(r,9.5))
        end=model.forward(m.plan.q1)
        self.assertLess(end.position[2],r.pose.position[2]-.001)
        self.assertLessEqual(np.linalg.norm(end.position-r.pose.position),.08+1e-8)
        self.assertGreater(1.5-end.position[0],.08)
        event=next(e for e in m.approach_transit.events if e['event']=='partial_approach_staged')
        self.assertTrue(event['actual_and_reference_full_gate_feasible'])
        self.assertTrue(event['backup_brake_evidence'])
        self.assertGreater(len(model.collision_calls),0)

    def test_dynamic_obstacle_still_blocks_the_diagonal(self):
        actor,model,r=self.setup_scene();model.dynamic_block=True
        self.assertFalse(actor.continuous_reference_manager.approach_transit.try_approach(r,9.5))
        self.assertIsNone(actor.continuous_reference_manager.pending)

    def test_diagonal_does_not_extend_global_deadline_or_consume_an_unverified_tick(self):
        actor,model,r=self.setup_scene();m=actor.continuous_reference_manager
        self.assertFalse(m.approach_transit.try_approach(r,.2))
        self.assertIsNone(m.plan)
        self.assertEqual(m.confirmations,[])


if __name__=='__main__':unittest.main()
