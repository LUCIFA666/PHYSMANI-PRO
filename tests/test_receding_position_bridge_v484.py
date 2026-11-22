from dataclasses import replace
import unittest
import numpy as np
import test_cup_reference_reserve as actual
from our_mani.intercept_planner import JointTrajectory
from our_mani.multitask.next_position_actual_v473 import check_sample
from our_mani.multitask.receding_position_bridge_v484 import repair_first,install,PiecewiseReference,capped_gate

class Tests(unittest.TestCase):
 def fixture(self):
  actor,model,robot=actual.Tests().fixture();m=actor.continuous_reference_manager
  q=robot.joints.copy();q[0]+=.009;v=np.zeros(7)
  plan=JointTrajectory(robot.timestamp,robot.timestamp+1.,q,v,q,v)
  return actor,model,robot,q,v,plan
 def test_projection_keeps_deadline_and_both_complete_path_branches(self):
  actor,model,robot,q,v,plan=self.fixture();m=actor.continuous_reference_manager
  self.assertFalse(check_sample(m.native_gate,robot,plan.at(robot.timestamp+.05)[0],v,.05).feasible)
  bridge,proof=repair_first(m.native_gate,m.reference_reserve_gate,robot,q,v,plan,.05)
  self.assertIsInstance(bridge,PiecewiseReference)
  self.assertEqual(bridge.end,plan.end)
  np.testing.assert_allclose(bridge.q1,plan.q1);np.testing.assert_allclose(bridge.v1,plan.v1)
  first_q,first_v=bridge.first.at(bridge.first.end);tail_q,tail_v=bridge.tail.at(bridge.tail.start)
  np.testing.assert_allclose(first_q,tail_q);np.testing.assert_allclose(first_v,tail_v)
  self.assertEqual(proof['trials'][-1]['actual_tail'],'feasible_sampled_trajectory')
  self.assertEqual(proof['trials'][-1]['reference_tail'],'feasible_sampled_trajectory')
  self.assertNotEqual(proof['actual_first_terminal_velocity'],proof['reference_first_terminal_velocity'])
  np.testing.assert_allclose(model.acceleration_limits,6.)
 def test_collision_empty_intersection_and_original_deadline_still_reject(self):
  actor,model,robot,q,v,plan=self.fixture();m=actor.continuous_reference_manager
  model.dynamic_block=True
  self.assertIsNone(repair_first(m.native_gate,m.reference_reserve_gate,robot,q,v,plan,.05)[0])
  model.dynamic_block=False
  qfar=q.copy();qfar[0]+=.1
  self.assertIsNone(repair_first(m.native_gate,m.reference_reserve_gate,robot,qfar,v,plan,.05)[0])
  short=JointTrajectory(robot.timestamp,robot.timestamp+.05,q,v,q,v)
  self.assertIsNone(repair_first(m.native_gate,m.reference_reserve_gate,robot,q,v,short,.05)[0])
 def test_actual_manager_emits_the_new_certificate_and_checked_offset_stop(self):
  actor,model,robot,q,v,plan=self.fixture();c,m=actor.controller,actor.continuous_reference_manager
  c.goal=m.plan=None;events=install(actor)
  m.last_confirmed['joints']=q.tolist();m.last_confirmed['nominal_velocity']=v.tolist()
  m.plan=plan;m.plan_kind='tracking';m.plan_candidate_id='current_fixed_goal';m.invalidated=False
  result=m.check_stream(robot,.05)
  self.assertTrue(result.feasible,result)
  self.assertTrue(events[-1]['admitted'])
  self.assertIsInstance(m.plan,PiecewiseReference)
  self.assertIn('measured_position_bridge',m.pending)
  self.assertIn('next_reference_stop_guard_v274',m.pending)
  np.testing.assert_allclose(m.plan.at(robot.timestamp+.05)[0],m.pending['reference_to_joints'])
  self.assertEqual(m.plan.end,plan.end)
 def test_existing_acquire_speed_reserve_is_also_kept_for_both_bridge_paths(self):
  actor,model,robot,q,v,plan=self.fixture();m=actor.continuous_reference_manager
  native=capped_gate(m.native_gate,np.full(7,.15));reference=capped_gate(m.reference_reserve_gate,np.full(7,.15))
  bridge,proof=repair_first(native,reference,robot,q,v,plan,.05)
  self.assertIsNotNone(bridge,proof)
  for field in ('actual_first','reference_first','actual_suffix','reference_suffix'):
   self.assertLessEqual(proof[field]['maximum_speed_fraction'],1.+1e-7)
  np.testing.assert_allclose(model.velocity_limits,m.native_gate.model.velocity_limits)
if __name__=='__main__':unittest.main()
