from dataclasses import replace
import unittest
import numpy as np
import test_cup_reference_reserve as actual
from our_mani.multitask.next_position_actual_v473 import check_sample,check_first,install
from our_mani.multitask.next_sample_actual_v455 import check_sample as old_check
class Tests(unittest.TestCase):
 def test_position_command_need_not_realize_nominal_endpoint_velocity(self):
  actor,model,robot=actual.Tests().fixture();gate=actor.continuous_reference_manager.native_gate
  q=robot.joints.copy();q[0]+=.0035;nominal=np.zeros(7)
  self.assertFalse(old_check(gate,robot,q,nominal,.05).feasible)
  proof=check_sample(gate,robot,q,nominal,.05)
  self.assertTrue(proof.feasible,proof)
  self.assertAlmostEqual(proof.joint_velocities[0],.14)
  self.assertEqual(nominal[0],0.)
 def test_large_held_target_error_collision_and_velocity_limit_remain_rejected(self):
  actor,model,robot=actual.Tests().fixture();gate=actor.continuous_reference_manager.native_gate
  q=robot.joints.copy();q[0]+=.04
  self.assertFalse(check_sample(gate,robot,q,np.zeros(7),.05).feasible)
  q=robot.joints.copy();model.dynamic_block=True
  self.assertFalse(check_sample(gate,robot,q,np.zeros(7),.05).feasible)
  model.dynamic_block=False
  fast=replace(robot,joint_velocities=np.array([3.,0,0,0,0,0,0]))
  self.assertFalse(check_sample(gate,fast,robot.joints+fast.joint_velocities*.05,np.zeros(7),.05).feasible)
 def test_same_issued_first_position_not_a_different_long_horizon_path(self):
  actor,model,robot=actual.Tests().fixture();gate=actor.continuous_reference_manager.native_gate
  q=robot.joints.copy();q[0]+=.04;v=np.zeros(7)
  self.assertTrue(gate.check_joints(robot,q,v,1.).feasible)
  self.assertFalse(check_first(gate,robot,q,v,q,v,1.,.05).feasible)
  self.assertTrue(check_first(gate,robot,robot.joints,v,q,v,1.,.05).feasible)
 def test_actual_cup_stream_preserves_separate_nominal_and_actual_velocity_proofs(self):
  actor,model,robot=actual.Tests().fixture();c,m=actor.controller,actor.continuous_reference_manager
  saved=(c.goal,m.plan,m.plan_kind);c.goal=m.plan=None;install(actor);c.goal,m.plan,m.plan_kind=saved
  result=m.check_stream(robot,.05);self.assertTrue(result.feasible,result)
  proof=m.pending['exact_next_actual_state_proof']
  self.assertIn('actual_proof_terminal_velocity',proof)
  self.assertEqual(proof['separate_nominal_reference_terminal_velocity'],m.pending['reference_to_velocity'])
  np.testing.assert_allclose(model.acceleration_limits,6.)
if __name__=='__main__':unittest.main()
