from dataclasses import replace
import unittest
import numpy as np
import test_cup_reference_reserve as actual
from our_mani.multitask.next_sample_actual_v455 import check_first,check_sample,install

class Tests(unittest.TestCase):
 def test_long_actual_endpoint_proof_does_not_prove_the_issued_first_sample(self):
  actor,model,robot=actual.Tests().fixture();m=actor.continuous_reference_manager
  q=robot.joints.copy();q[0]+=.04;v=np.zeros(7)
  self.assertTrue(m.native_gate.check_joints(robot,q,v,1.).feasible)
  self.assertFalse(check_first(m.native_gate,robot,q,v,q,v,1.,.05).feasible)
  self.assertTrue(check_first(m.native_gate,robot,robot.joints,v,q,v,1.,.05).feasible)
 def test_actual_current_collision_cannot_be_bypassed(self):
  actor,model,robot=actual.Tests().fixture();m=actor.continuous_reference_manager
  model.dynamic_block=True
  proof=check_sample(m.native_gate,robot,robot.joints,np.zeros(7),.05)
  self.assertFalse(proof.feasible)
 def test_install_keeps_actual_cup_stream_and_blocks_displaced_stationary_reference(self):
  actor,model,robot=actual.Tests().fixture();c,m=actor.controller,actor.continuous_reference_manager
  saved=(c.goal,m.plan,m.plan_kind);c.goal=m.plan=None
  events=install(actor);c.goal,m.plan,m.plan_kind=saved
  result=m.check_stream(robot,.05)
  self.assertTrue(result.feasible,result)
  self.assertIn('exact_next_actual_state_proof',m.pending)
  actual_robot=replace(robot,joints=robot.joints-np.array([.04,0,0,0,0,0,0]))
  m.current_state=replace(m.current_state,robot=actual_robot)
  result=m.check_stream(actual_robot,.05)
  self.assertFalse(result.feasible)
  self.assertIsNone(m.pending)
  np.testing.assert_allclose(model.acceleration_limits,6.)
  self.assertTrue(events)
if __name__=='__main__':unittest.main()
