from types import SimpleNamespace as NS
from dataclasses import replace
import unittest
import numpy as np
import test_cup_reference_reserve as actual
from our_mani.intercept_planner import JointTrajectory
from our_mani.multitask.block_receding_bridge_v488 import SmoothBlockEnvironment,public_decision_scope,MARGIN

class Tests(unittest.TestCase):
 def fixture(self):
  actor,model,robot=actual.Tests().fixture();b=SmoothBlockEnvironment.__new__(SmoothBlockEnvironment)
  model.collision_query=None
  b.env=NS(robot_model=model,ledger=NS(dt=.05),time=robot.timestamp)
  b.reference_q=robot.joints.copy();b.reference_q[0]+=.009;b.reference_v=np.zeros(7)
  b.public_robot=robot;b.bridge_events=[];b.backup=None
  return b,model,robot
 def evaluator(self,b,model,robot,end):
  class Workflow:
   def __init__(self,*args,**kw):
    self.selector=NS(reachability=None);self.collision_scene=args[2];self.dynamics=args[0]
    self.plan=JointTrajectory(robot.timestamp,robot.timestamp+1.,b.reference_q,b.reference_v,end,np.zeros(7))
    self.step=lambda *a,**kw:None
  return NS(CandidateInterception=Workflow),Workflow
 def test_actual_global_scope_repairs_setpoint_and_preserves_absolute_arrival(self):
  b,model,robot=self.fixture();e,original=self.evaluator(b,model,robot,b.reference_q)
  with public_decision_scope(e,b):
   w=e.CandidateInterception(NS(),model,NS(origin=robot.timestamp),None,{})
   result=w.joint_command(robot)
   self.assertEqual(result['command'],'joint_trajectory',result)
   self.assertLess(result['reference_joints'][0],b.reference_q[0])
   self.assertEqual(w.plan.end,robot.timestamp+1.)
   self.assertTrue(b.bridge_events[-1]['admitted'])
  self.assertIs(e.CandidateInterception,original)
 def test_global_planning_margin_rejects_near_limit_endpoint_without_changing_native_bounds(self):
  b,model,robot=self.fixture();model.upper=model.upper.copy();model.upper[0]=robot.joints[0]+.02
  upper=model.upper.copy();end=robot.joints.copy();end[0]=model.upper[0]-.001
  e,_=self.evaluator(b,model,robot,end)
  with public_decision_scope(e,b):
   w=e.CandidateInterception(NS(),model,NS(origin=robot.timestamp),None,{})
   result=w.joint_command(robot)
   self.assertEqual(result['command'],'hold')
   self.assertEqual(result['reason'],'global_interior_reference_reserve_rejected')
  np.testing.assert_array_equal(model.upper,upper)
  a,r=b._gates();np.testing.assert_allclose(a.model.upper,upper);np.testing.assert_allclose(r.model.upper,upper-MARGIN)
 def test_local_corrected_sample_has_complete_two_branch_offset_brake(self):
  b,model,robot=self.fixture();plan=JointTrajectory(robot.timestamp,robot.timestamp+1.,b.reference_q,b.reference_v,b.reference_q,np.zeros(7))
  bridge,proof=b.qualify(robot,plan,'local_reference');self.assertIsNotNone(bridge,proof)
  q,v=bridge.at(robot.timestamp+.05);stop=b.stopped_backup(robot,q,v)
  self.assertIsNotNone(stop)
  self.assertLessEqual(stop.end,10.);np.testing.assert_allclose(stop.v1,0.)
  model.dynamic_block=True
  self.assertIsNone(b.stopped_backup(robot,q,v))
 def test_global_current_time_and_suffix_collision_offsets_are_rebased_once(self):
  b,model,robot=self.fixture();seen=[]
  def query(offset):seen.append(offset);return False
  model.collision_query=query
  original_collision=model.collision
  model.collision=lambda q,offset:original_collision(q,offset) or bool(model.collision_query(offset))
  e,_=self.evaluator(b,model,robot,b.reference_q)
  with public_decision_scope(e,b):
   w=e.CandidateInterception(NS(),model,NS(origin=robot.timestamp-.1),None,{})
   result=w.joint_command(robot);self.assertEqual(result['command'],'joint_trajectory')
  self.assertIs(model.collision_query,query)
  self.assertAlmostEqual(min(seen),.1,places=8)
  self.assertLessEqual(max(seen),1.1+1e-8)
if __name__=='__main__':unittest.main()
