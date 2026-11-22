import unittest
import numpy as np
from our_mani.executor_contract_repair_20261112_v1.pose_executor import WorldPoseGoal,PoseGoalExecutor
from our_mani.execution_interface_20261111_v1 import PhysManiExecutor
from our_mani.local_mpc_20261102.executor import JointLimits
from our_mani.executor_contract_repair_20261112_v1.controller import execute_reference
from types import SimpleNamespace

class MotorFixture:
    def __init__(self):
        self.q=np.zeros(7);self.dq=np.zeros(7);self.target=self.q.copy();self.time=0.;self.ticks=0
        self.opening=1.;self.commands=[];self.ik_calls=0;self.success_at=None
    def get_joint_positions(self):return self.q.copy()
    def get_joint_velocities(self):return self.dq.copy()
    def set_joint_target_positions(self,q):self.target=np.array(q)
    def clock(self):return self.time
    def remaining_ticks(self):return 1200-self.ticks
    def terminal(self):return self.success_at is not None and self.ticks>=self.success_at
    def actuate_gripper(self,op):self.commands.append((self.ticks,op));self.opening=op;return False
    def solve_ik(self,p,q):self.ik_calls+=1;return np.full(7,.5)
    def tick(self,phase):
        moved=self.q+.15*(self.target-self.q);self.dq=(moved-self.q)/.05;self.q=moved
        self.ticks+=1;self.time+=.05

def make():
    b=MotorFixture();e=PhysManiExecutor(b,JointLimits([-3.]*7,[3.]*7,[2.]*7,acceleration=12.,jerk=1000.))
    return b,PoseGoalExecutor(b,e)

class PoseContract(unittest.TestCase):
    def action(self,at=0.,op=0.):return WorldPoseGoal(at,[.2,0,1.],[0,0,0,1.],op)
    def test_opening_at_tick10_without_full_path_or_arrival_wait(self):
        b,e=make();r=e.execute(self.action())
        self.assertEqual(r['executed_ticks'],10)
        self.assertEqual(next(t for t,op in b.commands if op==0.),9)
        self.assertGreater(np.max(np.abs(b.q-.5)),.01)
        self.assertFalse(r['measured_arrival_claimed'])
        self.assertFalse(r['whole_RML_path_waited'])
        self.assertFalse(r['gripper_done'])
    def test_same_pose_preserves_goal_and_pending_gripper_across_chunks(self):
        b,e=make();e.execute(self.action());r=e.execute(self.action(b.clock()))
        self.assertEqual(b.ik_calls,1);self.assertTrue(r['IK_reused'])
        self.assertEqual(b.commands[10][1],0.)
    def test_terminal_before_schedule_prevents_gripper_switch(self):
        b,e=make();b.success_at=4;r=e.execute(self.action())
        self.assertEqual(r['executed_ticks'],4)
        self.assertFalse(r['gripper_intent_issued'])
        self.assertTrue(all(op==1. for _,op in b.commands))
    def test_budget_before_schedule_preserves_existing_gripper_request(self):
        b,e=make();b.remaining_ticks=lambda:6-b.ticks
        r=e.execute(self.action())
        self.assertEqual(r['executed_ticks'],6)
        self.assertFalse(r['gripper_intent_issued'])
        self.assertTrue(all(op==1. for _,op in b.commands))
    def test_existing_policy_reference_reaches_same_typed_executor(self):
        b,e=make()
        reference=SimpleNamespace(timestamp=0.,position=[.2,0,1.],quaternion=[0,0,0,1.],opening_intent=.1)
        r=execute_reference(e,reference)
        self.assertEqual(r['executed_ticks'],10)
        self.assertEqual(next(t for t,op in b.commands if op==0.),9)
    def test_stale_pose_rejected_before_IK_or_any_write(self):
        b,e=make();b.time=.1
        with self.assertRaises(ValueError):e.execute(self.action())
        self.assertFalse(b.commands);self.assertEqual(b.ik_calls,0)
    def test_IK_failure_has_no_arm_or_gripper_dispatch(self):
        b,e=make();b.solve_ik=lambda p,q:None
        r=e.execute(self.action())
        self.assertEqual(r['status'],'IK_failed');self.assertEqual(b.ticks,0)
        self.assertFalse(b.commands)
    def test_rejected_pose_can_retain_accepted_goal_and_gripper_then_retry(self):
        b,e=make();e.execute(self.action())
        accepted=e._cached_joint_goal.copy()
        b.solve_ik=lambda p,q:None
        rejected=WorldPoseGoal(b.clock(),[.3,0,1.],[0,0,0,1.],1.)
        self.assertEqual(e.execute(rejected)['status'],'IK_failed')
        r=e.advance_retained_goal()
        self.assertEqual(r['executed_ticks'],10)
        self.assertEqual(r['retained_goal_source'],'previously_accepted_joint_goal')
        self.assertTrue(np.array_equal(r['joint_goal'],accepted))
        self.assertTrue(all(op==0. for _,op in b.commands[10:]))
        b.solve_ik=lambda p,q:np.full(7,.6)
        r=e.execute(WorldPoseGoal(b.clock(),[.3,0,1.],[0,0,0,1.],1.))
        self.assertEqual(r['status'],'chunk_complete');self.assertEqual(b.opening,1.)
    def test_no_prior_goal_has_explicit_measured_initial_hold(self):
        b,e=make();r=e.advance_retained_goal()
        self.assertEqual(r['retained_goal_source'],'measured_initial_joint_hold')
        self.assertEqual(r['executed_ticks'],10)
        self.assertTrue(np.array_equal(b.q,np.zeros(7)))
        self.assertTrue(all(op==1. for _,op in b.commands))
    def test_infeasible_retarget_preserves_accepted_reference_and_can_continue(self):
        b,e=make();e.execute(self.action())
        old_goal=e._cached_joint_goal.copy();old_state=e.executor.state();writes=len(b.commands)
        b.q[-1]=2.999;b.dq[-1]=2.
        b.solve_ik=lambda p,q:np.full(7,.6)
        r=e.execute(WorldPoseGoal(b.clock(),[.3,0,1.],[0,0,0,1.],1.))
        self.assertEqual(r['status'],'reference_rejected')
        self.assertEqual(r['failure_code'],5);self.assertEqual(len(b.commands),writes)
        self.assertTrue(np.array_equal(e._cached_joint_goal,old_goal))
        self.assertEqual(e.executor.state(),old_state)
        retained=e.advance_retained_goal();self.assertEqual(retained['executed_ticks'],10)
        self.assertEqual(b.opening,0.)
    def test_state_changing_IK_faults_before_any_physical_dispatch(self):
        b,e=make()
        def corrupt(p,q):b.q[0]+=.01;return np.full(7,.5)
        b.solve_ik=corrupt
        with self.assertRaisesRegex(RuntimeError,'IK query changed'):e.execute(self.action())
        self.assertEqual(b.ticks,0);self.assertFalse(b.commands)
        with self.assertRaisesRegex(RuntimeError,'faulted'):e.execute(self.action())
    def test_frame_quaternion_and_opening_cannot_be_silently_reinterpreted(self):
        with self.assertRaises(ValueError):WorldPoseGoal(0,[0,0,1],[0,0,0,1],0,frame='object')
        with self.assertRaises(ValueError):WorldPoseGoal(0,[0,0,1],[0,0,0,2],0)
        with self.assertRaises(ValueError):self.action(op=.3)

if __name__=='__main__':unittest.main()
