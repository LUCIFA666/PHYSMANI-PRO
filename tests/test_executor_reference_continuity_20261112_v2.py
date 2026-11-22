import unittest
import numpy as np
from tests.test_executor_pose_contract_repair_20261112_v1 import MotorFixture
from our_mani.execution_interface_20261111_v1 import PhysManiExecutor
from our_mani.local_mpc_20261102.executor import JointLimits
from our_mani.executor_contract_repair_20261112_v2.pose_executor import WorldPoseGoal,PoseGoalExecutor
from our_mani.executor_contract_repair_20261112_v2.reference import ReferenceState,step,preview


def make():
    b=MotorFixture();b.solve_ik=lambda p,q:np.full(7,p[0])
    core=PhysManiExecutor(b,JointLimits([-3.]*7,[3.]*7,[2.]*7,acceleration=12.,jerk=1000.))
    return b,PoseGoalExecutor(b,core)

def action(b,position,opening=0.):
    return WorldPoseGoal(b.clock(),[position,0,1.],[0,0,0,1.],opening)

class ReferenceContinuityTests(unittest.TestCase):
    def test_retarget_with_large_tracking_error_cannot_jump_command(self):
        b,e=make();first=e.execute(action(b,.8));last=first['execution'][-1]['ticks'][0]
        b.q[:]=-1.5;b.dq[:]=0.
        second=e.execute(action(b,-.8))
        p=np.array(second['execution'][0]['ticks'][0]['reference_positions'])
        previous=np.array(last['reference_positions'])
        self.assertLessEqual(np.max(abs(p-previous)),.1+1e-9)
        self.assertGreater(np.max(abs(p-b.q)),.5)
        self.assertFalse(second['reference_reseeded_on_target_change'])

    def test_repeated_abrupt_goals_obey_actual_discrete_speed_acceleration_jerk(self):
        b,e=make();commands=[]
        for goal in (.8,-.8,1.2,-1.2,.3,-.3,0.):
            r=e.execute(action(b,goal))
            self.assertEqual(r['executed_ticks'],10)
            commands.extend(x['ticks'][0]['reference_positions']for x in r['execution'])
        p=np.array(commands);v=np.diff(np.vstack((np.zeros(7),p)),axis=0)/.05
        a=np.diff(np.vstack((np.zeros(7),v)),axis=0)/.05
        j=np.diff(np.vstack((np.zeros(7),a)),axis=0)/.05
        self.assertLessEqual(abs(v).max(),2.+1e-8)
        self.assertLessEqual(abs(a).max(),12.+1e-8)
        self.assertLessEqual(abs(j).max(),1000.+1e-8)

    def test_preflight_and_rejected_IK_do_not_modify_reference_or_gripper(self):
        b,e=make();e.execute(action(b,.5));before=e.executor._last_direct
        b.solve_ik=lambda p,q:None
        r=e.execute(action(b,-.5,1.))
        self.assertEqual(r['status'],'IK_failed');self.assertEqual(e.executor._last_direct,before)
        self.assertEqual(b.opening,0.)
        self.assertEqual(e.advance_retained_goal()['executed_ticks'],10)
        self.assertEqual(b.opening,0.)

    def test_gripper_is_scheduled_at_tenth_tick_and_native_success_truncates(self):
        b,e=make();e.execute(action(b,.5))
        self.assertEqual(next(t for t,o in b.commands if o==0.),9)
        b.success_at=14;r=e.execute(action(b,-.5,1.))
        self.assertEqual(r['executed_ticks'],4);self.assertFalse(r['gripper_intent_issued'])
        self.assertEqual(b.opening,0.)

    def test_reference_preview_is_pure_and_arrives_without_measured_state_resets(self):
        _,e=make();s=ReferenceState(np.zeros(7),np.zeros(7),np.zeros(7));goal=np.full(7,.7)
        self.assertTrue(preview(s,goal,e.executor.limits)['feasible'])
        self.assertTrue(np.array_equal(s.position,np.zeros(7)))
        for _ in range(200):s=step(s,goal,e.executor.limits)
        np.testing.assert_allclose(s.position,goal,atol=1e-8)

if __name__=='__main__':unittest.main()
