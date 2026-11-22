from types import SimpleNamespace
import unittest
import numpy as np
from our_mani.multitask_executor_cycle_20261112_v4.backend import RestoringArmQuery


class ArmFixture:
    def __init__(self, fail):
        self.q=np.zeros(7);self.dq=np.full(7,.03);self.servo=np.full(7,.1)
        self.ik=[0.,0.,0.,0.,0.,0.,1.];self.fail=fail
        self._ik_target=SimpleNamespace(get_pose=lambda:self.ik.copy(),set_pose=lambda x:setattr(self,'ik',list(x)))
        self.dynamic_resets=[]
    def get_joint_positions(self):return self.q.copy()
    def get_joint_velocities(self):return self.dq.copy()
    def get_joint_target_positions(self):return self.servo.copy()
    def set_joint_target_positions(self,q):self.servo=np.array(q)
    def set_joint_positions(self,q,disable_dynamics=False):
        self.q=np.array(q);self.dynamic_resets.append(disable_dynamics)
    def get_path(self,*args,**kwargs):
        self.q[:]=.4;self.servo[:]=.7;self.ik[:3]=[.2,.3,.4]
        if self.fail:raise ValueError('fixture: native IK/path failed after temporary q change')
        return 'original path object'


class AtomicQueryRegression(unittest.TestCase):
    def test_current_native_launcher_consumes_atomic_query_worker(self):
        from pathlib import Path
        from our_mani.multitask_executor_cycle_20261112_v4.common import ROOT
        source=(ROOT/'our_mani/multitask_executor_cycle_20261112_v4/owner.py').read_text()
        self.assertIn('run_multitask_executor_cycle_native_20261112_v4.sh',source)
        launcher=(ROOT/'scripts/run_multitask_executor_cycle_native_20261112_v4.sh').read_text()
        self.assertIn('multitask_executor_cycle_20261112_v4.native_worker',launcher)
        worker=(ROOT/'our_mani/multitask_executor_cycle_20261112_v4/native_worker.py').read_text()
        self.assertIn('from .backend import TaskExecutionBackend',worker)

    def test_failed_native_path_does_not_corrupt_actual_q_or_servo(self):
        arm=ArmFixture(True);query=RestoringArmQuery(arm,lambda:(1.,1.,20))
        with self.assertRaisesRegex(ValueError,'native IK/path failed'):
            query.get_path([0,0,0])
        self.assertTrue(np.array_equal(arm.q,np.zeros(7)))
        self.assertTrue(np.array_equal(arm.dq,np.full(7,.03)))
        self.assertTrue(np.array_equal(arm.servo,np.full(7,.1)))
        self.assertEqual(arm.dynamic_resets,[False])
        self.assertTrue(query.last_query_receipt['temporary_q_changed'])
    def test_success_restores_state_and_returns_original_path(self):
        arm=ArmFixture(False);query=RestoringArmQuery(arm,lambda:(1.,1.,20))
        self.assertEqual(query.get_path([0,0,0]),'original path object')
        self.assertTrue(np.array_equal(arm.q,np.zeros(7)))
        self.assertEqual(query.last_query_receipt['physics_steps'],0)
    def test_unrestorable_velocity_change_is_rejected(self):
        arm=ArmFixture(False)
        original=arm.get_path
        def corrupt(*args,**kwargs):
            r=original(*args,**kwargs);arm.dq[:]=.9;return r
        arm.get_path=corrupt
        query=RestoringArmQuery(arm,lambda:(1.,1.,20))
        with self.assertRaisesRegex(RuntimeError,'could not restore'):
            query.get_path([0,0,0])

    def test_real_overshoot_needs_inward_reference_not_a_fabricated_measurement(self):
        from our_mani.multitask_executor_cycle_20261112_v4.joint_recovery import mechanical_recovery_reference
        actual=np.zeros(7);actual[-1]=2.897313117980957
        lower=np.full(7,-3.);upper=np.full(7,2.8973000049591064)
        reference=np.zeros(7);reference[-1]=upper[-1]
        goal,audit=mechanical_recovery_reference(actual,reference,lower,upper)
        self.assertAlmostEqual(goal[-1],upper[-1]-.002)
        self.assertTrue(np.array_equal(goal[:-1],reference[:-1]))
        self.assertGreater(audit['actual_q'][-1],upper[-1])
        self.assertFalse(audit['measured_q_clipped'])
        self.assertFalse(audit['native_limits_relaxed'])
        self.assertIsNone(mechanical_recovery_reference(np.zeros(7),reference,lower,upper))


if __name__=='__main__':unittest.main()
