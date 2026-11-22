"""Feedforward affects dispatched references; original limits remain binding."""
import unittest
import numpy as np
from our_mani.contracts import Forecast
from our_mani.highspeed_control_20261111_v1.tracking import TrackingReferencePlanner,TrackingConfig
from our_mani.execution_interface_20261111_v1 import PhysManiExecutor,JointTrajectory
from our_mani.local_mpc_20261102.executor import JointLimits
from tests.test_local_mpc_control_20261103_joint_goal_cem import fixture


class Response:
    def predict_joint(self,q,dq,qref,vref,aref):
        after=q+.8*(qref-q)
        return dict(q=after,dq=(after-q)/.05)


def setup(scale=1.):
    base,s,c=fixture()
    s['tcp_angular_velocity']=np.zeros(3)
    s['object_linear_velocity']=np.array([.2,0.,0.])
    f=Forecast(('target',),np.arange(17)*.05,
        s['object_position'][None,None]+np.arange(17)[:,None,None]*.05*s['object_linear_velocity'][None,None],
        np.tile(np.eye(3),(17,1,1,1)),np.tile(s['object_linear_velocity'],(17,1,1)),np.zeros((17,1,3)))
    bounds=base.goal_helper.limits
    limits=JointLimits(bounds.lower,bounds.upper,bounds.velocity)
    p=TrackingReferencePlanner(base.goal_helper.kinematics,Response(),limits,TrackingConfig(feedforward_scale=scale))
    return p,s,c[0],f


class TrackingContracts(unittest.TestCase):
    def test_target_velocity_changes_actual_reference_at_zero_pose_error(self):
        p,s,c,f=setup(1.);r=p.plan(s,c,f,pregrasp=True)
        p0,s0,c0,f0=setup(0.);r0=p0.plan(s0,c0,f0,pregrasp=True)
        self.assertLess(np.linalg.norm(r['position_error']),1e-12)
        self.assertGreater(np.linalg.norm(r['reference']['q']-r0['reference']['q']),1e-4)
        self.assertGreater(r['predicted_linear_velocity'][0],r0['predicted_linear_velocity'][0]+.05)

    def test_reference_is_accepted_unchanged_by_repaired_executor(self):
        p,s,c,f=setup();r=p.plan(s,c,f,pregrasp=True)
        class Backend:
            def __init__(self):self.q=s['q'].copy();self.dq=s['dq'].copy();self.time=0.;self.sent=None
            def get_joint_positions(self):return self.q
            def get_joint_velocities(self):return self.dq
            def clock(self):return self.time
            def terminal(self):return False
            def remaining_ticks(self):return 10
            def set_joint_target_positions(self,q):self.sent=np.array(q)
            def actuate_gripper(self,opening):return False
            def tick(self,phase):self.q=self.sent.copy();self.time+=.05
        b=Backend();ex=PhysManiExecutor(b,p.limits)
        receipt=ex.execute_command(JointTrajectory(0.,[r['reference']['q']],[.05]))
        np.testing.assert_array_equal(b.sent,r['reference']['q'])
        p.observe_execution(receipt)
        np.testing.assert_array_equal(p.last_applied['q'],b.sent)
        self.assertEqual(p.last_applied['time'],.05)

    def test_large_tracking_error_respects_original_reference_derivative_limits(self):
        p,s,c,f=setup();s['tcp_position']=s['tcp_position']+np.array([.5,-.4,.2])
        r=p.plan(s,c,f,pregrasp=False);ref=r['reference']
        self.assertTrue(np.all(np.abs(ref['v'])<=np.asarray(p.limits.effective_velocity)+1e-8))
        self.assertTrue(np.all(np.abs(ref['a'])<=np.asarray(p.limits.acceleration)+1e-8))
        self.assertTrue(np.all(np.abs(ref['a'])/.05<=np.asarray(p.limits.jerk)+1e-8))
        self.assertLessEqual(np.linalg.norm(r['desired_twist'][:3]),.2+1e-9)

    def test_virtual_reference_cannot_seed_a_different_actual_clock(self):
        p,s,c,f=setup();p.last_applied=dict(time=.05,q=s['q'],v=s['dq'],a=np.zeros(7))
        with self.assertRaisesRegex(ValueError,'actual previous applied'):p.plan(s,c,f,pregrasp=True)


if __name__=='__main__':unittest.main()
