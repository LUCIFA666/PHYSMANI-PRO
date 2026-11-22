"""Public future trajectory changes first command, with bounded execution."""
import unittest
import numpy as np
from our_mani.contracts import Forecast
from our_mani.local_mpc_20261102.executor import JointLimits
from our_mani.highspeed_control_20261111_v1.predictive_tracking import PredictiveTrackingPlanner
from tests.test_local_mpc_control_20261103_joint_goal_cem import fixture


class Response:
    coefficients=np.tile([.6,.02,0.],(7,1))
    def predict_joint(self,q,dq,qref,vref,aref):
        d=.6*(qref-q)+.02*dq*.05
        return dict(q=q+d,dq=d/.05)


def setup(reverse=False):
    base,s,c=fixture();s['tcp_angular_velocity']=np.zeros(3)
    times=np.arange(17)*.05
    x=.1*times
    if reverse:x=np.where(times<=.2,.1*times,.02-.1*(times-.2))
    p=np.repeat(s['object_position'][None,None],17,axis=0);p[:,0,0]+=x
    v=np.zeros((17,1,3));v[:,0,0]=.1
    if reverse:v[times>.2,0,0]=-.1
    f=Forecast(('target',),times,p,np.tile(np.eye(3),(17,1,1,1)),v,np.zeros((17,1,3)))
    lim=base.goal_helper.limits;limits=JointLimits(lim.lower,lim.upper,lim.velocity)
    planner=PredictiveTrackingPlanner(base.goal_helper.kinematics,Response(),limits)
    return planner,s,c[0],f


class PredictiveContracts(unittest.TestCase):
    def test_future_reversal_changes_first_reference_without_changing_current_input(self):
        p,s,c,f=setup(False);a=p.plan(s,c,f,pregrasp=True)
        p2,s2,c2,f2=setup(True);b=p2.plan(s2,c2,f2,pregrasp=True)
        self.assertIsNotNone(a['action']);self.assertIsNotNone(b['action'])
        np.testing.assert_array_equal(f.positions[0],f2.positions[0])
        np.testing.assert_array_equal(f.linear_velocities[0],f2.linear_velocities[0])
        self.assertGreater(np.linalg.norm(a['reference']['q']-b['reference']['q']),1e-5)
        self.assertTrue(a['future_public_forecast_used'])
        for result in (a,b):
            self.assertLessEqual(result['maximum_violation'],1e-6)
            self.assertTrue(np.all(np.abs(result['reference']['v'])<=np.asarray(p.limits.effective_velocity)+1e-8))
            self.assertTrue(np.all(np.abs(result['reference']['a'])<=np.asarray(p.limits.acceleration)+1e-8))
            np.testing.assert_allclose(result['reference_sequence'][-1],result['reference_sequence'][-2],atol=1e-8)


if __name__=='__main__':unittest.main()
