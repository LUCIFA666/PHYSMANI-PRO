"""A public-vision recovery changes robot posture without an object state."""
import unittest
import numpy as np
from our_mani.execution_interface_20261111_v1.executor import JointLimits
from our_mani.highspeed_control_20261111_v1.tracking_v4 import StoppableTrackingReferencePlanner
from our_mani.highspeed_control_20261111_v1.reacquisition import reacquisition_reference


class ReacquisitionContracts(unittest.TestCase):
    def test_robot_only_recovery_respects_original_reference_limits(self):
        def fk(q):
            T=np.eye(4);T[:3,3]=np.asarray(q)[:3]+[0,0,.8];return T
        limits=JointLimits([-3.]*7,[3.]*7,[3.]*7)
        planner=StoppableTrackingReferencePlanner(fk,None,limits)
        q=np.zeros(7);v=q.copy();v[0]=-1.2
        state=dict(time=0.,q=q,dq=q,actuator_state=dict(reference_positions=q,reference_velocities=v,reference_accelerations=q,reference_clock=0.))
        action=reacquisition_reference(planner,state,np.ones(7))
        nv=(action['joint_goal']-q)/.05
        self.assertTrue(np.all(np.abs(nv-v)<=12*.05+1e-9))
        self.assertTrue(np.all(np.abs(nv)<=3.+1e-9))
        self.assertLess(nv[0],0.)  # A destination change cannot reverse instantly.
        self.assertFalse(action['Native_object_state_used'])
        self.assertEqual(action['reacquisition_target'],'measured_initial_robot_posture')


if __name__=='__main__':unittest.main()
