import unittest
from types import SimpleNamespace
import numpy as np
from our_mani.executor_differential_tracking_20261112_v1.pose_executor import PoseGoalExecutor
from our_mani.executor_contract_repair_20261112_v2.reference import ReferenceState

class DifferentialAdmissionTests(unittest.TestCase):
    def make(self):
        dispatch=object.__new__(PoseGoalExecutor);state=ReferenceState(np.zeros(7),np.zeros(7),np.zeros(7))
        housing=dict(name='Panda_gripper',reference_matrix=np.eye(4),vertices=np.array([[-.005,-.005,-.005],[.005,.005,.005]]))
        snapshot=dict(joint_origins_world=[[0,0,z]for z in (0,.1,.2,.3,.4,.5,.6)],tcp_reference_matrix=np.eye(4).tolist())
        dispatch.clearance=SimpleNamespace(snapshot=snapshot,bodies=[housing],fk=SimpleNamespace(reference_tcp=np.eye(4)),
            body_margins=[.02],table_matrix=np.eye(4),bounds=[-1,1,-1,1,0,.75],safe=lambda q:np.array([True]))
        dispatch.backend=SimpleNamespace(get_joint_positions=lambda:np.zeros(7))
        dispatch._state=lambda:state;dispatch._barriers=lambda s,q:(np.zeros((0,7)),np.zeros(0),[])
        dispatch._next=lambda *args,**kwargs:(state,dict())
        return dispatch

    def test_safe_differential_prefix_does_not_require_endpoint_IK(self):
        allowed,receipt=self.make()._admit([.2,.1,.9],[0,0,0,1])
        self.assertTrue(allowed);self.assertFalse(receipt['Native_IK_called'])
        self.assertFalse(receipt['global_pose_reachability_claimed'])

    def test_tool_pose_that_crosses_table_is_still_rejected(self):
        allowed,receipt=self.make()._admit([.2,.1,.75],[0,0,0,1])
        self.assertFalse(allowed);self.assertEqual(receipt['reason'],'hardware_table_intersection')

if __name__=='__main__':unittest.main()
