import unittest
from types import SimpleNamespace
import numpy as np
from our_mani.executor_contract_repair_20261112_v3.clearance import HardwareTableClearance,tabletop_frame
from our_mani.executor_contract_repair_20261112_v3.pose_executor import PoseGoalExecutor


def geometry():
    model=HardwareTableClearance.__new__(HardwareTableClearance)
    model.table_inverse=np.eye(4);model.bounds=np.array([-1,1,-1,1,0,.75]);model.margin=.02
    screws=np.zeros((7,6));screws[:,5]=1;screws[0,5]=0;screws[0,3]=1
    model.fk=SimpleNamespace(screws=screws,q_ref=np.zeros(7))
    vertices=np.array([[x,y,z]for x in (-.05,.05)for y in (-.05,.05)for z in (-.05,.05)])
    matrix=np.eye(4);matrix[2,3]=1.
    model.bodies=[dict(name='test_housing',depth=7,vertices=vertices,reference_matrix=matrix)]
    return model


class ClearanceTests(unittest.TestCase):
    def test_rotated_native_table_uses_world_up_and_correct_height(self):
        native=np.eye(4);native[:3,:3]=[[0,0,-1],[0,1,0],[1,0,0]];native[2,3]=.375
        frame,bounds=tabletop_frame(native,[-.375,.375,-.55,.55,-.8,.8])
        np.testing.assert_allclose(frame[:3,2],[0,0,1])
        self.assertAlmostEqual(frame[2,3]+bounds[5],.75)
        np.testing.assert_allclose(sorted([bounds[1]-bounds[0],bounds[3]-bounds[2]]),[1.1,1.6])
    def test_batch_distinguishes_free_housing_and_table_intersection(self):
        g=geometry();q=np.zeros((2,7));q[1,0]=np.pi/2
        self.assertEqual(g.safe(q).tolist(),[True,False])
        np.testing.assert_array_equal(q[0],np.zeros(7))

    def test_clear_endpoints_do_not_hide_intermediate_collision(self):
        g=geometry();g.bodies[0]['reference_matrix'][2,3]=-1.
        q=np.zeros((3,7));q[:,0]=[-2.7,0,2.7]
        self.assertEqual(g.safe(q).tolist(),[True,False,True])

    def test_table_frame_translation_is_respected(self):
        g=geometry();translation=np.eye(4);translation[:3,3]=[2,3,.1]
        g.table_inverse=np.linalg.inv(translation)
        g.bodies[0]['reference_matrix']=translation@g.bodies[0]['reference_matrix']
        self.assertTrue(g.safe(np.zeros(7))[0])
        g.bodies[0]['reference_matrix'][2,3]-=.3
        self.assertFalse(g.safe(np.zeros(7))[0])

    def test_alternative_query_restores_robot_without_physics(self):
        state=SimpleNamespace(q=np.zeros(7),dq=np.ones(7)*.1,servo=np.ones(7)*.2,ik=[0,0,1,0,0,0,1])
        def sampling(*a,**kw):state.q=np.ones(7);return np.full((1,7),.3)
        arm=SimpleNamespace(get_joint_positions=lambda:state.q.copy(),solve_ik_via_sampling=sampling,
            get_joint_target_positions=lambda:state.servo.copy(),
            set_joint_positions=lambda q,disable_dynamics:setattr(state,'q',np.array(q)),
            set_joint_target_positions=lambda q:setattr(state,'servo',np.array(q)),
            _ik_target=SimpleNamespace(get_pose=lambda:state.ik.copy(),set_pose=lambda p:setattr(state,'ik',list(p))))
        backend=SimpleNamespace(_clock_tuple=lambda:(0.,0.,0),get_joint_positions=lambda:state.q.copy(),
            get_joint_velocities=lambda:state.dq.copy(),arm=arm,_ik_errors=(ValueError,))
        executor=PoseGoalExecutor.__new__(PoseGoalExecutor);executor.backend=backend
        action=SimpleNamespace(position=(0,0,1),quaternion_xyzw=(0,0,0,1))
        result=executor._sample(action)
        np.testing.assert_array_equal(result,np.full((1,7),.3))
        np.testing.assert_array_equal(state.q,np.zeros(7));np.testing.assert_array_equal(state.dq,np.ones(7)*.1)

    def test_invalid_native_IK_candidate_is_rejected_before_geometry_or_motion(self):
        from tests.test_executor_reference_continuity_20261112_v2 import make
        _,continuous=make();executor=PoseGoalExecutor.__new__(PoseGoalExecutor)
        executor.executor=continuous.executor
        for goal in (np.full(7,100.),np.full(7,np.nan),np.zeros(6)):
            ok,receipt=executor._qualify(goal,None)
            self.assertFalse(ok);self.assertEqual(receipt['reason'],'IK_solution_outside_native_contract')

if __name__=='__main__':unittest.main()
