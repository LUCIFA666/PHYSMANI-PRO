from copy import deepcopy
from types import SimpleNamespace
import unittest
import numpy as np

from our_mani.local_mpc_p2_20261103.action_adapter import (
    BatchPoseJointAdapter, DLSConfig, matrix_quaternion)
from our_mani.local_mpc_p2_20261103.kinematics import PoEKinematics, SCHEMA
from our_mani.local_mpc_p2_20261103.kinematics import export_native_kinematic_snapshot
from tests.test_local_mpc_p2_20261103_kinematics import GetterOnlyArm,serial


def rz(angle):
    c,s=np.cos(angle),np.sin(angle)
    return np.array([[c,-s,0.],[s,c,0.],[0.,0.,1.]])


def direct(q):
    value=np.eye(4)
    for angle,length in zip(q,(.4,.3,.2)):
        turn=np.eye(4);turn[:3,:3]=rz(angle)
        shift=np.eye(4);shift[0,3]=length
        value=value@turn@shift
    return value


def model():
    origins=np.array([[0.,0.,0.],[.4,0.,0.],[.7,0.,0.]])
    axes=np.tile([0.,0.,1.],(3,1))
    return PoEKinematics(dict(schema=SCHEMA,twist_order='v_omega',q_ref=np.zeros(3),
        space_screws_vw=np.c_[-np.cross(axes,origins),axes],tcp_reference_matrix=direct(np.zeros(3))))


class AdapterTests(unittest.TestCase):
    def test_rotation_quaternion_identity_pi_and_random_roundtrip(self):
        matrices=np.stack([rz(0),rz(np.pi),rz(-2.4),rz(.1)])
        q=matrix_quaternion(matrices)
        expected=np.c_[np.zeros((4,2)),np.sin(np.array([0,np.pi,-2.4,.1])/2),np.cos(np.array([0,np.pi,-2.4,.1])/2)]
        np.testing.assert_allclose(np.abs(np.sum(q*expected,axis=-1)),1,atol=1e-12)
        np.testing.assert_allclose(matrix_quaternion(np.eye(3)),[0,0,0,1],atol=1e-12)

    def test_known_serial_chain_targets_recovered_in_batch_without_mutation(self):
        chain=model();limits=SimpleNamespace(lower=np.full(3,-2.5),upper=np.full(3,2.5))
        adapter=BatchPoseJointAdapter(chain,limits,DLSConfig(iterations=30,position_tolerance=1e-5,angle_tolerance=1e-5))
        desired=np.array([[.4,-.7,.3],[-.3,.6,-.2],[.2,-.4,.5]])
        poses=np.stack([direct(row) for row in desired])
        seeds=desired+np.array([.15,-.1,.1])
        before=seeds.copy()
        result=adapter.solve(seeds,poses[:,:3,3],matrix_quaternion(poses[:,:3,:3]))
        self.assertTrue(result['joint_goal_feasible'].all())
        np.testing.assert_array_equal(seeds,before)
        np.testing.assert_allclose(chain(result['joint_goal']),poses,atol=2e-5)
        self.assertFalse(adapter.metadata['Native_IK_called'])
        self.assertFalse(adapter.metadata['collision_qualified'])

    def test_unreachable_target_and_invalid_seed_retained_as_explicit_infeasible(self):
        adapter=BatchPoseJointAdapter(model(),SimpleNamespace(lower=np.full(3,-2.),upper=np.full(3,2.)))
        result=adapter.solve(np.array([[0.,0.,0.],[3.,0.,0.]]),
            np.array([[5.,0.,0.],[.9,0.,0.]]),np.tile([0.,0.,0.,1.],(2,1)))
        self.assertFalse(result['joint_goal_feasible'].any())
        self.assertTrue(np.isfinite(result['joint_goal']).all())
        self.assertFalse(result['ik_seed_valid'][1])
        self.assertTrue(np.all(np.abs(result['joint_goal'])<=2.))

    def test_spatial_seven_joint_pose_orientation_and_position_recovered(self):
        # GetterOnlyArm is an independent algebraic CPU fixture, not PyRep.
        chain=PoEKinematics(export_native_kinematic_snapshot(GetterOnlyArm()))
        adapter=BatchPoseJointAdapter(chain,SimpleNamespace(lower=np.full(7,-2.),upper=np.full(7,2.)),
            DLSConfig(iterations=35,position_tolerance=1e-5,angle_tolerance=1e-5,damping=.005))
        desired=np.array([[.3,-.4,.2,.1,-.2,.5,-.1],[-.2,.5,.1,-.3,.4,-.2,.3]])
        poses=np.stack([serial(row)[1] for row in desired])
        result=adapter.solve(desired+.04,poses[:,:3,3],matrix_quaternion(poses[:,:3,:3]))
        self.assertTrue(result['joint_goal_feasible'].all())
        np.testing.assert_allclose(chain(result['joint_goal']),poses,atol=2e-5)

    def test_uses_predicted_actual_q_and_does_not_read_reference_or_contact_labels(self):
        adapter=BatchPoseJointAdapter(model(),SimpleNamespace(lower=np.full(3,-2.),upper=np.full(3,2.)))
        q=np.array([[.3,-.5,.2]])
        pose=direct(q[0]);u=dict(tcp_position=pose[None,:3,3],tcp_quaternion=matrix_quaternion(pose[None,:3,:3]))
        a=adapter(dict(q=q),u,dict(q_ref=np.full((1,3),999.)),dict(step=0))
        b=adapter(dict(q=q),u,dict(q_ref=np.zeros((1,3))),dict(step=0))
        np.testing.assert_array_equal(a['joint_goal'],q)
        np.testing.assert_array_equal(a['joint_goal'],b['joint_goal'])
        with self.assertRaises(ValueError):adapter({},u,{}, {})
        with self.assertRaises(ValueError):adapter.solve(q,pose[None,:3,3],np.zeros((1,4)))


if __name__=='__main__':unittest.main()
