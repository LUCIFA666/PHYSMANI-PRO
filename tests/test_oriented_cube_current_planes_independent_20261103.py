"""Independent synthetic current-plane math, no scene/GT/NN/Native access."""
from copy import deepcopy
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from our_mani.local_mpc_control_20261103 import oriented_cube_current_planes_prototype_20261103 as p


def fixture(faces=((0,1),(1,1),(2,1)),span=.014):
    R=Rotation.from_euler('xyz',[22.,-17.,31.],degrees=True).as_matrix();centre=np.array([.15,-.09,.8])
    points=[];normals=[];origins=[];views=[]
    for view,(j,sign)in enumerate(faces):
        other=[k for k in range(3)if k!=j]
        for a in(-span,0.,span):
            for b in(-span,0.,span):
                local=np.zeros(3);local[j]=sign*.025;local[other]=[a,b]
                points.append(centre+R@local);normals.append(R[:,j]*sign)
                origins.append(centre+R[:,j]*sign*.3);views.append(view%4)
    return np.asarray(points),np.asarray(normals),np.asarray(origins),np.asarray(views,int),centre


def result(points,normals,origins,views,transform=None):
    estimate=p.estimate_oriented_support(points,normals,origins,views)
    T=np.eye(4)if transform is None else transform
    # Invalid depths contribute no free-space evidence; supported planes must
    # stand alone, and a missing interval cannot be silently completed.
    refined=p.refine_current_oriented_support(np.zeros((64,64),np.float32),np.array([[-100.,0,32.],[0,-100.,32.],[0,0,1.]]),T,estimate)
    return estimate,refined


class IndependentCurrentPlaneGeometry(unittest.TestCase):
    def test_three_faces_rotated_far_from_world_axes_and_no_qualification(self):
        points,normals,origins,views,centre=fixture();e,r=result(points,normals,origins,views)
        self.assertEqual(e['status'],'UNKNOWN');self.assertIsNone(e['centre']);self.assertEqual(len(e['modes']),1)
        self.assertEqual(r['status'],'AVAILABLE');np.testing.assert_allclose(r['centre'],centre,rtol=0,atol=1e-12)
        self.assertLess(r['joint_radius'],.005);self.assertTrue(r['conditional_geometry_only'])
        for key in('qualified','position_measurement_qualified','model_qualified','raw_mask_model_qualified','force_or_contact_qualified'):
            self.assertIs(r[key],False)
        self.assertIs(r['orientation_observed'],False)

    def test_two_faces_partial_missing_axis_remains_unknown_without_free_depth(self):
        points,normals,origins,views,_=fixture(faces=((0,1),(1,1)),span=.010)
        e,r=result(points,normals,origins,views)
        self.assertEqual(len(e['modes']),1);self.assertEqual(sum(e['modes'][0]['support']['observed_axis']),2)
        self.assertGreater(e['modes'][0]['support']['joint_radius'],.005)
        self.assertEqual(r['status'],'UNKNOWN');self.assertIsNone(r['centre'])
        self.assertTrue(all(step['free_pixel_count']==0 and not step['constraints'] for step in r['modes'][0]['ray_support']['axis_general_trace']['rounds']))

    def test_opposing_faces_normal_signs_and_input_order_leave_centre_same(self):
        points,normals,origins,views,centre=fixture(faces=((0,1),(0,-1),(1,1),(2,-1)))
        _,base=result(points,normals,origins,views);self.assertEqual(base['status'],'AVAILABLE')
        rng=np.random.default_rng(261103);flips=rng.choice([-1.,1.],len(points));order=rng.permutation(len(points))
        e,r=result(points[order],(normals*flips[:,None])[order],origins[order],views[order])
        self.assertEqual(len(e['modes']),1);self.assertEqual(r['status'],'AVAILABLE')
        np.testing.assert_allclose(r['centre'],centre,rtol=0,atol=1e-12)
        np.testing.assert_allclose(r['centre'],base['centre'],rtol=0,atol=1e-12)

    def test_common_world_SE3_transform_equivariance_for_three_observed_faces(self):
        points,normals,origins,views,centre=fixture();Q=Rotation.from_euler('xyz',[-33.,12.,49.],degrees=True).as_matrix();translation=np.array([-.3,.21,.4])
        T=np.eye(4);T[:3,:3]=Q;T[:3,3]=translation
        e,r=result(points@Q.T+translation,normals@Q.T,origins@Q.T+translation,views,T)
        self.assertEqual(len(e['modes']),1);self.assertEqual(r['status'],'AVAILABLE')
        np.testing.assert_allclose(r['centre'],Q@centre+translation,rtol=0,atol=1e-12)
        self.assertTrue(r['full_orientation_feasible_set_proven']is False)

    def test_same_normal_family_does_not_invent_two_missing_faces(self):
        points,normals,origins,views,_=fixture(faces=((0,1),))
        e,r=result(points,normals,origins,views)
        self.assertEqual(len(e['modes']),0);self.assertEqual(r['status'],'UNKNOWN');self.assertIsNone(r['centre'])
        self.assertIn('fewer_than_two_independent_current_normal_families',e['reasons'])

    def test_all_current_rows_must_be_explained_and_outlier_not_discarded(self):
        points,normals,origins,views,_=fixture()
        changed=points.copy();changed[0]+=normals[0]*.002
        e,r=result(changed,normals,origins,views)
        self.assertEqual(r['status'],'UNKNOWN');self.assertIsNone(r['centre'])
        # A fourth non-Cube plane cannot disappear through an old world-axis
        # gate or majority fit. It is a current input that must be explained.
        extra=(normals[0]+normals[9])/np.sqrt(2)
        nn=normals.copy();nn[0]=extra
        e,r=result(points,nn,origins,views)
        self.assertEqual(r['status'],'UNKNOWN');self.assertIsNone(r['centre'])
        for values in((points,np.zeros_like(normals),origins,views),(points,normals,origins,views.astype(float))):
            with self.assertRaises(ValueError):p.estimate_oriented_support(*values)

if __name__=='__main__':unittest.main()
