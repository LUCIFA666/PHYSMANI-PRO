"""Pure CPU containment and cavity-preservation checks for uncertain CAD."""
import unittest
import numpy as np
from scipy.spatial.transform import Rotation

from our_mani.multitask.collision_geometry import collision_geometry_for_orientation,mesh_sha256


def ring():
    vertices=np.array([[-2,-2,0],[2,-2,0],[2,2,0],[-2,2,0],
                       [-1,-1,0],[1,-1,0],[1,1,0],[-1,1,0]],float)
    triangles=np.array([[0,1,5],[0,5,4],[1,2,6],[1,6,5],
                        [2,3,7],[2,7,6],[3,0,4],[3,4,7]],np.int32)
    return vertices,triangles


class CollisionGeometryChecks(unittest.TestCase):
    def test_full_retains_concave_original_triangle_surface_exactly(self):
        vertices,triangles=ring();before=vertices.copy(),triangles.copy()
        result=collision_geometry_for_orientation(vertices,triangles,orientation_observability='full')
        np.testing.assert_array_equal(result.vertices,vertices);np.testing.assert_array_equal(result.triangles,triangles)
        self.assertEqual(result.receipt['collision_geometry_kind'],'exact_TRAIN_triangles')
        self.assertTrue(result.receipt['cavities_preserved'])
        self.assertEqual(result.receipt['original_mesh_sha256'],result.receipt['registered_mesh_sha256'])
        np.testing.assert_array_equal(vertices,before[0]);np.testing.assert_array_equal(triangles,before[1])

    def test_cube_contains_off_origin_vertices_and_triangle_points_under_random_rotations(self):
        vertices,triangles=ring();vertices+=np.array([.37,-.23,.19])
        rotations=Rotation.random(300,random_state=1042).as_matrix()
        points=np.r_[vertices,vertices[triangles].mean(1)]
        for mode in ('none','axis'):
            result=collision_geometry_for_orientation(vertices,triangles,orientation_observability=mode)
            r=result.receipt['registered_cube_half_extent_m']
            self.assertGreaterEqual(r,float(np.linalg.norm(vertices,axis=1).max()))
            moved=np.einsum('tij,nj->tni',rotations,points)
            self.assertTrue((np.abs(moved)<=r).all())
            self.assertTrue((np.abs(moved)<=float(np.float32(r))).all())
            self.assertEqual(result.vertices.dtype,np.float32);self.assertEqual(len(result.triangles),12)
            self.assertFalse(result.receipt['cavities_preserved'])
            self.assertTrue(result.receipt['unknown_rotation_identity_is_not_exact_CAD'])
            self.assertFalse(result.receipt['native_pose_or_future_state_used'])

    def test_float32_bound_is_rounded_up_not_inward(self):
        value=float(np.float32(.1))+1e-10
        vertices=np.array([[value,0,0],[0,.01,0],[0,0,.01]])
        result=collision_geometry_for_orientation(vertices,[[0,1,2]],orientation_observability='none')
        self.assertGreater(result.receipt['registered_cube_half_extent_m'],value)
        self.assertEqual(result.receipt['registered_mesh_sha256'],mesh_sha256(result.vertices,result.triangles))

    def test_missing_or_invalid_orientation_and_mesh_fail_closed(self):
        vertices,triangles=ring()
        with self.assertRaises(TypeError):collision_geometry_for_orientation(vertices,triangles)
        for flag in (None,'symmetry','unknown'):
            with self.assertRaises(ValueError):
                collision_geometry_for_orientation(vertices,triangles,orientation_observability=flag)
        with self.assertRaises(ValueError):
            collision_geometry_for_orientation(vertices,[[0,1,99]],orientation_observability='full')
        with self.assertRaises(ValueError):
            collision_geometry_for_orientation(vertices,[[0.,1.,2.]],orientation_observability='none')


if __name__=='__main__':unittest.main()
