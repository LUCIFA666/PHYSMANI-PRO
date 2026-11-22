"""Analytic counterexamples and random yaw/triangle containment, CPU only."""
import itertools
import unittest
import numpy as np

from our_mani.multitask.bin_swept_geometry import (
    axis_yaw_swept_prototype,points_inside_prototype,projected_radial_bounds,
    partial_collision_axis_ready,_prism,
    verify_stored_sweep_exact,SweptPrototype,
)


def cup():
    angles=np.arange(24)*2*np.pi/24
    directions=np.c_[np.cos(angles),np.sin(angles)]
    walls,wt=_prism(0.,2.,directions*2,directions)
    floor,ft=_prism(0.,.25,directions*2,None)
    return np.r_[walls,floor],np.r_[wt,ft+len(walls)]


def random_surface_yaw_points(vertices,triangles,n=4000):
    rng=np.random.default_rng(4291);faces=vertices[triangles[rng.integers(0,len(triangles),n)]]
    weights=rng.dirichlet([1,1,1],n);points=(faces*weights[:,:,None]).sum(1)
    angles=rng.uniform(-np.pi,np.pi,n);y,z=points[:,1].copy(),points[:,2].copy()
    points[:,1]=y*np.cos(angles)-z*np.sin(angles);points[:,2]=y*np.sin(angles)+z*np.cos(angles)
    return points


class BinSweptPrototypeChecks(unittest.TestCase):
    def test_projected_polygon_origin_distance_is_not_vertex_minimum(self):
        low,high=projected_radial_bounds([[-2,-2],[2,-2],[0,2]],margin=0.)
        self.assertEqual(low,0.);self.assertGreater(high,2.)
        low,_=projected_radial_bounds([[2,-1],[2,1]],margin=0.)
        self.assertEqual(low,2.)

    def test_open_cup_keeps_aperture_and_bottom_and_contains_random_rotated_triangles(self):
        vertices,triangles=cup();result=axis_yaw_swept_prototype(vertices,triangles,slab_count=16,angular_segments=32)
        self.assertGreater(result.receipt['retained_cavity_slabs'],0)
        self.assertFalse(points_inside_prototype([[1.5,0,0],[2.,0,0]],result).any())
        self.assertTrue(points_inside_prototype([[.1,0,0]],result).all())
        self.assertTrue(points_inside_prototype(random_surface_yaw_points(vertices,triangles),result).all())
        self.assertTrue(all(row['exact_float32_polygon_containment_certificates'] for row in result.slabs))
        certificate=verify_stored_sweep_exact(vertices,triangles,result)
        self.assertTrue(certificate['exact_rational_source_triangle_slab_clips'])
        import copy
        bad=copy.deepcopy(list(result.slabs))
        cavity=next(row for row in bad if row['inner_polygon_yz'] is not None)
        cavity['inner_polygon_yz'][0]=[100.,0.]
        with self.assertRaises(ValueError):
            verify_stored_sweep_exact(vertices,triangles,SweptPrototype(result.vertices,result.triangles,tuple(bad),result.receipt))

    def test_closed_solid_does_not_acquire_a_false_hole_in_middle_slabs(self):
        vertices=np.asarray(list(itertools.product((-1.,1.),repeat=3)))
        triangles=np.asarray(((0,1,3),(0,3,2),(4,6,7),(4,7,5),(0,4,5),(0,5,1),
            (2,3,7),(2,7,6),(0,2,6),(0,6,4),(1,5,7),(1,7,3)))
        result=axis_yaw_swept_prototype(vertices,triangles,slab_count=8,angular_segments=32)
        self.assertEqual(result.receipt['retained_cavity_slabs'],0)
        self.assertTrue(points_inside_prototype([[0,0,0],[.5,0,0]],result).all())
        self.assertTrue(points_inside_prototype(random_surface_yaw_points(vertices,triangles),result).all())

    def test_directed_current_axis_is_mandatory_and_agreement_is_not_accuracy(self):
        evidence=dict(axis_observable=True,axis_polarity='directed',axis_world=[0.,0.,1.],
            evidence='current RGBD near-optimal registration agreement',accuracy_certified=False)
        self.assertTrue(partial_collision_axis_ready(evidence,axis_local=[1.,0.,0.]))
        for change in (dict(axis_observable=False),dict(axis_polarity='undirected'),dict(axis_world=[0.,0.,0.])):
            self.assertFalse(partial_collision_axis_ready({**evidence,**change},axis_local=[1.,0.,0.]))
        self.assertFalse(partial_collision_axis_ready(evidence,axis_local=[0.,0.,1.]))


if __name__=='__main__':unittest.main()
