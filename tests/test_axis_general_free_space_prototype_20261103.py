import unittest
from copy import deepcopy
import numpy as np
from our_mani.local_mpc_control_20261103.axis_general_free_space_prototype_20261103 import whole_pixel_face_footprints,refine_axis_general
from our_mani.near_goal_cube_geometry_20261103_v1.free_space import whole_pixel_top_footprints,subtract_open_forbidden


def camera():
    K=np.array([[50.,0.,16.],[0.,50.,16.],[0.,0.,1.]])
    T=np.eye(4);T[:3,:3]=np.diag([1.,-1.,-1.]);T[:3,3]=[0.,0.,1.]
    return K,T

class AxisGeneralTests(unittest.TestCase):
    def test_knownZ_reduces_to_existing_whole_pixel_eight_corner_geometry(self):
        K,T=camera();d=np.full((32,32),2.,np.float32)
        old=whole_pixel_top_footprints(d,K,T,0.)
        new=whole_pixel_face_footprints(d,K,T,2,np.array([.025-.00025,.025+.00025]),1)
        for k in ('free','footprint_min','footprint_max','max_expected_depth'):
            np.testing.assert_array_equal(new[k],old[k])
    def test_proper_axis_permutation_maps_knownZ_geometry_to_knownX(self):
        K,T=camera();d=np.full((32,32),2.,np.float32)
        z=whole_pixel_face_footprints(d,K,T,2,[.02475,.02525],1)
        P=np.array([[0.,0.,1.],[1.,0.,0.],[0.,1.,0.]])
        other=T.copy();other[:3,:3]=P@T[:3,:3];other[:3,3]=P@T[:3,3]
        x=whole_pixel_face_footprints(d,K,other,0,[.02475,.02525],1)
        np.testing.assert_array_equal(x['free'],z['free'])
        np.testing.assert_array_equal(x['max_expected_depth'],z['max_expected_depth'])
        np.testing.assert_array_equal(x['footprint_min'],z['footprint_min'][:,[2,0,1]])
        np.testing.assert_array_equal(x['footprint_max'],z['footprint_max'][:,[2,0,1]])
    def test_occluding_or_invalid_depth_is_never_free(self):
        K,T=camera()
        for value in (0.,float('nan'),.5):
            out=whole_pixel_face_footprints(np.full((32,32),value,np.float32),K,T,2,[.02475,.02525],1)
            self.assertFalse(out['free'].any())
    def test_E_pass_keeps_every_original_value(self):
        support={'support_available':True,'centre':[1.,2.,3.],'radius_each':[.00025]*3,'intervals':[[1.,1.],[2.,2.],[3.,3.]],'original_extra':'same'}
        out=refine_axis_general(None,None,None,support)
        for k,v in support.items():self.assertEqual(out[k],v)
        self.assertTrue(out['axis_general_trace']['E_passthrough'])
    def test_all_current_qualified_axes_considered_and_near_depth_does_not_shrink(self):
        K,T=camera();T[:3,3]=[1.,1.,1.]
        support={'support_available':False,'axis_counts':[3,3,0],'intervals':[[-.00025,.00025],[-.00025,.00025],[-.01,.01]],'coordinate_sources':['visible_face_median','visible_face_median','bounded_current_support']}
        out=refine_axis_general(np.full((32,32),.001,np.float32),K,T,support)
        self.assertEqual(out['axis_general_trace']['usable_anchor_axes'],[0,1])
        self.assertEqual([(s['round'],s['known_face_axis'])for s in out['axis_general_trace']['rounds']],[(0,0),(0,1),(1,0),(1,1)])
        self.assertEqual(out['feasible_components'][2],[[-.01,.01]])
        self.assertFalse(out['support_available'])
    def test_no_faces_and_strict_open_components_are_not_guessed(self):
        K,T=camera();s={'support_available':False,'axis_counts':[0,0,0],'intervals':[[-.01,.01]]*3,'coordinate_sources':['bounded_current_support']*3}
        out=refine_axis_general(np.full((32,32),2.,np.float32),K,T,s)
        self.assertFalse(out['support_available']);self.assertEqual(out['axis_general_trace']['usable_anchor_axes'],[])
        self.assertEqual(subtract_open_forbidden([[0.,3.]],[[1.,2.]]),[[0.,1.],[2.,3.]])
        self.assertEqual(subtract_open_forbidden([[0.,3.]],[[0.,1.],[1.,3.]]),[[0.,0.],[1.,1.],[3.,3.]])

if __name__=='__main__':unittest.main()
