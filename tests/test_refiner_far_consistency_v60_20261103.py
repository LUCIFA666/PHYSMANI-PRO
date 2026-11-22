import unittest
import numpy as np

from our_mani.local_mpc_20261103.refiner_far_consistency_v60 import component_far_consistency


def vertices(center):
    return np.asarray(center)+np.array([[x,y,z] for x in (-.025,.025)
        for y in (-.025,.025) for z in (-.025,.025)])


class FarConsistencyTests(unittest.TestCase):
    def test_complete_component_can_avoid_fixed_half_diagonal_gap(self):
        p=vertices([.10,0,0]); r=component_far_consistency(p,[0,0,0])
        self.assertTrue(r['far_consistent'])
        self.assertAlmostEqual(r['minimum_compatible_centre_distance_m'],.094)
        self.assertLess(.10-np.sqrt(3)*.025,.08)
        self.assertFalse(r['NN_measurement_available'])
        self.assertFalse(r['statistical_or_physical_guarantee'])

    def test_nearest_single_face_leaves_near_centres_compatible(self):
        p=vertices([.10,0,0]); r=component_far_consistency(p[p[:,0]<.1],[0,0,0])
        self.assertFalse(r['far_consistent'])
        self.assertAlmostEqual(r['centre_lower_m'][0],.044)

    def test_empty_outlier_and_nonfinite_remain_unknown(self):
        for p,reason in [(np.empty((0,3)),'empty_current_component'),
            (np.array([[0,0,0],[.063,0,0]]),'empty_compatible_centre_box'),
            (np.array([[np.nan,0,0]]),'nonfinite_component_or_initial_sphere')]:
            r=component_far_consistency(p,[0,0,0])
            self.assertEqual(r['status'],'UNKNOWN');self.assertEqual(r['reason'],reason)

    def test_touching_scope_and_one_ulp_remain_unknown(self):
        p=vertices([.086,0,0])
        self.assertFalse(component_far_consistency(p,[0,0,0])['far_consistent'])
        p[:,0]=np.nextafter(p[:,0],np.inf)
        self.assertFalse(component_far_consistency(p,[0,0,0])['far_consistent'])
        self.assertTrue(component_far_consistency(vertices([.086001,0,0]),[0,0,0])['far_consistent'])

    def test_conditional_centre_containment(self):
        rng=np.random.default_rng(261103700)
        for _ in range(30):
            c=rng.uniform(-.3,.3,3)
            true=c+rng.uniform(-.025,.025,(100,3))
            p=true+rng.uniform(-.006,.006,(100,3))
            r=component_far_consistency(p,np.zeros(3))
            self.assertTrue(np.all(c>=np.asarray(r['centre_lower_m'])-1e-15))
            self.assertTrue(np.all(c<=np.asarray(r['centre_upper_m'])+1e-15))
            if r['far_consistent']:self.assertGreater(np.linalg.norm(c),.08)

    def test_translation_and_permutation_invariant(self):
        p=vertices([.13,.04,0]);s=np.array([0.,0.,0.]);shift=np.array([.2,-.5,1.])
        a=component_far_consistency(p,s);b=component_far_consistency(p[::-1]+shift,s+shift)
        self.assertEqual(a['status'],b['status'])
        self.assertAlmostEqual(a['minimum_compatible_centre_distance_m'],b['minimum_compatible_centre_distance_m'])

    def test_far_box_cannot_authenticate_wrong_component(self):
        r=component_far_consistency(vertices([.2,0,0]),[0,0,0])
        self.assertTrue(r['far_consistent'])
        self.assertFalse(r['semantic_ownership_verified'])

    def test_finite_extremes_cannot_overflow_into_far_classification(self):
        for p,s in [(np.full((2,3),1e308),np.zeros(3)),
                    (np.full((2,3),1e308),np.full(3,-1e308))]:
            r=component_far_consistency(p,s)
            self.assertFalse(r['far_consistent'])
            self.assertEqual(r['reason'],'derived_geometry_nonfinite_or_overflow')
            self.assertIsNone(r['minimum_compatible_centre_distance_m'])

    def test_inputs_not_mutated_and_malformed_rejected(self):
        p=vertices([.1,0,0]);s=np.zeros(3);before=p.copy()
        component_far_consistency(p,s);np.testing.assert_array_equal(p,before)
        for p,s in [(np.zeros(3),s),(np.zeros((2,4)),s),(np.zeros((2,3)),np.zeros(2)),(np.ones((2,3),bool),np.zeros(3))]:
            with self.assertRaises(ValueError):component_far_consistency(p,s)


if __name__=='__main__':unittest.main()
