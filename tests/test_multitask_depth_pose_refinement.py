import unittest
import numpy as np

from our_mani.contracts import Pose
from our_mani.perception import CameraCalibration
from our_mani.multitask.depth_pose_refinement import (JointCurrentDepthObjective,
    JointRefinementConfig,refine_current_depth_pose,refine_pose_candidates)


class JointDepthRefinementTests(unittest.TestCase):
    def setUp(self):
        self.pose=Pose(np.zeros(3))
        self.cal=CameraCalibration(np.array([[10.,0.,5.],[0.,10.,5.],[0.,0.,1.]]),self.pose)
        self.surface={'body':np.array([[-.1,0.,1.],[0.,0.,1.],[.1,0.,1.]])}

    def views(self,depth):return {'front':(self.cal,depth)}

    def test_joint_optimization_improves_its_actual_depth_and_observed_objective(self):
        result=refine_current_depth_pose(self.surface,self.surface,self.pose,self.views(np.full((11,11),1.02)))
        self.assertLess(result['final']['joint_cost_m2'],result['initial']['joint_cost_m2']*.5)
        self.assertGreater(result['pose'].position[2],.005)
        self.assertLess(result['pose'].position[2],.02)
        self.assertLessEqual(result['nfev'],30)
        self.assertFalse(result['GT_native_future_or_outcome_used'])

    def test_bilinear_sampling_has_xy_gradient_inside_pixel(self):
        depth=np.tile(1.02+.01*(np.arange(11)-5),(11,1))
        objective=JointCurrentDepthObjective(self.surface,self.surface,self.views(depth))
        _,left=objective.evaluate(Pose(np.array([-.0001,0.,0.])))
        _,right=objective.evaluate(Pose(np.array([.0001,0.,0.])))
        self.assertLess(left['current_depth_free_space_mse_m2'],right['current_depth_free_space_mse_m2'])

    def test_true_occlusion_no_force_and_no_false_pose_certificate(self):
        result=refine_current_depth_pose(self.surface,self.surface,self.pose,self.views(np.full((11,11),.5)))
        self.assertEqual(result['initial']['current_depth_free_space_mse_m2'],0.)
        self.assertTrue(np.allclose(result['pose'].position,0.))
        self.assertFalse(result['pose_observability_proven'])

    def test_invalid_bilinear_support_and_out_of_frustum_zero(self):
        depth=np.full((11,11),np.nan)
        objective=JointCurrentDepthObjective(self.surface,self.surface,self.views(depth))
        _,metric=objective.evaluate(self.pose)
        self.assertEqual(metric['current_depth_free_space_mse_m2'],0.)
        self.assertEqual(metric['current_depth_counts']['invalid_depth_pixels'],3)
        _,outside=objective.evaluate(Pose(np.array([100.,0.,0.])))
        self.assertEqual(outside['current_depth_free_space_mse_m2'],0.)
        self.assertEqual(outside['current_depth_counts']['outside_frustum_samples'],3)

    def test_fixed_four_candidates_selection_uses_same_joint_cost(self):
        seeds=(self.pose,Pose(np.array([0.,0.,.005])),Pose(np.array([0.,0.,-.005])),Pose(np.array([0.,0.,.01])))
        result=refine_pose_candidates(self.surface,self.surface,seeds,self.views(np.full((11,11),1.02)))
        self.assertEqual(result['candidate_count'],4)
        self.assertEqual(result['selection_cost_m2'],min(row['final']['joint_cost_m2'] for row in result['candidates']))
        self.assertTrue(all(row['nfev']<=30 for row in result['candidates']))

    def test_no_future_or_GT_seed_keyword_and_no_bounds_input(self):
        with self.assertRaises(TypeError):
            refine_current_depth_pose(self.surface,self.surface,self.pose,self.views(np.ones((11,11))),GT_pose=self.pose)
        with self.assertRaises(TypeError):
            refine_current_depth_pose(self.surface,self.surface,self.pose,self.views(np.ones((11,11))),bounds_from_future=np.zeros(6))
        with self.assertRaises(ValueError):JointRefinementConfig(max_nfev=0)


if __name__=='__main__':unittest.main()
