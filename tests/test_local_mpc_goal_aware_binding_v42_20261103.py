"""Exact-source binding, unknown handling and runner-delta contracts; CPU only."""
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch
import ast,hashlib,inspect
import unittest
import numpy as np
from our_mani.contracts import ObjectState,Pose
from our_mani.perception import CameraCalibration
from our_mani.benchmark_perception import RedBoxTracker
from our_mani.local_mpc_control_20261103.goal_aware_perception import GoalAwareRedBoxTracker,VisualGoalSphere
from our_mani.local_mpc_control_20261103.public_grasp_state import ApprovedTrackerCacheBinding
from our_mani.local_mpc_control_20261103.goal_aware_binding_v42 import (
    ApprovedGoalAwareTrackerCacheBinding,GoalAwarePublicGraspStateAssembler,RAW_TRACKER_KEYS,APPROVED_COMPONENT_SHA256)
from our_mani.local_mpc_control_20261103 import public_grasp_v41 as v41
from our_mani.local_mpc_control_20261103 import public_grasp_v42 as v42
from our_mani.local_mpc_control_20261103.public_motion_bridge import PublicRobotSensors
from tests.test_local_mpc_goal_aware_perception_20261103 import observation,metadata
from tests.test_local_mpc_active_visual_attachment_20261103 import h


def target(t):
    p=np.array([.25,-.15,.775])
    return ObjectState('target',t,Pose(p),np.zeros((3,3)),np.zeros(3),np.zeros(3))


def robot(t):return PublicRobotSensors(t,[.25,-.15,.80],[0,0,0,1],np.zeros(7),np.zeros(7),[1.,1.])


def raw_metadata():return dict(backend='rgbd_red_known_axis_aligned_box',measured_points=64,visible_views=4,
    geometry_completion=True,simulator_pose_used=False,future_observations_used=False,known_dimensions_m=[.05,.05,.05])


class GoalAwareBindingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.wrapper=v42.GoalAwareActiveVisualPublicBridge()
    def setUp(self):self.wrapper.reset()
    def feed(self,t=0.,obs=None):
        obs=observation(t) if obs is None else obs
        with patch.object(self.wrapper.tracker.marker,'observe',return_value=(Pose([.25,.1,1.015]),metadata())),\
             patch.object(self.wrapper.tracker.tracker,'observe',return_value=(target(t),raw_metadata())),\
             patch.object(self.wrapper.bridge.goal_marker,'observe',return_value=(Pose([.25,.1,1.015]),metadata())):
            return self.wrapper.observe(obs,robot(t),close_requested=False)

    def test_explicit_new_types_source_and_frozen_model_not_broad_injection_exception(self):
        binding=self.wrapper.binding;binding.validate_dependencies()
        assembler=GoalAwarePublicGraspStateAssembler(binding)
        self.assertFalse(assembler.allow_test_doubles)
        with self.assertRaises(ValueError):ApprovedTrackerCacheBinding(self.wrapper.bridge,self.wrapper.tracker,self.wrapper.cache)
        class Unapproved(GoalAwareRedBoxTracker):pass
        original=binding.tracker;binding.tracker=Unapproved()
        with self.assertRaises(ValueError):binding.validate_dependencies()
        binding.tracker=original
        for attr in ('_mock_forecaster','_goal_marker_injected'):
            setattr(self.wrapper.bridge,attr,True)
            with self.assertRaises(ValueError):binding.validate_dependencies()
            setattr(self.wrapper.bridge,attr,False)
        with patch.dict(APPROVED_COMPONENT_SHA256,{'our_mani/benchmark_perception.py':'0'*64}):
            with self.assertRaisesRegex(ValueError,'SHA'):binding.validate_dependencies()
        binding.validate_dependencies()

    def test_extra_filter_trace_is_separate_from_whitelisted_sameframe_raw7(self):
        seen=[];stage=self.wrapper.cache.stage_public_target
        def record(value,**kwargs):seen.append(kwargs['diagnostic']);return stage(value,**kwargs)
        with patch.object(self.wrapper.cache,'stage_public_target',side_effect=record):result=self.feed()
        self.assertEqual(set(seen[0]),RAW_TRACKER_KEYS)
        self.assertTrue(result['tracker_dependency_injected']);self.assertFalse(result['forecaster_is_injected_test_double'])
        self.assertTrue(result['goal_aware_extra_metadata_separate_from_raw7_cache'])
        self.assertTrue(result['goal_aware_filter_trace']['filter_available'])
        self.assertIsNotNone(self.wrapper.binding.receipt()['scene_identity']['initial_raw_RGBD_sha256'])
        self.assertEqual(result['goal_aware_binding_version'],'v42')

    def test_static_sphere_and_signed_camera_identity_cannot_silently_refit(self):
        self.feed();binding=self.wrapper.binding;sphere=self.wrapper.tracker.sphere
        self.wrapper.tracker._sphere=VisualGoalSphere(sphere.center,sphere.radius_m,sphere.observed_at,sphere.median_fit_residual_m,sphere.surface_tolerance_m,sphere.marker_to_goal_offset)
        with self.assertRaisesRegex(ValueError,'sphere identity'):binding.validate_dependencies()
        self.wrapper.tracker._sphere=sphere
        changed=observation(.05);c=changed['observation/calibrations']['front']
        K=c.intrinsics.copy();K[0,0]*=1.01
        changed['observation/calibrations']['front']=CameraCalibration(K,c.camera_to_world)
        with self.assertRaisesRegex(ValueError,'calibration identity'):binding.begin_frame(changed)
        old_episode=binding.episode_id;self.wrapper.reset();self.assertNotEqual(old_episode,binding.episode_id)
        self.feed();self.assertIsNot(sphere,self.wrapper.tracker.sphere)

    def test_failed_initial_fit_never_uses_unfiltered_fallback_or_refits_later(self):
        with patch.object(self.wrapper.tracker.marker,'observe',side_effect=ValueError('explicit initial public sphere unavailable')) as fit,\
             patch.object(self.wrapper.tracker.tracker,'observe',return_value=(target(0.),raw_metadata())):
            result=self.wrapper.observe(observation(),robot(0.),close_requested=False)
        self.assertEqual(fit.call_count,1);self.assertFalse(result['observed_target_available'])
        self.assertIsNone(self.wrapper.binding.latest_visual_target);self.assertFalse(result['free_forecast_available'])
        with patch.object(self.wrapper.tracker.marker,'observe',return_value=(Pose([.25,.1,1.015]),metadata())) as fit,\
             patch.object(self.wrapper.tracker.tracker,'observe',return_value=(target(.05),raw_metadata())):
            result=self.wrapper.observe(observation(.05),robot(.05),close_requested=False)
        self.assertEqual(fit.call_count,0);self.assertIsNone(self.wrapper.tracker.sphere)
        self.assertTrue(result['goal_aware_filter_trace']['strict_initial_fit_unavailable_output_rejected_for_policy'])

    def test_shell_can_remove_all_observed_support_and_unknown_is_preserved_with_raw_unchanged(self):
        obs=observation();before={name:value.copy() for name,value in obs['observation/images'].items()}
        for name in obs['observation/calibrations']:
            obs['observation/calibrations'][name]=CameraCalibration(np.diag([-10000.,-10000.,1.]),Pose([0,0,0]))
        with patch.object(self.wrapper.tracker.marker,'observe',return_value=(Pose([0.,0.,.975]),metadata())),\
             patch.object(self.wrapper.bridge.goal_marker,'observe',return_value=(Pose([0.,0.,.975]),metadata())):
            result=self.wrapper.observe(obs,robot(0.),close_requested=False)
        self.assertFalse(result['observed_target_available']);self.assertFalse(result['free_forecast_available'])
        self.assertIn('not sufficiently visible',result['goal_aware_tracker_error'])
        for name in before:
            np.testing.assert_array_equal(before[name],obs['observation/images'][name])
            self.assertEqual(result['goal_aware_filter_trace']['per_camera'][name]['remaining_valid_red_depth_pixels'],0)
        self.assertTrue(result['goal_aware_filter_trace']['target_surface_overlap_with_sphere_shell_may_remove_target_pixels'])
        assembler=GoalAwarePublicGraspStateAssembler(self.wrapper.binding);assembler.observe_robot(robot(0.))
        self.assertIsNone(assembler.build(result,robot(0.),h(0.,target=1.)))

    def test_observation_channels_and_result_identity_are_strict(self):
        obs=observation();obs['Native_held']=True
        with self.assertRaises(ValueError):self.wrapper.binding.begin_frame(obs)
        result=self.feed()
        with self.assertRaises(ValueError):self.wrapper.binding.approve(deepcopy(result))
        result['goal_marker_dependency_injected']=True
        with self.assertRaises(ValueError):self.wrapper.binding.approve(result)

    def test_runner_AST_diff_is_only_explicit_version_type_and_log_names(self):
        new=inspect.getsource(v42.run_public_grasp_v42)
        for old,replacement in v42.RUNNER_APPROVED_RENAMES.items():new=new.replace(replacement,old)
        self.assertEqual(ast.dump(ast.parse(new)),ast.dump(ast.parse(inspect.getsource(v41.run_public_grasp_v41))))
        self.assertEqual(hashlib.sha256(Path(v41.__file__).read_bytes()).hexdigest(),v42.RUNNER_BASE_SOURCE_SHA256)
        self.assertIs(v42.build_response_planner,v41.build_response_planner)
        self.assertIs(v42.ActiveVisualGraspV41Coordinator,v41.ActiveVisualGraspV41Coordinator)


if __name__=='__main__':unittest.main()
