"""V70 pure geometry/source/transaction contracts, no model construction."""
from collections import deque
from copy import deepcopy
from types import SimpleNamespace
import ast
import inspect
import textwrap
import unittest

import numpy as np

from our_mani.contracts import Pose
from our_mani.perception import CameraCalibration
from our_mani.benchmark_perception import RedBoxTracker, BoxPrior
from our_mani.local_mpc_control_20261103 import depth_pixel_centre_goal_perception_v70 as new
from our_mani.local_mpc_control_20261103 import depth_scope12_goal_perception_v67 as old
from our_mani.local_mpc_control_20261103 import goal_aware_pixel_centre_binding_v70 as binding
from our_mani.local_mpc_control_20261103 import public_candidate_response_seed_v70 as main
from our_mani.local_mpc_control_20261103 import public_candidate_response_seed_v69 as control
from our_mani.local_mpc_control_20261103 import public_candidate_pixel_centre_measurement_v70 as primitive
from our_mani.local_mpc_control_20261103.goal_aware_perception import VisualGoalSphere
from our_mani.local_mpc_control_20261103.depth_refiner_public_inputs_v60 import pack_current_public_refiner_inputs


def node(function):
    return ast.parse(textwrap.dedent(inspect.getsource(function))).body[0]


def dump(value):
    return ast.dump(value, include_attributes=False)


def observation():
    K = np.array([[-200., 1.5, 128.], [0., -210., 128.], [0., 0., 1.]])
    R = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    calibrations = {name:CameraCalibration(K, Pose([.1, -.2, .4], R)) for name in ('front', '8', '16', '36')}
    return {'timestamp':1., 'observation/calibrations':calibrations,
            'observation/images':{name:np.zeros((256, 256, 3), np.uint8) for name in calibrations},
            'observation/depths':{name:np.ones((256, 256), np.float32) for name in calibrations}}


def scratch_function(filter_function):
    """Execute the production AST's far transaction, without any NN prefix.

    This deliberately does not construct or authenticate a live producer.
    Only the actual previous/raw/try-finally nodes are transplanted verbatim.
    """
    source = node(new.CurrentDepthPixelCentreTrackerV70.observe)
    start = next(i for i, item in enumerate(source.body) if isinstance(item, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == 'previous' for t in item.targets))
    end = next(i for i in range(start, len(source.body)) if isinstance(source.body[i], ast.Try))
    body = deepcopy(source.body[start:end+1])
    body.append(ast.Return(value=ast.Tuple(elts=[ast.Name(id=x, ctx=ast.Load()) for x in ('raw', 'metadata', 'raw_error')], ctx=ast.Load())))
    function = ast.FunctionDef(name='scratch', args=ast.arguments(posonlyargs=[], args=[ast.arg(arg='self'), ast.arg(arg='observation')],
        vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[]), body=body, decorator_list=[])
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    namespace = {'filter_observed_sphere_surface':filter_function}
    exec(compile(module, '<exact_V70_far_transaction_AST>', 'exec'), namespace)
    return namespace['scratch']


class FarPixelV70IndependentTests(unittest.TestCase):
    def test_private_calibration_only_original_raw_and_NN_packer_are_unchanged(self):
        obs = observation()
        sphere = VisualGoalSphere([0., 0., 1.], .038, 0., .001, .004, [0., 0., .015])
        packed_before = pack_current_public_refiner_inputs(obs, sphere)
        raw_before = binding.public_RGBD_digest(obs)
        calibration_before = binding.digest(binding.calibration_identity(obs))
        private = new.CurrentDepthPixelCentreTrackerV70._private_far_observation(None, obs)
        self.assertIsNot(private, obs)
        self.assertIsNot(private['observation/calibrations'], obs['observation/calibrations'])
        self.assertIs(private['observation/images'], obs['observation/images'])
        self.assertIs(private['observation/depths'], obs['observation/depths'])
        for name, original in obs['observation/calibrations'].items():
            self.assertIs(type(original), CameraCalibration)
            self.assertIs(type(private['observation/calibrations'][name]), new.PixelCentreFarCalibration)
            np.testing.assert_array_equal(private['observation/calibrations'][name].intrinsics, original.intrinsics)
            self.assertIsNot(private['observation/calibrations'][name].intrinsics, original.intrinsics)
            self.assertIs(private['observation/calibrations'][name].camera_to_world, original.camera_to_world)
        self.assertEqual(binding.public_RGBD_digest(obs), raw_before)
        self.assertEqual(binding.digest(binding.calibration_identity(obs)), calibration_before)
        packed_after = pack_current_public_refiner_inputs(obs, sphere)
        for key in packed_before:
            np.testing.assert_array_equal(packed_after[key], packed_before[key])

    def test_signed_K_half_pixel_is_only_ray_delta_and_preserves_all_valid_pixels(self):
        original = observation()['observation/calibrations']['front']
        private = new.PixelCentreFarCalibration(original.intrinsics, original.camera_to_world)
        depth = np.array([[.7, np.nan, .9], [0., 1.1, .5]])
        mask = np.array([[True, True, False], [True, True, True]])
        y, x = np.nonzero(mask & np.isfinite(depth) & (depth > 0))
        rays = np.c_[x+.5, y+.5, np.ones(len(x))] @ np.linalg.inv(original.intrinsics).T
        expected = original.camera_to_world.apply(rays * depth[y, x, None])
        actual = private.backproject(depth, mask)
        self.assertEqual(len(actual), len(x))
        np.testing.assert_array_equal(actual, expected)
        self.assertFalse(np.array_equal(actual, original.backproject(depth, mask)))

    def test_private_double_wrapping_and_foreign_public_fields_are_rejected(self):
        obs = observation()
        private = new.CurrentDepthPixelCentreTrackerV70._private_far_observation(None, obs)
        with self.assertRaises(ValueError):
            new.CurrentDepthPixelCentreTrackerV70._private_far_observation(None, private)
        obs['robot_q'] = np.zeros(7)
        with self.assertRaises(ValueError):
            new.CurrentDepthPixelCentreTrackerV70._private_far_observation(None, obs)

    def test_actual_far_transaction_rolls_back_then_final_accept_keeps_true_history_and_full_cloud(self):
        obs = observation()
        for fail in (False, True):
            with self.subTest(raw_failure=fail):
                # Unconstructed unit fixture: never a qualified runtime/binding.
                tracker = object.__new__(new.CurrentDepthPixelCentreTrackerV70)
                class RawProbe(RedBoxTracker):
                    def observe(self, current):
                        self.seen_history = [(t, p.copy()) for t, p in self.history]
                        self.seen_calibrations = current['observation/calibrations']
                        self.history.append((.999, np.full(3, 999.)))
                        self.last_velocity[:] = 999.
                        if fail:raise RuntimeError('synthetic raw failure after mutation')
                        return SimpleNamespace(current_component=True), {'geometry_completion':False}
                tracker.tracker = RawProbe(BoxPrior(minimum_red=100), complete_geometry=False)
                tracker.prior = tracker.tracker.prior
                tracker.history.extend([(.9, np.array([-.005, 0., .775])), (.95, np.array([0., 0., .775]))])
                tracker.last_velocity = [.1, 0., 0.]
                tracker._sphere = SimpleNamespace(public_only=True)
                tracker._accepted_source = 'FAR_R100_DEVELOPMENT_CONSISTENCY'
                tracker.last_measurement_metadata = {}
                before = [(t, p.copy()) for t, p in tracker.history]
                def filter_spy(private_obs, sphere, prior):
                    self.assertIs(private_obs['observation/images'], obs['observation/images'])
                    self.assertTrue(all(type(c) is new.PixelCentreFarCalibration for c in private_obs['observation/calibrations'].values()))
                    return private_obs, {'synthetic_public_filter':True}
                raw, metadata, error = scratch_function(filter_spy)(tracker, obs)
                self.assertEqual([t for t, _ in tracker.history], [.9, .95])
                for (_, got), (_, expected) in zip(tracker.history, before):np.testing.assert_array_equal(got, expected)
                np.testing.assert_array_equal(tracker.last_velocity, [.1, 0., 0.])
                self.assertEqual([t for t, _ in tracker.tracker.seen_history], [.9, .95])
                if fail:
                    self.assertIsNone(raw)
                    self.assertIn('synthetic raw failure', error)
                    with self.assertRaises(ValueError):tracker._unknown('missing_current_measurement')
                    self.assertEqual(len(tracker.history), 0)
                    self.assertIsNone(tracker.last_current_target)
                else:
                    self.assertIsNone(error)
                    cloud = np.c_[np.linspace(0., .01, 400), np.zeros(400), np.full(400, .8)]
                    target, public_meta = tracker._accept([.005, 0., .775], 1., cloud, 4, 1., 'FAR_R100_DEVELOPMENT_CONSISTENCY')
                    self.assertEqual([t for t, _ in tracker.history], [.9, .95, 1.])
                    np.testing.assert_array_equal(tracker.last_measured_cloud, cloud)
                    self.assertEqual(public_meta['measured_points'], 400)
                    self.assertEqual(len(target.points_world), 256)
                    self.assertFalse(tracker.last_measurement_metadata['template_is_measured_cloud'])

    def test_observe_NN_Sphere_and_history_math_exact_except_private_far_call_and_three_flags(self):
        before, after = node(old.CurrentDepthScope12TrackerV67.observe), node(new.CurrentDepthPixelCentreTrackerV70.observe)
        flags = {'far_backprojection_pixel_centre_offset', 'far_calibration_is_private', 'raw_RGBD_calibration_rewritten'}
        class Normalize(ast.NodeTransformer):
            def visit_Call(self, item):
                item.keywords = [keyword for keyword in item.keywords if keyword.arg not in flags]
                return self.generic_visit(item)
            def visit_Assign(self, item):
                if any(isinstance(target, ast.Name) and target.id == 'private_far' for target in item.targets):return None
                return self.generic_visit(item)
            def visit_Name(self, item):
                if item.id == 'private_far':item.id = 'observation'
                return item
        self.assertEqual(dump(before), dump(Normalize().visit(after)))

    def test_new_binding_rejects_old_tracker_and_replaced_private_method_before_runtime(self):
        cap = object.__new__(binding.ApprovedDepthPixelCentreTrackerCacheBindingV70)
        cap.bridge = object.__new__(binding.PublicMotionBridge)
        cap.cache = object.__new__(binding.PublicTargetFrameCache)
        cap.tracker = object.__new__(old.CurrentDepthScope12TrackerV67)
        with self.assertRaises(ValueError):cap.validate_dependencies()
        tracker = object.__new__(new.CurrentDepthPixelCentreTrackerV70)
        tracker.tracker = RedBoxTracker(BoxPrior(minimum_red=100), complete_geometry=False)
        tracker.prior = tracker.tracker.prior
        tracker.marker = binding.RedGoalMarker()
        tracker.config = binding.GoalSphereFilterConfig()
        tracker._private_far_observation = lambda obs:obs
        cap.tracker = tracker;cap.bridge.tracker = cap.cache;cap.cache.public_tracker = tracker
        with self.assertRaisesRegex(ValueError, 'Exact R100'):cap.validate_dependencies()

    def test_public_whitelist_drops_full_private_tree_and_rejects_unregistered_key(self):
        metadata = {'timestamp':1., 'far_backprojection_pixel_centre_offset':.5,
                    'far_calibration_is_private':True, 'raw_RGBD_calibration_rewritten':False,
                    'NN_current_measurement':{'source':{'GT_private_diagnostic':False}},
                    'far_consistency':{'semantic_ownership_verified':False}}
        public = new.project_public_position_evidence(metadata)
        self.assertNotIn('NN_current_measurement', public)
        self.assertNotIn('far_consistency', public)
        self.assertEqual(public['far_backprojection_pixel_centre_offset'], .5)
        metadata['unregistered_private_tree'] = {}
        with self.assertRaises(ValueError):new.project_public_position_evidence(metadata)

    def test_original_V69_hold_dispatch_and_raw_P4_functions_are_actual_original_objects(self):
        self.assertIs(main.ResponseSeedHoldPlannerV69, control.ResponseSeedHoldPlannerV69)
        self.assertIn(control._V69V4Dispatch, main.CandidateResponseSeedV70Coordinator.__mro__)
        self.assertNotIn('decide', main.CandidateResponseSeedV70Coordinator.__dict__)
        from our_mani.local_mpc_p4_20261103.runtime import pack_public_row, PublicRowCache
        self.assertIs(primitive.pack_public_row, pack_public_row)
        self.assertIs(primitive.PublicRowCache, PublicRowCache)


if __name__ == '__main__':
    unittest.main()
