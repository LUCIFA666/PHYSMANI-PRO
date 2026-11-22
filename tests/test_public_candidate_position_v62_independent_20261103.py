"""Independent V62 negative and control-equivalence checks, synthetic CPU only.

No passing position artifact or live constructor is fabricated. Bare frame
objects below isolate reject guards and cannot satisfy the real capability.
"""
import ast
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from our_mani.contracts import Pose
from our_mani.perception import CameraCalibration
from our_mani.local_mpc_control_20261103 import depth_position_goal_perception_v62 as tracker
from our_mani.local_mpc_control_20261103 import goal_aware_position_binding_v62 as binding
from our_mani.local_mpc_control_20261103 import public_candidate_position_v62 as producer
from our_mani.local_mpc_control_20261103 import public_candidate_refiner_v60 as old


ROOT = Path(__file__).resolve().parents[1]


def observation(timestamp=0.):
    names = ('front', '8', '16', '36')
    K = np.array([[-200., 0., 128.], [0., -200., 128.], [0., 0., 1.]])
    return dict(timestamp=timestamp, **{
        'observation/images': {n: np.zeros((256, 256, 3), np.uint8) for n in names},
        'observation/depths': {n: np.ones((256, 256), np.float32) for n in names},
        'observation/calibrations': {n: CameraCalibration(K.copy(), Pose(np.zeros(3))) for n in names},
    })


def unbound_frame_guard():
    obj = object.__new__(binding.ApprovedDepthPositionTrackerCacheBindingV62)
    obj.reset_episode()
    return obj


def current_cloud():
    cloud = np.arange(36, dtype=float).reshape(12, 3) / 1000
    rows = [dict(camera=i, baseline_candidate_points=3, accepted=[True] * 3,
                 accepted_points=3, point_world=cloud[3*i:3*i+3].tolist()) for i in range(4)]
    return dict(geometry_profile=tracker.GEOMETRY_VERSION, camera_point_ledger=rows,
                current_points_world=cloud.tolist(), measured_points=12, visible_views=4)


class PositionV62IndependentTests(unittest.TestCase):
    def test_constructor_exposes_no_diagnostic_or_runtime_injection(self):
        for extra in ({'refiner': object()}, {'allow_unqualified_diagnostic': True}):
            with self.assertRaises(TypeError):
                tracker.CurrentDepthPositionTrackerV62('missing.pt', 'a' * 64, **extra)
        for digest in (True, None, '', 'A' * 64):
            with self.assertRaises(ValueError):
                tracker.CurrentDepthPositionTrackerV62('missing.pt', digest)
        with self.assertRaises(FileNotFoundError):
            tracker.CurrentDepthPositionTrackerV62('/nonexistent/V62.pt', 'a' * 64)
        with self.assertRaisesRegex(ValueError, 'genuine'):
            object.__new__(tracker.CurrentDepthPositionTrackerV62).validate_refiner()
        for cls in (producer.PublicCandidatePositionStateAssemblerV62,
                    binding.DepthPositionPublicGraspStateAssemblerV62):
            with self.assertRaises(TypeError):
                cls(object())

    def test_position_capability_is_not_generic_or_raw_model_qualification(self):
        fields = dict(position_measurement_qualified=True, qualified=False,
                      model_qualified=False, raw_mask_model_qualified=False)
        self.assertTrue(tracker._position_flags(SimpleNamespace(**fields)))
        for key, bad in [('position_measurement_qualified', 1), ('qualified', True),
                         ('model_qualified', True), ('raw_mask_model_qualified', True)]:
            changed = dict(fields); changed[key] = bad
            self.assertFalse(tracker._position_flags(SimpleNamespace(**changed)))
        self.assertFalse(tracker._position_flags(SimpleNamespace(qualified=True)))

    def test_exact_F_accepted_cloud_not_raw_integer_cloud_or_template(self):
        measurement = current_cloud()
        cloud, views = tracker.current_position_depth_cloud(measurement)
        np.testing.assert_array_equal(cloud, measurement['current_points_world'])
        self.assertEqual((len(cloud), views), (12, 4))
        for mutate in (
            lambda m: m.update(geometry_profile='legacy_raw_cloud'),
            lambda m: m.update(current_points_world=[[99., 99., 99.]] * 12),
            lambda m: m.update(visible_views=3),
            lambda m: m['camera_point_ledger'][0].update(accepted=[1, 1, 1]),
            lambda m: m['camera_point_ledger'][0].update(accepted_points=2),
            lambda m: m['camera_point_ledger'][0]['point_world'][0].__setitem__(0, float('nan')),
            lambda m: m['camera_point_ledger'][1].update(camera=0),
        ):
            changed = deepcopy(measurement); mutate(changed)
            with self.assertRaises(ValueError):
                tracker.current_position_depth_cloud(changed)

    def test_unbound_frame_guard_rejects_duplicate_clock_calibration_and_depth(self):
        obj = unbound_frame_guard()
        with patch.object(obj, 'validate_dependencies'):
            obj.begin_frame(observation())
            for timestamp in (0., -.05, True, np.bool_(True), float('nan')):
                with self.assertRaises(ValueError):
                    obj.begin_frame(observation(timestamp))
            changed = observation(.05)
            K = changed['observation/calibrations']['front'].intrinsics.copy(); K[0, 0] -= 1
            changed['observation/calibrations']['front'] = CameraCalibration(K, Pose(np.zeros(3)))
            with self.assertRaises(ValueError):
                obj.begin_frame(changed)
        for mutate in (
            lambda o: o['observation/depths'].update(front=np.ones((256, 256), np.float64)),
            lambda o: o.update(Native_inputs_used=True),
            lambda o: o.update(future_observations_used=True),
        ):
            obj = unbound_frame_guard(); changed = observation(); mutate(changed)
            with patch.object(obj, 'validate_dependencies'):
                with self.assertRaises(ValueError):
                    obj.begin_frame(changed)

    def test_unbound_current_result_rejects_alias_provenance_and_RGBD_mutation(self):
        obj = unbound_frame_guard(); obs = observation()
        trace = dict(timestamp=0., model_checkpoint_sha256='a' * 64,
                     position_measurement_qualified=True, model_qualified=False,
                     raw_mask_model_qualified=False, packed_current_input_sha256=None)
        obj.tracker = SimpleNamespace(_last_observed_timestamp=0., expected_refiner_sha256='a' * 64,
                                     last_measurement_metadata=trace, sphere=None)
        result = dict(goal_aware_binding_version='v62', tracker_dependency_injected=True,
                      forecaster_is_injected_test_double=False, goal_marker_dependency_injected=False,
                      Native_inputs_used=False, future_observations_used=False,
                      observations_fabricated=False, depth_refiner_trace=deepcopy(trace))
        with patch.object(obj, 'validate_dependencies'):
            obj.begin_frame(obs); obj.latest_result = result
            with self.assertRaises(ValueError): obj.approve(deepcopy(result))
            for key in ('Native_inputs_used', 'future_observations_used', 'observations_fabricated'):
                result[key] = True
                with self.assertRaises(ValueError): obj.approve(result)
                result[key] = False
            result['depth_refiner_trace']['model_qualified'] = True
            with self.assertRaises(ValueError): obj.approve(result)
            result['depth_refiner_trace'] = deepcopy(trace)
            obs['observation/images']['front'][0, 0, 0] = 1
            with self.assertRaisesRegex(ValueError, 'mutated'):
                obj.approve(result)

    def test_route_switch_clears_only_motion_and_requires_explicit_flags(self):
        bridge = SimpleNamespace(_clear_motion=Mock(), attachment=object(), goal=object(), cache=object())
        before = (bridge.attachment, bridge.goal, bridge.cache)
        producer._reset_motion_history_for_route_change(bridge, {'accepted_history_reset_for_route_change': False})
        bridge._clear_motion.assert_not_called()
        trace = {'accepted_history_reset_for_route_change': True}
        producer._reset_motion_history_for_route_change(bridge, trace)
        bridge._clear_motion.assert_called_once_with(clear_tracker=False)
        self.assertEqual(before, (bridge.attachment, bridge.goal, bridge.cache))
        self.assertFalse(trace['attachment_probe_maintenance_goal_Sphere_and_cache_reset'])
        for bad in (False, 1, 'true', None):
            with self.assertRaises(ValueError): producer.v62_enabled(goal_sphere_filter=bad)
            with self.assertRaises(ValueError): producer.v62_enabled(postconfirmed_maintenance=bad)
        backend = SimpleNamespace(bridge=SimpleNamespace(goal_sphere_filter=True, postconfirmed_maintenance=True))
        config = dict(candidate_position_measurement_v62=True, candidate_goal_sphere_filter_v56=True,
                      postconfirmed_position_maintenance_v56=True)
        for mutate in (lambda c: c.update(candidate_position_measurement_v62=1),
                       lambda c: c.update(public_grasp_active_visual_v42=True),
                       lambda c: c.update(candidate_goal_sphere_filter_v56=False)):
            changed = dict(config); mutate(changed)
            with self.assertRaises(ValueError): producer._validate_v62_runner(backend, changed)

    def test_control_runner_sample_and_coordinator_AST_unchanged(self):
        replacements = {
            'CandidatePositionV62Coordinator': 'CandidateGraspDepthRefinerV60Coordinator',
            'ObservedPublicCandidatePositionV62Backend': 'ObservedPublicCandidateDepthRefinerV60Backend',
            'run_public_candidate_position_v62': 'run_public_candidate_depth_refiner_v60',
            '_validate_v62_runner': '_validate_v60_runner',
            'public_candidate_position_v62': 'public_candidate_depth_refiner_v60',
            'P4_candidate_transformer_v62_': 'P4_candidate_transformer_v60_',
            'candidate_position_measurement_v62': 'candidate_depth_refiner_target_v60',
            'v62_enabled': 'v60_enabled', 'V62': 'V60',
        }
        source = Path(producer.__file__).read_text()
        for key in sorted(replacements, key=len, reverse=True):
            source = source.replace(key, replacements[key])
        new_tree, old_tree = ast.parse(source), ast.parse(Path(old.__file__).read_text())
        def find(tree, name, parent=None):
            nodes = tree.body if parent is None else next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == parent).body
            return next(n for n in nodes if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name)
        pairs = [('run_public_candidate_depth_refiner_v60', None),
                 ('sample', 'ObservedPublicCandidateDepthRefinerV60Backend'),
                 ('decide', 'CandidateGraspDepthRefinerV60Coordinator')]
        for name, parent in pairs:
            self.assertEqual(ast.dump(find(new_tree, name, parent), include_attributes=False),
                             ast.dump(find(old_tree, name, parent), include_attributes=False), name)


if __name__ == '__main__': unittest.main()
