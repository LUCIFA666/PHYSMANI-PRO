"""Explicit development capability for one V64 current-depth producer.

This authenticates code/model/current public frame provenance.  It does not
certify semantic ownership, physical far separation or closed-loop readiness.
Existing V59 and other approved tracker capabilities remain unchanged.
"""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from uuid import uuid4
import hashlib
import json
import numpy as np

from ..benchmark_perception import RedBoxTracker,BoxPrior
from ..contracts import ObjectState
from ..goal_perception import RedGoalMarker
from ..perception import CameraCalibration
from .goal_aware_perception import GoalSphereFilterConfig,_validate_public
from .depth_position_goal_perception_v64 import CurrentDepthPositionTrackerV64,_packed_digest,_file_stat
from .depth_refiner_public_inputs_v60 import pack_current_public_refiner_inputs
from .public_attachment import PublicTargetFrameCache
from .public_motion_bridge import PublicMotionBridge,MultiBodyContextDynamics,CHECKPOINT_SHA256,GOAL_CALIBRATION,CAMERAS
from .public_grasp_state import PublicGraspStateAssembler
from .public_state import PublicStateAssembler

ROOT=Path(__file__).resolve().parents[2]
RAW_TRACKER_KEYS=frozenset(('backend','measured_points','visible_views','geometry_completion',
    'simulator_pose_used','future_observations_used','known_dimensions_m'))
APPROVED_COMPONENT_SHA256={'our_mani/local_mpc_control_20261103/depth_position_goal_perception_v64.py': '0b075eb8397a0437c8a2352e5e906c855de794070aa2ac30498bd4e4e87abf0f', 'our_mani/local_mpc_control_20261103/depth_refiner_public_inputs_v60.py': 'df8109dd409706e616e611f33a36bf110c07acb6d573bfc190bb5bcb24ddc3c1', 'our_mani/local_mpc_20261103/refiner_far_consistency_v60.py': '621cc6c9f498e7295ad59888f55128dc51137bc2b1fc22183deb1ab8c5857cc4', 'our_mani/local_mpc_control_20261103/goal_aware_perception.py': 'e2105eb7b80269324ff77c8ca28a3cac1a480cf006c1c3b1324867179af05f8c', 'our_mani/benchmark_perception.py': 'e1eb4cff08f96117df50f22672362eb7b4bbf8274eb7b25a8bb2b2b1a863adf2', 'our_mani/goal_perception.py': 'f542fe8671e60bae2d41bb2bc64b85e2ac2147ec1b54879ad8eea8f43b37c1ed', 'our_mani/local_mpc_control_20261103/public_attachment.py': '8cbea0c0c5c2db65022767e13960180a4025540f81f8b2f788b000461a746208', 'our_mani/local_mpc_control_20261103/public_motion_bridge.py': 'e00868f68cbfcae2facb4b5b215202e7986b19c57b7b50ed185f97a6df89b6de', 'our_mani/multitask/context_se3.py': '8f882d93b63c9cd5bb7112f9c89bda0ec10fce8d07224364032e57529be33023', 'artifacts/framework/local_policy/goal_marker_calibration.json': '3973330d5de75f25d049fdcebc433d3f826926f37d4db1e790195b35c55bbc00', 'our_mani/cube_position_measurement_20261103_v8/runtime.py': '22452a1abf64410fdd5f1e215b7ee0e6f12969391a0d04319b6fd68832629810', 'our_mani/cube_position_measurement_20261103_v8/qualification.py': 'd7351ba7c105e0b70f750142e33eac3c2a655e7497364135566f69b2c3cfecea', 'our_mani/cube_position_measurement_20261103_v8/evaluation.py': '18adaa82906ee4f58af24e7de47974d987f918df9c56d3be16cfab55e2b12544', 'our_mani/cube_position_measurement_20261103_v8/prediction.py': 'db33daa0e34340583bcf01f64abac2eb23f2f35246cf4a24e51837beebe75d32'}


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False,
        default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x.item() if isinstance(x,np.generic) else None).encode()).hexdigest()


def calibration_identity(observation):
    return {n:dict(intrinsics=np.asarray(observation['observation/calibrations'][n].intrinsics).tolist(),
        camera_to_world=observation['observation/calibrations'][n].camera_to_world.xyzw().tolist(),
        depth_scale=observation['observation/calibrations'][n].depth_scale,
        timestamp_offset=observation['observation/calibrations'][n].timestamp_offset) for n in sorted(CAMERAS)}


def public_RGBD_digest(observation):
    h=hashlib.sha256()
    for group in ('observation/images','observation/depths'):
        for n in sorted(CAMERAS):
            a=np.asarray(observation[group][n]);h.update((group+':'+n+':'+str(a.dtype)+':'+str(a.shape)).encode());h.update(a.tobytes())
    return h.hexdigest()


class ApprovedDepthPositionTrackerCacheBindingV64:
    def __init__(self,bridge,tracker,cache):
        self.bridge,self.tracker,self.cache=bridge,tracker,cache
        for relative,expected in APPROVED_COMPONENT_SHA256.items():
            if hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()!=expected:
                raise ValueError('Unapproved V64 position dependency SHA: '+relative)
        self._approved_stats={relative:_file_stat(ROOT/relative) for relative in APPROVED_COMPONENT_SHA256}
        self.binding_id=uuid4().hex;self.reset_episode();self.validate_dependencies()

    def reset_episode(self):
        self.episode_id=uuid4().hex;self.scene_identity=None
        self.latest_result=self.latest_visual_target=None;self.last_frame_timestamp=None
        self._sphere_object=None;self._sphere_digest=None;self.current_raw_RGBD_sha256=None
        self._current_public_observation=None

    def validate_dependencies(self):
        if (type(self.bridge) is not PublicMotionBridge or type(self.tracker) is not CurrentDepthPositionTrackerV64
            or type(self.tracker.tracker) is not RedBoxTracker or type(self.tracker.marker) is not RedGoalMarker
            or type(self.cache) is not PublicTargetFrameCache or self.bridge.tracker is not self.cache
            or self.cache.public_tracker is not self.tracker):
            raise ValueError('Only the exact real V64 tracker/model and same-frame cache capability is accepted')
        t=self.tracker;inner=t.tracker
        if (t.prior is not inner.prior or type(inner.prior) is not BoxPrior or asdict(inner.prior)!=asdict(BoxPrior(minimum_red=100))
            or inner.object_id!='target' or inner.history.maxlen!=5 or inner.complete_geometry is not False or inner.points!=256
            or type(t.config) is not GoalSphereFilterConfig or asdict(t.config)!=asdict(GoalSphereFilterConfig())
            or getattr(t.observe,'__func__',None) is not CurrentDepthPositionTrackerV64.observe
            or getattr(t.validate_refiner,'__func__',None) is not CurrentDepthPositionTrackerV64.validate_refiner
            or getattr(t.model_sources,'__func__',None) is not CurrentDepthPositionTrackerV64.model_sources
            or getattr(t.public_position_evidence,'__func__',None) is not CurrentDepthPositionTrackerV64.public_position_evidence
            or getattr(inner.observe,'__func__',None) is not RedBoxTracker.observe
            or getattr(t.marker.observe,'__func__',None) is not RedGoalMarker.observe):
            raise ValueError('Exact R100 measured-component/knownshape/current real methods required')
        t.validate_refiner()
        offset=np.asarray(json.loads(GOAL_CALIBRATION.read_text())['offset_train_calibrated_m'])
        if (not np.array_equal(t.marker.offset,offset) or not np.array_equal(self.bridge.goal_marker.offset,offset)
            or self.bridge._mock_forecaster is not False or self.bridge._goal_marker_injected is not False
            or self.bridge._tracker_injected is not True or type(self.bridge.goal_marker) is not RedGoalMarker
            or type(self.bridge.forecaster) is not MultiBodyContextDynamics
            or self.bridge.forecaster.checkpoint_sha256!=CHECKPOINT_SHA256):
            raise ValueError('Original frozen real GRU/default public marker required, no forecaster/goal injection')
        for relative,expected in self._approved_stats.items():
            if _file_stat(ROOT/relative)!=expected:
                raise ValueError('V64 dependency changed after full constructor SHA check: '+relative)
        if self._sphere_object is not None and (t.sphere is not self._sphere_object or digest(asdict(t.sphere))!=self._sphere_digest):
            raise ValueError('First real public Sphere cannot be replaced or silently refitted')

    def begin_frame(self,observation):
        _validate_public(observation);self.validate_dependencies();now=observation['timestamp']
        if isinstance(now,(bool,np.bool_)) or not np.isfinite(now):raise ValueError('Actual numeric current sensor time required')
        now=float(now)
        if self.last_frame_timestamp is not None and now<=self.last_frame_timestamp:
            raise ValueError('One unique causal actual RGBD frame required')
        for n in CAMERAS:
            if (type(observation['observation/calibrations'][n]) is not CameraCalibration
                or observation['observation/calibrations'][n].depth_scale!=1.
                or observation['observation/calibrations'][n].timestamp_offset!=0.
                or np.asarray(observation['observation/images'][n]).shape!=(256,256,3)
                or np.asarray(observation['observation/depths'][n]).shape!=(256,256)
                or np.asarray(observation['observation/depths'][n]).dtype!=np.float32):
                raise ValueError('Exact original camera calibration/256 RGB/float32 depth required')
        calibration=calibration_identity(observation);cal_sha=digest(calibration)
        raw_sha=public_RGBD_digest(observation)
        if self.scene_identity is None:
            self.scene_identity=dict(episode_id=self.episode_id,first_public_timestamp=now,
                initial_raw_RGBD_sha256=raw_sha,camera_calibration_sha256=cal_sha,cameras=calibration,
                static_visual_goal_from_first_frame_only=True)
        elif cal_sha!=self.scene_identity['camera_calibration_sha256']:
            raise ValueError('Camera calibration changed; new episode required')
        self.last_frame_timestamp=now;self.current_raw_RGBD_sha256=raw_sha
        self._current_public_observation=observation

    def record_sphere(self):
        sphere=self.tracker.sphere
        if sphere is not None and self._sphere_object is None:
            if self.scene_identity is None or abs(sphere.observed_at-self.scene_identity['first_public_timestamp'])>1e-8:
                raise ValueError('Sphere must originate in the first actual public RGBD frame')
            self._sphere_object=sphere;self._sphere_digest=digest(asdict(sphere))
        self.validate_dependencies()

    def bind_current(self,result,target):
        self.record_sphere();now=float(result['timestamp'])
        trace=self.tracker.last_measurement_metadata
        if (self.last_frame_timestamp is None or abs(now-self.last_frame_timestamp)>1e-8
            or self.tracker._last_observed_timestamp is None or abs(self.tracker._last_observed_timestamp-now)>1e-8
            or not isinstance(trace,dict) or abs(trace.get('timestamp',-1)-now)>1e-8):
            raise ValueError('Only the exact current observed tracker frame can bind')
        if target is not None and (type(target) is not ObjectState or target is not self.tracker.last_current_target
            or abs(target.timestamp-now)>1e-8):raise ValueError('Only the actual same-frame final measured target may bind')
        self.latest_result=result;self.latest_visual_target=target;self.approve(result)

    def approve(self,result):
        self.validate_dependencies()
        if result is not self.latest_result or result.get('goal_aware_binding_version')!='v64':
            raise ValueError('Only the exact current V64-bound result accepted')
        if (result.get('tracker_dependency_injected') is not True
            or result.get('forecaster_is_injected_test_double') is not False
            or result.get('goal_marker_dependency_injected') is not False
            or any(result.get(k) is not False for k in ('Native_inputs_used','future_observations_used','observations_fabricated'))):
            raise ValueError('Truthful narrow tracker injection and public-only causal provenance required')
        trace=result.get('depth_refiner_trace')
        if (not isinstance(trace,dict) or trace.get('timestamp')!=self.tracker._last_observed_timestamp
            or trace.get('model_checkpoint_sha256')!=self.tracker.expected_refiner_sha256
            or trace.get('position_measurement_qualified') is not True
            or trace.get('model_qualified') is not False or trace.get('raw_mask_model_qualified') is not False
            or trace.get('packed_current_input_sha256')!=self.tracker.last_measurement_metadata.get('packed_current_input_sha256')):
            raise ValueError('Current input/model/result trace identity mismatch')
        expected=self.tracker.public_position_evidence()
        if self.tracker.sphere is None or expected.get('producer_ready') is not True:
            expected['strict_refiner_or_initial_fit_unavailable_output_rejected_for_policy']=True
        if expected.get('accepted_history_reset_for_route_change') is True:
            expected.update(motion_history_reset_for_route_change=True,
                motion_history_reset_protocol='accepted_public_measurement_source_change_resets_motion_history_only_v64',
                attachment_probe_maintenance_goal_Sphere_and_cache_reset=False)
        if digest(trace)!=digest(expected):
            raise ValueError('Only the exact same-frame whitelisted public position evidence can bind')
        observation=self._current_public_observation
        if (observation is None or public_RGBD_digest(observation)!=self.current_raw_RGBD_sha256
            or digest(calibration_identity(observation))!=self.scene_identity['camera_calibration_sha256']):
            raise ValueError('Current raw RGBD/calibration mutated after frame binding')
        if self.tracker.sphere is not None and trace.get('producer_ready') is True:
            if _packed_digest(pack_current_public_refiner_inputs(observation,self.tracker.sphere))!=trace.get('packed_current_input_sha256'):
                raise ValueError('Refiner result belongs to a different current public input')
        return True

    def receipt(self):
        return dict(binding_version='current_depth_refiner_public_binding_v64_development',
            binding_id=self.binding_id,episode_id=self.episode_id,tracker_dependency_injected=True,
            allow_test_doubles_enabled=False,forecaster_injection_approved=False,goal_injection_approved=False,
            model_checkpoint_sha256=self.tracker.expected_refiner_sha256,
            position_measurement_qualified=True,model_qualified=False,raw_mask_model_qualified=False,
            position_proposal_thresholds=[.995,.9],adaptive_proposal_changes_are_same_position_source=True,
            model_sources_sha256=self.tracker.model_sources(),approved_source_sha256=deepcopy(APPROVED_COMPONENT_SHA256),
            binding_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            scene_identity=deepcopy(self.scene_identity),current_raw_RGBD_sha256=self.current_raw_RGBD_sha256,
            initial_sphere=None if self._sphere_object is None else asdict(self._sphere_object),initial_sphere_sha256=self._sphere_digest,
            public_input_packer_is_provenance_authenticator=False,invalid_observation_is_unknown_not_prediction=True,
            known_shape_proxy_points=256,known_shape_dimensions_m=[.05,.05,.05],template_is_measured_cloud=False,
            far_consistency_is_development_hypothesis=True,far_consistency_is_physical_or_ownership_certificate=False,
            known_prior_counterexamples=73,closed_loop_Native_development_qualified=False,
            raw_RGBD_rewritten=False,NN_inputs_sphere_shell_filtered=False,
            source_switch_resets_motion_history_only=True,current_measurement_not_suppressed_during_GRU_warmup=True)


class DepthPositionPublicGraspStateAssemblerV64(PublicGraspStateAssembler):
    def __init__(self,binding):
        PublicStateAssembler.__init__(self)
        if type(binding) is not ApprovedDepthPositionTrackerCacheBindingV64:
            raise TypeError('Only explicit exact V64 development capability accepted')
        binding.validate_dependencies();self.binding=binding
