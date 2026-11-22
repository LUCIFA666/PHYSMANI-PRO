"""Explicit v42 capability for one frozen goal-aware public RGBD tracker.

No blanket test-double approval. Scene identity is public initial RGBD plus
signed camera calibration and one immutable initial visual sphere fit. Source
and episode identities are provenance, never task/speed/model input features.
"""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from uuid import uuid4
import hashlib,json
import numpy as np
from ..benchmark_perception import RedBoxTracker,BoxPrior
from ..goal_perception import RedGoalMarker
from ..contracts import ObjectState
from .goal_aware_perception import GoalAwareRedBoxTracker,GoalSphereFilterConfig,_validate_public
from .public_attachment import PublicTargetFrameCache
from .public_motion_bridge import PublicMotionBridge,MultiBodyContextDynamics,CHECKPOINT_SHA256,GOAL_CALIBRATION,CAMERAS
from .public_grasp_state import PublicGraspStateAssembler
from .public_state import PublicStateAssembler

ROOT=Path(__file__).resolve().parents[2]
APPROVED_COMPONENT_SHA256={
 'our_mani/local_mpc_control_20261103/goal_aware_perception.py':'e2105eb7b80269324ff77c8ca28a3cac1a480cf006c1c3b1324867179af05f8c',
 'our_mani/benchmark_perception.py':'e1eb4cff08f96117df50f22672362eb7b4bbf8274eb7b25a8bb2b2b1a863adf2',
 'our_mani/goal_perception.py':'f542fe8671e60bae2d41bb2bc64b85e2ac2147ec1b54879ad8eea8f43b37c1ed',
 'our_mani/local_mpc_control_20261103/public_attachment.py':'8cbea0c0c5c2db65022767e13960180a4025540f81f8b2f788b000461a746208',
 'our_mani/local_mpc_control_20261103/public_motion_bridge.py':'e00868f68cbfcae2facb4b5b215202e7986b19c57b7b50ed185f97a6df89b6de',
 'our_mani/multitask/context_se3.py':'8f882d93b63c9cd5bb7112f9c89bda0ec10fce8d07224364032e57529be33023',
 'artifacts/framework/local_policy/goal_marker_calibration.json':'3973330d5de75f25d049fdcebc433d3f826926f37d4db1e790195b35c55bbc00'}
RAW_TRACKER_KEYS=frozenset(('backend','measured_points','visible_views','geometry_completion','simulator_pose_used','future_observations_used','known_dimensions_m'))


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False,
        default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x.item() if isinstance(x,np.generic) else None).encode()).hexdigest()


def calibration_identity(observation):
    return {name:dict(intrinsics=np.asarray(observation['observation/calibrations'][name].intrinsics).tolist(),
        camera_to_world=observation['observation/calibrations'][name].camera_to_world.xyzw().tolist()) for name in sorted(CAMERAS)}


def public_RGBD_digest(observation):
    h=hashlib.sha256()
    for group in ('observation/images','observation/depths'):
        for name in sorted(CAMERAS):
            value=np.asarray(observation[group][name]);h.update((group+':'+name+':'+str(value.dtype)+':'+str(value.shape)).encode());h.update(value.tobytes(order='C'))
    return h.hexdigest()


class ApprovedGoalAwareTrackerCacheBinding:
    def __init__(self,bridge,tracker,cache):
        self.bridge,self.tracker,self.cache=bridge,tracker,cache
        self.binding_id=uuid4().hex;self.reset_episode();self.validate_dependencies()
    def reset_episode(self):
        self.episode_id=uuid4().hex;self.scene_identity=None;self.latest_result=None;self.latest_visual_target=None
        self.last_frame_timestamp=None;self._sphere_object=None;self._sphere_digest=None
    def validate_dependencies(self):
        if (type(self.bridge) is not PublicMotionBridge or type(self.tracker) is not GoalAwareRedBoxTracker
                or type(self.tracker.tracker) is not RedBoxTracker or type(self.tracker.marker) is not RedGoalMarker
                or type(self.cache) is not PublicTargetFrameCache or self.bridge.tracker is not self.cache
                or self.cache.public_tracker is not self.tracker):raise ValueError('only explicitly approved exact real v42 tracker/components/cache are accepted')
        inner=self.tracker.tracker
        if (self.tracker.prior is not inner.prior or asdict(inner.prior)!=asdict(BoxPrior()) or inner.object_id!='target' or inner.history.maxlen!=5
                or inner.complete_geometry is not True or inner.points!=256
                or type(self.tracker.config) is not GoalSphereFilterConfig or asdict(self.tracker.config)!=asdict(GoalSphereFilterConfig())):
            raise ValueError('unchanged declared box prior and sphere-shell configuration required')
        offset=np.asarray(json.loads(GOAL_CALIBRATION.read_text())['offset_train_calibrated_m'])
        if (not np.array_equal(self.tracker.marker.offset,offset) or not np.array_equal(self.bridge.goal_marker.offset,offset)
                or self.bridge._mock_forecaster is not False or self.bridge._goal_marker_injected is not False
                or self.bridge._tracker_injected is not True or type(self.bridge.goal_marker) is not RedGoalMarker
                or type(self.bridge.forecaster) is not MultiBodyContextDynamics or self.bridge.forecaster.checkpoint_sha256!=CHECKPOINT_SHA256):
            raise ValueError('real frozen forecaster and default public calibrated marker only; injection exceptions are narrow')
        for relative,expected in APPROVED_COMPONENT_SHA256.items():
            if hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()!=expected:raise ValueError('unapproved component source SHA: '+relative)
        sphere=self.tracker.sphere
        if self._sphere_object is not None and (sphere is not self._sphere_object or digest(asdict(sphere))!=self._sphere_digest):
            raise ValueError('initial public sphere identity changed within one episode; silent refit forbidden')
    def begin_frame(self,observation):
        _validate_public(observation);self.validate_dependencies();now=float(observation['timestamp'])
        if self.last_frame_timestamp is not None and now<=self.last_frame_timestamp:raise ValueError('one actual causal RGBD frame required')
        calibration=calibration_identity(observation);calibration_sha=digest(calibration)
        if self.scene_identity is None:
            self.scene_identity=dict(episode_id=self.episode_id,first_public_timestamp=now,
                initial_raw_RGBD_sha256=public_RGBD_digest(observation),camera_calibration_sha256=calibration_sha,
                cameras=calibration,static_visual_goal_from_first_frame_only=True)
        elif calibration_sha!=self.scene_identity['camera_calibration_sha256']:
            raise ValueError('camera/scene calibration identity changed; new episode binding required')
        self.last_frame_timestamp=now
    def record_sphere(self):
        sphere=self.tracker.sphere
        if sphere is not None and self._sphere_object is None:
            if self.scene_identity is None or abs(sphere.observed_at-self.scene_identity['first_public_timestamp'])>1e-8:
                raise ValueError('sphere must originate in exact first public frame')
            self._sphere_object=sphere;self._sphere_digest=digest(asdict(sphere))
        self.validate_dependencies()
    def bind_current(self,result,target):
        self.record_sphere()
        if target is not None and (not isinstance(target,ObjectState) or abs(target.timestamp-result['timestamp'])>1e-8):raise ValueError('same real visual ObjectState or None only')
        self.latest_result=result;self.latest_visual_target=target;self.approve(result)
    def approve(self,result):
        self.validate_dependencies()
        if result is not self.latest_result or result.get('goal_aware_binding_version')!='v42':raise ValueError('only current exact v42 bound result accepted')
        if (result.get('tracker_dependency_injected') is not True or result.get('forecaster_is_injected_test_double') is not False
                or result.get('goal_marker_dependency_injected') is not False or any(result.get(k) is not False for k in
                    ('Native_inputs_used','future_observations_used','observations_fabricated'))):raise ValueError('truthful narrow injected tracker provenance required')
    def receipt(self):
        return dict(binding_version='goal_aware_public_binding_v42',binding_id=self.binding_id,episode_id=self.episode_id,
            tracker_dependency_injected=True,approved_dependency='exact_GoalAwareRedBoxTracker_with_exact_original_RedBoxTracker_and_RedGoalMarker_in_exact_sameframe_cache',
            allow_test_doubles_enabled=False,forecaster_injection_approved=False,goal_injection_approved=False,
            filter_config=asdict(self.tracker.config),strict_initial_fit_required=True,unfiltered_fallback_used_for_policy=False,
            scene_identity=deepcopy(self.scene_identity),initial_sphere=None if self._sphere_object is None else asdict(self._sphere_object),
            initial_sphere_sha256=self._sphere_digest,approved_source_sha256=deepcopy(APPROVED_COMPONENT_SHA256),
            binding_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            invalid_observation_is_unknown_not_prediction=True,raw_RGBD_rewritten=False)


class GoalAwarePublicGraspStateAssembler(PublicGraspStateAssembler):
    """Only constructor capability changes; all strict public input checks stay."""
    def __init__(self,binding):
        PublicStateAssembler.__init__(self)
        if type(binding) is not ApprovedGoalAwareTrackerCacheBinding:raise TypeError('explicit exact v42 binding required')
        self.binding=binding
