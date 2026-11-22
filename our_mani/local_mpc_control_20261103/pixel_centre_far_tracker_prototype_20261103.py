"""Offline-only pixel-centre R100 far component research.

The original public sphere remains fixed. Both shell suppression and the
unchanged RedBoxTracker see the same private half-pixel calibration. No
existing calibration, tracker, model, certificate, binding or consumer is
modified. This class is deliberately outside every approved live exact type.
"""
from collections import deque
from copy import deepcopy

import numpy as np

from ..benchmark_perception import BoxPrior, RedBoxTracker, box_surface
from ..contracts import ObjectState, Pose
from ..perception import CameraCalibration
from ..local_mpc_20261103.refiner_far_consistency_v60 import component_far_consistency
from .goal_aware_perception import VisualGoalSphere, _validate_public, filter_observed_sphere_surface

VERSION = 'offline_pixel_centre_far_R100_consistency_research_20261103_v1'
PUBLIC_EVIDENCE_KEYS = frozenset((
    'version', 'timestamp', 'current_measurement_available', 'reason', 'current_component_points',
    'visible_views', 'pixel_centre_offset', 'initial_sphere_fixed', 'sphere_refitted',
    'orientation_observed', 'template_is_measured_cloud', 'qualified',
    'position_measurement_qualified', 'raw_mask_model_qualified',
    'semantic_ownership_verified', 'physical_or_statistical_certificate',
    'far_consistency_is_development_hypothesis', 'live_consumer_admission'))


class PixelCentreFarCalibration(CameraCalibration):
    """Only arithmetic delta: integer u/v -> OpenGL sample u+.5/v+.5."""
    def backproject(self, depth, mask):
        z = np.asarray(depth, dtype=float)*self.depth_scale
        if z.ndim != 2 or np.asarray(mask).shape != z.shape:
            raise ValueError('depth and segmentation must be aligned H x W images')
        y, x = np.nonzero(np.asarray(mask, dtype=bool) & np.isfinite(z) & (z > 0))
        uv = np.c_[x+.5, y+.5, np.ones(len(x))]
        rays = uv @ np.linalg.inv(self.intrinsics).T
        return self.camera_to_world.apply(rays*z[y, x, None])


def project_public_far_evidence(metadata):
    """Positive whitelist: no private component, shell, proof or source tree."""
    if not isinstance(metadata, dict) or set(metadata) != PUBLIC_EVIDENCE_KEYS:
        raise ValueError('Exact small research far summary required')
    if (metadata['version'] != VERSION or metadata['live_consumer_admission'] is not False
        or any(metadata[k] is not False for k in (
            'qualified', 'position_measurement_qualified', 'raw_mask_model_qualified',
            'semantic_ownership_verified', 'physical_or_statistical_certificate', 'sphere_refitted'))):
        raise ValueError('Prototype cannot imply certification or live admission')
    return deepcopy(metadata)


class PixelCentreFarTrackerPrototype:
    """Far-only causal replay, no NN, prediction, attachment or live adapter.

    observe accepts the exact original four public RGBD/calibration/time keys.
    Returned target uses the unchanged identity/50 mm template prior; complete
    actual current component is kept separately in last_measured_cloud. An
    unavailable frame clears only this prototype's own causal motion history.
    """
    def __init__(self, initial_public_sphere):
        if type(initial_public_sphere) is not VisualGoalSphere:
            raise TypeError('Exact previously observed public sphere required')
        self.sphere = initial_public_sphere
        self.prior = BoxPrior(minimum_red=100)
        self._raw_tracker = RedBoxTracker(prior=self.prior, complete_geometry=False, history=5, points=256)
        self.history = deque(maxlen=5)
        self.last_velocity = np.zeros(3)
        self.last_timestamp = None
        self.last_current_target = None
        self.last_measured_cloud = None
        self.last_private = None
        self.last_public = None
        self.last_filtered_observation = None
        self.last_history_before = []
        self.last_velocity_before = np.zeros(3)

    def reset(self):
        self.history.clear()
        self.last_velocity = np.zeros(3)

    def public_far_evidence(self):
        return project_public_far_evidence(self.last_public)

    def _unknown(self, reason):
        self.last_public.update(current_measurement_available=False, reason=reason)
        self.last_current_target = None
        self.last_measured_cloud = None
        self.reset()
        return None, self.public_far_evidence()

    def observe(self, observation):
        _validate_public(observation)
        now = float(observation['timestamp'])
        if self.last_timestamp is not None and now <= self.last_timestamp:
            raise ValueError('One unique causal public frame required')
        gap = None if self.last_timestamp is None else now-self.last_timestamp
        self.last_timestamp = now
        self.last_current_target = None
        self.last_measured_cloud = None
        self.last_filtered_observation = None
        self.last_history_before = [(t, p.copy()) for t, p in self.history]
        self.last_velocity_before = self.last_velocity.copy()
        self.last_private = dict(timestamp=now, raw_component=None, raw_metadata=None,
            sphere_filter=None, far_consistency=None, raw_error=None,
            original_private_calibration_modified=False)
        self.last_public = dict(version=VERSION, timestamp=now, current_measurement_available=False,
            reason=None, current_component_points=0, visible_views=0, pixel_centre_offset=.5,
            initial_sphere_fixed=True, sphere_refitted=False, orientation_observed=False,
            template_is_measured_cloud=False, qualified=False, position_measurement_qualified=False,
            raw_mask_model_qualified=False, semantic_ownership_verified=False,
            physical_or_statistical_certificate=False,
            far_consistency_is_development_hypothesis=True, live_consumer_admission=False)
        if gap is not None and (gap > .075 or abs(gap-.05) > 1e-6):
            return self._unknown('non50ms_or_stale_actual_frame_no_interpolation')
        # The one private calibration convention reaches BOTH original
        # functions. Original arrays/calibrations remain untouched.
        private = dict(observation)
        private['observation/calibrations'] = {
            name: PixelCentreFarCalibration(c.intrinsics, c.camera_to_world, c.depth_scale, c.timestamp_offset)
            for name, c in observation['observation/calibrations'].items()}
        raw = meta = None
        try:
            filtered, trace = filter_observed_sphere_surface(private, self.sphere, self.prior)
            self.last_filtered_observation = filtered
            self.last_private['sphere_filter'] = trace
            self._raw_tracker.history.clear()
            self._raw_tracker.history.extend(self.last_history_before)
            self._raw_tracker.last_velocity = self.last_velocity_before.copy()
            raw, meta = self._raw_tracker.observe(filtered)
        except (ValueError, RuntimeError) as error:
            self.last_private['raw_error'] = type(error).__name__+': '+str(error)
        if raw is None:
            return self._unknown('no_current_far_component')
        if meta['geometry_completion'] is not False:
            raise ValueError('Complete current component, never template, required for far consistency')
        self.last_private.update(raw_component=raw, raw_metadata=meta)
        proof = component_far_consistency(raw.points_world, self.sphere.center)
        self.last_private['far_consistency'] = proof
        self.last_public.update(current_component_points=len(raw.points_world), visible_views=meta['visible_views'])
        if not proof['far_consistent'] or raw.confidence < .5:
            return self._unknown('current_component_without_far_consistency')
        # Original outer _accept LS velocity, history length and template math.
        center = raw.pose.position
        samples = list(self.history)+[(now, center)]
        velocity = np.zeros(3)
        if len(samples) > 1:
            times = np.array([p[0] for p in samples]); times -= times.mean()
            positions = np.array([p[1] for p in samples])
            velocity = (times[:, None]*(positions-positions.mean(0))).sum(0)/(times@times)
        self.history.append((now, center.copy())); self.last_velocity = velocity.copy()
        pose = Pose(center)
        self.last_current_target = ObjectState('target', now, pose, box_surface(pose, self.prior.dimensions, 256),
            velocity, np.zeros(3), raw.confidence)
        self.last_measured_cloud = raw.points_world.copy()
        self.last_public.update(current_measurement_available=True, reason='current_far_consistent_development_hypothesis')
        return self.last_current_target, self.public_far_evidence()
