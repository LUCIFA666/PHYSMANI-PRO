"""Causal multi-body RGB-D registration using declared, shared geometry priors.

Runtime input consists only of observation arrays, public camera calibration,
fixed semantic/instance IDs and immutable local geometry. No environment, task,
native object, live goal sensor or hidden attachment state is accepted.

ObjectState requires a rotation even for symmetric objects. Such rotations are
explicit gauge representatives, with observability disclosed in metadata;
callers must never interpret an unobservable sphere rotation as ground truth.
"""
from collections import deque, defaultdict
from dataclasses import dataclass, field
import hashlib
import itertools
import json
from typing import Mapping, Sequence

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares

from ..contracts import ObjectState, Pose, array
from ..perception import CameraCalibration
from .rotation_equivalence import VerifiedFiniteSymmetry, CurrentRotationEquivalence


@dataclass(frozen=True)
class PriorProvenance:
    source_kind: str
    source_reference: str
    frame_definition: str
    shared_with_baseline: bool
    calibration_split: str | None = None
    train_source_groups: tuple[str, ...] = ()
    evidence_sha256: str | None = None

    def __post_init__(self):
        if self.source_kind not in ('declared_cad', 'train_calibrated_geometry', 'train_calibrated_rigid_layout'):
            raise ValueError('geometry must be declared CAD or TRAIN-calibrated geometry')
        if not self.source_reference or not self.frame_definition or self.shared_with_baseline is not True:
            raise ValueError('declare geometry frame, source and equal information permission')
        if self.source_kind.startswith('train_') and (
                self.calibration_split != 'train' or not self.train_source_groups or not self.evidence_sha256):
            raise ValueError('TRAIN calibration requires TRAIN sources and an evidence fingerprint')
        if self.evidence_sha256 is not None and (len(self.evidence_sha256) != 64 or
                any(character not in '0123456789abcdef' for character in self.evidence_sha256)):
            raise ValueError('geometry calibration evidence must have a SHA256 fingerprint')
        if self.train_source_groups and (len(set(self.train_source_groups)) != len(self.train_source_groups)
                or any(not group.startswith('train:') for group in self.train_source_groups)):
            raise ValueError('geometry calibration groups must be unique TRAIN identities')
        if self.calibration_split not in (None, 'train'):
            raise ValueError('VAL/TEST geometry calibration is forbidden')


@dataclass(frozen=True)
class GeometryComponent:
    """Surface samples in the owning body's declared local CAD frame."""
    name: str
    instance_ids: tuple[int, ...]
    points_body: np.ndarray

    def __post_init__(self):
        points = array(self.points_body)
        if points.ndim != 2 or points.shape[1] != 3 or len(points) < 6:
            raise ValueError('a component needs at least six finite CAD surface samples')
        ids = tuple(self.instance_ids)
        if not ids or len(set(ids)) != len(ids) or any(type(value) is not int or not 0 <= value < 2**24 for value in ids):
            raise ValueError('components require unique declared integer instance IDs')
        object.__setattr__(self, 'points_body', points)
        object.__setattr__(self, 'instance_ids', ids)


@dataclass(frozen=True)
class BodyTemplate:
    role_id: str
    components: tuple[GeometryComponent, ...]
    provenance: PriorProvenance
    orientation_observability: str = 'full'
    symmetry_axis: np.ndarray = field(default_factory=lambda: np.array([0., 0., 1.]))
    sphere_radius: float | None = None
    visual_geometry_status: str = 'renderable_components'
    axis_polarity: str = 'undirected'
    finite_rotation_symmetry: VerifiedFiniteSymmetry | None = None
    cup_feature_model: object | None = None

    def __post_init__(self):
        if (not self.role_id or not all(isinstance(c, GeometryComponent) for c in self.components)
                or self.visual_geometry_status not in ('renderable_components', 'native_nonrenderable')
                or bool(self.components) != (self.visual_geometry_status == 'renderable_components')):
            raise ValueError('body templates require a semantic role and explicit geometry components')
        if self.visual_geometry_status == 'native_nonrenderable' and (
                self.provenance.calibration_split != 'train' or not self.provenance.evidence_sha256
                or not self.provenance.train_source_groups or self.sphere_radius is not None):
            raise ValueError('nonrenderable roles require explicit TRAIN evidence and no invented visual surface')
        if len({c.name for c in self.components}) != len(self.components):
            raise ValueError('component names must be unique inside a body')
        if self.orientation_observability not in ('full', 'axis', 'none'):
            raise ValueError('declare full, axis-only, or unobservable rotation')
        if self.axis_polarity not in ('undirected','directed') or (
                self.axis_polarity=='directed' and self.orientation_observability!='axis'):
            raise ValueError('directed polarity applies only to an explicitly observed axis')
        if self.finite_rotation_symmetry is not None:
            symmetry = self.finite_rotation_symmetry
            if (not isinstance(symmetry, VerifiedFiniteSymmetry) or self.role_id != 'ring'
                    or self.orientation_observability != 'none' or self.sphere_radius is not None
                    or symmetry.certificate.get('CAD_asset_sha256') != self.provenance.evidence_sha256
                    or symmetry.certificate.get('body_id') != self.role_id):
                raise ValueError('finite pose quotient requires the same certified TRAIN ring CAD and no full-R claim')
        if self.cup_feature_model is not None:
            from .cup_features import CupCADFeatures
            model = self.cup_feature_model
            if (not isinstance(model, CupCADFeatures) or self.role_id not in ('cup0','cup1','cup2')
                    or self.orientation_observability != 'full'
                    or model.receipt.get('asset_internal_sha256') != self.provenance.evidence_sha256
                    or model.receipt.get('role') != self.role_id):
                raise ValueError('cup feature registration requires its exact TRAIN CAD and explicit per-frame full-R gate')
        object.__setattr__(self, 'components', tuple(self.components))
        axis = array(self.symmetry_axis, (3,))
        if np.linalg.norm(axis) < 1e-8:
            raise ValueError('invalid symmetry axis')
        object.__setattr__(self, 'symmetry_axis', array(axis / np.linalg.norm(axis)))
        if self.sphere_radius is not None and (self.sphere_radius <= 0 or self.orientation_observability != 'none'):
            raise ValueError('sphere fitting requires a positive radius and unobservable orientation')

    def metadata(self):
        provenance = dict(source_kind=self.provenance.source_kind,
            source_reference=self.provenance.source_reference,
            frame_definition=self.provenance.frame_definition,
            shared_with_baseline=self.provenance.shared_with_baseline,
            calibration_split=self.provenance.calibration_split,
            train_source_groups=list(self.provenance.train_source_groups),
            evidence_sha256=self.provenance.evidence_sha256)
        descriptor = dict(role_id=self.role_id, orientation_observability=self.orientation_observability,
            symmetry_axis=self.symmetry_axis.tolist(), sphere_radius=self.sphere_radius,
            visual_geometry_status=self.visual_geometry_status,
            axis_polarity=self.axis_polarity,
            finite_rotation_certificate_sha256=(self.finite_rotation_symmetry.certificate['certificate_sha256']
                if self.finite_rotation_symmetry is not None else None),
            cup_feature_prior=self.cup_feature_model.receipt if self.cup_feature_model is not None else None,
            provenance=provenance, components=[dict(name=component.name,
                instance_ids=list(component.instance_ids), points_shape=list(component.points_body.shape),
                local_surface_sha256=hashlib.sha256(component.points_body.astype('<f8').tobytes()).hexdigest())
                for component in self.components])
        descriptor['template_sha256'] = hashlib.sha256(json.dumps(descriptor,sort_keys=True).encode()).hexdigest()
        return descriptor

    @classmethod
    def from_train_geometry(cls, role_id, components, *, source_episodes, evidence_sha256,
                            source_reference, frame_definition, **kwargs):
        """Interface for an offline TRAIN-only CAD/reconstruction calibrator.

        The caller supplies local surface samples already reconstructed/aligned
        from TRAIN RGB-D and documents that frame. This factory validates
        lineage; it does not claim to have calibrated supplied geometry.
        """
        sources = tuple(source_episodes)
        if not sources or any(row.get('split') != 'train' or not row.get('task')
                              or not row.get('raw_episode_id') for row in sources):
            raise ValueError('template construction accepts TRAIN source identities only')
        groups = tuple(f"train:{row['task']}:{row['raw_episode_id']}" for row in sources)
        if len(groups) != len(set(groups)):
            raise ValueError('duplicate TRAIN template source')
        provenance = PriorProvenance('train_calibrated_geometry', source_reference, frame_definition, True,
                                     'train', groups, evidence_sha256)
        return cls(role_id, tuple(components), provenance, **kwargs)


@dataclass(frozen=True)
class RigidChildConstraint:
    parent_role: str
    child_role: str
    child_to_parent: Pose
    provenance: PriorProvenance

    def __post_init__(self):
        if self.parent_role == self.child_role or self.provenance.source_kind != 'train_calibrated_rigid_layout':
            raise ValueError('rigid layout requires distinct roles and TRAIN validation evidence')


@dataclass(frozen=True)
class RegistrationConfig:
    required_cameras: tuple[str, ...] = ('front', '8', '16', '36')
    minimum_visible_points: int = 12
    maximum_points_per_component: int = 384
    maximum_camera_skew_seconds: float = .035
    voxel_size_m: float = .002
    maximum_correspondence_m: float = .06
    maximum_registration_rms_m: float = .012
    minimum_inlier_fraction: float = .65
    iterations: int = 25
    retained_initializations: int = 4
    history_window: int = 5
    maximum_history_gap_seconds: float = .10
    rotation_ambiguity_radians: float = .3
    ambiguity_rms_slack_m: float = .0005
    free_space_weight: float = 0.
    free_space_tolerance_m: float = .004
    free_space_maximum_gap_m: float = .03

    def __post_init__(self):
        if (min(self.minimum_visible_points, self.maximum_points_per_component, self.iterations,
                self.retained_initializations, self.history_window) < 2
                or min(self.voxel_size_m, self.maximum_correspondence_m, self.maximum_registration_rms_m,
                       self.maximum_history_gap_seconds) <= 0
                or not 0 < self.minimum_inlier_fraction <= 1):
            raise ValueError('invalid registration parameters')
        if not self.required_cameras or len(set(self.required_cameras)) != len(self.required_cameras):
            raise ValueError('declare a unique public camera set')
        if (not np.isfinite([self.free_space_weight,self.free_space_tolerance_m,self.free_space_maximum_gap_m]).all()
                or min(self.free_space_weight,self.free_space_tolerance_m)<0 or self.free_space_maximum_gap_m<=0):
            raise ValueError('finite nonnegative current-depth pose scoring settings required')


def decode_instance_mask(mask):
    value = np.asarray(mask)
    if value.ndim == 2:
        if not np.isfinite(value).all() or not np.all(value == np.floor(value)):
            raise ValueError('integer instance-mask labels required')
        return value.astype(np.int64)
    if value.ndim != 3 or value.shape[2] != 3 or not np.isfinite(value).all():
        raise ValueError('instance masks must be HxW IDs or HxWx3 color-coded RGB')
    if value.dtype.kind == 'f':
        if value.min() < 0 or value.max() > 1 + 1e-6:
            raise ValueError('floating color-coded masks must be normalized to [0,1]')
        value = np.rint(value * 255)
    if value.min() < 0 or value.max() > 255:
        raise ValueError('invalid mask color byte')
    rgb = value.astype(np.int64)
    return rgb[..., 0] + 256 * rgb[..., 1] + 65536 * rgb[..., 2]


def _compose(a, b):
    return Pose(a.apply(b.position), a.rotation @ b.rotation)


def _weighted_fit(source, target, weights):
    weights = weights / weights.sum()
    a, b = weights @ source, weights @ target
    u, _, vt = np.linalg.svd((source-a).T @ (weights[:, None] * (target-b)))
    fix = np.diag([1., 1., np.linalg.det(vt.T @ u.T)])
    rotation = vt.T @ fix @ u.T
    return Pose(b - rotation @ a, rotation)


def _sample(points, maximum, voxel=0.):
    points = np.asarray(points, dtype=float)
    if voxel and len(points):
        _, indices = np.unique(np.floor(points / voxel).astype(np.int64), axis=0, return_index=True)
        points = points[np.sort(indices)]
    if len(points) > maximum:
        points = points[np.linspace(0, len(points)-1, maximum, dtype=int)]
    return points


def _axes(points):
    _, vectors = np.linalg.eigh(np.cov(points.T))
    axes = vectors[:, ::-1]
    axes[:, -1] *= np.linalg.det(axes)
    return axes


def _world_twist(history):
    """CPU counterpart of context_se3.finite_difference_twist (same world convention)."""
    times = np.asarray([r[0] for r in history], dtype=float)
    positions = np.asarray([r[1].position for r in history])
    rotations = np.asarray([r[1].rotation for r in history])
    centered = times-times.mean()
    velocity = np.sum(centered[:, None]*(positions-positions.mean(0)), axis=0) / np.sum(centered**2)
    increments = rotations[1:] @ np.swapaxes(rotations[:-1], -1, -2)
    angular = Rotation.from_matrix(increments.copy()).as_rotvec() / np.diff(times)[:, None]
    return velocity, np.average(angular, axis=0, weights=np.diff(times))


class MultiBodySE3Tracker:
    def __init__(self, templates: Sequence[BodyTemplate], *, task_spec=None,
                 rigid_children: Sequence[RigidChildConstraint] = (), config=RegistrationConfig()):
        self.templates = {template.role_id: template for template in templates}
        if not templates or len(self.templates) != len(templates) or not all(isinstance(t, BodyTemplate) for t in templates):
            raise ValueError('provide unique data-only body templates')
        self.config = config
        ids = [value for template in templates for component in template.components for value in component.instance_ids]
        if len(ids) != len(set(ids)):
            raise ValueError('instance IDs cannot belong to two components or independent roles')
        if task_spec is not None:
            names = {role.name for role in task_spec.roles}
            if set(self.templates)-names or set(self.templates)&set(task_spec.audit_only_goal_roles):
                raise ValueError('templates must use TaskSpec physical role names, excluding native audit-only goals/joints')
        self.constraints = tuple(rigid_children)
        self.parent = {}
        for constraint in self.constraints:
            if not isinstance(constraint, RigidChildConstraint) or {constraint.parent_role, constraint.child_role}-set(self.templates):
                raise ValueError('unknown rigid constraint role')
            if constraint.child_role in self.parent:
                raise ValueError('a rigid child cannot have two parents')
            if task_spec is not None:
                safe = (constraint.parent_role == 'frame' and constraint.child_role.startswith(('peg', 'spoke'))
                        or constraint.parent_role == 'button' and constraint.child_role == 'wrap')
                if not safe:
                    raise ValueError('TaskSpec does not permit a permanent rigid constraint for articulated/attachable roles')
            self.parent[constraint.child_role] = constraint
        self.groups = {}
        for role in self.templates:
            current, transform, seen = role, Pose(np.zeros(3)), set()
            while current in self.parent:
                if current in seen:
                    raise ValueError('cyclic rigid layout')
                seen.add(current)
                edge = self.parent[current]
                transform = _compose(edge.child_to_parent, transform)
                current = edge.parent_role
            self.groups.setdefault(current, {})[role] = transform
        self.reset()

    def reset(self):
        self.history = {role: deque(maxlen=self.config.history_window) for role in self.templates}
        self.axis_history = {role: deque(maxlen=self.config.history_window) for role in self.templates}
        self.last_states = {}
        self._current_views = None
        self._current_timestamp = None
        self.last_observed = {}
        self.last_pose_evidence = {}
        self.last_modes = {}
        self.last_timestamp = None

    def _clouds(self, observation, now):
        for key in ('observation/depths', 'observation/calibrations', 'observation/instance_masks'):
            if key not in observation:
                raise ValueError('missing public observation channel: ' + key)
        depths, calibrations, masks = [observation[key] for key in
            ('observation/depths', 'observation/calibrations', 'observation/instance_masks')]
        if set(depths) != set(calibrations) or set(depths) != set(masks):
            raise ValueError('depth/mask/public calibration camera sets differ')
        if set(depths) != set(self.config.required_cameras):
            raise ValueError('camera set differs from the declared tracker protocol')
        times = observation.get('observation/camera_timestamps', {})
        views = {}
        for camera in depths:
            calibration = calibrations[camera]
            if not isinstance(calibration, CameraCalibration):
                raise TypeError('supply numeric public CameraCalibration objects, never native sensor handles')
            time = float(times.get(camera, now)) + calibration.timestamp_offset
            if not np.isfinite(time) or time > now + 1e-9 or now-time > self.config.maximum_camera_skew_seconds:
                raise ValueError('camera frame is future or stale relative to the current observation')
            depth = np.asarray(depths[camera], dtype=float)
            labels = decode_instance_mask(masks[camera])
            if depth.ndim != 2 or labels.shape != depth.shape:
                raise ValueError('aligned metric HxW depth and instance masks required')
            views[camera] = (calibration, depth, labels)
        clouds, visibility = {}, {}
        for role, template in self.templates.items():
            per_component, counts = {}, {camera: 0 for camera in views}
            for component in template.components:
                chunks = []
                for camera, (calibration, depth, labels) in views.items():
                    mask = np.isin(labels, component.instance_ids)
                    counts[camera] = counts.get(camera, 0) + int(mask.sum())
                    chunks.append(calibration.backproject(depth, mask))
                points = np.concatenate(chunks) if chunks else np.empty((0, 3))
                per_component[component.name] = _sample(points, self.config.maximum_points_per_component,
                                                       self.config.voxel_size_m)
            clouds[role] = per_component
            visibility[role] = dict(mask_pixels_by_camera=counts,
                observed_point_count=sum(len(points) for points in per_component.values()))
        return clouds, visibility, views

    def _registration(self, members, clouds, previous=None):
        feature_roles = [role for role in members if self.templates[role].cup_feature_model is not None]
        if feature_roles:
            if len(members) != 1:
                raise ValueError('cup feature poses must not be permanently bound into another rigid group')
            role = feature_roles[0]
            local = members[role]
            if not np.allclose(local.position,0.,rtol=0.,atol=1e-12) or not np.allclose(
                    local.rotation,np.eye(3),rtol=0.,atol=1e-12):
                raise ValueError('cup feature registration expects its own observed body frame')
            return self._cup_feature_registration(role)
        pieces = []
        for role, local in members.items():
            for component in self.templates[role].components:
                observed = clouds[role][component.name]
                if len(observed) < 3:
                    continue
                model = _sample(local.apply(component.points_body), self.config.maximum_points_per_component)
                pieces.append((role, component.name, model, observed, cKDTree(model)))
        if sum(len(piece[3]) for piece in pieces) < self.config.minimum_visible_points:
            return None
        model = np.concatenate([piece[2] for piece in pieces])
        observed = np.concatenate([piece[3] for piece in pieces])
        def residual(pose):
            return [tree.query(pose.inverse_apply(points))[0] for _, _, _, points, tree in pieces]
        def observed_score(pose):
            return float(np.mean([np.mean(np.minimum(d, self.config.maximum_correspondence_m)**2)
                                  for d in residual(pose)]))
        surface_pieces={role+':'+component.name:local.apply(component.points_body)
            for role,local in members.items() for component in self.templates[role].components}
        def depth_cost(pose):
            from .depth_pose_consistency import current_depth_pose_cost,FreeSpaceConfig
            if self._current_views is None:
                raise ValueError('current-depth pose scoring requires synchronized observation views')
            return current_depth_pose_cost(surface_pieces,pose,self._current_views,
                config=FreeSpaceConfig(tolerance_m=self.config.free_space_tolerance_m,
                    maximum_gap_m=self.config.free_space_maximum_gap_m,weight=self.config.free_space_weight))
        def score(pose):
            return observed_score(pose)+(depth_cost(pose)['cost_m2'] if self.config.free_space_weight>0 else 0.)
        seeds = []
        if previous is not None:
            seeds.append(previous)
        if len(pieces) >= 3:
            a = np.asarray([piece[2].mean(0) for piece in pieces])
            b = np.asarray([piece[3].mean(0) for piece in pieces])
            if np.linalg.matrix_rank(a-a.mean(0), tol=1e-6) >= 2:
                seeds.append(_weighted_fit(a, b, np.ones(len(a))))
        source_axes, target_axes = _axes(model), _axes(observed)
        for permutation in itertools.permutations(range(3)):
            for signs in itertools.product((-1., 1.), repeat=3):
                transform = np.eye(3)[:, permutation] * signs
                rotation = target_axes @ transform @ source_axes.T
                if np.linalg.det(rotation) > 0:
                    seeds.append(Pose(observed.mean(0)-rotation@model.mean(0), rotation))
        seeds.sort(key=score)
        seeds = seeds[:self.config.retained_initializations]
        if previous is not None:
            seeds.append(previous)  # history is not discarded by coarse PCA ranking
        fitted = []
        for pose in seeds:
            for _ in range(self.config.iterations):
                local_points, targets, weights = [], [], []
                for _, _, geometry, points, tree in pieces:
                    distances, index = tree.query(pose.inverse_apply(points))
                    valid = distances < self.config.maximum_correspondence_m
                    if valid.sum() < 3:
                        continue
                    local_points.append(geometry[index[valid]])
                    targets.append(points[valid])
                    weights.append(np.full(valid.sum(), 1./valid.sum()))
                if not local_points or sum(len(value) for value in local_points) < 6:
                    break
                new = _weighted_fit(np.concatenate(local_points), np.concatenate(targets), np.concatenate(weights))
                delta = np.linalg.norm(new.position-pose.position) + np.linalg.norm(
                    Rotation.from_matrix(new.rotation @ pose.rotation.T).as_rotvec())
                pose = new
                if delta < 1e-6:
                    break
            fitted.append((score(pose), pose))
        fitted.sort(key=lambda item: item[0])
        best_score, best = fitted[0]
        finite_quotient = any(self.templates[role].finite_rotation_symmetry is not None for role in members)
        if previous is not None and not finite_quotient:
            near = [item for item in fitted if np.sqrt(item[0]) <= np.sqrt(best_score)+self.config.ambiguity_rms_slack_m]
            _, best = min(near, key=lambda item: np.linalg.norm(
                Rotation.from_matrix(item[1].rotation @ previous.rotation.T).as_rotvec()))
        per_role = defaultdict(list)
        for piece, distances in zip(pieces, residual(best)):
            per_role[piece[0]].extend(distances.tolist())
        quality = {}
        for role, distances in per_role.items():
            distances = np.asarray(distances)
            inliers = distances <= self.config.maximum_registration_rms_m * 2
            rms = float(np.sqrt(np.mean(distances[inliers]**2))) if inliers.any() else float('inf')
            quality[role] = dict(rms_m=rms, inlier_fraction=float(inliers.mean()),
                valid=rms <= self.config.maximum_registration_rms_m and inliers.mean() >= self.config.minimum_inlier_fraction)
        ambiguous = any(np.sqrt(value) <= np.sqrt(best_score)+self.config.ambiguity_rms_slack_m
                        and np.linalg.norm(Rotation.from_matrix(pose.rotation @ best.rotation.T).as_rotvec())
                        > self.config.rotation_ambiguity_radians for value, pose in fitted)
        from .axis_evidence import near_optimal_axis_evidence
        axis_evidence={role:near_optimal_axis_evidence(
            [(float(np.sqrt(value)),pose.rotation@local.rotation) for value,pose in fitted],
            best.rotation@local.rotation,self.templates[role].symmetry_axis,
            directed=self.templates[role].axis_polarity=='directed',best_rms_m=float(np.sqrt(best_score)),
            rms_slack_m=self.config.ambiguity_rms_slack_m,ambiguity_radians=self.config.rotation_ambiguity_radians)
            for role,local in members.items() if self.templates[role].orientation_observability=='axis'}
        return dict(pose=best, rms_m=float(np.sqrt(observed_score(best))), per_role=quality,
                    valid=all(row['valid'] for row in quality.values()), rotation_ambiguous=bool(ambiguous),
                    axis_evidence_by_role=axis_evidence,registration_selection_cost_m2=float(score(best)),
                    retained_root_candidate_poses=tuple(pose for _,pose in fitted),
                    retained_candidate_rms_m=tuple(float(np.sqrt(value)) for value,_ in fitted),
                    current_depth_pose_cost=depth_cost(best) if self.config.free_space_weight>0 else None)

    def _cup_feature_registration(self, role):
        """Current semantic RGB-D features, with no native/teacher pose input."""
        from .cup_features import estimate_cup_feature_pose
        template = self.templates[role]
        if self._current_views is None:
            raise ValueError('cup feature registration requires current validated observation views')
        ids = tuple(identity for component in template.components for identity in component.instance_ids)
        points = np.concatenate([calibration.backproject(depth,np.isin(labels,ids))
            for calibration,depth,labels in self._current_views.values()])
        try:
            result = estimate_cup_feature_pose(points,template.cup_feature_model,self._current_views,
                reference_time=self._current_timestamp)
        except (ValueError,np.linalg.LinAlgError,FloatingPointError):
            return None
        if result.get('pose_xyzw') is None:
            return None
        pose = Pose.from_xyzw(result['pose_xyzw'])
        local_surface = np.concatenate([component.points_body for component in template.components])
        distances = cKDTree(local_surface).query(pose.inverse_apply(points))[0]
        inliers = distances <= 2*self.config.maximum_registration_rms_m
        rms = float(np.sqrt(np.mean(distances[inliers]**2))) if inliers.any() else float('inf')
        fraction = float(inliers.mean())
        valid = bool(rms<=self.config.maximum_registration_rms_m and fraction>=self.config.minimum_inlier_fraction)
        quality = dict(rms_m=rms,inlier_fraction=fraction,valid=valid)
        best = result['candidates'][0]
        diagnostic = {key:result.get(key) for key in ('version','status','ambiguous','candidate_count',
            'candidate_cost_gap_m2','current_point_count','outside_handle_points','readiness_is_not_accuracy_certification')}
        diagnostic.update(cylinder_rms_m=result.get('cylinder',{}).get('rms_m'),
            raw_current_points=len(points),observed_surface_rms_m=rms,observed_surface_inlier_fraction=fraction,
            full_rotation_ready=bool(valid and not result['ambiguous']),GT_or_native_pose_used=False,
            parameter_search_used=False,pose_estimator='TRAIN_CAD_cylinder_handle_mouth')
        return dict(pose=pose,rms_m=rms,per_role={role:quality},valid=valid,
            rotation_ambiguous=bool(result['ambiguous']),axis_evidence_by_role={},
            registration_selection_cost_m2=float(best['cost_m2']),
            current_depth_pose_cost=best.get('current_depth_pose_cost'),cup_feature_registration=diagnostic)

    def _symmetry_pose(self, role, pose, points, previous):
        template = self.templates[role]
        if template.sphere_radius is not None:
            radius = template.sphere_radius
            fit = least_squares(lambda center: np.linalg.norm(points-center, axis=1)-radius,
                                pose.position, loss='soft_l1', f_scale=.003)
            if not fit.success or not np.isfinite(fit.x).all():
                raise ValueError('sphere surface registration failed')
            pose = Pose(fit.x, np.eye(3))
        if template.orientation_observability == 'none' and template.finite_rotation_symmetry is None:
            return Pose(pose.position, np.eye(3))
        if template.orientation_observability == 'axis':
            axis = pose.rotation @ template.symmetry_axis
            reference = previous.rotation if previous is not None else np.eye(3)
            old_axis = reference @ template.symmetry_axis
            if template.axis_polarity=='undirected' and np.dot(axis, old_axis) < 0:
                axis = -axis  # undirected symmetry axis; polarity is a gauge
            cross = np.cross(old_axis, axis)
            magnitude = np.linalg.norm(cross)
            if magnitude>1e-10:
                correction=Rotation.from_rotvec(cross/magnitude*np.arctan2(magnitude,np.dot(old_axis,axis))).as_matrix()
            elif np.dot(old_axis,axis)<0:
                basis=np.eye(3)[int(np.argmin(np.abs(old_axis)))];normal=np.cross(old_axis,basis)
                correction=Rotation.from_rotvec(np.pi*normal/np.linalg.norm(normal)).as_matrix()
            else:correction=np.eye(3)
            return Pose(pose.position, correction @ reference)
        return pose

    def _occlusion(self, role, views):
        if self.templates[role].visual_geometry_status == 'native_nonrenderable':
            return 'TRAIN_native_nonrenderable_no_direct_visual_surface'
        previous = self.last_states.get(role)
        if previous is None:
            return 'not_initialized_no_visible_geometry'
        points = previous.pose.apply(np.concatenate([c.points_body for c in self.templates[role].components]))
        projected, blocked = 0, 0
        for calibration, depth, _ in views.values():
            camera_points = calibration.camera_to_world.inverse_apply(points)
            front = camera_points[:, 2] > 1e-6
            camera_points = camera_points[front]
            if not len(camera_points):
                continue
            uv = camera_points @ calibration.intrinsics.T
            uv = np.rint(uv[:, :2]/uv[:, 2:]).astype(int)
            inside = (uv[:, 0] >= 0) & (uv[:, 0] < depth.shape[1]) & (uv[:, 1] >= 0) & (uv[:, 1] < depth.shape[0])
            uv, camera_points = uv[inside], camera_points[inside]
            measured = depth[uv[:, 1], uv[:, 0]] * calibration.depth_scale
            valid = np.isfinite(measured) & (measured > 0)
            projected += len(uv)
            blocked += int(np.sum(valid & (measured < camera_points[:, 2]-.015)))
        if not projected:
            return 'past_pose_projects_outside_camera_frusta'
        return 'past_pose_projection_occluded' if blocked > .5 * projected else 'mask_or_depth_missing_or_sparse'

    def observe(self, observation: Mapping, *, modes: Mapping[str, str] | None = None):
        """Return ``(ObjectState list, metadata)`` for current/past sensor evidence.

        ``modes`` is controller-owned touch/action/proprioception evidence. It
        must never be read from native task latches. Mode changes clear twist
        history; missing modes are explicitly reported as unknown.
        """
        now = float(observation['timestamp'])
        if not np.isfinite(now) or self.last_timestamp is not None and now <= self.last_timestamp:
            raise ValueError('reset tracker at episode boundaries; observation timestamps must increase')
        modes = dict(modes or {})
        if set(modes)-set(self.templates) or any(mode not in ('external','attached','released','unknown') for mode in modes.values()):
            raise ValueError('unknown role or controller-observed motion mode')
        clouds, visibility, views = self._clouds(observation, now)
        self._current_views = views
        self._current_timestamp = now
        registrations, group_diagnostics = {}, []
        for root, members in self.groups.items():
            old = self.last_states.get(root)
            initial = old.pose if old is not None else None
            group = self._registration(members, clouds, initial)
            group_valid = group is not None and group['valid']
            group_diagnostics.append(dict(root=root, members=list(members),
                registration_valid=group_valid, rigidity_consistency_tested=len(group['per_role']) >= 2 if group else False,
                failure='insufficient_geometry_or_constraint_conflict' if not group_valid else None,
                TRAIN_constraints=[dict(parent=c.parent_role,child=c.child_role,evidence_sha256=c.provenance.evidence_sha256)
                                   for c in self.constraints if c.child_role in members]))
            if group_valid:
                for role, transform in members.items():
                    registrations[role] = dict(group, pose=_compose(group['pose'], transform),
                        directly_fitted=role in group['per_role'], rigid_root=root)
            else:
                # A detached/articulated child never forces an inconsistent
                # root pose. Track members independently when the prior fails.
                for role in members:
                    old = self.last_states.get(role)
                    fitted = self._registration({role: Pose(np.zeros(3))}, clouds, old.pose if old else None)
                    if fitted is not None and fitted['valid']:
                        registrations[role] = dict(fitted, directly_fitted=True, rigid_root=None)
        objects, body_diagnostics = [], {}
        for role, template in self.templates.items():
            points = np.concatenate(list(clouds[role].values())) if clouds[role] else np.empty((0, 3))
            direct = len(points) >= self.config.minimum_visible_points
            fit = registrations.get(role)
            previous = self.last_states.get(role)
            mode = modes.get(role, 'unknown')
            history = self.history[role]
            axis_history = self.axis_history[role]
            if self.last_modes.get(role) != mode or history and now-history[-1][0] > self.config.maximum_history_gap_seconds:
                history.clear()
                axis_history.clear()
            self.last_modes[role] = mode
            rotation_observed = bool(fit and template.orientation_observability == 'full' and not fit['rotation_ambiguous'])
            axis_evidence=fit.get('axis_evidence_by_role',{}).get(role) if fit else None
            diagnostic = dict(**visibility[role], directly_visible=direct,
                registration_valid=fit is not None, orientation_observable=rotation_observed,
                orientation_observability=template.orientation_observability,
                visual_geometry_status=template.visual_geometry_status,
                axis_polarity=template.axis_polarity,
                axis_observable=bool(axis_evidence and axis_evidence['axis_observable']),
                axis_evidence=axis_evidence,
                current_depth_pose_cost=fit.get('current_depth_pose_cost') if fit else None,
                cup_feature_registration=fit.get('cup_feature_registration') if fit else None,
                registration_selection_cost_m2=fit.get('registration_selection_cost_m2') if fit else None,
                rotation_gauge='measured_CAD_frame' if rotation_observed else 'unobservable_or_ambiguous_gauge_representative',
                pose_source='current_group_RGBD_and_declared_rigid_layout' if fit and not fit.get('directly_fitted')
                    else 'current_RGBD_CAD_registration' if fit else 'past_evidence_only',
                motion_mode=mode, mode_provenance='caller_controller_touch_action_proprioception' if role in modes else 'not_observed',
                rigid_root=fit.get('rigid_root') if fit else None,
                template_sha256=template.metadata()['template_sha256'],
                geometry_prior_source=template.provenance.source_reference,
                geometry_prior_shared_with_baseline=True)
            if fit is None:
                diagnostic.update(visible=False, twist_observed=False,
                    occlusion=self._occlusion(role, views), pose_observed_until=self.last_observed.get(role),
                    pose_evidence_until=self.last_pose_evidence.get(role))
                history.clear()
                axis_history.clear()
                if previous is not None:
                    age = now-self.last_pose_evidence[role]
                    objects.append(ObjectState(role, now, previous.pose, previous.points_world,
                        previous.linear_velocity, previous.angular_velocity,
                        float(previous.confidence*np.exp(-age/.15)), visible=False))
                body_diagnostics[role] = diagnostic
                continue
            pose = fit['pose']
            if direct:
                pose = self._symmetry_pose(role, pose, points, previous.pose if previous else None)
            elif template.orientation_observability == 'none' and template.finite_rotation_symmetry is None:
                pose = Pose(pose.position, np.eye(3))
            if template.finite_rotation_symmetry is not None:
                root = fit.get('rigid_root')
                body_in_root = self.groups[root][role] if root is not None else Pose(np.zeros(3))
                evidence = CurrentRotationEquivalence.from_root_hypotheses(template.finite_rotation_symmetry,
                    body_id=role, timestamp=now, root_candidate_poses=fit['retained_root_candidate_poses'],
                    candidate_rms_m=fit['retained_candidate_rms_m'], body_in_root=body_in_root,
                    registration_valid=True, directly_visible=bool(direct), all_retained_hypotheses_provided=True,
                    representative_body_pose=pose, rms_slack_m=self.config.ambiguity_rms_slack_m,
                    maximum_registration_rms_m=self.config.maximum_registration_rms_m)
                diagnostic.update(rotation_equivalence_evidence=evidence,
                    rotation_equivalence_receipt=evidence.receipt(),
                    orientation_equivalence_observable=evidence.observable,
                    rotation_gauge='current_RGBD_certified_D4_representative; full orientation remains unobserved')
            if direct:
                history.append((now, pose))
                self.last_observed[role] = now
            else:
                history.clear()
            velocity, angular = _world_twist(history) if len(history) > 1 else (np.zeros(3), np.zeros(3))
            if template.orientation_observability == 'full':
                if direct and rotation_observed:
                    axis_history.append((now,pose))
                else:
                    axis_history.clear()
                angular = _world_twist(axis_history)[1] if len(axis_history)>1 else np.zeros(3)
            elif template.orientation_observability == 'none':
                angular = np.zeros(3)
            elif template.orientation_observability == 'axis':
                if direct and diagnostic['axis_observable']:
                    axis_history.append((now, pose))
                else:
                    axis_history.clear()
                # Translation remains measurable when the axis is ambiguous.
                # Angular increments may only join consecutive valid axes.
                angular = (_world_twist(axis_history)[1] if len(axis_history)>1 else np.zeros(3))
                axis = pose.rotation @ template.symmetry_axis
                angular = angular - axis*np.dot(angular,axis)
                if not diagnostic['axis_observable']:
                    angular=np.zeros(3)
            quality = fit['per_role'].get(role, dict(rms_m=fit['rms_m'],inlier_fraction=1.))
            confidence = float(np.clip(quality['inlier_fraction']*np.exp(-quality['rms_m']/self.config.maximum_registration_rms_m),0.,1.))
            if len(points) >= 3:
                state_points, points_source = points, 'current_RGBD_masks'
            elif template.components:
                state_points = pose.apply(np.concatenate([c.points_body for c in template.components]))
                points_source = 'rigid_prior_geometry_transformed_by_observed_group'
            else:
                # ObjectState carries supporting observed points, not a fake
                # child surface. A nonrenderable role can only arrive here
                # through its explicitly evidenced rigid group.
                members = self.groups[fit['rigid_root']]
                support = [cloud for member in members for cloud in clouds[member].values() if len(cloud)]
                state_points = np.concatenate(support) if support else np.empty((0, 3))
                if len(state_points) < 3:
                    raise ValueError('nonrenderable role requires current RGB-D rigid-group support')
                points_source = 'current_rigid_group_RGBD_support_not_child_surface'
            state = ObjectState(role, now, pose, state_points,
                                velocity, angular, confidence, visible=direct)
            objects.append(state)
            self.last_states[role] = state
            self.last_pose_evidence[role] = now
            diagnostic.update(visible=direct, pose_observed_until=now if direct else self.last_observed.get(role),
                pose_evidence_until=now,
                occlusion=None if direct else self._occlusion(role,views),
                twist_observed=direct and len(history)>1, rotation_ambiguous=fit['rotation_ambiguous'],
                angular_velocity_observability='none' if template.orientation_observability=='none' else
                    'perpendicular_to_symmetry_axis_only' if template.orientation_observability=='axis' else
                    'world_twist_under_causal_rotation_gauge',
                registration_rms_m=quality['rms_m'], inlier_fraction=quality['inlier_fraction'],
                points_source=points_source)
            body_diagnostics[role] = diagnostic
        self.last_timestamp = now
        return objects, dict(bodies=body_diagnostics, rigid_groups=group_diagnostics,
            template_priors={role:template.metadata() for role,template in self.templates.items()},
            input_information='current/past RGB-D, current declared instance masks, public camera calibration and shared CAD/TRAIN geometry priors',
            camera_names=list(views), RGB_color_used=False, native_state_or_goal_sensor_poses_used=False,
            live_expert_or_hidden_attachment_state_used=False,
            twist_frame='world; angular increments log(R_current @ R_previous.T)/dt',
            symmetry_warning='Gauge rotations for sphere/axis/symmetry-ambiguous bodies are not observed ground-truth orientation.')
