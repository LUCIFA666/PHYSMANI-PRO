"""Explicit held-target collision semantics for the two legacy block tasks.

The original RGBD tracker result/history is returned unchanged. A first-contact
measurement may establish a support-consistent rigid transform using the same
raw selected cloud and known box prior. This additional quality gate uses the
shared RegistrationConfig defaults; the legacy tracker had no such RMS gate.
Only the gripper subtree is permitted to touch a qualified held block. Arm and
held-block collisions with every immutable static obstacle remain checked.
"""
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import replace
from types import MethodType
import numpy as np

from ..benchmark_perception import RedBoxTracker, BoxPrior, box_surface
from ..contracts import Pose
from .block_exact_reference_v201 import SmoothBlockEnvironment as OriginalBoundary
from .block_exact_reference_v201 import public_decision_scope
from .perception import RegistrationConfig
from .safe_native_mesh import read_native_mesh
from .supported_sphere_attachment_v10 import horizontal_rectangle
from .evaluation import file_sha

SCHEMA = 'r10_block_first_contact_RGBD_support_and_gripper_only_contact_v298'


def binding(root):
    from pathlib import Path
    names = ('block_held_support_v298.py', 'block_exact_reference_v201.py',
             'block_reference_v197.py', 'safe_native_mesh.py',
             'supported_sphere_attachment_v10.py', 'perception.py')
    q = RegistrationConfig()
    return dict(schema=SCHEMA, sources_sha256={
        'our_mani/multitask/' + n: file_sha(Path(__file__).with_name(n)) for n in names},
        original_tracker_and_history_unchanged=True,
        original_known_axis_aligned_BoxPrior_retained=True,
        support_admission='first public target contact, own closed request, same current selected RGBD cloud, finite static workspace',
        additional_quality_gate=dict(maximum_surface_rms_m=q.maximum_registration_rms_m,
            minimum_inlier_fraction=q.minimum_inlier_fraction,
            minimum_visible_points=q.minimum_visible_points,
            source='shared RegistrationConfig defaults; additional to legacy tracker'),
        contact_permission='qualified held target versus gripper-owned shape subtree only',
        held_target_vs_forearm_and_all_static_obstacles_checked=True,
        unheld_collision_behavior_unchanged=True, extra_camera_or_physics_calls=False,
        native_dynamic_truth_as_policy_input=False, original_native_grasp_and_success_unchanged=True,
        original_reference4_actual6_and10s_unchanged=True, formal_VAL_TEST=False)


def supported_box(pose, points, vertices, plane, prior):
    points = np.asarray(points, float)
    q = RegistrationConfig()
    world = pose.apply(vertices)
    height = plane['height']
    row = dict(accepted=False, source='same_current_raw_RGBD_and_static_workspace')
    if (points.ndim != 2 or points.shape[1] != 3 or len(points) < q.minimum_visible_points
            or not np.isfinite(points).all()
            or np.mean(points[:, 2] >= height) < q.minimum_inlier_fraction
            or np.any(world[:, :2].min(0) < plane['xy_min'])
            or np.any(world[:, :2].max(0) > plane['xy_max'])):
        return pose, dict(row, reason='current_surface_or_finite_footprint_unqualified')
    margin = 16 * float(np.spacing(np.float32(max(1., abs(height)))))
    shift = max(0., height + margin - world[:, 2].min())
    candidate = Pose(pose.position + [0., 0., shift], pose.rotation)
    # Exact distance to the declared box surface, without surface-sampling bias.
    excess = np.abs(candidate.inverse_apply(points)) - np.asarray(prior.dimensions) / 2
    signed = np.linalg.norm(np.maximum(excess, 0.), axis=1) + np.minimum(excess.max(1), 0.)
    residual = np.abs(signed)
    mask = residual <= 2 * q.maximum_registration_rms_m
    rms = float(np.sqrt(np.mean(residual[mask] ** 2))) if mask.any() else float('inf')
    fraction = float(mask.mean())
    accepted = bool(rms <= q.maximum_registration_rms_m and fraction >= q.minimum_inlier_fraction)
    row.update(accepted=accepted, z_correction_m=float(shift), surface_rms_m=rms,
        inlier_fraction=fraction, minimum_corrected_vertex_z=float(candidate.apply(vertices)[:, 2].min()))
    return (candidate if accepted else pose), row


class MeasuredTracker(RedBoxTracker):
    def __init__(self, boundary, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.boundary = boundary

    def observe(self, observation):
        completed = self.complete_geometry
        self.complete_geometry = False
        try:
            raw, metadata = super().observe(observation)
        finally:
            self.complete_geometry = completed
        self.boundary.visual_measurement = dict(timestamp=raw.timestamp, pose=raw.pose,
            points=raw.points_world.copy(), prior=self.prior, object_id=raw.object_id)
        state = replace(raw, points_world=box_surface(raw.pose, self.prior.dimensions, self.points)) if completed else raw
        return state, dict(metadata, geometry_completion=completed)


def held_collision(service, pose, arm_shapes, gripper_handles):
    """Owned/restoring sandbox; only held target/gripper pairs are exempt."""
    service.queries += 1
    with service.estimated_target(pose):
        service.target.set_collidable(False)
        arm_world = bool(service.env.arm.check_arm_collision())
        service.target.set_collidable(True)
        held_world = any(service.target.check_collision(g) for g in service.ghosts)
        held_arm = any(service.target.check_collision(g) for g in arm_shapes
                       if g.get_handle() not in gripper_handles)
        return bool(arm_world or held_world or held_arm)


class SmoothBlockEnvironment(OriginalBoundary):
    def __init__(self, env, context=None):
        super().__init__(env, context)
        self.visual_measurement = None
        self.held_local = None
        self.contact_started_at = None
        self.admission_attempted = False
        self.support_events = []
        self.held_queries = 0
        self.service = None

    def observe(self):
        result = super().observe()
        robot = result[1]
        if not robot.contact or self.env._requested_opening >= .5:
            self.held_local = None
            self.contact_started_at = None
            self.admission_attempted = False
        elif self.contact_started_at is None:
            self.contact_started_at = float(robot.timestamp)
        return result

    def _install_map(self):
        service = self.env.map_service
        if service is self.service:
            return
        if self.service is not None:
            raise ValueError('held geometry scope cannot cross a native reset')
        from pyrep.const import ObjectType
        matches = [g for g, info in zip(service.ghosts, service.map) if info['name'] == 'workspace']
        if len(matches) != 1:
            raise ValueError('one immutable workspace support required')
        v, f, _ = read_native_mesh(matches[0])
        self.plane = horizontal_rectangle(v, f, matches[0].get_matrix())
        self.vertices = np.asarray(read_native_mesh(service.target)[0], float).reshape(-1, 3)
        half = np.asarray(BoxPrior().dimensions) / 2
        if not (np.allclose(self.vertices.min(0), -half, atol=1e-5, rtol=0)
                and np.allclose(self.vertices.max(0), half, atol=1e-5, rtol=0)):
            raise ValueError('actual collision prototype differs from original known 5cm box prior')
        gripper = self.env.gripper.get_objects_in_tree(object_type=ObjectType.SHAPE, exclude_base=False)
        self.gripper_handles = {x.get_handle() for x in gripper}
        self.arm_shapes = [x for x in self.env.arm.get_objects_in_tree(
            object_type=ObjectType.SHAPE, exclude_base=False) if x.is_collidable()]
        if not self.gripper_handles or not self.arm_shapes:
            raise ValueError('complete robot and gripper-owned shape sets required')
        self.service, self.original_check = service, service.check
        def check(instance, pose=None):
            if self._held_valid():
                self.held_queries += 1
                tool = Pose.from_xyzw(self.env.arm.get_tip().get_pose())
                local = self.held_local
                return held_collision(instance, Pose(tool.apply(local.position), tool.rotation @ local.rotation),
                    self.arm_shapes, self.gripper_handles)
            return self.original_check(pose)
        service.check = MethodType(check, service)

    def _held_valid(self):
        return bool(self.held_local is not None and self.public_robot is not None
            and self.public_robot.contact and self.env._requested_opening < .5
            and abs(self.public_robot.timestamp - self.env.time) < 1e-8)

    def supplied_decision(self, robot, obj, decision, dynamics):
        super().supplied_decision(robot, obj, decision, dynamics)
        self._install_map()
        if self.contact_started_at is None or self.admission_attempted:
            return
        self.admission_attempted = True
        measurement = self.visual_measurement
        if (not robot.contact or self.env._requested_opening >= .5 or measurement is None
                or measurement['object_id'] != 'target' or obj is None or obj.object_id != 'target'
                or abs(measurement['timestamp'] - robot.timestamp) > 1e-8
                or abs(self.contact_started_at - robot.timestamp) > 1e-8):
            self.support_events.append(dict(time=float(robot.timestamp), accepted=False,
                reason='no_same_first_contact_RGBD_measurement'))
            return
        pose, row = supported_box(measurement['pose'], measurement['points'], self.vertices,
            self.plane, measurement['prior'])
        if row['accepted']:
            self.held_local = Pose(robot.pose.inverse_apply(pose.position), robot.pose.rotation.T @ pose.rotation)
        self.support_events.append(dict(row, time=float(robot.timestamp),
            rigid_transform_frozen_until_public_contact_lost=True))

    @contextmanager
    def _collision_scope(self):
        if not self._held_valid():
            with super()._collision_scope():
                yield
            return
        model = self.env.robot_model
        previous = model.collision_query
        model.collision_query = lambda offset: self.service.check()
        try:
            yield
        finally:
            model.collision_query = previous

    def restore_map(self):
        if self.service is not None:
            self.service.check = self.original_check

    def receipt(self):
        return dict(super().receipt(), block_held_support_v298=dict(schema=SCHEMA,
            support_events=deepcopy(self.support_events), held_collision_queries=self.held_queries,
            original_tracker_history_changed=False, shared_static_map_changed=False,
            gripper_only_held_contact_permission=True,
            held_vs_arm_and_all_static_obstacles_checked=True,
            additional_physics_or_camera_calls=0, actual_motor_motion_guaranteed=False))


@contextmanager
def measured_tracker_scope(evaluator, boundary):
    original = evaluator.RedBoxTracker
    if original is not RedBoxTracker:
        raise ValueError('fresh original tracker scope required')
    evaluator.RedBoxTracker = lambda *a, **k: MeasuredTracker(boundary, *a, **k)
    try:
        yield
    finally:
        evaluator.RedBoxTracker = original
        boundary.restore_map()
