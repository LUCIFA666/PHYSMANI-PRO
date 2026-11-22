"""Public held-block admission and phase prototype on frozen1153/1114.

Only an applied own close and the existing public target-grasp bool advance
the original TCP lift.  A current measured known-axis box must pass the
original exact-surface quality before its transform can enable transport.
No simulator calls, new pose prior, support-plane correction or success claim
are introduced by the new observation/phase methods.  Native validation is
separate; this module alone is not a qualification for VAL/TEST.
"""
from contextlib import contextmanager
from copy import deepcopy
import hashlib
from pathlib import Path

import numpy as np

from ..contracts import Pose
from . import retained_preclose_block_integration_v1153 as retained
from .perception import RegistrationConfig
from .physmani_block_policy_v538 import (
    PublicBlockCollision, BlockPublicFrontend, BlockGeometryIntents,
)

SCHEMA = 'r10_public_held_block_phase_v1203'
SOURCE = 'our_mani/multitask/public_held_block_phase_v1203.py'
PARENT_SOURCE = retained.SOURCE
PARENT_SHA256 = 'ef646686984cf2de785e00ae39ec460bb68ac2fae9b3b4c1b9ca8e5a303a1b92'
SURFACE_SOURCE = 'our_mani/multitask/block_held_support_v298.py'
SURFACE_SHA256 = 'f28e067d6e1c3ffd3355602e26903d07bf8005b91d282fbd8b486fb988e63c97'
SOURCES = tuple(dict.fromkeys((SOURCE, SURFACE_SOURCE, *retained.SOURCES,
                              'our_mani/multitask/perception.py')))


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def binding(root, native_repo):
    root = Path(root)
    if (_sha(root/PARENT_SOURCE) != PARENT_SHA256
            or _sha(root/SURFACE_SOURCE) != SURFACE_SHA256):
        raise retained.ProtocolInvariantError('frozen1153/298 public block parents changed')
    q = RegistrationConfig()
    return dict(schema=SCHEMA, source_sha256=_sha(root/SOURCE),
        parent=retained.binding(root, native_repo),
        sources_sha256={name: _sha(root/name) for name in SOURCES},
        tasks=list(retained.parent.BLOCKS),
        changed_factor='public_grasp_phase_and_inhand_pose_admission',
        factory='explicit_subclasses_once_then1120_adapt_and1153_retained_facade',
        frozen_parent_sources_edited=False, frozen_parent_classes_patched=False,
        original_pregrasp_tracker_floor_ROI_association_and_preclose_unchanged=True,
        original_known_axis_box_dimensions_m=[.05, .05, .05],
        exact_box_quality=dict(source=SURFACE_SOURCE, source_sha256=SURFACE_SHA256,
            minimum_visible_points=q.minimum_visible_points,
            maximum_registration_rms_m=q.maximum_registration_rms_m,
            minimum_inlier_fraction=q.minimum_inlier_fraction,
            residual='original298_analytic_box_surface', numerical_thresholds_changed=False),
        held_admission='same_counted_public_raw_pose_and_selected_measured_cloud',
        held_workspace_plane_XY_support_or_Z_correction_required=False,
        admission_retry='once_per_new_counted_frame_while_current_contact_and_confirmed_close',
        lift='original_TCP_plus_original_lift_height_after_confirmed_close_and_public_target_grasp',
        held_transform_required_for_transport=True,
        release='original1153_dispatch_with_active_capture_invalidation_after_confirmation',
        public_joint_collision_hard_veto=False, collision_diagnostic_limit=0,
        additional_native_getters_observations_predictions_or_success_calls=0,
        original_native_physics_task_predicates_tick_budget_and_exceptions_unchanged=True,
        native_success_claimed=False, formal_VAL_TEST_qualification=False)


def qualify_observed_inhand_box(pose, points_world, prior):
    """Quality-check the current public known-axis candidate, without support.

    The pose is already estimated by the original RedBoxTracker from this
    selected measured cloud.  This is neither arbitrary6DoF ICP nor a fit to
    completed/forecast geometry.  The analytic residual and inequalities are
    exactly298's post-support quality, evaluated without its Z translation.
    """
    q = RegistrationConfig()
    dimensions = np.asarray(prior.dimensions, float)
    if dimensions.shape != (3,) or not np.array_equal(dimensions, [.05, .05, .05]):
        raise retained.ProtocolInvariantError('original public5cm box prior required')
    row = dict(accepted=False, source='same_current_raw_RGBD_known_axis_box',
        support_plane_used=False, support_XY_footprint_used=False, z_correction_m=0.,
        minimum_visible_points=q.minimum_visible_points,
        maximum_registration_rms_m=q.maximum_registration_rms_m,
        minimum_inlier_fraction=q.minimum_inlier_fraction)
    try:
        points = np.asarray(points_world, float)
    except (TypeError, ValueError):
        return None, dict(row, reason='current_finite_measured_surface_required')
    if (type(pose) is not Pose or points.ndim != 2 or points.shape[1] != 3
            or len(points) < q.minimum_visible_points or not np.isfinite(points).all()):
        return None, dict(row, reason='current_finite_measured_surface_required',
                          measured_points=int(points.shape[0]) if points.ndim else 0)
    excess = np.abs(pose.inverse_apply(points)) - dimensions / 2
    signed = np.linalg.norm(np.maximum(excess, 0.), axis=1) + np.minimum(excess.max(1), 0.)
    residual = np.abs(signed)
    mask = residual <= 2 * q.maximum_registration_rms_m
    rms = float(np.sqrt(np.mean(residual[mask] ** 2))) if mask.any() else float('inf')
    fraction = float(mask.mean())
    accepted = bool(rms <= q.maximum_registration_rms_m and fraction >= q.minimum_inlier_fraction)
    row.update(accepted=accepted, measured_points=len(points), surface_rms_m=rms,
        inlier_fraction=fraction,
        reason='current_exact_box_quality_passed' if accepted else 'current_exact_box_quality_unqualified')
    return (pose if accepted else None), row


class PublicHeldBlockCollision(PublicBlockCollision):
    """Keep the original static construction, change only held admission."""
    def __init__(self, env):
        super().__init__(env)
        self.last_contact_frame = None
        self.held_admission_events = []
        self.capture_epoch = 0

    def clear_public_capture(self):
        self.held_local = None
        self.attempted = False

    def reset_public_capture(self):
        self.clear_public_capture()
        self.last_contact_frame = None
        self.capture_epoch += 1

    def observe_contact(self, robot, raw, prior, own_close):
        when = float(robot.timestamp)
        if self.last_contact_frame is not None:
            if when < self.last_contact_frame - 1e-8:
                raise retained.ProtocolInvariantError('monotonic counted public capture frames required')
            if abs(when-self.last_contact_frame) <= 1e-8:
                return
        self.last_contact_frame = when
        if not robot.contact or not own_close:
            self.clear_public_capture()
            return
        if self.held_local is not None:
            return
        self.attempted = True  # A receipt fact, never a permanent admission veto.
        if raw is None or raw.object_id != 'target' or abs(raw.timestamp-when) > 1e-8:
            proof = dict(accepted=False, reason='current_selected_public_RGBD_unavailable')
        else:
            pose, proof = qualify_observed_inhand_box(raw.pose, raw.points_world, prior)
            if pose is not None:
                self.held_local = Pose(robot.pose.inverse_apply(pose.position),
                                       robot.pose.rotation.T @ pose.rotation)
        self.held_admission_events.append(dict(proof, time=when,
            source='confirmed_close_public_target_grasp_same_current_RGBD',
            capture_epoch=self.capture_epoch, native_success_claimed=False))

    def receipt(self):
        result = super().receipt()
        # The old298 admission was replaced, so never present its inherited
        # constant description as the active execution truth.
        result.update(original_held_support_geometry_reused=False,
            held_pose_admission='current_measured_known_axis_exact_box_without_workspace_support',
            held_admission_events=deepcopy(self.held_admission_events),
            capture_epoch=self.capture_epoch,
            workspace_support_pose_used_for_held_admission=False)
        return result


class PublicHeldBlockFrontend(BlockPublicFrontend):
    """One original tracker/predict pipeline; invalidate stale capture epochs."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.public_capture_seen = False
        self.public_capture_lost = False

    def invalidate_public_capture(self):
        self.close_confirmed = False
        self.public_capture_seen = False
        self.public_capture_lost = True
        self.collision.clear_public_capture()

    def reset_public_capture(self):
        self.close_confirmed = False
        self.public_capture_seen = False
        self.public_capture_lost = False
        self.collision.reset_public_capture()

    def observe(self, observation, robot):
        # Validate before mutating epoch state; the original method enforces
        # these same counted-frame conditions and then calls the tracker once.
        if (robot.task_success or abs(float(observation['timestamp'])-robot.timestamp) > 1e-8
                or self.last_time is not None and robot.timestamp <= self.last_time+1e-8):
            return super().observe(observation, robot)
        if self.public_capture_seen and not robot.contact:
            self.invalidate_public_capture()
        result = super().observe(observation, robot)
        if self.close_confirmed and robot.contact:
            self.public_capture_seen = True
        return result


class PublicHeldBlockIntents(BlockGeometryIntents):
    """Original acquisition, then TF-independent lift and TF-gated transport."""
    def reset_public_capture(self):
        self.frontend.reset_public_capture()
        self.phase = 'approach'
        self.selected = self.close_witness = self.lift_goal = self.transport_goal = None

    def _transport_goal(self):
        f = self.frontend
        local = f.collision.held_local
        if local is None or f.goal_pose is None:
            return None
        return Pose(f.goal_pose.position-f.goal_pose.rotation @ local.position,
                    f.goal_pose.rotation)

    def select(self):
        f = self.frontend
        r = f.robot
        if (r is None or f.goal_pose is None or f.frames < f.config['warmup_ticks']
                or self.phase == 'released'):
            return None
        active = self.close_witness is not None and f.close_confirmed
        if active and r.contact and self.phase in ('approach', 'acquire'):
            self.phase = 'lift'
            self.lift_goal = Pose(r.pose.position+[0, 0, self.local.lift_height], r.pose.rotation)
            self.transport_goal = self._transport_goal()
            self.events.append(dict(time=float(r.timestamp),
                event='confirmed_own_close_and_public_target_grasp_enter_lift',
                held_transform_available=f.collision.held_local is not None,
                native_success_claimed=False))
        if self.phase not in ('lift', 'transport'):
            return super().select()
        if not active or not r.contact:
            return None
        if self.phase == 'lift' and np.linalg.norm(r.pose.position-self.lift_goal.position) <= .015:
            if f.collision.held_local is None:
                return None  # Original no-new-intent hold; never a blind transport/open.
            if self.transport_goal is None:
                self.transport_goal = self._transport_goal()
            self.phase = 'transport'
        if self.phase == 'transport' and f.collision.held_local is None:
            return None
        target = self.lift_goal if self.phase == 'lift' else self.transport_goal
        action = np.r_[target.xyzw(), 0.]
        self.selected = dict(pose=target, opening=0., selected_at=float(r.timestamp),
                             phase=self.phase, action=action.copy())
        self.events.append(dict(time=float(r.timestamp), phase=self.phase,
            event='public_block_RML_pose_intent', action=action.tolist(),
            predicted_reference_time=None, method=f.method, native_state_or_outcome_read=False))
        return action

    def confirm_opening(self, opening, proof, proposed, *, confirmed_after):
        f = self.frontend
        was_active = f.close_confirmed or f.public_capture_seen or self.phase in ('lift', 'transport')
        result = super().confirm_opening(opening, proof, proposed, confirmed_after=confirmed_after)
        if opening >= .5 and was_active:
            f.invalidate_public_capture()
            self.transport_goal = self.lift_goal = self.selected = None
            self.phase = 'released'
        elif opening < .5 and proposed is not None:
            f.public_capture_seen = False
            f.public_capture_lost = False
        return result

    def receipt(self):
        result = super().receipt()
        result['public_held_block_phase_v1203'] = dict(schema=SCHEMA,
            phase=self.phase, active_confirmed_close=bool(self.frontend.close_confirmed),
            public_capture_seen=bool(self.frontend.public_capture_seen),
            public_capture_lost=bool(self.frontend.public_capture_lost),
            lift_requires_held_transform=False, transport_requires_held_transform=True,
            original_pregrasp_and_preclose_unchanged=True,
            original_lift_height_and_arrival_threshold_unchanged=True,
            current_measured_cloud_only=True, native_success_claimed=False,
            additional_native_calls=0, formal_VAL_TEST_qualification=False)
        return result


class PublicHeldBlockWitness(retained.RetainedBlockPrecloseWitness):
    """Keep1153's exact dispatch; label the replaced phase receipt honestly."""
    def receipt(self):
        result = super().receipt()
        row = result['retained_preclose_block_integration_v1153']
        row['original_frontend_support_phase_and_native_actuator_unchanged'] = False
        row['original_preclose_and_native_actuator_unchanged'] = True
        row['active_public_phase_source'] = SOURCE
        result['held_geometry']['active_use'] = 'current_public_inhand_box_admission_without_workspace_support'
        return result


def construct(root, config, task, method, env):
    """Explicit1153-compatible factory; no patch of its frozen constructors."""
    if task not in retained.parent.BLOCKS or method not in ('cv_geometry', 'context_geometry'):
        raise ValueError('exactly the two original block tasks and geometry predictors')
    collision = PublicHeldBlockCollision(env)
    frontend = PublicHeldBlockFrontend(Path(root), config, task, method, env.robot_model, collision)
    policy = PublicHeldBlockIntents(frontend)
    frontend, dispatched, executor = retained.parent.adapt(config, frontend, policy)
    active = PublicHeldBlockWitness(dispatched)
    executor.skill_policy = active
    return frontend, active, executor


@contextmanager
def integrated_block_components(root, config, task, method, env):
    """Same owned scope signature; the parent still owns env/reset/outcomes."""
    frontend, active, executor = construct(root, config, task, method, env)
    try:
        yield frontend, active, executor
    finally:
        # Mirror1153's final facade restoration on the SAME new executor.
        executor.skill_policy = active.parent_policy
