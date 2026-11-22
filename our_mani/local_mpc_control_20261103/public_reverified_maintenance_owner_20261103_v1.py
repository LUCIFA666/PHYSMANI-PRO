"""Unwired current-public maintenance after an explicit re-verification commit.

The failed original owner and original references are retained. This component
never resets them, admits another epoch, executes an action, or certifies force.
"""
from copy import deepcopy
from dataclasses import fields, is_dataclass
import hashlib
import json
import numpy as np

from .public_reverification_epoch_prototype_20261103 import (
    PublicReverificationEpochPrototype, VERSION as PROTOTYPE_VERSION)
from .active_visual_attachment import ActiveVisualAttachment
from .public_motion_bridge import PublicRobotSensors, PublicMotionEvidence
from ..contracts import Pose

VERSION = 'public_reverified_maintenance_owner_20261103_v1'


def _plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if is_dataclass(value):
        return {f.name: _plain(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if value is None or type(value) in (str, bool, int, float):
        return value
    raise TypeError('Only explicit public proof data can be fingerprinted')


def _digest(value):
    return hashlib.sha256(json.dumps(_plain(value), sort_keys=True,
        allow_nan=False).encode()).hexdigest()


class PublicReverifiedMaintenanceOwner:
    """Construct immediately after a genuine current-frame explicit commit.

    Constructor publishes that commit frame once in ``latest``. Subsequent
    observe calls require distinct consecutive real50ms frames. Failure is
    permanent for this owner; physical release needs a separate future owner,
    and is never an automatic reset here. Outer source binding is still needed
    before any production integration; exact Python types are not sensors.
    """
    def __init__(self, prototype):
        if type(prototype) is not PublicReverificationEpochPrototype:
            raise TypeError('Exact independent re-verification prototype required')
        self.prototype = prototype
        self._assert_prototype_methods()
        prototype._unchanged()
        p = prototype
        if (type(p.verification_epoch) is not int or p.verification_epoch != 1
                or type(p.committed) is not dict or type(p.inner) is not ActiveVisualAttachment
                or type(p.last_robot) is not PublicRobotSensors or type(p.last_result) is not dict
                or p.last_close is not True or p.witness_failed or p.latest is None
                or p.latest.get('current_fresh_witness_passed') is not True):
            raise ValueError('A current explicit committed epoch, never static reappearance, is required')
        robot, result, commit = p.last_robot, p.last_result, p.committed
        target, error = p._current(result, robot, True)
        if (error or target is not p.last_target or p.last_timestamp != robot.timestamp
                or p._frame_identity(result, robot) != p._frame_digest):
            raise ValueError('Exact current public commit frame required')
        evidence = commit.get('evidence')
        metrics = dict(timestamp=robot.timestamp, **p._metrics())
        if (commit.get('version') != PROTOTYPE_VERSION
                or type(commit.get('verification_epoch')) is not int or commit['verification_epoch'] != 1
                or commit.get('physical_attempt_id') != p.attempt_id
                or commit.get('timestamp') != robot.timestamp
                or type(evidence) is not PublicMotionEvidence
                or evidence.event != 'attachment_confirmed' or evidence.timestamp != robot.timestamp
                or evidence.sources != ('rgbd_relative_pose_tracking', 'public_robot_pose', 'public_gripper_opening')
                or evidence.orientation_observable is not False
                or _digest(commit.get('fresh_witness')) != _digest(p.inner.latest)
                or _digest(commit.get('current_whole_trace_confirmation')) != _digest(metrics)
                or _digest(commit.get('original_failed_proof')) != _digest(p.original_proof)
                or _digest(commit.get('original_fixed_r0')) != _digest(p.r0)
                or _digest(commit.get('original_object_anchor')) != _digest(p.object_anchor)
                or _digest(commit.get('original_preclose_baseline')) != _digest(p.original_preclose_baseline)
                or any(commit.get(k) is not False for k in
                    ('old_failed_owner_reset', 'maintenance_replaced', 'production_integrated'))):
            raise ValueError('Exact explicit-commit identity, source and complete witness proof required')
        self.original_failed_owner = p.old
        self._inner = p.inner
        self._commit = commit
        self._commit_digest = _digest(commit)
        self._references_digest = _digest(self._references())
        self._r0 = p.r0.copy()
        self._object_anchor = p.object_anchor.copy()
        self._preclose_baseline = p.original_preclose_baseline
        self.attempt_id = p.attempt_id
        self.verification_epoch = 1
        self.started_at = float(robot.timestamp)
        self.last_timestamp = self.started_at
        self.failed = False
        self.failure_time = None
        self.first_failure_reason = None
        self.latest = None
        self._receipt(robot.timestamp, 'current_explicit_epoch_commit_bound', True,
            target=target, robot=robot, integrity=True)

    def _assert_prototype_methods(self):
        p = self.prototype
        for name in ('_current', '_unchanged', '_frame_identity', '_metrics'):
            if getattr(getattr(p, name), '__func__', None) is not getattr(PublicReverificationEpochPrototype, name):
                raise ValueError('Original prototype public-contract methods required')

    def _references(self):
        p = self.prototype
        return dict(old_started=p.old.started, old_failed=p.old.failed,
            old_attempt_id=p.old.attempt_id, old_failure_time=p.old.failure_time,
            old_r0=p.old.r0, old_object_anchor=p.old.object_anchor, old_proof=p.old.proof,
            prototype_r0=p.r0, prototype_object_anchor=p.object_anchor,
            original_proof=p.original_proof, original_preclose_baseline=p.original_preclose_baseline)

    def _integrity(self):
        p = self.prototype
        self._assert_prototype_methods()
        p._unchanged()
        if (type(p) is not PublicReverificationEpochPrototype
                or type(p.verification_epoch) is not int or p.verification_epoch != 1
                or p.old is not self.original_failed_owner or p.old.failed is not True
                or p.inner is not self._inner or p.committed is not self._commit
                or _digest(p.committed) != self._commit_digest
                or _digest(self._references()) != self._references_digest):
            raise ValueError('Committed identity and original failed proof/references are immutable')

    def _receipt(self, now, reason, passed, *, target=None, robot=None, integrity=True):
        event = None
        drift = rise = None
        if target is not None and robot is not None:
            relative = robot.tcp_pose.inverse_apply(target.pose.position)
            drift = float(np.linalg.norm(relative-self._r0))
            rise = float(target.pose.position[2]-self._object_anchor[2])
            if passed:
                event = PublicMotionEvidence('attachment_confirmed', now,
                    ('rgbd_relative_pose_tracking', 'public_robot_pose', 'public_gripper_opening'),
                    Pose(relative, robot.tcp_pose.rotation.T@target.pose.rotation),
                    orientation_observable=False)
        self.latest = dict(version=VERSION, timestamp=float(now),
            protocol='explicit_reverified_epoch_current_public_position_maintenance_v1',
            verification_epoch=1, physical_attempt_id=self.attempt_id,
            current_reverified_position_maintenance_passed=bool(passed),
            mode='ATTACHED_PUBLIC' if passed else 'UNRESOLVED', evidence=event,
            failed=self.failed, failure_time=self.failure_time,
            first_failure_reason=self.first_failure_reason, reason=reason,
            current_fixed_original_r0_deviation_m=drift,
            current_visual_height_above_original_probe_anchor_m=rise,
            original_fixed_r0=self._r0.copy(), original_object_anchor=self._object_anchor.copy(),
            original_preclose_baseline=deepcopy(self._preclose_baseline),
            original_reference_integrity_passed=bool(integrity),
            original_failed_owner_remains_failed=self.original_failed_owner.failed is True,
            old_failed_owner_reset=False, anchor_rolled_forward=False,
            original_success_height_baseline_not_rolled=True,
            fresh_commit_timestamp=self.started_at,
            committed_proof_sha256=self._commit_digest,
            real_sensor_dt_seconds=.05, maximum_original_r0_deviation_m=.010,
            minimum_visual_height_above_original_probe_anchor_m=.008,
            source='current_typed_visible_public_body_and_actual_public_robot',
            controlled_or_predicted_pose_used=False, Native_inputs_used=False,
            force_or_frictional_proof=False, full_SE3_attachment_qualified=False,
            production_integrated=False, outer_source_binding_required_for_production=True,
            permanent_failure_until_physical_release_no_automatic_restart=True)
        return deepcopy(self.latest)

    def _fail(self, now, reason, *, integrity=True):
        if not self.failed:
            self.failure_time = float(now)
            self.first_failure_reason = reason
        self.failed = True
        return self._receipt(now, reason, False, integrity=integrity)

    def observe(self, result, robot, *, close_requested):
        if type(robot) is not PublicRobotSensors or type(close_requested) is not bool:
            self._fail(self.last_timestamp, 'untyped_actual_robot_or_close_intent')
            raise TypeError('Exact current PublicRobotSensors and known close-intent bool required')
        now = robot.timestamp
        try:
            self._integrity()
        except (TypeError, ValueError, AttributeError):
            return self._fail(now, 'committed_identity_or_original_reference_changed', integrity=False)
        if now <= self.last_timestamp:
            return self._fail(now, 'duplicate_or_nonincreasing_real_frame')
        gap = now-self.last_timestamp
        self.last_timestamp = now
        if self.failed:
            return self._receipt(now, 'failed_reverified_owner_latched_no_automatic_restart', False)
        if abs(gap-.05) > 1e-6:
            return self._fail(now, 'missing_non50ms_real_feedback_no_interpolation')
        try:
            target, error = self.prototype._current(result, robot, close_requested)
        except (TypeError, ValueError, AttributeError):
            return self._fail(now, 'current_public_contract_rejected')
        if error:
            return self._fail(now, error)
        return self._receipt(now, 'current_reverified_visible_position_maintained', True,
            target=target, robot=robot, integrity=True)
