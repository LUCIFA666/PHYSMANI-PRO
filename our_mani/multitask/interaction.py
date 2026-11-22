"""TRAIN-fitted relative interaction curves, feasible candidates and SE3 MPC.

This module consumes observed body states and a fitted artifact. It has no
native-task/scene handles, expert waypoint lookup, goal-sensor read or hidden
phase input. Fitting may inspect future TRAIN body tracks; inference cannot.
No artifact means no trained skill: constructors fail closed.
"""
from collections import Counter
from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation, Slerp

from ..contracts import ObjectState, Pose
from ..intercept_planner import JointTrajectory
from .specs import get_task_spec
from .candidate_success import (ContinuationAttempts, candidate_features,
                                select_candidate, CONTINUATION_POLICY)
from .reference_time import (SCHEMA as REFERENCE_TIME_SCHEMA, CURRENT, COMMAND_ISSUE,COMMAND_GOAL,
    validate_basis,validate_template_basis, selected_reference, valid_sha)

LIBRARY_SCHEMA = 'r10_TRAIN_body_relative_phase_curves_v1'


class InteractionArtifactError(ValueError):
    pass


class InteractionFittingPending(InteractionArtifactError):
    def __init__(self,message,diagnostics):
        super().__init__(message)
        self.diagnostics=_json(diagnostics)


def _json(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: _json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json(v) for v in value]
    return value


def _pose_at(times, poses, time):
    if time < times[0]-1e-8 or time > times[-1]+1e-8:
        raise InteractionArtifactError('TRAIN body supervision does not cover requested lag')
    time = float(np.clip(time, times[0], times[-1]))
    if len(times) == 1:
        return Pose.from_xyzw(poses[0])
    position = np.array([np.interp(time, times, poses[:, axis]) for axis in range(3)])
    rotation = Slerp(times, Rotation.from_quat(poses[:, 3:]))([time]).as_matrix()[0]
    return Pose(position, rotation)


@dataclass(frozen=True)
class TRAINBodyTrack:
    object_id: str
    times: np.ndarray
    poses_xyzw: np.ndarray
    motion_modes: tuple[str, ...]
    orientation_observable: np.ndarray
    pose_valid: np.ndarray | None = None
    maximum_gap_seconds: float = .075

    def __post_init__(self):
        times, poses = np.asarray(self.times, float), np.asarray(self.poses_xyzw, float)
        observed = np.asarray(self.orientation_observable, bool)
        valid = np.ones(len(times),bool) if self.pose_valid is None else np.asarray(self.pose_valid,bool)
        if (not self.object_id or 'stage_run' in self.object_id.lower()
                or times.ndim != 1 or not len(times) or np.any(np.diff(times) <= 0)
                or poses.shape != (len(times), 7) or observed.shape != times.shape or valid.shape != times.shape
                or len(self.motion_modes) != len(times)
                or any(mode not in ('external', 'attached', 'released') for mode in self.motion_modes)
                or not np.isfinite(times).all() or not np.isfinite(poses).all()
                or np.any(np.linalg.norm(poses[:, 3:], axis=1) < 1e-8)
                or not np.isfinite(self.maximum_gap_seconds) or self.maximum_gap_seconds <= 0):
            raise InteractionArtifactError('stable aligned TRAIN body identities/SE3 are required')
        object.__setattr__(self, 'times', times.copy())
        object.__setattr__(self, 'poses_xyzw', poses.copy())
        object.__setattr__(self, 'orientation_observable', observed.copy())
        object.__setattr__(self, 'pose_valid', valid.copy())

    def at(self, time, *, orientation_required, same_mode_as=None, not_before=None):
        time = float(time)
        if time < self.times[0]-1e-8 or time > self.times[-1]+1e-8:
            raise InteractionArtifactError('TRAIN body supervision does not cover requested lag')
        index = int(np.clip(np.searchsorted(self.times, time), 0, len(self.times)-1))
        # An exact boundary observation is valid on its own side; interpolation
        # between two different modes is not a physical body trajectory.
        left = index if abs(time-self.times[index]) <= 1e-8 else max(0, index-1)
        begin = left
        if not_before is not None:
            if not_before < self.times[0]-1e-8 or not_before > time+1e-8:
                raise InteractionArtifactError('invalid TRAIN lag interval')
            begin = max(0,int(np.searchsorted(self.times,float(not_before),side='right'))-1)
        if not self.pose_valid[begin:index+1].all():
            raise InteractionArtifactError('TRAIN lag crosses invalid offline reference supervision')
        if np.any(np.diff(self.times[begin:index+1]) > self.maximum_gap_seconds):
            raise InteractionArtifactError('TRAIN lag crosses a supervision clock gap')
        if (len(set(self.motion_modes[begin:index+1])) != 1
                or (same_mode_as is not None and self.motion_modes[index] != same_mode_as)):
            raise InteractionArtifactError('TRAIN lag crosses an attachment/contact mode boundary')
        if orientation_required and not self.orientation_observable[begin:index+1].all():
            raise InteractionArtifactError('SE3 anchor requires observable/supervised orientation')
        return _pose_at(self.times, self.poses_xyzw, time)


@dataclass(frozen=True)
class TRAINReferenceBinding:
    reference_role: str
    reference_id: str | None
    reference_kind: str  # observed_se3, translation_only, phase_origin, world
    anchor_id: str
    manipulated_role: str
    reference_time_basis: str = CURRENT
    reference_time_rule_sha256: str | None = None

    def __post_init__(self):
        if (self.reference_kind not in ('observed_se3', 'translation_only', 'phase_origin', 'world')
                or not self.reference_role or not self.anchor_id or not self.manipulated_role
                or (self.reference_kind not in ('phase_origin', 'world') and not self.reference_id)):
            raise InteractionArtifactError('explicit TRAIN reference/anchor binding required')
        validate_basis(self.reference_time_basis,self.reference_kind,self.reference_time_rule_sha256)


@dataclass(frozen=True)
class TRAINPhaseCompletionEvent:
    """Explicit offline skill completion; command arrival alone is insufficient.

    The terminal body pose is a TRAIN metric label, never an external-dynamics
    observation. In particular an acquire endpoint may already be attached.
    """
    phase: str
    to_phase: str
    evidence: str
    origin_command_id: int
    eligible_command_ids: tuple[int, ...]
    attempt_start_time: float
    attempt_end_time: float
    binding: TRAINReferenceBinding
    reference_mode: str
    origin_measured_tcp_xyzw: np.ndarray
    terminal_tcp_xyzw: np.ndarray
    terminal_reference_xyzw: np.ndarray
    committed_world_waypoint_xyzw: np.ndarray
    executed_tick_count: int
    command_completed: bool
    label_namespace: str
    issued_reference_xyzw: np.ndarray | None = None
    reference_time_provenance: dict | None = None

    def __post_init__(self):
        if (not self.phase or not self.to_phase or self.label_namespace != 'TRAIN_metric_only'
                or self.evidence not in ('completed_command_then_ordered_teacher_phase_transition',
                    'TRAIN_native_success_during_executed_final_required_phase')
                or type(self.origin_command_id) is not int or self.origin_command_id < 0
                or not self.eligible_command_ids
                or any(type(c) is not int or c < 0 for c in self.eligible_command_ids)
                or len(set(self.eligible_command_ids)) != len(self.eligible_command_ids)
                or self.eligible_command_ids != (self.origin_command_id,)
                or type(self.executed_tick_count) is not int or self.executed_tick_count < 1
                or type(self.command_completed) is not bool
                or not np.isfinite([self.attempt_start_time,self.attempt_end_time]).all()
                or self.attempt_end_time <= self.attempt_start_time
                or self.reference_mode not in ('external','attached','released','not_applicable')
                or ((self.binding.reference_kind in ('phase_origin','world')) !=
                    (self.reference_mode == 'not_applicable'))):
            raise InteractionArtifactError('explicit executed TRAIN phase completion evidence required')
        for field in ('origin_measured_tcp_xyzw','terminal_tcp_xyzw','terminal_reference_xyzw',
                      'committed_world_waypoint_xyzw'):
            value=np.asarray(getattr(self,field),float)
            if value.shape != (7,) or not np.isfinite(value).all() or np.linalg.norm(value[3:]) < 1e-8:
                raise InteractionArtifactError('finite TRAIN completion reference/TCP SE3 required')
            object.__setattr__(self,field,value.copy())
        if self.binding.reference_kind == 'translation_only':
            if Rotation.from_quat(self.terminal_reference_xyzw[3:]).magnitude() > 1e-8:
                raise InteractionArtifactError('translation-only terminal label must retain its identity gauge')
        if self.binding.reference_kind == 'phase_origin':
            expected=Pose.from_xyzw(self.origin_measured_tcp_xyzw)
            actual=Pose.from_xyzw(self.terminal_reference_xyzw)
            if (np.linalg.norm(expected.position-actual.position) > 1e-8 or
                    Rotation.from_matrix(expected.rotation@actual.rotation.T).magnitude() > 1e-8):
                raise InteractionArtifactError('phase-origin terminal reference must be the issue TCP')
        if self.binding.reference_kind == 'world':
            actual=Pose.from_xyzw(self.terminal_reference_xyzw)
            if np.linalg.norm(actual.position) > 1e-8 or Rotation.from_matrix(actual.rotation.copy()).magnitude() > 1e-8:
                raise InteractionArtifactError('world terminal reference must be the public identity frame')
        if self.binding.reference_time_basis == COMMAND_ISSUE:
            issued=np.asarray(self.issued_reference_xyzw,float)
            proof=self.reference_time_provenance
            if (issued.shape!=(7,) or not np.isfinite(issued).all()
                    or abs(np.linalg.norm(issued[3:])-1.)>1e-6
                    or Rotation.from_quat(issued[3:]).magnitude()>1e-8
                    or not isinstance(proof,dict) or proof.get('schema')!=REFERENCE_TIME_SCHEMA
                    or proof.get('rule_sha256')!=self.binding.reference_time_rule_sha256
                    or proof.get('command_id')!=self.origin_command_id
                    or proof.get('issued_at')!=self.attempt_start_time
                    or not valid_sha(proof.get('command_receipt_sha256'))
                    or proof.get('original_terminal_metric_pose_preserved') is not True
                    or proof.get('source_masks_preserved') is not True):
                raise InteractionArtifactError('command_issue needs separate valid issued reference pose and original ledger provenance')
            object.__setattr__(self,'issued_reference_xyzw',issued.copy())
        elif self.issued_reference_xyzw is not None or self.reference_time_provenance is not None:
            raise InteractionArtifactError('current event cannot carry an undeclared issue reference derivation')


@dataclass(frozen=True)
class TRAINSemanticAcquireSegment:
    """Verified actual mode segment, distinct from a raw command event."""
    segment_id: str
    phase: str
    binding: TRAINReferenceBinding
    reference_mode: str
    frame_indices: tuple[int, ...]
    raw_command_ids: tuple[int, ...]
    terminal_frame: int
    attempt_start_time: float
    attempt_end_time: float
    origin_measured_tcp_xyzw: np.ndarray
    terminal_tcp_xyzw: np.ndarray
    terminal_reference_xyzw: np.ndarray
    evidence: str
    reference_time_provenance: dict | None = None
    issued_reference_xyzw: np.ndarray | None = None

    def __post_init__(self):
        if (self.phase!='acquire' or self.reference_mode not in ('external','attached')
                or not self.segment_id or not isinstance(self.binding,TRAINReferenceBinding)
                or self.binding.reference_kind!='translation_only' or self.binding.reference_id!='ball'
                or self.binding.reference_time_basis!=CURRENT
                or not self.frame_indices or any(type(i) is not int or i<0 for i in self.frame_indices)
                or self.frame_indices!=tuple(range(self.frame_indices[0],self.frame_indices[-1]+1))
                or not self.raw_command_ids or any(type(i) is not int or i<0 for i in self.raw_command_ids)
                or len(set(self.raw_command_ids))!=len(self.raw_command_ids)
                or type(self.terminal_frame) is not int or self.terminal_frame<=self.frame_indices[-1]
                or not np.isfinite([self.attempt_start_time,self.attempt_end_time]).all()
                or self.attempt_end_time<=self.attempt_start_time
                or self.evidence!=('public_external_to_attached_boundary' if self.reference_mode=='external'
                    else 'original_acquire_to_lift_event')):
            raise InteractionArtifactError('explicit verified semantic acquire mode segment required')
        for name in ('origin_measured_tcp_xyzw','terminal_tcp_xyzw','terminal_reference_xyzw'):
            value=np.asarray(getattr(self,name),float)
            if value.shape!=(7,) or not np.isfinite(value).all() or np.linalg.norm(value[3:])<1e-8:
                raise InteractionArtifactError('finite acquired semantic segment SE3 required')
            object.__setattr__(self,name,value.copy())


@dataclass(frozen=True)
class TRAINSemanticAcquireEvidence:
    task_name: str
    source_id: str
    parent_source_group: str
    raw_bindings: dict
    derivation_sha256: str
    derivation_module_sha256: str
    status: str
    segments: tuple[TRAINSemanticAcquireSegment, ...]
    derivation_rule_sha256: str
    derivation_rule_pointer: dict | None = None

    def __post_init__(self):
        from .semantic_acquire import TASKS
        if (self.task_name not in TASKS or not self.source_id or not self.parent_source_group
                or set(self.raw_bindings)!= {'canonical_receipt','cache','commands'}
                or any(set(p)!= {'path','sha256'} or not valid_sha(p['sha256']) for p in self.raw_bindings.values())
                or not all(valid_sha(x) for x in (self.derivation_sha256,self.derivation_module_sha256,
                    self.derivation_rule_sha256)) or not self.status
                or any(not isinstance(s,TRAINSemanticAcquireSegment) for s in self.segments)):
            raise InteractionArtifactError('explicit TRAIN semantic acquire raw/source evidence required')
        if self.segments:
            if (len(self.segments)!=2 or tuple(s.reference_mode for s in self.segments)!=('external','attached')
                    or self.segments[0].terminal_frame!=self.segments[1].frame_indices[0]
                    or self.segments[0].attempt_end_time!=self.segments[1].attempt_start_time
                    or len({i for s in self.segments for i in s.frame_indices})!=sum(len(s.frame_indices) for s in self.segments)
                    or self.status!='derived_mode_segments_pending_explicit_fitter_interface'):
                raise InteractionArtifactError('semantic acquire segments overlap or erase an actual mode boundary')
        object.__setattr__(self,'raw_bindings',_json(self.raw_bindings))


@dataclass(frozen=True)
class TRAINInteractionSource:
    task_name: str
    source_id: str
    split: str
    source_kind: str  # TRAIN_demonstration, TRAIN_native_generated_augmentation, synthetic (tests only)
    times: np.ndarray
    expert_tcp_xyzw: np.ndarray
    expert_opening: np.ndarray
    measured_tcp_xyzw: np.ndarray
    stages: tuple[str, ...]
    references: tuple[TRAINReferenceBinding | None, ...]
    body_tracks: dict[str, TRAINBodyTrack]
    parent_source_group: str | None = None
    augmentation_id: str | None = None
    raw_episode_id: str | None = None
    parent_raw_episode_id: str | None = None
    command_ids: np.ndarray | None = None
    completion_events: tuple[TRAINPhaseCompletionEvent, ...] = ()
    curve_tcp_xyzw: np.ndarray | None = None
    curve_label_kind: str = 'raw_expert_waypoint'
    command_ledger: tuple[dict, ...] = ()
    excluded_completion_events: tuple[dict, ...] = ()
    semantic_acquire_evidence: TRAINSemanticAcquireEvidence | None = None
    command_goal_reference_evidence: object | None = None

    def __post_init__(self):
        t = np.asarray(self.times, float)
        n = len(t)
        if self.split != 'train' or self.source_kind not in (
                'TRAIN_demonstration', 'TRAIN_native_generated_augmentation', 'synthetic'):
            raise InteractionArtifactError('interaction fitting accepts TRAIN supervision only')
        if (self.source_kind == 'TRAIN_native_generated_augmentation'
                and (not self.parent_source_group or not self.augmentation_id
                     or not self.raw_episode_id or not self.parent_raw_episode_id)):
            raise InteractionArtifactError('generated TRAIN augmentation needs explicit parent group and rollout ID')
        if self.parent_raw_episode_id and not self.raw_episode_id:
            raise InteractionArtifactError('parent raw identity requires a separate rollout raw identity')
        if (self.parent_source_group and self.raw_episode_id
                and self.parent_source_group != self.grouping_source):
            raise InteractionArtifactError('parent source group must equal (train, task, parent raw episode)')
        if (not self.source_id or t.ndim != 1 or n < 1 or np.any(np.diff(t) <= 0)
                or not np.isfinite(t).all() or len(self.stages) != n or len(self.references) != n):
            raise InteractionArtifactError('invalid source-aligned TRAIN interaction arrays')
        for field in ('expert_tcp_xyzw', 'measured_tcp_xyzw'):
            value = np.asarray(getattr(self, field), float)
            if (value.shape != (n, 7) or not np.isfinite(value).all()
                    or np.any(np.linalg.norm(value[:, 3:], axis=1) < 1e-8)):
                raise InteractionArtifactError('TRAIN expert/actual TCP poses must be distinct finite SE3 arrays')
            object.__setattr__(self, field, value.copy())
        opening = np.asarray(self.expert_opening, float)
        if opening.shape != (n,) or not np.isfinite(opening).all() or np.any((opening < 0) | (opening > 1)):
            raise InteractionArtifactError('invalid TRAIN expert gripper actions')
        for body_id, track in self.body_tracks.items():
            if body_id != track.object_id:
                raise InteractionArtifactError('body track key differs from stable object identity')
        object.__setattr__(self, 'times', t.copy())
        object.__setattr__(self, 'expert_opening', opening.copy())
        if self.curve_label_kind not in ('raw_expert_waypoint','measured_executed_TCP'):
            raise InteractionArtifactError('explicit interaction curve-label semantics required')
        if self.curve_tcp_xyzw is not None:
            value=np.asarray(self.curve_tcp_xyzw,float)
            if (value.shape != (n,7) or not np.isfinite(value).all()
                    or np.any(np.linalg.norm(value[:,3:],axis=1) < 1e-8)):
                raise InteractionArtifactError('finite explicit curve TCP labels required')
            object.__setattr__(self,'curve_tcp_xyzw',value.copy())
        if self.command_ids is not None:
            ids=np.asarray(self.command_ids)
            if ids.shape != (n,) or ids.dtype.kind not in 'iu' or np.any(ids < -1):
                raise InteractionArtifactError('aligned executed command identities required')
            object.__setattr__(self,'command_ids',ids.astype(np.int64,copy=True))
        if self.completion_events and self.command_ids is None:
            raise InteractionArtifactError('completion events require command-aligned TRAIN observations')
        ledger_ids=[row.get('command_id') for row in self.command_ledger]
        if (any(type(c) is not int or c < 0 for c in ledger_ids) or len(set(ledger_ids)) != len(ledger_ids)):
            raise InteractionArtifactError('unique raw TRAIN command ledger identities required')
        object.__setattr__(self,'command_ledger',tuple(_json(row) for row in self.command_ledger))
        object.__setattr__(self,'excluded_completion_events',tuple(_json(row) for row in self.excluded_completion_events))
        phase_order=tuple(phase.name for phase in get_task_spec(self.task_name).phase_skills)
        required=tuple(p.name for p in get_task_spec(self.task_name).phase_skills if not p.optional_after_native_completion)
        seen=set()
        for event in self.completion_events:
            if not isinstance(event,TRAINPhaseCompletionEvent) or event.origin_command_id in seen:
                raise InteractionArtifactError('unique typed phase completion events required')
            seen.add(event.origin_command_id)
            if event.phase not in phase_order:
                raise InteractionArtifactError('completion references an undeclared phase')
            if event.evidence == 'completed_command_then_ordered_teacher_phase_transition':
                index=phase_order.index(event.phase)
                if (not event.command_completed or index+1 >= len(phase_order)
                        or event.to_phase != phase_order[index+1]):
                    raise InteractionArtifactError('only completed commands followed by adjacent forward phase count')
            elif event.phase != required[-1] or event.to_phase != 'native_success':
                raise InteractionArtifactError('native success can label only the final required executed phase')
            if (event.attempt_start_time < t[0]-1e-8 or event.attempt_end_time < t[0]
                    or event.attempt_start_time > t[-1]+1e-8
                    or not np.any(self.command_ids == event.origin_command_id)):
                raise InteractionArtifactError('completion event lies outside its executed TRAIN command')
        semantic=self.semantic_acquire_evidence
        if semantic is not None:
            if (not isinstance(semantic,TRAINSemanticAcquireEvidence)
                    or semantic.task_name!=self.task_name or semantic.source_id!=self.source_id
                    or semantic.parent_source_group!=self.grouping_source or self.command_ids is None):
                raise InteractionArtifactError('semantic acquire evidence source/task/original parent differs')
            semantic_phase_map={stage:phase.name for phase in get_task_spec(self.task_name).phase_skills
                for stage in phase.TRAIN_expert_stages}
            for segment in semantic.segments:
                indices=segment.frame_indices;end=segment.terminal_frame
                if (end>=n or abs(t[indices[0]]-segment.attempt_start_time)>1e-8
                        or abs(t[end]-segment.attempt_end_time)>1e-8
                        or tuple(dict.fromkeys(map(int,self.command_ids[list(indices)])))!=segment.raw_command_ids
                        or any(semantic_phase_map.get(self.stages[i])!='acquire'
                            or self.references[i]!=segment.binding for i in indices)):
                    raise InteractionArtifactError('semantic segment frames/commands/reference differ from actual source')
                track=self.body_tracks.get(segment.binding.reference_id)
                if track is None:
                    raise InteractionArtifactError('semantic segment physical body track missing')
                for i in indices:
                    track.at(t[i],orientation_required=False,same_mode_as=segment.reference_mode,not_before=t[i])
                terminal_reference=track.at(t[end],orientation_required=False,same_mode_as='attached',not_before=t[end])
                terminal_reference=Pose(terminal_reference.position,np.eye(3))
                if not np.allclose(terminal_reference.xyzw(),segment.terminal_reference_xyzw,rtol=0,atol=1e-8):
                    raise InteractionArtifactError('semantic terminal reference differs from original metric track')
                if (not np.allclose(self.measured_tcp_xyzw[indices[0]],segment.origin_measured_tcp_xyzw,rtol=0,atol=1e-8)
                        or not np.allclose(self.measured_tcp_xyzw[end],segment.terminal_tcp_xyzw,rtol=0,atol=1e-8)):
                    raise InteractionArtifactError('semantic segment TCP snapshot differs from actual acquired source')
        goal=self.command_goal_reference_evidence
        if goal is not None:
            from .command_goal_reference import TRAINCommandGoalEvidence
            if (not isinstance(goal,TRAINCommandGoalEvidence) or goal.task_name!=self.task_name
                    or goal.source_id!=self.source_id or goal.parent_source_group!=self.grouping_source):
                raise InteractionArtifactError('command-goal evidence source/task/parent differs')
            ledger={row['command_id']:row for row in self.command_ledger}
            for reference in goal.references:
                row=ledger.get(reference.command_id)
                if row is None or row['generated_at']!=reference.issued_at or reference.binding!=TRAINReferenceBinding(
                    row['reference_role'],row['reference_body'],row['reference_kind'],row['anchor_id'],row['manipulated_role']):
                    raise InteractionArtifactError('command-goal reference differs from original command binding/clock')

    @property
    def grouping_source(self):
        if self.raw_episode_id:
            return ':'.join(('train', self.task_name, self.parent_raw_episode_id or self.raw_episode_id))
        return self.parent_source_group or self.source_id


def bind_verified_semantic_acquire(source,verification,*,rule_sha256,rule_pointer=None):
    """Bind a separately raw-recomputed TRAIN derivation; no raw IDs rewritten."""
    from . import semantic_acquire as semantic
    from . import acquisition_lineage as lineage
    if verification.get('raw_evidence_verified') is not True:
        raise InteractionArtifactError('semantic acquire binding needs actual archived/raw recomputation')
    context=verification.get('validation_context')
    if not isinstance(context,dict):
        raise InteractionArtifactError('semantic acquire binding lacks its actual archived validation context')
    root=Path(context['root']).resolve();files={}
    auxiliary_pointer=context.get('auxiliary_registration')
    if auxiliary_pointer is not None:
        from . import teacher_supplement
        auxiliary=teacher_supplement.verify_registration(root,auxiliary_pointer,live=False,require_ready=True)
        actual=auxiliary['current']
    else:actual=lineage.load_archived_plan(root,context['actual_registration'],files)
    expected=next((c for c in actual['plan']['cases'] if c['case_id']==context['expected_case_id']),None)
    if expected is None:
        raise InteractionArtifactError('semantic acquisition not in its actual archived roster')
    previous=None
    if context['predecessor_registration'] is not None:
        previous=lineage.load_archived_plan(root,context['predecessor_registration'],files)
        lineage.verify_recovery_transition(root,previous,actual,files)
    verification=semantic.verify_derived_receipt(root,verification['receipt'],registered=actual,
        expected_case=expected,predecessor=previous,preceding=context['preceding'],auxiliary_registration=auxiliary_pointer)
    receipt=verification['receipt'];arrays=verification['verified_arrays']
    if (receipt['task']!=source.task_name or receipt['source_id']!=source.source_id
            or receipt['parent_source_group']!=source.grouping_source):
        raise InteractionArtifactError('semantic acquire derivation belongs to another actual source')
    for value,raw in ((source.times,arrays['times']),
            (source.measured_tcp_xyzw,arrays['measured_tcp_xyzw']),
            (source.curve_tcp_xyzw,arrays['measured_tcp_xyzw']),
            (source.expert_opening,arrays['teacher_applied_opening']),
            (source.command_ids,np.where(arrays['teacher_action_valid'],arrays['teacher_command_id'],-1))):
        if not np.array_equal(value,raw):
            raise InteractionArtifactError('semantic interaction source arrays differ from verified raw cache')
    j=list(map(str,arrays['body_ids'])).index('ball');track=source.body_tracks.get('ball')
    if (track is None or not np.array_equal(track.times,arrays['times'])
            or not np.allclose(track.poses_xyzw[:,:3],arrays['target_positions'][:,j],rtol=0,atol=1e-8)
            or not np.allclose(Rotation.from_quat(track.poses_xyzw[:,3:]).as_matrix(),
                Rotation.from_matrix(arrays['target_rotations'][:,j]).as_matrix(),rtol=0,atol=1e-8)
            or track.motion_modes!=tuple(('external','attached','released')[int(m)] for m in arrays['modes'][:,j])
            or not np.array_equal(track.pose_valid,arrays['native_pose_valid'][:,j]&arrays['mode_valid'][:,j])):
        raise InteractionArtifactError('semantic physical body track differs from verified native TRAIN labels')
    segments=[]
    for value in verification['segments']:
        indices=value.frame_indices;binding=source.references[indices[0]]
        if binding is None:
            raise InteractionArtifactError('semantic acquire segment has no valid original reference')
        track=source.body_tracks[binding.reference_id];end=value.terminal_frame
        if not track.pose_valid[end]:
            raise InteractionArtifactError('semantic segment terminal reference metric label invalid')
        reference=Pose.from_xyzw(track.poses_xyzw[end])
        if binding.reference_kind=='translation_only':reference=Pose(reference.position,np.eye(3))
        sid=f'acquire:{value.mode}:{indices[0]}-{indices[-1]}:terminal{end}'
        segments.append(TRAINSemanticAcquireSegment(sid,'acquire',binding,value.mode,indices,
            value.raw_command_ids,end,value.start_time,value.end_time,source.measured_tcp_xyzw[indices[0]],
            source.measured_tcp_xyzw[end],reference.xyzw(),value.terminal_kind))
    digest=hashlib.sha256(json.dumps(receipt,sort_keys=True,separators=(',',':'),ensure_ascii=False,
        allow_nan=False).encode()).hexdigest()
    evidence=TRAINSemanticAcquireEvidence(source.task_name,source.source_id,source.grouping_source,
        receipt['raw_bindings'],digest,hashlib.sha256(Path(semantic.__file__).read_bytes()).hexdigest(),
        receipt['status'],tuple(segments),rule_sha256,rule_pointer)
    return replace(source,semantic_acquire_evidence=evidence)


@dataclass(frozen=True)
class InteractionFitConfig:
    progress_knots: int
    lag_grid_seconds: tuple[float, ...]
    duration_quantile: float
    tolerance_quantile: float
    minimum_sources: int
    minimum_position_tolerance_m: float
    minimum_orientation_tolerance_rad: float
    minimum_lag_sample_fraction: float
    minimum_completed_parent_sources: int | None = None
    maximum_position_tolerance_m: float | None = None
    maximum_orientation_tolerance_rad: float | None = None
    development_only: bool = False

    def __post_init__(self):
        if (not np.isfinite([*self.lag_grid_seconds, self.duration_quantile,
                self.tolerance_quantile, self.minimum_position_tolerance_m,
                self.minimum_orientation_tolerance_rad, self.minimum_lag_sample_fraction]).all()
                or self.progress_knots < 2 or self.minimum_sources < 2
                or not self.lag_grid_seconds or min(self.lag_grid_seconds) < 0
                or 0. not in self.lag_grid_seconds
                or not 0 < self.duration_quantile <= 1 or not 0 < self.tolerance_quantile <= 1
                or min(self.minimum_position_tolerance_m, self.minimum_orientation_tolerance_rad) <= 0
                or not 0 < self.minimum_lag_sample_fraction <= 1):
            raise InteractionArtifactError('explicit valid TRAIN fitting configuration required')
        optional=(self.minimum_completed_parent_sources,self.maximum_position_tolerance_m,
                  self.maximum_orientation_tolerance_rad)
        if any(value is not None for value in optional):
            if (any(value is None for value in optional) or
                    type(self.minimum_completed_parent_sources) is not int or
                    self.minimum_completed_parent_sources < 2 or
                    not np.isfinite(optional).all() or
                    self.maximum_position_tolerance_m < self.minimum_position_tolerance_m or
                    self.maximum_orientation_tolerance_rad < self.minimum_orientation_tolerance_rad):
                raise InteractionArtifactError('explicit completed-parent support and finite terminal caps required')
        if type(self.development_only) is not bool:
            raise InteractionArtifactError('development_only must be an explicit boolean')


@dataclass(frozen=True)
class PhaseTemplate:
    phase: str
    anchor_id: str
    reference_role: str
    reference_kind: str
    manipulated_role: str
    lag_seconds: float
    progress: np.ndarray
    positions: np.ndarray
    rotations: np.ndarray
    opening: np.ndarray
    duration_seconds: float
    duration_budget_seconds: float
    position_tolerance_m: float
    orientation_tolerance_rad: float
    reference_mode: str
    reference_time_basis: str = CURRENT
    reference_time_rule_sha256: str | None = None
    completion_geometry_xyzw: np.ndarray | None = None

    def __post_init__(self):
        validate_template_basis(self.reference_time_basis,self.reference_kind,self.reference_time_rule_sha256)
        if self.reference_time_basis==COMMAND_GOAL:
            if self.lag_seconds!=0. or self.completion_geometry_xyzw is None:
                raise InteractionArtifactError('fixed command-goal template needs canonical public completion geometry')
            object.__setattr__(self,'completion_geometry_xyzw',Pose.from_xyzw(self.completion_geometry_xyzw).xyzw())
        elif self.completion_geometry_xyzw is not None:
            raise InteractionArtifactError('legacy template cannot carry undeclared command-goal completion geometry')
        p = np.asarray(self.progress, float)
        values = {'positions': np.asarray(self.positions, float),
                  'rotations': np.asarray(self.rotations, float), 'opening': np.asarray(self.opening, float)}
        if (not np.isfinite([self.lag_seconds, self.duration_seconds, self.duration_budget_seconds,
                            self.position_tolerance_m, self.orientation_tolerance_rad]).all()
                or len(p) < 2 or p[0] != 0 or p[-1] != 1 or np.any(np.diff(p) <= 0)
                or values['positions'].shape != (len(p), 3)
                or values['rotations'].shape != (len(p), 3, 3)
                or values['opening'].shape != p.shape
                or not all(np.isfinite(x).all() for x in (p, *values.values()))
                or np.any((values['opening'] < 0) | (values['opening'] > 1))
                or self.reference_kind not in ('observed_se3', 'translation_only', 'phase_origin', 'world')
                or self.reference_mode not in ('external', 'attached', 'released', 'not_applicable')
                or (self.reference_kind in ('phase_origin', 'world')) != (self.reference_mode == 'not_applicable')
                or self.lag_seconds < 0 or self.duration_seconds <= 0
                or self.duration_budget_seconds < self.duration_seconds
                or min(self.position_tolerance_m, self.orientation_tolerance_rad) <= 0):
            raise InteractionArtifactError('invalid fitted phase curve')
        for rotation in values['rotations']:
            Pose(np.zeros(3), rotation)
        object.__setattr__(self, 'progress', p.copy())
        for key, value in values.items():
            object.__setattr__(self, key, value.copy())

    def at(self, progress):
        u = float(np.clip(progress, 0., 1.))
        i = min(len(self.progress)-2, max(0, int(np.searchsorted(self.progress, u)-1)))
        span = self.progress[i+1]-self.progress[i]
        fraction = (u-self.progress[i])/span
        position = (1-fraction)*self.positions[i]+fraction*self.positions[i+1]
        rotation = Slerp(self.progress, Rotation.from_matrix(self.rotations.copy()))([u]).as_matrix()[0]
        velocity = (self.positions[i+1]-self.positions[i])/(span*self.duration_seconds)
        angular = Rotation.from_matrix(self.rotations[i+1] @ self.rotations[i].T).as_rotvec()/(span*self.duration_seconds)
        opening = float(np.interp(u, self.progress, self.opening))
        return Pose(position, rotation), velocity, angular, opening


class InteractionLibrary:
    def __init__(self, payload, *, allow_synthetic=False):
        self.metadata = payload['metadata']
        m = self.metadata
        if (m.get('schema') != LIBRARY_SCHEMA or m.get('training_complete') is not True
                or m.get('fit_split') != 'train' or m.get('selection_split') != 'train'
                or m.get('TEST_used_for_fitting') is not False
                or (m.get('synthetic') is not False and not allow_synthetic)
                or not m.get('TRAIN_source_ids')):
            raise InteractionArtifactError('completed TRAIN-fitted interaction library required')
        if (m.get('fitting_module_sha256') != hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
                or m.get('task_spec_module_sha256') != hashlib.sha256((Path(__file__).parent/'specs.py').read_bytes()).hexdigest()):
            raise InteractionArtifactError('interaction fitting/schema source differs from artifact')
        self.task_spec = get_task_spec(m['task_name'])
        declared_order = tuple(phase.name for phase in self.task_spec.phase_skills)
        self.phase_order = tuple(m.get('fitted_phase_order', ()))
        omitted = self.task_spec.phase_skills[len(self.phase_order):]
        if (not self.phase_order or self.phase_order != declared_order[:len(self.phase_order)]
                or any(not phase.optional_after_native_completion for phase in omitted)
                or {row['phase'] for row in m.get('omitted_optional_phases', ())} != {phase.name for phase in omitted}):
            raise InteractionArtifactError('only missing trailing native-optional phases may be omitted')
        self.templates = tuple(PhaseTemplate(**item) for item in payload['templates'])
        if any(t.reference_time_basis==COMMAND_GOAL for t in self.templates):
            proof=m.get('command_goal_reference_derivation',{})
            helper=Path(__file__).parent/'command_goal_reference.py'
            if (proof.get('helper_sha256')!=hashlib.sha256(helper.read_bytes()).hexdigest()
                    or sorted({t.reference_time_rule_sha256 for t in self.templates if t.reference_time_basis==COMMAND_GOAL})
                    !=proof.get('rule_sha256_values')
                    or m.get('public_completion_verifier_calibrated') is not False):
                raise InteractionArtifactError('command-goal library lacks matching typed helper/rule/public-verifier scope')
        if any(t.reference_time_basis==COMMAND_ISSUE for t in self.templates):
            if (m.get('reference_time_schema')!=REFERENCE_TIME_SCHEMA
                    or m.get('reference_time_helper_sha256')!=hashlib.sha256((Path(__file__).parent/'reference_time.py').read_bytes()).hexdigest()
                    or sorted({t.reference_time_rule_sha256 for t in self.templates if t.reference_time_basis==COMMAND_ISSUE})
                       !=m.get('reference_time_rule_sha256_values')):
                raise InteractionArtifactError('command_issue library lacks matching explicit reference-time source/rule binding')
        if set(t.phase for t in self.templates) != set(self.phase_order):
            raise InteractionArtifactError('missing TRAIN-fitted phase; no untrained default skill allowed')
        self.payload = _json(payload)
        self.sha256 = hashlib.sha256(json.dumps(self.payload, sort_keys=True, allow_nan=False).encode()).hexdigest()
        if m.get('synthetic') is False:
            path = Path(m['TRAIN_source_manifest_path'])
            if hashlib.sha256(path.read_bytes()).hexdigest() != m['TRAIN_source_manifest_sha256']:
                raise InteractionArtifactError('TRAIN source manifest changed')

    @classmethod
    def load(cls, path):
        path = Path(path)
        if not path.is_file():
            raise InteractionArtifactError('TRAIN interaction artifact is missing: ' + str(path))
        return cls(json.loads(path.read_text()))

    def save(self, path):
        path = Path(path)
        if path.exists():
            raise InteractionArtifactError('do not overwrite a fitted interaction artifact')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.payload, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False)+'\n')

    def for_phase(self, phase):
        return tuple(template for template in self.templates if template.phase == phase)

    def suffix_budget(self, phase_index):
        return sum(max(t.duration_budget_seconds for t in self.for_phase(phase))
                   for phase in self.phase_order[phase_index+1:])


def _fit_curve(samples, progress_grid):
    positions, rotations, opening = [], [], []
    source_counts = Counter(row['source'] for row in samples)
    width = 1./(len(progress_grid)-1)
    for u in progress_grid:
        chosen = [row for row in samples if abs(row['progress']-u) <= width]
        if not chosen:
            chosen = [min(samples, key=lambda row: abs(row['progress']-u))]
        weights = np.array([1./source_counts[row['source']] for row in chosen])
        weights /= weights.sum()
        positions.append(np.average([row['position'] for row in chosen], axis=0, weights=weights))
        rotations.append(Rotation.from_matrix(np.array([row['rotation'] for row in chosen])).mean(weights).as_matrix())
        opening.append(np.average([row['opening'] for row in chosen], weights=weights))
    return np.asarray(positions), np.asarray(rotations), np.asarray(opening)


def _fit_completed_curve(samples,progress_grid):
    """Completed attempts provide an explicit endpoint, not a transient window."""
    p,r,o=_fit_curve(samples,progress_grid)
    terminal=[row for row in samples if row.get('terminal_event')]
    if not terminal:
        raise InteractionArtifactError('a completed curve requires current-reference terminal labels')
    counts=Counter(row['source'] for row in terminal)
    weights=np.asarray([1./counts[row['source']] for row in terminal]);weights/=weights.sum()
    p[-1]=np.average([row['position'] for row in terminal],axis=0,weights=weights)
    r[-1]=Rotation.from_matrix(np.asarray([row['rotation'] for row in terminal])).mean(weights).as_matrix()
    o[-1]=np.average([row['opening'] for row in terminal],weights=weights)
    return p,r,o


def _completion_reference(source,event,lag):
    binding=event.binding
    if binding.reference_time_basis==COMMAND_ISSUE:
        if lag!=0.:
            raise InteractionArtifactError('command_issue reference has no forward motion lag')
        if event.issued_reference_xyzw is None:
            raise InteractionArtifactError('command_issue reference pose is missing')
        return Pose.from_xyzw(event.issued_reference_xyzw)
    if lag == 0. or binding.reference_kind in ('world','phase_origin'):
        if lag and binding.reference_kind in ('world','phase_origin'):
            raise InteractionArtifactError('fixed references have no motion lag')
        if binding.reference_kind=='observed_se3':
            track=source.body_tracks[binding.reference_id]
            index=int(np.searchsorted(track.times,event.attempt_end_time))
            if (index>=len(track.times) or abs(track.times[index]-event.attempt_end_time)>1e-8
                    or not track.orientation_observable[index]):
                raise InteractionArtifactError('terminal interaction reference lacks full orientation supervision')
        return Pose.from_xyzw(event.terminal_reference_xyzw)
    # Positive lag requires real continuous same-mode body supervision. The
    # explicit terminal snapshot alone cannot manufacture future attachment or
    # external dynamics after native termination.
    track=source.body_tracks[binding.reference_id]
    reference=track.at(event.attempt_end_time+lag,
        orientation_required=binding.reference_kind=='observed_se3',
        same_mode_as=event.reference_mode,not_before=event.attempt_end_time)
    return Pose(reference.position,np.eye(3)) if binding.reference_kind=='translation_only' else reference


def _semantic_acquire_curve_rows(source,segment,config,attempt_id):
    """Actual same-mode rows plus one explicitly acquired metric terminal."""
    binding=segment.binding;event_rows={};excluded=Counter()
    duration=segment.attempt_end_time-segment.attempt_start_time
    for lag in config.lag_grid_seconds:
        try:terminal_reference=_completion_reference(source,segment,float(lag))
        except (InteractionArtifactError,KeyError):
            excluded['lag_terminal_reference_unavailable']+=1;continue
        rows=[]
        for i in segment.frame_indices:
            try:
                reference=source.body_tracks[binding.reference_id].at(source.times[i]+lag,
                    orientation_required=False,same_mode_as=segment.reference_mode,not_before=source.times[i])
                reference=Pose(reference.position,np.eye(3))
                curve=Pose.from_xyzw(source.curve_tcp_xyzw[i])
            except (InteractionArtifactError,KeyError):
                excluded['invalid_mode_gap_or_reference_row']+=1;continue
            rows.append(dict(source=source.grouping_source,rollout_id=source.source_id,attempt_id=attempt_id,
                progress=float((source.times[i]-segment.attempt_start_time)/duration),
                position=reference.rotation.T@(curve.position-reference.position),
                rotation=reference.rotation.T@curve.rotation,opening=source.expert_opening[i],terminal_event=False))
        if not rows:continue
        terminal=Pose.from_xyzw(segment.terminal_tcp_xyzw)
        rows.append(dict(source=source.grouping_source,rollout_id=source.source_id,attempt_id=attempt_id,
            progress=1.,position=terminal_reference.rotation.T@(terminal.position-terminal_reference.position),
            rotation=terminal_reference.rotation.T@terminal.rotation,
            opening=source.expert_opening[segment.frame_indices[-1]],terminal_event=True))
        event_rows[float(lag)]=rows
    return event_rows,excluded


def _fit_completed_interactions(sources,config,*,TRAIN_source_manifest_path,synthetic):
    spec=get_task_spec(sources[0].task_name)
    if (config.minimum_completed_parent_sources is None or config.maximum_position_tolerance_m is None
            or config.maximum_orientation_tolerance_rad is None):
        raise InteractionArtifactError('production completed-attempt fitting needs explicit parent support and terminal caps')
    if not config.development_only and config.minimum_completed_parent_sources < 8:
        raise InteractionArtifactError('formal completion fitting requires at least eight completed TRAIN parents per phase')
    manifest_hash=None
    if not synthetic:
        if TRAIN_source_manifest_path is None:
            raise InteractionArtifactError('production fit requires an actual TRAIN acquisition manifest')
        path=Path(TRAIN_source_manifest_path);manifest=json.loads(path.read_text());entries=manifest.get('entries',[])
        if not entries or any(row.get('split')!='train' for row in entries):
            raise InteractionArtifactError('interaction fit manifest must contain TRAIN acquisitions only')
        declared={row['source_id']:row for row in entries if row.get('task')==spec.task_name
            and row.get('interaction_eligible',True)}
        if set(declared)!={source.source_id for source in sources}:
            raise InteractionArtifactError('every declared teacher acquisition must be audited, including incomplete attempts')
        for source in sources:
            row=declared[source.source_id]
            group=':'.join(('train',row['task'],row.get('parent_raw_episode_id') or row['raw_episode_id']))
            if group!=source.grouping_source:
                raise InteractionArtifactError('TRAIN source parent groups differ from acquisition manifest')
        manifest_hash=hashlib.sha256(path.read_bytes()).hexdigest()
    stage_phase={stage:phase.name for phase in spec.phase_skills for stage in phase.TRAIN_expert_stages}
    semantic_flags={source.semantic_acquire_evidence is not None for source in sources}
    if semantic_flags=={True,False}:
        raise InteractionArtifactError('cannot mix legacy and semantic acquire supervision within one task fit')
    semantic_enabled=semantic_flags=={True}
    if semantic_enabled and len({s.semantic_acquire_evidence.derivation_rule_sha256 for s in sources})!=1:
        raise InteractionArtifactError('mixed semantic acquire derivation rules within one task fit')
    groups={};durations={};event_records={};curation=[];failures=[]
    goal_flags={source.command_goal_reference_evidence is not None for source in sources}
    if goal_flags=={True,False}:
        raise InteractionArtifactError('cannot mix legacy and command-goal evidence within one task fit')
    if goal_flags=={True} and len({source.command_goal_reference_evidence.rule_sha256 for source in sources})!=1:
        raise InteractionArtifactError('cannot mix command-goal rules within one task fit')
    goal_records={};canonical_groups={}
    for source in sources:
        goal_evidence=source.command_goal_reference_evidence
        command_goals={} if goal_evidence is None else {row.command_id:row for row in goal_evidence.references}
        pending_goals=set() if goal_evidence is None else {row['command_id'] for row in goal_evidence.pending_commands}
        if source.command_ids is None or source.curve_tcp_xyzw is None or source.curve_label_kind!='measured_executed_TCP':
            raise InteractionArtifactError('production BC requires explicit executed TCP curve labels and command IDs')
        events={event.origin_command_id:event for event in source.completion_events}
        command_ids=set(map(int,source.command_ids[source.command_ids>=0]))
        command_ids.update(row['command_id'] for row in source.command_ledger)
        for command_id in sorted(command_ids):
            indices=np.flatnonzero(source.command_ids==command_id)
            event=events.get(command_id)
            ledger=next((row for row in source.command_ledger if row['command_id']==command_id),{})
            raw_phase=event.phase if event is not None else ledger.get('raw_phase',
                stage_phase.get(source.stages[indices[0]]) if len(indices) else None)
            if semantic_enabled and raw_phase=='acquire':
                excluded_original=next((row for row in source.excluded_completion_events
                    if row['command_id']==command_id),None)
                begin=float(ledger.get('generated_at',source.times[indices[0]] if len(indices) else source.times[0]))
                end=float(ledger.get('cutoff_at',ledger.get('last_executed_at',source.times[-1])))
                curation.append(dict(source_id=source.source_id,parent_source_group=source.grouping_source,
                    command_id=command_id,phase='acquire',included_in_BC=False,
                    reason='represented_by_verified_semantic_acquire_evidence',
                    command_waypoint_completed='command_completed_at' in ledger,
                    original_phase_completion_event_present=event is not None or excluded_original is not None,
                    original_completion_label_exclusion=excluded_original,
                    duration_is_right_censored_lower_bound=event is None and excluded_original is None,
                    observed_duration_lower_bound_seconds=max(0.,end-begin),
                    original_completion_events_unchanged=True,raw_rows_preserved=True,
                    semantic_evidence_status=source.semantic_acquire_evidence.status,
                    legacy_acquire_curve_rows_double_counted=False))
                continue
            if event is None:
                excluded_event=next((row for row in source.excluded_completion_events if row['command_id']==command_id),None)
                begin=float(ledger.get('generated_at',source.times[indices[0]] if len(indices) else source.times[0]))
                end=float(ledger.get('cutoff_at',ledger.get('last_executed_at',source.times[-1])))
                curation.append(dict(source_id=source.source_id,parent_source_group=source.grouping_source,
                    command_id=command_id,phase=ledger.get('raw_phase'),included_in_BC=False,
                    reason='no_explicit_successful_skill_completion_event' if excluded_event is None else excluded_event['reason'],
                    command_waypoint_completed='command_completed_at' in ledger,
                    skill_completion_observed=excluded_event is not None,
                    duration_is_right_censored_lower_bound=excluded_event is None,
                    observed_duration_lower_bound_seconds=max(0.,end-begin),raw_rows_preserved=True))
                continue
            if not len(indices):
                failures.append(dict(source_id=source.source_id,command_id=command_id,reason='completion has no executed rows'))
                continue
            if command_id in pending_goals:
                failures.append(dict(source_id=source.source_id,command_id=command_id,
                    reason='completed original command has invalid command-goal intent supervision'))
                continue
            binding=event.binding
            command_goal=command_goals.get(command_id)
            if (event.attempt_end_time>source.times[-1]+1e-8 or
                    any(stage_phase.get(source.stages[i])!=event.phase or source.references[i]!=binding for i in indices
                        if source.references[i] is not None)):
                failures.append(dict(source_id=source.source_id,command_id=command_id,reason='completion interval/phase/reference differs'))
                continue
            indices=np.asarray([i for i in indices if source.references[i]==binding and
                event.attempt_start_time-1e-8<=source.times[i]<event.attempt_end_time-1e-8],int)
            if not len(indices):
                failures.append(dict(source_id=source.source_id,command_id=command_id,reason='completion has no valid reference rows'))
                continue
            key=(event.phase,binding.anchor_id,binding.reference_role,binding.reference_kind,
                binding.manipulated_role,event.reference_mode,COMMAND_GOAL if command_goal else binding.reference_time_basis)
            attempt_id=source.source_id+':command'+str(command_id)
            duration=float(event.attempt_end_time-event.attempt_start_time)
            event_rows={};excluded=Counter()
            for lag in config.lag_grid_seconds:
                if command_goal and lag!=0.:
                    excluded['fixed_command_goal_has_no_forward_lag']+=1;continue
                try: terminal_reference=(Pose.from_xyzw(command_goal.intent_reference_xyzw)
                    if command_goal else _completion_reference(source,event,float(lag)))
                except (InteractionArtifactError,KeyError):
                    excluded['lag_terminal_reference_unavailable']+=1;continue
                rows=[]
                for i in indices:
                    try:
                        if binding.reference_kind=='world':reference=Pose(np.zeros(3))
                        elif binding.reference_kind=='phase_origin':reference=Pose.from_xyzw(event.origin_measured_tcp_xyzw)
                        else:
                            track=source.body_tracks[binding.reference_id]
                            reference=track.at(source.times[i]+lag,
                                orientation_required=binding.reference_kind=='observed_se3',
                                same_mode_as=event.reference_mode,not_before=source.times[i])
                            if binding.reference_kind=='translation_only':reference=Pose(reference.position,np.eye(3))
                            if command_goal or binding.reference_time_basis==COMMAND_ISSUE:
                                # Preserve original sample/mode validity; change
                                # only the declared coordinate reference time.
                                reference=terminal_reference
                        curve=Pose.from_xyzw(source.curve_tcp_xyzw[i])
                    except (InteractionArtifactError,KeyError):
                        excluded['invalid_mode_gap_or_reference_row']+=1;continue
                    rows.append(dict(source=source.grouping_source,rollout_id=source.source_id,attempt_id=attempt_id,
                        progress=float((source.times[i]-event.attempt_start_time)/duration),
                        position=reference.rotation.T@(curve.position-reference.position),
                        rotation=reference.rotation.T@curve.rotation,opening=source.expert_opening[i],terminal_event=False))
                if not rows:
                    continue
                terminal=Pose.from_xyzw(event.terminal_tcp_xyzw)
                # Opening is the last genuinely issued command, not a new
                # synthetic gripper label at the native terminal observation.
                rows.append(dict(source=source.grouping_source,rollout_id=source.source_id,attempt_id=attempt_id,
                    progress=1.,position=terminal_reference.rotation.T@(terminal.position-terminal_reference.position),
                    rotation=terminal_reference.rotation.T@terminal.rotation,
                    opening=source.expert_opening[indices[-1]],terminal_event=True))
                event_rows[float(lag)]=rows
            included=0. in event_rows
            curation.append(dict(source_id=source.source_id,parent_source_group=source.grouping_source,
                command_id=command_id,phase=event.phase,included_in_BC=included,
                reason='explicit_successful_skill_attempt' if included else 'completed_event_without_usable_reference_curve',
                completion_evidence=event.evidence,command_waypoint_completed=event.command_completed,
                completed_duration_seconds=duration,duration_is_right_censored_lower_bound=False,
                issue_reference_mode=event.reference_mode,
                reference_time_basis=COMMAND_GOAL if command_goal else binding.reference_time_basis,
                reference_time_rule_sha256=goal_evidence.rule_sha256 if command_goal else binding.reference_time_rule_sha256,
                reference_time_derivation=event.reference_time_provenance,
                terminal_reference_is_interaction_metric_label=True,
                terminal_reference_is_external_dynamics_observation=False,
                excluded_fit_rows=dict(excluded),raw_rows_preserved=True))
            if not included:
                continue
            durations.setdefault(key,[]).append((source.grouping_source,duration))
            event_records[attempt_id]=(source,event)
            if command_goal:
                goal_records[attempt_id]=command_goal
                reference=Pose.from_xyzw(command_goal.intent_reference_xyzw)
                old=Pose.from_xyzw(event.committed_world_waypoint_xyzw)
                canonical_groups.setdefault(key,[]).append(dict(source=source.grouping_source,progress=1.,terminal_event=True,
                    position=reference.rotation.T@(old.position-reference.position),rotation=reference.rotation.T@old.rotation,
                    opening=source.expert_opening[indices[-1]]))
            for lag,rows in event_rows.items():groups.setdefault((key,lag),[]).extend(rows)
        if semantic_enabled:
            evidence=source.semantic_acquire_evidence
            for segment in evidence.segments:
                binding=segment.binding
                key=('acquire',binding.anchor_id,binding.reference_role,binding.reference_kind,
                    binding.manipulated_role,segment.reference_mode,binding.reference_time_basis)
                attempt_id=source.source_id+':semantic:'+segment.segment_id
                event_rows,excluded=_semantic_acquire_curve_rows(source,segment,config,attempt_id)
                included=0. in event_rows
                duration=segment.attempt_end_time-segment.attempt_start_time
                curation.append(dict(source_id=source.source_id,parent_source_group=source.grouping_source,
                    phase='acquire',segment_id=segment.segment_id,attempt_id=attempt_id,
                    raw_command_ids=list(segment.raw_command_ids),actual_frame_indices=list(segment.frame_indices),
                    terminal_frame=segment.terminal_frame,reference_partition=list(key),included_in_BC=included,
                    reason='verified_semantic_acquire_mode_segment' if included else 'semantic_segment_without_usable_same_mode_curve',
                    completion_evidence=segment.evidence,completed_duration_seconds=duration,
                    duration_is_right_censored_lower_bound=False,issue_reference_mode=segment.reference_mode,
                    reference_time_basis=binding.reference_time_basis,reference_time_rule_sha256=binding.reference_time_rule_sha256,
                    terminal_reference_is_interaction_metric_label=True,
                    terminal_reference_is_external_dynamics_observation=False,
                    excluded_fit_rows=dict(excluded),raw_rows_preserved=True,
                    raw_bindings=evidence.raw_bindings,semantic_derivation_sha256=evidence.derivation_sha256,
                    legacy_acquire_curve_rows_double_counted=False))
                if not included:continue
                durations.setdefault(key,[]).append((source.grouping_source,duration))
                event_records[attempt_id]=(source,segment)
                for lag,rows in event_rows.items():groups.setdefault((key,lag),[]).extend(rows)
    report=dict(supervision='explicit_completed_skill_attempts_only',all_acquisitions_preserved=True,
        incomplete_attempt_durations_are_right_censored=True,curation=curation,invalid_events=list(failures),
        curve_labels='measured_executed_TCP',held_raw_waypoints_are_independent_teacher_decisions=False,
        phase_success_is_not_task_success=True,task_outcomes_do_not_filter_earlier_completed_skills=True,
        TRAIN_acquisition_sources=sorted(source.source_id for source in sources),
        completion_tolerance_source='parent-group crossfit current-reference terminal endpoint residual',
        old_committed_world_waypoint_residual_is_separate=True,phase_fits={})
    report['duration_semantics']='eligible completed command-level skill attempt interval; not total semantic phase path duration'
    if semantic_enabled:
        report['duration_semantics']='original command intervals outside acquire; verified acquired mode-segment intervals for acquire; censored raw command durations remain audit-only'
        report['semantic_acquire_derivation']=dict(default_legacy_behavior_changed=False,
            original_events_command_ids_masks_and_censoring_preserved=True,
            rule_sha256_values=sorted({s.semantic_acquire_evidence.derivation_rule_sha256 for s in sources}),
            helper_sha256_values=sorted({s.semantic_acquire_evidence.derivation_module_sha256 for s in sources}),
            per_source={s.source_id:dict(derivation_sha256=s.semantic_acquire_evidence.derivation_sha256,
                raw_bindings=s.semantic_acquire_evidence.raw_bindings,status=s.semantic_acquire_evidence.status,
                segment_ids=[x.segment_id for x in s.semantic_acquire_evidence.segments]) for s in sources})
    if goal_flags=={True}:
        from .command_goal_reference import evidence_proof
        report['command_goal_reference_derivation']=dict(
            schema='r10_TRAIN_command_goal_reference_intent_v1',
            rule_sha256_values=sorted({s.command_goal_reference_evidence.rule_sha256 for s in sources}),
            helper_sha256=hashlib.sha256((Path(__file__).parent/'command_goal_reference.py').read_bytes()).hexdigest(),
            original_raw_completion_metric_preserved=True,
            per_source={s.source_id:evidence_proof(s.command_goal_reference_evidence) for s in sources})
        report['completion_tolerance_source']='parent crossfit declared intent tracking; original actual-current metric remains separate'
        report['public_completion_verifier_calibrated']=False
    if failures:
        raise InteractionFittingPending('invalid explicit completion evidence',report)
    grid=np.linspace(0.,1.,config.progress_knots);templates=[]
    for key,completed in durations.items():
        label='|'.join(key);base=groups[(key,0.)];parents={r['source'] for r in base}
        rules={(event_records[r['attempt_id']][0].command_goal_reference_evidence.rule_sha256
            if r['attempt_id'] in goal_records else event_records[r['attempt_id']][1].binding.reference_time_rule_sha256) for r in base}
        if len(rules)!=1:
            raise InteractionArtifactError('mixed reference-time derivation rules within one fit partition')
        rule_sha=next(iter(rules))
        diagnostic=dict(completed_parent_sources=sorted(parents),completed_parent_count=len(parents),
            minimum_completed_parent_sources=config.minimum_completed_parent_sources,
            completed_attempt_count=len({r['attempt_id'] for r in base}),
            completed_rollouts=sorted({r['rollout_id'] for r in base}),reference_mode=key[-2],
            source_grouping='parent raw TRAIN source; all augmentations share a fold',
            reference_time_basis=key[-1],reference_time_rule_sha256=rule_sha)
        report['phase_fits'][label]=diagnostic
        if len(parents)<max(config.minimum_sources,config.minimum_completed_parent_sources):
            diagnostic.update(passed=False,reason='insufficient completed TRAIN parent support');continue
        folds={name:i%2 for i,name in enumerate(sorted(parents))};diagnostic['source_to_fold']=folds
        candidates=[]
        for lag in config.lag_grid_seconds:
            if key[-1] in (COMMAND_ISSUE,COMMAND_GOAL) and lag!=0.:
                candidates.append(dict(lag=float(lag),valid=False,reason='command_issue has a fixed reference; forward motion lag is inapplicable',samples=0))
                continue
            rows=groups.get((key,float(lag)),[])
            if ({r['attempt_id'] for r in rows}!={r['attempt_id'] for r in base} or
                    len(rows)<config.minimum_lag_sample_fraction*len(base)):
                candidates.append(dict(lag=lag,valid=False,reason='completed attempt/sample coverage differs',samples=len(rows)));continue
            errors=[];endpoint_errors=[]
            for fold in (0,1):
                fit=[r for r in rows if folds[r['source']]!=fold];held=[r for r in rows if folds[r['source']]==fold]
                p,r,o=_fit_completed_curve(fit,grid)
                for sample in held:
                    position=np.asarray([np.interp(sample['progress'],grid,p[:,a]) for a in range(3)])
                    rotation=Slerp(grid,Rotation.from_matrix(r))([sample['progress']]).as_matrix()[0]
                    pe=float(np.linalg.norm(position-sample['position']))
                    re=float(Rotation.from_matrix(rotation@sample['rotation'].T).magnitude())
                    errors.append((sample['source'],pe**2+re**2))
                    if sample['terminal_event']:
                        source,event=event_records[sample['attempt_id']]
                        terminal=Pose.from_xyzw(event.terminal_tcp_xyzw)
                        old=Pose.from_xyzw(source.expert_tcp_xyzw[event.frame_indices[-1]] if
                            isinstance(event,TRAINSemanticAcquireSegment) else event.committed_world_waypoint_xyzw)
                        current_reference=Pose.from_xyzw(event.terminal_reference_xyzw)
                        current_p=current_reference.rotation.T@(terminal.position-current_reference.position)
                        current_r=current_reference.rotation.T@terminal.rotation
                        endpoint_errors.append(dict(source=sample['source'],attempt_id=sample['attempt_id'],fold=fold,
                            reference_time_basis=key[-1],
                            fitted_reference_endpoint_position_error_m=pe,fitted_reference_endpoint_orientation_error_rad=re,
                            current_reference_endpoint_position_error_m=float(np.linalg.norm(position-current_p)),
                            current_reference_endpoint_orientation_error_rad=float(Rotation.from_matrix(rotation@current_r.T).magnitude()),
                            command_issue_reference_endpoint_position_error_m=pe if event.binding.reference_time_basis==COMMAND_ISSUE else None,
                            command_goal_intent_endpoint_position_error_m=pe if sample['attempt_id'] in goal_records else None,
                            command_goal_intent_endpoint_orientation_error_rad=re if sample['attempt_id'] in goal_records else None,
                            original_current_terminal_metric_pose_preserved=True,
                            committed_world_waypoint_position_error_m=float(np.linalg.norm(terminal.position-old.position)),
                            committed_world_waypoint_orientation_error_rad=float(Rotation.from_matrix(terminal.rotation@old.rotation.T).magnitude())))
            error=float(np.mean([np.mean([e for s,e in errors if s==parent]) for parent in parents]))
            candidates.append(dict(lag=float(lag),valid=True,grouped_TRAIN_CV_error=error,samples=len(rows),
                terminal_crossfit_errors=endpoint_errors))
        valid=[c for c in candidates if c['valid']];diagnostic['candidates']=candidates
        if not valid:diagnostic.update(passed=False,reason='no valid completed-attempt lag');continue
        chosen=min(valid,key=lambda row:(row['grouped_TRAIN_CV_error'],row['lag']))
        terminal_errors=chosen['terminal_crossfit_errors']
        by_parent={parent:dict(position=float(np.median([e['fitted_reference_endpoint_position_error_m'] for e in terminal_errors
            if e['source']==parent])),orientation=float(np.median([e['fitted_reference_endpoint_orientation_error_rad'] for e in terminal_errors
            if e['source']==parent]))) for parent in sorted(parents)}
        pt=max(config.minimum_position_tolerance_m,float(np.quantile([e['position'] for e in by_parent.values()],config.tolerance_quantile)))
        rt=max(config.minimum_orientation_tolerance_rad,float(np.quantile([e['orientation'] for e in by_parent.values()],config.tolerance_quantile)))
        diagnostic.update(selected_lag=chosen['lag'],terminal_parent_residuals=by_parent,
            position_tolerance_m=pt,orientation_tolerance_rad=rt,
            maximum_position_tolerance_m=config.maximum_position_tolerance_m,
            maximum_orientation_tolerance_rad=config.maximum_orientation_tolerance_rad,
            fitted_samples=len(groups[(key,chosen['lag'])]),fitted_sources=sorted(parents),
            fitted_rollouts=sorted({r['rollout_id'] for r in groups[(key,chosen['lag'])]}))
        if pt>config.maximum_position_tolerance_m or rt>config.maximum_orientation_tolerance_rad:
            diagnostic.update(passed=False,reason='terminal residual exceeds TRAIN preregistered cap; no clipping');continue
        p,r,o=_fit_completed_curve(groups[(key,chosen['lag'])],grid)
        values=[float(np.median([duration for s,duration in completed if s==parent])) for parent in sorted(parents)]
        duration=float(np.median(values));budget=max(duration,float(np.quantile(values,config.duration_quantile)))
        diagnostic.update(passed=True,completed_duration_by_parent=dict(zip(sorted(parents),values)),
            duration_seconds=duration,duration_budget_seconds=budget,
            incomplete_durations_used_for_quantile=False)
        phase,anchor,role,kind,manipulated,mode,basis=key
        template=dict(phase=phase,anchor_id=anchor,reference_role=role,reference_kind=kind,
            manipulated_role=manipulated,lag_seconds=chosen['lag'],progress=grid,positions=p,rotations=r,opening=o,
            duration_seconds=duration,duration_budget_seconds=budget,position_tolerance_m=pt,
            orientation_tolerance_rad=rt,reference_mode=mode,reference_time_basis=basis,
            reference_time_rule_sha256=rule_sha)
        if basis==COMMAND_GOAL:
            cp,cr,_=_fit_completed_curve(canonical_groups[key],grid)
            template['completion_geometry_xyzw']=Pose(cp[-1],cr[-1]).xyzw()
            diagnostic['completion_geometry_source']='parent-balanced original committed TRAIN waypoint relative to declared command-goal intent'
            diagnostic['public_completion_verifier_calibrated']=False
        templates.append(template)
    available={t['phase'] for t in templates};order=[];omitted=[]
    for phase in spec.phase_skills:
        if phase.name in available:
            if omitted:failures.append(dict(phase=phase.name,reason='missing optional phase is not trailing'))
            order.append(phase.name)
        elif phase.optional_after_native_completion:
            omitted.append(dict(phase=phase.name,reason='no supported completed TRAIN attempt for trailing native-optional phase',
                no_label_was_fabricated=True))
        else:failures.append(dict(phase=phase.name,reason='missing supported completed TRAIN skill attempts'))
    failed_keys=[key for key,d in report['phase_fits'].items() if not d.get('passed')]
    report['required_phase_failures']=failures;report['failed_fit_keys']=failed_keys
    if failures or failed_keys:
        raise InteractionFittingPending('completed TRAIN interaction fit pending support/terminal-cap validation',report)
    used={r['rollout_id'] for (key,lag),rows in groups.items() if lag==0. for r in rows}
    metadata=dict(schema=LIBRARY_SCHEMA,task_name=spec.task_name,training_complete=True,
        fit_split='train',selection_split='train',TEST_used_for_fitting=False,synthetic=synthetic,
        TRAIN_source_ids=sorted(source.source_id for source in sources),
        TRAIN_parent_source_groups=sorted({source.grouping_source for source in sources}),
        TRAIN_source_provenance={s.source_id:dict(source_kind=s.source_kind,parent_source_group=s.grouping_source,
            augmentation_id=s.augmentation_id,raw_episode_id=s.raw_episode_id,parent_raw_episode_id=s.parent_raw_episode_id) for s in sources},
        TRAIN_source_manifest_path=str(TRAIN_source_manifest_path) if TRAIN_source_manifest_path is not None else None,
        TRAIN_source_manifest_sha256=manifest_hash,
        fitting_module_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        task_spec_module_sha256=hashlib.sha256((Path(__file__).parent/'specs.py').read_bytes()).hexdigest(),
        fit_configuration=asdict(config),source_grouped_lag_selection=report['phase_fits'],
        completed_skill_fitting=report,BC_used_source_ids=sorted(used),
        BC_excluded_source_ids=sorted({s.source_id for s in sources}-used),
        all_acquisition_sources_used_for_BC=used=={s.source_id for s in sources},
        completion_tolerances_are_terminal_only=True,
        reference_time_schema=REFERENCE_TIME_SCHEMA,
        reference_time_helper_sha256=hashlib.sha256((Path(__file__).parent/'reference_time.py').read_bytes()).hexdigest(),
        reference_time_rule_sha256_values=sorted({event.binding.reference_time_rule_sha256 for source in sources
            for event in source.completion_events if event.binding.reference_time_basis==COMMAND_ISSUE}),
        formal_comparison_ready=False,
        development_support_override=config.development_only,
        target_label_column_is_not_a_body_track=True,expert_actions_are_training_labels_not_runtime_inputs=True,
        fitted_phase_order=order,omitted_optional_phases=omitted,
        TRAIN_stage_coverage={s.source_id:dict(Counter(s.stages)) for s in sources})
    if semantic_enabled:
        metadata['semantic_acquire_derivation']=report['semantic_acquire_derivation']
        metadata['semantic_acquire_runtime_schema_requires_explicit_release_review']=True
    if goal_flags=={True}:
        metadata['command_goal_reference_derivation']=report['command_goal_reference_derivation']
        metadata['public_completion_verifier_calibrated']=False
        metadata['command_goal_tolerance_scope']='intent_tracking_TRAIN_crossfit; current_public_canonical_geometry_verifier_not_calibrated'
    return InteractionLibrary(dict(metadata=metadata,templates=templates),allow_synthetic=synthetic)


def fit_interaction_library(sources, config, *, TRAIN_source_manifest_path=None):
    """Fit source-grouped lag/relative curves from explicit TRAIN body tracks.

    Native state labels or RGB-D tracks are legal TRAIN supervision. The caller
    must supply their body identities and observation modes; stage_run indices
    and the HDF5 debug XYZ column cannot substitute for these tracks.
    """
    sources = tuple(sources)
    if not sources or len({s.task_name for s in sources}) != 1 or len({s.source_id for s in sources}) != len(sources):
        raise InteractionArtifactError('fit one task with unique TRAIN source identities')
    spec = get_task_spec(sources[0].task_name)
    synthetic = all(s.source_kind == 'synthetic' for s in sources)
    if any(s.source_kind == 'synthetic' for s in sources) and not synthetic:
        raise InteractionArtifactError('cannot mix synthetic fixtures with production TRAIN')
    if not synthetic or any(source.completion_events or source.semantic_acquire_evidence is not None for source in sources):
        return _fit_completed_interactions(sources,config,
            TRAIN_source_manifest_path=TRAIN_source_manifest_path,synthetic=synthetic)
    manifest_hash = None
    if not synthetic:
        if TRAIN_source_manifest_path is None:
            raise InteractionArtifactError('production fit requires an actual TRAIN tracks manifest')
        path = Path(TRAIN_source_manifest_path)
        manifest = json.loads(path.read_text())
        entries = manifest.get('entries', [])
        if not entries or any(row.get('split') != 'train' for row in entries):
            raise InteractionArtifactError('fit manifest must contain TRAIN entries only')
        expected = {row['source_id'] for row in entries if row.get('task') == spec.task_name}
        if expected != {source.source_id for source in sources}:
            raise InteractionArtifactError('not every declared task TRAIN source participates')
        declared = {row['source_id']: row for row in entries if row.get('task') == spec.task_name}
        def declared_group(row):
            if row.get('raw_episode_id'):
                return ':'.join(('train', row['task'], row.get('parent_raw_episode_id') or row['raw_episode_id']))
            return row.get('parent_source_group', row['source_id'])
        if any(declared_group(declared[source.source_id]) != source.grouping_source for source in sources):
            raise InteractionArtifactError('TRAIN source parent groups differ from the frozen manifest')
        manifest_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    stage_phase = {stage: phase.name for phase in spec.phase_skills for stage in phase.TRAIN_expert_stages}
    groups, runs = {}, {}
    for source in sources:
        unknown = set(source.stages)-set(stage_phase)
        if unknown:
            raise InteractionArtifactError('unmapped TRAIN expert stages: ' + str(sorted(unknown)))
        i = 0
        while i < len(source.times):
            binding = source.references[i]
            phase = stage_phase[source.stages[i]]
            if binding is None:
                i += 1
                continue
            track = None
            if binding.reference_kind not in ('phase_origin', 'world'):
                if binding.reference_id not in source.body_tracks:
                    raise InteractionArtifactError('missing TRAIN body SE3 track for ' + str(binding.reference_id))
                track = source.body_tracks[binding.reference_id]

            def reference_mode(index):
                if track is None:
                    return 'not_applicable'
                time = source.times[index]
                body_index = int(np.clip(np.searchsorted(track.times, time), 0, len(track.times)-1))
                if abs(track.times[body_index]-time) > 1e-8:
                    raise InteractionArtifactError('TRAIN reference mode must be aligned with command observations')
                return track.motion_modes[body_index]

            def reference_valid(index):
                if track is None:
                    return True
                body_index = int(np.searchsorted(track.times,source.times[index]))
                return (body_index < len(track.times) and
                    abs(track.times[body_index]-source.times[index]) <= 1e-8 and track.pose_valid[body_index])

            if not reference_valid(i):
                i += 1
                continue

            mode = reference_mode(i)
            j = i+1
            while (j < len(source.times) and stage_phase[source.stages[j]] == phase
                   and source.references[j] == binding and reference_valid(j)
                   and (source.times[j]-source.times[j-1] <= (track.maximum_gap_seconds if track else .075))
                   and reference_mode(j) == mode):
                j += 1
            key = (phase, binding.anchor_id, binding.reference_role,
                   binding.reference_kind, binding.manipulated_role, mode)
            duration = float(source.times[j-1]-source.times[i]+np.median(np.diff(source.times)))
            runs.setdefault(key, []).append((source.grouping_source, duration))
            origin = Pose.from_xyzw(source.measured_tcp_xyzw[i])
            for index in range(i, j):
                command = Pose.from_xyzw(source.expert_tcp_xyzw[index])
                actual = Pose.from_xyzw(source.measured_tcp_xyzw[index])
                progress = float((source.times[index]-source.times[i])/max(duration, 1e-8))
                progress = min(1., progress)
                for lag in config.lag_grid_seconds:
                    if binding.reference_kind in ('phase_origin', 'world'):
                        if lag != 0:
                            continue
                        reference = origin if binding.reference_kind == 'phase_origin' else Pose(np.zeros(3))
                    else:
                        try:
                            current_index = min(len(track.times)-1, int(np.searchsorted(track.times, source.times[index])))
                            reference = track.at(source.times[index]+lag,
                                orientation_required=binding.reference_kind == 'observed_se3',
                                same_mode_as=track.motion_modes[current_index],not_before=source.times[index])
                        except InteractionArtifactError:
                            continue
                        if binding.reference_kind == 'translation_only':
                            reference = Pose(reference.position, np.eye(3))
                    local_position = reference.rotation.T @ (command.position-reference.position)
                    local_rotation = reference.rotation.T @ command.rotation
                    row = dict(source=source.grouping_source, rollout_id=source.source_id, progress=progress,
                        position=local_position, rotation=local_rotation,
                        opening=source.expert_opening[index],
                        position_error=np.linalg.norm(actual.position-command.position),
                        rotation_error=Rotation.from_matrix(actual.rotation @ command.rotation.T).magnitude())
                    groups.setdefault((key, float(lag)), []).append(row)
            i = j
    fitted_sources = {row['rollout_id'] for (key, lag), rows in groups.items() if lag == 0. for row in rows}
    if fitted_sources != {source.source_id for source in sources}:
        raise InteractionArtifactError('declared TRAIN sources lack usable stable-body interaction supervision: '+
            str(sorted({source.source_id for source in sources}-fitted_sources)))
    templates, diagnostics = [], {}
    grid = np.linspace(0., 1., config.progress_knots)
    for key, durations in runs.items():
        base = groups.get((key, 0.), [])
        base_sources = {row['source'] for row in base}
        if len(base_sources) < config.minimum_sources:
            raise InteractionArtifactError('insufficient TRAIN source tracks for ' + str(key))
        candidates = []
        source_to_fold = {name: index % 2 for index, name in enumerate(sorted(base_sources))}
        for lag in config.lag_grid_seconds:
            rows = groups.get((key, float(lag)), [])
            if ({row['source'] for row in rows} != base_sources
                    or {row['rollout_id'] for row in rows} != {row['rollout_id'] for row in base}
                    or len(rows) < config.minimum_lag_sample_fraction*len(base)):
                candidates.append(dict(lag=lag, valid=False, reason='TRAIN source/sample coverage differs', samples=len(rows)))
                continue
            errors = []
            for fold in (0, 1):
                fit = [r for r in rows if source_to_fold[r['source']] != fold]
                held = [r for r in rows if source_to_fold[r['source']] == fold]
                p, r, o = _fit_curve(fit, grid)
                for sample in held:
                    position = np.array([np.interp(sample['progress'], grid, p[:, axis]) for axis in range(3)])
                    rotation = Slerp(grid, Rotation.from_matrix(r))([sample['progress']]).as_matrix()[0]
                    errors.append((sample['source'], np.linalg.norm(position-sample['position'])**2 +
                                   Rotation.from_matrix(rotation @ sample['rotation'].T).magnitude()**2))
            by_source = {s: np.mean([error for source_id, error in errors if source_id == s]) for s in base_sources}
            candidates.append(dict(lag=float(lag), valid=True, grouped_TRAIN_CV_error=float(np.mean(list(by_source.values()))), samples=len(rows)))
        valid = [item for item in candidates if item['valid']]
        if not valid:
            raise InteractionArtifactError('no valid TRAIN lag selection')
        selected = min(valid, key=lambda item: (item['grouped_TRAIN_CV_error'], item['lag']))
        rows = groups[(key, selected['lag'])]
        p, r, opening = _fit_curve(rows, grid)
        values = [float(np.median([duration for source, duration in durations if source == source_id]))
                  for source_id in sorted({source for source, _ in durations})]
        duration = float(np.median(values))
        budget = max(duration, float(np.quantile(values, config.duration_quantile)))
        phase, anchor, role, kind, manipulated, mode = key
        templates.append(dict(phase=phase, anchor_id=anchor, reference_role=role,
            reference_kind=kind, manipulated_role=manipulated, lag_seconds=selected['lag'],
            progress=grid, positions=p, rotations=r, opening=opening,
            duration_seconds=duration, duration_budget_seconds=budget,
            position_tolerance_m=max(config.minimum_position_tolerance_m,
                float(np.quantile([row['position_error'] for row in rows], config.tolerance_quantile))),
            orientation_tolerance_rad=max(config.minimum_orientation_tolerance_rad,
                float(np.quantile([row['rotation_error'] for row in rows], config.tolerance_quantile))),
            reference_mode=mode))
        diagnostics['|'.join(key)] = dict(candidates=candidates, selected_lag=selected['lag'],
            source_grouping='parent raw TRAIN source; all augmentations share a fold', source_to_fold=source_to_fold,
            fitted_sources=sorted(base_sources), fitted_rollouts=sorted({row['rollout_id'] for row in rows}),
            reference_mode=mode, fitted_samples=len(rows))
    payload = dict(metadata=dict(schema=LIBRARY_SCHEMA, task_name=spec.task_name,
        training_complete=True, fit_split='train', selection_split='train',
        TEST_used_for_fitting=False, synthetic=synthetic,
        TRAIN_source_ids=sorted(source.source_id for source in sources),
        TRAIN_parent_source_groups=sorted({source.grouping_source for source in sources}),
        TRAIN_source_provenance={source.source_id: dict(source_kind=source.source_kind,
            parent_source_group=source.grouping_source, augmentation_id=source.augmentation_id,
            raw_episode_id=source.raw_episode_id, parent_raw_episode_id=source.parent_raw_episode_id)
            for source in sources},
        TRAIN_source_manifest_path=str(TRAIN_source_manifest_path) if TRAIN_source_manifest_path is not None else None,
        TRAIN_source_manifest_sha256=manifest_hash,
        fitting_module_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        task_spec_module_sha256=hashlib.sha256((Path(__file__).parent/'specs.py').read_bytes()).hexdigest(),
        fit_configuration=asdict(config), source_grouped_lag_selection=diagnostics,
        target_label_column_is_not_a_body_track=True,
        expert_actions_are_training_labels_not_runtime_inputs=True), templates=templates)
    available = {template['phase'] for template in templates}
    fitted_order, omitted = [], []
    for phase in spec.phase_skills:
        if phase.name in available:
            if omitted:
                raise InteractionArtifactError('a missing optional phase is not a trailing phase')
            fitted_order.append(phase.name)
        elif phase.optional_after_native_completion:
            omitted.append(dict(phase=phase.name,
                reason='no usable TRAIN supervision for trailing optional skill; native success does not require it',
                no_label_was_fabricated=True))
        else:
            raise InteractionArtifactError('missing required TRAIN interaction phase '+phase.name)
    payload['metadata'].update(fitted_phase_order=fitted_order, omitted_optional_phases=omitted,
        TRAIN_stage_coverage={source.source_id:dict(Counter(source.stages)) for source in sources})
    return InteractionLibrary(payload, allow_synthetic=synthetic)


@dataclass(frozen=True)
class PolicyState:
    robot: object
    bodies: dict[str, ObjectState]
    perception_metadata: dict
    role_bindings: dict[str, tuple[str, ...]]

    def __post_init__(self):
        if (self.perception_metadata.get('metric_only') is True
                or any(key in self.perception_metadata for key in
                       ('native_result', 'active_motion_schedule', 'task_internal_fields', 'reset_receipt'))):
            raise ValueError('privileged audit data cannot be a policy observation')
        if any(not isinstance(body, ObjectState) or body_id != body.object_id
               for body_id, body in self.bodies.items()):
            raise ValueError('policy bodies require observed stable ObjectState identities')
        if any(not isinstance(body_id, str) for ids in self.role_bindings.values() for body_id in ids):
            raise ValueError('role bindings must be observed body IDs, never native handles')


@dataclass(frozen=True)
class CandidateConfig:
    arrival_offsets_seconds: tuple[float, ...]
    dt: float
    maximum_linear_speed: float
    maximum_angular_speed: float
    maximum_observation_age: float
    minimum_observation_confidence: float
    geometry_time_penalty: float
    geometry_joint_effort_penalty: float
    ranking_minimum_gain: float
    ranking_disagreement_multiplier: float

    def __post_init__(self):
        values = (self.dt, self.maximum_linear_speed, self.maximum_angular_speed,
                  self.maximum_observation_age, *self.arrival_offsets_seconds)
        if (not self.arrival_offsets_seconds or not np.isfinite(values).all() or min(values) <= 0
                or not np.isfinite([self.minimum_observation_confidence, self.geometry_time_penalty,
                    self.geometry_joint_effort_penalty, self.ranking_minimum_gain,
                    self.ranking_disagreement_multiplier]).all()
                or not 0 <= self.minimum_observation_confidence <= 1
                or min(self.geometry_time_penalty, self.geometry_joint_effort_penalty,
                       self.ranking_minimum_gain, self.ranking_disagreement_multiplier) < 0):
            raise ValueError('explicit frozen candidate/control limits are required')


@dataclass(frozen=True)
class InteractionCandidate:
    candidate_id: str
    phase: str
    phase_index: int
    reference_id: str
    manipulated_ids: tuple[str, ...]
    template: PhaseTemplate
    selected_at: float
    arrival_time: float
    pose: Pose
    linear_velocity: np.ndarray
    angular_velocity: np.ndarray
    reference_pose: Pose
    reference_velocity: np.ndarray
    reference_angular_velocity: np.ndarray
    current_reference: ObjectState
    reachability: object
    geometry_utility: float
    phase_duration: float
    phase_budget: float
    suffix_budget: float
    translation_only_reference: bool
    collision_evidence: dict


def _observed_reference(state, body_id, template, origin, config, *,
        ring_affordance=None,library_sha256=None,task_name=None,allow_development=False):
    now = state.robot.timestamp
    if template.reference_kind in ('phase_origin', 'world'):
        pose = origin if template.reference_kind == 'phase_origin' else Pose(np.zeros(3))
        return ObjectState(body_id, now, pose, np.tile(pose.position, (3, 1)),
                           np.zeros(3), np.zeros(3), 1., True)
    body = state.bodies[body_id]
    metadata = state.perception_metadata.get('bodies', {}).get(body_id, {})
    if (not body.visible or metadata.get('directly_visible') is not True
            or metadata.get('registration_valid') is not True
            or body.confidence < config.minimum_observation_confidence
            or now-body.timestamp < -1e-8 or now-body.timestamp > config.maximum_observation_age):
        raise ValueError('fresh directly observed reference required')
    if template.reference_kind == 'observed_se3':
        if metadata.get('orientation_observable') is not True:
            from .ring_affordance import VerifiedRingAffordance
            allowed=isinstance(ring_affordance,VerifiedRingAffordance) and ring_affordance.permits_reference(
                task=task_name,body_id=body_id,phase=template.phase,template=template,
                library_sha256=library_sha256,evidence=metadata.get('rotation_equivalence_evidence'),
                observed_pose=body.pose,observed_timestamp=body.timestamp,now=now,
                maximum_age=config.maximum_observation_age,allow_development=allow_development)
            if not allowed:raise ValueError('required reference rotation is unobservable')
        return body
    # Sphere/axial gauge is not a true orientation measurement. A translation-
    # only template uses its learned world orientation, not the gauge pose.
    return ObjectState(body.object_id, body.timestamp, Pose(body.pose.position), body.points_world,
                       body.linear_velocity, np.zeros(3), body.confidence, body.visible)


def _reference_at(forecast, reference_id, template, time, origin):
    if template.reference_kind == 'phase_origin':
        return origin, np.zeros(3), np.zeros(3)
    if template.reference_kind == 'world':
        return Pose(np.zeros(3)), np.zeros(3), np.zeros(3)
    pose, velocity, angular = forecast.at(reference_id, time)
    if template.reference_kind == 'translation_only':
        pose, angular = Pose(pose.position), np.zeros(3)
    return pose, np.asarray(velocity), np.asarray(angular)


def _selected_reference_at(candidate,forecast,time,origin):
    return selected_reference(candidate,lambda:_reference_at(forecast,candidate.reference_id,
        candidate.template,time,origin))


def _world_target(reference, velocity, angular, template, progress):
    local, local_velocity, local_angular, opening = template.at(progress)
    offset = reference.rotation @ local.position
    pose = Pose(reference.position+offset, reference.rotation @ local.rotation)
    return (pose, velocity+np.cross(angular, offset)+reference.rotation @ local_velocity,
            angular+reference.rotation @ local_angular, opening)


class CandidateGenerator:
    """IK/joint proofs and caller-estimated dynamic collision precede ranking.

    collision_check(pose,time,phase,observed_bodies,forecast,manipulated_ids)
    returns {'feasible': bool, 'reason': str, 'evidence': dict}. It must use the
    shared static map and predicted/observed dynamic geometry, not live truth.
    """
    def __init__(self, library, reachability, collision_check, config, *, allow_synthetic=False,
                 ring_affordance=None,allow_development_ring_affordance=False):
        if not isinstance(library, InteractionLibrary) or not callable(collision_check):
            raise ValueError('a fitted library and explicit collision service are required')
        if library.metadata.get('synthetic') is not False and not allow_synthetic:
            raise InteractionArtifactError('synthetic fixture is not a trained production interaction library')
        if (not callable(getattr(collision_check, 'bind_candidate', None))
                and not (allow_synthetic and library.metadata.get('synthetic') is True)):
            raise ValueError('production collision service requires a per-candidate scene bind hook')
        self.library, self.reachability = library, reachability
        self.collision_check, self.config = collision_check, config
        if ring_affordance is not None:
            from .ring_affordance import VerifiedRingAffordance
            if not isinstance(ring_affordance,VerifiedRingAffordance):
                raise InteractionArtifactError('ring reference permission must be a reproduced frozen TRAIN capability')
        self.ring_affordance=ring_affordance
        self.allow_development_ring_affordance=allow_development_ring_affordance
        self.last_diagnostics = {}

    def observed_reference(self,state,body_id,template,origin):
        return _observed_reference(state,body_id,template,origin,self.config,
            ring_affordance=self.ring_affordance,library_sha256=self.library.sha256,
            task_name=self.library.task_spec.task_name,
            allow_development=self.allow_development_ring_affordance)

    def bind_candidate_scene(self, state, forecast, phase, reference_id, manipulated_ids, arrival_time):
        hook = getattr(self.collision_check, 'bind_candidate', None)
        if callable(hook):
            hook(state=state, forecast=forecast, phase=phase, reference_id=reference_id,
                 manipulated_ids=tuple(manipulated_ids), arrival_time=float(arrival_time))

    def generate(self, state, forecast, phase_index, remaining_seconds, origin):
        phase = self.library.phase_order[phase_index]
        now, rejected, result = float(state.robot.timestamp), Counter(), []
        minimum_arrival = min(self.config.arrival_offsets_seconds)
        suffix = sum(max(minimum_arrival, max(t.duration_budget_seconds
                     for t in self.library.for_phase(name)))+self.config.dt
                     for name in self.library.phase_order[phase_index+1:])
        for template in self.library.for_phase(phase):
            ids = ((template.reference_kind + ':' + phase,) if template.reference_kind in ('phase_origin', 'world')
                   else tuple(state.role_bindings.get(template.reference_role, ())))
            if not ids:
                rejected['missing_observed_semantic_reference'] += 1
            for body_id in ids:
                try:
                    current = self.observed_reference(state, body_id, template, origin)
                except (KeyError, ValueError) as error:
                    rejected[str(error)] += 1
                    continue
                if template.reference_kind not in ('phase_origin', 'world'):
                    mode = state.perception_metadata.get('bodies', {}).get(body_id, {}).get('motion_mode')
                    if mode != template.reference_mode:
                        rejected['TRAIN_template_reference_mode_differs_from_controller_observation'] += 1
                        continue
                    if mode == 'attached':
                        rejected['attached_reference_requires_measured_attachment_continuation'] += 1
                        continue
                manipulated = tuple(state.role_bindings.get(template.manipulated_role, ()))
                if template.manipulated_role == template.reference_role:
                    manipulated = (body_id,)
                if not manipulated:
                    rejected['missing_observed_manipulated_body'] += 1
                    continue
                if len(manipulated) != 1:
                    rejected['ambiguous_observed_manipulated_identity'] += 1
                    continue
                for delay in self.config.arrival_offsets_seconds:
                    budget = max(float(delay), template.duration_budget_seconds)+self.config.dt
                    if budget+suffix > remaining_seconds+1e-8:
                        rejected['insufficient_full_continuation_time_budget'] += 1
                        continue
                    arrival = now+delay
                    try:
                        reference, v, w = _reference_at(forecast, body_id, template,
                                                       arrival+template.lag_seconds, origin)
                        if template.reference_time_basis in (COMMAND_ISSUE,COMMAND_GOAL):
                            # Candidate reference is predicted at arrival once;
                            # the selected command then owns this fixed pose.
                            v,w=np.zeros(3),np.zeros(3)
                        progress = min(1., delay/template.duration_seconds)
                        pose, velocity, angular, _ = _world_target(reference, v, w, template, progress)
                    except (IndexError, KeyError, ValueError):
                        rejected['forecast_horizon_or_body_unavailable'] += 1
                        continue
                    if (np.linalg.norm(velocity) > self.config.maximum_linear_speed+1e-8
                            or np.linalg.norm(angular) > self.config.maximum_angular_speed+1e-8):
                        rejected['target_twist_outside_control_limits'] += 1
                        continue
                    self.bind_candidate_scene(state, forecast, phase, body_id, manipulated, arrival)
                    check = self.reachability.check(state.robot, pose, velocity, angular, delay)
                    if not check.feasible:
                        rejected['IK_joint_' + str(check.reason)] += 1
                        continue
                    if check.joints is None or check.joint_velocities is None:
                        raise ValueError('reachability proof lacks joint position/velocity evidence')
                    collision = self.collision_check(pose=pose, time=arrival, phase=phase,
                        observed_bodies=state.bodies, forecast=forecast, manipulated_ids=manipulated)
                    if type(collision.get('feasible')) is not bool:
                        raise ValueError('dynamic collision service must return explicit feasibility')
                    if not collision['feasible']:
                        rejected['dynamic_collision_' + str(collision.get('reason'))] += 1
                        continue
                    effort = float(np.linalg.norm(check.joints-state.robot.joints))
                    utility = -self.config.geometry_time_penalty*delay-self.config.geometry_joint_effort_penalty*effort
                    result.append(InteractionCandidate(
                        str(len(result))+':'+template.anchor_id+':'+body_id, phase, phase_index,
                        body_id, manipulated, template, now, arrival, pose, velocity, angular,
                        reference, v, w, current, check, float(utility), template.duration_seconds,
                        budget, suffix, template.reference_kind != 'observed_se3',
                        collision.get('evidence', {})))
        self.last_diagnostics = dict(feasible_candidates=len(result), rejections=dict(rejected),
            geometry_rule='negative arrival delay and joint effort after hard time/twist/IK/joint/dynamic-collision feasibility',
            full_continuation_budget_seconds=suffix)
        return result


@dataclass(frozen=True)
class MPCConfig:
    dt: float
    horizon_steps: int
    maximum_linear_speed: float
    maximum_linear_acceleration: float
    maximum_angular_speed: float
    maximum_angular_acceleration: float
    position_weight: float
    rotation_weight: float
    smoothness_weight: float
    optimizer_iterations: int

    def __post_init__(self):
        if (self.horizon_steps < 1 or self.optimizer_iterations < 1
                or not np.isfinite([value for key, value in asdict(self).items()]).all()
                or min(value for key, value in asdict(self).items()) <= 0):
            raise ValueError('explicit positive SE3 MPC configuration required')


class SE3InteractionMPC:
    """Kinematic SE3 integrator MPC; learned contact dynamics are not claimed."""
    def __init__(self, config):
        self.config = config

    def command(self, robot, desired_poses):
        c, n = self.config, self.config.horizon_steps
        if len(desired_poses) != n:
            raise ValueError('MPC horizon does not match target pose sequence')
        initial = np.r_[robot.linear_velocity, robot.angular_velocity]
        limits = np.array([c.maximum_linear_speed]*3+[c.maximum_angular_speed]*3)
        seed = np.tile(np.clip(initial, -limits, limits), (n, 1))
        def roll(flat):
            u = flat.reshape(n, 6)
            positions = robot.pose.position+np.cumsum(u[:, :3], axis=0)*c.dt
            rotations, rotation = [], robot.pose.rotation.copy()
            for angular in u[:, 3:]:
                rotation = Rotation.from_rotvec(angular*c.dt).as_matrix() @ rotation
                rotations.append(rotation)
            return u, positions, rotations
        def objective(flat):
            u, p, r = roll(flat)
            position_error = p-np.asarray([pose.position for pose in desired_poses])
            angular_error = np.array([Rotation.from_matrix(a @ b.rotation.T).as_rotvec()
                                      for a, b in zip(r, desired_poses)])
            delta = np.diff(np.vstack((initial, u)), axis=0)
            return (c.position_weight*np.square(position_error).sum()
                    + c.rotation_weight*np.square(angular_error).sum()
                    + c.smoothness_weight*np.square(delta).sum())
        def constraints(flat):
            u = flat.reshape(n, 6)
            delta = np.diff(np.vstack((initial, u)), axis=0)
            return np.r_[c.maximum_linear_speed**2-np.square(u[:, :3]).sum(1),
                c.maximum_angular_speed**2-np.square(u[:, 3:]).sum(1),
                (c.maximum_linear_acceleration*c.dt)**2-np.square(delta[:, :3]).sum(1),
                (c.maximum_angular_acceleration*c.dt)**2-np.square(delta[:, 3:]).sum(1)]
        result = minimize(objective, seed.ravel(), method='SLSQP',
            bounds=[(-limit, limit) for _ in range(n) for limit in limits],
            constraints=[dict(type='ineq', fun=constraints)],
            options=dict(maxiter=c.optimizer_iterations, ftol=1e-7))
        if (not np.isfinite(result.fun) or not np.isfinite(result.x).all()
                or np.min(constraints(result.x)) < -1e-5):
            raise RuntimeError('SE3 MPC infeasible: '+str(result.message))
        twist = result.x.reshape(n, 6)[0]
        pose = Pose(robot.pose.position+twist[:3]*c.dt,
                    Rotation.from_rotvec(twist[3:]*c.dt).as_matrix() @ robot.pose.rotation)
        return pose, twist[:3], twist[3:], dict(controller='SE3_kinematic_MPC',
            nominal_constraints_verified=True, actual_motor_tracking_guaranteed=False,
            optimizer_converged=bool(result.success), cost=float(result.fun), iterations=int(result.nit),
            learned_contact_dynamics=False)


def continuation_fingerprint(library, candidate_config, mpc_config,gripper_timing_config=None,
                             joint_limit_recovery_margin_rad=None):
    digest = hashlib.sha256(Path(__file__).read_bytes())
    digest.update((Path(__file__).parent/'candidate_success.py').read_bytes())
    digest.update((Path(__file__).parent/'reference_time.py').read_bytes())
    digest.update(json.dumps(dict(library_sha256=library.sha256,
        candidate_config=asdict(candidate_config), mpc_config=asdict(mpc_config),
        gripper_timing_config=asdict(gripper_timing_config) if gripper_timing_config is not None else None,
        joint_limit_recovery_margin_rad=joint_limit_recovery_margin_rad,
        continuation_selection_policy=CONTINUATION_POLICY,
        conditional_probability_scope='selected_candidate_then_frozen_geometry_downstream'), sort_keys=True).encode())
    return digest.hexdigest()


class TRAINUniformFirstCandidateSampler:
    """Pure TRAIN exploration of one executed candidate, with no outcome input."""
    def __init__(self, *, seed, purpose):
        if purpose != 'train' or type(seed) is not int:
            raise ValueError('first-candidate exploration is explicitly TRAIN-only and seeded')
        self.metadata = dict(purpose='train',sampler='uniform_first_actionable_candidate',seed=seed,
                             labels_or_future_truth_used=False)
        self.rng, self.draw_count = np.random.default_rng(seed), 0

    def select(self, candidates, features):
        x = np.asarray(features,float)
        if (self.draw_count or not candidates or x.ndim != 2 or len(x) != len(candidates)
                or not np.isfinite(x).all()):
            raise ValueError('TRAIN sampler accepts exactly one current observed candidate set')
        self.draw_count += 1
        return int(self.rng.integers(len(candidates)))


@dataclass(frozen=True)
class GripperTimingConfig:
    close_position_tolerance_m: float
    close_orientation_tolerance_rad: float

    def __post_init__(self):
        if (not np.isfinite(list(asdict(self).values())).all()
                or min(asdict(self).values()) <= 0):
            raise ValueError('explicit finite positive public close-arrival tolerances required')


class InteractionController:
    """Frozen observed-skill continuation; parent alone supplies native labels.

    Output commands use the shared counted actuator. No native success/hidden
    phase enters step(). Native success, failure or infrastructure censoring is
    passed separately to attempts.finish() by the collector/evaluator.
    """
    def __init__(self, library, generator, mpc, *, predictor, ranking,
                 continuation_sha256, success_model=None, allow_synthetic=False,
                 first_candidate_sampler=None, collection_purpose=None,gripper_timing_config=None):
        if predictor not in ('cv', 'context') or ranking not in ('geometry', 'value'):
            raise ValueError('unknown fixed comparison method')
        if library.metadata.get('synthetic') is not False and not allow_synthetic:
            raise InteractionArtifactError('production controller rejects synthetic skill fixtures')
        if (gripper_timing_config is None and not allow_synthetic) or (gripper_timing_config is not None
                and not isinstance(gripper_timing_config,GripperTimingConfig)):
            raise ValueError('production continuation requires explicit GripperTimingConfig')
        if ranking == 'value' and success_model is None:
            raise InteractionArtifactError('fitted matching native-completion critic required for value variant')
        if ranking == 'value':
            from .candidate_success import FEATURE_SCHEMA_SHA256, LABEL_SCOPE
            metadata = getattr(success_model, 'metadata', {})
            if (metadata.get('continuation_sha256') != continuation_sha256
                    or metadata.get('interaction_library_sha256') != library.sha256
                    or metadata.get('feature_schema_sha256') != FEATURE_SCHEMA_SHA256
                    or metadata.get('label_scope') != LABEL_SCOPE
                    or metadata.get('continuation_selection_policy') != CONTINUATION_POLICY):
                raise InteractionArtifactError('critic does not match this frozen full continuation')
        if first_candidate_sampler is not None:
            if (collection_purpose != 'train' or ranking != 'geometry'
                    or not isinstance(first_candidate_sampler, TRAINUniformFirstCandidateSampler)):
                raise ValueError('uniform first-candidate injection is only legal for TRAIN geometry collection')
        self.library, self.generator, self.mpc = library, generator, mpc
        self.predictor, self.ranking, self.success_model = predictor, ranking, success_model
        self.continuation_sha256 = continuation_sha256
        self.attempts = ContinuationAttempts(task_name=library.task_spec.task_name,
            predictor=predictor, continuation_sha256=continuation_sha256, library_sha256=library.sha256)
        self.phase_index, self.goal, self.trajectory, self.origin = 0, None, None, None
        self.events, self.last_diagnostics = [], {}
        self.selection_events=[]
        self.committed_roles, self.measured_attachments = {}, {}
        self.value_selection_consumed = False
        self.first_candidate_sampler = first_candidate_sampler
        self.first_selection_consumed = False
        self.first_selection_policy = ('TRAIN_explore_once' if first_candidate_sampler
                                       else 'value_once' if ranking == 'value' else 'geometry')
        self.gripper_timing_config=gripper_timing_config
        self.gripper_timing_events=[]
        self._gripper_goal_token=self._close_edge_issued_at=None

    def _gripper_opening(self,state,requested,final_target):
        """Acquire closes only at observed final grasp SE3, then stays latched.

        Grasp requests still occur only on a native open-to-close transition.
        This public timing rule never retries attachment while already closed.
        Other phases honor their learned opening immediately.
        """
        requested=float(requested>=.5);robot=state.robot;template=self.goal.template
        command=requested;reason='learned_opening_unmodified';distance=angle=None
        token=(self.goal.candidate_id,self.goal.selected_at,len(self.attempts.records))
        if token!=self._gripper_goal_token:
            self._gripper_goal_token=token;self._close_edge_issued_at=None
        config=self.gripper_timing_config
        if config is not None and template.phase=='acquire':
            if final_target is not None:
                distance=float(np.linalg.norm(robot.pose.position-final_target.position))
                angle=float(Rotation.from_matrix(robot.pose.rotation@final_target.rotation.T).magnitude())
            if self._public_acquire_contact(state):
                command=0.;reason='preserve_public_contact_while_verifying_acquire'
            elif requested>=.5:
                command=1.;self._close_edge_issued_at=None;reason='acquire_open_approach'
            elif self._close_edge_issued_at is not None:
                command=0.;reason='one_close_edge_already_issued_this_attempt'
            elif not robot.gripper_open:
                command=1.;reason='reopen_empty_gripper_before_new_close_edge'
            elif (distance is None or distance>config.close_position_tolerance_m
                    or angle>config.close_orientation_tolerance_rad):
                command=1.;reason='defer_close_until_observed_final_grasp_SE3'
            else:
                command=0.;self._close_edge_issued_at=float(robot.timestamp)
                reason='observed_final_grasp_SE3_reached_issue_one_close_edge'
        event=dict(time=float(robot.timestamp),phase=template.phase,candidate_id=self.goal.candidate_id,
            desired_opening=requested,commanded_opening=command,observed_gripper_open=bool(robot.gripper_open),
            public_contact=bool(robot.contact),reason=reason,final_grasp_position_error_m=distance,
            final_grasp_orientation_error_rad=angle,close_edge_issued_at=self._close_edge_issued_at,
            native_object_or_goal_read=False,gripper_width_observed=False)
        self.gripper_timing_events.append(event);self.last_diagnostics['gripper_timing']=event
        return command

    def _reopen_failed_acquire(self,state,reason):
        # One stationary public open command starts recovery. The next actual
        # attempt still requires an observed open gripper before a new close.
        self._gripper_goal_token=self._close_edge_issued_at=None
        self.gripper_timing_events.append(dict(time=float(state.robot.timestamp),phase='acquire',
            desired_opening=1.,commanded_opening=1.,observed_gripper_open=bool(state.robot.gripper_open),
            public_contact=bool(state.robot.contact),reason=reason,native_object_or_goal_read=False,
            gripper_width_observed=False))
        return dict(command='joint_trajectory',joints=state.robot.joints.copy(),opening=1.,
            phase='acquire',reason=reason,nominal_constraints_verified=True,
            actual_motor_tracking_guaranteed=False)

    def _hold(self, reason):
        return dict(command='hold', reason=reason, phase=self.library.phase_order[self.phase_index]
                    if self.phase_index < len(self.library.phase_order) else 'verify')

    def _phase_done(self, state, target, *, public_geometry=False):
        template, robot = self.goal.template, state.robot
        # A fixed intention endpoint is an execution target, not evidence that
        # the moving object is currently at the required physical geometry.
        if template.reference_time_basis==COMMAND_GOAL and not public_geometry:
            return False
        progress = (robot.timestamp-self.goal.selected_at)/template.duration_seconds
        if progress < 1.-1e-8:
            return False
        if template.phase == 'acquire':
            return bool(self._public_acquire_contact(state) and
                        np.linalg.norm(robot.pose.position-target.position) <= template.position_tolerance_m
                        and Rotation.from_matrix(robot.pose.rotation@target.rotation.T).magnitude()
                            <=template.orientation_tolerance_rad)
        if template.phase in ('release', 'deposit', 'hang', 'place'):
            opening = template.at(1.)[3]
            if opening >= .5 and (not robot.gripper_open or robot.contact):
                return False
        return (np.linalg.norm(robot.pose.position-target.position) <= template.position_tolerance_m
                and Rotation.from_matrix(robot.pose.rotation @ target.rotation.T).magnitude()
                <= template.orientation_tolerance_rad)

    def _complete_phase(self, state):
        goal, template, now = self.goal, self.goal.template, float(state.robot.timestamp)
        if template.phase == 'acquire':
            for body_id in goal.manipulated_ids:
                body = state.bodies.get(body_id)
                metadata = state.perception_metadata.get('bodies', {}).get(body_id, {})
                if body is None or not body.visible or metadata.get('directly_visible') is not True:
                    existing=self._causal_attachment_pose(state,body_id)
                    if existing is None:
                        return self._hold('observed_body_needed_to_measure_attachment_transform')
                    # Already measured before occlusion; current contact and
                    # robot proprioception maintain that causal transform.
                    continue
                local = Pose(state.robot.pose.rotation.T @ (body.pose.position-state.robot.pose.position),
                             state.robot.pose.rotation.T @ body.pose.rotation)
                self.measured_attachments[body_id] = dict(
                    local_pose_xyzw=local.xyzw().tolist(), observed_at=now,
                    source='current_RGBD_body_pose_and_robot_proprioception_at_public_contact',
                    orientation_observable=metadata.get('orientation_observable') is True,
                    rotation_gauge=metadata.get('rotation_gauge'))
        elif not state.robot.contact and state.robot.gripper_open:
            for body_id in goal.manipulated_ids:
                self.measured_attachments.pop(body_id, None)
        self.attempts.skill_completed(template.phase, now)
        self.events.append(dict(time=now, phase=template.phase, event='observable_skill_completed',
                                official_task_success_claimed=False))
        self.phase_index += 1
        self.goal = self.trajectory = self.origin = None
        return self._hold('advance_to_next_TRAIN_fitted_skill')

    def _causal_attachment_pose(self,state,body_id):
        item=self.measured_attachments.get(body_id)
        if (not state.robot.contact or not item
                or item.get('source') != 'current_RGBD_body_pose_and_robot_proprioception_at_public_contact'
                or not np.isfinite(item.get('observed_at',np.nan))
                or not 0 <= item['observed_at'] <= state.robot.timestamp+1e-8):
            return None
        local=Pose.from_xyzw(item['local_pose_xyzw'])
        return Pose(state.robot.pose.position+state.robot.pose.rotation@local.position,
                    state.robot.pose.rotation@local.rotation)

    def _public_acquire_contact(self,state):
        """Own close + unique causal identity, independent of open-width proxy.

        The unchanged shared contact bit indicates held presence without IDs.
        RGBD/public-mode evidence must associate that bit with the chosen body.
        """
        goal=self.goal;now=float(state.robot.timestamp)
        closed=getattr(self,'_close_edge_issued_at',None)
        ids=getattr(goal,'manipulated_ids',())
        if (goal.template.phase!='acquire' or not state.robot.contact or closed is None
                or not np.isfinite(closed) or closed>now+1e-8 or len(ids)!=1
                or ids[0]!=getattr(goal,'reference_id',None)):
            return False
        identity=ids[0];metadata=state.perception_metadata
        invalid=set(metadata.get('invalid_mode_ids',()))
        bodies=metadata.get('bodies',{})
        associated={body for body,info in bodies.items()
            if info.get('motion_mode')=='attached' and body not in invalid}
        info=bodies.get(identity,{})
        if (identity in invalid or associated!={identity}
                or tuple(state.role_bindings.get(goal.template.manipulated_role,()))!=(identity,)
                or info.get('mode_provenance')!='caller_controller_touch_action_proprioception'):
            return False
        try:
            self.generator.observed_reference(state,identity,goal.template,self.origin)
            return True
        except (KeyError,ValueError):
            return self._causal_attachment_pose(state,identity) is not None

    def step(self, state, forecast, remaining_seconds):
        # Chosen cup/peg identity survives phase transitions. These are policy
        # decisions among observed IDs, never native handles or hidden parents.
        state = PolicyState(state.robot, state.bodies, state.perception_metadata,
                            {**state.role_bindings, **self.committed_roles})
        now = float(state.robot.timestamp)
        if self.phase_index >= len(self.library.phase_order):
            return self._hold('trained_skills_completed_await_parent_native_outcome')
        phase = self.library.phase_order[self.phase_index]
        if (phase=='acquire' and self.gripper_timing_config is not None and self.goal is None
                and not state.robot.gripper_open and not state.robot.contact):
            return self._reopen_failed_acquire(state,'observe_open_empty_gripper_before_new_acquire_attempt')
        if self.origin is None:
            self.origin = state.robot.pose
        if self.goal is None:
            candidates = self.generator.generate(state, forecast, self.phase_index,
                                                  remaining_seconds, self.origin)
            self.last_diagnostics = dict(candidate_search=self.generator.last_diagnostics)
            if not candidates:
                return self._hold('no_feasible_fitted_interaction_candidate')
            features = []
            contexts = forecast.metadata.get('contexts', {})
            for candidate in candidates:
                context = contexts.get(candidate.reference_id) if self.predictor == 'context' else None
                if self.predictor == 'context' and candidate.template.reference_kind not in ('phase_origin', 'world') and context is None:
                    raise ValueError('observed reference lacks its causal SimNorm context')
                features.append(candidate_features(candidate, state.robot, candidate.current_reference,
                    remaining_seconds, predictor=self.predictor, context=context))
            value_query = self.ranking == 'value' and not self.value_selection_consumed
            predictions = (self.success_model.predict(np.stack(features), task_name=self.library.task_spec.task_name,
                predictor=self.predictor, phase=phase) if value_query else None)
            if value_query:
                # One episode-level critic decision, including a guarded
                # fallback, is followed by the same geometry continuation that
                # generated the full-native-completion TRAIN labels.
                self.value_selection_consumed = True
            config = self.generator.config
            selected = select_candidate(candidates, predictions,
                minimum_gain=config.ranking_minimum_gain,
                disagreement_multiplier=config.ranking_disagreement_multiplier)
            exploration = None
            if self.first_candidate_sampler is not None and not self.first_selection_consumed:
                chosen = self.first_candidate_sampler.select(candidates,features)
                selected = replace(selected,selected_index=chosen,proposed_index=chosen,
                    actual_override=chosen != selected.geometry_index,success_probability=None,
                    selected_supported=None,calibrated_ensemble_std=None,
                    geometry_utility=float(candidates[chosen].geometry_utility),
                    fallback_reason='TRAIN_uniform_first_candidate_exploration')
                exploration = dict(self.first_candidate_sampler.metadata,selected_index=chosen,
                    candidate_count=len(candidates),only_executed_candidate_receives_label=True)
            self.first_selection_consumed = True
            self.goal = candidates[selected.selected_index]
            if self.goal.template.reference_kind not in ('phase_origin', 'world'):
                self.committed_roles[self.goal.template.reference_role] = (self.goal.reference_id,)
            self.committed_roles[self.goal.template.manipulated_role] = self.goal.manipulated_ids
            self.trajectory = JointTrajectory(now, self.goal.arrival_time, state.robot.joints.copy(),
                state.robot.joint_velocities.copy(), self.goal.reachability.joints.copy(),
                self.goal.reachability.joint_velocities.copy())
            decision_id=self.attempts.begin(phase, self.goal.candidate_id, now, features[selected.selected_index],
                probability=selected.success_probability, selected_supported=selected.selected_supported)
            self.selection_events.append(dict(decision_id=decision_id,time=float(now),phase=phase,
                candidate_count=len(candidates),candidate_ids=[candidate.candidate_id for candidate in candidates],
                selected_candidate_id=self.goal.candidate_id,
                geometry_candidate_id=candidates[selected.geometry_index].candidate_id,
                proposed_candidate_id=(candidates[selected.proposed_index].candidate_id if selected.proposed_index is not None else None),
                **asdict(selected),value_query_this_selection=bool(value_query),
                episode_first_actionable_decision=decision_id==0,
                first_selection_policy=self.first_selection_policy,
                continuation_selection_policy=CONTINUATION_POLICY,
                probability_is_unpenalized=True,all_downstream_selections_geometry=True,
                unexecuted_candidates_labelled=False))
            self.last_diagnostics.update(selection=asdict(selected),
                probability_is_unpenalized=True, selection_utility_kind='geometry_with_guarded_override',
                value_query_this_selection=value_query,
                TRAIN_exploration=exploration,
                continuation_selection_policy=CONTINUATION_POLICY,
                all_downstream_selections_geometry=True)
        goal, template = self.goal, self.goal.template
        self.generator.bind_candidate_scene(state, forecast, phase, goal.reference_id,
                                             goal.manipulated_ids, max(now, goal.arrival_time))
        # Public contact can change this reference from external to attached.
        # Evaluate observed grasp completion before asking the exogenous model
        # for another prediction of a body it correctly refuses to roll out.
        if (template.reference_time_basis!=COMMAND_GOAL and template.phase == 'acquire'
                and self._public_acquire_contact(state)):
            try:
                try:
                    measured = self.generator.observed_reference(state,goal.reference_id,template,self.origin)
                    reference_pose=measured.pose
                    reference_source='current_direct_RGBD'
                except (KeyError,ValueError):
                    reference_pose=self._causal_attachment_pose(state,goal.reference_id)
                    if reference_pose is None:
                        raise ValueError('no directly observed or causally measured attached reference')
                    if template.reference_kind == 'translation_only':
                        reference_pose=Pose(reference_pose.position)
                    reference_source='causal_measured_attachment_and_current_robot_proprioception'
                observed_target = _world_target(reference_pose,np.zeros(3),np.zeros(3),template,1.)[0]
                if self._phase_done(state, observed_target):
                    self.last_diagnostics['acquire_completion_reference_source']=reference_source
                    return self._complete_phase(state)
            except (KeyError, ValueError):
                return self._hold('observed_body_needed_to_verify_public_grasp_completion')
        # Recheck a fresh observation even while a verified joint plan is active.
        try:
            current_reference=self.generator.observed_reference(state,goal.reference_id,template,self.origin)
            reference, v, w = _selected_reference_at(goal,forecast,now+template.lag_seconds,self.origin)
            progress = np.clip((now-goal.selected_at)/template.duration_seconds, 0, 1)
            target, _, _, opening = _world_target(reference, v, w, template, progress)
            if template.reference_time_basis==COMMAND_GOAL:
                canonical=Pose.from_xyzw(template.completion_geometry_xyzw)
                completion_target=Pose(current_reference.pose.position+current_reference.pose.rotation@canonical.position,
                    current_reference.pose.rotation@canonical.rotation)
                self.last_diagnostics['public_completion_geometry']=dict(
                    source='fresh_current_RGBD_pose_and_explicit_TRAIN_canonical_geometry',
                    actual_current_metric_calibrated=False,official_task_success_claimed=False,
                    position_error_m=float(np.linalg.norm(state.robot.pose.position-completion_target.position)),
                    orientation_error_rad=float(Rotation.from_matrix(state.robot.pose.rotation@completion_target.rotation.T).magnitude()))
            else:completion_target=target
            final_grasp_target=(completion_target if template.reference_time_basis==COMMAND_GOAL else
                _world_target(current_reference.pose,np.zeros(3),np.zeros(3),template,1.)[0]) if template.phase=='acquire' else None
        except (KeyError, IndexError, ValueError):
            if template.reference_time_basis==COMMAND_GOAL and now>goal.selected_at+goal.phase_budget+1e-8:
                self.attempts.abandon(phase,now,'public_geometry_unavailable_at_skill_timeout')
                self.goal=self.trajectory=self.origin=None
                if phase=='acquire' and self.gripper_timing_config is not None:
                    return self._reopen_failed_acquire(state,'unverified_acquire_geometry_timeout_reopen_before_reselection')
                return self._hold('fitted_skill_budget_exhausted_reselect_same_phase')
            return self._hold('fresh_observed_reference_or_forecast_required')
        if self._phase_done(state,completion_target,public_geometry=template.reference_time_basis==COMMAND_GOAL):
            return self._complete_phase(state)
        if now > goal.selected_at+goal.phase_budget+1e-8:
            self.attempts.abandon(phase, now, 'observable_skill_timeout_same_phase_abandonment')
            self.goal = self.trajectory = self.origin = None
            if (phase=='acquire' and self.gripper_timing_config is not None
                    and (not state.robot.contact or template.reference_time_basis==COMMAND_GOAL)):
                return self._reopen_failed_acquire(state,'failed_acquire_timeout_reopen_before_reselection')
            return self._hold('fitted_skill_budget_exhausted_reselect_same_phase')
        collision = self.generator.collision_check(pose=target, time=now, phase=phase,
            observed_bodies=state.bodies, forecast=forecast, manipulated_ids=goal.manipulated_ids)
        if collision.get('feasible') is not True:
            self.attempts.abandon(phase, now, 'fresh_dynamic_collision_rejected')
            self.goal = self.trajectory = self.origin = None
            return self._hold('fresh_dynamic_collision_rejected')
        if now+self.generator.config.dt < goal.arrival_time-1e-8:
            joints, velocities = self.trajectory.at(now+self.generator.config.dt)
            # Shared model validates the next measured-state segment again.
            check = self.generator.reachability.check_joints(state.robot, joints, velocities,
                                                            self.generator.config.dt)
            if not check.feasible:
                return self._hold('verified_plan_next_joint_segment_rejected')
            approach_opening=(1. if phase=='acquire' and self.gripper_timing_config is not None
                              else float(state.robot.gripper_open))
            commanded_opening=self._gripper_opening(state,approach_opening,final_grasp_target)
            return dict(command='joint_trajectory', joints=joints, opening=commanded_opening,
                phase=phase, trajectory_end=goal.arrival_time, nominal_constraints_verified=True,
                actual_motor_tracking_guaranteed=False)
        desired = []
        try:
            for offset in np.arange(1, self.mpc.config.horizon_steps+1)*self.mpc.config.dt:
                ref, rv, rw = _selected_reference_at(goal,forecast,now+offset+template.lag_seconds,self.origin)
                p = np.clip((now+offset-goal.selected_at)/template.duration_seconds, 0, 1)
                desired.append(_world_target(ref, rv, rw, template, p)[0])
            pose, velocity, angular, details = self.mpc.command(state.robot, desired)
        except (KeyError, IndexError, ValueError, RuntimeError) as error:
            self.last_diagnostics['local_control_rejection'] = str(error)
            return self._hold('SE3_MPC_or_trained_forecast_unavailable')
        check = self.generator.reachability.check(state.robot, pose, velocity, angular, self.mpc.config.dt)
        if not check.feasible:
            return self._hold('local_MPC_IK_joint_constraints_rejected')
        self.last_diagnostics['local_control'] = details
        commanded_opening=self._gripper_opening(state,opening,final_grasp_target)
        return dict(command='joint_trajectory', joints=check.joints, opening=commanded_opening,
                    phase=phase, nominal_constraints_verified=True,
                    actual_motor_tracking_guaranteed=False, local_control=details)
