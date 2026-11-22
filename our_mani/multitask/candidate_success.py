"""R10 native-completion critic contracts and multi-phase outcome accounting.

Skill completion is a separate label. A completed early skill remains pending
until its frozen downstream continuation succeeds or fails. Only abandoning a
candidate within the same phase closes that decision's continuation early.
"""
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from scipy.special import expit, logit

SCHEMA = 'r10_observed_SE3_candidate_native_continuation_v1'
LABEL_SCOPE = 'native_task_completion_under_frozen_continuation_before_same_phase_abandonment'
CONTINUATION_POLICY = 'single_value_selection_then_frozen_geometry'


def _names(prefix, count):
    return tuple(prefix + '_' + str(i) for i in range(count))


FEATURE_NAMES = (
    *_names('current_reference_position', 3), *_names('current_reference_rotation6', 6),
    *_names('current_reference_velocity', 3), *_names('current_reference_angular_velocity', 3),
    *_names('robot_relative_position', 3), *_names('robot_relative_rotation_vector', 3),
    *_names('robot_relative_velocity', 3), *_names('robot_angular_velocity', 3),
    *_names('robot_joint_position', 7), *_names('robot_joint_velocity', 7),
    *_names('future_reference_position', 3), *_names('future_reference_rotation6', 6),
    *_names('future_reference_velocity', 3), *_names('future_reference_angular_velocity', 3),
    *_names('target_relative_position', 3), *_names('target_relative_rotation_vector', 3),
    *_names('target_velocity', 3), *_names('target_angular_velocity', 3),
    *_names('required_joint_delta', 7), *_names('required_joint_velocity', 7),
    'arrival_delay', 'remaining_episode_seconds', 'phase_duration_seconds',
    'suffix_budget_seconds', 'phase_index', 'reference_confidence',
    'phase_progress', 'reference_translation_only', 'robot_gripper_open', 'public_contact',
    'context_predictor', 'context_available',
    *_names('causal_dynamics_context', 16),
)
FEATURE_SCHEMA_SHA256 = hashlib.sha256(json.dumps(
    dict(schema=SCHEMA, features=FEATURE_NAMES), sort_keys=True).encode()).hexdigest()


def candidate_features(candidate, robot, current_reference, remaining_seconds,
                       *, predictor, context=None, phase_progress=0.):
    """Observed/current reference + predicted future + robot data; no native truth."""
    if predictor not in ('cv', 'context'):
        raise ValueError('unknown predictor')
    if len(robot.joints) != 7 or len(robot.joint_velocities) != 7:
        raise ValueError('R10 candidate feature schema requires seven robot joints')
    context_available = predictor == 'context' and context is not None
    if context_available:
        context = np.asarray(context, dtype=float)
        if context.shape != (16,):
            raise ValueError('candidate requires its declared causal reference context')
    else:
        context = np.zeros(16)
    current = current_reference.pose
    future = candidate.reference_pose
    rotation_error = lambda a, b: Rotation.from_matrix(a @ b.T).as_rotvec()
    values = np.r_[
        current.position, current.rotation[:, :2].reshape(6),
        current_reference.linear_velocity, current_reference.angular_velocity,
        robot.pose.position-current.position, rotation_error(robot.pose.rotation, current.rotation),
        robot.linear_velocity-current_reference.linear_velocity, robot.angular_velocity,
        robot.joints, robot.joint_velocities,
        future.position, future.rotation[:, :2].reshape(6),
        candidate.reference_velocity, candidate.reference_angular_velocity,
        future.rotation.T @ (candidate.pose.position-future.position),
        rotation_error(candidate.pose.rotation, future.rotation),
        candidate.linear_velocity, candidate.angular_velocity,
        candidate.reachability.joints-robot.joints, candidate.reachability.joint_velocities,
        candidate.arrival_time-robot.timestamp, remaining_seconds,
        candidate.phase_duration, candidate.suffix_budget, candidate.phase_index,
        current_reference.confidence, phase_progress, candidate.translation_only_reference,
        robot.gripper_open, robot.contact, predictor == 'context', context_available, context,
    ].astype(np.float32)
    if values.shape != (len(FEATURE_NAMES),) or not np.isfinite(values).all():
        raise ValueError('invalid R10 candidate feature vector')
    return values


class ContinuationAttempts:
    """Label ledger owned by collection/evaluation, never read as policy input."""
    def __init__(self, *, task_name, predictor, continuation_sha256, library_sha256):
        if not continuation_sha256 or not library_sha256:
            raise ValueError('attempt labels must bind a frozen controller and interaction library')
        self.task_name, self.predictor = task_name, predictor
        self.continuation_sha256, self.library_sha256 = continuation_sha256, library_sha256
        self.records, self.active_by_phase = [], {}
        self.finalized = False

    def begin(self, phase, candidate_id, observed_until, feature_vector, *,
              probability=None, selected_supported=None):
        if self.finalized:
            raise ValueError('a finalized episode ledger cannot be reused')
        if not np.isfinite(observed_until):
            raise ValueError('invalid selection time')
        index = len(self.records)
        feature_vector = np.asarray(feature_vector, dtype=float)
        if feature_vector.shape != (len(FEATURE_NAMES),) or not np.isfinite(feature_vector).all():
            raise ValueError('attempt feature schema differs')
        if probability is not None and (not np.isfinite(probability) or not 0 <= probability <= 1):
            raise ValueError('probability must be the unpenalized model probability')
        if self.records and observed_until < max(row['observed_until'] for row in self.records):
            raise ValueError('selection time precedes ledger history')
        previous = self.active_by_phase.get(phase)
        if previous is not None:
            row = self.records[previous]
            if row['official_task_success'] is None and row['skill_success'] is not True:
                self.abandon(phase, observed_until, 'same_phase_candidate_replaced')
        row = dict(decision_id=index, task=self.task_name, predictor=self.predictor,
                   phase=phase, candidate_id=str(candidate_id), selected_at=float(observed_until),
                   observed_until=float(observed_until), features=feature_vector.tolist(),
                   feature_schema=SCHEMA, feature_schema_sha256=FEATURE_SCHEMA_SHA256,
                   label_scope=LABEL_SCOPE, continuation_sha256=self.continuation_sha256,
                   continuation_selection_policy=CONTINUATION_POLICY,
                   episode_first_actionable_decision=index == 0,
                   interaction_library_sha256=self.library_sha256,
                   predicted_probability=probability, selected_supported=selected_supported,
                   skill_success=None, skill_reason=None, official_task_success=None,
                   outcome_reason=None, complete=False, censored=False)
        self.records.append(row)
        self.active_by_phase[phase] = index
        return index

    def skill_completed(self, phase, time, reason='observable_skill_completed'):
        if self.finalized or not np.isfinite(time):
            raise ValueError('invalid skill event after episode end')
        row = self.records[self.active_by_phase[phase]]
        if row['complete']:
            raise ValueError('cannot complete a skill after abandoning its continuation')
        if time < row['selected_at']:
            raise ValueError('skill event precedes candidate selection')
        row.update(skill_success=True, skill_reason=reason, observed_until=float(time))
        # Crucially, official_task_success remains None across the next phase.

    def abandon(self, phase, time, reason):
        if self.finalized or not np.isfinite(time):
            raise ValueError('invalid abandonment after episode end')
        index = self.active_by_phase.get(phase)
        if index is None:
            return
        row = self.records[index]
        if row['complete']:
            return
        if row['skill_success'] is True:
            raise ValueError('phase transition is not abandonment of its completed skill')
        if time < row['selected_at']:
            raise ValueError('abandonment precedes candidate selection')
        row.update(official_task_success=False, outcome_reason=reason,
                   observed_until=float(time), complete=True, skill_success=False,
                   skill_reason='candidate_abandoned_before_skill_completion')

    def finish(self, native_success, time, reason):
        if self.finalized or not np.isfinite(time):
            raise ValueError('episode ledger already finalized or invalid end time')
        if native_success is not None and type(native_success) is not bool:
            raise ValueError('native task outcome must be bool or censored None')
        for row in self.records:
            if row['complete']:
                continue
            if time < row['observed_until']:
                raise ValueError('task outcome precedes recorded observation')
            row.update(official_task_success=native_success, observed_until=float(time),
                       outcome_reason=reason, complete=native_success is not None,
                       censored=native_success is None)
        self.finalized = True
        return deepcopy(self.records)

    def to_report(self):
        return deepcopy(self.records)


class CandidateSuccessModel:
    """Load a fitted CPU linear bootstrap critic; never create random heads.

    The later TRAIN collector/trainer writes this explicit format. Calibration
    may use source-separated VAL; classifier selection and normalization must
    remain TRAIN-only. Unsupported phases/tasks fail closed rather than borrowing
    a moving-pick head. Production loading rejects synthetic fixtures.
    """
    def __init__(self, artifact_path, *, continuation_sha256, library_sha256,
                 allow_synthetic=False):
        path = Path(artifact_path)
        payload = json.loads(path.read_text())
        self.metadata = payload['metadata']
        m = self.metadata
        if (m.get('training_complete') is not True or m.get('classifier_fit_split') != 'train'
                or m.get('classifier_selection_split') != 'train'
                or m.get('normalization_fit_split') != 'train'
                or m.get('TEST_used_for_fitting') is not False
                or m.get('label_scope') != LABEL_SCOPE
                or m.get('continuation_selection_policy') != CONTINUATION_POLICY
                or m.get('feature_schema_sha256') != FEATURE_SCHEMA_SHA256
                or m.get('continuation_sha256') != continuation_sha256
                or m.get('interaction_library_sha256') != library_sha256
                or not m.get('TRAIN_source_manifest_sha256')
                or (m.get('synthetic') is not False and not allow_synthetic)):
            raise ValueError('completed matching TRAIN-only native-continuation critic required')
        self.artifact_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        if m.get('synthetic') is False:
            manifest_path = Path(m['TRAIN_source_manifest_path'])
            if hashlib.sha256(manifest_path.read_bytes()).hexdigest() != m['TRAIN_source_manifest_sha256']:
                raise ValueError('TRAIN critic source manifest changed')
            entries = json.loads(manifest_path.read_text()).get('entries', [])
            if not entries or any(row.get('split') != 'train' for row in entries):
                raise ValueError('critic training manifest must contain TRAIN entries only')
            declared = {':'.join(('train', row['task'], row.get('parent_raw_episode_id') or row['raw_episode_id']))
                        for row in entries}
            if (m.get('TRAIN_parent_source_grouping') != ['split', 'task', 'parent_raw_episode_id_or_raw_episode_id']
                    or set(m.get('TRAIN_parent_source_groups', ())) != declared):
                raise ValueError('critic source grouping/coverage does not match raw-parent TRAIN manifest')
            validation_path = Path(m['VAL_calibration_manifest_path'])
            if hashlib.sha256(validation_path.read_bytes()).hexdigest() != m['VAL_calibration_manifest_sha256']:
                raise ValueError('VAL calibration manifest changed')
            validation_entries = json.loads(validation_path.read_text()).get('entries', [])
            if (not validation_entries or any(row.get('split') != 'validation' for row in validation_entries)
                    or m.get('classifier_selection_uses_VAL') is not False
                    or m.get('calibration_selector_used_VAL') is not True):
                raise ValueError('critic CAL provenance does not preserve TRAIN classifier selection')
        self.heads = {}
        for key, head in payload['heads'].items():
            mean, scale = np.asarray(head['TRAIN_mean'], float), np.asarray(head['TRAIN_scale'], float)
            weights, bias = np.asarray(head['ensemble_weights'], float), np.asarray(head['ensemble_bias'], float)
            support = np.asarray(head['standardized_TRAIN_support'], float)
            d = len(FEATURE_NAMES)
            if (mean.shape != (d,) or scale.shape != (d,) or np.any(scale <= 0)
                    or weights.ndim != 2 or weights.shape[1] != d or len(weights) < 2
                    or bias.shape != (len(weights),) or support.ndim != 2
                    or support.shape[1] != d or not len(support)
                    or not all(np.isfinite(a).all() for a in (mean, scale, weights, bias, support))):
                raise ValueError('invalid trained critic/support dimensions')
            calibration = head['calibration']
            a, b = float(calibration['slope']), float(calibration['intercept'])
            if (not np.isfinite([a, b]).all() or a <= 0
                    or calibration.get('fit_split') not in ('validation', 'identity')
                    or head.get('source_coverage_complete') is not True
                    or type(head.get('ranking_enabled')) is not bool
                    or head.get('support_threshold_fit_split') != 'train'
                    or head.get('support_threshold_excludes_same_parent_source') is not True
                    or not np.isfinite(head['support_max_distance']) or head['support_max_distance'] <= 0):
                raise ValueError('invalid critic calibration/source support')
            if (head['ranking_enabled'] and (head.get('TRAIN_ranking_guard_passed') is not True
                    or head.get('calibration_crossfit_gate_passed') is not True)):
                raise ValueError('learned ranking cannot bypass TRAIN discrimination or VAL calibration guards')
            support_sources = head.get('support_parent_sources', [])
            if (len(support_sources) != len(support) or any(not source.startswith('train:') for source in support_sources)
                    or (m.get('synthetic') is False and set(support_sources)-declared)):
                raise ValueError('nearest-neighbour support requires actual distinct TRAIN parent identities')
            if calibration['fit_split'] == 'identity' and (a != 1. or b != 0.):
                raise ValueError('identity calibration must preserve raw probabilities')
            self.heads[key] = (head, mean, scale, weights, bias, cKDTree(support))

    def predict(self, features, *, task_name, predictor, phase):
        key = '|'.join((task_name, predictor, phase))
        if key not in self.heads:
            raise ValueError('no fitted R10 critic for ' + key)
        head, mean, scale, weights, bias, tree = self.heads[key]
        x = np.asarray(features, float)
        if x.ndim != 2 or x.shape[1] != len(FEATURE_NAMES) or not np.isfinite(x).all():
            raise ValueError('critic input schema differs')
        normalized = (x-mean)/scale
        members = expit(normalized @ weights.T + bias)
        raw = members.mean(1)
        a, b = head['calibration']['slope'], head['calibration']['intercept']
        correction = lambda p: expit(a*logit(np.clip(p, 1e-7, 1-1e-7))+b)
        distance = np.asarray(tree.query(normalized)[0])
        return dict(probability=correction(raw), raw_probability=raw,
                    supported=distance <= head['support_max_distance'],
                    ensemble_std=correction(members).std(1),
                    nearest_fit_distance=distance, ranking_enabled=head['ranking_enabled'])


@dataclass(frozen=True)
class CandidateSelection:
    selected_index: int
    geometry_index: int
    proposed_index: int | None
    actual_override: bool
    success_probability: float | None
    selected_supported: bool | None
    calibrated_ensemble_std: float | None
    geometry_utility: float
    fallback_reason: str | None


def select_candidate(candidates, predictions=None, *, minimum_gain,
                     disagreement_multiplier):
    if not candidates:
        return None
    if (not np.isfinite([minimum_gain, disagreement_multiplier]).all()
            or minimum_gain < 0 or disagreement_multiplier < 0):
        raise ValueError('ranking guard thresholds must be nonnegative')
    geometry = np.asarray([candidate.geometry_utility for candidate in candidates], float)
    if not np.isfinite(geometry).all():
        raise ValueError('invalid geometry utility')
    base = selected = int(np.argmax(geometry))
    proposed, probability, supported, std, fallback = None, None, None, None, None
    if predictions is not None:
        p = np.asarray(predictions['probability'], float)
        mask = np.asarray(predictions['supported'], bool)
        sigma = np.asarray(predictions['ensemble_std'], float)
        if (any(a.shape != geometry.shape for a in (p, mask, sigma))
                or not np.isfinite(p).all() or np.any((p < 0) | (p > 1))
                or not np.isfinite(sigma).all() or np.any(sigma < 0)):
            raise ValueError('invalid calibrated candidate predictions')
        if predictions.get('ranking_enabled') is not True:
            fallback = 'TRAIN_selection_does_not_support_learned_ranking'
        elif not mask[base]:
            fallback = 'geometry_reference_outside_TRAIN_support'
        else:
            # Geometry only breaks an exact probability tie. Adding a scaled
            # geometry term would silently change the advertised probabilities.
            proposed = int(max(np.flatnonzero(mask), key=lambda index:
                               (p[index], geometry[index], -index)))
            required = minimum_gain + disagreement_multiplier*np.hypot(sigma[proposed], sigma[base])
            if proposed == base:
                fallback = 'learned_and_geometry_agree'
            elif p[proposed]-p[base] > required:
                selected = proposed
            else:
                fallback = 'gain_below_frozen_disagreement_margin'
        probability, supported, std = float(p[selected]), bool(mask[selected]), float(sigma[selected])
    return CandidateSelection(selected, base, proposed, selected != base,
                              probability, supported, std, float(geometry[selected]), fallback)
