"""Small task-specific candidate critics with TRAIN-only support checks.

Only observable r06 candidate features enter the critic. Derived norms retain
the original causal context coordinates and introduce no new information.
The .pt artifact is a trusted local Python pickle containing sklearn models.
"""
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit
from scipy.spatial import cKDTree

TASKS = ('pick_moving_target_on_the_table', 'pick_moving_target_on_the_table_high_speed')
FORMAT = 'candidate_success_r07_sklearn_v1'
DERIVED_NAMES = (
    'current_target_speed', 'current_robot_target_distance', 'current_relative_speed',
    'robot_angular_speed', 'robot_max_joint_speed', 'predicted_target_speed',
    'interception_offset_distance', 'terminal_relative_speed', 'orientation_error_angle',
    'joint_displacement_l2', 'joint_displacement_linf', 'terminal_max_joint_speed',
    'transport_distance', 'minimum_workspace_margin', 'actual_approach_distance',
    'approach_distance_per_second', 'joint_displacement_per_second',
    'transport_distance_per_remaining_second', 'arrival_fraction_of_remaining_budget',
)
FEATURE_SCHEMA = dict(version='observable_r06_raw_plus_norms_v1', raw_dimensions=[71, 87],
    context_coordinates='71:87 for Context, unchanged', derived_names=DERIVED_NAMES)
FEATURE_SCHEMA_SHA256 = hashlib.sha256(json.dumps(FEATURE_SCHEMA, sort_keys=True).encode()).hexdigest()


def transform_features(features, raw_dim=None):
    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] not in (71, 87) or (raw_dim and x.shape[1] != raw_dim):
        raise ValueError('candidate critic expects the original 71 or 87 observable features')
    if not np.isfinite(x).all():
        raise ValueError('nonfinite candidate critic features')
    norm = lambda lo, hi: np.linalg.norm(x[:, lo:hi], axis=1)
    approach = np.linalg.norm(x[:, 29:32] + x[:, 35:38] - x[:, :3] - x[:, 6:9], axis=1)
    arrival = np.maximum(x[:, 61], .001)
    transport_budget = np.maximum(x[:, 63], .001)
    derived = np.column_stack((norm(3, 6), norm(6, 9), norm(9, 12), norm(12, 15),
        np.max(np.abs(x[:, 22:29]), axis=1), norm(32, 35), norm(35, 38), norm(38, 41),
        norm(41, 44), norm(44, 51), np.max(np.abs(x[:, 44:51]), axis=1),
        np.max(np.abs(x[:, 51:58]), axis=1), norm(58, 61), np.min(x[:, 65:69], axis=1),
        approach, approach / arrival, norm(44, 51) / arrival, norm(58, 61) / transport_budget,
        x[:, 61] / np.maximum(x[:, 62], .001)))
    return np.column_stack((x, derived))


def fit_affine(probability, labels):
    """Attempt-uniform loss within one task, with a strictly positive slope."""
    p, y = np.asarray(probability, float), np.asarray(labels, float)
    if p.ndim != 1 or p.shape != y.shape or not len(y) or not np.isfinite(p).all():
        raise ValueError('invalid affine calibration observations')
    z = logit(np.clip(p, 1e-6, 1 - 1e-6))
    def objective(theta):
        calibrated = theta[0] * z + theta[1]
        return float(np.mean(np.logaddexp(0., calibrated) - y * calibrated)
            + .005 * (theta[0] - 1.)**2 + .0001 * theta[1]**2)
    fitted = minimize(objective, [1., 0.], method='L-BFGS-B', bounds=[(.05, 5.), (-8., 8.)])
    if not fitted.success:
        raise ValueError('task affine calibration optimization failed: ' + str(fitted.message))
    return dict(slope=float(fitted.x[0]), intercept=float(fitted.x[1]),
        target='uniform candidate attempts within this task',
        regularization=dict(slope_to_one=.005, intercept_to_zero=.0001),
        slope_bounds=[.05, 5.], intercept_bounds=[-8., 8.])


def calibrate(probability, correction):
    return expit(correction['slope'] * logit(np.clip(np.asarray(probability, float), 1e-6, 1 - 1e-6))
        + correction['intercept'])


def build_task_support(features, source_ids, raw_dim):
    """Calibrate support radius on TRAIN neighbors from a different raw source."""
    x = transform_features(features, raw_dim)
    sources = np.asarray(source_ids)
    if len(x) != len(sources) or len(np.unique(sources)) < 2:
        raise ValueError('support requires two or more TRAIN sources with features')
    mean, scale = x.mean(0), np.maximum(x.std(0), .05)
    normalized = (x - mean) / scale
    tree = cKDTree(normalized)
    _, counts = np.unique(sources, return_counts=True)
    k = min(len(x), int(counts.max()) + 1)
    distances, indices = tree.query(normalized, k=k)
    other = sources[indices] != sources[:, None]
    if not other.any(axis=1).all():
        raise ValueError('TRAIN support could not find a different-source neighbor')
    nearest_other = np.where(other, distances, np.inf).min(axis=1)
    slack = np.maximum(.1 * x.std(0), .02)
    return dict(mean=mean.tolist(), scale=scale.tolist(), lower=(x.min(0) - slack).tolist(),
        upper=(x.max(0) + slack).tolist(), normalized_fit_features=normalized.tolist(),
        maximum_nearest_fit_distance=max(float(np.quantile(nearest_other, .99)), 1e-6),
        source_ids=sources.tolist(), source_count=int(len(np.unique(sources))),
        different_source_distance_quantiles={str(q): float(np.quantile(nearest_other, q)) for q in (.5, .9, .99, 1.)},
        rule='TRAIN-only bounds and q99 nearest other raw-source TRAIN distance; sensor replicas share a source',
        formal_test_used=False, validation_used=False)


class CandidateProbability:
    def __init__(self, checkpoint, *, controller_sha256=None):
        path = Path(checkpoint)
        with path.open('rb') as stream:
            saved = pickle.load(stream)
        if saved.get('format') != FORMAT:
            raise ValueError('unknown candidate critic artifact format')
        self.metadata = saved['metadata']
        if self.metadata['feature_schema_sha256'] != FEATURE_SCHEMA_SHA256:
            raise ValueError('candidate feature schema differs')
        if controller_sha256 and self.metadata['controller_sha256'] != controller_sha256:
            raise ValueError('candidate critic continuation controller differs')
        self.tasks = saved['tasks']
        if any(len(head['models']) != 5 for head in self.tasks.values()):
            raise ValueError('candidate critic requires five positive-bootstrap members per task')
        support = json.loads(path.with_name('value_support.json').read_text())
        if support['value_checkpoint_sha256'] != hashlib.sha256(path.read_bytes()).hexdigest():
            raise ValueError('candidate support belongs to another checkpoint')
        if support['feature_schema_sha256'] != FEATURE_SCHEMA_SHA256:
            raise ValueError('candidate support feature schema differs')
        if support['selection_sha256'] != self.metadata['selection_sha256']:
            raise ValueError('candidate support selection version differs')
        if set(self.tasks) != set(TASKS) or set(support['tasks']) != set(TASKS):
            raise ValueError('candidate critic requires both task-specific heads')
        self.support = support['tasks']
        self.trees = {task: cKDTree(self.support[task]['normalized_fit_features']) for task in TASKS}

    def predict(self, features, task):
        if task not in self.tasks:
            raise ValueError('unknown task for candidate critic')
        x = transform_features(features, self.metadata['input_dim'])
        if not len(x):
            return dict(probability=np.array([]), supported=np.array([], bool),
                ensemble_std=np.array([]), nearest_fit_distance=np.array([]),
                ranking_enabled=bool(self.metadata['per_task'][task]['ranking_enabled']))
        head = self.tasks[task]
        members = np.stack([model.predict_proba(x)[:, 1] for model in head['models']])
        probability = calibrate(members.mean(0), head['calibration'])
        calibrated_members = calibrate(members, head['calibration'])
        support = self.support[task]
        normalized = (x - np.asarray(support['mean'])) / np.asarray(support['scale'])
        distance = self.trees[task].query(normalized)[0]
        supported = distance <= support['maximum_nearest_fit_distance']
        supported &= (x >= np.asarray(support['lower'])).all(1)
        supported &= (x <= np.asarray(support['upper'])).all(1)
        return dict(probability=probability, supported=supported,
            ensemble_std=calibrated_members.std(0), nearest_fit_distance=distance,
            ranking_enabled=bool(self.metadata['per_task'][task]['ranking_enabled']))
