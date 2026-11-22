"""Small frozen local servo and success-probability models trained on rollouts."""
from pathlib import Path
import hashlib
import json
import numpy as np
import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, inputs, outputs, hidden=128):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(inputs, hidden), nn.SiLU(),
                                     nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, outputs))

    def forward(self, inputs):
        return self.network(inputs)


class LocalServo:
    def __init__(self, checkpoint):
        self.mean, self.std = checkpoint['feature_mean'], checkpoint['feature_std']
        self.action_mean, self.action_std = checkpoint['action_mean'], checkpoint['action_std']
        self.model = MLP(17, 6, checkpoint['hidden'])
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.eval()
        self.metadata = checkpoint['metadata']

    @classmethod
    def load(cls, path):
        return cls(torch.load(Path(path), map_location='cpu', weights_only=False))

    @torch.inference_mode()
    def __call__(self, features):
        x = torch.tensor((features-self.mean)/self.std, dtype=torch.float32)
        return self.model(x).numpy()*self.action_std+self.action_mean


class SuccessValue:
    def __init__(self, checkpoint):
        self.mean, self.std = checkpoint['feature_mean'], checkpoint['feature_std']
        if np.asarray(self.mean).shape != (18,):
            raise ValueError('context-conditioned checkpoints require SuccessProbability and causal context inputs')
        self.model = MLP(18, 1, checkpoint['hidden'])
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.eval()
        self.temperature = checkpoint.get('temperature', 1.)
        self.metadata = checkpoint['metadata']

    @classmethod
    def load(cls, path):
        return cls(torch.load(Path(path), map_location='cpu', weights_only=False))

    @torch.inference_mode()
    def __call__(self, features):
        array = np.asarray(features)
        x = torch.tensor((array-self.mean)/self.std, dtype=torch.float32)
        return torch.sigmoid(self.model(x)/self.temperature).numpy()[..., 0]


class CandidateSuccessValue:
    """Evaluate each region/time/twist using its predicted local capture state."""
    def __init__(self, checkpoint, *, controller_sha256, capture_budget=3.):
        self.value = SuccessValue.load(checkpoint)
        if self.value.metadata['controller_sha256'] != controller_sha256:
            raise ValueError('Value was trained for a different local controller')
        self.capture_budget = capture_budget
        from scipy.spatial import cKDTree
        path = Path(checkpoint).with_name('value_support.json')
        if not path.exists():
            raise ValueError('TRAIN-only Value input support guard is required')
        self.support = json.loads(path.read_text())
        if self.support['value_checkpoint_sha256'] != hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest():
            raise ValueError('Value support guard belongs to different weights')
        self.support_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        self.support_tree = cKDTree(np.asarray(self.support['normalized_fit_features']))

    def in_support(self, features):
        features = np.asarray(features)
        return (np.isfinite(features).all() and np.all(features >= self.support['lower'])
                and np.all(features <= self.support['upper'])
                and self.support_tree.query((features-self.support['mean'])/self.support['scale'])[0]
                    <= self.support['maximum_nearest_fit_distance'])

    def __call__(self, *, region, object_pose, object_velocity, object_angular_velocity,
                 robot_pose, robot_velocity, robot_angular_velocity):
        from scipy.spatial.transform import Rotation
        from .contracts import Pose
        if (Rotation.from_matrix(object_pose.rotation.copy()).magnitude() > .1
                or np.linalg.norm(object_angular_velocity) > .1):
            return float('nan')  # Only world-aligned moving-pick boxes were trained.
        orientation = Rotation.from_matrix(robot_pose.rotation@Pose.from_xyzw([0, 0, 0, 0, 1, 0, 0]).rotation.T).as_rotvec()
        features = np.r_[robot_pose.position-object_pose.position, robot_velocity-object_velocity,
                         object_velocity, object_angular_velocity, orientation, 1., 0., self.capture_budget]
        if not self.in_support(features):
            return float('nan')
        return float(self.value(features))


def probability_metrics(probabilities, labels):
    p, y = np.asarray(probabilities), np.asarray(labels)
    from scipy.stats import rankdata
    positive, negative = int(y.sum()), int((1-y).sum())
    auc = float((rankdata(p)[y == 1].sum()-positive*(positive+1)/2)/(positive*negative)) if positive and negative else None
    clipped = np.clip(p, 1e-6, 1-1e-6)
    ece, bins = 0., []
    for i in range(5):
        chosen = (p >= i/5) & (p < (i+1)/5 if i < 4 else p <= 1.)
        if chosen.any():
            mean, actual = float(p[chosen].mean()), float(y[chosen].mean())
            ece += chosen.mean()*abs(mean-actual)
            bins.append({'low': i/5, 'high': (i+1)/5, 'n': int(chosen.sum()), 'probability': mean, 'actual': actual})
    return {'n': len(y), 'positive': positive, 'brier': float(np.mean((p-y)**2)),
            'log_loss': float(-np.mean(y*np.log(clipped)+(1-y)*np.log(1-clipped))),
            'auroc': auc, 'ece_5_bins': float(ece), 'calibration_bins': bins,
            'accuracy_at_half': float(np.mean((p >= .5) == y))}
