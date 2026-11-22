"""TRAIN-supervised driver/support residual over a frozen dynamics model.

The released moving targets are externally driven; the collision/free-flight
NGFF checkpoint has no driver input. This optional translation adapter is a
separate model. The original NGFF weights are never updated.
"""
import numpy as np
import torch
from torch import nn


class DriverResidual(nn.Module):
    def __init__(self, dimension=25, hidden=128):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(dimension, hidden), nn.SiLU(),
                                     nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 3))
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, features):
        return self.network(features)


def causal_features(times, positions, velocity, horizon, ngff_minus_cv):
    positions = np.asarray(positions)[-5:]
    times = np.asarray(times)[-5:]
    if len(positions) != 5:
        raise ValueError('driver adapter needs five causal RGB-D positions')
    relative = times-times[-1]
    design = np.c_[np.ones(5), relative, .5*relative**2]
    acceleration = np.clip(np.linalg.lstsq(design, positions, rcond=None)[0][2], -2., 2.)
    return np.r_[positions[-1], velocity, acceleration,
                 (positions[:-1]-positions[-1]).reshape(-1), horizon, ngff_minus_cv].astype(np.float32)


class LoadedDriverAdapter:
    def __init__(self, path):
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        self.mode = checkpoint['mode']
        self.mean, self.std = checkpoint['mean'], checkpoint['std']
        self.model = DriverResidual(hidden=checkpoint['hidden'])
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.eval()
        self.metadata = checkpoint['metadata']

    @torch.inference_mode()
    def predict(self, times, positions, velocity, offsets, frozen_ngff_positions=None):
        position = np.asarray(positions)[-1]
        offsets = np.asarray(offsets)
        cv = position+offsets[:, None]*velocity
        if self.mode == 'ngff_driver_adapter':
            if frozen_ngff_positions is None:
                raise ValueError('NGFF residual requires the frozen NGFF forecast')
            base = np.asarray(frozen_ngff_positions)
            differences = base-cv
        else:
            base, differences = cv, np.zeros_like(cv)
        features = np.array([causal_features(times, positions, velocity, h, d) for h, d in zip(offsets, differences)])
        normalized = torch.as_tensor((features-self.mean)/self.std, dtype=torch.float32)
        residual = self.model(normalized).numpy()
        return base+offsets[:, None]*residual
