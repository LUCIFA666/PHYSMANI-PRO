"""Pipeline v1: causal history context and SimNorm latent dynamics.

Position/velocity retain SI units; a learned latent transition predicts their
acceleration. The history context conditions every transition. No pixels are
generated, and no NGFF weights are used.
"""
from collections import deque
import hashlib
from pathlib import Path
import threading

import numpy as np
import torch
from torch import nn

from .contracts import Forecast

POSITION_CENTER = np.array([.2, 0., .775], dtype=np.float32)
POSITION_SCALE = np.array([.25, .6, .1], dtype=np.float32)
VELOCITY_SCALE = np.array([.5, .5, .5], dtype=np.float32)


class SimNorm(nn.Module):
    def __init__(self, group_size=8):
        super().__init__()
        self.group_size = group_size

    def forward(self, values):
        if values.shape[-1] % self.group_size:
            raise ValueError('latent dimension must be divisible by SimNorm group size')
        return values.reshape(*values.shape[:-1], -1, self.group_size).softmax(-1).reshape_as(values)


def normalized_state(position, velocity):
    return np.concatenate([(np.asarray(position)-POSITION_CENTER)/POSITION_SCALE,
                           np.asarray(velocity)/VELOCITY_SCALE], axis=-1).astype(np.float32)


def history_features(times, positions, velocities, confidence, maximum=20):
    times, positions, velocities, confidence = [np.asarray(x)[-maximum:] for x in (times, positions, velocities, confidence)]
    if len(times) < 5 or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError('context requires five or more increasing causal observations')
    if times[-1]-times[-2] > .075 or np.any(np.diff(times) > .075):
        raise ValueError('context history has missing observations')
    state = normalized_state(positions, velocities)
    dt = np.r_[.05, np.diff(times)]/.05
    features = np.c_[state, dt, confidence].astype(np.float32)
    if not np.isfinite(features).all():
        raise ValueError('non-finite causal context history')
    padded = np.zeros((maximum, 8), dtype=np.float32)
    padded[:len(features)] = features
    return padded, len(features), state[-1]


class ContextDynamicsNetwork(nn.Module):
    def __init__(self, hidden=64, latent_dim=64, context_dim=16, normalization='simnorm'):
        super().__init__()
        self.spec = dict(hidden=hidden, latent_dim=latent_dim, context_dim=context_dim, normalization=normalization)
        norm = lambda dim: SimNorm(8) if normalization == 'simnorm' else nn.LayerNorm(dim)
        self.history_encoder = nn.GRU(8, hidden, batch_first=True)
        self.context_projection = nn.Sequential(nn.Linear(hidden, context_dim), norm(context_dim))
        self.state_encoder = nn.Sequential(nn.Linear(6, hidden), nn.SiLU(), nn.Linear(hidden, latent_dim), norm(latent_dim))
        self.transition = nn.Sequential(nn.Linear(latent_dim+context_dim+7, hidden), nn.SiLU(),
                                        nn.Linear(hidden, latent_dim), norm(latent_dim))
        self.acceleration = nn.Sequential(nn.Linear(latent_dim+context_dim+6, hidden), nn.SiLU(), nn.Linear(hidden, 3))
        self.state_decoder = nn.Sequential(nn.Linear(latent_dim, hidden), nn.SiLU(), nn.Linear(hidden, 6))
        nn.init.zeros_(self.acceleration[-1].weight)
        nn.init.zeros_(self.acceleration[-1].bias)
        self.register_buffer('position_center', torch.tensor(POSITION_CENTER))
        self.register_buffer('position_scale', torch.tensor(POSITION_SCALE))
        self.register_buffer('velocity_scale', torch.tensor(VELOCITY_SCALE))

    def encode_context(self, history, lengths):
        packed = nn.utils.rnn.pack_padded_sequence(history, lengths.detach().cpu(), batch_first=True, enforce_sorted=False)
        _, hidden = self.history_encoder(packed)
        return self.context_projection(hidden[-1])

    def forward(self, history, lengths, initial_state, steps=22, dt=.05, context_override=None):
        context = self.encode_context(history, lengths)
        if context_override is not None:
            if context_override.shape != context.shape:
                raise ValueError('context intervention shape differs')
            context = context_override
        z = self.state_encoder(initial_state)
        initial_z = z
        position = initial_state[:, :3]*self.position_scale+self.position_center
        velocity = initial_state[:, 3:]*self.velocity_scale
        positions, velocities, latents = [position], [velocity], [z]
        for _ in range(steps):
            state = torch.cat([(position-self.position_center)/self.position_scale, velocity/self.velocity_scale], -1)
            z = self.transition(torch.cat([z, context, state, torch.full_like(state[:, :1], dt/.05)], -1))
            acceleration = 40.*torch.tanh(self.acceleration(torch.cat([z, context, state], -1)))
            position = position+dt*velocity+.5*dt*dt*acceleration
            velocity = velocity+dt*acceleration
            positions.append(position)
            velocities.append(velocity)
            latents.append(z)
        return {'positions': torch.stack(positions, 1), 'velocities': torch.stack(velocities, 1),
                'latents': torch.stack(latents, 1), 'context': context,
                'reconstructed_state': self.state_decoder(initial_z)}

    def encoder_sha256(self):
        digest = hashlib.sha256()
        for name, value in sorted(self.state_dict().items()):
            if name.startswith(('history_encoder.', 'context_projection.')):
                digest.update(name.encode())
                digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()


class ContextLatentDynamics:
    name = 'pipeline_v1_context_simnorm'

    def __init__(self, checkpoint):
        self.path = Path(checkpoint)
        data = torch.load(self.path, map_location='cpu', weights_only=False)
        self.model = ContextDynamicsNetwork(**data['spec'])
        self.model.load_state_dict(data['state_dict'])
        self.model.eval()
        self.metadata = data['metadata']
        self.checkpoint_sha256 = hashlib.sha256(self.path.read_bytes()).hexdigest()
        self.encoder_sha256 = self.model.encoder_sha256()
        if self.metadata['encoder_sha256'] != self.encoder_sha256:
            raise ValueError('causal context encoder fingerprint differs')
        self.maximum_validated_horizon = self.metadata['maximum_training_horizon_seconds']
        self.history, self.lock = {}, threading.RLock()

    def reset(self):
        with self.lock:
            self.history.clear()

    def observe(self, objects):
        with self.lock:
            for obj in objects:
                h = self.history.setdefault(obj.object_id, deque(maxlen=20))
                if h and obj.timestamp < h[-1][0]:
                    raise ValueError('context history must reset at episode boundaries')
                if h and obj.timestamp == h[-1][0]:
                    continue
                if h and obj.timestamp-h[-1][0] > .075:
                    h.clear()
                h.append((obj.timestamp, obj.pose.position.copy(), obj.linear_velocity.copy(), obj.confidence))

    def inputs(self, object_id):
        with self.lock:
            rows = list(self.history.get(object_id, ()))
        if not rows:
            raise ValueError('context history is empty')
        features, length, state = history_features(*zip(*rows))
        return torch.tensor(features[None]), torch.tensor([length]), torch.tensor(state[None]), rows[-1][0]

    @torch.inference_mode()
    def context_snapshot(self, object_id='target'):
        history, lengths, _, cutoff = self.inputs(object_id)
        values = self.model.encode_context(history, lengths)[0].numpy()
        return {'values': values.tolist(), 'encoder_sha256': self.encoder_sha256,
                'source': 'past_observations', 'observed_until': float(cutoff),
                'history_frames': int(lengths[0]), 'dynamics_checkpoint_sha256': self.checkpoint_sha256}

    @torch.inference_mode()
    def predict(self, objects, horizon, dt):
        if len(objects) != 1 or not np.isclose(dt, .05) or horizon > self.maximum_validated_horizon+1e-8:
            raise ValueError('v1 supports one object, 50ms steps and the trained forecast horizon')
        obj = objects[0]
        history, lengths, state, cutoff = self.inputs(obj.object_id)
        if abs(cutoff-obj.timestamp) > 1e-8:
            raise ValueError('context must include the current observation')
        steps = int(round(horizon/dt))
        prediction = self.model(history, lengths, state, steps=steps, dt=dt)
        positions = prediction['positions'][0].numpy()
        velocities = prediction['velocities'][0].numpy()
        rotations = np.broadcast_to(obj.pose.rotation, (steps+1, 1, 3, 3)).copy()
        return Forecast((obj.object_id,), obj.timestamp+np.arange(steps+1)*dt,
            positions[:, None], rotations, velocities[:, None], np.zeros((steps+1, 1, 3)),
            metadata={'method': self.name, 'checkpoint_sha256': self.checkpoint_sha256,
                      'context': prediction['context'][0].numpy().tolist(),
                      'context_encoder_sha256': self.encoder_sha256,
                      'history_frames': int(lengths[0]), 'ngff_used': False})
