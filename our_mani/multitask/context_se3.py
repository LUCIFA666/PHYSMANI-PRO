"""Causal per-body context + SimNorm dynamics for the R10 SE(3) interface.

This module supplies an architecture, not a trained model. Inference requires
a checkpoint that declares completed TRAIN fitting and its exact state schema.
Attached bodies are handled by the controller's measured attachment transform,
not extrapolated as if their original external motion continued.
"""
from collections import deque
import hashlib
import json
from pathlib import Path
import threading

import numpy as np
from scipy.spatial.transform import Rotation
import torch
from torch import nn

from ..context_dynamics import SimNorm
from ..contracts import Forecast

SCHEMA = 'r10_se3_world_twist_rot6_mode_observability_v2'
MODES = {'external': 0, 'attached': 1, 'released': 2}
STATE_DIM = 19
HISTORY_DIM = 21


def so3_exp(vector):
    """Differentiable exponential; vector is a world-frame rotation increment."""
    x, y, z = vector.unbind(-1)
    zero = torch.zeros_like(x)
    skew = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), -1)
    skew = skew.reshape(*vector.shape[:-1], 3, 3)
    theta = torch.linalg.vector_norm(vector, dim=-1)
    a = torch.sinc(theta / torch.pi)[..., None, None]
    b = (.5 * torch.sinc(theta / (2 * torch.pi)).square())[..., None, None]
    identity = torch.eye(3, device=vector.device, dtype=vector.dtype)
    return identity + a * skew + b * (skew @ skew)


def rotation_from_6d(values):
    columns = values.reshape(*values.shape[:-1], 3, 2)
    first = nn.functional.normalize(columns[..., 0], dim=-1)
    second = columns[..., 1] - (first * columns[..., 1]).sum(-1, keepdim=True) * first
    second = nn.functional.normalize(second, dim=-1)
    return torch.stack((first, second, torch.linalg.cross(first, second, dim=-1)), -1)


def finite_difference_twist(times, positions, rotations, window=5):
    """Estimate the current world twist using past poses only."""
    t, p, r = [np.asarray(v, dtype=np.float64)[-window:] for v in (times, positions, rotations)]
    if len(t) < 2 or p.shape != (len(t), 3) or r.shape != (len(t), 3, 3):
        raise ValueError('two or more aligned poses are required')
    if not all(np.isfinite(v).all() for v in (t, p, r)) or np.any(np.diff(t) <= 0):
        raise ValueError('nonfinite or noncausal pose history')
    centered = t - t.mean()
    velocity = np.sum(centered[:, None] * (p - p.mean(0)), axis=0) / np.sum(centered ** 2)
    increments = r[1:] @ np.swapaxes(r[:-1], -1, -2)
    angular = Rotation.from_matrix(increments.copy()).as_rotvec() / np.diff(t)[:, None]
    return velocity, np.average(angular, axis=0, weights=np.diff(t))


class SE3ContextNetwork(nn.Module):
    def __init__(self, hidden=96, latent_dim=64, context_dim=16,
                 position_center=(0., 0., .8), position_scale=(.5, .6, .5),
                 velocity_scale=(1., 1., 1.), angular_velocity_scale=(2., 2., 2.),
                 maximum_acceleration=40., maximum_angular_acceleration=80.):
        super().__init__()
        if latent_dim % 8 or context_dim % 8:
            raise ValueError('SimNorm dimensions must be divisible by eight')
        if min(maximum_acceleration, maximum_angular_acceleration) <= 0:
            raise ValueError('acceleration bounds must be positive')
        for values in (position_center, position_scale, velocity_scale, angular_velocity_scale):
            if np.asarray(values).shape != (3,) or not np.isfinite(values).all():
                raise ValueError('normalization requires finite xyz triples')
        for values in (position_scale, velocity_scale, angular_velocity_scale):
            if min(values) <= 0:
                raise ValueError('normalization scales must be positive')
        self.spec = dict(hidden=hidden, latent_dim=latent_dim, context_dim=context_dim,
            position_center=list(position_center), position_scale=list(position_scale),
            velocity_scale=list(velocity_scale), angular_velocity_scale=list(angular_velocity_scale),
            maximum_acceleration=maximum_acceleration,
            maximum_angular_acceleration=maximum_angular_acceleration)
        self.maximum_acceleration = float(maximum_acceleration)
        self.maximum_angular_acceleration = float(maximum_angular_acceleration)
        self.history_encoder = nn.GRU(HISTORY_DIM, hidden, batch_first=True)
        self.context_projection = nn.Sequential(nn.Linear(hidden, context_dim), SimNorm(8))
        self.state_encoder = nn.Sequential(nn.Linear(STATE_DIM, hidden), nn.SiLU(),
                                          nn.Linear(hidden, latent_dim), SimNorm(8))
        self.transition = nn.Sequential(nn.Linear(latent_dim + context_dim + STATE_DIM + 1, hidden),
                                       nn.SiLU(), nn.Linear(hidden, latent_dim), SimNorm(8))
        self.acceleration = nn.Sequential(nn.Linear(latent_dim + context_dim + STATE_DIM, hidden),
                                         nn.SiLU(), nn.Linear(hidden, 6))
        self.state_decoder = nn.Sequential(nn.Linear(latent_dim, hidden), nn.SiLU(),
                                          nn.Linear(hidden, STATE_DIM))
        nn.init.zeros_(self.acceleration[-1].weight)
        nn.init.zeros_(self.acceleration[-1].bias)
        for name, values in (('position_center', position_center), ('position_scale', position_scale),
                             ('velocity_scale', velocity_scale), ('angular_velocity_scale', angular_velocity_scale)):
            self.register_buffer(name, torch.tensor(values, dtype=torch.float32))

    def normalized(self, positions, rotations, velocities, angular_velocities, modes,
                   rotation_observable=None):
        mode = nn.functional.one_hot(modes.to(torch.long), num_classes=3).to(positions.dtype)
        observed = (torch.ones_like(positions[..., :1]) if rotation_observable is None
                    else rotation_observable.to(positions.dtype).reshape(*positions.shape[:-1], 1))
        return torch.cat(((positions - self.position_center) / self.position_scale,
                          rotations[..., :, :2].reshape(*positions.shape[:-1], 6),
                          velocities / self.velocity_scale,
                          angular_velocities / self.angular_velocity_scale, mode, observed), -1)

    def encode_context(self, history, lengths):
        packed = nn.utils.rnn.pack_padded_sequence(history, lengths.detach().cpu(),
                                                   batch_first=True, enforce_sorted=False)
        _, hidden = self.history_encoder(packed)
        return self.context_projection(hidden[-1])

    def forward(self, history, lengths, initial_state, steps=22, dt=.05, context_override=None):
        if steps < 1 or dt <= 0:
            raise ValueError('positive rollout horizon and dt required')
        context = self.encode_context(history, lengths)
        if context_override is not None:
            if context_override.shape != context.shape:
                raise ValueError('context intervention shape differs')
            context = context_override
        position = initial_state[:, :3] * self.position_scale + self.position_center
        rotation = rotation_from_6d(initial_state[:, 3:9])
        velocity = initial_state[:, 9:12] * self.velocity_scale
        angular = initial_state[:, 12:15] * self.angular_velocity_scale
        modes = initial_state[:, 15:18].argmax(-1)
        rotation_observable = initial_state[:, 18]
        z = self.state_encoder(initial_state)
        initial_z = z
        positions, rotations, velocities, angular_velocities, latents = [position], [rotation], [velocity], [angular], [z]
        for _ in range(steps):
            state = self.normalized(position, rotation, velocity, angular, modes, rotation_observable)
            z = self.transition(torch.cat((z, context, state, torch.full_like(state[:, :1], dt / .05)), -1))
            acceleration = torch.tanh(self.acceleration(torch.cat((z, context, state), -1)))
            linear_acceleration = self.maximum_acceleration * acceleration[:, :3]
            # Unobservable spin/gauge has no supervised angular residual. Retain
            # only the causal measured twist (axis-gauge motion uses CV).
            angular_acceleration = (self.maximum_angular_acceleration * acceleration[:, 3:]
                                    * rotation_observable[:, None])
            position = position + dt * velocity + .5 * dt ** 2 * linear_acceleration
            rotation = so3_exp(dt * angular + .5 * dt ** 2 * angular_acceleration) @ rotation
            velocity = velocity + dt * linear_acceleration
            angular = angular + dt * angular_acceleration
            positions.append(position); rotations.append(rotation)
            velocities.append(velocity); angular_velocities.append(angular); latents.append(z)
        return dict(positions=torch.stack(positions, 1), rotations=torch.stack(rotations, 1),
                    velocities=torch.stack(velocities, 1), angular_velocities=torch.stack(angular_velocities, 1),
                    latents=torch.stack(latents, 1), context=context,
                    reconstructed_state=self.state_decoder(initial_z))

    def encoder_sha256(self):
        digest = hashlib.sha256()
        for name, value in sorted(self.state_dict().items()):
            if name.startswith(('history_encoder.', 'context_projection.')):
                digest.update(name.encode()); digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()


def history_inputs(rows, model, maximum=20):
    rows = rows[-maximum:]
    if len(rows) < 5:
        raise ValueError('context requires at least five causal poses')
    times = np.asarray([r['timestamp'] for r in rows], dtype=np.float64)
    if np.any(np.diff(times) <= 0) or np.any(np.diff(times) > .075):
        raise ValueError('context history is noncausal or has a missing observation')
    if len({r['mode'] for r in rows}) != 1:
        raise ValueError('motion modes must reset their context history')
    tensor = lambda key: torch.tensor(np.asarray([r[key] for r in rows]), dtype=torch.float32)
    with torch.no_grad():
        state = model.normalized(tensor('position'), tensor('rotation'), tensor('velocity'),
                                 tensor('angular_velocity'), torch.tensor([r['mode'] for r in rows]),
                                 torch.tensor([r.get('rotation_observable', True) for r in rows]))
    features = np.c_[state.numpy(), np.r_[.05, np.diff(times)] / .05,
                     np.asarray([r['confidence'] for r in rows])].astype(np.float32)
    if not np.isfinite(features).all():
        raise ValueError('nonfinite history')
    padded = np.zeros((maximum, HISTORY_DIM), dtype=np.float32)
    padded[:len(features)] = features
    return torch.tensor(padded[None]), torch.tensor([len(features)]), state[-1:].clone(), float(times[-1])


class MultiBodyContextDynamics:
    name = 'context_simnorm_se3_r10'

    def __init__(self, checkpoint):
        self.path = Path(checkpoint)
        data = torch.load(self.path, map_location='cpu', weights_only=False)
        self.metadata = data['metadata']
        if (self.metadata.get('state_schema') != SCHEMA
                or self.metadata.get('normalization_fit_split') != 'train'
                or self.metadata.get('training_fit_split') != 'train'
                or self.metadata.get('training_complete') is not True
                or self.metadata.get('TEST_used_for_fitting') is not False):
            raise ValueError('a completed TRAIN-only SE3 checkpoint is required')
        source = self.metadata['training_source_manifest']
        source_path = self.path.parent / source['path_relative_to_checkpoint']
        if hashlib.sha256(source_path.read_bytes()).hexdigest() != source['sha256']:
            raise ValueError('actual TRAIN source manifest fingerprint differs')
        entries = json.loads(source_path.read_text())['entries']
        if not entries or any(row['split'] != 'train' for row in entries):
            raise ValueError('fitted state/normalization sources must be TRAIN only')
        keys = [(row['task'], row['raw_episode_id']) for row in entries]
        parents = {(row['task'], row.get('parent_raw_episode_id', row['raw_episode_id'])) for row in entries}
        if (len(keys) != len(set(keys)) or len(keys) != source['cache_episode_count']
                or len(parents) != source['source_episode_count']):
            raise ValueError('TRAIN source identity/coverage differs')
        if self.metadata.get('architecture_sha256') != hashlib.sha256(Path(__file__).read_bytes()).hexdigest():
            raise ValueError('SE3 architecture changed since fitting')
        self.model = SE3ContextNetwork(**data['spec']).eval()
        self.model.load_state_dict(data['state_dict'])
        self.encoder_sha256 = self.model.encoder_sha256()
        if self.metadata['encoder_sha256'] != self.encoder_sha256:
            raise ValueError('context encoder fingerprint differs')
        self.checkpoint_sha256 = hashlib.sha256(self.path.read_bytes()).hexdigest()
        self.maximum_validated_horizon = float(self.metadata['maximum_training_horizon_seconds'])
        self.history, self.lock = {}, threading.RLock()

    def reset(self):
        with self.lock:
            self.history.clear()

    def observe(self, objects, modes=None, orientation_observability=None):
        modes = modes or {}
        orientation_observability = orientation_observability or {}
        if (len({obj.object_id for obj in objects}) != len(objects)
                or any(not obj.visible for obj in objects)
                or any(modes.get(obj.object_id, 'external') not in MODES for obj in objects)
                or any(obj.object_id not in orientation_observability
                       or not isinstance(orientation_observability[obj.object_id], (bool, np.bool_))
                       for obj in objects)):
            raise ValueError('fresh visible unique bodies and known motion modes are required')
        with self.lock:
            for obj in objects:
                mode_name = modes.get(obj.object_id, 'external')
                if mode_name not in MODES:
                    raise ValueError('unknown observed motion mode')
                mode = MODES[mode_name]
                history = self.history.setdefault(obj.object_id, deque(maxlen=20))
                if history and obj.timestamp < history[-1]['timestamp']:
                    raise ValueError('reset context at episode boundaries')
                if history and (mode != history[-1]['mode'] or obj.timestamp - history[-1]['timestamp'] > .075):
                    history.clear()
                if history and obj.timestamp == history[-1]['timestamp']:
                    continue
                history.append(dict(timestamp=float(obj.timestamp), position=obj.pose.position.copy(),
                    rotation=obj.pose.rotation.copy(), velocity=obj.linear_velocity.copy(),
                    angular_velocity=obj.angular_velocity.copy(), confidence=float(obj.confidence), mode=mode,
                    rotation_observable=bool(orientation_observability[obj.object_id])))

    def inputs(self, object_id):
        with self.lock:
            rows = list(self.history.get(object_id, ()))
        return history_inputs(rows, self.model)

    @torch.inference_mode()
    def context_snapshot(self, object_id):
        history, lengths, _, cutoff = self.inputs(object_id)
        return dict(values=self.model.encode_context(history, lengths)[0].numpy().tolist(),
                    source='past_and_current_observed_SE3_only', observed_until=cutoff,
                    history_frames=int(lengths[0]), encoder_sha256=self.encoder_sha256,
                    dynamics_checkpoint_sha256=self.checkpoint_sha256, object_id=object_id)

    @torch.inference_mode()
    def predict(self, objects, horizon, dt):
        if (not objects or any(not obj.visible for obj in objects)
                or not np.isclose(dt, .05) or horizon <= 0
                or horizon > self.maximum_validated_horizon + 1e-8
                or not np.isclose(horizon / dt, round(horizon / dt))):
            raise ValueError('use a nonempty body set and a trained 50ms horizon')
        if len({obj.object_id for obj in objects}) != len(objects):
            raise ValueError('duplicate object IDs')
        if len({float(obj.timestamp) for obj in objects}) != 1:
            raise ValueError('body observations must be time aligned')
        tensors = [self.inputs(obj.object_id) for obj in objects]
        for obj, (_, _, state, cutoff) in zip(objects, tensors):
            if abs(cutoff - obj.timestamp) > 1e-8:
                raise ValueError('history must include each current observation')
            if int(state[0, 15:18].argmax()) == MODES['attached']:
                raise ValueError('attached-body prediction requires measured attachment kinematics from the controller')
        history = torch.cat([v[0] for v in tensors])
        lengths = torch.cat([v[1] for v in tensors])
        state = torch.cat([v[2] for v in tensors])
        steps = int(round(horizon / dt))
        result = self.model(history, lengths, state, steps=steps, dt=dt)
        values = {key: result[key].transpose(0, 1).numpy() for key in
                  ('positions', 'rotations', 'velocities', 'angular_velocities')}
        cutoff = objects[0].timestamp
        return Forecast(tuple(obj.object_id for obj in objects), cutoff + np.arange(steps + 1) * dt,
            values['positions'], values['rotations'], values['velocities'], values['angular_velocities'],
            metadata=dict(method=self.name, state_schema=SCHEMA, checkpoint_sha256=self.checkpoint_sha256,
                context_encoder_sha256=self.encoder_sha256, contexts={obj.object_id: context for obj, context in
                    zip(objects, result['context'].numpy().tolist())},
                full_orientation_observable={obj.object_id: bool(state[i, 18]) for i, obj in enumerate(objects)},
                unobservable_angular_residual_learned=False,
                history_frames={obj.object_id: int(n) for obj, n in zip(objects, lengths)},
                observed_until=float(cutoff), rigid_group_relationships='must be preserved by task adapter',
                learned_contact_or_attachment_dynamics=False, ngff_used=False))
