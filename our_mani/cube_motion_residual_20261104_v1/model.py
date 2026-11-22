"""Original GRU/c16/z64 dynamics, refitted to current public physical-clock data.

The old network already integrates a CV prior plus learned acceleration. This
family keeps that architecture. It changes the data/loss/initialization contract,
not a CV wrapper presented as trained dynamics. Public position histories produce
all input channels; object orientation remains an explicitly unobserved identity.
"""
from dataclasses import asdict, dataclass
from pathlib import Path
import hashlib
import inspect
import math

import torch
from torch import nn

from . import VERSION
from ..multitask.context_se3 import SE3ContextNetwork, HISTORY_DIM, STATE_DIM

LEGACY_SHA256 = 'ebec3d1b3a83ceb0d4fb3bc889f8b62376cb4581071f53de29e0b6bd71c162ef'


@dataclass(frozen=True)
class MotionConfig:
    hidden: int = 96
    latent_dim: int = 64
    context_dim: int = 16
    position_center: tuple = (0., 0., .8)
    position_scale: tuple = (.5, .6, .5)
    velocity_scale: tuple = (1., 1., 1.)
    angular_velocity_scale: tuple = (2., 2., 2.)
    maximum_acceleration: float = 5.
    maximum_angular_acceleration: float = 80.

    def __post_init__(self):
        if type(self.hidden) is not int or self.hidden < 8:
            raise ValueError('positive integer GRU width required')
        if self.latent_dim != 64 or self.context_dim != 16:
            raise ValueError('This family retains z64 and c16')
        for key in ('position_center', 'position_scale', 'velocity_scale', 'angular_velocity_scale'):
            values = getattr(self, key)
            if (len(values) != 3 or any(isinstance(v, bool) or not math.isfinite(float(v)) for v in values)
                    or (key != 'position_center' and min(values) <= 0)):
                raise ValueError('finite xyz normalization required: ' + key)
            object.__setattr__(self, key, tuple(float(v) for v in values))
        for value in (self.maximum_acceleration, self.maximum_angular_acceleration):
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError('positive finite residual acceleration bound required')

    def to_dict(self):
        result = asdict(self)
        for key in ('position_center', 'position_scale', 'velocity_scale', 'angular_velocity_scale'):
            result[key] = list(result[key])
        return result

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict) or set(value) != set(cls().to_dict()):
            raise ValueError('Exact motion configuration required')
        return cls(**value)


def public_inputs(model, *, positions, timestamps, confidence, lengths):
    """Prepare original 21/19 tensors from right-padded observed positions only.

    Velocities use least-squares regression over at most five *past* samples.
    The first row has no measured derivative/interval: both placeholders are 0.
    No observation is interpolated, duplicated, supplied by a forecast or GT.
    """
    if not isinstance(positions, torch.Tensor) or positions.ndim != 3 or positions.shape[-1] != 3:
        raise ValueError('public positions[B,T,3] required')
    batch, count, _ = positions.shape
    if batch < 1 or not 5 <= count <= 20 or positions.dtype != torch.float32:
        raise ValueError('FP32 positions, nonempty batch and 5..20 history slots required')
    if (not isinstance(timestamps, torch.Tensor) or timestamps.shape != (batch, count)
            or timestamps.dtype not in (torch.float32, torch.float64)
            or not isinstance(confidence, torch.Tensor) or confidence.shape != (batch, count)
            or confidence.dtype != positions.dtype or not isinstance(lengths, torch.Tensor)
            or lengths.shape != (batch,) or lengths.dtype != torch.int64):
        raise ValueError('public times/confidence/actual lengths have invalid shape or dtype')
    if any(v.device != positions.device for v in (timestamps, confidence, lengths)):
        raise ValueError('public tensors must share one device')
    if any(not bool(torch.isfinite(v).all()) for v in (positions, timestamps, confidence)):
        raise ValueError('all public values including padding must be finite')
    if bool(((lengths < 5) | (lengths > count)).any()):
        raise ValueError('at least five actual observations, no invented frames')
    valid = torch.arange(count, device=positions.device)[None] < lengths[:, None]
    if bool((timestamps[valid] < 0).any()) or bool(((confidence[valid] <= 0) | (confidence[valid] > 1)).any()):
        raise ValueError('nonnegative actual times and positive public confidence required')
    if bool((positions[~valid] != 0).any()) or bool((timestamps[~valid] != 0).any()) or bool((confidence[~valid] != 0).any()):
        raise ValueError('right padding must be zero, never repeated observations')
    intervals = timestamps[:, 1:] - timestamps[:, :-1]
    if not bool(torch.allclose(intervals[valid[:, 1:]], intervals.new_full(intervals[valid[:, 1:]].shape, .05), atol=1e-5, rtol=0)):
        raise ValueError('contiguous actual 50ms public history required; reset on gaps')
    velocity = torch.zeros_like(positions)
    for end in range(1, count):
        start = max(0, end - 4)
        t = timestamps[:, start:end+1]
        p = positions[:, start:end+1]
        centered = t - t.mean(1, keepdim=True)
        denominator = centered.square().sum(1).clamp_min(1e-12)
        slope = (centered[..., None] * (p - p.mean(1, keepdim=True))).sum(1) / denominator[:, None]
        velocity[:, end] = slope.to(positions.dtype)
    velocity = velocity * valid[..., None]
    rotation = torch.eye(3, device=positions.device, dtype=positions.dtype).expand(batch, count, 3, 3)
    angular = torch.zeros_like(positions)
    mode = torch.zeros((batch, count), device=positions.device, dtype=torch.int64)
    observable = torch.zeros((batch, count), device=positions.device, dtype=positions.dtype)
    state = model.normalized(positions, rotation, velocity, angular, mode, observable)
    delta = torch.zeros_like(confidence)
    delta[:, 1:] = intervals.to(positions.dtype) / .05
    history = torch.cat((state, delta[..., None], confidence[..., None]), -1) * valid[..., None]
    ids = lengths - 1
    row = torch.arange(batch, device=positions.device)
    return history, lengths, state[row, ids], timestamps[row, ids]


class CubeMotionResidualGRU(SE3ContextNetwork):
    """Same trainable GRU/SimNorm/state-transition architecture as v671."""

    def __init__(self, config=None):
        self.configuration = config if config is not None else MotionConfig()
        if type(self.configuration) is not MotionConfig:
            raise TypeError('Exact MotionConfig required')
        super().__init__(**self.configuration.to_dict())

    def model_spec(self):
        return dict(version=VERSION, config=self.configuration.to_dict(), history_dim=HISTORY_DIM,
            state_dim=STATE_DIM, time_step_seconds=.05, maximum_steps=16,
            public_history='current/past observed positions and confidence only; 5-point causal OLS velocity',
            first_derivative_and_interval='zero unavailable placeholders, not measured',
            orientation='identity/unobserved; zero spin prior; not Native body axes',
            scope='Cube externally driven free motion only; no attachment/contact dynamics',
            initialization='CV acceleration head zero; optional explicit v671 state transfer')

    @classmethod
    def from_legacy_checkpoint(cls, checkpoint, *, expected_sha256=LEGACY_SHA256, maximum_acceleration=5.):
        path = Path(checkpoint).resolve(strict=True)
        actual_sha = hashlib.sha256(path.read_bytes()).hexdigest()
        if expected_sha256 != LEGACY_SHA256 or actual_sha != expected_sha256:
            raise ValueError('Exact original ebec checkpoint required for declared transfer')
        payload = torch.load(path, map_location='cpu', weights_only=False)
        metadata = payload['metadata']
        if (metadata.get('training_complete') is not True or metadata.get('TEST_used_for_fitting') is not False
                or metadata.get('architecture_sha256') != hashlib.sha256(Path(inspect.getfile(SE3ContextNetwork)).read_bytes()).hexdigest()):
            raise ValueError('Original TRAIN-only architecture provenance required')
        settings = dict(payload['spec'], maximum_acceleration=maximum_acceleration)
        model = cls(MotionConfig.from_dict(settings))
        model.load_state_dict(payload['state_dict'], strict=True)
        nn.init.zeros_(model.acceleration[-1].weight)
        nn.init.zeros_(model.acceleration[-1].bias)
        receipt = dict(checkpoint=str(path), sha256=actual_sha, all_parameter_shapes_transferred=True,
            explicitly_reset_prefix='acceleration.2.', baseline='constant_velocity_before_fine_tuning',
            fine_tune_trainable_parameters='all original GRU/context/latent/dynamics parameters',
            normalization_transferred=True, old_checkpoint_or_source_modified=False)
        return model, receipt

    def forward(self, history, lengths, initial_state, steps=16, dt=.05, context_override=None):
        if type(steps) is not int or not 1 <= steps <= 16 or isinstance(dt, bool) or abs(dt-.05) > 1e-10:
            raise ValueError('This new family supports actual 50ms steps up to .8s only')
        if (not isinstance(history, torch.Tensor) or history.ndim != 3 or history.shape[-1] != 21
                or history.dtype != torch.float32 or not 5 <= history.shape[1] <= 20
                or initial_state.shape != (history.shape[0], 19) or initial_state.dtype != history.dtype
                or lengths.dtype != torch.int64 or lengths.shape != (history.shape[0],)
                or bool(((lengths < 5) | (lengths > history.shape[1])).any())):
            raise ValueError('Original FP32 21-history/19-state with 5..20 actual rows required')
        if any(not bool(torch.isfinite(v).all()) for v in (history, initial_state)):
            raise ValueError('finite physical model input required')
        if bool((initial_state[:, 18] != 0).any()) or not bool((initial_state[:, 15:18] == initial_state.new_tensor([1.,0.,0.])).all()):
            raise ValueError('External free mode and unobserved Cube orientation required')
        identity6 = initial_state.new_tensor([1.,0.,0.,1.,0.,0.])
        if not torch.allclose(initial_state[:, 3:9], identity6.expand(history.shape[0], 6), atol=1e-6, rtol=0):
            raise ValueError('Explicit public identity orientation prior required')
        if bool((initial_state[:, 12:15] != 0).any()):
            raise ValueError('Unobserved angular velocity is the zero prior, not a label')
        if context_override is not None:
            raise ValueError('Production/training forward cannot inject motion context')
        result = super().forward(history, lengths, initial_state, steps=steps, dt=dt)
        at = torch.arange(steps+1, device=history.device, dtype=history.dtype) * dt
        result['cv_positions'] = result['positions'][:, :1] + at[None, :, None] * result['velocities'][:, :1]
        result['residual_accelerations'] = (result['velocities'][:, 1:] - result['velocities'][:, :-1]) / dt
        result['relative_times'] = at
        return result

    def forward_public(self, *, positions, timestamps, confidence, lengths):
        history, lengths, state, cutoff = public_inputs(self, positions=positions, timestamps=timestamps,
            confidence=confidence, lengths=lengths)
        result = self(history, lengths, state, steps=16, dt=.05)
        result['times'] = cutoff[:, None] + torch.arange(17, device=positions.device, dtype=timestamps.dtype)[None] * .05
        return result


def model_from_spec(spec):
    if not isinstance(spec, dict) or spec.get('version') != VERSION:
        raise ValueError('Exact independently versioned Cube residual spec required')
    model = CubeMotionResidualGRU(MotionConfig.from_dict(spec['config']))
    if model.model_spec() != spec:
        raise ValueError('Exact canonical physical model contract required')
    return model
