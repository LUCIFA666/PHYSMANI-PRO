"""SE3 dynamics objectives and parent-source balanced fitting helpers."""
from collections import defaultdict

import numpy as np
import torch


def batch_tensors(samples, indices, device):
    rows = [samples[int(i)] for i in indices]
    result = {key: torch.as_tensor(np.asarray([r[key] for r in rows]), device=device)
              for key in ('history', 'length', 'state', 'mask', 'rotation_mask', 'rotation_observable')}
    result['targets'] = {key: torch.as_tensor(np.asarray([r['targets'][key] for r in rows]),
                                            dtype=torch.float32, device=device)
                         for key in ('positions', 'rotations', 'linear_velocities', 'angular_velocities')}
    for key, value in result['targets'].items():
        mask = result['rotation_mask'] if key in ('rotations', 'angular_velocities') else result['mask']
        while mask.ndim < value.ndim:
            mask = mask[..., None]
        result['targets'][key] = torch.where(mask, value, torch.zeros_like(value))
    return result


def masked_window_mean(values, mask):
    counts = mask.sum(-1)
    return (values * mask).sum(-1) / counts.clamp_min(1)


def loss_per_window(model, data, steps=22, prediction=None):
    prediction = prediction or model(data['history'], data['length'], data['state'], steps=steps)
    mask, rm = data['mask'][:, :steps + 1], data['rotation_mask'][:, :steps + 1]
    targets = {key: value[:, :steps + 1] for key, value in data['targets'].items()}
    p = ((prediction['positions'] - targets['positions']) / .05).square().mean(-1)
    v = ((prediction['velocities'] - targets['linear_velocities']) / .5).square().mean(-1)
    # Chordal SO3 loss has a finite derivative at identity and ignores rotation gauges.
    r = ((prediction['rotations'] - targets['rotations']) / .5).square().mean((-1, -2))
    w = ((prediction['angular_velocities'] - targets['angular_velocities']) / 2.).square().mean(-1)
    state_mask = torch.ones_like(data['state'])
    state_mask[:, 3:9] *= data['rotation_observable'][:, None]
    state_mask[:, 12:15] *= data['rotation_observable'][:, None]
    reconstruction = ((prediction['reconstructed_state'] - data['state']).square() * state_mask).sum(-1)
    reconstruction /= state_mask.sum(-1).clamp_min(1)
    return (masked_window_mean(p, mask) + .05 * masked_window_mean(v, mask)
            + .1 * masked_window_mean(r, rm) + .01 * masked_window_mean(w, rm)
            + .01 * reconstruction)


def balanced_parent_indices(samples, windows_per_parent, rng):
    """Every real TRAIN parent contributes equally, regardless of replica count."""
    if windows_per_parent < 1:
        raise ValueError('at least one optimization example per parent required')
    grouped = defaultdict(list)
    for i, row in enumerate(samples):
        if row['source_key'][0] != 'train':
            raise ValueError('optimizer sampler only accepts TRAIN sources')
        grouped[row['source_key']].append(i)
    indices = np.concatenate([rng.choice(values, windows_per_parent,
                                         replace=len(values) < windows_per_parent)
                              for _, values in sorted(grouped.items())])
    rng.shuffle(indices)
    return indices


def macro_parent(values, samples):
    grouped = defaultdict(list)
    for value, row in zip(values, samples):
        grouped[row['source_key']].append(float(value))
    if not grouped:
        raise ValueError('no supervised sources')
    return float(np.mean([np.mean(v) for v in grouped.values()]))


@torch.inference_mode()
def evaluate(model, samples, device, steps=22, batch_size=256, interventions=False):
    """VAL diagnostics equal-weight real source episodes, not long trajectories."""
    was_training = model.training
    model.eval()
    losses = []
    metrics = defaultdict(lambda: defaultdict(list))
    horizons = sorted(set(k for k in (1, 2, 5, 10, 22, steps) if k <= steps))
    for begin in range(0, len(samples), batch_size):
        selected = samples[begin:begin + batch_size]
        data = batch_tensors(samples, range(begin, begin + len(selected)), device)
        result = model(data['history'], data['length'], data['state'], steps=steps)
        losses.extend(loss_per_window(model, data, steps, result).cpu().tolist())
        dt = torch.arange(steps + 1, device=device, dtype=torch.float32) * .05
        state = data['state']
        p0 = state[:, :3] * model.position_scale + model.position_center
        v0 = state[:, 9:12] * model.velocity_scale
        from .context_se3 import rotation_from_6d, so3_exp
        r0 = rotation_from_6d(state[:, 3:9])
        w0 = state[:, 12:15] * model.angular_velocity_scale
        predictions = {'context': result, 'CV': {
            'positions': p0[:, None] + dt[None, :, None] * v0[:, None],
            'rotations': so3_exp(dt[None, :, None] * w0[:, None]) @ r0[:, None]}}
        if interventions:
            # Uniform SimNorm groups are a valid context intervention, not all-zero vectors.
            predictions['uniform_context'] = model(data['history'], data['length'], state, steps=steps,
                context_override=torch.full_like(result['context'], 1 / 8))
        for name, pred in predictions.items():
            distance = torch.linalg.vector_norm(pred['positions'] - data['targets']['positions'], dim=-1)
            relative = pred['rotations'] @ data['targets']['rotations'].transpose(-1, -2)
            cosine = ((relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(-1, 1)
            angles = torch.acos(cosine)
            for k in horizons:
                for i, row in enumerate(selected):
                    if not bool(data['mask'][i, k]):
                        continue
                    valid = data['mask'][i, 1:k + 1]
                    values = [float(distance[i, 1:k + 1][valid].mean()), float(distance[i, k])]
                    rotation = float(angles[i, k]) if bool(data['rotation_mask'][i, k]) else None
                    metrics[(name, k)][row['source_key']].append((*values, rotation))
    output = {}
    for (name, k), parents in metrics.items():
        rows = [np.mean(np.asarray(v, dtype=object)[:, :2].astype(float), axis=0) for v in parents.values()]
        rotation = [np.mean([x[2] for x in v if x[2] is not None]) for v in parents.values()
                    if any(x[2] is not None for x in v)]
        per_task = {}
        for task in sorted({key[1] for key in parents}):
            values = [np.mean(np.asarray(v, dtype=object)[:, :2].astype(float), axis=0)
                      for key, v in parents.items() if key[1] == task]
            per_task[task] = dict(source_episodes=len(values),
                macro_ADE_m=float(np.mean(values, axis=0)[0]), macro_FDE_m=float(np.mean(values, axis=0)[1]))
        output.setdefault(name, {})[str(round(k * .05, 3))] = dict(
            source_episodes=len(parents), windows=sum(map(len, parents.values())),
            macro_ADE_m=float(np.mean(rows, axis=0)[0]), macro_FDE_m=float(np.mean(rows, axis=0)[1]),
            observable_rotation_endpoint_radians=float(np.mean(rotation)) if rotation else None,
            rotation_source_episodes=len(rotation), by_task=per_task)
    model.train(was_training)
    return dict(selection_loss=macro_parent(losses, samples), metrics=output)
