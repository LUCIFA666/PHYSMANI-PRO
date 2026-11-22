"""Independent encoder-only Cube student. Explicit train/evaluate, CPU check default.

All context/forecast metrics are development diagnostics. This module does not
modify the old GRU/P4 source pins or construct a live controller/Native backend.
"""
from copy import deepcopy
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import time

import numpy as np
import torch

from ..multitask.context_se3 import SE3ContextNetwork
from . import VERSION
from .data import ROOT, TEACHER_SOURCES, sha, canonical, load_registry, load_windows, features, dropout_indices, load_development_validation

PREFIXES = ('history_encoder.', 'context_projection.')
CONFIG = Path(__file__).with_name('fixed_config.json')


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def atomic_status(path, value):
    temporary = path.with_name(path.name+'.temporary')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def config():
    value = json.loads(CONFIG.read_text())
    if value['version'] != VERSION or value['trainable_prefixes'] != list(PREFIXES):
        raise ValueError('Exact independently versioned fixed configuration required')
    return value


def implementation_sources():
    names = ('our_mani/context_gap_dropout_20261103_v1/__init__.py',
             'our_mani/context_gap_dropout_20261103_v1/data.py',
             'our_mani/context_gap_dropout_20261103_v1/training.py',
             'our_mani/context_gap_dropout_20261103_v1/compatibility.py',
             'our_mani/context_gap_dropout_20261103_v1/fixed_config.json',
             'scripts/train_context_gap_dropout_20261103_v1.py',
             'our_mani/multitask/context_se3.py', 'our_mani/context_dynamics.py',
             'our_mani/contracts.py', 'our_mani/local_mpc_p4_20261103/model.py',
             'our_mani/physmani_data.py', 'scripts/prepare_domain_tracks.py',
             'scripts/prepare_joint16_track_adapter_v31_r10.py')
    return {str((ROOT/name).resolve(strict=True)): sha(ROOT/name) for name in names}


def tensor_digest(state, *, encoder=False):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        if name.startswith(PREFIXES) != encoder:
            continue
        data = value.detach().cpu().contiguous().numpy()
        digest.update((name+str(data.dtype)+str(data.shape)).encode())
        digest.update(data.tobytes())
    return digest.hexdigest()


def load_teacher_student():
    settings = config()
    path = ROOT/settings['teacher_checkpoint']
    if sha(path) != settings['teacher_sha256']:
        raise ValueError('Original frozen ebec teacher identity required')
    payload = torch.load(path, map_location='cpu', weights_only=False)
    metadata = payload['metadata']
    if (metadata['architecture_sha256'] != sha(ROOT/'our_mani/multitask/context_se3.py')
            or metadata['encoder_sha256'] is None or metadata['training_complete'] is not True
            or metadata['TEST_used_for_fitting'] is not False
            or metadata['training_source_manifest']['sha256'] != sha(TEACHER_SOURCES)):
        raise ValueError('Exact original teacher architecture/completed TRAIN provenance required')
    teacher = SE3ContextNetwork(**payload['spec']).eval()
    if (payload['spec']['context_dim'], payload['spec']['latent_dim']) != (16, 64):
        raise ValueError('Original SimNorm c16/z64 interface required')
    teacher.load_state_dict(payload['state_dict'], strict=True)
    teacher.requires_grad_(False)
    if teacher.encoder_sha256() != metadata['encoder_sha256']:
        raise ValueError('Original context encoder bytes differ')
    student = deepcopy(teacher)
    for name, parameter in student.named_parameters():
        parameter.requires_grad_(name.startswith(PREFIXES))
    return teacher, student, payload['spec']


def make_batch(windows, indices, model, rng, *, mode=None):
    dense, dropped, traces = [], [], []
    for index in indices:
        window = windows[int(index)]
        dense.append(features(window, model))
        keep, trace = dropout_indices(window, rng, mode)
        dropped.append(features(window, model, keep, recompute_twist=True))
        traces.append(trace)
    def stack(values):
        return (torch.from_numpy(np.stack([x['history'] for x in values])),
                torch.tensor([x['length'] for x in values], dtype=torch.long),
                torch.from_numpy(np.stack([x['state'] for x in values])))
    return stack(dense), stack(dropped), traces


def loss_components(teacher, student, dense, dropped, device):
    dh, dl, _ = dense
    gh, gl, _ = dropped
    with torch.no_grad():
        target = teacher.encode_context(dh.to(device), dl)
    dense_c = student.encode_context(dh.to(device), dl)
    gap_c = student.encode_context(gh.to(device), gl)
    drop_loss = (gap_c-target).square().mean(-1)
    replay_loss = (dense_c-target).square().mean(-1)
    return dict(total=drop_loss+replay_loss, drop_context=drop_loss, dense_replay_context=replay_loss)


def frozen_identity(registry_path, files):
    settings = config()
    return dict(version=VERSION, source_sha256=implementation_sources(),
        registry=dict(path=str(Path(registry_path).resolve()), sha256=sha(registry_path)),
        teacher=dict(path=str(ROOT/settings['teacher_checkpoint']), sha256=settings['teacher_sha256']),
        teacher_training_sources=dict(path=str(TEACHER_SOURCES), sha256=sha(TEACHER_SOURCES)),
        TRAIN_cache_sha256=files, configuration=settings)


def verify_identity(identity):
    if implementation_sources() != identity['source_sha256']:
        raise ValueError('Student method source changed')
    pointers = [identity['registry'], identity['teacher'], identity['teacher_training_sources']]
    if any(sha(p['path']) != p['sha256'] for p in pointers):
        raise ValueError('Teacher/partition identity changed')
    if any(sha(name) != digest for name, digest in identity['TRAIN_cache_sha256'].items()):
        raise ValueError('Original parent cache changed')


def selection_context_metrics(teacher, student, windows):
    """Fixed per-anchor seed; independent of training RNG/epoch/batch order."""
    teacher.eval()
    student.eval()
    grouped = defaultdict(list)
    with torch.no_grad():
        for index, window in enumerate(windows):
            values = []
            for mode in ('isolated', 'burst'):
                seed = int(canonical([list(window.parent), window.anchor, mode, config()['seed']])[:16], 16)
                dense, gap, _ = make_batch(windows, [index], teacher.cpu(), np.random.default_rng(seed), mode=mode)
                result = loss_components(teacher, student, dense, gap, 'cpu')
                values.append(float(result['total'][0]))
            grouped[window.parent].append(float(np.mean(values)))
    parent_means = {':'.join(key): float(np.mean(value)) for key, value in grouped.items()}
    return dict(selection_loss=float(np.mean(list(parent_means.values()))), parent_mean_context_loss=parent_means,
                parents=len(grouped), anchors=len(windows), fixed_dropout_modes=['isolated', 'burst'])


def prediction_metrics(teacher, student, windows):
    """Full selection anchors, tracked future observations only; no physical claim."""
    teacher.cpu().eval()
    student.cpu().eval()
    errors = defaultdict(list)
    per_parent_errors = defaultdict(lambda: defaultdict(list))
    context_differences = defaultdict(list)
    retained_ledger = []
    counts = CounterLike()
    with torch.inference_mode():
        for index, window in enumerate(windows):
            dense = features(window, teacher)
            td = (torch.from_numpy(dense['history'][None]), torch.tensor([dense['length']]), torch.from_numpy(dense['state'][None]))
            teacher_c = teacher.encode_context(td[0], td[1])
            variants = [('teacher_dense', teacher, td), ('student_dense', student, td)]
            for mode in ('isolated', 'burst'):
                seed = int(canonical([list(window.parent), window.anchor, mode, config()['seed']])[:16], 16)
                _, dropped, traces = make_batch(windows, [index], teacher, np.random.default_rng(seed), mode=mode)
                variants.extend([('teacher_'+mode, teacher, dropped), ('student_'+mode, student, dropped)])
                retained_ledger.append(dict(parent=list(window.parent), anchor=window.anchor, mode=mode,
                    seed=seed, trace=traces[0], paired_input_sha256=hashlib.sha256(dropped[0].numpy().tobytes()+dropped[2].numpy().tobytes()).hexdigest()))
            for mode, model, batch in variants:
                prediction = model(batch[0], batch[1], batch[2], steps=16, dt=.05)
                current_c = prediction['context']
                context_differences[mode].append(float(torch.linalg.vector_norm(current_c-teacher_c)))
                for step in (2, 4, 8, 16):
                    key = mode+':'+str(step*.05)+'s'
                    if not window.future_valid[step]:
                        counts[key+':label_unavailable'] += 1
                        continue
                    error = float(np.linalg.norm(prediction['positions'][0, step].numpy()-window.future_positions[step]))
                    errors[key].append(error)
                    per_parent_errors[':'.join(window.parent)][key].append(error)
                    counts[key+':scored'] += 1
    def stats(value):
        return dict(count=len(value), mean=float(np.mean(value)), p95=float(np.quantile(value, .95)), max=float(np.max(value))) if value else dict(count=0)
    return dict(position_errors_m={key: stats(value) for key, value in errors.items()},
                per_parent_position_errors_m={parent: {key: stats(value) for key, value in values.items()}
                                              for parent, values in per_parent_errors.items()},
                context_L2={key: stats(value) for key, value in context_differences.items()}, counts=dict(counts),
                paired_fixed_seed_retained_observation_ledger=retained_ledger,
                source='tracked_future_observations_not_Native_GT', selection_parent_count=len(set(w.parent for w in windows)),
                horizon_time_basis=config()['time_basis'], physical_time_forecasting_qualified=False,
                teacher_and_student_forward_calls=len(windows)*6,
                frozen_teacher_forward_calls=len(windows)*3, student_forward_calls=len(windows)*3,
                candidate_compatibility=dict(status='not_evaluated_without_frozen_selection_public_feature_bundles', qualified=False),
                live_control_admitted=False, other14_tasks_compatibility_claimed=False)


class CounterLike(defaultdict):
    def __init__(self):
        super().__init__(int)


def prepare(registry_path):
    torch.set_num_threads(2)
    registry = load_registry(registry_path)
    windows, ledger, files = load_windows(registry)
    teacher, student, spec = load_teacher_student()
    identity = frozen_identity(registry_path, files)
    return registry, windows, ledger, teacher, student, spec, identity


def check(registry_path, output=None):
    registry, windows, ledger, teacher, student, spec, identity = prepare(registry_path)
    verify_identity(identity)
    report = dict(version=VERSION, CPU_check=True, GPU_used=False, optimizer_constructed=False,
        optimizer_steps=0, warmup_or_model_forward_calls=0, registry_roles=dict(CounterLikeFromRows(registry['rows'])),
        windows_by_role=dict(CounterLikeFromRows([dict(role=w.role) for w in windows])),
        actual_timestamp_parent_ledger=ledger, teacher_spec=spec,
        trainable_parameter_names=[name for name, p in student.named_parameters() if p.requires_grad],
        frozen_body_digest=tensor_digest(student.state_dict()), identity=identity,
        P4_or_live_runtime_admitted=False, other14_tasks_compatibility_claimed=False, TEST_read=False)
    report.update(time_basis=config()['time_basis'], physical_time_robustness_qualified=False,
                  metric_3cm_position_training=False)
    if output is not None:
        output = Path(output)
        output.mkdir(parents=True, exist_ok=False)
        write(output/'CPU_check.json', report)
    return report


def CounterLikeFromRows(rows):
    values = CounterLike()
    for row in rows:
        values[row['role']] += 1
    return values


def train(registry_path, output, *, device='cuda'):
    """Only an explicit CLI --train calls this function or constructs AdamW."""
    settings = config()
    torch.manual_seed(settings['seed'])
    rng = np.random.default_rng(settings['seed'])
    registry, windows, ledger, teacher, student, spec, identity = prepare(registry_path)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    write(output/'identity.json', identity)
    write(output/'data_access.json', dict(parent_ledger=ledger, TEST_read=False, success_based_filter=False))
    verify_identity(identity)
    frozen_body = tensor_digest(student.state_dict())
    original_teacher = {key: value.clone() for key, value in teacher.state_dict().items()}
    fit = [w for w in windows if w.role == 'fit']
    selection = [w for w in windows if w.role == 'selection']
    groups = defaultdict(list)
    for index, window in enumerate(fit):
        groups[window.parent].append(index)
    # Teacher evaluation and selection are CPU2; only the explicitly requested
    # fitting batches use CUDA. No calibration cohort is loaded or scored.
    student.to(device)
    normalizer = deepcopy(teacher).cpu().eval()
    teacher.to(device)
    optimizer = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad],
                                 lr=settings['learning_rate'], weight_decay=settings['weight_decay'])
    history, actual_steps, selected, best = [], 0, None, math.inf
    selected_state = None
    for epoch in range(settings['epochs']):
        indices = np.concatenate([rng.choice(groups[key], settings['windows_per_fit_parent_per_epoch'], replace=True)
                                  for key in sorted(groups)])
        rng.shuffle(indices)
        student.train()
        total = 0.
        for begin in range(0, len(indices), settings['batch_size']):
            chosen = indices[begin:begin+settings['batch_size']]
            # Feature normalization uses frozen CPU buffers; no trainable layer.
            dense, dropped, _ = make_batch(fit, chosen, normalizer, rng)
            result = loss_components(teacher, student, dense, dropped, device)
            loss = result['total'].mean()
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite student loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in student.parameters() if p.requires_grad], settings['gradient_norm_cap'], error_if_nonfinite=True)
            optimizer.step()
            actual_steps += 1
            total += float(loss.detach())*len(chosen)
        teacher.cpu()
        student.cpu()
        metrics = selection_context_metrics(teacher, student, selection)
        if metrics['selection_loss'] < best:
            best, selected = metrics['selection_loss'], epoch+1
            selected_state = {name: value.detach().cpu().clone() for name, value in student.state_dict().items()}
        row = dict(epoch=epoch+1, optimizer_steps=actual_steps, fit_context_loss=total/len(indices),
                   selection=metrics, selected_epoch=selected)
        history.append(row)
        atomic_status(output/'status.json', dict(stage='training', **row))
        print(json.dumps(dict(epoch=epoch+1, optimizer_steps=actual_steps, selection_loss=metrics['selection_loss'])), flush=True)
        teacher.to(device)
        student.to(device)
    student.cpu().load_state_dict(selected_state, strict=True)
    teacher.cpu()
    if tensor_digest(student.state_dict()) != frozen_body:
        raise ValueError('Frozen dynamics/normalization bytes changed')
    if any(not torch.equal(original_teacher[key], value.cpu()) for key, value in teacher.state_dict().items()):
        raise ValueError('Frozen teacher changed')
    expected_steps = settings['epochs']*math.ceil(len(groups)*settings['windows_per_fit_parent_per_epoch']/settings['batch_size'])
    if actual_steps != expected_steps:
        raise ValueError('Actual optimizer step budget differs')
    verify_identity(identity)
    write(output/'history.json', history)
    evaluation = prediction_metrics(teacher, student, selection)
    write(output/'selection_prediction_diagnostic.json', evaluation)
    dev_windows, dev_ledger, dev_files = load_development_validation()
    dev_identity = dict(files=dev_files, parent_ledger=dev_ledger, used_for_epoch_selection=False,
                        original_teacher_selection_exposed=True, fresh_TEST=False)
    write(output/'development_VAL_data.json', dev_identity)
    write(output/'development_VAL_prediction_diagnostic.json', prediction_metrics(teacher, student, dev_windows))
    if any(sha(name) != digest for name, digest in dev_files.items()):
        raise ValueError('Original development VAL files changed during evaluation')
    metadata = dict(version=VERSION, selected_epoch=selected, completed_epochs=settings['epochs'],
        actual_optimizer_steps=actual_steps, expected_optimizer_steps=expected_steps,
        teacher_sha256=settings['teacher_sha256'], teacher_frozen=True, trainable_prefixes=list(PREFIXES),
        frozen_body_digest=frozen_body, state_schema='r10_se3_world_twist_rot6_mode_observability_v2',
        training_complete=True, TEST_used_for_fitting=False, selection_split='predeclared_TRAIN_internal_parent_selection',
        target_tasks=list(registry.get('tasks', ())), other14_tasks_compatibility_claimed=False,
        P4_or_live_runtime_admitted=False, status='development_student_not_policy_admitted',
        time_basis=settings['time_basis'], physical_time_gap_robustness_qualified=False,
        metric_3cm_position_training_or_qualification=False,
        identity_sha256=sha(output/'identity.json'), selection_prediction_diagnostic_sha256=sha(output/'selection_prediction_diagnostic.json'))
    torch.save(dict(spec=spec, state_dict=selected_state, metadata=metadata), output/'selected_student.pt')
    write(output/'source_verification.json', dict(before=identity, after=frozen_identity(registry_path, identity['TRAIN_cache_sha256']), equal=True))
    atomic_status(output/'status.json', dict(stage='complete', **metadata, selected_checkpoint_sha256=sha(output/'selected_student.pt')))
    return dict(metadata, selected_checkpoint=str(output/'selected_student.pt'), selected_checkpoint_sha256=sha(output/'selected_student.pt'))


def evaluate_checkpoint(registry_path, checkpoint, expected_sha256, output):
    if sha(checkpoint) != expected_sha256:
        raise ValueError('Exact independently selected student required')
    _, windows, _, teacher, student, _, identity = prepare(registry_path)
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if payload['metadata'].get('version') != VERSION or payload['metadata'].get('P4_or_live_runtime_admitted') is not False:
        raise ValueError('Exact unadmitted development student version required')
    student.load_state_dict(payload['state_dict'], strict=True)
    if tensor_digest(student.state_dict()) != tensor_digest(teacher.state_dict()):
        raise ValueError('Frozen body/normalization differ from original teacher')
    report = prediction_metrics(teacher, student, [w for w in windows if w.role == 'selection'])
    dev_windows, dev_ledger, dev_files = load_development_validation()
    report['old_development_validation'] = prediction_metrics(teacher, student, dev_windows)
    report['old_development_validation_identity'] = dict(files=dev_files, parent_ledger=dev_ledger,
        used_for_epoch_selection=False, original_teacher_selection_exposed=True, fresh_TEST=False)
    verify_identity(identity)
    write(output, report)
    return report
