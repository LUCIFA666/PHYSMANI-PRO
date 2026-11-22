"""Train a new Cube future-contact proposal model on the existing TRAIN split.

Only a real-time-labelled candidate module is trained. Public vision, the
GRU/context forecaster, execution dynamics and MPC remain their original
versions. Best-of-eight validation measures supervised coverage, not policy
success. This is a development run; no TEST payload is accessed.
"""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch

FEATURES = ('future_tokens', 'robot', 'geometry', 'context_c', 'task_e')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pointer(path):
    return dict(path=str(Path(path).resolve(strict=True)), sha256=sha(path))


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def verify(mapping):
    for path, digest in mapping.items():
        if sha(path) != digest:
            raise ValueError('Frozen source or input changed: ' + path)


def legacy_loader():
    path = ROOT / 'scripts/train_p4_candidate_20261103.py'
    spec = importlib.util.spec_from_file_location('frozen_p4_dataset_reader', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_inputs(plan):
    loader = legacy_loader()
    train, tr = loader.load_dataset(plan['train_dataset'], 'train')
    selection, sr = loader.load_dataset(plan['selection_dataset'], 'val')
    if set(tr['parents']) & set(sr['parents']):
        raise ValueError('Whole-episode train/selection overlap')
    if len(tr['parents']) != 14 or len(sr['parents']) != 7:
        raise ValueError('Fixed 14 training and 7 development parents required')
    if len(train['robot']) != 224 or len(selection['robot']) != 112:
        raise ValueError('The complete fixed TRAIN224/development112 cohorts are required')
    for arrays in (train, selection):
        if not np.all(arrays['task_e'] == np.float32(1. / 8.)):
            raise ValueError('Single-task pilot must preserve explicit fixed task embedding')
    return train, tr, selection, sr


def quantiles(values):
    a = np.asarray(values, dtype=np.float64)
    return dict(mean=float(a.mean()), p50=float(np.percentile(a, 50)),
                p95=float(np.percentile(a, 95)), maximum=float(a.max()))


def tensors(arrays, device):
    return {k: torch.from_numpy(v).to(device) for k, v in arrays.items()}


def features(arrays, indices):
    return {k: arrays[k][indices] for k in FEATURES}


def loss_for(output, arrays, indices):
    from our_mani.candidate_transformer_cube_20261104_v1.losses import matching_loss
    return matching_loss(output, arrays['target_position'][indices],
                         arrays['target_rotation'][indices], arrays['target_dt'][indices])


@torch.no_grad()
def evaluate(model, arrays, parents, batch_size=32):
    model.eval()
    parts = {k: [] for k in ('position', 'rotation', 'time', 'loss', 'slot')}
    for begin in range(0, len(arrays['robot']), batch_size):
        ids = slice(begin, begin + batch_size)
        result = loss_for(model(**features(arrays, ids)), arrays, ids)
        for key, src in (('position', 'matched_position_error_m'),
                         ('rotation', 'matched_rotation_error_rad'),
                         ('time', 'matched_time_error_seconds'),
                         ('loss', 'per_example'), ('slot', 'matching_index')):
            parts[key].append(result[src].detach().cpu().numpy())
    parts = {k: np.concatenate(v) for k, v in parts.items()}
    weights = arrays['sample_weight'].detach().cpu().numpy()
    indices = arrays['parent_index'].detach().cpu().numpy()
    return dict(parent_balanced_loss=float(np.sum(parts['loss'] * weights) / weights.sum()),
        oracle_matched_relative_position_error_m=quantiles(parts['position']),
        oracle_matched_parallel_gripper_rotation_error_rad=quantiles(parts['rotation']),
        oracle_matched_time_error_seconds=quantiles(parts['time']),
        supervised_slot_counts=np.bincount(parts['slot'], minlength=8).tolist(),
        per_parent={name: dict(samples=int(np.sum(indices == i)),
            relative_position_error_m=quantiles(parts['position'][indices == i]),
            time_error_seconds=quantiles(parts['time'][indices == i])) for i, name in enumerate(parents)},
        samples=len(indices), independent_parents=len(parents),
        scope='joint pose/time oracle-matched proposal coverage; not executed selection accuracy or task success',
        task_classifier_trained=False, success_probability_calibrated=False,
        formal_VAL_TEST=False, accuracy_hard_gate_used=False)


def stopped(output):
    for path in (output / 'STOP', output.parent / 'STOP', ROOT / 'runtime/framework/STOP',
                 ROOT / 'runtime/framework/local_mpc_20261102/STOP'):
        if path.exists():
            raise InterruptedError('Administrative STOP: ' + str(path))


def train(plan_path, expected_sha):
    if sha(plan_path) != expected_sha:
        raise ValueError('External frozen training plan SHA required')
    plan = read(plan_path)
    verify(plan['source_sha256']); verify(plan['input_sha256'])
    if plan.get('schema') != 'cube_future_candidate_training_plan_20261104_v1':
        raise ValueError('Explicit new training family required')
    output = Path(plan['output'])
    stopped(output)
    output.mkdir(parents=True, exist_ok=False)
    write(output / 'request.json', dict(plan=pointer(plan_path), configuration=plan,
        Native_started=False, TEST_read=False,
        component_trained='future-contact candidate Transformer only'))
    start = time.perf_counter()
    steps = 0
    try:
        from our_mani.current_cube_rgbd_centre_large_20261104_v2.training import exclusive_resources
        from our_mani.candidate_transformer_cube_20261104_v1.model import CandidateConfig, CubeFutureCandidateTransformer
        from our_mani.local_mpc_p4_20261103.model import CandidateTransformerPilot
        with exclusive_resources() as resource:
            if not torch.cuda.is_available():
                raise RuntimeError('Actual CUDA is required; no CPU fallback')
            torch.set_num_threads(2)
            os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
            torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.benchmark = False
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_math_sdp(True)
            random.seed(plan['seed']); np.random.seed(plan['seed']); torch.manual_seed(plan['seed'])
            torch.cuda.manual_seed_all(plan['seed'])
            torch.cuda.reset_peak_memory_stats()
            train_np, tr, selection_np, sr = load_inputs(plan)
            train_data, selection_data = tensors(train_np, 'cuda:0'), tensors(selection_np, 'cuda:0')
            model = CubeFutureCandidateTransformer(CandidateConfig.from_dict(plan['model_config'])).to('cuda:0')
            model_spec = model.spec()
            baseline_payload = torch.load(plan['baseline_checkpoint'], map_location='cpu', weights_only=True)
            baseline = CandidateTransformerPilot().to('cuda:0')
            baseline.load_state_dict(baseline_payload['model_state'], strict=True)
            if baseline_payload['model_spec'] != baseline.spec():
                raise ValueError('Exact original P4 baseline specification required')
            baseline_metrics = evaluate(baseline, selection_data, sr['parents'])
            write(output / 'baseline_selection_report.json', baseline_metrics)
            del baseline, baseline_payload
            initial = evaluate(model, selection_data, sr['parents'])
            write(output / 'initial_selection_report.json', initial)
            # The untrained initialization is diagnostic only. Select among
            # the fixed trained epochs 5, 10, ..., 200.
            best_loss, best_epoch, best_state = float('inf'), None, None
            optimizer = torch.optim.AdamW(model.parameters(), lr=plan['learning_rate'],
                                          weight_decay=plan['weight_decay'])
            rng = np.random.default_rng(plan['seed'])
            count = len(train_np['robot'])
            n_batches = math.ceil(count / plan['batch_size'])
            final_state = None
            write(output / 'data_scope.json', dict(
                train_dataset=pointer(Path(plan['train_dataset']) / 'dataset.npz'),
                selection_dataset=pointer(Path(plan['selection_dataset']) / 'dataset.npz'),
                train_parents=tr['parents'], selection_parents=sr['parents'],
                total_training_teacher_cases=len(tr['cases']), supervised_training_parents=14,
                failed_training_cases_retained=[c for c in tr['cases'] if c['accepted_contact_labels'] == 0],
                total_selection_teacher_cases=len(sr['cases']), supervised_selection_parents=7,
                failed_selection_cases_retained=[c for c in sr['cases'] if c['accepted_contact_labels'] == 0],
                ordinary_task_custom_motion_training=True, official_high_speed_training=False,
                contact_labels='first actual 50ms attachment followed by same-segment goal plus lifted hold',
                task_e='fixed uniform one-task embedding; no task classifier training',
                sources_and_inputs_unchanged=True))
            with (output / 'history.jsonl').open('x') as log:
                for epoch in range(1, plan['epochs'] + 1):
                    stopped(output)
                    model.train()
                    order = rng.permutation(count)
                    epoch_sum = 0.
                    for begin in range(0, count, plan['batch_size']):
                        stopped(output)
                        ids = torch.as_tensor(order[begin:begin + plan['batch_size']], device='cuda:0')
                        optimizer.zero_grad(set_to_none=True)
                        output_batch = model(**features(train_data, ids))
                        values = loss_for(output_batch, train_data, ids)['per_example']
                        # Existing windows have equal counts per parent. Retain
                        # explicit per-parent sample weights at the loss boundary.
                        weights = train_data['sample_weight'][ids]
                        loss = (values * weights).sum() / weights.sum()
                        if not bool(torch.isfinite(loss)):
                            raise ValueError('Nonfinite supervised candidate loss')
                        loss.backward()
                        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                        optimizer.step(); steps += 1
                        epoch_sum += float(loss.detach())
                    entry = dict(epoch=epoch, optimizer_steps=steps,
                        train_mean_batch_loss=epoch_sum / n_batches,
                        last_gradient_norm=float(norm), elapsed_seconds=time.perf_counter() - start)
                    if epoch % plan['selection_interval'] == 0 or epoch == plan['epochs']:
                        metrics = evaluate(model, selection_data, sr['parents'])
                        entry['selection_loss'] = metrics['parent_balanced_loss']
                        if metrics['parent_balanced_loss'] < best_loss:
                            best_loss, best_epoch = metrics['parent_balanced_loss'], epoch
                            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                        entry['selected_epoch'] = best_epoch
                        entry['best_selection_loss'] = best_loss
                        print(json.dumps(entry), flush=True)
                    log.write(json.dumps(entry, allow_nan=False) + '\n'); log.flush()
            if best_state is None or best_epoch is None:
                raise RuntimeError('No trained selection checkpoint was evaluated')
            final_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            common = dict(schema='Cube_candidate_transformer_20261104_v1', model_spec=model_spec,
                model_config=plan['model_config'], source_sha256=plan['source_sha256'],
                train_parents=tr['parents'], train_dataset_sha256=tr['dataset_sha256'],
                selection_parents=sr['parents'], selection_dataset_sha256=sr['dataset_sha256'],
                training_updates=steps, epochs_completed=plan['epochs'], seed=plan['seed'],
                task_classifier_trained=False, formal_VAL_TEST=False,
                selection='minimum_parent_balanced_loss_on_existing_development_selection',
                pretrained_weights_loaded=False, physical_handoff_time_supervised=True,
                success_probability_calibrated=False)
            torch.save(dict(common, model_state=final_state, selected_epoch=plan['epochs']), output / 'final.pt')
            torch.save(dict(common, model_state=best_state, selected_epoch=best_epoch), output / 'selected.pt')
            model.load_state_dict(best_state, strict=True)
            train_metrics = evaluate(model, train_data, tr['parents'])
            selected_metrics = evaluate(model, selection_data, sr['parents'])
            write(output / 'train_report.json', train_metrics)
            write(output / 'selection_report.json', selected_metrics)
            verify(plan['source_sha256']); verify(plan['input_sha256'])
            status = dict(complete=True, training_complete=True, epochs_completed=plan['epochs'],
                actual_optimizer_steps=steps, selected_epoch=best_epoch,
                selected_checkpoint=pointer(output / 'selected.pt'), final_checkpoint=pointer(output / 'final.pt'),
                model_spec=model_spec, parameters=sum(p.numel() for p in model.parameters()),
                training_dataset_parents=14, selection_dataset_parents=7,
                train_report=pointer(output / 'train_report.json'), selection_report=pointer(output / 'selection_report.json'),
                baseline_selection_report=pointer(output / 'baseline_selection_report.json'),
                wall_seconds=time.perf_counter() - start, device=str(next(model.parameters()).device), precision='float32',
                peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                resources=resource, source_unchanged=True, input_unchanged=True,
                Native_started=False, full_task_success_evaluated=False, TEST_read=False)
            write(output / 'fit_status.json', status)
            print(json.dumps({k: status[k] for k in ('complete','epochs_completed','actual_optimizer_steps','selected_epoch','parameters','wall_seconds')}), flush=True)
            return status
    except BaseException as error:
        write(output / 'failure.json', dict(complete=False, actual_optimizer_steps=steps,
            error_type=type(error).__name__, error=str(error), wall_seconds=time.perf_counter() - start,
            Native_started=False, automatic_retry=False))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if not args.execute:
        if sha(args.plan) != args.plan_sha256:
            raise ValueError('Frozen plan SHA required')
        plan = read(args.plan); verify(plan['source_sha256']); verify(plan['input_sha256'])
        data = load_inputs(plan)
        print(json.dumps(dict(complete=True, preparation_only=True, train_windows=len(data[0]['robot']),
            selection_windows=len(data[2]['robot']), GPU=False, model_forwards=0, Native=False)))
    else:
        train(args.plan, args.plan_sha256)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
