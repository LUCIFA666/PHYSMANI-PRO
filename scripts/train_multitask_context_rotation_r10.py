"""Experimental same-data rotation emphasis; standard grouped VAL selection.

This separate trainer does not replace the frozen full-v6 candidate or authorize
formal model release. The original architecture, labels and evaluation stay fixed.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from our_mani.multitask import context_se3
from our_mani.multitask.context_se3 import SCHEMA, SE3ContextNetwork
from our_mani.multitask.fitting import batch_tensors, evaluate
from our_mani.multitask import rotation_training_objective as objective
from our_mani.multitask.rotation_training_objective import balanced_parent_indices, loss_per_window
from our_mani.multitask.track_dataset import read_tracks, train_normalization, examples


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def gpu_preflight():
    report_path = ROOT / 'artifacts/framework/aligned_r09/final_report.json'
    if not report_path.exists():
        raise RuntimeError('R09 formal evaluation must finish before R10 GPU fitting')
    report = json.loads(report_path.read_text())
    # A frozen development performance screen can fail while evaluation protocol passes.
    if (report.get('complete') is not True or report.get('protocol_acceptance_passed') is not True
            or report.get('total_test_runs') != 400 or report.get('expected_test_runs') != 400):
        raise RuntimeError('R09 evaluation is incomplete')
    status = json.loads((report_path.parent / 'status.json').read_text())
    if status.get('stage') != 'complete':
        raise RuntimeError('R09 orchestrator has not completed final checks')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', required=True)
    p.add_argument('--out', default='artifacts/framework/multitask_r10/context_rotation_experimental')
    p.add_argument('--device', default='cuda')
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--windows-per-parent', type=int, default=64)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--steps', type=int, default=22)
    p.add_argument('--expected-train', type=int, default=1400)
    p.add_argument('--expected-validation', type=int, default=275)
    p.add_argument('--seed', type=int, default=1018)
    p.add_argument('--learning-rate', type=float, default=3e-4)
    p.add_argument('--rotation-loss-multiplier', type=float, default=1.)
    p.add_argument('--angular-loss-multiplier', type=float, default=1.)
    p.add_argument('--rotation-sampling-multiplier', type=float, default=1.)
    args = p.parse_args()
    for name in ('rotation_loss_multiplier','angular_loss_multiplier','rotation_sampling_multiplier'):
        objective.positive(getattr(args,name),name)
    if min(args.epochs, args.windows_per_parent, args.batch_size, args.steps) < 1:
        raise ValueError('positive optimization budgets required')
    if args.steps != 22:
        raise ValueError('R10 requires the declared full 1.1s rollout horizon')
    if args.device.startswith('cuda'):
        gpu_preflight()
    out, manifest_path = (ROOT / args.out).resolve(), (ROOT / args.manifest).resolve()
    if not out.is_relative_to(ROOT) or not manifest_path.is_relative_to(ROOT):
        raise ValueError('experimental files must remain inside the desktop project')
    baseline_path=ROOT/'artifacts/framework/multitask_r10/context_se3_full_v6/report.json'
    baseline_hash=sha(baseline_path)
    baseline=json.loads(baseline_path.read_text())
    if (sha(manifest_path)!=baseline['metadata']['track_manifest_sha256']
            or sha(context_se3.__file__)!=baseline['metadata']['architecture_sha256']
            or (args.epochs,args.windows_per_parent,args.batch_size,args.steps,
                args.expected_train,args.expected_validation,args.seed,args.learning_rate)
                !=(30,64,256,22,1400,275,1018,3e-4)):
        raise ValueError('same-data comparison requires the original v6 manifest, architecture and complete optimization budget')
    out.mkdir(parents=True, exist_ok=True)
    if any((out / name).exists() for name in ('dynamics.pt', 'training_sources.json', 'status.json')):
        raise FileExistsError('use a fresh version directory; fitted candidates cannot be silently overwritten')
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    manifest, episodes = read_tracks(ROOT, manifest_path)
    original_manifest_hash = sha(manifest_path)
    normalization, normalization_coverage = train_normalization(episodes)
    model = SE3ContextNetwork(**normalization)
    train_episodes = [x for x in episodes if x.entry['split'] == 'train']
    validation_episodes = [x for x in episodes if x.entry['split'] == 'validation']
    train, train_coverage = examples(train_episodes, model, args.steps)
    validation, validation_coverage = examples(validation_episodes, model, args.steps)
    if len(train_coverage) != args.expected_train or len(validation_coverage) != args.expected_validation:
        raise ValueError(f'parent source coverage differs: TRAIN={len(train_coverage)}, VAL={len(validation_coverage)}')
    for split, samples in (('train', train), ('validation', validation)):
        tasks = {row['source_key'][1] for row in samples}
        full_tasks = {row['source_key'][1] for row in samples if row['mask'][-1]}
        if full_tasks != tasks:
            raise ValueError(f'{split} tasks lack supervised 1.1s windows: {tasks - full_tasks}')
    if set(train_coverage) != set(normalization_coverage):
        raise ValueError('all normalization TRAIN parents must contribute optimizer examples')
    rotation_flags=np.asarray([objective.moving_rotation_window(row) for row in train],dtype=bool)
    source_path = out / 'training_sources.json'
    write(source_path, dict(version='r10_fitted_TRAIN_sources_v1', entries=[e.entry for e in train_episodes],
        cache_episode_count=len(train_episodes), source_episode_count=len(train_coverage),
        body_track_manifest_sha256=sha(manifest_path), normalization_fit_split='train',
        synthetic_augmentations_are_not_original_source_replays=True))
    device = torch.device(args.device)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    started = time.perf_counter()
    best_score, best_state, best_epoch = float('inf'), None, None
    history, actual_coverage = [], Counter()
    actual_moving_rotation_examples=0
    objective_hash=sha(objective.__file__)
    architecture_hash = sha(context_se3.__file__)
    fitting_hash = sha(ROOT / 'our_mani/multitask/fitting.py')
    dataset_hash = sha(ROOT / 'our_mani/multitask/track_dataset.py')
    script_hash = sha(__file__)
    motion_labels_hash = sha(ROOT / 'our_mani/multitask/motion_labels.py')
    for epoch in range(args.epochs):
        model.train()
        indices = balanced_parent_indices(train, args.windows_per_parent, rng,
            rotation_sampling_multiplier=args.rotation_sampling_multiplier)
        total_loss, updates = 0., 0
        # Early short rollouts reduce unstable acceleration updates; every candidate
        # is selected at the complete horizon and later epochs optimize that horizon.
        steps = args.steps if epoch == args.epochs - 1 else min(args.steps, 6 + epoch * 2)
        for begin in range(0, len(indices), args.batch_size):
            ii = indices[begin:begin + args.batch_size]
            data = batch_tensors(train, ii, device)
            loss = loss_per_window(model, data, steps,
                rotation_loss_multiplier=args.rotation_loss_multiplier,
                angular_loss_multiplier=args.angular_loss_multiplier).mean()
            if not torch.isfinite(loss):
                raise ValueError('nonfinite dynamics loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            if not torch.isfinite(norm):
                raise ValueError('nonfinite dynamics gradient')
            optimizer.step()
            actual_coverage.update(train[int(i)]['source_key'] for i in ii)
            actual_moving_rotation_examples+=int(np.count_nonzero(rotation_flags[ii]))
            total_loss += float(loss.detach()) * len(ii)
            updates += 1
        selected = evaluate(model, validation, device, args.steps, args.batch_size)
        score = selected['selection_loss']
        if not np.isfinite(score):
            raise ValueError('nonfinite VAL selection score')
        if steps == args.steps and score < best_score:
            best_score, best_epoch = score, epoch + 1
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        row = dict(epoch=epoch + 1, curriculum_steps=steps, updates=updates,
            TRAIN_loss=total_loss / len(indices), VAL_selection_loss=score,
            best_epoch=best_epoch, seconds=time.perf_counter() - started,
            optimizer_parent_source_coverage=len(actual_coverage),
            optimizer_moving_rotation_examples=actual_moving_rotation_examples,training_complete=False,
            TEST_used_for_fitting=False)
        history.append(row)
        write(out / 'status.json', dict(stage='training', **row))
        print(json.dumps(row), flush=True)
    if set(actual_coverage) != set(train_coverage):
        raise ValueError('some real TRAIN parents never reached an optimizer update')
    if sha(context_se3.__file__) != architecture_hash:
        raise ValueError('architecture changed during fitting')
    if sha(manifest_path) != original_manifest_hash:
        raise ValueError('body-track manifest changed during fitting')
    for episode in episodes:
        if sha(ROOT / episode.entry['cache']) != episode.entry['cache_sha256']:
            raise ValueError('body-track cache changed during fitting')
    for path, digest in ((ROOT / 'our_mani/multitask/fitting.py', fitting_hash),
                         (ROOT / 'our_mani/multitask/track_dataset.py', dataset_hash), (Path(__file__), script_hash),
                         (ROOT / 'our_mani/multitask/motion_labels.py', motion_labels_hash),
                         (Path(objective.__file__),objective_hash),(baseline_path,baseline_hash)):
        if sha(path) != digest:
            raise ValueError('training source code changed during fitting')
    model.load_state_dict(best_state)
    model.eval()
    final_validation = evaluate(model, validation, device, args.steps, args.batch_size, interventions=True)
    metadata = dict(created_utc=datetime.now(timezone.utc).isoformat(), version='multitask_r10_context_simnorm_se3',
        state_schema=SCHEMA, normalization_fit_split='train', training_fit_split='train',
        normalization_weighting='all valid TRAIN body frames; optimizer and VAL metrics balance real parent sources',
        training_complete=True, TEST_used_for_fitting=False, selection_split='validation',
        selected_epoch=best_epoch, completed_epochs=args.epochs, maximum_training_horizon_seconds=.05 * args.steps,
        architecture_sha256=architecture_hash, fitting_sha256=fitting_hash, track_dataset_sha256=dataset_hash,
        training_script_sha256=script_hash,track_manifest_sha256=sha(manifest_path),
        training_objective_schema=objective.SCHEMA,rotation_objective_sha256=objective_hash,
        comparison_baseline_report_sha256=baseline_hash,
        rotation_loss_multiplier=args.rotation_loss_multiplier,angular_loss_multiplier=args.angular_loss_multiplier,
        rotation_sampling_multiplier=args.rotation_sampling_multiplier,
        optimizer_moving_rotation_examples=actual_moving_rotation_examples,
        same_original1400_TRAIN_and275_VAL_as_full_v6=True,
        original_per_parent_optimizer_budget_preserved=True,
        standard_original_VAL_selection_loss_unchanged=True,
        experimental_objective_not_formal_R10_release=True,
        dynamics_target_label_schema=manifest.get('dynamics_target_label_schema'),
        motion_labels_sha256=motion_labels_hash,
        encoder_sha256=model.encoder_sha256(),
        training_source_manifest=dict(path_relative_to_checkpoint=source_path.name, sha256=sha(source_path),
            cache_episode_count=len(train_episodes), source_episode_count=len(train_coverage)),
        VAL_source_episodes=len(validation_coverage), TRAIN_windows=len(train), VAL_windows=len(validation),
        parameters=sum(p.numel() for p in model.parameters()),
        optimizer_examples_per_parent_min=min(actual_coverage.values()),
        optimizer_examples_per_parent_max=max(actual_coverage.values()),
        attached_body_dynamics_learned=False, physical_context_identifiability_verified=False,
        synthetic_augmentations_are_not_original_source_replays=True, ngff_used=False,
        seed=args.seed, seconds=time.perf_counter() - started)
    temporary = out / 'dynamics.partial.pt'
    torch.save(dict(spec=model.spec, state_dict=best_state, metadata=metadata), temporary)
    temporary.replace(out / 'dynamics.pt')
    report = dict(metadata=metadata, checkpoint_sha256=sha(out / 'dynamics.pt'), history=history,
        validation=final_validation, optimizer_coverage={':'.join(key): value for key, value in sorted(actual_coverage.items())},
        selection_disclosure='Same original grouped VAL score selects checkpoint for all rotation-emphasis experiments; objective coefficients differ only during TRAIN. Development experiment, not a formal R10 model release or TEST success result.')
    write(out / 'report.json', report)
    write(out / 'status.json', dict(stage='complete', metadata=metadata, checkpoint_sha256=report['checkpoint_sha256']))
    print(json.dumps(dict(stage='complete', checkpoint=str(out / 'dynamics.pt'), metadata=metadata)), flush=True)


if __name__ == '__main__':
    main()
