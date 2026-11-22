"""Joint16 shared Context trainer with explicit independent label shards.

--check validates all1600 TRAIN and315 offline VAL parents on CPU; it does not
create an optimizer or use GPU. --train keeps the original30x64 budget and
selects the checkpoint on complete1.1s offline VAL. It never reads TEST.
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
from our_mani.multitask.fitting import batch_tensors, balanced_parent_indices, evaluate, loss_per_window
from our_mani.multitask.track_dataset import train_normalization, examples
from our_mani.multitask import joint16_tracks_v31
from our_mani.multitask.joint16_tracks_v31 import read_joint16_tracks


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
    mode=p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check',action='store_true')
    mode.add_argument('--train',action='store_true')
    p.add_argument('--manifest', required=True)
    p.add_argument('--out', default='artifacts/framework/multitask_r10/context_joint16_v73')
    p.add_argument('--device', default='cuda')
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--windows-per-parent', type=int, default=64)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--steps', type=int, default=22)
    p.add_argument('--expected-train', type=int, default=1600)
    p.add_argument('--expected-validation', type=int, default=315)
    p.add_argument('--seed', type=int, default=1018)
    p.add_argument('--learning-rate', type=float, default=3e-4)
    args = p.parse_args()
    if min(args.epochs, args.windows_per_parent, args.batch_size, args.steps) < 1:
        raise ValueError('positive optimization budgets required')
    if args.steps != 22:
        raise ValueError('R10 requires the declared full 1.1s rollout horizon')
    if (args.expected_train,args.expected_validation,args.epochs,args.windows_per_parent,args.batch_size,args.steps,args.seed,args.learning_rate)!=(1600,315,30,64,256,22,1018,3e-4):
        raise ValueError('joint16 keeps the complete declared TRAIN/VAL roster and original optimization budget')
    if args.train and args.device.startswith('cuda'):
        gpu_preflight()
    out, manifest_path = (ROOT / args.out).resolve(), (ROOT / args.manifest).resolve()
    if not out.is_relative_to(ROOT) or not manifest_path.is_relative_to(ROOT):
        raise ValueError('joint16 inputs/outputs must remain in project')
    out.mkdir(parents=True, exist_ok=True)
    if any((out / name).exists() for name in ('dynamics.pt', 'training_sources.json', 'status.json')):
        raise FileExistsError('use a fresh version directory; fitted candidates cannot be silently overwritten')
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    manifest, episodes = read_joint16_tracks(ROOT, manifest_path)
    adapter_hash=sha(joint16_tracks_v31.__file__)
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
        if len(tasks)!=16 or full_tasks != tasks:
            raise ValueError(f'{split} tasks lack supervised 1.1s windows: {tasks - full_tasks}')
    if set(train_coverage) != set(normalization_coverage):
        raise ValueError('all normalization TRAIN parents must contribute optimizer examples')
    if args.check:
        proof=dict(complete=True,mode='CPU_check_only',TRAIN_parents=len(train_coverage),offline_VAL_parents=len(validation_coverage),
            tasks=16,TRAIN_windows=len(train),offline_VAL_windows=len(validation),full_horizon_seconds=1.1,
            manifest=dict(path=str(manifest_path.relative_to(ROOT)),sha256=original_manifest_hash),
            admitted_shard_schemas=manifest['admitted_shard_schemas'],model_parameters=sum(p.numel() for p in model.parameters()),
            source_code_sha256={str(Path(__file__).relative_to(ROOT)):sha(__file__),
                str(Path(joint16_tracks_v31.__file__).relative_to(ROOT)):adapter_hash,
                'our_mani/multitask/context_se3.py':sha(context_se3.__file__),
                'our_mani/multitask/track_dataset.py':sha(ROOT/'our_mani/multitask/track_dataset.py'),
                'our_mani/multitask/fitting.py':sha(ROOT/'our_mani/multitask/fitting.py')},
            normalization_fit_split='train',optimizer_steps=0,GPU_used=False,model_training_started=False,
            executor_qualification_granted=False,formal_release=False,TEST_read=False)
        write(out/'CPU_check.json',proof)
        print(json.dumps(proof),flush=True)
        return
    source_path = out / 'training_sources.json'
    write(source_path, dict(version='r10_joint16_fitted_TRAIN_sources_v73', entries=[e.entry for e in train_episodes],
        cache_episode_count=len(train_episodes), source_episode_count=len(train_coverage),
        body_track_manifest_sha256=sha(manifest_path), normalization_fit_split='train',
        synthetic_augmentations_are_not_original_source_replays=True))
    device = torch.device(args.device)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    started = time.perf_counter()
    best_score, best_state, best_epoch = float('inf'), None, None
    history, actual_coverage = [], Counter()
    architecture_hash = sha(context_se3.__file__)
    fitting_hash = sha(ROOT / 'our_mani/multitask/fitting.py')
    dataset_hash = sha(ROOT / 'our_mani/multitask/track_dataset.py')
    script_hash = sha(__file__)
    motion_labels_hash = sha(ROOT / 'our_mani/multitask/motion_labels.py')
    for epoch in range(args.epochs):
        model.train()
        indices = balanced_parent_indices(train, args.windows_per_parent, rng)
        total_loss, updates = 0., 0
        # Early short rollouts reduce unstable acceleration updates; every candidate
        # is selected at the complete horizon and later epochs optimize that horizon.
        steps = args.steps if epoch == args.epochs - 1 else min(args.steps, 6 + epoch * 2)
        for begin in range(0, len(indices), args.batch_size):
            ii = indices[begin:begin + args.batch_size]
            data = batch_tensors(train, ii, device)
            loss = loss_per_window(model, data, steps).mean()
            if not torch.isfinite(loss):
                raise ValueError('nonfinite dynamics loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            if not torch.isfinite(norm):
                raise ValueError('nonfinite dynamics gradient')
            optimizer.step()
            actual_coverage.update(train[int(i)]['source_key'] for i in ii)
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
            optimizer_parent_source_coverage=len(actual_coverage), training_complete=False,
            TEST_used_for_fitting=False)
        history.append(row)
        write(out / 'status.json', dict(stage='training', **row))
        print(json.dumps(row), flush=True)
    if set(actual_coverage) != set(train_coverage):
        raise ValueError('some real TRAIN parents never reached an optimizer update')
    if sha(context_se3.__file__) != architecture_hash:
        raise ValueError('architecture changed during fitting')
    if sha(joint16_tracks_v31.__file__) != adapter_hash:
        raise ValueError('joint16 dataset adapter changed during fitting')
    if sha(manifest_path) != original_manifest_hash:
        raise ValueError('body-track manifest changed during fitting')
    for episode in episodes:
        if sha(ROOT / episode.entry['cache']) != episode.entry['cache_sha256']:
            raise ValueError('body-track cache changed during fitting')
    for path, digest in ((ROOT / 'our_mani/multitask/fitting.py', fitting_hash),
                         (ROOT / 'our_mani/multitask/track_dataset.py', dataset_hash), (Path(__file__), script_hash),
                         (ROOT / 'our_mani/multitask/motion_labels.py', motion_labels_hash)):
        if sha(path) != digest:
            raise ValueError('training source code changed during fitting')
    model.load_state_dict(best_state)
    model.eval()
    final_validation = evaluate(model, validation, device, args.steps, args.batch_size, interventions=True)
    metadata = dict(created_utc=datetime.now(timezone.utc).isoformat(), version='multitask_r10_joint16_context_simnorm_se3_v73',
        state_schema=SCHEMA, normalization_fit_split='train', training_fit_split='train',
        normalization_weighting='all valid TRAIN body frames; optimizer and VAL metrics balance real parent sources',
        training_complete=True, TEST_used_for_fitting=False, selection_split='validation',
        selected_epoch=best_epoch, completed_epochs=args.epochs, maximum_training_horizon_seconds=.05 * args.steps,
        architecture_sha256=architecture_hash, fitting_sha256=fitting_hash, track_dataset_sha256=dataset_hash,
        training_script_sha256=script_hash, track_manifest_sha256=sha(manifest_path),
        dynamics_target_label_schema='explicit_independent_shards_no_single_schema_claim',
        joint16_label_shards=manifest['admitted_shard_schemas'],joint16_adapter_sha256=adapter_hash,
        jointly_trained_tasks=sorted({e.entry['task'] for e in train_episodes}),
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
        selection_disclosure='VAL selects checkpoint and is development evidence; success is evaluated separately on frozen official TEST')
    write(out / 'report.json', report)
    write(out / 'status.json', dict(stage='complete', metadata=metadata, checkpoint_sha256=report['checkpoint_sha256']))
    print(json.dumps(dict(stage='complete', checkpoint=str(out / 'dynamics.pt'), metadata=metadata)), flush=True)


if __name__ == '__main__':
    main()
