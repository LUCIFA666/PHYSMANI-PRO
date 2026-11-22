"""Independent full32 schedule160. No model/optimizer is created on import."""
from pathlib import Path
import json
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..near_goal_depth_refiner_20261103.data import (
    DepthInfluenceDataset, FEATURE_NAMES, CLASS_NAMES, LABEL_PROTOCOL, sha256, checked_pointer,
)
from ..near_goal_depth_refiner_20261103.model import TinyDepthInfluenceUNet, predict_probabilities
from ..near_goal_depth_refiner_20261103.training import _collate, _loss_batch
from ..near_goal_depth_refiner_20261103.evaluation import (
    oracle_diagnostic, evaluate_probabilities, choose_calibrated_threshold, THRESHOLD_GRID, QUALITY_PROTOCOL,
)
from .cache import CachedRoleDataset, canonical_sha
from .calibration import (calibrate_cpu_selected_model, CALIBRATION_DEVICE, CALIBRATION_NUM_THREADS,
                          check_library_environment)

ROOT = Path(__file__).resolve().parents[2]
VERSION = 'near_goal_binary_current_depth_influence_schedule160_cpu2_20261103_v3'
SEED, EPOCHS, BATCH_SHOTS, OPTIMIZER_STEPS = 261103700, 160, 16, 1920
LR, WEIGHT_DECAY = 3e-4, 1e-4
PROTOCOL = ROOT/'artifacts/framework/near_goal_segmentation_20261103/binary_depth_refiner_schedule160_protocol_v2/protocol.json'
PROTOCOL_SHA256 = '0b1063e8dc52fee604da35b6358b2d84f6a9656df4c1a74e399159ec97b1dfb6'
CPU_CALIBRATION_ADDENDUM = ROOT/'artifacts/framework/near_goal_segmentation_20261103/binary_depth_refiner_cpu_calibration_addendum_v3/protocol.json'
CPU_CALIBRATION_ADDENDUM_SHA256 = 'fe6e427276e077465f879ee894afb7c91b6fa63e672d46aa7418854e73fff2ac'
V1_RECEIPT = ROOT/'artifacts/framework/near_goal_segmentation_20261103/binary_depth_refiner_code_v1/receipt.json'
V1_RECEIPT_SHA256 = '8c1bdaaee9f7e15d0a79c33ef0113d4356a6eb74aea4a2b3c14df62fcbb4129a'
FULL_BUILDER_SHA256 = '511fba972d07ce3ba6c229ceddfd55d059f7965b8de6e7db55e17bea7b577c3a'
FULL_REGISTRY_SHA256 = '57106527829fa9c23d796a0e6e5ceb265ada804fcc3dd48cdc6f460af669c317'


def write_new(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def check_frozen_definitions():
    if (sha256(PROTOCOL) != PROTOCOL_SHA256 or sha256(V1_RECEIPT) != V1_RECEIPT_SHA256
            or sha256(CPU_CALIBRATION_ADDENDUM) != CPU_CALIBRATION_ADDENDUM_SHA256):
        raise ValueError('Exact pre-full32 schedule protocol and frozenv1 definitions required')
    receipt = json.loads(V1_RECEIPT.read_text())
    sources = {str(ROOT/name): digest for name, digest in receipt['production_source_sha256'].items()}
    sources.update({str(PROTOCOL): PROTOCOL_SHA256, str(V1_RECEIPT): V1_RECEIPT_SHA256})
    sources[str(CPU_CALIBRATION_ADDENDUM)] = CPU_CALIBRATION_ADDENDUM_SHA256
    for name, digest in sources.items():
        if sha256(name) != digest:
            raise ValueError('Frozen v1 network/loss/crop/quality/source changed: '+name)
    return sources


def fixed_config():
    return dict(seed=SEED, epochs=EPOCHS, batch_shots=BATCH_SHOTS, actual_optimizer_steps_required=OPTIMIZER_STEPS,
        optimizer='AdamW', lr=LR, weight_decay=WEIGHT_DECAY, lr_schedule='constant',
        initialization='fresh_random_no_pilot_warmstart', augmentation='none',
        loss='unchanged_knownBinaryCE+CubeDice', network='frozen_TinyDepthInfluenceUNet_base8_9in2logits',
        feature_names=list(FEATURE_NAMES), class_names=list(CLASS_NAMES),
        label_protocol=LABEL_PROTOCOL, quality_protocol=QUALITY_PROTOCOL, threshold_grid=list(THRESHOLD_GRID),
        cache_roles=['fit', 'selection'], cache_fields_only=['features_float32', 'semantic_uint8'],
        calibration_in_training_cache=False, equal_parent_shot_view_weighting=True,
        schedule_protocol_sha256=PROTOCOL_SHA256,
        calibration_device=CALIBRATION_DEVICE, calibration_num_threads=CALIBRATION_NUM_THREADS,
        calibration_addendum_sha256=CPU_CALIBRATION_ADDENDUM_SHA256, completion=False)


def load_full32_dataset(manifest):
    """No old pilot records or an unverified manually mixed full32 manifest."""
    check_frozen_definitions()
    check_library_environment()
    dataset = DepthInfluenceDataset(manifest)
    if dataset.scope != 'full32' or len(dataset) != 256:
        raise ValueError('Schedule160 supports exact full32 only; pilot cannot fit or qualify')
    if (dataset.capture_plan.get('schema') != 'P3_near_goal_depth_owner_full32_plan_20261103_v1'
            or dataset.capture_plan.get('scope') != 'full32'
            or dataset.capture_plan['TRAIN_q_registry']['sha256'] != FULL_REGISTRY_SHA256):
        raise ValueError('One new full32 capture plan and fixed32 q registry required')
    receipt_path = dataset.path.with_name('receipt.json')
    receipt = json.loads(receipt_path.read_text())
    if (receipt.get('version') != 'near_goal_depth_influence_new_full32_TRAIN_manifest_builder_20261103_v1'
            or receipt.get('complete') is not True or receipt.get('structural_check_complete') is not True
            or receipt.get('scope') != 'full32' or receipt.get('parent_count') != 32 or receipt.get('shot_count') != 256
            or receipt.get('old_pilot_DATA_reused') is not False
            or receipt.get('builder', {}).get('sha256') != FULL_BUILDER_SHA256
            or receipt.get('manifest', {}).get('sha256') != sha256(dataset.path)
            or Path(receipt['manifest']['path']).resolve() != dataset.path
            or receipt.get('all32_cases_bound_to_one_NEW_plan_sha256') != sha256(dataset.capture_plan_path)):
        raise ValueError('Exact complete new full32 aggregate receipt required')
    dataset.bound_pointers[str(receipt_path)] = sha256(receipt_path)
    for family in receipt.get('original_owner_evidence', []):
        for pointer in family.values():
            path = checked_pointer(ROOT, pointer)
            dataset.bound_pointers[str(path)] = pointer['sha256']
    if len(receipt.get('original_owner_evidence', [])) != 32:
        raise ValueError('All32 owner/materialization evidence ledgers required')
    dataset.revalidate()
    return dataset


def implementation_source_map():
    sources = check_frozen_definitions()
    package = Path(__file__).parent
    for path in sorted([*package.glob('*.py'), *package.glob('*.json')]):
        sources[str(path)] = sha256(path)
    cli = ROOT/'scripts/train_near_goal_depth_refiner_20261103_v2.py'
    sources[str(cli)] = sha256(cli)
    return sources


def source_identity(dataset):
    sources = implementation_source_map()
    return json.loads(json.dumps(dict(manifest_path=str(dataset.path), manifest_sha256=sha256(dataset.path),
        code_sha256=sources, bound_data_sha256=dataset.bound_pointers,
        capture_source_identity=dataset.capture_source_identity, config=fixed_config())))


def _verify_identity(dataset, identity):
    dataset.revalidate()
    if dataset.bound_pointers != identity['bound_data_sha256']:
        raise ValueError('Raw-data/source bindings changed')
    for name, digest in identity['code_sha256'].items():
        if sha256(name) != digest:
            raise ValueError('Schedule/source implementation changed during fit')


def _cache_report(fit, selection):
    fit.revalidate()
    selection.revalidate()
    roles = dict(fit=fit.attestation(), selection=selection.attestation())
    total_bytes = sum(r['bytes'] for r in roles.values())
    if total_bytes != 135790592:
        raise ValueError('Exact192+32 currentfeature/label cache footprint required')
    return dict(schema='schedule160_fit_selection_cache_attestation_v2', roles=roles,
        total_bytes=total_bytes, total_MiB=total_bytes/(1024*1024), calibration_cached=False,
        roles_content_sha256=canonical_sha(roles))


def train_capture(manifest, output, *, device='cpu', workers=0):
    if type(workers) is not int or not 0 <= workers <= 8:
        raise ValueError('Bounded explicit loaderworker count required')
    complete = load_full32_dataset(manifest)
    out = Path(output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    identity = source_identity(complete)
    write_new(out/'identity.json', identity)
    for name, digest in identity['code_sha256'].items():
        source = Path(name)
        target = out/'source_snapshot'/source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as stream:
            stream.write(source.read_bytes())
        if sha256(target) != digest:
            raise ValueError('Source snapshot byte mismatch')
    oracle = oracle_diagnostic(complete)
    write_new(out/'oracle_before_fit.json', oracle)
    if not oracle['ready_for_segmentation_fit']:
        write_new(out/'status.json', dict(version=VERSION, complete=False, training_started=False,
            training_complete=False, actual_optimizer_steps=0, model_qualified=False,
            reason='unchanged_rawMask_eroded_oracle_gate_failed'))
        raise ValueError('Original oracle gate failed; no model/optimizer/cache fit constructed')
    fit = CachedRoleDataset.from_dataset(complete, 'fit')
    selection = CachedRoleDataset.from_dataset(complete, 'selection')
    cache_before = _cache_report(fit, selection)
    write_new(out/'cache_before.json', cache_before)
    _verify_identity(complete, identity)
    torch.set_num_threads(2)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if str(device).startswith('cuda'):
        torch.cuda.manual_seed_all(SEED)
    loader = DataLoader(fit, batch_size=BATCH_SHOTS, shuffle=True, num_workers=workers,
        generator=torch.Generator().manual_seed(SEED), collate_fn=_collate, drop_last=False)
    select_loader = DataLoader(selection, batch_size=BATCH_SHOTS, shuffle=False,
        num_workers=workers, collate_fn=_collate, drop_last=False)
    if len(loader) != 12 or len(select_loader) != 2:
        raise ValueError('Exactly12fit/2selection minibatches per epoch required')
    model = TinyDepthInfluenceUNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    actual_steps, best_state, best_epoch, best_loss = 0, None, None, float('inf')
    history, started = [], time.perf_counter()
    try:
        for epoch in range(1, EPOCHS+1):
            model.train()
            fit_sum, fit_count, epoch_steps = 0., 0, 0
            for batch in loader:
                optimizer.zero_grad(set_to_none=True)
                loss = _loss_batch(model, batch, device)
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite fitloss; no retry')
                loss.backward()
                optimizer.step()
                actual_steps += 1
                epoch_steps += 1
                count = len(batch['features'])
                fit_sum += float(loss.detach())*count
                fit_count += count
            if fit_count != 192 or epoch_steps != 12 or actual_steps != epoch*12:
                raise ValueError('Missing/filtered TRAIN cache samples or incorrect actual optimizer count')
            model.eval()
            select_sum, select_count = 0., 0
            with torch.inference_mode():
                for batch in select_loader:
                    loss = _loss_batch(model, batch, device)
                    if not torch.isfinite(loss):
                        raise ValueError('Nonfinite selectionloss; no retry')
                    count = len(batch['features'])
                    select_sum += float(loss)*count
                    select_count += count
            if select_count != 32:
                raise ValueError('Selection must retain all32 fixed samples')
            value = select_sum/select_count
            if value < best_loss:
                best_loss, best_epoch = value, epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            row = dict(epoch=epoch, fit_loss=fit_sum/fit_count, selection_loss=value,
                       actual_optimizer_steps=actual_steps)
            history.append(row)
            with (out/'progress.jsonl').open('a') as stream:
                stream.write(json.dumps(row)+'\n')
            print(json.dumps(row), flush=True)
        if actual_steps != OPTIMIZER_STEPS:
            raise ValueError('Completed160epochs must mean exactly1920 actual optimizer steps')
        cache_after = _cache_report(fit, selection)
        if cache_after != cache_before:
            raise ValueError('Ordered cache content/source changed during fit')
        _verify_identity(complete, identity)
        write_new(out/'cache_after.json', cache_after)
        model.load_state_dict(best_state, strict=True)
        model.eval()
        # Calibration is loaded afresh, not included in either training cache.
        samples = [complete[i] for i, r in enumerate(complete.records) if r['role'] == 'calib']
        if len(samples) != 32 or len({s['parent_id'] for s in samples}) != 4:
            raise ValueError('Exactly4calibration parents/all32shots required')
        calibration_result = calibrate_cpu_selected_model(model, samples)
        calibration = calibration_result['selection']
        write_new(out/'calibration.json', calibration_result)
        _verify_identity(complete, identity)
        fit.revalidate()
        selection.revalidate()
        write_new(out/'source_verification.json', dict(complete=True,
            raw_data_and_sources_verified_before_after=True, original_identity_sha256=canonical_sha(identity),
            source_bindings_sha256=canonical_sha(complete.bound_pointers), cache_content_unchanged=True,
            actual_optimizer_steps=actual_steps, calibration_cached=False,
            calibration_device=CALIBRATION_DEVICE, calibration_num_threads=CALIBRATION_NUM_THREADS,
            calibration_environment=calibration_result['execution']))
        write_new(out/'history.json', history)
        qualified = calibration['calibration_passed']
        metadata = dict(identity=identity, dataset_split='TRAIN', TEST_read=False,
            training_complete=True, epochs_completed=EPOCHS, actual_optimizer_steps=actual_steps,
            selected_epoch=best_epoch, scope='full32', model_qualified=qualified,
            schedule_protocol_sha256=PROTOCOL_SHA256, label_protocol=LABEL_PROTOCOL,
            calibration_addendum_sha256=CPU_CALIBRATION_ADDENDUM_SHA256,
            calibration_device=CALIBRATION_DEVICE, calibration_num_threads=CALIBRATION_NUM_THREADS,
            calibration_environment=calibration_result['execution'],
            quality_protocol=QUALITY_PROTOCOL, current_depth_influence_only=True,
            completion=False, complete_visible_instance_claimed=False, geometry_erosions=1,
            TCP_GRU_GT_input=False, absolute_worldXYZ_input=False, Native_control_qualified=False,
            feature_names=list(FEATURE_NAMES), class_names=list(CLASS_NAMES),
            model_config=dict(input_channels=9, classes=2, base_channels=8),
            selected_threshold=calibration['selected_threshold'], calibration_cached=False,
            calibration_parent_ids=sorted({s['parent_id'] for s in samples}),
            evidence={name: dict(path=name+'.json', sha256=sha256(out/(name+'.json')))
                      for name in ('calibration', 'oracle_before_fit', 'cache_before', 'cache_after',
                                   'source_verification', 'history')})
        checkpoint = out/'best.pt'
        torch.save(dict(version=VERSION, model_state_dict=best_state, metadata=metadata), checkpoint)
        write_new(out/'status.json', dict(version=VERSION, complete=True, training_started=True,
            training_complete=True, completed_epochs=EPOCHS, actual_optimizer_steps=actual_steps,
            selected_epoch=best_epoch, identity=identity, scope='full32', model_qualified=qualified,
            calibration_device=CALIBRATION_DEVICE, calibration_num_threads=CALIBRATION_NUM_THREADS,
            calibration_addendum_sha256=CPU_CALIBRATION_ADDENDUM_SHA256,
            best_checkpoint_sha256=sha256(checkpoint), TEST_read=False, Native_control_qualified=False,
            wall_seconds=time.perf_counter()-started))
        return checkpoint
    except BaseException as exc:
        write_new(out/'failure.json', dict(version=VERSION, complete=False, training_complete=False,
            actual_optimizer_steps=actual_steps, completed_epochs=len(history), error_type=type(exc).__name__,
            error=str(exc), model_qualified=False, no_retry=True))
        raise
