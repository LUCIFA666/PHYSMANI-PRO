"""Fresh v5 fit/selection only; never recompute consumed calibration scores."""
from pathlib import Path
import json
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..near_goal_depth_refiner_20261103.data import FEATURE_NAMES, CLASS_NAMES, LABEL_PROTOCOL, PARTITION_IDS, sha256
from ..near_goal_depth_refiner_20261103.model import TinyDepthInfluenceUNet
from ..near_goal_depth_refiner_20261103.training import _collate
from ..near_goal_depth_refiner_20261103_v4.training import (
    load_full32_dataset, implementation_source_map as v4_source_map,
    checked_geometry_sources, EXPECTED_GEOMETRY_VERSION, FULL_MANIFEST_SHA256,
)
from ..near_goal_depth_refiner_20261103_v4.cache import CachedRoleDataset, canonical_sha
from ..near_goal_depth_refiner_20261103_v4.calibration import check_library_environment
from .losses import loss_batch, PROTOCOL_SHA256, UNKNOWN_WEIGHT

ROOT = Path(__file__).resolve().parents[2]
VERSION = 'near_goal_binary_UNKNOWN_uniformKL_schedule160_selected_20261103_v5'
SEED, EPOCHS, BATCH_SHOTS, OPTIMIZER_STEPS = 261103700, 160, 16, 1920
LR, WEIGHT_DECAY = 3e-4, 1e-4
PROTOCOL = ROOT/'artifacts/framework/near_goal_segmentation_20261103/binary_depth_refiner_unknown_abstention_v5_protocol_v1/protocol.json'
V4_RECEIPT = ROOT/'artifacts/framework/near_goal_segmentation_20261103/binary_depth_refiner_geometryF_schedule160_code_v4/receipt.json'
V4_RECEIPT_SHA256 = '791285a38c5bf2d180cfa56d550f72d5aabf4c8b557374be227da88dc2032acd'
PRIOR_V4 = ROOT/'artifacts/framework/near_goal_segmentation_20261103/full32_geometryF_fit_v4'
PRIOR_ORACLE_SHA256 = '9a0cc2aac207986377d01c12130903867569a0be99b56ff522276caeb53d3c26'
PRIOR_STATUS_SHA256 = 'b866bd423b2f0da818876447fff99b5597ded09e177ba59f332dbac931165212'
PRIOR_CHECKPOINT_SHA256 = '4a47971eea2be2add4039d9bb058ed7d0966c61f4c15ba312b973ddfc9912bcd'
TRAINING_FILES = ('__init__.py', 'losses.py', 'training.py', 'training_contract.json')


class FitSelectionAccessView:
    """Log actual sample extraction and reject old calibration before decode.

    The frozen loader has already validated all32 initial public RGBDs,
    including4 consumed-calibration initial frames, for provenance only.
    This view governs shot sample materialization, not those initial checks.
    """
    def __init__(self, dataset):
        self.dataset = dataset
        self.access = []

    def __getattr__(self, name):
        return getattr(self.dataset, name)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        row = self.dataset.records[index]
        if row['role'] not in ('fit', 'selection'):
            raise ValueError('Consumed calibration shots cannot be materialized by v5 fit')
        self.access.append(dict(index=int(index), parent_id=row['parent_id'],
                                scene_id=row['scene_id'], role=row['role']))
        return self.dataset[index]


def write_new(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False); stream.write('\n')


def fixed_config():
    return dict(seed=SEED, epochs=EPOCHS, actual_optimizer_steps_required=OPTIMIZER_STEPS,
        batch_shots=BATCH_SHOTS, optimizer='AdamW', lr=LR, weight_decay=WEIGHT_DECAY,
        lr_schedule='constant', initialization='fresh_random_same_seed_no_warmstart',
        fit_parent_ids=list(PARTITION_IDS[:24]), selection_parent_ids=list(PARTITION_IDS[24:28]),
        loss='unchanged_knownCE+CubeDice_plus_realUNKNOWN_Uniform2_KL', unknown_weight=UNKNOWN_WEIGHT,
        unknown_domain='semantic255_and_public_feature8_equals1',
        reduction='equal_views_then_equal_shots_parents_equal_eight_shots',
        protocol_sha256=PROTOCOL_SHA256, geometry_version=EXPECTED_GEOMETRY_VERSION,
        model_config=dict(input_channels=9, classes=2, base_channels=8),
        feature_names=list(FEATURE_NAMES), class_names=list(CLASS_NAMES), label_protocol=LABEL_PROTOCOL,
        source_manifest_sha256=FULL_MANIFEST_SHA256, cache_fields_only=['features', 'semantic'],
        CPU_num_threads=2, calibration_performed=False, qualification_stage='pending_fresh_calibration')


def implementation_source_map():
    if sha256(PROTOCOL) != PROTOCOL_SHA256 or sha256(V4_RECEIPT) != V4_RECEIPT_SHA256:
        raise ValueError('Exact pre-fit v5 protocol and frozen v4 source receipt required')
    sources = v4_source_map()
    old = json.loads(V4_RECEIPT.read_text())['implementation_source_sha256']
    if sources != old:
        raise ValueError('Reuse the exact unchanged36 v4/model/data/geometry dependencies')
    package = Path(__file__).parent
    for name in TRAINING_FILES:
        path = package/name; sources[str(path)] = sha256(path)
    cli = ROOT/'scripts/train_near_goal_depth_refiner_20261103_v5.py'
    sources.update({str(cli): sha256(cli), str(PROTOCOL): PROTOCOL_SHA256,
                    str(V4_RECEIPT): V4_RECEIPT_SHA256})
    return sources


def validate_prior_oracle(dataset, path, expected_sha256):
    """Reuse a passed geometry gate BY SHA, without decoding old calib JSON.

    Exact completed v4/status+source guarantees the oracle preceded its1920
    steps. Only that previously fixed gate is reused; no old weights, losses,
    poses or calibration reports are loaded or selected from.
    """
    path = Path(path).resolve(strict=True)
    status_path = PRIOR_V4/'status.json'
    if (expected_sha256 != PRIOR_ORACLE_SHA256 or sha256(path) != expected_sha256
            or sha256(status_path) != PRIOR_STATUS_SHA256):
        raise ValueError('Exact pre-existing passed F geometry oracle/status SHA evidence required')
    status = json.loads(status_path.read_text())
    if (status.get('complete') is not True or status.get('training_complete') is not True
            or status.get('completed_epochs') != 160 or status.get('actual_optimizer_steps') != 1920
            or status.get('best_checkpoint_sha256') != PRIOR_CHECKPOINT_SHA256
            or status.get('identity', {}).get('manifest_sha256') != sha256(dataset.path)
            or status['identity'].get('bound_data_sha256') != dataset.bound_pointers
            or status['identity'].get('code_sha256') != v4_source_map()
            or status.get('geometry_source_sha256') != checked_geometry_sources()):
        raise ValueError('Completed same-data/source v4 must attest the reused gate')
    return dict(oracle=dict(path=str(path), sha256=expected_sha256),
        passed_gate_attested_by_completed_v4_status=dict(path=str(status_path), sha256=PRIOR_STATUS_SHA256),
        prior_v4_checkpoint_sha256=PRIOR_CHECKPOINT_SHA256,
        old_oracle_or_calibration_JSON_decoded=False, old_model_weights_loaded=False,
        old_calibration_loss_or_pose_recomputed=False)


def _verify(dataset, identity, caches=()):
    dataset.revalidate()
    if dataset.bound_pointers != identity['bound_data_sha256']:
        raise ValueError('Raw source/data binding changed')
    for name, digest in identity['code_sha256'].items():
        if sha256(name) != digest: raise ValueError('Training source changed during fit')
    for pointer in identity['prior_geometry_gate'].values():
        if isinstance(pointer, dict) and set(pointer) == {'path', 'sha256'}:
            if sha256(pointer['path']) != pointer['sha256']: raise ValueError('Prior gate evidence changed')
    for cache in caches: cache.revalidate()


def _cache_report(fit, selection):
    roles = dict(fit=fit.attestation(), selection=selection.attestation())
    size = sum(x['bytes'] for x in roles.values())
    if size != 135790592: raise ValueError('Exact original fit192/selection32 cache required')
    return dict(roles=roles, roles_content_sha256=canonical_sha(roles), total_bytes=size,
        calibration_cached=False, source_and_order_verified=True)


def train_selected(manifest, output, *, oracle_evidence, oracle_evidence_sha256,
                   device='cpu', workers=0):
    if type(workers) is not int or not 0 <= workers <= 8:
        raise ValueError('Explicit bounded CPU loader worker count required')
    sources = implementation_source_map()
    check_library_environment()
    dataset = load_full32_dataset(manifest)
    prior = validate_prior_oracle(dataset, oracle_evidence, oracle_evidence_sha256)
    out = Path(output).resolve(); out.mkdir(parents=True, exist_ok=False)
    identity = json.loads(json.dumps(dict(config=fixed_config(), code_sha256=sources,
        manifest_path=str(dataset.path), manifest_sha256=sha256(dataset.path),
        bound_data_sha256=dataset.bound_pointers, capture_source_identity=dataset.capture_source_identity,
        geometry_source_sha256=checked_geometry_sources(), prior_geometry_gate=prior)))
    write_new(out/'identity.json', identity)
    for name, digest in sources.items():
        source = Path(name); target = out/'source_snapshot'/source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as stream: stream.write(source.read_bytes())
        if sha256(target) != digest: raise ValueError('Source archive mismatch')
    access = FitSelectionAccessView(dataset)
    fit = CachedRoleDataset.from_dataset(access, 'fit')
    selection = CachedRoleDataset.from_dataset(access, 'selection')
    expected_access = [dict(index=i, parent_id=r['parent_id'], scene_id=r['scene_id'], role=r['role'])
                       for i, r in enumerate(dataset.records) if r['role'] in ('fit', 'selection')]
    if access.access != expected_access or len(access.access) != 224:
        raise ValueError('Exactly FIT192+selection32 raw shot extractions required')
    write_new(out/'sample_access.json', dict(raw_shot_access=access.access,
        consumed_calibration_32shot_getitem_calls=0,
        initial_32parent_RGBD_provenance_validated_by_frozen_loader=True,
        consumed_calibration_four_initial_RGBDs_provenance_only=True,
        no_calibration_forward_loss_or_quality_evaluation=True))
    before = _cache_report(fit, selection); write_new(out/'cache_before.json', before)
    _verify(dataset, identity, (fit, selection))
    torch.set_num_threads(2); random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    if str(device).startswith('cuda'): torch.cuda.manual_seed_all(SEED)
    loader = DataLoader(fit, batch_size=BATCH_SHOTS, shuffle=True, num_workers=workers,
        generator=torch.Generator().manual_seed(SEED), collate_fn=_collate, drop_last=False)
    select_loader = DataLoader(selection, batch_size=BATCH_SHOTS, shuffle=False,
        num_workers=workers, collate_fn=_collate, drop_last=False)
    if len(loader) != 12 or len(select_loader) != 2: raise ValueError('Fixed12/2 minibatch ledger required')
    model = TinyDepthInfluenceUNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    history, actual_steps, best_state, best_loss, best_epoch = [], 0, None, float('inf'), None
    started = time.perf_counter()
    try:
        for epoch in range(1, EPOCHS+1):
            model.train(); fit_sum, fit_count, epoch_steps = 0., 0, 0
            for batch in loader:
                optimizer.zero_grad(set_to_none=True)
                loss = loss_batch(model, batch, device)
                if not torch.isfinite(loss): raise ValueError('Nonfinite fit loss, no retry')
                loss.backward(); optimizer.step(); actual_steps += 1; epoch_steps += 1
                count = len(batch['features']); fit_sum += float(loss.detach())*count; fit_count += count
            if fit_count != 192 or epoch_steps != 12 or actual_steps != epoch*12:
                raise ValueError('Fixed192 samples/12actualsteps per epoch required')
            model.eval(); select_sum, select_count = 0., 0
            with torch.inference_mode():
                for batch in select_loader:
                    loss = loss_batch(model, batch, device)
                    if not torch.isfinite(loss): raise ValueError('Nonfinite selection loss, no retry')
                    count = len(batch['features']); select_sum += float(loss)*count; select_count += count
            if select_count != 32: raise ValueError('All32 selection samples must be retained')
            value = select_sum/select_count
            if value < best_loss:
                best_loss, best_epoch = value, epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            row = dict(epoch=epoch, fit_loss=fit_sum/fit_count, selection_loss=value,
                       actual_optimizer_steps=actual_steps)
            history.append(row)
            with (out/'progress.jsonl').open('a') as stream: stream.write(json.dumps(row)+'\n')
            print(json.dumps(row), flush=True)
        if actual_steps != OPTIMIZER_STEPS: raise ValueError('Completed160 means exactly1920actualsteps')
        _verify(dataset, identity, (fit, selection))
        after = _cache_report(fit, selection)
        if after != before: raise ValueError('Ordered original cache changed')
        write_new(out/'cache_after.json', after); write_new(out/'history.json', history)
        write_new(out/'source_verification.json', dict(complete=True,
            identity_sha256=canonical_sha(identity), raw_data_source_cache_before_after_verified=True,
            actual_optimizer_steps=actual_steps, calibration_forward_calls=0,
            consumed_calibration_model_metrics_recomputed=False))
        metadata = dict(stage='selected_pending_fresh_calibration', identity=identity,
            training_complete=True, epochs_completed=EPOCHS, actual_optimizer_steps=actual_steps,
            selected_epoch=best_epoch, selection_loss=best_loss, dataset_split='TRAIN', scope='full32',
            model_qualified=False, selected_threshold=None, calibration_state='pending_fresh4',
            calibration_forward_calls=0, old_calibration_used_for_epoch_selection=False,
            consumed_calibration_four_initial_RGBDs_provenance_only=True,
            consumed_calibration_32shot_getitem_calls=0,
            geometry_version=EXPECTED_GEOMETRY_VERSION, geometry_source_sha256=checked_geometry_sources(),
            protocol_sha256=PROTOCOL_SHA256, feature_names=list(FEATURE_NAMES),
            class_names=list(CLASS_NAMES), label_protocol=LABEL_PROTOCOL,
            model_config=dict(input_channels=9, classes=2, base_channels=8),
            CPU_calibration_device='cpu', CPU_calibration_num_threads=2,
            fresh_calibration_parent_seeds=list(range(261103632, 261103636)),
            selected_artifact_immutable=True, requires_new_qualified_artifact=True,
            TEST_read=False, Native_control_qualified=False,
            evidence={name: dict(path=name+'.json', sha256=sha256(out/(name+'.json')))
                for name in ('identity', 'sample_access', 'cache_before', 'cache_after', 'history', 'source_verification')})
        checkpoint = out/'selected.pt'
        with checkpoint.open('xb') as stream:
            torch.save(dict(version=VERSION, model_state_dict=best_state, metadata=metadata), stream)
        write_new(out/'fit_status.json', dict(version=VERSION, complete=True, training_complete=True,
            completed_epochs=EPOCHS, actual_optimizer_steps=actual_steps, identity=identity,
            selected_epoch=best_epoch, selected_checkpoint_sha256=sha256(checkpoint),
            model_qualified=False, calibration_state='pending_fresh4', calibration_forward_calls=0,
            old_calibration_new_model_scored=False, TEST_read=False, Native_control_qualified=False,
            wall_seconds=time.perf_counter()-started))
        return checkpoint
    except BaseException as exc:
        write_new(out/'failure.json', dict(version=VERSION, complete=False, training_complete=False,
            actual_optimizer_steps=actual_steps, completed_epochs=len(history),
            error_type=type(exc).__name__, error=str(exc), no_retry=True,
            model_qualified=False, calibration_state='not_performed'))
        raise
