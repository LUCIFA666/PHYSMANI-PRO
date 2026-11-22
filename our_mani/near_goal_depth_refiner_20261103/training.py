"""Explicit fixed TRAIN fit. Import/CPU check never constructs an optimizer."""
from pathlib import Path
import json
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import (DepthInfluenceDataset, FEATURE_NAMES, CLASS_NAMES, LABEL_PROTOCOL,
                   PROTOCOL_FILES, sha256)
from .model import TinyDepthInfluenceUNet, binary_segmentation_loss, predict_probabilities
from .evaluation import (oracle_diagnostic, evaluate_probabilities, choose_calibrated_threshold,
                         THRESHOLD_GRID, QUALITY_PROTOCOL)

VERSION = 'near_goal_binary_current_depth_influence_20261103_v1'
SEED, EPOCHS, LR, WEIGHT_DECAY = 261103700, 40, 3e-4, 1e-4


def write_new(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def fixed_config():
    return dict(seed=SEED, epochs=EPOCHS, optimizer='AdamW', lr=LR, weight_decay=WEIGHT_DECAY,
        batch_shots=16, loss='knownBinaryCE+CubeDice', unet_base_channels=8,
        feature_names=FEATURE_NAMES, class_names=CLASS_NAMES,
        parent_weighting='eachparent8shots_equal_fourviews_equal', augmentation='none', pretrained_weights=False,
        label_protocol=LABEL_PROTOCOL, quality_protocol=QUALITY_PROTOCOL, threshold_grid=THRESHOLD_GRID,
        geometry_erosions=1, near_scope_m=.08, minimum_conservative_PPV=.995,
        minimum_oracle_eligible_coverage=.90, maximum_centre_p95_m=.005,
        unknown_owner_frame_falseavailable=0, completion=False)


def source_identity(dataset):
    package = Path(__file__).parent
    repository = package.parents[1]
    files = sorted(package.glob('*.py')) + sorted(package.glob('*.json'))
    files += [repository/'scripts/train_near_goal_depth_refiner_20261103.py',
              repository/'our_mani/near_goal_segmentation_20261103/data.py']
    code = {str(path.resolve()): sha256(path) for path in files}
    return json.loads(json.dumps(dict(manifest_path=str(dataset.path), manifest_sha256=sha256(dataset.path),
        partition_sha256=dataset.partition_sha256, code_sha256=code,
        bound_data_sha256=dataset.bound_pointers, capture_source_identity=dataset.capture_source_identity,
        fixed_protocols=PROTOCOL_FILES, config=fixed_config())))


def _collate(samples):
    return dict(features=torch.as_tensor(np.stack([s['features'] for s in samples]), dtype=torch.float32),
                labels=torch.as_tensor(np.stack([s['semantic'] for s in samples]), dtype=torch.long))


def _loss_batch(model, batch, device):
    x, y = batch['features'].to(device), batch['labels'].to(device)
    count = len(x)
    logits = model(x.reshape(count*4, 9, 64, 64))
    return binary_segmentation_loss(logits, y.reshape(count*4, 64, 64)).reshape(count, 4).mean(1).mean()


def train_capture(manifest, output, *, device='cpu', workers=0, allow_pilot_smoke=False):
    if type(allow_pilot_smoke) is not bool or type(workers) is not int or not 0 <= workers <= 8:
        raise ValueError('Explicit bool pilot scope and bounded loaderworkers required')
    complete = DepthInfluenceDataset(manifest)
    if complete.scope != 'full32' and not allow_pilot_smoke:
        raise ValueError('Pilot requires explicit smoke opt-in and can never runtimequalify')
    out = Path(output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    identity = source_identity(complete)
    write_new(out/'identity.json', identity)
    repository = Path(__file__).resolve().parents[2]
    for name, digest in identity['code_sha256'].items():
        source = Path(name)
        target = out/'source_snapshot'/source.relative_to(repository)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as stream:
            stream.write(source.read_bytes())
        if sha256(target) != digest:
            raise ValueError('Source archive byte mismatch')
    oracle = oracle_diagnostic(complete)
    write_new(out/'oracle_before_fit.json', oracle)
    if not oracle['ready_for_segmentation_fit']:
        write_new(out/'status.json', dict(version=VERSION, complete=False, training_started=False,
            training_complete=False, model_qualified=False, reason='rawMask_eroded_oracle_geometry_gate_failed'))
        raise ValueError('Oracle5mm/coverage support insufficient; no model or optimizer constructed')
    # Everything above is data/readiness; only this explicit train call can optimize.
    torch.set_num_threads(2)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if str(device).startswith('cuda'):
        torch.cuda.manual_seed_all(SEED)
    fit, selection, calib = (DepthInfluenceDataset(manifest, r) for r in ('fit', 'selection', 'calib'))
    loader = DataLoader(fit, batch_size=16, shuffle=True, num_workers=workers,
        generator=torch.Generator().manual_seed(SEED), collate_fn=_collate)
    select_loader = DataLoader(selection, batch_size=16, shuffle=False, num_workers=workers, collate_fn=_collate)
    model = TinyDepthInfluenceUNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    history, best_state, best_epoch, best_loss = [], None, None, float('inf')
    started = time.perf_counter()
    for epoch in range(1, EPOCHS+1):
        model.train()
        fit_sum, fit_count = 0., 0
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = _loss_batch(model, batch, device)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite TRAINloss; no retry')
            loss.backward()
            optimizer.step()
            count = len(batch['features'])
            fit_sum += float(loss.detach())*count
            fit_count += count
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
        value = select_sum/select_count
        if value < best_loss:
            best_loss, best_epoch = value, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        history.append(dict(epoch=epoch, fit_loss=fit_sum/fit_count, selection_loss=value))
        print(json.dumps(history[-1]), flush=True)
    model.load_state_dict(best_state, strict=True)
    model.eval()
    # Calibration labels never choose an epoch or update parameters.
    samples = [calib[i] for i in range(len(calib))]
    probabilities = [predict_probabilities(model, s['features']) for s in samples]
    reports = [evaluate_probabilities(samples, probabilities, t) for t in THRESHOLD_GRID]
    calibration = choose_calibrated_threshold(reports)
    write_new(out/'calibration.json', dict(selection=calibration, reports=reports))
    qualified = complete.scope == 'full32' and calibration['calibration_passed']
    metadata = dict(identity=identity, dataset_split='TRAIN', TEST_read=False,
        training_complete=True, epochs_completed=EPOCHS, selected_epoch=best_epoch,
        scope=complete.scope, pilot_is_smoke_only=complete.scope != 'full32', model_qualified=qualified,
        label_protocol=LABEL_PROTOCOL, quality_protocol=QUALITY_PROTOCOL,
        label_semantics='current_measured_depth_influence_not_allRGBinstance_or_amodal',
        selected_threshold=calibration['selected_threshold'], feature_names=list(FEATURE_NAMES),
        class_names=list(CLASS_NAMES), model_config=dict(input_channels=9, classes=2, base_channels=8),
        completion=False, current_depth_influence_only=True, geometry_erosions=1,
        complete_visible_instance_claimed=False, pose_regression=False, orientation_observed=False,
        absolute_worldXYZ_input=False, TCP_GRU_GT_input=False, raw_RGBD_modified=False,
        Native_control_qualified=False, calibration_evidence=dict(path='calibration.json', sha256=sha256(out/'calibration.json')),
        oracle_evidence=dict(path='oracle_before_fit.json', sha256=sha256(out/'oracle_before_fit.json')))
    complete.revalidate()
    for name, digest in identity['code_sha256'].items():
        if sha256(name) != digest:
            raise ValueError('Source changed during fit; no complete checkpoint published')
    checkpoint = out/'best.pt'
    torch.save(dict(version=VERSION, model_state_dict=best_state, metadata=metadata), checkpoint)
    write_new(out/'history.json', history)
    write_new(out/'status.json', dict(version=VERSION, complete=True, training_started=True,
        training_complete=True, completed_epochs=EPOCHS, selected_epoch=best_epoch, identity=identity,
        best_checkpoint_sha256=sha256(checkpoint), model_qualified=qualified,
        scope=complete.scope, TEST_read=False, Native_control_qualified=False,
        wall_seconds=time.perf_counter()-started))
    return checkpoint
