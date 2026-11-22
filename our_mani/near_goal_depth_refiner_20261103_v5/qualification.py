"""One-use fresh4 qualification of immutable, selected v5 weights.

No optimizer, Native, or old calibration scoring. A failed request consumes the
fresh cohort too: changing methods requires independently preassigned data.
"""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import os
import time

import numpy as np
import torch

from ..near_goal_depth_refiner_20261103.data import (
    FEATURE_NAMES, CLASS_NAMES, LABEL_PROTOCOL, PARTITION_IDS, SHOT_IDS, sha256,
)
from ..near_goal_depth_refiner_20261103.model import TinyDepthInfluenceUNet, predict_probabilities
from ..near_goal_depth_refiner_20261103_v4.cache import canonical_sha
from ..near_goal_depth_refiner_20261103_v4.runtime import _cache_evidence
from ..near_goal_depth_refiner_20261103_v4.calibration import (
    assert_cpu_prediction_contract, check_library_environment,
)
from ..near_goal_depth_refiner_20261103_v4.evaluation import (
    THRESHOLD_GRID, QUALITY_PROTOCOL, BASELINE_ELIGIBILITY_PROTOCOL,
)
from ..near_goal_fresh4_calibration_20261103.data import (
    Fresh4CalibrationDataset, PARTITION_IDS as FRESH_PARENT_IDS,
)
from ..near_goal_fresh4_calibration_20261103.evaluation import (
    freeze_legacy_eligibility, evaluate_fresh_probabilities, choose_fresh_calibrated_threshold,
)
from .training import (
    VERSION as FIT_VERSION, EPOCHS, OPTIMIZER_STEPS, PROTOCOL_SHA256,
    fixed_config, implementation_source_map, checked_geometry_sources, EXPECTED_GEOMETRY_VERSION,
)

ROOT = Path(__file__).resolve().parents[2]
VERSION = 'near_goal_binary_UNKNOWN_uniformKL_fresh4_qualified_cpu2_20261103_v5'
CONSUMPTION_NAME = 'calibration_consumption.request.json'
QUALIFICATION_FILES = (
    'our_mani/near_goal_depth_refiner_20261103_v5/qualification.py',
    'our_mani/near_goal_depth_refiner_20261103_v5/runtime.py',
    'scripts/qualify_near_goal_depth_refiner_20261103_v5.py',
    'our_mani/near_goal_fresh4_calibration_20261103/__init__.py',
    'our_mani/near_goal_fresh4_calibration_20261103/data.py',
    'our_mani/near_goal_fresh4_calibration_20261103/evaluation.py',
)


def qualification_source_map():
    """Explicit fit43 plus the six qualification/loader files; no V60 glob."""
    sources = implementation_source_map()
    for name in QUALIFICATION_FILES:
        path = ROOT/name
        sources[str(path)] = sha256(path)
    return sources


def write_new(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def pointer(path):
    path = Path(path).resolve(strict=True)
    return dict(path=str(path), sha256=sha256(path))


def checked_sha(value):
    if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError('Explicit lowercase SHA256 required')
    return value


def state_digest(state):
    rows = {}
    for name, value in sorted(state.items()):
        if not isinstance(value, torch.Tensor) or not torch.isfinite(value).all():
            raise ValueError('Finite tensor-only model state required')
        a = value.detach().cpu().contiguous().numpy()
        rows[name] = dict(shape=list(a.shape), dtype=a.dtype.str,
                          sha256=hashlib.sha256(a.tobytes()).hexdigest())
    return canonical_sha(rows)


def _local_evidence(directory, pointers, names):
    if set(pointers) != set(names):
        raise ValueError('All and only declared evidence files required')
    result = {}
    for name, ref in pointers.items():
        path = (directory/ref['path']).resolve(strict=True)
        if path.parent != directory or sha256(path) != checked_sha(ref['sha256']):
            raise ValueError('Same-directory immutable evidence missing or changed')
        result[name] = json.loads(path.read_text())
    return result


def validate_selected_checkpoint(selected, expected_sha256):
    """Check actual fit160/1920, fit43, six evidence files, and selection rule."""
    selected = Path(selected).resolve(strict=True)
    if sha256(selected) != checked_sha(expected_sha256):
        raise ValueError('Immutable selected checkpoint SHA mismatch')
    payload = torch.load(selected, map_location='cpu', weights_only=True)
    meta = payload.get('metadata', {})
    if (payload.get('version') != FIT_VERSION or meta.get('stage') != 'selected_pending_fresh_calibration'
            or meta.get('training_complete') is not True
            or type(meta.get('epochs_completed')) is not int or meta['epochs_completed'] != EPOCHS
            or type(meta.get('actual_optimizer_steps')) is not int or meta['actual_optimizer_steps'] != OPTIMIZER_STEPS
            or meta.get('scope') != 'full32' or meta.get('dataset_split') != 'TRAIN'
            or meta.get('model_qualified') is not False or meta.get('selected_threshold') is not None
            or meta.get('calibration_state') != 'pending_fresh4' or meta.get('calibration_forward_calls') != 0
            or meta.get('old_calibration_used_for_epoch_selection') is not False
            or meta.get('consumed_calibration_32shot_getitem_calls') != 0
            or meta.get('selected_artifact_immutable') is not True
            or meta.get('requires_new_qualified_artifact') is not True
            or meta.get('TEST_read') is not False or meta.get('Native_control_qualified') is not False
            or meta.get('protocol_sha256') != PROTOCOL_SHA256
            or tuple(meta.get('feature_names', ())) != FEATURE_NAMES
            or tuple(meta.get('class_names', ())) != CLASS_NAMES
            or meta.get('label_protocol') != LABEL_PROTOCOL
            or meta.get('model_config') != dict(input_channels=9, classes=2, base_channels=8)
            or meta.get('geometry_version') != EXPECTED_GEOMETRY_VERSION
            or meta.get('geometry_source_sha256') != checked_geometry_sources()
            or meta.get('CPU_calibration_device') != 'cpu' or meta.get('CPU_calibration_num_threads') != 2
            or meta.get('fresh_calibration_parent_seeds') != list(range(261103632, 261103636))):
        raise ValueError('Only exact immutable selected v5 fit160/1920 pending fresh4 accepted')
    identity = meta.get('identity', {})
    if identity.get('config') != fixed_config() or identity.get('code_sha256') != implementation_source_map():
        raise ValueError('Exact current explicit fit43/config identity required')
    status = json.loads(selected.with_name('fit_status.json').read_text())
    if (status.get('version') != FIT_VERSION or status.get('complete') is not True
            or status.get('training_complete') is not True
            or type(status.get('completed_epochs')) is not int or status['completed_epochs'] != EPOCHS
            or type(status.get('actual_optimizer_steps')) is not int or status['actual_optimizer_steps'] != OPTIMIZER_STEPS
            or status.get('selected_checkpoint_sha256') != expected_sha256
            or status.get('identity') != identity or status.get('selected_epoch') != meta.get('selected_epoch')
            or status.get('model_qualified') is not False or status.get('calibration_state') != 'pending_fresh4'
            or status.get('calibration_forward_calls') != 0 or status.get('old_calibration_new_model_scored') is not False
            or status.get('TEST_read') is not False):
        raise ValueError('Actual completed selected-fit status required; never relabel a failed/partial fit')
    evidence = _local_evidence(selected.parent, meta.get('evidence', {}),
        ('identity', 'sample_access', 'cache_before', 'cache_after', 'history', 'source_verification'))
    if evidence['identity'] != identity:
        raise ValueError('Saved selected identity differs')
    _cache_evidence(evidence['cache_before'], evidence['cache_after'], identity)
    access = evidence['sample_access']
    expected = [dict(index=i, parent_id=p, scene_id=p+'/'+s, role='fit' if i < 192 else 'selection')
                for i, (p, s) in enumerate((p, s) for p in PARTITION_IDS[:28] for s in SHOT_IDS)]
    if (access.get('raw_shot_access') != expected
            or access.get('consumed_calibration_32shot_getitem_calls') != 0
            or access.get('no_calibration_forward_loss_or_quality_evaluation') is not True
            or access.get('consumed_calibration_four_initial_RGBDs_provenance_only') is not True):
        raise ValueError('Exactly FIT192/selection32 causal sample extraction; no old calibration reuse')
    history = evidence['history']
    if not isinstance(history, list) or len(history) != EPOCHS:
        raise ValueError('All160 actual epoch records required')
    for epoch, row in enumerate(history, 1):
        if (type(row.get('epoch')) is not int or row['epoch'] != epoch
                or type(row.get('actual_optimizer_steps')) is not int or row['actual_optimizer_steps'] != epoch*12
                or not np.isfinite([row.get('fit_loss'), row.get('selection_loss')]).all()):
            raise ValueError('Actual history epoch/step/loss inconsistent')
    chosen = min(range(EPOCHS), key=lambda i: history[i]['selection_loss'])+1
    if meta.get('selected_epoch') != chosen or meta.get('selection_loss') != history[chosen-1]['selection_loss']:
        raise ValueError('Only earliest minimum selection loss chooses selected checkpoint')
    verify = evidence['source_verification']
    if (verify.get('complete') is not True or verify.get('identity_sha256') != canonical_sha(identity)
            or verify.get('raw_data_source_cache_before_after_verified') is not True
            or verify.get('actual_optimizer_steps') != OPTIMIZER_STEPS
            or verify.get('calibration_forward_calls') != 0
            or verify.get('consumed_calibration_model_metrics_recomputed') is not False):
        raise ValueError('Fit raw-source/cache before-after evidence required')
    for ref in identity.get('prior_geometry_gate', {}).values():
        if isinstance(ref, dict) and set(ref) == {'path', 'sha256'} and sha256(ref['path']) != ref['sha256']:
            raise ValueError('Prior already-passed geometry attestation changed')
    state_digest(payload['model_state_dict'])
    return payload, dict(selected=pointer(selected), fit_status=pointer(selected.with_name('fit_status.json')),
                         evidence={k: pointer(selected.parent/v['path']) for k, v in meta['evidence'].items()})


def claim_calibration_once(dataset_manifest, selected_sha256, sources, output, *, capture_root):
    """Original capture-root claim also prevents copied-manifest aliases."""
    manifest = Path(dataset_manifest).resolve(strict=True)
    capture_root = Path(capture_root).resolve(strict=True)
    path = capture_root/CONSUMPTION_NAME
    sibling = manifest.parent/CONSUMPTION_NAME
    if sibling.exists():
        raise FileExistsError('Fresh calibration already consumed at dataset sibling')
    value = dict(schema='v5_fresh4_calibration_single_consumption_request_v1',
        created_utc=datetime.now(timezone.utc).isoformat(), manifest=pointer(manifest),
        selected_checkpoint_sha256=checked_sha(selected_sha256), output=str(Path(output).resolve()),
        qualification_version=VERSION, protocol_sha256=PROTOCOL_SHA256, capture_root=str(capture_root),
        method_source_sha256=sources, no_retry=True, request_consumes_cohort_even_if_failure=True)
    write_new(path, value)  # open('x') is the inter-process exclusive claim.
    if sibling != path:
        write_new(sibling, value)  # Any failure keeps the original claim consumed.
    return pointer(path), pointer(sibling)


def check_qualification_inputs(selected, expected_sha256, fresh_manifest):
    payload, selected_evidence = validate_selected_checkpoint(selected, expected_sha256)
    dataset = Fresh4CalibrationDataset(fresh_manifest)
    sources = qualification_source_map()
    check_library_environment()
    roots = {Path(r['inputs_path']).resolve().parent.parent.parent for r in dataset.records}
    if len(roots) != 1:
        raise ValueError('One original capture-root required')
    capture_root = roots.pop()
    return dict(complete=True, qualification_started=False, Native_started=False, fit_started=False,
        selected=selected_evidence['selected'], fresh_manifest=pointer(dataset.path),
        samples=len(dataset), parents=4, method_source_sha256=sources,
        consumption_exists=any((p/CONSUMPTION_NAME).exists() for p in (dataset.path.parent, capture_root)),
        model_qualified=False, selected_epoch=payload['metadata']['selected_epoch'])


def _verify_unchanged(dataset, sources, selected_evidence):
    dataset.revalidate()
    if qualification_source_map() != sources:
        raise ValueError('Qualification method source changed')
    refs = [selected_evidence['selected'], selected_evidence['fit_status']]
    refs.extend(selected_evidence['evidence'].values())
    if any(sha256(ref['path']) != ref['sha256'] for ref in refs):
        raise ValueError('Immutable selected checkpoint or its fit evidence changed')


def qualify_selected_checkpoint(selected, expected_sha256, fresh_manifest, output):
    started = time.perf_counter()
    payload, selected_evidence = validate_selected_checkpoint(selected, expected_sha256)
    sources = qualification_source_map()
    check_library_environment()
    dataset = Fresh4CalibrationDataset(fresh_manifest)
    if len(dataset) != 32:
        raise ValueError('Exactly new four-parent/32shot cohort required')
    out = Path(output).resolve()
    if out.exists() or out == dataset.path.parent or out == Path(selected).resolve().parent:
        raise ValueError('New immutable output directory required')
    capture_roots = {Path(r['inputs_path']).resolve().parent.parent.parent for r in dataset.records}
    if len(capture_roots) != 1:
        raise ValueError('One common original fresh4 capture-root required')
    capture_root = capture_roots.pop()
    request, dataset_request = claim_calibration_once(dataset.path, expected_sha256, sources, out,
                                                     capture_root=capture_root)
    out.mkdir(parents=True, exist_ok=False)
    write_new(out/'request.json', json.loads(Path(request['path']).read_text()))
    fresh_identity = dict(manifest=pointer(dataset.path), bound_data_sha256=dict(dataset.bound_pointers),
        method_source_sha256=sources, request=request, dataset_request=dataset_request,
        capture_root=str(capture_root), selected_evidence=selected_evidence)
    write_new(out/'fresh_identity.json', fresh_identity)
    try:
        samples = [dataset[i] for i in range(len(dataset))]
        eligibility = freeze_legacy_eligibility(samples)
        write_new(out/'eligibility_before_predictions.json', eligibility)
        model = TinyDepthInfluenceUNet().float().cpu()
        model.load_state_dict(payload['model_state_dict'], strict=True)
        model.eval(); torch.set_num_threads(2)
        environment = assert_cpu_prediction_contract(model)
        probabilities = []
        for sample in samples:
            features = sample['features']
            if (not isinstance(features, np.ndarray) or features.dtype != np.float32
                    or features.shape != (4, 9, 64, 64) or not np.isfinite(features).all()):
                raise ValueError('Unchanged four-view float32 current features required')
            assert_cpu_prediction_contract(model, environment)
            probabilities.append(predict_probabilities(model, features))
        assert_cpu_prediction_contract(model, environment)
        if len(probabilities) != 32 or state_digest(model.state_dict()) != state_digest(payload['model_state_dict']):
            raise ValueError('Exactly32 forwards must preserve selected model state')
        with (out/'probabilities.npz').open('xb') as stream:
            np.savez_compressed(stream, probabilities=np.stack(probabilities),
                                scene_ids=np.array([s['scene_id'] for s in samples]))
        execution = dict(calibration_device='cpu', calibration_num_threads=2,
            model_eval=True, model_dtype='float32', forward_calls=len(probabilities), samples=len(samples),
            parent_count=4, threshold_grid=list(THRESHOLD_GRID), GPU_CPU_selection_performed=False,
            **environment)
        _verify_unchanged(dataset, sources, selected_evidence)
        write_new(out/'prediction_freeze.json', dict(complete=True, frozen_before_scores=True,
            selected=selected_evidence['selected'], fresh_manifest=pointer(dataset.path),
            probabilities=pointer(out/'probabilities.npz'), eligibility=pointer(out/'eligibility_before_predictions.json'),
            execution=execution, method_source_sha256=sources, request=request))
        with np.load(out/'probabilities.npz', allow_pickle=False) as stored:
            frozen = stored['probabilities']
            if stored['scene_ids'].tolist() != [s['scene_id'] for s in samples]:
                raise ValueError('Frozen probability order changed')
        reports = [evaluate_fresh_probabilities(samples, frozen, threshold, eligibility) for threshold in THRESHOLD_GRID]
        selection = choose_fresh_calibrated_threshold(reports, eligibility)
        qualified = selection['calibration_passed']
        _verify_unchanged(dataset, sources, selected_evidence)
        report = dict(version=VERSION, complete=True, execution=execution, reports=reports, selection=selection,
            source_before_after_verified=True, fresh_identity=fresh_identity,
            old_selected_artifact_unchanged=True, predictions_frozen_before_scores=True,
            qualification_is_internal_static_known_50mm_axisCube_only=True,
            Native_control_qualified=False, grasp_qualified=False, TEST_read=False)
        write_new(out/'qualification.json', report)
        metadata = dict(payload['metadata'])
        metadata.update(stage='fresh4_calibrated', model_qualified=qualified,
            selected_threshold=selection['selected_threshold'], calibration_state='passed' if qualified else 'failed_raw995',
            completion=False, current_depth_influence_only=True, complete_visible_instance_claimed=False,
            geometry_erosions=1, TCP_GRU_GT_input=False, absolute_worldXYZ_input=False,
            quality_protocol=QUALITY_PROTOCOL, baseline_eligibility_protocol=BASELINE_ELIGIBILITY_PROTOCOL,
            calibration_parent_ids=list(FRESH_PARENT_IDS), calibration_scope='fresh4_calibration',
            calibration_device='cpu', calibration_num_threads=2,
            calibration_environment=execution, calibration_forward_calls=32,
            fresh_identity=fresh_identity, selected_weights_sha256=state_digest(payload['model_state_dict']),
            selected_checkpoint_sha256=expected_sha256, qualification_version=VERSION,
            grasp_qualified=False, model_scope='known_opaque_50mm_axisalignedCube_current_static_RGBD',
            qualification_evidence={name:dict(path=name+'.json',sha256=sha256(out/(name+'.json')))
                for name in ('fresh_identity', 'request', 'eligibility_before_predictions', 'prediction_freeze', 'qualification')})
        with (out/'calibrated.pt').open('xb') as stream:
            torch.save(dict(version=VERSION, model_state_dict=payload['model_state_dict'], metadata=metadata), stream)
        write_new(out/'status.json', dict(version=VERSION, complete=True, qualification_complete=True,
            model_qualified=qualified, calibrated_checkpoint_sha256=sha256(out/'calibrated.pt'),
            selected_checkpoint_sha256=expected_sha256, selected_weights_sha256=metadata['selected_weights_sha256'],
            method_source_sha256=sources, fresh_identity=fresh_identity, execution=execution,
            calibration_forward_calls=32, Native_control_qualified=False, grasp_qualified=False, TEST_read=False,
            wall_seconds=time.perf_counter()-started))
        return report
    except BaseException as exc:
        write_new(out/'failure.json', dict(version=VERSION, complete=False, model_qualified=False,
            error_type=type(exc).__name__, error=str(exc), request=request,
            cohort_consumed=True, no_retry=True, Native_control_qualified=False))
        raise
