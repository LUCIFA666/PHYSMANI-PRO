"""One fixed .995 profile, one fresh4 verification; immutable V5 fit proof."""
from datetime import datetime, timezone
from pathlib import Path
import json
import time
import numpy as np
import torch
from ..near_goal_depth_refiner_20261103.data import sha256
from ..near_goal_depth_refiner_20261103.model import TinyDepthInfluenceUNet, predict_probabilities
from ..near_goal_depth_refiner_20261103_v4.calibration import assert_cpu_prediction_contract, check_library_environment
from ..near_goal_fresh4_calibration_20261103.data import Fresh4CalibrationDataset, PARTITION_IDS as FRESH_PARENT_IDS
from ..near_goal_depth_refiner_20261103_v5.qualification import (
    validate_selected_checkpoint, pointer, checked_sha, state_digest, write_new, _local_evidence,
)
from ..near_goal_depth_refiner_20261103_v5.training import (
    implementation_source_map, PROTOCOL_SHA256 as FIT_PROTOCOL_SHA256,
)
from .evaluation import THRESHOLD_GRID, QUALITY_PROTOCOL, BASELINE_ELIGIBILITY_PROTOCOL
from .fresh_evaluation import freeze_legacy_eligibility, evaluate_fresh_probabilities, choose_fresh_calibrated_threshold

ROOT = Path(__file__).resolve().parents[2]
VERSION = 'near_goal_binary_V5weights_fixed_scalar995_fresh4_cpu2_20261103_v6'
PROTOCOL = ROOT/'artifacts/framework/near_goal_segmentation_20261103/v5_weights_single_threshold_inference_profile_v6/protocol.json'
PROTOCOL_SHA256 = 'cfb3785d6888e2f73c3c972c4f0780664d4af93c5f48f15330bd707e39e18725'
CONSUMPTION_NAME = 'calibration_consumption.request.json'
QUALIFICATION_FILES = (
    'our_mani/near_goal_depth_refiner_20261103_v5/qualification.py',
    'our_mani/near_goal_fresh4_calibration_20261103/__init__.py',
    'our_mani/near_goal_fresh4_calibration_20261103/data.py',
    'our_mani/near_goal_fresh4_calibration_20261103/evaluation.py',
    'our_mani/near_goal_depth_refiner_20261103_v6/__init__.py',
    'our_mani/near_goal_depth_refiner_20261103_v6/evaluation.py',
    'our_mani/near_goal_depth_refiner_20261103_v6/fresh_evaluation.py',
    'our_mani/near_goal_depth_refiner_20261103_v6/qualification.py',
    'our_mani/near_goal_depth_refiner_20261103_v6/runtime.py',
    'scripts/qualify_near_goal_depth_refiner_20261103_v6.py',
)


def check_profile_bindings(selected=None, expected_sha256=None, fresh_manifest=None):
    if sha256(PROTOCOL) != PROTOCOL_SHA256:
        raise ValueError('Exact pre-fresh fixed inference-profile protocol required')
    profile = json.loads(PROTOCOL.read_text())
    if (profile.get('fixed_threshold') != .995 or profile.get('fresh_threshold_grid') != [.995]
            or profile.get('training_protocol_sha256') != FIT_PROTOCOL_SHA256
            or profile.get('selection_basis', {}).get('selection_raw_gate_pass') is not False
            or profile.get('qualification', {}).get('threshold_selection_from_fresh') is not False):
        raise ValueError('Only the predeclared same-V5-weights fixed scalar profile')
    if selected is not None:
        ref = profile['selected_checkpoint']
        if Path(selected).resolve(strict=True) != (ROOT/ref['path']).resolve(strict=True) or expected_sha256 != ref['sha256']:
            raise ValueError('Profile binds the exact unchanged V5 selected checkpoint')
    if fresh_manifest is not None:
        ref = profile['fresh_calibration']
        if (Path(fresh_manifest).resolve(strict=True) != (ROOT/ref['manifest']).resolve(strict=True)
                or sha256(fresh_manifest) != ref['manifest_sha256']):
            raise ValueError('Only the exact preassigned full fresh4 manifest can be consumed')
    return profile


def qualification_source_map():
    check_profile_bindings()
    sources = implementation_source_map()
    for name in QUALIFICATION_FILES:
        path = ROOT/name; sources[str(path)] = sha256(path)
    sources[str(PROTOCOL)] = PROTOCOL_SHA256
    return sources


def claim_calibration_once(dataset_manifest, selected_sha256, sources, output, *, capture_root):
    """Original capture-root claim also prevents copied-manifest aliases."""
    manifest = Path(dataset_manifest).resolve(strict=True)
    capture_root = Path(capture_root).resolve(strict=True)
    path = capture_root/CONSUMPTION_NAME
    sibling = manifest.parent/CONSUMPTION_NAME
    if sibling.exists():
        raise FileExistsError('Fresh calibration already consumed at dataset sibling')
    value = dict(schema='v6_fixed_scalar_fresh4_single_consumption_request_v1',
        created_utc=datetime.now(timezone.utc).isoformat(), manifest=pointer(manifest),
        selected_checkpoint_sha256=checked_sha(selected_sha256), output=str(Path(output).resolve()),
        qualification_version=VERSION, protocol_sha256=PROTOCOL_SHA256, capture_root=str(capture_root),
        training_protocol_sha256=FIT_PROTOCOL_SHA256, threshold_fixed_before_fresh=.995,
        method_source_sha256=sources, no_retry=True, request_consumes_cohort_even_if_failure=True)
    write_new(path, value)  # open('x') is the inter-process exclusive claim.
    if sibling != path:
        write_new(sibling, value)  # Any failure keeps the original claim consumed.
    return pointer(path), pointer(sibling)

def check_qualification_inputs(selected, expected_sha256, fresh_manifest):
    check_profile_bindings(selected, expected_sha256, fresh_manifest)
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
    check_profile_bindings(selected, expected_sha256, fresh_manifest)
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
            threshold_selected_from_fresh=False, **environment)
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
            same_V5_weights=True, new_training=False, selection_stage_raw_gate_pass=False,
            profile=pointer(PROTOCOL), threshold_selected_from_fresh=False,
            source_before_after_verified=True, fresh_identity=fresh_identity,
            old_selected_artifact_unchanged=True, predictions_frozen_before_scores=True,
            qualification_is_internal_static_known_50mm_axisCube_only=True,
            Native_control_qualified=False, grasp_qualified=False, TEST_read=False)
        write_new(out/'qualification.json', report)
        metadata = dict(payload['metadata'])
        metadata.update(stage='fresh4_calibrated', model_qualified=qualified,
            same_V5_weights=True, new_training=False, selection_stage_raw_gate_pass=False,
            profile=pointer(PROTOCOL), profile_protocol_sha256=PROTOCOL_SHA256,
            threshold_fixed_before_fresh=.995, threshold_selected_from_fresh=False,
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
            same_V5_weights=True, new_training=False, selection_stage_raw_gate_pass=False,
            profile_protocol_sha256=PROTOCOL_SHA256, training_protocol_sha256=FIT_PROTOCOL_SHA256,
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
