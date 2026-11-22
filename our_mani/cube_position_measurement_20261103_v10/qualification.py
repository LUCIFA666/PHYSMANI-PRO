"""One-use position-only qualification; no raw-mask promotion or new training."""
from datetime import datetime, timezone
from pathlib import Path
import json
import os
import sys
import time
import numpy as np
import torch

from ..near_goal_depth_refiner_20261103.data import sha256
from ..near_goal_depth_refiner_20261103.model import TinyDepthInfluenceUNet, predict_probabilities
from ..near_goal_depth_refiner_20261103_v4.calibration import check_library_environment, assert_cpu_prediction_contract
from ..near_goal_depth_refiner_20261103_v5.qualification import (
    validate_selected_checkpoint, pointer, checked_sha, state_digest, write_new, _local_evidence,
)
from ..near_goal_depth_refiner_20261103_v5.training import (
    implementation_source_map, PROTOCOL_SHA256 as FIT_PROTOCOL_SHA256, checked_geometry_sources,
)
from .data import PositionCalibration8V10Dataset, PARTITION_IDS, validate_position_dataset as validate_data_dataset
from .evaluation import (
    QUALITY_PROTOCOL, freeze_legacy_eligibility, validate_prediction_route,
    score_frozen_positions, position_gate, validate_report_integrity,
)
from ..local_mpc_control_20261103.scope12_support_quality_20261103_v10 import extract_current_support_masks
from .prediction import predict_with_legacy_audit, validate_legacy_audit, PROPOSAL_THRESHOLDS, VERSION as PROPOSAL_VERSION

ROOT = Path(__file__).resolve().parents[2]
from . import VERSION
PROTOCOL = ROOT/'artifacts/framework/near_goal_segmentation_20261103/current_cube_position_scope12_v10/protocol.json'
PROTOCOL_SHA256 = '09cfe386d68b0d262c7af0d5d42404e7b350eec5e5411d501a38c88037d06500'
CONSUMPTION_NAME = 'calibration_consumption.request.json'
SOURCE_FILES = tuple('our_mani/cube_position_measurement_20261103_v10/'+name for name in ('__init__.py','data.py','prediction.py','evaluation.py','qualification.py','runtime.py')) + ('scripts/qualify_cube_position_measurement_20261103_v10.py','our_mani/local_mpc_control_20261103/scope12_support_quality_20261103_v10.py')



def check_profile_bindings(selected=None, expected_sha256=None, position_manifest=None):
    if sha256(PROTOCOL)!=PROTOCOL_SHA256:raise ValueError('Immutable pre-capture scope12-only profile changed')
    profile=json.loads(PROTOCOL.read_text())
    if (profile.get('version')!=VERSION or profile.get('proposal_thresholds')!=[.995,.9]
        or profile.get('scope12_EF_radius_m')!=.12 or profile.get('legacy_scope_m')!=.08
        or profile.get('axis_general_runtime_enabled')is not False or profile.get('axis_general_qualified')is not False
        or profile.get('new_training')is not False
        or profile.get('qualification',{}).get('required_position_fixed_denominator')!=24
        or profile['qualification'].get('scope_annulus_fixed_denominator')!=16
        or profile['qualification'].get('required_position_minimum_coverage')!=.9
        or profile['qualification'].get('scope_annulus_minimum_coverage')!=.9
        or 'legacy_control_minimum_coverage'in profile['qualification']):
        raise ValueError('Scope12-only fixed24/annulus16 profile, never axis or control8 coverage gate')
    if expected_sha256 is not None and expected_sha256!=profile['selected_weights_sha256']:
        raise ValueError('Only original dbf selected weights, no new training')
    return profile


def validate_position_dataset(dataset):
    validate_data_dataset(dataset)
    if [p['seed']for p in dataset.capture_plan['parents']]!=list(range(261103656,261103664)):
        raise ValueError('Only new8/80 fixed V10 capture cohort')


def qualification_source_map():
    from ..cube_position_measurement_20261103_v8.qualification import qualification_source_map as old_map
    from ..near_goal_cube_geometry_scope12_20261103_v1 import geometry_source_map
    from ..local_mpc_control_20261103.scope12_support_quality_20261103_v10 import support_source_map
    profile=check_profile_bindings();sources=old_map();sources.update(geometry_source_map())
    helper=profile['support_compatibility']
    if sha256(ROOT/helper['module'])!=helper['sha256']:
        raise ValueError('Exact new two-profile support compatibility implementation required')
    sources.update(support_source_map())
    package=ROOT/'our_mani/near_goal_cube_geometry_scope12_20261103_v1'
    if (sha256(package/'measurement.py')!=profile['source_geometry']['measurement_sha256']
        or sha256(package/'__init__.py')!=profile['source_geometry']['init_sha256']):
        raise ValueError('Exact single scope-change geometry required')
    for name in SOURCE_FILES:sources[str(ROOT/name)]=sha256(ROOT/name)
    sources[str(PROTOCOL)]=PROTOCOL_SHA256
    return sources


def check_execution_environment(model=None, expected=None):
    profile=check_profile_bindings()['environment'];environment=check_library_environment()
    if Path(sys.executable).resolve()!=Path('/home/ubuntu/physmani-conda/envs/physmani5090/bin/python').resolve():
        raise ValueError('NativePython CPU2 environment required')
    for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
        if os.environ.get(name)!=profile[name]:raise ValueError('Native policy environment4 required')
        environment[name]=os.environ[name]
    if model is not None:
        assert_cpu_prediction_contract(model)
        environment.update(torch_num_threads=torch.get_num_threads(),model_eval=not model.training,model_dtype='float32',device='cpu')
    if expected is not None and any(environment.get(k)!=v for k,v in expected.items()):raise ValueError('Calibration/runtime CPU2 identity drift')
    return environment


def _capture_root(dataset):
    roots = {Path(r['inputs_path']).resolve().parent.parent.parent for r in dataset.records}
    if len(roots) != 1:
        raise ValueError('Single original eight-case capture-root required')
    return roots.pop()


def claim_calibration_once(dataset_manifest, selected_sha256, sources, output, *, capture_root):
    manifest = Path(dataset_manifest).resolve(strict=True)
    capture_root = Path(capture_root).resolve(strict=True)
    path, sibling = capture_root/CONSUMPTION_NAME, manifest.parent/CONSUMPTION_NAME
    if sibling.exists():
        raise FileExistsError('Position calibration already consumed')
    value = dict(schema='position_scope12_calibration_single_consumption_request_v10',
        created_utc=datetime.now(timezone.utc).isoformat(), manifest=pointer(manifest),
        selected_checkpoint_sha256=checked_sha(selected_sha256), output=str(Path(output).resolve()),
        capture_root=str(capture_root), qualification_version=VERSION,
        position_profile_sha256=PROTOCOL_SHA256, training_protocol_sha256=FIT_PROTOCOL_SHA256,
        method_source_sha256=sources, position_measurement_only=True,
        model_qualified=False, raw_mask_model_qualified=False, no_retry=True,
        request_consumes_cohort_even_if_failure=True)
    write_new(path, value)
    if sibling != path: write_new(sibling, value)
    return pointer(path), pointer(sibling)


def check_qualification_inputs(selected, expected_sha256, position_manifest):
    check_profile_bindings(selected, expected_sha256, position_manifest)
    payload, evidence = validate_selected_checkpoint(selected, expected_sha256)
    dataset = PositionCalibration8V10Dataset(position_manifest)
    validate_position_dataset(dataset)
    sources = qualification_source_map(); environment = check_execution_environment()
    root = _capture_root(dataset)
    return dict(complete=True, qualification_started=False, model_forward_calls=0,
        position_measurement_qualified=False, model_qualified=False, raw_mask_model_qualified=False,
        selected=evidence['selected'], selected_epoch=payload['metadata']['selected_epoch'],
        position_manifest=pointer(dataset.path), samples=len(dataset), environment=environment,
        method_source_sha256=sources,
        consumption_exists=any((p/CONSUMPTION_NAME).exists() for p in (root, dataset.path.parent)))


def _verify_unchanged(dataset, sources, evidence, claims):
    dataset.revalidate()
    if qualification_source_map() != sources:
        raise ValueError('Position method source changed')
    refs = [evidence['selected'], evidence['fit_status']]+list(evidence['evidence'].values())+list(claims)
    if any(sha256(ref['path']) != ref['sha256'] for ref in refs):
        raise ValueError('Original selected fit or one-use request changed')


def qualify_selected_checkpoint(selected, expected_sha256, position_manifest, output):
    started = time.perf_counter()
    check_profile_bindings(selected, expected_sha256, position_manifest)
    payload, selected_evidence = validate_selected_checkpoint(selected, expected_sha256)
    sources = qualification_source_map(); check_execution_environment()
    dataset = PositionCalibration8V10Dataset(position_manifest)
    validate_position_dataset(dataset)
    if len(dataset) != 80: raise ValueError('All80 new position snapshots required')
    out = Path(output).resolve()
    if out.exists() or out in (dataset.path.parent, Path(selected).resolve().parent):
        raise ValueError('Independent new immutable qualification output required')
    capture = _capture_root(dataset)
    request, sibling = claim_calibration_once(dataset.path, expected_sha256, sources, out, capture_root=capture)
    out.mkdir(parents=True, exist_ok=False)
    write_new(out/'request.json', json.loads(Path(request['path']).read_text()))
    identity = dict(manifest=pointer(dataset.path), bound_data_sha256=dict(dataset.bound_pointers),
        capture_root=str(capture), request=request, dataset_request=sibling,
        selected_evidence=selected_evidence, method_source_sha256=sources, profile=pointer(PROTOCOL))
    write_new(out/'position_identity.json', identity)
    completed_forward_calls=0
    current_scene=None
    try:
        samples = [dataset[i] for i in range(len(dataset))]
        eligibility = freeze_legacy_eligibility(samples)
        write_new(out/'eligibility_before_predictions.json', eligibility)
        model = TinyDepthInfluenceUNet().float().cpu()
        model.load_state_dict(payload['model_state_dict'], strict=True)
        model.eval(); torch.set_num_threads(2)
        environment = check_execution_environment(model)
        probabilities, predictions, raw, points, stencils = [], [], [], [], []
        for sample in samples:
            current_scene=sample['scene_id']
            features = sample['features']
            if (not isinstance(features, np.ndarray) or features.dtype != np.float32
                    or features.shape != (4, 9, 64, 64) or not np.isfinite(features).all()):
                raise ValueError('Original float32 four-view current features required')
            check_execution_environment(model, environment)
            probability = predict_probabilities(model, features)
            completed_forward_calls+=1
            prediction, audit = predict_with_legacy_audit(sample['inputs'], sample['crops'], probability)
            prediction['legacy_audit'] = audit
            validate_prediction_route(prediction)
            masks = extract_current_support_masks(sample['crops'], prediction['measurement'])
            raw.append(prediction.pop('raw_Cube_mask'))
            points.append(masks['points']); stencils.append(masks['stencils'])
            prediction.update(scene_id=sample['scene_id'], support_array_index=len(predictions))
            predictions.append(prediction); probabilities.append(probability)
        if len(probabilities) != 80 or state_digest(model.state_dict()) != state_digest(payload['model_state_dict']):
            raise ValueError('Exactly80 forwards must preserve immutable selected weights')
        check_execution_environment(model, environment)
        ids = np.array([s['scene_id'] for s in samples])
        with (out/'probabilities.npz').open('xb') as stream:
            np.savez_compressed(stream, probabilities=np.stack(probabilities), scene_ids=ids)
        with (out/'public_support_masks.npz').open('xb') as stream:
            np.savez_compressed(stream, raw_Cube_mask=np.stack(raw), points=np.stack(points),
                                stencils=np.stack(stencils), scene_ids=ids)
        write_new(out/'public_positions.json', predictions)
        execution = dict(environment=environment, forward_calls=80, samples=80, parent_count=8,
                         proposal_thresholds=list(PROPOSAL_THRESHOLDS), proposal_version=PROPOSAL_VERSION, threshold_selected_from_fresh=False)
        _verify_unchanged(dataset, sources, selected_evidence, (request, sibling))
        freeze = dict(complete=True, frozen_before_GT_scores=True, execution=execution,
            selected=selected_evidence['selected'], position_identity=identity,
            files={name:pointer(out/name) for name in ('probabilities.npz', 'public_support_masks.npz',
                'public_positions.json', 'eligibility_before_predictions.json')})
        write_new(out/'prediction_freeze.json', freeze)
        frozen = json.loads((out/'public_positions.json').read_text())
        with np.load(out/'public_support_masks.npz', allow_pickle=False) as stored:
            if stored['scene_ids'].tolist() != ids.tolist(): raise ValueError('Frozen support order changed')
            for i, (sample, prediction) in enumerate(zip(samples, frozen)):
                masks = extract_current_support_masks(sample['crops'], prediction['measurement'])
                if not np.array_equal(masks['points'], stored['points'][i]) or not np.array_equal(masks['stencils'], stored['stencils'][i]):
                    raise ValueError('Frozen masks must be the exact public supports, before GT scoring')
                prediction['raw_Cube_mask'] = stored['raw_Cube_mask'][i]
                validate_prediction_route(prediction)
        report = score_frozen_positions(samples, frozen, eligibility)
        decision = position_gate(report['positions'], report['used_available_support']['points'],
                                 report['used_available_support']['stencils'])
        if validate_report_integrity(report,eligibility)!=decision:
            raise ValueError('Fixed80 scope/annulus/support reports must rebuild exactly')
        if decision != report['decision']: raise ValueError('Position gate must match unchanged frozen evaluator')
        _verify_unchanged(dataset, sources, selected_evidence, (request, sibling))
        qualified = decision['position_measurement_qualified']
        certificate = dict(version=VERSION, complete=True, position_identity=identity,
            execution=execution, report=report, decision=decision,
            position_measurement_qualified=qualified,scope12_EF_qualified=qualified,
            axis_general_qualified=False,active_certified_routes=decision['active_certified_routes'],
            composite_global_coverage_qualified=decision['composite_global_coverage_qualified'],
            model_qualified=False, raw_mask_model_qualified=False,
            force_or_contact_qualified=False,
            same_V5_weights=True, new_training=False, predictions_frozen_before_GT_scores=True,
            source_before_after_verified=True, Native_control_qualified=False, grasp_qualified=False, TEST_read=False)
        write_new(out/'qualification.json', certificate)
        metadata = dict(payload['metadata'])
        metadata.update(stage='position_calibrated', qualification_version=VERSION,
            position_measurement_qualified=qualified, model_qualified=False, raw_mask_model_qualified=False,
            force_or_contact_qualified=False,
            scope12_EF_qualified=qualified,axis_general_qualified=False,
            active_certified_routes=decision['active_certified_routes'],scope_radius_m=.12,
            composite_global_coverage_qualified=decision['composite_global_coverage_qualified'],
            position_calibration_state='passed' if qualified else 'failed', qualified=False,
            same_V5_weights=True, new_training=False, position_profile=pointer(PROTOCOL),
            position_profile_sha256=PROTOCOL_SHA256, position_proposal_thresholds=list(PROPOSAL_THRESHOLDS), position_prediction_version=PROPOSAL_VERSION,
            position_calibration_parent_ids=list(PARTITION_IDS), position_calibration_environment=environment,
            position_calibration_forward_calls=80, position_identity=identity,
            selected_checkpoint_sha256=expected_sha256, selected_weights_sha256=state_digest(payload['model_state_dict']),
            completion=False, orientation_observed=False, known_axisaligned_50mm_Cube_prior=True,
            raw_mask_failures_preserved=True, prior_position_failures_preserved=True, Native_control_qualified=False, grasp_qualified=False,
            qualification_evidence={name:dict(path=name+'.json',sha256=sha256(out/(name+'.json')))
                for name in ('position_identity', 'request', 'eligibility_before_predictions', 'prediction_freeze', 'qualification')})
        with (out/'calibrated.pt').open('xb') as stream:
            torch.save(dict(version=VERSION, model_state_dict=payload['model_state_dict'], metadata=metadata), stream)
        write_new(out/'status.json', dict(version=VERSION, complete=True, qualification_complete=True,
            position_measurement_qualified=qualified, model_qualified=False, raw_mask_model_qualified=False,
            force_or_contact_qualified=False,
            scope12_EF_qualified=qualified,axis_general_qualified=False,
            active_certified_routes=decision['active_certified_routes'],
            composite_global_coverage_qualified=decision['composite_global_coverage_qualified'],
            calibrated_checkpoint_sha256=sha256(out/'calibrated.pt'), selected_checkpoint_sha256=expected_sha256,
            selected_weights_sha256=metadata['selected_weights_sha256'], position_identity=identity,
            execution=execution, Native_control_qualified=False, grasp_qualified=False, TEST_read=False,
            wall_seconds=time.perf_counter()-started))
        return certificate
    except BaseException as exc:
        write_new(out/'failure.json', dict(version=VERSION, complete=False, position_measurement_qualified=False,
            model_qualified=False, raw_mask_model_qualified=False, request=request, cohort_consumed=True,
            actual_completed_forward_calls=completed_forward_calls,current_scene=current_scene,
            error_type=type(exc).__name__, error=str(exc), no_retry=True))
        raise
