"""New large-model selected-VAL and single-use reserved public freeze/score.

Summaries reuse the old arithmetic only. No old VERSION, model certificate,
threshold selection or runtime admission is inherited.
"""
import json
from pathlib import Path
import numpy as np
from . import VERSION
from .config import fixed_config, ENCODER_SHA256
from .training import training_source_map, pointer, write_new, tensor_batch, INPUT_KEYS
from ..current_cube_rgbd_centre_20261103_v1.data import DualCropCubeDataset, checked_pointer, load_npz, sha256
from ..current_cube_rgbd_centre_20261103_v1.evaluation import summarize_rows

ROLES = {'selection': (16, 208), 'reserved_fresh_centre_VAL': (8, 104)}


def _validate_selected(selected, expected_sha256, manifest):
    if sha256(selected) != expected_sha256:
        raise ValueError('Exact selected whole checkpoint SHA required')
    import torch
    payload = torch.load(selected, map_location='cpu', weights_only=True)
    meta = payload.get('metadata', {}); config = fixed_config(); sources = training_source_map()
    if (meta.get('version') != VERSION or meta.get('training_complete') is not True
            or meta.get('epochs_completed') != 40 or meta.get('actual_optimizer_steps') != 520
            or meta.get('config') != config or meta.get('source_sha256') != sources
            or meta.get('manifest') != pointer(manifest) or meta.get('model_qualified') is not False
            or meta.get('runtime_or_control_admitted') is not False
            or meta.get('model_spec', {}).get('parameters') != config['expected_parameters']):
        raise ValueError('Completed new large-model same-source520step checkpoint required')
    side = Path(selected).resolve().parent
    status = json.loads((side / 'fit_status.json').read_text())
    if (status.get('version') != VERSION or status.get('training_complete') is not True
            or status.get('actual_optimizer_steps') != 520 or status.get('selected_checkpoint') != pointer(selected)):
        raise ValueError('Actual final large-model fit status required')
    expected = {'identity', 'cache_before', 'cache_after', 'history', 'sample_access', 'source_verification'}
    if set(meta.get('evidence', {})) != expected:
        raise ValueError('Complete six original fit evidence pointers required')
    for name, ref in meta['evidence'].items():
        if checked_pointer(side, ref) != side / (name + '.json'):
            raise ValueError('Original sibling fit evidence identity required')
    checked_pointer(side, meta['actual_gpu_preflight'])
    return payload, meta


def predict(selected, expected_sha256, manifest, output, encoder_checkpoint, *, role='reserved_fresh_centre_VAL'):
    if role not in ROLES: raise ValueError('Only fixed selected-VAL or reserved8 roles allowed')
    parents, count = ROLES[role]
    data = DualCropCubeDataset(manifest, role, allow_reserved=role == 'reserved_fresh_centre_VAL')
    if len(data) != count: raise ValueError('Complete fixed role denominator required')
    out = Path(output).resolve()
    if out.exists(): raise FileExistsError('Fresh prediction directory required')
    if sha256(encoder_checkpoint) != ENCODER_SHA256: raise ValueError('Original initializer changed')
    payload, meta = _validate_selected(selected, expected_sha256, manifest)
    sources = training_source_map(); before = data.revalidate(); config = fixed_config()
    request = dict(version=VERSION, role=role, selected=pointer(selected), manifest=pointer(manifest),
        source_sha256=sources, parents=parents, samples=count, availability_threshold=.5,
        no_threshold_search=True, no_runtime_qualification=True,
        one_attempt=role == 'reserved_fresh_centre_VAL', failures_consume=True)
    claim = None
    if role == 'reserved_fresh_centre_VAL':
        # Canonical SAME data claim as frozen small pipeline: a new namespace
        # cannot bypass prior consumption of these eight reserved parents.
        claim = data.path.parent / 'reserved_centre_evaluation_consumption.request.json'
        write_new(claim, request)
    out.mkdir(parents=True, exist_ok=False); write_new(out / 'request.json', request)
    import torch
    from .model import LargeDualCropCubeCentreV1, CentreConfig
    torch.set_num_threads(config['evaluation_num_threads'])
    model = LargeDualCropCubeCentreV1(CentreConfig(**config['architecture']), encoder_trainable=True)
    model.load_selected_v5_encoder(encoder_checkpoint, ENCODER_SHA256)
    model.load_state_dict(payload['model_state_dict'], strict=True); model.eval()
    for p in model.parameters(): p.requires_grad_(False)
    ledger, outputs, calls = [], [], 0
    keys = ('centre_world', 'measurement_availability_logit', 'uncertainty_score_m', 'uncertainty_score_valid')
    with torch.no_grad():
        for start in range(0, count, config['micro_batch_size']):
            samples = [data.public_sample(i) for i in range(start, min(start + config['micro_batch_size'], count))]
            arrays = {k: np.stack([s['model_inputs'][k] for s in samples]) for k in INPUT_KEYS}
            batch = tensor_batch(arrays, np.arange(len(samples)), 'cpu')
            value = model(**batch); calls += 1
            outputs.append({k: value[k].detach().cpu().numpy() for k in keys})
            ledger.extend(dict(scene_id=s['scene_id'], parent_id=s['parent_id'], role=s['role']) for s in samples)
    arrays = {k: np.concatenate([v[k] for v in outputs]) for k in keys}
    if any(not np.isfinite(v).all() for v in arrays.values()): raise ValueError('All role outputs must be finite')
    np.savez_compressed(out / 'predictions.npz', **arrays)
    write_new(out / 'ordered_scenes.json', ledger)
    data.revalidate()
    if sources != training_source_map(): raise ValueError('Method changed during public predictions')
    freeze = dict(version=VERSION, complete=True, role=role, parents=parents, samples=count,
        selected=pointer(selected), manifest=pointer(manifest), consumption=None if claim is None else pointer(claim),
        files={n: pointer(out / p) for n, p in (('predictions', 'predictions.npz'),
            ('ordered_scenes', 'ordered_scenes.json'), ('request', 'request.json'))},
        method_source_sha256=sources, bound_data_sha256=data.bound_pointers,
        data_before=before, data_after=data.revalidate(), model_forward_batch_calls=calls,
        model_forward_rows=count, device='cpu', torch_num_threads=2, autocast=False,
        environment=dict(torch=str(torch.__version__), numpy=np.__version__),
        target_label_array_decodes_before_freeze=0,
        label_proof_GT_metadata_validated_before_prediction=True,
        label_proof_metadata_not_model_input_or_prediction_selection=True,
        public_model_INPUT_KEYS=list(INPUT_KEYS), threshold=.5, threshold_search=False,
        model_qualified=False, position_measurement_qualified=False, uncertainty_qualified=False,
        runtime_or_control_admitted=False, TEST_read=False, Native_started=False, training_started=False)
    write_new(out / 'prediction_freeze.json', freeze)
    return dict(prediction_freeze=pointer(out / 'prediction_freeze.json'), role=role,
        samples=count, model_forward_batch_calls=calls, qualified=False)


def validate_freeze(output):
    out = Path(output).resolve(strict=True); frozen = json.loads((out / 'prediction_freeze.json').read_text())
    role = frozen.get('role')
    if role not in ROLES: raise ValueError('Fixed role in public freeze required')
    parents, count = ROLES[role]
    if (frozen.get('version') != VERSION or frozen.get('complete') is not True
            or frozen.get('samples') != count or frozen.get('parents') != parents
            or frozen.get('target_label_array_decodes_before_freeze') != 0
            or frozen.get('device') != 'cpu' or frozen.get('torch_num_threads') != 2
            or frozen.get('threshold') != .5 or frozen.get('threshold_search') is not False):
        raise ValueError('Full new large-model fixed-role prediction freeze required')
    refs = [frozen['selected'], frozen['manifest'], *frozen['files'].values()]
    if role == 'reserved_fresh_centre_VAL': refs.append(frozen['consumption'])
    for ref in refs: checked_pointer(out, ref)
    if (frozen['method_source_sha256'] != training_source_map()
            or any(sha256(p) != v for p, v in frozen['bound_data_sha256'].items())):
        raise ValueError('Frozen model/data/method changed')
    arrays = load_npz(frozen['files']['predictions']['path'])
    ledger = json.loads(Path(frozen['files']['ordered_scenes']['path']).read_text())
    shapes = dict(centre_world=(count, 3), measurement_availability_logit=(count,),
        uncertainty_score_m=(count, 3), uncertainty_score_valid=(count,))
    if (set(arrays) != set(shapes) or len(ledger) != count
            or len({v['scene_id'] for v in ledger}) != count
            or arrays['uncertainty_score_valid'].dtype != np.bool_
            or any(arrays[k].shape != shape or not np.isfinite(arrays[k]).all() for k, shape in shapes.items())):
        raise ValueError('All finite ordered current predictions retained before any scoring')
    return frozen, arrays, ledger


def score(output):
    frozen, arrays, ledger = validate_freeze(output)  # Must finish BEFORE target NPZ access.
    role = frozen['role']; out = Path(output)
    write_new(out / 'score_consumption.request.json', dict(version=VERSION,
        prediction_freeze=pointer(out / 'prediction_freeze.json'), one_attempt=True))
    data = DualCropCubeDataset(frozen['manifest']['path'], role, allow_reserved=role == 'reserved_fresh_centre_VAL')
    if [r['scene_id'] for r in data.records] != [r['scene_id'] for r in ledger]:
        raise ValueError('Exact ordered role identities changed')
    rows = []
    for i in range(len(data)):
        sample = data[i]; target = sample['targets']; record = sample['record']
        probability = float(np.exp(-np.logaddexp(0., -float(arrays['measurement_availability_logit'][i]))))
        valid_score = bool(arrays['uncertainty_score_valid'][i]); available = probability >= .5 and valid_score
        rows.append(dict(scene_id=sample['scene_id'], parent_id=sample['parent_id'], task=record['source_task'],
            shot_id=record['shot_id'], condition=record['condition'], condition_detail=record['condition_detail'],
            presence_target=target['presence_target'], target_reason=target['reason'], centre_valid=target['centre_valid'],
            availability_probability=probability, availability_classifier_positive=probability >= .5,
            uncertainty_score_valid=valid_score, candidate_available=available,
            uncertainty_score_m=arrays['uncertainty_score_m'][i].tolist(),
            centre_error_m=None if not target['centre_valid'] else float(np.linalg.norm(arrays['centre_world'][i] - target['centre_target_world'])),
            near_owner_pixels=target['near_owner_pixels'], wide_owner_pixels=target['wide_owner_pixels']))
    groups = {field: {name: summarize_rows([r for r in rows if r[field] == name])
        for name in sorted({r[field] for r in rows})} for field in ('task', 'parent_id', 'shot_id', 'condition')}
    report = dict(schema='large_dual_crop_centre_fixed_role_development_report_20261104_v1',
        version=VERSION, role=role, prediction_freeze=pointer(out / 'prediction_freeze.json'),
        rows=rows, aggregate=summarize_rows(rows), groups=groups,
        all_fixed_role_samples_retained=True, fixed_availability_threshold=.5,
        target_NPZ_pose_scoring_only_after_public_freeze=True,
        label_proof_GT_metadata_validated_before_prediction=True,
        uncertainty_is_uncalibrated_score_not_CI=True, model_qualified=False,
        position_measurement_qualified=False, runtime_or_control_admitted=False,
        new_model_forward_calls=0, Native_started=False, training_started=False, TEST_read=False)
    data.revalidate(); write_new(out / 'report.json', report)
    return dict(report=pointer(out / 'report.json'), aggregate=report['aggregate'])
