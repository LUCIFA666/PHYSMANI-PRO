"""Public prediction freeze first, then optional privileged offline scoring."""
from pathlib import Path
from collections import defaultdict
import hashlib
import json

import numpy as np
import torch

from . import VERSION
from .data import ROOT, load_public_dataset, read_rows
from ..context_gap_dropout_20261103_v1.data import sha, canonical, features, dropout_indices
from ..context_gap_dropout_20261103_v1.training import load_teacher_student, tensor_digest, verify_identity

VARIANTS = ('teacher_dense', 'student_dense', 'teacher_isolated', 'student_isolated', 'teacher_burst', 'student_burst')
FROZEN_PREPARATION = ROOT/'artifacts/framework/local_mpc_20261102/Cube_context_gap_dropout_preparation_v1/CPU_check_recorded_frame_clock_v2/CPU_check.json'


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def source_map():
    old = json.loads(FROZEN_PREPARATION.read_text())['identity']['source_sha256']
    if any(sha(path) != value for path, value in old.items()):
        raise ValueError('STOP13 training dependencies changed')
    new = ('our_mani/context_physical_eval_20261103_v1/__init__.py',
           'our_mani/context_physical_eval_20261103_v1/data.py',
           'our_mani/context_physical_eval_20261103_v1/evaluation.py',
           'scripts/evaluate_context_physical_TRAIN16_20261103_v1.py')
    return {**old, **{str(ROOT/p):sha(ROOT/p) for p in new}, str(FROZEN_PREPARATION):sha(FROZEN_PREPARATION)}


def check(output=None):
    windows, ledger, _, identity = load_public_dataset()
    report = dict(version=VERSION, parents=16, frames=1300, dense_history_anchors=len(windows),
        complete800ms_public_future_anchors=sum(int(w.future_valid[1:].all()) for w in windows),
        eligible_current_public_frames=sum(r['eligible_free_current'] for values in ledger.values() for r in values),
        frame_ledger=ledger, identity=identity, source_sha256=source_map(),
        actual_model_forward_calls=0, optimizer_constructed=False, Native_started=False,
        evaluation_labels_opened_by_this_check=False, TEST_read=False,
        no_independent_teacher_or_TF_generalization_claim=True)
    if output is not None:
        write(output, report)
    return report


def window_manifest(windows):
    values = []
    for window in windows:
        traces = {}
        for mode in ('isolated', 'burst'):
            seed = int(canonical([list(window.parent), window.anchor, mode, 1018])[:16], 16)
            keep, trace = dropout_indices(window, np.random.default_rng(seed), mode)
            traces[mode] = dict(seed=seed, retained_indices=keep.tolist(), trace=trace)
        values.append(dict(parent_id=window.parent[0], anchor_row=window.anchor,
            anchor_time=float(window.times[-1]), dense_times=window.times.tolist(), dropout=traces))
    return values


def load_models(student_path, expected_sha256):
    torch.set_num_threads(2)
    path = Path(student_path).resolve(strict=True)
    if sha(path) != expected_sha256:
        raise ValueError('Exact completed selected student SHA required')
    teacher, student, spec = load_teacher_student()
    data = torch.load(path, map_location='cpu', weights_only=False)
    metadata = data['metadata']
    if (metadata.get('training_complete') is not True or metadata.get('completed_epochs') != 30
            or metadata.get('actual_optimizer_steps') != 2100 or metadata.get('P4_or_live_runtime_admitted') is not False
            or metadata.get('physical_time_gap_robustness_qualified') is not False or data['spec'] != spec):
        raise ValueError('Completed offline frame-clock-only student, without policy/physical qualification required')
    identity_path = path.with_name('identity.json')
    if sha(identity_path) != metadata['identity_sha256']:
        raise ValueError('Selected student training identity changed')
    training_identity = json.loads(identity_path.read_text())
    verify_identity(training_identity)
    student.load_state_dict(data['state_dict'], strict=True)
    if tensor_digest(student.state_dict()) != tensor_digest(teacher.state_dict()):
        raise ValueError('Original frozen dynamics/normalization bytes must match')
    teacher.cpu().eval().requires_grad_(False)
    student.cpu().eval().requires_grad_(False)
    return teacher, student, dict(path=str(path), sha256=expected_sha256,
        original_metadata_target_tasks=metadata.get('target_tasks'),
        actual_training_scope=['pick_moving_target_on_the_table', 'pick_moving_target_on_the_table_high_speed'],
        scope_evidence='training identity preassigned registry and fit data-access ledger; original empty target_tasks metadata is not silently reinterpreted as global scope',
        training_identity=dict(path=str(identity_path), sha256=sha(identity_path)))


def predict(student_path, expected_sha256, output):
    """Explicit invocation only. Never opens evaluation_labels or native_ticks."""
    windows, ledger, _, identity = load_public_dataset()
    sources, manifest = source_map(), window_manifest(windows)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    write(output/'public_input_identity.json', identity)
    write(output/'all_frame_ledger.json', ledger)
    write(output/'window_manifest.json', manifest)
    write(output/'prediction_request.json', dict(version=VERSION, source_sha256=sources,
        public_input_identity_sha256=sha(output/'public_input_identity.json'),
        window_manifest_sha256=sha(output/'window_manifest.json'), all_frame_ledger_sha256=sha(output/'all_frame_ledger.json'),
        student=dict(path=str(Path(student_path).resolve()), sha256=expected_sha256),
        variants=list(VARIANTS), parents=16, anchors=969, CPU_threads=2,
        labels_are_not_model_inputs=True, training_or_Native_requested=False))
    teacher, student, student_identity = load_models(student_path, expected_sha256)
    arrays = {key: [] for key in ('positions', 'rotations', 'velocities', 'angular_velocities', 'contexts')}
    teacher_before, student_before = teacher.encoder_sha256(), student.encoder_sha256()
    with torch.inference_mode():
        for window, record in zip(windows, manifest):
            dense = features(window, teacher)
            batches = {'dense':dense}
            for mode in ('isolated', 'burst'):
                keep = np.asarray(record['dropout'][mode]['retained_indices'])
                batches[mode] = features(window, teacher, keep, recompute_twist=True)
            own = {key: [] for key in arrays}
            for variant in VARIANTS:
                name, mode = variant.split('_', 1)
                model, batch = (teacher if name == 'teacher' else student), batches[mode]
                result = model(torch.from_numpy(batch['history'][None]), torch.tensor([batch['length']]),
                    torch.from_numpy(batch['state'][None]), steps=16, dt=.05)
                for key in own:
                    own[key].append(result['context' if key == 'contexts' else key][0].cpu().numpy())
            for key in arrays:
                arrays[key].append(np.stack(own[key]))
    arrays = {key:np.stack(value) for key, value in arrays.items()}
    if not all(np.isfinite(value).all() for value in arrays.values()):
        raise ValueError('Nonfinite offline prediction')
    if teacher.encoder_sha256() != teacher_before or student.encoder_sha256() != student_before:
        raise ValueError('Model mutated during prediction')
    if sources != source_map() or any(sha(p) != value for p,value in identity['public_and_closed_source_sha256'].items()):
        raise ValueError('Sources/public cohort changed during prediction')
    np.savez_compressed(output/'public_predictions.npz', **arrays)
    files = {name:dict(path=str(output/name), sha256=sha(output/name)) for name in
        ('prediction_request.json', 'public_input_identity.json', 'all_frame_ledger.json', 'window_manifest.json', 'public_predictions.npz')}
    freeze = dict(version=VERSION, complete=True, files=files, source_sha256=sources, student=student_identity,
        teacher_sha256='ebec3d1b3a83ceb0d4fb3bc889f8b62376cb4581071f53de29e0b6bd71c162ef',
        variants=list(VARIANTS), all_parent_ids=identity['parent_ids'], anchors=969, frames=1300,
        actual_teacher_forward_calls=969*3, actual_student_forward_calls=969*3, actual_total_forward_calls=969*6,
        CPU_threads=2, GPU_used=False, labels_opened_during_prediction=False, Native_started=False,
        P4_candidate_forward_calls=0, qualification=False, TEST_read=False)
    write(output/'prediction_freeze.json', freeze)
    return freeze


def validate_freeze(directory):
    directory = Path(directory).resolve(strict=True)
    freeze = json.loads((directory/'prediction_freeze.json').read_text())
    if (freeze.get('version') != VERSION or freeze.get('complete') is not True
            or freeze.get('variants') != list(VARIANTS) or freeze.get('anchors') != 969 or freeze.get('frames') != 1300
            or set(freeze.get('files', {})) != {'prediction_request.json', 'public_input_identity.json', 'all_frame_ledger.json', 'window_manifest.json', 'public_predictions.npz'}
            or freeze.get('source_sha256') != source_map()):
        raise ValueError('Complete exact-cohort public prediction freeze required before any label read')
    for name, pointer in freeze['files'].items():
        if Path(pointer['path']).resolve() != directory/name or sha(pointer['path']) != pointer['sha256']:
            raise ValueError('Same-output exact prediction artifacts required')
    request = json.loads((directory/'prediction_request.json').read_text())
    if (request.get('version') != VERSION or request.get('source_sha256') != freeze['source_sha256']
            or request.get('variants') != list(VARIANTS) or request.get('anchors') != 969
            or request.get('parents') != 16 or request.get('student') != {k:freeze['student'][k] for k in ('path','sha256')}
            or any(request.get(key+'_sha256') != sha(directory/name) for key,name in
                (('public_input_identity','public_input_identity.json'), ('window_manifest','window_manifest.json'), ('all_frame_ledger','all_frame_ledger.json')))):
        raise ValueError('Pre-forward source/input/model request differs from final freeze')
    windows, ledger, cases, identity = load_public_dataset()
    if (freeze['all_parent_ids'] != identity['parent_ids']
            or json.loads((directory/'window_manifest.json').read_text()) != window_manifest(windows)
            or json.loads((directory/'all_frame_ledger.json').read_text()) != ledger
            or json.loads((directory/'public_input_identity.json').read_text()) != identity
            or sha(freeze['student']['path']) != freeze['student']['sha256']):
        raise ValueError('Prediction inputs/window cohort/student identity changed')
    with np.load(directory/'public_predictions.npz', allow_pickle=False) as stored:
        expected = dict(positions=(969,6,17,3), rotations=(969,6,17,3,3), velocities=(969,6,17,3),
                        angular_velocities=(969,6,17,3), contexts=(969,6,16))
        if set(stored.files) != set(expected):
            raise ValueError('Exact complete public prediction arrays required')
        arrays = {name:np.array(stored[name]) for name in stored.files}
        if any(value.shape != expected[name] or not np.isfinite(value).all() for name,value in arrays.items()):
            raise ValueError('Finite complete969x6 prediction arrays required')
    return freeze, windows, cases, arrays


def score(directory, output, *, include_native_labels=False):
    # This validation and all public cohort reconstruction completes before
    # the first evaluation-label file is opened, including failure paths.
    freeze, windows, cases, arrays = validate_freeze(directory)
    labels, label_sha = {}, {}
    if include_native_labels:
        for parent, case in cases.items():
            path = Path(case['case'])/'public_shadow/evaluation_labels.jsonl'
            label_sha[str(path)] = sha(path)
            values = read_rows(path)
            times = [float(row['time']) for row in values]
            if len(times) != len(set(times)):
                raise ValueError('Duplicate posthoc label clock')
            labels[parent] = {round(row['time'],8):row for row in values}
    errors, counts, parent_errors = defaultdict(list), defaultdict(int), defaultdict(lambda:defaultdict(list))
    scene_errors = defaultdict(lambda:defaultdict(list))
    for index, window in enumerate(windows):
        parent = window.parent[0]
        for vi, variant in enumerate(VARIANTS):
            for step in (2,4,8,16):
                key = variant+':'+str(step*.05)+'s'
                if not window.future_valid[step]:
                    counts[key+':public_future_unavailable'] += 1
                    continue
                prediction = arrays['positions'][index,vi,step]
                error = float(np.linalg.norm(prediction-window.future_positions[step]))
                errors['public_tracked_future:'+key].append(error)
                parent_errors[parent]['public_tracked_future:'+key].append(error)
                scene_errors[cases[parent]['scene']]['public_tracked_future:'+key].append(error)
                if include_native_labels:
                    label = labels[parent].get(round(float(window.times[-1]+step*.05),8))
                    if label is None:
                        counts[key+':posthoc_label_missing'] += 1
                    else:
                        error = float(np.linalg.norm(prediction-np.asarray(label['object_position'],float)))
                        errors['privileged_posthoc_native_position:'+key].append(error)
                        parent_errors[parent]['privileged_posthoc_native_position:'+key].append(error)
                        scene_errors[cases[parent]['scene']]['privileged_posthoc_native_position:'+key].append(error)
    def stats(x):
        return dict(count=len(x),mean=float(np.mean(x)),p95=float(np.quantile(x,.95)),max=float(np.max(x)))
    context = arrays['contexts']
    paired = {}
    for source in ('public_tracked_future','privileged_posthoc_native_position'):
        for mode in ('dense','isolated','burst'):
            for horizon in ('.1','.2','.4','.8'):
                suffix=':'+str(float(horizon))+'s'
                teacher_errors=errors.get(source+':teacher_'+mode+suffix, [])
                student_errors=errors.get(source+':student_'+mode+suffix, [])
                if teacher_errors and len(teacher_errors)==len(student_errors):
                    paired[source+':'+mode+suffix]=stats((np.asarray(student_errors)-teacher_errors).tolist())
    input_identity=json.loads((Path(directory)/'public_input_identity.json').read_text())
    scene_coverage=defaultdict(lambda:defaultdict(int))
    for parent in input_identity['parents']:
        scene_coverage[parent['scene']]['parents']+=1
        for key in ('frames','eligible_current_public_free_frames','dense5_history_anchors','anchors_with_complete800ms_public_future'):
            scene_coverage[parent['scene']][key]+=parent[key]
    report = dict(version=VERSION, prediction_freeze_sha256=sha(Path(directory)/'prediction_freeze.json'),
        parents=16, anchors=969, frames=1300, complete800ms_public_future_anchors=sum(int(w.future_valid[1:].all()) for w in windows),
        errors_m={key:stats(value) for key,value in errors.items()}, missing_future_label_counts=dict(counts),
        per_parent_errors_m={p:{k:stats(v) for k,v in values.items()} for p,values in parent_errors.items()},
        per_scene_errors_m={p:{k:stats(v) for k,v in values.items()} for p,values in scene_errors.items()},
        scene_coverage={key:dict(value) for key,value in scene_coverage.items()},
        paired_error_difference_student_minus_teacher_m=paired,
        context_L2_to_dense_teacher={variant:stats(np.linalg.norm(context[:,i]-context[:,0],axis=-1).tolist()) for i,variant in enumerate(VARIANTS)},
        context_L2_student_minus_paired_teacher={mode:stats(np.linalg.norm(context[:,i+1]-context[:,i],axis=-1).tolist())
            for i,mode in ((0,'dense'),(2,'isolated'),(4,'burst'))},
        label_files_posthoc_sha256=label_sha, public_histories_and_anchor_selection_use_GT=False,
        physical_clock=True, ordinary_pick_only=True, high_speed_coverage=False,
        old_teacher_and_TF_prior_development_exposure_not_claimed_independent=True,
        native_held_or_success_not_used_to_filter_windows=True, P4_candidate_forward_calls=0,
        candidate_compatibility='not_evaluated_without_frozen_public_feature_pair_diagnostic',
        qualification=False, live_controller_admitted=False, TEST_read=False)
    if any(sha(path) != digest for path,digest in label_sha.items()):
        raise ValueError('Posthoc labels changed')
    write(output, report)
    return report
