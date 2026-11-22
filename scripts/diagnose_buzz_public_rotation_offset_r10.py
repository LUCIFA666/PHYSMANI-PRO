"""Fixed TRAIN-only parent-crossfit test of a constant CAD gauge offset.

All original teacher frames and modes remain in the output. Public matrices
are compared as stored representatives; their none observability permission
is neither changed nor treated as full orientation evidence.
"""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'artifacts/framework/multitask_r10'
PRIOR = BASE/'buzz_two_center_TRAIN_models_and_teacher_scope_v3/report.json'
PRIOR_SHA = '0136c7f482e59d7090e52d7ed32dedeb2db2402c29749451ec34c841c92da5e4'
TASKS = ('beat_the_rotating_buzz', 'beat_the_rotating_buzz_high_speed')
ROLES = ('frame', 'wand')
BEHAVIOR = 'TRAIN_teacher_committed_waypoint'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def describe(values):
    x = np.asarray(values, dtype=float).reshape(-1)
    x = x[np.isfinite(x)]
    if not len(x):
        return dict(count=0, mean=None, q50=None, q95=None, max=None, above_point15=0)
    return dict(count=len(x), mean=float(x.mean()), q50=float(np.quantile(x, .5)),
        q95=float(np.quantile(x, .95)), max=float(x.max()), above_point15=int((x > .15).sum()))


def matrix_valid(rotations):
    finite = np.isfinite(rotations).all(axis=(-2, -1))
    safe = np.where(finite[..., None, None], rotations, np.eye(3))
    return finite & (np.max(np.abs(np.swapaxes(safe, -2, -1) @ safe - np.eye(3)), axis=(-2, -1)) < 1e-6) & (np.abs(np.linalg.det(safe)-1) < 1e-6)


def errors(prediction, label, mask):
    result = np.full(mask.shape, np.nan)
    result[mask] = np.arccos(np.clip((np.einsum('...ij,...ij->...', prediction[mask], label[mask])-1)/2, -1, 1))
    return result


def fit_offset(public_rotations, label_rotations, valid):
    # One predetermined chordal SO(3) mean, uniform over eligible TRAIN rows.
    # No held labels, outlier removal, phase selection, or threshold search.
    relative = np.swapaxes(public_rotations[valid], -2, -1) @ label_rotations[valid]
    if not len(relative):
        raise ValueError('No eligible static-offset TRAIN rows')
    mean = relative.mean(axis=0)
    u, singular, vh = np.linalg.svd(mean)
    correction = np.eye(3)
    correction[-1, -1] = np.linalg.det(u @ vh)
    return u @ correction @ vh, singular, len(relative)


def main(output):
    output = Path(output).resolve()
    if output.exists() or not output.is_relative_to(BASE):
        raise ValueError('Fresh ROOT artifact directory required')
    output.mkdir(parents=True)
    files = {}

    def pointer(path, expected=None):
        path = Path(path)
        checksum = sha(path)
        if expected is not None and checksum != expected:
            raise ValueError('Prior audited input changed: '+str(path))
        name = str(path.relative_to(ROOT))
        if name in files and files[name] != checksum:
            raise ValueError('Conflicting input SHA')
        files[name] = checksum
        return dict(path=name, sha256=checksum)

    pointer(PRIOR, PRIOR_SHA)
    prior = json.loads(PRIOR.read_text())
    assert prior['all400_original_hold_and_teacher_cases_retained'] is True
    assert prior['all_input_SHA_stable_before_after'] is True
    cv_pointer = prior['parent_crossvalidation']
    pointer(ROOT/cv_pointer['path'], cv_pointer['sha256'])
    cv = json.loads((ROOT/cv_pointer['path']).read_text())
    fold_pointer = cv['fixed_plan']
    pointer(ROOT/fold_pointer['path'], fold_pointer['sha256'])
    fold_plan = json.loads((ROOT/fold_pointer['path']).read_text())
    folds = fold_plan['parent_folds']
    registry_pointer = fold_plan['registrations']
    pointer(ROOT/registry_pointer['path'], registry_pointer['sha256'])
    cases = [row for row in prior['actual_case_provenance'] if row['behavior'] == BEHAVIOR]
    assert len(cases) == len({row['case_id'] for row in cases}) == 200
    for task in TASKS:
        task_cases = [c for c in cases if c['task'] == task]
        assert len(task_cases) == len({c['parent_source_group'] for c in task_cases}) == 100
        assert Counter(folds[c['parent_source_group']] for c in task_cases) == {0:50, 1:50}
    plan = dict(schema='r10_fixed_buzz_public_R_constant_right_offset_diagnostic_plan_v1',
        created_utc=datetime.now(timezone.utc).isoformat(), prior_strict_400_case_audit=pointer(PRIOR),
        fixed_original_teacher_case_ids=sorted(c['case_id'] for c in cases),
        tasks=list(TASKS), parents_per_task=100, parent_folds=folds,
        runtime_input='Stored current public rotations/valid/mode_valid/modes only; orientation none remains none',
        method='R_pred=R_public Q; Q=SO3 chordal mean of R_public.T R_TRAIN_label on other50 teacher parents',
        training_weight='Uniform eligible TRAIN frames, all semantic phases and public modes',
        fit_eligibility='Public valid & mode_valid & known mode & numeric SO3, native TRAIN label valid & numeric SO3',
        held_prediction_eligibility='Public valid & mode_valid & known mode & numeric SO3 only',
        missing_failed_censored_frames_retained=True, outlier_filtering=False, parameter_search=False,
        performance_gate_rad=.15, native_or_GPU_started=False, production_sources_modified=False)
    write(output/'fixed_plan.json', plan)

    # Validate raw/canonical/parent/cache hashes against the prior strict audit,
    # reusing its immutable source/transition proof rather than rescanning the
    # huge code closure. Public-only columns are loaded for inference here.
    public = {}
    canonical = {}
    for case in cases:
        for key in ('canonical_receipt','raw_child_receipt','actual_parent_receipt','cache'):
            p = case[key]
            pointer(ROOT/p['path'], p['sha256'])
        record = json.loads((ROOT/case['canonical_receipt']['path']).read_text())
        assert record['entry']['cache_sha256'] == case['cache']['sha256']
        assert record['entry']['cache'] == case['cache']['path']
        canonical[case['case_id']] = record
        with np.load(ROOT/case['cache']['path'], allow_pickle=False) as stored:
            ids = list(map(str, stored['body_ids']))
            order = [ids.index(role) for role in ROLES]
            data = {name:stored[name].copy() for name in (
                'rotations','valid','mode_valid','modes','rotation_observable','times','teacher_phase','robot_contact','robot_gripper_open')}
        for name in ('rotations','valid','mode_valid','modes','rotation_observable'):
            data[name] = data[name][:,order]
        data['matrix_valid'] = matrix_valid(data['rotations'])
        data['eligible'] = data['valid'] & data['mode_valid'] & np.isin(data['modes'], (0,1,2)) & data['matrix_valid']
        data['order'] = order
        public[case['case_id']] = data

    offsets = {}
    predictions = {}
    # Fold fitting reads labels only from the other50 parents. The predictor
    # receives only public current matrices and the frozen resulting Q.
    for task in TASKS:
        task_cases = [c for c in cases if c['task'] == task]
        offsets[task] = {}
        for fold in (0,1):
            training = [c for c in task_cases if folds[c['parent_source_group']] != fold]
            held = [c for c in task_cases if folds[c['parent_source_group']] == fold]
            labels, rotations, eligible = [], [], []
            for case in training:
                data = public[case['case_id']]
                with np.load(ROOT/case['cache']['path'], allow_pickle=False) as stored:
                    target = stored['target_rotations'][:,data['order']].copy()
                    label_valid = (stored['native_pose_valid'][:,data['order']] & stored['target_rotation_observable'][:,data['order']]).copy()
                label_valid &= matrix_valid(target)
                labels.append(target); rotations.append(data['rotations']); eligible.append(data['eligible'] & label_valid)
            labels, rotations, eligible = map(np.concatenate, (labels, rotations, eligible))
            qs = []
            diagnostics = {}
            for j, role in enumerate(ROLES):
                q, singular, count = fit_offset(rotations[:,j], labels[:,j], eligible[:,j])
                qs.append(q)
                diagnostics[role] = dict(Q=q.tolist(), training_rows=count, mean_matrix_singular_values=singular.tolist())
            Q = np.stack(qs)
            offsets[task][str(fold)] = dict(training_parent_source_groups=sorted(c['parent_source_group'] for c in training),
                held_parent_source_groups=sorted(c['parent_source_group'] for c in held), roles=diagnostics)
            assert not set(offsets[task][str(fold)]['training_parent_source_groups']) & set(offsets[task][str(fold)]['held_parent_source_groups'])
            for case in held:
                data = public[case['case_id']]
                prediction = data['rotations'] @ Q
                prediction[~data['eligible']] = np.nan
                predictions[case['case_id']] = prediction
    assert len(predictions) == 200
    write(output/'frozen_parent_fold_offsets.json', offsets)

    # Only now score held predictions against native TRAIN label columns.
    tasks = {}
    prediction_files = []
    for task in TASKS:
        task_cases = [c for c in cases if c['task'] == task]
        rows = []
        all_raw_error, all_corrected_error, all_label, all_eligible, all_external = [], [], [], [], []
        all_phases, all_modes, all_observable, all_rawvalid, all_modevalid = [], [], [], [], []
        all_caseids, all_frameindices, all_prediction = [], [], []
        for case in task_cases:
            data = public[case['case_id']]
            with np.load(ROOT/case['cache']['path'],allow_pickle=False) as stored:
                target = stored['target_rotations'][:,data['order']].copy()
                label_valid = (stored['native_pose_valid'][:,data['order']] & stored['target_rotation_observable'][:,data['order']]).copy()
            label_valid &= matrix_valid(target)
            raw_error = errors(data['rotations'], target, data['matrix_valid'] & label_valid)
            corrected = errors(predictions[case['case_id']], target, data['eligible'] & label_valid)
            external = data['eligible'] & (data['modes'] == 0)
            rows.append(dict(case_id=case['case_id'],parent_source_group=case['parent_source_group'],fold=folds[case['parent_source_group']],
                frames=len(data['times']),roles={role:dict(public_eligible=int(data['eligible'][:,j].sum()),
                    raw_stored_matrix_SO3_error_rad=describe(raw_error[:,j]),
                    corrected_public_eligible_SO3_error_rad=describe(corrected[:,j]),
                    corrected_external_SO3_error_rad=describe(corrected[external[:,j],j])) for j,role in enumerate(ROLES)}))
            all_raw_error.append(raw_error); all_corrected_error.append(corrected);all_label.append(label_valid)
            all_eligible.append(data['eligible']); all_external.append(external); all_phases.extend(map(str,data['teacher_phase']))
            all_modes.append(data['modes']); all_observable.append(data['rotation_observable']);all_rawvalid.append(data['valid']);all_modevalid.append(data['mode_valid'])
            all_caseids.extend([case['case_id']]*len(data['times']));all_frameindices.extend(range(len(data['times'])));all_prediction.append(predictions[case['case_id']])
        raw_error, corrected, label_valid, eligible, external, modes, observable, valid, modevalid = map(np.concatenate,
            (all_raw_error,all_corrected_error,all_label,all_eligible,all_external,all_modes,all_observable,all_rawvalid,all_modevalid))
        phases = np.asarray(all_phases)
        def summary(mask):
            return {role:dict(total_frames=int(mask.sum()),public_valid=int(valid[mask,j].sum()),
                mode_valid=int(modevalid[mask,j].sum()),public_eligible=int(eligible[mask,j].sum()),
                public_eligible_coverage=float(eligible[mask,j].mean()) if mask.any() else None,
                public_rotation_observable=int(observable[mask,j].sum()),
                label_valid=int(label_valid[mask,j].sum()),public_mode_counts=dict(Counter(map(int,modes[mask,j]))),
                raw_stored_matrix_SO3_error_rad=describe(raw_error[mask,j]),
                raw_public_eligible_SO3_error_rad=describe(raw_error[mask & eligible[:,j],j]),
                raw_external_SO3_error_rad=describe(raw_error[mask & external[:,j],j]),
                corrected_public_eligible_SO3_error_rad=describe(corrected[mask,j]),
                corrected_external_SO3_error_rad=describe(corrected[mask & external[:,j],j]),
                corrected_by_mode={str(m):describe(corrected[mask & eligible[:,j] & (modes[:,j]==m),j]) for m in (0,1,2)}) for j,role in enumerate(ROLES)}
        tasks[task] = dict(parents=100,total_frames=len(phases),roles=summary(np.ones(len(phases),bool)),
            per_phase={phase:summary(phases==phase) for phase in sorted(set(phases))},per_parent=rows,
            all_original_frames_and_modes_retained=True,
            fixed_offset_satisfies_point15_q95_all_roles=all(summary(np.ones(len(phases),bool))[r]['corrected_public_eligible_SO3_error_rad']['q95'] is not None and
                summary(np.ones(len(phases),bool))[r]['corrected_public_eligible_SO3_error_rad']['q95']<.15 for r in ROLES))
        prediction_path=output/(task+'__all100_teacher_parent_crossfit.npz')
        np.savez_compressed(prediction_path,case_id=np.asarray(all_caseids),frame_index=np.asarray(all_frameindices),
            predicted_rotations=np.concatenate(all_prediction),raw_SO3_error_rad=raw_error,corrected_SO3_error_rad=corrected,
            public_eligible=eligible,public_external=external,label_valid=label_valid,public_rotation_observable=observable,
            public_modes=modes,teacher_phase=phases)
        prediction_files.append(pointer(prediction_path))
    for name, checksum in files.items():
        assert sha(ROOT/name)==checksum, 'Input changed during diagnostic: '+name
    result=dict(schema='r10_buzz_public_rotation_constant_right_offset_TRAIN_parent_crossfit_diagnostic_v1',
        created_utc=datetime.now(timezone.utc).isoformat(),fixed_plan=pointer(output/'fixed_plan.json'),
        prior_strict_actual400_case_audit=pointer(PRIOR),reused_source_lineage_and_arrays_proof=True,
        actual_teacher_case_provenance=cases, actual_registration_list=registry_pointer,
        frozen_offsets=pointer(output/'frozen_parent_fold_offsets.json'),tasks=tasks,prediction_archives=prediction_files,
        held_GT_used_to_fit_or_predict=False,all_predictions_finished_before_held_scoring=True,
        all200_original_teacher_cases_retained=True,outlier_filtering=False,parameter_search=False,
        production_orientation_none_changed=False,live_tracker_connected=False,native_or_GPU_started=False,
        full_source_closure_rehashed=False,all_selected_raw_parent_cache_SHA_stable_before_after=True,
        verified_files_sha256=files,diagnostic_source=pointer(Path(__file__)),
        limitations=['Stored rotations are RGBD/CAD gauge representatives with none permission, not certified full-R observations.',
            'This is TRAIN parent crossvalidation only, not VAL/TEST or pipeline performance.',
            'A failed fixed right-offset hypothesis is retained without selecting a different model or filtering hard modes.'])
    write(output/'report.json',result)
    lines=['Public-R constant right-offset TRAIN diagnostic: all200 teacher parents, fixed2-fold, no native/GPU.']
    for task,data in tasks.items():
        for role,m in data['roles'].items():
            lines.append(f"{task}/{role}: raw q95={m['raw_public_eligible_SO3_error_rad']['q95']:.6f}, corrected q95={m['corrected_public_eligible_SO3_error_rad']['q95']:.6f}, external corrected q95={m['corrected_external_SO3_error_rad']['q95']:.6f}; coverage={m['public_eligible']}/{m['total_frames']}.")
    lines.append('All frames, modes and tails retained; orientation none remains unchanged. No held labels are used for fitting or inference.')
    (output/'report.txt').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(report=pointer(output/'report.json'),tasks={task:{role:m['corrected_public_eligible_SO3_error_rad'] for role,m in data['roles'].items()} for task,data in tasks.items()}),ensure_ascii=False))


if __name__=='__main__':
    if len(sys.argv)!=2:
        raise SystemExit('Usage: diagnose_buzz_public_rotation_offset_r10.py FRESH_ROOT_OUTPUT_DIRECTORY')
    main(sys.argv[1])
