"""Read-only R10 TRAIN smoke diagnostics; never collect, fit, or rewrite labels.

The canonical collector receipts, preserved child receipts, and every declared
output are SHA checked before an episode contributes any numerical result.
Missing/invalid/short-terminal episodes stay visible in the coverage report.
Native pose differences are offline label diagnostics, not policy observations.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from our_mani.multitask.specs import TASK_SPECS, get_task_spec
from scripts.collect_multitask_training_r10 import context_window_eligibility

BASE = ROOT / 'artifacts/framework/multitask_r10'
VERSION = 'r10_offline_TRAIN_smoke_analysis_v1'


class SmokeAuditError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise SmokeAuditError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        allow_nan=False, separators=(',', ':')).encode()).hexdigest()


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            result.update(block)
    return result.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def absolute(path):
    return (ROOT / path).resolve()


def artifact_path(path):
    path = absolute(path)
    require(path.is_relative_to(BASE.resolve()), 'collector output is outside desktop R10: ' + str(path))
    return path


def pointer(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=sha(path))


def fraction(numerator, denominator):
    return None if denominator == 0 else float(numerator / denominator)


def scalar_stats(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return dict(count=0, mean=None, median=None, p95=None, maximum=None)
    return dict(count=len(values), mean=float(values.mean()), median=float(np.median(values)),
                p95=float(np.quantile(values, .95)), maximum=float(values.max()))


def continuity(mask, times, modes=None):
    """Exact observed-frame runs; a gap or public mode change breaks a run."""
    mask = np.asarray(mask, dtype=bool)
    runs, length = [], 0
    for i, valid in enumerate(mask):
        same = i == 0 or (abs(times[i] - times[i-1] - .05) <= 1e-6
                         and (modes is None or modes[i] == modes[i-1]))
        if length and (not valid or not same):
            runs.append(length); length = 0
        if valid:
            length += 1
    if length:
        runs.append(length)
    return dict(frames=int(mask.sum()), fraction=fraction(int(mask.sum()), len(mask)),
        run_count=len(runs), run_lengths_frames=runs, longest_run_frames=max(runs, default=0),
        runs_at_least_five_frames=sum(n >= 5 for n in runs),
        contiguous_five_frame_endpoints=sum(max(0, n-4) for n in runs),
        duration_of_observed_intervals_seconds=float(sum(max(0, n-1) for n in runs)*.05))


def validate_arrays(arrays, diagnostics, spec):
    require('times' in arrays and 'body_ids' in arrays, 'NPZ lacks clock/body identities')
    times = np.asarray(arrays['times'])
    bodies = list(map(str, arrays['body_ids']))
    t, b = len(times), len(bodies)
    expected = {role.name for role in spec.roles if role.name not in spec.audit_only_goal_roles and role.kind == 'body'}
    require(t >= 1 and t <= 201 and times.shape == (t,) and np.isfinite(times).all()
        and abs(times[0]) <= 1e-6 and np.allclose(np.diff(times), .05, atol=1e-6, rtol=0),
        'NPZ does not use the 0..200 counted 50ms clock')
    require(len(set(bodies)) == b and set(bodies) == expected, 'NPZ body IDs differ from physical TaskSpec roles')
    shapes = dict(positions=(t,b,3), rotations=(t,b,3,3), linear_velocities=(t,b,3),
        angular_velocities=(t,b,3), valid=(t,b), confidence=(t,b), modes=(t,b),
        rotation_observable=(t,b), mode_valid=(t,b), native_pose_valid=(t,b),
        dynamics_segment_valid=(t,b), target_positions=(t,b,3), target_rotations=(t,b,3,3),
        target_linear_velocities=(t,b,3), target_angular_velocities=(t,b,3), target_valid=(t,b),
        target_rotation_observable=(t,b), teacher_action_valid=(t,))
    for name, shape in shapes.items():
        require(name in arrays and arrays[name].shape == shape, 'NPZ shape mismatch: ' + name)
    for name in ('teacher_stage', 'teacher_phase', 'teacher_reference_body', 'teacher_reference_role',
                 'teacher_reference_kind', 'teacher_manipulated_body', 'teacher_manipulated_role', 'public_command_kind'):
        require(name in arrays and arrays[name].shape == (t,), 'NPZ teacher/public-command shape mismatch: ' + name)
    for name in ('valid', 'rotation_observable', 'mode_valid', 'native_pose_valid', 'dynamics_segment_valid',
                 'target_valid', 'target_rotation_observable', 'teacher_action_valid'):
        require(arrays[name].dtype == np.bool_, 'NPZ masks must remain boolean: ' + name)
    require(np.isin(arrays['modes'], (0,1,2)).all(), 'unknown numerical public motion mode')
    frames = diagnostics.get('frames')
    require(isinstance(frames, list) and len(frames) == t, 'diagnostic frame coverage differs from NPZ')
    for i, frame in enumerate(frames):
        require(abs(float(frame['timestamp']) - times[i]) <= 1e-6
            and set(frame['bodies']) == set(bodies), 'diagnostic clock/body coverage differs')
        require(frame['public_command_kind'] == str(arrays['public_command_kind'][i]),
                'public command differs between diagnostics and NPZ')
        for j,body in enumerate(bodies):
            evidence=frame['bodies'][body]
            require(not arrays['valid'][i,j] or (evidence.get('directly_visible') is True
                and evidence.get('registration_valid') is True), 'history validity falsely marks an occluded/unregistered body fresh')
    require(not np.any(arrays['rotation_observable'] & ~arrays['valid']),
            'full rotation observability requires valid current history')
    require(not np.any(arrays['valid'] & (~arrays['mode_valid'] | ~arrays['dynamics_segment_valid'])),
            'invalid mode/public segment entered history validity')
    require(not np.any(arrays['target_valid'] & (~arrays['native_pose_valid'] | ~arrays['mode_valid']
            | ~arrays['dynamics_segment_valid'])), 'invalid mode/public segment entered target validity')
    for prefix, valid in (('', arrays['valid']), ('target_', arrays['native_pose_valid'])):
        for suffix in ('positions','rotations','linear_velocities','angular_velocities'):
            require(np.isfinite(arrays[prefix+suffix][valid]).all(), 'nonfinite valid array: ' + prefix+suffix)
        rotations = arrays[prefix+'rotations'][valid]
        require(np.allclose(rotations @ np.swapaxes(rotations,-1,-2), np.eye(3), atol=1e-5, rtol=0)
                and np.allclose(np.linalg.det(rotations), 1., atol=1e-5, rtol=0),
                'valid rotation is outside SO(3): ' + prefix)
    return times, bodies, frames


def velocity_comparison(reported, derived, mask, moving_threshold, zero_threshold):
    a, b = reported[mask], derived[mask]
    if not len(a):
        return dict(pairs=0, vector_rms_error=None, component_bias_reported_minus_pose_difference=None,
            mean_error_norm=None, maximum_error_norm=None, pose_motion_pairs=0,
            pose_motion_but_reported_zero_pairs=0, pose_motion_but_reported_zero_fraction=None,
            reported_speed=scalar_stats([]), pose_difference_speed=scalar_stats([]))
    error = a-b
    speed, reported_speed = np.linalg.norm(b,axis=1), np.linalg.norm(a,axis=1)
    moving = speed > moving_threshold
    missed = moving & (reported_speed <= zero_threshold)
    return dict(pairs=len(a), vector_rms_error=float(np.sqrt(np.mean(np.sum(error*error,axis=1)))),
        component_bias_reported_minus_pose_difference=error.mean(axis=0).tolist(),
        mean_error_norm=float(np.linalg.norm(error,axis=1).mean()),
        maximum_error_norm=float(np.linalg.norm(error,axis=1).max()),
        pose_motion_pairs=int(moving.sum()), pose_motion_but_reported_zero_pairs=int(missed.sum()),
        pose_motion_but_reported_zero_fraction=fraction(int(missed.sum()),int(moving.sum())),
        reported_speed=scalar_stats(reported_speed), pose_difference_speed=scalar_stats(speed))


def native_twist_audit(arrays, body, thresholds):
    times = arrays['times']; dt = np.diff(times)
    positions, rotations = arrays['target_positions'][:,body], arrays['target_rotations'][:,body]
    valid = arrays['native_pose_valid'][:,body]
    pair_valid = valid[:-1] & valid[1:]
    derived_v = np.diff(positions,axis=0)/dt[:,None]
    derived_w = np.zeros_like(derived_v)
    if pair_valid.any():
        derived_w[pair_valid] = Rotation.from_matrix(
            rotations[1:][pair_valid] @ np.swapaxes(rotations[:-1][pair_valid],-1,-2)).as_rotvec()/dt[pair_valid,None]
    reported_v = (arrays['target_linear_velocities'][:-1,body]+arrays['target_linear_velocities'][1:,body])*.5
    reported_w = (arrays['target_angular_velocities'][:-1,body]+arrays['target_angular_velocities'][1:,body])*.5
    modes = arrays['modes'][:,body]
    mode_valid = arrays['mode_valid'][:,body]
    segment = arrays['dynamics_segment_valid'][:,body]
    stages, phases = arrays['teacher_stage'].astype(str), arrays['teacher_phase'].astype(str)
    action = arrays['teacher_action_valid']
    phase_continuous = action[:-1] & action[1:] & (phases[:-1] == phases[1:]) & (stages[:-1] == stages[1:])
    segmented = pair_valid & mode_valid[:-1] & mode_valid[1:] & (modes[:-1] == modes[1:])
    segmented &= segment[:-1] & segment[1:] & phase_continuous
    result = dict(pair_clock='adjacent native pose samples at 50ms; derivative located at interval midpoint',
        reported_velocity_comparator='mean of native get_velocity at the two interval endpoints',
        angular_frame='world: log(R[t+1] @ R[t].T) / dt', raw_native_pairs=int(pair_valid.sum()),
        segmented_pairs=int(segmented.sum()),
        segmentation='both mode-valid, same public mode, both exogenous public-command mask, same actually observed teacher stage/phase',
        note='terminal row has no teacher action and is omitted only from segmented comparison; raw comparison retains it',
        thresholds=thresholds, raw_native={}, segmented={})
    for label, mask in (('raw_native',pair_valid), ('segmented',segmented)):
        result[label]['linear_m_per_s'] = velocity_comparison(reported_v,derived_v,mask,
            thresholds['linear_motion_m_per_s'],thresholds['reported_zero_m_per_s'])
        result[label]['angular_rad_per_s'] = velocity_comparison(reported_w,derived_w,mask,
            thresholds['angular_motion_rad_per_s'],thresholds['reported_zero_rad_per_s'])
    result['segmented_by_public_mode'] = {name:dict(
        linear_m_per_s=velocity_comparison(reported_v,derived_v,segmented & (modes[:-1] == mode),
            thresholds['linear_motion_m_per_s'],thresholds['reported_zero_m_per_s']),
        angular_rad_per_s=velocity_comparison(reported_w,derived_w,segmented & (modes[:-1] == mode),
            thresholds['angular_motion_rad_per_s'],thresholds['reported_zero_rad_per_s']))
        for mode,name in enumerate(('external','attached','released'))}
    return result


def phase_coverage(arrays, spec, frames, ids):
    action = arrays['teacher_action_valid']; stages = arrays['teacher_stage'].astype(str)
    phases = arrays['teacher_phase'].astype(str)
    mapping = {stage:phase.name for phase in spec.phase_skills for stage in phase.TRAIN_expert_stages}
    unknown = sorted(set(stages[action])-set(mapping))
    mismatches = [int(i) for i in np.flatnonzero(action) if mapping.get(stages[i]) != phases[i]]
    require(not unknown and not mismatches, 'actual teacher stage/phase differs from TaskSpec')
    result = dict(teacher_action_rows=int(action.sum()), stages=dict(Counter(stages[action])),
        phases=dict(Counter(phases[action])), unknown_stages=unknown,
        required_phases_missing=[phase.name for phase in spec.phase_skills
            if not phase.optional_after_native_completion and not np.any(action & (phases == phase.name))],
        optional_phases_missing=[phase.name for phase in spec.phase_skills
            if phase.optional_after_native_completion and not np.any(action & (phases == phase.name))],
        role_availability_by_phase=[])
    role = arrays['teacher_reference_role'].astype(str); body = arrays['teacher_reference_body'].astype(str)
    kind = arrays['teacher_reference_kind'].astype(str)
    for phase, reference_role, reference_body, reference_kind in sorted(set(zip(phases[action],role[action],body[action],kind[action]))):
        selected = action & (phases == phase) & (role == reference_role) & (body == reference_body) & (kind == reference_kind)
        indices = np.flatnonzero(selected); count = len(indices)
        if reference_kind in ('world','phase_origin'):
            direct = valid = rotation = count
        else:
            require(reference_body in ids, 'teacher reference body is missing from tracked roles')
            j = ids.index(reference_body)
            direct = sum(frames[i]['bodies'][reference_body].get('directly_visible') is True for i in indices)
            valid = int(arrays['valid'][selected,j].sum())
            rotation = int((arrays['valid'][selected,j] & arrays['rotation_observable'][selected,j]).sum())
        needs_rotation = reference_kind == 'observed_se3'
        result['role_availability_by_phase'].append(dict(phase=phase, reference_role=reference_role,
            actual_TRAIN_supervision_body=reference_body, reference_kind=reference_kind,
            action_rows=count, direct_visible_rows=int(direct), history_valid_rows=valid,
            full_rotation_observable_rows=rotation, requires_full_reference_rotation=needs_rotation,
            required_reference_available_rows=rotation if needs_rotation else valid,
            required_reference_available_fraction=fraction(rotation if needs_rotation else valid,count),
            reference_is_virtual_public_frame=reference_kind in ('world','phase_origin')))
    result['interpretation'] = ('Actual TRAIN teacher bindings only; these are offline diagnostics, not hidden phase/body inputs '
        'to the actor. This availability measure does not apply confidence/reachability/collision or guarantee task completion.')
    return result


def rigid_layout_audit(arrays, spec, layout):
    ids = list(map(str,arrays['body_ids']))
    pairs = [('frame',child) for child in ids if child.startswith(('peg','spoke'))]
    if 'button' in ids and 'wrap' in ids:
        pairs.append(('button','wrap'))
    result = []
    for parent,child in pairs:
        p,c = ids.index(parent),ids.index(child)
        valid = arrays['native_pose_valid'][:,p] & arrays['native_pose_valid'][:,c]
        row = dict(parent=parent,child=child,native_pose_samples=int(valid.sum()),
            relative_position_rms_m=None,relative_position_max_m=None,
            relative_rotation_rms_rad=None,relative_rotation_max_rad=None)
        if valid.any():
            pr,cr = arrays['target_rotations'][valid,p],arrays['target_rotations'][valid,c]
            local_p = np.einsum('tji,tj->ti',pr,arrays['target_positions'][valid,c]-arrays['target_positions'][valid,p])
            local_r = np.einsum('tji,tjk->tik',pr,cr)
            center = np.median(local_p,axis=0); rotation = Rotation.from_matrix(local_r).mean().as_matrix()
            delta_p = np.linalg.norm(local_p-center,axis=1)
            delta_r = np.linalg.norm(Rotation.from_matrix(local_r @ rotation.T).as_rotvec(),axis=1)
            row.update(relative_position_rms_m=float(np.sqrt(np.mean(delta_p**2))),
                relative_position_max_m=float(delta_p.max()),relative_rotation_rms_rad=float(np.sqrt(np.mean(delta_r**2))),
                relative_rotation_max_rad=float(delta_r.max()))
        result.append(row)
    if layout is not None:
        require(layout.get('version') == 'r10_TRAIN_observed_rigid_layout_v1' and layout.get('task') == spec.task_name
            and layout.get('split') == 'train' and layout.get('evidence_sha256') == digest(
                {k:v for k,v in layout.items() if k != 'evidence_sha256'}), 'TRAIN rigid layout evidence is invalid')
        require(layout.get('arrays_sha256') == digest({key:arrays[key].tolist() for key in
            ('body_ids','times','target_positions','target_rotations','target_valid')}), 'rigid layout refers to different label arrays')
    return dict(native_raw_pose_relative_consistency=result,collector_layout=layout,
        native_pose_report_does_not_promote_current_tracker_orientation=True,
        derived_layout_is_not_retroactively_applied_to_history=True)


def audit_case(case_path, plan, case, thresholds):
    record = read(case_path)
    require(record.get('version') == 'r10_single_case_collection_receipt_v1' and record.get('status') == 'completed'
        and record.get('case_id') == case['case_id'] and record.get('plan_sha256') == plan['plan_sha256']
        and record.get('task') == case['task'] and record.get('split') == case['split']
        and record.get('raw_episode_id') == case['raw_episode_id']
        and record.get('parent_source_group') == case['parent_source_group']
        and record.get('case_isolation') == 'fresh_simulator_process_per_episode'
        and type(record.get('process_id')) is int and record['process_id'] > 0, 'case identity/status differs')
    require(record.get('policy_history_contains_native_GT') is False and record.get('teacher_paths_requested') is False
        and record.get('teacher_actions_used') is True, 'TRAIN causal/teacher namespace receipt differs')
    files = record.get('files_sha256',{})
    require(isinstance(files,dict) and files, 'case has no output SHA receipts')
    inputs = [pointer(case_path)]
    for name,expected in files.items():
        path = artifact_path(name)
        require(path.is_file() and sha(path) == expected, 'declared output changed/missing: ' + str(path))
        inputs.append(pointer(path))
    parent = record.get('parent_process_receipt')
    if parent is not None:
        path = artifact_path(parent['child_case_path'])
        require(sha(path) == parent['child_case_sha256'], 'preserved raw child receipt changed')
        child = read(path)
        require({key:value for key,value in record.items() if key != 'parent_process_receipt'} == child,
                'canonical parent is not the exact preserved raw child')
        launcher = absolute(parent['launcher'])
        require(sha(launcher) == parent['launcher_sha256'], 'parent launcher changed')
        command = parent.get('command',[])
        require(isinstance(command,list) and '--case-id' in command and command[command.index('--case-id')+1] == case['case_id'],
                'parent command case identity differs')
        inputs += [pointer(path),pointer(launcher)]
    entry = record['entry'];cache = artifact_path(entry['cache'])
    require(entry.get('task') == case['task'] and entry.get('split') == 'train'
        and entry.get('raw_episode_id') == case['raw_episode_id']
        and entry.get('parent_raw_episode_id') == case['parent_raw_episode_id']
        and entry.get('parent_source_group') == case['parent_source_group']
        and entry.get('observation_source') == 'causal_RGBD_and_proprioception'
        and entry.get('target_source') == 'TRAIN_VAL_native_supervision'
        and files.get(str(cache.relative_to(ROOT))) == entry.get('cache_sha256') == sha(cache),
        'track entry identity/SHA/information source differs')
    diagnostics_paths = [artifact_path(name) for name in files if Path(name).name == 'tracker_diagnostics.json']
    require(len(diagnostics_paths) == 1, 'one SHA-bound tracker diagnostic file required')
    with np.load(cache,allow_pickle=False) as loaded:
        arrays = {key:loaded[key] for key in loaded.files}
    spec = get_task_spec(case['task'])
    times,ids,frames = validate_arrays(arrays,read(diagnostics_paths[0]),spec)
    alignment = record['alignment'];clock = alignment['clock']
    require(clock['ticks'] == len(times)-1 and clock['dt'] == .05 and clock['budget_seconds'] == 10.
        and abs(clock['simulation_seconds']-times[-1]) <= 1e-6, 'native counted receipt differs from NPZ')
    for name in ('actual_physics_seconds','actual_task_seconds'):
        value = alignment[name]
        require(np.isfinite(value) and abs(value-times[-1]) <= 2e-5, 'native clock differs from NPZ: ' + name)
    eligibility = context_window_eligibility(arrays)
    require(entry.get('frames') == len(times) and entry.get('context_window_eligibility') == eligibility
        and entry.get('context_training_eligible') == (eligibility['usable_context_windows'] > 0),
        'collector eligible-window receipt differs from independently recomputed windows')
    phase = phase_coverage(arrays,spec,frames,ids)
    bodies = {}
    for j,body in enumerate(ids):
        diagnostic = [frame['bodies'][body] for frame in frames]
        direct = np.array([row.get('directly_visible') is True for row in diagnostic])
        registered = np.array([row.get('registration_valid') is True for row in diagnostic])
        ambiguous = np.array([row.get('rotation_ambiguous') is True for row in diagnostic])
        available = arrays['valid'][:,j]
        modes = arrays['modes'][:,j]
        bodies[body] = dict(direct_visible=continuity(direct,times),
            directly_visible_and_registered=continuity(direct & registered,times),
            history_valid_same_mode=continuity(available,times,modes),
            full_orientation_history=continuity(available & arrays['rotation_observable'][:,j],times,modes),
            mode_rows={name:int((modes == mode).sum()) for mode,name in enumerate(('external','attached','released'))},
            ambiguous_mode_rows=int((~arrays['mode_valid'][:,j]).sum()),
            public_non_exogenous_masked_rows=int((~arrays['dynamics_segment_valid'][:,j]).sum()),
            native_pose_valid_rows=int(arrays['native_pose_valid'][:,j].sum()),
            target_valid_rows=int(arrays['target_valid'][:,j].sum()),
            rotation_ambiguous_rows=int(ambiguous.sum()),
            rotation_ambiguity_flag_missing_rows=sum('rotation_ambiguous' not in row for row in diagnostic),
            configured_orientation_observability=dict(Counter(str(row.get('orientation_observability','missing')) for row in diagnostic)),
            registration_rms_m=scalar_stats([row.get('registration_rms_m',np.nan) for row in diagnostic]),
            registration_rms_missing_rows=sum('registration_rms_m' not in row for row in diagnostic),
            eligible_windows=eligibility['by_body'][body],native_twist=native_twist_audit(arrays,j,thresholds))
    press = arrays['public_command_kind'].astype(str) == 'press'
    press_checks = dict(issued_press_rows=int(press.sum()),plate_present='plate' in ids,passed=True,
        expected_rule='issued press latches plate non-exogenous through all subsequent rows; raw native_pose_valid is preserved')
    if 'plate' in ids:
        p = ids.index('plate');masked = np.maximum.accumulate(press)
        bad = np.flatnonzero(masked & (arrays['dynamics_segment_valid'][:,p] | arrays['valid'][:,p] | arrays['target_valid'][:,p]))
        press_checks.update(first_press_time=float(times[np.flatnonzero(press)[0]]) if press.any() else None,
            rows_after_first_press=int(masked.sum()),violating_rows=bad.tolist(),passed=not len(bad),
            raw_native_pose_valid_after_press=int(arrays['native_pose_valid'][masked,p].sum()))
        require(not len(bad), 'plate public press mask was not latched')
    layouts = [artifact_path(name) for name in files if Path(name).name == 'TRAIN_rigid_layout.json']
    require(len(layouts) == 1, 'one SHA-bound TRAIN rigid-layout audit required')
    rigid = rigid_layout_audit(arrays,spec,read(layouts[0]))
    terminal = alignment.get('native_terminal_latched') is True
    short = terminal and len(times) < 27
    return dict(case_id=case['case_id'],task=case['task'],raw_episode_id=case['raw_episode_id'],
        parent_source_group=case['parent_source_group'],status='verified',provenance=inputs,
        process_id=record['process_id'],frames=len(times),ticks=clock['ticks'],seconds=float(times[-1]),
        native_terminal_latched=terminal,native_success_latched=alignment.get('native_success_latched'),
        native_terminal_failure_latched=alignment.get('native_terminal_failure_latched'),
        short_terminal=short,immediate_terminal=terminal and len(times) == 1,
        short_terminal_is_infrastructure_failure=False,automatic_resampling_recommended=False,
        context_window_eligibility=eligibility,teacher=phase,bodies=bodies,public_press_mask=press_checks,
        rigid_layout=rigid,registration_group_frames=sum(bool(frame.get('rigid_groups')) for frame in frames),
        selection_by_native_success=False,labels_modified=False)


def analyze(plan_path,queue_path,thresholds=None):
    thresholds = thresholds or dict(linear_motion_m_per_s=.001,angular_motion_rad_per_s=.01,
        reported_zero_m_per_s=1e-6,reported_zero_rad_per_s=1e-6)
    plan_path,queue_path = absolute(plan_path),artifact_path(queue_path)
    plan = read(plan_path)
    require(plan.get('version') == 'r10_source_grouped_native_augmentation_plan_v1'
        and plan.get('plan_sha256') == digest({k:v for k,v in plan.items() if k != 'plan_sha256'}),
        'collection plan digest/version is invalid')
    cases = plan.get('cases',[])
    require(plan.get('split') == 'train' and len(cases) == 14
        and {case['task'] for case in cases} == set(TASK_SPECS)
        and len({case['case_id'] for case in cases}) == 14
        and all(case['split'] == 'train' for case in cases), 'smoke analysis expects one real TRAIN case per remaining task')
    results,missing,invalid = [],[],[]
    for case in cases:
        path = queue_path/'cases'/case['split']/case['task']/case['case_id']/'case.json'
        if not path.is_file():
            missing.append(dict(case_id=case['case_id'],task=case['task'],path=str(path),status='missing'))
            continue
        try:
            results.append(audit_case(path,plan,case,thresholds))
        except Exception as error:
            invalid.append(dict(case_id=case['case_id'],task=case['task'],path=str(path),
                case_sha256=sha(path),status='invalid',error=type(error).__name__+': '+str(error)))
    provenance = [pointer(plan_path),pointer(Path(__file__)),pointer(ROOT/'scripts/collect_multitask_training_r10.py'),
                  pointer(ROOT/'our_mani/multitask/specs.py')]
    queue_report_path = queue_path/'queue_report.json'
    queue_report = read(queue_report_path) if queue_report_path.is_file() else None
    if queue_report is not None:
        provenance.append(pointer(queue_report_path))
    manifest_path = queue_path/'manifest.json'
    if manifest_path.is_file():
        provenance.append(pointer(manifest_path))
        manifest = read(manifest_path)
        if manifest.get('plan_sha256') != plan['plan_sha256']:
            invalid.append(dict(status='invalid',error='queue manifest refers to a different plan'))
        actual = {row['case_id']:row for row in manifest.get('case_receipts',[])}
        for case in results:
            source = case['provenance'][0]
            row = actual.get(case['case_id'])
            if row is None or absolute(row['path']) != Path(source['path']) or row['sha256'] != source['sha256']:
                invalid.append(dict(case_id=case['case_id'],task=case['task'],status='invalid',
                                    error='queue merged manifest case path/SHA differs'))
    short = [dict(case_id=row['case_id'],task=row['task'],frames=row['frames'],
        seconds=row['seconds'],immediate_terminal=row['immediate_terminal'],
        native_success_latched=row['native_success_latched'],teacher_action_rows=row['teacher']['teacher_action_rows'],
        eligible_windows=row['context_window_eligibility']['usable_context_windows']) for row in results if row['short_terminal']]
    complete = len(results) == len(cases) and not missing and not invalid
    frozen_sources = []
    for name,expected in plan.get('inputs',{}).items():
        path=absolute(name)
        if path.suffix in ('.py','.sh'):
            actual=sha(path) if path.is_file() else None
            frozen_sources.append(dict(path=str(path),frozen_sha256=expected,current_sha256=actual,
                                       current_matches_collection_plan=actual == expected))
    return dict(version=VERSION,created_utc=datetime.now(timezone.utc).isoformat(),
        expected_cases=14,verified_cases=len(results),missing_cases=len(missing),invalid_cases=len(invalid),
        complete=complete,diagnostics_valid=not invalid,
        status='complete' if complete else 'invalid' if invalid else 'incomplete' if results else 'not_started_or_no_completed_cases',
        queue_path=str(queue_path),queue_directory_exists=queue_path.is_dir(),plan_sha256=plan['plan_sha256'],
        provenance=provenance,frozen_collection_inputs=plan.get('inputs',{}),
        current_source_receipts=frozen_sources,
        current_sources_match_collection_plan=all(row['current_matches_collection_plan'] for row in frozen_sources),
        missing=missing,invalid=invalid,cases=results,short_terminal_cases=short,
        cases_with_usable_context_windows=sum(row['context_window_eligibility']['usable_context_windows'] > 0 for row in results),
        cases_with_full_horizon_windows=sum(row['context_window_eligibility']['full_horizon_windows'] > 0 for row in results),
        cases_missing_required_teacher_phases=sum(bool(row['teacher']['required_phases_missing']) for row in results),
        queue_infrastructure_receipts=[] if queue_report is None else [row for row in queue_report.get('receipts',[])
            if row.get('status') not in ('completed','resumed_verified')],
        limitations=['One TRAIN source per task is a smoke diagnostic, not full source coverage or generalization evidence.',
            'Native task success never filters acquisition or this analysis; infrastructure errors are not SR negatives.',
            'All native pose/twist comparisons are offline labels/metrics; no GT is written into history or forecasts.',
            'Configured none/axis/ambiguous rotations are not promoted to full orientation by native-label availability.',
            'Thresholds here only describe motion/zero counts; they do not select model/control thresholds.',
            'Short terminal cases are preserved without teacher after terminal, padding, or automatic infrastructure retry.'],
        diagnostic_thresholds=thresholds,model_fitted=False,native_or_GPU_started_by_analysis=False,
        source_files_or_labels_modified=False,TEST_read=False)


def report_text(report):
    lines=['R10 TRAIN smoke 离线数据诊断',
        f"状态：{report['status']}；SHA验收 {report['verified_cases']}/14；缺失 {report['missing_cases']}；错误 {report['invalid_cases']}。",
        '本脚本只读既有 TRAIN 产物；未启动仿真/GPU，未训练、改写标签或调整阈值。',
        '当前没有完成的 case 时，不产生任何真实效果或速度一致性结论。' if not report['cases'] else
            '所有完成 case 均按原样分析；native success 不用于筛选数据。',
        f"具有可用 context 窗口：{report['cases_with_usable_context_windows']}；具有完整1.1s窗口：{report['cases_with_full_horizon_windows']}。",
        f"缺少必需 teacher phase 的 case：{report['cases_missing_required_teacher_phases']}；真实短 terminal：{len(report['short_terminal_cases'])}。",
        '']
    for case in report['cases']:
        q=case['context_window_eligibility']
        lines += [f"{case['task']}：{case['frames']}帧/{case['seconds']:.2f}s；窗口 {q['usable_context_windows']}（完整 {q['full_horizon_windows']}）；短terminal={case['short_terminal']}。",
            '  实际阶段：'+json.dumps(case['teacher']['phases'],ensure_ascii=False)+'；缺失必需阶段：'+str(case['teacher']['required_phases_missing'])]
        for body,row in case['bodies'].items():
            v=row['native_twist']['segmented']['linear_m_per_s'];w=row['native_twist']['segmented']['angular_rad_per_s']
            lines.append(f"  {body}：直接可见 {row['direct_visible']['frames']}/{case['frames']}；有效最长 {row['history_valid_same_mode']['longest_run_frames']}帧；完整朝向 {row['full_orientation_history']['frames']}；原生速度差分RMS v={v['vector_rms_error']}，w={w['vector_rms_error']}；运动但报0 v={v['pose_motion_but_reported_zero_pairs']}，w={w['pose_motion_but_reported_zero_pairs']}。")
        for role in case['teacher']['role_availability_by_phase']:
            if role['requires_full_reference_rotation']:
                lines.append(f"  完整SE3阶段 {role['phase']} / {role['reference_role']}→{role['actual_TRAIN_supervision_body']}：可用 {role['required_reference_available_rows']}/{role['action_rows']}。")
    if report['missing']:
        lines += ['', '缺失任务：'+', '.join(row['task'] for row in report['missing'])]
    if report['invalid']:
        lines += ['', '数据错误：']+[json.dumps(row,ensure_ascii=False) for row in report['invalid']]
    lines += ['', '速度比较使用相邻50ms原生位姿差分、world SO(3)增量和两端reported速度均值；分段子集不跨公开mode、按压mask或实际teacher stage/phase边界。',
        '未知/对称朝向仍按unknown/axis/none处理。完整SE3可用率只检查感知与记录，不代表规划可达、接触安全或任务收益。',
        '一例每任务仅用于smoke排错；短terminal单列保留，不伪装基础设施失败。']
    return '\n'.join(lines)+'\n'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',default=str(BASE/'TRAIN_smoke_collection_plan_v3.json'))
    parser.add_argument('--queue',required=True,help='collector queue output root, not the policy evaluation queue')
    parser.add_argument('--out',default=str(BASE/'TRAIN_smoke_analysis_v3.json'))
    args=parser.parse_args()
    report=analyze(args.plan,args.queue)
    out=artifact_path(args.out);out.parent.mkdir(parents=True,exist_ok=True)
    require(out.suffix == '.json','analysis output must have a .json suffix')
    input_paths={absolute(args.plan),absolute(args.queue)/'manifest.json',absolute(args.queue)/'queue_report.json'}
    input_paths.update(absolute(path) for path in report['frozen_collection_inputs'])
    for case in report['cases']:
        input_paths.update(Path(row['path']) for row in case['provenance'])
    input_paths.update(Path(row['path']) for row in report['missing']+report['invalid'] if 'path' in row)
    require(out not in input_paths and out.with_suffix('.txt') not in input_paths,
            'read-only analysis cannot overwrite source/plan/case evidence')
    if out.exists():
        require(read(out).get('version') == VERSION,'refusing to overwrite a different artifact')
    for path,contents in ((out,json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n'),
                          (out.with_suffix('.txt'),report_text(report))):
        temporary=path.with_suffix(path.suffix+'.tmp');temporary.write_text(contents);temporary.replace(path)
    print(json.dumps({key:report[key] for key in ('status','expected_cases','verified_cases','missing_cases','invalid_cases','complete')},ensure_ascii=False))
    if report['invalid']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
