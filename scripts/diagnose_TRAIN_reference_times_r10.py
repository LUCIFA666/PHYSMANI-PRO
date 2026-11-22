"""Read-only actual700 TRAIN endpoint/reference-time diagnosis.

Hypothetical issue/world coordinate comparisons are endpoint diagnostics only.
They do not change typed sources, rules, production fitting, raw labels or gates.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from our_mani.contracts import Pose
from our_mani.multitask import acquisition_lineage as a
from our_mani.multitask import artifact_fitting, interaction
from our_mani.multitask.training_view import validate_arrays


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def stats(values):
    values = np.asarray(values, float)
    if not len(values):
        return dict(count=0)
    a.require(values.ndim == 1 and np.isfinite(values).all(), 'invalid diagnostic metrics')
    return dict(count=len(values), minimum=float(values.min()), median=float(np.median(values)),
        mean=float(values.mean()), quantile90=float(np.quantile(values, .9)),
        quantile95=float(np.quantile(values, .95)), quantile99=float(np.quantile(values, .99)),
        maximum=float(values.max()))


def endpoint_crossfit(rows, configuration):
    """Production parent folds and exact unchanged terminal-endpoint solver."""
    parents = sorted({row['source'] for row in rows})
    result = dict(parent_count=len(parents), event_count=len(rows),
        minimum_completed_parents=configuration.minimum_completed_parent_sources,
        maximum_position_tolerance_m=configuration.maximum_position_tolerance_m,
        maximum_orientation_tolerance_rad=configuration.maximum_orientation_tolerance_rad,
        original_parent_grouping_unchanged=True, full_curve_lag_selection_or_library_fit=False)
    if len(parents) < 2:
        return dict(result, status='insufficient_independent_parents_for_crossfit',
            passes_existing_endpoint_caps=False)
    folds = {parent: index % 2 for index, parent in enumerate(parents)}
    residuals = []
    means = {}
    for fold in (0, 1):
        fit = [row for row in rows if folds[row['source']] != fold]
        held = [row for row in rows if folds[row['source']] == fold]
        p, r, _ = interaction._fit_completed_curve(fit, np.linspace(0, 1, configuration.progress_knots))
        means[str(fold)] = dict(position=p[-1].tolist(), rotation_xyzw=Rotation.from_matrix(r[-1]).as_quat().tolist())
        for row in held:
            residuals.append(dict(parent=row['source'], case_id=row['case_id'], command_id=row['command_id'], fold=fold,
                position_error_m=float(np.linalg.norm(p[-1]-row['position'])),
                orientation_error_rad=float(Rotation.from_matrix(r[-1]@row['rotation'].T).magnitude())))
    per_parent = {parent: dict(position=float(np.median([r['position_error_m'] for r in residuals if r['parent'] == parent])),
        orientation=float(np.median([r['orientation_error_rad'] for r in residuals if r['parent'] == parent]))) for parent in parents}
    p = max(configuration.minimum_position_tolerance_m,
        float(np.quantile([r['position'] for r in per_parent.values()], configuration.tolerance_quantile)))
    angle = max(configuration.minimum_orientation_tolerance_rad,
        float(np.quantile([r['orientation'] for r in per_parent.values()], configuration.tolerance_quantile)))
    return dict(result, status='actual_parent_endpoint_crossfit',
        source_to_fold=folds, fit_terminal_means=means,
        position_tolerance_m=p, orientation_tolerance_rad=angle,
        position_parent_residual_distribution_m=stats([r['position'] for r in per_parent.values()]),
        orientation_parent_residual_distribution_rad=stats([r['orientation'] for r in per_parent.values()]),
        parents_position_residual_above_cap=sum(r['position'] > configuration.maximum_position_tolerance_m for r in per_parent.values()),
        parents_orientation_residual_above_cap=sum(r['orientation'] > configuration.maximum_orientation_tolerance_rad for r in per_parent.values()),
        passes_existing_endpoint_caps=(len(parents) >= max(configuration.minimum_sources,configuration.minimum_completed_parent_sources)
            and p <= configuration.maximum_position_tolerance_m and angle <= configuration.maximum_orientation_tolerance_rad),
        parent_terminal_residuals=per_parent, residuals=residuals,
        largest_position_residual_parents=sorted((dict(parent=parent,**value) for parent,value in per_parent.items()),
            key=lambda row:(-row['position'],row['parent']))[:10],
        calls_unchanged_production_completed_endpoint_solver=True)


def relative(reference, target):
    return dict(position=reference.rotation.T@(target.position-reference.position),
                rotation=reference.rotation.T@target.rotation)


def pose_error(first, second):
    return dict(position_m=float(np.linalg.norm(first.position-second.position)),
        orientation_rad=float(Rotation.from_matrix(first.rotation@second.rotation.T).magnitude()),
        position_vector_world_m=(first.position-second.position).tolist(),
        rotation_vector_world_rad=Rotation.from_matrix(first.rotation@second.rotation.T).as_rotvec().tolist())


def frame_at(times, value):
    found = np.flatnonzero(np.isclose(times, float(value), atol=1e-8, rtol=0))
    a.require(len(found) == 1, 'original command/event clock has no unique actual frame')
    return int(found[0])


def mode_timeline(arrays, body, commands):
    j = tuple(map(str, arrays['body_ids'])).index(body)
    n = len(arrays['times'])
    selected = np.zeros(n, bool)
    selected[0] = True
    for values in (arrays['modes'][:,j], arrays['mode_valid'][:,j], arrays['robot_contact'],
                   arrays['teacher_command_id'], arrays['teacher_applied_opening']):
        selected[1:] |= values[1:] != values[:-1]
    return [dict(frame=int(i), time=float(arrays['times'][i]),
        public_mode=('external','attached','released')[int(arrays['modes'][i,j])],
        public_mode_valid=bool(arrays['mode_valid'][i,j]), public_contact=bool(arrays['robot_contact'][i]),
        gripper_open_proxy=bool(arrays['robot_gripper_open'][i]),
        issued_opening=float(arrays['teacher_applied_opening'][i]),
        command_id=int(arrays['teacher_command_id'][i]), teacher_phase=str(arrays['teacher_phase'][i]),
        mode_is_causal_public_association_not_native_held_ID=True) for i in np.flatnonzero(selected)]


def run(prior_path, output):
    begun = time.perf_counter()
    started = datetime.now(timezone.utc).isoformat()
    output = a.local(ROOT, output)
    a.require(not output.exists(), 'fresh endpoint diagnosis output required')
    output.mkdir(parents=True)
    files = {}
    prior_pointer = a.pointer(ROOT, a.local(ROOT, prior_path))
    prior = a.read(a.checked_pointer(ROOT, prior_pointer, files))
    a.require(prior['schema'] == 'r10_complete_original100_teacher_tasks_CPU_fit_snapshot_v1'
        and prior['actual_new_task_fits'] == 7 and prior['TRAIN_only'] is True
        and prior['VAL_or_TEST_used'] is False and prior['thresholds_or_parent_support_relaxed'] is False,
        'strict existing complete-task700 TRAIN diagnostic required')
    files.update(prior['verified_files_sha256'])
    # Re-read immutable bytes used by the actual previous700-case full audit.
    # This does not inspect any later completion or change that fixed snapshot.
    for path, checksum in files.items():
        a.require(a.file_sha(path) == checksum, 'prior full raw/source audit input changed: '+path)
    configuration = a.read(a.checked_pointer(ROOT, prior['configuration'], files))
    config = interaction.InteractionFitConfig(**configuration['interaction'])
    a.require(config.minimum_completed_parent_sources == 8 and config.maximum_position_tolerance_m == .03
        and config.maximum_orientation_tolerance_rad == .15 and config.development_only is False,
        'original8/.03/.15 requirements changed')
    sources = {str(Path(module.__file__).resolve()):a.file_sha(module.__file__)
        for module in (artifact_fitting, interaction, a)}
    sources[str(Path(__file__).resolve())] = a.file_sha(__file__)
    for name in ('mode_observer.py','pipeline.py','train_teacher.py','specs.py'):
        p=ROOT/'our_mani/multitask'/name;sources[str(p)]=a.file_sha(p)
    files.update(sources)
    tasks = {}
    for task, previous in prior['tasks'].items():
        if previous.get('real_fit_performed') is not True:
            continue
        a.require(previous['verified_original_teacher_cases'] == previous['verified_distinct_original_parents'] == 100
            and not previous['raw_or_adapter_errors'], 'complete actual100 raw audit required per target task')
        manifest = a.read(a.checked_pointer(ROOT, previous['fit_manifest'], files))
        fitted = a.read(a.checked_pointer(ROOT, previous['detailed_fit_report'], files))['completed_skill_fitting']
        admitted = {(row['source_id'],row['command_id']) for row in fitted['curation']
                    if row['included_in_BC'] is True}
        comparisons = {}
        events_all = []
        cases_all = []
        raw_parent_seen = set()
        for proof in manifest['case_proofs']:
            path = a.checked_pointer(ROOT, proof['canonical_receipt'], files)
            record = a.read(path)
            entry = record['entry']
            command_path = a.checked_pointer(ROOT, proof['command_receipt'], files)
            cache = a.checked_pointer(ROOT, proof['cache'], files)
            with np.load(cache, allow_pickle=False) as stored:
                arrays = {key:stored[key].copy() for key in stored.files}
            validate_arrays(arrays, entry, record)
            command_receipt = a.read(command_path)
            case = dict(path=path, record=record, entry=entry, arrays=arrays,
                command_receipt=command_receipt, command_receipt_verified=True,
                command_receipt_sha256=proof['command_receipt']['sha256'])
            source, audit = artifact_fitting.interaction_source(case)
            a.require(source.grouping_source == proof['parent_source_group'] and source.grouping_source not in raw_parent_seen,
                      'original parent is missing or duplicated')
            raw_parent_seen.add(source.grouping_source)
            ledger = {row['command_id']:row for row in source.command_ledger}
            raw_events = command_receipt['phase_completion_events']
            driver = record['native_reset_receipt'].get('native_initial_snapshot',{}).get('active_motion_schedule',{})
            event_ids = {row.origin_command_id for row in source.completion_events}
            raw_events_by_id = {row['origin_command_id']:row for row in raw_events}
            case_row = dict(case_id=proof['case_id'],parent=source.grouping_source,source_id=source.source_id,
                canonical_receipt=proof['canonical_receipt'],command_receipt=proof['command_receipt'],cache=proof['cache'],
                actual_plan_sha256=proof['actual_plan_sha256'],original_commands=command_receipt['commands'],
                original_completion_events=raw_events,raw_completion_count=len(raw_events),
                typed_valid_completion_count=len(source.completion_events),
                excluded_original_completion_events=list(source.excluded_completion_events),
                original_zero_action_or_failed_source_preserved=True,
                commands_without_typed_completion=[dict(command_id=cid,raw_phase=row['raw_phase'],raw_stage=row['raw_stage'],
                    waypoint_completed='command_completed_at' in row,
                    original_completion_event_present=cid in raw_events_by_id,
                    right_censored=cid not in raw_events_by_id,
                    generated_at=row['generated_at'],cutoff_at=row.get('cutoff_at'),cutoff_reason=row.get('cutoff_reason'))
                    for cid,row in ledger.items() if cid not in event_ids],
                native_initial_motion_schedule_metric_only=driver,adapter_audit=audit)
            cases_all.append(case_row)
            for event in source.completion_events:
                command = ledger[event.origin_command_id]
                binding = event.binding
                key = '|'.join((event.phase,binding.anchor_id,binding.reference_role,binding.reference_kind,
                                binding.manipulated_role,event.reference_mode,binding.reference_time_basis))
                start = frame_at(source.times,event.attempt_start_time)
                end = frame_at(source.times,event.attempt_end_time)
                terminal = Pose.from_xyzw(event.terminal_tcp_xyzw)
                target = Pose.from_xyzw(event.committed_world_waypoint_xyzw)
                current = Pose.from_xyzw(event.terminal_reference_xyzw)
                issue = None
                issue_invalid = None
                if binding.reference_kind == 'world':
                    issue = Pose(np.zeros(3))
                elif binding.reference_kind == 'phase_origin':
                    issue = Pose.from_xyzw(event.origin_measured_tcp_xyzw)
                else:
                    j = tuple(map(str,arrays['body_ids'])).index(binding.reference_id)
                    raw_issue = Pose.from_xyzw(command['issued_reference_native_pose_xyzw'])
                    stored = Pose(arrays['target_positions'][start,j],arrays['target_rotations'][start,j])
                    a.require(pose_error(raw_issue,stored)['position_m'] <= 1e-8 and
                        pose_error(raw_issue,stored)['orientation_rad'] <= 1e-7,
                        'original issued reference disagrees with acquired native label')
                    if not arrays['native_pose_valid'][start,j] or not arrays['mode_valid'][start,j]:
                        issue_invalid = 'original issue native pose or mode mask invalid'
                    elif binding.reference_kind == 'observed_se3' and not arrays['target_rotation_observable'][start,j]:
                        issue_invalid = 'original full-rotation issue label unavailable'
                    else:
                        issue = Pose(raw_issue.position,np.eye(3)) if binding.reference_kind == 'translation_only' else raw_issue
                admission = (source.source_id,event.origin_command_id) in admitted
                event_row = dict(case_id=proof['case_id'],parent=source.grouping_source,source_id=source.source_id,
                    command_id=event.origin_command_id,partition=key,raw_stage=command['raw_stage'],
                    reference_body=binding.reference_id,reference_mode=event.reference_mode,
                    start_frame=start,terminal_frame=end,duration_seconds=event.attempt_end_time-event.attempt_start_time,
                    originally_included_in_production_BC=admission,issue_reference_unavailable_reason=issue_invalid,
                    original_native_motion_schedule_metric_only=driver,
                    actual_terminal_TCP_xyzw=terminal.xyzw().tolist(),original_committed_world_waypoint_xyzw=target.xyzw().tolist(),
                    current_native_reference_xyzw=current.xyzw().tolist(),
                    issued_native_reference_xyzw=None if issue is None else issue.xyzw().tolist(),
                    actual_terminal_error_to_original_world_waypoint=pose_error(terminal,target))
                if command.get('measured_arrival_at') is not None:
                    arrival = Pose.from_xyzw(arrays['measured_tcp_xyzw'][frame_at(source.times,command['measured_arrival_at'])])
                    arrival_error = pose_error(arrival,target)
                    a.require(abs(arrival_error['position_m']-command['arrival_position_error_m']) < 1e-8 and
                        abs(arrival_error['orientation_rad']-command['arrival_orientation_error_rad']) < 1e-7,
                        'raw measured arrival residual disagrees with actual array frame')
                    event_row['actual_original_measured_arrival_error_to_world_waypoint'] = arrival_error
                options = dict(current_native_reference=relative(current,terminal),
                    terminal_world=relative(Pose(np.zeros(3)),terminal),
                    committed_waypoint_world=relative(Pose(np.zeros(3)),target),
                    committed_waypoint_relative_current=relative(current,target),
                    actual_error_relative_committed_world_waypoint=relative(target,terminal))
                if issue is not None:
                    options.update(issued_native_reference=relative(issue,terminal),
                        committed_waypoint_relative_issue=relative(issue,target))
                    event_row['actual_issue_to_terminal_reference_motion'] = pose_error(current,issue)
                original_fit = fitted['phase_fits'].get(key)
                if original_fit and 'selected_lag' in original_fit:
                    chosen = interaction._completion_reference(source,event,float(original_fit['selected_lag']))
                    options['production_selected_lag_reference'] = relative(chosen,terminal)
                if event.reference_mode == 'released' and binding.reference_id is not None:
                    event_row['complete_public_contact_mode_command_timeline'] = mode_timeline(arrays,binding.reference_id,ledger)
                event_row['endpoint_diagnostic_coordinates'] = {name:dict(position=value['position'].tolist(),
                    rotation_xyzw=Rotation.from_matrix(value['rotation']).as_quat().tolist()) for name,value in options.items()}
                events_all.append(event_row)
                if admission:
                    for name,value in options.items():
                        comparisons.setdefault(key,{}).setdefault(name,[]).append(dict(source=source.grouping_source,
                            case_id=proof['case_id'],command_id=event.origin_command_id,progress=1.,terminal_event=True,
                            opening=float(source.expert_opening[max(start,end-1)]),**value))
        a.require(len(cases_all) == len(raw_parent_seen) == 100,'all100 original source parents must remain retained')
        partitions = {}
        for key, variants in comparisons.items():
            original_fit = fitted['phase_fits'][key]
            endpoint = {name:endpoint_crossfit(rows,config) for name,rows in variants.items()}
            expected_events = {(row['case_id'],row['command_id']) for row in variants['current_native_reference']}
            for name,result in endpoint.items():
                result['covers_every_originally_included_completed_event'] = (
                    {(row['case_id'],row['command_id']) for row in variants[name]} == expected_events)
                result['is_diagnostic_hypothesis_not_new_reference_rule'] = True
            if 'production_selected_lag_reference' in endpoint:
                measured = endpoint['production_selected_lag_reference']
                a.require(abs(measured['position_tolerance_m']-original_fit['position_tolerance_m']) < 1e-12
                    and abs(measured['orientation_tolerance_rad']-original_fit['orientation_tolerance_rad']) < 1e-12,
                    'actual endpoint crossfit does not reproduce original production selected-lag residual')
            matching = [row for row in events_all if row['partition'] == key and row['originally_included_in_production_BC']]
            direct = [row['actual_terminal_error_to_original_world_waypoint'] for row in matching]
            motion = [row['actual_issue_to_terminal_reference_motion'] for row in matching
                      if 'actual_issue_to_terminal_reference_motion' in row]
            subgroups = {}
            for field in ('reference_body','raw_stage'):
                for value in sorted({str(row[field]) for row in matching}):
                    ids={(row['case_id'],row['command_id']) for row in matching if str(row[field])==value}
                    subgroups[field+'='+value]={name:endpoint_crossfit([row for row in rows
                        if (row['case_id'],row['command_id']) in ids],config) for name,rows in variants.items()
                        if name in ('current_native_reference','issued_native_reference','terminal_world')}
            partitions[key] = dict(original_actual_production_phase_fit=original_fit,
                original_completed_events=len(matching),endpoint_reference_comparisons=endpoint,
                observed_reference_body_counts=dict(Counter(row['reference_body'] for row in matching)),
                observed_raw_stage_counts=dict(Counter(row['raw_stage'] for row in matching)),
                duration_distribution_seconds=stats([row['duration_seconds'] for row in matching]),
                actual_terminal_to_committed_waypoint_error=dict(position_m=stats([row['position_m'] for row in direct]),
                    orientation_rad=stats([row['orientation_rad'] for row in direct]),
                    mean_position_error_vector_world_m=np.mean([row['position_vector_world_m'] for row in direct],axis=0).tolist(),
                    mean_rotation_error_vector_world_rad=np.mean([row['rotation_vector_world_rad'] for row in direct],axis=0).tolist()),
                actual_issue_to_terminal_reference_motion=dict(position_m=stats([row['position_m'] for row in motion]),
                    orientation_rad=stats([row['orientation_rad'] for row in motion])),
                descriptive_subgroup_crossfits_not_parent_filtering_or_task_release=subgroups)
        details=output/task/'endpoint_reference_details.json'
        task_result=dict(task=task,original_teacher_cases=100,original_distinct_parents=100,
            all_original_sources_commands_events_censor_and_masks_retained=True,
            partitions=partitions,per_source=cases_all,per_completed_event=events_all,
            original_production_required_phase_failures=fitted.get('required_phase_failures',[]),
            original_production_failed_fit_keys=fitted.get('failed_fit_keys',[]),
            no_hypothetical_library_fitted_or_exported=True)
        write(details,task_result)
        tasks[task]=dict(actual_sources=100,original_distinct_parents=100,
            complete_raw_completion_events=sum(row['raw_completion_count'] for row in cases_all),
            typed_valid_completion_events=len(events_all),details=a.pointer(ROOT,details),
            partitions={key:dict(original_passed=value['original_actual_production_phase_fit'].get('passed'),
                original_selected_lag=value['original_actual_production_phase_fit'].get('selected_lag'),
                actual_terminal_to_committed_waypoint_error=value['actual_terminal_to_committed_waypoint_error'],
                reference_motion=value['actual_issue_to_terminal_reference_motion'],
                comparisons={name:{field:comparison.get(field) for field in ('parent_count','event_count',
                    'position_tolerance_m','orientation_tolerance_rad','passes_existing_endpoint_caps',
                    'parents_position_residual_above_cap','parents_orientation_residual_above_cap',
                    'covers_every_originally_included_completed_event')} for name,comparison
                    in value['endpoint_reference_comparisons'].items()}) for key,value in partitions.items()})
        write(output/'progress.json',dict(tasks=tasks,complete=False,native_or_GPU_started=False))
    a.require(len(tasks) == 7,'exact seven previously wholly verified task100 datasets required')
    for path,checksum in files.items():a.require(a.file_sha(path)==checksum,'diagnostic input bytes changed: '+path)
    report=dict(schema='r10_actual700_TRAIN_physical_reference_time_endpoint_diagnosis_v1',
        started_utc=started,finished_utc=datetime.now(timezone.utc).isoformat(),prior_strict700_fit_report=prior_pointer,
        tasks=tasks,original_teacher_cases=700,all_original_sources_commands_events_masks_and_censoring_retained=True,
        actual_selected_lag_endpoint_math_reproduced_to_1e_minus12=True,
        parent_group_crossfit_used=True,minimum_completed_parent_sources=8,maximum_position_tolerance_m=.03,
        maximum_orientation_tolerance_rad=.15,thresholds_rules_or_production_source_modified=False,
        TRAIN_only=True,VAL_TEST_or_auxiliary_read=False,native_or_GPU_started=False,formal_model_or_policy_release=False,
        hypothetical_issue_SO3_comparison_does_not_authorize_existing_translation_only_reference_time_schema=True,
        rare_released_partitions_preserved=True,source_sha256=sources,verified_files_sha256=files,
        wall_seconds=time.perf_counter()-begun)
    write(output/'report.json',report)
    write(output/'progress.json',dict(tasks=tasks,complete=True,native_or_GPU_started=False))
    return dict(report=a.pointer(ROOT,output/'report.json'),actual_original_teacher_cases=700,
        tasks={task:dict(details=value['details'],failed_partitions={key:row['comparisons'] for key,row
            in value['partitions'].items() if row['original_passed'] is not True}) for task,value in tasks.items()},
        production_source_or_rule_changed=False,native_or_GPU_started=False)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prior-report',required=True)
    parser.add_argument('--out',required=True)
    args=parser.parse_args()
    print(json.dumps(run(args.prior_report,args.out),ensure_ascii=False))
