"""CPU-only source-declared forecast and rare-mode analysis of audited700.

No parameter search, raw/source edits, parent filtering or model release.
The forecast comparison uses privileged TRAIN driver labels for diagnosis only.
"""
import argparse
from collections import Counter
from datetime import datetime,timezone
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[1]
for p in (ROOT,ROOT/'scripts'):
    sys.path.insert(0,str(p))
from our_mani.contracts import Pose
from our_mani.multitask import acquisition_lineage as a
from our_mani.multitask.interaction import InteractionFitConfig
from diagnose_TRAIN_reference_times_r10 import endpoint_crossfit,relative,pose_error,stats


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def run(report_path,out):
    out=a.local(ROOT,out);a.require(not out.exists(),'fresh interpretation namespace required');out.mkdir()
    files={};p=a.local(ROOT,report_path);report=a.read(a.checked_pointer(ROOT,a.pointer(ROOT,p),files))
    a.require(report['original_teacher_cases']==700 and report['actual_selected_lag_endpoint_math_reproduced_to_1e_minus12'] is True,
        'strict actual700 endpoint comparison required')
    prior=a.read(a.checked_pointer(ROOT,report['prior_strict700_fit_report'],files))
    config=InteractionFitConfig(**a.read(a.checked_pointer(ROOT,prior['configuration'],files))['interaction'])
    registrations=a.read(a.checked_pointer(ROOT,prior['actual_registrations'],files))
    native_sources={}
    wanted=set(report['tasks'])|{'place_cups_on_rotating_frame'}
    for registration in registrations:
        snapshot=a.read(a.checked_pointer(ROOT,registration['source_snapshot'],files))
        for entry in snapshot['entries']:
            original=Path(entry['original_path'])
            if (original.parent.name=='tasks' and original.stem in wanted) or str(original).endswith('/pyrep/objects/object.py'):
                pointer={k:entry[k] for k in ('path','sha256')}
                a.checked_pointer(ROOT,pointer,files)
                if str(original) in native_sources:
                    a.require(native_sources[str(original)]['source']['sha256']==pointer['sha256'],
                        'native expert/rotation source differs between actual archived profiles')
                native_sources[str(original)]=dict(source=pointer,actual_source_snapshot=registration['source_snapshot'],
                    original_v11_execution_source_bound=any(row['original_path']==str(original)
                        for row in a.read(a.local(ROOT,registrations[0]['source_snapshot']['path']))['entries']))
    a.require(len(native_sources)>=8,'archived native teacher tasks/rotation helper/Object.rotate must be bound')
    forecast={};rare={};bin_rows=[]
    for task,summary in report['tasks'].items():
        detail=a.read(a.checked_pointer(ROOT,summary['details'],files))
        case_map={row['case_id']:row for row in detail['per_source']}
        rows_by_key={};rare_by_case={}
        for event in detail['per_completed_event']:
            key=event['partition'];phase=key.split('|')[0]
            if not event['originally_included_in_production_BC']:continue
            # These exact original expert branches use compute_target_pose(.5).
            source_delay_applies=(task.startswith('beat_the_rotating_buzz') and phase in ('acquire','follow','release')) or (
                task.startswith('insert_onto_rotating_peg') and phase in ('align','insert','release'))
            if source_delay_applies and event['issued_native_reference_xyzw'] is not None:
                case=case_map[event['case_id']];cache=a.checked_pointer(ROOT,case['cache'],files)
                with np.load(cache,allow_pickle=False) as stored:
                    ids=tuple(map(str,stored['body_ids']));j=ids.index('frame');i=event['start_frame']
                    center=stored['target_positions'][i,j].copy();frame_rotation=stored['target_rotations'][i,j].copy()
                    valid=bool(stored['native_pose_valid'][i,j])
                a.require(valid,'source-declared forecast diagnostic needs valid original frame label')
                yaw=float(event['original_native_motion_schedule_metric_only']['yaw_speed'])
                delta=Rotation.from_rotvec(frame_rotation[:,2]*np.deg2rad(yaw)*.5).as_matrix()
                issue=Pose.from_xyzw(event['issued_native_reference_xyzw'])
                predicted=Pose(center+delta@(issue.position-center),delta@issue.rotation)
                terminal=Pose.from_xyzw(event['actual_terminal_TCP_xyzw'])
                goal=Pose.from_xyzw(event['original_committed_world_waypoint_xyzw'])
                common=dict(source=event['parent'],case_id=event['case_id'],command_id=event['command_id'],
                    progress=1.,terminal_event=True,opening=1.)
                for name,target in [('actual_terminal',terminal),('original_committed_waypoint',goal)]:
                    rows_by_key.setdefault(key,{}).setdefault(name,[]).append(dict(common,**relative(predicted,target)))
            if event['reference_mode']=='released':
                rare_by_case.setdefault(event['case_id'],[]).append(event)
            if task=='put_rubbish_in_moving_bin' and phase=='deposit':
                case=case_map[event['case_id']];cache=a.checked_pointer(ROOT,case['cache'],files)
                with np.load(cache,allow_pickle=False) as stored:
                    ids=tuple(map(str,stored['body_ids']));j=ids.index('bin');positions=stored['target_positions'][:,j,:2].copy()
                    times=stored['times'].copy()
                velocity=np.diff(positions,axis=0)/.05
                flips=np.flatnonzero(np.any((velocity[1:]*velocity[:-1]<0)&
                    (np.abs(velocity[1:])>1e-3)&(np.abs(velocity[:-1])>1e-3),axis=1))+1
                begin,end=event['start_frame'],event['terminal_frame']
                fit=detail['partitions'][key]['endpoint_reference_comparisons']['current_native_reference']
                error=fit['parent_terminal_residuals'][event['parent']]
                bin_rows.append(dict(parent=event['parent'],case_id=event['case_id'],command_id=event['command_id'],
                    current_reference_parent_CV_position_error_m=error['position'],
                    actual_terminal_to_world_waypoint_error=event['actual_terminal_error_to_original_world_waypoint'],
                    duration_seconds=event['duration_seconds'],native_reference_motion=event['actual_issue_to_terminal_reference_motion'],
                    boundary_like_observed_velocity_sign_changes_during_command=[dict(frame=int(i),time=float(times[i]),
                        xy_position=positions[i].tolist(),previous_displacement_velocity=velocity[i-1].tolist(),
                        next_displacement_velocity=velocity[i].tolist()) for i in flips if begin<=i<=end],
                    velocity_sign_change_is_descriptive_not_native_boundary_sensor=True))
        if rows_by_key:
            forecast[task]={key:{name:endpoint_crossfit(rows,config) for name,rows in options.items()}
                for key,options in rows_by_key.items()}
        if rare_by_case:
            rare[task]={}
            for cid,events in rare_by_case.items():
                timeline=events[0]['complete_public_contact_mode_command_timeline']
                first_start=min(event['start_frame'] for event in events)
                release=[row for row in timeline if row['public_mode']=='released' and row['frame']<=first_start]
                release_first=release[0] if release else None
                commands=case_map[cid]['original_commands']
                prior_last=[row for row in commands if row['generated_at']<first_start*.05
                    and row['raw_phase'] in ('release','hang')]
                rare[task][cid]=dict(parent=events[0]['parent'],
                    original_released_completion_events=[dict(command_id=e['command_id'],phase=e['partition'].split('|')[0],
                        raw_stage=e['raw_stage'],start_frame=e['start_frame'],terminal_frame=e['terminal_frame']) for e in events],
                    first_public_release_transition=release_first,
                    earlier_release_or_hang_commands=prior_last,complete_public_mode_timeline=timeline,
                    released_completion_occurs_after_an_earlier_release_or_hang=bool(prior_last))
    for task in forecast:
        for key in forecast[task]:
            for name in forecast[task][key]:
                forecast[task][key][name].update(source_declared_prediction_seconds=.5,
                    privileged_TRAIN_driver_yaw_speed_used=True,source_fixed_parameter_not_searched=True,
                    causal_forecast_model_or_runtime_permission_proven=False,
                    full_library_fitting_not_performed=True)
    bin_key=next(key for key in report['tasks']['put_rubbish_in_moving_bin']['partitions'] if key.startswith('deposit|'))
    bins=report['tasks']['put_rubbish_in_moving_bin']['partitions'][bin_key]
    tail=[row for row in bin_rows if row['current_reference_parent_CV_position_error_m']>.03]
    tail.sort(key=lambda row:-row['current_reference_parent_CV_position_error_m'])
    for name in ('mode_observer.py','interaction.py','pipeline.py'):
        p=ROOT/'our_mani/multitask'/name;files[str(p)]=a.file_sha(p)
    for p in (Path(__file__),ROOT/'scripts/diagnose_TRAIN_reference_times_r10.py'):
        files[str(p)]=a.file_sha(p)
    result=dict(schema='r10_actual700_reference_time_source_root_cause_readonly_v1',
        created_utc=datetime.now(timezone.utc).isoformat(),numerical_report=a.pointer(ROOT,a.local(ROOT,report_path)),
        native_sources_are_historical_executed_archive=native_sources,
        source_declared_half_second_rotating_forecast_counterdiagnostic=forecast,
        bin_normal_deposit=dict(original_parent_count=90,current_reference_distribution=bins['comparisons']['current_native_reference'],
            actual_waypoint_arrival_errors=bins['actual_terminal_to_committed_waypoint_error'],
            parents_above_existing3cm=tail,all90_parent_rows=bin_rows,
            all90_parent_rows_retained=True,outliers_removed_or_thresholds_relaxed=False,
            parents_with_velocity_sign_change_during_command=sum(bool(row['boundary_like_observed_velocity_sign_changes_during_command']) for row in bin_rows),
            above3cm_parents_with_velocity_sign_change=sum(bool(row['boundary_like_observed_velocity_sign_changes_during_command']) for row in tail)),
        actual_rare_released_sources=rare,
        runtime_released_partition_review=dict(
            public_observer='PublicModeObserver.observe keeps released after known contact clears until a later attachment',
            actual_rare_source_cause='Every observed rare released approach/acquire completion follows a previous explicit release/hang in the same teacher episode; these are second teacher cycles.',
            current_controller_completed_phase_order='InteractionController.step holds at phase_index>=len(library.phase_order); it does not restart approach after final phase.',
            released_approach_from_this_second_cycle_reachable_in_current_one_pass_controller=False,
            released_acquire_cannot_be_declared_unreachable=True,
            released_acquire_reason='An acquire that closes but cannot verify the selected body can call _reopen_failed_acquire, remain in acquire and clear contact. CandidateGenerator compares exact public mode but does not categorically reject released.',
            original_completed_released_partitions_removed=False,
            gate_or_runtime_policy_change_authorized=False),
        production_sources_rules_raw_labels_masks_parent_folds_or_thresholds_modified=False,
        TRAIN_only=True,native_GPU_or_new_collection_started=False,formal_policy_or_model_release=False,
        verified_files_sha256=files)
    for name,checksum in files.items():a.require(a.file_sha(name)==checksum,'source or interpretation input changed: '+name)
    p=out/'report.json';write(p,result)
    return dict(report=a.pointer(ROOT,p),forecast_counterdiagnostic={task:{key:{name:[row.get('parent_count'),
        row.get('position_tolerance_m'),row.get('orientation_tolerance_rad'),row['passes_existing_endpoint_caps']]
        for name,row in variants.items()} for key,variants in partitions.items()} for task,partitions in forecast.items()},
        bin_tail_parents=len(tail),bin_tail_velocity_sign_changes=result['bin_normal_deposit']['above3cm_parents_with_velocity_sign_change'],
        rare_original_parent_counts={task:len(rows) for task,rows in rare.items()},native_or_GPU_started=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',required=True);parser.add_argument('--out',required=True)
    args=parser.parse_args();print(json.dumps(run(args.report,args.out),ensure_ascii=False))
