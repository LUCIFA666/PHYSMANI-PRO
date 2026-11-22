"""Read-only evidence audit of the already completed fixed v15 native pair.

No live full-collection inventory, simulation launch, fit or outcome selection.
"""
from copy import deepcopy
from datetime import datetime,timezone
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
for path in (ROOT,ROOT/'scripts'):
    if str(path) not in sys.path:sys.path.insert(0,str(path))
import collect_multitask_recovery_r10 as runner
from our_mani.multitask import acquisition_lineage as lineage
from our_mani.multitask.training_view import validate_arrays
from our_mani.multitask.artifact_fitting import interaction_source
from report_multitask_phase_support_r10 import classify_attempts,summarize_attempts
from our_mani.multitask.specs import get_task_spec

BASE=ROOT/'artifacts/framework/multitask_r10'
IDS=('03026b372ecf56ec','626949c9370829b5')
SETUP=BASE/'TRAIN_recovery_v15_initial_pair_setup'
OUTPUT=BASE/'v15integration_acceptance.json'


def snapshots(value,path='reset'):
    result=[]
    if isinstance(value,dict):
        if all(key in value for key in ('arm','gripper','bodies','physics_time','task_time')):result.append((path,value))
        for key,item in value.items():result.extend(snapshots(item,path+'.'+key))
    elif isinstance(value,list):
        for index,item in enumerate(value):result.extend(snapshots(item,path+'['+str(index)+']'))
    return result


def physical_audit(reset,permission):
    require=lineage.require;rows=snapshots(reset);maxima={kind:0. for kind in ('arm','gripper')}
    require(reset['complete'] is True and reset['recovery_applied'] is True and reset['source_resampled'] is False,
        'fixed pair must preserve its actual successful same-source physical recovery')
    require(reset['failure_eligibility']['original_native_attempts_observed']==40
        and reset['failure_eligibility']['waypoint_index']==0,'original40 waypoint0 exhaustion differs')
    for label,state in rows:
        for kind in maxima:
            joint=state[kind];q=np.asarray(joint['q']);bounds=np.asarray(joint['intervals']);cyclic=np.asarray(joint['cyclic'])
            violation=np.where(cyclic,0.,np.maximum(np.maximum(bounds[:,0]-q,q-bounds.sum(axis=1)),0.))
            require(np.array_equal(violation,np.asarray(joint['native_interval_violation']))
                and float(violation.max())==joint['maximum_native_interval_violation'],'raw joint violation receipt differs from raw q/native intervals')
            tolerance=permission['joint_interval_numerical_tolerance'][kind]
            require(joint['native_interval_numerical_tolerance']==tolerance and float(violation.max())<=tolerance
                and joint['raw_joint_positions_clipped'] is False,'raw joint guard exceeds frozen numerical tolerance or clips evidence')
            maxima[kind]=max(maxima[kind],float(violation.max()))
    steps=reset['settling_steps'];require(len(steps)==11 and [row['phase'] for row in steps]==['original_native10']*10+['original_owned_development1'],
        'original10 plus owned1 settle inventory differs')
    deltas=[row['after']['physics_time']-row['before']['physics_time'] for row in steps]
    require(max(abs(value-.05) for value in deltas)<1e-6 and reset['task_delta_seconds']==0.
        and abs(reset['physics_delta_seconds']-.55)<1e-6,'same-scene settling clock differs')
    require(all(row['raw_state_unchanged'] is True and row['before']==row['after'] for row in reset['native_observation_queries']),
        'native reset observation query changed physical state')
    for key in ('physical_guards_before_settling','physical_guards_after_settling'):
        require(all(value is False for value in reset[key].values()),'physical collision/query guard differs')
    return dict(snapshot_count=len(rows),raw_joint_positions_clipped=False,maximum_native_interval_violation=maxima,
        frozen_numerical_tolerance=permission['joint_interval_numerical_tolerance'],settling_steps=len(steps),
        settling_physics_delta_seconds=reset['physics_delta_seconds'],settling_task_delta_seconds=reset['task_delta_seconds'],
        maximum_single_settle_dt_error_seconds=max(abs(value-.05) for value in deltas),
        final_gripper_violation_m=reset['final_settled_state']['gripper']['maximum_native_interval_violation'],
        physical_guards_passed=True,observation_queries_preserve_raw_state=True)


def main():
    require=lineage.require;require(not OUTPUT.exists() and not OUTPUT.with_suffix('.txt').exists(),'fresh fixed-pair audit output required')
    files={};context=runner.verify_execution(SETUP/'plan.json',SETUP/'frozen/execution_registration.json',case_ids=list(IDS))
    files.update(context['files']);actual=context['current'];previous=context['previous']
    plan=context['plan'];by_id={case['case_id']:case for case in plan['cases']}
    permission=lineage.read(lineage.checked_pointer(ROOT,plan['protocol']['development_reset_profile'],files))
    rows=[];records={};arrays={};resets={};history=[]
    for version in (12,13,14):
        registration_path=BASE/('TRAIN_recovery_v'+str(version)+'_initial_pair_setup/frozen/execution_registration.json')
        registry=lineage.read(lineage.checked_pointer(ROOT,lineage.pointer(ROOT,registration_path),files))
        historical=lineage.load_archived_plan(ROOT,registry['recovery'],files)
        for cid in IDS:
            path=historical['collection']/'infrastructure_attempts'/cid/'0/parent_receipt.json'
            _,receipt=lineage.verify_attempt_parent(ROOT,lineage.pointer(ROOT,path),historical,cid,files,status='infrastructure_error')
            log=path.with_name('native.log');lineage.checked_pointer(ROOT,lineage.pointer(ROOT,log),files)
            history.append(dict(version=version,case_id=cid,actual_plan_sha256=historical['plan']['plan_sha256'],
                parent_receipt=lineage.pointer(ROOT,path),native_log=lineage.pointer(ROOT,log),status=receipt['status'],
                policy_failure_label_assigned=False,source_snapshot=registry['recovery']['source_snapshot'],
                historical_canonical_exists=(historical['collection']/'cases/train'/by_id[cid]['task']/cid/'case.json').exists()))
    for cid in IDS:
        case=by_id[cid];path=runner.collection.case_directory(actual['collection'],case)/'case.json'
        record=runner.validate_recovery_result(path,context,case);records[cid]=record
        command=record['parent_process_receipt']['command'];attempt=Path(lineage.command_argument(command,'--output')).parent
        parent=lineage.read(attempt/'parent_receipt.json');lineage.checked_pointer(ROOT,lineage.pointer(ROOT,attempt/'native.log'),files)
        row=dict(case_id=cid,canonical_receipt=lineage.pointer(ROOT,path),parent_receipt=lineage.pointer(ROOT,attempt/'parent_receipt.json'))
        completion=lineage.verify_completed(ROOT,row,actual,case,files);preceding=[]
        for ptr in context['failures'][cid]:
            raw_path,failed=lineage.verify_attempt_parent(ROOT,ptr,previous,cid,files,status='infrastructure_error')
            log=raw_path.with_name('native.log');lineage.checked_pointer(ROOT,lineage.pointer(ROOT,log),files)
            preceding.append(dict(actual_plan_sha256=previous['plan']['plan_sha256'],parent_receipt=ptr,native_log=lineage.pointer(ROOT,log),
                started_utc=failed['started_utc'],finished_utc=failed['finished_utc']))
        lineage.verify_recovery_execution(ROOT,completion,actual,previous,preceding,files)
        with np.load(ROOT/record['entry']['cache'],allow_pickle=False) as stored:arrays[cid]={key:stored[key].copy() for key in stored.files}
        data=arrays[cid];stats=validate_arrays(data,record['entry'],record)
        hooks=lineage.read(ROOT/record['recovery_execution']['hook_receipt']['path']);reset=hooks['reset_recovery_receipts'][0];resets[cid]=reset
        require(reset==record['native_reset_receipt']['reset_recovery_receipt'],'hook and original embedded reset differ')
        require(all(reset.get(key)==value for key,value in lineage.expected_reset(case,plan['protocol']).items()),'same-case reset identity differs')
        native_rows=int(data['teacher_action_valid'].sum());require(native_rows==record['teacher_action_rows'],'teacher row count differs from raw masks')
        phase=dict(teacher_used=case['teacher_used'],teacher_valid_rows=native_rows,RML_paths=len(hooks['RML_path_recovery_receipts']))
        if case['teacher_used']:
            command_path=ROOT/record['entry']['teacher_command_receipt'];commands=lineage.read(command_path)
            source,source_audit=interaction_source(dict(path=path,record=record,entry=record['entry'],arrays=data,
                command_receipt_verified=True,command_receipt=commands,command_receipt_sha256=lineage.file_sha(command_path)))
            attempts=classify_attempts(source,task_success=record['alignment']['native_success_latched'])
            phase.update(command_count=len(commands['commands']),commands=commands['commands'],completion_event_count=len(commands['phase_completion_events']),
                attempts=attempts,phase_parent_support=summarize_attempts(attempts,[skill.name for skill in get_task_spec(case['task']).phase_skills]),
                interaction_source_audit=source_audit,BC_model_fitted=False)
            require(native_rows==0 and not commands['phase_completion_events'] and len(commands['commands'])==1
                and commands['commands'][0]['planning_failure'] and commands['commands'][0]['executed_tick_count']==0
                and record['alignment']['clock']['phases']=={'TRAIN_teacher_failed_no_resample':200},
                'fixed teacher path failure was not preserved as zero-execution right-censored evidence')
        else:require(native_rows==0 and record['entry']['interaction_eligible'] is False,'hold cannot supply teacher BC labels')
        rows.append(dict(case_id=cid,behavior=case['behavior'],parent_source_group=case['parent_source_group'],actual_plan_sha256=plan['plan_sha256'],
            canonical_receipt=lineage.pointer(ROOT,path),raw_child_receipt=completion['child_receipt'],parent_receipt=row['parent_receipt'],
            actual_child_python_process_id=record['process_id'],actual_parent_Popen_process_id=parent['process_id'],
            parent_child_PID_equality_required=False,runner_validate_recovery_result_passed=True,array_and_clock_validation_passed=True,
            actual_ticks=record['alignment']['clock']['ticks'],frames=record['entry']['frames'],actual_clock=record['alignment']['clock'],
            actual_physics_seconds=record['alignment']['actual_physics_seconds'],actual_task_seconds=record['alignment']['actual_task_seconds'],
            context_primary_view=case['context_primary_view'],recorded_context_eligible=record['entry']['context_training_eligible'],
            independently_computed_causal_window_statistics=stats,physical_reset=physical_audit(reset,permission),teacher_phase_BC=phase,
            native_task_success=record['alignment']['native_success_latched'],native_task_success_is_training_compatibility_gate=False,
            preceding_original_infrastructure_attempts=preceding))
    teacher,hold=IDS;left,right=resets[teacher],resets[hold]
    require(left['final_settled_state']==right['final_settled_state'] and left['settling_steps']==right['settling_steps']
        and records[teacher]['native_reset_receipt']['native_initial_snapshot']==records[hold]['native_reset_receipt']['native_initial_snapshot'],
        'saved paired actual initial physical states differ')
    fields=('measured_tcp_xyzw','joint_positions','robot_gripper_open','positions','rotations','target_positions','target_rotations',
        'native_reported_linear_velocities','native_reported_angular_velocities')
    deltas={name:float(np.max(np.abs(arrays[teacher][name][0].astype(float)-arrays[hold][name][0].astype(float)))) for name in fields}
    require(all(value==0. for value in deltas.values()),'paired frame-zero body/robot arrays differ')
    sensor=[str(arrays[cid]['sensor_sha256'][0]) for cid in IDS];require(sensor[0]==sensor[1],'paired frame-zero RGBD/mask digest differs')
    require(left['seed_identity']==right['seed_identity'] and left['seed']==right['seed'] and left['variation']==right['variation']
        and left['source']==right['source'],'teacher/hold source-derived initialization differs')
    source_HDF5=[records[cid]['native_reset_receipt']['source_hdf5_sha256'] for cid in IDS];require(source_HDF5[0]==source_HDF5[1],'source camera-calibration HDF5 metric SHA differs')
    require(not any(row['historical_canonical_exists'] for row in history),'earlier failed version has been silently promoted')
    total_RML=sum(row['teacher_phase_BC']['RML_paths'] for row in rows)
    for name in ('our_mani/multitask/acquisition_lineage.py','our_mani/multitask/validation_lineage.py',
        'our_mani/multitask/training_view.py','our_mani/multitask/artifact_fitting.py','scripts/report_multitask_phase_support_r10.py',str(Path(__file__).relative_to(ROOT))):
        lineage.checked_pointer(ROOT,lineage.pointer(ROOT,ROOT/name),files)
    require(lineage.file_sha(ROOT/'our_mani/multitask/acquisition_lineage.py')=='9351ec6cb0a3da907ddf51c2a84d7fdf979a45a2186b8427e9433f757af9e8d4','frozen historical lineage changed')
    for name,checksum in files.items():require(lineage.file_sha(name)==checksum,'audited fixed-pair input changed while reading')
    report=dict(schema='r10_v15_fixed_pair_training_acquisition_integration_acceptance_v1',created_utc=datetime.now(timezone.utc).isoformat(),
        complete=True,integration_acceptance_passed=True,scope='two already completed actual native TRAIN acquisitions; same single parent',
        fixed_case_ids=list(IDS),actual_plan=context['current']['registration']['plan'],actual_source_snapshot=context['current']['registration']['source_snapshot'],
        actual_plan_sha256=plan['plan_sha256'],actual_archived_source_count=len(actual['sources']),actual_plan_input_count=len(plan['inputs']),
        source_archive_transition_permissions_and_all_fixed_pair_raw_outputs_verified=True,executed_source_files_modified=False,
        cases=rows,earlier_failed_v12_v13_v14_attempts_preserved=history,
        paired_initialization=dict(seed=left['seed'],source_variation=left['variation'],scene_variation=left['final_settled_state']['scene_variation'],
            source=left['source'],driver=left['final_settled_state']['driver'],source_calibration_HDF5_metric_SHA_equal=source_HDF5[0],
            actual_frame_zero_RGBD_and_instance_masks_digest=sensor[0],sensor_digest_does_not_include_camera_calibration_matrices=True,
            runtime_camera_matrices_separately_saved=False,runtime_camera_matrix_full_equality_claimed=False,
            final_settled_state_and_all11_settle_step_snapshots_exactly_identical=True,native_initial_snapshot_exactly_identical=True,
            frame_zero_body_and_robot_array_max_abs_differences=deltas,opaque_contact_solver_state_full_equality_claimed=False),
        actual_RML_paths_observed=total_RML,actual_RML_recovery_validated_by_this_pair=False,
        teacher_BC_completed_phase_support_added=0,teacher_path_failure_preserved_right_censored=True,
        allow_expand_same_source_physical_initialization_acquisition_recovery=True,
        RML_failure_case_native_validation_still_required=True,full_roster_phase_or_model_release_proven=False,
        full_live_composite_constructed=False,native_or_GPU_started_by_this_audit=False,
        policy_success_rate_or_method_comparison_claimed=False,verified_files_sha256=files)
    OUTPUT.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    OUTPUT.with_suffix('.txt').write_text('\n'.join([
        'PASS: fixed v15 pair TRAIN acquisition compatibility (one original TRAIN parent; two actual native cases).',
        'Teacher:200ticks/201frames;hold:60ticks/61frames. SHA/source/transition/base/hooks/raw/parent/arrays/clocks verified.',
        'Same seed2990062122/source variation0, exact saved paired initial body+robot states, identical frame0 RGBD/mask digest.',
        'Gripper raw interval residual max2.820044755935669e-6m < frozen1e-5m;arm residual0 < frozen1e-6rad; no clipping.',
        'Teacher wp0 path generation failed:0 executed teacher rows,0 completed skill events; retained right-censored/no resample.',
        'RML paths observed:0. This pair does not validate an actual RML fault repair or learned-policy success.',
        'May extend declared same-source physical initialization recovery. Preserve earlier failed receipts and validate a genuine RML case separately.',
        'No full live composite scanned; no new native/GPU process launched; no fitting, phase release or policy success-rate claim.'])+'\n')
    print(json.dumps(dict(integration_acceptance_passed=True,cases=2,actual_RML_paths_observed=total_RML,report=lineage.pointer(ROOT,OUTPUT))))


if __name__=='__main__':main()
