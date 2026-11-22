"""Read-only strict audit of the already executed v16 first RML case."""
from datetime import datetime,timezone
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
for directory in (ROOT,ROOT/'scripts'):
    if str(directory) not in sys.path:sys.path.insert(0,str(directory))
import collect_multitask_recovery_r10 as runner
from our_mani.multitask import acquisition_lineage as lineage
from our_mani.multitask.training_view import validate_arrays
from our_mani.multitask.artifact_fitting import interaction_source
from our_mani.multitask.specs import get_task_spec
from report_multitask_phase_support_r10 import classify_attempts,summarize_attempts

BASE=ROOT/'artifacts/framework/multitask_r10'
SETUP=BASE/'TRAIN_recovery_v16_known_failures_setup'
CID='70a9c9f68520d04b'
OUT=BASE/'first_RML_v16_integration_acceptance.json'


def main():
    require=lineage.require;require(not OUT.exists() and not OUT.with_suffix('.txt').exists(),'new fixed-case audit outputs required')
    context=runner.verify_execution(SETUP/'plan.json',SETUP/'frozen/execution_registration.json',case_ids=[CID])
    current=context['current'];previous=context['previous'];plan=context['plan'];files=dict(context['files'])
    case=next(value for value in plan['cases'] if value['case_id']==CID)
    path=runner.collection.case_directory(current['collection'],case)/'case.json'
    record=runner.validate_recovery_result(path,context,case)
    process=record['parent_process_receipt'];attempt=Path(lineage.command_argument(process['command'],'--output')).parent
    parent_pointer=lineage.pointer(ROOT,attempt/'parent_receipt.json');parent=lineage.read(attempt/'parent_receipt.json')
    lineage.checked_pointer(ROOT,lineage.pointer(ROOT,attempt/'native.log'),files)
    completion=lineage.verify_completed(ROOT,dict(case_id=CID,canonical_receipt=lineage.pointer(ROOT,path),parent_receipt=parent_pointer),current,case,files)
    preceding=[];old_log_evidence=[]
    for ptr in context['failures'][CID]:
        old_path,failed=lineage.verify_attempt_parent(ROOT,ptr,previous,CID,files,status='infrastructure_error')
        old_log=old_path.with_name('native.log');lineage.checked_pointer(ROOT,lineage.pointer(ROOT,old_log),files)
        preceding.append(dict(actual_plan_sha256=previous['plan']['plan_sha256'],parent_receipt=ptr,native_log=lineage.pointer(ROOT,old_log),
            started_utc=failed['started_utc'],finished_utc=failed['finished_utc']))
        lines=old_log.read_text(errors='replace').splitlines()
        selected=[line for line in lines if any(token in line for token in ('rMax=','vel_correction=','目标:','超过最大步数'))]
        old_log_evidence.append(dict(parent_receipt=ptr,native_log=lineage.pointer(ROOT,old_log),
            original_failure_status=failed['status'],original_failure_policy_label_assigned=failed['policy_failure_label_assigned'],
            RML_diagnostic_lines=selected))
    lineage.verify_recovery_execution(ROOT,completion,current,previous,preceding,files)
    with np.load(ROOT/record['entry']['cache'],allow_pickle=False) as stored:arrays={key:stored[key].copy() for key in stored.files}
    statistics=validate_arrays(arrays,record['entry'],record)
    hooks=lineage.read(ROOT/record['recovery_execution']['hook_receipt']['path'])
    reset=hooks['reset_recovery_receipts'][0];derived=lineage.expected_reset(case,plan['protocol'])
    require(reset==record['native_reset_receipt']['reset_recovery_receipt']
        and all(reset.get(key)==value for key,value in derived.items()) and reset['source']==case['source']
        and reset['source_resampled'] is False and reset['complete'] is True,'actual RML case reset/source identity differs')
    paths=hooks['RML_path_recovery_receipts'];require(len(paths)==1,'fixed first-RML case must preserve its one actual executed native path')
    rml=paths[0];factory=rml['initialization_receipt'];probes=factory['probes'];ticks=record['alignment']['clock']['ticks']
    require(rml['initialization_count']==1 and rml['base_adapter_step_calls']==rml['native_step_calls']==ticks==26
        and rml['close_called'] is True and rml['close_complete'] is True and rml['close_errors']==[] and rml['errors']==[],
        'real native RML execution or cleanup differs from counted actor ticks')
    require(rml['physics_step_called_by_adapter'] is False and rml['real_native_step_motion_replaced'] is False
        and factory['complete'] is True and factory['previous_q_reset_before_every_probe'] is True
        and factory['physics_step_called'] is False and factory['arm_joint_or_target_setter_called'] is False
        and factory['physical_parameters_modified'] is False and factory['final_execution_handle_stepped_by_factory'] is False
        and factory['live_execution_handle_transferred'] is True and factory['cleanup_failures']==[],
        'RML virtual factory may not replace the actual native motion or advance physical clocks')
    require(len(probes)==1 and factory['accepted_velocity_correction']==1. and probes[0]['velocity_correction']==1.
        and probes[0]['virtual_steps']==35 and probes[0]['final_state']==1
        and probes[0]['maximum_joint_velocity_ratio']<=factory['velocity_limit_acceptance_ratio']
        and probes[0]['probe_handle_deleted'] is True
        and factory['deleted_probe_handles']==[probes[0]['probe_handle']]
        and factory['live_execution_handle'] not in factory['deleted_probe_handles'],
        'virtual probe acceptance, ownership or deletion receipt differs')
    require(rml['native_execution_completion_observed'] is False
        and rml['historical_fault_reproduced_in_native_diagnostic'] is False
        and rml['native_velocity_correction_second_round_observed'] is False,
        'this early-censored, single-probe run must not claim waypoint completion or historical fault reproduction')
    teacher_mask=arrays['teacher_action_valid'];limits=np.asarray(factory['joint_upper_velocity_limits'],float)
    require(int(teacher_mask.sum())==record['teacher_action_rows']==26 and np.all(arrays['teacher_command_id'][teacher_mask]==0)
        and np.isfinite(arrays['applied_joint_reference'][teacher_mask]).all(),'actual teacher/action rows or joint reference evidence differs')
    joint_motion=np.abs(arrays['joint_positions']-arrays['joint_positions'][0]).max(axis=0)
    actual_velocities=np.abs(np.diff(arrays['joint_positions'],axis=0)/np.diff(arrays['times'])[:,None])
    tcp_motion=float(np.linalg.norm(arrays['measured_tcp_xyzw'][-1,:3]-arrays['measured_tcp_xyzw'][0,:3]))
    require(float(joint_motion.max())>0. and tcp_motion>0.,'actual counted RML path has no measured robot motion')
    require(np.array_equal(np.asarray(probes[0]['previous_q_initial']),arrays['joint_positions'][0]),'RML probe did not start from the measured initial robot q')
    command_path=ROOT/record['entry']['teacher_command_receipt'];commands=lineage.read(command_path)
    source,source_audit=interaction_source(dict(path=path,record=record,entry=record['entry'],arrays=arrays,
        command_receipt_verified=True,command_receipt=commands,command_receipt_sha256=lineage.file_sha(command_path)))
    attempts=classify_attempts(source,task_success=record['alignment']['native_success_latched'])
    phases=summarize_attempts(attempts,[phase.name for phase in get_task_spec(case['task']).phase_skills])
    require(len(commands['commands'])==1 and commands['commands'][0]['executed_tick_count']==26
        and commands['commands'][0]['command_incomplete_at_cutoff'] is True
        and commands['commands'][0]['cutoff_reason']=='native_terminal' and commands['phase_completion_events']==[]
        and not source.completion_events and len(attempts)==1 and attempts[0]['right_censored'] is True,
        'native-terminal incomplete command must remain right-censored without completed-skill BC promotion')
    archived_sources={}
    for relative in ('scripts/collect_multitask_recovery_r10.py','our_mani/multitask/rml_path_repair.py',
        'our_mani/multitask/rml_path_recovery.py','our_mani/multitask/train_teacher.py','our_mani/multitask/development_reset_recovery.py'):
        archived_sources[relative]=current['sources'][str(ROOT/relative)]
    for source_path in current['sources']:
        if source_path.endswith('/arm_configuration_path.py'):archived_sources['author_arm_configuration_path']=current['sources'][source_path]
    for name in (str(Path(__file__).relative_to(ROOT)),'our_mani/multitask/training_view.py',
        'our_mani/multitask/artifact_fitting.py','scripts/report_multitask_phase_support_r10.py'):
        lineage.checked_pointer(ROOT,lineage.pointer(ROOT,ROOT/name),files)
    require(lineage.file_sha(ROOT/'our_mani/multitask/acquisition_lineage.py')=='9351ec6cb0a3da907ddf51c2a84d7fdf979a45a2186b8427e9433f757af9e8d4','historical lineage source changed')
    remaining=[cid for cid in context['allowed'] if cid!=CID];require(len(context['allowed'])==54 and len(remaining)==53,'original fixed recovery54 roster differs')
    for name,checksum in files.items():require(lineage.file_sha(name)==checksum,'fixed-case audit input changed while reading')
    report=dict(schema='r10_first_RML_v16_training_acquisition_integration_acceptance_v1',created_utc=datetime.now(timezone.utc).isoformat(),
        complete=True,integration_acceptance_passed=True,scope='one predeclared originally failed TRAIN case; infrastructure compatibility only',
        case_id=CID,task=case['task'],parent_source_group=case['parent_source_group'],actual_plan=current['registration']['plan'],
        actual_source_snapshot=current['registration']['source_snapshot'],actual_archived_source_count=len(current['sources']),
        actual_plan_input_count=len(plan['inputs']),source_archive_transition_permissions_raw_base_hook_parent_validation_passed=True,
        runner_validate_recovery_result_passed=True,independent_array_and_clock_validation_passed=True,
        canonical_receipt=lineage.pointer(ROOT,path),raw_child_receipt=completion['child_receipt'],parent_receipt=parent_pointer,
        actual_child_python_process_id=record['process_id'],actual_parent_Popen_process_id=parent['process_id'],
        parent_child_PID_equality_required=False,same_source_seed_variation=dict(source=case['source'],expected=derived,
            actual_seed=reset['seed'],actual_variation=reset['variation'],source_resampled=False,
            original_native_reset_succeeded=reset['original_reset_succeeded'],owned_physical_recovery_applied=reset['recovery_applied'],
            original_failed_attempt_full_initial_state_comparison_available=False,native_Cpp_planner_RNG_observable=False),
        actual_execution=dict(ticks=ticks,frames=len(arrays['times']),clock=record['alignment']['clock'],
            actual_physics_seconds=record['alignment']['actual_physics_seconds'],actual_task_seconds=record['alignment']['actual_task_seconds'],
            actual_teacher_action_rows=int(teacher_mask.sum()),measured_joint_position_max_abs_change_rad=joint_motion.tolist(),
            measured_TCP_start_to_end_translation_m=tcp_motion,measured_finite_difference_joint_speed_max_rad_s=actual_velocities.max(axis=0).tolist(),
            measured_joint_speed_to_native_limit_max_ratio=float((actual_velocities/limits).max()),
            real_native_RML_steps=rml['native_step_calls'],virtual_probe_steps_are_not_physics_ticks=True),
        actual_RML_factory_and_execution_witness=rml,archived_RML_execution_source_bindings=archived_sources,
        cleanup=dict(virtual_probe_handle=probes[0]['probe_handle'],probe_deletion_receipted=True,
            live_execution_handle=factory['live_execution_handle'],transferred_to_original_native_path=True,
            early_terminal_path_closed=True,close_complete=True,cleanup_errors=[],
            cleanup_basis='actual close receipt plus archived NativeRMLPath.close simRMLRemove implementation',
            independent_native_global_handle_inventory_saved=False),
        causal_window_statistics=statistics,context_primary_view=case['context_primary_view'],
        teacher_phase_BC=dict(commands=commands,attempts=attempts,phase_parent_support=phases,interaction_source_audit=source_audit,
            raw_teacher_rows_retained=26,completed_skill_phase_support_added=0,incomplete_native_terminal_attempt_retained_right_censored=True,
            native_terminal_failure_relabelled_as_infrastructure=False,BC_model_fitted=False),
        observed_native_outcome={key:record['alignment'][key] for key in ('native_success_latched','native_terminal_failure_latched','native_terminal_tick')},
        native_task_success_required_for_infrastructure_acceptance=False,prior_original_failure_evidence=old_log_evidence,
        historical_fake_velocity_fault_reproduced_in_this_native_run=False,second_velocity_correction_probe_observed=False,
        current_replanned_native_path_is_identical_to_old_failed_path_claimed=False,
        compatibility_of_owned_RML_initialization_real_execution_and_early_cleanup_validated=True,
        original_failure_counterfactual_cure_proven=False,may_expand_remaining_declared53_infrastructure_cases=True,
        remaining_predeclared_case_ids=remaining,automatic_resampling_or_retry_authorized=False,existing_completed_replacement_authorized=False,
        full_live_composite_constructed=False,full_roster_or_eight_parent_phase_release_proven=False,
        policy_success_rate_or_method_comparison_claimed=False,frozen_sources_modified=False,native_or_GPU_started_by_this_audit=False,
        verified_files_sha256=files)
    OUT.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    OUT.with_suffix('.txt').write_text('\n'.join([
        'PASS: first fixed v16 RML TRAIN acquisition infrastructure integration; case70a9c9f68520d04b.',
        'Actual owned RML initialization1/path1, virtual probe35steps, real native execution26ticks/1.3s, measured robot motion present.',
        'Probe0 deletion receipted; live handle1 transferred unstepped by factory, then native early-terminal close complete without errors.',
        'Same original TRAIN sourceepisode123,seed1796081873,variation0. Original reset succeeded; physical recovery branch not applied.',
        'One probe at velocity correction1.0,maxratio0.7016306373536695. Historical fake-velocity fault and second correction round NOT reproduced.',
        'Old failed trajectory was not saved/reused and current path length differs; this is compatibility, not a counterfactual cure proof.',
        '26 raw teacher rows retained, native-terminal incomplete approach right-censored,0 completed-skill BC phase support; no relabelling.',
        'May extend only the remaining53 predeclared same-source infra cases; preserve failures/early terminals and existing canonicals.',
        'No full live composite scanned, no native/GPU started, no policy-success or formal training-release claim.'])+'\n')
    print(json.dumps(dict(integration_acceptance_passed=True,real_RML_steps=26,remaining_cases_may_expand=53,report=lineage.pointer(ROOT,OUT))))


if __name__=='__main__':main()
