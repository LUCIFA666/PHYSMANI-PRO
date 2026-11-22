"""Independent actual first replica2 raw/parent/feedback integration audit.

Never starts a worker or simulator, changes frozen inputs, fits a model, or
uses the native task outcome as an infrastructure acceptance threshold.
"""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from our_mani.multitask import acquisition_lineage as a,periodic_acquisition as audit
from our_mani.multitask import artifact_fitting as adapter,interaction,release_preflight as release
from scripts.report_multitask_phase_support_r10 import classify_attempts
BASE=ROOT/'artifacts/framework/multitask_r10'
CID='d00ab7e60a3ef5a7'
REGISTRY=BASE/'TRAIN_high_cup_periodic10_all100_candidate_v2_setup/execution_registration_approved.json'
REGISTRY_SHA='7e31d1c653985670b3e2d310b3c49d302e05c537ea1a2e42575fd94ef67946d8'


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(interaction._json(value),ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def run(output):
    started=time.perf_counter();inputs=release._Inputs(ROOT)
    a.require(a.file_sha(REGISTRY)==REGISTRY_SHA,'exact approved first-case registry required')
    context=audit.verify_registration(ROOT,a.pointer(ROOT,REGISTRY),live=True,require_ready=True)
    inputs.include(context['files'])
    directory=context['current']['collection'];canonical=directory/'cases/train'/audit.TASK/CID/'case.json'
    parent_path=directory/'infrastructure_attempts'/CID/'0/parent_receipt.json'
    a.require(canonical.exists() and parent_path.exists(),'actual first canonical and terminal parent must exist')
    parent=inputs.read(str(parent_path),a.file_sha(parent_path))
    a.require(parent['status']=='completed' and parent['returncode']==0,'actual first infrastructure parent is not completed')
    out=a.local(ROOT,output);a.require(not out.exists(),'fresh first actual integration report required');out.mkdir(parents=True)
    case=context['expected_cases'][CID];validation={};files={}
    record=audit.validate_result(ROOT,canonical,context,case,files=files,validation_proofs=validation)
    completed=a.verify_completed(ROOT,dict(case_id=CID,canonical_receipt=a.pointer(ROOT,canonical),
        parent_receipt=a.pointer(ROOT,parent_path)),context['current'],case,files)
    proof=audit.audit_collection(ROOT,a.pointer(ROOT,REGISTRY),require_complete=False)
    a.require(proof['audit_passed'] is True and proof['complete'] is False
        and proof['acquisition_projection']['expected']==100
        and proof['acquisition_projection']['completed']==1
        and set(proof['per_case_feedback_audit'])=={CID}
        and not proof['infrastructure_errors'] and not proof['pending'],
        'first integration must preserve actual1/100 full denominator without extra attempts')
    inputs.include(files);inputs.include(proof['verified_files_sha256'])
    own=record['periodic_replanning_execution'];hooks=inputs.read_pointer(own['hook_receipt'])
    base=inputs.read_pointer(own['base_collector_receipt']);scope=hooks['periodic_replanning_factory']
    a.require(len(context['current']['sources'])==82,'actual approved source closure differs from82')
    source_pairs={}
    for owner,archived in context['current']['sources'].items():
        a.require(a.file_sha(owner)==archived['sha256'],'live executable differs from actual frozen archive')
        inputs.file(archived['path'],archived['sha256'])
        source_pairs[owner]=dict(sha256=archived['sha256'],archive=archived)
    source_snapshot=a.read(BASE/'command_goal_reference_rule_v6_counted_final_setup_v1/preparation.json')['source_snapshot']
    inputs.source_snapshot(source_snapshot)
    inputs.file(__file__,a.file_sha(__file__))
    with np.load(inputs.file(record['entry']['cache'],record['entry']['cache_sha256']),allow_pickle=False) as saved:
        arrays={name:saved[name].copy() for name in saved.files}
    ledger_path=inputs.file(record['entry']['teacher_command_receipt'],record['files_sha256'][record['entry']['teacher_command_receipt']])
    ledger=a.read(ledger_path);ticks=record['alignment']['clock']['ticks']
    raw_feedback=audit.validate_counted_commands(ledger,scope,ticks=ticks,arrays=arrays,
        native_outcome=record['alignment']['native_outcome'])
    a.require(raw_feedback==validation[CID] and raw_feedback==proof['per_case_feedback_audit'][CID],
        'independent complete actual feedback reconstruction differs')
    paths=hooks['RML_path_recovery_receipts'];path_reports=[]
    for index,path in enumerate(paths):
        factory=path['initialization_receipt']
        a.require(path['close_called'] is True and path['close_complete'] is True
            and not path['close_errors'] and not path['errors']
            and path['base_adapter_step_calls']==path['native_step_calls']
            and path['physics_step_called_by_adapter'] is False
            and path['real_native_step_motion_replaced'] is False
            and factory['complete'] is True and not factory['cleanup_failures']
            and factory['physics_step_called'] is False and factory['arm_joint_or_target_setter_called'] is False
            and factory['physical_parameters_modified'] is False
            and factory['final_execution_handle_stepped_by_factory'] is False
            and factory['live_execution_handle_transferred'] is True,
            'actual periodic RML initialization, native motion or close witness differs')
        a.require(all(probe['probe_handle_deleted'] is True for probe in factory['probes'])
            and factory['live_execution_handle'] not in factory['deleted_probe_handles'],
            'RML virtual probe cleanup/live ownership differs')
        path_reports.append(dict(index=index,real_native_step_calls=path['native_step_calls'],
            native_execution_completion_observed=path['native_execution_completion_observed'],
            close_complete=True,virtual_probe_steps=sum(row['virtual_steps'] for row in factory['probes']),
            virtual_probe_steps_are_not_physics_ticks=True))
    follow_rows=int(np.count_nonzero((arrays['teacher_execution_stage']=='follow_path')
        & ~arrays['teacher_joint_limit_recovery']))
    # The result's stage is sampled after path.step(): its final step can
    # already report wait_arrival/actuate.  Count the first path_done edge per
    # real command as well, rather than treating those saved stages as no step.
    native_step_frames=set(map(int,np.flatnonzero((arrays['teacher_execution_stage']=='follow_path')
        & ~arrays['teacher_joint_limit_recovery'])))
    for command in ledger['commands']:
        rows=np.flatnonzero(arrays['teacher_command_id']==command['command_id'])
        ended=rows[arrays['teacher_path_done'][rows]&~arrays['teacher_joint_limit_recovery'][rows]]
        if len(ended):native_step_frames.add(int(ended[0]))
    a.require(sum(row['real_native_step_calls'] for row in path_reports)==len(native_step_frames),
        'actual RML native path steps differ from active follow feedback plus first actual path-done edges')
    source,source_audit=adapter.interaction_source(dict(path=canonical,record=record,entry=record['entry'],arrays=arrays,
        command_receipt=ledger,command_receipt_verified=True,command_receipt_sha256=a.file_sha(ledger_path)))
    attempts=classify_attempts(source,task_success=record['alignment']['native_success_latched'])
    interrupted={command['command_id'] for command in ledger['commands'] if 'periodic_interruption' in command}
    for command in ledger['commands']:
        if command['command_id'] in interrupted:
            a.require('command_completed_at' not in command and command.get('command_incomplete_at_cutoff') is True,
                'interrupted raw command was promoted to completed command')
    ordered={event['origin_command_id'] for event in ledger['phase_completion_events']
        if event['evidence']=='completed_command_then_ordered_teacher_phase_transition'}
    a.require(not interrupted&ordered,'periodic interruption created a false ordered completed-phase event')
    joint_motion=np.abs(arrays['joint_positions']-arrays['joint_positions'][0]).max(axis=0)
    tcp_motion=float(np.linalg.norm(arrays['measured_tcp_xyzw'][-1,:3]-arrays['measured_tcp_xyzw'][0,:3]))
    reset=hooks['reset_recovery_receipts'][0];expected_reset=a.expected_reset(case,context['plan']['protocol'])
    a.require(all(reset[key]==value for key,value in expected_reset.items()) and reset['source_resampled'] is False,
        'actual first replica2 reset differs from fixed source-derived seed/variation')
    inputs.rehash_all()
    a.require(all(a.file_sha(owner)==row['sha256'] for owner,row in source_pairs.items()),
        'one of the82 live execution sources changed during independent audit')
    result=dict(schema='r10_periodic10_actual_first_replica2_native_integration_acceptance_v1',
        created_utc=datetime.now(timezone.utc).isoformat(),integration_acceptance_passed=True,
        case_id=CID,parent_source_group=case['parent_source_group'],native_infrastructure_cases_audited=1,
        full100_complete=False,actual_collection_completed=1,actual_collection_expected=100,
        actual_plan=context['current']['registration']['plan'],actual_registration=a.pointer(ROOT,REGISTRY),
        actual_source_snapshot=context['current']['registration']['source_snapshot'],actual_source_count=82,
        actual_source_pairs=source_pairs,canonical_receipt=a.pointer(ROOT,canonical),parent_receipt=a.pointer(ROOT,parent_path),
        raw_child_receipt=completed['child_receipt'],base_receipt=own['base_collector_receipt'],hook_receipt=own['hook_receipt'],
        raw_base_and_parent_bytes_not_rewritten=True,actual_child_python_pid=record['process_id'],actual_parent_Popen_pid=parent['process_id'],
        reset=dict(expected=expected_reset,actual_seed=reset['seed'],actual_variation=reset['variation'],
            source_resampled=False,original_reset_succeeded=reset['original_reset_succeeded'],
            owned_physical_recovery_applied=reset['recovery_applied'],replica2_exact_r1_scene_pair_claimed=False),
        actual_clock=record['alignment'],actual_feedback_proof=raw_feedback,
        command_ids=[row['command_id'] for row in ledger['commands']],raw_command_ledger=a.pointer(ROOT,ledger_path),
        original_phase_events=ledger['phase_completion_events'],interrupted_command_ids=sorted(interrupted),
        interrupted_ordered_completed_phase_events=0,raw_attempt_classifications=attempts,
        actual_RML_paths=path_reports,actual_RML_close_all_complete=True,actual_active_follow_feedback_rows=follow_rows,
        actual_native_path_step_frames_reconstructed=sorted(native_step_frames),
        actual_joint_position_max_abs_change_rad=joint_motion.tolist(),actual_TCP_start_end_distance_m=tcp_motion,
        cleanup_basis='actual wrapper and source-bound native RML close receipts; no independent engine-global handle inventory',
        current_original22_fitter_sources_unchanged=True,original3100_or_Context_data_modified=False,
        native_outcome_used_as_infrastructure_acceptance_threshold=False,
        native_task_outcome_is_policy_success_rate=False,phase_support_or_publiccompletion_or_model_release_qualified=False,
        technical_extension_of_fixed_remaining99_is_acceptable=True,new_seed_or_outcome_selected_expansion_allowed=False,
        actual_periodic_cut_and_fresh_query_flow_observed=raw_feedback['actual_periodic_requeries']>0,
        remaining_fixed_case_ids=[cid for cid in context['expected_cases'] if cid!=CID],
        expanded_queue_or_native_GPU_started_by_auditor=False,model_fitted=False,
        verified_files_sha256=inputs.files,final_direct_input_rehash_passed=True,seconds=time.perf_counter()-started)
    write(out/'report.json',result)
    write(out/'actual_partial_acquisition_audit.json',proof)
    return a.pointer(ROOT,out/'report.json')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);args=p.parse_args();print(json.dumps(run(args.out)))
