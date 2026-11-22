"""Report-only fixed4 motion ablation versus the sealed old4 baseline.

Default reads plans only. Evaluate only after Root's all4 actual-wait signal (owner itself failed a retained shutdown race);
all four child closures, common live/archive/Native sources are gated before
opening any initial truth snapshot/tick. No simulator or NN is constructed.
"""
import argparse
from collections import Counter
from copy import deepcopy
import hashlib,json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.summarize_cube_candidate_paired_20261104_v1 import (
    sha,read,pointer,checked,lines,stats,decision_summary,initial_comparison)
OWNER=ROOT/'config/framework/cube_motion_only_native_owner_20261104_v2/plan.json'
BASELINE=ROOT/'artifacts/framework/local_mpc_20261102/cube_candidate_paired_closed8_posthoc_v1/report.json'
BASELINE_SHA='88827968d7e2fbbcc1aa2cb5d45df45a6a9caef414f9f5a4c6fcb524ebb07d9f'
IDS=('ordinary_00','ordinary_01','highspeed_00','highspeed_01')


def metadata(owner_path,expected_sha=None):
    if expected_sha and sha(owner_path)!=expected_sha:raise ValueError('External owner plan changed')
    owner=read(owner_path)
    if [j['id']for j in owner['jobs']]!=list(IDS):raise ValueError('Whole original fixed4 required')
    argv=owner['jobs'][0]['argv']
    plan_path=Path(argv[argv.index('--plan')+1]);freeze_path=Path(argv[argv.index('--source-freeze')+1])
    if sha(freeze_path)!=argv[argv.index('--source-freeze-sha256')+1]:raise ValueError('Worker freeze pointer changed')
    plan=read(plan_path);freeze=read(freeze_path)
    if (plan['profiles_order']!=['corrected_legacy'] or plan['formal_VAL_TEST']is not False
        or [c['id']for c in plan['cases']]!=list(IDS)):raise ValueError('Explicit whole4 motion-only development plan required')
    if sha(BASELINE)!=BASELINE_SHA:raise ValueError('Sealed baseline report changed')
    return owner,plan,freeze,dict(owner_plan=pointer(owner_path),execution_plan=pointer(plan_path),source_freeze=pointer(freeze_path),baseline_report=pointer(BASELINE))


def gate(owner,plan,freeze):
    out=Path(owner['output_root']);terminal=read(out/'terminal_status.json');failure=read(out/'failure.json')
    expected_error='Stage returned with live descendants; stop/cleanup required'
    if (terminal.get('complete')is not False or terminal.get('jobs_completed')!=3
            or terminal.get('error_type')!='RuntimeError' or terminal.get('error')!=expected_error
            or terminal.get('phase')!='cube_motion_only_native_20261104_v2'
            or [r['id']for r in terminal.get('records',[])]!=list(IDS[:3])
            or failure.get('complete')is not False or failure.get('jobs_completed')!=3
            or failure.get('error_type')!='RuntimeError' or failure.get('error')!=expected_error
            or (out/'completion.json').exists()):
        raise ValueError('Only exact retained owner shutdown race is allowed for staged posthoc diagnostics')
    for record in terminal['records']:checked(record['actual_exit'])
    prepared=[];live={};archive_count=0
    for job,spec in zip(owner['jobs'],plan['cases']):
        case=Path(job['output']);actual=read(out/'owners'/job['id']/'actual_exit.json')
        worker=read(case/'worker_exit.json');run=read(case/'run_inputs.json');result=read(case/'result.json')
        expected_child_error=None if job['id']!='highspeed_01'else dict(type='RuntimeError',message=expected_error)
        reaped=actual.get('adopted_descendants_reaped')
        if (actual.get('error')!=expected_child_error or not isinstance(reaped,list)
                or any(type(p.get('wait_status'))is not int or p['wait_status']!=0 for p in reaped)
                or (job['id']=='highspeed_01'and not reaped)):
            raise ValueError('Only exact last-child shutdown error with all adopted children wait0 allowed')
        if (type(actual.get('actual_process_returncode'))is not int or actual['actual_process_returncode']!=0
            or actual.get('actual_wait_completed')is not True or actual.get('unreaped_descendants')!=[]
            or actual.get('source_unchanged')is not True or actual.get('stop_reason')is not None
            or type(worker.get('exit_code'))is not int or worker['exit_code']!=0 or worker.get('native_closed')is not True
            or result.get('complete')is not True or result.get('source_unchanged')is not True
            or result.get('native_task')!=spec['task'] or result.get('seed')!=spec['seed']
            or run.get('TEST_read')is not False or run.get('Native_labels_used_by_policy')is not False
            or run.get('candidate_profile')!='corrected_legacy'):
            raise ValueError('Fixed case actual0/wait/Nativeclosed/source required: '+job['id'])
        sources=run['source_sha256']
        if sources!=freeze['files_sha256']:raise ValueError('Captured common source graph differs')
        for name,digest in sources.items():
            p=Path(name)
            if p.is_absolute()or'..'in p.parts:raise ValueError('ROOT-relative archive required')
            if name not in live:
                if sha(ROOT/p)!=digest:raise ValueError('Live source changed: '+name)
                live[name]=digest
            if sha(case/'source_snapshot'/p)!=digest:raise ValueError('Archived source changed: '+name)
            archive_count+=1
        if run['native_files_sha256']!=freeze['native_files_sha256']:raise ValueError('Native7 source graph differs')
        for name,digest in freeze['native_files_sha256'].items():
            if sha(name)!=digest:raise ValueError('Original Native task/condition changed: '+name)
        if result['candidate_transformer']['checkpoint_sha256']!=plan['candidate_profiles']['corrected_legacy']['sha256']:
            raise ValueError('Actual old candidate weight mismatch')
        prepared.append(dict(case=case,spec=spec,result=result,run=run,actual=actual,
            evidence=dict(worker_exit=pointer(case/'worker_exit.json'),actual_exit=pointer(out/'owners'/job['id']/'actual_exit.json'),
                run_inputs=pointer(case/'run_inputs.json'),result=pointer(case/'result.json'))))
    return prepared,dict(all4_actual0_wait_Nativeclosed=True,all4_source_gates_before_GT=True,
        live_repo_files=len(live),archived_files=archive_count,Native_files=len(freeze['native_files_sha256']),
        owner_complete=False,owner_actual_returncode=1,owner_actual_wait_confirmed_by_Root=True,
        owner_lifecycle_passed=False,last_child_shutdown_race_retained=True,
        last_child_error=dict(type='RuntimeError',message=expected_error),
        all_adopted_descendants_wait_status_zero=True,final_unreaped_descendants=[],
        failure=pointer(out/'failure.json'),terminal=pointer(out/'terminal_status.json'))


def visibility(case):
    packets=list(lines(case/'public_shadow/public_outputs.jsonl'))
    observed=[];losses=[];forecast_gaps=[];previous=False;ever=False;prev_forecast=False;ever_forecast=False
    for p in packets:
        r=p['result'];now=r['timestamp'];obj=r.get('object_observation')
        available=(r.get('observed_target_available')is True and isinstance(obj,dict)and obj.get('visible')is True)
        ready=r.get('free_forecast_available')is True
        if previous and not available:losses.append(dict(timestamp=now,reason=r.get('reason'),mode=r.get('mode'),close_requested=p['close_requested']))
        if prev_forecast and not ready:forecast_gaps.append(dict(timestamp=now,reason=r.get('reason'),mode=r.get('mode'),close_requested=p['close_requested']))
        if available:observed.append(now)
        previous=available;prev_forecast=ready;ever|=available;ever_forecast|=ready
    return dict(real_frames=len(packets),current_measured_frames=len(observed),
        first_measurement_time=observed[0]if observed else None,last_measurement_time=observed[-1]if observed else None,
        first_current_visual_loss_after_seen=losses[0]if losses else None,current_visual_loss_edges=losses,
        first_free_forecast_loss_after_ready=forecast_gaps[0]if forecast_gaps else None,
        measured_presence_is_not_attachment_or_physical_held=True)


def control_details(decisions):
    from scipy.spatial.transform import Rotation
    import numpy as np
    close=[];reopen=[];old=1.;rows=[]
    for record in decisions:
        opening=record['executed_action']['opening']
        if old>=.5 and opening<.5:close.append(record['time'])
        if old<.5 and opening>=.5:reopen.append(record['time'])
        old=opening;d=record['decision'];s=d.get('state');a=d.get('action')
        if d['mode']!='FREE_GRU'or not isinstance(s,dict)or not isinstance(a,dict):continue
        pool=(d.get('candidate_pool')or{}).get('candidates',[])
        selected=[c for c in pool if c['id']==a.get('candidate_id')]
        if len(selected)!=1:continue
        c=selected[0];lever=Rotation.from_quat(s['object_quaternion']).as_matrix()@np.asarray(c['object_tcp_position'])
        distance=float(np.linalg.norm(np.asarray(s['tcp_position'])-(np.asarray(s['object_position'])+lever)))
        relative=float(np.linalg.norm(np.asarray(s['tcp_velocity'])-(np.asarray(s['object_linear_velocity'])+np.cross(s['object_angular_velocity'],lever))))
        rows.append(dict(timestamp=record['time'],candidate_id=c['id'],deadline=c.get('deadline'),
            seconds_until_deadline=None if c.get('deadline')is None else c['deadline']-record['time'],
            current_public_selected_point_distance_m=distance,current_public_selected_point_relative_speed_m_per_s=relative,
            selected_opening=a['opening'],exact_action_dispatched=record['executed_action']['source']=='exact_selected_modeled_q_and_opening',
            phase=(d.get('plan')or{}).get('phase_state'),
            current_derivative_binding=record.get('current_derivative_binding')))
    return dict(actual_close_dispatch_times=close,actual_reopen_dispatch_times=reopen,
        first_close_dispatch_time=close[0]if close else None,
        current_public_selected_point_distance=stats([x['current_public_selected_point_distance_m']for x in rows]),
        minimum_distance_frame=min(rows,key=lambda x:x['current_public_selected_point_distance_m'])if rows else None,
        last_FREE_selected_frame=rows[-1]if rows else None,
        selected_dispatch_with_expired_deadline_count=sum(x['seconds_until_deadline']is not None and x['seconds_until_deadline']< -1e-8 for x in rows),
        this_is_alignment_to_model_selected_point_not_unique_correct_grasp_error=True)


def metrics(native,public):
    return dict(Native_goal_ever=native['native_success_any'],Native_goal_final=native['native_success_final'],
        Native_held_final=native['final']['held'],Native_lifted_hold_final=native['final']['current_lifted_hold_passed'],
        Native_final_goal_AND_final_held=native['native_success_final']and native['final']['held'],
        Native_final_goal_AND_current_lifted_hold=native['native_success_final']and native['final']['current_lifted_hold_passed'],
        Native_ever_goal_AND_current_lifted_hold=native['strict_current_joint_condition']['final'],
        P1_lifted_hold_ever=native['p1_hold_metric']['passed'],
        public_lifted_hold_ever=public['public_lift_hold_ever'],public_lifted_hold_final=public['public_lift_hold_final'],public_goal_final=public['public_goal_complete_final'])


def evaluate(owner_path,output,expected_sha=None):
    owner,plan,freeze,inputs=metadata(owner_path,expected_sha);out=Path(output)
    if out.exists():raise FileExistsError('Fresh independent artifact required')
    prepared,source_gate=gate(owner,plan,freeze)
    # Every new case source/closure checked above. No initial snapshot/ticks were
    # read by metadata() or gate(). Only now read privileged recorded outcomes.
    from our_mani.local_mpc_control_20261103.native_task_evaluation import evaluate_states
    from scripts.evaluate_public_position_closed_cases_20261103 import summarize_public
    from scripts.evaluate_public_position_closed_cases_with_alignment_20261103 import alignment_from_records
    baseline=read(BASELINE);old={c['case_id']:c for c in baseline['cases']if c['profile']=='legacy_pilot'}
    if len(old)!=4:raise ValueError('All four old baseline cases required')
    if sha(ROOT/'our_mani/local_mpc_control_20261103/native_task_evaluation.py')!=baseline['original_native_judge']['sha256']:
        raise ValueError('Original judge must match sealed old report')
    summaries=[];pairs=[]
    for item in prepared:
        case=item['case'];spec=item['spec'];result=item['result'];b=old[spec['id']]
        if(b['task'],b['seed'])!=(spec['task'],spec['seed']):raise ValueError('Same original task/seed required')
        snapshot=read(case/'initial_snapshot.json');ticks=list(lines(case/'native_ticks.jsonl'));decisions=list(lines(case/'public_candidate_position_v72_decisions.jsonl'))
        if len(decisions)!=result['cycles']or sum(d['executor']['executed_ticks']for d in decisions)!=len(ticks):raise ValueError('All actual cycles/ticks required')
        native,_=evaluate_states(snapshot['state'],ticks)
        patched=dict(result)
        if patched.get('public_goal_completion')is None:patched['public_goal_completion']={}
        public=summarize_public(case,patched)
        metric=metrics(native,public)
        old_snapshot_path=checked(b['evidence']['initial_snapshot.json']);old_case=old_snapshot_path.parent
        old_snapshot=read(old_snapshot_path)
        comparison=initial_comparison(old_snapshot,snapshot)
        evidence=dict(item['evidence'],initial_snapshot=pointer(case/'initial_snapshot.json'),native_ticks=pointer(case/'native_ticks.jsonl'),
            decisions=pointer(case/'public_candidate_position_v72_decisions.jsonl'),public_outputs=pointer(case/'public_shadow/public_outputs.jsonl'))
        row=dict(case_id=spec['id'],task=spec['task'],seed=spec['seed'],speed_group=b['speed_group'],metrics=metric,
            original_native_hold_metrics=native,public=public,visibility=visibility(case),first_close_alignment=alignment_from_records(snapshot['state'],ticks,decisions),
            first_close_alignment_axis_semantics='Original helper: posthoc Native body axes transform selected local point; unobserved public axis prior may differ. Alignment is diagnostic, not grasp-success qualification.',
            control=decision_summary(decisions,result),public_control_details=control_details(decisions),actual_ticks=len(ticks),
            current_derivative_binding=result.get('current_derivative_binding'),evidence=evidence)
        summaries.append(row)
        old_decisions=list(lines(checked(b['evidence']['public_candidate_position_v72_decisions.jsonl'])))
        pairs.append(dict(case_id=spec['id'],task=spec['task'],seed=spec['seed'],initial_conditions=comparison,
            full_initial_snapshot_bytes_equal=old_snapshot_path.read_bytes()==(case/'initial_snapshot.json').read_bytes(),
            baseline_metrics=b['metrics'],new_metrics=metric,baseline_first_close=b['first_close_alignment'],new_first_close=row['first_close_alignment'],
            baseline_visibility=visibility(old_case),new_visibility=row['visibility'],
            baseline_public_control=control_details(old_decisions),new_public_control=row['public_control_details'],
            baseline_modes=b['control']['mode_counts'],new_modes=row['control']['mode_counts']))
    aggregate={}
    for group in('ordinary','high_speed','all'):
        chosen=[c for c in summaries if group=='all'or c['speed_group']==group]
        base=[old[c['case_id']]for c in chosen]
        aggregate[group]=dict(denominator=len(chosen),baseline={k:sum(c['metrics'][k]for c in base)for k in base[0]['metrics']},
            new={k:sum(c['metrics'][k]for c in chosen)for k in chosen[0]['metrics']})
    report=dict(schema='Cube_motion_only_same_seed_staged_closed4_vs_old4_shutdown_race_posthoc_20261104_v3',complete=True,inputs=inputs,
        source_gate=source_gate,cases=summaries,paired=pairs,aggregate=aggregate,
        intervention='New actual-clock GRU703 + derivative anchor ABI consistency fix; original old candidate/V10vision/V72strategy/physics/proof retained. No new candidate and no bounded reobserve enabled.',
        metric_contract='Native final task flag AND current original lifted hold, Native ever/final/attachment, public proof kept separate. No outcome-dependent rerun or denominator removal.',
        formal_VAL_TEST=False,full_owner_lifecycle_passed=False,staged_diagnostic_only=True,shutdown_race_preserved=True,GT_only_posthoc=True,heldout_or_TEST_read=False,NN_forward_calls=0,GPU_used=False,Native_started=False,
        helper=pointer(__file__))
    out.mkdir(parents=True,exist_ok=False)
    (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    (out/'receipt.json').write_text(json.dumps(dict(complete=True,report=pointer(out/'report.json'),helper=pointer(__file__),
        inputs=inputs,all4_retained=True,full_owner_lifecycle_passed=False,owner_completion_rewritten=False,staged_diagnostic_only=True,original_sources_modified=False,Native_started=False,NN_forward_calls=0,heldout_or_TEST_read=False),indent=2)+'\n')
    return dict(report=pointer(out/'report.json'),receipt=pointer(out/'receipt.json'),aggregate=aggregate)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--owner-plan',type=Path,default=OWNER)
    p.add_argument('--expected-sha256');p.add_argument('--output',type=Path);p.add_argument('--evaluate',action='store_true');a=p.parse_args()
    if not a.evaluate:
        _,plan,_,inputs=metadata(a.owner_plan,a.expected_sha256)
        print(json.dumps(dict(ready_to_attempt_after_Root_all4_actual_wait_signal=True,case_count=len(plan['cases']),
            actual_case_results_read=False,GT_read=False,Native_started=False,NN_forward_calls=0,inputs=inputs)));return 0
    if a.output is None:p.error('--evaluate requires fresh --output')
    print(json.dumps(evaluate(a.owner_plan,a.output,a.expected_sha256),indent=2,allow_nan=False));return 0
if __name__=='__main__':raise SystemExit(main())
