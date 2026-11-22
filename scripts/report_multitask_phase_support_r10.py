"""Read-only, in-progress TRAIN completed-skill support audit for interaction BC.

This reports necessary support before fitting, not fitted endpoint tolerance,
native task success, a model release, or an acquisition-completeness gate.
"""
import argparse
from collections import Counter,defaultdict
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from our_mani.contracts import Pose
from our_mani.multitask.artifact_fitting import interaction_source
from our_mani.multitask.interaction import InteractionArtifactError,_completion_reference
from our_mani.multitask.specs import get_task_spec

VERSION='r10_observed_completed_TRAIN_skill_parent_support_audit_v1'
FORMAL_MINIMUM_COMPLETED_PARENTS=8  # Existing interaction._fit_completed_interactions hard minimum.


def utc():return datetime.now(timezone.utc).isoformat()
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def require(condition,message):
    if not condition:raise ValueError(message)
def inside(root,name):
    path=(Path(root)/name).resolve();require(path.is_relative_to(Path(root).resolve()),'phase audit input escaped project root')
    return path


def verify_canonical_case(root,path,expected,plan_sha256):
    """Bind canonical/raw receipt, exact phase/track/source bytes and clocks.

    Unrelated rendering/geometry outputs are not reread by this support report;
    the formal fitting loader later verifies the entire acquisition manifest.
    """
    path=inside(root,path);record=read(path);entry=record.get('entry',{})
    require(record.get('status')=='completed' and record.get('split')=='train'
        and record.get('case_id')==expected['case_id'] and record.get('plan_sha256')==plan_sha256
        and record.get('task')==expected['task'] and record.get('raw_episode_id')==expected['raw_episode_id']
        and record.get('parent_source_group')==expected['parent_source_group']
        and record.get('case_isolation')=='fresh_simulator_process_per_episode'
        and record.get('teacher_actions_used') is expected['teacher_used']
        and record.get('policy_history_contains_native_GT') is False,'canonical TRAIN identity/boundary differs')
    require(all(entry.get(field)==expected[field] for field in
        ('task','raw_episode_id','parent_raw_episode_id','parent_source_group','behavior','teacher_used','interaction_eligible')),
        'canonical TRAIN entry differs from fixed planned behavior/parent')
    require(entry.get('split')=='train' and entry.get('parent_source_group')=='train:'+entry['task']+':'+entry['parent_raw_episode_id'],
        'native augmentation must retain its original TRAIN parent identity')
    process=record.get('parent_process_receipt');require(isinstance(process,dict),'fresh raw child/process receipt missing')
    child_path=inside(root,process['child_case_path']);require(sha(child_path)==process['child_case_sha256'],'raw child receipt hash changed')
    require({k:v for k,v in record.items() if k!='parent_process_receipt'}==read(child_path),
        'canonical receipt differs from the exact raw child')
    require(type(record.get('process_id')) is int and record['process_id']>0
        and type(process.get('process_id')) is int and process['process_id']>0,'fresh process IDs missing')
    launcher=inside(root,process['launcher']);require(sha(launcher)==process['launcher_sha256'],'TRAIN launcher source changed')
    files=record.get('files_sha256',{});cache=inside(root,entry['cache'])
    require(files.get(entry['cache'])==entry['cache_sha256']==sha(cache),'actual body-track bytes changed')
    original=entry['source_demonstration'];source_path=inside(root,original['path'])
    require(sha(source_path)==original['sha256'],'original TRAIN parent source cache changed')
    with np.load(cache,allow_pickle=False) as stored:arrays={key:stored[key].copy() for key in stored.files}
    clock=record.get('alignment',{}).get('clock',{});ticks=clock.get('ticks');n=len(arrays['times'])
    require(type(ticks) is int and 0<=ticks<=expected['maximum_collection_ticks']
        and n==ticks+1==entry.get('frames') and clock.get('dt')==.05 and clock.get('budget_seconds')==10.
        and abs(clock.get('simulation_seconds',-1)-ticks*.05)<=1e-6
        and all(abs(record['alignment'].get(field,-1)-ticks*.05)<=2e-5
            for field in ('actual_physics_seconds','actual_task_seconds')),'actual TRAIN acquisition clock/cache differs')
    inputs={str(path):sha(path),str(child_path):sha(child_path),str(cache):sha(cache),str(launcher):sha(launcher),str(source_path):sha(source_path)}
    case=dict(path=path,record=record,entry=entry,arrays=arrays)
    if expected['teacher_used']:
        command_name=entry.get('phase_completion_events_receipt')
        require(command_name is not None and command_name==entry.get('teacher_command_receipt')
            and command_name in files,'actual command and completion ledger must be hash-bound')
        command_path=inside(root,command_name);require(sha(command_path)==files[command_name],'command/completion ledger bytes changed')
        case.update(command_receipt=read(command_path),command_receipt_sha256=files[command_name],command_receipt_verified=True)
        inputs[str(command_path)]=sha(command_path)
    return case,inputs


def classify_attempts(source,*,task_success):
    """Mirror fitter's lag-zero curve eligibility without fitting or curation."""
    stage_phase={stage:phase.name for phase in get_task_spec(source.task_name).phase_skills for stage in phase.TRAIN_expert_stages}
    events={event.origin_command_id:event for event in source.completion_events}
    excluded={row['command_id']:row for row in source.excluded_completion_events}
    ledger={row['command_id']:row for row in source.command_ledger}
    command_ids=set(map(int,source.command_ids[source.command_ids>=0]))|set(ledger)
    rows=[]
    for cid in sorted(command_ids):
        indices=np.flatnonzero(source.command_ids==cid);command=ledger.get(cid,{});event=events.get(cid)
        row=dict(source_id=source.source_id,parent_source_group=source.grouping_source,command_id=cid,
            phase=command.get('raw_phase'),waypoint_command_done='command_completed_at' in command,
            native_task_success=bool(task_success),completed_skill_support=False,right_censored=False,
            raw_rows_preserved=True,reference_partition=None)
        if event is None:
            bad=excluded.get(cid)
            begin=float(command.get('generated_at',source.times[indices[0]] if len(indices) else source.times[0]))
            end=float(command.get('cutoff_at',command.get('last_executed_at',source.times[-1])))
            row.update(status='completed_event_label_excluded' if bad else 'right_censored_incomplete_attempt',
                right_censored=bad is None,reason=bad['reason'] if bad else 'no_explicit_successful_skill_completion_event',
                observed_duration_lower_bound_seconds=max(0.,end-begin))
            rows.append(row);continue
        row['phase']=event.phase;binding=event.binding
        row.update(completion_evidence=event.evidence,completed_duration_seconds=event.attempt_end_time-event.attempt_start_time,
            reference_partition=[event.phase,binding.anchor_id,binding.reference_role,binding.reference_kind,
                binding.manipulated_role,event.reference_mode])
        if (not len(indices) or event.attempt_end_time>source.times[-1]+1e-8 or
                any(stage_phase.get(source.stages[i])!=event.phase or source.references[i]!=binding
                    for i in indices if source.references[i] is not None)):
            row.update(status='invalid_completed_event',reason='completion interval, executed rows, phase or reference differs')
            rows.append(row);continue
        selected=[i for i in indices if source.references[i]==binding
            and event.attempt_start_time-1e-8<=source.times[i]<event.attempt_end_time-1e-8]
        usable=0
        try:
            _completion_reference(source,event,0.)
            for i in selected:
                try:
                    if binding.reference_kind not in ('world','phase_origin'):
                        source.body_tracks[binding.reference_id].at(source.times[i],
                            orientation_required=binding.reference_kind=='observed_se3',same_mode_as=event.reference_mode,
                            not_before=source.times[i])
                    Pose.from_xyzw(source.curve_tcp_xyzw[i]);usable+=1
                except (InteractionArtifactError,KeyError,ValueError):continue
        except (InteractionArtifactError,KeyError,ValueError):pass
        row.update(status='completed_skill_with_usable_lag_zero_curve' if usable else 'completed_event_without_usable_reference_curve',
            completed_skill_support=usable>0,lag_zero_usable_executed_rows=usable,
            reason='explicit successful skill event and actual measured TCP curve' if usable else 'no usable lag-zero reference/curve rows')
        rows.append(row)
    return rows


def summarize_attempts(attempts,phases):
    result={}
    for phase in phases:
        rows=[row for row in attempts if row['phase']==phase];completed=[row for row in rows if row['completed_skill_support']]
        parents=sorted({row['parent_source_group'] for row in completed});partitions={}
        for key in sorted({tuple(row['reference_partition']) for row in completed}):
            selected=[row for row in completed if tuple(row['reference_partition'])==key]
            groups=sorted({row['parent_source_group'] for row in selected})
            partitions['|'.join(key)]=dict(reference_partition=list(key),completed_attempts=len(selected),
                distinct_completed_TRAIN_parents=len(groups),parent_source_groups=groups,
                existing_formal_minimum_completed_parents=FORMAL_MINIMUM_COMPLETED_PARENTS,
                necessary_parent_support_met=len(groups)>=FORMAL_MINIMUM_COMPLETED_PARENTS)
        result[phase]=dict(completed_skill_attempts=len(completed),distinct_completed_TRAIN_parents=len(parents),
            completed_parent_source_groups=parents,
            completed_skill_attempts_from_eventually_failed_tasks=sum(not row['native_task_success'] for row in completed),
            waypoint_command_done_without_usable_completed_skill=sum(row['waypoint_command_done'] and not row['completed_skill_support'] for row in rows),
            right_censored_incomplete_attempts=sum(row['right_censored'] for row in rows),
            right_censored_distinct_parents=len({row['parent_source_group'] for row in rows if row['right_censored']}),
            completed_but_label_or_curve_excluded_attempts=sum(row['status'] in ('completed_event_label_excluded','completed_event_without_usable_reference_curve') for row in rows),
            invalid_completed_events=sum(row['status']=='invalid_completed_event' for row in rows),
            attempt_status_counts=dict(Counter(row['status'] for row in rows)),
            existing_formal_minimum_completed_parents=FORMAL_MINIMUM_COMPLETED_PARENTS,
            aggregate_phase_parent_support_met=len(parents)>=FORMAL_MINIMUM_COMPLETED_PARENTS,
            reference_partitions=partitions,
            phase_aggregate_does_not_replace_per_partition_support=True,formal_fitting_acceptance_proven=False)
    return result


def original_prior_summary(root,path,tasks):
    path=inside(root,path);prior=read(path)
    require(prior.get('version')=='r10_original_TRAIN_action_phase_priors_v1' and prior.get('split')=='train'
        and prior.get('sources_read')==1400 and prior.get('VAL_expert_arrays_read') is False
        and prior.get('TEST_read') is False and prior.get('future_debug_target_XYZ_used') is False
        and prior.get('artifact_sha256')==digest({k:v for k,v in prior.items() if k!='artifact_sha256'}),
        'original1400 TRAIN-only phase prior identity differs')
    inputs={str(path):sha(path)};counts=Counter();seen=set()
    for source in prior['sources']:
        group=source['source_group'];require(group not in seen and group.startswith('train:'),'duplicate/non-TRAIN original prior parent')
        seen.add(group);task=group.split(':')[1];counts[task]+=1
        cached=inside(root,source['path']);require(sha(cached)==source['sha256'],'original prior cache changed')
        inputs[str(cached)]=sha(cached)
    require(len(seen)==1400 and all(counts[task]==100 for task in tasks),'original TRAIN parent coverage differs')
    return dict(source_parent_count=1400,per_task_original_parent_count=dict(counts),
        per_task_phase_prior_parent_count={task:{phase:row['source_count'] for phase,row in prior['per_task'][task].items()} for task in tasks},
        actual_completed_skill_event_supervision=False,contributes_to_native_eight_parent_support=False),inputs


def run(plan_path,collection,output,prior_path):
    started=time.perf_counter();started_utc=utc();plan_path=inside(ROOT,plan_path);plan=read(plan_path)
    require(plan.get('split')=='train' and plan.get('plan_sha256')==digest({k:v for k,v in plan.items() if k!='plan_sha256'}),'immutable full TRAIN plan identity differs')
    cases=plan['cases'];declared={case['case_id']:case for case in cases}
    require(len(declared)==len(cases) and all(case['split']=='train' for case in cases),'unique planned TRAIN case identities required')
    tasks=sorted({case['task'] for case in cases});collection=inside(ROOT,collection)
    canonical_paths=sorted((collection/'cases/train').glob('*/*/case.json'))
    by_task={task:[] for task in tasks};acquired=Counter();errors=[];attempts=[];verified_ids=set()
    inputs={str(plan_path):sha(plan_path),str(Path(__file__).resolve()):sha(__file__)}
    original,original_inputs=original_prior_summary(ROOT,prior_path,tasks);inputs.update(original_inputs)
    for path in canonical_paths:
        try:
            raw=read(path);expected=declared.get(raw.get('case_id'))
            require(expected is not None,'canonical case is not declared in this fixed full TRAIN plan')
            require(expected['case_id'] not in verified_ids,'duplicate canonical acquisition identity')
            case,used=verify_canonical_case(ROOT,path,expected,plan['plan_sha256']);inputs.update(used)
            entry=case['entry'];acquired[(entry['task'],'teacher' if expected['teacher_used'] else 'hold')]+=1
            verified_ids.add(expected['case_id'])
            if expected['teacher_used']:
                source,source_audit=interaction_source(case)
                outcome=case['record']['alignment'].get('native_outcome',{})
                rows=classify_attempts(source,task_success=outcome.get('success') is True)
                for row in rows:row['task']=entry['task'];row['canonical_case_id']=expected['case_id']
                by_task[entry['task']].extend(rows);attempts.extend(rows)
        except (ValueError,KeyError,TypeError,OSError,InteractionArtifactError) as error:
            errors.append(dict(path=str(path.relative_to(ROOT)),error_type=type(error).__name__,error=str(error)))
    expected_counts=Counter((case['task'],'teacher' if case['teacher_used'] else 'hold') for case in cases)
    task_reports={}
    for task in tasks:
        spec=get_task_spec(task);phase_rows=summarize_attempts(by_task[task],[phase.name for phase in spec.phase_skills])
        for phase,row in phase_rows.items():
            row['optional_after_native_completion']=next(p.optional_after_native_completion for p in spec.phase_skills if p.name==phase)
            row['original_demonstration_phase_prior_parent_count']=original['per_task_phase_prior_parent_count'][task].get(phase,0)
        task_reports[task]=dict(expected_teacher_acquisitions=expected_counts[(task,'teacher')],observed_teacher_acquisitions=acquired[(task,'teacher')],
            expected_hold_acquisitions=expected_counts[(task,'hold')],observed_hold_acquisitions=acquired[(task,'hold')],phases=phase_rows)
    complete=len(verified_ids)==len(declared) and not errors
    imported=(ROOT/'our_mani/multitask/artifact_fitting.py',ROOT/'our_mani/multitask/interaction.py',ROOT/'our_mani/multitask/specs.py')
    inputs.update({str(path):sha(path) for path in imported})
    result=dict(version=VERSION,audit_complete=True,scope='observed_so_far' if not complete else 'full_acquisition_observed',
        fitting_status='support_diagnostic_only',full_TRAIN_acquisition_complete=complete,
        acquisition_status='complete' if complete else 'pending',snapshot_started_utc=started_utc,snapshot_finished_utc=utc(),
        plan_sha256=plan['plan_sha256'],expected_native_TRAIN_cases=len(declared),
        canonical_cases_seen_at_snapshot_start=len(canonical_paths),verified_canonical_acquisitions=len(verified_ids),
        missing_case_count=len(set(declared)-verified_ids),case_audit_errors=errors,
        original_demonstration_phase_priors=original,tasks=task_reports,attempts=attempts,
        TRAIN_only=True,VAL_used=False,TEST_used=False,native_simulator_started=False,GPU_used=False,
        outcomes_do_not_filter_earlier_completed_skills=True,waypoint_done_is_completed_skill=False,
        incomplete_attempts_are_right_censored=True,completed_label_invalid_is_separate_from_right_censoring=True,
        original_demonstration_prior_counts_are_not_native_completed_skill_support=True,
        configured_existing_formal_minimum_completed_parents=FORMAL_MINIMUM_COMPLETED_PARENTS,
        support_requirement_source='our_mani/multitask/interaction.py:_fit_completed_interactions',
        aggregate_phase_counts_do_not_replace_reference_partition_counts=True,
        lag_grid_tolerance_crossfit_and_final_library_acceptance_not_performed=True,
        canonical_phase_track_source_and_child_hashes_verified=True,unrelated_geometry_render_output_hashes_not_reverified=True,
        formal_fitting_loader_must_still_verify_all_manifest_outputs=True,
        inputs_sha256=inputs,wall_seconds=time.perf_counter()-started)
    output=inside(ROOT,output);require(not output.exists(),'preserve prior phase support snapshots; choose a fresh output filename')
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    lines=[VERSION,f"{result['scope']}; acquisition {len(verified_ids)}/{len(declared)}; case audit errors {len(errors)}",
        'Completed TRAIN parent support is necessary only; no fitting/VAL/TEST/native task success claim.',
        'Original1400 phase priors are reported separately and never added to native completed-skill counts.']
    for task,row in task_reports.items():
        lines.append(task+' teacher '+str(row['observed_teacher_acquisitions'])+'/'+str(row['expected_teacher_acquisitions'])+
            ': '+', '.join(phase+'='+str(p['distinct_completed_TRAIN_parents'])+' parents ('+str(p['right_censored_incomplete_attempts'])+' censored)'
                for phase,p in row['phases'].items()))
    output.with_suffix('.txt').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(scope=result['scope'],verified=len(verified_ids),expected=len(declared),errors=len(errors),
        report=str(output.relative_to(ROOT)),sha256=sha(output),wall_seconds=result['wall_seconds'])))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',default='artifacts/framework/multitask_r10/TRAIN_full_collection_plan_v11.json')
    parser.add_argument('--collection',default='artifacts/framework/multitask_r10/TRAIN_full_collection_v11')
    parser.add_argument('--original-prior',default='artifacts/framework/multitask_r10/original_TRAIN_action_phase_priors.json')
    parser.add_argument('--out',default='artifacts/framework/multitask_r10/phase_support/observed_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'.json')
    args=parser.parse_args();run(args.plan,args.collection,args.out,args.original_prior)
