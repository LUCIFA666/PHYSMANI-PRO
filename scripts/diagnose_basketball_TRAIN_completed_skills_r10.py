"""TRAIN-only basketball teacher100 support and real CPU fit diagnostics.

No VAL/TEST reads, native launch, roster resampling or frozen source changes.
"""
from collections import Counter
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
for folder in (ROOT,ROOT/'scripts'):
    if str(folder) not in sys.path:sys.path.insert(0,str(folder))
from our_mani.multitask import acquisition_lineage as lineage
from our_mani.multitask.artifact_fitting import interaction_source
from our_mani.multitask.interaction import InteractionFitConfig,InteractionFittingPending,InteractionArtifactError,InteractionLibrary,fit_interaction_library
from our_mani.multitask.training_view import validate_arrays
from our_mani.multitask.specs import get_task_spec
from report_multitask_phase_support_r10 import verify_canonical_case,classify_attempts,summarize_attempts

BASE=ROOT/'artifacts/framework/multitask_r10'
OUT=BASE/'basketball_TRAIN_completed_skill_diagnostic_v1'
REGISTRATIONS=BASE/'TRAIN_composite_registration_preflight_v3/registrations.json'
CONFIG=ROOT/'config/framework/multitask_r10_artifact_fit_v1.json'
TASKS=('moving_basketball_in_hoop','moving_basketball_in_hoop_high_speed')


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def main():
    require=lineage.require;require(not OUT.exists(),'fresh targeted TRAIN diagnostic output required')
    begun=time.perf_counter();files={};registrations=lineage.read(lineage.checked_pointer(ROOT,lineage.pointer(ROOT,REGISTRATIONS),files))
    require(len(registrations)==7,'actual v3 seven-plan chain required')
    plans=[];by_sha={}
    for index,registration in enumerate(registrations):
        current=lineage.load_archived_plan(ROOT,registration,files);key=current['plan']['plan_sha256']
        require(key not in by_sha,'duplicate actual execution plan')
        if index:lineage.verify_recovery_transition(ROOT,by_sha[registration['predecessor_plan_sha256']],current,files)
        by_sha[key]=current;plans.append(current)
    base=plans[0]['plan'];require(all(plan['plan']['cases']==base['cases'] for plan in plans),'actual recovery case identity changed')
    config=lineage.read(lineage.checked_pointer(ROOT,lineage.pointer(ROOT,CONFIG),files));fit_config=InteractionFitConfig(**config['interaction'])
    require(lineage.file_sha(CONFIG)=='5c6a74fcf10f559423d320911a7e58f6f600aa7061334ebfa2d76e64f0b7d008'
        and fit_config.minimum_completed_parent_sources==8 and fit_config.maximum_position_tolerance_m==.03
        and fit_config.maximum_orientation_tolerance_rad==.15 and fit_config.development_only is False,'fixed production fit configuration differs')
    OUT.mkdir();reports={};errors=[]
    for task in TASKS:
        expected=[case for case in base['cases'] if case['task']==task and case['teacher_used']]
        require(len(expected)==100 and len({case['parent_source_group'] for case in expected})==100
            and all(case['replica']==case['behavior_replica']==0 for case in expected),'original100 K1 teacher roster required')
        cases=[];sources=[];attempts=[];case_rows=[];missing=[];pending=[];task_errors=[]
        for case in expected:
            cid=case['case_id'];hits=[(plan,plan['collection']/'cases/train'/task/cid/'case.json') for plan in plans]
            hits=[item for item in hits if item[1].is_file()]
            if not hits:missing.append(cid);continue
            if len(hits)!=1:task_errors.append(dict(case_id=cid,error='duplicate or replaced canonical across actual plans'));continue
            actual,path=hits[0]
            try:
                record=lineage.read(path);process=record['parent_process_receipt']
                attempt_path=Path(lineage.command_argument(process['command'],'--output')).parent
                parent_path=attempt_path/'parent_receipt.json'
                if not parent_path.is_file() or lineage.read(parent_path).get('status')!='completed':
                    pending.append(dict(case_id=cid,reason='raw parent commit pending'));continue
                completion=lineage.verify_completed(ROOT,dict(case_id=cid,canonical_receipt=lineage.pointer(ROOT,path),
                    parent_receipt=lineage.pointer(ROOT,parent_path)),actual,case,files)
                preceding=[]
                for plan in plans:
                    for failed_path in (plan['collection']/'infrastructure_attempts'/cid).glob('*/parent_receipt.json'):
                        failed=lineage.read(failed_path)
                        if failed.get('status')!='infrastructure_error':continue
                        ptr=lineage.pointer(ROOT,failed_path);lineage.verify_attempt_parent(ROOT,ptr,plan,cid,files,status='infrastructure_error')
                        log=failed_path.with_name('native.log');lineage.checked_pointer(ROOT,lineage.pointer(ROOT,log),files)
                        require(lineage.instant(failed['finished_utc'])<=lineage.instant(record['started_utc']),'infra attempt follows completed teacher')
                        preceding.append(dict(actual_plan_sha256=plan['plan']['plan_sha256'],parent_receipt=ptr,native_log=lineage.pointer(ROOT,log),
                            started_utc=failed['started_utc'],finished_utc=failed['finished_utc']))
                preceding.sort(key=lambda row:(lineage.instant(row['finished_utc']),row['parent_receipt']['path']))
                if actual!=plans[0]:
                    previous=by_sha[actual['registration']['predecessor_plan_sha256']]
                    lineage.verify_recovery_execution(ROOT,completion,actual,previous,preceding,files)
                parsed,phase_inputs=verify_canonical_case(ROOT,path,case,actual['plan']['plan_sha256']);files.update(phase_inputs)
                stats=validate_arrays(parsed['arrays'],parsed['entry'],parsed['record'])
                source,adapter_audit=interaction_source(parsed);classified=classify_attempts(source,task_success=record['alignment']['native_success_latched'])
                for row in classified:row.update(task=task,case_id=cid)
                raw_commands=parsed['command_receipt']['commands'];zero_actions=int(parsed['arrays']['teacher_action_valid'].sum())==0
                case_rows.append(dict(case_id=cid,parent_source_group=case['parent_source_group'],actual_plan_sha256=actual['plan']['plan_sha256'],
                    canonical_receipt=lineage.pointer(ROOT,path),raw_child_receipt=completion['child_receipt'],parent_receipt=lineage.pointer(ROOT,parent_path),
                    prior_infrastructure_attempts=preceding,teacher_action_rows=int(parsed['arrays']['teacher_action_valid'].sum()),
                    command_count=len(raw_commands),zero_command=len(raw_commands)==0,zero_action=zero_actions,
                    raw_completion_events=len(parsed['command_receipt']['phase_completion_events']),
                    actual_ticks=record['alignment']['clock']['ticks'],native_success=record['alignment']['native_success_latched'],
                    native_terminal_failure=record['alignment']['native_terminal_failure_latched'],
                    interaction_adapter=adapter_audit,causal_windows_diagnostic_only=stats))
                cases.append(parsed);sources.append(source);attempts.extend(classified)
            except (ValueError,KeyError,TypeError,OSError) as error:
                task_errors.append(dict(case_id=cid,path=str(path.relative_to(ROOT)),error_type=type(error).__name__,error=str(error)))
        full=len(cases)==100 and not missing and not pending and not task_errors
        spec=get_task_spec(task);support=summarize_attempts(attempts,[phase.name for phase in spec.phase_skills])
        for phase,row in support.items():row['optional_after_native_completion']=next(skill.optional_after_native_completion for skill in spec.phase_skills if skill.name==phase)
        task_report=dict(expected_teacher_cases=100,verified_committed_teacher_cases=len(cases),full_100_teacher_observed=full,
            distinct_original_teacher_parents=len({case['entry']['parent_source_group'] for case in cases}),missing_case_ids=missing,
            pending_parent_commits=pending,errors=task_errors,zero_command_cases=[row['case_id'] for row in case_rows if row['zero_command']],
            zero_action_cases=[row['case_id'] for row in case_rows if row['zero_action']],
            phase_usable_lag0_support=support,attempts=attempts,case_receipts_and_adapter_diagnostics=case_rows,
            completed_skill_support_from_eventually_failed_tasks_retained=True,
            incomplete_attempts_preserved_right_censored=True,task_success_does_not_select_fit_sources=True)
        if full:
            manifest_path=OUT/task/'TRAIN_task_teacher_fit_manifest.json'
            write(manifest_path,dict(schema='r10_actual_one_task100_teacher_diagnostic_fit_manifest_v1',fit_split='train',VAL_used=False,TEST_used=False,
                diagnostic_only=True,complete_task_teacher_coverage=True,expected_teacher_cases=100,entries=[case['entry'] for case in cases],
                case_receipts=[dict(case_id=row['case_id'],**row['canonical_receipt'],actual_plan_sha256=row['actual_plan_sha256']) for row in case_rows],
                actual_registration_list=lineage.pointer(ROOT,REGISTRATIONS),configuration=lineage.pointer(ROOT,CONFIG),
                actual_plan_sha256_values=sorted({row['actual_plan_sha256'] for row in case_rows}),
                full1400_TRAIN_or_formal_release_proven=False))
            fitting_started=time.perf_counter()
            try:
                library=fit_interaction_library(sources,fit_config,TRAIN_source_manifest_path=manifest_path)
                library.payload['metadata'].update(diagnostic_only=True,complete_task_teacher_case_count=100,
                    full1400_TRAIN_or_formal_release_proven=False,formal_comparison_ready=False,
                    scope='all100 original teacher cases of this task; not all14 task full-data artifact release')
                library=InteractionLibrary(library.payload);artifact=OUT/task/'interaction_diagnostic.json';library.save(artifact)
                fitting=dict(status='task100_real_CPU_fit_passed_diagnostic_only',artifact=lineage.pointer(ROOT,artifact),
                    completed_skill_fitting=library.metadata['completed_skill_fitting'],templates=len(library.templates),
                    formal_comparison_ready=False,full1400_release_proven=False)
            except (InteractionFittingPending,InteractionArtifactError,ValueError) as error:
                fitting=dict(status='task100_real_CPU_fit_pending',error_type=type(error).__name__,error=str(error),
                    diagnostics=getattr(error,'diagnostics',None),thresholds_relaxed=False,formal_comparison_ready=False)
            fitting['CPU_seconds']=time.perf_counter()-fitting_started
            task_report['real_CPU_fit']=fitting;write(OUT/task/'real_CPU_fit_report.json',fitting)
        else:task_report['real_CPU_fit']=dict(status='not_started_incomplete100_teacher_commit_or_audit',real_fit_performed=False)
        reports[task]=task_report;errors.extend(task_errors)
    # New launch metadata binds v3 seven registrations; previous plans remain immutable.
    old_launch=BASE/'launch_preparation_v2/launch_plan.json';launch=lineage.read(old_launch)
    launch['schema']='r10_training_launch_registration_binding_update_v3';launch['prior_launch_plan']=lineage.pointer(ROOT,old_launch)
    launch['known_registrations']=lineage.pointer(ROOT,REGISTRATIONS);launch['registrations_must_include_any_future_real_recovery_plans']=True
    launch['actual_training_launched']=False
    for job in launch['command_jobs']:
        job['argv']=[value.replace('TRAIN_composite_registration_preflight_v2/registrations.json','TRAIN_composite_registration_preflight_v3/registrations.json') for value in job['argv']]
    write(OUT/'launch_plan_v3_registration_binding.json',launch)
    for source in (Path(__file__),ROOT/'scripts/report_multitask_phase_support_r10.py',ROOT/'our_mani/multitask/artifact_fitting.py',ROOT/'our_mani/multitask/interaction.py'):
        lineage.checked_pointer(ROOT,lineage.pointer(ROOT,source),files)
    for name,checksum in files.items():require(lineage.file_sha(name)==checksum,'targeted committed teacher input changed during diagnostic')
    result=dict(schema='r10_basketball_original100_teacher_completed_skill_diagnostic_v1',created_utc=datetime.now(timezone.utc).isoformat(),
        actual_registrations=lineage.pointer(ROOT,REGISTRATIONS),actual_registration_count=7,fixed_fit_configuration=lineage.pointer(ROOT,CONFIG),
        targeted_original_tasks=list(TASKS),TRAIN_only=True,VAL_or_TEST_read=False,native_or_GPU_started=False,
        scope='only two task families original100 teacher cases each; strict source archive/transition/raw/array audits',
        tasks=reports,case_audit_errors=errors,full_live_composite_constructed=False,full1400_TRAIN_or_formal_release_claimed=False,
        additional_training_replicas_launched=False,original_K1_cases_rewritten=False,
        launch_v3_seven_registration_metadata=lineage.pointer(ROOT,OUT/'launch_plan_v3_registration_binding.json'),
        verified_files_sha256=files,wall_seconds=time.perf_counter()-begun)
    write(OUT/'report.json',result)
    lines=['TRAIN-only original100 basketball teacher support/real CPU fit diagnostic; no native/GPU launched.']
    for task,row in reports.items():
        lines.append(task+': verified'+str(row['verified_committed_teacher_cases'])+'/100; full100='+str(row['full_100_teacher_observed'])+
            '; '+','.join(phase+' usablelag0parents='+str(support['distinct_completed_TRAIN_parents']) for phase,support in row['phase_usable_lag0_support'].items())+
            '; realCPUfit='+row['real_CPU_fit']['status'])
    lines.extend(['Every zero-action/zero-command/censored original teacher case remains audited; no outcome source selection.',
        'Existing min8/.03m/.15rad remains unchanged. Full100 task diagnostic is not full1400 or a formal model release.',
        'Launch registration binding updated to actualv3 seven plans; previous launch artifacts retained.'])
    (OUT/'report.txt').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(report=lineage.pointer(ROOT,OUT/'report.json'),tasks={task:dict(teachers=row['verified_committed_teacher_cases'],
        full100=row['full_100_teacher_observed'],deposit=row['phase_usable_lag0_support']['deposit']['distinct_completed_TRAIN_parents'],
        fit=row['real_CPU_fit']['status']) for task,row in reports.items()})))


if __name__=='__main__':main()
