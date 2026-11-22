"""Execute the draft typed command-goal helper on actual complete task100 data."""
import argparse
from dataclasses import asdict
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from our_mani.multitask import acquisition_lineage as a
from our_mani.multitask import artifact_fitting,interaction,command_goal_reference as goal


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def run(prior_path,rule_path,output):
    started=datetime.now(timezone.utc).isoformat();begun=time.perf_counter()
    output=a.local(ROOT,output);a.require(not output.exists(),'fresh full-curve diagnostic namespace required');output.mkdir()
    files={};prior=a.read(a.checked_pointer(ROOT,a.pointer(ROOT,a.local(ROOT,prior_path)),files))
    a.require(prior['actual_new_task_fits']==7 and prior['TRAIN_only'] is True,'previous actual700 full-raw task audit required')
    config_path=a.checked_pointer(ROOT,prior['configuration'],files)
    config=interaction.InteractionFitConfig(**a.read(config_path)['interaction'])
    rule=goal.load_rule(ROOT,a.pointer(ROOT,a.local(ROOT,rule_path)),purpose='CPU_diagnostic')
    files.update(rule.verified_files_sha256)
    for path,checksum in prior['source_sha256'].items():
        a.require(a.file_sha(path)==checksum,'original strict raw/fitter source changed before draft diagnostic')
    code_before={str(Path(module.__file__).resolve()):a.file_sha(module.__file__)
        for module in (goal,interaction,artifact_fitting)}
    code_before[str(Path(__file__).resolve())]=a.file_sha(__file__);files.update(code_before)
    tasks={}
    for task,previous in prior['tasks'].items():
        if previous.get('real_fit_performed') is not True:continue
        manifest=a.read(a.checked_pointer(ROOT,previous['fit_manifest'],files))
        baseline=a.read(a.checked_pointer(ROOT,previous['detailed_fit_report'],files))['completed_skill_fitting']
        a.require(len(manifest['case_proofs'])==100,'complete original task100 required')
        sources=[];proofs=[];per_source=[]
        for row in manifest['case_proofs']:
            path=a.checked_pointer(ROOT,row['canonical_receipt'],files);record=a.read(path);entry=record['entry']
            cache=a.checked_pointer(ROOT,row['cache'],files)
            with np.load(cache,allow_pickle=False) as stored:arrays={key:stored[key].copy() for key in stored.files}
            commands=a.read(a.checked_pointer(ROOT,row['command_receipt'],files))
            case=dict(path=path,record=record,entry=entry,arrays=arrays,command_receipt=commands,
                command_receipt_verified=True,command_receipt_sha256=row['command_receipt']['sha256'])
            source,audit=artifact_fitting.interaction_source(case)
            evidence=goal.derive_source(case,source,rule)
            sources.append(source);proofs.append(evidence);files.update(evidence.derivation_proof['verified_files_sha256'])
            per_source.append(dict(source_id=source.source_id,parent=source.grouping_source,
                raw_bindings=evidence.raw_bindings,original_masks_sha256=evidence.original_masks_sha256,
                original_command_ledger_sha256=evidence.original_command_ledger_sha256,
                original_completion_events_sha256=evidence.original_completion_events_sha256,
                pending_commands=list(evidence.pending_commands),
                references=[dict(command_id=value.command_id,issued_at=value.issued_at,
                    original_binding=asdict(value.binding),original_reference_mode=value.reference_mode,
                    intent_kind=value.intent_kind,intent_reference_xyzw=value.intent_reference_xyzw.tolist(),
                    original_issue_reference_xyzw=value.original_issue_reference_xyzw.tolist(),
                    original_completion_metric=value.original_completion_metric,provenance=value.provenance)
                    for value in evidence.references],original_adapter_audit=audit))
        result=goal.fit_diagnostic(sources,proofs,config)
        # Every unaffected partition must reproduce the actual original full
        # measured-curve lag selection and parent residuals, not just endpoints.
        reproduced=[]
        for key,row in result['phase_fits'].items():
            if key.endswith('|'+goal.INTENT_BASIS):continue
            previous_fit=baseline['phase_fits'][key]
            a.require(row['completed_parent_count']==previous_fit['completed_parent_count']
                and row['passed']==previous_fit['passed'],'unconfigured partition coverage/acceptance changed')
            for name in ('selected_lag','position_tolerance_m','orientation_tolerance_rad'):
                if name in previous_fit:a.require(abs(row[name]-previous_fit[name])<1e-12,'unconfigured actual kernel differs: '+key+':'+name)
            if 'selected_lag' in previous_fit:
                original_cv=next(item['grouped_TRAIN_CV_error'] for item in previous_fit['candidates']
                    if item.get('valid') and item['lag']==previous_fit['selected_lag'])
                a.require(abs(row['grouped_TRAIN_CV_error']-original_cv)<1e-12,'unconfigured full-curve CV differs')
            reproduced.append(key)
        task_dir=output/task
        evidence_path=task_dir/'TRAIN_command_goal_derivation.json'
        write(evidence_path,dict(schema=goal.SCHEMA,rule=rule.pointer,task=task,original_teacher_cases=100,
            original_distinct_parents=100,original100_source_objects_and_raw_events_unchanged=True,per_source=per_source,
            helper_source=a.pointer(ROOT,Path(goal.__file__)),production_enabled=False))
        details=task_dir/'real_full_curve_fit_report.json';write(details,result)
        tasks[task]=dict(original_teacher_cases=100,original_distinct_parents=100,
            derivation=a.pointer(ROOT,evidence_path),actual_full_curve_fit=a.pointer(ROOT,details),
            all_required_phases_and_observed_partitions_passed=result['all_required_phases_and_observed_partitions_passed'],
            required_phase_failures=result['required_phase_failures'],failed_fit_keys=result['failed_fit_keys'],
            unconfigured_actual_kernel_reproduced_to_1e_minus12=reproduced,
            phase_fits={key:{field:row.get(field) for field in ('passed','reason','completed_parent_count',
                'completed_attempt_count','curve_rows_including_metric_terminal','selected_lag','grouped_TRAIN_CV_error',
                'position_tolerance_m','orientation_tolerance_rad','duration_seconds','duration_budget_seconds')}
                for key,row in result['phase_fits'].items()})
        write(output/'progress.json',dict(tasks=tasks,complete=False,native_or_GPU_started=False))
    a.require(len(tasks)==7,'all7 original task100 datasets required')
    for path,checksum in files.items():a.require(a.file_sha(path)==checksum,'draft full-curve input/source changed: '+path)
    report=dict(schema='r10_actual700_command_goal_full_curve_CPU_draft_v1',started_utc=started,
        finished_utc=datetime.now(timezone.utc).isoformat(),wall_seconds=time.perf_counter()-begun,
        prior_full_raw_audit=a.pointer(ROOT,a.local(ROOT,prior_path)),rule=rule.pointer,
        original_numeric_fit_configuration=a.pointer(ROOT,config_path),tasks=tasks,
        actual_original_teacher_cases=700,all_original_sources_failed_censored_and_rare_modes_retained=True,
        source_sha256=code_before,verified_files_sha256=files,production_enabled=False,
        full_real_measured_curves_parent_CV_not_endpoint_only=True,native_GPU_or_new_collection_started=False,
        runtime_forecast_accuracy_or_closed_loop_performance_proven=False,formal_model_or_policy_release=False)
    write(output/'report.json',report);write(output/'progress.json',dict(tasks=tasks,complete=True,native_or_GPU_started=False))
    return dict(report=a.pointer(ROOT,output/'report.json'),tasks=tasks,native_or_GPU_started=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prior-report',required=True);parser.add_argument('--rule',required=True);parser.add_argument('--out',required=True)
    args=parser.parse_args();print(json.dumps(run(args.prior_report,args.rule,args.out),ensure_ascii=False))
