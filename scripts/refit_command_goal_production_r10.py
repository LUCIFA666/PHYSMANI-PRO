"""TRAIN-only actual700 production interaction refit; no simulation or GPU."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from our_mani.multitask import acquisition_lineage as a
from our_mani.multitask import artifact_fitting as adapter,interaction,command_goal_reference as goal,release_preflight as release


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(interaction._json(value),ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def run(prior_path,config_path,output):
    output=a.local(ROOT,output);a.require(not output.exists(),'fresh production refit namespace required');output.mkdir()
    begun=time.perf_counter();started=datetime.now(timezone.utc).isoformat();files={}
    prior=a.read(a.checked_pointer(ROOT,a.pointer(ROOT,a.local(ROOT,prior_path)),files))
    a.require(prior['actual_new_task_fits']==7 and prior['TRAIN_only'] is True,'original actual700 audit required')
    config_path=a.local(ROOT,config_path);configuration=a.read(a.checked_pointer(ROOT,a.pointer(ROOT,config_path),files))
    config=interaction.InteractionFitConfig(**configuration['interaction'])
    a.require(config.minimum_completed_parent_sources==8 and config.maximum_position_tolerance_m==.03
        and config.maximum_orientation_tolerance_rad==.15 and config.development_only is False,'unchanged original gates required')
    rule=goal.load_rule(ROOT,configuration['command_goal_reference_rule'],purpose='production_fitting')
    files.update(rule.verified_files_sha256)
    code_before={str(Path(m.__file__).resolve()):a.file_sha(m.__file__) for m in (adapter,interaction,goal,release)}
    code_before[str(Path(__file__).resolve())]=a.file_sha(__file__);files.update(code_before);tasks={}
    for task,previous in prior['tasks'].items():
        if previous.get('real_fit_performed') is not True:continue
        original_manifest=a.read(a.checked_pointer(ROOT,previous['fit_manifest'],files))
        a.require(len(original_manifest['case_proofs'])==100,'all100 original teachers required')
        sources=[];proofs={};case_proofs=[];task_files={}
        for row in original_manifest['case_proofs']:
            path=a.checked_pointer(ROOT,row['canonical_receipt'],files);record=a.read(path);entry=record['entry']
            cache=a.checked_pointer(ROOT,row['cache'],files)
            with np.load(cache,allow_pickle=False) as stored:arrays={key:stored[key].copy() for key in stored.files}
            commands=a.read(a.checked_pointer(ROOT,row['command_receipt'],files))
            case=dict(path=path,record=record,entry=entry,arrays=arrays,command_receipt=commands,
                command_receipt_verified=True,command_receipt_sha256=row['command_receipt']['sha256'])
            source,audit=adapter.interaction_source(case,command_goal_rule=rule)
            evidence=source.command_goal_reference_evidence
            if evidence is None:
                # Unconfigured tasks still reverify the same complete actual
                # raw/archive/parent/array execution; no new basis is attached.
                evidence=goal.derive_source(case,source,rule)
            task_files.update(evidence.derivation_proof['verified_files_sha256'])
            if source.command_goal_reference_evidence is not None:proofs[source.source_id]=goal.evidence_proof(evidence)
            sources.append(source);case_proofs.append(dict(row,command_goal_proof=goal.evidence_proof(evidence)))
        a.require(len(sources)==100 and len({s.grouping_source for s in sources})==100,'distinct actual100 parents required')
        directory=output/task;manifest=directory/'TRAIN_fit_manifest.json'
        value=dict(schema='r10_actual_task100_command_goal_production_kernel_diagnostic_manifest_v2',
            task=task,fit_split='train',VAL_used=False,TEST_used=False,formal_model_or_policy_release=False,
            fitting_configuration=a.pointer(ROOT,config_path),entries=[s['entry'] for s in
                [dict(entry=a.read(a.local(ROOT,r['canonical_receipt']['path']))['entry']) for r in case_proofs]],
            case_proofs=case_proofs,command_goal_reference_derivation=dict(rule=rule.pointer,
                helper=adapter.pointer(Path(goal.__file__)),derivation_schema=goal.SCHEMA,
                original_raw_metric_events_and_masks_preserved=True,per_source=proofs,
                verified_files_sha256=rule.verified_files_sha256),
            all_original100_failed_censored_and_rare_modes_retained=True,source_sha256=code_before,
            verified_files_sha256=task_files)
        # Complete source proof is frozen before any numeric fit is attempted.
        write(manifest,value);manifest_sha=a.file_sha(manifest)
        row=dict(original_teacher_sources=100,distinct_original_parents=100,fit_manifest=a.pointer(ROOT,manifest),
            formal_model_or_policy_release=False,public_full_R_support_is_separate=True)
        library=None
        try:
            library=interaction.fit_interaction_library(sources,config,TRAIN_source_manifest_path=manifest)
            report=library.metadata['completed_skill_fitting'];path=directory/'interaction_library_diagnostic.json'
            library.save(path);row.update(status='production_kernel_fit_passed_diagnostic_only',library=a.pointer(ROOT,path))
        except interaction.InteractionFittingPending as error:
            report=error.diagnostics;row.update(status='production_kernel_fit_pending_unchanged_gates',error=str(error))
        a.require(a.file_sha(manifest)==manifest_sha,'source proof changed after numeric fitting')
        # Formal support and canonical geometry are independently reconstructed
        # from actual per-source typed evidence for each admitted library.
        if library is not None:
            release._command_goal_library_metadata(library.metadata,sources,rule if proofs else None)
            counts={}
            for template in library.templates:
                key,parents,attempts=release._completion_support(sources,template)
                release._command_goal_geometry(sources,template)
                diagnostic=report['phase_fits']['|'.join(key)]
                a.require(parents==set(diagnostic['completed_parent_sources'])
                    and len(attempts)==diagnostic['completed_attempt_count'],'actual producer/consumer support mismatch')
                counts['|'.join(key)]=len(parents)
            row['actual_formal_consumer_support_and_canonical_geometry_checked']=counts
        detail=directory/'real_production_fit_report.json';write(detail,report)
        row.update(detailed_fit_report=a.pointer(ROOT,detail),failed_fit_keys=report.get('failed_fit_keys',[]),
            required_phase_failures=report.get('required_phase_failures',[]),
            phase_fits={key:{name:fit.get(name) for name in ('passed','reason','completed_parent_count','completed_attempt_count',
                'selected_lag','position_tolerance_m','orientation_tolerance_rad','duration_seconds','duration_budget_seconds')}
                for key,fit in report['phase_fits'].items()})
        tasks[task]=row;files.update(task_files);write(output/'progress.json',dict(tasks=tasks,complete=False))
    a.require(len(tasks)==7,'all7 complete original task100 datasets required')
    for name,checksum in files.items():a.require(a.file_sha(name)==checksum,'production diagnostic input changed: '+name)
    report=dict(schema='r10_actual700_command_goal_production_kernel_refit_v2',started_utc=started,
        finished_utc=datetime.now(timezone.utc).isoformat(),wall_seconds=time.perf_counter()-begun,
        prior_actual700_audit=a.pointer(ROOT,a.local(ROOT,prior_path)),configuration=a.pointer(ROOT,config_path),
        command_goal_rule=rule.pointer,source_sha256=code_before,tasks=tasks,
        actual_original_teacher_cases=700,all_failed_censored_rare_modes_and_raw_metrics_retained=True,
        actual_production_fitter_invoked_without_mocks=True,public_current_geometry_verifier_calibrated=False,
        runtime_forecast_or_closed_loop_performance_proven=False,native_GPU_or_collection_started=False,
        formal_model_or_policy_release=False,verified_files_sha256=files)
    write(output/'report.json',report);write(output/'progress.json',dict(tasks=tasks,complete=True))
    return dict(report=a.pointer(ROOT,output/'report.json'),tasks={t:r['status'] for t,r in tasks.items()})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--prior-report',required=True)
    p.add_argument('--fit-config',required=True);p.add_argument('--out',required=True);args=p.parse_args()
    print(json.dumps(run(args.prior_report,args.fit_config,args.out),ensure_ascii=False))
