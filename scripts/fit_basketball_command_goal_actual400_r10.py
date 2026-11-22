"""Actual original200+aux200 production fit with two verified TRAIN evidence views."""
import argparse
from collections import Counter
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from our_mani.multitask import acquisition_lineage as a,semantic_acquire as semantic,interaction,release_preflight as release
from our_mani.multitask import artifact_fitting as adapter,teacher_supplement as supplement
from our_mani.multitask.reference_time import load_rule as load_reference_rule
from our_mani.multitask import command_goal_reference as goal,basketball_command_goal_reference as basketball


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(interaction._json(value),ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def run(configuration_path,output):
    begun=time.perf_counter();output=a.local(ROOT,output);a.require(not output.exists(),'fresh actual400 fit namespace required');output.mkdir()
    files={};configuration_path=a.local(ROOT,configuration_path);configuration=a.read(a.checked_pointer(ROOT,a.pointer(ROOT,configuration_path),files))
    input_closure=release._Inputs(ROOT)
    for field in ('fitting_core_source_snapshot','fitting_producer_source_snapshot'):
        a.require(field in configuration,'explicit current core/producer execution snapshot required')
        input_closure.source_snapshot(configuration[field])
    evidence=a.read(a.checked_pointer(ROOT,configuration['semantic_auxiliary_complete_audit'],files))
    a.require(evidence['auxiliary_complete200'] is True and evidence['original_Context_primary_K1_changed'] is False,
        'explicit complete200 audit with unchanged primary Context view required')
    approved=configuration['fixed_auxiliary_teacher_fit_scope']['explicit_auxiliary_registration']
    a.require(approved['sha256']=='e210884ed2520e348441ac16e77feccc0cc456bd7e32f4bb4f35a2abc17ae0e8','exact approved fixed200 registry required')
    registration=a.read(a.checked_pointer(ROOT,approved,files))
    a.require(registration['execution']['plan']['sha256']=='073d3d08906e3e045a74cb25003dc426becef2dbde1830f514549965dec8bdc1'
        and registration['execution']['source_snapshot']['sha256']=='7504ccb379fa74d86dc9d4931c6e839e185173d426908e62f23801ebcbab86e0',
        'approved aux actual plan/archive differs')
    actual_proof=a.read(a.checked_pointer(ROOT,evidence['auxiliary_audit'],files))
    a.checked_pointer(ROOT,evidence['auxiliary_manifest'],files)
    a.require(actual_proof['audit_passed'] is True and actual_proof['complete'] is True
        and actual_proof['acquisition_projection']['expected']==actual_proof['acquisition_projection']['completed']==200,
        'strict all200 source/raw/arrays complete audit required')
    rule=semantic.load_rule(ROOT,configuration['semantic_acquire_derivation_rule'])
    a.require(len(rule.auxiliary_contexts)==1 and next(iter(rule.auxiliary_contexts.values()))['pointer']==approved,
        'semantic rule must explicitly bind this one approved auxiliary execution')
    reference=load_reference_rule(ROOT,configuration['reference_time_derivation_rule'])
    goal_rule=goal.load_rule(ROOT,configuration['command_goal_reference_rule'],purpose='production_fitting')
    a.require(len(goal_rule.contexts)==9 and len(goal_rule.auxiliary_contexts)==1,'nine primary plus explicit auxiliary contexts required')
    config=interaction.InteractionFitConfig(**configuration['interaction'])
    a.require(config.minimum_completed_parent_sources==8 and config.maximum_position_tolerance_m==.03
        and config.maximum_orientation_tolerance_rad==.15 and config.development_only is False,'unchanged8/.03/.15 required')
    code={str(Path(m.__file__).resolve()):a.file_sha(m.__file__) for m in (semantic,interaction,adapter,release,goal,basketball,supplement)}
    code[str(Path(__file__).resolve())]=a.file_sha(__file__)
    cases={task:[] for task in semantic.TASKS};primary=next(iter(rule.contexts.values()));selection=[]
    for expected in primary['plan']['cases']:
        if expected['task'] not in semantic.TASKS or expected['teacher_used'] is not True:continue
        found=[c['collection']/'cases/train'/expected['task']/expected['case_id']/'case.json'
            for c in rule.contexts.values() if (c['collection']/'cases/train'/expected['task']/expected['case_id']/'case.json').is_file()]
        a.require(len(found)==1,'each original fixed teacher has exactly one actual canonical receipt')
        selection.append(('original_r0',found[0],expected))
    for auxiliary in rule.auxiliary_contexts.values():
        current=auxiliary['context']['current']
        for expected in current['plan']['cases']:
            a.require(expected['task'] in semantic.TASKS and expected['teacher_used'] is True,'aux must contain only fixed basketball teacher sources')
            path=current['collection']/'cases/train'/expected['task']/expected['case_id']/'case.json'
            a.require(path.is_file(),'missing fixed auxiliary canonical receipt');selection.append(('auxiliary_r1',path,expected))
    a.require(Counter(kind for kind,p,e in selection)=={'original_r0':200,'auxiliary_r1':200},'exact400 fixed sources required')
    global_proofs={};goal_proofs={};all_cases=[];all_files=dict(rule.verified_files_sha256);all_files.update(goal_rule.verified_files_sha256)
    for kind,path,expected in selection:
        record=a.read(path);entry=record['entry'];cache=a.local(ROOT,entry['cache']);commands=a.local(ROOT,entry['teacher_command_receipt'])
        with np.load(cache,allow_pickle=False) as stored:arrays={k:stored[k].copy() for k in stored.files}
        native=dict(path=path,record=record,entry=entry,arrays=arrays,command_receipt=a.read(commands),
            command_receipt_verified=True,command_receipt_sha256=a.file_sha(commands))
        source,audit=adapter.interaction_source(native,reference_time_rule=reference,semantic_acquire_rule=rule,command_goal_rule=goal_rule)
        a.require(source.source_id not in global_proofs,'duplicate original/aux rollout source')
        proof=release._semantic_source_proof(source);global_proofs[source.source_id]=proof
        goal_proofs[source.source_id]=goal.evidence_proof(source.command_goal_reference_evidence)
        all_files.update(audit['command_goal_reference_derivation']['verified_files_sha256'])
        all_files.update(audit['semantic_acquire_derivation']['verified_files_sha256'])
        row=dict(kind=kind,canonical_receipt=a.pointer(ROOT,path),cache=a.pointer(ROOT,cache),command_receipt=a.pointer(ROOT,commands),
            source_id=source.source_id,parent_source_group=source.grouping_source,semantic_proof=proof,goal_proof=goal_proofs[source.source_id])
        cases[entry['task']].append((native,source,row));all_cases.append(native)
        if len(all_cases)%25==0:write(output/'progress.json',dict(stage='strict_actual400_derivation',completed_sources=len(all_cases),expected=400))
    a.require(all(len(v)==200 and len({s.grouping_source for c,s,r in v})==100
        and set(Counter(s.grouping_source for c,s,r in v).values())=={2} for v in cases.values()),
        'two exact replicas share each original parent, with100 parents per task')
    semantic_manifest=dict(rule=dict(path=rule.path,sha256=rule.sha256),helper=rule.helper,derivation_schema=semantic.SCHEMA,
        raw_masks_and_events_preserved=True,per_source=global_proofs,verified_files_sha256=rule.verified_files_sha256)
    goal_manifest=dict(rule=goal_rule.pointer,helper=adapter.pointer(Path(goal.__file__)),derivation_schema=goal.SCHEMA,
        original_raw_metric_events_and_masks_preserved=True,per_source=goal_proofs,verified_files_sha256=goal_rule.verified_files_sha256)
    global_path=output/'TRAIN_400_derivation_proof_before_fit.json'
    write(global_path,dict(schema='r10_explicit_original200_aux200_semantic_fit_source_manifest_v1',fit_split='train',VAL_used=False,TEST_used=False,
        fitting_configuration=a.pointer(ROOT,configuration_path),semantic_acquire_derivation=semantic_manifest,
        command_goal_reference_derivation=goal_manifest,
        complete_aux200_audit=configuration['semantic_auxiliary_complete_audit'],auxiliary_registration=approved,
        original_sources=200,auxiliary_sources=200,distinct_original_parent_groups=200,
        entries=[c['entry'] for c in all_cases],case_proofs=[r for v in cases.values() for c,s,r in v],
        primary2800_or_combined3000_complete_claimed=False,formal_model_policy_release=False))
    frozen=a.file_sha(global_path);tasks={}
    for task,values in cases.items():
        directory=output/task;manifest=directory/'TRAIN_fit_manifest.json'
        write(manifest,dict(schema='r10_task_original100_aux100_semantic_fit_diagnostic_v1',fit_split='train',VAL_used=False,TEST_used=False,
            entries=[c['entry'] for c,s,r in values],case_proofs=[r for c,s,r in values],
            semantic_acquire_derivation=semantic_manifest,command_goal_reference_derivation=goal_manifest,
            complete400_proof=a.pointer(ROOT,global_path),fitting_configuration=a.pointer(ROOT,configuration_path),
            all_original_failed_censored_and_empty_semantic_sources_retained=True,original_parents=100,actual_teacher_rollouts=200,
            parent_fold_unit='original raw TRAIN parent; r0/r1 always share fold',formal_model_policy_release=False))
        row=dict(original_teacher_sources=100,auxiliary_teacher_sources=100,total_sources=200,distinct_original_parents=100,
            fit_manifest=a.pointer(ROOT,manifest),semantic_physical_rollouts=sum(bool(s.semantic_acquire_evidence.segments) for c,s,r in values))
        try:
            library=interaction.fit_interaction_library([s for c,s,r in values],config,TRAIN_source_manifest_path=manifest)
            details=library.metadata['completed_skill_fitting'];path=directory/'interaction_library_diagnostic.json';library.save(path)
            row.update(status='actual200_teacher_rollouts_production_kernel_passed_diagnostic_only',library=a.pointer(ROOT,path))
        except interaction.InteractionFittingPending as error:
            details=error.diagnostics;row.update(status='actual200_teacher_rollouts_fit_pending_unchanged_gates',error=str(error))
        detail=directory/'real_production_fit_report.json';write(detail,details)
        row.update(detailed_fit_report=a.pointer(ROOT,detail),failed_fit_keys=details.get('failed_fit_keys',[]),
            phase_fits={k:{field:v.get(field) for field in ('passed','reason','completed_parent_count','completed_attempt_count',
                'position_tolerance_m','orientation_tolerance_rad','selected_lag','duration_seconds','duration_budget_seconds')}
                for k,v in details['phase_fits'].items()})
        tasks[task]=row;write(output/'progress.json',dict(stage='actual400_full_curve_fitting',tasks=tasks,complete=False))
    a.require(a.file_sha(global_path)==frozen,'pre-fit400 source proof changed')
    input_closure.include(all_files);input_closure.include(files);input_closure.rehash_all()
    for name,sha in code.items():a.require(a.file_sha(name)==sha,'fitting/consumer source changed during actual400 fit')
    report=dict(schema='r10_original200_aux200_actual400_semantic_and_command_goal_production_CPU_fit_v1',created_utc=datetime.now(timezone.utc).isoformat(),
        wall_seconds=time.perf_counter()-begun,configuration=a.pointer(ROOT,configuration_path),rule=configuration['semantic_acquire_derivation_rule'],
        command_goal_rule=configuration['command_goal_reference_rule'],
        complete3000_bundle_audit_reference=configuration.get('complete_primary2800_aux200_bundle_audit'),
        all675_original_deposit_commands_retained=sum(len(s.command_goal_reference_evidence.references)+len(s.command_goal_reference_evidence.pending_commands) for values in cases.values() for c,s,r in values)==675,
        complete_aux200_audit=configuration['semantic_auxiliary_complete_audit'],actual_approved_aux_registration=approved,
        prefit400_source_proof=a.pointer(ROOT,global_path),original200_aux200_all_sources_retained=True,
        distinct_original_parent_groups=200,actual_rollout_sources=400,fold_unit_original_parent=True,
        tasks=tasks,source_sha256=code,verified_files_sha256=input_closure.files,old_v4_rule_or_negative_report_modified=False,
        formal_primary2800_or_combined3000_manifest_created=False,Context_K1_unchanged=True,native_GPU_or_collection_started=False,
        formal_model_policy_or_success_rate_release=False)
    path=output/'report.json';write(path,report);write(output/'progress.json',dict(complete=True,tasks=tasks))
    return dict(report=a.pointer(ROOT,path),tasks={t:dict(status=v['status'],phase_fits=v['phase_fits']) for t,v in tasks.items()})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--fit-config',required=True);p.add_argument('--out',required=True)
    args=p.parse_args();print(json.dumps(run(args.fit_config,args.out),ensure_ascii=False))
