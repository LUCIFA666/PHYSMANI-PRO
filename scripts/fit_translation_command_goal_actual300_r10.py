"""Real original300 TRAIN full-curve candidate fits, no native/GPU/release."""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from our_mani.multitask import acquisition_lineage as a,artifact_fitting as adapter,interaction,command_goal_reference as goal,translation_command_goal_reference as trans,release_preflight as release


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(interaction._json(value),ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def run(configpath,output):
    began=time.perf_counter();output=a.local(ROOT,output);a.require(not output.exists(),'fresh actual300 output required');output.mkdir()
    inputs=release._Inputs(ROOT);configpath=a.local(ROOT,configpath);config=inputs.read(a.pointer(ROOT,configpath)['path'],a.file_sha(configpath))
    inputs.source_snapshot(config['fitting_core_source_snapshot']);inputs.source_snapshot(config['fitting_producer_source_snapshot'])
    rule=goal.load_rule(ROOT,config['command_goal_reference_rule'],purpose='production_fitting')
    a.require(rule.tasks==goal.TASK_PHASES_V5 and len(rule.contexts)==9 and not rule.auxiliary_contexts,'explicit original-only v5 tasks/context required')
    inputs.include(rule.verified_files_sha256)
    ic=interaction.InteractionFitConfig(**config['interaction']);a.require(ic.minimum_completed_parent_sources==8 and ic.maximum_position_tolerance_m==.03 and ic.maximum_orientation_tolerance_rad==.15,'original gates changed')
    selection=[]
    for expected in next(iter(rule.contexts.values()))['plan']['cases']:
        if expected['task'] not in trans.TASKS or expected['teacher_used'] is not True:continue
        found=[c['collection']/'cases/train'/expected['task']/expected['case_id']/'case.json' for c in rule.contexts.values() if (c['collection']/'cases/train'/expected['task']/expected['case_id']/'case.json').is_file()]
        a.require(len(found)==1,'fixed original teacher missing or duplicated');selection.append((expected,found[0]))
    a.require(len(selection)==300,'exact all100×3 original teachers required')
    groups={t:[] for t in trans.TASKS};proofs={};rows=[];cases=[]
    for expected,path in selection:
        record=a.read(path);entry=record['entry'];cache=a.local(ROOT,entry['cache']);commands=a.local(ROOT,entry['teacher_command_receipt'])
        with np.load(cache,allow_pickle=False) as z:arrays={k:z[k].copy() for k in z.files}
        case=dict(path=path,record=record,entry=entry,arrays=arrays,command_receipt=a.read(commands),command_receipt_verified=True,command_receipt_sha256=a.file_sha(commands))
        source,audit=adapter.interaction_source(case,command_goal_rule=rule);sid=source.source_id
        a.require(sid not in proofs and source.grouping_source==expected['parent_source_group'],'source/parent identity changed')
        proofs[sid]=goal.evidence_proof(source.command_goal_reference_evidence);inputs.include(audit['command_goal_reference_derivation']['verified_files_sha256'])
        row=dict(source_id=sid,parent_source_group=source.grouping_source,canonical_receipt=a.pointer(ROOT,path),cache=a.pointer(ROOT,cache),command_receipt=a.pointer(ROOT,commands),goal_proof=proofs[sid])
        rows.append(row);cases.append(case);groups[entry['task']].append((case,source,row))
        if len(rows)%25==0:write(output/'progress.json',dict(stage='actual300_strict_raw_derivation',completed=len(rows),expected=300))
    a.require(all(len(v)==len({s.grouping_source for c,s,r in v})==100 for v in groups.values()),'complete original100 per task required')
    meta=dict(rule=rule.pointer,helper=adapter.pointer(Path(goal.__file__)),derivation_schema=goal.SCHEMA,original_raw_metric_events_and_masks_preserved=True,per_source=proofs,verified_files_sha256=rule.verified_files_sha256)
    globalpath=output/'TRAIN_300_derivation_proof_before_fit.json'
    write(globalpath,dict(schema='r10_original300_translation_intent_prefit_v1',fit_split='train',VAL_used=False,TEST_used=False,
        fitting_configuration=a.pointer(ROOT,configpath),command_goal_reference_derivation=meta,entries=[c['entry'] for c in cases],case_proofs=rows,
        all_original_failed_censored_sources_retained=True,distinct_original_parents=300,actual_teacher_sources=300,formal_release=False))
    frozen=a.file_sha(globalpath);tasks={}
    for task,values in groups.items():
        manifest=output/task/'TRAIN_fit_manifest.json';write(manifest,dict(schema='r10_original100_translation_intent_fit_v1',fit_split='train',VAL_used=False,TEST_used=False,
            fitting_configuration=a.pointer(ROOT,configpath),complete300_proof=a.pointer(ROOT,globalpath),entries=[c['entry'] for c,s,r in values],case_proofs=[r for c,s,r in values],
            command_goal_reference_derivation=meta,all_original_failed_censored_sources_retained=True,formal_release=False))
        row=dict(original_teacher_sources=100,original_parent_groups=100,fit_manifest=a.pointer(ROOT,manifest))
        try:
            library=interaction.fit_interaction_library([s for c,s,r in values],ic,TRAIN_source_manifest_path=manifest)
            details=library.metadata['completed_skill_fitting'];libpath=output/task/'interaction_library_candidate.json';library.save(libpath)
            row.update(status='actual_original100_production_kernel_passed_candidate_only',library=a.pointer(ROOT,libpath))
        except interaction.InteractionFittingPending as error:
            details=error.diagnostics;row.update(status='actual_original100_pending_unchanged_caps',error=str(error))
        detailpath=output/task/'real_production_fit_report.json';write(detailpath,details)
        row.update(detailed_fit_report=a.pointer(ROOT,detailpath),failed_fit_keys=details['failed_fit_keys'],phase_fits={k:{n:v.get(n) for n in ('passed','reason','completed_parent_count','completed_attempt_count','position_tolerance_m','orientation_tolerance_rad','selected_lag','duration_seconds','duration_budget_seconds','grouped_TRAIN_CV_error')} for k,v in details['phase_fits'].items()})
        tasks[task]=row;write(output/'progress.json',dict(stage='actual300_complete_curve_fits',tasks=tasks,complete=False))
    a.require(a.file_sha(globalpath)==frozen,'prefit actual300 proofs changed');inputs.rehash_all()
    result=dict(schema='r10_original300_translation_intent_actual_production_CPU_fit_v1',configuration=a.pointer(ROOT,configpath),
        source_rule=rule.pointer,prefit300_source_proof=a.pointer(ROOT,globalpath),actual_teacher_sources=300,distinct_original_parents=300,
        tasks=tasks,all_original_failed_censored_sources_retained=True,button_press_reference_or_native_events_modified=False,
        thresholds_or_parent_support_relaxed=False,verified_files_sha256=inputs.files,wall_seconds=time.perf_counter()-began,
        full1400_or3000_refit_claimed=False,native_or_GPU_started=False,formal_policy_or_SR_release=False)
    report=output/'report.json';write(report,result);write(output/'progress.json',dict(complete=True,tasks=tasks));return a.pointer(ROOT,report)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--fit-config',required=True);p.add_argument('--out',required=True);args=p.parse_args();print(json.dumps(run(args.fit_config,args.out)))
