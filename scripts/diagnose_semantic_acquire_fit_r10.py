"""Real TRAIN task100x2 CPU refit with explicitly verified semantic acquire."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from our_mani.multitask import acquisition_lineage as a
from our_mani.multitask import semantic_acquire as semantic
from our_mani.multitask import interaction
from our_mani.multitask.artifact_fitting import interaction_source
from our_mani.multitask.reference_time import load_rule as load_reference_rule


def diagnose(rule_pointer,fit_config,reference_rule_pointer,out):
    out=a.local(ROOT,out);a.require(not out.exists(),'fresh actual CPU diagnostic namespace required')
    out.mkdir(parents=True);started=datetime.now(timezone.utc).isoformat()
    rule=semantic.load_rule(ROOT,rule_pointer)
    reference=load_reference_rule(ROOT,reference_rule_pointer) if reference_rule_pointer else None
    configuration=a.read(a.local(ROOT,fit_config));config=interaction.InteractionFitConfig(**configuration['interaction'])
    code_before={str(Path(module.__file__).resolve()):a.file_sha(module.__file__)
        for module in (semantic,interaction)}
    cases_by_task={task:[] for task in semantic.TASKS};rows={};files=dict(rule.verified_files_sha256)
    original=next(iter(rule.contexts.values()))
    for case in original['plan']['cases']:
        if case['task'] not in semantic.TASKS or case['teacher_used'] is not True:continue
        cid=case['case_id'];found=[]
        for context in rule.contexts.values():
            p=context['collection']/'cases/train'/case['task']/cid/'case.json'
            if p.exists():found.append(p)
        a.require(len(found)==1,'every original teacher must have one actual canonical result')
        p=found[0];record=a.read(p);entry=record['entry'];cache=a.local(ROOT,entry['cache'])
        with np.load(cache,allow_pickle=False) as stored:arrays={key:stored[key].copy() for key in stored.files}
        command_path=a.local(ROOT,entry['teacher_command_receipt'])
        native=dict(path=p,record=record,entry=entry,arrays=arrays,command_receipt=a.read(command_path),
            command_receipt_sha256=record['files_sha256'][entry['teacher_command_receipt']],command_receipt_verified=True)
        source,audit=interaction_source(native,reference_time_rule=reference,semantic_acquire_rule=rule)
        cases_by_task[case['task']].append((native,source,audit))
    a.require(all(len(cases)==100 for cases in cases_by_task.values()),'actual original100 teachers per task required')
    for task,cases in cases_by_task.items():
        directory=out/task;directory.mkdir();manifest=dict(version='r10_actual_teacher_task100_semantic_fit_diagnostic_v1',
            fit_split='train',VAL_used=False,TEST_used=False,entries=[case['entry'] for case,source,audit in cases],
            semantic_acquire_derivation=dict(rule=rule_pointer,helper=rule.helper,
                per_source={source.source_id:audit['semantic_acquire_derivation'] for case,source,audit in cases}),
            all_original100_parents_retained=True,formal_comparison_release=False)
        path=directory/'TRAIN_fit_manifest.json';path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
        row=dict(original_teacher_acquisitions=100,distinct_original_parents=len({s.grouping_source for c,s,d in cases}),
            all_original_sources_retained=True,semantic_source_count=100,
            semantic_physical_parent_sources=sum(bool(s.semantic_acquire_evidence.segments) for c,s,d in cases),
            fit_manifest=a.pointer(ROOT,path),native_or_GPU_started=False)
        try:
            library=interaction.fit_interaction_library([s for c,s,d in cases],config,TRAIN_source_manifest_path=path)
            row.update(status='real_CPU_fit_passed_pending_full_project_and_formal_release',
                completed_skill_fitting=library.metadata['completed_skill_fitting'],template_count=len(library.templates))
        except interaction.InteractionFittingPending as error:
            row.update(status='real_CPU_fit_pending_unchanged_support_or_terminal_caps',error=str(error),
                completed_skill_fitting=error.diagnostics)
        except (ValueError,interaction.InteractionArtifactError) as error:
            row.update(status='real_CPU_fit_rejected',error_type=type(error).__name__,error=str(error))
        rows[task]=row
    a.require(all(a.file_sha(name)==sha for name,sha in code_before.items()),'semantic fitter source changed during diagnostic')
    report=dict(schema='r10_actual_original100x2_semantic_acquire_CPU_refit_v1',started_utc=started,
        finished_utc=datetime.now(timezone.utc).isoformat(),tasks=rows,rule=rule_pointer,
        reference_time_rule=reference_rule_pointer,fit_config=a.pointer(ROOT,a.local(ROOT,fit_config)),
        source_sha256=code_before,all200_original_teacher_sources_retained=True,
        thresholds_or_parent_support_relaxed=False,native_GPU_or_supplement_started=False,
        formal_model_or_policy_release=False,Context_K1_view_or_data_changed=False)
    p=out/'report.json';p.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    return dict(report=a.pointer(ROOT,p),tasks={task:dict(status=row['status'],
        phase_fits={key:{name:value.get(name) for name in ('passed','completed_parent_count','reason',
            'position_tolerance_m','orientation_tolerance_rad','duration_seconds','duration_budget_seconds')}
            for key,value in row.get('completed_skill_fitting',{}).get('phase_fits',{}).items()})
        for task,row in rows.items()},native_or_GPU_started=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rule',required=True)
    parser.add_argument('--fit-config',required=True)
    parser.add_argument('--reference-time-rule')
    parser.add_argument('--out',required=True)
    args=parser.parse_args()
    def pointer(value):return a.pointer(ROOT,a.local(ROOT,value)) if value else None
    print(json.dumps(diagnose(pointer(args.rule),args.fit_config,pointer(args.reference_time_rule),args.out),ensure_ascii=False))
