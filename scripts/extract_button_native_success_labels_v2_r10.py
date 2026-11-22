"""All original200 TRAIN button outcomes; independent labels, no BC override."""
import argparse
from dataclasses import asdict
import json,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from our_mani.multitask import acquisition_lineage as a,artifact_fitting as adapter,command_goal_reference as goal,native_task_success_labels as labels,release_preflight as release,interaction


def run(out):
    began=time.perf_counter();out=a.local(ROOT,out);a.require(not out.exists(),'fresh native-label output required');out.mkdir()
    inputs=release._Inputs(ROOT);ruleptr=a.pointer(ROOT,ROOT/'artifacts/framework/multitask_r10/command_goal_reference_rule_v5_translation_revision2_setup/command_goal_rule_original_only.json')
    rule=goal.load_rule(ROOT,ruleptr,purpose='production_fitting');inputs.include(rule.verified_files_sha256)
    for module in (goal,adapter,labels):inputs.file(module.__file__,a.file_sha(module.__file__))
    inputs.file(__file__,a.file_sha(__file__));rows=[]
    for task in labels.TASKS:
        cases=inputs.read(f'artifacts/framework/multitask_r10/full14_pending_reference_readonly_diagnostic_v6/{task}/cases.json',a.file_sha(ROOT/f'artifacts/framework/multitask_r10/full14_pending_reference_readonly_diagnostic_v6/{task}/cases.json'))
        for row in cases:
            path=inputs.pointer(row['canonical_receipt']);record=a.read(path);cache=inputs.pointer(row['cache']);commands=inputs.pointer(row['command_receipt'])
            with np.load(cache,allow_pickle=False) as z:arrays={k:z[k].copy() for k in z.files}
            case=dict(path=path,record=record,entry=record['entry'],arrays=arrays,command_receipt=a.read(commands),command_receipt_verified=True,command_receipt_sha256=row['command_receipt']['sha256'])
            source=adapter.interaction_source(case)[0];value=labels.extract(case,source,rule)
            inputs.include(value.derivation_proof['verified_files_sha256']);rows.append(asdict(value))
    a.require(len(rows)==len({r['parent_source_group'] for r in rows})==200,'complete original200 parent coverage required')
    inputs.rehash_all();path=out/'labels.json';path.write_text(json.dumps(interaction._json(rows),ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    report=dict(schema='r10_original200_button_independent_native_success_labels_v1',actual_original_teacher_sources=200,actual_original_parents=200,
        per_task={t:dict(all_original_sources=sum(r['task_name']==t for r in rows),success=sum(r['task_name']==t and r['raw_task_success'] for r in rows),
            budget_censored=sum(r['task_name']==t and r['budget_censored'] for r in rows),native_terminal_failure=sum(r['task_name']==t and r['raw_terminal_failure'] for r in rows),
            command_completed_success=sum(r['task_name']==t and r['success_completion'] is not None and r['success_completion']['raw_command_completed'] for r in rows)) for t in labels.TASKS},
        labels=a.pointer(ROOT,path),helper=a.pointer(ROOT,Path(labels.__file__)),verified_files_sha256=inputs.files,
        command_completed_at_fabricated=False,geometry_caps_or_BC_admission_modified=False,
        native_success_is_public_runtime_observation=False,all_failed_censored_sources_preserved=True,
        native_or_GPU_started=False,formal_release=False,seconds=time.perf_counter()-began)
    rp=out/'report.json';rp.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps(a.pointer(ROOT,rp)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);run(p.parse_args().out)
