"""Independent original200 remove-cup candidate fits; no runtime/spec mutation."""
import argparse
from dataclasses import asdict
import json,shutil,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from our_mani.multitask import acquisition_lineage as a,artifact_fitting as adapter,command_goal_reference as goal,interaction,remove_command_goal_candidates_v46 as candidate,release_preflight as release
BASE=ROOT/'artifacts/framework/multitask_r10'


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(interaction._json(value),ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def run(out):
    begun=time.perf_counter();out=a.local(ROOT,out);a.require(not out.exists(),'fresh independent remove candidate namespace required');out.mkdir()
    inputs=release._Inputs(ROOT);ruleptr=a.pointer(ROOT,BASE/'command_goal_reference_rule_v6_counted_final_setup_v1/command_goal_rule.json')
    rule=goal.load_rule(ROOT,ruleptr,purpose='production_fitting');inputs.include(rule.verified_files_sha256)
    for mod in (goal,candidate,adapter,interaction):inputs.file(mod.__file__,a.file_sha(mod.__file__))
    inputs.file(__file__,a.file_sha(__file__))
    native=ROOT.parent/'phsimani/PhysMani/third_party/RLBench/rlbench/tasks';sourceptrs={}
    for task in candidate.TASKS:
        old=native/(task+'.py');new=out/'readonly_source'/old.name;new.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(old,new);sourceptrs[task]=a.pointer(ROOT,new)
    old=native/'place_cups_on_rotating_frame.py';new=out/'readonly_source'/old.name;shutil.copyfile(old,new)
    permission=out/'candidate_permission.json';write(permission,dict(schema=candidate.SCHEMA,TRAIN_only=True,runtime_enabled=False,production_enabled=False,
        helper=a.pointer(ROOT,Path(candidate.__file__)),task_sources=sourceptrs,rotating_helper=a.pointer(ROOT,new),
        original_readonly_evidence=a.pointer(ROOT,BASE/'full14_pending_reference_readonly_diagnostic_v6/report.json'),
        waypoint_parent_hierarchy_recorded=False,allow_unqualified_rotating_numeric_diagnostic=True))
    inputs.pointer(a.pointer(ROOT,permission))
    configpath=BASE/'command_goal_reference_rule_v6_counted_final_setup_v1/artifact_fit_configuration_full3100.json'
    config=inputs.read(str(configpath),a.file_sha(configpath));ic=interaction.InteractionFitConfig(**config['interaction'])
    allproofs=[];sources={};evidences={}
    for task in candidate.TASKS:
        rows=a.read(BASE/f'full14_pending_reference_readonly_diagnostic_v6/{task}/cases.json');sources[task]=[];evidences[task]=[]
        for row in rows:
            path=inputs.pointer(row['canonical_receipt']);record=a.read(path);cache=inputs.pointer(row['cache']);commands=inputs.pointer(row['command_receipt'])
            with np.load(cache,allow_pickle=False) as z:arrays={k:z[k].copy() for k in z.files}
            case=dict(path=path,record=record,entry=record['entry'],arrays=arrays,command_receipt=a.read(commands),command_receipt_verified=True,command_receipt_sha256=row['command_receipt']['sha256'])
            source=adapter.interaction_source(case)[0];e=candidate.derive_candidate(case,source,rule,a.pointer(ROOT,permission),include_unqualified_rotating=True)
            inputs.include(e.derivation_proof['verified_files_sha256']);sources[task].append(source);evidences[task].append(e)
            allproofs.append(dict(source_id=source.source_id,parent_source_group=source.grouping_source,
                original_raw_bindings=e.original_verified_evidence.raw_bindings,original_command_sha256=e.original_verified_evidence.original_command_ledger_sha256,
                original_completion_events_sha256=e.original_verified_evidence.original_completion_events_sha256,
                original_masks_sha256=e.original_verified_evidence.original_masks_sha256,
                original_references_and_events_retained=True,candidate_references=[asdict(r) for r in e.references],pending_commands=e.pending_commands,
                original_censored_command_count=len(source.command_ledger)-len(source.completion_events)))
        a.require(len(rows)==len(sources[task])==len({s.grouping_source for s in sources[task]})==100,'all original100 parents retained')
    prefit=out/'TRAIN_200_independent_candidate_prefit_proof.json';write(prefit,dict(schema=candidate.SCHEMA,actual_teacher_sources=200,distinct_original_parents=200,case_proofs=allproofs,
        source_rule=ruleptr,candidate_permission=a.pointer(ROOT,permission),original_all_failed_censored_sources_retained=True,runtime_enabled=False,production_enabled=False))
    frozen=a.file_sha(prefit);tasks={};previous=a.read(BASE/'TRAIN_artifacts_full_goalv4_semanticv6_v1/report.json')
    for task in candidate.TASKS:
        old=goal.fit_diagnostic(sources[task],[e.original_verified_evidence for e in evidences[task]],ic)
        unchanged=[]
        expected=previous['per_task'][task]['completed_skill_fitting']['phase_fits']
        for key,row in old['phase_fits'].items():
            for name in ('selected_lag','position_tolerance_m','orientation_tolerance_rad'):
                if name in row and name in expected[key]:a.require(abs(row[name]-expected[key][name])<=1e-12,'legacy numeric baseline changed')
            unchanged.append(key)
        current=candidate.fit_candidate_diagnostic(sources[task],evidences[task],ic)
        baselinepath=out/task/'legacy_original100_full_curve.json';write(baselinepath,old)
        currentpath=out/task/'independent_candidate_original100_full_curve.json';write(currentpath,current)
        tasks[task]=dict(actual_teacher_sources=100,actual_original_parents=100,legacy_numeric_partitions_unchanged=unchanged,
            legacy=a.pointer(ROOT,baselinepath),candidate=a.pointer(ROOT,currentpath),candidate_all_numeric_partitions_passed=current['all_required_phases_and_observed_partitions_passed'],
            source_hierarchy_qualified=False,causal_runtime_nearest_spoke_selection_verified=False,
            runtime_or_production_model_release=False,
            phases={k:{f:v.get(f) for f in ('passed','completed_parent_count','completed_attempt_count','position_tolerance_m','orientation_tolerance_rad','selected_lag','duration_seconds','duration_budget_seconds','grouped_TRAIN_CV_error')} for k,v in current['phase_fits'].items()})
    a.require(a.file_sha(prefit)==frozen,'pre-fit typed candidate proofs changed');inputs.rehash_all()
    report=dict(schema='r10_original200_remove_independent_reference_candidate_CPU_fit_v1',actual_teacher_sources=200,actual_original_parent_groups=200,
        prefit_proof=a.pointer(ROOT,prefit),candidate_permission=a.pointer(ROOT,permission),tasks=tasks,verified_files_sha256=inputs.files,
        all_original_failed_censored_sources_preserved=True,original_reference_events_commands_masks_not_rewritten=True,
        actual_full_curves_and_original_parent_group_CV=True,production_curve_and_rotation_numeric_kernel_unchanged=True,
        rotating_hierarchy_evidence_gap_retained=True,runtime_reference_role_or_specs_changed=False,
        native_or_GPU_started=False,formal_or_runtime_release=False,seconds=time.perf_counter()-begun)
    rp=out/'report.json';write(rp,report);print(json.dumps(a.pointer(ROOT,rp)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);run(p.parse_args().out)
