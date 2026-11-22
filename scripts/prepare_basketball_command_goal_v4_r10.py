"""Freeze explicit goalv4/semanticv6 TRAIN rules; metadata only, no simulation."""
from copy import deepcopy
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from our_mani.multitask import acquisition_lineage as a,command_goal_reference as goal
from our_mani.multitask import basketball_command_goal_reference as basketball,semantic_acquire,interaction,artifact_fitting

BASE=ROOT/'artifacts/framework/multitask_r10'


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def run(revision=1):
    a.require(type(revision) is int and revision>=1,'explicit positive revision required')
    out=BASE/('command_goal_reference_rule_v4_setup' if revision==1 else f'command_goal_reference_rule_v4_revision{revision}_setup')
    semout=BASE/'semantic_acquire_rule_v6_aux200_setup'
    a.require(not out.exists() and (revision>1 or not semout.exists()),'fresh goalv4/semanticv6 rule namespaces required')
    out.mkdir();files={}
    if not semout.exists():semout.mkdir()
    registration=BASE/'TRAIN_recovery_v20_final_remaining_setup/registrations_with_final_recovery.json'
    registration_pointer=a.pointer(ROOT,registration);roster=a.read(registration)
    a.require(len(roster)==9 and registration_pointer['sha256']=='43aa6a28a8c8ea24c51e0d1b02d13e924200aac7f8cfb611c66986c7ff553a8f',
        'exact nine actual primary executions required')
    contexts=[a.load_archived_plan(ROOT,row,files) for row in roster]
    draft_path=BASE/'basketball_deposit_intention_actual400_draft_v2/report.json';draft=a.read(draft_path)
    old=BASE/'command_goal_reference_rule_v3_setup';oldrule=a.read(old/'command_goal_rule.json')
    native=a.read(old/'archived_source_evidence.json');native['tasks']=goal.TASK_PHASES
    native['predecessor_evidence']=a.pointer(ROOT,old/'archived_source_evidence.json')
    native['created_utc']=datetime.now(timezone.utc).isoformat()
    for name,item in draft['source'].items():
        if Path(name).stem not in basketball.TASKS:continue
        candidates=[c['sources'][name] for c in contexts if name in c['sources']]
        a.require(candidates and all(row['sha256']==item['sha256'] for row in candidates),'task source differs from actual archives')
        native['archived_sources'][name]=a.pointer(ROOT,Path(candidates[0]['path']))
    native_path=out/'archived_source_evidence.json';write(native_path,native)
    bin_permission=deepcopy(a.read(old/'bin_intention_permission.json'))
    bin_path=out/'bin_intention_permission.json';write(bin_path,bin_permission)
    kernels={task:next(item['archive'] for name,item in draft['source'].items() if Path(name).stem==task) for task in basketball.TASKS}
    kernels['motion_helper']=next(item['archive'] for name,item in draft['source'].items() if Path(name).stem not in basketball.TASKS)
    for key in ('pyrep_getter','sim_getter','C_ABI_header','C_ABI_builder'):kernels[key]=bin_permission['kernel_sources'][key]
    review=out/'basketball_readonly_current_source_review.json'
    write(review,dict(schema='r10_basketball_readonly_current_source_derivation_review_v1',kernel_sources=kernels,
        historical_execution_dependency_claimed=False,new_TRAIN_derivation_sources_explicitly_bound=True,
        native_task_sources_also_independently_bound_to_actual_acquisition_archives=True,
        original_math_and_ABI_snapshot_gap_not_retroactively_filled=True,original400_replay=a.pointer(ROOT,draft_path)))
    aux_audit=a.pointer(ROOT,BASE/'TRAIN_teacher_supplement_complete200_audit_20261018T0751Z/report.json')
    permission=out/'basketball_intention_permission.json'
    write(permission,dict(schema=basketball.PERMISSION_SCHEMA,TRAIN_only=True,VAL_used=False,TEST_used=False,
        production_fitting_enabled=True,tasks=list(basketball.TASKS),helper=a.pointer(ROOT,Path(basketball.__file__)),
        kernel_sources=kernels,readonly_current_source_review=a.pointer(ROOT,review),
        exact_all400_replay=a.pointer(ROOT,draft_path),complete_aux200_audit=aux_audit))
    auxiliary=a.pointer(ROOT,BASE/'TRAIN_teacher_supplement_basketball_r1_v2_setup/execution_registration_approved.json')
    oldrule.update(schema=goal.FIT_RULE_SCHEMA,tasks=goal.TASK_PHASES,helper=a.pointer(ROOT,Path(goal.__file__)),
        original_adapter_source=a.pointer(ROOT,Path(artifact_fitting.__file__)),actual_registrations=registration_pointer,
        source_evidence=a.pointer(ROOT,native_path),bin_intention_derivation_permission=a.pointer(ROOT,bin_path),
        basketball_intention_derivation_permission=a.pointer(ROOT,permission),auxiliary_execution_registrations=[auxiliary])
    rule_path=out/'command_goal_rule.json';write(rule_path,oldrule)
    sem=deepcopy(a.read(BASE/'semantic_acquire_rule_v5_aux200_setup/semantic_acquire_rule.json'))
    evidence=deepcopy(a.read(a.local(ROOT,sem['evidence_report']['path'])))
    evidence['predecessor_original200_design_evidence']=sem['evidence_report']
    evidence['actual_registration_list']=registration_pointer
    evidence['archive_roster_extension_only']=dict(original_basket200_raw_sources_unchanged=True,
        current400_raw_consumer_proof=a.pointer(ROOT,BASE/'semantic_acquire_original200_aux200_consumer_v5_scoped_integrity_v1/report.json'),
        added_primary_registration_does_not_replace_canonical_basket200=True)
    evidence_path=semout/'original200_design_evidence_with_nine_actual_registrations.json'
    if revision==1:write(evidence_path,evidence)
    sem['actual_registration_list']=registration_pointer;sem['evidence_report']=a.pointer(ROOT,evidence_path)
    sem['source_bindings']['adapter']=a.pointer(ROOT,Path(artifact_fitting.__file__))
    sempath=semout/'semantic_acquire_rule.json'
    if revision==1:write(sempath,sem)
    else:a.require(a.read(sempath)==sem,'semantic source binding changed; a new semantic namespace is required')
    original_only=deepcopy(sem);original_only['auxiliary_execution_registrations']=[]
    if revision==1:write(semout/'semantic_acquire_original_only_negative_rule.json',original_only)
    config=deepcopy(a.read(BASE/'semantic_acquire_rule_v5_aux200_setup/artifact_fit_configuration.json'))
    config['version']='r10_full_TRAIN_command_goal_v4_semantic_v6_explicit_aux200'
    config['command_goal_reference_rule']=a.pointer(ROOT,rule_path);config['semantic_acquire_derivation_rule']=a.pointer(ROOT,sempath)
    config['command_goal_intent_scope']['basketball_source_scope']='deposit closed/open original unreflected t+.5 hoop intention; TRAIN-only driver; runtime causal candidate forecast only'
    config['basketball_source_intention_draft_evidence']=a.pointer(ROOT,draft_path)
    configuration=out/'artifact_fit_configuration.json';write(configuration,config)
    sources={}
    names=('interaction.py','reference_time.py','artifact_fitting.py','command_goal_reference.py','bin_command_goal_reference.py',
        'basketball_command_goal_reference.py','semantic_acquire.py','release_preflight.py','evaluation.py','acquisition_lineage.py',
        'training_view.py','teacher_supplement.py','acquisition_manifest.py','specs.py')
    for name in names:
        source=ROOT/'our_mani/multitask'/name;target=out/'sources/our_mani/multitask'/name
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
        sources[str(source.relative_to(ROOT))]=dict(live_source=a.pointer(ROOT,source),archived_source=a.pointer(ROOT,target))
    for source in (ROOT/'our_mani/contracts.py',Path(__file__).resolve()):
        target=out/'sources'/source.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
        sources[str(source.relative_to(ROOT))]=dict(live_source=a.pointer(ROOT,source),archived_source=a.pointer(ROOT,target))
    snapshot=out/'source_snapshot.json';write(snapshot,dict(schema='r10_command_goal_production_core_source_snapshot_v4',
        created_utc=datetime.now(timezone.utc).isoformat(),sources=sources,
        predecessor_source_snapshot=a.pointer(ROOT,old/'source_snapshot.json'),goal_rule=a.pointer(ROOT,rule_path),
        basketball_permission=a.pointer(ROOT,permission),semantic_rule_v6=a.pointer(ROOT,sempath),configuration=a.pointer(ROOT,configuration),
        interaction_reference_time_and_controller_numerics_unchanged=True,native_GPU_or_collection_started=False))
    # Full permission/raw archive validation is an explicit CPU action.  No
    # optimizer/native process is started by this metadata builder.
    verified=goal.load_rule(ROOT,a.pointer(ROOT,rule_path),purpose='production_fitting')
    semantic=semantic_acquire.load_rule(ROOT,a.pointer(ROOT,sempath))
    a.require(len(verified.contexts)==len(semantic.contexts)==9 and len(verified.auxiliary_contexts)==len(semantic.auxiliary_contexts)==1,
        'the exact primary9+one approved auxiliary contexts must load')
    return dict(source_snapshot=a.pointer(ROOT,snapshot),goal_rule=a.pointer(ROOT,rule_path),
        semantic_rule=a.pointer(ROOT,sempath),configuration=a.pointer(ROOT,configuration),primary_actual_plans=9,auxiliary_actual_plans=1)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--revision',type=int,default=1)
    print(json.dumps(run(parser.parse_args().revision),ensure_ascii=False))
