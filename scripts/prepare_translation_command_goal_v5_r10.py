"""Freeze new explicit translation TRAIN scope; no fit/native process launch."""
from copy import deepcopy
from datetime import datetime,timezone
import argparse,json,shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from our_mani.multitask import acquisition_lineage as a,command_goal_reference as goal,translation_command_goal_reference as trans,artifact_fitting as adapter,interaction
from our_mani.multitask import semantic_acquire
BASE=ROOT/'artifacts/framework/multitask_r10'


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def run(revision=1):
    a.require(type(revision) is int and revision>=1,'explicit positive metadata revision required')
    out=BASE/('command_goal_reference_rule_v5_translation_setup' if revision==1 else f'command_goal_reference_rule_v5_translation_revision{revision}_setup');a.require(not out.exists(),'fresh v5 setup required');out.mkdir()
    old=BASE/'command_goal_reference_rule_v4_revision2_setup';oldrule=a.read(old/'command_goal_rule.json')
    evidence=BASE/'full14_pending_reference_readonly_diagnostic_v6/report.json';pending=a.read(evidence)
    native=ROOT.parent/'phsimani/PhysMani/third_party/RLBench/rlbench/tasks'
    sources={}
    for task in trans.TASKS:
        actual=native/(task+'.py');target=out/'readonly_source'/actual.name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(actual,target)
        a.require(a.file_sha(actual)==pending['source_evidence'][task]['sha256'],'readonly task differs from all100 actual evidence');sources[task]=a.pointer(ROOT,target)
    oldbin=a.read(old/'bin_intention_permission.json')
    sources['motion_helper']=oldbin['kernel_sources']['motion_helper']
    for key in ('pyrep_getter','sim_getter','C_ABI_header','C_ABI_builder'):sources[key]=oldbin['kernel_sources'][key]
    review=out/'translation_readonly_source_review.json';write(review,dict(schema='r10_translation_readonly_source_derivation_review_v1',
        kernel_sources=sources,historical_execution_dependency_claimed=False,new_TRAIN_derivation_sources_explicitly_bound=True,
        actual_all100_evidence=a.pointer(ROOT,evidence),source_parameter_searched=False,
        public_runtime_access_to_private_source_forecast_allowed=False,original_reflection_tail_or_failures_removed=False))
    permission=out/'translation_intention_permission.json';write(permission,dict(schema=trans.PERMISSION_SCHEMA,TRAIN_only=True,VAL_used=False,TEST_used=False,
        production_fitting_enabled=True,tasks=list(trans.TASKS),helper=a.pointer(ROOT,Path(trans.__file__)),kernel_sources=sources,
        readonly_current_source_review=a.pointer(ROOT,review),actual_original100_evidence=a.pointer(ROOT,evidence)))
    rule=deepcopy(oldrule);rule.update(schema=goal.FIT_RULE_SCHEMA_V5,tasks=goal.TASK_PHASES_V5,helper=a.pointer(ROOT,Path(goal.__file__)),
        original_adapter_source=a.pointer(ROOT,Path(adapter.__file__)),original_interaction_source=a.pointer(ROOT,Path(interaction.__file__)),translation_intention_derivation_permission=a.pointer(ROOT,permission),auxiliary_execution_registrations=[])
    rulepath=out/'command_goal_rule_original_only.json';write(rulepath,rule)
    full=deepcopy(rule);full['auxiliary_execution_registrations']=oldrule['auxiliary_execution_registrations'];write(out/'command_goal_rule_explicit_complete_aux200.json',full)
    sem=deepcopy(a.read(BASE/'semantic_acquire_rule_v6_aux200_setup/semantic_acquire_rule.json'))
    sem['source_bindings']['adapter']=a.pointer(ROOT,Path(adapter.__file__))
    sempath=out/'semantic_acquire_rule_v7_explicit_aux200.json';write(sempath,sem)
    config=deepcopy(a.read(BASE/'full14_goalv4_semanticv6_execution_setup_v1/artifact_fit_configuration_v3.json'))
    config['version']='r10_original300_translation_goal_v5_CPU_candidate'
    config['command_goal_reference_rule']=a.pointer(ROOT,rulepath)
    config['semantic_acquire_derivation_rule']=a.pointer(ROOT,sempath)
    config['translation_candidate_scope']=dict(tasks=list(trans.TASKS),original_teacher_sources=300,distinct_original_parents=300,
        aux_sources_used=0,all_original_failed_censored_sources_retained=True,formal_release=False,
        original2800_and_3000_bundle_not_recreated=True,button_press_not_derived=True,
        old_v4_current_reference_counterevidence=a.pointer(ROOT,BASE/'TRAIN_artifacts_full_goalv4_semanticv6_v1/report.json'))
    configpath=out/'artifact_fit_configuration_original300.json';write(configpath,config)
    archive={}
    names=('command_goal_reference.py','translation_command_goal_reference.py','interaction.py','reference_time.py','artifact_fitting.py',
        'release_preflight.py','semantic_acquire.py','bin_command_goal_reference.py','basketball_command_goal_reference.py',
        'acquisition_lineage.py','training_view.py','acquisition_manifest.py','teacher_supplement.py','specs.py','public_roles.py','evaluation.py')
    for name in names:
        source=ROOT/'our_mani/multitask'/name;target=out/'core_source_archive'/source.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
        archive[str(source.relative_to(ROOT))]=dict(live_source=a.pointer(ROOT,source),archived_source=a.pointer(ROOT,target))
    for source in (ROOT/'our_mani/contracts.py',Path(__file__).resolve()):
        target=out/'core_source_archive'/source.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
        archive[str(source.relative_to(ROOT))]=dict(live_source=a.pointer(ROOT,source),archived_source=a.pointer(ROOT,target))
    snapshot=out/'source_snapshot.json';write(snapshot,dict(schema='r10_command_goal_production_core_source_snapshot_v5',sources=archive,
        created_utc=datetime.now(timezone.utc).isoformat(),predecessor_source_snapshot=a.pointer(ROOT,old/'source_snapshot_v2.json'),
        translation_permission=a.pointer(ROOT,permission),goal_rule=a.pointer(ROOT,rulepath),configuration=a.pointer(ROOT,configpath),
        interaction_reference_time_numeric_logic_changed=False,specs_or_collector_or_Context_changed=False,native_GPU_or_collection_started=False))
    # This short permission audit never repeats the approved aux200 raw audit.
    loaded=trans.load_permission(ROOT,a.pointer(ROOT,permission),purpose='production_fitting')
    for task in trans.TASKS:trans._kernel(loaded,task)
    return dict(rule=a.pointer(ROOT,rulepath),permission=a.pointer(ROOT,permission),configuration=a.pointer(ROOT,configpath),snapshot=a.pointer(ROOT,snapshot))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--revision',type=int,default=1);args=p.parse_args();print(json.dumps(run(args.revision),ensure_ascii=False))
