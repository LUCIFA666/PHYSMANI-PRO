"""CPU metadata-only full-source budget draft; cannot launch native collection.

DRAFT versions deliberately differ from the executable collector plan schema.
No expert arrays, images, TEST trajectories, simulator or GPU are read here.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import collect_multitask_training_r10 as c
from our_mani.multitask.specs import TASK_SPECS


def build_drafts(demonstrations,protocol_path,output):
    metadata,rows=c.demonstration_registry(demonstrations)
    protocol=json.loads(Path(protocol_path).read_text());c.validate_collection_protocol(protocol)
    if c.sha(protocol['experiment_plan_path'])!=protocol['experiment_plan_sha256']:
        raise c.CollectionError('experiment plan changed since protocol source binding')
    output=c.destination(output)
    counts={split:Counter(row['task'] for row in rows if row['split']==split) for split in ('train','validation')}
    if counts['train']!=dict.fromkeys(TASK_SPECS,100) or sum(counts['validation'].values())!=275:
        raise c.CollectionError('all1400 TRAIN/275 true VAL sources required, never a smoke-subset replacement')
    original=c.BASE/'original_TRAIN_action_phase_priors.json'
    priors=json.loads(original.read_text())
    if priors.get('sources_read')!=1400 or priors.get('demonstration_manifest_sha256')!=c.sha(demonstrations):
        raise c.CollectionError('all1400 original TRAIN priors must remain separately bound')
    draft_protocol=dict(protocol,draft=True,plan_frozen=False,native_execution_authorized=False,
        version='r10_full_source_teacher_hold_protocol_DRAFT_v1',execution_scope='full_source_preparation_DRAFT',
        full_source_training_started=False,full_source_training_parents_completed=0,
        authorized_scope='metadata preparation only; full native execution awaits root after geometry/rigid validation',
        orientation_and_component_priors_ready_for_full_collection=False,
        full_collection_prerequisites=['TRAIN visual/renderable component flags audit and geometry registration validated',
            'TRAIN rigid layout/group orientation priors validated and explicit full/axis/none annotations',
            'required role coverage and current/past public tracking without native poses',
            'updated source/experiment/asset SHA freezes for each actual executable plan',
            'VAL uses frozen TRAIN templates/layouts; no VAL teacher or geometry calibration'])
    draft_protocol.pop('smoke_budget',None)
    draft_protocol['full_collection_budget']=dict(TRAIN_parents=1400,TRAIN_teacher_cases=1400,
        TRAIN_hold_cases=1400,VAL_hold_cases=275,total_native_cases=3075)
    c.write_json(output/'common_full_training_config_r10_DRAFT.json',draft_protocol)
    drafts={}
    for split in ('train','validation'):
        profiles=c.behavior_profiles(protocol,split);cases=[]
        for row in rows:
            if row['split']!=split:continue
            _,task,parent=c.source_identity(row)
            for behavior,profile in profiles.items():
                for replica in profile['replica_indices']:
                    identity=dict(split=split,task=task,raw=parent,replica=replica,behavior=behavior)
                    cases.append(dict(case_id=c.canonical_sha(identity)[:16],split=split,task=task,
                        raw_episode_id=f'{parent}__{behavior}_r{replica}',parent_raw_episode_id=parent,
                        parent_source_group=f'{split}:{task}:{parent}',behavior=behavior,behavior_replica=replica,
                        teacher_used=profile['teacher_used'],context_primary_view=profile['context_primary_view'],
                        interaction_eligible=profile['interaction_eligible'],maximum_collection_ticks=profile['maximum_ticks'],
                        source_kind=('TRAIN_native_generated_augmentation' if profile['teacher_used'] else 'TRAIN_native_generated_hold')
                            if split=='train' else 'VAL_native_generated_non_teacher',
                        source=dict(task=task,path=row['source_path'],variation=row['variation'],raw_episode_id=parent),
                        offline_demonstration=dict(path=row['npz_path'],sha256=row['npz_sha256'],
                            source_file_stat=row['source_file_stat'],whole_HDF5_sha256_recomputed=False),
                        frozen_TRAIN_template_asset='PENDING' if split=='validation' else None,
                        frozen_TRAIN_rigid_layout='PENDING',TRAIN_orientation_annotations_approved=False))
        drafts[split]=dict(version='r10_full_source_behavior_collection_DRAFT_v1',created_utc=c.utc(),
            split=split,draft=True,plan_frozen=False,native_execution_authorized=False,
            collector_plan_version_intentionally_unsupported=True,
            dynamics_target_label_schema=protocol.get('dynamics_target_label_schema','legacy_raw_native_reported_velocity'),
            demonstration_manifest=dict(path=str(Path(demonstrations).resolve()),sha256=c.sha(demonstrations)),
            original_TRAIN_priors=dict(path=str(original),sha256=c.sha(original),source_count=1400),
            source_counts_by_task=dict(counts[split]),real_parent_sources=sum(counts[split].values()),
            cases=cases,case_count=len(cases),behavior_budgets=profiles,
            maximum_physics_ticks=sum(case['maximum_collection_ticks'] for case in cases),
            maximum_simulated_seconds=sum(case['maximum_collection_ticks']*.05 for case in cases),
            acquisition_parent_group='split:task:original_parent_raw_episode_id',
            original_HDF5_motion_replay=False,teacher_only_interaction=True,hold_only_primary_context=True,
            model_training_started=False,automatic_outcome_selected_resampling=False,
            smoke_parent_replaces_full_source_inventory=False,
            source_integrity='manifest/cache hashes and original stat bound; no new full HDF5 hash or expert array read',
            remaining_hard_gates=draft_protocol['full_collection_prerequisites'])
        if len({(row['task'],row['raw_episode_id']) for row in cases})!=len(cases):raise c.CollectionError('draft raw identity duplicates')
        c.write_json(output/f'{split}_collection_DRAFT.json',drafts[split])
    train_keys={(x['task'],x['parent_raw_episode_id']) for x in drafts['train']['cases']}
    val_keys={(x['task'],x['parent_raw_episode_id']) for x in drafts['validation']['cases']}
    if train_keys&val_keys:raise c.CollectionError('TRAIN/VAL parent sources overlap')
    summary=dict(version='r10_full_collection_budget_DRAFT_v1',draft=True,native_execution_authorized=False,
        train_parent_sources=1400,TRAIN_teacher_cases=1400,TRAIN_hold_cases=1400,VAL_hold_cases=275,
        TRAIN_max_ticks=364000,VAL_max_ticks=16500,total_native_cases=3075,
        actual_full_source_training_completed=0,source_counts_by_task={s:dict(v) for s,v in counts.items()},
        validation_shortfall_against_nominal20_per_task={task:20-counts['validation'][task] for task in TASK_SPECS
            if counts['validation'][task]!=20},
        budget_excludes_critic_and_five_method_evaluation=True,wall_time_estimate_claimed=False,
        TEST_read=False,VAL_expert_arrays_read=False,native_started=False,GPU_used=False,
        experiment_plan=dict(path=protocol['experiment_plan_path'],sha256=protocol['experiment_plan_sha256']),
        draft_inputs={str(Path(demonstrations).resolve()):c.sha(demonstrations),str(Path(protocol_path).resolve()):c.sha(protocol_path),
            str(Path(__file__).resolve()):c.sha(__file__)},
        future_actual_plan_command_template=[
            '.conda-env/bin/python -B scripts/collect_multitask_training_r10.py --prepare-plan --split train --protocol APPROVED_FINAL_CONFIG --plan NEW_TRAIN_EXECUTABLE_PLAN --offline-priors artifacts/framework/multitask_r10/original_TRAIN_action_phase_priors.json',
            '.conda-env/bin/python -B scripts/collect_multitask_training_r10.py --prepare-plan --split validation --protocol APPROVED_FINAL_CONFIG --plan NEW_VAL_EXECUTABLE_PLAN --template-assets FROZEN_TRAIN_ASSETS --rigid-layouts FROZEN_TRAIN_LAYOUTS --validation-mode hold'],
        execution_commands_not_run=True)
    summary['dynamics_target_label_schema']=protocol.get('dynamics_target_label_schema','legacy_raw_native_reported_velocity')
    c.write_json(output/'budget_summary_DRAFT.json',summary)
    text=(f"全量准备草案：TRAIN 1400真实父源 × (teacher1条200ticks + hold1条60ticks) = 2800case；VAL 275真实父源 × hold1条60ticks。\n"
        "共3075次native补采上限；完整原1400示教priors另列，不把smoke样本替代全源。\n"
        "VAL真实短缺：普通peg18、高速peg19、高速bin18；其余20。不得补造或用TRAIN顶替。\n"
        "本工具仅metadata预算，不冻结可执行计划、不启动native/GPU、不读VAL专家数组或TEST。\n"
        "geometry/renderable flags与TRAIN rigid/orientation验证后，由root批准新配置和源/资产锁；失败样本保留，不按成功补采。\n")
    (output/'budget_summary_DRAFT.txt').write_text(text)
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--demonstrations',type=Path,default=c.BASE/'demonstrations/manifest.json')
    parser.add_argument('--protocol',type=Path,default=c.BASE/'common_training_smoke_config_v5.json')
    parser.add_argument('--output',type=Path,default=c.BASE/'full_collection_preparation_DRAFT')
    args=parser.parse_args();summary=build_drafts(args.demonstrations,args.protocol,args.output)
    print(json.dumps({k:summary[k] for k in ('train_parent_sources','TRAIN_teacher_cases','TRAIN_hold_cases','VAL_hold_cases','total_native_cases','native_started')}))
if __name__=='__main__':main()
