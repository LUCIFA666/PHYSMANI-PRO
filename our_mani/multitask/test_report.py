"""Publish only complete frozen R10 official TEST and its seventy fixed demos.

Previous R09 results/demos remain a separate protocol block. This module never
executes a policy, trains a model, chooses a new demo source, or changes a gate.
"""
from collections import Counter
import json
from pathlib import Path

import numpy as np

from .evaluation import METHODS,atomic_json,case_identity,comparison_protocol_sha256,digest,file_sha
from .release_preflight import normalized_lock,require_frozen_coverage_certificate
from .reset_parity import compare_five_way,profile_problems
from .specs import TASK_SPECS
from .trace_evidence import audit_selection_usage
from .validation_release import Proof,_cases,_performance,proof_directory
from .video_validation import validate_video

SCHEMA='r10_complete_fixed_official_TEST_report_v1'


def require(value,message):
    if not value:raise ValueError(message)


def official_inventory(proof,path):
    manifest=proof.read(path);rows=manifest.get('episodes',[])
    require(manifest.get('split')=='test' and manifest.get('mapping_status')=='verified'
        and manifest.get('episode_count')==len(rows)==560,'complete verified 560-source official alias mapping required')
    canonical={(row['task'],row['raw_episode_id']):row for row in rows};counts=Counter(task for task,raw in canonical)
    require(len(canonical)==560 and counts==dict.fromkeys(TASK_SPECS,40),'official TEST requires exactly forty unique sources per remaining task')
    aliases=set();first={}
    for row in rows:
        require(row.get('mapping_status')=='verified' and row.get('variation') in TASK_SPECS[row['task']].supported_variations
            and type(row.get('official_episode_index')) is int and row['official_episode_index']>=0,
            'official alias variation/index mapping is incomplete')
        alias=(row['task'],row['variation'],row['official_episode_index'])
        require(alias not in aliases,'duplicate official alias identity');aliases.add(alias)
        first.setdefault(row['task'],row['raw_episode_id'])
    return canonical,first


def official_plan(proof,path,canonical,first):
    plan=proof.read(path)
    require(plan.get('version')=='r10_fresh_policy_queue_v1'
        and plan.get('sha256')==digest({k:v for k,v in plan.items() if k!='sha256'})
        and plan.get('split')=='test' and set(plan.get('methods',[]))==set(METHODS) and len(plan['methods'])==5
        and plan.get('replicas')==1 and plan.get('source_episodes')==560 and plan.get('expected_cases')==2800
        and plan.get('official_TEST_used_for_fitting') is False
        and plan.get('case_isolation')=='fresh_simulator_process_per_episode',
        'immutable complete five-method 2800-case official TEST plan required')
    for field in ('config','manifest','frozen_inputs','parent_source','child_source'):proof.pointer(plan[field])
    source=proof.read_pointer(plan['manifest'])
    require(source.get('episodes')==list(canonical.values()),'TEST plan must keep exact canonical alias rows/order')
    identities=set();ids=set();demos=set()
    for job in plan.get('jobs',[]):
        row=job['row'];key=(row['task'],row['raw_episode_id']);identity=job['identity'];method=identity['method']
        require(key in canonical,'TEST job refers to an unknown canonical source')
        expected_demo=key[1]==first[key[0]]
        require(row==canonical[key] and method in METHODS
            and job.get('replica')==0 and job.get('exploration_seed') is None
            and job.get('source_raw_episode_id')==key[1] and job.get('record_demo') is expected_demo,
            'TEST source/method/replica or fixed first-demo selection changed')
        expected=case_identity(split='test',task=key[0],raw_episode_id=key[1],method=method,source_row=row,
            config_sha256=plan['config']['sha256'],inputs_sha256=plan['frozen_inputs']['sha256'])
        require(identity==expected and job['job_id']==expected['sha256'][:20]
            and job['job_id'] not in ids and (key,method) not in identities,'duplicate/changed official TEST identity')
        ids.add(job['job_id']);identities.add((key,method))
        if expected_demo:demos.add((key[0],method))
    require(len(plan['jobs'])==2800 and identities=={(key,m) for key in canonical for m in METHODS}
        and demos=={(task,m) for task in TASK_SPECS for m in METHODS},'complete 2800 cases/seventy first demos required')
    return plan


def _VAL_release(proof,config,test_files):
    require(config.get('formal_comparison_ready') is True,'formal TEST configuration was not released')
    pointer=config['r10_validation_release'];value=proof.read_pointer(pointer)
    require(value.get('complete') is True and value.get('protocol_acceptance_passed') is True
        and value.get('formal_test_permitted') is True and value.get('split')=='validation' and value.get('TEST_used') is False
        and value.get('expected_source_cases')==275
        and value.get('expected_method_cases')==value.get('completed_method_cases')==1375,
        'complete full-five VAL release must precede official TEST')
    proof.locked(pointer['path'],test_files,pointer['sha256'])
    old_lock=proof.read_pointer(value['validated_input_lock']);old_files=normalized_lock(proof.root,old_lock['files'])
    proof.locked(value['validated_input_lock']['path'],test_files,value['validated_input_lock']['sha256'])
    require(all(test_files.get(name)==checksum for name,checksum in old_files.items()),'TEST lock omitted or changed an original VAL input')
    proof.include(old_files);validated=proof.read_pointer(value['validated_config'])
    proof.locked(value['validated_config']['path'],old_files,value['validated_config']['sha256'])
    require(value['config_sha256']==value['validated_config']['sha256']
        and value['comparison_protocol_sha256']==comparison_protocol_sha256(validated)==comparison_protocol_sha256(config),
        'official TEST changed a policy/model/external setting from full VAL')
    require(value.get('input_files_sha256') and all(test_files.get(name)==checksum for name,checksum in
        normalized_lock(proof.root,value['input_files_sha256']).items()),'TEST lock must retain the complete VAL acceptance proof graph')
    proof.include(value['input_files_sha256'])
    require_frozen_coverage_certificate(proof.root,value['model_coverage_certificate'],config,test_files)
    return value


def _official_source_evidence(proof,row,records):
    required=row.get('source_required_files',{})
    require(required and 'target_state.pkl' in required,'official source hash evidence missing')
    for item in required.values():proof.file(item['path'],item['sha256'])
    proof.file(row['task_source_path'],row['task_source_sha256']);proof.file(row['task_scene_path'],row['task_scene_sha256'])
    for record in records.values():
        reset=record.get('recorded_initial_conditions',{}).get('reset_receipt',{})
        require(reset.get('task')==row['task'] and reset.get('raw_episode_id')==row['raw_episode_id']
            and reset.get('variation')==row['variation'] and reset.get('official_episode_index')==row['official_episode_index']
            and reset.get('reset_kind')=='released_official_TEST_family_dispatch'
            and reset.get('original_source_replayed') is True and reset.get('expert_actions_used') is False
            and reset.get('future_observations_used') is False and reset.get('target_state_sha256')==row['target_state_sha256'],
            'recorded official reset does not bind the canonical alias/initial schedule')
        source=reset.get('verified_source_inputs',{})
        require(source.get('source_required_sha256')=={name:item['sha256'] for name,item in required.items()}
            and source.get('task_source_sha256')==row['task_source_sha256']
            and source.get('task_scene_sha256')==row['task_scene_sha256'],
            'recorded reset source/scene hashes differ from official mapping')


def _demos(proof,groups,canonical,first):
    entries=[]
    for task in TASK_SPECS:
        raw=first[task]
        for method in METHODS:
            record=groups[(task,raw)][method];video=record['video'];validation=validate_video(proof.root,video)
            require(validation['decoded_frame_count']==record['counted_clock']['ticks']+1
                and record.get('demo_selection')=='fixed first official manifest source, independent of outcome',
                'formal demo must use reset plus actual counted frames of the fixed first source')
            proof.file(video['path'],video['sha256']);proof.file(validation['decoder'],validation['decoder_sha256'])
            entries.append(dict(task=task,method=method,raw_episode_id=raw,
                selection='fixed first canonical official TEST source, independent of outcome',
                formal_metrics_included=True,demo_only_replay=False,replay_result_in_formal_statistics=False,
                formal_result=dict(success=record['official_task_success'],fatal=record['native_terminal_failure']),
                identity_sha256=record['identity']['sha256'],video=video,video_validation=validation))
    require(len(entries)==70 and len({(e['task'],e['method']) for e in entries})==70,'one and only one demo per remaining task/method')
    return dict(version='r10_fixed_seventy_formal_demo_index_v1',complete=True,expected=70,completed=70,entries=entries,
        demo_selection_uses_outcome=False,additional_native_rollouts_started=False,extra_sensor_or_physics_steps_added=False,
        formal_statistics_replaced_by_replays=False,frame_convention='reset plus every counted 50ms tick; existing mandatory sensor frames only')


def _strata(groups):
    result={}
    for speed in ('ordinary','high_speed'):
        subset={key:records for key,records in groups.items() if key[0].endswith('_high_speed')==(speed=='high_speed')}
        result[speed]={method:dict(successes=sum(row[method]['official_task_success'] for row in subset.values()),
            episodes=len(subset),success_rate=sum(row[method]['official_task_success'] for row in subset.values())/len(subset)) for method in METHODS}
    return result


def finalize_official_test(root,queue_plan,out,*,
        official_manifest='artifacts/framework/multitask_r10/official_test_manifest.json'):
    """Only a complete clean 2800-case queue plus 70 decodable demos publishes."""
    proof=Proof(root);out=(proof.root/out).resolve();stage='official_inventory'
    require(out.is_relative_to(proof.root/'artifacts/framework/multitask_r10') and not out.exists(),
        'use a fresh immutable final R10 report path')
    try:
        canonical,first=official_inventory(proof,official_manifest);stage='official_queue_plan'
        plan=official_plan(proof,queue_plan,canonical,first);config=proof.read_pointer(plan['config']);plan['_config']=config
        lock=proof.read_pointer(plan['frozen_inputs']);files=normalized_lock(proof.root,lock['files']);proof.include(files)
        for field in ('config','manifest','child_source'):proof.locked(plan[field]['path'],files,plan[field]['sha256'])
        stage='prior_full_VAL_release';val=_VAL_release(proof,config,files)
        stage='complete_official_cases';groups,receipts,status=_cases(proof,plan,split='test',expected_cases=2800)
        require(set(groups)==set(canonical),'official completed-case source coverage differs')
        stage='TRAIN_tolerances';profile=proof.read_pointer(config['initial_condition_tolerance_profile'])
        require(not profile_problems(profile,evidence_root=proof.root),'frozen TRAIN reset tolerance evidence is invalid')
        proof.pointer(profile['TRAIN_evidence']);stage='official_recorded_initial_conditions';pairs=[]
        for key,records in groups.items():
            _official_source_evidence(proof,canonical[key],records)
            comparison=compare_five_way(records,profile,evidence_root=proof.root,require_frozen=True)
            require(comparison.get('complete') is True and comparison.get('accepted') is True
                and len(comparison.get('checks',[]))==10,'official recorded initial conditions differ for '+repr(key))
            pairs.append(dict(task=key[0],raw_episode_id=key[1],**comparison))
        stage='selection_trace';usage=audit_selection_usage(groups);performance=_performance(groups)
        stage='fixed_seventy_demos';demos=_demos(proof,groups,canonical,first)
        directory=proof_directory(out)
        save=lambda name,value:_save_proof(proof,directory/name,value)
        pair_pointer=save('recorded_initial_conditions.json',dict(complete=True,accepted=True,sources=560,pairs=5600,
            entries=pairs,full_native_physics_state_identical_claimed=False,metric_only=True,policy_use_allowed=False))
        case_pointer=save('canonical_cases.json',dict(complete=True,expected=2800,entries=receipts))
        usage_pointer=save('selection_usage.json',usage);demo_pointer=save('demo_index.json',demos)
        proof.file(Path(__file__),file_sha(Path(__file__)))
        for name in ('trace_evidence.py','validation_release.py','video_validation.py'):proof.file(proof.root/'our_mani/multitask'/name)
        proof.file(proof.root/'scripts/finalize_multitask_test_r10.py');proof.include(dict(proof.files))
        report=dict(schema=SCHEMA,complete=True,protocol_acceptance_passed=True,split='test',formal_TEST_used=True,
            expected_source_cases=560,expected_method_cases=2800,completed_method_cases=2800,source_cases=560,
            validated_config=plan['config'],comparison_protocol_sha256=comparison_protocol_sha256(config),
            queue_plan=dict(path=proof.name(queue_plan),sha256=file_sha(proof.root/queue_plan)),
            frozen_input_lock=plan['frozen_inputs'],validation_release=config['r10_validation_release'],
            model_coverage_certificate=val['model_coverage_certificate'],canonical_cases=case_pointer,
            recorded_initial_conditions=pair_pointer,selection_usage=usage_pointer,demo_index=demo_pointer,
            per_task=performance['per_task'],pooled=performance['pooled'],by_speed=_strata(groups),
            development_performance_screen=val.get('development_performance_screen'),
            development_performance_screen_passed=val.get('development_performance_screen_passed'),
            TEST_negative_value_geometry_differences=performance['negative_value_geometry_differences'],
            TEST_result_used_to_select_models_thresholds_or_demos=False,success_module_benefit_assumed=False,
            baseline_checkpoint_sha256=proof.files[proof.name(config['baseline_checkpoint'])],baseline_executor=config['baseline_executor'],
            comparison_interpretation='adapted shared-external-condition comparison; not original paper Table 1 reproduction',
            input_files_sha256=proof.files,native_or_GPU_launched_by_publisher=False,model_fitting_performed=False,
            full_native_physics_state_identical_claimed=False,
            limitations=['Recorded conditions are matched; opaque native solver/contact caches are unrecorded.',
                'A zero learned-override count does not establish equality of full independent trajectories.',
                'Actual success-rate differences are reported without claiming critic causality from probability calibration.'])
        atomic_json(out,report);return report
    except Exception as error:
        pending=dict(schema=SCHEMA,complete=False,protocol_acceptance_passed=False,split='test',stage=stage,
            expected_source_cases=560,expected_method_cases=2800,errors=[repr(error)],no_final_report_written=True,
            input_files_sha256=proof.files,native_or_GPU_launched_by_publisher=False,model_fitting_performed=False,
            no_infrastructure_failure_scored_as_policy_negative=True)
        directory=out.parent/'pending_TEST_report';directory.mkdir(parents=True,exist_ok=True);i=0
        while (directory/(out.stem+f'__{i:04d}.json')).exists():i+=1
        path=directory/(out.stem+f'__{i:04d}.json');atomic_json(path,pending)
        pending['pending_report']=dict(path=proof.name(path),sha256=file_sha(path));return pending


def _save_proof(proof,path,value):
    atomic_json(path,value);proof.file(path);return dict(path=proof.name(path),sha256=file_sha(path))


def combine_published_blocks(root,r10_report,out,*,r09_report='artifacts/framework/aligned_r09/final_report.json',
        r09_demos='artifacts/framework/aligned_r09/demo_index.json'):
    """A delivery index, preserving the distinct R09/R10 metric definitions."""
    proof=Proof(root);out=(proof.root/out).resolve()
    require(out.is_relative_to(proof.root/'artifacts/framework/multitask_r10') and not out.exists(),'new combined delivery path required')
    r10=proof.read(r10_report);old=proof.read(r09_report);old_demos=proof.read(r09_demos)
    require(r10.get('schema')==SCHEMA and r10.get('complete') is True and r10.get('protocol_acceptance_passed') is True
        and r10.get('expected_source_cases')==r10.get('source_cases')==560
        and r10.get('expected_method_cases')==r10.get('completed_method_cases')==2800
        and old.get('complete') is True and old.get('protocol_acceptance_passed') is True
        and old.get('total_test_runs')==old.get('expected_test_runs')==400 and not old.get('acceptance_errors'),
        'both protocol blocks must already be fully published')
    # Recheck the accepted evidence graph, rather than trusting headline counts
    # after an input, raw child or video has been removed or changed.
    require(r10.get('input_files_sha256'),'R10 published acceptance input graph is missing')
    proof.include(r10['input_files_sha256'])
    for field in ('canonical_cases','recorded_initial_conditions','selection_usage'):
        accepted=proof.read_pointer(r10[field])
        require(accepted.get('complete') is True,'published R10 '+field+' is incomplete')
    require(accepted.get('diagnostics_valid') is True,'published R10 selection usage is invalid')
    for owner,path_field,sha_field in ((old,'parallel_evaluation_inputs','parallel_evaluation_inputs_sha256'),
            (old_demos,'execution_inputs','execution_inputs_sha256')):
        graph=proof.read(owner[path_field],owner[sha_field])
        require(graph.get('files'),'published R09 frozen acceptance/demo input graph is missing')
        proof.include(graph['files'])
    new_demos=proof.read_pointer(r10['demo_index'])
    require(new_demos.get('complete') is True and new_demos.get('expected')==new_demos.get('completed')==len(new_demos['entries'])==70
        and old_demos.get('complete') is True and old_demos.get('expected')==old_demos.get('completed')==len(old_demos['entries'])==10
        and old_demos.get('formal_final_report_sha256')==file_sha(proof.root/r09_report)
        and old_demos.get('original_formal_statistics_replaced') is False
        and old_demos.get('replay_results_in_formal_statistics') is False,'published demo blocks missing or metric definitions changed')
    entries=[]
    for entry in old_demos['entries']:
        require(entry.get('task') in ('pick_moving_target_on_the_table','pick_moving_target_on_the_table_high_speed')
            and entry.get('method') in METHODS and entry.get('replay_included_in_formal_statistics') is False,
            'R09 demo identity or replay metric scope differs')
        formal=proof.read(entry['formal_record'],entry['formal_record_sha256'])
        proof.file(entry['original_formal_child_record'],entry['original_formal_child_record_sha256'])
        require(formal.get('official_task_success')==entry['formal_result']['official_task_success'],
            'R09 formal result was replaced by a demo replay result')
        if entry['source']=='independent_demo_only_replay':
            replay=proof.read(entry['replay_record'],entry['replay_record_sha256'])
            proof.file(entry['recording_receipt'],entry['recording_receipt_sha256'])
            require(entry.get('formal_metrics_included') is False
                and replay.get('official_task_success')==entry['replay_result']['official_task_success'],
                'R09 independent replay result or metric exclusion differs')
        else:
            require(entry['source']=='formal_rollout_video' and entry.get('formal_metrics_included') is True
                and entry.get('replay_record') is None and entry.get('replay_result') is None,
                'unknown R09 recording provenance')
        video=dict(path=entry['video'],sha256=entry['video_sha256'],decoded_frame_count=entry['video_validation']['decoded_frame_count'])
        decoded=validate_video(proof.root,video);proof.file(video['path'],video['sha256'])
        proof.file(decoded['decoder'],decoded['decoder_sha256'])
        entries.append(dict(protocol_block='R09',**entry))
    for entry in new_demos['entries']:
        require(entry.get('task') in TASK_SPECS and entry.get('method') in METHODS
            and entry.get('formal_metrics_included') is True and entry.get('demo_only_replay') is False
            and entry.get('replay_result_in_formal_statistics') is False,'R10 demo formal metric scope differs')
        decoded=validate_video(proof.root,entry['video']);proof.file(entry['video']['path'],entry['video']['sha256'])
        proof.file(decoded['decoder'],decoded['decoder_sha256'])
        entries.append(dict(protocol_block='R10',**entry))
    require(len(entries)==80 and len({(row['task'],row['method']) for row in entries})==80,'exactly one demo for sixteen tasks/five methods required')
    proof.file(Path(__file__),file_sha(Path(__file__)))
    proof.file(proof.root/'our_mani/multitask/video_validation.py')
    proof.file(proof.root/'scripts/finalize_multitask_test_r10.py')
    proof.include(dict(proof.files))
    result=dict(version='r09_r10_separate_protocol_delivery_index_v1',complete=True,formal_run_count_only=3200,
        demo_expected=80,demo_completed=80,entries=entries,blocks=dict(
            R09=dict(report=dict(path=proof.name(r09_report),sha256=file_sha(proof.root/r09_report)),formal_runs=400,
                demos=10,baseline_executor=old['baseline_executor'],development_performance_screen_passed=old.get('development_performance_screen_passed'),
                metric_definition='original frozen two-task R09 statistics; eight independent demo-only replays stay excluded'),
            R10=dict(report=dict(path=proof.name(r10_report),sha256=file_sha(proof.root/r10_report)),formal_runs=2800,
                demos=70,baseline_executor=r10['baseline_executor'],metric_definition='full remaining14 official native tasks; first-case videos are formal case recordings')),
        pooled_cross_protocol_success_rate_reported=False,old_formal_statistics_modified=False,
        replay_outcomes_relabelled_as_formal_results=False,protocol_definitions_combined=False,input_files_sha256=proof.files)
    atomic_json(out,result);return result
