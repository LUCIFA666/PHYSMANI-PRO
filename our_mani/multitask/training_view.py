"""SHA-verified causal training views over fixed TRAIN/VAL acquisitions.

Short native terminal acquisitions remain in an immutable ledger with zero
training weight. No cache is copied, padded, repaired, outcome-filtered or
resampled. A parent must have an actual usable example in another declared
fixed-budget acquisition, otherwise the coverage gate fails.
"""
from collections import Counter,defaultdict
from pathlib import Path
import json

import numpy as np

from .evaluation import atomic_json,file_sha
from .acquisition_manifest import load_acquisition_manifest,actual_case_plan_sha,COMPOSITE_VERSION
from .specs import TASK_SPECS,get_task_spec


def require(value,message):
    if not value: raise ValueError(message)


def read(path): return json.loads(Path(path).read_text())


def local(root,path):
    root=Path(root).resolve();path=(root/path).resolve()
    require(path.is_relative_to(root),'new cache/receipt inputs must remain inside desktop project')
    return path


def window_statistics(arrays,*,steps=22,stride=2):
    """Exact examples() mask/count logic on a counted native 50ms clock.

    Pose normalization does not determine eligibility; all finite history and
    labels are independently checked below. A mode/unknown-target boundary is
    permanently accumulated, exactly as track_dataset.examples does.
    """
    require(steps==22 and stride==2,'R10 view is fixed at 1.1s / stride2')
    times=arrays['times'];n=len(times);per_body={}
    for j,body in enumerate(map(str,arrays['body_ids'])):
        valid,modes=arrays['valid'][:,j],arrays['modes'][:,j]
        usable=full=0
        for anchor in range(4,n-1,stride):
            mode=int(modes[anchor])
            if mode==1 or not valid[anchor]: continue
            begin=anchor
            while (begin>max(0,anchor-19) and valid[begin-1] and modes[begin-1]==mode
                   and times[begin]-times[begin-1]<=.075):
                begin-=1
            if anchor-begin+1<5: continue
            end=min(n,anchor+steps+1);count=end-anchor
            boundary=np.maximum.accumulate((modes[anchor:end]!=mode)|~arrays['target_valid'][anchor:end,j])
            mask=np.zeros(steps+1,bool)
            mask[:count]=~boundary & np.isclose(times[anchor:end],times[anchor]+np.arange(count)*.05,atol=1e-6)
            mask[0]=False
            if not mask.any(): continue
            usable+=1;full+=int(mask[-1])
        per_body[body]=dict(usable_causal_examples=usable,full_1p1s_examples=full)
    return dict(usable_causal_examples=sum(row['usable_causal_examples'] for row in per_body.values()),
        full_1p1s_examples=sum(row['full_1p1s_examples'] for row in per_body.values()),per_body=per_body,
        steps=22,stride=2,minimum_history_frames=5,attached_external_dynamics_examples=False)


def validate_arrays(arrays,entry,receipt):
    required=('times','body_ids','positions','rotations','linear_velocities','angular_velocities','confidence',
        'valid','modes','rotation_observable','target_positions','target_rotations','target_linear_velocities',
        'target_angular_velocities','target_valid','target_rotation_observable')
    require(all(name in arrays for name in required),'native causal body cache lacks explicit history or target arrays')
    times=np.asarray(arrays['times']);n=len(times);ids=list(map(str,arrays['body_ids']));b=len(ids)
    require(times.shape==(n,) and 1<=n<=201 and np.isfinite(times).all() and abs(times[0])<=1e-6
        and np.allclose(np.diff(times),.05,atol=1e-6,rtol=0),'native cache must preserve its full counted 50ms clock')
    expected={role.name for role in get_task_spec(entry['task']).roles if role.kind=='body'}
    expected-=set(get_task_spec(entry['task']).audit_only_goal_roles)
    require(len(ids)==len(set(ids)) and set(ids)==expected,'physical body IDs differ from native task spec')
    shapes={name:(n,b,3) for name in ('positions','linear_velocities','angular_velocities',
        'target_positions','target_linear_velocities','target_angular_velocities')}
    shapes.update(rotations=(n,b,3,3),target_rotations=(n,b,3,3))
    shapes.update({name:(n,b) for name in ('confidence','valid','modes','rotation_observable',
        'target_valid','target_rotation_observable')})
    for name,shape in shapes.items(): require(arrays[name].shape==shape,'native cache shape differs: '+name)
    for name in ('valid','rotation_observable','target_valid','target_rotation_observable'):
        require(arrays[name].dtype==np.bool_,'native label masks must remain boolean')
    schema=entry.get('dynamics_target_label_schema')
    if schema is not None:
        from .motion_labels import SCHEMA,RAW_LINEAR,RAW_ANGULAR
        require(schema==SCHEMA,'unknown dynamics target velocity-label schema')
        for name in (RAW_LINEAR,RAW_ANGULAR):
            require(name in arrays and arrays[name].shape==(n,b,3),'v6 raw native reported twist evidence missing')
        for name in ('target_twist_valid','target_pose_valid_before_twist_mask'):
            require(name in arrays and arrays[name].shape==(n,b) and arrays[name].dtype==np.bool_,
                'v6 explicit pre-twist/interval mask missing')
        require(not arrays['target_twist_valid'][0].any() and np.array_equal(arrays['target_valid'],
            arrays['target_pose_valid_before_twist_mask']&arrays['target_twist_valid']),
            'v6 target validity must conservatively preserve its pose/derived-twist boundary mask')
    require(np.isin(arrays['modes'],(0,1,2)).all(),'unknown motion mode code')
    require(not np.any(arrays['rotation_observable']&~arrays['valid']),
        'history orientation evidence requires valid current history')
    for mask in ('mode_valid','dynamics_segment_valid'):
        if mask in arrays:
            require(arrays[mask].shape==(n,b) and arrays[mask].dtype==np.bool_,'invalid native mask shape')
            require(not np.any(arrays['valid']&~arrays[mask]) and not np.any(arrays['target_valid']&~arrays[mask]),
                'an invalid mode/controlled segment entered history or dynamics targets')
    if 'native_pose_valid' in arrays:
        require(arrays['native_pose_valid'].shape==(n,b) and arrays['native_pose_valid'].dtype==np.bool_
            and not np.any(arrays['target_valid']&~arrays['native_pose_valid']),
            'invalid native supervision was promoted to a valid target')
    valid=arrays['valid'];confidence=arrays['confidence'][valid]
    require(np.isfinite(confidence).all() and np.all((confidence>=0)&(confidence<=1)),
        'nonfinite or out-of-range causal history confidence')
    for prefix,mask in (('',valid),('target_',arrays['target_valid'])):
        for suffix in ('positions','rotations','linear_velocities','angular_velocities'):
            require(np.isfinite(arrays[prefix+suffix][mask]).all(),'nonfinite valid history/target labels')
        rotations=arrays[prefix+'rotations'][mask]
        require(np.allclose(rotations@np.swapaxes(rotations,-1,-2),np.eye(3),atol=1e-4,rtol=0)
            and np.all(np.linalg.det(rotations)>.9999),'valid orientation is not SO3')
    alignment=receipt.get('alignment',{});clock=alignment.get('clock',{});ticks=clock.get('ticks')
    require(type(ticks) is int and 0<=ticks<=200 and n==ticks+1 and entry.get('frames')==n
        and clock.get('dt')==.05 and clock.get('budget_seconds')==10.
        and abs(clock.get('simulation_seconds',-1)-ticks*.05)<=1e-6,
        'acquisition ledger does not match original cache frame axis')
    for name in ('actual_physics_seconds','actual_task_seconds'):
        value=alignment.get(name)
        require(isinstance(value,(int,float)) and np.isfinite(value) and abs(value-ticks*.05)<=2e-5,
            'acquisition actual native clock differs: '+name)
    return window_statistics(arrays)


def parent_sources(root,path,split,expected):
    path=local(root,path);manifest=read(path);rows=manifest.get('episodes',[])
    require(manifest.get('split')==split,'parent source manifest split differs')
    rows=[row for row in rows if row.get('task') in TASK_SPECS]
    parents={(row['task'],row['raw_episode_id']) for row in rows}
    require(len(rows)==len(parents)==expected,'real original parent source coverage differs')
    return parents,dict(path=str(path),sha256=file_sha(path))


def acquisition_profiles(values,budget,split):
    """Validate the full predeclared budget, before inspecting any outcomes."""
    behavioral=budget.get('version')=='r10_fixed_behavior_budget_v1'
    if behavioral:
        expected={'TRAIN_teacher_committed_waypoint':(200,True,False,True),
            'TRAIN_hold_motion_identification':(60,False,True,False)} if split=='train' else {
            'VAL_hold_motion_identification':(60,False,True,False)}
        profiles=budget.get('behaviors',{})
        require(isinstance(profiles,dict) and set(profiles)==set(expected),
            'fixed behavior budget must declare teacher+hold TRAIN or nonteacher hold VAL')
        for behavior,(ticks,teacher,context,interaction) in expected.items():
            profile=profiles[behavior];registered=profile.get('replica_indices')
            require(type(profile.get('replicas')) is int and profile['replicas']>=1
                and isinstance(registered,list) and len(registered)==profile['replicas']
                and all(type(replica) is int and replica>=0 for replica in registered)
                and len(set(registered))==len(registered)
                and type(profile.get('maximum_ticks')) is int and profile['maximum_ticks']==ticks
                and profile.get('teacher_used') is teacher and profile.get('context_primary_view') is context
                and profile.get('interaction_eligible') is interaction,
                'behavior budget/primary view/teacher/replica declaration differs: '+behavior)
        parents={(row['task'],row.get('parent_raw_episode_id')) for row in values}
        require(all(row.get('behavior') in profiles for row in values),'undeclared acquisition behavior')
        per_behavior={behavior:sum(row.get('behavior')==behavior for row in values) for behavior in profiles}
        require(budget.get('selected_parent_sources')==len(parents)
            and budget.get('planned_cases_by_behavior')==per_behavior
            and per_behavior=={name:len(parents)*profile['replicas'] for name,profile in profiles.items()},
            'every parent must retain every predeclared behavior acquisition')
        for parent in parents:
            for behavior,profile in profiles.items():
                rows=[row for row in values if (row['task'],row.get('parent_raw_episode_id'))==parent
                    and row.get('behavior')==behavior]
                replicas=[row.get('behavior_replica') for row in rows]
                require(len(rows)==profile['replicas'] and len(set(replicas))==len(replicas)
                    and all(type(replica) is int for replica in replicas)
                    and set(replicas)==set(profile['replica_indices']),
                    'missing/duplicate or undeclared replica within a fixed parent behavior batch')
        return profiles
    require('version' not in budget and not any('behavior' in row for row in values),
        'unknown or partially declared acquisition behavior budget')
    require(type(budget.get('replicas')) is int and budget['replicas']>=1,
        'fixed acquisition replica budget is missing')
    counts=Counter((row['task'],row.get('parent_raw_episode_id')) for row in values)
    require(budget.get('selected_parent_sources')==len(counts)
        and len(values)==len(counts)*budget['replicas'] and set(counts.values())=={budget['replicas']},
        'fixed K replicas must account for every selected acquisition parent')
    registered=budget.get('registered_replica_indices',list(range(budget['replicas'])))
    require(isinstance(registered,list) and len(registered)==budget['replicas']
        and all(type(replica) is int and replica>=0 for replica in registered)
        and len(set(registered))==len(registered),'invalid predeclared fixed replica index set')
    for parent in counts:
        replicas=[row.get('augmentation_replica') for row in values
            if (row['task'],row.get('parent_raw_episode_id'))==parent]
        require(all(type(replica) is int and replica>=0 for replica in replicas)
            and len(set(replicas))==len(replicas) and set(replicas)==set(registered),
            'duplicate/missing replica or undeclared replica index within a fixed batch')
    return None


def load_acquisitions(root,manifest_paths,split,files):
    entries=[];ledger=[];lineage=[];seen=set()
    for name in manifest_paths:
        path=local(root,name);manifest,composite_proof=load_acquisition_manifest(root,path)
        files[str(path)]=file_sha(path)
        if composite_proof is not None:
            for filename,checksum in composite_proof['verified_files_sha256'].items():
                require(filename not in files or files[filename]==checksum,'conflicting historical lineage input SHA')
                files[filename]=checksum
        values=manifest.get('entries',[]);receipts=manifest.get('case_receipts',[])
        require(manifest.get('version') in ('r10_causal_body_tracks_v1',COMPOSITE_VERSION) and values
            and manifest.get('complete') is True and manifest.get('TEST_used') is False
            and not manifest.get('missing_case_ids')
            and manifest.get('expected')==manifest.get('completed')==len(values)==len(receipts),
            'every fixed-budget declared acquisition must be complete before making a training view')
        budget=manifest.get('augmentation_budget',{})
        require(budget.get('planned_native_cases')==len(values)
            and budget.get('completed_native_cases')==len(values),
            'fixed acquisition replica budget/coverage is missing or incomplete')
        require(all(row.get('split')==split and row.get('task') in TASK_SPECS for row in values),
            'TRAIN and VAL acquisition manifests must be separate; TEST is forbidden')
        require(all(row.get('dynamics_target_label_schema')==manifest.get('dynamics_target_label_schema') for row in values),
            'acquisition manifest and every entry must use the same dynamics target label schema')
        profiles=acquisition_profiles(values,budget,split)
        declared={(row['task'],row['raw_episode_id']):row for row in values}
        require(len(declared)==len(values),'duplicate acquisition rollout identity')
        matched=set()
        for item in receipts:
            case_path=local(root,item['path']);require(file_sha(case_path)==item['sha256'],'native case receipt changed')
            files[str(case_path)]=item['sha256'];record=read(case_path);entry=record.get('entry',{})
            profile=profiles.get(entry.get('behavior')) if profiles is not None else dict(
                teacher_used=split=='train',context_primary_view=True,interaction_eligible=split=='train',maximum_ticks=200)
            key=(entry.get('task'),entry.get('raw_episode_id'))
            require(key in declared and key not in matched and key not in seen and entry==declared[key],
                'native acquisition identity duplicated or differs from its manifest')
            require(record.get('status')=='completed' and record.get('split')==split
                and record.get('case_id')==item.get('case_id')
                and record.get('plan_sha256')==actual_case_plan_sha(manifest,composite_proof,item.get('case_id'))
                and record.get('case_isolation')=='fresh_simulator_process_per_episode'
                and type(record.get('process_id')) is int and record['process_id']>0
                and record.get('teacher_actions_used') is profile['teacher_used']
                and record.get('policy_history_contains_native_GT') is False,
                'incomplete, wrongly supervised or non-fresh acquisition receipt')
            parent=entry.get('parent_raw_episode_id')
            require(isinstance(parent,str) and parent and entry.get('parent_source_group')==
                f"{split}:{entry['task']}:{parent}" and entry.get('original_HDF5_motion_replayed') is False
                and entry.get('observation_source')=='causal_RGBD_and_proprioception'
                and entry.get('target_source')=='TRAIN_VAL_native_supervision'
                and entry.get('source_kind')==(('TRAIN_native_generated_augmentation' if profile['teacher_used'] else
                    'TRAIN_native_generated_hold') if split=='train'
                    else 'VAL_native_generated_non_teacher'),
                'acquisition parent/history/target namespaces differ')
            if profiles is not None:
                require(all(entry.get(field) is profile[field] for field in
                    ('teacher_used','context_primary_view','interaction_eligible'))
                    and entry.get('maximum_collection_ticks')==profile['maximum_ticks']
                    and entry.get('augmentation_replica')==entry['behavior_replica']
                    and entry['raw_episode_id']==f"{parent}__{entry['behavior']}_r{entry['behavior_replica']}",
                    'acquisition does not match its predeclared behavior/replica/primary-view identity')
            outputs=record.get('files_sha256',{});require(outputs,'native outputs are not SHA-bound')
            for filename,checksum in outputs.items():
                output=local(root,filename);require(file_sha(output)==checksum,'native acquisition output changed')
                require(str(output) not in files or files[str(output)]==checksum,'conflicting cache SHA in fixed batches')
                files[str(output)]=checksum
            cache=local(root,entry['cache'])
            require(outputs.get(entry['cache'])==entry['cache_sha256'],'NPZ is not bound to native receipt')
            process=record.get('parent_process_receipt')
            require(isinstance(process,dict),'native raw child/process receipt is required')
            raw_path=local(root,process['child_case_path']);launcher=local(root,process['launcher'])
            require(file_sha(raw_path)==process['child_case_sha256'],'raw child changed')
            if composite_proof is None:
                require(file_sha(launcher)==process['launcher_sha256'],'launch source changed')
            require(read(raw_path)=={k:v for k,v in record.items() if k!='parent_process_receipt'},
                'canonical acquisition is not its exact unmodified raw child receipt')
            files[str(raw_path)]=process['child_case_sha256']
            if composite_proof is None:
                files[str(launcher)]=process['launcher_sha256']
            with np.load(cache,allow_pickle=False) as stored:
                arrays={name:stored[name].copy() for name in stored.files}
            stats=validate_arrays(arrays,entry,record)
            require(record['alignment']['clock']['ticks']<=profile['maximum_ticks'],
                'acquisition exceeded its fixed behavior tick budget')
            selected=stats['usable_causal_examples']>0 and profile['context_primary_view']
            reason=(None if selected else 'not_primary_hold_motion_context_view' if not profile['context_primary_view']
                else 'fewer_than_five_observed_frames' if len(arrays['times'])<5 else 'no_usable_causal_future_window')
            ledger.append(dict(split=split,task=entry['task'],raw_episode_id=entry['raw_episode_id'],
                parent_raw_episode_id=parent,parent_source_group=entry['parent_source_group'],
                cache=entry['cache'],cache_sha256=entry['cache_sha256'],frames=len(arrays['times']),
                acquisition_receipt=dict(path=str(case_path),sha256=item['sha256']),
                raw_child_receipt=dict(path=str(raw_path),sha256=process['child_case_sha256']),
                native_acquisition_alignment=record['alignment'],included_in_training_view=selected,
                exclusion_reason=reason,training_weight=1 if selected else 0,window_statistics=stats,
                behavior=entry.get('behavior','legacy_teacher' if split=='train' else 'legacy_nonteacher'),
                behavior_replica=entry.get('behavior_replica',entry.get('augmentation_replica')),
                context_primary_view=profile['context_primary_view'],teacher_actions_used=profile['teacher_used'],
                interaction_eligible=profile['interaction_eligible'],maximum_collection_ticks=profile['maximum_ticks'],
                dynamics_target_label_schema=entry.get('dynamics_target_label_schema'),
                native_outcome_used_for_selection=False,original_cache_modified=False))
            if selected: entries.append(entry)
            seen.add(key);matched.add(key)
        require(matched==set(declared),'declared native attempts were silently omitted')
        lineage.append(dict(path=str(path),sha256=file_sha(path),split=split,declared_acquisitions=len(values),
            fixed_augmentation_budget=budget,plan_sha256=manifest.get('plan_sha256'),
            actual_plan_sha256_values=(composite_proof.get('actual_plan_sha256_values')
                if composite_proof is not None else None),
            historical_composite=manifest.get('version')==COMPOSITE_VERSION,
            execution_kind=(composite_proof.get('execution_kind','historical_composite')
                if composite_proof is not None else 'ordinary_single_plan'),
            first_execution_VAL_shared_profile=(composite_proof is not None
                and composite_proof.get('first_execution') is True),
            shared_profile_execution_registration=manifest.get('shared_profile_execution_registration'),
            declared_protocol_transitions=(composite_proof.get('declared_protocol_transitions')
                if composite_proof is not None else None),
            original_receipts_rewritten=False,
            dynamics_target_label_schema=manifest.get('dynamics_target_label_schema')))
    return entries,ledger,lineage


def require_production_behavior_budget(lineage):
    """Current preregistered experiment has exactly one teacher/hold replica."""
    for row in lineage:
        budget=row.get('fixed_augmentation_budget',{})
        require(budget.get('version')=='r10_fixed_behavior_budget_v1'
            and budget.get('behaviors')
            and all(profile.get('replicas')==1 and profile.get('replica_indices')==[0]
                for profile in budget['behaviors'].values()),
            'formal context acquisition must preserve preregistered K1/[0] behavior budget')


def merge_training_view(root,train_manifests,validation_manifests,train_sources,validation_sources,out,
                        *,expected_train=1400,expected_validation=275,development_only=False):
    root=Path(root).resolve();out=local(root,out)
    require(development_only or (expected_train,expected_validation)==(1400,275),
        'formal training view requires the full real 1400/275 parents')
    require(not out.exists() or not any(out.iterdir()),'preserve prior merger evidence; use a fresh version')
    out.mkdir(parents=True,exist_ok=True);files={};problems=[]
    try:
        train_expected,tp=parent_sources(root,train_sources,'train',expected_train)
        val_expected,vp=parent_sources(root,validation_sources,'validation',expected_validation)
        files[tp['path']]=tp['sha256'];files[vp['path']]=vp['sha256']
        require(not train_expected&val_expected,'TRAIN/VAL raw parent sources overlap')
        train,tl,tline=load_acquisitions(root,train_manifests,'train',files)
        validation,vl,vline=load_acquisitions(root,validation_manifests,'validation',files)
        behavior_versions=[row['fixed_augmentation_budget'].get('version')=='r10_fixed_behavior_budget_v1'
            for row in tline+vline]
        require(not any(behavior_versions) or all(behavior_versions),
            'legacy teacher views cannot replace missing hold parents in a v5 primary-context view')
        require(development_only or (behavior_versions and all(behavior_versions)),
            'production context training view requires explicit fixed teacher/hold behavior acquisitions')
        if not development_only:require_production_behavior_budget(tline+vline)
        schemas={row.get('dynamics_target_label_schema') for row in tline+vline}
        require(len(schemas)==1,'v5 raw and v6 derived-twist acquisition batches cannot be mixed')
        dynamics_schema=next(iter(schemas))
        ledger=tl+vl;entries=train+validation
        by_split={split:[row for row in ledger if row['split']==split] for split in ('train','validation')}
        coverage={};full={}
        for split,expected in (('train',train_expected),('validation',val_expected)):
            acquired={(row['task'],row['parent_raw_episode_id']) for row in by_split[split]}
            usable={(row['task'],row['parent_raw_episode_id']) for row in by_split[split] if row['included_in_training_view']}
            selected_rows=[row for row in by_split[split] if row['included_in_training_view']]
            full_tasks={row['task'] for row in selected_rows if row['window_statistics']['full_1p1s_examples']>0}
            tasks={task for task,_ in expected}
            if acquired-expected: problems.append(split+': undeclared original parent sources')
            if expected-acquired: problems.append(split+': missing acquired original parents')
            if expected-usable: problems.append(split+': original parents without an actual usable causal example')
            if tasks-full_tasks: problems.append(split+': tasks without full 1.1s supervision')
            coverage[split]=dict(expected_parent_sources=len(expected),acquired_parent_sources=len(acquired),
                usable_parent_sources=len(usable),missing_acquired=sorted(expected-acquired),
                missing_usable=sorted(expected-usable),unexpected_parents=sorted(acquired-expected),
                usable_examples=sum(row['window_statistics']['usable_causal_examples'] for row in selected_rows),
                diagnostic_examples_in_all_acquisitions=sum(row['window_statistics']['usable_causal_examples'] for row in by_split[split]),
                included_augmentations=sum(row['included_in_training_view'] for row in by_split[split]),
                excluded_augmentations=sum(not row['included_in_training_view'] for row in by_split[split]))
            full[split]=dict(full_horizon_tasks=sorted(full_tasks),missing_full_horizon_tasks=sorted(tasks-full_tasks),
                full_horizon_examples=sum(row['window_statistics']['full_1p1s_examples'] for row in selected_rows))
        for path in (Path(__file__),root/'our_mani/multitask/track_dataset.py',root/'our_mani/multitask/context_se3.py'):
            files[str(path)]=file_sha(path)
        for name,checksum in files.items(): require(file_sha(name)==checksum,'training-view input changed while merging')
        ledger_path=out/'acquisition_ledger.json'
        atomic_json(ledger_path,dict(version='r10_all_fixed_acquisition_ledger_v1',entries=ledger,
            declared_attempts=len(ledger),retained_original_cache_paths=True,automatic_resampling=False,
            native_success_or_failure_filtering=False,excluded_caches_are_zero_weight=True))
        ready=not problems
        report=dict(version='r10_causal_training_view_report_v1',complete=True,training_view_ready=ready,
            formal_comparison_ready=False,development_only=development_only,coverage_errors=problems,
            parent_coverage=coverage,full_horizon_coverage=full,acquisition_lineage=tline+vline,
            original_parent_manifests=dict(train=tp,validation=vp),acquisition_ledger=dict(path=str(ledger_path),sha256=file_sha(ledger_path)),
            original_acquisitions=len(ledger),training_view_augmentations=len(entries),
            primary_hold_motion_context_view=bool(behavior_versions and all(behavior_versions)),
            dynamics_target_label_schema=dynamics_schema,
            teacher_acquisitions_can_replace_missing_hold_parents=False if any(behavior_versions) else None,
            excluded_augmentations=[row for row in ledger if not row['included_in_training_view']],
            input_files_sha256=files,TEST_read=False,VAL_teacher_actions_used=False,
            normalization_or_model_fitting_performed=False,automatic_resampling=False,
            excluded_short_terminal_is_not_infrastructure_failure=True)
        if ready:
            manifest=dict(version='r10_causal_body_tracks_v1',complete=True,entries=entries,TEST_used=False,
                training_view_only=True,formal_comparison_ready=False,development_only=development_only,
                primary_hold_motion_context_view=bool(behavior_versions and all(behavior_versions)),
                dynamics_target_label_schema=dynamics_schema,
                teacher_acquisitions_can_replace_missing_hold_parents=False if any(behavior_versions) else None,
                cache_episode_count=len(entries),source_episode_count=expected_train+expected_validation,
                parent_source_coverage_passed=True,full_1p1s_each_task_passed=True,
                acquisition_lineage=tline+vline,acquisition_ledger=report['acquisition_ledger'],
                all_declared_acquisitions_retained_in_ledger=True,automatic_resampling=False,
                source_grouping='split/task/original parent; distinct augmentation body histories never merged')
            manifest_path=out/'manifest.json';atomic_json(manifest_path,manifest)
            report['training_view_manifest']=dict(path=str(manifest_path),sha256=file_sha(manifest_path))
        atomic_json(out/'report.json',report)
        return report
    except Exception as error:
        atomic_json(out/'infrastructure_error.json',dict(complete=False,training_view_ready=False,
            formal_comparison_ready=False,error=repr(error),no_acquisition_scored_as_policy_failure=True,
            no_training_view_manifest_written=True))
        raise
