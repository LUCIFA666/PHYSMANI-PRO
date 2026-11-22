"""Independent first-execution VAL shared-profile acquisition audit.

This module does not relax or modify the frozen historical TRAIN auditor. VAL
uses its original 275 hold cases and an explicitly shared initialization
profile. No previous infrastructure failure, recovery input or teacher is
invented. Historical executable hashes refer to archived bytes only.
"""
from pathlib import Path

import numpy as np

from .acquisition_lineage import (AcquisitionLineageError,canonical_sha,checked_pointer,command_argument,
    file_sha,instant,load_archived_plan,local,pointer,read,require,validate_fixed_roster,verify_attempt_parent,
    verify_completed,expected_reset,VAL_HOLD)

TRACK_VERSION='r10_causal_body_tracks_v1'
REGISTRATION_SCHEMA='r10_first_execution_development_initialization_registration_v1'
EXECUTION_VERSION='r10_first_execution_VAL_development_initialization_runner_v1'
AUDIT_SCHEMA='r10_first_execution_VAL_shared_profile_lineage_audit_v1'
REGISTRATION_FIELD='shared_profile_execution_registration'
HOOK_KEYS=('development_reset_profile','RML_factory_hook')
REGISTRATION_KEYS=frozenset(('schema','original_plan','original_collection','source_closure_plan','execution',
    'original_TRAIN_parents','original_VAL_parents'))
PLAN_CREATION_FIELDS=frozenset(('created_utc','plan_sha256','inputs','protocol_path'))


def checked_plan(root,value,files):
    path=checked_pointer(root,value,files);plan=read(path)
    require(plan.get('plan_sha256')==canonical_sha({key:item for key,item in plan.items() if key!='plan_sha256'}),
        'first-execution plan internal SHA differs')
    return path,plan


def bind_source(registered,name,checksum):
    """Bind a historical executable without comparing it to today's source."""
    name=str(Path(name).resolve())
    require(registered['plan']['inputs'].get(name)==checksum and name in registered['sources']
        and registered['sources'][name]['sha256']==checksum,
        'shared-profile executable is not bound to its actual archived source: '+name)


def verify_permissions(root,registered,files):
    protocol=registered['plan']['protocol'];payloads={}
    for key in HOOK_KEYS:
        value=protocol[key];path=checked_pointer(root,value,files)
        require(registered['plan']['inputs'].get(str(path))==value['sha256'],'shared profile permission is not an actual plan input')
        payload=read(path);payloads[key]=payload
        for name,checksum in payload.get('inputs_sha256',{}).items():
            if Path(name).suffix in ('.py','.sh'):bind_source(registered,name,checksum)
            else:
                artifact=checked_pointer(root,dict(path=name,sha256=checksum),files)
                require(registered['plan']['inputs'].get(str(artifact))==checksum,'shared profile data input is unbound')
        if 'diagnostic_report' in payload:checked_pointer(root,payload['diagnostic_report'],files)
    physical=payloads['development_reset_profile']
    require('validation' in physical.get('allowed_splits',[]) and physical.get('official_TEST_allowed') is False
        and physical.get('resample_source_seed_variation_or_scene') is False and physical.get('strict_physical_guards') is True,
        'physical initialization permission must preserve VAL, guards, same source and official TEST isolation')
    native=physical.get('native_inputs_sha256',{})
    require(isinstance(native,dict) and native,'native initialization source bindings are required')
    for relative,checksum in native.items():
        bind_source(registered,Path(protocol['baseline_repo'])/relative,checksum)
    if 'base_simulation_sha256' in physical:
        bind_source(registered,Path(root)/'our_mani/multitask/simulation.py',physical['base_simulation_sha256'])
    rml=payloads['RML_factory_hook']
    require(rml.get('physics_step_called_by_factory') is False and rml.get('real_native_step_motion_replaced') is False
        and rml.get('historical_fault_reproduced_in_native_diagnostic') is False,
        'shared RML permission must preserve native execution and its limited diagnostic claim')
    return payloads


def verify_validation_execution_registration(root,registration,files=None):
    """Audit the first-execution plan/roster/archive before or after acquisition."""
    root=Path(root).resolve();files={} if files is None else files
    registration_pointer=registration if isinstance(registration,dict) else pointer(root,registration)
    path=checked_pointer(root,registration_pointer,files);registry=read(path)
    require(set(registry)==REGISTRATION_KEYS and registry.get('schema')==REGISTRATION_SCHEMA,
        'exact first-execution VAL registry and original TRAIN/VAL parent pointers required')
    original_path,original=checked_plan(root,registry['original_plan'],files)
    closure_path,closure=checked_plan(root,registry['source_closure_plan'],files)
    actual=load_archived_plan(root,registry['execution'],files);plan=actual['plan']
    train=read(checked_pointer(root,registry['original_TRAIN_parents'],files))
    validation=read(checked_pointer(root,registry['original_VAL_parents'],files))
    demonstration_path=local(root,'artifacts/framework/multitask_r10/demonstrations/manifest.json')
    require(str(demonstration_path) in plan['inputs'],'audited original demonstration metadata must be an actual VAL plan input')
    demonstrations=read(checked_pointer(root,dict(path=str(demonstration_path),sha256=plan['inputs'][str(demonstration_path)]),files))
    parent_hashes=demonstrations.get('inputs',{}).get('manifests_sha256',{})
    require(all(registry[key]['sha256']==parent_hashes.get(split)
        and plan['inputs'].get(str(local(root,registry[key]['path'])))==registry[key]['sha256']
        for split,key in (('train','original_TRAIN_parents'),('validation','original_VAL_parents'))),
        'original TRAIN/VAL parent pointers must match the audited demonstration metadata and actual plan inputs')
    for value in (original,plan):
        validate_fixed_roster(value,train,validation)
        require(value.get('split')=='validation' and value.get('validation_mode')=='hold'
            and value.get('frozen_policy') is None and value.get('context_primary_behavior')==VAL_HOLD,
            'ordinary full275 VAL hold plan required')
        require('recovery_input' not in value['protocol'],'first VAL execution cannot invent an infrastructure recovery input')
    require(not any(key in original['protocol'] for key in HOOK_KEYS),'original VAL plan must precede shared-profile execution')
    require(all(key in plan['protocol'] and plan['protocol'][key]==closure['protocol'].get(key) for key in HOOK_KEYS),
        'actual VAL permissions must equal the declared shared TRAIN source-closure profile')
    expected=dict(original);expected['protocol']=dict(original['protocol'])
    expected['protocol'].update({key:plan['protocol'][key] for key in HOOK_KEYS})
    require({key:value for key,value in expected.items() if key not in PLAN_CREATION_FIELDS}
        =={key:value for key,value in plan.items() if key not in PLAN_CREATION_FIELDS},
        'first VAL execution changes original case, seed, variation, geometry, label, budget or protocol')
    for name,checksum in original['inputs'].items():
        require(plan['inputs'].get(name)==checksum,'original VAL input was removed or rebound')
    for name,checksum in closure['inputs'].items():
        if Path(name).suffix in ('.py','.sh'):bind_source(actual,name,checksum)
    for name,value in ((original_path,registry['original_plan']),(closure_path,registry['source_closure_plan'])):
        require(plan['inputs'].get(str(name))==value['sha256'],'original VAL/source-closure evidence is not an actual plan input')
    for case in plan['cases']:
        asset_path=local(root,case.get('template_asset',''));require(str(asset_path) in plan['inputs'],'VAL requires its original frozen TRAIN geometry input')
        asset=read(asset_path)
        require(asset.get('split')=='train' and asset.get('task')==case['task']
            and str(asset.get('source_group','')).startswith('train:'+case['task']+':')
            and asset.get('world_pose_or_motion_schedule_in_asset') is False and asset.get('shared_with_baseline') is True,
            'VAL CAD must be a same-task shared TRAIN artifact without world-pose or motion-schedule information')
    original_collection=local(root,registry['original_collection'])
    require(original_collection!=actual['collection'] and not original_collection.is_relative_to(actual['collection'])
        and not actual['collection'].is_relative_to(original_collection),'first VAL execution requires a separate declared namespace')
    require(not any(original_collection.glob('cases/*/*/*/case.json'))
        and not (original_collection/'infrastructure_attempts').exists(),
        'original VAL already has native attempts; first-execution audit cannot replace recovery lineage')
    verify_permissions(root,actual,files)
    return dict(registry=registry,registration_pointer=registration_pointer,actual=actual,original=original,
        source_closure=closure,coverage_contract=validate_fixed_roster(plan,train,validation),files=files)


def verify_initialization_witness(root,record,case,context,parent,files):
    actual=context['actual'];protocol=actual['plan']['protocol'];registry=context['registry']
    own=record.get('development_initialization_execution',{})
    require(set(own)=={'version','hook_receipt','base_collector_receipt','execution_registration'}
        and own.get('version')==EXECUTION_VERSION and own.get('execution_registration')==context['registration_pointer']
        and 'recovery_execution' not in record,'ordinary VAL lacks its actual first-execution initialization witness')
    hook_path=checked_pointer(root,own['hook_receipt'],files);base_path=checked_pointer(root,own['base_collector_receipt'],files)
    raw_path=local(root,record['parent_process_receipt']['child_case_path'])
    require(hook_path==raw_path.parent/'development_initialization_hook_receipt.json'
        and base_path==raw_path.parent/'base_collector_receipt.json','VAL hook/base witnesses must remain with the actual raw child')
    hooks=read(hook_path);base=read(base_path)
    require(hooks.get('version')==EXECUTION_VERSION and hooks.get('complete') is True and hooks.get('error') is None
        and hooks.get('case_id')==case['case_id'] and hooks.get('plan')==actual['registration']['plan']
        and hooks.get('plan_sha256')==actual['plan']['plan_sha256']
        and hooks.get('execution_registration')==context['registration_pointer']
        and hooks.get('original_plan')==registry['original_plan'] and hooks.get('source_closure_plan')==registry['source_closure_plan']
        and hooks.get('permissions')=={key:protocol[key] for key in HOOK_KEYS}
        and hooks.get('first_execution') is True and hooks.get('previous_infrastructure_failure_required') is False
        and hooks.get('RML_path_recovery_receipts')==[] and 'failed_parent_receipts' not in hooks
        and all(hooks.get(key) is False for key in ('policy_failure_label_assigned','historical_outputs_modified',
            'teacher_actions_used','teacher_paths_requested','VAL_geometry_exported')),
        'first VAL hook identity, shared profile, no-teacher/no-RML/no-export scope differs')
    runner=hooks.get('actual_runner',{});name=str(local(root,runner.get('path','')))
    require(Path(name).name=='collect_multitask_validation_initialization_r10.py','actual first VAL runner identity differs')
    bind_source(actual,name,runner.get('sha256'))
    resets=hooks.get('reset_recovery_receipts',[])
    require(isinstance(resets,list) and len(resets)==1 and isinstance(resets[0],dict) and resets[0].get('complete') is True
        and resets[0]==record['native_reset_receipt'].get('reset_recovery_receipt')
        and all(resets[0].get(key)==value for key,value in expected_reset(case,protocol).items())
        and resets[0].get('permission')==protocol['development_reset_profile'] and resets[0].get('task')==case['task']
        and resets[0].get('split')=='validation' and resets[0].get('source')==case['source']
        and resets[0].get('replica')==case['replica'] and resets[0].get('source_resampled') is False
        and resets[0].get('official_TEST_allowed') is False,
        'VAL requires one measured complete same-source reset witness identical to the native receipt')
    require(instant(parent['started_utc'])<=instant(hooks['started_utc'])<=instant(record['started_utc'])
        <=instant(record['finished_utc'])<=instant(hooks['finished_utc'])<=instant(parent['finished_utc']),
        'first VAL initialization/child/parent clocks differ')
    stripped={key:value for key,value in record.items() if key not in ('development_initialization_execution','parent_process_receipt')}
    stripped['files_sha256']={key:value for key,value in stripped['files_sha256'].items()
        if key not in (own['hook_receipt']['path'],own['base_collector_receipt']['path'])}
    require(stripped==base,'first VAL wrapper changed its original base collector receipt or labels')
    for key in ('hook_receipt','base_collector_receipt'):
        require(record['files_sha256'].get(own[key]['path'])==own[key]['sha256'],'VAL initialization/base witness missing from original output hashes')


def audit_validation_acquisition_lineage(root,manifest,*,require_complete=True):
    """Validate an ordinary TRACK manifest without synthesizing a composite plan."""
    root=Path(root).resolve();files={}
    if not isinstance(manifest,dict):
        path=local(root,manifest);files[str(path)]=file_sha(path);manifest=read(path)
    require(manifest.get('version')==TRACK_VERSION and manifest.get('TEST_used') is False
        and manifest.get('validation_teacher_actions_used') is False,
        'ordinary nonteacher VAL body-track acquisition manifest required')
    context=verify_validation_execution_registration(root,manifest[REGISTRATION_FIELD],files)
    actual=context['actual'];plan=actual['plan'];roster={case['case_id']:case for case in plan['cases']}
    require(manifest.get('plan_sha256')==plan['plan_sha256'] and manifest.get('expected')==275
        and manifest.get('original_demonstration_budget')==plan['original_demonstration_budget']
        and manifest.get('dynamics_target_label_schema')==plan['protocol'].get('dynamics_target_label_schema'),
        'VAL manifest original budget, schema or actual single plan differs')
    values=manifest.get('entries',[]);rows=manifest.get('case_receipts',[])
    require(isinstance(values,list) and isinstance(rows,list) and len(values)==len(rows)==manifest.get('completed'),
        'VAL entries/receipt counts differ')
    inventory={};attempts={};infrastructure=[]
    for path in sorted((actual['collection']/'cases/validation').glob('*/*/case.json')):
        record=read(path);cid=record.get('case_id')
        require(cid in roster and cid not in inventory and record.get('plan_sha256')==plan['plan_sha256'],
            'duplicate, undeclared or wrongly planned VAL canonical completion')
        require(path==actual['collection']/'cases/validation'/roster[cid]['task']/cid/'case.json',
            'VAL canonical receipt must retain its declared task/case namespace')
        inventory[cid]=pointer(root,path)
    for path in sorted((actual['collection']/'infrastructure_attempts').glob('*/*/parent_receipt.json')):
        record=read(path);cid=record.get('case_id');require(cid in roster and cid not in attempts,
            'first VAL execution cannot hide duplicate, new-case or repeated native attempts')
        require(path==actual['collection']/'infrastructure_attempts'/cid/'0/parent_receipt.json',
            'first VAL execution permits only its original same-case attempt0')
        status=record.get('status');require(status in ('completed','infrastructure_error'),'VAL native attempt status differs')
        _,record=verify_attempt_parent(root,pointer(root,path),actual,cid,files,status=status)
        require(record.get('version')==EXECUTION_VERSION and record.get('attempt')==0
            and record.get('execution_registration')==context['registration_pointer']
            and record.get('actual_plan')==actual['registration']['plan'] and record.get('first_execution') is True
            and record.get('automatic_retries') is False,'VAL parent must bind actual first execution without hidden retries')
        log=path.with_name('native.log');checked_pointer(root,pointer(root,log),files)
        attempts[cid]=dict(path=path,record=record)
        if status=='infrastructure_error':infrastructure.append(dict(case_id=cid,parent_receipt=pointer(root,path),native_log=pointer(root,log)))
    accepted={};entries={}
    for row,entry in zip(rows,values):
        cid=row.get('case_id');require(cid in roster and cid not in accepted and row==dict(case_id=cid,**inventory.get(cid,{})),
            'VAL manifest omits, duplicates or replaces an existing canonical acquisition')
        require(cid in attempts and attempts[cid]['record']['status']=='completed','VAL canonical lacks its committed completed parent')
        case=roster[cid];receipt=dict(case_id=cid,canonical_receipt=inventory[cid],parent_receipt=pointer(root,attempts[cid]['path']))
        completion=verify_completed(root,receipt,actual,case,files);record=read(local(root,row['path']))
        require(entry==completion['entry'] and entry.get('split')=='validation'
            and type(record.get('process_id')) is int and record['process_id']>0
            and record.get('teacher_actions_used') is False and record.get('teacher_paths_requested') is False
            and record.get('teacher_action_rows')==0 and entry.get('teacher_command_receipt') is None
            and entry.get('phase_completion_events_receipt') is None,'VAL cannot introduce expert action, path or completion supervision')
        require(not any((Path(name).name.startswith('TRAIN_') and Path(name).name!='TRAIN_geometry.json')
            or Path(name).name in ('VAL_geometry.json','teacher_commands.json')
            for name in record['files_sha256']),'VAL cannot export TRAIN annotations/layout or teacher commands')
        with np.load(local(root,entry['cache']),allow_pickle=False) as stored:
            require('teacher_action_valid' in stored and stored['teacher_action_valid'].dtype==np.bool_
                and stored['teacher_action_valid'].shape==(entry['frames'],)
                and not stored['teacher_action_valid'].any(),'VAL cache contains teacher-valid rows or a missing per-frame teacher mask')
        command=attempts[cid]['record']['command']
        require(Path(command_argument(command,'--execution-registration')).resolve()
            ==local(root,context['registration_pointer']['path']),'VAL parent command differs from actual first-execution registry')
        verify_initialization_witness(root,record,case,context,attempts[cid]['record'],files)
        accepted[cid]=completion;entries[cid]=entry
    require(set(accepted)==set(inventory) and {cid for cid,value in attempts.items() if value['record']['status']=='completed'}==set(inventory),
        'every committed VAL completion must be retained; parent and canonical commits differ')
    missing=sorted(set(roster)-set(accepted));complete=not missing
    require(manifest.get('complete') is complete and sorted(manifest.get('missing_case_ids',[]))==missing,
        'VAL complete/pending declaration differs from the full fixed275 roster')
    require(not require_complete or complete,'full275 first-execution VAL acquisition remains incomplete')
    expected_budget=dict(plan['augmentation_budget'],completed_native_cases=len(accepted),native_started=bool(accepted))
    require(manifest.get('augmentation_budget')==expected_budget and manifest.get('cache_episode_count')==len(accepted)
        and manifest.get('source_episode_count')==len({(entry['task'],entry['parent_raw_episode_id']) for entry in entries.values()}),
        'VAL fixed behavior budget or distinct source count differs')
    require(not complete or manifest['source_episode_count']==275,'complete VAL must retain all original275 distinct parents')
    actual_map={cid:plan['plan_sha256'] for cid in accepted}
    return dict(schema=AUDIT_SCHEMA,audit_passed=True,complete=complete,status='complete_VAL_lineage' if complete else 'partial_VAL_lineage_pending_acquisitions',
        execution_kind='first_execution_VAL_shared_development_initialization',expected_acquisitions=275,verified_completed_acquisitions=len(accepted),
        missing_case_ids=missing,actual_plan_sha256_values=[plan['plan_sha256']],per_case_actual_plan_sha256=actual_map,
        acquisition_projection=manifest,verified_files_sha256=files,inputs_sha256=files,coverage_contract=context['coverage_contract'],
        original_augmentation_budget=plan['augmentation_budget'],original_VAL_plan=context['registry']['original_plan'],
        source_closure_plan=context['registry']['source_closure_plan'],execution_registration=context['registration_pointer'],
        infrastructure_attempts=infrastructure,prior_infrastructure_failure_required=False,first_execution=True,
        completed_acquisitions=[accepted[cid] for cid in roster if cid in accepted],original_case_receipts_rewritten=False,
        original_labels_rewritten=False,VAL_teacher_used=False,VAL_RML_paths_created=False,VAL_geometry_exported=False,
        original_VAL_parent_coverage=275,downstream_training_view_and_all_array_validation_required=True,
        acquisition_complete_is_not_model_performance_or_phase_support=True)
