"""Strict historical/composite acquisition audit, independent of live loaders.

Original receipts and labels are never rewritten. A recovered execution has
its own actual plan and archived sources; only the original full fixed roster
defines coverage. This module is deliberately not imported by live collection,
training_view, or artifact_fitting.
"""
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path

import numpy as np

SCHEMA='r10_historical_composite_acquisition_lineage_v1'
COMPOSITE_VERSION=SCHEMA
AUDIT_SCHEMA='r10_historical_composite_acquisition_lineage_audit_v1'
SOURCE_SCHEMA='r10_frozen_live_acquisition_execution_sources_v1'
TRAIN_TEACHER='TRAIN_teacher_committed_waypoint'
TRAIN_HOLD='TRAIN_hold_motion_identification'
VAL_HOLD='VAL_hold_motion_identification'
TRAIN_PARENTS=1400
VAL_PARENTS=275
MINIMUM_COMPLETED_PHASE_PARENTS=8
# No case field is discarded. These plan/protocol fields are review/version
# metadata only. Unknown differences, including seeds, labels and budgets,
# are never silently ignored.
PLAN_METADATA_ALLOWLIST=frozenset(('version','created_utc','plan_sha256','inputs','protocol_path'))
PROTOCOL_METADATA_ALLOWLIST=frozenset(('version','created_utc','previous_preparation_config','code_review',
    'infrastructure_recovery','full_source_training_started','full_source_training_parents_completed',
    'full_source_collection_started'))
PROTOCOL_TRANSITION_ALLOWLIST=frozenset(('/protocol/development_reset_profile','/protocol/RML_factory_hook',
    '/protocol/recovery_input'))
TASKS=tuple(task+suffix for task in ('beat_the_rotating_buzz','insert_onto_rotating_peg',
    'moving_basketball_in_hoop','place_cups_on_rotating_frame','push_moving_button',
    'put_rubbish_in_moving_bin','remove_cups_from_rotating_frame') for suffix in ('','_high_speed'))


class AcquisitionLineageError(ValueError):pass


def require(value,message):
    if not value:raise AcquisitionLineageError(message)
def canonical_sha(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
def file_sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def local(root,name):
    path=(Path(root)/name).resolve();require(path.is_relative_to(Path(root).resolve()),'lineage artifact escaped project root')
    return path
def pointer(root,path):
    path=local(root,path);return dict(path=str(path.relative_to(Path(root).resolve())),sha256=file_sha(path))
def checked_pointer(root,value,files):
    require(isinstance(value,dict) and isinstance(value.get('path'),str)
        and isinstance(value.get('sha256'),str) and len(value['sha256'])==64,'explicit immutable path/SHA256 pointer required')
    path=local(root,value['path']);require(path.is_file(),'lineage file missing: '+str(path))
    if str(path) in files:require(files[str(path)]==value['sha256'],'conflicting lineage file SHA')
    else:require(file_sha(path)==value['sha256'],'lineage bytes changed: '+str(path));files[str(path)]=value['sha256']
    return path
def instant(value):
    result=datetime.fromisoformat(value);require(result.tzinfo is not None,'timezone-aware execution timestamp required')
    return result
def command_argument(command,flag):
    require(isinstance(command,list) and command.count(flag)==1,'exact acquisition command argument required: '+flag)
    index=command.index(flag);require(index+1<len(command),'truncated acquisition command')
    return command[index+1]


def expected_reset(case,protocol):
    identity=dict(task=case['task'],raw_episode_id=case['parent_raw_episode_id'],split=case['split'],
        base_seed=int(protocol['seed']),replica=int(case['replica']),version='r10_native_generated_development_v1')
    digest=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    return dict(seed=int(digest[:8],16),seed_identity=identity,variation=case['source']['variation'])


def validate_fixed_roster(plan,train_parents,validation_parents):
    """Coverage never accepts a smoke subset or a teacher/hold substitute."""
    train=[row for row in train_parents['episodes'] if row['task'] in TASKS]
    validation=[row for row in validation_parents['episodes'] if row['task'] in TASKS]
    train_ids={(r['task'],r['raw_episode_id']) for r in train}
    validation_ids={(r['task'],r['raw_episode_id']) for r in validation}
    require(train_parents.get('split')=='train' and len(train)==len(train_ids)==TRAIN_PARENTS
        and Counter(r['task'] for r in train)==dict.fromkeys(TASKS,100),'exact original1400 TRAIN parents required')
    require(validation_parents.get('split')=='validation' and len(validation)==len(validation_ids)==VAL_PARENTS,
        'exact original275 VAL parents required')
    original=plan.get('original_demonstration_budget',{})
    require(original.get('train_sources')==TRAIN_PARENTS and original.get('validation_sources')==VAL_PARENTS,
        '1400 TRAIN /275 VAL coverage budget cannot be relaxed')
    split=plan.get('split');require(split in ('train','validation'),'only TRAIN/VAL acquisition lineage is allowed')
    parent_rows=train if split=='train' else validation
    lookup={(row['task'],row['raw_episode_id']):row for row in parent_rows}
    profiles={TRAIN_TEACHER:(200,True,False,True),TRAIN_HOLD:(60,False,True,False)} if split=='train' else {
        VAL_HOLD:(60,False,True,False)}
    expected={(task,raw,behavior,0) for task,raw in lookup for behavior in profiles}
    seen=set();ids=set()
    for case in plan['cases']:
        key=(case['task'],case['parent_raw_episode_id'],case['behavior'],case['behavior_replica'])
        require(key in expected and key not in seen and case['case_id'] not in ids,'duplicate, unknown or missing teacher/hold roster identity')
        seen.add(key);ids.add(case['case_id']);ticks,teacher,context,interaction=profiles[case['behavior']]
        require(case['split']==split and case['replica']==case['behavior_replica']==0
            and case['raw_episode_id']==case['parent_raw_episode_id']+'__'+case['behavior']+'_r0'
            and case['parent_source_group']==split+':'+case['task']+':'+case['parent_raw_episode_id']
            and case['maximum_collection_ticks']==ticks and case['teacher_used'] is teacher
            and case['context_primary_view'] is context and case['interaction_eligible'] is interaction,
            'fixed behavior/labels/budget/parent identity changed')
        original_row=lookup[(case['task'],case['parent_raw_episode_id'])]
        require(case['source']['task']==case['task'] and case['source']['raw_episode_id']==case['parent_raw_episode_id']
            and Path(case['source']['path']).resolve()==Path(original_row['path']).resolve()
            and type(case['source'].get('variation')) is int,'source/variation identity differs from original parent roster')
    require(seen==expected,'missing teacher-hold acquisition or incomplete original parent roster')
    budget=plan['augmentation_budget']
    require(budget.get('version')=='r10_fixed_behavior_budget_v1'
        and budget.get('selected_parent_sources')==len(lookup) and budget.get('planned_native_cases')==len(expected)
        and set(budget.get('behaviors',{}))==set(profiles),'full fixed acquisition budget differs')
    for name,values in profiles.items():
        profile=budget['behaviors'][name];ticks,teacher,context,interaction=values
        require(profile.get('replicas')==1 and profile.get('replica_indices')==[0] and profile.get('maximum_ticks')==ticks
            and profile.get('teacher_used') is teacher and profile.get('context_primary_view') is context
            and profile.get('interaction_eligible') is interaction,'registered behavior budget changed')
    return dict(split=split,original_TRAIN_parents=TRAIN_PARENTS,original_VAL_parents=VAL_PARENTS,
        planned_acquisitions=len(expected),planned_parent_sources=len(lookup),minimum_completed_phase_parents=8)


def load_archived_plan(root,registration,files):
    path=checked_pointer(root,registration['plan'],files);plan=read(path)
    require(plan.get('plan_sha256')==canonical_sha({k:v for k,v in plan.items() if k!='plan_sha256'}),'actual execution plan internal SHA differs')
    snapshot_path=checked_pointer(root,registration['source_snapshot'],files);snapshot=read(snapshot_path)
    require(snapshot.get('version')==SOURCE_SCHEMA and snapshot.get('all_plan_inputs_verified_before_and_after') is True
        and snapshot.get('source_files_modified') is False and snapshot.get('case_records_or_labels_modified') is False
        and (Path(root)/snapshot['plan']['path']).resolve()==path
        and snapshot['plan']['sha256']==registration['plan']['sha256']
        and snapshot['plan']['internal_sha256']==plan['plan_sha256'],'actual plan/source archive identity differs')
    sources={};expected={str(Path(name).resolve()):checksum for name,checksum in plan['inputs'].items() if Path(name).suffix in ('.py','.sh')}
    require(snapshot.get('source_count')==len(snapshot['entries'])==len(expected),'every actual executable plan source must be archived')
    for item in snapshot['entries']:
        original=str(Path(item['original_path']).resolve())
        require(original in expected and original not in sources and expected[original]==item['sha256'],
            'archived source is duplicate, undeclared or differs from actual plan')
        archived=checked_pointer(root,item,files);sources[original]=dict(path=str(archived),sha256=item['sha256'])
    require(set(sources)==set(expected),'actual executable source archive incomplete')
    for name,checksum in plan['inputs'].items():
        if Path(name).suffix not in ('.py','.sh'):checked_pointer(root,dict(path=name,sha256=checksum),files)
    return dict(plan=plan,path=path,registration=registration,snapshot_path=snapshot_path,
        sources=sources,collection=local(root,registration['collection']))


def semantic_plan(plan):
    value={key:item for key,item in plan.items() if key not in PLAN_METADATA_ALLOWLIST}
    value['protocol']={key:item for key,item in plan['protocol'].items() if key not in PROTOCOL_METADATA_ALLOWLIST}
    return value


def protocol_value(plan,json_pointer):
    require(json_pointer in PROTOCOL_TRANSITION_ALLOWLIST,'protocol transition is outside the exact reset/RML/lineage allowlist')
    key=json_pointer.split('/')[-1]
    return dict(present=True,value=plan['protocol'][key]) if key in plan['protocol'] else dict(present=False)


def declared_pointer_dependencies(root,value,files,seen=None,*,follow_json=True):
    """Hash-bind explicitly declared new hook/evidence artifacts, never rewrite."""
    seen=set() if seen is None else seen;result=set()
    if isinstance(value,dict):
        if {'path','sha256'}<=set(value):
            path=checked_pointer(root,value,files);result.add(str(path))
            if follow_json and path.suffix=='.json' and path not in seen:
                # Inspect the declared permission itself, but treat its
                # historical diagnostic/evidence pointers as immutable files,
                # not as claims that old source hashes equal today's source.
                seen.add(path);result|=declared_pointer_dependencies(root,read(path),files,seen,follow_json=False)
        for field,item in value.items():
            if field=='inputs_sha256' and isinstance(item,dict):
                for name,checksum in item.items():
                    path=checked_pointer(root,dict(path=name,sha256=checksum),files);result.add(str(path))
            else:result|=declared_pointer_dependencies(root,item,files,seen,follow_json=follow_json)
    elif isinstance(value,list):
        for item in value:result|=declared_pointer_dependencies(root,item,files,seen,follow_json=follow_json)
    return result


def verify_recovery_transition(root,previous,current,files):
    before,after=previous['plan'],current['plan'];registration=current['registration']
    require(registration.get('predecessor_plan_sha256')==before['plan_sha256'],'recovery predecessor must be an already registered actual plan')
    change=read(checked_pointer(root,registration['infrastructure_change_declaration'],files))
    require(change.get('schema')=='r10_declared_infrastructure_only_acquisition_transition_v1'
        and change.get('from_plan_sha256')==before['plan_sha256'] and change.get('to_plan_sha256')==after['plan_sha256']
        and change.get('preserves_case_seed_variation_geometry_labels_budget') is True
        and isinstance(change.get('reason'),str) and bool(change['reason'])
        and isinstance(change.get('allowed_recovery_case_ids'),list)
        and len(change['allowed_recovery_case_ids'])==len(set(change['allowed_recovery_case_ids'])),
        'explicit immutable infrastructure-only change and recovery case roster required')
    protocol_changes=change.get('protocol_changes',[])
    require(isinstance(protocol_changes,list),'explicit protocol_changes list required')
    altered=semantic_plan(after);altered['protocol']=dict(altered['protocol'])
    declared_protocol=set();new_dependencies=set()
    for item in protocol_changes:
        path=item.get('json_pointer');require(path in PROTOCOL_TRANSITION_ALLOWLIST and path not in declared_protocol,
            'duplicate or forbidden protocol transition; seed/dt/geometry/labels/budgets/gripper/action/cameras are immutable')
        declared_protocol.add(path)
        require(item.get('old')==protocol_value(before,path) and item.get('new')==protocol_value(after,path)
            and item['old']!=item['new'] and isinstance(item.get('reason'),str) and bool(item['reason']),
            'protocol transition must bind exact old/new presence/value and a reason')
        checked_pointer(root,item['evidence'],files)
        new_dependencies|=declared_pointer_dependencies(root,item['evidence'],files,follow_json=False)
        if item['new']['present']:new_dependencies|=declared_pointer_dependencies(root,item['new']['value'],files)
        key=path.split('/')[-1]
        if item['old']['present']:altered['protocol'][key]=item['old']['value']
        else:altered['protocol'].pop(key,None)
    require(semantic_plan(before)==altered,
        'undeclared semantic case/seed/variation/geometry/label/budget/protocol change; only explicitly declared reset/RML transitions permitted')
    if '/protocol/recovery_input' in declared_protocol:
        value=after['protocol']['recovery_input'];payload=read(checked_pointer(root,value,files))
        require(payload.get('original_plan')==previous['registration']['plan']
            and isinstance(payload.get('failed_parent_receipts'),list) and payload['failed_parent_receipts']
            and payload.get('development_reset_profile')==after['protocol'].get('development_reset_profile')
            and payload.get('RML_factory_hook')==after['protocol'].get('RML_factory_hook'),
            'recovery_input must bind exact original plan/failed receipts/two hook values')
        allowed={'schema','original_plan','failed_parent_receipts','development_reset_profile','RML_factory_hook',
            'created_utc','case_ids','reason','scope','source_snapshots','infrastructure_only'}
        require(set(payload)<=allowed,'recovery_input is lineage metadata and cannot introduce physical rule fields')
        failed_cases=set();failed_paths=set()
        for ptr in payload['failed_parent_receipts']:
            failed=read(checked_pointer(root,ptr,files))
            require(failed.get('status')=='infrastructure_error' and failed.get('case_id') in change['allowed_recovery_case_ids']
                and Path(command_argument(failed['command'],'--plan')).resolve()==previous['path']
                and str(local(root,ptr['path'])) not in failed_paths,
                'recovery_input failure pointer is not a same-case predecessor infra attempt')
            failed_paths.add(str(local(root,ptr['path'])));failed_cases.add(failed['case_id'])
        require(failed_cases==set(change['allowed_recovery_case_ids']),
            'recovery_input must bind every allowed recovery case to a preserved predecessor failure')
    before_sources={name:item['sha256'] for name,item in previous['sources'].items()}
    after_sources={name:item['sha256'] for name,item in current['sources'].items()}
    changed={name:(before_sources.get(name),after_sources.get(name)) for name in set(before_sources)|set(after_sources)
        if before_sources.get(name)!=after_sources.get(name)}
    declared={row['original_path']:(row.get('before_sha256'),row.get('after_sha256')) for row in change.get('source_changes',[])}
    require(len(declared)==len(change.get('source_changes',[])) and declared==changed,
        'every source change must be explicitly declared with original and actual SHA; no hidden source override')
    for name in new_dependencies:
        if Path(name).suffix in ('.py','.sh'):
            require(name in after_sources and after_sources[name]==files[name],
                'new executable permission dependency must be an actual plan source and source snapshot member')
    # Only the protocol's versioned JSON file may change among noncode inputs.
    before_inputs=before['inputs'];after_inputs=after['inputs']
    metadata={str(Path(before['protocol_path']).resolve()),str(Path(after['protocol_path']).resolve())}|new_dependencies
    for name in set(before_inputs)|set(after_inputs):
        if Path(name).suffix not in ('.py','.sh') and before_inputs.get(name)!=after_inputs.get(name):
            require(str(Path(name).resolve()) in metadata
                and (name not in before_inputs or name not in after_inputs or str(Path(name).resolve()) in
                    {str(Path(before['protocol_path']).resolve()),str(Path(after['protocol_path']).resolve())}),
                'non-source geometry/label/data input changed across recovery')
    return change


def verify_attempt_parent(root,pointer_value,registered,case_id,files,*,status):
    path=checked_pointer(root,pointer_value,files);record=read(path)
    require(record.get('case_id')==case_id and record.get('status')==status
        and record.get('policy_failure_label_assigned') is False
        and type(record.get('process_id')) is int and record['process_id']>0,
        'exact raw infra/completed parent receipt identity required')
    if status=='completed':require(record.get('returncode')==0,'completed parent must preserve a successful native return code')
    command=record.get('command');require(Path(command_argument(command,'--plan')).resolve()==registered['path']
        and command_argument(command,'--case-id')==case_id,'parent command does not bind this actual plan/case')
    require(len(command)>=4 and command[0]=='bash' and '--collect-case' in command
        and all(str(Path(name).resolve()) in registered['sources'] for name in command[1:3])
        and Path(command_argument(command,'--output')).resolve()==path.parent/'output',
        'parent command must bind the archived launcher/collector and preserved attempt output')
    require(instant(record['finished_utc'])>=instant(record['started_utc']),'parent receipt clock reversed')
    return path,record


def verify_completed(root,row,registered,expected,files):
    path=checked_pointer(root,row['canonical_receipt'],files);record=read(path);plan=registered['plan'];entry=record.get('entry',{})
    require(record.get('status')=='completed' and record.get('case_id')==expected['case_id']
        and record.get('plan_sha256')==plan['plan_sha256'] and record.get('split')==expected['split']
        and record.get('case_isolation')=='fresh_simulator_process_per_episode'
        and record.get('teacher_actions_used') is expected['teacher_used']
        and record.get('policy_history_contains_native_GT') is False,'actual completed acquisition receipt identity differs')
    for name in ('task','raw_episode_id','parent_raw_episode_id','parent_source_group','behavior','behavior_replica',
        'maximum_collection_ticks','teacher_used','context_primary_view','interaction_eligible'):
        require(entry.get(name)==expected[name],'actual completed entry differs from original fixed case: '+name)
    require(entry.get('dynamics_target_label_schema')==plan['protocol'].get('dynamics_target_label_schema'),
        'actual target label schema differs from declared fixed plan')
    require(entry.get('augmentation_replica')==expected['replica']
        and entry.get('source_demonstration')==expected['offline_demonstration']
        and entry.get('original_HDF5_motion_replayed') is False
        and entry.get('observation_source')=='causal_RGBD_and_proprioception'
        and entry.get('target_source')=='TRAIN_VAL_native_supervision'
        and entry.get('source_kind')==(('TRAIN_native_generated_augmentation' if expected['teacher_used'] else
            'TRAIN_native_generated_hold') if expected['split']=='train' else 'VAL_native_generated_non_teacher'),
        'original source, observation, supervision or augmentation labels changed')
    reset=record['native_reset_receipt'];derived=expected_reset(expected,plan['protocol'])
    require(all(reset.get(name)==value for name,value in derived.items())
        and reset.get('task')==expected['task'] and reset.get('raw_episode_id')==expected['parent_raw_episode_id']
        and reset.get('source_split')==expected['split'] and reset.get('replica')==expected['replica']
        and Path(reset['source_path']).resolve()==Path(expected['source']['path']).resolve()
        and reset.get('official_TEST') is False and reset.get('original_source_replayed') is False,
        'actual seed/variation/source reset identity changed')
    process=record['parent_process_receipt'];raw=checked_pointer(root,dict(path=process['child_case_path'],sha256=process['child_case_sha256']),files)
    require(read(raw)=={k:v for k,v in record.items() if k!='parent_process_receipt'},'canonical receipt rewrites its historical raw child')
    parent_path,parent=verify_attempt_parent(root,row['parent_receipt'],registered,expected['case_id'],files,status='completed')
    require(parent['command']==process['command'] and parent['process_id']==process['process_id']
        and raw.is_relative_to(parent_path.parent/'output'), 'canonical child and raw parent/process receipts differ')
    require(instant(parent['started_utc'])<=instant(record['started_utc'])<=instant(record['finished_utc'])<=instant(parent['finished_utc']),
        'completed child/parent chronological lineage differs')
    launcher=str((Path(root)/process['launcher']).resolve())
    require(launcher in registered['sources'] and registered['sources'][launcher]['sha256']==process['launcher_sha256'],
        'historical launcher SHA is not in actual archived execution sources')
    outputs=record.get('files_sha256',{});require(outputs,'actual output hashes missing')
    for name,checksum in outputs.items():checked_pointer(root,dict(path=name,sha256=checksum),files)
    require(outputs.get(entry['cache'])==entry['cache_sha256'],'actual cache is not hash-bound by original receipt')
    geometry=[name for name in outputs if Path(name).name=='TRAIN_geometry.json']
    require(len(geometry)==1 and expected.get('template_asset') in plan['inputs'],'frozen geometry source/output identity missing')
    original_geometry=read(local(root,expected['template_asset']));actual_geometry=read(local(root,geometry[0]))
    require(original_geometry==actual_geometry,'actual geometry differs from original frozen same-task TRAIN CAD')
    clock=record['alignment']['clock'];ticks=clock['ticks']
    require(type(ticks) is int and 0<=ticks<=expected['maximum_collection_ticks'] and clock.get('dt')==.05
        and clock.get('budget_seconds')==10. and abs(clock.get('simulation_seconds',-1)-ticks*.05)<=1e-6,
        'actual acquisition exceeded or changed fixed behavior clock/budget')
    require(all(abs(record['alignment'].get(field,-1)-ticks*.05)<=2e-5
        for field in ('actual_physics_seconds','actual_task_seconds')),'actual native clocks differ from recorded original frame axis')
    with np.load(local(root,entry['cache']),allow_pickle=False) as stored:
        times=stored['times'];require(len(times)==ticks+1==entry['frames'] and abs(float(times[0]))<=1e-6
            and np.allclose(np.diff(times),.05,rtol=0,atol=1e-6),'actual original NPZ frame axis changed')
    return dict(case_id=expected['case_id'],case_identity_sha256=canonical_sha(expected),
        actual_plan_sha256=plan['plan_sha256'],actual_plan=registered['registration']['plan'],
        actual_source_snapshot=registered['registration']['source_snapshot'],
        canonical_receipt=row['canonical_receipt'],parent_receipt=row['parent_receipt'],
        child_receipt=dict(path=str(raw.relative_to(Path(root).resolve())),sha256=process['child_case_sha256']),
        parent_source_group=expected['parent_source_group'],behavior=expected['behavior'],reset_identity=derived,
        original_output_hashes=outputs,entry=entry,started_utc=record['started_utc'],finished_utc=record['finished_utc'])


def verify_recovery_execution(root,completion,registered,predecessor,preceding,files):
    """Audit raw hook witnesses using archived execution identities, not live code."""
    keys=('development_reset_profile','RML_factory_hook','recovery_input');protocol=registered['plan']['protocol']
    if not any(key in protocol for key in keys):return
    require(all(key in protocol for key in keys),'recovery execution must declare both hooks and the lineage input')
    record=read(checked_pointer(root,completion['canonical_receipt'],files));own=record.get('recovery_execution',{})
    version='r10_explicit_same_case_infrastructure_recovery_runner_v1'
    require(own.get('version')==version,'hook-enabled plan lacks its actual recovery execution witness')
    hook_path=checked_pointer(root,own['hook_receipt'],files);base_path=checked_pointer(root,own['base_collector_receipt'],files)
    registry=read(checked_pointer(root,own['execution_registration'],files))
    require(set(registry)=={'schema','original','recovery'}
        and registry['schema']=='r10_explicit_recovery_execution_registration_v1'
        and registry['original']==predecessor['registration'] and registry['recovery']==registered['registration'],
        'raw recovery execution registration differs from actual historical plan/source registrations')
    hooks=read(hook_path);base=read(base_path);cid=completion['case_id']
    original_failures=[row['parent_receipt'] for row in preceding if row['actual_plan_sha256']==predecessor['plan']['plan_sha256']]
    actual_failures=hooks.get('failed_parent_receipts',[])
    require(isinstance(actual_failures,list) and len(actual_failures)==len(original_failures)
        and {canonical_sha(row) for row in actual_failures}=={canonical_sha(row) for row in original_failures},
        'raw hook receipt must preserve every preceding same-case predecessor infrastructure failure')
    require(hooks.get('version')==version and hooks.get('complete') is True and hooks.get('error') is None
        and hooks.get('case_id')==cid and hooks.get('plan')==registered['registration']['plan']
        and hooks.get('plan_sha256')==registered['plan']['plan_sha256']
        and hooks.get('original_plan')==predecessor['registration']['plan']
        and hooks.get('execution_registration')==own['execution_registration']
        and hooks.get('permissions')=={key:protocol[key] for key in keys}
        and all(hooks.get(key) is False for key in ('policy_failure_label_assigned','historical_v11_outputs_modified',
            'partial_historical_rows_reconstructed','new_attempt_is_original_v11')),
        'raw recovery hook identity, permissions or historical preservation claims differ')
    runner=hooks.get('actual_runner',{});original=str(local(root,runner.get('path','')))
    require(Path(original).name=='collect_multitask_recovery_r10.py' and original in registered['sources']
        and runner.get('sha256')==registered['sources'][original]['sha256'],
        'actual recovery runner must match its archived execution source SHA')
    reset=hooks.get('reset_recovery_receipts',[]);paths=hooks.get('RML_path_recovery_receipts',[])
    require(isinstance(reset,list) and len(reset)==1 and isinstance(reset[0],dict) and reset[0].get('complete') is True
        and reset[0].get('permission')==protocol['development_reset_profile']
        and reset[0]==record['native_reset_receipt'].get('reset_recovery_receipt')
        and all(reset[0].get(key)==value for key,value in completion['reset_identity'].items())
        and reset[0].get('task')==record['entry']['task'] and reset[0].get('split')==record['split']
        and reset[0].get('source_resampled') is False and reset[0].get('official_TEST_allowed') is False,
        'raw recovery requires one complete same-seed/variation physical-reset witness without resampling')
    require(isinstance(paths,list) and all(isinstance(row,dict) and row.get('close_complete') is True
        and row.get('physics_step_called_by_adapter') is False and row.get('real_native_step_motion_replaced') is False
        and row.get('historical_fault_reproduced_in_native_diagnostic') is False for row in paths),
        'raw RML witnesses must preserve cleanup, native motion and limited diagnostic scope')
    parent=read(checked_pointer(root,completion['parent_receipt'],files))
    require(instant(parent['started_utc'])<=instant(hooks['started_utc'])<=instant(completion['started_utc'])
        <=instant(completion['finished_utc'])<=instant(hooks['finished_utc'])<=instant(parent['finished_utc']),
        'recovery hook/child/parent execution clocks differ')
    stripped={key:value for key,value in record.items() if key not in ('recovery_execution','parent_process_receipt')}
    stripped['files_sha256']={key:value for key,value in stripped['files_sha256'].items()
        if key not in (own['hook_receipt']['path'],own['base_collector_receipt']['path'])}
    require(stripped==base,'recovery wrapper changed its original base collector receipt or labels')
    for key in ('hook_receipt','base_collector_receipt'):
        require(record['files_sha256'].get(own[key]['path'])==own[key]['sha256'],
            'actual recovery hook/base witness must be hash-bound in original child outputs')


def audit_acquisition_lineage(root,manifest,*,require_complete=False):
    root=Path(root).resolve();files={}
    if not isinstance(manifest,dict):
        path=local(root,manifest);files[str(path)]=file_sha(path);manifest=read(path)
    require(manifest.get('schema')==SCHEMA and manifest.get('lineage_sha256')==canonical_sha(
        {k:v for k,v in manifest.items() if k!='lineage_sha256'}),'immutable composite lineage identity differs')
    require(manifest.get('case_records_or_labels_rewritten') is False,
        'composite must explicitly preserve original case receipts and labels')
    train=read(checked_pointer(root,manifest['original_TRAIN_parents'],files))
    validation=read(checked_pointer(root,manifest['original_VAL_parents'],files))
    registrations=manifest.get('actual_plans',[]);require(registrations,'actual historical plan/source registrations required')
    plans={};changes={}
    for index,registration in enumerate(registrations):
        current=load_archived_plan(root,registration,files);key=current['plan']['plan_sha256']
        require(key not in plans,'duplicate actual plan registration')
        if index:
            predecessor=registration.get('predecessor_plan_sha256');require(predecessor in plans,'recovery predecessor must precede its actual new plan')
            changes[key]=verify_recovery_transition(root,plans[predecessor],current,files)
        plans[key]=current
    base=next(iter(plans.values()));require(manifest['base_plan_sha256']==base['plan']['plan_sha256'],'original fixed roster plan differs')
    contract=validate_fixed_roster(base['plan'],train,validation)
    roster={case['case_id']:case for case in base['plan']['cases']}
    completions={};infra={};all_known_completed={}
    # Filesystem inventory independently protects any existing completed case;
    # omitting an old receipt from a proposed composite cannot hide replacement.
    for key,registered in plans.items():
        for path in sorted((registered['collection']/'cases'/contract['split']).glob('*/*/case.json')):
            record=read(path);cid=record.get('case_id')
            require(cid in roster and record.get('plan_sha256')==key,'collection contains an undeclared or wrongly planned completion')
            require(cid not in all_known_completed,'duplicate completed acquisition or replacement of an existing completed case')
            all_known_completed[cid]=pointer(root,path)
        for path in sorted((registered['collection']/'infrastructure_attempts').glob('*/*/parent_receipt.json')):
            record=read(path)
            if record.get('status')!='infrastructure_error':continue
            cid=record.get('case_id');require(cid in roster,'infra receipt is outside original fixed roster')
            ptr=pointer(root,path);_,record=verify_attempt_parent(root,ptr,registered,cid,files,status='infrastructure_error')
            log=path.with_name('native.log');require(log.is_file(),'raw infrastructure native log must be preserved')
            item=dict(actual_plan_sha256=key,parent_receipt=ptr,native_log=pointer(root,log),
                started_utc=record['started_utc'],finished_utc=record['finished_utc'])
            checked_pointer(root,item['native_log'],files);infra.setdefault(cid,[]).append(item)
    for key,change in changes.items():
        previous=plans[key]['registration']['predecessor_plan_sha256']
        for cid in change['allowed_recovery_case_ids']:
            require(cid in roster and any(row['actual_plan_sha256']==previous for row in infra.get(cid,[])),
                'new plan override has no preceding same-case infrastructure failure')
    for row in manifest.get('completed_acquisitions',[]):
        cid=row.get('case_id');key=row.get('actual_plan_sha256')
        require(cid in roster and cid not in completions and key in plans,'duplicate/undeclared composite completion identity')
        require(row.get('canonical_receipt')==all_known_completed.get(cid),
            'composite omitted or replaced the existing historical canonical completion')
        current=verify_completed(root,row,plans[key],roster[cid],files)
        previous=sorted(infra.get(cid,[]),key=lambda item:(instant(item['finished_utc']),item['parent_receipt']['path']))
        require(all(instant(item['finished_utc'])<=instant(current['started_utc']) for item in previous),
            'infrastructure attempt after completed acquisition is not a permitted recovery')
        declared_chain=row.get('preceding_infrastructure_attempts',[])
        require(declared_chain==previous,'preceding infrastructure receipt/log chain was omitted, changed or fabricated')
        if key!=manifest['base_plan_sha256']:
            change=changes[key];predecessor=plans[key]['registration']['predecessor_plan_sha256']
            require(cid in change['allowed_recovery_case_ids'] and any(p['actual_plan_sha256']==predecessor for p in previous),
                'new plan completion cannot override an unfailed or already completed original case')
            require(instant(change['created_utc'])<=instant(current['started_utc']),
                'recovery source/roster declaration must precede execution, not be selected after outcomes')
            verify_recovery_execution(root,current,plans[key],plans[predecessor],previous,files)
        current['preceding_infrastructure_attempts']=previous;completions[cid]=current
    require(set(completions)==set(all_known_completed),'every existing historical completion must be retained in the composite')
    missing=sorted(set(roster)-set(completions));complete=not missing
    require(not require_complete or complete,'full fixed roster has missing teacher/hold acquisitions')
    declared_contract=manifest.get('coverage_contract',{})
    require(declared_contract.get('original_TRAIN_parents')==TRAIN_PARENTS
        and declared_contract.get('original_VAL_parents')==VAL_PARENTS
        and declared_contract.get('minimum_completed_phase_parents')==MINIMUM_COMPLETED_PHASE_PARENTS,
        '1400 TRAIN /275 VAL /eight completed-parent phase requirements cannot be relaxed')
    ordered=[completions[cid] for cid in roster if cid in completions]
    actual_map={row['case_id']:row['actual_plan_sha256'] for row in ordered}
    receipts=[dict(case_id=row['case_id'],**row['canonical_receipt'],actual_plan_sha256=row['actual_plan_sha256'],
        actual_source_snapshot=row['actual_source_snapshot']) for row in ordered]
    original_budget=base['plan']['augmentation_budget']
    projected_budget=dict(original_budget,completed_native_cases=len(ordered),native_started=bool(ordered))
    projection=dict(version=COMPOSITE_VERSION,complete=complete,TEST_used=False,expected=len(roster),completed=len(ordered),
        entries=[row['entry'] for row in ordered],case_receipts=receipts,missing_case_ids=missing,
        plan_sha256=None,base_plan_sha256=manifest['base_plan_sha256'],actual_plan_sha256_values=list(plans),
        per_case_actual_plan_sha256=actual_map,augmentation_budget=projected_budget,
        original_augmentation_budget=original_budget,original_demonstration_budget=base['plan']['original_demonstration_budget'],
        dynamics_target_label_schema=base['plan']['protocol'].get('dynamics_target_label_schema'),
        single_plan_SHA_claimed=False,projection_rewrites_original_receipts=False)
    return dict(schema=AUDIT_SCHEMA,audit_passed=True,lineage_complete=complete,complete=complete,
        status='complete_lineage' if complete else 'partial_lineage_pending_acquisitions',
        base_plan_sha256=manifest['base_plan_sha256'],coverage_contract=contract,
        expected_acquisitions=len(roster),verified_completed_acquisitions=len(completions),missing_case_ids=missing,
        actual_plan_sha256_values=list(plans),completed_acquisitions=ordered,
        per_case_actual_plan_sha256=actual_map,acquisition_projection=projection,
        original_augmentation_budget=original_budget,
        declared_protocol_transitions={key:value.get('protocol_changes',[]) for key,value in changes.items()},
        protocol_fully_unchanged=not any(value.get('protocol_changes') for value in changes.values()),
        original_case_receipts_rewritten=False,original_labels_rewritten=False,old_SHA_forged=False,
        source_semantic_equivalence_claims_are_declarations_not_automatic_proof=True,
        downstream_training_view_and_all_array_validation_required=True,
        original275_VAL_coverage_retained_as_requirement=True,VAL_execution_coverage_proven=False,
        minimum_eight_parent_phase_support_retained=True,phase_support_or_fitted_library_acceptance_proven=False,
        inputs_sha256=files,verified_files_sha256=files)
