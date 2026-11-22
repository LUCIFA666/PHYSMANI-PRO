"""Candidate, unactivated independent full1920 four-tier753/902/956/1011 audit.

This auditor starts no work, retries no case, and never filters nativeFalse,
unknown motion or zero counted ticks. Original admission, complete byte gates,
RML, author outcome and cleanup checks remain mandatory. The new observation
proofs are environment point observations, not TEST qualification or release.
"""
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import argparse
import json
import os
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from our_mani.multitask.evaluation import atomic_json,digest,validate_completed_case
from our_mani.multitask.evaluation_roster16_v283 import TASKS,METHODS,validate_roster
from our_mani.multitask.input_stat_cache_v79 import verify_inputs
from our_mani.multitask.frozen_lookup_batch_v900 import LookupBatch
from our_mani.multitask.formal_suite_admission_v728 import read_formal_source
from our_mani.multitask.common_rml_protocol_v530 import fingerprint,require_attestation
from our_mani.multitask.three_method_release16 import measured_motion
from our_mani.multitask.three_method_release_v531 import policy_fingerprint
from our_mani.multitask.rml_case_preflight_v742 import outcome_protocol_binding
from scripts import run_concurrent_resumed_formal_TEST_owner_v902_r10 as physical_owner
from scripts import audit_concurrent_resumed_formal_TEST_v906_r10 as previous_audit
from scripts import audit_composite_TEST_results_v957_r10 as parent_audit
from scripts import run_composite_TEST_owner_v1011_r10 as current_owner
from scripts import audit_formal_suite16_results_v754_r10 as original_audit
from scripts import audit_environment_resumed_formal_VAL_v824_r10 as environment_audit
from scripts.run_formal_case_author_outcome_v751_r10 import BINDING_KEYS,require_native_protocol

SOURCE='scripts/audit_composite_TEST_results_v1012_r10.py'
TEST_SOURCE='tests/test_composite_TEST_results_v1012.py'
CORE='our_mani/multitask/composite_TEST_resume_v1010.py'
OWNER='scripts/run_composite_TEST_owner_v1011_r10.py'
CONTROLLER='scripts/continue_composite_TEST_v1013_r10.py'
ORIGINAL=physical_owner.PREVIOUS
INTERMEDIATE=physical_owner.RUNTIME
PREVIOUS='runtime/framework/formal_TEST_composite_resume_v956'
RUNTIME='runtime/framework/formal_TEST_composite_resume_v1011'
SETUP='artifacts/framework/multitask_r10/formal_TEST_composite_resume_v1011_setup'
CONTROL=SETUP+'/resume_control.json';LOCK=SETUP+'/resume_inputs.json'
OUTPUT='artifacts/framework/multitask_r10/formal_TEST_composite_resume_v1011/TEST_results.json'
SCHEMA='r10_four_tier_composite_formal_TEST_results_v1012'
SOURCES=tuple(dict.fromkeys((SOURCE,TEST_SOURCE,CORE,OWNER,CONTROLLER,
    'scripts/prepare_composite_TEST_resume_v1010_r10.py',
    'tests/test_composite_TEST_controller_v1013.py',*parent_audit.SOURCES)))
EvidenceLedger=previous_audit.EvidenceLedger
require_clean_terminal=previous_audit.require_clean_terminal


def require(value,message):
    if not value:raise ValueError(message)


def core_module():
    from our_mani.multitask import composite_TEST_resume_v1010
    return composite_TEST_resume_v1010


def require_not_stopped():
    names=('runtime/framework/multitask_r10_STOP',RUNTIME+'/STOP',PREVIOUS+'/STOP',INTERMEDIATE+'/STOP',
        'runtime/framework/ordered_composite_TEST_v1013/STOP','runtime/framework/ordered_composite_TEST_v958/STOP',
        ORIGINAL+'/STOP','runtime/framework/ordered_concurrent_TEST_v907/STOP',
        'runtime/framework/ordered_env_resume_VAL_TEST_v827/STOP')
    require(not any((ROOT/name).exists() for name in names),'STOP has priority over composite1012 audit')


def tier(row):
    name=row.get('evidence_runtime')
    require(name in (ORIGINAL,INTERMEDIATE,PREVIOUS,RUNTIME),'only the four actual physical evidence owners are admitted')
    kind=row.get('execution_entry',{}).get('kind')
    require(kind==('original751' if name==ORIGINAL else 'batched901'),
        'actual753 direct751 and actual902/956 batched901 provenance required')
    return name


def require_four_tier_plan(control,parent_control,old_control,original_records,middle_records,parent_records,new_plan):
    """Close each actual parent transformation, never rewrite either old lineage."""
    parent_audit.require_three_tier_plan(parent_control,old_control,
        original_records['plan'],original_records['status'],original_records['terminal'],
        middle_records['plan'],middle_records['status'],middle_records['terminal'],parent_records['plan'])
    core_module().require_resume_plan(control,parent_records['plan'],parent_records['status'],parent_records['terminal'],new_plan)
    completed=set(control['completed_job_ids']);prior={r['job_id']:r for r in parent_records['plan']['jobs']}
    require(set(middle_records['status']['completed'])<=set(parent_records['status']['completed'])<=completed,
        'every historical completion must remain adopted, without source/outcome filtering')
    for name,record in middle_records['status']['completed'].items():
        require(parent_records['status']['completed'][name]==record,'exact902 historical terminal cannot change in956')
    for row in new_plan['jobs']:
        parent=prior[row['job_id']];lineage=row.get('resume_lineage_v1010',{})
        require(lineage.get('schema')==core_module().LINEAGE_SCHEMA
            and lineage.get('parent_row_sha256')==digest(parent)
            and lineage.get('parent_imported_completed') is parent['imported_completed']
            and lineage.get('parent_evidence_runtime')==parent['evidence_runtime']
            and lineage.get('parent_runtime')==PREVIOUS
            and row.get('resume_lineage')==parent.get('resume_lineage')
            and row.get('previous_row_sha256')==parent.get('previous_row_sha256'),
            'new lineage must bind the entire956 parent while preserving both older lineage fields')
        if row['job_id'] in completed:
            require(tier(row)==parent['evidence_runtime'],'an adopted case keeps its real753/902/956 owner')
    counts=Counter(tier(r) for r in new_plan['jobs'])
    require(counts[ORIGINAL]==len(old_control['completed_job_ids'])
        and counts[INTERMEDIATE]==len(parent_control['completed_job_ids'])-counts[ORIGINAL]
        and counts[PREVIOUS]==len(completed)-len(parent_control['completed_job_ids'])
        and counts[RUNTIME]==1920-len(completed),'four actual source counts must match each literal parent partition')
    return counts


def require_tier_launch(actual,row,bindings,*,check,identity,stage_owner,control):
    """The source runtime selects the real owner PID, never current imported flag."""
    original_audit.require_actual_launch(actual,row,bindings,check=check,identity=identity)
    require(type(actual.get('launcher_pid')) is int and actual['launcher_pid']>0
        and type(actual.get('launcher_start_ticks')) is int and actual['launcher_start_ticks']>0
        and actual.get('owner_pid')==stage_owner.get('pid') and type(stage_owner.get('pid')) is int,
        'physical leaf must bind its real evidence-tier owner PID and unique PID/start')
    if tier(row)==ORIGINAL:
        require(row['physical_argv']==row['argv'] and all(key not in actual for key in
            ('execution_entry','physical_argv','inherited_frozen_environment','current_mount_namespace')),
            'old753 format cannot acquire invented901 or namespace fields')
    else:
        receipt=environment_audit.require_environment_receipt(stage_owner.get('frozen_environment'),control,
            owner_pid=stage_owner['pid'])
        require(receipt['observer_start_ticks']==stage_owner.get('start_ticks')
            and actual.get('execution_entry')==row['execution_entry']
            and actual.get('physical_argv')==row['physical_argv']+(['--check'] if check else [])
            and actual.get('inherited_frozen_environment')==receipt
            and actual.get('current_mount_namespace')==receipt['current_mount_namespace'],
            'batched child must retain real outer/physical argv and its own stage-owner namespace')


def require_case_outcome(case,identity,protocol,policy,profile):
    """Original749/530/clock gates; motion None and nativeFalse are not filters."""
    require(identity.get('sha256')==digest({k:v for k,v in identity.items() if k!='sha256'}),'actual identity digest differs')
    validate_completed_case(case,identity);require_native_protocol(case,protocol)
    require_attestation(case.get('common_author_RML_parameters',{}),profile)
    domain=case.get('component_domain',{});witness=case.get('full16_frozen_execution',{})
    motion=case.get('all_stage_counted_joint_motion',{});clock=case['counted_clock'];ticks=clock['ticks']
    observed=measured_motion(case)
    require(type(case.get('official_task_success')) is bool and (observed is None or type(observed) is bool)
        and case.get('cleanup_completed') is True and domain.get('purpose')=='evaluation'
        and domain.get('split')==identity.get('split')=='test'
        and witness.get('policy_sha256')==policy and witness.get('source_row_sha256')==identity['source_row_sha256']
        and isinstance(motion.get('events'),list) and len(motion['events'])==ticks,
        'native cleanup, source, original evaluation scope and every counted motion window must agree')
    return dict(native_success=case['official_task_success'],motion_pass=observed,ticks=ticks)


def summarize(cases):
    result=previous_audit.summarize(cases)
    for task in result:
        for method,value in task['methods'].items():
            selected=[c for c in cases if c['task']==task['task'] and c['method']==method]
            value['measured_motion_unknown_cases']=sum(measured_motion(c) is None for c in selected)
    return result


def audit_complete_grid(cases,manifest,routes):
    """Original283/531 scientific grid, allowing unknown diagnostic motion.

    Frozen531 delegates to an older grid that requires every diagnostic to be
    bool. This version makes that sole reporting-presence adaptation explicit;
    it does not edit cases or coerce unknown to a pass, failure or zero.
    """
    sources=validate_roster(manifest,'test',completed_VAL_audit=True)
    policy=policy_fingerprint(routes);profile=fingerprint(routes['common_author_RML_profile'])
    expected={(r['task'],r['raw_episode_id'],m) for r in sources for m in METHODS}
    source_sha={(r['task'],r['raw_episode_id']):digest(r) for r in sources};keys=[];rows=[]
    for case in cases:
        raw=case.get('source_raw_episode_id',case.get('parent_raw_episode_id',case.get('raw_episode_id')))
        key=(case.get('task'),raw,case.get('method'));keys.append(key)
        require(key in expected and case.get('split')=='test','case outside original640source/three-method TEST grid')
        require(case.get('complete') is True and case.get('cleanup_completed') is True and case.get('infrastructure_error') is None,
            'infrastructure or partial cases cannot enter full results')
        witness=case.get('full16_frozen_execution',{})
        require(witness.get('policy_sha256')==policy and witness.get('source_row_sha256')==source_sha[key[:2]],
            'case policy/source must equal original frozen full16 suite')
        require(not case.get('demo_only') and not case.get('development_demo_only') and case.get('TEST_used_for_fitting') is False,
            'formal grid cannot contain development or fitted TEST cases')
        native=case.get('official_task_success');motion=measured_motion(case)
        require(type(native) is bool and (motion is None or type(motion) is bool),'strict native outcome and bool-or-unknown motion required')
        require_attestation(case.get('common_author_RML_parameters',{}),profile)
        rows.append(dict(task=key[0],source=raw,method=key[2],native_success=native,motion_pass=motion))
    require(len(keys)==len(set(keys)) and set(keys)==expected,'complete unique original16/640/1920 cohort required')
    ours=[r for r in rows if r['method']!='physmani']
    return dict(source_cases=len(sources),method_cases=len(rows),completed_method_cases=len(rows),methods=list(METHODS),tasks=list(TASKS),
        policy_sha256=policy,native_successes_by_method={m:sum(r['native_success'] for r in rows if r['method']==m) for m in METHODS},
        measured_motion_violations_by_method={m:sum(r['motion_pass'] is False for r in rows if r['method']==m) for m in METHODS},
        measured_motion_unknown_by_method={m:sum(r['motion_pass'] is None for r in rows if r['method']==m) for m in METHODS},
        ours_measured_motion_passed=(None if any(r['motion_pass'] is None for r in ours) else all(r['motion_pass'] for r in ours)),
        native_success_is_not_a_release_threshold=True,baseline_algorithm_unchanged=True,complete=True,split='test',TEST_used_for_fitting=False,
        common_RML_profile_sha256=profile,motion_is_diagnostic_only=True,motion_metrics_required_for_all_methods=True,
        native_task_success_unchanged=True,unknown_motion_is_not_replaced=True,
        original531_diagnostic_presence_adapted_for_unknown=True,original_scientific_grid_and_RML_rules_retained=True)


def require_bindings(control,parent_control,old_control,original_launch,middle_records,parent_records,new_records,
                     *,source_sha,driver,guard,control_pointer,lock_pointer,audit_environment):
    """All four owner identities and actual quotas; only1011 is full complete."""
    protocol=control['author_outcome_protocol']
    fixed=dict(split='test',policy_sha256=control['policy_sha256'],admission_driver=driver,admission_guard=guard,
        execution_protocol_changed=True,native_runtime=protocol['runtime_consumer'],original_native_runtime=protocol['original_runtime'],
        author_outcome_protocol=protocol,automatic_retry=False)
    stages=((ORIGINAL,original_launch,'scripts/run_formal_suite16_owner_v753_r10.py'),
        (INTERMEDIATE,middle_records['launch'],'scripts/run_concurrent_resumed_formal_TEST_owner_v902_r10.py'),
        (PREVIOUS,parent_records['launch'],'scripts/run_composite_TEST_owner_v956_r10.py'),(RUNTIME,new_records['launch'],OWNER))
    for value in (original_launch,*middle_records.values(),*parent_records.values(),*new_records.values()):
        require(all(value.get(k)==v for k,v in fixed.items()),'each real physical owner protocol/scope binding required')
    for runtime,launch,source in stages:
        require(launch.get('source_sha256')==physical_owner.pointer(ROOT/source)['sha256']
            and launch.get('suite_lock_sha256')==control['suite_lock']['sha256']
            and launch.get('split_source_lock_sha256')==source_sha
            and type(launch.get('pid')) is int and launch['pid']>0
            and type(launch.get('start_ticks')) is int and launch['start_ticks']>0,'actual owner source/PID/start/input binding required')
    require(len({(v['pid'],v['start_ticks']) for _,v,_ in stages})==4,'four distinct actual physical owner identities required')
    require(all(original_launch.get(k)==v for k,v in dict(max_native_slots=2,max_GPU_slots=1,max_CPU_preflight_slots=2).items()),
        'original753 quotas must remain actual2/1/2')
    configurations=((middle_records,old_control,parent_control['previous_resume_control'],parent_control['previous_resume_control_lock'],
        'scripts/run_concurrent_resumed_formal_TEST_owner_v902_r10.py',INTERMEDIATE,physical_owner.OUTPUT,physical_owner),
        (parent_records,parent_control,control['previous_resume_control'],control['previous_resume_control_lock'],
        'scripts/run_composite_TEST_owner_v956_r10.py',PREVIOUS,parent_audit.core_module().OUTPUT,physical_owner),
        (new_records,control,control_pointer,lock_pointer,OWNER,RUNTIME,core_module().OUTPUT,current_owner))
    for records,ctl,cp,lp,source,runtime,out,validator in configurations:
        common=dict(resume_control=cp,resume_control_lock=lp,resource_profile=ctl['resource_profile'],
            resource_qualification=ctl['resource_qualification'],execution_implementation=ctl['execution_implementation'])
        require(all(all(v.get(k)==x for k,x in common.items()) for v in records.values()),'each tier keeps its real resource/control receipts')
        launch=records['launch'];profile=validator.validate_resource_profile(ctl['resource_profile'])
        require(launch.get('execution_entry')==ctl['execution_entry']
            and all(launch.get(k)==profile[k] for k in ('max_native_slots','max_GPU_slots','max_CPU_preflight_slots')),
            'actual owner entry and independently qualified quotas must match')
        receipt=environment_audit.require_environment_receipt(launch.get('frozen_environment'),ctl,owner_pid=launch['pid'])
        require(receipt['observer_start_ticks']==launch['start_ticks']
            and all(v.get('frozen_environment')==receipt for v in records.values()),'owner records must preserve their own actual namespace capture')
        expected={'--suite':ctl['suite']['path'],'--suite-lock':ctl['suite_lock']['path'],'--source-inputs':ctl['source_inputs']['path'],
            '--validation-release':ctl['validation_release']['path'],'--runtime':runtime,'--out':out,
            '--resume-control':cp['path'],'--resume-lock':lp['path'],'--resume-lock-sha256':lp['sha256']}
        argv=launch.get('argv')
        require(isinstance(argv,list) and len(argv)==1+2*len(expected) and argv[0] in (source,str(ROOT/source))
            and all(physical_owner.option(argv,k)==v for k,v in expected.items()),'exact actual owner argv/control pin required')
    for records,ctl in ((parent_records,parent_control),(new_records,control)):
        require(all(all(v.get(k)==ctl[k] for k in ('previous_resume_control','previous_resume_control_lock','environment_recheck'))
            for v in records.values()),'each later tier retains its actual parent control and point observation')
    require(middle_records['launch'].get('base_stage_owner')==physical_owner.pointer(ROOT/'scripts/run_formal_suite16_owner_v753_r10.py')
        and parent_records['launch'].get('base_stage_owner')==middle_records['launch']['base_stage_owner']
        and parent_records['launch'].get('parent_stage_owner')==physical_owner.pointer(ROOT/'scripts/run_concurrent_resumed_formal_TEST_owner_v902_r10.py')
        and new_records['launch'].get('base_stage_owner')==middle_records['launch']['base_stage_owner']
        and new_records['launch'].get('parent_stage_owner')==physical_owner.pointer(ROOT/'scripts/run_composite_TEST_owner_v956_r10.py'),
        'base/parent stage owner fields are real source pointers, never launch pointers')
    require(all(v.get('resource_profile_changed') is True and v.get('previous_resource_profile')==parent_control['resource_profile']
        and v.get('previous_resource_qualification')==parent_control['resource_qualification'] for v in new_records.values()),
        'only new1011 discloses the qualified quota change, historical physical policy is unchanged')
    plan,status,terminal=(new_records[k] for k in ('plan','status','terminal'));count=len(control['completed_job_ids'])
    require(new_records['launch'].get('imported_completed_cases')==plan.get('imported_completed_cases')==terminal.get('imported_completed_cases')==status.get('imported_completed')==count
        and terminal.get('resumed_new_cases')==1920-count and terminal.get('complete') is True
        and terminal.get('expected')==terminal.get('completed')==1920 and terminal.get('failed')==0
        and terminal.get('owner_error') is None and terminal.get('stop_requested') is False
        and terminal.get('drain_requested') is False and terminal.get('drained') is False
        and terminal.get('CPU_not_launched')==0 and terminal.get('CPU_ready_unexecuted')==[]
        and terminal.get('formal_release_requires_independent_audit') is True
        and status.get('expected')==1920 and len(status.get('completed',{}))==1920 and status.get('failed')=={}
        and status.get('active')==status.get('checking')==status.get('ready_jobs')==[]
        and status.get('CPU_ready')==status.get('native_slots')==status.get('GPU_slots')==0
        and status.get('CPU_launched')==1920-count and status.get('owner_error') is None
        and status.get('stop_requested') is False and status.get('drain_requested') is False,
        'only new1011 complete clean unique1920 owner may supply final results')
    environment_audit.require_environment_receipt(audit_environment,control)
    return {runtime:value for runtime,value,_ in stages}


def require_control(control,old_control,files,batch,ledger,control_pointer):
    """Freeze both old authorization and new provenance; no future SHA invented."""
    core=core_module()
    require(control.get('schema')==core.CONTROL_SCHEMA and control.get('split')=='test'
        and control.get('expected_cases')==1920 and control.get('source_count')==640
        and control.get('previous_runtime')==PREVIOUS
        and control.get('previous_controller_runtime')=='runtime/framework/ordered_composite_TEST_v958'
        and control.get('user_authorized_resume') is True and control.get('standing_authorization_reused') is True
        and control.get('new_user_event_invented') is False and control.get('physical_policy_changed') is False,
        'genuine same-policy fourth-tier TEST continuation required')
    inherited=('suite','suite_lock','source_inputs','validation_release','policy_sha256','author_outcome_protocol',
        'execution_entry','execution_implementation','standing_continuation_record',*previous_audit.ENVIRONMENT_FIELDS)
    require(all(control.get(k)==old_control.get(k) for k in inherited),'original scientific/physical implementation scope cannot change')
    require(control.get('resource_profile_changed') is True
        and control.get('previous_resource_profile')==old_control.get('resource_profile')
        and control.get('previous_resource_qualification')==old_control.get('resource_qualification'),
        'actual historical resource tier and explicit newly qualified profile required')
    require(current_owner.validate_resource_profile(control['resource_profile'])==dict(
        schema=current_owner.PROFILE_SCHEMA,max_native_slots=8,max_GPU_slots=3,max_CPU_preflight_slots=4,ready_limit=16),
        'exact independently qualified8/3/4/16 resource profile required')
    require(control.get('resume_owner')==physical_owner.pointer(ROOT/OWNER)
        and control.get('resume_auditor')==control.get('result_auditor')==physical_owner.pointer(ROOT/SOURCE)
        and control.get('resume_controller')==physical_owner.pointer(ROOT/CONTROLLER),
        'exact1011/1012/1013 control sources required')
    require(control.get('previous_resume_control',{}).get('path')==core.PREVIOUS_CONTROL
        and control.get('previous_resume_control_lock',{}).get('path')==core.PREVIOUS_LOCK,
        'exact original955 control and input-lock paths required')
    required=('resume_owner','resume_auditor','result_auditor','resume_controller','suite','suite_lock','source_inputs',
        'validation_release','resource_qualification','resource_qualifier','standing_continuation_record',
        'concurrency_authorization_record','previous_resume_control','previous_resume_control_lock','environment_recheck',
        'previous_launch','previous_plan','previous_status','previous_terminal','previous_controller_launch',
        'previous_controller_status','previous_controller_failure','previous_DRAIN','original_launch','original_plan','original_status','original_terminal',
        *previous_audit.ENVIRONMENT_FIELDS)
    for key in required:
        p=control.get(key);require(physical_owner.valid_pointer(p),'exact frozen control pointer required: '+key)
        ledger.frozen(batch,files,p,parse=False)
    ledger.frozen(batch,files,control_pointer,parse=False)
    for name in SOURCES:ledger.frozen(batch,files,physical_owner.pointer(ROOT/name),parse=False)
    for key,leaf in (('launch','launch.json'),('plan','plan.json'),('status','status.json'),('terminal','terminal.json')):
        require(control['previous_'+key]['path']==PREVIOUS+'/'+leaf
            and control['original_'+key]==old_control['original_'+key],
            'original753 and parent956 metadata must retain actual frozen paths')
    for key,leaf in (('launch','launch.json'),('status','status.json'),('failure','failure.json')):
        require(control['previous_controller_'+key]['path']=='runtime/framework/ordered_composite_TEST_v958/'+leaf,
            'previous958 must retain genuine partial-failure controller metadata')
    require(control['previous_DRAIN']['path']==PREVIOUS+'/DRAIN','actual old956 DRAIN evidence must remain frozen')
    state=ledger.json(control['previous_controller_status']['path']);failure=ledger.json(control['previous_controller_failure']['path'])
    require(state.get('stage')=='requires_concrete_repair' and state.get('automatic_retry') is False
        and failure.get('complete') is False and failure.get('user_stop') is False and failure.get('automatic_retry') is False,
        'old958 failure is retained, never replaced with an invented successful terminal')
    observation=ledger.json(control['environment_recheck']['path']);core.require_environment_recheck(observation,control)
    require(observation.get('schema')=='r10_fresh_frozen_TEST_environment_CPU_observation_v954'
        and observation.get('complete') is True and observation.get('Native_started') is False
        and observation.get('GPU_started') is False and observation.get('admission_or_TEST_started') is False
        and observation.get('declared_model_case_raw_bytes_read') is False
        and observation.get('scientific_inputs_or_old_locks_modified') is False
        and observation.get('parsed_suite_lock_is_exact_hashed_payload') is True
        and observation.get('original_v900_topology_constructor_passed') is True
        and observation.get('original_full_topology_final_check_passed') is True
        and observation.get('topology_is_point_observation_not_continuous') is True
        and observation.get('suite_lock')==control['suite_lock'],
        '954 is only a real same-lock OS/topology point observation, never qualification')
    require(physical_owner.valid_pointer(observation.get('source'))
        and observation['source']['path']=='scripts/check_frozen_TEST_environment_v954_r10.py','actual954 observer source required')
    ledger.frozen(batch,files,observation['source'],parse=False)
    environment_audit.require_environment_receipt(observation.get('actual_environment'),control)
    entry=physical_owner.validate_entry(control['execution_entry']);require(entry['kind']=='batched901','new1011 actual901 entry required')
    from scripts import run_batched_formal_driver_v901_r10 as bootstrap
    for key in ('batch_control','batch_lock'):ledger.frozen(batch,files,entry[key],parse=False)
    small=ledger.json(entry['batch_lock']['path'])
    require(small.get('schema')==bootstrap.LOCK_SCHEMA and small.get('execution_enabled') is True,'actual enabled901 small lock required')
    sources={name:physical_owner.pointer(ROOT/name) for name in bootstrap.SOURCES}
    for p in sources.values():
        ledger.frozen(batch,files,p,parse=False);ledger.frozen(batch,small['files'],p,parse=False)
    require(control['execution_implementation']==physical_owner.execution_implementation(entry,sources),
        'actual901 implementation must remain the independently qualified904 one')
    from scripts import prepare_composite_TEST_resume_v1010_r10 as preparation
    qualification=ledger.json(control['resource_qualification']['path'])
    audit_pointer=qualification.get('actual_independent_audit')
    require(physical_owner.valid_pointer(audit_pointer)
        and audit_pointer['path']==preparation.QUALIFICATION_LAUNCH,'real independent1005 audit launch required')
    audit_launch=ledger.frozen(batch,files,audit_pointer)
    preparation.require_resource_proof(qualification,control,audit_launch=audit_launch)
    for name,checksum in qualification['inputs'].items():
        require(preparation.qualification_path_allowed(name),'only declared1005 qualification evidence admitted')
        ledger.frozen(batch,files,dict(path=name,sha256=checksum),parse=False)
    for key in ('control','lock'):ledger.frozen(batch,files,qualification[key],parse=False)
    standing=ledger.json(control['standing_continuation_record']['path'])
    authorization=ledger.json(control['concurrency_authorization_record']['path'])
    require(standing.get('continue_until_user_stop') is True and standing.get('user_stop_received') is False
        and authorization.get('user_authorized_concurrency_change') is True,
        'standing continuation and genuine resource-change authorization must remain explicit')
    return sources


def require_historical_pointer(batch,files,parent_files,old_files,p,runtime):
    batch.locked_pointer(ROOT,files,p)
    if runtime in (ORIGINAL,INTERMEDIATE):batch.locked_pointer(ROOT,parent_files,p)
    if runtime==ORIGINAL:batch.locked_pointer(ROOT,old_files,p)


def require_final_owner(records,count):
    """Small completion gate before any full graph or TEST source is opened."""
    launch,plan,status,terminal=(records[k] for k in ('launch','plan','status','terminal'))
    require(launch.get('imported_completed_cases')==plan.get('imported_completed_cases')
        ==terminal.get('imported_completed_cases')==status.get('imported_completed')==count
        and terminal.get('complete') is True and terminal.get('expected')==terminal.get('completed')==1920
        and terminal.get('failed')==0 and terminal.get('resumed_new_cases')==1920-count
        and terminal.get('automatic_retry') is False and terminal.get('stop_requested') is False
        and terminal.get('drain_requested') is False and terminal.get('drained') is False
        and terminal.get('owner_error') is None and terminal.get('CPU_not_launched')==0
        and terminal.get('CPU_ready_unexecuted')==[]
        and status.get('expected')==1920 and isinstance(status.get('completed'),dict)
        and len(status['completed'])==1920 and status.get('failed')=={}
        and status.get('active')==status.get('checking')==status.get('ready_jobs')==[]
        and status.get('CPU_ready')==status.get('native_slots')==status.get('GPU_slots')==0
        and status.get('stop_requested') is False and status.get('drain_requested') is False
        and status.get('owner_error') is None and status.get('automatic_retry') is False,
        'actual clean complete1011 owner required before full TEST graph access')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime',default=RUNTIME);parser.add_argument('--out',default=OUTPUT)
    parser.add_argument('--resume-control',default=CONTROL);parser.add_argument('--resume-lock',default=LOCK)
    parser.add_argument('--resume-lock-sha256',required=True);args=parser.parse_args(argv)
    require(args.runtime==RUNTIME and args.out==OUTPUT and args.resume_control==CONTROL and args.resume_lock==LOCK,
        'fixed new1012 output/control/runtime namespace required')
    out=ROOT/OUTPUT;require(not out.exists(),'never overwrite prior composite TEST results')
    require_not_stopped();ledger=EvidenceLedger(ROOT)
    lock_pointer=ledger.pointer(LOCK)
    require(physical_owner.valid_pointer(lock_pointer) and lock_pointer['sha256']==args.resume_lock_sha256,
        'external exact1010 input-lock byte SHA required before following pointers')
    lock=ledger.json(LOCK);files=lock.get('files')
    require(lock.get('schema')==core_module().LOCK_SCHEMA and lock.get('approved_for_execution') is True
        and isinstance(files,dict),'approved1010 frozen source/control/evidence supplement required')
    control_pointer=ledger.pointer(CONTROL)
    matches=[files[n] for n in (CONTROL,str(ROOT/CONTROL)) if n in files]
    require(matches==[control_pointer['sha256']],'control must have one exact declaration before data/release reads')
    require(lock.get('control')==control_pointer,'actual1010 input lock must bind the exact parsed control bytes')
    control=ledger.json(CONTROL)
    # The genuine complete VAL gate precedes any verify_inputs touching TEST.
    release=physical_owner.require_complete_VAL_release(control,control['validation_release']['path'])
    new_records={k:ledger.json(RUNTIME+'/'+k+'.json') for k in ('launch','plan','status','terminal')}
    require_final_owner(new_records,len(control.get('completed_job_ids',[])))
    verify_inputs(ROOT,files);audit_environment=environment_audit.capture_environment(control)
    with LookupBatch(ROOT) as batch:
        parent_control=ledger.frozen(batch,files,control['previous_resume_control'])
        parent_lock=ledger.frozen(batch,files,control['previous_resume_control_lock']);parent_files=parent_lock['files']
        require(parent_lock.get('schema')==parent_audit.core_module().LOCK_SCHEMA and parent_lock.get('approved_for_execution') is True,
            'actual immutable955 input lock required')
        require(all(files.get(k)==v for k,v in parent_files.items()),'every declared955 key/SHA must remain unchanged in1010')
        verify_inputs(ROOT,parent_files)
        old_control=ledger.frozen(batch,parent_files,parent_control['previous_resume_control'])
        old_lock=ledger.frozen(batch,parent_files,parent_control['previous_resume_control_lock']);old_files=old_lock['files']
        require(old_lock.get('schema')==physical_owner.LOCK_SCHEMA and old_lock.get('approved_for_execution') is True,
            'actual immutable915 input lock and frozen graph required')
        require(all(parent_files.get(k)==v for k,v in old_files.items()),'every declared915 key/SHA remains unchanged in955')
        verify_inputs(ROOT,old_files)
        old_sources=previous_audit.require_control(old_control,old_files,batch,ledger,
            parent_control['previous_resume_control'],parent_control['previous_resume_control_lock'])
        parent_sources=parent_audit.require_control(parent_control,old_control,parent_files,batch,ledger,control['previous_resume_control'])
        batch_sources=require_control(control,parent_control,files,batch,ledger,control_pointer)
        require(batch_sources==parent_sources==old_sources,'all original901/900 implementation sources must stay identical across four tiers')
        suite_lock=ledger.json(control['suite_lock']['path']);suite_files=suite_lock['files'];verify_inputs(ROOT,suite_files)
        suite=ledger.frozen(batch,suite_files,control['suite'])
        for key,name in (('formal_case_driver',physical_owner.ORIGINAL_ENTRY),
                ('formal_admission_guard','our_mani/multitask/formal_suite_admission_v728.py'),
                ('formal_result_auditor','scripts/audit_formal_suite16_results_v754_r10.py')):
            require(suite.get(key)==physical_owner.pointer(ROOT/name),'original frozen physical suite pointer differs: '+key)
            ledger.frozen(batch,suite_files,suite[key],parse=False)
        protocol=outcome_protocol_binding(ROOT)
        require(suite.get('author_outcome_protocol')==control['author_outcome_protocol']==protocol,
            'unchanged741/743/744 protocol required')
        for key in ('implementation','runtime_consumer','original_runtime'):ledger.frozen(batch,suite_files,protocol[key],parse=False)
        driver,guard=suite['formal_case_driver'],suite['formal_admission_guard']
        manifest,sources,proof=read_formal_source(ROOT,suite,suite_files,'test',validation_release=release)
        policy=proof['policy_sha256'];require(policy==control['policy_sha256'] and len(sources)==640,'original640 same-policy TEST cohort required')
        source_meta=ledger.json(control['source_inputs']['path']);source_sha=control['source_inputs']['sha256']
        require(source_meta.get('schema')=='r10_full16_post_admission_split_source_lock_v1'
            and source_meta.get('split')=='test' and source_meta.get('source_count')==640
            and source_meta.get('policy_sha256')==policy and source_meta.get('TEST_used_for_fitting') is False
            and source_meta.get('source_manifest_sha256')==suite['source_manifests']['test']['sha256'],
            'same original admitted640 split lock required')
        verify_inputs(ROOT,source_meta['files'])
        original_records={k:ledger.json(control['original_'+k]['path']) for k in ('launch','plan','status','terminal')}
        middle_records={k:ledger.json(parent_control['previous_'+k]['path']) for k in ('launch','plan','status','terminal')}
        parent_records={k:ledger.json(control['previous_'+k]['path']) for k in ('launch','plan','status','terminal')}
        counts=require_four_tier_plan(control,parent_control,old_control,original_records,middle_records,parent_records,new_records['plan'])
        previous_audit.require_original_plan(original_records['plan'],control,suite,sources,proof,driver,guard)
        require(original_records['terminal'].get('author_outcome_protocol')==protocol
            and original_records['terminal'].get('execution_protocol_changed') is True,
            'original753 terminal protocol must stay original, even though it is an incomplete drain')
        stage_owners=require_bindings(control,parent_control,old_control,original_records['launch'],middle_records,parent_records,new_records,source_sha=source_sha,
            driver=driver,guard=guard,control_pointer=control_pointer,lock_pointer=lock_pointer,audit_environment=audit_environment)
        for launch in stage_owners.values():
            require(launch.get('source_manifest')==suite['source_manifests']['test'] and not physical_owner.alive(launch),
                'each actual physical owner must bind the source manifest and have exited')
        require(middle_records['launch'].get('batch_sources')==parent_records['launch'].get('batch_sources')==new_records['launch'].get('batch_sources')==batch_sources,
            'old902 and new956 must disclose actual unchanged901 source declarations')
        for pointer in (control['previous_controller_launch'],parent_control['previous_controller_launch'],old_control['previous_controller_launch']):
            require(not physical_owner.alive(ledger.json(pointer['path'])),'historical physical controllers must have exited')
        from scripts import run_batched_formal_driver_v901_r10 as bootstrap
        # Each distinct formal901 control is independently checked on an actual
        # row. No per-case whole-graph construction is introduced by this audit.
        representatives={}
        for row in new_records['plan']['jobs']:
            if tier(row)!=ORIGINAL:representatives.setdefault(digest(row['execution_entry']),row)
        for row in representatives.values():
            entry=row['execution_entry'];repeated=bootstrap.load_control(entry['batch_control']['path'],entry['batch_lock']['path'],
                entry['batch_lock']['sha256'],'formal',row['physical_argv'][3:])
            require(repeated['sources']==batch_sources,'actual old/new901 source/control gate differs')
        profile=fingerprint(proof['routes']['common_author_RML_profile']);cases=[];case_pointers=[];driver_pointers=[];batch_pointers=[];tier_rows=[]
        seen_processes=set();status=new_records['status'];plan=new_records['plan'];old_ids=set(control['completed_job_ids'])
        require(set(status['completed'])=={r['job_id'] for r in plan['jobs']},'all1920 unique actual completion keys required')
        for row in plan['jobs']:
            require_not_stopped();name=row['job_id'];runtime=tier(row);stage_owner=stage_owners[runtime]
            bindings={key:row[key] for key in BINDING_KEYS};leaf=ROOT/runtime
            cpu_leaf=leaf/'CPU'/name;native_leaf=leaf/'native'/name
            cpu=ledger.json(cpu_leaf/'CPU_preflight.json');require(cpu==ledger.json(cpu_leaf/'stdout.log'),'actual CPU JSON/stdout must be identical')
            identity=physical_owner.accept_CPU(cpu,row,source_sha,{} if runtime==ORIGINAL else batch_sources)
            require(identity.get('task')==row['task'] and identity.get('source_raw_episode_id')==row['raw']
                and identity.get('split')=='test' and identity.get('demo_only') is False,'original source/task/formal CPU identity required')
            for child,check in ((cpu_leaf,True),(native_leaf,False)):
                actual=ledger.json(child/'launch.json')
                require_tier_launch(actual,row,bindings,check=check,identity=identity,stage_owner=stage_owner,control=control)
                process=(actual['launcher_pid'],actual['launcher_start_ticks']);require(process not in seen_processes,'one unique actual PID/start per CPU/native request')
                seen_processes.add(process);require_clean_terminal(ledger.json(child/'terminal.json'))
                for filename in ('launch.json','terminal.json','stdout.log','stderr.log'):
                    p=ledger.pointer(child/filename)
                    if name in old_ids:require_historical_pointer(batch,files,parent_files,old_files,p,runtime)
            if name in old_ids:require_historical_pointer(batch,files,parent_files,old_files,ledger.pointer(cpu_leaf/'CPU_preflight.json'),runtime)
            terminal=ledger.json(native_leaf/'terminal.json');record=status['completed'][name];require_clean_terminal(record)
            require(all(record.get(k)==v for k,v in terminal.items()),'new status must preserve the true native terminal')
            if name in old_ids:require(record==parent_records['status']['completed'][name],'all old nativeTrue/False outcomes must remain unchanged')
            case_path=ROOT/physical_owner.option(row['physical_argv'],'--out')/'case.json';driver_path=case_path.parent/'admission_driver_receipt.json'
            cp,dp=ledger.pointer(case_path),ledger.pointer(driver_path)
            require(terminal.get('case')==record.get('case')==cp and terminal.get('admission_driver_receipt')==record.get('admission_driver_receipt')==dp,
                'each terminal/status must bind its actual case and independent751 receipt')
            if name in old_ids:
                require_historical_pointer(batch,files,parent_files,old_files,cp,runtime);require_historical_pointer(batch,files,parent_files,old_files,dp,runtime)
            case=ledger.json(case_path)
            original_audit.require_driver_evidence(ledger.json(driver_path),driver,guard,dict(worker=row['case_worker'],consumer=row['native_runtime']),case_path,'test')
            metrics=require_case_outcome(case,identity,protocol,policy,profile)
            require(record.get('native_success') is metrics['native_success'] and 'motion_pass' in record
                and record['motion_pass'] is metrics['motion_pass'],'recorded native/motion values must preserve actual case, including unknown')
            bp=None
            if runtime!=ORIGINAL:
                batch_path=case_path.parent/'batch_driver_receipt.json';bp=ledger.pointer(batch_path)
                physical_owner.require_batch_execution(ledger.json(batch_path),row,check=False,expected_sources=batch_sources,case=cp,original_receipt=dp)
                require(terminal.get('batch_driver_receipt')==record.get('batch_driver_receipt')==bp,'each901 sidecar needs true terminal/status binding')
                if name in old_ids:require_historical_pointer(batch,files,parent_files,old_files,bp,runtime)
                batch_pointers.append(bp)
            else:require('batch_driver_receipt' not in terminal and 'batch_driver_receipt' not in record,'original753 cannot invent901 evidence')
            cases.append(case);case_pointers.append(cp);driver_pointers.append(dp)
            tier_rows.append(dict(job_id=name,evidence_runtime=runtime,imported_completed=row['imported_completed'],
                resume_lineage_v1010=row['resume_lineage_v1010'],
                actual_stage_owner_pid=stage_owner['pid'],actual_stage_owner_start_ticks=stage_owner['start_ticks'],
                resume_lineage=row['resume_lineage'],previous_row_sha256=row['previous_row_sha256'],
                execution_entry=row['execution_entry'],actual_outer_argv=row['argv'],physical_argv=row['physical_argv'],
                case=cp,admission_driver_receipt=dp,batch_driver_receipt=bp))
        result=audit_complete_grid(cases,manifest,proof['routes'])
        require(result.get('complete') is True and result.get('source_cases')==640
            and result.get('method_cases')==result.get('completed_method_cases')==1920,'complete original16/640/1920 composite TEST cohort required')
        require(new_records['terminal'].get('native_successes')==sum(c['official_task_success'] for c in cases),'newowner success count must equal independent native cases')
        for frozen in (files,parent_files,old_files,suite_files,source_meta['files']):verify_inputs(ROOT,frozen)
        inputs=ledger.close()
    owner_pointers={ORIGINAL:(control['original_launch'],control['original_terminal']),
        INTERMEDIATE:(parent_control['previous_launch'],parent_control['previous_terminal']),
        PREVIOUS:(control['previous_launch'],control['previous_terminal']),
        RUNTIME:(inputs_pointer(inputs,RUNTIME+'/launch.json'),inputs_pointer(inputs,RUNTIME+'/terminal.json'))}
    result.update(schema=SCHEMA,created_utc=datetime.now(timezone.utc).isoformat(),per_task=summarize(cases),
        cases=case_pointers,admission_driver_receipts=driver_pointers,batch_driver_receipts=batch_pointers,evidence_tiers=tier_rows,
        inputs_sha256=inputs,resume_control=control_pointer,resume_control_lock=lock_pointer,
        base_result_auditor=physical_owner.pointer(ROOT/'scripts/audit_formal_suite16_results_v754_r10.py'),
        previous_composite_auditors=[physical_owner.pointer(ROOT/name) for name in
            ('scripts/audit_concurrent_resumed_formal_TEST_v906_r10.py','scripts/audit_composite_TEST_results_v957_r10.py')],
        actual_stage_owners=[dict(runtime=name,launch=owner_pointers[name][0],terminal=owner_pointers[name][1],
            pid=stage_owners[name]['pid'],start_ticks=stage_owners[name]['start_ticks'],completed=counts[name],
            resource_limits={k:stage_owners[name][k] for k in ('max_native_slots','max_GPU_slots','max_CPU_preflight_slots')})
            for name in (ORIGINAL,INTERMEDIATE,PREVIOUS,RUNTIME)],source_input_lock=control['source_inputs'],validation_release=control['validation_release'],
        resource_profile=control['resource_profile'],resource_qualification=control['resource_qualification'],
        resource_profile_changed=True,previous_resource_profile=control['previous_resource_profile'],
        previous_resource_qualification=control['previous_resource_qualification'],
        execution_implementation=control['execution_implementation'],frozen_environment=audit_environment,
        environment_recheck=control['environment_recheck'],environment_recheck_is_point_observation_not_TEST_qualification=True,
        auditor_invocation=dict(argv=list(sys.argv) if argv is None else [SOURCE,*argv],pid=os.getpid(),
            start_ticks=physical_owner.ticks(os.getpid()),source=control['resume_auditor'],inherited_physical_stage_identity_claimed=False),
        admission_driver=driver,admission_guard=guard,native_runtime=protocol['runtime_consumer'],original_native_runtime=protocol['original_runtime'],
        author_outcome_protocol=protocol,execution_protocol_changed=True,source_sha256=inputs[str(ROOT/SOURCE)],
        frozen_lookup_batch_metrics=dict(batch.metrics),used_evidence_rechecked_at_close=True,
        one_fixed_original_checkpoint_not_top3_selected=True,source_outcome_filtering=False,native_failures_retained=True,
        motion_unknown_retained=True,zero_tick_cases_retained=True,motion_is_diagnostic_only=True,
        historical_infrastructure_attempts=[dict(runtime=runtime,job_ids=ctl['failed_job_ids'],
            previous_status=ctl['previous_status'],previous_terminal=ctl['previous_terminal'],
            counted_as_native_task_failures=False,all_keys_reexecuted_once_in_new_namespace=True)
            for runtime,ctl in ((INTERMEDIATE,parent_control),(PREVIOUS,control))],
        Value_NGFF_FASTER_used=False,automatic_retry=False,historical_completed_cases_retained=len(control['completed_job_ids']),old_completed_results_not_rerun=True,independent_composite_audit=True,
        old907_or958_successful_terminal_claimed=False,original754_906_or957_complete_result_claimed=False)
    require_not_stopped();atomic_json(out,result)
    print(json.dumps(dict(complete=True,schema=SCHEMA,split='test',cases=1920,
        native_successes_by_method=result['native_successes_by_method'],motion_is_diagnostic_only=True,out=OUTPUT)),flush=True)
    return result


def inputs_pointer(inputs,name):return dict(path=name,sha256=inputs[str(ROOT/name)])


if __name__=='__main__':main()
