"""Independent full1920 six-tier audit after an observed host reboot.

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
from scripts import audit_composite_TEST_results_v957_r10 as grandparent_audit
from scripts import audit_composite_TEST_results_v1012_r10 as parent_audit
from scripts import run_composite_TEST_owner_v1011_r10 as prior_owner
from scripts import run_composite_TEST_owner_v1041_r10 as fifth_owner
from scripts import audit_composite_TEST_results_v1042_r10 as fifth_audit
from scripts import run_reboot_TEST_owner_v1056_r10 as current_owner
from scripts import audit_formal_suite16_results_v754_r10 as original_audit
from scripts import audit_environment_resumed_formal_VAL_v824_r10 as environment_audit
from scripts.run_formal_case_author_outcome_v751_r10 import BINDING_KEYS,require_native_protocol

SOURCE='scripts/audit_reboot_TEST_results_v1057_r10.py'
TEST_SOURCE='tests/test_reboot_TEST_results_v1057.py'
CORE='our_mani/multitask/composite_TEST_reboot_resume_v1055.py'
OWNER='scripts/run_reboot_TEST_owner_v1056_r10.py'
CONTROLLER='scripts/continue_reboot_TEST_v1058_r10.py'
ORIGINAL=physical_owner.PREVIOUS
INTERMEDIATE=physical_owner.RUNTIME
GRANDPARENT='runtime/framework/formal_TEST_composite_resume_v956'
PREVIOUS='runtime/framework/formal_TEST_composite_resume_v1011'
FIFTH='runtime/framework/formal_TEST_composite_resume_v1041'
RUNTIME='runtime/framework/formal_TEST_reboot_resume_v1056'
SETUP='artifacts/framework/multitask_r10/formal_TEST_reboot_resume_v1056_setup'
CONTROL=SETUP+'/resume_control.json';LOCK=SETUP+'/resume_inputs.json'
OUTPUT='artifacts/framework/multitask_r10/formal_TEST_reboot_resume_v1056/TEST_results.json'
SCHEMA='r10_six_tier_reboot_formal_TEST_results_v1057'
SOURCES=tuple(dict.fromkeys((SOURCE,TEST_SOURCE,CORE,OWNER,CONTROLLER,
    'scripts/prepare_composite_TEST_reboot_resume_v1055_r10.py',
    'scripts/audit_reboot_TEST_evidence_v1054_r10.py',
    'tests/test_reboot_TEST_controller_v1058.py',*fifth_audit.SOURCES)))
EvidenceLedger=previous_audit.EvidenceLedger
require_clean_terminal=previous_audit.require_clean_terminal


def require(value,message):
    if not value:raise ValueError(message)


def core_module():
    from our_mani.multitask import composite_TEST_reboot_resume_v1055
    return composite_TEST_reboot_resume_v1055


def require_not_stopped():
    names=('runtime/framework/multitask_r10_STOP',RUNTIME+'/STOP',FIFTH+'/STOP',PREVIOUS+'/STOP',GRANDPARENT+'/STOP',INTERMEDIATE+'/STOP',
        'runtime/framework/ordered_reboot_TEST_v1058/STOP','runtime/framework/ordered_composite_TEST_v1043/STOP',
        'runtime/framework/ordered_composite_TEST_v1013/STOP','runtime/framework/ordered_composite_TEST_v958/STOP',
        ORIGINAL+'/STOP','runtime/framework/ordered_concurrent_TEST_v907/STOP',
        'runtime/framework/ordered_env_resume_VAL_TEST_v827/STOP')
    require(not any((ROOT/name).exists() for name in names),'STOP has priority over reboot1057 audit')


def tier(row):
    name=row.get('evidence_runtime')
    require(name in (ORIGINAL,INTERMEDIATE,GRANDPARENT,PREVIOUS,FIFTH,RUNTIME),'only the six actual physical evidence owners are admitted')
    kind=row.get('execution_entry',{}).get('kind')
    require(kind==('original751' if name==ORIGINAL else 'batched901'),
        'actual753 direct751 and actual902/956 batched901 provenance required')
    return name


def require_six_tier_plan(control,fifth_control,parent_control,grandparent_control,old_control,original_records,middle_records,grandparent_records,parent_records,fifth_records,reboot_audit,new_plan):
    """Close the five frozen historical transformations before reboot recovery."""
    fifth_audit.require_five_tier_plan(fifth_control,parent_control,grandparent_control,old_control,
        original_records,middle_records,grandparent_records,parent_records,fifth_records['plan'])
    core=core_module()
    core.require_resume_plan(control,fifth_records['plan'],fifth_records['status'],reboot_audit,new_plan)
    adopted=set(control['completed_job_ids']);prior={r['job_id']:r for r in fifth_records['plan']['jobs']}
    historical=set(parent_records['status']['completed'])
    require(historical<=adopted,'all intact historical753/902/956/1011 completions remain adopted')
    for name,record in parent_records['status']['completed'].items():
        require(fifth_records['status']['completed'][name]==record,
            'literal historical completions cannot change at the reboot boundary')
    for row in new_plan['jobs']:
        parent=prior[row['job_id']];lineage=row.get(core.LINEAGE_FIELD,{})
        require(lineage.get('schema')==core.LINEAGE_SCHEMA
            and lineage.get('parent_row_sha256')==digest(parent)
            and lineage.get('parent_imported_completed') is parent['imported_completed']
            and lineage.get('parent_evidence_runtime')==parent['evidence_runtime']
            and lineage.get('parent_runtime')==FIFTH
            and all(row.get(k)==parent.get(k) for k in
                ('resume_lineage','resume_lineage_v1010','resume_lineage_v1040','previous_row_sha256')),
            'reboot lineage binds the complete1041 row and preserves every older provenance field')
        if row['job_id'] in adopted:
            require(tier(row)==parent['evidence_runtime'],'adopted cases retain actual five-tier ownership')
    counts=Counter(tier(r) for r in new_plan['jobs'])
    require(counts[ORIGINAL]==len(old_control['completed_job_ids'])
        and counts[INTERMEDIATE]==len(grandparent_control['completed_job_ids'])-counts[ORIGINAL]
        and counts[GRANDPARENT]==len(parent_control['completed_job_ids'])-len(grandparent_control['completed_job_ids'])
        and counts[PREVIOUS]==len(fifth_control['completed_job_ids'])-len(parent_control['completed_job_ids'])
        and counts[FIFTH]==len(adopted)-len(fifth_control['completed_job_ids'])
        and counts[RUNTIME]==1920-len(adopted),'six actual runtime counts must match all literal partitions')
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


def require_bindings(control,fifth_control,parent_control,grandparent_control,old_control,original_launch,middle_records,grandparent_records,parent_records,fifth_records,new_records,
                     *,source_sha,driver,guard,control_pointer,lock_pointer,audit_environment):
    """Six real identities; the interrupted1041 owner has no invented terminal."""
    current_owner.require_capacity(control)
    protocol=control['author_outcome_protocol']
    fixed=dict(split='test',policy_sha256=control['policy_sha256'],admission_driver=driver,admission_guard=guard,
        execution_protocol_changed=True,native_runtime=protocol['runtime_consumer'],original_native_runtime=protocol['original_runtime'],
        author_outcome_protocol=protocol,automatic_retry=False)
    stages=((ORIGINAL,original_launch,'scripts/run_formal_suite16_owner_v753_r10.py'),
        (INTERMEDIATE,middle_records['launch'],'scripts/run_concurrent_resumed_formal_TEST_owner_v902_r10.py'),
        (GRANDPARENT,grandparent_records['launch'],'scripts/run_composite_TEST_owner_v956_r10.py'),
        (PREVIOUS,parent_records['launch'],'scripts/run_composite_TEST_owner_v1011_r10.py'),
        (FIFTH,fifth_records['launch'],'scripts/run_composite_TEST_owner_v1041_r10.py'),
        (RUNTIME,new_records['launch'],OWNER))
    require(set(fifth_records)=={'launch','plan','status'},'interrupted1041 retains only its real nonterminal records')
    for value in (original_launch,*middle_records.values(),*grandparent_records.values(),*parent_records.values(),*fifth_records.values(),*new_records.values()):
        require(all(value.get(k)==v for k,v in fixed.items()),'each true owner retains the original protocol and scope')
    for runtime,launch,source in stages:
        require(launch.get('source_sha256')==physical_owner.pointer(ROOT/source)['sha256']
            and launch.get('suite_lock_sha256')==control['suite_lock']['sha256']
            and launch.get('split_source_lock_sha256')==source_sha
            and type(launch.get('pid')) is int and launch['pid']>0
            and type(launch.get('start_ticks')) is int and launch['start_ticks']>0,'real source/PID/start/input binding required')
    require(len({(v['pid'],v['start_ticks']) for _,v,_ in stages})==6,'six distinct physical owner identities required')
    require(all(original_launch.get(k)==v for k,v in dict(max_native_slots=2,max_GPU_slots=1,max_CPU_preflight_slots=2).items()),
        'original753 actual2/1/2 limits cannot change')
    configurations=((middle_records,old_control,grandparent_control['previous_resume_control'],grandparent_control['previous_resume_control_lock'],
        'scripts/run_concurrent_resumed_formal_TEST_owner_v902_r10.py',INTERMEDIATE,physical_owner.OUTPUT,physical_owner),
        (grandparent_records,grandparent_control,parent_control['previous_resume_control'],parent_control['previous_resume_control_lock'],
        'scripts/run_composite_TEST_owner_v956_r10.py',GRANDPARENT,grandparent_audit.core_module().OUTPUT,physical_owner),
        (parent_records,parent_control,fifth_control['previous_resume_control'],fifth_control['previous_resume_control_lock'],
        'scripts/run_composite_TEST_owner_v1011_r10.py',PREVIOUS,parent_audit.core_module().OUTPUT,prior_owner),
        (fifth_records,fifth_control,control['previous_resume_control'],control['previous_resume_control_lock'],
        'scripts/run_composite_TEST_owner_v1041_r10.py',FIFTH,fifth_audit.core_module().OUTPUT,fifth_owner),
        (new_records,control,control_pointer,lock_pointer,OWNER,RUNTIME,core_module().OUTPUT,current_owner))
    for records,ctl,cp,lp,source,runtime,out,validator in configurations:
        common=dict(resume_control=cp,resume_control_lock=lp,resource_profile=ctl['resource_profile'],
            resource_qualification=ctl['resource_qualification'],execution_implementation=ctl['execution_implementation'])
        require(all(all(v.get(k)==x for k,x in common.items()) for v in records.values()),'each tier keeps actual resource/control receipts')
        launch=records['launch'];profile=validator.validate_resource_profile(ctl['resource_profile'])
        require(launch.get('execution_entry')==ctl['execution_entry']
            and all(launch.get(k)==profile[k] for k in ('max_native_slots','max_GPU_slots','max_CPU_preflight_slots')),
            'actual quotas, never the capacity envelope, govern each owner')
        receipt=environment_audit.require_environment_receipt(launch.get('frozen_environment'),ctl,owner_pid=launch['pid'])
        require(receipt['observer_start_ticks']==launch['start_ticks']
            and all(v.get('frozen_environment')==receipt for v in records.values()),'each owner keeps its own real environment receipt')
        expected={'--suite':ctl['suite']['path'],'--suite-lock':ctl['suite_lock']['path'],'--source-inputs':ctl['source_inputs']['path'],
            '--validation-release':ctl['validation_release']['path'],'--runtime':runtime,'--out':out,
            '--resume-control':cp['path'],'--resume-lock':lp['path'],'--resume-lock-sha256':lp['sha256']}
        argv=launch.get('argv')
        require(isinstance(argv,list) and len(argv)==1+2*len(expected) and argv[0] in (source,str(ROOT/source))
            and all(physical_owner.option(argv,k)==v for k,v in expected.items()),'exact owner argv/control pin required')
    for records,ctl in ((grandparent_records,grandparent_control),(parent_records,parent_control),(fifth_records,fifth_control),(new_records,control)):
        require(all(all(v.get(k)==ctl[k] for k in ('previous_resume_control','previous_resume_control_lock','environment_recheck'))
            for v in records.values()),'later tiers retain their real parent control and point observation')
    base=physical_owner.pointer(ROOT/'scripts/run_formal_suite16_owner_v753_r10.py')
    for records,parent_source in ((middle_records,None),(grandparent_records,'scripts/run_concurrent_resumed_formal_TEST_owner_v902_r10.py'),
            (parent_records,'scripts/run_composite_TEST_owner_v956_r10.py'),(fifth_records,'scripts/run_composite_TEST_owner_v1011_r10.py'),
            (new_records,'scripts/run_composite_TEST_owner_v1041_r10.py')):
        require(records['launch'].get('base_stage_owner')==base
            and (parent_source is None or records['launch'].get('parent_stage_owner')==physical_owner.pointer(ROOT/parent_source)),
            'base/parent owners are real source pointers, never launch pointers')
    for records,ctl,prior in ((parent_records,parent_control,grandparent_control),(fifth_records,fifth_control,parent_control),(new_records,control,fifth_control)):
        changed=ctl['resource_profile']!=prior['resource_profile']
        require(type(ctl.get('resource_profile_changed')) is bool and ctl['resource_profile_changed'] is changed
            and all(v.get('resource_profile_changed') is changed and v.get('previous_resource_profile')==prior['resource_profile']
                and v.get('previous_resource_qualification')==prior['resource_qualification'] for v in records.values()),
            'each quota transition or same-quota continuation remains explicit')
    capacity=('resource_capacity_envelope','resource_profile_within_capacity','resource_capacity_source','resource_qualifier')
    for records,ctl in ((fifth_records,fifth_control),(new_records,control)):
        require(ctl.get('resource_profile_within_capacity') is True
            and all(all(v.get(k)==ctl.get(k) for k in capacity) for v in records.values()),
            'actual quotas and qualified capacity remain separately disclosed')
    require(all(v.get('reboot_audit')==control.get('reboot_audit') and v.get('infrastructure_recovery') is True
        and v.get('boot_id')==new_records['launch'].get('boot_id') for v in new_records.values())
        and isinstance(new_records['launch'].get('boot_id'),str) and bool(new_records['launch']['boot_id']),
        'new physical owner keeps the exact reboot proof and its actual boot identity')
    require_final_owner(new_records,len(control['completed_job_ids']))
    require(new_records['terminal'].get('formal_release_requires_independent_audit') is True
        and new_records['status'].get('CPU_launched')==1920-len(control['completed_job_ids']),
        'new1056 full completion still requires independent audit')
    environment_audit.require_environment_receipt(audit_environment,control)
    return {runtime:value for runtime,value,_ in stages}


def require_qualification_inputs(qualification,control,preparation,batch,files,ledger):
    """Consume metadata only from the genuinely selected qualification source."""
    require(isinstance(qualification.get('inputs'),dict),'selected capacity evidence map required')
    for name,checksum in qualification['inputs'].items():
        require(preparation.qualification_path_allowed(name,capacity_source=control['resource_capacity_source']),
            'only selected-capacity qualification metadata admitted')
        ledger.frozen(batch,files,dict(path=name,sha256=checksum),parse=False)


def require_control(control,fifth_control,files,batch,ledger,control_pointer):
    """The reboot proof supplements all immutable1040 inputs; it is no terminal."""
    core=core_module()
    require(control.get('schema')==core.CONTROL_SCHEMA and control.get('split')=='test'
        and control.get('expected_cases')==1920 and control.get('source_count')==640
        and control.get('previous_runtime')==FIFTH
        and control.get('previous_controller_runtime')=='runtime/framework/ordered_composite_TEST_v1043'
        and control.get('user_authorized_resume') is True and control.get('standing_authorization_reused') is True
        and control.get('new_user_event_invented') is False and control.get('physical_policy_changed') is False
        and control.get('explicit_host_reboot_recovery') is True,
        'same-policy sixth-tier reboot recovery required')
    inherited=('suite','suite_lock','source_inputs','validation_release','policy_sha256','author_outcome_protocol',
        'execution_entry','execution_implementation','standing_continuation_record',*previous_audit.ENVIRONMENT_FIELDS)
    require(all(control.get(k)==fifth_control.get(k) for k in inherited),'original scientific/physical scope cannot change')
    require(type(control.get('resource_profile_changed')) is bool
        and control['resource_profile_changed'] is (control['resource_profile']!=fifth_control['resource_profile'])
        and control.get('previous_resource_profile')==fifth_control.get('resource_profile')
        and control.get('previous_resource_qualification')==fifth_control.get('resource_qualification'),
        'literal previous quota and capacity evidence required')
    current_owner.require_capacity(control)
    require(control.get('resume_owner')==physical_owner.pointer(ROOT/OWNER)
        and control.get('resume_auditor')==control.get('result_auditor')==physical_owner.pointer(ROOT/SOURCE)
        and control.get('resume_controller')==physical_owner.pointer(ROOT/CONTROLLER),
        'exact1056/1057/1058 source pointers required')
    require(control.get('previous_resume_control',{}).get('path')==core.PREVIOUS_CONTROL
        and control.get('previous_resume_control_lock',{}).get('path')==core.PREVIOUS_LOCK,
        'literal immutable1040 control/input paths required')
    required=('resume_owner','resume_auditor','result_auditor','resume_controller','suite','suite_lock','source_inputs',
        'validation_release','resource_qualification','resource_qualifier','standing_continuation_record',
        'concurrency_authorization_record','previous_resume_control','previous_resume_control_lock','environment_recheck',
        'previous_launch','previous_plan','previous_status','previous_controller_launch','previous_controller_status',
        'reboot_audit','host_reboot_observation','original_launch','original_plan','original_status','original_terminal',*previous_audit.ENVIRONMENT_FIELDS)
    for key in required:
        p=control.get(key);require(physical_owner.valid_pointer(p),'exact frozen control pointer required: '+key)
        ledger.frozen(batch,files,p,parse=False)
    require(not any(key in control for key in ('previous_terminal','previous_controller_failure','previous_DRAIN')),
        'reboot loss must never acquire invented terminal, failure or DRAIN pointers')
    ledger.frozen(batch,files,control_pointer,parse=False)
    for name in SOURCES:ledger.frozen(batch,files,physical_owner.pointer(ROOT/name),parse=False)
    for key,leaf in (('launch','launch.json'),('plan','plan.json'),('status','status.json')):
        require(control['previous_'+key]['path']==FIFTH+'/'+leaf
            and control['original_'+key]==fifth_control['original_'+key],
            'frozen original753 and partial1041 paths remain exact')
    require(control['original_terminal']==fifth_control['original_terminal'],'original753 terminal pointer cannot change')
    for key,leaf in (('launch','launch.json'),('status','status.json')):
        require(control['previous_controller_'+key]['path']=='runtime/framework/ordered_composite_TEST_v1043/'+leaf,
            'old1043 interrupted controller metadata must remain literal')
    reboot=ledger.json(control['reboot_audit']['path'])
    require(control['reboot_audit']['path']=='runtime/framework/reboot_TEST_evidence_v1054.json'
        and reboot.get('schema')=='r10_readonly_host_reboot_TEST_evidence_observation_v1054'
        and reboot.get('audit_complete') is True and reboot.get('formal_full_TEST_complete') is False
        and reboot.get('old_outputs_modified') is False and reboot.get('missing_evidence_reconstructed') is False
        and reboot.get('unknown_counted_as_NativeFalse') is False and reboot.get('success_based_case_selection') is False
        and reboot.get('new_terminal_claimed_for_interrupted_owner') is False,
        'real1054 reboot evidence is only recoverability proof, never a full result or terminal')
    source=reboot.get('source');require(physical_owner.valid_pointer(source)
        and source['path']=='scripts/audit_reboot_TEST_evidence_v1054_r10.py','real1054 evidence source required')
    ledger.frozen(batch,files,source,parse=False)
    from scripts import prepare_composite_TEST_reboot_resume_v1055_r10 as recovery_preparation
    for runtime in (FIFTH,'runtime/framework/ordered_composite_TEST_v1043'):
        for leaf in ('terminal.json','failure.json','DRAIN','STOP'):
            name=runtime+'/'+leaf;observed=reboot.get('absent_evidence',{}).get(name,{})
            require(observed.get('path')==name and observed.get('exists') is False and not (ROOT/name).exists(),
                'the actual interrupted owner/controller must retain absent terminal/failure/DRAIN/STOP evidence')
    recovery_preparation.require_host_reboot_observation(ledger.json(control['host_reboot_observation']['path']),
        dict(launch=ledger.json(control['previous_launch']['path'])),reboot)
    observation=ledger.json(control['environment_recheck']['path']);core.require_environment_recheck(observation,control)
    ledger.frozen(batch,files,observation['source'],parse=False)
    environment_audit.require_environment_receipt(observation.get('actual_environment'),control)
    entry=physical_owner.validate_entry(control['execution_entry']);require(entry['kind']=='batched901','unchanged real901 entry required')
    from scripts import run_batched_formal_driver_v901_r10 as bootstrap
    for key in ('batch_control','batch_lock'):ledger.frozen(batch,files,entry[key],parse=False)
    small=ledger.json(entry['batch_lock']['path'])
    require(small.get('schema')==bootstrap.LOCK_SCHEMA and small.get('execution_enabled') is True,'actual enabled901 small lock required')
    sources={name:physical_owner.pointer(ROOT/name) for name in bootstrap.SOURCES}
    for p in sources.values():
        ledger.frozen(batch,files,p,parse=False);ledger.frozen(batch,small['files'],p,parse=False)
    require(control['execution_implementation']==physical_owner.execution_implementation(entry,sources),
        'original qualified physical implementation cannot change after reboot')
    from scripts import prepare_composite_TEST_resume_v1040_r10 as preparation
    qualification=ledger.json(control['resource_qualification']['path']);audit_pointer=qualification.get('actual_independent_audit')
    require(physical_owner.valid_pointer(audit_pointer)
        and audit_pointer['path'] in ('runtime/framework/concurrent_TRAIN_resource_audit_v1031/launch.json',
            'runtime/framework/concurrent_TRAIN_resource_audit_v1005/launch.json'),'real selected capacity audit launch required')
    audit_launch=ledger.frozen(batch,files,audit_pointer)
    preparation.require_resource_proof(qualification,control,audit_launch=audit_launch)
    require_qualification_inputs(qualification,control,preparation,batch,files,ledger)
    for key in ('control','lock'):ledger.frozen(batch,files,qualification[key],parse=False)
    standing=ledger.json(control['standing_continuation_record']['path'])
    authorization=ledger.json(control['concurrency_authorization_record']['path'])
    require(standing.get('continue_until_user_stop') is True and standing.get('user_stop_received') is False
        and authorization.get('user_authorized_concurrency_change') is True,'standing authorization is retained without inventing a new event')
    return sources


def require_historical_pointer(batch,files,fifth_files,parent_files,grandparent_files,old_files,p,runtime):
    """Every historical generation retains its complete declaration unchanged."""
    batch.locked_pointer(ROOT,files,p)
    if runtime in (ORIGINAL,INTERMEDIATE,GRANDPARENT,PREVIOUS):batch.locked_pointer(ROOT,fifth_files,p)
    if runtime in (ORIGINAL,INTERMEDIATE,GRANDPARENT):batch.locked_pointer(ROOT,parent_files,p)
    if runtime in (ORIGINAL,INTERMEDIATE):batch.locked_pointer(ROOT,grandparent_files,p)
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
        'actual clean complete1056 owner required before full TEST graph access')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime',default=RUNTIME);parser.add_argument('--out',default=OUTPUT)
    parser.add_argument('--resume-control',default=CONTROL);parser.add_argument('--resume-lock',default=LOCK)
    parser.add_argument('--resume-lock-sha256',required=True);args=parser.parse_args(argv)
    require(args.runtime==RUNTIME and args.out==OUTPUT and args.resume_control==CONTROL and args.resume_lock==LOCK,
        'fixed new1057 output/control/runtime namespace required')
    out=ROOT/OUTPUT;require(not out.exists(),'never overwrite prior composite TEST results')
    require_not_stopped();ledger=EvidenceLedger(ROOT)
    lock_pointer=ledger.pointer(LOCK)
    require(physical_owner.valid_pointer(lock_pointer) and lock_pointer['sha256']==args.resume_lock_sha256,
        'external exact1055 input-lock byte SHA required before following pointers')
    lock=ledger.json(LOCK);files=lock.get('files')
    require(lock.get('schema')==core_module().LOCK_SCHEMA and lock.get('approved_for_execution') is True
        and isinstance(files,dict),'approved1055 frozen source/control/evidence supplement required')
    control_pointer=ledger.pointer(CONTROL)
    matches=[files[n] for n in (CONTROL,str(ROOT/CONTROL)) if n in files]
    require(matches==[control_pointer['sha256']],'control must have one exact declaration before data/release reads')
    require(lock.get('control')==control_pointer,'actual1055 input lock must bind the exact parsed control bytes')
    control=ledger.json(CONTROL)
    # The genuine complete VAL gate precedes any verify_inputs touching TEST.
    release=physical_owner.require_complete_VAL_release(control,control['validation_release']['path'])
    new_records={k:ledger.json(RUNTIME+'/'+k+'.json') for k in ('launch','plan','status','terminal')}
    require_final_owner(new_records,len(control.get('completed_job_ids',[])))
    verify_inputs(ROOT,files);audit_environment=environment_audit.capture_environment(control)
    with LookupBatch(ROOT) as batch:
        fifth_control=ledger.frozen(batch,files,control['previous_resume_control'])
        fifth_lock=ledger.frozen(batch,files,control['previous_resume_control_lock']);fifth_files=fifth_lock['files']
        require(fifth_lock.get('schema')==fifth_audit.core_module().LOCK_SCHEMA and fifth_lock.get('approved_for_execution') is True,
            'literal immutable1040 input lock required')
        require(all(files.get(k)==v for k,v in fifth_files.items()),'every1040 input key/SHA must remain unchanged in1055')
        verify_inputs(ROOT,fifth_files)
        parent_control=ledger.frozen(batch,fifth_files,fifth_control['previous_resume_control'])
        parent_lock=ledger.frozen(batch,fifth_files,fifth_control['previous_resume_control_lock']);parent_files=parent_lock['files']
        require(parent_lock.get('schema')==parent_audit.core_module().LOCK_SCHEMA and parent_lock.get('approved_for_execution') is True,
            'actual immutable1010 input lock required')
        require(all(fifth_files.get(k)==v for k,v in parent_files.items()),'every1010 declared key/SHA stays unchanged in1040')
        verify_inputs(ROOT,parent_files)
        grandparent_control=ledger.frozen(batch,parent_files,parent_control['previous_resume_control'])
        grandparent_lock=ledger.frozen(batch,parent_files,parent_control['previous_resume_control_lock']);grandparent_files=grandparent_lock['files']
        require(grandparent_lock.get('schema')==grandparent_audit.core_module().LOCK_SCHEMA and grandparent_lock.get('approved_for_execution') is True,
            'actual immutable955 input lock required')
        require(all(parent_files.get(k)==v for k,v in grandparent_files.items()),'every955 declared key/SHA stays unchanged in1010')
        verify_inputs(ROOT,grandparent_files)
        old_control=ledger.frozen(batch,grandparent_files,grandparent_control['previous_resume_control'])
        old_lock=ledger.frozen(batch,grandparent_files,grandparent_control['previous_resume_control_lock']);old_files=old_lock['files']
        require(old_lock.get('schema')==physical_owner.LOCK_SCHEMA and old_lock.get('approved_for_execution') is True,
            'actual immutable915 input lock and frozen graph required')
        require(all(grandparent_files.get(k)==v for k,v in old_files.items()),'every915 declared key/SHA stays unchanged in955')
        verify_inputs(ROOT,old_files)
        old_sources=previous_audit.require_control(old_control,old_files,batch,ledger,
            grandparent_control['previous_resume_control'],grandparent_control['previous_resume_control_lock'])
        grandparent_sources=grandparent_audit.require_control(grandparent_control,old_control,grandparent_files,batch,ledger,parent_control['previous_resume_control'])
        parent_sources=parent_audit.require_control(parent_control,grandparent_control,parent_files,batch,ledger,fifth_control['previous_resume_control'])
        fifth_sources=fifth_audit.require_control(fifth_control,parent_control,fifth_files,batch,ledger,control['previous_resume_control'])
        batch_sources=require_control(control,fifth_control,files,batch,ledger,control_pointer)
        require(batch_sources==fifth_sources==parent_sources==grandparent_sources==old_sources,'original901/900 implementation stays identical across six tiers')
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
        middle_records={k:ledger.json(grandparent_control['previous_'+k]['path']) for k in ('launch','plan','status','terminal')}
        grandparent_records={k:ledger.json(parent_control['previous_'+k]['path']) for k in ('launch','plan','status','terminal')}
        parent_records={k:ledger.json(fifth_control['previous_'+k]['path']) for k in ('launch','plan','status','terminal')}
        fifth_records={k:ledger.json(control['previous_'+k]['path']) for k in ('launch','plan','status')}
        reboot_audit=ledger.json(control['reboot_audit']['path'])
        require(new_records['launch'].get('boot_id')==reboot_audit.get('current_boot_id'),
            'new1056 launch binds the actual recovery boot recorded by1054')
        counts=require_six_tier_plan(control,fifth_control,parent_control,grandparent_control,old_control,original_records,middle_records,grandparent_records,parent_records,fifth_records,reboot_audit,new_records['plan'])
        previous_audit.require_original_plan(original_records['plan'],control,suite,sources,proof,driver,guard)
        require(original_records['terminal'].get('author_outcome_protocol')==protocol
            and original_records['terminal'].get('execution_protocol_changed') is True,
            'original753 terminal protocol must stay original, even though it is an incomplete drain')
        stage_owners=require_bindings(control,fifth_control,parent_control,grandparent_control,old_control,original_records['launch'],middle_records,grandparent_records,parent_records,fifth_records,new_records,source_sha=source_sha,
            driver=driver,guard=guard,control_pointer=control_pointer,lock_pointer=lock_pointer,audit_environment=audit_environment)
        for launch in stage_owners.values():
            require(launch.get('source_manifest')==suite['source_manifests']['test'] and not physical_owner.alive(launch),
                'each actual physical owner must bind the source manifest and have exited')
        require(middle_records['launch'].get('batch_sources')==grandparent_records['launch'].get('batch_sources')==parent_records['launch'].get('batch_sources')==fifth_records['launch'].get('batch_sources')==new_records['launch'].get('batch_sources')==batch_sources,
            'old902 and new956 must disclose actual unchanged901 source declarations')
        for pointer in (control['previous_controller_launch'],fifth_control['previous_controller_launch'],parent_control['previous_controller_launch'],grandparent_control['previous_controller_launch'],old_control['previous_controller_launch']):
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
                    if name in old_ids:require_historical_pointer(batch,files,fifth_files,parent_files,grandparent_files,old_files,p,runtime)
            if name in old_ids:require_historical_pointer(batch,files,fifth_files,parent_files,grandparent_files,old_files,ledger.pointer(cpu_leaf/'CPU_preflight.json'),runtime)
            terminal=ledger.json(native_leaf/'terminal.json');record=status['completed'][name];require_clean_terminal(record)
            require(all(record.get(k)==v for k,v in terminal.items()),'new status must preserve the true native terminal')
            if name in old_ids:require(record==fifth_records['status']['completed'][name],'all adopted nativeTrue/False outcomes remain literal')
            case_path=ROOT/physical_owner.option(row['physical_argv'],'--out')/'case.json';driver_path=case_path.parent/'admission_driver_receipt.json'
            cp,dp=ledger.pointer(case_path),ledger.pointer(driver_path)
            require(terminal.get('case')==record.get('case')==cp and terminal.get('admission_driver_receipt')==record.get('admission_driver_receipt')==dp,
                'each terminal/status must bind its actual case and independent751 receipt')
            if name in old_ids:
                require_historical_pointer(batch,files,fifth_files,parent_files,grandparent_files,old_files,cp,runtime);require_historical_pointer(batch,files,fifth_files,parent_files,grandparent_files,old_files,dp,runtime)
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
                if name in old_ids:require_historical_pointer(batch,files,fifth_files,parent_files,grandparent_files,old_files,bp,runtime)
                batch_pointers.append(bp)
            else:require('batch_driver_receipt' not in terminal and 'batch_driver_receipt' not in record,'original753 cannot invent901 evidence')
            cases.append(case);case_pointers.append(cp);driver_pointers.append(dp)
            tier_rows.append(dict(job_id=name,evidence_runtime=runtime,imported_completed=row['imported_completed'],
                resume_lineage_v1055=row['resume_lineage_v1055'],resume_lineage_v1040=row['resume_lineage_v1040'],resume_lineage_v1010=row['resume_lineage_v1010'],
                actual_stage_owner_pid=stage_owner['pid'],actual_stage_owner_start_ticks=stage_owner['start_ticks'],
                resume_lineage=row['resume_lineage'],previous_row_sha256=row['previous_row_sha256'],
                execution_entry=row['execution_entry'],actual_outer_argv=row['argv'],physical_argv=row['physical_argv'],
                case=cp,admission_driver_receipt=dp,batch_driver_receipt=bp))
        result=audit_complete_grid(cases,manifest,proof['routes'])
        require(result.get('complete') is True and result.get('source_cases')==640
            and result.get('method_cases')==result.get('completed_method_cases')==1920,'complete original16/640/1920 composite TEST cohort required')
        require(new_records['terminal'].get('native_successes')==sum(c['official_task_success'] for c in cases),'newowner success count must equal independent native cases')
        for frozen in (files,fifth_files,parent_files,grandparent_files,old_files,suite_files,source_meta['files']):verify_inputs(ROOT,frozen)
        inputs=ledger.close()
    owner_pointers={ORIGINAL:(control['original_launch'],control['original_terminal']),
        INTERMEDIATE:(grandparent_control['previous_launch'],grandparent_control['previous_terminal']),
        GRANDPARENT:(parent_control['previous_launch'],parent_control['previous_terminal']),
        PREVIOUS:(fifth_control['previous_launch'],fifth_control['previous_terminal']),
        FIFTH:(control['previous_launch'],None),
        RUNTIME:(inputs_pointer(inputs,RUNTIME+'/launch.json'),inputs_pointer(inputs,RUNTIME+'/terminal.json'))}
    result.update(schema=SCHEMA,created_utc=datetime.now(timezone.utc).isoformat(),per_task=summarize(cases),
        cases=case_pointers,admission_driver_receipts=driver_pointers,batch_driver_receipts=batch_pointers,evidence_tiers=tier_rows,
        inputs_sha256=inputs,resume_control=control_pointer,resume_control_lock=lock_pointer,
        base_result_auditor=physical_owner.pointer(ROOT/'scripts/audit_formal_suite16_results_v754_r10.py'),
        previous_composite_auditors=[physical_owner.pointer(ROOT/name) for name in
            ('scripts/audit_concurrent_resumed_formal_TEST_v906_r10.py','scripts/audit_composite_TEST_results_v957_r10.py','scripts/audit_composite_TEST_results_v1012_r10.py','scripts/audit_composite_TEST_results_v1042_r10.py')],
        actual_stage_owners=[dict(runtime=name,launch=owner_pointers[name][0],terminal=owner_pointers[name][1],
            pid=stage_owners[name]['pid'],start_ticks=stage_owners[name]['start_ticks'],completed=counts[name],
            resource_limits={k:stage_owners[name][k] for k in ('max_native_slots','max_GPU_slots','max_CPU_preflight_slots')})
            for name in (ORIGINAL,INTERMEDIATE,GRANDPARENT,PREVIOUS,FIFTH,RUNTIME)],source_input_lock=control['source_inputs'],validation_release=control['validation_release'],
        resource_profile=control['resource_profile'],resource_qualification=control['resource_qualification'],
        resource_qualifier=control['resource_qualifier'],
        resource_profile_changed=control['resource_profile_changed'],previous_resource_profile=control['previous_resource_profile'],
        resource_capacity_envelope=control['resource_capacity_envelope'],resource_profile_within_capacity=control['resource_profile_within_capacity'],
        resource_capacity_source=control['resource_capacity_source'],
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
            for runtime,ctl in ((INTERMEDIATE,grandparent_control),(GRANDPARENT,parent_control),(PREVIOUS,fifth_control))],
        host_reboot_interruption=dict(runtime=FIFTH,reboot_audit=control['reboot_audit'],
            saved_completed_job_ids=control['saved_completed_job_ids'],
            rejected_saved_completed_job_ids=control['rejected_saved_completed_job_ids'],
            unknown_interrupted_cases=reboot_audit['unknown_interrupted_cases'],
            unknown_counted_as_native_task_failures=False,previous_terminal=None,previous_terminal_invented=False,
            damaged_outputs_retained=True,all_remaining_keys_executed_once_in_new_namespace=True),
        Value_NGFF_FASTER_used=False,automatic_retry=False,historical_completed_cases_retained=len(control['completed_job_ids']),adopted_completed_results_not_rerun=True,
        saved_registered_but_damaged_cases_rerun=len(control['rejected_saved_completed_job_ids']),independent_composite_audit=True,
        reboot_audit=control['reboot_audit'],host_reboot_observation=control['host_reboot_observation'],infrastructure_recovery=True,
        old907_or958_or1013_or1043_successful_terminal_claimed=False,original754_906_957_1012_or1042_complete_result_claimed=False)
    require_not_stopped();atomic_json(out,result)
    print(json.dumps(dict(complete=True,schema=SCHEMA,split='test',cases=1920,
        native_successes_by_method=result['native_successes_by_method'],motion_is_diagnostic_only=True,out=OUTPUT)),flush=True)
    return result


def inputs_pointer(inputs,name):return dict(path=name,sha256=inputs[str(ROOT/name)])


if __name__=='__main__':main()
