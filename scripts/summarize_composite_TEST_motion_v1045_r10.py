"""Read-only saved motion after the actual complete1043/1041/1042 TEST chain.

The frozen838 reducer and920 weighting/definitions are reused without another
joint differencing pass. Five actual753/902/956/1011/1041 evidence tiers remain distinct.
No admission, model, Native/GPU job, release, fitting or observer is started.
Development must use source code and CPU/temp fakes, never partial live data.
"""
import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime,timezone
import hashlib
import json
import math
from pathlib import Path,PurePosixPath
import types

ROOT=Path(__file__).resolve().parents[1]
SOURCE='scripts/summarize_composite_TEST_motion_v1045_r10.py'
TEST_SOURCE='tests/test_composite_TEST_motion_v1045.py'
LEGACY_SOURCE='scripts/summarize_full_TEST_motion_v920_r10.py'
LEGACY_SHA='684c8c3f799b3177f0e8d328808d36ddd584f697bb6c37a7467ec4fbe79d463a'
CORE='our_mani/multitask/composite_TEST_resume_v1040.py'
CORE_SHA='e8ff7c4930b55147785d90e14d1b7ff6550c8ebcbc7431942dcd2ae49a5d44b4'
OWNER='scripts/run_composite_TEST_owner_v1041_r10.py'
OWNER_SHA='1da6d0b295403a258f187e0a35ee81fea552cfad8b74d640d38897ab62f53e9f'
AUDITOR='scripts/audit_composite_TEST_results_v1042_r10.py'
AUDITOR_SHA='b4f9443ac19891ba257ca14ca137c9350037777aa20e90b0af092af91f36dac2'
CONTROLLER_SOURCE='scripts/continue_composite_TEST_v1043_r10.py'
CONTROLLER_SHA='7d09d2e7350c1a41dca37e905cefe91398c066d8047de5a73a70a2541cb607c9'
PREPARER='scripts/prepare_composite_TEST_resume_v1040_r10.py'
PREPARER_SHA='31303d107062c029101a8a2511bc0d7157e20d1ae9d5f84973812ad5bc268d54'
ORIGINAL_OWNER_SHA='ece7f0d57f4e533b22a62d35a4b5220deedb88dbfc44767b060fc889b30cb183'
CHECK_SOURCE='scripts/check_frozen_TEST_environment_v954_r10.py'
PARENT_CORE='our_mani/multitask/composite_TEST_resume_v1010.py'
PARENT_CORE_SHA='19f917bb4675c62dfbd969d9d37c0cf64eaa108a8456914499677d0f08f08685'
PARENT_OWNER='scripts/run_composite_TEST_owner_v1011_r10.py'
PARENT_OWNER_SHA='a721cfd3b3b7611d3c57d233f416fdf9b8e2a13c72f41846df9d3ebd2eb838e0'
PREVIOUS_ADAPTER='scripts/summarize_composite_TEST_motion_v1015_r10.py'
PREVIOUS_ADAPTER_SHA='860c2c639154e35386c13ec2cbc9bbbef0d66463c0fb3834ff5ccc577f55b9f5'


GRANDPARENT_CORE='our_mani/multitask/composite_TEST_resume_v955.py'
GRANDPARENT_CORE_SHA='e636ea89d27bb497bf716062cc6fa2ccdcbee5811e9aacd50c9e29e82dc431bc'
GRANDPARENT_OWNER='scripts/run_composite_TEST_owner_v956_r10.py'
GRANDPARENT_OWNER_SHA='03159361511fdae6eb3591d03afb6fabd9199fbfdf7b60dd65ac17e7e945d83d'

def load_source(name,expected):
    raw=(ROOT/name).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=expected:raise ValueError('reviewed saved-motion source SHA changed: '+name)
    module=types.ModuleType('composite_motion_'+Path(name).stem);module.__file__=str(ROOT/name)
    exec(compile(raw,str(ROOT/name),'exec'),module.__dict__);return module


LEGACY=load_source(LEGACY_SOURCE,LEGACY_SHA);BASE=LEGACY.BASE;H=LEGACY.H;P=LEGACY.P
C=load_source(CORE,CORE_SHA);PC=load_source(PARENT_CORE,PARENT_CORE_SHA);GC=load_source(GRANDPARENT_CORE,GRANDPARENT_CORE_SHA)
require=BASE.require;digest=BASE.digest;sha_bytes=BASE.sha_bytes;valid_sha=BASE.valid_sha;option=BASE.option
METHODS=BASE.METHODS;TASKS=BASE.TASKS;POLICY=BASE.POLICY;PROFILE=BASE.PROFILE;PARAMETERS=BASE.PARAMETERS
REDUCER=BASE.REDUCER;SUBSET=BASE.SUBSET
SETUP=C.SETUP;CONTROL=C.CONTROL;LOCK=C.LOCK;RUNTIME=C.RUNTIME;CONTROLLER='runtime/framework/ordered_composite_TEST_v1043'
ORIGINAL=C.ORIGINAL;MIDDLE=C.INTERMEDIATE;PARENT=C.PREVIOUS;OLD_CONTROLLER=C.PREVIOUS_CONTROLLER
PARENT_CONTROL=C.PREVIOUS_CONTROL;PARENT_LOCK=C.PREVIOUS_LOCK
GRANDPARENT=C.OLDER;GRANDPARENT_CONTROL=PC.PREVIOUS_CONTROL;GRANDPARENT_LOCK=PC.PREVIOUS_LOCK;OLDER_CONTROLLER=PC.PREVIOUS_CONTROLLER
MIDDLE_CONTROL=LEGACY.CONTROL;MIDDLE_LOCK=LEGACY.LOCK
ART='artifacts/framework/multitask_r10/formal_TEST_composite_resume_v1041'
RESULT=ART+'/TEST_results.json';OUTPUT=RUNTIME+'/full_sixteen_task_motion_summary_v1045.json'
OWNER_STAGE=CONTROLLER+'/stages/formal_TEST_1920';AUDIT_STAGE=CONTROLLER+'/stages/independent_TEST_results'
SCHEMA='r10_full1920_five_tier_TEST_saved_motion_summary_v1045'
RESULT_SCHEMA='r10_five_tier_composite_formal_TEST_results_v1042'
REPORTING_DEPENDENCIES=dict(LEGACY.REPORTING_DEPENDENCIES)
PINS={**LEGACY.PINS,LEGACY_SOURCE:LEGACY_SHA,CORE:CORE_SHA,OWNER:OWNER_SHA,AUDITOR:AUDITOR_SHA,
    CONTROLLER_SOURCE:CONTROLLER_SHA,PREPARER:PREPARER_SHA,BASE.OWNER:ORIGINAL_OWNER_SHA,
    PARENT_CORE:PARENT_CORE_SHA,PARENT_OWNER:PARENT_OWNER_SHA,PREVIOUS_ADAPTER:PREVIOUS_ADAPTER_SHA,
    GRANDPARENT_CORE:GRANDPARENT_CORE_SHA,GRANDPARENT_OWNER:GRANDPARENT_OWNER_SHA}
SOURCES={*LEGACY.SOURCES,LEGACY_SOURCE,CORE,OWNER,AUDITOR,CONTROLLER_SOURCE,PREPARER,CHECK_SOURCE,PARENT_CORE,PARENT_OWNER,PREVIOUS_ADAPTER,GRANDPARENT_CORE,GRANDPARENT_OWNER}


def allowed(name):
    if name in {*PINS,*SOURCES,CONTROL,LOCK,PARENT_CONTROL,PARENT_LOCK,GRANDPARENT_CONTROL,GRANDPARENT_LOCK,MIDDLE_CONTROL,MIDDLE_LOCK,RESULT,SUBSET,
            BASE.SUITE,BASE.SUITE_LOCK,BASE.SOURCE_LOCK,BASE.MANIFEST,BASE.ORIGINAL_MANIFEST,BASE.ROUTES,BASE.VAL_RELEASE,
            LEGACY.BATCH_CONTROL,LEGACY.BATCH_LOCK,'runtime/framework/frozen_TEST_environment_recheck_v954.json'}:return True
    for runtime in (CONTROLLER,RUNTIME,ORIGINAL,MIDDLE,GRANDPARENT,PARENT,OLD_CONTROLLER,OLDER_CONTROLLER,LEGACY.CONTROLLER,BASE.CONTROLLER):
        if name in {runtime+'/'+n for n in ('launch.json','plan.json','status.json','terminal.json','failure.json','DRAIN')}:return True
    for stage in (OWNER_STAGE,AUDIT_STAGE):
        if name in {stage+'/'+n for n in ('launch.json','terminal.json','stdout.log')}:return True
    for runtime in (RUNTIME,ORIGINAL,MIDDLE,GRANDPARENT,PARENT):
        parts=name[len(runtime)+1:].split('/') if name.startswith(runtime+'/') else []
        if len(parts)==3 and parts[0] in ('CPU','native') and parts[2] in ('launch.json','terminal.json','CPU_preflight.json','stdout.log','stderr.log'):return True
    for out in (ART+'/test','artifacts/framework/multitask_r10/formal_TEST_composite_resume_v1011/test','artifacts/framework/multitask_r10/formal_TEST_composite_resume_v956/test',LEGACY.ART+'/test',BASE.ART+'/test'):
        parts=name[len(out)+1:].split('/') if name.startswith(out+'/') else []
        if len(parts)==4 and parts[0] in TASKS and parts[2] in METHODS and parts[3] in ('case.json','admission_driver_receipt.json','batch_driver_receipt.json'):return True
    return False


class Reader(BASE.Reader):
    """Read only explicit metadata/cases, not a transitive input/model graph."""
    def __init__(self,root):
        super().__init__(Path(root).resolve());self.resumed={};self.signatures={}
        self.inputs.update({LEGACY_SOURCE:LEGACY_SHA,CORE:CORE_SHA})
    def raw(self,name,*,expected=None,evidence=False,frozen=False,original=False,resumed=False):
        require(isinstance(name,str) and not Path(name).is_absolute() and str(PurePosixPath(name))==name
            and '..' not in PurePosixPath(name).parts and allowed(name),'non-evidence input refused: '+str(name))
        path=(self.root/name).resolve(strict=True)
        require(path.is_relative_to(self.root) and str(path.relative_to(self.root))==name,'canonical evidence path required')
        signature=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
        before=signature(path.stat());value=path.read_bytes();require(before==signature(path.stat()),'input changed while reading: '+name)
        actual=sha_bytes(value)
        wanted=[v for v in (expected,PINS.get(name),self.lookup(self.evidence,name) if evidence else None,
            self.lookup(self.frozen,name) if frozen else None,self.lookup(self.original,name) if original else None,
            self.lookup(self.resumed,name) if resumed else None) if v is not None]
        require(all(valid_sha(v) and actual==v for v in wanted),'bound saved-motion input SHA changed: '+name)
        require(name not in self.inputs or self.inputs[name]==actual,'saved-motion input changed during collection')
        self.inputs[name]=actual;self.signatures[name]=before;return value
    close=LEGACY.Reader.close


pointer=LEGACY.pointer;pid=LEGACY.pid;environment=LEGACY.environment;clean=LEGACY.clean
recorded_motion_flag=LEGACY.recorded_motion_flag


def verify_reporting_dependencies(reader):
    records=[]
    for name,expected in REPORTING_DEPENDENCIES.items():
        membership={}
        for key in ('original','frozen'):
            mapping=getattr(reader,key);values=[mapping[k] for k in (name,str(reader.root/name)) if k in mapping]
            require(all(valid_sha(v) and v==expected for v in values),'old declaration conflicts with reporting helper')
            membership[key]=bool(values)
        reader.raw(name,expected=expected,resumed=True,evidence=True)
        records.append(dict(path=name,sha256=expected,scope='supplementary_readonly_reporting_dependency',
            original_physical_input_lock_member=membership['original'],original_TEST_suite_input_lock_member=membership['frozen'],
            bound_by_actual1040_supplement_and1042_evidence=True,independent_reporting_source_pin=True,
            old_input_lock_amended=False,proof_of_original_physical_freeze_membership_claimed=False))
    reader.reporting_dependencies=records


def require_stage(value,end,argv,source_sha,controller):
    pid(value);clean(end,stage=True)
    require(value.get('argv')==argv and value.get('source_sha256')==source_sha and value.get('automatic_retry') is False
        and value.get('inherited_frozen_environment')==controller['frozen_environment']
        and value.get('current_mount_namespace')==controller['frozen_environment']['current_mount_namespace'],
        'actual958 stage argv/source/PID/environment required')


def require_capacity_metadata(control):
    """Recorded actual quotas and capacity are distinct; no live resource probe."""
    fields=('max_native_slots','max_GPU_slots','max_CPU_preflight_slots','ready_limit')
    source=control.get('resource_capacity_source')
    if source=='qualified1031':schema='r10_frozen_concurrent_native_resource_profile_v1031';limits=(12,5,6,24)
    elif source=='qualified1005_fallback':schema='r10_frozen_concurrent_native_resource_profile_v1005';limits=(8,3,4,16)
    else:raise ValueError('actual qualified1031 or real1005 capacity source required')
    capacity=dict(schema=schema,**dict(zip(fields,limits)));actual=control.get('resource_profile')
    require(control.get('resource_capacity_envelope')==capacity and control.get('resource_profile_within_capacity') is True
        and isinstance(actual,dict) and set(actual)=={'schema',*fields} and actual['schema']==schema
        and all(type(actual.get(k)) is int and 1<=actual[k]<=capacity[k] for k in fields)
        and actual['max_GPU_slots']<=actual['max_native_slots']
        and (source!='qualified1005_fallback' or actual==capacity)
        and type(control.get('resource_profile_changed')) is bool
        and control['resource_profile_changed'] is (actual!=control.get('previous_resource_profile')),
        'actual formal quotas must fit the selected genuine capacity; same-quota fallback is unchanged')
    return actual


def require_owner_completion(launch,plan,status,terminal,stage,control,cp,lp):
    common=dict(split='test',policy_sha256=POLICY,author_outcome_protocol=control['author_outcome_protocol'],execution_protocol_changed=True,
        resume_control=cp,resume_control_lock=lp,resource_profile=control['resource_profile'],resource_qualification=control['resource_qualification'],
        execution_implementation=control['execution_implementation'],automatic_retry=False)
    require(all(all(v.get(k)==x for k,x in common.items()) for v in (launch,plan,status,terminal)),'immutable1011 owner bindings required')
    count=len(control['completed_job_ids'])
    require(pid(launch)==pid(stage) and launch.get('source_sha256')==control['resume_owner']['sha256']
        and launch.get('argv')==stage['argv'][3:] and launch.get('execution_entry')==control['execution_entry']
        and launch.get('imported_completed_cases')==plan.get('imported_completed_cases')==count,'actual1011 invocation/import count required')
    require(terminal.get('complete') is True and terminal.get('expected')==terminal.get('completed')==1920
        and terminal.get('failed')==0 and terminal.get('owner_error') is None
        and terminal.get('stop_requested') is False and terminal.get('drain_requested') is False and terminal.get('drained') is False
        and terminal.get('CPU_ready_unexecuted')==[] and terminal.get('CPU_not_launched')==0
        and terminal.get('imported_completed_cases')==count and terminal.get('resumed_new_cases')==1920-count
        and terminal.get('formal_release_requires_independent_audit') is True,'clean full1011 terminal required before TEST source/case reads')
    require(status.get('expected')==1920 and len(status.get('completed',{}))==1920 and status.get('failed')=={}
        and status.get('active')==status.get('checking')==status.get('ready_jobs')==[]
        and status.get('CPU_ready')==status.get('native_slots')==status.get('GPU_slots')==0
        and status.get('CPU_launched')==1920-count and status.get('imported_completed')==count
        and status.get('owner_error') is None and status.get('stop_requested') is False and status.get('drain_requested') is False,
        'clean unique1920 status required before TEST source/case reads')
    first=environment(launch.get('frozen_environment',{}),control,launch,expected_mount=stage['current_mount_namespace'])
    require(all(v.get('frozen_environment')==first for v in (plan,status,terminal)),'actual1011 namespace stays its own recorded proof')
    expected_profile=require_capacity_metadata(control)
    require(control.get('resource_profile')==expected_profile
        and all(launch.get(k)==expected_profile[k] for k in ('max_native_slots','max_GPU_slots','max_CPU_preflight_slots')),
        'actual frozen launch quotas must match separately qualified capacity')
    for value in (launch,plan,status,terminal):
        require(all(value.get(k)==control.get(k) for k in ('previous_resume_control','previous_resume_control_lock','environment_recheck','resource_profile_changed','previous_resource_profile','previous_resource_qualification',
                'resource_capacity_envelope','resource_profile_within_capacity','resource_capacity_source','resource_qualifier')),
            'actual956 parent control and environment observation must remain explicit')
    require(launch.get('base_stage_owner')==dict(path=BASE.OWNER,sha256=ORIGINAL_OWNER_SHA)
        and launch.get('parent_stage_owner')==dict(path=PARENT_OWNER,sha256=PARENT_OWNER_SHA),
        'actual956 original753/parent902 source pointers required')


def require_result(result,control,cp,lp,audit_stage):
    BASE.require_result_header(result)
    require(result.get('schema')==RESULT_SCHEMA and result.get('source_sha256')==control['resume_auditor']['sha256']
        and result.get('resume_control')==cp and result.get('resume_control_lock')==lp
        and result.get('source_input_lock')==control['source_inputs'] and result.get('validation_release')==control['validation_release']
        and result.get('author_outcome_protocol')==control['author_outcome_protocol']
        and result.get('resource_profile')==control['resource_profile'] and result.get('resource_qualification')==control['resource_qualification']
        and result.get('resource_qualifier')==control['resource_qualifier']
        and all(result.get(k)==control.get(k) for k in ('resource_capacity_envelope','resource_profile_within_capacity','resource_capacity_source'))
        and result.get('execution_implementation')==control['execution_implementation']
        and result.get('environment_recheck')==control['environment_recheck']
        and type(control.get('resource_profile_changed')) is bool and result.get('resource_profile_changed') is control['resource_profile_changed'] and result.get('previous_resource_profile')==control['previous_resource_profile']
        and result.get('previous_resource_qualification')==control['previous_resource_qualification'],'actual independent957 result bindings required')
    for key,value in dict(used_evidence_rechecked_at_close=True,old_completed_results_not_rerun=True,independent_composite_audit=True,
            motion_unknown_retained=True,zero_tick_cases_retained=True,old907_or958_or1013_successful_terminal_claimed=False,
            original754_906_957_or1012_complete_result_claimed=False,automatic_retry=False).items():
        require(result.get(key) is value,'actual three-tier957 evidence required: '+key)
    invocation=result.get('auditor_invocation',{})
    require(pid(invocation)==pid(audit_stage) and invocation.get('argv')==audit_stage['argv'][3:]
        and invocation.get('source')==control['resume_auditor'] and invocation.get('inherited_physical_stage_identity_claimed') is False,
        'actual957 independent auditor PID/start/argv required')
    environment(result.get('frozen_environment',{}),control,invocation,expected_mount=audit_stage['current_mount_namespace'])


def require_saved_case(case,row,source_sha,protocol,source_lock_sha):
    """Original851 identity/clock/RML checks with explicit unknown diagnostic."""
    identity=case.get('identity',{});raw=row['raw'];task=row['task'];method=row['method']
    require(case.get('complete') is True and case.get('cleanup_completed') is True and case.get('infrastructure_error') is None
        and type(case.get('official_task_success')) is bool and type(case.get('native_terminal_failure')) is bool,'clean saved native outcome required')
    require(case.get('split')==identity.get('split')=='test' and case.get('task')==identity.get('task')==task
        and case.get('method')==identity.get('method')==method and case.get('source_raw_episode_id')==identity.get('source_raw_episode_id')==raw
        and case.get('parent_raw_episode_id') is None and identity.get('parent_raw_episode_id') is None
        and identity.get('raw_episode_id')==raw and identity.get('replica')==0 and identity.get('source_row_sha256')==source_sha
        and identity.get('config_sha256')==row['configuration_sha256'] and identity.get('frozen_inputs_sha256')==row['input_lock_sha256']
        and identity.get('source_input_lock_sha256')==source_lock_sha,'actual original TEST source/CPU identity required')
    for key,value in dict(demo_only=False,development_demo_only=False,formal_validation_metric=False,formal_TEST_metric=True,
            formal_test_used=True,TEST_used_for_fitting=False,Value_NGFF_FASTER_used=False).items():
        require(case.get(key) is value,'formal saved TEST case required: '+key)
    domain=case.get('component_domain',{});frozen=case.get('full16_frozen_execution',{})
    require(domain.get('purpose')=='evaluation' and domain.get('split')=='test' and domain.get('task')==task
        and domain.get('method')==method and domain.get('raw_episode_id')==raw and case.get('author_outcome_protocol')==protocol
        and frozen.get('policy_sha256')==POLICY and frozen.get('source_row_sha256')==source_sha and frozen.get('actual_split')=='test'
        and frozen.get('VAL_TEST_relabelled_as_TRAIN') is False,'saved formal evaluation/domain/protocol required')
    rml=case.get('common_author_RML_parameters',{});motion=case.get('all_stage_counted_joint_motion',{})
    require(rml.get('complete') is True and rml.get('native_parameters_mutated') is False and rml.get('actual_motion_bound_claimed') is False
        and rml.get('profile_sha256')==PROFILE and rml.get('actual_parameters')==PARAMETERS
        and 'all_stage_actual_window_checks_passed' in motion
        and (motion['all_stage_actual_window_checks_passed'] is None or type(motion['all_stage_actual_window_checks_passed']) is bool)
        and motion.get('actual_window_acceleration_criterion_rad_s2')==6.
        and motion.get('planning_metadata_used_as_actual_acceleration_cap') is False and motion.get('policy_or_physics_modified') is False,
        'same RML and bool-or-unknown diagnostic metadata required; no fabricated zero/pass')
    clock=case.get('counted_clock',{});alignment=case.get('alignment_receipt',{});native=alignment.get('clock',{});ticks=clock.get('ticks')
    require(type(ticks) is int and 0<=ticks<=200 and clock.get('dt')==.05 and type(clock.get('seconds')) in (int,float)
        and math.isfinite(clock['seconds']) and abs(clock['seconds']-ticks*.05)<=1e-9 and native.get('ticks')==ticks
        and native.get('dt')==.05 and native.get('budget_seconds')==10.
        and abs(native.get('simulation_seconds',float('inf'))-clock['seconds'])<=1e-9
        and sum(native.get('phases',{}).values())==ticks and isinstance(motion.get('events'),list) and len(motion['events'])==ticks,
        'saved physical clock/window coverage must remain unchanged')
    for key in ('actual_physics_seconds','actual_task_seconds'):
        value=alignment.get(key);require(type(value) in (int,float) and math.isfinite(value) and abs(value-clock['seconds'])<=1e-5,
            'saved native clock mismatch: '+key)


def require_released_cohort(plan,result,status,stage_owners,old_ids):
    """Metadata-only1920 pointer/tier gate, before manifest or case bytes."""
    rows=plan.get('jobs');tiers=result.get('evidence_tiers');cases=result.get('cases');drivers=result.get('admission_driver_receipts')
    require(all(isinstance(v,list) and len(v)==1920 for v in (rows,tiers,cases,drivers)),
        'complete1920 ordered plan/case/driver/tier pointer grid required first')
    seen=set();batches=[]
    for row,tier,cp,dp in zip(rows,tiers,cases,drivers):
        name=row['job_id'];runtime=row['evidence_runtime'];owner=stage_owners.get(runtime);imported=name in old_ids
        require(name not in seen and owner is not None and row['imported_completed'] is imported,'actual unique metadata tiers required');seen.add(name)
        out=option(row['physical_argv'],'--out');pointer(cp,out+'/case.json');pointer(dp,out+'/admission_driver_receipt.json')
        record=status['completed'].get(name,{});clean(record)
        require(type(record.get('native_success')) is bool and 'motion_pass' in record
            and (record['motion_pass'] is None or type(record['motion_pass']) is bool)
            and record.get('case')==cp and record.get('admission_driver_receipt')==dp,'actual status must bind explicit native outcome and saved pointers')
        bp=None
        if runtime!=ORIGINAL:
            bp=pointer(record.get('batch_driver_receipt'),out+'/batch_driver_receipt.json');batches.append(bp)
        else:require('batch_driver_receipt' not in record,'original753 cannot acquire901 metadata')
        expected=dict(job_id=name,evidence_runtime=runtime,imported_completed=imported,actual_stage_owner_pid=owner['pid'],
            actual_stage_owner_start_ticks=owner['start_ticks'],resume_lineage=row['resume_lineage'],resume_lineage_v1010=row['resume_lineage_v1010'],resume_lineage_v1040=row['resume_lineage_v1040'],previous_row_sha256=row['previous_row_sha256'],
            execution_entry=row['execution_entry'],actual_outer_argv=row['argv'],physical_argv=row['physical_argv'],case=cp,
            admission_driver_receipt=dp,batch_driver_receipt=bp)
        require(tier==expected,'released957 tier must match actual ordered case/status/owner metadata')
    require(set(status['completed'])==seen and result.get('batch_driver_receipts')==batches,
        'all1920 records and actual three-tier901 sidecars must be complete before source/case reads')


def collect(reader,result_sha,resume_control=CONTROL,resume_lock=LOCK,resume_lock_sha=None):
    require(resume_control==CONTROL and resume_lock==LOCK and valid_sha(result_sha) and valid_sha(resume_lock_sha),
        'actual955 namespace and external final957-result/955-lock SHAs required')
    locked=reader.json(LOCK,expected=resume_lock_sha)
    require(locked.get('schema')==C.LOCK_SCHEMA and locked.get('approved_for_execution') is True
        and isinstance(locked.get('files'),dict),'actual955 frozen continuation lock required')
    reader.resumed=locked['files'];control=reader.json(CONTROL,resumed=True);cp,lp=reader.pointer(CONTROL),reader.pointer(LOCK)
    require(locked.get('control')==cp and control.get('schema')==C.CONTROL_SCHEMA and control.get('split')=='test'
        and control.get('expected_cases')==1920 and control.get('source_count')==640 and control.get('policy_sha256')==POLICY
        and control.get('physical_policy_changed') is False,'actual same-policy full1920 composite control required')
    for key,name in (('resume_controller',CONTROLLER_SOURCE),('resume_owner',OWNER),('resume_auditor',AUDITOR)):
        p=pointer(control.get(key),name);reader.raw(name,expected=p['sha256'],resumed=True)
    reader.raw(CORE,expected=CORE_SHA,resumed=True)
    controller=reader.json(CONTROLLER+'/launch.json');final=reader.json(CONTROLLER+'/terminal.json')
    require(controller.get('source_sha256')==control['resume_controller']['sha256'] and controller.get('resume_control')==cp
        and controller.get('resume_control_lock')==lp and controller.get('automatic_retry') is False,'actual958 launch/source/control required')
    environment(controller.get('frozen_environment',{}),control,controller)
    require(final.get('complete') is True and final.get('automatic_retry') is False
        and pointer(final.get('TEST_results'),RESULT)['sha256']==result_sha,'complete958 final result pointer required before saved TEST data')
    state=reader.json(CONTROLLER+'/status.json')
    require(state.get('stage')=='complete' and state.get('automatic_retry') is False
        and state.get('frozen_environment')==controller['frozen_environment'],'actual958 complete status required')
    owner_args=[]
    for key,flag in (('suite','--suite'),('suite_lock','--suite-lock'),('source_inputs','--source-inputs'),('validation_release','--validation-release')):
        owner_args.extend((flag,pointer(control[key])['path']))
    owner_args+=['--runtime',RUNTIME,'--out',ART+'/test','--resume-control',CONTROL,'--resume-lock',LOCK,'--resume-lock-sha256',resume_lock_sha]
    audit_args=['--resume-control',CONTROL,'--resume-lock',LOCK,'--resume-lock-sha256',resume_lock_sha]
    stages={}
    for stage,source,args in ((OWNER_STAGE,OWNER,owner_args),(AUDIT_STAGE,AUDITOR,audit_args)):
        launch,end=reader.json(stage+'/launch.json'),reader.json(stage+'/terminal.json')
        require_stage(launch,end,[str(reader.root/'.conda-env/bin/python'),'-B','-u',source,*args],reader.inputs[source],controller);stages[source]=launch
    launch,plan,status,terminal=(reader.json(RUNTIME+'/'+name+'.json') for name in ('launch','plan','status','terminal'))
    require_owner_completion(launch,plan,status,terminal,stages[OWNER],control,cp,lp)
    # First final-result read. No manifest, case, raw or model was opened above.
    result=reader.json(RESULT,expected=result_sha);require_result(result,control,cp,lp,stages[AUDITOR]);reader.evidence=result['inputs_sha256']
    reader.raw(PARENT_CORE,expected=PARENT_CORE_SHA,resumed=True,evidence=True)
    reader.raw(GRANDPARENT_CORE,expected=GRANDPARENT_CORE_SHA,resumed=True,evidence=True)
    # The earlier motion adapter is a separately pinned reporting dependency,
    # never a claimed member of an older physical or TEST input lock.
    reader.raw(PREVIOUS_ADAPTER,expected=PREVIOUS_ADAPTER_SHA)
    for name in (CONTROL,LOCK,CORE,OWNER,AUDITOR,*[RUNTIME+'/'+n+'.json' for n in ('launch','plan','status','terminal')]):
        require(reader.lookup(reader.evidence,name)==reader.inputs[name],'actual957 consumed evidence SHA differs: '+name)
    stdout=reader.json(AUDIT_STAGE+'/stdout.log')
    require(stdout.get('complete') is True and stdout.get('schema')==RESULT_SCHEMA and stdout.get('split')=='test'
        and stdout.get('cases')==1920 and stdout.get('out')==RESULT and stdout.get('motion_is_diagnostic_only') is True
        and stdout.get('native_successes_by_method')==result.get('native_successes_by_method'),'actual957 full stdout must match released result')
    parent_control=reader.pointed(pointer(control['previous_resume_control'],PARENT_CONTROL),resumed=True,evidence=True)
    parent_lock=reader.pointed(pointer(control['previous_resume_control_lock'],PARENT_LOCK),resumed=True,evidence=True)
    require(parent_lock.get('schema')==PC.LOCK_SCHEMA and parent_lock.get('approved_for_execution') is True,'actual immutable1010 parent lock required')
    grandparent_control=reader.pointed(pointer(parent_control['previous_resume_control'],GRANDPARENT_CONTROL),resumed=True,evidence=True)
    grandparent_lock=reader.pointed(pointer(parent_control['previous_resume_control_lock'],GRANDPARENT_LOCK),resumed=True,evidence=True)
    require(grandparent_lock.get('schema')==GC.LOCK_SCHEMA and grandparent_lock.get('approved_for_execution') is True,'actual immutable955 grandparent lock required')
    middle_control=reader.pointed(pointer(grandparent_control['previous_resume_control'],MIDDLE_CONTROL),resumed=True,evidence=True)
    middle_lock=reader.pointed(pointer(grandparent_control['previous_resume_control_lock'],MIDDLE_LOCK),resumed=True,evidence=True)
    require(middle_lock.get('schema')==P.LOCK_SCHEMA and middle_lock.get('approved_for_execution') is True,'actual immutable915 middle lock required')
    histories={}
    for runtime,ctl,prefix in ((ORIGINAL,control,'original_'),(MIDDLE,grandparent_control,'previous_'),(GRANDPARENT,parent_control,'previous_'),(PARENT,control,'previous_')):
        histories[runtime]={}
        for name in ('launch','plan','status','terminal'):
            p=pointer(ctl[prefix+name],runtime+'/'+name+'.json')
            if runtime==ORIGINAL:require(p==parent_control['original_'+name]==grandparent_control['original_'+name]==middle_control['previous_'+name],
                'all original753 pointers remain the same actual evidence')
            histories[runtime][name]=reader.pointed(p,resumed=True,evidence=True)
    original,middle,grandparent,parent=(histories[r] for r in (ORIGINAL,MIDDLE,GRANDPARENT,PARENT))
    P.require_partition(middle_control,original['plan'],original['status'],original['terminal'])
    require(middle['plan'].get('jobs')==P.resumed_jobs(original['plan'],middle_control,middle_control['execution_entry']),
        'actual753 to902 whole-row provenance cannot be flattened')
    GC.require_resume_plan(grandparent_control,middle['plan'],middle['status'],middle['terminal'],grandparent['plan'])
    PC.require_resume_plan(parent_control,grandparent['plan'],grandparent['status'],grandparent['terminal'],parent['plan'])
    C.require_resume_plan(control,parent['plan'],parent['status'],parent['terminal'],plan)
    require(control['execution_entry']==parent_control['execution_entry']==grandparent_control['execution_entry']==middle_control['execution_entry'],
        'unchanged original915 formal901 controls required')
    for ctl,runtime in ((control,OLD_CONTROLLER),(parent_control,OLDER_CONTROLLER),(grandparent_control,LEGACY.CONTROLLER)):
        for key,name in (('previous_controller_launch','launch.json'),('previous_controller_status','status.json'),('previous_controller_failure','failure.json')):
            value=reader.pointed(pointer(ctl[key],runtime+'/'+name),resumed=True,evidence=True)
            if name=='status.json':require(value.get('stage')=='requires_concrete_repair' and value.get('automatic_retry') is False,'real partial historical controller status required')
            if name=='failure.json':require(value.get('complete') is False and value.get('user_stop') is False and value.get('automatic_retry') is False,
                'partial historical failure cannot become invented full terminal')
    reader.pointed(pointer(control['previous_DRAIN'],PARENT+'/DRAIN'),resumed=True,evidence=True)
    recheck=reader.pointed(pointer(control['environment_recheck'],'runtime/framework/frozen_TEST_environment_recheck_v954.json'),resumed=True,evidence=True)
    C.require_environment_recheck(recheck,control)
    reader.raw(CHECK_SOURCE,expected=pointer(recheck['source'],CHECK_SOURCE)['sha256'],resumed=True,evidence=True)
    for runtime,records,source in ((ORIGINAL,original,BASE.OWNER),(MIDDLE,middle,LEGACY.OWNER),(GRANDPARENT,grandparent,GRANDPARENT_OWNER),(PARENT,parent,PARENT_OWNER)):
        reader.raw(source,expected=PINS[source],resumed=True,evidence=True)
        require(records['launch'].get('source_sha256')==reader.inputs[source],'actual historical owner source differs')
        if runtime!=ORIGINAL:environment(records['launch'].get('frozen_environment',{}),control,records['launch'])
    expected_counts=Counter(r['evidence_runtime'] for r in plan['jobs'])
    stage_owners={ORIGINAL:original['launch'],MIDDLE:middle['launch'],GRANDPARENT:grandparent['launch'],PARENT:parent['launch'],RUNTIME:launch}
    owners=result.get('actual_stage_owners');require(isinstance(owners,list) and len(owners)==5,'actual five physical owners required')
    expected_owners=[]
    for runtime in (ORIGINAL,MIDDLE,GRANDPARENT,PARENT,RUNTIME):
        owner=stage_owners[runtime]
        expected_owners.append(dict(runtime=runtime,launch=reader.pointer(runtime+'/launch.json'),terminal=reader.pointer(runtime+'/terminal.json'),
            pid=owner['pid'],start_ticks=owner['start_ticks'],completed=expected_counts[runtime],
            resource_limits={k:owner[k] for k in ('max_native_slots','max_GPU_slots','max_CPU_preflight_slots')}))
    require(owners==expected_owners and len({pid(v) for v in stage_owners.values()})==5,'five real owner sources and actual limits required')
    old_ids=set(control['completed_job_ids'])
    require(set(parent['status']['completed'])==old_ids
        and all(status['completed'].get(k)==v for k,v in parent['status']['completed'].items())
        and all(parent['status']['completed'].get(k)==v for k,v in grandparent['status']['completed'].items())
        and all(grandparent['status']['completed'].get(k)==v for k,v in middle['status']['completed'].items())
        and all(middle['status']['completed'].get(k)==v for k,v in original['status']['completed'].items()),
        'all historical nativeFalse/True and motionNone records remain literal')
    require(result.get('historical_infrastructure_attempts')==[dict(runtime=runtime,job_ids=ctl['failed_job_ids'],
        previous_status=ctl['previous_status'],previous_terminal=ctl['previous_terminal'],counted_as_native_task_failures=False,
        all_keys_reexecuted_once_in_new_namespace=True) for runtime,ctl in ((MIDDLE,grandparent_control),(GRANDPARENT,parent_control),(PARENT,control))],
        'three historical infrastructure tiers remain attempts, never native outcomes')
    require_released_cohort(plan,result,status,stage_owners,old_ids)
    # All full-chain and provenance gates above precede source manifest/cases.
    original_control=reader.json(BASE.CONTROL);original_lock=reader.json(BASE.LOCK);reader.original=original_lock['files']
    require(original_control.get('policy_sha256')==POLICY and original_lock.get('approved_for_execution') is True,'actual unchanged825 original graph required')
    suite=reader.pointed(pointer(control['suite'],BASE.SUITE),evidence=True)
    suite_lock=reader.pointed(pointer(control['suite_lock'],BASE.SUITE_LOCK),evidence=True);reader.frozen=suite_lock['files']
    require(suite_lock.get('approved_for_execution') is True and suite_lock.get('actual_formal_split')=='test'
        and suite_lock.get('policy_sha256')==POLICY and suite_lock.get('model_or_policy_parameters_changed') is False
        and reader.lookup(reader.frozen,BASE.SUITE)==reader.inputs[BASE.SUITE]
        and suite.get('validation_release_path')==BASE.VAL_RELEASE,'original unchanged826 TEST suite required')
    val=reader.pointed(pointer(control['validation_release'],BASE.VAL_RELEASE),evidence=True,frozen=True);H.require_release_header(val);valp=reader.pointer(BASE.VAL_RELEASE)
    require(suite_lock.get('original_945_VAL_release')==valp and val.get('source_sha256')==original_control['resume_auditor']['sha256']
        and val.get('resume_control')==reader.pointer(BASE.CONTROL) and val.get('resume_control_lock')==reader.pointer(BASE.LOCK),'real original824 completeVAL prerequisite required')
    protocol=control['author_outcome_protocol']
    require(protocol==suite['author_outcome_protocol']==original_control['author_outcome_protocol']
        and result.get('admission_driver')==suite.get('formal_case_driver') and result.get('admission_guard')==suite.get('formal_admission_guard')
        and result.get('base_result_auditor')==suite.get('formal_result_auditor')
        and pointer(suite['formal_result_auditor'],BASE.AUDITOR)['sha256']==reader.lookup(reader.frozen,BASE.AUDITOR),
        'original754 suite graph and actual957 independent result remain distinct')
    fixed=dict(split='test',policy_sha256=POLICY,author_outcome_protocol=protocol,execution_protocol_changed=True,
        admission_driver=result['admission_driver'],admission_guard=result['admission_guard'],native_runtime=protocol['runtime_consumer'],
        original_native_runtime=protocol['original_runtime'],automatic_retry=False)
    require(all(all(record.get(k)==v for k,v in fixed.items()) for record in
        (original['launch'],middle['launch'],middle['plan'],middle['status'],middle['terminal'],
         grandparent['launch'],grandparent['plan'],grandparent['status'],grandparent['terminal'],
         parent['launch'],parent['plan'],parent['status'],parent['terminal'],launch,plan,status,terminal)),
        'three real owners must preserve original policy/admission/runtime bindings')
    manifest=reader.pointed(suite['source_manifests']['test'],frozen=True)
    require(suite['source_manifests']['test']['path']==BASE.MANIFEST and manifest.get('original_source',{}).get('path')==BASE.ORIGINAL_MANIFEST,'original TEST source scope required')
    original_manifest=reader.pointed(manifest['original_source'],frozen=True)
    source_lock=reader.pointed(pointer(control['source_inputs'],BASE.SOURCE_LOCK),evidence=True)
    reader.raw(BASE.SOURCE_PREPARER,frozen=True,original=True,resumed=True)
    require(source_lock.get('source_manifest_sha256')==reader.inputs[BASE.MANIFEST],'actual756 source manifest SHA required')
    sources=BASE.source_grid(manifest,original_manifest,source_lock,valp,reader.inputs[BASE.SOURCE_PREPARER])
    routes=reader.pointed(suite['routes'],frozen=True,original=True);BASE.require_plan(original['plan'],sources,routes,protocol)
    for owner in stage_owners.values():
        require(owner.get('suite_lock_sha256')==reader.inputs[BASE.SUITE_LOCK]
            and owner.get('split_source_lock_sha256')==reader.inputs[BASE.SOURCE_LOCK]
            and owner.get('source_manifest')==suite['source_manifests']['test'],'each actual physical owner source locks must agree')
    entry=P.validate_entry(control['execution_entry']);batch_sources=control['execution_implementation']['sources']
    require(control['execution_implementation']==P.execution_implementation(entry,batch_sources)
        and launch.get('batch_sources')==parent['launch'].get('batch_sources')==grandparent['launch'].get('batch_sources')==middle['launch'].get('batch_sources')==batch_sources,'same actual901 implementation required')
    for name,p in batch_sources.items():reader.raw(name,expected=pointer(p,name)['sha256'],resumed=True,evidence=True)
    verify_reporting_dependencies(reader)
    small=reader.pointed(pointer(entry['batch_lock'],LEGACY.BATCH_LOCK),resumed=True,evidence=True)
    reader.pointed(pointer(entry['batch_control'],LEGACY.BATCH_CONTROL),resumed=True,evidence=True)
    require(small.get('schema')=='r10_scoped_frozen_lookup_bootstrap_inputs_v901' and small.get('execution_enabled') is True,'actual unchanged915/901 supplement required')
    for name,p in batch_sources.items():require(reader.lookup(small['files'],name)==p['sha256'],'901 source must belong to actual small lock')
    rows=plan['jobs'];tiers=result.get('evidence_tiers');casepointers=result['cases'];drivers=result['admission_driver_receipts']
    require(isinstance(tiers,list) and len(tiers)==len(rows)==len(casepointers)==len(drivers)==1920,'exact ordered1920 three-tier evidence required')
    reducer=H.load_reducer(reader);selected=[];seen=set();processes=set();batch_pointers=[]
    for row,observed,casep,driverp in zip(rows,tiers,casepointers,drivers):
        name=row['job_id'];runtime=row['evidence_runtime'];imported=name in old_ids;owner=stage_owners.get(runtime)
        require(owner is not None and runtime==(row['resume_lineage_v1040']['parent_evidence_runtime'] if imported else RUNTIME)
            and type(row['imported_completed']) is bool and row['imported_completed'] is imported and name not in seen,'exact actual evidence tier required');seen.add(name)
        expected=dict(job_id=name,evidence_runtime=runtime,imported_completed=imported,actual_stage_owner_pid=owner['pid'],
            actual_stage_owner_start_ticks=owner['start_ticks'],resume_lineage=row['resume_lineage'],resume_lineage_v1010=row['resume_lineage_v1010'],resume_lineage_v1040=row['resume_lineage_v1040'],previous_row_sha256=row['previous_row_sha256'],
            execution_entry=row['execution_entry'],actual_outer_argv=row['argv'],physical_argv=row['physical_argv'],case=casep,
            admission_driver_receipt=driverp,batch_driver_receipt=observed.get('batch_driver_receipt'))
        require(observed==expected,'actual957 whole-row provenance differs')
        out=option(row['physical_argv'],'--out');pointer(casep,out+'/case.json');pointer(driverp,out+'/admission_driver_receipt.json')
        require(row.get('admission_driver')==result['admission_driver'] and row.get('admission_guard')==result['admission_guard'],'same original751/728 graph required')
        cpu_leaf=runtime+'/CPU/'+name;native_leaf=runtime+'/native/'+name
        cpu=reader.json(cpu_leaf+'/CPU_preflight.json',evidence=True,resumed=imported)
        require(cpu==reader.json(cpu_leaf+'/stdout.log',evidence=True,resumed=imported) and cpu.get('complete') is True
            and cpu.get('native_started') is False and cpu.get('GPU_started') is False and cpu.get('actual_split')=='test'
            and cpu.get('execution_protocol_changed') is True,'actual CPU-only proof/stdout required')
        bindings={k:row[k] for k in H.BINDINGS}
        for leaf,check in ((cpu_leaf,True),(native_leaf,False)):
            child=reader.json(leaf+'/launch.json',evidence=True,resumed=imported);end=reader.json(leaf+'/terminal.json',evidence=True,resumed=imported);clean(end)
            process=(child.get('launcher_pid'),child.get('launcher_start_ticks'))
            require(all(type(v) is int and v>0 for v in process) and process not in processes,'actual unique CPU/native PID/start required');processes.add(process)
            require(child.get('argv')==row['argv']+(['--check'] if check else []) and child.get('check_only') is check
                and child.get('owner_pid')==owner['pid'] and child.get('native_slots')==(0 if check else 1)
                and child.get('GPU_slots')==(0 if check else int(row['method']=='physmani'))
                and child.get('expected_identity_sha256')==(None if check else cpu['identity']['sha256'])
                and child.get('execution_protocol_changed') is True and all(child.get(k)==v and cpu.get(k)==v for k,v in bindings.items()),
                'actual CPU/native owner/argv/identity required')
            if runtime==ORIGINAL:
                require(all(k not in child for k in ('execution_entry','physical_argv','inherited_frozen_environment','current_mount_namespace')),'753 must not acquire invented901/environment fields')
            else:
                require(child.get('execution_entry')==row['execution_entry'] and child.get('physical_argv')==row['physical_argv']+(['--check'] if check else [])
                    and child.get('inherited_frozen_environment')==owner['frozen_environment']
                    and child.get('current_mount_namespace')==owner['frozen_environment']['current_mount_namespace'],'real902/956 forwarded argv/namespace required')
            if not check:native_end=end
        record=status['completed'][name];clean(record)
        require(all(record.get(k)==v for k,v in native_end.items()) and (not imported or record==parent['status']['completed'][name]),'completed record must preserve actual native outcome')
        P.require_batch_execution(cpu.get('batch_execution'),row,check=True,expected_sources={} if runtime==ORIGINAL else batch_sources)
        case=reader.pointed(casep,evidence=True,resumed=imported);driver=reader.pointed(driverp,evidence=True,resumed=imported)
        require(native_end.get('case')==record.get('case')==casep and native_end.get('admission_driver_receipt')==record.get('admission_driver_receipt')==driverp
            and cpu.get('identity')==case.get('identity') and cpu.get('component_domain')==case.get('component_domain'),'original CPU/native/case/driver bindings required')
        require(driver.get('schema')=='r10_explicit_author_outcome_formal_driver_v751' and driver.get('complete') is True
            and driver.get('actual_split')=='test' and driver.get('execution_protocol_changed') is True and driver.get('case')==casep
            and all(driver.get(k)==v for k,v in bindings.items()),'actual751 original driver receipt required')
        bp=None
        if runtime!=ORIGINAL:
            bp=pointer(native_end.get('batch_driver_receipt'),out+'/batch_driver_receipt.json')
            require(record.get('batch_driver_receipt')==observed.get('batch_driver_receipt')==bp,'real901 sidecar must bind terminal/status/tier')
            P.require_batch_execution(reader.pointed(bp,evidence=True,resumed=imported),row,check=False,expected_sources=batch_sources,case=casep,original_receipt=driverp);batch_pointers.append(bp)
        else:require('batch_driver_receipt' not in native_end and 'batch_driver_receipt' not in record and observed['batch_driver_receipt'] is None,'753 cannot claim a901 sidecar')
        require_saved_case(case,row,sources[row['task'],row['raw']],protocol,reader.inputs[BASE.SOURCE_LOCK])
        require(record.get('native_success') is case['official_task_success'] and 'motion_pass' in record
            and record['motion_pass'] is recorded_motion_flag(case),'nativeFalse and bool-or-unknown diagnostic must remain original')
        metrics,diagnostics=reducer.reduce_saved_case(case);metrics={k:(v if reducer.finite(v) else None) for k,v in metrics.items()}
        selected.append(dict(task=row['task'],method=row['method'],source_raw_episode_id=row['raw'],source_row_sha256=row['source_row_sha256'],
            case=casep,admission_driver_receipt=driverp,batch_driver_receipt=bp,evidence_runtime=runtime,imported_completed=imported,
            resume_lineage=row['resume_lineage'],resume_lineage_v1010=row['resume_lineage_v1010'],resume_lineage_v1040=row['resume_lineage_v1040'],actual_owner_pid=owner['pid'],actual_owner_start_ticks=owner['start_ticks'],execution_entry=row['execution_entry'],
            CPU_receipt=reader.pointer(cpu_leaf+'/CPU_preflight.json'),native_launch=reader.pointer(native_leaf+'/launch.json'),
            native_terminal=reader.pointer(native_leaf+'/terminal.json'),official_task_success=case['official_task_success'],
            native_terminal_failure=case['native_terminal_failure'],metrics=metrics,recorded_window_diagnostics=diagnostics))
    require(result.get('batch_driver_receipts')==batch_pointers and set(status['completed'])==seen and len(selected)==1920,'complete original901-sidecar/1920-case cohort required')
    for method in METHODS:require(sum(r['official_task_success'] for r in selected if r['method']==method)==result['native_successes_by_method'][method],'all native outcomes must remain in motion cohort')
    require(terminal.get('native_successes')==sum(r['official_task_success'] for r in selected),'actual956 native total differs')
    subset=reader.json(SUBSET)
    if hasattr(reader,'close'):reader.close()
    return selected,result,subset,reducer


def build_report(reader,rows,result,subset,reducer):
    return dict(schema=SCHEMA,complete=True,created_utc=datetime.now(timezone.utc).isoformat(),read_only_analysis=True,split='test',
        task_count=16,source_count=640,case_count=1920,methods=list(METHODS),task_source_counts={t:40 for t in TASKS},policy_sha256=POLICY,
        test_results=reader.pointer(RESULT),actual_test_auditor=dict(path=AUDITOR,sha256=result['source_sha256']),
        resume_control=reader.pointer(CONTROL),resume_control_lock=reader.pointer(LOCK),
        provenance_runtime_counts=dict(Counter(r['evidence_runtime'] for r in rows)),actual_stage_owners=result['actual_stage_owners'],
        metric_definitions=subset['metric_definitions'],statistical_definition=dict(subset['statistical_definition'],
            overall='pool640 equally weighted cases per method; never average task statistics'),
        diagnostic_definition=subset['diagnostic_definition'],hard_budget_disclosure=subset['hard_budget_disclosure'],
        common_RML_planner_configuration=dict(parameters=PARAMETERS,profile_sha256=PROFILE,cases=1920,
            parameter_kind='saved path-parameter planning configuration, not a per-joint measured bound'),
        resource_profile=result['resource_profile'],resource_qualification=result['resource_qualification'],resource_qualifier=result['resource_qualifier'],
        resource_capacity_envelope=result['resource_capacity_envelope'],resource_profile_within_capacity=result['resource_profile_within_capacity'],resource_capacity_source=result['resource_capacity_source'],
        resource_profile_changed=result['resource_profile_changed'],previous_resource_profile=result['previous_resource_profile'],
        previous_resource_qualification=result['previous_resource_qualification'],
        execution_implementation=result['execution_implementation'],historical_infrastructure_attempts=result['historical_infrastructure_attempts'],
        overall_by_method={method:reducer.summarize([r for r in rows if r['method']==method]) for method in METHODS},
        by_task_method=[dict(task=task,method=method,**reducer.summarize([r for r in rows if (r['task'],r['method'])==(task,method)]))
            for task in TASKS for method in METHODS],case_evidence=rows,inputs_sha256=reader.inputs,source_sha256=reader.inputs[SOURCE],
        saved_motion_algorithm_source=dict(path=LEGACY_SOURCE,sha256=LEGACY_SHA),
        previous_saved_motion_adapter=dict(path=PREVIOUS_ADAPTER,sha256=PREVIOUS_ADAPTER_SHA),
        audited_TEST_helper=dict(path=LEGACY.BASE_SOURCE,sha256=LEGACY.BASE_SHA),audited_VAL_helper=dict(path=BASE.HELPER,sha256=BASE.HELPER_SHA),
        reducer_source=dict(path=REDUCER,sha256=PINS[REDUCER]),
        pure_provenance_helpers=[dict(path=CORE,sha256=CORE_SHA),dict(path=PARENT_CORE,sha256=PARENT_CORE_SHA),dict(path=GRANDPARENT_CORE,sha256=GRANDPARENT_CORE_SHA),dict(path=LEGACY.OWNER,sha256=LEGACY.OWNER_SHA)],
        reporting_dependencies=reader.reporting_dependencies,original_physical_locks_amended=False,
        saved_case_diagnostic_presence_allows_unknown=True,unknown_motion_not_filled_as_zero_or_boolean=True,
        namespace_evidence_scope='753 retains original fields;902/956/1011/1041 children retain real own proofs;1042 auditor binds1043 stage',
        full1920_five_tier_TEST_results_and_complete1043_observed=True,new_release_granted_by_this_summary=False,
        native_false_retained=True,success_rate_or_acceleration_filter_used=False,new_joint_differencing_used=False,
        alignment_motion_mirror_pooled=False,transitive_TRAIN_model_OS_inputs_walked=False,
        raw_reset_PKL_image_HDF5_contents_opened=False,TEST_used_for_fitting=False,native_GPU_or_jobs_started=False,
        frozen_source_or_lock_modified=False,TXT_or_handoff_written=False)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--test-results-sha256',required=True);parser.add_argument('--resume-lock-sha256',required=True)
    parser.add_argument('--resume-control',default=CONTROL);parser.add_argument('--resume-lock',default=LOCK)
    args=parser.parse_args(argv);require(not (ROOT/OUTPUT).exists(),'preserve existing1045 output; never overwrite')
    reader=Reader(ROOT);rows,result,subset,reducer=collect(reader,args.test_results_sha256,args.resume_control,args.resume_lock,args.resume_lock_sha256)
    reader.inputs[SOURCE]=sha_bytes(Path(__file__).read_bytes());report=build_report(reader,rows,result,subset,reducer)
    # No directory is created on failure, and no existing result is replaced.
    with (ROOT/OUTPUT).open('x') as stream:stream.write(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(complete=True,output=OUTPUT,sha256=sha_bytes((ROOT/OUTPUT).read_bytes()),case_count=1920)))
    return report


if __name__=='__main__':main()
