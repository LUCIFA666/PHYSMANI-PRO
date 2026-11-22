"""Saved motion from the complete, independently audited mixed753/902 TEST grid.

No launch, release, fitting or policy change. A future main needs externally
bound906 result and915 lock SHAs, clean907 stages/final and full902 completion.
Only then are source metadata and the1920 explicit saved cases consumed. Old
drained753 evidence stays old;901 bootstrap receipts stay separate from751.
The unchanged838 reducer uses saved fields, retaining False/zero-tick/unknown.
Preparation and tests must use source code and CPU fakes only.
"""
import argparse
import ast
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import types

ROOT=Path(__file__).resolve().parents[1]
BASE_SOURCE='scripts/summarize_formal_TEST_motion_v851_r10.py'
BASE_SHA='d31fb9cad92f3346ccdd55290f72c853365719de5f883ca9ff33e3710d4839eb'
OWNER='scripts/run_concurrent_resumed_formal_TEST_owner_v902_r10.py'
OWNER_SHA='ef55aaf86e9334dcae0f02113e76fcbdcc360709a769aaa34f2b1f33a77c6c43'
AUDITOR='scripts/audit_concurrent_resumed_formal_TEST_v906_r10.py'
CONTROLLER_SOURCE='scripts/continue_concurrent_TEST_v907_r10.py'
PREPARER='scripts/prepare_concurrent_TEST_resume_v915_r10.py'


def load_source(name,expected):
    raw=(ROOT/name).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=expected:raise ValueError('audited source SHA changed: '+name)
    module=types.ModuleType('saved_motion_'+Path(name).stem);module.__file__=str(ROOT/name)
    exec(compile(raw,str(ROOT/name),'exec'),module.__dict__)
    return module


BASE=load_source(BASE_SOURCE,BASE_SHA);H=BASE.H
require=BASE.require;digest=BASE.digest;sha_bytes=BASE.sha_bytes;valid_sha=BASE.valid_sha;option=BASE.option
METHODS=BASE.METHODS;TASKS=BASE.TASKS;POLICY=BASE.POLICY;PROFILE=BASE.PROFILE;PARAMETERS=BASE.PARAMETERS
REDUCER=BASE.REDUCER;SUBSET=BASE.SUBSET
SETUP='artifacts/framework/multitask_r10/formal_TEST_concurrent_resume_v902_setup'
CONTROL=SETUP+'/resume_control.json';LOCK=SETUP+'/resume_inputs.json'
BATCH_CONTROL=SETUP+'/formal_batch_control.json';BATCH_LOCK=SETUP+'/formal_batch_inputs.json'
RUNTIME='runtime/framework/formal_TEST_concurrent_resume_v902'
CONTROLLER='runtime/framework/ordered_concurrent_TEST_v907'
PREVIOUS=BASE.RUNTIME;OLD_CONTROLLER=BASE.CONTROLLER
ART='artifacts/framework/multitask_r10/formal_TEST_concurrent_resume_v902'
RESULT=ART+'/TEST_results.json'
OUTPUT=RUNTIME+'/full_sixteen_task_motion_summary_v920.json'
OWNER_STAGE=CONTROLLER+'/stages/formal_TEST_1920';AUDIT_STAGE=CONTROLLER+'/stages/independent_TEST_results'
SCHEMA='r10_full1920_composite_TEST_saved_motion_summary_v920'
RESULT_SCHEMA='r10_composite_concurrent_resumed_formal_TEST_results_v906'
REPORTING_DEPENDENCIES={'our_mani/multitask/three_method_release16.py':'faf77babdd431be41c5765a34dfb1b08a18e7bf8ec1a70b6f26b78f7210477f3'}
PINS={**BASE.PINS,BASE_SOURCE:BASE_SHA,OWNER:OWNER_SHA,**REPORTING_DEPENDENCIES}
SOURCES={*BASE.SOURCES,BASE_SOURCE,OWNER,AUDITOR,CONTROLLER_SOURCE,PREPARER,
    'scripts/run_batched_formal_driver_v901_r10.py','our_mani/multitask/frozen_lookup_batch_v900.py',
    'our_mani/multitask/rml_case_preflight_v742.py','our_mani/multitask/three_method_release16.py'}


def load_pure_owner_helpers():
    """Compile exact selected902 definitions; no imports, main or IO helper.

    This is not an AST transformation. Only these source-pinned pure functions
    and literal bindings enter the namespace; the original full module never
    executes and no source/control/data graph is followed at import.
    """
    raw=(ROOT/OWNER).read_bytes();require(sha_bytes(raw)==OWNER_SHA,'audited902 source SHA changed')
    constants={'CONTROL_SCHEMA','LOCK_SCHEMA','PROFILE_SCHEMA','QUALIFICATION_SCHEMA','PREVIOUS','PREVIOUS_CONTROLLER',
        'RUNTIME','OUTPUT','ORIGINAL_ENTRY','BATCH_ENTRY','BATCH_SCHEMA','COMMON_SOURCES','ROW_PROVENANCE'}
    functions={'option','validate_resource_profile','require_partition','validate_entry','valid_pointer','physical_argv',
        'outer_argv','resumed_jobs','require_batch_execution','require_batch_metrics','execution_implementation'}
    selected=[];found=set()
    for node in ast.parse(raw,filename=str(ROOT/OWNER)).body:
        names={t.id for t in node.targets if isinstance(t,ast.Name)} if isinstance(node,ast.Assign) else set()
        if names and names<=constants:selected.append(node);found.update(names)
        elif isinstance(node,ast.FunctionDef) and node.name in functions:selected.append(node);found.add(node.name)
    require(found==constants|functions,'exact pure902 source helper set required')
    namespace=dict(deepcopy=deepcopy,Path=Path,math=math,TASKS=TASKS,METHODS=METHODS,digest=digest)
    exec(compile(ast.Module(body=selected,type_ignores=[]),str(ROOT/OWNER),'exec'),namespace)
    return types.SimpleNamespace(**{k:namespace[k] for k in constants|functions})


P=load_pure_owner_helpers()


def allowed(name):
    fixed={*PINS,*SOURCES,CONTROL,LOCK,BATCH_CONTROL,BATCH_LOCK,RESULT,SUBSET,
        BASE.SUITE,BASE.SUITE_LOCK,BASE.SOURCE_LOCK,BASE.MANIFEST,BASE.ORIGINAL_MANIFEST,BASE.ROUTES,BASE.VAL_RELEASE}
    if name in fixed:return True
    for runtime in (CONTROLLER,RUNTIME,PREVIOUS,OLD_CONTROLLER):
        if name in {runtime+'/'+n for n in ('launch.json','plan.json','status.json','terminal.json','failure.json','DRAIN')}:return True
    for stage in (OWNER_STAGE,AUDIT_STAGE):
        if name in {stage+'/'+n for n in ('launch.json','terminal.json','stdout.log')}:return True
    for runtime in (RUNTIME,PREVIOUS):
        parts=name[len(runtime)+1:].split('/') if name.startswith(runtime+'/') else []
        if len(parts)==3 and parts[0] in ('CPU','native') and parts[2] in ('launch.json','terminal.json','CPU_preflight.json','stdout.log','stderr.log'):return True
    for out in (ART+'/test',BASE.ART+'/test'):
        parts=name[len(out)+1:].split('/') if name.startswith(out+'/') else []
        if len(parts)==4 and parts[0] in TASKS and parts[2] in METHODS and parts[3] in ('case.json','admission_driver_receipt.json','batch_driver_receipt.json'):return True
    return False


class Reader(BASE.Reader):
    """Only explicit consumed evidence; never walk a transitive frozen map."""
    def __init__(self,root):
        super().__init__(Path(root).resolve());self.resumed={};self.signatures={}
        self.inputs.update({BASE_SOURCE:BASE_SHA,OWNER:OWNER_SHA})
    def raw(self,name,*,expected=None,evidence=False,frozen=False,original=False,resumed=False):
        require(isinstance(name,str) and not Path(name).is_absolute() and str(PurePosixPath(name))==name
            and '..' not in PurePosixPath(name).parts and allowed(name),'non-evidence or undeclared input refused: '+str(name))
        path=(self.root/name).resolve(strict=True)
        require(path.is_relative_to(self.root) and str(path.relative_to(self.root))==name,'canonical evidence required; symlink aliases refused before open')
        signature=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
        before=signature(path.stat());raw=path.read_bytes();require(before==signature(path.stat()),'input changed while reading: '+name)
        actual=sha_bytes(raw)
        wanted=[v for v in (expected,PINS.get(name),self.lookup(self.evidence,name) if evidence else None,
            self.lookup(self.frozen,name) if frozen else None,self.lookup(self.original,name) if original else None,
            self.lookup(self.resumed,name) if resumed else None) if v is not None]
        require(all(valid_sha(v) and actual==v for v in wanted),'input SHA changed: '+name)
        require(name not in self.inputs or self.inputs[name]==actual,'input changed during summary: '+name)
        self.inputs[name]=actual;self.signatures[name]=before;return raw
    def close(self):
        for name,previous in self.signatures.items():
            path=(self.root/name).resolve(strict=True);s=path.stat()
            require(str(path.relative_to(self.root))==name and previous==(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns),
                'consumed evidence path/stat changed before summary end: '+name)


def pointer(value,name=None):
    require(P.valid_pointer(value) and (name is None or value['path']==name),'exact canonical project path/SHA pointer required')
    return value


def pid(value):
    require(type(value.get('pid')) is int and value['pid']>0 and type(value.get('start_ticks')) is int and value['start_ticks']>0,'real PID/start required')
    return value['pid'],value['start_ticks']


def environment(value,control,record,*,expected_mount=None):
    H.require_environment(value,control,observer=pid(record)[0])
    require(value.get('observer_start_ticks')==record['start_ticks'],'recorded environment observer start changed')
    if expected_mount is not None:require(value['current_mount_namespace']==expected_mount,'recorded stage namespace changed')
    return value


def clean(value,*,stage=False):
    H.require_clean_terminal(value,controller_stage=stage)
    require(value.get('error') is None and value.get('leaf_future_exception') is not True,'infrastructure error is not complete native evidence')


def recorded_motion_flag(case):
    # Exact precedence of frozen three_method_release16.measured_motion; this
    # is the status consistency flag, never a filter or replacement reducer.
    actor=case.get('actor_metrics') or {}
    value=(actor.get('task_dispatch_v43') or {}).get('actual_observed_window_limits_passed')
    if type(value) is bool:return value
    feed=actor.get('continuous_reference_feedforward') or {};failed=feed.get('observed_motion_qualification_failed')
    if type(failed) is bool and feed.get('observed_events'):return not failed
    for report in (case.get('all_stage_counted_joint_motion'),(case.get('alignment_receipt') or {}).get('all_stage_counted_joint_motion_v191')):
        if isinstance(report,dict) and type(report.get('all_stage_actual_window_checks_passed')) is bool:return report['all_stage_actual_window_checks_passed']
    return None


def verify_reporting_dependencies(reader):
    """Supplementary reporting source, never invented old lock membership."""
    records=[]
    for name,expected in REPORTING_DEPENDENCIES.items():
        membership={}
        for key in ('original','frozen'):
            mapping=getattr(reader,key)
            values=[mapping[k] for k in (name,str(reader.root/name)) if k in mapping]
            require(all(valid_sha(v) and v==expected for v in values),'existing old declaration conflicts with supplementary reporting helper: '+name)
            membership[key]=bool(values)
        reader.raw(name,expected=expected,resumed=True,evidence=True)
        records.append(dict(path=name,sha256=expected,scope='supplementary_readonly_reporting_dependency',
            used_for='check saved diagnostic status flag with the frozen measured_motion precedence',
            original_physical_input_lock_member=membership['original'],original_TEST_suite_input_lock_member=membership['frozen'],
            bound_by_actual915_supplement_and906_evidence=True,independent_reporting_source_pin=True,
            old_input_lock_amended=False,proof_of_original_physical_freeze_membership_claimed=False))
    reader.reporting_dependencies=records
    return records


def require_stage(value,end,argv,source_sha,controller):
    pid(value);clean(end,stage=True)
    require(value.get('argv')==argv and value.get('source_sha256')==source_sha and value.get('automatic_retry') is False
        and value.get('inherited_frozen_environment')==controller['frozen_environment']
        and value.get('current_mount_namespace')==controller['frozen_environment']['current_mount_namespace'],'actual907 stage argv/source/PID/environment required')


def require_owner_completion(launch,plan,status,terminal,stage,control,control_pointer,lock_pointer):
    protocol=control['author_outcome_protocol'];count=len(control['completed_job_ids'])
    common=dict(split='test',policy_sha256=POLICY,author_outcome_protocol=protocol,execution_protocol_changed=True,
        resume_control=control_pointer,resume_control_lock=lock_pointer,resource_profile=control['resource_profile'],
        resource_qualification=control['resource_qualification'],execution_implementation=control['execution_implementation'],automatic_retry=False)
    require(all(all(v.get(k)==x for k,x in common.items()) for v in (launch,plan,status,terminal)),'immutable902 owner bindings required')
    require(pid(launch)==pid(stage) and launch.get('source_sha256')==control['resume_owner']['sha256']
        and launch.get('argv')==stage['argv'][3:] and launch.get('execution_entry')==control['execution_entry']
        and launch.get('imported_completed_cases')==count and plan.get('imported_completed_cases')==count,
        'actual902 owner stage/source/argv/import binding required')
    require(terminal.get('complete') is True and terminal.get('expected')==terminal.get('completed')==1920
        and terminal.get('failed')==0 and terminal.get('owner_error') is None and terminal.get('stop_requested') is False
        and terminal.get('drain_requested') is False and terminal.get('drained') is False
        and terminal.get('CPU_ready_unexecuted')==[] and terminal.get('CPU_not_launched')==0
        and terminal.get('imported_completed_cases')==count and terminal.get('resumed_new_cases')==1920-count
        and terminal.get('formal_release_requires_independent_audit') is True,'clean full1920 mixed owner terminal required before TEST metadata')
    require(status.get('expected')==1920 and len(status.get('completed',{}))==1920 and status.get('failed')=={}
        and status.get('active')==status.get('checking')==status.get('ready_jobs')==[]
        and status.get('CPU_ready')==status.get('native_slots')==status.get('GPU_slots')==0
        and status.get('CPU_launched')==1920-count and status.get('imported_completed')==count
        and status.get('owner_error') is None and status.get('stop_requested') is False and status.get('drain_requested') is False,
        'clean full1920 mixed owner status required before TEST metadata')
    first=environment(launch.get('frozen_environment',{}),control,launch,expected_mount=stage['current_mount_namespace'])
    require(all(v.get('frozen_environment')==first for v in (plan,status,terminal)),'actual902 environment must be unchanged across owner records')
    quotas=P.validate_resource_profile(control['resource_profile'])
    require(all(launch.get(k)==quotas[k] for k in ('max_native_slots','max_GPU_slots','max_CPU_preflight_slots')),'actual902 resource profile differs')


def require_result(result,control,control_pointer,lock_pointer,audit_stage):
    BASE.require_result_header(result)
    require(result.get('schema')==RESULT_SCHEMA and result.get('source_sha256')==control['resume_auditor']['sha256']
        and result.get('resume_control')==control_pointer and result.get('resume_control_lock')==lock_pointer
        and result.get('source_input_lock')==control['source_inputs'] and result.get('validation_release')==control['validation_release']
        and result.get('author_outcome_protocol')==control['author_outcome_protocol']
        and result.get('resource_profile')==control['resource_profile'] and result.get('resource_qualification')==control['resource_qualification']
        and result.get('execution_implementation')==control['execution_implementation'],'actual independent906 result graph required')
    for key,value in dict(used_evidence_rechecked_at_close=True,old753_results_not_rerun=True,independent_composite_audit=True,
        original754_single_owner_result_claimed=False,automatic_retry=False).items():require(result.get(key) is value,'composite audit evidence required: '+key)
    invocation=result.get('auditor_invocation',{})
    require(pid(invocation)==pid(audit_stage) and invocation.get('argv')==audit_stage['argv'][3:]
        and invocation.get('source')==control['resume_auditor'] and invocation.get('inherited_physical_stage_identity_claimed') is False,
        'actual906 independent auditor PID/start/argv required')
    environment(result.get('frozen_environment',{}),control,invocation,expected_mount=audit_stage['current_mount_namespace'])


def collect(reader,result_sha,resume_control=CONTROL,resume_lock=LOCK,resume_lock_sha=None):
    require(resume_control==CONTROL and resume_lock==LOCK and valid_sha(result_sha) and valid_sha(resume_lock_sha),'exact915 control and externally pinned lock/result SHAs required')
    lock=reader.json(LOCK,expected=resume_lock_sha)
    require(lock.get('schema')==P.LOCK_SCHEMA and lock.get('approved_for_execution') is True and isinstance(lock.get('files'),dict),'actual frozen915 continuation lock required')
    reader.resumed=lock['files'];control=reader.json(CONTROL,resumed=True)
    cp,lp=reader.pointer(CONTROL),reader.pointer(LOCK)
    require(lock.get('control')==cp,'actual915 lock must bind its explicit control pointer')
    require(control.get('schema')==P.CONTROL_SCHEMA and control.get('split')=='test' and control.get('expected_cases')==1920
        and control.get('source_count')==640 and control.get('policy_sha256')==POLICY and control.get('physical_policy_changed') is False,
        'unchanged full1920 continuation required')
    for key,name in (('resume_controller',CONTROLLER_SOURCE),('resume_owner',OWNER),('resume_auditor',AUDITOR)):
        p=pointer(control.get(key),name);reader.raw(name,expected=p['sha256'],resumed=True)
    controller=reader.json(CONTROLLER+'/launch.json');final=reader.json(CONTROLLER+'/terminal.json')
    require(controller.get('source_sha256')==control['resume_controller']['sha256'] and controller.get('resume_control')==cp
        and controller.get('resume_control_lock')==lp and controller.get('automatic_retry') is False,'real907 source/control binding required')
    environment(controller.get('frozen_environment',{}),control,controller)
    require(final.get('complete') is True and final.get('automatic_retry') is False
        and pointer(final.get('TEST_results'),RESULT)['sha256']==result_sha,'complete907 final result pointer required before TEST metadata')
    controller_status=reader.json(CONTROLLER+'/status.json')
    require(controller_status.get('stage')=='complete' and controller_status.get('automatic_retry') is False
        and controller_status.get('frozen_environment')==controller['frozen_environment'],
        'actual907 status must be complete in the same recorded environment before TEST metadata')
    owner_args=[]
    for key,flag in (('suite','--suite'),('suite_lock','--suite-lock'),('source_inputs','--source-inputs'),('validation_release','--validation-release')):
        owner_args.extend((flag,pointer(control.get(key))['path']))
    owner_args += ['--runtime',RUNTIME,'--out',ART+'/test','--resume-control',CONTROL,'--resume-lock',LOCK,'--resume-lock-sha256',resume_lock_sha]
    audit_args=['--resume-control',CONTROL,'--resume-lock',LOCK,'--resume-lock-sha256',resume_lock_sha]
    stages={}
    for stage,source,args in ((OWNER_STAGE,OWNER,owner_args),(AUDIT_STAGE,AUDITOR,audit_args)):
        launch=reader.json(stage+'/launch.json');end=reader.json(stage+'/terminal.json')
        require_stage(launch,end,[str(reader.root/'.conda-env/bin/python'),'-B','-u',source,*args],reader.inputs[source],controller);stages[source]=launch
    launch,plan,status,terminal=(reader.json(RUNTIME+'/'+n) for n in ('launch.json','plan.json','status.json','terminal.json'))
    require_owner_completion(launch,plan,status,terminal,stages[OWNER],control,cp,lp)
    # Completion receipts precede the first result/source-manifest/case read.
    result=reader.json(RESULT,expected=result_sha);require_result(result,control,cp,lp,stages[AUDITOR]);reader.evidence=result['inputs_sha256']
    for name in (CONTROL,LOCK,OWNER,AUDITOR,*[RUNTIME+'/'+n for n in ('launch.json','plan.json','status.json','terminal.json')]):
        require(reader.lookup(reader.evidence,name)==reader.inputs[name],'actual906 consumed evidence SHA differs: '+name)
    stdout=reader.json(AUDIT_STAGE+'/stdout.log')
    require(stdout.get('complete') is True and stdout.get('schema')==RESULT_SCHEMA and stdout.get('split')=='test'
        and stdout.get('cases')==1920 and stdout.get('out')==RESULT and stdout.get('motion_is_diagnostic_only') is True
        and stdout.get('native_successes_by_method')==result.get('native_successes_by_method'),'actual906 stdout required; it has no TEST_permitted field')
    old={}
    for key,name in (('previous_launch','launch.json'),('previous_plan','plan.json'),('previous_status','status.json'),('previous_terminal','terminal.json')):
        old[name]=reader.pointed(pointer(control[key],PREVIOUS+'/'+name),resumed=True,evidence=True)
    P.require_partition(control,old['plan.json'],old['status.json'],old['terminal.json'])
    require(plan.get('jobs')==P.resumed_jobs(old['plan.json'],control,control['execution_entry']) and plan.get('expected_cases')==1920
        and plan.get('source_outcome_filtering') is False,'exact old753/new902 continuation plan required')
    for key,name in (('previous_controller_launch','launch.json'),('previous_controller_status','status.json'),('previous_controller_failure','failure.json')):
        value=reader.pointed(pointer(control[key],OLD_CONTROLLER+'/'+name),resumed=True,evidence=True)
        if name=='status.json':require(value.get('stage')=='requires_concrete_repair' and value.get('automatic_retry') is False,'real old827 drain status required')
        if name=='failure.json':require(value.get('complete') is False and value.get('user_stop') is False and value.get('automatic_retry') is False,'real old827 failure, not a fabricated full terminal, required')
    reader.pointed(pointer(control['previous_DRAIN'],PREVIOUS+'/DRAIN'),resumed=True,evidence=True)
    # The original suite still advertises754.906 is the separately frozen actual auditor.
    original_control=reader.json(BASE.CONTROL);original_lock=reader.json(BASE.LOCK);reader.original=original_lock['files']
    require(original_control.get('policy_sha256')==POLICY and original_lock.get('approved_for_execution') is True,'actual frozen825 VAL graph required')
    suite=reader.pointed(pointer(control['suite'],BASE.SUITE),evidence=True)
    suite_lock=reader.pointed(pointer(control['suite_lock'],BASE.SUITE_LOCK),evidence=True);reader.frozen=suite_lock['files']
    require(suite_lock.get('approved_for_execution') is True and suite_lock.get('actual_formal_split')=='test'
        and suite_lock.get('policy_sha256')==POLICY and suite_lock.get('model_or_policy_parameters_changed') is False,'original unchanged826 TEST suite required')
    require(reader.lookup(reader.frozen,BASE.SUITE)==reader.inputs[BASE.SUITE] and suite.get('validation_release_path')==BASE.VAL_RELEASE,'original frozen suite scope required')
    val=reader.pointed(pointer(control['validation_release'],BASE.VAL_RELEASE),evidence=True,frozen=True);H.require_release_header(val)
    valp=reader.pointer(BASE.VAL_RELEASE)
    require(suite_lock.get('original_945_VAL_release')==valp and val.get('source_sha256')==original_control['resume_auditor']['sha256']
        and val.get('resume_control')==reader.pointer(BASE.CONTROL) and val.get('resume_control_lock')==reader.pointer(BASE.LOCK),'real824 VAL prerequisite required')
    protocol=control['author_outcome_protocol']
    require(protocol==suite['author_outcome_protocol']==original_control['author_outcome_protocol']
        and result.get('admission_driver')==suite.get('formal_case_driver') and result.get('admission_guard')==suite.get('formal_admission_guard')
        and result.get('base_result_auditor')==suite.get('formal_result_auditor')
        and pointer(suite['formal_result_auditor'],BASE.AUDITOR)['sha256']==reader.lookup(reader.frozen,BASE.AUDITOR),
        'original754 suite auditor and independent906 must remain distinct')
    stage_bindings=dict(split='test',policy_sha256=POLICY,author_outcome_protocol=protocol,execution_protocol_changed=True,
        admission_driver=result['admission_driver'],admission_guard=result['admission_guard'],native_runtime=protocol['runtime_consumer'],
        original_native_runtime=protocol['original_runtime'],automatic_retry=False)
    require(all(all(record.get(k)==v for k,v in stage_bindings.items()) for record in (old['launch.json'],launch,plan,status,terminal)),
        'both real physical owners must retain original policy/admission/runtime bindings')
    manifest=reader.pointed(suite['source_manifests']['test'],frozen=True)
    require(suite['source_manifests']['test']['path']==BASE.MANIFEST and manifest.get('original_source',{}).get('path')==BASE.ORIGINAL_MANIFEST,'original TEST manifest scope required')
    original=reader.pointed(manifest['original_source'],frozen=True)
    source_lock=reader.pointed(pointer(control['source_inputs'],BASE.SOURCE_LOCK),evidence=True)
    reader.raw(BASE.SOURCE_PREPARER,frozen=True,original=True,resumed=True)
    require(source_lock.get('source_manifest_sha256')==reader.inputs[BASE.MANIFEST],'same756 source manifest SHA required')
    sources=BASE.source_grid(manifest,original,source_lock,valp,reader.inputs[BASE.SOURCE_PREPARER])
    routes=reader.pointed(suite['routes'],frozen=True,original=True)
    BASE.require_plan(old['plan.json'],sources,routes,protocol)
    for record in (old['launch.json'],launch):
        require(record.get('suite_lock_sha256')==reader.inputs[BASE.SUITE_LOCK] and record.get('split_source_lock_sha256')==reader.inputs[BASE.SOURCE_LOCK]
            and record.get('source_manifest')==suite['source_manifests']['test'],'real evidence owner source/lock scope required')
    old_source=reader.raw(BASE.OWNER,resumed=True,evidence=True)
    require(old['launch.json'].get('source_sha256')==sha_bytes(old_source) and launch.get('base_stage_owner')==reader.pointer(BASE.OWNER),'real original753 owner provenance required')
    count=len(control['completed_job_ids']);owners=result.get('actual_stage_owners')
    expected_owners=[dict(runtime=PREVIOUS,launch=control['previous_launch'],terminal=control['previous_terminal'],pid=old['launch.json']['pid'],
        start_ticks=old['launch.json']['start_ticks'],completed=count,resource_limits={k:old['launch.json'][k] for k in ('max_native_slots','max_GPU_slots','max_CPU_preflight_slots')}),
        dict(runtime=RUNTIME,launch=reader.pointer(RUNTIME+'/launch.json'),terminal=reader.pointer(RUNTIME+'/terminal.json'),pid=launch['pid'],
        start_ticks=launch['start_ticks'],completed=1920-count,resource_profile=control['resource_profile'])]
    require(owners==expected_owners,'two real evidence-tier owners and exact counts required')
    entry=P.validate_entry(control['execution_entry']);batch_sources=control['execution_implementation']['sources']
    require(control['execution_implementation']==P.execution_implementation(entry,batch_sources) and launch.get('batch_sources')==(batch_sources if entry['kind']=='batched901' else {}),'same qualified execution implementation required')
    for name,p in batch_sources.items():reader.raw(name,expected=pointer(p,name)['sha256'],resumed=True,evidence=True)
    verify_reporting_dependencies(reader)
    if entry['kind']=='batched901':
        small=reader.pointed(pointer(entry['batch_lock'],BATCH_LOCK),resumed=True,evidence=True)
        reader.pointed(pointer(entry['batch_control'],BATCH_CONTROL),resumed=True,evidence=True)
        require(small.get('schema')=='r10_scoped_frozen_lookup_bootstrap_inputs_v901' and small.get('execution_enabled') is True,'actual901 enabled supplement required')
        for name,p in batch_sources.items():require(reader.lookup(small['files'],name)==p['sha256'],'source must be fixed by both resume and901 locks')
    rows=plan['jobs'];tiers=result.get('evidence_tiers');cases=result['cases'];drivers=result['admission_driver_receipts']
    require(isinstance(tiers,list) and len(tiers)==1920 and len(rows)==len(cases)==len(drivers)==1920,'exact1920 mixed ordered evidence required')
    reducer=H.load_reducer(reader);selected=[];seen=set();seen_processes=set();batch_pointers=[]
    for row,tier,casep,driverp in zip(rows,tiers,cases,drivers):
        name=row['job_id'];runtime=row['evidence_runtime'];imported=row['imported_completed'];stage_owner=old['launch.json'] if imported else launch
        out=option(row['physical_argv'],'--out');expected_tier=dict(job_id=name,evidence_runtime=runtime,imported_completed=imported,
            actual_stage_owner_pid=stage_owner['pid'],actual_stage_owner_start_ticks=stage_owner['start_ticks'],previous_row_sha256=row['previous_row_sha256'],
            execution_entry=row['execution_entry'],case=casep,admission_driver_receipt=driverp)
        require(type(imported) is bool and runtime==(PREVIOUS if imported else RUNTIME) and tier==expected_tier and name not in seen,'exact unique mixed evidence tier required');seen.add(name)
        pointer(casep,out+'/case.json');pointer(driverp,out+'/admission_driver_receipt.json')
        require(row.get('admission_driver')==result['admission_driver'] and row.get('admission_guard')==result['admission_guard'],
            'every case must use the independently released original751/728 graph')
        cpu_leaf=runtime+'/CPU/'+name;native_leaf=runtime+'/native/'+name
        cpu=reader.json(cpu_leaf+'/CPU_preflight.json',evidence=True,resumed=imported)
        require(cpu==reader.json(cpu_leaf+'/stdout.log',evidence=True,resumed=imported) and cpu.get('complete') is True
            and cpu.get('native_started') is False and cpu.get('GPU_started') is False and cpu.get('actual_split')=='test'
            and cpu.get('execution_protocol_changed') is True,'actual CPU-only receipt/stdout required')
        expected={k:row[k] for k in H.BINDINGS}
        for leaf,check in ((cpu_leaf,True),(native_leaf,False)):
            child=reader.json(leaf+'/launch.json',evidence=True,resumed=imported);end=reader.json(leaf+'/terminal.json',evidence=True,resumed=imported);clean(end)
            process=(child.get('launcher_pid'),child.get('launcher_start_ticks'))
            require(all(type(v) is int and v>0 for v in process) and process not in seen_processes,'unique actual CPU/native PID/start required');seen_processes.add(process)
            require(child.get('argv')==row['argv']+(['--check'] if check else []) and child.get('check_only') is check
                and child.get('owner_pid')==stage_owner['pid'] and child.get('native_slots')==(0 if check else 1)
                and child.get('GPU_slots')==(0 if check else int(row['method']=='physmani'))
                and child.get('expected_identity_sha256')==(None if check else cpu['identity']['sha256'])
                and child.get('execution_protocol_changed') is True and all(child.get(k)==v and cpu.get(k)==v for k,v in expected.items()),'actual mixed child invocation/identity required')
            if imported:require(all(k not in child for k in ('execution_entry','physical_argv','inherited_frozen_environment','current_mount_namespace')),'old753 cannot invent new entry/namespace evidence')
            else:require(child.get('execution_entry')==row['execution_entry'] and child.get('physical_argv')==row['physical_argv']+(['--check'] if check else [])
                and child.get('inherited_frozen_environment')==launch['frozen_environment'] and child.get('current_mount_namespace')==launch['frozen_environment']['current_mount_namespace'],'actual new902 entry/forwarded argv/environment required')
            if not check:native_end=end
        record=status['completed'][name];clean(record)
        require(all(record.get(k)==v for k,v in native_end.items()) and (not imported or record==old['status.json']['completed'][name]),'completed native record must preserve its original outcome')
        P.require_batch_execution(cpu.get('batch_execution'),row,check=True,expected_sources={} if imported else batch_sources)
        case=reader.pointed(casep,evidence=True,resumed=imported);driver=reader.pointed(driverp,evidence=True,resumed=imported)
        require(native_end.get('case')==record.get('case')==casep and native_end.get('admission_driver_receipt')==record.get('admission_driver_receipt')==driverp
            and cpu.get('identity')==case.get('identity') and cpu.get('component_domain')==case.get('component_domain'),'native/CPU/case/751 receipt must agree')
        require(driver.get('schema')=='r10_explicit_author_outcome_formal_driver_v751' and driver.get('complete') is True
            and driver.get('actual_split')=='test' and driver.get('execution_protocol_changed') is True and driver.get('case')==casep
            and all(driver.get(k)==v for k,v in expected.items()),'actual original751 driver receipt required')
        if row['execution_entry']['kind']=='batched901':
            bp=pointer(native_end.get('batch_driver_receipt'),out+'/batch_driver_receipt.json')
            require(record.get('batch_driver_receipt')==bp,'actual901 sidecar must be in native terminal/status')
            value=reader.pointed(bp,evidence=True);P.require_batch_execution(value,row,check=False,expected_sources=batch_sources,case=casep,original_receipt=driverp);batch_pointers.append(bp)
        else:require('batch_driver_receipt' not in native_end and 'batch_driver_receipt' not in record,'old751 evidence cannot gain a901 sidecar')
        BASE.require_case(case,row,sources[row['task'],row['raw']],protocol,reader.inputs[BASE.SOURCE_LOCK])
        require(record.get('native_success') is case['official_task_success'] and record.get('motion_pass') is recorded_motion_flag(case),'status must retain native and diagnostic outcomes')
        metrics,diagnostics=reducer.reduce_saved_case(case);metrics={k:(v if reducer.finite(v) else None) for k,v in metrics.items()}
        selected.append(dict(task=row['task'],method=row['method'],source_raw_episode_id=row['raw'],source_row_sha256=row['source_row_sha256'],case=casep,
            admission_driver_receipt=driverp,evidence_runtime=runtime,imported_completed=imported,actual_owner_pid=stage_owner['pid'],actual_owner_start_ticks=stage_owner['start_ticks'],
            execution_entry=row['execution_entry'],CPU_receipt=reader.pointer(cpu_leaf+'/CPU_preflight.json'),native_launch=reader.pointer(native_leaf+'/launch.json'),
            native_terminal=reader.pointer(native_leaf+'/terminal.json'),official_task_success=case['official_task_success'],native_terminal_failure=case['native_terminal_failure'],
            metrics=metrics,recorded_window_diagnostics=diagnostics))
    require(result.get('batch_driver_receipts')==batch_pointers,'exact new901 sidecar cohort required')
    require(set(status['completed'])==seen and len(selected)==1920,'exact unique1920 completed cohort required')
    for m in METHODS:require(sum(r['official_task_success'] for r in selected if r['method']==m)==result['native_successes_by_method'][m],'all native outcomes must remain in cohort')
    require(terminal.get('native_successes')==sum(r['official_task_success'] for r in selected),'owner native total differs')
    subset=reader.json(SUBSET)
    if hasattr(reader,'close'):reader.close()
    return selected,result,subset,reducer


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--test-results-sha256',required=True);parser.add_argument('--resume-lock-sha256',required=True)
    parser.add_argument('--resume-control',default=CONTROL);parser.add_argument('--resume-lock',default=LOCK)
    args=parser.parse_args(argv);require(not (ROOT/OUTPUT).exists(),'preserve existing920 output; never overwrite')
    reader=Reader(ROOT);rows,result,subset,reducer=collect(reader,args.test_results_sha256,args.resume_control,args.resume_lock,args.resume_lock_sha256)
    source=str(Path(__file__).relative_to(ROOT));reader.inputs[source]=sha_bytes(Path(__file__).read_bytes())
    report=dict(schema=SCHEMA,complete=True,created_utc=datetime.now(timezone.utc).isoformat(),read_only_analysis=True,split='test',
        task_count=16,source_count=640,case_count=1920,methods=list(METHODS),task_source_counts={t:40 for t in TASKS},policy_sha256=POLICY,
        test_results=reader.pointer(RESULT),actual_test_auditor=dict(path=AUDITOR,sha256=result['source_sha256']),
        resume_control=reader.pointer(CONTROL),resume_control_lock=reader.pointer(LOCK),
        provenance_runtime_counts=dict(Counter(r['evidence_runtime'] for r in rows)),actual_stage_owners=result['actual_stage_owners'],
        metric_definitions=subset['metric_definitions'],statistical_definition=dict(subset['statistical_definition'],overall='pool640 equally weighted cases per method; never average task statistics'),
        diagnostic_definition=subset['diagnostic_definition'],hard_budget_disclosure=subset['hard_budget_disclosure'],
        common_RML_planner_configuration=dict(parameters=PARAMETERS,profile_sha256=PROFILE,cases=1920,parameter_kind='saved path-parameter planning configuration, not a per-joint measured bound'),
        resource_profile=result['resource_profile'],resource_qualification=result['resource_qualification'],execution_implementation=result['execution_implementation'],
        overall_by_method={m:reducer.summarize([r for r in rows if r['method']==m]) for m in METHODS},
        by_task_method=[dict(task=t,method=m,**reducer.summarize([r for r in rows if (r['task'],r['method'])==(t,m)])) for t in TASKS for m in METHODS],
        case_evidence=rows,inputs_sha256=reader.inputs,source_sha256=reader.inputs[source],
        audited_TEST_helper=dict(path=BASE_SOURCE,sha256=BASE_SHA),audited_VAL_helper=dict(path=BASE.HELPER,sha256=BASE.HELPER_SHA),
        reducer_source=dict(path=REDUCER,sha256=PINS[REDUCER]),pure_provenance_helpers=dict(path=OWNER,sha256=OWNER_SHA),
        reporting_dependencies=reader.reporting_dependencies,original_physical_locks_amended=False,
        namespace_evidence_scope='old753 retains original writer fields; new902 children inherit real902 proof;906 auditor capture binds907 stage PID/start',
        full1920_composite_TEST_results_and_complete907_observed=True,new_release_granted_by_this_summary=False,native_false_retained=True,
        success_rate_or_acceleration_filter_used=False,new_joint_differencing_used=False,alignment_motion_mirror_pooled=False,
        transitive_TRAIN_model_OS_inputs_walked=False,raw_reset_PKL_image_HDF5_contents_opened=False,TEST_used_for_fitting=False,
        native_GPU_or_jobs_started=False,frozen_source_or_lock_modified=False,TXT_or_handoff_written=False)
    with (ROOT/OUTPUT).open('x') as stream:stream.write(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(complete=True,output=OUTPUT,sha256=sha_bytes((ROOT/OUTPUT).read_bytes()),case_count=1920)))


if __name__=='__main__':main()
