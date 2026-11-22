"""Inert metadata/resource gate after actual three-tier TEST and saved motion.

Only an explicit enabled scope reads fixed completion metadata. No case, raw,
model, manifest or transitive input graph is opened. Frozen963/964 pure field
validators retain the published contract. Frozen924 is used only for its
canonical Permit registry, never for its obsolete full completion gate.
"""
from contextlib import ExitStack, contextmanager
from copy import deepcopy
import hashlib
import importlib
import json
import os
from pathlib import Path, PurePosixPath
from types import ModuleType
from weakref import WeakKeyDictionary

SOURCE='our_mani/multitask/post_composite_TEST_resource_guard_v965.py'
VERSION='r10_inert_post_complete_three_tier_TEST_and_motion_guard_v965'
LEGACY_SOURCE='our_mani/multitask/post_TEST_resource_guard_v924.py'
MOTION_SOURCE='scripts/summarize_composite_TEST_motion_v963_r10.py'
POST_SOURCE='scripts/continue_post_composite_TEST_motion_v964_r10.py'
CORE_SOURCE='our_mani/multitask/composite_TEST_resume_v955.py'
CONTROLLER_SOURCE='scripts/continue_composite_TEST_v958_r10.py'
OWNER_SOURCE='scripts/run_composite_TEST_owner_v956_r10.py'
AUDITOR_SOURCE='scripts/audit_composite_TEST_results_v957_r10.py'
SOURCE_SHA={
    LEGACY_SOURCE:'635d7e7a570982599ffb912a8cb65b5badb25907b4100677159955e4eb3afbed',
    MOTION_SOURCE:'08ecc1b5846422fa870e8f9939b8ff817573ded4ca67f4911a101d8283966ef3',
    POST_SOURCE:'f88b0e2c5a7b14b2b71295853b19636ecbbc9074d0c502337bd2cb6a60117a83',
    CORE_SOURCE:'e636ea89d27bb497bf716062cc6fa2ccdcbee5811e9aacd50c9e29e82dc431bc',
    CONTROLLER_SOURCE:'d4b2a52e8c4d5553bf1a6722f71ed415bb6882757c9c0c83e9fe7296c632bde6',
    OWNER_SOURCE:'03159361511fdae6eb3591d03afb6fabd9199fbfdf7b60dd65ac17e7e945d83d',
    AUDITOR_SOURCE:'97b7f2a20654877f89471b7752311ec8241525d0f64249e280e62fb692bb484a'}
CONTROLLER='runtime/framework/ordered_composite_TEST_v958'
TEST_RUNTIME='runtime/framework/formal_TEST_composite_resume_v956'
ORIGINAL='runtime/framework/formal_TEST_after_environment_resume_v753'
PARENT='runtime/framework/formal_TEST_concurrent_resume_v902'
PARENT_CONTROLLER='runtime/framework/ordered_concurrent_TEST_v907'
SETUP='artifacts/framework/multitask_r10/formal_TEST_composite_resume_v956_setup'
CONTROL=SETUP+'/resume_control.json';CONTROL_LOCK=SETUP+'/resume_inputs.json'
PARENT_SETUP='artifacts/framework/multitask_r10/formal_TEST_concurrent_resume_v902_setup'
PARENT_CONTROL=PARENT_SETUP+'/resume_control.json';PARENT_LOCK=PARENT_SETUP+'/resume_inputs.json'
ART='artifacts/framework/multitask_r10/formal_TEST_composite_resume_v956'
RESULT=ART+'/TEST_results.json'
OWNER_STAGE=CONTROLLER+'/stages/formal_TEST_1920'
AUDIT_STAGE=CONTROLLER+'/stages/independent_TEST_results'
POST_RUNTIME='runtime/framework/post_complete_composite_TEST_motion_v964'
POST_SETUP='artifacts/framework/multitask_r10/post_complete_composite_TEST_motion_v964_setup'
POST_CONTROL=POST_SETUP+'/control.json';POST_LOCK=POST_SETUP+'/inputs.json'
POST_STAGE=POST_RUNTIME+'/stages/saved_motion_v963'
MOTION=TEST_RUNTIME+'/full_sixteen_task_motion_summary_v963.json'
POST_PATHS=dict(control=POST_CONTROL,input_lock=POST_LOCK,launch=POST_RUNTIME+'/launch.json',
    status=POST_RUNTIME+'/status.json',terminal=POST_RUNTIME+'/terminal.json',
    analysis_launch=POST_STAGE+'/launch.json',analysis_terminal=POST_STAGE+'/terminal.json',analysis_stdout=POST_STAGE+'/stdout.log')
POLICY='eb8f768d99975a7c2de2c36de5283d64898c2296f33fb342e332310eb32db6ef'
LOCKS=(CONTROLLER+'.lock',PARENT_CONTROLLER+'.lock','runtime/framework/ordered_resume_VAL_TEST_v775.lock',
    'runtime/framework/ordered_env_resume_VAL_TEST_v827.lock','runtime/framework/fixed_TRAIN104_geometry_native.lock',POST_RUNTIME+'.lock')
STOP_PATHS=('runtime/framework/multitask_r10_STOP',CONTROLLER+'/STOP',TEST_RUNTIME+'/STOP',POST_RUNTIME+'/STOP',
    PARENT_CONTROLLER+'/STOP',PARENT+'/STOP',ORIGINAL+'/STOP','runtime/framework/ordered_env_resume_VAL_TEST_v827/STOP',
    'runtime/framework/formal_VAL_env_resume_v823/STOP','runtime/framework/ordered_resume_VAL_TEST_v775/STOP',
    'runtime/framework/formal_VAL_resume_v772/STOP','runtime/framework/concurrent_TRAIN_resource_canary_v903/STOP',
    'runtime/framework/post_TEST_resource_guard_v924/STOP','runtime/framework/post_composite_TEST_resource_guard_v965/STOP')
EXTRA_PREFIXES=('run_composite_TEST_owner_','continue_composite_TEST_','audit_composite_TEST_results_',
    'continue_post_composite_TEST_motion_','summarize_composite_TEST_motion_','continue_post_TEST_motion_',
    'summarize_full_TEST_motion_')
_LEASES=WeakKeyDictionary()


def require(value,message):
    if not value:raise ValueError('post composite TEST965: '+message)


def valid_sha(value):return isinstance(value,str) and len(value)==64 and not set(value)-set('0123456789abcdef')
def sha(raw):return hashlib.sha256(raw).hexdigest()
def _pointer(value,name=None):
    require(isinstance(value,dict) and set(value)=={'path','sha256'} and isinstance(value.get('path'),str)
        and value['path'] and not Path(value['path']).is_absolute() and '..' not in PurePosixPath(value['path']).parts
        and str(PurePosixPath(value['path']))==value['path'] and valid_sha(value.get('sha256'))
        and (name is None or value['path']==name),'exact canonical project pointer required')
    return value


def _allowed(name):
    if name in {*SOURCE_SHA,SOURCE,CONTROL,CONTROL_LOCK,PARENT_CONTROL,PARENT_LOCK,RESULT,MOTION,*POST_PATHS.values()}:return True
    if name in {'runtime/framework/concurrent_TRAIN_resource_qualification_v904.json',
        'runtime/framework/concurrent_TRAIN_resource_audit_v904/launch.json',
        'runtime/framework/frozen_TEST_environment_recheck_v954.json'}:return True
    sources=('scripts/run_formal_suite16_owner_v753_r10.py','scripts/run_concurrent_resumed_formal_TEST_owner_v902_r10.py',
        'scripts/continue_concurrent_TEST_v907_r10.py','scripts/continue_environment_resumed_VAL_TEST_v827_r10.py',
        'scripts/run_environment_resumed_formal_VAL_owner_v823_r10.py','scripts/audit_concurrent_TRAIN_resource_canary_v904_r10.py',
        'scripts/check_frozen_TEST_environment_v954_r10.py')
    if name in sources:return True
    runtimes=(CONTROLLER,TEST_RUNTIME,ORIGINAL,PARENT,PARENT_CONTROLLER,'runtime/framework/ordered_env_resume_VAL_TEST_v827')
    if any(name==runtime+'/'+leaf for runtime in runtimes for leaf in ('launch.json','plan.json','status.json','terminal.json','failure.json','DRAIN')):return True
    return name=='runtime/framework/ordered_env_resume_VAL_TEST_v827/stages/environment_resume_VAL_945/launch.json' or any(
        name==stage+'/'+leaf for stage in (OWNER_STAGE,AUDIT_STAGE) for leaf in ('launch.json','terminal.json','stdout.log'))


class _Metadata:
    def __init__(self,root,callback):self.root,self.callback,self.inputs,self.signatures=Path(root).resolve(),callback,{},{}
    def raw(self,name,expected=None):
        require(isinstance(name,str) and _allowed(name) and not Path(name).is_absolute()
            and '..' not in PurePosixPath(name).parts and str(PurePosixPath(name))==name,'non-metadata read forbidden')
        if self.callback is not None:raw=self.callback(name)
        else:
            path=self.root/name;require(path.resolve(strict=True)==path,'metadata alias refused')
            signature=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
            before=signature(path.stat());raw=path.read_bytes()
            require(before==signature(path.stat()) and (name not in self.signatures or self.signatures[name]==before),'metadata identity changed')
            self.signatures[name]=before
        require(type(raw) is bytes,'exact bytes required');actual=sha(raw)
        require((expected is None or valid_sha(expected) and actual==expected)
            and (name not in self.inputs or self.inputs[name]==actual),'metadata/source SHA changed: '+name)
        self.inputs[name]=actual;return raw
    def json(self,name,expected=None):
        value=json.loads(self.raw(name,expected));require(isinstance(value,dict),'JSON object required');return value
    def pointer(self,name):return dict(path=name,sha256=self.inputs[name])
    def pointed(self,p,name=None):_pointer(p,name);return self.json(p['path'],p['sha256'])
    def close(self):
        for name,expected in tuple(self.inputs.items()):self.raw(name,expected)


def _legacy(root):
    """One canonical registry shared with frozen934, never a second exec copy."""
    path=Path(root)/LEGACY_SOURCE
    require(path.resolve(strict=True)==path,'canonical924 source required')
    before=path.stat();raw=path.read_bytes();after=path.stat()
    signature=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
    require(signature(before)==signature(after) and sha(raw)==SOURCE_SHA[LEGACY_SOURCE],'frozen924 source changed')
    module=importlib.import_module('our_mani.multitask.post_TEST_resource_guard_v924')
    require(Path(module.__file__).resolve(strict=True)==path,'shared canonical924 module identity required')
    return module


def _contracts(reader):
    for name,expected in SOURCE_SHA.items():reader.raw(name,expected)
    legacy=_legacy(reader.root)
    def load(name):
        module=ModuleType('post965_'+Path(name).stem);module.__file__=str(reader.root/name)
        exec(compile(reader.raw(name,SOURCE_SHA[name]),str(reader.root/name),'exec'),module.__dict__);return module
    return legacy,load(MOTION_SOURCE),load(POST_SOURCE)


def _postprocessing(r,old,post,control,cp,lp,completion,motion_sha,post_lock_sha):
    lock=r.json(POST_LOCK,post_lock_sha)
    require(lock.get('schema')==post.LOCK_SCHEMA and lock.get('execution_enabled') is True,'real964 small activation lock required')
    files=lock.get('files');names={*post.SOURCES,POST_CONTROL,CONTROL,CONTROL_LOCK,CONTROLLER+'/launch.json'}
    require(isinstance(files,dict) and len(files)==len(names),'exact11 postprocessing bindings required')
    canonical={}
    for key,value in files.items():
        require(isinstance(key,str),'small-map path required');path=Path(key)
        name=str(path.relative_to(r.root)) if path.is_absolute() and path.is_relative_to(r.root) else key
        require(name in names and name not in canonical and valid_sha(value),'postprocessing alias/extra/non-SHA refused')
        canonical[name]=value
    require(set(canonical)==names,'exact postprocessing source set required')
    ctl=r.json(POST_CONTROL,canonical[POST_CONTROL]);pcp,plp=r.pointer(POST_CONTROL),r.pointer(POST_LOCK)
    require(ctl.get('schema')==post.SCHEMA and ctl.get('execution_enabled') is True and ctl.get('read_only_analysis') is True
        and ctl.get('automatic_retry') is False and ctl.get('policy_sha256')==POLICY and type(ctl.get('poll_seconds')) is int
        and 1<=ctl['poll_seconds']<=30,'frozen read-only964 control required')
    fixed=dict(runtime=POST_RUNTIME,controller_runtime=CONTROLLER,owner_runtime=TEST_RUNTIME,test_results=RESULT,motion_summary=MOTION,
        resume_control=cp,resume_control_lock=lp,controller_launch=r.pointer(CONTROLLER+'/launch.json'))
    require(all(ctl.get(k)==v for k,v in fixed.items()),'actual complete TEST and postprocessing batch must match')
    sources=ctl.get('sources');require(isinstance(sources,dict) and set(sources)==set(post.SOURCES),'seven source/test bindings required')
    for name,p in sources.items():
        _pointer(p,name);require(canonical[name]==p['sha256'],'post source differs from lock')
        if name in SOURCE_SHA:require(p['sha256']==SOURCE_SHA[name],'reviewed postprocessing source changed')
        # Two CPU-test sources are declared/pinned metadata, not executable dependencies.
    for name in (CONTROL,CONTROL_LOCK,CONTROLLER+'/launch.json'):
        require(canonical[name]==r.inputs[name],'selected complete batch differs from post small lock')
    launch,state,end=(r.json(POST_RUNTIME+'/'+n+'.json') for n in ('launch','status','terminal'))
    known=[old._pid(launch)]
    host=dict(standard_host_analysis=True,frozen_environment_inherited_claimed=False,automatic_retry=False)
    require(all(launch.get(k) is v for k,v in host.items()) and launch.get('source_sha256')==SOURCE_SHA[POST_SOURCE]
        and launch.get('control')==pcp and launch.get('input_lock')==plp
        and launch.get('monitored_controller_launch')==ctl['controller_launch'],'real host964 identity and batch required')
    argv=launch.get('argv');require(isinstance(argv,list) and argv and all(type(v) is str for v in argv)
        and argv[0] in (POST_SOURCE,str(r.root/POST_SOURCE)),'real964 argv required')
    import argparse
    parser=argparse.ArgumentParser(add_help=False,allow_abbrev=False);parser.add_argument('--execute',action='store_true')
    parser.add_argument('--control',default=POST_CONTROL);parser.add_argument('--lock',default=POST_LOCK);parser.add_argument('--lock-sha256')
    try:args=parser.parse_args(argv[1:])
    except SystemExit as error:raise ValueError('invalid actual964 argv') from error
    require(args.execute and argv.count('--execute')==1 and argv.count('--lock-sha256')==1
        and argv.count('--control')<=1 and argv.count('--lock')<=1 and args.lock_sha256==post_lock_sha
        and args.control==POST_CONTROL and args.lock==POST_LOCK,'external964 lock pin required')
    sp=dict(path=MOTION,sha256=motion_sha)
    require(state.get('stage')=='complete' and state.get('automatic_retry') is False and state.get('motion_summary')==sp
        and end.get('schema')=='r10_readonly_post_complete_composite_TEST_motion_terminal_v964' and end.get('complete') is True
        and end.get('motion_summary')==sp and end.get('TEST_results')==completion['result']
        and end.get('control')==pcp and end.get('input_lock')==plp and end.get('resume_control')==cp and end.get('resume_control_lock')==lp
        and all(end.get(k) is v for k,v in {**host,'Native_GPU_TRAIN_started':False,'TXT_modified':False}.items()),'actual complete964 final required before summary read')
    child,child_end=r.json(POST_STAGE+'/launch.json'),r.json(POST_STAGE+'/terminal.json');known.append(old._pid(child));old._clean_stage(child_end)
    expected=[str(r.root/'.conda-env/bin/python'),'-B','-u',MOTION_SOURCE,'--test-results-sha256',completion['result_sha256'],
        '--resume-lock-sha256',lp['sha256'],'--resume-control',CONTROL,'--resume-lock',CONTROL_LOCK]
    require(child.get('argv')==expected and child.get('source')==sources[MOTION_SOURCE]
        and child.get('source_sha256')==SOURCE_SHA[MOTION_SOURCE] and child.get('owner_pid')==launch['pid']
        and child.get('owner_start_ticks')==launch['start_ticks'] and child.get('control')==pcp and child.get('input_lock')==plp
        and child.get('TEST_results')==completion['result'] and child_end.get('interrupted') is False
        and all(child.get(k) is v for k,v in host.items()),'real host963 child source/PID/argv/batch required')
    summary=r.json(MOTION,motion_sha);stdout=r.json(POST_STAGE+'/stdout.log')
    post.validate_summary(summary,sp,stdout,ctl,dict(sources=sources),completion,r.root)
    return known,{key:r.pointer(name) for key,name in POST_PATHS.items()}


def require_complete_TEST_metadata(root,expected_test_result_sha256,*,resume_lock_sha256,motion_summary_sha256,
    postprocessing_lock_sha256,read_bytes=None):
    """Complete small metadata only; no numerical score or motion threshold."""
    require(all(valid_sha(v) for v in (expected_test_result_sha256,resume_lock_sha256,motion_summary_sha256,postprocessing_lock_sha256)),
        'external957 result,955 lock,963 summary and964 lock SHA pins required')
    r=_Metadata(root,read_bytes);old,m,post=_contracts(r);C=m.C;P=m.P
    locked=r.json(CONTROL_LOCK,resume_lock_sha256)
    require(locked.get('schema')==C.LOCK_SCHEMA and locked.get('approved_for_execution') is True,'actual955 frozen lock required')
    files=locked.get('files');control=r.json(CONTROL,old._lookup(files,r.root,CONTROL));cp,lp=r.pointer(CONTROL),r.pointer(CONTROL_LOCK)
    require(locked.get('control')==cp and control.get('schema')==C.CONTROL_SCHEMA and control.get('split')=='test'
        and control.get('expected_cases')==1920 and control.get('source_count')==640 and control.get('policy_sha256')==POLICY
        and control.get('author_outcome_protocol')==old.PROTOCOL and control.get('resource_profile')==old.PROFILE_LIMITS
        and control.get('user_authorized_resume') is True and control.get('standing_authorization_reused') is True
        and control.get('new_user_event_invented') is False and control.get('physical_policy_changed') is False,'actual same-policy955 continuation required')
    for key,name in (('resume_controller',CONTROLLER_SOURCE),('resume_owner',OWNER_SOURCE),('resume_auditor',AUDITOR_SOURCE),('result_auditor',AUDITOR_SOURCE)):
        require(control.get(key)==dict(path=name,sha256=SOURCE_SHA[name]) and old._lookup(files,r.root,name)==SOURCE_SHA[name],'frozen actual writer binding required')
    for key,field in (('environment_wrapper','wrapper'),('environment_manifest','manifest'),('environment_namespace_proof','namespace_qualification')):
        require(control.get(key)==old.ENVIRONMENT_POINTERS[field],'original qualified820 binding changed')
    entry,implementation=old._entry(control,files,r.root)
    controller,state,final=(r.json(CONTROLLER+'/'+n+'.json') for n in ('launch','status','terminal'));known=[old._pid(controller)]
    require(controller.get('source_sha256')==SOURCE_SHA[CONTROLLER_SOURCE] and controller.get('resume_control')==cp
        and controller.get('resume_control_lock')==lp and controller.get('automatic_retry') is False,'actual958 launch required')
    env=old._environment(controller.get('frozen_environment'),controller)
    require(state.get('stage')=='complete' and state.get('automatic_retry') is False and state.get('frozen_environment')==env
        and not state.get('infrastructure_failure_detected') and state.get('owner_error') is None
        and final.get('complete') is True and final.get('automatic_retry') is False
        and _pointer(final.get('TEST_results'),RESULT)['sha256']==expected_test_result_sha256,'complete actual958 required before result read')
    owner_args=[]
    for key,flag in (('suite','--suite'),('suite_lock','--suite-lock'),('source_inputs','--source-inputs'),('validation_release','--validation-release')):
        owner_args.extend((flag,_pointer(control[key])['path']))
    owner_args+=['--runtime',TEST_RUNTIME,'--out',ART+'/test','--resume-control',CONTROL,'--resume-lock',CONTROL_LOCK,'--resume-lock-sha256',resume_lock_sha256]
    audit_args=['--resume-control',CONTROL,'--resume-lock',CONTROL_LOCK,'--resume-lock-sha256',resume_lock_sha256];stages={}
    for stage,source,args in ((OWNER_STAGE,OWNER_SOURCE,owner_args),(AUDIT_STAGE,AUDITOR_SOURCE,audit_args)):
        launch,end=r.json(stage+'/launch.json'),r.json(stage+'/terminal.json');known.append(old._pid(launch))
        m.require_stage(launch,end,[str(r.root/'.conda-env/bin/python'),'-B','-u',source,*args],SOURCE_SHA[source],controller);stages[source]=launch
    launch,plan,status,terminal=(r.json(TEST_RUNTIME+'/'+n+'.json') for n in ('launch','plan','status','terminal'))
    m.require_owner_completion(launch,plan,status,terminal,stages[OWNER_SOURCE],control,cp,lp)
    require(launch.get('execution_entry')==entry,'unchanged qualified901 entry required');old._row_grid(plan)
    # The first957 result read is after complete958 and clean full1920 owner/stages.
    result=r.json(RESULT,expected_test_result_sha256);m.require_result(result,control,cp,lp,stages[AUDITOR_SOURCE])
    evidence=result.get('inputs_sha256')
    required=(CONTROL,CONTROL_LOCK,CORE_SOURCE,OWNER_SOURCE,AUDITOR_SOURCE,*[TEST_RUNTIME+'/'+n+'.json' for n in ('launch','plan','status','terminal')])
    require(all(old._lookup(evidence,r.root,n)==r.inputs[n] for n in required),'actual957 metadata ledger binding changed')
    require(old._lookup(evidence,r.root,control['source_inputs']['path'])==control['source_inputs']['sha256'],'original640 input lock must remain bound, not opened')
    stdout=r.json(AUDIT_STAGE+'/stdout.log')
    require(stdout.get('complete') is True and stdout.get('schema')==m.RESULT_SCHEMA and stdout.get('split')=='test'
        and stdout.get('cases')==1920 and stdout.get('out')==RESULT and stdout.get('motion_is_diagnostic_only') is True
        and stdout.get('native_successes_by_method')==result.get('native_successes_by_method'),'actual957 stdout required')
    def frozen(p,name=None,*,audited=True):
        _pointer(p,name);require(old._lookup(files,r.root,p['path'])==p['sha256']
            and (not audited or old._lookup(evidence,r.root,p['path'])==p['sha256']),'frozen historical metadata and actual957 evidence required')
        return r.pointed(p)
    prior=frozen(control['previous_resume_control'],PARENT_CONTROL);prior_lock=frozen(control['previous_resume_control_lock'],PARENT_LOCK)
    require(prior_lock.get('approved_for_execution') is True and prior_lock.get('schema')==P.LOCK_SCHEMA
        and prior_lock.get('control')==control['previous_resume_control'],'actual915 provenance required')
    histories={}
    for runtime,prefix in ((ORIGINAL,'original_'),(PARENT,'previous_')):
        histories[runtime]={n:frozen(control[prefix+n],runtime+'/'+n+'.json') for n in ('launch','plan','status','terminal')}
        if runtime==ORIGINAL:require(all(control[prefix+n]==prior['previous_'+n] for n in histories[runtime]),'original753 tier differs from915')
    original,parent=histories[ORIGINAL],histories[PARENT]
    P.require_partition(prior,original['plan'],original['status'],original['terminal'])
    require(parent['plan'].get('jobs')==P.resumed_jobs(original['plan'],prior,prior['execution_entry']) and entry==prior['execution_entry'],'literal753 to902 provenance and unchanged901 controls required')
    C.require_resume_plan(control,parent['plan'],parent['status'],parent['terminal'],plan)
    require(all(status['completed'].get(k)==v for k,v in parent['status']['completed'].items())
        and all(parent['status']['completed'].get(k)==v for k,v in original['status']['completed'].items()),'all historical False/unknown records must remain literal')
    recheck=frozen(control['environment_recheck'],'runtime/framework/frozen_TEST_environment_recheck_v954.json');C.require_environment_recheck(recheck,control)
    for key,leaf in (('previous_controller_launch','launch.json'),('previous_controller_status','status.json'),('previous_controller_failure','failure.json')):
        record=frozen(control[key],PARENT_CONTROLLER+'/'+leaf)
        if leaf=='launch.json':known.append(old._pid(record));require(record.get('source_sha256')==old.SOURCE_SHA[old.CONTROLLER_SOURCE],'real old907 source required')
        elif leaf=='status.json':require(record.get('stage')=='requires_concrete_repair' and record.get('automatic_retry') is False,'preserve real907 repair status')
        else:require(record.get('complete') is False and record.get('user_stop') is False and record.get('automatic_retry') is False,'never invent successful907 terminal')
    frozen(control['previous_DRAIN'],PARENT+'/DRAIN')
    owners={ORIGINAL:original['launch'],PARENT:parent['launch'],TEST_RUNTIME:launch}
    m.require_released_cohort(plan,result,status,owners,set(control['completed_job_ids']))
    counts,stage_owners=post.require_three_tier_evidence(result,control,status,r.pointer(TEST_RUNTIME+'/launch.json'),r.pointer(TEST_RUNTIME+'/terminal.json'))
    for runtime,source in ((ORIGINAL,old.OLD_OWNER_SOURCE),(PARENT,old.OWNER_SOURCE)):
        known.append(old._pid(owners[runtime]));r.raw(source,old.SOURCE_SHA[source])
        require(owners[runtime].get('source_sha256')==old.SOURCE_SHA[source],'actual historical owner source required')
    expected_owners=[dict(runtime=runtime,launch=r.pointer(runtime+'/launch.json'),terminal=r.pointer(runtime+'/terminal.json'),
        **old._pid(owners[runtime]),completed=counts[runtime],resource_limits={k:owners[runtime][k] for k in
        ('max_native_slots','max_GPU_slots','max_CPU_preflight_slots')}) for runtime in (ORIGINAL,PARENT,TEST_RUNTIME)]
    require(stage_owners==expected_owners,'three real physical owner sources and limits required')
    # Additional known ancestors/qualification use original915 pointers, never fake907 success.
    for key,leaf in (('previous_controller_launch','launch.json'),('previous_controller_status','status.json'),('previous_controller_failure','failure.json')):
        record=frozen(prior[key],old.OLD_CONTROLLER+'/'+leaf)
        if leaf=='launch.json':known.append(old._pid(record));require(record.get('source_sha256')==old.SOURCE_SHA[old.OLD_CONTROLLER_SOURCE],'real827 source required')
        elif leaf=='status.json':require(record.get('stage')=='requires_concrete_repair' and record.get('automatic_retry') is False,'preserve real827 repair status')
        else:require(record.get('complete') is False and record.get('user_stop') is False and record.get('automatic_retry') is False,'never invent successful827 terminal')
    historic=r.json(old.VAL_OWNER_STAGE+'/launch.json');known.append(old._pid(historic))
    require(historic.get('source_sha256')==old.SOURCE_SHA[old.VAL_OWNER_SOURCE]
        and old.option(historic.get('argv'),'--split')=='validation','real historical823 VAL stage required')
    q=frozen(control['resource_qualification'],old.QUALIFICATION)
    require(q.get('schema')=='r10_concurrent_TRAIN_resource_qualification_v902' and q.get('complete') is True and q.get('split')=='train'
        and type(q.get('case_count')) is int and q['case_count']==6 and q.get('source_sha256')==old.SOURCE_SHA[old.QUALIFIER_SOURCE]
        and q.get('resource_profile')==old.PROFILE_LIMITS and q.get('execution_entry')==implementation
        and q.get('policy_sha256')==POLICY and q.get('author_outcome_protocol')==old.PROTOCOL
        and type(q.get('infrastructure_failed')) is int and q['infrastructure_failed']==0
        and all(q.get(k) is True for k in ('cleanup_completed','resource_bounds_verified','overlap_verified','independent_audit_complete'))
        and type(q.get('observed_native_peak')) is int and q['observed_native_peak']==4
        and type(q.get('observed_GPU_peak')) is int and q['observed_GPU_peak']==2
        and q.get('formal_score_contribution') is False and q.get('TEST_read') is False,'independent unchanged904 resource qualification required')
    #957 consumes the qualification body; its904 launch is frozen by955 but is
    #not itself consumed by957. Never invent that absent ledger membership.
    qa=frozen(q.get('actual_independent_audit'),old.QUALIFICATION_LAUNCH,audited=False);known.append(old._pid(qa))
    require(qa.get('source_sha256')==old.SOURCE_SHA[old.QUALIFIER_SOURCE],'actual904 audit identity required')
    completion=dict(result=r.pointer(RESULT),result_sha256=expected_test_result_sha256,provenance_runtime_counts=counts,actual_stage_owners=stage_owners)
    post_known,post_pointers=_postprocessing(r,old,post,control,cp,lp,completion,motion_summary_sha256,postprocessing_lock_sha256);known+=post_known
    require(len({(p['pid'],p['start_ticks']) for p in known})==len(known),'all eleven known process identities must be distinct')
    r.close()
    return dict(schema=VERSION,complete=True,expected_cases=1920,source_cases=640,task_count=16,policy_sha256=POLICY,
        TEST_results=r.pointer(RESULT),resume_control=cp,resume_control_lock=lp,motion_summary=r.pointer(MOTION),
        postprocessing_input_lock=r.pointer(POST_LOCK),postprocessing=post_pointers,actual_controller=r.pointer(CONTROLLER+'/launch.json'),
        actual_controller_terminal=r.pointer(CONTROLLER+'/terminal.json'),actual_owner=r.pointer(TEST_RUNTIME+'/launch.json'),
        actual_owner_terminal=r.pointer(TEST_RUNTIME+'/terminal.json'),actual_audit_stage=r.pointer(AUDIT_STAGE+'/launch.json'),
        actual_stage_owners=stage_owners,provenance_runtime_counts=counts,known_processes=known,inputs_sha256=dict(r.inputs),
        historical_infrastructure_attempts=deepcopy(result['historical_infrastructure_attempts']),
        source_case_bytes_read=False,source_manifest_or_raw_bytes_read=False,transitive_input_map_walked=False,
        scores_used_for_gate=False,success_rate_threshold_used=False,motion6_used_for_gate=False,new_release_granted=False,
        old907_success_terminal_required=False,old929_success_terminal_required=False,
        legacy924_compatibility_scope='Permit registration only; no old924 completion or resource gate called',
        gate_chain='actual958/956/957 then actual964/963',registration_only_compatibility=True)


def _stop_guard(root):
    if any((Path(root)/name).exists() for name in STOP_PATHS):raise InterruptedError('STOP prevents a new TRAIN diagnostic')


@contextmanager
def _reserve(root):
    import fcntl
    with ExitStack() as stack:
        for name in LOCKS:
            path=Path(root)/name
            require(path.parent.resolve()==path.parent and path.parent.is_relative_to(Path(root).resolve()) and not path.is_symlink(),'canonical lock path required')
            stream=stack.enter_context(path.open('a+'))
            try:fcntl.flock(stream.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError as error:raise RuntimeError('formal/global resource lock busy: '+name) from error
        yield


def _is_alive(record):
    require(type(record.get('pid')) is int and type(record.get('start_ticks')) is int,'real process identity required')
    try:fields=Path('/proc',str(record['pid']),'stat').read_text().rsplit(')',1)[1].split()
    except (FileNotFoundError,ProcessLookupError):return False
    return fields[0] not in ('Z','X') and int(fields[19])==record['start_ticks']


def _inventory(root):
    """Original924 inventory algorithm with new controller/analysis names added."""
    old=_legacy(root);prefixes=old.NATIVE_PREFIXES+EXTRA_PREFIXES
    root,proc,uid=Path(root).resolve(),Path('/proc'),os.getuid()
    def stat(path):
        f=(path/'stat').read_text().rsplit(')',1)[1].split();return f[0],int(f[1]),int(f[19])
    own,parent=set(),os.getpid()
    while parent>0 and parent not in own:own.add(parent);parent=stat(proc/str(parent))[1]
    comm=(proc/'1/comm').read_text().strip();active,unresolved=[],[]
    for entry in proc.iterdir():
        if not entry.name.isdigit() or int(entry.name) in own:continue
        try:
            before=stat(entry)
            if before[0] in ('Z','X'):continue
            process_uid=entry.stat().st_uid
            try:tokens=(entry/'cmdline').read_bytes().decode(errors='replace').split('\0')
            except PermissionError:
                if process_uid==uid:unresolved.append(dict(pid=int(entry.name),reason='same-UID cmdline unreadable'))
                continue
            matches=[Path(x).name for x in tokens if '\n' not in x and '\r' not in x and Path(x).name.endswith('.py') and Path(x).name.startswith(prefixes)]
            if not matches or process_uid!=uid:continue
            cwd=Path(os.readlink(entry/'cwd')).resolve();after=stat(entry)
            if before[2]==after[2] and after[0] not in ('Z','X') and cwd==root:active.append(dict(pid=int(entry.name),start_ticks=after[2],scripts=matches))
        except (FileNotFoundError,ProcessLookupError):continue
        except (OSError,ValueError,IndexError) as error:unresolved.append(dict(pid=int(entry.name),reason=repr(error)))
    return dict(host_visibility_available=comm in ('systemd','init'),same_uid_inventory_incomplete=bool(unresolved),live_project_native_processes=active,
        unresolved_processes=unresolved,excluded_self_and_ancestor_pids=sorted(own),ancestor_chain_complete=parent==0,host_namespace_links_claimed_readable=False)


def require_permit(permit,root,*,allow_test_callbacks=False):
    # Do not call a forged object's __hash__/__eq__ while rejecting it.
    owned=next((key for key in _LEASES if key is permit),None)
    require(owned is not None,'live965-owned permit required; old924 permit is insufficient')
    lease=_LEASES[owned]
    require(lease['pid']==os.getpid() and lease['root']==Path(root).resolve(),'same-process canonical-root965 lease required')
    return lease['legacy'].require_permit(permit,root,allow_test_callbacks=allow_test_callbacks)


@contextmanager
def reserved_resources_after_TEST(root,expected_test_result_sha256,*,resume_lock_sha256,motion_summary_sha256,
    postprocessing_lock_sha256,enabled=False,read_bytes=None,stop_check=None,reserve=None,inventory=None,is_alive=None):
    """Hold through original worker cleanup; disabled construction is inert."""
    require(enabled is True,'explicit enabled=True required; default remains inert')
    require(all(valid_sha(v) for v in (expected_test_result_sha256,resume_lock_sha256,motion_summary_sha256,postprocessing_lock_sha256)),'all four external pins required')
    root=Path(root).resolve();test_only=any(v is not None for v in (read_bytes,stop_check,reserve,inventory,is_alive))
    stop,hold,scan,alive=stop_check or _stop_guard,reserve or _reserve,inventory or _inventory,is_alive or _is_alive
    stop(root)
    with hold(root):
        stop(root)
        proof=require_complete_TEST_metadata(root,expected_test_result_sha256,resume_lock_sha256=resume_lock_sha256,
            motion_summary_sha256=motion_summary_sha256,postprocessing_lock_sha256=postprocessing_lock_sha256,read_bytes=read_bytes)
        require(not any(alive(p) for p in proof['known_processes']),'formal/physical/analysis/resource-audit process still live')
        snapshot=scan(root)
        require(isinstance(snapshot,dict) and snapshot.get('host_visibility_available') is True and snapshot.get('ancestor_chain_complete') is True
            and snapshot.get('same_uid_inventory_incomplete') is False and snapshot.get('live_project_native_processes')==[],'complete empty host-visible inventory required')
        stop(root);legacy=_legacy(root);permit=legacy._Permit()
        record=dict(pid=os.getpid(),root=root,proof=deepcopy(proof),stop=stop,test_only=test_only)
        legacy._LEASES[permit]=record;_LEASES[permit]=dict(pid=os.getpid(),root=root,legacy=legacy)
        try:yield permit
        finally:_LEASES.pop(permit,None);legacy._LEASES.pop(permit,None)
