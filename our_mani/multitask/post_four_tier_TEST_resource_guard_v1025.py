"""Inert metadata/resource gate after actual four-tier TEST and saved motion.

Only an explicit enabled scope reads fixed completion metadata. No case, raw,
model, manifest or transitive input graph is opened. Frozen1015/1016 pure field
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

SOURCE='our_mani/multitask/post_four_tier_TEST_resource_guard_v1025.py'
VERSION='r10_inert_post_complete_four_tier_TEST_and_motion_guard_v1025'
LEGACY_SOURCE='our_mani/multitask/post_TEST_resource_guard_v924.py'
MOTION_SOURCE='scripts/summarize_composite_TEST_motion_v1015_r10.py'
POST_SOURCE='scripts/continue_post_composite_TEST_motion_v1016_r10.py'
CORE_SOURCE='our_mani/multitask/composite_TEST_resume_v1010.py'
CONTROLLER_SOURCE='scripts/continue_composite_TEST_v1013_r10.py'
OWNER_SOURCE='scripts/run_composite_TEST_owner_v1011_r10.py'
AUDITOR_SOURCE='scripts/audit_composite_TEST_results_v1012_r10.py'
PARENT_CORE_SOURCE='our_mani/multitask/composite_TEST_resume_v955.py'
PARENT_OWNER_SOURCE='scripts/run_composite_TEST_owner_v956_r10.py'
PARENT_CONTROLLER_SOURCE='scripts/continue_composite_TEST_v958_r10.py'
QUALIFIER_SOURCE='scripts/audit_concurrent_TRAIN_resource_canary_v1005_r10.py'
QUALIFICATION='runtime/framework/concurrent_TRAIN_resource_qualification_v1005.json'
QUALIFICATION_LAUNCH='runtime/framework/concurrent_TRAIN_resource_audit_v1005/launch.json'
CANARY='runtime/framework/concurrent_TRAIN_resource_canary_v1005'
PROFILE=dict(schema='r10_frozen_concurrent_native_resource_profile_v1005',max_native_slots=8,max_GPU_slots=3,max_CPU_preflight_slots=4,ready_limit=16)
SOURCE_SHA={'our_mani/multitask/post_TEST_resource_guard_v924.py': '635d7e7a570982599ffb912a8cb65b5badb25907b4100677159955e4eb3afbed',
 'scripts/summarize_composite_TEST_motion_v1015_r10.py': '860c2c639154e35386c13ec2cbc9bbbef0d66463c0fb3834ff5ccc577f55b9f5',
 'scripts/continue_post_composite_TEST_motion_v1016_r10.py': '86b47f8c2101e40c4fe38a01a0074946b27583b6ce0b37f0acfe2e908e9c853f',
 'our_mani/multitask/composite_TEST_resume_v1010.py': '19f917bb4675c62dfbd969d9d37c0cf64eaa108a8456914499677d0f08f08685',
 'scripts/continue_composite_TEST_v1013_r10.py': 'ec63efdb9288b31ff0112c9999fda1edfa53e923068c83d49f069ef91f74af32',
 'scripts/run_composite_TEST_owner_v1011_r10.py': 'a721cfd3b3b7611d3c57d233f416fdf9b8e2a13c72f41846df9d3ebd2eb838e0',
 'scripts/audit_composite_TEST_results_v1012_r10.py': '2f200aa93ff4ee1748764da2283ba3c10294fb52b81cd1667978969af56ff79a',
 'our_mani/multitask/composite_TEST_resume_v955.py': 'e636ea89d27bb497bf716062cc6fa2ccdcbee5811e9aacd50c9e29e82dc431bc',
 'scripts/run_composite_TEST_owner_v956_r10.py': '03159361511fdae6eb3591d03afb6fabd9199fbfdf7b60dd65ac17e7e945d83d',
 'scripts/continue_composite_TEST_v958_r10.py': 'd4b2a52e8c4d5553bf1a6722f71ed415bb6882757c9c0c83e9fe7296c632bde6',
 'scripts/audit_concurrent_TRAIN_resource_canary_v1005_r10.py': '24899b2f35000e24838801b920f623fa4aa9a8ae6e04f712e77aa7cd2d9a4a4a'}
CONTROLLER='runtime/framework/ordered_composite_TEST_v1013'
TEST_RUNTIME='runtime/framework/formal_TEST_composite_resume_v1011'
ORIGINAL='runtime/framework/formal_TEST_after_environment_resume_v753'
MIDDLE='runtime/framework/formal_TEST_concurrent_resume_v902'
MIDDLE_CONTROLLER='runtime/framework/ordered_concurrent_TEST_v907'
PARENT='runtime/framework/formal_TEST_composite_resume_v956'
PARENT_CONTROLLER='runtime/framework/ordered_composite_TEST_v958'
SETUP='artifacts/framework/multitask_r10/formal_TEST_composite_resume_v1011_setup'
CONTROL=SETUP+'/resume_control.json';CONTROL_LOCK=SETUP+'/resume_inputs.json'
MIDDLE_SETUP='artifacts/framework/multitask_r10/formal_TEST_concurrent_resume_v902_setup'
MIDDLE_CONTROL=MIDDLE_SETUP+'/resume_control.json';MIDDLE_LOCK=MIDDLE_SETUP+'/resume_inputs.json'
PARENT_SETUP='artifacts/framework/multitask_r10/formal_TEST_composite_resume_v956_setup'
PARENT_CONTROL=PARENT_SETUP+'/resume_control.json';PARENT_LOCK=PARENT_SETUP+'/resume_inputs.json'
ART='artifacts/framework/multitask_r10/formal_TEST_composite_resume_v1011'
RESULT=ART+'/TEST_results.json'
OWNER_STAGE=CONTROLLER+'/stages/formal_TEST_1920'
AUDIT_STAGE=CONTROLLER+'/stages/independent_TEST_results'
POST_RUNTIME='runtime/framework/post_complete_composite_TEST_motion_v1016'
POST_SETUP='artifacts/framework/multitask_r10/post_complete_composite_TEST_motion_v1016_setup'
POST_CONTROL=POST_SETUP+'/control.json';POST_LOCK=POST_SETUP+'/inputs.json'
POST_STAGE=POST_RUNTIME+'/stages/saved_motion_v1015'
MOTION=TEST_RUNTIME+'/full_sixteen_task_motion_summary_v1015.json'
POST_PATHS=dict(control=POST_CONTROL,input_lock=POST_LOCK,launch=POST_RUNTIME+'/launch.json',
    status=POST_RUNTIME+'/status.json',terminal=POST_RUNTIME+'/terminal.json',
    analysis_launch=POST_STAGE+'/launch.json',analysis_terminal=POST_STAGE+'/terminal.json',analysis_stdout=POST_STAGE+'/stdout.log')
POLICY='eb8f768d99975a7c2de2c36de5283d64898c2296f33fb342e332310eb32db6ef'
LOCKS=(CONTROLLER+'.lock',PARENT_CONTROLLER+'.lock',MIDDLE_CONTROLLER+'.lock','runtime/framework/ordered_resume_VAL_TEST_v775.lock',
    'runtime/framework/ordered_env_resume_VAL_TEST_v827.lock','runtime/framework/fixed_TRAIN104_geometry_native.lock',POST_RUNTIME+'.lock')
STOP_PATHS=('runtime/framework/multitask_r10_STOP',CONTROLLER+'/STOP',TEST_RUNTIME+'/STOP',POST_RUNTIME+'/STOP',
    PARENT_CONTROLLER+'/STOP',PARENT+'/STOP',MIDDLE_CONTROLLER+'/STOP',MIDDLE+'/STOP',ORIGINAL+'/STOP','runtime/framework/ordered_env_resume_VAL_TEST_v827/STOP',
    'runtime/framework/formal_VAL_env_resume_v823/STOP','runtime/framework/ordered_resume_VAL_TEST_v775/STOP',
    'runtime/framework/formal_VAL_resume_v772/STOP','runtime/framework/concurrent_TRAIN_resource_canary_v903/STOP',
    'runtime/framework/concurrent_TRAIN_resource_canary_v1005/STOP',
    'runtime/framework/post_TEST_resource_guard_v924/STOP','runtime/framework/post_four_tier_TEST_resource_guard_v1025/STOP')
EXTRA_PREFIXES=('run_composite_TEST_owner_','continue_composite_TEST_','audit_composite_TEST_results_',
    'continue_post_composite_TEST_motion_','summarize_composite_TEST_motion_','continue_post_TEST_motion_',
    'summarize_full_TEST_motion_')
_LEASES=WeakKeyDictionary()


def require(value,message):
    if not value:raise ValueError('post composite TEST1025: '+message)


def valid_sha(value):return isinstance(value,str) and len(value)==64 and not set(value)-set('0123456789abcdef')
def sha(raw):return hashlib.sha256(raw).hexdigest()
def _pointer(value,name=None):
    require(isinstance(value,dict) and set(value)=={'path','sha256'} and isinstance(value.get('path'),str)
        and value['path'] and not Path(value['path']).is_absolute() and '..' not in PurePosixPath(value['path']).parts
        and str(PurePosixPath(value['path']))==value['path'] and valid_sha(value.get('sha256'))
        and (name is None or value['path']==name),'exact canonical project pointer required')
    return value


def _allowed(name):
    if name in {*SOURCE_SHA,SOURCE,CONTROL,CONTROL_LOCK,PARENT_CONTROL,PARENT_LOCK,MIDDLE_CONTROL,MIDDLE_LOCK,QUALIFICATION,QUALIFICATION_LAUNCH,RESULT,MOTION,*POST_PATHS.values()}:return True
    if name in {'runtime/framework/concurrent_TRAIN_resource_qualification_v904.json',
        'runtime/framework/concurrent_TRAIN_resource_audit_v904/launch.json',
        'runtime/framework/frozen_TEST_environment_recheck_v954.json'}:return True
    sources=('scripts/run_formal_suite16_owner_v753_r10.py','scripts/run_concurrent_resumed_formal_TEST_owner_v902_r10.py',
        'scripts/continue_concurrent_TEST_v907_r10.py','scripts/continue_environment_resumed_VAL_TEST_v827_r10.py',
        'scripts/run_environment_resumed_formal_VAL_owner_v823_r10.py','scripts/audit_concurrent_TRAIN_resource_canary_v904_r10.py',
        'scripts/check_frozen_TEST_environment_v954_r10.py')
    if name in sources:return True
    runtimes=(CONTROLLER,TEST_RUNTIME,ORIGINAL,MIDDLE,MIDDLE_CONTROLLER,PARENT,PARENT_CONTROLLER,'runtime/framework/ordered_env_resume_VAL_TEST_v827')
    if any(name==runtime+'/'+leaf for runtime in runtimes for leaf in ('launch.json','plan.json','status.json','terminal.json','failure.json','DRAIN')):return True
    if name in {CANARY+'/CPU_owner_launch.json',CANARY+'/native_owner_launch.json'}:return True
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
        module=ModuleType('post1025_'+Path(name).stem);module.__file__=str(reader.root/name)
        exec(compile(reader.raw(name,SOURCE_SHA[name]),str(reader.root/name),'exec'),module.__dict__);return module
    return legacy,load(MOTION_SOURCE),load(POST_SOURCE)


def _postprocessing(r,old,post,control,cp,lp,completion,motion_sha,post_lock_sha):
    lock=r.json(POST_LOCK,post_lock_sha)
    require(lock.get('schema')==post.LOCK_SCHEMA and lock.get('execution_enabled') is True,'real1016 small activation lock required')
    files=lock.get('files');names={*post.SOURCES,POST_CONTROL,CONTROL,CONTROL_LOCK,PARENT_CONTROL,CONTROLLER+'/launch.json'}
    require(isinstance(files,dict) and len(files)==len(names),'exact12 postprocessing bindings required')
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
        and 1<=ctl['poll_seconds']<=30,'frozen read-only1016 control required')
    fixed=dict(runtime=POST_RUNTIME,controller_runtime=CONTROLLER,owner_runtime=TEST_RUNTIME,test_results=RESULT,motion_summary=MOTION,
        resume_control=cp,resume_control_lock=lp,controller_launch=r.pointer(CONTROLLER+'/launch.json'))
    require(all(ctl.get(k)==v for k,v in fixed.items()),'actual complete TEST and postprocessing batch must match')
    sources=ctl.get('sources');require(isinstance(sources,dict) and set(sources)==set(post.SOURCES),'seven source/test bindings required')
    for name,p in sources.items():
        _pointer(p,name);require(canonical[name]==p['sha256'],'post source differs from lock')
        if name in SOURCE_SHA:require(p['sha256']==SOURCE_SHA[name],'reviewed postprocessing source changed')
        # Two CPU-test sources are declared/pinned metadata, not executable dependencies.
    for name in (CONTROL,CONTROL_LOCK,PARENT_CONTROL,CONTROLLER+'/launch.json'):
        require(canonical[name]==r.inputs[name],'selected complete batch differs from post small lock')
    launch,state,end=(r.json(POST_RUNTIME+'/'+n+'.json') for n in ('launch','status','terminal'))
    known=[old._pid(launch)]
    host=dict(standard_host_analysis=True,frozen_environment_inherited_claimed=False,automatic_retry=False)
    require(all(launch.get(k) is v for k,v in host.items()) and launch.get('source_sha256')==SOURCE_SHA[POST_SOURCE]
        and launch.get('control')==pcp and launch.get('input_lock')==plp
        and launch.get('monitored_controller_launch')==ctl['controller_launch'],'real host1016 identity and batch required')
    argv=launch.get('argv');require(isinstance(argv,list) and argv and all(type(v) is str for v in argv)
        and argv[0] in (POST_SOURCE,str(r.root/POST_SOURCE)),'real1016 argv required')
    import argparse
    parser=argparse.ArgumentParser(add_help=False,allow_abbrev=False);parser.add_argument('--execute',action='store_true')
    parser.add_argument('--control',default=POST_CONTROL);parser.add_argument('--lock',default=POST_LOCK);parser.add_argument('--lock-sha256')
    try:args=parser.parse_args(argv[1:])
    except SystemExit as error:raise ValueError('invalid actual1016 argv') from error
    require(args.execute and argv.count('--execute')==1 and argv.count('--lock-sha256')==1
        and argv.count('--control')<=1 and argv.count('--lock')<=1 and args.lock_sha256==post_lock_sha
        and args.control==POST_CONTROL and args.lock==POST_LOCK,'external1016 lock pin required')
    sp=dict(path=MOTION,sha256=motion_sha)
    require(state.get('stage')=='complete' and state.get('automatic_retry') is False and state.get('motion_summary')==sp
        and end.get('schema')=='r10_readonly_post_complete_composite_TEST_motion_terminal_v1016' and end.get('complete') is True
        and end.get('motion_summary')==sp and end.get('TEST_results')==completion['result']
        and end.get('control')==pcp and end.get('input_lock')==plp and end.get('resume_control')==cp and end.get('resume_control_lock')==lp
        and all(end.get(k) is v for k,v in {**host,'Native_GPU_TRAIN_started':False,'TXT_modified':False}.items()),'actual complete1016 final required before summary read')
    child,child_end=r.json(POST_STAGE+'/launch.json'),r.json(POST_STAGE+'/terminal.json');known.append(old._pid(child));old._clean_stage(child_end)
    expected=[str(r.root/'.conda-env/bin/python'),'-B','-u',MOTION_SOURCE,'--test-results-sha256',completion['result_sha256'],
        '--resume-lock-sha256',lp['sha256'],'--resume-control',CONTROL,'--resume-lock',CONTROL_LOCK]
    require(child.get('argv')==expected and child.get('source')==sources[MOTION_SOURCE]
        and child.get('source_sha256')==SOURCE_SHA[MOTION_SOURCE] and child.get('owner_pid')==launch['pid']
        and child.get('owner_start_ticks')==launch['start_ticks'] and child.get('control')==pcp and child.get('input_lock')==plp
        and child.get('TEST_results')==completion['result'] and child_end.get('interrupted') is False
        and all(child.get(k) is v for k,v in host.items()),'real host1015 child source/PID/argv/batch required')
    summary=r.json(MOTION,motion_sha);stdout=r.json(POST_STAGE+'/stdout.log')
    post.validate_summary(summary,sp,stdout,ctl,dict(sources=sources),completion,r.root)
    return known,{key:r.pointer(name) for key,name in POST_PATHS.items()}


def require_complete_TEST_metadata(root,expected_test_result_sha256,*,resume_lock_sha256,motion_summary_sha256,
    postprocessing_lock_sha256,read_bytes=None):
    """Actual four-tier TEST then saved-motion completion; metadata only."""
    require(all(valid_sha(v) for v in (expected_test_result_sha256,resume_lock_sha256,motion_summary_sha256,postprocessing_lock_sha256)),
        'external1012 result,1010 lock,1015 summary and1016 lock SHA pins required')
    r=_Metadata(root,read_bytes);old,m,post=_contracts(r);C,PC,P=m.C,m.PC,m.P
    locked=r.json(CONTROL_LOCK,resume_lock_sha256)
    require(locked.get('schema')==C.LOCK_SCHEMA and locked.get('approved_for_execution') is True,'actual1010 frozen lock required')
    files=locked.get('files');control=r.json(CONTROL,old._lookup(files,r.root,CONTROL));cp,lp=r.pointer(CONTROL),r.pointer(CONTROL_LOCK)
    require(locked.get('control')==cp and control.get('schema')==C.CONTROL_SCHEMA and control.get('split')=='test'
        and control.get('expected_cases')==1920 and control.get('source_count')==640 and control.get('policy_sha256')==POLICY
        and control.get('author_outcome_protocol')==old.PROTOCOL and control.get('resource_profile')==PROFILE
        and control.get('resource_qualifier')==dict(path=QUALIFIER_SOURCE,sha256=SOURCE_SHA[QUALIFIER_SOURCE])
        and control.get('user_authorized_resume') is True and control.get('standing_authorization_reused') is True
        and control.get('new_user_event_invented') is False and control.get('physical_policy_changed') is False,
        'actual same-policy1010 continuation required')
    for key,name in (('resume_controller',CONTROLLER_SOURCE),('resume_owner',OWNER_SOURCE),('resume_auditor',AUDITOR_SOURCE),('result_auditor',AUDITOR_SOURCE)):
        require(control.get(key)==dict(path=name,sha256=SOURCE_SHA[name]) and old._lookup(files,r.root,name)==SOURCE_SHA[name],
            'frozen actual writer binding required')
    for key,field in (('environment_wrapper','wrapper'),('environment_manifest','manifest'),('environment_namespace_proof','namespace_qualification')):
        require(control.get(key)==old.ENVIRONMENT_POINTERS[field],'original qualified820 binding changed')
    entry,implementation=old._entry(control,files,r.root)
    controller,state,final=(r.json(CONTROLLER+'/'+n+'.json') for n in ('launch','status','terminal'));known=[old._pid(controller)]
    require(controller.get('source_sha256')==SOURCE_SHA[CONTROLLER_SOURCE] and controller.get('resume_control')==cp
        and controller.get('resume_control_lock')==lp and controller.get('automatic_retry') is False,'actual1013 launch required')
    env=old._environment(controller.get('frozen_environment'),controller)
    require(state.get('stage')=='complete' and state.get('automatic_retry') is False and state.get('frozen_environment')==env
        and not state.get('infrastructure_failure_detected') and state.get('owner_error') is None
        and final.get('complete') is True and final.get('automatic_retry') is False
        and _pointer(final.get('TEST_results'),RESULT)['sha256']==expected_test_result_sha256,
        'complete actual1013 required before result read')
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
    # No incomplete/current TEST can reach any published result or post data.
    result=r.json(RESULT,expected_test_result_sha256);m.require_result(result,control,cp,lp,stages[AUDITOR_SOURCE])
    evidence=result.get('inputs_sha256')
    required=(CONTROL,CONTROL_LOCK,CORE_SOURCE,OWNER_SOURCE,AUDITOR_SOURCE,*[TEST_RUNTIME+'/'+n+'.json' for n in ('launch','plan','status','terminal')])
    require(all(old._lookup(evidence,r.root,n)==r.inputs[n] for n in required),'actual1012 metadata ledger binding changed')
    require(old._lookup(evidence,r.root,control['source_inputs']['path'])==control['source_inputs']['sha256'],
        'original640 input lock must remain bound, not opened')
    stdout=r.json(AUDIT_STAGE+'/stdout.log')
    require(stdout.get('complete') is True and stdout.get('schema')==m.RESULT_SCHEMA and stdout.get('split')=='test'
        and stdout.get('cases')==1920 and stdout.get('out')==RESULT and stdout.get('motion_is_diagnostic_only') is True
        and stdout.get('native_successes_by_method')==result.get('native_successes_by_method'),'actual1012 stdout required')
    def frozen(p,name=None,*,audited=True):
        _pointer(p,name);require(old._lookup(files,r.root,p['path'])==p['sha256']
            and (not audited or old._lookup(evidence,r.root,p['path'])==p['sha256']),
            'frozen historical metadata and actual1012 evidence required')
        return r.pointed(p)
    parent=frozen(control['previous_resume_control'],PARENT_CONTROL)
    parent_lock=frozen(control['previous_resume_control_lock'],PARENT_LOCK)
    require(parent_lock.get('approved_for_execution') is True and parent_lock.get('schema')==PC.LOCK_SCHEMA
        and parent_lock.get('control')==control['previous_resume_control'],'actual955 provenance required')
    middle=frozen(parent['previous_resume_control'],MIDDLE_CONTROL)
    middle_lock=frozen(parent['previous_resume_control_lock'],MIDDLE_LOCK)
    require(middle_lock.get('approved_for_execution') is True and middle_lock.get('schema')==P.LOCK_SCHEMA
        and middle_lock.get('control')==parent['previous_resume_control'],'actual915 provenance required')
    require(parent.get('schema')==PC.CONTROL_SCHEMA and parent.get('policy_sha256')==POLICY
        and parent.get('author_outcome_protocol')==old.PROTOCOL and parent.get('resource_profile')==old.PROFILE_LIMITS
        and control.get('resource_profile_changed') is True
        and control.get('previous_resource_profile')==parent['resource_profile']
        and control.get('previous_resource_qualification')==parent['resource_qualification']
        and parent.get('execution_implementation')==implementation and middle.get('execution_entry')==entry,
        'unchanged original915 implementation and explicit qualified resource transition required')
    histories={}
    for runtime,ctl,prefix in ((ORIGINAL,control,'original_'),(MIDDLE,parent,'previous_'),(PARENT,control,'previous_')):
        histories[runtime]={n:frozen(ctl[prefix+n],runtime+'/'+n+'.json') for n in ('launch','plan','status','terminal')}
    original,mid,previous=histories[ORIGINAL],histories[MIDDLE],histories[PARENT]
    require(all(control['original_'+n]==middle['previous_'+n] for n in original),'original753 tier differs from915')
    P.require_partition(middle,original['plan'],original['status'],original['terminal'])
    require(mid['plan'].get('jobs')==P.resumed_jobs(original['plan'],middle,middle['execution_entry']),
        'literal753 to902 provenance required')
    PC.require_resume_plan(parent,mid['plan'],mid['status'],mid['terminal'],previous['plan'])
    C.require_resume_plan(control,previous['plan'],previous['status'],previous['terminal'],plan)
    for older,newer in ((original,mid),(mid,previous),(previous,{'status':status})):
        require(all(newer['status']['completed'].get(k)==v for k,v in older['status']['completed'].items()),
            'all historical False/None records must remain literal')
    recheck=frozen(control['environment_recheck'],'runtime/framework/frozen_TEST_environment_recheck_v954.json')
    C.require_environment_recheck(recheck,control)
    for ctl,runtime,source in ((control,PARENT_CONTROLLER,PARENT_CONTROLLER_SOURCE),
            (parent,MIDDLE_CONTROLLER,old.CONTROLLER_SOURCE),(middle,old.OLD_CONTROLLER,old.OLD_CONTROLLER_SOURCE)):
        for key,leaf in (('previous_controller_launch','launch.json'),('previous_controller_status','status.json'),('previous_controller_failure','failure.json')):
            record=frozen(ctl[key],runtime+'/'+leaf)
            if leaf=='launch.json':
                known.append(old._pid(record));expected=SOURCE_SHA.get(source,old.SOURCE_SHA.get(source))
                r.raw(source,expected);require(record.get('source_sha256')==expected,'real historical controller source required')
            elif leaf=='status.json':require(record.get('stage')=='requires_concrete_repair' and record.get('automatic_retry') is False,'preserve real historical repair status')
            else:require(record.get('complete') is False and record.get('user_stop') is False and record.get('automatic_retry') is False,
                'never invent successful old827/907/958 terminals')
    frozen(control['previous_DRAIN'],PARENT+'/DRAIN');frozen(parent['previous_DRAIN'],MIDDLE+'/DRAIN')
    owners={runtime:value['launch'] for runtime,value in histories.items()};owners[TEST_RUNTIME]=launch
    m.require_released_cohort(plan,result,status,owners,set(control['completed_job_ids']))
    counts,stage_owners=post.require_four_tier_evidence(result,control,status,r.pointer(TEST_RUNTIME+'/launch.json'),r.pointer(TEST_RUNTIME+'/terminal.json'),parent)
    for runtime,source in ((ORIGINAL,old.OLD_OWNER_SOURCE),(MIDDLE,old.OWNER_SOURCE),(PARENT,PARENT_OWNER_SOURCE)):
        known.append(old._pid(owners[runtime]));expected=SOURCE_SHA.get(source,old.SOURCE_SHA.get(source))
        r.raw(source,expected);require(owners[runtime].get('source_sha256')==expected,'actual historical owner source required')
    expected_owners=[dict(runtime=runtime,launch=r.pointer(runtime+'/launch.json'),terminal=r.pointer(runtime+'/terminal.json'),
        **old._pid(owners[runtime]),completed=counts[runtime],resource_limits={k:owners[runtime][k] for k in
        ('max_native_slots','max_GPU_slots','max_CPU_preflight_slots')}) for runtime in (ORIGINAL,MIDDLE,PARENT,TEST_RUNTIME)]
    require(stage_owners==expected_owners,'four real physical owner sources and limits required')
    historic=r.json(old.VAL_OWNER_STAGE+'/launch.json');known.append(old._pid(historic))
    require(historic.get('source_sha256')==old.SOURCE_SHA[old.VAL_OWNER_SOURCE]
        and old.option(historic.get('argv'),'--split')=='validation','real historical823 VAL stage required')
    # Only qualification metadata is read.12 TRAIN cases are never opened here.
    q=frozen(control['resource_qualification'],QUALIFICATION)
    require(q.get('schema')=='r10_independent_concurrent_TRAIN_resource_qualification_v1005'
        and q.get('complete') is True and q.get('split')=='train' and type(q.get('case_count')) is int and q['case_count']==12
        and q.get('source_sha256')==SOURCE_SHA[QUALIFIER_SOURCE] and q.get('resource_profile')==PROFILE
        and q.get('execution_entry')==implementation and q.get('policy_sha256')==POLICY and q.get('author_outcome_protocol')==old.PROTOCOL
        and type(q.get('infrastructure_failed')) is int and q['infrastructure_failed']==0
        and all(q.get(k) is True for k in ('cleanup_completed','resource_bounds_verified','overlap_verified','independent_audit_complete'))
        and q.get('observed_native_peak')==8 and q.get('observed_GPU_peak')==3 and q.get('actual_CPU_preflight_peak')==4
        and q.get('joint_supply_overlap',{}).get('complete') is True
        and q.get('hardware_observation',{}).get('actual_three_compute_contexts_during_joint8_3_4_observed') is True
        and type(q['hardware_observation'].get('unknown_samples')) is int and q['hardware_observation']['unknown_samples']==0
        and q['hardware_observation'].get('required_headroom_mib')==2048
        and type(q['hardware_observation'].get('minimum_sampled_headroom_mib')) is int
        and q['hardware_observation']['minimum_sampled_headroom_mib']>=2048
        and q.get('formal_score_contribution') is False and q.get('TEST_read') is False,'independent actual1005 resource qualification required')
    qa=frozen(q.get('actual_independent_audit'),QUALIFICATION_LAUNCH,audited=False);known.append(old._pid(qa))
    require(qa.get('source_sha256')==SOURCE_SHA[QUALIFIER_SOURCE],'actual1005 auditor identity required')
    # Historical904 also remains a real, exited source;1012 need not consume
    # its audit launch, so do not invent that absent ledger membership.
    oldq=frozen(parent['resource_qualification'],old.QUALIFICATION,audited=False)
    oldqa=frozen(oldq.get('actual_independent_audit'),old.QUALIFICATION_LAUNCH,audited=False);known.append(old._pid(oldqa))
    require(oldqa.get('source_sha256')==old.SOURCE_SHA[old.QUALIFIER_SOURCE],'actual historical904 identity required')
    completion=dict(result=r.pointer(RESULT),result_sha256=expected_test_result_sha256,provenance_runtime_counts=counts,actual_stage_owners=stage_owners)
    post_known,post_pointers=_postprocessing(r,old,post,control,cp,lp,completion,motion_summary_sha256,postprocessing_lock_sha256);known+=post_known
    require(len({(p['pid'],p['start_ticks']) for p in known})==len(known),'all known process identities must be distinct')
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
        old907_success_terminal_required=False,old958_success_terminal_required=False,old964_success_terminal_required=False,
        legacy924_compatibility_scope='Permit registration only; no old924 completion or resource gate called',
        gate_chain='actual1013/1011/1012 then actual1016/1015',registration_only_compatibility=True)


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
    require(owned is not None,'live1025-owned permit required; old924 permit is insufficient')
    lease=_LEASES[owned]
    require(lease['pid']==os.getpid() and lease['root']==Path(root).resolve(),'same-process canonical-root1025 lease required')
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
