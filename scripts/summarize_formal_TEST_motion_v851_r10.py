"""Summarize saved TEST motion only after the genuine complete754/827 finish.

Preparation/tests read source code and in-memory fakes only. A future main
requires an explicit final TEST_results SHA, complete827 controller and real753
owner/auditor receipts before reading the640-source TEST metadata. Raw reset,
PKL/image/HDF5/model contents are never opened. No job or release is created.
Native False, motion False and zero counted ticks remain in the1920-case grid.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import types

ROOT = Path(__file__).resolve().parents[1]
HELPER = 'scripts/summarize_formal_VAL_motion_v847_r10.py'
HELPER_SHA = '65bbcae3ac2a3ebb569fb7083292561ba205dc36b2e8fd6def14053a286792f3'


def load_source(name, expected):
    path=ROOT/name;raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=expected: raise ValueError('audited source SHA changed: '+name)
    module=types.ModuleType('validated_motion_'+Path(name).stem);module.__file__=str(path)
    exec(compile(raw,str(path),'exec'),module.__dict__)
    return module


H=load_source(HELPER,HELPER_SHA)
require=H.require;sha_bytes=H.sha_bytes;digest=H.digest;valid_sha=H.valid_sha;option=H.option
METHODS=H.METHODS;TASKS=H.TASKS;POLICY=H.POLICY;PROFILE=H.PROFILE;PARAMETERS=H.PARAMETERS
CONTROL=H.CONTROL;LOCK=H.LOCK;VAL_RELEASE=H.RELEASE;CONTROLLER=H.CONTROLLER;CONTROLLER_SOURCE=H.CONTROLLER_SOURCE
REDUCER=H.REDUCER;SUBSET=H.SUBSET
SETUP='artifacts/framework/multitask_r10/formal_TEST_after_environment_resume_v826_setup'
SUITE=SETUP+'/suite.json';SUITE_LOCK=SETUP+'/frozen_inputs.json'
SOURCE_LOCK=SETUP+'/test_source_inputs.json';MANIFEST=SETUP+'/official_test_manifest16.json'
ORIGINAL_MANIFEST='artifacts/framework/protocol/test_manifest_all16.json'
ROUTES='artifacts/framework/multitask_r10/author_outcome_qualification_v745_setup/routes.json'
RUNTIME='runtime/framework/formal_TEST_after_environment_resume_v753'
ART='artifacts/framework/multitask_r10/formal_VAL_env_resume_v823'
RESULT=ART+'/TEST_results.json'
OUTPUT=RUNTIME+'/full_sixteen_task_motion_summary_v851.json'
OWNER='scripts/run_formal_suite16_owner_v753_r10.py'
AUDITOR='scripts/audit_formal_suite16_results_v754_r10.py'
PREPARER='scripts/prepare_formal_TEST_after_environment_resume_v826_r10.py'
SOURCE_PREPARER='scripts/prepare_formal_split_source_lock_v756_r10.py'
AUDIT_STAGE=CONTROLLER+'/stages/independent_TEST_results'
OWNER_STAGE=CONTROLLER+'/stages/formal_TEST_1920'
SOURCES={HELPER,REDUCER,CONTROLLER_SOURCE,OWNER,AUDITOR,PREPARER,SOURCE_PREPARER,
    'scripts/run_formal_case_author_outcome_v751_r10.py','scripts/run_task_executor_smoke_author_outcome_v743_r10.py',
    'scripts/run_task_executor_smoke_author_suite16_v744_r10.py','scripts/run_task_executor_smoke_rml_suite16_v664_r10.py',
    'our_mani/multitask/formal_suite_admission_v728.py','our_mani/multitask/native_outcome_sampling_v741.py'}
PINS={CONTROL:H.PINS[CONTROL],LOCK:H.PINS[LOCK],HELPER:HELPER_SHA,REDUCER:H.PINS[REDUCER],SUBSET:H.PINS[SUBSET]}


def allowed(name):
    if name in {*PINS,*SOURCES,RESULT,VAL_RELEASE,SUITE,SUITE_LOCK,SOURCE_LOCK,MANIFEST,ORIGINAL_MANIFEST,ROUTES,
                CONTROLLER+'/launch.json',CONTROLLER+'/terminal.json',RUNTIME+'/launch.json',RUNTIME+'/terminal.json',RUNTIME+'/plan.json'}:return True
    for stage in (AUDIT_STAGE,OWNER_STAGE):
        if name.startswith(stage+'/') and name[len(stage)+1:] in ('launch.json','terminal.json','stdout.log'):return True
    suffix=name[len(RUNTIME)+1:] if name.startswith(RUNTIME+'/') else ''
    parts=suffix.split('/')
    if len(parts)==3 and parts[0] in ('CPU','native') and parts[2] in ('launch.json','terminal.json','CPU_preflight.json','stdout.log'):return True
    if name.startswith(ART+'/test/') and Path(name).name in ('case.json','admission_driver_receipt.json'):return True
    return False


class Reader:
    """Strict metadata/evidence allowlist; never traverse transitive input maps."""
    def __init__(self,root):
        self.root=Path(root);self.inputs={HELPER:HELPER_SHA};self.evidence={};self.frozen={};self.original={}
    def lookup(self,mapping,name):
        values={mapping[k] for k in (name,str(self.root/name)) if k in mapping}
        require(len(values)==1 and valid_sha(next(iter(values))),'missing/conflicting evidence SHA: '+name)
        return next(iter(values))
    def raw(self,name,*,expected=None,evidence=False,frozen=False,original=False):
        require(isinstance(name,str) and not Path(name).is_absolute() and str(PurePosixPath(name))==name
            and '..' not in PurePosixPath(name).parts and allowed(name),'non-evidence or undeclared input refused: '+str(name))
        path=(self.root/name).resolve(strict=True)
        require(path.is_relative_to(self.root) and str(path.relative_to(self.root))==name,'canonical evidence required; symlink aliases refused before open')
        raw=path.read_bytes();actual=sha_bytes(raw)
        wanted=[v for v in (expected,PINS.get(name),self.lookup(self.evidence,name) if evidence else None,
            self.lookup(self.frozen,name) if frozen else None,self.lookup(self.original,name) if original else None) if v is not None]
        require(all(valid_sha(v) and actual==v for v in wanted),'input SHA changed: '+name)
        require(name not in self.inputs or self.inputs[name]==actual,'input changed during summary: '+name)
        self.inputs[name]=actual;return raw
    def json(self,name,**kwargs):return json.loads(self.raw(name,**kwargs))
    def pointer(self,name):return dict(path=name,sha256=self.inputs[name])
    def pointed(self,pointer,**kwargs):
        require(isinstance(pointer,dict) and set(pointer)=={'path','sha256'},'explicit path/SHA pointer required')
        return self.json(pointer['path'],expected=pointer['sha256'],**kwargs)


def require_result_header(result):
    #754 TEST calls531.audit_grid directly, so no VAL-release schema/TEST_used flag exists.
    require(result.get('complete') is True and result.get('split')=='test' and result.get('source_cases')==640
        and result.get('method_cases')==1920 and result.get('completed_method_cases')==1920
        and result.get('methods')==list(METHODS) and result.get('tasks')==list(TASKS)
        and result.get('policy_sha256')==POLICY and result.get('common_RML_profile_sha256')==PROFILE,
        'real complete16/640/1920 TEST results required')
    for key,value in dict(TEST_used_for_fitting=False,motion_is_diagnostic_only=True,motion_metrics_required_for_all_methods=True,
        native_task_success_unchanged=True,native_success_is_not_a_release_threshold=True,source_outcome_filtering=False,
        native_failures_retained=True,Value_NGFF_FASTER_used=False,execution_protocol_changed=True).items():
        require(result.get(key) is value,'unchanged native/diagnostic semantics required: '+key)
    require(isinstance(result.get('cases'),list) and len(result['cases'])==1920
        and isinstance(result.get('admission_driver_receipts'),list) and len(result['admission_driver_receipts'])==1920
        and isinstance(result.get('inputs_sha256'),dict) and result['inputs_sha256'],'1920 actual case/driver pointers and evidence map required')


def require_finished_controller(controller,terminal,control):
    require(controller.get('resume_control')==dict(path=CONTROL,sha256=PINS[CONTROL])
        and controller.get('resume_control_lock')==dict(path=LOCK,sha256=PINS[LOCK]),'actual frozen827 control binding required')
    H.require_environment(controller.get('frozen_environment',{}),control,observer=controller.get('pid'))
    require(terminal.get('complete') is True and terminal.get('automatic_retry') is False
        and terminal.get('TEST_results')==RESULT,'full827 TEST completion required before TEST metadata reads')


def require_stage(launch,terminal,argv,source_sha,controller):
    H.require_clean_terminal(terminal,controller_stage=True)
    require(launch.get('argv')==argv and launch.get('source_sha256')==source_sha and launch.get('automatic_retry') is False
        and launch.get('inherited_frozen_environment')==controller['frozen_environment']
        and launch.get('current_mount_namespace')==controller['frozen_environment']['current_mount_namespace']
        and type(launch.get('pid')) is int and type(launch.get('start_ticks')) is int,'actual827 stage argv/PID/inherited environment required')


def require_owner_completion(launch,terminal,stage,result,source_sha,protocol,terminal_pointer):
    require(launch.get('source_sha256')==source_sha and launch.get('split')=='test'
        and launch.get('pid')==stage['pid'] and launch.get('start_ticks')==stage['start_ticks']
        and launch.get('policy_sha256')==POLICY and launch.get('author_outcome_protocol')==protocol
        and launch.get('execution_protocol_changed') is True and launch.get('automatic_retry') is False
        and terminal.get('complete') is True and terminal.get('split')=='test' and terminal.get('expected')==1920
        and terminal.get('completed')==1920 and terminal.get('failed')==0 and terminal.get('stop_requested') is False
        and terminal.get('automatic_retry') is False and terminal.get('policy_sha256')==POLICY
        and terminal.get('author_outcome_protocol')==protocol and terminal.get('execution_protocol_changed') is True
        and result.get('actual_stage_owner')==terminal_pointer,
        'actual clean full1920 original753 owner required before TEST metadata')


def source_grid(manifest,original,source_lock,val_pointer,source_preparer_sha):
    require(manifest.get('split')=='test' and manifest.get('mapping_status')=='verified' and manifest.get('episode_count')==640
        and manifest.get('training_calibration_or_parameter_selection_used') is False
        and manifest.get('complete_unchanged_VAL_release_observed_before_source_selection')==val_pointer,
        'post-VAL original verified640 TEST manifest required')
    require(source_lock.get('schema')=='r10_full16_post_admission_split_source_lock_v1'
        and source_lock.get('split')=='test' and source_lock.get('source_count')==640
        and source_lock.get('policy_sha256')==POLICY and source_lock.get('complete_VAL_release_observed') is True
        and source_lock.get('full16_admission_observed_before_source_selection') is True
        and source_lock.get('source_content_used_for_policy_fitting') is False and source_lock.get('TEST_used_for_fitting') is False
        and source_lock.get('native_started') is False and source_lock.get('GPU_started') is False
        and source_lock.get('source_lock_preparer_sha256')==source_preparer_sha,'real756 post-admission TEST source metadata required')
    originals=[(r.get('task'),r.get('raw_episode_id')) for r in original.get('episodes',[])]
    #prepare_framework_protocol.py:51 writes this original roster without split.
    require(original.get('split','test')=='test' and len(originals)==len(set(originals))==640
        and Counter(k[0] for k in originals)==Counter({t:40 for t in TASKS}),'exact original640 TEST roster required')
    sources={};aliases=set();files=source_lock.get('files',{})
    for row in manifest.get('episodes',[]):
        key=(row.get('task'),row.get('raw_episode_id'));variation=row.get('variation');index=row.get('official_episode_index')
        require(key in originals and key not in sources and isinstance(key[1],str) and key[1]
            and row.get('split')=='test' and row.get('mapping_status')=='verified'
            and type(variation) is int and variation>=0 and type(index) is int and index>=0
            and (key[0],variation,index) not in aliases,'unique original TEST raw/variation-local reset identity required')
        required=row.get('source_required_files',{})
        require(isinstance(required,dict) and required and all(isinstance(v,dict) and valid_sha(v.get('sha256'))
            and files.get(v.get('path'))==v['sha256'] for v in required.values()),'reset metadata file SHA bindings required; raw bytes remain unopened')
        for field in ('task_source_path','task_scene_path'):
            if row.get(field):require(valid_sha(files.get(row[field])),'declared task source/scene SHA required')
        aliases.add((key[0],variation,index));sources[key]=digest(row)
    require(len(sources)==640 and set(sources)==set(originals),'complete original640 TEST source bijection required')
    return sources


def require_plan(plan,sources,routes,protocol):
    require(plan.get('split')=='test' and plan.get('expected_cases')==1920 and plan.get('policy_sha256')==POLICY
        and plan.get('source_outcome_filtering') is False and plan.get('automatic_retry') is False,'complete unchanged TEST plan required')
    route_rows=routes.get('routes',[]);by_route={(r['task'],r['method']):r for r in route_rows}
    require(len(route_rows)==len(by_route)==48 and set(by_route)=={(t,m) for t in TASKS for m in METHODS},'exact frozen48 physical routes required')
    seen=set();ids=set();rows=plan.get('jobs',[])
    expected={(task,raw,m) for task,raw in sources for m in METHODS}
    for row in rows:
        key=(row['task'],row['raw'],row['method']);route=by_route[(row['task'],row['method'])];argv=row['argv']
        require(key in expected and key not in seen and row['job_id']=='_'.join(key) and row['job_id'] not in ids
            and row['source_row_sha256']==sources[key[:2]],'unique original source/method TEST plan required')
        require(not any(k in row for k in ('evidence_runtime','imported_completed','previous_row_sha256')),'TEST is one fresh753 cohort; no VAL imports')
        seen.add(key);ids.add(row['job_id'])
        require(argv[:3]==['bash','scripts/run_aligned_gpu.sh' if row['method']=='physmani' else 'scripts/run_local_sim.sh',row['admission_driver']['path']]
            and row.get('worker_sha256')==row['admission_driver']['sha256'] and row.get('case_worker')==route['worker']
            and row.get('native_runtime')=={k:route['consumer'][k] for k in ('path','sha256')}
            and row.get('original_native_runtime')==protocol['original_runtime'] and row.get('author_outcome_protocol')==protocol
            and row.get('configuration_sha256')==route['configuration']['sha256'] and row.get('input_lock_sha256')==route['frozen_inputs']['sha256'],
            'unchanged751/743/744 route and configuration required')
        for flag,value in (('--config',route['configuration']['path']),('--frozen-inputs',route['frozen_inputs']['path']),
            ('--manifest',MANIFEST),('--suite',SUITE),('--suite-lock',SUITE_LOCK),('--source-inputs',SOURCE_LOCK),
            ('--split','test'),('--task',key[0]),('--raw-episode-id',key[1]),('--method',key[2]),
            ('--out',ART+'/test/'+'/'.join(key)),('--validation-release',VAL_RELEASE)):
            require(option(argv,flag)==value,'actual TEST argv binding changed: '+flag)
    require(len(rows)==1920 and seen==expected,'exact original640 sources x3 TEST bijection required')
    return rows


def require_case(case,row,source_sha,protocol,source_lock_sha):
    #742:187-189: TEST uses original raw, parent=None, replica0; VAL augmentation is absent.
    identity=case.get('identity',{});raw=row['raw'];task=row['task'];method=row['method']
    require(case.get('complete') is True and case.get('cleanup_completed') is True and case.get('infrastructure_error') is None
        and type(case.get('official_task_success')) is bool and type(case.get('native_terminal_failure')) is bool,'complete TEST native outcome/cleanup required')
    require(case.get('split')==identity.get('split')=='test' and case.get('task')==identity.get('task')==task
        and case.get('method')==identity.get('method')==method and case.get('source_raw_episode_id')==identity.get('source_raw_episode_id')==raw
        and case.get('parent_raw_episode_id') is None and identity.get('parent_raw_episode_id') is None
        and identity.get('raw_episode_id')==raw and identity.get('replica')==0 and identity.get('source_row_sha256')==source_sha
        and identity.get('config_sha256')==row['configuration_sha256'] and identity.get('frozen_inputs_sha256')==row['input_lock_sha256']
        and identity.get('source_input_lock_sha256')==source_lock_sha,'actual original TEST source/CPU identity required')
    for k,v in dict(demo_only=False,development_demo_only=False,formal_validation_metric=False,formal_TEST_metric=True,
        formal_test_used=True,TEST_used_for_fitting=False,Value_NGFF_FASTER_used=False).items():require(case.get(k) is v,'formal TEST-only case required: '+k)
    domain=case.get('component_domain',{});frozen=case.get('full16_frozen_execution',{})
    require(domain.get('purpose')=='evaluation' and domain.get('split')=='test' and domain.get('task')==task
        and domain.get('method')==method and domain.get('raw_episode_id')==raw and case.get('author_outcome_protocol')==protocol
        and frozen.get('policy_sha256')==POLICY and frozen.get('source_row_sha256')==source_sha
        and frozen.get('actual_split')=='test' and frozen.get('VAL_TEST_relabelled_as_TRAIN') is False,'unchanged TEST evaluation protocol/domain required')
    #Same numeric RML/clock checks as frozen847, with the actual TEST object unchanged.
    rml=case.get('common_author_RML_parameters',{});motion=case.get('all_stage_counted_joint_motion',{})
    require(rml.get('complete') is True and rml.get('native_parameters_mutated') is False
        and rml.get('actual_motion_bound_claimed') is False and rml.get('profile_sha256')==PROFILE
        and rml.get('actual_parameters')==PARAMETERS and type(motion.get('all_stage_actual_window_checks_passed')) is bool
        and motion.get('actual_window_acceleration_criterion_rad_s2')==6.
        and motion.get('planning_metadata_used_as_actual_acceleration_cap') is False and motion.get('policy_or_physics_modified') is False,
        'common RML and diagnostic-only measured-motion metadata required')
    clock=case.get('counted_clock',{});alignment=case.get('alignment_receipt',{});native_clock=alignment.get('clock',{});ticks=clock.get('ticks')
    require(type(ticks) is int and 0<=ticks<=200 and clock.get('dt')==.05 and type(clock.get('seconds')) in (int,float)
        and math.isfinite(clock['seconds']) and abs(clock['seconds']-ticks*.05)<=1e-9
        and native_clock.get('ticks')==ticks and native_clock.get('dt')==.05 and native_clock.get('budget_seconds')==10.
        and abs(native_clock.get('simulation_seconds',float('inf'))-clock['seconds'])<=1e-9
        and sum(native_clock.get('phases',{}).values())==ticks
        and isinstance(motion.get('events'),list) and len(motion['events'])==ticks,'saved counted-clock/window coverage required')
    for key in ('actual_physics_seconds','actual_task_seconds'):
        value=alignment.get(key)
        require(type(value) in (int,float) and math.isfinite(value) and abs(value-clock['seconds'])<=1e-5,'native counted-clock mismatch: '+key)


def require_pointers(rows,cases,drivers):
    require(len(rows)==len(cases)==len(drivers)==1920,'exact1920 case/driver pointers required')
    seen=set();seen_driver=set()
    for row,case,driver in zip(rows,cases,drivers):
        out=option(row['argv'],'--out')
        require(isinstance(case,dict) and set(case)=={'path','sha256'} and valid_sha(case['sha256'])
            and case['path']==out+'/case.json' and case['path'] not in seen
            and isinstance(driver,dict) and set(driver)=={'path','sha256'} and valid_sha(driver['sha256'])
            and driver['path']==out+'/admission_driver_receipt.json' and driver['path'] not in seen_driver,
            'unique ordered TEST case/driver bijection required')
        seen.add(case['path']);seen_driver.add(driver['path'])


def collect(reader,result_sha):
    result=reader.json(RESULT,expected=result_sha);require_result_header(result);reader.evidence=result['inputs_sha256']
    control=reader.json(CONTROL);original_lock=reader.json(LOCK);reader.original=original_lock['files']
    require(control.get('policy_sha256')==POLICY and original_lock.get('approved_for_execution') is True,'actual frozen823 graph required')
    for name in (CONTROLLER_SOURCE,OWNER,AUDITOR,PREPARER,SOURCE_PREPARER):reader.raw(name,original=True)
    require(result.get('source_sha256')==reader.inputs[AUDITOR],'actual original754 TEST auditor required')
    controller=reader.json(CONTROLLER+'/launch.json');controller_terminal=reader.json(CONTROLLER+'/terminal.json')
    require(controller.get('source_sha256')==reader.inputs[CONTROLLER_SOURCE],'actual frozen827 source required')
    require_finished_controller(controller,controller_terminal,control)
    common=['--suite',SUITE,'--suite-lock',SUITE_LOCK,'--source-inputs',SOURCE_LOCK,'--split','test','--validation-release',VAL_RELEASE,'--runtime',RUNTIME]
    stages={}
    for stage,source,tail in ((OWNER_STAGE,OWNER,['--out',ART+'/test']),(AUDIT_STAGE,AUDITOR,['--out',RESULT])):
        launch=reader.json(stage+'/launch.json');terminal=reader.json(stage+'/terminal.json')
        argv=[str(reader.root/'.conda-env/bin/python'),'-u',source,*common,*tail]
        require_stage(launch,terminal,argv,reader.inputs[source],controller);stages[source]=launch
    stdout=reader.json(AUDIT_STAGE+'/stdout.log')
    require(stdout.get('complete') is True and stdout.get('split')=='test' and stdout.get('cases')==1920
        and stdout.get('TEST_permitted') is False and stdout.get('out')==RESULT
        and stdout.get('native_successes_by_method')==result.get('native_successes_by_method'),'actual754 TEST stdout required')
    launch=reader.json(RUNTIME+'/launch.json',evidence=True);terminal=reader.json(RUNTIME+'/terminal.json',evidence=True)
    require_owner_completion(launch,terminal,stages[OWNER],result,reader.inputs[OWNER],control['author_outcome_protocol'],
        reader.pointer(RUNTIME+'/terminal.json'))
    #Only after genuine full TEST completion do we read dynamic TEST setup/source metadata.
    suite=reader.json(SUITE,evidence=True);test_lock=reader.json(SUITE_LOCK,evidence=True);reader.frozen=test_lock['files']
    require(test_lock.get('approved_for_execution') is True and test_lock.get('actual_formal_split')=='test'
        and test_lock.get('policy_sha256')==POLICY and test_lock.get('model_or_policy_parameters_changed') is False,'actual frozen826 TEST setup required')
    require(reader.lookup(reader.frozen,SUITE)==reader.inputs[SUITE] and suite.get('validation_release_path')==VAL_RELEASE
        and suite.get('source_manifests',{}).get('test',{}).get('path')==MANIFEST,'TEST suite must bind actual sources and completed VAL release')
    val=reader.json(VAL_RELEASE,frozen=True);H.require_release_header(val)
    val_pointer=reader.pointer(VAL_RELEASE)
    require(test_lock.get('original_945_VAL_release')==val_pointer and val.get('source_sha256')==control['resume_auditor']['sha256']
        and val.get('resume_control')==reader.pointer(CONTROL) and val.get('resume_control_lock')==reader.pointer(LOCK),'actual824 VAL prerequisite required')
    protocol=suite['author_outcome_protocol']
    require(protocol==control['author_outcome_protocol'] and result.get('author_outcome_protocol')==protocol
        and result.get('admission_driver')==suite.get('formal_case_driver') and result.get('admission_guard')==suite.get('formal_admission_guard')
        and suite.get('formal_result_auditor')==dict(path=AUDITOR,sha256=reader.inputs[AUDITOR]),'unchanged TEST executable/protocol graph required')
    pointers=[result['admission_driver'],result['admission_guard'],protocol['implementation'],protocol['runtime_consumer'],protocol['original_runtime']]
    for pointer in pointers:reader.raw(pointer['path'],expected=pointer['sha256'],frozen=True,original=True)
    for name in (OWNER,AUDITOR,PREPARER,SOURCE_PREPARER):reader.raw(name,frozen=True,original=True)
    manifest=reader.pointed(suite['source_manifests']['test'],frozen=True)
    require(manifest.get('original_source',{}).get('path')==ORIGINAL_MANIFEST,'exact original TEST manifest pointer required')
    original=reader.pointed(manifest['original_source'],frozen=True)
    source_lock=reader.json(SOURCE_LOCK,evidence=True)
    require(result.get('source_input_lock')==reader.pointer(SOURCE_LOCK)
        and source_lock.get('source_manifest_sha256')==reader.inputs[MANIFEST],'real756 TEST source lock pointer required')
    sources=source_grid(manifest,original,source_lock,val_pointer,reader.inputs[SOURCE_PREPARER])
    routes=reader.pointed(suite['routes'],frozen=True,original=True)
    plan=reader.json(RUNTIME+'/plan.json',evidence=True)
    rows=require_plan(plan,sources,routes,protocol)
    require(launch.get('suite_lock_sha256')==reader.inputs[SUITE_LOCK]
        and launch.get('split_source_lock_sha256')==reader.inputs[SOURCE_LOCK]
        and launch.get('source_manifest')==suite['source_manifests']['test'] and launch.get('author_outcome_protocol')==protocol,
        'actual753 TEST owner PID/source/manifest/lock binding required')
    require_pointers(rows,result['cases'],result['admission_driver_receipts'])
    reducer=H.load_reducer(reader);selected=[]
    for row,case_pointer,driver_pointer in zip(rows,result['cases'],result['admission_driver_receipts']):
        expected={key:row[key] for key in H.BINDINGS}
        require(row['admission_driver']==result['admission_driver'] and row['admission_guard']==result['admission_guard'],'exact released driver/guard required')
        reader.raw(row['case_worker']['path'],expected=row['case_worker']['sha256'],frozen=True,original=True)
        cpu_leaf=RUNTIME+'/CPU/'+row['job_id'];native_leaf=RUNTIME+'/native/'+row['job_id']
        cpu=reader.json(cpu_leaf+'/CPU_preflight.json',evidence=True)
        require(cpu==reader.json(cpu_leaf+'/stdout.log') and cpu.get('complete') is True and cpu.get('native_started') is False
            and cpu.get('GPU_started') is False and cpu.get('actual_split')=='test' and cpu.get('execution_protocol_changed') is True,
            'actual CPU-only TEST receipt/stdout required')
        native_terminal=None
        for leaf,check in ((cpu_leaf,True),(native_leaf,False)):
            child=reader.json(leaf+'/launch.json',evidence=True)
            end=reader.json(leaf+'/terminal.json',evidence=not check);H.require_clean_terminal(end)
            require(child.get('argv')==row['argv']+(['--check'] if check else []) and child.get('check_only') is check
                and child.get('owner_pid')==launch['pid'] and child.get('native_slots')==(0 if check else 1)
                and child.get('GPU_slots')==(0 if check else int(row['method']=='physmani'))
                and child.get('execution_protocol_changed') is True
                and child.get('expected_identity_sha256')==(None if check else cpu['identity']['sha256'])
                and all(child.get(k)==v and cpu.get(k)==v for k,v in expected.items()),'actual753 child argv/identity/resources/protocol required')
            if not check:native_terminal=end
        #754 appends the case pointer but does not duplicate its SHA in inputs.
        #The explicit released pointer and native terminal independently bind it.
        case=reader.pointed(case_pointer);driver=reader.pointed(driver_pointer,evidence=True)
        require(native_terminal.get('case')==case_pointer and native_terminal.get('admission_driver_receipt')==driver_pointer
            and cpu.get('identity')==case.get('identity') and cpu.get('component_domain')==case.get('component_domain'),
            'actual TEST CPU/native/case pointers required')
        require(driver.get('schema')=='r10_explicit_author_outcome_formal_driver_v751' and driver.get('complete') is True
            and driver.get('actual_split')=='test' and driver.get('execution_protocol_changed') is True
            and driver.get('case')==case_pointer and all(driver.get(k)==v for k,v in expected.items()),'actual751 TEST driver receipt required')
        require_case(case,row,sources[(row['task'],row['raw'])],protocol,reader.inputs[SOURCE_LOCK])
        metrics,diagnostics=reducer.reduce_saved_case(case)
        metrics={k:(v if reducer.finite(v) else None) for k,v in metrics.items()}
        selected.append(dict(task=row['task'],method=row['method'],source_raw_episode_id=row['raw'],source_row_sha256=row['source_row_sha256'],
            case=case_pointer,admission_driver_receipt=driver_pointer,evidence_runtime=RUNTIME,actual_owner_pid=launch['pid'],
            CPU_receipt=reader.pointer(cpu_leaf+'/CPU_preflight.json'),native_launch=reader.pointer(native_leaf+'/launch.json'),
            native_terminal=reader.pointer(native_leaf+'/terminal.json'),official_task_success=case['official_task_success'],
            native_terminal_failure=case['native_terminal_failure'],metrics=metrics,recorded_window_diagnostics=diagnostics))
    for method in METHODS:require(sum(r['official_task_success'] for r in selected if r['method']==method)==result['native_successes_by_method'][method],
        'all TEST native outcomes must remain in released cohort')
    require(len(selected)==1920,'exact1920 TEST cases required')
    subset=reader.json(SUBSET)
    return selected,result,subset,reducer


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--test-results-sha256',required=True)
    args=parser.parse_args(argv);require(valid_sha(args.test_results_sha256),'explicit complete754 TEST_results SHA required')
    require(not (ROOT/OUTPUT).exists(),'preserve existing851 output; never overwrite')
    reader=Reader(ROOT);rows,result,subset,reducer=collect(reader,args.test_results_sha256)
    source=str(Path(__file__).relative_to(ROOT));reader.inputs[source]=sha_bytes(Path(__file__).read_bytes())
    report=dict(schema='r10_full1920_TEST_saved_motion_summary_v851',complete=True,created_utc=datetime.now(timezone.utc).isoformat(),
        read_only_analysis=True,split='test',task_count=16,source_count=640,case_count=1920,methods=list(METHODS),
        task_source_counts={t:40 for t in TASKS},policy_sha256=POLICY,test_results=reader.pointer(RESULT),
        actual_test_auditor=dict(path=AUDITOR,sha256=result['source_sha256']),provenance_runtime_counts={RUNTIME:1920},
        metric_definitions=subset['metric_definitions'],statistical_definition=dict(subset['statistical_definition'],
            overall='pool640 equally weighted cases per method; never average task statistics'),
        diagnostic_definition=subset['diagnostic_definition'],hard_budget_disclosure=subset['hard_budget_disclosure'],
        common_RML_planner_configuration=dict(parameters=PARAMETERS,profile_sha256=PROFILE,cases=1920,
            parameter_kind='saved path-parameter planning configuration, not a per-joint measured bound'),
        overall_by_method={m:reducer.summarize([r for r in rows if r['method']==m]) for m in METHODS},
        by_task_method=[dict(task=t,method=m,**reducer.summarize([r for r in rows if (r['task'],r['method'])==(t,m)])) for t in TASKS for m in METHODS],
        case_evidence=rows,inputs_sha256=reader.inputs,source_sha256=reader.inputs[source],
        audited_helper=dict(path=HELPER,sha256=HELPER_SHA),reducer_source=dict(path=REDUCER,sha256=PINS[REDUCER]),
        namespace_evidence_scope='827 stage launch records inherit controller820 proof; original753/754 writers do not capture per-child/auditor live namespace receipts',
        full1920_TEST_results_and_complete827_observed=True,new_release_granted_by_this_summary=False,native_false_retained=True,
        success_rate_or_acceleration_filter_used=False,new_joint_differencing_used=False,alignment_motion_mirror_pooled=False,
        transitive_TRAIN_model_OS_inputs_walked=False,raw_reset_PKL_image_HDF5_contents_opened=False,TEST_used_for_fitting=False,
        native_GPU_or_jobs_started=False,frozen_source_or_lock_modified=False,TXT_or_handoff_written=False)
    with (ROOT/OUTPUT).open('x') as f:f.write(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(complete=True,output=OUTPUT,sha256=sha_bytes((ROOT/OUTPUT).read_bytes()),case_count=1920)))


if __name__=='__main__':main()
