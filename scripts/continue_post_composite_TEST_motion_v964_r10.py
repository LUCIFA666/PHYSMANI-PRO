"""Inactive, one-time saved-motion postprocessing of one frozen TEST batch.

Without --execute, main performs no I/O. An activated controller only reads
fixed completion metadata and starts the already reviewed963 analysis once.
It never starts Native/GPU/TRAIN or changes another controller or its flags.
Its own host process does not claim to inherit a recorded820 namespace.
"""
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import signal
import subprocess
import sys
import tempfile
import threading
import time
import traceback

ROOT=Path(__file__).parent.parent
SOURCE='scripts/continue_post_composite_TEST_motion_v964_r10.py'
TEST_SOURCE='tests/test_post_composite_TEST_motion_controller_v964.py'
SUMMARY_SOURCE='scripts/summarize_composite_TEST_motion_v963_r10.py'
SUMMARY_TEST='tests/test_composite_TEST_motion_v963.py'
CONTROLLER_SOURCE='scripts/continue_composite_TEST_v958_r10.py'
OWNER_SOURCE='scripts/run_composite_TEST_owner_v956_r10.py'
AUDITOR_SOURCE='scripts/audit_composite_TEST_results_v957_r10.py'
SOURCES=(SOURCE,TEST_SOURCE,SUMMARY_SOURCE,SUMMARY_TEST,CONTROLLER_SOURCE,OWNER_SOURCE,AUDITOR_SOURCE)
RUNTIME='runtime/framework/post_complete_composite_TEST_motion_v964'
SETUP='artifacts/framework/multitask_r10/post_complete_composite_TEST_motion_v964_setup'
CONTROL=SETUP+'/control.json'
LOCK=SETUP+'/inputs.json'
CONTROLLER='runtime/framework/ordered_composite_TEST_v958'
OWNER_RUNTIME='runtime/framework/formal_TEST_composite_resume_v956'
ORIGINAL_SETUP='artifacts/framework/multitask_r10/formal_TEST_composite_resume_v956_setup'
RESUME_CONTROL=ORIGINAL_SETUP+'/resume_control.json'
RESUME_LOCK=ORIGINAL_SETUP+'/resume_inputs.json'
RESULT='artifacts/framework/multitask_r10/formal_TEST_composite_resume_v956/TEST_results.json'
SUMMARY_OUTPUT=OWNER_RUNTIME+'/full_sixteen_task_motion_summary_v963.json'
STAGE='saved_motion_v963'
SCHEMA='r10_readonly_post_complete_composite_TEST_motion_control_v964'
LOCK_SCHEMA='r10_readonly_post_complete_composite_TEST_motion_inputs_v964'
RESULT_SCHEMA='r10_three_tier_composite_formal_TEST_results_v957'
SUMMARY_SCHEMA='r10_full1920_three_tier_TEST_saved_motion_summary_v963'
POLICY='eb8f768d99975a7c2de2c36de5283d64898c2296f33fb342e332310eb32db6ef'
METHODS=('cv_geometry','context_geometry','physmani')
TASKS=('beat_the_rotating_buzz','beat_the_rotating_buzz_high_speed','insert_onto_rotating_peg','insert_onto_rotating_peg_high_speed',
    'moving_basketball_in_hoop','moving_basketball_in_hoop_high_speed','pick_moving_target_on_the_table','pick_moving_target_on_the_table_high_speed',
    'place_cups_on_rotating_frame','place_cups_on_rotating_frame_high_speed','push_moving_button','push_moving_button_high_speed',
    'put_rubbish_in_moving_bin','put_rubbish_in_moving_bin_high_speed','remove_cups_from_rotating_frame','remove_cups_from_rotating_frame_high_speed')
STOP_PATHS=('runtime/framework/multitask_r10_STOP',CONTROLLER+'/STOP',OWNER_RUNTIME+'/STOP',RUNTIME+'/STOP')
HISTORICAL_ORIGINAL='runtime/framework/formal_TEST_after_environment_resume_v753'
HISTORICAL_PREVIOUS='runtime/framework/formal_TEST_concurrent_resume_v902'
STOP_PATHS=(*STOP_PATHS,HISTORICAL_PREVIOUS+'/STOP','runtime/framework/ordered_concurrent_TEST_v907/STOP')
EVENT=threading.Event()


def require(value,message):
    if not value:raise ValueError('post TEST motion964: '+message)


def utc():return datetime.now(timezone.utc).isoformat()
def sha(raw):return hashlib.sha256(raw).hexdigest()
def valid_sha(value):return isinstance(value,str) and len(value)==64 and not set(value)-set('0123456789abcdef')
def root_path(root=None):return Path(ROOT if root is None else root).resolve()

def pointer(value,name=None):
    require(isinstance(value,dict) and set(value)=={'path','sha256'} and isinstance(value.get('path'),str)
        and value['path'] and not Path(value['path']).is_absolute() and '..' not in PurePosixPath(value['path']).parts
        and str(PurePosixPath(value['path']))==value['path'] and valid_sha(value.get('sha256'))
        and (name is None or value['path']==name),'exact project path/SHA pointer required')
    return value


def _allowed(name):
    if name in {*SOURCES,CONTROL,LOCK,RESUME_CONTROL,RESUME_LOCK,RESULT,SUMMARY_OUTPUT}:return True
    if name in {runtime+'/'+leaf for runtime in (CONTROLLER,OWNER_RUNTIME) for leaf in ('launch.json','status.json','terminal.json','failure.json')}:return True
    return name in {CONTROLLER+'/stages/'+stage+'/'+leaf for stage in ('formal_TEST_1920','independent_TEST_results')
        for leaf in ('launch.json','terminal.json')}


def read_bytes(root,name,expected=None,*,missing=False):
    require(isinstance(name,str) and _allowed(name) and not Path(name).is_absolute() and '..' not in PurePosixPath(name).parts,
        'only fixed postprocessing metadata may be read: '+str(name))
    path=Path(root)/name
    try:
        require(path.resolve(strict=True)==path,'metadata aliases refused')
        # Dynamic status files are atomically replaced. The single bytes read is
        # the observation; it is never advertised as a frozen future pointer.
        raw=path.read_bytes()
    except FileNotFoundError:
        if missing:return None
        raise
    require(expected is None or valid_sha(expected) and sha(raw)==expected,'bound metadata/source SHA changed: '+name)
    return raw


def read_json(root,name,expected=None,*,missing=False):
    raw=read_bytes(root,name,expected,missing=missing)
    if raw is None:return None,None
    value=json.loads(raw);require(isinstance(value,dict),'JSON metadata object required')
    return value,dict(path=name,sha256=sha(raw))


def _canonical_argument(root,value,name):
    require(isinstance(value,str),'fixed control path required')
    path=Path(value) if Path(value).is_absolute() else Path(root)/value
    require(path==Path(root)/name,'only exact fixed control namespace accepted')
    return name


def _lookup(files,root,name):
    require(isinstance(files,dict),'explicit small input map required')
    matches=[files[key] for key in (name,str(Path(root)/name)) if key in files]
    require(len(matches)==1 and valid_sha(matches[0]),'unique selected metadata SHA required: '+name)
    return matches[0]


def _pid(value):
    require(type(value.get('pid')) is int and value['pid']>0 and type(value.get('start_ticks')) is int and value['start_ticks']>0,'actual PID/start required')
    return value['pid'],value['start_ticks']


def option(argv,name):
    require(isinstance(argv,list) and argv.count(name)==1,'one '+name+' required')
    i=argv.index(name);require(i+1<len(argv),'missing '+name+' value');return argv[i+1]


def stopped(root):return EVENT.is_set() or any((Path(root)/name).exists() for name in STOP_PATHS)
def guard(root):
    if stopped(root):raise InterruptedError('global/958/956/private964 STOP')


def load_control(control_path=CONTROL,lock_path=LOCK,lock_sha=None,*,root=None):
    root=root_path(root);guard(root)
    _canonical_argument(root,control_path,CONTROL);_canonical_argument(root,lock_path,LOCK)
    require(valid_sha(lock_sha),'external964 small-lock SHA required')
    lock,lp=read_json(root,LOCK,lock_sha)
    require(lock.get('schema')==LOCK_SCHEMA and lock.get('execution_enabled') is True,'explicit frozen964 activation required')
    files=lock.get('files');expected_names={*SOURCES,CONTROL,RESUME_CONTROL,RESUME_LOCK,CONTROLLER+'/launch.json'}
    require(isinstance(files,dict) and len(files)==len(expected_names),'exact small source/control map required; no transitive graph')
    canonical={}
    for key,value in files.items():
        require(isinstance(key,str),'input map path required')
        name=key
        if Path(key).is_absolute():
            require(Path(key).is_relative_to(root),'external input refused')
            name=str(Path(key).relative_to(root))
        require(name in expected_names and name not in canonical and valid_sha(value),'unknown/duplicate/non-SHA input refused before open')
        canonical[name]=value
    require(set(canonical)==expected_names,'complete small source binding required')
    control,cp=read_json(root,CONTROL,canonical[CONTROL])
    require(control.get('schema')==SCHEMA and control.get('execution_enabled') is True and control.get('read_only_analysis') is True
        and control.get('automatic_retry') is False and control.get('policy_sha256')==POLICY,'explicit read-only unchanged TEST postprocessing required')
    fixed=dict(runtime=RUNTIME,controller_runtime=CONTROLLER,owner_runtime=OWNER_RUNTIME,test_results=RESULT,motion_summary=SUMMARY_OUTPUT)
    require(all(control.get(k)==v for k,v in fixed.items()),'one fixed controller/owner/result/output required')
    require(type(control.get('poll_seconds')) is int and 1<=control['poll_seconds']<=30,'poll interval must be1..30 seconds')
    sources=control.get('sources');require(isinstance(sources,dict) and set(sources)==set(SOURCES),'all seven source/test pointers required')
    for name,p in sources.items():
        pointer(p,name);require(canonical[name]==p['sha256'],'source pointer differs from small lock')
        read_bytes(root,name,p['sha256'])
    for key,name in (('resume_control',RESUME_CONTROL),('resume_control_lock',RESUME_LOCK),('controller_launch',CONTROLLER+'/launch.json')):
        p=pointer(control.get(key),name);require(canonical[name]==p['sha256'],'frozen batch pointer differs: '+key)
    original,op=read_json(root,RESUME_CONTROL,control['resume_control']['sha256'])
    original_lock,olp=read_json(root,RESUME_LOCK,control['resume_control_lock']['sha256'])
    require(original_lock.get('schema')=='r10_composite_formal_TEST_resume_inputs_v955' and original_lock.get('approved_for_execution') is True
        and _lookup(original_lock.get('files'),root,RESUME_CONTROL)==op['sha256'],'actual955 frozen continuation required')
    require(original.get('schema')=='r10_composite_formal_TEST_resume_control_v955' and original.get('split')=='test'
        and original.get('expected_cases')==1920 and original.get('source_count')==640 and original.get('policy_sha256')==POLICY
        and original.get('physical_policy_changed') is False,'actual full1920 TEST continuation required')
    for key,name in (('resume_controller',CONTROLLER_SOURCE),('resume_owner',OWNER_SOURCE),('resume_auditor',AUDITOR_SOURCE)):
        require(original.get(key)==sources[name] and _lookup(original_lock['files'],root,name)==sources[name]['sha256'],'original955 actual writer source binding changed')
    launch,_=read_json(root,CONTROLLER+'/launch.json',control['controller_launch']['sha256']);_pid(launch)
    require(launch.get('source_sha256')==sources[CONTROLLER_SOURCE]['sha256'] and launch.get('resume_control')==op
        and launch.get('resume_control_lock')==olp and launch.get('automatic_retry') is False,'actual selected958 launch/batch required')
    guard(root)
    return control,dict(control=cp,lock=lp,sources=deepcopy(sources),original_control=original,original_lock=original_lock,
        controller_launch=deepcopy(control['controller_launch']),canonical=canonical)


def _clean_stage(value):
    require(value.get('complete') is True and type(value.get('returncode')) is int and value['returncode']==0
        and value.get('automatic_retry') is False,'actual958 stage did not complete cleanly')


def require_three_tier_evidence(result,original,status,owner_launch,owner_terminal):
    """Bind published957 metadata; no case or source payload is opened here."""
    runtimes=(HISTORICAL_ORIGINAL,HISTORICAL_PREVIOUS,OWNER_RUNTIME)
    stages=result.get('actual_stage_owners')
    require(isinstance(stages,list) and len(stages)==3
        and [row.get('runtime') for row in stages]==list(runtimes),'three actual physical evidence tiers required')
    by_runtime={row['runtime']:row for row in stages}
    require(len({_pid(row) for row in stages})==3,'three distinct recorded physical owner identities required')
    counts={row['runtime']:row.get('completed') for row in stages}
    require(all(type(value) is int and value>0 for value in counts.values()) and sum(counts.values())==1920
        and counts[HISTORICAL_ORIGINAL]+counts[HISTORICAL_PREVIOUS]==len(original['completed_job_ids'])
        and counts[OWNER_RUNTIME]==1920-len(original['completed_job_ids']),
        'three-tier counts must retain every old outcome exactly once')
    for runtime,launch,end in ((HISTORICAL_ORIGINAL,original['original_launch'],original['original_terminal']),
            (HISTORICAL_PREVIOUS,original['previous_launch'],original['previous_terminal']),
            (OWNER_RUNTIME,owner_launch,owner_terminal)):
        require(by_runtime[runtime].get('launch')==launch and by_runtime[runtime].get('terminal')==end,
            'actual evidence owner pointers must match their original or current physical metadata')
    entries=result['evidence_tiers'];case_pointers=result['cases'];drivers=result['admission_driver_receipts']
    require(isinstance(entries,list) and len(entries)==1920
        and len({row.get('job_id') for row in entries})==1920
        and {row.get('job_id') for row in entries}==set(status['completed']),
        'published evidence must cover every actual completed key without duplicates')
    require(len({pointer(p)['path'] for p in case_pointers})==1920
        and len({pointer(p)['path'] for p in drivers})==1920,'unique physical case and751 receipt paths required')
    observed={name:0 for name in runtimes};batch=[];imported=set();new=set()
    for index,row in enumerate(entries):
        key=row['job_id'];runtime=row.get('evidence_runtime')
        require(runtime in observed,'unrecognized physical evidence runtime')
        observed[runtime]+=1;owner=by_runtime[runtime]
        require(type(row.get('imported_completed')) is bool and row['imported_completed'] is (runtime!=OWNER_RUNTIME)
            and row.get('actual_stage_owner_pid')==owner['pid']
            and row.get('actual_stage_owner_start_ticks')==owner['start_ticks'],
            'each case must retain its real physical owner and import marker')
        pointer(case_pointers[index]);pointer(drivers[index])
        record=status['completed'][key]
        require(row.get('case')==case_pointers[index]==record.get('case')
            and row.get('admission_driver_receipt')==drivers[index]==record.get('admission_driver_receipt')
            and record.get('complete') is True and type(record.get('native_success')) is bool
            and type(record.get('returncode')) is int and record['returncode']==0
            and record.get('interrupted') is False and record.get('timed_out') is False
            and record.get('automatic_retry') is False
            and 'motion_pass' in record and (record['motion_pass'] is None or type(record['motion_pass']) is bool),
            'nativeFalse/motion unknown and physical case pointers must remain literal')
        entry=row.get('execution_entry',{})
        if runtime==HISTORICAL_ORIGINAL:
            require(entry.get('kind')=='original751' and row.get('batch_driver_receipt') is None
                and 'batch_driver_receipt' not in record,'original753 cannot acquire901 evidence')
        else:
            require(entry.get('kind')=='batched901','actual902/956 must retain real901 entry')
            bp=pointer(row.get('batch_driver_receipt'))
            require(record.get('batch_driver_receipt')==bp,'actual901 sidecar pointer differs')
            batch.append(bp)
        (imported if runtime!=OWNER_RUNTIME else new).add(key)
    require(observed==counts and imported==set(original['completed_job_ids'])
        and result.get('batch_driver_receipts')==batch,'actual three-tier evidence and batch receipts differ')
    history=result.get('historical_infrastructure_attempts',{})
    require(history.get('job_ids')==original.get('failed_job_ids')
        and set(history.get('job_ids',[]))<=new
        and history.get('previous_status')==original.get('previous_status')
        and history.get('previous_terminal')==original.get('previous_terminal')
        and history.get('counted_as_native_task_failures') is False
        and history.get('all_keys_reexecuted_once_in_new_namespace') is True,
        'historical infrastructure attempts stay separate and their keys must finish in the new tier')
    return counts,deepcopy(stages)


def completion_snapshot(root,control,binding):
    """Return None while pending; only read957 results after all completion gates."""
    guard(root);sources=binding['sources'];original=binding['original_control']
    controller,_=read_json(root,CONTROLLER+'/launch.json',binding['controller_launch']['sha256'])
    failure,_=read_json(root,CONTROLLER+'/failure.json',missing=True)
    state,sp=read_json(root,CONTROLLER+'/status.json',missing=True)
    if failure:
        if failure.get('user_stop') is True:raise InterruptedError('selected958 recorded a user stop')
        raise RuntimeError('selected958 recorded failure; analysis will not retry it')
    if state is None:return None
    stage=state.get('stage')
    if stage=='user_stopped':raise InterruptedError('selected958 user stopped')
    require(stage in ('formal_TEST_1920','independent_TEST_results','complete'),'selected958 failed/draining/unrecognized stage')
    require(state.get('automatic_retry') is False and not state.get('infrastructure_failure_detected')
        and state.get('owner_error') is None,'selected958 infrastructure failure')
    launch,lptr=read_json(root,OWNER_RUNTIME+'/launch.json',missing=True)
    status,osp=read_json(root,OWNER_RUNTIME+'/status.json',missing=True)
    if launch is not None:
        require(launch.get('source_sha256')==sources[OWNER_SOURCE]['sha256'] and launch.get('resume_control')==control['resume_control']
            and launch.get('resume_control_lock')==control['resume_control_lock'] and launch.get('policy_sha256')==POLICY
            and launch.get('author_outcome_protocol')==original.get('author_outcome_protocol'),'wrong actual956 owner/batch')
    if status is not None:
        require(status.get('failed')=={} and status.get('owner_error') is None and status.get('stop_requested') is False
            and status.get('drain_requested') is False,'selected956 stopped/draining/failed; no analysis')
    if stage!='complete':return None
    require(launch is not None and status is not None,'complete958 is missing956 owner records')
    final,fp=read_json(root,CONTROLLER+'/terminal.json')
    require(final.get('complete') is True and final.get('automatic_retry') is False,'genuine selected958 final completion required')
    result_pointer=pointer(final.get('TEST_results'),RESULT)
    terminal,tp=read_json(root,OWNER_RUNTIME+'/terminal.json')
    for value in (status,terminal):
        require(value.get('split')=='test' and value.get('policy_sha256')==POLICY
            and value.get('author_outcome_protocol')==original.get('author_outcome_protocol')
            and value.get('resume_control')==control['resume_control'] and value.get('resume_control_lock')==control['resume_control_lock']
            and value.get('automatic_retry') is False,'complete956 records must bind the selected batch')
    require(terminal.get('complete') is True and terminal.get('split')=='test' and terminal.get('expected')==terminal.get('completed')==1920
        and terminal.get('failed')==0 and terminal.get('owner_error') is None and terminal.get('stop_requested') is False
        and terminal.get('drain_requested') is False and terminal.get('drained') is False and terminal.get('automatic_retry') is False
        and terminal.get('CPU_ready_unexecuted')==[] and type(terminal.get('CPU_not_launched')) is int and terminal['CPU_not_launched']==0
        and terminal.get('formal_release_requires_independent_audit') is True,'complete1920 clean956 terminal required before result read')
    require(status.get('expected')==1920 and isinstance(status.get('completed'),dict) and len(status['completed'])==1920
        and status.get('active')==status.get('checking')==status.get('ready_jobs')==[]
        and status.get('CPU_ready')==status.get('native_slots')==status.get('GPU_slots')==0,'complete1920 idle956 status required before result read')
    recorded=controller.get('frozen_environment',{})
    stages={}
    for name,source in (('formal_TEST_1920',OWNER_SOURCE),('independent_TEST_results',AUDITOR_SOURCE)):
        leaf=CONTROLLER+'/stages/'+name
        actual,ap=read_json(root,leaf+'/launch.json');end,ep=read_json(root,leaf+'/terminal.json');_pid(actual);_clean_stage(end)
        argv=actual.get('argv')
        require(isinstance(argv,list) and argv[:4]==[str(Path(root)/'.conda-env/bin/python'),'-B','-u',source]
            and actual.get('source_sha256')==sources[source]['sha256'] and actual.get('automatic_retry') is False
            and actual.get('inherited_frozen_environment')==recorded
            and actual.get('current_mount_namespace')==recorded.get('current_mount_namespace'),'actual958 source/stage receipt changed')
        stages[source]=actual
    require(_pid(stages[OWNER_SOURCE])==_pid(launch) and stages[OWNER_SOURCE]['argv'][3:]==launch.get('argv'),'956 must be the actual selected958 child')
    require(stages[AUDITOR_SOURCE]['argv'][4:]==['--resume-control',RESUME_CONTROL,'--resume-lock',RESUME_LOCK,
        '--resume-lock-sha256',control['resume_control_lock']['sha256']],'actual957 stage command must bind the original955 lock')
    # First result read: an incomplete/failed/wrong batch never reaches this line.
    result,rp=read_json(root,RESULT,result_pointer['sha256'])
    require(result.get('schema')==RESULT_SCHEMA and result.get('complete') is True and result.get('split')=='test'
        and result.get('source_cases')==640 and result.get('method_cases')==result.get('completed_method_cases')==1920
        and result.get('methods')==list(METHODS) and result.get('tasks')==list(TASKS)
        and result.get('source_sha256')==sources[AUDITOR_SOURCE]['sha256'] and result.get('policy_sha256')==POLICY
        and result.get('resume_control')==control['resume_control'] and result.get('resume_control_lock')==control['resume_control_lock']
        and result.get('source_input_lock')==original.get('source_inputs') and result.get('validation_release')==original.get('validation_release')
        and result.get('author_outcome_protocol')==original.get('author_outcome_protocol') and result.get('independent_composite_audit') is True
        and result.get('resource_profile')==original.get('resource_profile') and result.get('resource_qualification')==original.get('resource_qualification')
        and result.get('execution_implementation')==original.get('execution_implementation')
        and result.get('native_failures_retained') is True and result.get('source_outcome_filtering') is False
        and result.get('Value_NGFF_FASTER_used') is False and result.get('automatic_retry') is False
        and result.get('used_evidence_rechecked_at_close') is True and result.get('original754_or906_single_result_claimed') is False
        and all(isinstance(result.get(k),list) and len(result[k])==1920 for k in ('cases','admission_driver_receipts','evidence_tiers'))
        and isinstance(result.get('batch_driver_receipts'),list),
        'actual complete bound957 result required')
    invocation=result.get('auditor_invocation',{})
    require(_pid(invocation)==_pid(stages[AUDITOR_SOURCE]) and invocation.get('source')==sources[AUDITOR_SOURCE]
        and invocation.get('argv')==stages[AUDITOR_SOURCE]['argv'][3:],'actual957 invocation differs from selected958 stage')
    counts,stage_owners=require_three_tier_evidence(result,original,status,lptr,tp)
    require(_pid(stage_owners[-1])==_pid(launch),'published new physical owner must be the actual selected958 child')
    guard(root)
    return dict(result=rp,result_sha256=rp['sha256'],controller_status=sp,controller_terminal=fp,
        owner_launch=lptr,owner_status=osp,owner_terminal=tp,completed_utc=utc(),
        provenance_runtime_counts=counts,actual_stage_owners=stage_owners)


def validate_summary(summary,sp,stdout,control,binding,completion,root):
    require(summary.get('schema')==SUMMARY_SCHEMA and summary.get('complete') is True and summary.get('read_only_analysis') is True
        and summary.get('split')=='test' and summary.get('task_count')==16 and summary.get('source_count')==640 and summary.get('case_count')==1920
        and summary.get('methods')==list(METHODS) and summary.get('task_source_counts')=={task:40 for task in TASKS}
        and summary.get('policy_sha256')==POLICY and summary.get('source_sha256')==binding['sources'][SUMMARY_SOURCE]['sha256']
        and summary.get('test_results')==completion['result'] and summary.get('resume_control')==control['resume_control']
        and summary.get('resume_control_lock')==control['resume_control_lock'] and summary.get('actual_test_auditor')==binding['sources'][AUDITOR_SOURCE]
        and summary.get('full1920_three_tier_TEST_results_and_complete958_observed') is True
        and summary.get('provenance_runtime_counts')==completion['provenance_runtime_counts']
        and summary.get('actual_stage_owners')==completion['actual_stage_owners']
        and summary.get('new_release_granted_by_this_summary') is False and summary.get('native_false_retained') is True
        and summary.get('success_rate_or_acceleration_filter_used') is False and summary.get('native_GPU_or_jobs_started') is False
        and summary.get('TEST_used_for_fitting') is False and summary.get('TXT_or_handoff_written') is False,
        'actual963 full saved-motion output/binding required')
    inputs=summary.get('inputs_sha256')
    for name,expected in ((RESULT,completion['result_sha256']),(RESUME_CONTROL,control['resume_control']['sha256']),
        (RESUME_LOCK,control['resume_control_lock']['sha256']),(SUMMARY_SOURCE,binding['sources'][SUMMARY_SOURCE]['sha256'])):
        require(_lookup(inputs,root,name)==expected,'actual963 consumed input binding differs')
    require(stdout.get('complete') is True and stdout.get('output')==SUMMARY_OUTPUT and stdout.get('sha256')==sp['sha256']
        and stdout.get('case_count')==1920,'actual963 stdout/output SHA differs')


def _write(root,name,value,*,overwrite=False):
    require(name.startswith(RUNTIME+'/') and '..' not in PurePosixPath(name).parts,'write outside own964 runtime refused')
    path=Path(root)/name;require(path.parent.resolve()==path.parent and not path.is_symlink(),'canonical own output required')
    text=json.dumps(value,ensure_ascii=False,sort_keys=True,allow_nan=False,indent=2)+'\n'
    if not overwrite:
        with path.open('x') as f:f.write(text)
    else:
        with tempfile.NamedTemporaryFile('w',dir=path.parent,prefix='.964-',delete=False) as f:
            f.write(text);temp=f.name
        os.replace(temp,path)


@contextmanager
def observer_lock(root):
    path=Path(root)/(RUNTIME+'.lock')
    require(path.parent.resolve()==path.parent and not path.is_symlink(),'canonical private964 lock required')
    with path.open('a+') as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        yield


def ticks(pid):return int(Path('/proc',str(pid),'stat').read_text().rsplit(')',1)[1].split()[19])
def _wait(seconds):return EVENT.wait(min(30.,seconds))
def _popen(argv,**kwargs):return subprocess.Popen(argv,**kwargs)


def _stop_child(child,start):
    if child.poll() is None:
        require(ticks(child.pid)==start,'refuse to signal a changed child PID/start')
        child.terminate()
        try:child.wait(timeout=5)
        except subprocess.TimeoutExpired:
            require(child.poll() is None and ticks(child.pid)==start,'refuse to kill a changed child PID/start')
            child.kill();child.wait(timeout=5)


def run_analysis(root,control,binding,completion):
    # Independent full reread before the only permitted child launch.
    fresh,fresh_binding=load_control(CONTROL,LOCK,binding['lock']['sha256'],root=root)
    require(fresh==control and fresh_binding==binding,'activation source/control changed before963 launch')
    repeated=completion_snapshot(root,control,binding)
    require(repeated is not None and repeated['result']==completion['result'],'completion evidence changed before963 launch')
    guard(root);require(not (Path(root)/SUMMARY_OUTPUT).exists(),'963 output already exists; never duplicate manual analysis')
    leaf=RUNTIME+'/stages/'+STAGE;(Path(root)/leaf).mkdir(parents=True,exist_ok=False)
    argv=[str(Path(root)/'.conda-env/bin/python'),'-B','-u',SUMMARY_SOURCE,'--test-results-sha256',completion['result_sha256'],
        '--resume-lock-sha256',control['resume_control_lock']['sha256'],'--resume-control',RESUME_CONTROL,'--resume-lock',RESUME_LOCK]
    child=None;start=None;interrupted=False;started=time.monotonic()
    try:
        with (Path(root)/leaf/'stdout.log').open('xb') as stdout,(Path(root)/leaf/'stderr.log').open('xb') as stderr:
            guard(root)
            # Completion metadata reads may take time. Recheck the actual
            # source/control bytes at the final launch boundary as well.
            final_control,final_binding=load_control(CONTROL,LOCK,binding['lock']['sha256'],root=root)
            require(final_control==control and final_binding==binding,'activation source/control changed at963 launch boundary')
            require(not (Path(root)/SUMMARY_OUTPUT).exists(),'963 output appeared before launch; never duplicate analysis')
            guard(root)
            child=_popen(argv,cwd=root,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),stdout=stdout,stderr=stderr,start_new_session=True)
            start=ticks(child.pid)
            _write(root,leaf+'/launch.json',dict(created_utc=utc(),argv=argv,pid=child.pid,start_ticks=start,
                source=binding['sources'][SUMMARY_SOURCE],source_sha256=binding['sources'][SUMMARY_SOURCE]['sha256'],
                owner_pid=os.getpid(),owner_start_ticks=ticks(os.getpid()),control=binding['control'],input_lock=binding['lock'],
                TEST_results=completion['result'],standard_host_analysis=True,frozen_environment_inherited_claimed=False,automatic_retry=False))
            while child.poll() is None:
                if stopped(root):interrupted=True;_stop_child(child,start);break
                _wait(1.)
            code=child.wait()
        _write(root,leaf+'/terminal.json',dict(complete=code==0 and not interrupted,returncode=code,interrupted=interrupted,
            finished_utc=utc(),wall_seconds=time.monotonic()-started,automatic_retry=False))
        if interrupted:raise InterruptedError('STOP interrupted only owned963 analysis')
        require(code==0,'963 analysis exited unsuccessfully; no retry')
        guard(root)
        summary,sp=read_json(root,SUMMARY_OUTPUT)
        stdout_path=Path(root)/leaf/'stdout.log';raw=stdout_path.read_bytes();out=json.loads(raw)
        require(isinstance(out,dict),'one actual963 JSON stdout required')
        validate_summary(summary,sp,out,control,binding,completion,root)
        return sp
    except BaseException:
        if child is not None and child.poll() is None and start is not None:_stop_child(child,start)
        if child is not None and not (Path(root)/leaf/'terminal.json').exists():
            _write(root,leaf+'/terminal.json',dict(complete=False,returncode=child.poll(),interrupted=stopped(root),
                finished_utc=utc(),wall_seconds=time.monotonic()-started,automatic_retry=False))
        raise


def execute(control_path=CONTROL,lock_path=LOCK,lock_sha=None,*,root=None):
    root=root_path(root)
    guard(root)
    with observer_lock(root):
        control,binding=load_control(control_path,lock_path,lock_sha,root=root)
        require(not (root/RUNTIME).exists(),'preserve existing964 namespace; no restart')
        require(not (root/SUMMARY_OUTPUT).exists(),'preserve existing963 output; no duplicate child')
        (root/RUNTIME).mkdir(parents=True,exist_ok=False)
        _write(root,RUNTIME+'/launch.json',dict(created_utc=utc(),argv=list(sys.argv),pid=os.getpid(),start_ticks=ticks(os.getpid()),
            source_sha256=binding['sources'][SOURCE]['sha256'],control=binding['control'],input_lock=binding['lock'],
            monitored_controller_launch=binding['controller_launch'],standard_host_analysis=True,
            frozen_environment_inherited_claimed=False,automatic_retry=False))
        try:
            while True:
                guard(root);completion=completion_snapshot(root,control,binding)
                if completion is not None:break
                _write(root,RUNTIME+'/status.json',dict(updated_utc=utc(),stage='await_complete_TEST',automatic_retry=False,
                    monitored_controller_launch=binding['controller_launch'],Native_GPU_TRAIN_started=False),overwrite=True)
                _wait(control['poll_seconds'])
            _write(root,RUNTIME+'/status.json',dict(updated_utc=utc(),stage=STAGE,automatic_retry=False,TEST_results=completion['result']),overwrite=True)
            output=run_analysis(root,control,binding,completion);guard(root)
            terminal=dict(schema='r10_readonly_post_complete_composite_TEST_motion_terminal_v964',complete=True,finished_utc=utc(),
                motion_summary=output,TEST_results=completion['result'],control=binding['control'],input_lock=binding['lock'],
                resume_control=control['resume_control'],resume_control_lock=control['resume_control_lock'],
                standard_host_analysis=True,frozen_environment_inherited_claimed=False,
                automatic_retry=False,Native_GPU_TRAIN_started=False,TXT_modified=False)
            _write(root,RUNTIME+'/terminal.json',terminal)
            _write(root,RUNTIME+'/status.json',dict(updated_utc=utc(),stage='complete',automatic_retry=False,motion_summary=output),overwrite=True)
            return terminal
        except BaseException as error:
            user_stop=isinstance(error,InterruptedError) or stopped(root)
            _write(root,RUNTIME+'/failure.json',dict(complete=False,user_stop=user_stop,finished_utc=utc(),error=repr(error),
                traceback=traceback.format_exc(),automatic_retry=False,Native_GPU_TRAIN_started=False))
            _write(root,RUNTIME+'/status.json',dict(updated_utc=utc(),stage='user_stopped' if user_stop else 'requires_concrete_repair',
                error=repr(error),automatic_retry=False),overwrite=True)
            raise


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true');parser.add_argument('--control',default=CONTROL)
    parser.add_argument('--lock',default=LOCK);parser.add_argument('--lock-sha256')
    args=parser.parse_args(argv)
    if not args.execute:
        print(json.dumps(dict(execution_enabled=False,read_only_analysis=True,jobs_started=False,message='inactive; explicit frozen activation required')))
        return
    require(valid_sha(args.lock_sha256),'--execute requires external --lock-sha256')
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,lambda *_:EVENT.set())
    value=execute(args.control,args.lock,args.lock_sha256)
    print(json.dumps(dict(complete=True,motion_summary=value['motion_summary'],automatic_retry=False)))


if __name__=='__main__':main()
