"""Fresh composite continuation of TEST; original902 scheduling is retained.

Every adopted753/902/956/1011 outcome keeps its actual entry, evidence owner and bytes.
All historical infrastructure attempts remain separate attempts, never nativeFalse.
This inactive source never repairs, overwrites or restarts an earlier runtime.
"""
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack,contextmanager
from pathlib import Path
from datetime import datetime,timezone
import argparse,fcntl,hashlib,json,math,os,signal,subprocess,sys,threading,time,traceback

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from our_mani.multitask.evaluation import atomic_json,file_sha,digest,validate_completed_case
from our_mani.multitask.evaluation_roster16_v283 import TASKS,METHODS
from our_mani.multitask.input_stat_cache_v79 import verify_inputs
from our_mani.multitask.formal_suite_admission_v728 import read_formal_source,locked_pointer
from our_mani.multitask.common_rml_protocol_v530 import fingerprint,require_attestation
from our_mani.multitask.three_method_release16 import measured_motion
from scripts import run_formal_suite16_owner_v753_r10 as original_owner
from scripts import run_environment_resumed_formal_VAL_owner_v823_r10 as environment_owner
from scripts.run_native_resource_owner_v19_r10 import interrupt_owned_tree
from scripts.run_native_actuator_diagnostic_pair_v5_2_r10 import exclusive_pair
from scripts.run_formal_case_author_outcome_v751_r10 import BINDING_KEYS
from our_mani.multitask import composite_TEST_resume_v1040 as composite

CONTROL_SCHEMA=composite.CONTROL_SCHEMA
LOCK_SCHEMA=composite.LOCK_SCHEMA
PROFILE_SCHEMA='r10_frozen_concurrent_native_resource_profile_v1031'
FALLBACK_PROFILE_SCHEMA='r10_frozen_concurrent_native_resource_profile_v1005'
QUALIFICATION_SCHEMA='r10_independent_concurrent_TRAIN_resource_qualification_v1031'
PREVIOUS=composite.PREVIOUS
PREVIOUS_CONTROLLER=composite.PREVIOUS_CONTROLLER
RUNTIME=composite.RUNTIME
OUTPUT=composite.OUTPUT
ORIGINAL_ENTRY='scripts/run_formal_case_author_outcome_v751_r10.py'
BATCH_ENTRY='scripts/run_batched_formal_driver_v901_r10.py'
BATCH_SCHEMA='r10_scoped_frozen_lookup_bootstrap_receipt_v901'
COMMON_SOURCES=(ORIGINAL_ENTRY,'our_mani/multitask/formal_suite_admission_v728.py',
    'our_mani/multitask/rml_case_preflight_v742.py','scripts/run_task_executor_smoke_author_outcome_v743_r10.py',
    'scripts/run_task_executor_smoke_author_suite16_v744_r10.py')
ROW_PROVENANCE={'evidence_runtime','imported_completed','previous_row_sha256','execution_entry',
                'physical_argv','previous_worker_sha256'}
ENVIRONMENT_RECEIPT=None


def utc():return datetime.now(timezone.utc).isoformat()
def ticks(pid):return int(Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[19])


def pointer(path):
    path=Path(path).resolve()
    if not path.is_relative_to(ROOT):raise ValueError('project-only control evidence required')
    return dict(path=str(path.relative_to(ROOT)),sha256=file_sha(path))


def option(argv,name):
    if not isinstance(argv,list) or argv.count(name)!=1:raise ValueError('one explicit '+name+' required')
    i=argv.index(name)
    if i+1==len(argv):raise ValueError('missing '+name+' value')
    return argv[i+1]


def validate_resource_profile(value):
    fields={'schema','max_native_slots','max_GPU_slots','max_CPU_preflight_slots','ready_limit'}
    limits={PROFILE_SCHEMA:(12,5,6,24),FALLBACK_PROFILE_SCHEMA:(8,3,4,16)}
    if not isinstance(value,dict) or set(value)!=fields or value.get('schema') not in limits:
        raise ValueError('explicit1031 capacity-bounded or actual1005 fallback profile required')
    for name,maximum in zip(('max_native_slots','max_GPU_slots','max_CPU_preflight_slots','ready_limit'),limits[value['schema']]):
        if type(value.get(name)) is not int or not 1<=value[name]<=maximum:
            raise ValueError('bounded strict-integer resource quota required: '+name)
    if value['max_GPU_slots']>value['max_native_slots']:
        raise ValueError('GPU quota cannot exceed native quota')
    return deepcopy(value)


def require_capacity(control):
    actual=validate_resource_profile(control['resource_profile'])
    capacity=validate_resource_profile(control['resource_capacity_envelope'])
    kind=control.get('resource_capacity_source')
    limits=(12,5,6,24) if kind=='qualified1031' else (8,3,4,16) if kind=='qualified1005_fallback' else None
    schema=PROFILE_SCHEMA if kind=='qualified1031' else FALLBACK_PROFILE_SCHEMA
    names=('max_native_slots','max_GPU_slots','max_CPU_preflight_slots','ready_limit')
    if (limits is None or capacity!={'schema':schema,**dict(zip(names,limits))}
            or actual['schema']!=capacity['schema'] or any(actual[k]>capacity[k] for k in names)
            or control.get('resource_profile_within_capacity') is not True):
        raise ValueError('actual frozen quotas must be explicitly within the genuinely selected qualification capacity')
    if kind=='qualified1005_fallback' and actual!=capacity:
        raise ValueError('fallback must retain the actually qualified original1005 profile')
    return actual,capacity,kind


def select_native_jobs(ready,active,profile):
    """Pure stable selection, respecting both aggregate native and GPU quotas."""
    profile=validate_resource_profile(profile);selected=[]
    all_rows=[*ready,*active]
    names=[r['job_id'] for r in all_rows]
    if len(names)!=len(set(names)):raise ValueError('ready/active keys must be unique and disjoint')
    if any(r.get('method') not in METHODS for r in all_rows):raise ValueError('exact three methods required')
    native=len(active);gpu=sum(r['method']=='physmani' for r in active)
    if native>profile['max_native_slots'] or gpu>profile['max_GPU_slots']:
        raise ValueError('existing resource use already exceeds frozen quota')
    for row in ready:
        if native==profile['max_native_slots']:break
        if row['method']=='physmani' and gpu==profile['max_GPU_slots']:continue
        selected.append(row);native+=1;gpu+=int(row['method']=='physmani')
    return selected


def dispatch_control(runtime,cancelled,draining):
    runtime=Path(runtime)
    stops=(ROOT/'runtime/framework/multitask_r10_STOP',runtime/'STOP',ROOT/PREVIOUS/'STOP',ROOT/PREVIOUS_CONTROLLER/'STOP',
           ROOT/'runtime/framework/formal_TEST_composite_resume_v956/STOP',ROOT/'runtime/framework/ordered_composite_TEST_v958/STOP',ROOT/composite.ORIGINAL/'STOP',ROOT/'runtime/framework/formal_TEST_concurrent_resume_v902/STOP',ROOT/'runtime/framework/ordered_concurrent_TEST_v907/STOP',ROOT/'runtime/framework/ordered_env_resume_VAL_TEST_v827/STOP')
    if any(p.exists() for p in stops):cancelled.set()
    return bool(draining or (runtime/'DRAIN').exists())


def drained(draining,checking,active):return bool(draining and not checking and not active)


def require_partition(control,plan,status,terminal):
    return composite.require_partition(control,plan,status,terminal)


def validate_entry(entry):
    if not isinstance(entry,dict) or entry.get('kind') not in ('original751','batched901'):
        raise ValueError('explicit original751 or real901 entry required')
    target=ORIGINAL_ENTRY if entry['kind']=='original751' else BATCH_ENTRY
    p=entry.get('pointer')
    if not valid_pointer(p) or p.get('path')!=target:
        raise ValueError('exact selected entry pointer required')
    if entry['kind']=='original751':
        if set(entry)!={'kind','pointer'}:raise ValueError('original751 must not invent a batch sidecar')
    elif set(entry)!={'kind','pointer','batch_control','batch_lock'}:
        raise ValueError('901 requires explicit immutable control and lock pointers')
    elif any(not valid_pointer(entry[key]) for key in ('batch_control','batch_lock')):
        raise ValueError('901 control/lock must be exact project-relative SHA pointers')
    return deepcopy(entry)


def valid_pointer(value):
    if not isinstance(value,dict) or set(value)!={'path','sha256'}:return False
    name,checksum=value.get('path'),value.get('sha256')
    return (isinstance(name,str) and bool(name) and not Path(name).is_absolute() and '..' not in Path(name).parts
            and isinstance(checksum,str) and len(checksum)==64 and not set(checksum)-set('0123456789abcdef'))


def physical_argv(row,new_out):
    argv=deepcopy(row['physical_argv'])
    if len(argv)<3 or argv[2]!=ORIGINAL_ENTRY:raise ValueError('old753 exact751 entry required')
    option(argv,'--out')
    argv[argv.index('--out')+1]=str(new_out)
    return argv


def outer_argv(inner,entry):
    entry=validate_entry(entry)
    if entry['kind']=='original751':return deepcopy(inner)
    return [*inner[:2],entry['pointer']['path'],'--batch-control',entry['batch_control']['path'],
            '--batch-lock',entry['batch_lock']['path'],'--batch-lock-sha256',entry['batch_lock']['sha256'],
            '--entry','formal','--',*inner[3:]]


def resumed_jobs(old_plan,control,entry):
    return composite.resumed_jobs(old_plan,control,entry)


def require_resume_plan(control,old_plan,old_status,old_terminal,new_plan):
    return composite.require_resume_plan(control,old_plan,old_status,old_terminal,new_plan)


def require_batch_execution(value,row,*,check,expected_sources,case=None,original_receipt=None):
    """Pure v901 sidecar binding. No physical751 receipt is relabelled."""
    entry=validate_entry(row['execution_entry'])
    if entry['kind']=='original751':
        if value is not None:raise ValueError('original751 cannot claim a901 sidecar')
        return None
    expected_names={*COMMON_SOURCES,BATCH_ENTRY,'our_mani/multitask/frozen_lookup_batch_v900.py'}
    if (not isinstance(expected_sources,dict) or set(expected_sources)!=expected_names
            or any(not valid_pointer(p) or p['path']!=name for name,p in expected_sources.items())):
        raise ValueError('all seven real901/900/original implementation sources must be bound')
    tail=['--check'] if check else []
    if (not isinstance(value,dict) or value.get('schema')!=BATCH_SCHEMA or value.get('complete') is not True
            or type(value.get('check_only')) is not bool or value['check_only'] is not check
            or value.get('entry')!='formal' or value.get('original_entry')!=ORIGINAL_ENTRY
            or value.get('actual_outer_argv')!=row['argv'][2:]+tail
            or value.get('forwarded_original_argv')!=row['physical_argv'][2:]+tail
            or value.get('control')!=entry['batch_control'] or value.get('lock')!=entry['batch_lock']
            or value.get('sources')!=expected_sources
            or expected_sources.get(BATCH_ENTRY)!=entry['pointer']
            or value.get('scientific_policy_changed') is not False
            or value.get('full_verify_inputs_calls_retained') is not True
            or value.get('native_entry_claim')!='original_function_called_under_explicit_v901_bootstrap'):
        raise ValueError('actual901 outer/forwarded/control/source evidence mismatch')
    phases=value.get('phases')
    if (not isinstance(phases,list) or len(phases)!=2 or not all(isinstance(p,dict) for p in phases)
            or [p.get('phase') for p in phases]
            !=['original751_bindings','original742_prepare']):
        raise ValueError('exact751-bindings/742-prepare batch boundary receipts required')
    for phase in phases:
        elapsed=phase.get('wall_seconds')
        if (phase.get('boundary_recheck_complete') is not True or not isinstance(phase.get('metrics'),dict)
                or type(elapsed) not in (int,float) or not math.isfinite(elapsed) or elapsed<0):
            raise ValueError('completed finite batch phase and boundary recheck required')
        require_batch_metrics(phase['metrics'])
    if check:
        if 'case' in value or 'original_admission_driver_receipt' in value:
            raise ValueError('CPU check cannot claim native output or sidecar')
    elif (not valid_pointer(case) or not valid_pointer(original_receipt)
            or value.get('case')!=case or value.get('original_admission_driver_receipt')!=original_receipt):
        raise ValueError('native901 receipt must bind actual case and original751 sidecar')
    return value


def require_batch_metrics(value):
    counts=('maps','declared_files','watched_directories','pointer_checks','full_graph_boundary_checks',
            'used_path_end_checks','used_external','actual_SHA_checks','strong_stat_reuses','exact_JSON_payloads')
    if (not isinstance(value,dict) or value.get('schema')!='r10_scoped_frozen_lookup_metrics_v900' or value.get('closed') is not True
            or value.get('complete') is not True or any(type(value.get(k)) is not int or value[k]<0 for k in counts)
            or value['maps']<1 or value['declared_files']<value['maps'] or value['watched_directories']<1
            or value['pointer_checks']<1 or not 1<=value['used_path_end_checks']<=value['pointer_checks']
            or value['full_graph_boundary_checks']!=2*value['maps']
            or value['actual_SHA_checks']!=value['pointer_checks']+value['used_path_end_checks']
            or value['strong_stat_reuses']>value['actual_SHA_checks']
            or value['used_external']>value['pointer_checks']):
        raise ValueError('complete closed900 lookup metrics with every map boundary and used-input recheck required')


def require_resource_qualification(value,control):
    """Capacity evidence is separate from actual quota; nativeFalse never filters."""
    actual,capacity,kind=require_capacity(control)
    high=kind=='qualified1031';schema=QUALIFICATION_SCHEMA if high else 'r10_independent_concurrent_TRAIN_resource_qualification_v1005'
    qualifier='scripts/audit_concurrent_TRAIN_resource_canary_v'+('1031' if high else '1005')+'_r10.py'
    qualification='runtime/framework/concurrent_TRAIN_resource_qualification_v'+('1031' if high else '1005')+'.json'
    if (not isinstance(value,dict) or value.get('schema')!=schema
            or value.get('complete') is not True or value.get('split')!='train'
            or value.get('resource_profile')!=capacity or value.get('execution_entry')!=control['execution_implementation']
            or (high and (value.get('capacity_envelope')!=capacity or value.get('capacity_envelope_only') is not True))
            or value.get('policy_sha256')!=control['policy_sha256']
            or value.get('author_outcome_protocol')!=control['author_outcome_protocol']
            or type(value.get('case_count')) is not int or value['case_count']!=(18 if high else 12)
            or (high and (type(value.get('source_count')) is not int or value['source_count']!=6))
            or type(value.get('infrastructure_failed')) is not int or value['infrastructure_failed']!=0
            or type(value.get('actual_CPU_preflight_peak')) is not int or value['actual_CPU_preflight_peak']!=capacity['max_CPU_preflight_slots']
            or value.get('cleanup_completed') is not True or value.get('resource_bounds_verified') is not True
            or value.get('overlap_verified') is not True or value.get('independent_audit_complete') is not True
            or not isinstance(value.get('joint_supply_overlap'),dict) or value['joint_supply_overlap'].get('complete') is not True
            or value.get('formal_score_contribution') is not False or value.get('TEST_read') is not False
            or type(value.get('observed_native_peak')) is not int or value['observed_native_peak']!=capacity['max_native_slots']
            or type(value.get('observed_GPU_peak')) is not int or value['observed_GPU_peak']!=capacity['max_GPU_slots']
            or control['resource_qualifier'].get('path')!=qualifier
            or control['resource_qualification'].get('path')!=qualification
            or value.get('source_sha256')!=control['resource_qualifier']['sha256']):
        raise ValueError('real selected independent TRAIN capacity qualification required; no success-rate selection')
    return value


def execution_implementation(entry,source_bindings):
    """Common TRAIN/formal source identity without either split's control lock."""
    if not isinstance(entry,dict) or entry.get('kind') not in ('original751','batched901'):
        raise ValueError('explicit shared execution implementation required')
    required=set(COMMON_SOURCES)
    if entry['kind']=='batched901':required.update((BATCH_ENTRY,'our_mani/multitask/frozen_lookup_batch_v900.py'))
    if (not isinstance(source_bindings,dict) or set(source_bindings)!=required
            or any(not valid_pointer(p) or p['path']!=name for name,p in source_bindings.items())):
        raise ValueError('complete exact shared implementation source pointers required')
    target=BATCH_ENTRY if entry['kind']=='batched901' else ORIGINAL_ENTRY
    if entry.get('pointer')!=source_bindings[target]:raise ValueError('shared actual entry source pointer mismatch')
    return dict(schema='r10_resource_execution_implementation_v902',kind=entry['kind'],
        entrypoint=deepcopy(entry['pointer']),
        lookup_batch=deepcopy(source_bindings['our_mani/multitask/frozen_lookup_batch_v900.py']) if entry['kind']=='batched901' else None,
        original_formal_driver=deepcopy(source_bindings[ORIGINAL_ENTRY]),
        original_case_worker=deepcopy(source_bindings['scripts/run_task_executor_smoke_author_outcome_v743_r10.py']),
        original_native_runtime=deepcopy(source_bindings['scripts/run_task_executor_smoke_author_suite16_v744_r10.py']),
        sources=deepcopy(source_bindings))


def alive(record):
    pid,start=record.get('pid'),record.get('start_ticks')
    if type(pid) is not int or type(start) is not int:raise ValueError('explicit prior PID/start identity required')
    try:
        fields=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
        return fields[0] not in ('Z','X') and int(fields[19])==start
    except FileNotFoundError:return False


@contextmanager
def controller_locks():
    """The new owner holds old controller locks; its parent must not hold them."""
    with ExitStack() as stack:
        for name in ('runtime/framework/ordered_composite_TEST_v1013.lock','runtime/framework/ordered_composite_TEST_v958.lock','runtime/framework/ordered_concurrent_TEST_v907.lock',
                     'runtime/framework/ordered_resume_VAL_TEST_v775.lock',
                     'runtime/framework/ordered_env_resume_VAL_TEST_v827.lock'):
            handle=stack.enter_context((ROOT/name).open('a+'))
            fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        yield


def process_visibility():
    # Reuse823's complete host/PID check with only an explicit new script name.
    # The scope restores the frozen helper's globals before returning.
    from unittest.mock import patch
    prefixes=(*environment_owner.NATIVE_PREFIXES,'run_concurrent_resumed_formal_TEST_owner_',
              'run_batched_formal_driver_','run_concurrent_TRAIN_','continue_concurrent_TEST_',
              'run_composite_TEST_owner_','continue_composite_TEST_')
    with patch.object(environment_owner,'NATIVE_PREFIXES',prefixes):
        return environment_owner.process_visibility()


def read_pinned_resume_lock(path,expected):
    if not isinstance(expected,str) or len(expected)!=64 or set(expected)-set('0123456789abcdef'):
        raise ValueError('explicit external resume-lock SHA256 required')
    payload=Path(path).read_bytes()
    if hashlib.sha256(payload).hexdigest()!=expected:
        raise ValueError('resume-lock bytes differ from externally pinned SHA256')
    return json.loads(payload)


def read_bound_json(files,p):
    """Parse the very bytes matching the declared pointer, after original lookup."""
    locked_pointer(ROOT,files,p)
    path=ROOT/p['path'];payload=path.read_bytes()
    if hashlib.sha256(payload).hexdigest()!=p['sha256']:
        raise ValueError('bound JSON payload changed before parsing')
    return json.loads(payload)


def require_history(control,files):
    """Close all four historical owners using unchanged1010 and older validators."""
    from scripts import run_composite_TEST_owner_v1011_r10 as parent_owner
    from our_mani.multitask import composite_TEST_resume_v1010 as parent_core
    prior=read_bound_json(files,control['previous_resume_control'])
    prior_lock=read_bound_json(files,control['previous_resume_control_lock'])
    if (prior.get('schema')!=parent_core.CONTROL_SCHEMA or prior_lock.get('schema')!=parent_core.LOCK_SCHEMA
            or prior_lock.get('approved_for_execution') is not True):
        raise ValueError('actual immutable1010 control/lock required')
    verify_inputs(ROOT,prior_lock['files'])
    if any(files.get(k)!=v for k,v in prior_lock['files'].items()):
        raise ValueError('every original1010 frozen key/SHA must be preserved')
    for key in ('suite','suite_lock','source_inputs','validation_release','policy_sha256','author_outcome_protocol',
                'execution_entry','execution_implementation','environment_manifest','environment_wrapper',
                'environment_namespace_proof','standing_continuation_record'):
        if control.get(key)!=prior.get(key):raise ValueError('unchanged original science binding required: '+key)
    changed=control.get('resource_profile')!=prior.get('resource_profile')
    if (type(control.get('resource_profile_changed')) is not bool or control['resource_profile_changed'] is not changed
            or control.get('previous_resource_profile')!=prior.get('resource_profile')
            or control.get('previous_resource_qualification')!=prior.get('resource_qualification')):
        raise ValueError('actual profile change or truthful same-profile fallback and prior proof must be explicit')
    require_capacity(control)
    _,records,older=parent_owner.require_history(prior,prior_lock['files'])
    for key in ('launch','plan','status','terminal'):
        if control.get('original_'+key)!=prior.get('original_'+key):
            raise ValueError('original753 pointers must retain their exact nested lineage')
        if read_bound_json(files,control['original_'+key])!=records[key]:
            raise ValueError('original753 metadata changed')
    parent={key:read_bound_json(files,control['previous_'+key]) for key in ('launch','plan','status','terminal')}
    parent_core.require_resume_plan(prior,older['plan'],older['status'],older['terminal'],parent['plan'])
    composite.require_partition(control,parent['plan'],parent['status'],parent['terminal'])
    if any(parent['status']['completed'].get(k)!=v for k,v in older['status']['completed'].items()):
        raise ValueError('all historical753/902/956 outcomes must remain literal in1011')
    for name,value in (('753',records['launch']),('956',older['launch']),('1011',parent['launch'])):
        if alive(value):raise ValueError('historical'+name+' owner must remain exited')
    return prior,records,parent


def load_control(args):
    global ENVIRONMENT_RECEIPT
    event=threading.Event();dispatch_control(ROOT/RUNTIME,event,False)
    if event.is_set():raise InterruptedError('STOP before any composite continuation admission')
    control_path=(ROOT/args.resume_control).resolve();lock_path=(ROOT/args.resume_lock).resolve()
    if control_path!=ROOT/composite.CONTROL or lock_path!=ROOT/composite.LOCK:
        raise ValueError('exact fresh1040 control and externally pinned lock required')
    lock=read_pinned_resume_lock(lock_path,args.resume_lock_sha256)
    if lock.get('schema')!=LOCK_SCHEMA or lock.get('approved_for_execution') is not True:
        raise ValueError('1041 is inactive without an approved immutable1040 lock')
    files=lock.get('files');payload=control_path.read_bytes();control=json.loads(payload)
    expected=files.get(str(control_path),files.get(str(control_path.relative_to(ROOT)))) if isinstance(files,dict) else None
    if expected!=hashlib.sha256(payload).hexdigest():
        raise ValueError('same parsed control bytes must be pinned before any TEST input read')
    if lock.get('control')!=dict(path=composite.CONTROL,sha256=expected):
        raise ValueError('lock must explicitly bind these exact parsed control bytes')
    if (control.get('schema')!=CONTROL_SCHEMA or control.get('resume_owner')!=pointer(__file__)
            or control.get('resume_auditor')!=control.get('result_auditor')
            or control.get('result_auditor',{}).get('path')!=composite.AUDITOR
            or control.get('resume_controller',{}).get('path')!=composite.CONTROLLER
            or control.get('previous_runtime')!=PREVIOUS
            or control.get('previous_controller_runtime')!=PREVIOUS_CONTROLLER
            or control.get('previous_resume_control',{}).get('path')!=composite.PREVIOUS_CONTROL
            or control.get('previous_resume_control_lock',{}).get('path')!=composite.PREVIOUS_LOCK):
        raise ValueError('actual1041/1042/1043 and original1011/1013 lineage required')
    require_complete_VAL_release(control,args.validation_release)
    ENVIRONMENT_RECEIPT=environment_owner.require_frozen_environment(control)
    verify_inputs(ROOT,files);locked_pointer(ROOT,files,pointer(control_path))
    for key,arg in (('suite',args.suite),('suite_lock',args.suite_lock),('source_inputs',args.source_inputs),
                    ('validation_release',args.validation_release)):
        if control.get(key)!=pointer(ROOT/arg):raise ValueError('same frozen TEST source/release scope required: '+key)
        locked_pointer(ROOT,files,control[key])
    keys=('resume_owner','resume_auditor','result_auditor','resume_controller','resource_qualification','resource_qualifier',
          'standing_continuation_record','concurrency_authorization_record','previous_launch','previous_plan',
          'previous_status','previous_terminal','previous_controller_launch','previous_controller_status',
          'previous_controller_failure','previous_DRAIN','previous_resume_control','previous_resume_control_lock',
          'original_launch','original_plan','original_status','original_terminal','environment_manifest',
          'environment_wrapper','environment_namespace_proof','environment_recheck')
    for key in keys:locked_pointer(ROOT,files,control[key])
    _,_,capacity_kind=require_capacity(control)
    version='1031' if capacity_kind=='qualified1031' else '1005'
    if (control['resource_qualifier']!=pointer(ROOT/('scripts/audit_concurrent_TRAIN_resource_canary_v'+version+'_r10.py'))
            or control['resource_qualification'].get('path')!='runtime/framework/concurrent_TRAIN_resource_qualification_v'+version+'.json'):
        raise ValueError('only the actual selected independent capacity qualification is admitted')
    for key,name in (('previous_launch','launch.json'),('previous_plan','plan.json'),('previous_status','status.json'),('previous_terminal','terminal.json')):
        if control[key]!=pointer(ROOT/PREVIOUS/name):raise ValueError('real1011 immutable DRAIN snapshot required')
    for key,name in (('previous_controller_launch','launch.json'),('previous_controller_status','status.json'),('previous_controller_failure','failure.json')):
        if control[key]!=pointer(ROOT/PREVIOUS_CONTROLLER/name):raise ValueError('real1013 partial failure metadata required')
    if control['previous_DRAIN']!=pointer(ROOT/PREVIOUS/'DRAIN'):
        raise ValueError('actual1011 DRAIN file must remain unchanged')
    prior,original,parent=require_history(control,files)
    old_launch=parent['launch'];ctl_launch=read_bound_json(files,control['previous_controller_launch'])
    if (old_launch.get('source_sha256')!=pointer(ROOT/'scripts/run_composite_TEST_owner_v1011_r10.py')['sha256']
            or ctl_launch.get('source_sha256')!=pointer(ROOT/'scripts/continue_composite_TEST_v1013_r10.py')['sha256']
            or old_launch.get('resume_control')!=control['previous_resume_control']
            or old_launch.get('resume_control_lock')!=control['previous_resume_control_lock']
            or ctl_launch.get('resume_control')!=control['previous_resume_control']
            or ctl_launch.get('resume_control_lock')!=control['previous_resume_control_lock']
            or alive(old_launch) or alive(ctl_launch)):
        raise ValueError('actual exited1011/1013 source/PID/control evidence required')
    state=read_bound_json(files,control['previous_controller_status'])
    failure=read_bound_json(files,control['previous_controller_failure'])
    if (state.get('stage')!='requires_concrete_repair' or state.get('automatic_retry') is not False
            or failure.get('complete') is not False or failure.get('user_stop') is not False
            or failure.get('automatic_retry') is not False):
        raise ValueError('preserve true1013 partial failure; never invent a successful terminal')
    standing=read_bound_json(files,control['standing_continuation_record'])
    authorization=read_bound_json(files,control['concurrency_authorization_record'])
    if (standing.get('continue_until_user_stop') is not True or standing.get('user_stop_received') is not False
            or authorization.get('user_authorized_concurrency_change') is not True):
        raise ValueError('real standing/concurrency authorization must remain active')
    profile,capacity,capacity_kind=require_capacity(control)
    entry=validate_entry(control['execution_entry'])
    if entry['kind']!='batched901' or entry!=prior['execution_entry']:
        raise ValueError('retain original915 formal901 controls; no circular or changed bootstrap')
    from scripts import run_batched_formal_driver_v901_r10 as bootstrap
    for key in ('pointer','batch_control','batch_lock'):locked_pointer(ROOT,files,entry[key])
    batch_lock=read_bound_json(files,entry['batch_lock'])
    if batch_lock.get('schema')!=bootstrap.LOCK_SCHEMA or batch_lock.get('execution_enabled') is not True:
        raise ValueError('actual original enabled901 lock required')
    batch_sources={}
    for name in bootstrap.SOURCES:
        p=pointer(ROOT/name);locked_pointer(ROOT,files,p)
        if batch_lock['files'].get(str((ROOT/name).resolve()),batch_lock['files'].get(name))!=p['sha256']:
            raise ValueError('all original901/900 sources must retain both original locks')
        batch_sources[name]=p
    if control['execution_implementation']!=execution_implementation(entry,batch_sources):
        raise ValueError('actual original resource-qualified implementation changed')
    qualification=read_bound_json(files,control['resource_qualification']);require_resource_qualification(qualification,control)
    recheck=read_bound_json(files,control['environment_recheck'])
    composite.require_environment_recheck(recheck,control);locked_pointer(ROOT,files,recheck['source'])
    return control,files,batch_sources


def require_complete_VAL_release(control,path):
    """Small authentic aggregate gate before verify_inputs may touch TEST inputs."""
    p=ROOT/path;value=json.loads(p.read_text())
    if (control.get('validation_release')!=pointer(p)
            or value.get('schema')!='r10_full16_common_author_RML_validation_release_v531'
            or value.get('complete') is not True or value.get('split')!='validation'
            or value.get('source_cases')!=315 or value.get('method_cases')!=945
            or value.get('completed_method_cases')!=945 or value.get('formal_test_permitted') is not True
            or value.get('TEST_used') is not False or value.get('TEST_used_for_fitting') is not False
            or value.get('policy_sha256')!=control.get('policy_sha256')
            or value.get('author_outcome_protocol')!=control.get('author_outcome_protocol')
            or value.get('methods')!=list(METHODS) or set(value.get('tasks',[]))!=set(TASKS)
            or value.get('source_sha256')!=file_sha(ROOT/'scripts/audit_environment_resumed_formal_VAL_v824_r10.py')):
        raise ValueError('genuine unchanged-policy complete824 VAL release required before TEST inputs')
    return value


def child(row,leaf,cancelled,*,check):
    leaf=Path(leaf);process=None;started=time.monotonic();timed_out=False
    try:
        leaf.mkdir(parents=True,exist_ok=False)
        argv=row['argv']+(['--check'] if check else [])
        dispatch_control(ROOT/RUNTIME,cancelled,False)
        if cancelled.is_set():raise InterruptedError('STOP before a new CPU/native child')
        if not ENVIRONMENT_RECEIPT or os.readlink('/proc/self/ns/mnt')!=ENVIRONMENT_RECEIPT['current_mount_namespace']:
            raise ValueError('child must inherit the actual verified820 namespace')
        with (leaf/'stdout.log').open('xb') as stdout,(leaf/'stderr.log').open('xb') as stderr:
            dispatch_control(ROOT/RUNTIME,cancelled,False)
            if cancelled.is_set():raise InterruptedError('STOP immediately before Popen')
            process=subprocess.Popen(argv,cwd=ROOT,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),
                                     stdout=stdout,stderr=stderr,start_new_session=True)
            atomic_json(leaf/'launch.json',dict(created_utc=utc(),argv=argv,launcher_pid=process.pid,
                launcher_start_ticks=ticks(process.pid),owner_pid=os.getpid(),check_only=check,
                native_slots=0 if check else 1,GPU_slots=0 if check else int(row['method']=='physmani'),
                execution_entry=row['execution_entry'],physical_argv=row['physical_argv']+(['--check'] if check else []),
                inherited_frozen_environment=ENVIRONMENT_RECEIPT,current_mount_namespace=os.readlink('/proc/self/ns/mnt'),
                execution_protocol_changed=True,expected_identity_sha256=None if check else row['expected_identity']['sha256'],
                **{key:row[key] for key in BINDING_KEYS}))
            while process.poll() is None:
                timed_out=time.monotonic()-started>1800
                if cancelled.is_set() or timed_out:interrupt_owned_tree(process);break
                cancelled.wait(.5)
            code=process.wait()
        result=dict(complete=code==0 and not cancelled.is_set() and not timed_out,returncode=code,timed_out=timed_out,
                    interrupted=cancelled.is_set(),finished_utc=utc(),wall_seconds=time.monotonic()-started,automatic_retry=False)
        if result['complete'] and not check:
            out=ROOT/option(row['physical_argv'],'--out')
            result.update(case=pointer(out/'case.json'),admission_driver_receipt=pointer(out/'admission_driver_receipt.json'))
            if row['execution_entry']['kind']=='batched901':result['batch_driver_receipt']=pointer(out/'batch_driver_receipt.json')
    except BaseException as error:
        if process is not None and process.poll() is None:interrupt_owned_tree(process)
        result=dict(complete=False,error=repr(error),traceback=traceback.format_exc(),finished_utc=utc(),
                    interrupted=cancelled.is_set() or isinstance(error,InterruptedError),timed_out=timed_out,automatic_retry=False)
    atomic_json(leaf/'terminal.json',result);return result


def future_result(future):
    """A leaf filesystem failure is infrastructure, not an uncaught scheduler exit."""
    try:return future.result()
    except BaseException as error:
        return dict(complete=False,error=repr(error),traceback=traceback.format_exc(),finished_utc=utc(),
                    interrupted=False,timed_out=False,automatic_retry=False,leaf_future_exception=True)


def terminal_is_complete(completed_count,failed,owner_error,cancelled):
    return completed_count==1920 and not failed and owner_error is None and not cancelled.is_set()


def accept_CPU(receipt,row,source_sha,batch_sources):
    original_owner.require_CPU_driver(receipt,row)
    identity=receipt.get('identity',{})
    if (receipt.get('complete') is not True or receipt.get('native_started') is not False
            or receipt.get('GPU_started') is not False or receipt.get('actual_split')!='test'
            or receipt.get('component_domain',{}).get('purpose')!='evaluation'
            or identity.get('source_row_sha256')!=row['source_row_sha256'] or identity.get('method')!=row['method']
            or identity.get('config_sha256')!=row['configuration_sha256']
            or identity.get('frozen_inputs_sha256')!=row['input_lock_sha256']
            or identity.get('source_input_lock_sha256')!=source_sha):
        raise ValueError('actual source/config/domain CPU identity mismatch')
    require_batch_execution(receipt.get('batch_execution'),row,check=True,expected_sources=batch_sources)
    return identity


def accept_native(row,result,identity,protocol,policy,profile,batch_sources):
    out=ROOT/option(row['physical_argv'],'--out');path=out/'case.json';case=json.loads(path.read_text())
    case_pointer=pointer(path);driver_path=out/'admission_driver_receipt.json';driver_pointer=pointer(driver_path)
    original_owner.require_native_driver(json.loads(driver_path.read_text()),row,path,'test')
    original_owner.require_native_protocol(case,protocol);validate_completed_case(case,identity)
    require_attestation(case.get('common_author_RML_parameters',{}),profile)
    if (result.get('case')!=case_pointer or result.get('admission_driver_receipt')!=driver_pointer
            or case.get('cleanup_completed') is not True or case.get('component_domain',{}).get('purpose')!='evaluation'
            or case['component_domain'].get('split')!='test'
            or case.get('full16_frozen_execution',{}).get('policy_sha256')!=policy
            or type(case.get('official_task_success')) is not bool
            or (measured_motion(case) is not None and type(measured_motion(case)) is not bool)):
        raise ValueError('actual native TEST cleanup/source/policy receipt mismatch')
    record=dict(result,case=case_pointer,admission_driver_receipt=driver_pointer,
                native_success=case['official_task_success'],motion_pass=measured_motion(case))
    if row['execution_entry']['kind']=='batched901':
        batch_path=out/'batch_driver_receipt.json';batch_pointer=pointer(batch_path)
        require_batch_execution(json.loads(batch_path.read_text()),row,check=False,expected_sources=batch_sources,
                                case=case_pointer,original_receipt=driver_pointer)
        if result.get('batch_driver_receipt')!=batch_pointer:raise ValueError('terminal must bind actual901 native sidecar')
        record['batch_driver_receipt']=batch_pointer
    return record


def main():
    global ENVIRONMENT_RECEIPT
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('suite','suite-lock','source-inputs','validation-release','runtime','out','resume-control','resume-lock','resume-lock-sha256'):
        parser.add_argument('--'+name,required=True)
    args=parser.parse_args();runtime=(ROOT/args.runtime).resolve();out=(ROOT/args.out).resolve()
    if runtime!=(ROOT/RUNTIME).resolve() or out!=(ROOT/OUTPUT).resolve():raise ValueError('exact fresh1011 runtime/output required')
    if runtime.exists() or out.exists():raise FileExistsError('one-time continuation; preserve all prior attempts')
    control,resume_files,batch_sources=load_control(args);quotas=validate_resource_profile(control['resource_profile'])
    policy=control['policy_sha256'];protocol=control['author_outcome_protocol']
    suite_files=json.loads((ROOT/args.suite_lock).read_text())['files'];verify_inputs(ROOT,suite_files)
    suite=json.loads((ROOT/args.suite).read_text());driver,guard=original_owner.driver_bindings(suite,suite_files)
    if suite['author_outcome_protocol']!=protocol:raise ValueError('original741 outcome protocol must stay unchanged')
    release=require_complete_VAL_release(control,args.validation_release)
    manifest,sources,proof=read_formal_source(ROOT,suite,suite_files,'test',validation_release=release)
    if proof['policy_sha256']!=policy:raise ValueError('original physical policy must stay unchanged')
    source_lock=ROOT/args.source_inputs;source_meta=json.loads(source_lock.read_text());source_sha=file_sha(source_lock)
    if (source_meta.get('schema')!='r10_full16_post_admission_split_source_lock_v1' or source_meta.get('split')!='test'
            or source_meta.get('source_count')!=640 or source_meta.get('policy_sha256')!=policy
            or source_meta.get('TEST_used_for_fitting') is not False
            or source_meta.get('source_manifest_sha256')!=suite['source_manifests']['test']['sha256']):
        raise ValueError('same original640 TEST source lock required')
    verify_inputs(ROOT,source_meta['files'])
    old=ROOT/PREVIOUS;old_plan=json.loads((old/'plan.json').read_text());old_status=json.loads((old/'status.json').read_text())
    old_terminal=json.loads((old/'terminal.json').read_text());old_by_id=require_partition(control,old_plan,old_status,old_terminal)
    roster={(r['task'],r['raw_episode_id']):r for r in sources}
    for row in old_plan['jobs']:
        route=next(r for r in proof['routes']['routes'] if (r['task'],r['method'])==(row['task'],row['method']))
        source=roster[(row['task'],row['raw'])]
        if (row['admission_driver']!=driver or row['admission_guard']!=guard
                or row['case_worker']!=route['worker'] or row['native_runtime']!={k:route['consumer'][k] for k in ('path','sha256')}
                or row['author_outcome_protocol']!=protocol or row['worker_sha256']!=row['execution_entry']['pointer']['sha256']
                or row['source_row_sha256']!=digest(source)
                or row['configuration_sha256']!=route['configuration']['sha256']
                or row['input_lock_sha256']!=route['frozen_inputs']['sha256']):
            raise ValueError('every original row must match same source/config/physical route')
        for flag,value in (('--config',route['configuration']['path']),('--frozen-inputs',route['frozen_inputs']['path']),
                ('--manifest',suite['source_manifests']['test']['path']),('--suite',args.suite),('--suite-lock',args.suite_lock),
                ('--source-inputs',args.source_inputs),('--validation-release',args.validation_release),('--split','test'),
                ('--task',row['task']),('--raw-episode-id',row['raw']),('--method',row['method'])):
            if option(row['physical_argv'],flag)!=value:raise ValueError('original actual argv scope changed: '+flag)
    jobs=resumed_jobs(old_plan,control,control['execution_entry']);rml_profile=fingerprint(proof['routes']['common_author_RML_profile'])
    completed=deepcopy(old_status['completed'])
    for row in jobs:
        if not row['imported_completed']:continue
        name=row['job_id'];historical=composite.parent_from_imported(row)
        cpu=json.loads((ROOT/row['evidence_runtime']/'CPU'/name/'CPU_preflight.json').read_text())
        historical_sources=batch_sources if historical['execution_entry']['kind']=='batched901' else {}
        identity=accept_CPU(cpu,historical,source_sha,historical_sources)
        if accept_native(historical,completed[name],identity,protocol,policy,rml_profile,historical_sources)!=completed[name]:
            raise ValueError('imported native record cannot be rewritten')
    cancelled=threading.Event()
    for signum in (signal.SIGINT,signal.SIGTERM):signal.signal(signum,lambda *_:cancelled.set())
    with controller_locks(),exclusive_pair():
        if alive(json.loads((ROOT/control['previous_launch']['path']).read_text())) or alive(json.loads((ROOT/control['previous_controller_launch']['path']).read_text())):
            raise ValueError('old owners must remain exited after acquiring shared locks')
        inventory=process_visibility()
        if not inventory['host_visibility_available'] or inventory['live_project_native_processes'] or inventory['same_uid_inventory_incomplete']:
            raise RuntimeError('complete host-native inventory must be clean before scheduling')
        dispatch_control(runtime,cancelled,False)
        if cancelled.is_set():raise InterruptedError('STOP before creating new runtime')
        runtime.mkdir()
        common=dict(split='test',policy_sha256=policy,author_outcome_protocol=protocol,execution_protocol_changed=True,
            resume_control=pointer(ROOT/args.resume_control),resume_control_lock=pointer(ROOT/args.resume_lock),
            frozen_environment=ENVIRONMENT_RECEIPT,resource_profile=quotas,resource_qualification=control['resource_qualification'],
            resource_qualifier=control['resource_qualifier'],
            execution_implementation=control['execution_implementation'],resource_profile_changed=control['resource_profile_changed'],
            resource_capacity_envelope=control['resource_capacity_envelope'],resource_profile_within_capacity=True,
            resource_capacity_source=control['resource_capacity_source'],
            previous_resource_profile=control['previous_resource_profile'],previous_resource_qualification=control['previous_resource_qualification'],
            previous_resume_control=control['previous_resume_control'],
            previous_resume_control_lock=control['previous_resume_control_lock'],
            environment_recheck=control['environment_recheck'],
            admission_driver=driver,admission_guard=guard,native_runtime=protocol['runtime_consumer'],
            original_native_runtime=protocol['original_runtime'],automatic_retry=False)
        atomic_json(runtime/'launch.json',dict(common,created_utc=utc(),pid=os.getpid(),start_ticks=ticks(os.getpid()),
            argv=list(sys.argv),source_sha256=file_sha(__file__),suite_lock_sha256=file_sha(ROOT/args.suite_lock),source_manifest=suite['source_manifests']['test'],
            split_source_lock_sha256=source_sha,execution_entry=control['execution_entry'],batch_sources=batch_sources,
            base_stage_owner=pointer(ROOT/'scripts/run_formal_suite16_owner_v753_r10.py'),
            parent_stage_owner=pointer(ROOT/'scripts/run_composite_TEST_owner_v1011_r10.py'),
            imported_completed_cases=len(completed),inventory=inventory,**{k:v for k,v in quotas.items() if k.startswith('max_')}))
        atomic_json(runtime/'plan.json',dict(common,jobs=jobs,expected_cases=1920,imported_completed_cases=len(completed),
            source_outcome_filtering=False))
        pending=[r for r in jobs if not r['imported_completed']]
        checking={};ready=[];active={};failed={};index=0;draining=False;owner_error=None
        with ThreadPoolExecutor(max_workers=quotas['max_CPU_preflight_slots']) as cpu_pool,ThreadPoolExecutor(max_workers=quotas['max_native_slots']) as native_pool:
            while index<len(pending) or checking or ready or active:
                draining=dispatch_control(runtime,cancelled,draining) or bool(failed)
                for name,(future,row) in list(checking.items()):
                    if not future.done():continue
                    result=future_result(future);del checking[name]
                    if not result['complete']:failed[name]=dict(stage='CPU_preflight',**result);continue
                    try:
                        receipt=json.loads((runtime/'CPU'/name/'stdout.log').read_text())
                        row['expected_identity']=accept_CPU(receipt,row,source_sha,batch_sources)
                        atomic_json(runtime/'CPU'/name/'CPU_preflight.json',receipt);ready.append(row)
                    except BaseException as error:failed[name]=dict(stage='CPU_receipt',error=repr(error),automatic_retry=False)
                for name,(future,row) in list(active.items()):
                    if not future.done():continue
                    result=future_result(future);del active[name]
                    if not result['complete']:failed[name]=dict(stage='native',**result);continue
                    try:completed[name]=accept_native(row,result,row['expected_identity'],protocol,policy,rml_profile,batch_sources)
                    except BaseException as error:failed[name]=dict(stage='native_case_contract',error=repr(error),automatic_retry=False)
                draining=draining or bool(failed)
                if not cancelled.is_set() and not draining:
                    try:
                        while index<len(pending) and len(checking)<quotas['max_CPU_preflight_slots'] and len(ready)+len(checking)<quotas['ready_limit']:
                            draining=dispatch_control(runtime,cancelled,draining)
                            if cancelled.is_set() or draining:break
                            row=pending[index];name=row['job_id']
                            if file_sha(ROOT/row['argv'][2])!=row['worker_sha256']:raise ValueError('actual frozen execution entry changed')
                            index+=1;checking[name]=(cpu_pool.submit(child,row,runtime/'CPU'/name,cancelled,check=True),row)
                        selected=select_native_jobs(ready,[r for _,r in active.values()],quotas)
                        for row in selected:
                            draining=dispatch_control(runtime,cancelled,draining)
                            if cancelled.is_set() or draining:break
                            ready.remove(row);name=row['job_id']
                            active[name]=(native_pool.submit(child,row,runtime/'native'/name,cancelled,check=False),row)
                    except BaseException as error:owner_error=repr(error);draining=True
                atomic_json(runtime/'status.json',dict(common,updated_utc=utc(),expected=1920,CPU_launched=index,
                    imported_completed=len(control['completed_job_ids']),checking=list(checking),CPU_ready=len(ready),
                    active=list(active),completed=completed,failed=failed,ready_jobs=[r['job_id'] for r in ready],
                    drain_requested=draining,stop_requested=cancelled.is_set(),native_slots=len(active),
                    GPU_slots=sum(r['method']=='physmani' for _,r in active.values()),owner_error=owner_error),overwrite=True)
                if cancelled.is_set() and not checking and not active:break
                if drained(draining,checking,active):break
                cancelled.wait(2.)
        draining=dispatch_control(runtime,cancelled,draining)
        terminal=dict(common,complete=terminal_is_complete(len(completed),failed,owner_error,cancelled),finished_utc=utc(),
            expected=1920,completed=len(completed),failed=len(failed),owner_error=owner_error,
            drain_requested=draining,drained=drained(draining,checking,active),stop_requested=cancelled.is_set(),
            CPU_ready_unexecuted=[r['job_id'] for r in ready],CPU_not_launched=len(pending)-index,
            imported_completed_cases=len(control['completed_job_ids']),resumed_new_cases=len(completed)-len(control['completed_job_ids']),
            native_successes=sum(v['native_success'] for v in completed.values()),formal_release_requires_independent_audit=True)
        atomic_json(runtime/'terminal.json',terminal);print(json.dumps(terminal),flush=True)


if __name__=='__main__':main()
