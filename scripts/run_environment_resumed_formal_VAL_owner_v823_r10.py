"""Resume591 completed VAL cases after a frozen system-environment repair.

Two CPU preflight slots feed two native slots, with at most one baseline GPU
case. The TRAIN owner must already have drained and released its native lock.
No TEST stage can be constructed without the complete frozen-policy VAL proof.
"""
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from datetime import datetime,timezone
import argparse,json,os,signal,subprocess,sys,threading,time,traceback
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from our_mani.multitask.evaluation import atomic_json,file_sha,digest,validate_completed_case
from our_mani.multitask.formal_suite_admission_v728 import read_formal_source,locked_pointer
from our_mani.multitask.input_stat_cache_v79 import verify_inputs
from our_mani.multitask.common_rml_protocol_v530 import require_attestation,fingerprint
from our_mani.multitask.three_method_release16 import measured_motion
from our_mani.multitask.evaluation_roster16_v283 import METHODS
from scripts.run_native_resource_owner_v19_r10 import interrupt_owned_tree
from scripts.run_native_actuator_diagnostic_pair_v5_2_r10 import exclusive_pair,process_visibility as legacy_process_visibility
from scripts.run_formal_case_author_outcome_v751_r10 import BINDING_KEYS,DRIVER_SCHEMA,require_native_protocol
from our_mani.multitask.rml_case_preflight_v742 import outcome_protocol_binding

CONTROL_SCHEMA='r10_frozen_environment_formal_VAL_resume_control_v823'
LOCK_SCHEMA='r10_frozen_environment_formal_VAL_resume_inputs_v823'
PREVIOUS='runtime/framework/formal_VAL_resume_v772'
RUNTIME='runtime/framework/formal_VAL_env_resume_v823'
OUTPUT='artifacts/framework/multitask_r10/formal_VAL_env_resume_v823/validation'
ENVIRONMENT_RECEIPT=None
NATIVE_PREFIXES=('run_geometry_','run_native_','run_continuous_reference_pair_','collect_multitask_',
    'run_ready_candidate_','launch_multitask_','run_multitask_fresh_queue','run_task_formal_',
    'run_trained_task_','evaluate_multitask_case_','run_PhysMani_task_','run_task_executor_smoke_',
    'run_formal_case_','run_formal_suite16_owner_','run_resumed_formal_VAL_owner_',
    'run_environment_resumed_formal_VAL_owner_','continue_resumed_VAL_TEST_',
    'continue_environment_resumed_VAL_TEST_','continue_ordered_','continue_author_outcome_',
    'run_frozen_environment_TRAIN_canary_')


def full_process_visibility(proc=Path('/proc'),*,pid=None,uid=None):
    """Read host PID inventory; unrelated processes never require cwd/ns links."""
    proc=Path(proc);pid=os.getpid() if pid is None else pid;uid=os.getuid() if uid is None else uid
    def stat(entry):
        fields=(entry/'stat').read_text().rsplit(')',1)[1].split()
        return fields[0],int(fields[1]),int(fields[19])
    own=set();ancestor=pid;ancestor_complete=True
    while ancestor>0 and ancestor not in own:
        own.add(ancestor)
        try:ancestor=stat(proc/str(ancestor))[1]
        except (OSError,ValueError,IndexError):ancestor_complete=False;break
    try:comm=(proc/'1/comm').read_text().strip()
    except OSError:comm=None
    active=[];unresolved=[];incomplete=not ancestor_complete
    for entry in proc.iterdir():
        if not entry.name.isdigit() or int(entry.name) in own:continue
        try:
            before=stat(entry)
            if before[0] in ('Z','X'):continue
            process_uid=entry.stat().st_uid
            try:tokens=(entry/'cmdline').read_bytes().decode(errors='replace').split('\0')
            except PermissionError:
                if process_uid==uid:incomplete=True;unresolved.append(dict(pid=int(entry.name),reason='same-UID cmdline unreadable'))
                continue
            scripts=[Path(x).name for x in tokens if '\n' not in x and '\r' not in x
                     and Path(x).name.endswith('.py') and Path(x).name.startswith(NATIVE_PREFIXES)]
            if not scripts:continue
            if process_uid!=uid:continue
            try:cwd=Path(os.readlink(entry/'cwd')).resolve()
            except PermissionError:
                incomplete=True;unresolved.append(dict(pid=int(entry.name),scripts=scripts,reason='matching same-UID cwd unreadable'));continue
            after=stat(entry)
            if before[2]==after[2] and cwd==ROOT.resolve():
                active.append(dict(pid=int(entry.name),start_ticks=before[2],scripts=scripts,UID=process_uid))
        except (FileNotFoundError,ProcessLookupError):continue
        except PermissionError:
            try:same_uid=entry.stat().st_uid==uid
            except OSError:same_uid=True
            if same_uid:incomplete=True;unresolved.append(dict(pid=int(entry.name),reason='same-UID process metadata unreadable'))
        except (OSError,ValueError,IndexError) as error:
            incomplete=True;unresolved.append(dict(pid=int(entry.name),reason=repr(error)))
    return dict(pid1_comm=comm,host_visibility_available=comm in ('systemd','init'),
        live_project_native_processes=active,same_uid_inventory_incomplete=incomplete,
        excluded_self_and_ancestor_pids=sorted(own),ancestor_chain_complete=ancestor_complete,
        unresolved_matching_or_same_uid_processes=unresolved,
        host_namespace_links_claimed_readable=False,
        note='stat/PPID/cmdline inventory; cwd only for actual same-UID native/controller script matches; denied matching cwd fails closed')


def process_visibility():
    try:legacy=legacy_process_visibility()
    except (OSError,ValueError,IndexError) as error:legacy=dict(error=repr(error),complete=False)
    return dict(full_process_visibility(),legacy_v5_2_actual_result=legacy)


def require_frozen_environment(control):
    """Require820's actual private namespace; never repair or launch it here."""
    from scripts.run_frozen_system_environment_v820_r10 import capture_namespace_evidence
    for key in ('environment_manifest','environment_wrapper','environment_namespace_proof'):
        if control.get(key)!=pointer(ROOT/control[key]['path']):raise ValueError('frozen environment evidence changed: '+key)
    if control['environment_wrapper']['path']!='scripts/run_frozen_system_environment_v820_r10.py':
        raise ValueError('exact820 environment wrapper required')
    manifest=control['environment_manifest'];proof=control['environment_namespace_proof']
    namespace=capture_namespace_evidence(ROOT/manifest['path'],manifest['sha256'],
        qualification_path=ROOT/proof['path'],qualification_sha=proof['sha256'])
    from scripts.audit_environment_resumed_formal_VAL_v824_r10 import require_environment_receipt
    require_environment_receipt(namespace,control,owner_pid=os.getpid())
    if (namespace.get('manifest')!=manifest or namespace.get('wrapper')!=control['environment_wrapper']
        or namespace.get('namespace_qualification')!=proof or namespace.get('observer_pid')!=os.getpid()
        or namespace.get('current_mount_namespace')!=os.readlink('/proc/self/ns/mnt')):
        raise ValueError('actual820 namespace capture differs from frozen control/current process')
    return namespace


def utc():return datetime.now(timezone.utc).isoformat()
def ticks(pid):return int(Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[19])


def pointer(path):
    path=Path(path).resolve()
    if not path.is_relative_to(ROOT):raise ValueError('project evidence pointer required')
    return dict(path=str(path.relative_to(ROOT)),sha256=file_sha(path))


def driver_bindings(suite,files):
    expected={'formal_case_driver':'scripts/run_formal_case_author_outcome_v751_r10.py',
              'formal_admission_guard':'our_mani/multitask/formal_suite_admission_v728.py',
              'formal_result_auditor':'scripts/audit_formal_suite16_results_v754_r10.py'}
    for key,name in expected.items():
        if locked_pointer(ROOT,files,suite[key])!=(ROOT/name).resolve():
            raise ValueError('exact separately frozen formal admission dependency required: '+key)
    protocol=outcome_protocol_binding(ROOT)
    if suite.get('author_outcome_protocol')!=protocol:raise ValueError('frozen author-outcome suite protocol required')
    for key in ('implementation','runtime_consumer','original_runtime'):locked_pointer(ROOT,files,protocol[key])
    locked_pointer(ROOT,files,pointer(ROOT/'our_mani/multitask/rml_case_preflight_v742.py'))
    locked_pointer(ROOT,files,pointer(ROOT/'scripts/run_formal_suite16_owner_v753_r10.py'))
    return suite['formal_case_driver'],suite['formal_admission_guard']


def require_CPU_driver(receipt,row):
    if receipt.get('execution_protocol_changed') is not True:raise ValueError('explicit new frozen execution policy required')
    for key in BINDING_KEYS:
        if receipt.get(key)!=row[key]:raise ValueError('actual CPU admission driver binding changed: '+key)


def require_native_driver(receipt,row,case_path,split):
    if (receipt.get('schema')!=DRIVER_SCHEMA or receipt.get('complete') is not True
        or receipt.get('execution_protocol_changed') is not True or receipt.get('actual_split')!=split
        or receipt.get('case')!=pointer(case_path)):
        raise ValueError('complete external admission receipt must bind the exact author-outcome native case')
    for key in BINDING_KEYS:
        if receipt.get(key)!=row[key]:raise ValueError('actual native admission binding changed: '+key)


def dispatch_control(runtime,cancelled,draining):
    """STOP cancels; DRAIN only prevents new CPU/native submissions."""
    runtime=Path(runtime)
    if (runtime/'STOP').exists() or (ROOT/'runtime/framework/multitask_r10_STOP').exists():cancelled.set()
    return draining or (runtime/'DRAIN').exists()


def drained(draining,checking,active):
    return bool(draining and not checking and not active)


def child(row,leaf,cancelled,*,check):
    leaf.mkdir(parents=True,exist_ok=False);process=None;started=time.monotonic()
    argv=row['argv']+(['--check'] if check else []);timed_out=False
    try:
        if cancelled.is_set() or (ROOT/'runtime/framework/multitask_r10_STOP').exists():
            raise InterruptedError('STOP before a new native/CPU child')
        if not ENVIRONMENT_RECEIPT or os.readlink('/proc/self/ns/mnt')!=ENVIRONMENT_RECEIPT['current_mount_namespace']:
            raise ValueError('new children must inherit the actual verified820 namespace')
        with (leaf/'stdout.log').open('xb') as log,(leaf/'stderr.log').open('xb') as err:
            process=subprocess.Popen(argv,cwd=ROOT,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),
                stdout=log,stderr=err,start_new_session=True)
            atomic_json(leaf/'launch.json',dict(created_utc=utc(),argv=argv,launcher_pid=process.pid,
                launcher_start_ticks=ticks(process.pid),owner_pid=os.getpid(),check_only=check,
                native_slots=0 if check else 1,GPU_slots=0 if check else int(row['method']=='physmani'),
                **{key:row[key] for key in BINDING_KEYS},
                inherited_frozen_environment=ENVIRONMENT_RECEIPT,
                current_mount_namespace=os.readlink('/proc/self/ns/mnt'),
                execution_protocol_changed=True,expected_identity_sha256=None if check else row['expected_identity']['sha256']))
            while process.poll() is None:
                timed_out=time.monotonic()-started>1800
                if cancelled.is_set() or timed_out:interrupt_owned_tree(process);break
                cancelled.wait(.5)
            code=process.wait()
        result=dict(complete=code==0 and not cancelled.is_set() and not timed_out,returncode=code,
            timed_out=timed_out,interrupted=cancelled.is_set(),finished_utc=utc(),wall_seconds=time.monotonic()-started,
            automatic_retry=False)
        if result['complete'] and not check:
            case_dir=ROOT/row['argv'][row['argv'].index('--out')+1]
            result.update(case=pointer(case_dir/'case.json'),
                admission_driver_receipt=pointer(case_dir/'admission_driver_receipt.json'))
    except BaseException as error:
        if process is not None and process.poll() is None:interrupt_owned_tree(process)
        result=dict(complete=False,error=repr(error),traceback=traceback.format_exc(),finished_utc=utc(),
                    interrupted=cancelled.is_set() or isinstance(error,InterruptedError),timed_out=timed_out,automatic_retry=False)
    atomic_json(leaf/'terminal.json',result);return result



def load_resume_control(control_path, lock_path, suite_path, suite_lock, source_inputs):
    global ENVIRONMENT_RECEIPT
    if (ROOT/'runtime/framework/multitask_r10_STOP').exists():raise InterruptedError('user STOP has priority over environment continuation')
    control_path=(ROOT/control_path).resolve();lock_path=(ROOT/lock_path).resolve()
    lock=json.loads(lock_path.read_text());files=lock['files']
    if (lock.get('schema')!=LOCK_SCHEMA
        or lock.get('approved_for_execution') is not True):raise ValueError('explicit frozen user resume inputs required')
    c=json.loads(control_path.read_text())
    ENVIRONMENT_RECEIPT=require_frozen_environment(c)
    verify_inputs(ROOT,files);locked_pointer(ROOT,files,pointer(control_path))
    if (c.get('schema')!=CONTROL_SCHEMA
        or c.get('user_authorized_resume') is not True or c.get('split')!='validation'
        or c.get('expected_cases')!=945 or c.get('resume_owner')!=pointer(Path(__file__))
        or c.get('standing_authorization_reused') is not True or c.get('new_user_event_invented') is not False
        or c.get('physical_policy_changed') is not False):
        raise ValueError('actual user-authorized same-policy VAL continuation required')
    for key,path in [('suite',suite_path),('suite_lock',suite_lock),('source_inputs',source_inputs)]:
        if c.get(key)!=pointer(ROOT/path):raise ValueError('original formal source scope changed: '+key)
        locked_pointer(ROOT,files,c[key])
    for key in ('resume_owner','resume_auditor','previous_launch','previous_plan','previous_status','previous_terminal',
                'user_resume_record','standing_continuation_record','previous_resume_control','previous_resume_control_lock',
                'environment_manifest','environment_wrapper','environment_namespace_proof'):
        locked_pointer(ROOT,files,c[key])
    permission=json.loads((ROOT/c['user_resume_record']['path']).read_text())
    if permission.get('user_authorized_resume') is not True or permission.get('revokes_previous_stop') is not True:
        raise ValueError('the real772 user continuation must remain authorized')
    standing=json.loads((ROOT/c['standing_continuation_record']['path']).read_text())
    if standing.get('continue_until_user_stop') is not True or standing.get('user_stop_received') is not False:
        raise ValueError('standing user continuation has not been preserved')
    if c.get('previous_runtime')!=PREVIOUS:raise ValueError('exact previous772 runtime required')
    for key,path in (('previous_resume_control','artifacts/framework/multitask_r10/formal_VAL_resume_v772_setup/resume_control.json'),
                     ('previous_resume_control_lock','artifacts/framework/multitask_r10/formal_VAL_resume_v772_setup/resume_inputs.json'),
                     ('user_resume_record','runtime/framework/user_resume_v772.json'),
                     ('standing_continuation_record','runtime/framework/latest_executor_continuation_v817.json'),
                     ('resume_auditor','scripts/audit_environment_resumed_formal_VAL_v824_r10.py')):
        if c[key]!=pointer(ROOT/path):raise ValueError('exact standing/nested continuation evidence required: '+key)
    old=ROOT/c['previous_runtime']
    for key,name in [('previous_launch','launch.json'),('previous_plan','plan.json'),('previous_status','status.json'),('previous_terminal','terminal.json')]:
        if c[key]!=pointer(old/name):raise ValueError('resume provenance changed: '+key)
    return c,files


def resumed_jobs(jobs,old_plan,control,new_runtime):
    old={r['job_id']:r for r in old_plan['jobs']}
    imported=set(control['completed_job_ids']);result=[]
    for row in jobs:
        name=row['job_id'];copy=deepcopy(old[name] if name in imported else row)
        previous=old[name]
        if previous.get('evidence_runtime') not in ('runtime/framework/formal_VAL_v753',PREVIOUS):
            raise ValueError('previous row lost its real753/772 evidence tier')
        copy.update(evidence_runtime=previous['evidence_runtime'] if name in imported else new_runtime,
                    imported_completed=name in imported,previous_row_sha256=digest(previous),
                    previous_imported_completed=previous['imported_completed'])
        result.append(copy)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('suite','suite-lock','source-inputs','split','runtime','out'):p.add_argument('--'+k,required=True)
    p.add_argument('--validation-release');p.add_argument('--resume-control',required=True);p.add_argument('--resume-lock',required=True);args=p.parse_args()
    if args.split!='validation':raise ValueError('resumed owner is VAL only; fresh TEST retains original753')
    runtime=(ROOT/args.runtime).resolve();out=(ROOT/args.out).resolve()
    if runtime!=(ROOT/RUNTIME).resolve() or out!=(ROOT/OUTPUT).resolve():raise ValueError('exact fresh823 namespace required')
    if not runtime.is_relative_to(ROOT/'runtime/framework') or not out.is_relative_to(ROOT/'artifacts/framework/multitask_r10'):
        raise ValueError('isolated project runtime and formal outputs required')
    if runtime.exists() or out.exists():raise FileExistsError('preserve existing stage; no implicit restart/retry')
    control,resume_files=load_resume_control(args.resume_control,args.resume_lock,args.suite,args.suite_lock,args.source_inputs)
    files=json.loads((ROOT/args.suite_lock).read_text())['files'];verify_inputs(ROOT,files)
    suite_path=(ROOT/args.suite).resolve();locked_pointer(ROOT,files,dict(path=str(suite_path.relative_to(ROOT)),sha256=file_sha(suite_path)))
    suite=json.loads(suite_path.read_text());release=None
    driver,guard=driver_bindings(suite,files)
    protocol=suite['author_outcome_protocol']
    if args.split=='test':
        if not args.validation_release or (ROOT/args.validation_release).resolve()!=(ROOT/suite['validation_release_path']).resolve():
            raise ValueError('exact completed VAL release required before TEST source selection')
        release=json.loads((ROOT/args.validation_release).read_text())
    manifest,rows,proof=read_formal_source(ROOT,suite,files,args.split,validation_release=release)
    profile=fingerprint(proof['routes']['common_author_RML_profile']);policy=proof['policy_sha256']
    if policy!=control['policy_sha256']:raise ValueError('physical policy changed while resuming')
    source_lock=(ROOT/args.source_inputs).resolve();source_inputs=json.loads(source_lock.read_text())
    if (source_inputs.get('schema')!='r10_full16_post_admission_split_source_lock_v1'
        or source_inputs.get('split')!=args.split or source_inputs.get('policy_sha256')!=policy
        or source_inputs.get('source_count')!=len(rows) or source_inputs.get('TEST_used_for_fitting') is not False
        or source_inputs.get('source_manifest_sha256')!=suite['source_manifests'][args.split]['sha256']):
        raise ValueError('exact admitted split source lock required')
    verify_inputs(ROOT,source_inputs['files'])
    inventory=process_visibility()
    if not inventory['host_visibility_available'] or inventory['live_project_native_processes'] or inventory['same_uid_inventory_incomplete']:
        raise RuntimeError('drain existing project native workers before formal ownership')
    cancelled=threading.Event()
    signal.signal(signal.SIGINT,lambda *_:cancelled.set());signal.signal(signal.SIGTERM,lambda *_:cancelled.set())
    jobs=[]
    for row in rows:
        for method in METHODS:
            route=next(r for r in proof['routes']['routes'] if (r['task'],r['method'])==(row['task'],method))
            if route['worker']['path']!='scripts/run_task_executor_smoke_author_outcome_v743_r10.py':
                raise ValueError('formal driver requires the exact qualified author-outcome worker743')
            if route['consumer']['path']!='scripts/run_task_executor_smoke_author_suite16_v744_r10.py':
                raise ValueError('formal driver requires the exact qualified author-outcome runtime744')
            if {k:route['consumer'][k] for k in ('path','sha256')}!=protocol['runtime_consumer']:
                raise ValueError('route runtime differs from frozen author-outcome protocol')
            argv=['bash','scripts/run_aligned_gpu.sh' if method=='physmani' else 'scripts/run_local_sim.sh',driver['path'],
                '--config',route['configuration']['path'],'--manifest',suite['source_manifests'][args.split]['path'],
                '--frozen-inputs',route['frozen_inputs']['path'],'--suite',args.suite,'--suite-lock',args.suite_lock,
                '--source-inputs',args.source_inputs,'--task',row['task'],'--raw-episode-id',row['raw_episode_id'],
                '--method',method,'--split',args.split,'--out',str((out/row['task']/row['raw_episode_id']/method).relative_to(ROOT))]
            if args.validation_release:argv+=['--validation-release',args.validation_release]
            jobs.append(dict(task=row['task'],raw=row['raw_episode_id'],method=method,argv=argv,
                source_row_sha256=digest(row),job_id=row['task']+'_'+row['raw_episode_id']+'_'+method,
                worker_sha256=driver['sha256'],admission_driver=driver,admission_guard=guard,
                case_worker=route['worker'],native_runtime={k:route['consumer'][k] for k in ('path','sha256')},
                original_native_runtime=protocol['original_runtime'],author_outcome_protocol=protocol,
                configuration_sha256=route['configuration']['sha256'],
                input_lock_sha256=route['frozen_inputs']['sha256']))
    expected=945 if args.split=='validation' else 1920
    if len(jobs)!=expected:raise ValueError('exact full16 three-method source grid required')
    old=ROOT/control['previous_runtime']
    old_plan=json.loads((old/'plan.json').read_text());old_status=json.loads((old/'status.json').read_text())
    old_terminal=json.loads((old/'terminal.json').read_text())
    jobs=resumed_jobs(jobs,old_plan,control,str(runtime.relative_to(ROOT)))
    from scripts.audit_environment_resumed_formal_VAL_v824_r10 import require_resume_plan
    require_resume_plan(control,old_plan,old_status,old_terminal,dict(jobs=jobs,split=args.split,expected_cases=945,policy_sha256=policy),runtime=runtime)
    # Validate reused native outcomes; no model/source/trajectory is changed.
    for row in jobs:
        if not row['imported_completed']:continue
        name=row['job_id'];record=old_status['completed'][name]
        evidence=ROOT/row['evidence_runtime']
        receipt=json.loads((evidence/'CPU'/name/'CPU_preflight.json').read_text())
        require_CPU_driver(receipt,row)
        path=ROOT/record['case']['path'];case=json.loads(path.read_text())
        if record['case']!=pointer(path):raise ValueError('previous completed native case changed')
        require_native_driver(json.loads((path.parent/'admission_driver_receipt.json').read_text()),row,path,args.split)
        require_native_protocol(case,protocol);validate_completed_case(case,receipt['identity'])
        require_attestation(case.get('common_author_RML_parameters',{}),profile)
        if case.get('cleanup_completed') is not True or case['full16_frozen_execution']['policy_sha256']!=policy:
            raise ValueError('previous native cleanup/policy changed')
    with exclusive_pair():
        inventory=process_visibility()
        if not inventory['host_visibility_available'] or inventory['live_project_native_processes'] or inventory['same_uid_inventory_incomplete']:
            raise RuntimeError('full host native inventory changed while acquiring its shared lock')
        runtime.mkdir();atomic_json(runtime/'launch.json',dict(created_utc=utc(),pid=os.getpid(),start_ticks=ticks(os.getpid()),
            source_sha256=file_sha(__file__),split=args.split,policy_sha256=policy,source_manifest=suite['source_manifests'][args.split],
            suite_lock_sha256=file_sha(ROOT/args.suite_lock),split_source_lock_sha256=file_sha(source_lock),
            admission_driver=driver,admission_guard=guard,execution_protocol_changed=True,
            native_runtime=protocol['runtime_consumer'],original_native_runtime=protocol['original_runtime'],author_outcome_protocol=protocol,
            max_native_slots=2,max_GPU_slots=1,max_CPU_preflight_slots=2,automatic_retry=False,inventory=inventory,
            resumed_after_environment_failure=True,imported_completed_cases=len(control['completed_job_ids']),
            frozen_environment=ENVIRONMENT_RECEIPT,
            resume_control=pointer(ROOT/args.resume_control),resume_control_lock=pointer(ROOT/args.resume_lock),
            base_stage_owner=pointer(ROOT/'scripts/run_formal_suite16_owner_v753_r10.py')))
        atomic_json(runtime/'plan.json',dict(split=args.split,jobs=jobs,expected_cases=expected,policy_sha256=policy,
            source_outcome_filtering=False,automatic_retry=False,resume_control=pointer(ROOT/args.resume_control),
            resume_control_lock=pointer(ROOT/args.resume_lock),imported_completed_cases=len(control['completed_job_ids'])))
        jobs=[r for r in jobs if not r['imported_completed']]
        checking={};ready=[];active={};completed=deepcopy(old_status['completed']);failed={};next_index=0;draining=False
        with ThreadPoolExecutor(max_workers=2) as cpu,ThreadPoolExecutor(max_workers=2) as native:
            while next_index<len(jobs) or checking or ready or active:
                draining=dispatch_control(runtime,cancelled,draining)
                for name,(future,row) in list(checking.items()):
                    if not future.done():continue
                    result=future.result();del checking[name]
                    if not result['complete']:failed[name]=dict(stage='CPU_preflight',**result);continue
                    try:
                        receipt=json.loads((runtime/'CPU'/name/'stdout.log').read_text())
                        require_CPU_driver(receipt,row)
                        identity=receipt['identity']
                        if (receipt.get('complete') is not True or receipt.get('native_started') is not False or receipt.get('GPU_started') is not False
                            or receipt.get('actual_split')!=args.split or receipt['component_domain'].get('purpose')!='evaluation'
                            or identity.get('source_row_sha256')!=row['source_row_sha256'] or identity.get('method')!=row['method']
                            or identity.get('config_sha256')!=row['configuration_sha256']
                            or identity.get('frozen_inputs_sha256')!=row['input_lock_sha256']
                            or identity.get('source_input_lock_sha256')!=file_sha(source_lock)):
                            raise ValueError('actual-argv CPU source/domain receipt mismatch')
                        atomic_json(runtime/'CPU'/name/'CPU_preflight.json',receipt);row['expected_identity']=identity;ready.append(row)
                    except BaseException as error:failed[name]=dict(stage='CPU_receipt',error=repr(error),automatic_retry=False)
                for name,(future,row) in list(active.items()):
                    if not future.done():continue
                    result=future.result();del active[name]
                    if result['complete']:
                        try:
                            path=ROOT/row['argv'][row['argv'].index('--out')+1]/'case.json';case=json.loads(path.read_text())
                            driver_path=path.parent/'admission_driver_receipt.json'
                            driver_receipt=json.loads(driver_path.read_text())
                            require_native_driver(driver_receipt,row,path,args.split)
                            require_native_protocol(case,protocol)
                            if result.get('case')!=pointer(path) or result.get('admission_driver_receipt')!=pointer(driver_path):
                                raise ValueError('native terminal case/driver receipt SHA changed')
                            validate_completed_case(case,row['expected_identity'])
                            require_attestation(case.get('common_author_RML_parameters',{}),profile)
                            if (case.get('cleanup_completed') is not True or case.get('component_domain',{}).get('purpose')!='evaluation'
                                or case.get('full16_frozen_execution',{}).get('policy_sha256')!=policy
                                or type(measured_motion(case)) is not bool):raise ValueError('actual native formal contract incomplete')
                            completed[name]=dict(result,case=dict(path=str(path.relative_to(ROOT)),sha256=file_sha(path)),
                                admission_driver_receipt=pointer(driver_path),
                                native_success=case['official_task_success'],motion_pass=measured_motion(case))
                        except BaseException as error:failed[name]=dict(stage='native_case_contract',error=repr(error),automatic_retry=False)
                    else:failed[name]=dict(stage='native',**result)
                if not cancelled.is_set() and not draining:
                    while next_index<len(jobs) and len(checking)<2 and len(ready)<8:
                        row=jobs[next_index];next_index+=1;name=row['job_id']
                        actual=file_sha(ROOT/row['argv'][2])
                        if actual!=row['worker_sha256']:raise ValueError('frozen formal worker changed')
                        checking[name]=(cpu.submit(child,row,runtime/'CPU'/name,cancelled,check=True),row)
                    for row in list(ready):
                        if len(active)>=2:break
                        if row['method']=='physmani' and any(r['method']=='physmani' for _,r in active.values()):continue
                        name=row['job_id'];ready.remove(row)
                        active[name]=(native.submit(child,row,runtime/'native'/name,cancelled,check=False),row)
                atomic_json(runtime/'status.json',dict(updated_utc=utc(),split=args.split,expected=expected,CPU_launched=next_index,imported_completed=len(control['completed_job_ids']),
                    previous_environment_failed_cases=control['failed_job_ids'],
                    frozen_environment=ENVIRONMENT_RECEIPT,
                    checking=list(checking),CPU_ready=len(ready),active=list(active),completed=completed,failed=failed,
                    ready_jobs=[row['job_id'] for row in ready],drain_requested=draining,
                    native_slots=len(active),GPU_slots=sum(r['method']=='physmani' for _,r in active.values()),
                    stop_requested=cancelled.is_set(),automatic_retry=False),overwrite=True)
                if cancelled.is_set() and not checking and not active:break
                if drained(draining,checking,active):break
                cancelled.wait(2.)
        terminal=dict(complete=len(completed)==expected and not failed,finished_utc=utc(),split=args.split,
            expected=expected,completed=len(completed),failed=len(failed),policy_sha256=policy,automatic_retry=False,
            drain_requested=draining,drained=drained(draining,checking,active),stop_requested=cancelled.is_set(),
            CPU_ready_unexecuted=[row['job_id'] for row in ready],CPU_not_launched=len(jobs)-next_index,
            execution_protocol_changed=True,author_outcome_protocol=protocol,
            native_successes=sum(r['native_success'] for r in completed.values()),formal_release_requires_independent_audit=True,
            imported_completed_cases=len(control['completed_job_ids']),resumed_new_cases=len(completed)-len(control['completed_job_ids']),
            previous_environment_failed_cases=control['failed_job_ids'],frozen_environment=ENVIRONMENT_RECEIPT,
            resume_control=pointer(ROOT/args.resume_control),resume_control_lock=pointer(ROOT/args.resume_lock))
        atomic_json(runtime/'terminal.json',terminal);print(json.dumps(terminal),flush=True)


if __name__=='__main__':main()
