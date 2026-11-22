"""Own the frozen explicit author-outcome formal policy and its actual cases.

Two CPU preflight slots feed two native slots, with at most one baseline GPU
case. The TRAIN owner must already have drained and released its native lock.
No TEST stage can be constructed without the complete frozen-policy VAL proof.
"""
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
from scripts.run_native_actuator_diagnostic_pair_v5_2_r10 import exclusive_pair,process_visibility
from scripts.run_formal_case_author_outcome_v751_r10 import BINDING_KEYS,DRIVER_SCHEMA,require_native_protocol
from our_mani.multitask.rml_case_preflight_v742 import outcome_protocol_binding


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
    locked_pointer(ROOT,files,pointer(Path(__file__)))
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
        with (leaf/'stdout.log').open('xb') as log,(leaf/'stderr.log').open('xb') as err:
            process=subprocess.Popen(argv,cwd=ROOT,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),
                stdout=log,stderr=err,start_new_session=True)
            atomic_json(leaf/'launch.json',dict(created_utc=utc(),argv=argv,launcher_pid=process.pid,
                launcher_start_ticks=ticks(process.pid),owner_pid=os.getpid(),check_only=check,
                native_slots=0 if check else 1,GPU_slots=0 if check else int(row['method']=='physmani'),
                **{key:row[key] for key in BINDING_KEYS},
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
        result=dict(complete=False,error=repr(error),traceback=traceback.format_exc(),finished_utc=utc(),automatic_retry=False)
    atomic_json(leaf/'terminal.json',result);return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('suite','suite-lock','source-inputs','split','runtime','out'):p.add_argument('--'+k,required=True)
    p.add_argument('--validation-release');args=p.parse_args()
    if args.split not in ('validation','test'):raise ValueError('formal owner only')
    runtime=(ROOT/args.runtime).resolve();out=(ROOT/args.out).resolve()
    if not runtime.is_relative_to(ROOT/'runtime/framework') or not out.is_relative_to(ROOT/'artifacts/framework/multitask_r10'):
        raise ValueError('isolated project runtime and formal outputs required')
    if runtime.exists() or out.exists():raise FileExistsError('preserve existing stage; no implicit restart/retry')
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
    with exclusive_pair():
        runtime.mkdir();atomic_json(runtime/'launch.json',dict(created_utc=utc(),pid=os.getpid(),start_ticks=ticks(os.getpid()),
            source_sha256=file_sha(__file__),split=args.split,policy_sha256=policy,source_manifest=suite['source_manifests'][args.split],
            suite_lock_sha256=file_sha(ROOT/args.suite_lock),split_source_lock_sha256=file_sha(source_lock),
            admission_driver=driver,admission_guard=guard,execution_protocol_changed=True,
            native_runtime=protocol['runtime_consumer'],original_native_runtime=protocol['original_runtime'],author_outcome_protocol=protocol,
            max_native_slots=2,max_GPU_slots=1,max_CPU_preflight_slots=2,automatic_retry=False,inventory=inventory))
        atomic_json(runtime/'plan.json',dict(split=args.split,jobs=jobs,expected_cases=expected,policy_sha256=policy,
            source_outcome_filtering=False,automatic_retry=False))
        checking={};ready=[];active={};completed={};failed={};next_index=0;draining=False
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
                atomic_json(runtime/'status.json',dict(updated_utc=utc(),split=args.split,expected=expected,CPU_launched=next_index,
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
            native_successes=sum(r['native_success'] for r in completed.values()),formal_release_requires_independent_audit=True)
        atomic_json(runtime/'terminal.json',terminal);print(json.dumps(terminal),flush=True)


if __name__=='__main__':main()
