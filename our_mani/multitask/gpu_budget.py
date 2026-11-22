"""Owned 96-SM MPS lifecycle; independent of policy/model/release gates.

Static MPS clients share one SM set.  This is a CUDA SM resource budget, not
a limit on graphics, memory, bandwidth, power or instantaneous GPU utilization.
No CUDA/Torch/native library is imported by this module at import time.
"""
from datetime import datetime, timezone
import ctypes
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

PROFILE_SCHEMA='r10_project_shared_96SM_MPS_profile_v1'
CONTRACT_SCHEMA='r10_project_MPS96_execution_environment_contract_v1'
APPROVAL_SCHEMA='r10_ROOT_project_MPS96_launch_approval_v1'
RECEIPT_SCHEMA='r10_actual_project_shared_96SM_MPS_execution_v1'
MPS_KEYS=('CUDA_MPS_PIPE_DIRECTORY','CUDA_MPS_LOG_DIRECTORY','CUDA_MPS_SM_PARTITION')
CONTROL=Path('/usr/bin/nvidia-cuda-mps-control')
SERVER=Path('/usr/bin/nvidia-cuda-mps-server')


def require(value,message):
    if not value:raise ValueError(message)


def now():return datetime.now(timezone.utc).isoformat()


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def pointer(path):
    path=Path(path).resolve()
    return dict(path=str(path),sha256=sha(path))


def bound(ref):
    path=Path(ref['path'])
    require(path.is_absolute() and path.is_file() and sha(path)==ref['sha256'],'bound execution input changed: '+str(path))
    return path.resolve()


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as handle:json.dump(value,handle,indent=2,sort_keys=True,allow_nan=False);handle.write('\n')


def canonical_inside(root,path):
    root=Path(root).resolve();path=Path(path)
    require(path.is_absolute() and path.resolve()==path and path.is_relative_to(root),
        'MPS path must be canonical inside project before any write')
    current=path
    while current!=root:
        require(not current.is_symlink(),'MPS namespace cannot traverse symlinks')
        current=current.parent
    return path


def process_identity(pid):
    require(type(pid) is int and pid>0,'positive process identity required')
    base=Path('/proc')/str(pid)
    stat=base.joinpath('stat').read_text();tail=stat[stat.rfind(')')+2:].split()
    environment={}
    for item in base.joinpath('environ').read_bytes().split(b'\0'):
        key,separator,value=item.partition(b'=')
        if separator and key.decode(errors='replace') in MPS_KEYS+('CUDA_VISIBLE_DEVICES',):
            environment[key.decode()]=value.decode()
    return dict(pid=pid,parent_pid=int(tail[1]),start_ticks=int(tail[19]),
        executable=str(base.joinpath('exe').resolve()),
        argv=[item.decode(errors='replace') for item in base.joinpath('cmdline').read_bytes().split(b'\0') if item],
        MPS_environment=environment)


def identity_is_live(identity):
    try:return process_identity(identity['pid'])['start_ticks']==identity['start_ticks']
    except (FileNotFoundError,ProcessLookupError,PermissionError):return False


def discover_daemon(namespace):
    identities=[]
    for path in Path('/proc').glob('[0-9]*/comm'):
        try:
            if not path.read_text().strip().startswith('nvidia-cuda-mps'):continue
            identity=process_identity(int(path.parent.name))
            if Path(identity['executable']).resolve()==CONTROL.resolve() and identity['MPS_environment'].get(
                'CUDA_MPS_PIPE_DIRECTORY')==str(Path(namespace)/'p'):identities.append(identity)
        except (FileNotFoundError,ProcessLookupError,PermissionError):pass
    return identities


def descendants(pid):
    rows={}
    for path in Path('/proc').glob('[0-9]*/stat'):
        try:
            text=path.read_text();tail=text[text.rfind(')')+2:].split()
            rows[int(path.parent.name)]=int(tail[1])
        except (FileNotFoundError,ProcessLookupError,PermissionError,ValueError):pass
    found={pid}
    while True:
        addition={child for child,parent in rows.items() if parent in found}-found
        if not addition:return found
        found|=addition


def existing_MPS_processes():
    result=[]
    for path in Path('/proc').glob('[0-9]*/comm'):
        try:
            if path.read_text().strip().startswith('nvidia-cuda-mps'):result.append(int(path.parent.name))
        except (FileNotFoundError,PermissionError):pass
    return result


def command(argv,*,env=None,input=None,timeout=30):
    p=subprocess.run(argv,env=env,input=input,text=True,capture_output=True,timeout=timeout)
    return dict(argv=list(argv),input=input,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr,utc=now())


def hardware_observation(run=command):
    result=run(['nvidia-smi','--query-gpu=uuid,name,driver_version,compute_mode','--format=csv,noheader'])
    require(result['returncode']==0,'cannot observe GPU hardware')
    lines=result['stdout'].strip().splitlines();require(len(lines)==1,'exact single reviewed GPU required')
    values=[value.strip() for value in lines[0].split(',')]
    require(len(values)==4,'GPU hardware query schema differs')
    return dict(uuid=values[0],name=values[1],driver_version=values[2],compute_mode=values[3])


def validate_profile(profile,root):
    root=Path(root).resolve()
    require(profile.get('schema')==PROFILE_SCHEMA and profile.get('root')==str(root), 'MPS profile schema/root differs')
    require(profile['static_partitioning'] is True and profile['chunks']==12
        and profile['physical_SM_count']==170 and profile['partition_SM_count']==96
        and profile['maximum_project_CUDA_SM_fraction']==.6
        and 96/170<=profile['maximum_project_CUDA_SM_fraction'], 'fixed reviewed 96/170SM budget required')
    namespace=canonical_inside(root,Path(profile['namespace']))
    require(namespace.is_relative_to(root/'runtime/mps') and namespace!=root/'runtime/mps'
        and len(os.fsencode(str(namespace/'p'/'control')))<100,'short dedicated runtime/mps namespace required')
    require(profile['GPU']['compute_mode']=='Default' and profile['GPU']['name']=='NVIDIA GeForce RTX 5090'
        and profile['GPU']['driver_version']=='595.99.02','reviewed actual GPU/driver/default compute mode required')
    require(isinstance(profile['allowed_command_argv'],list) and profile['allowed_command_argv']
        and all(isinstance(value,str) and value and '\0' not in value for value in profile['allowed_command_argv']),
        'exact nonempty workload argv required')
    require(profile.get('scope')=='CUDA_SM_resource_budget_only'
        and profile.get('replaces_policy_model_data_or_release_gates') is False,'resource wrapper cannot replace release gates')
    require(profile.get('foreign_CUDA_process_policy')=='observe_only_no_mutation'
        and profile.get('exclusive_SM_or_performance_guarantee') is False,
        'foreign CUDA workloads must remain untouched; no exclusivity claim')
    for ref in profile['inputs']:bound(ref)
    require(profile['inputs'] and len({ref['path'] for ref in profile['inputs']})==len(profile['inputs']),
        'unique source/runtime/execution inputs required')
    required={str(path.resolve()) for path in (Path(__file__),root/'scripts/run_project_GPU_budget_r10.py',
        CONTROL,SERVER,Path('/usr/lib/x86_64-linux-gnu/libcuda.so.1'),Path(profile['context_probe_python']))}
    require(required<={str(Path(ref['path']).resolve()) for ref in profile['inputs']},
        'budget source/controller/server/driver/interpreter closure is incomplete')
    return namespace


def prepare(root,output,namespace,argv,execution_inputs=(),*,python_executable=None,run=command):
    root=Path(root).resolve();output=canonical_inside(root,Path(output).resolve())
    require(not output.exists(),'fresh budget metadata namespace required')
    namespace=canonical_inside(root,Path(namespace))
    python_executable=Path(python_executable or sys.executable).resolve()
    owned=[Path(__file__),root/'scripts/run_project_GPU_budget_r10.py',CONTROL,SERVER,
           Path('/usr/lib/x86_64-linux-gnu/libcuda.so.1'),python_executable]
    inputs={str(Path(path).resolve()):pointer(path) for path in owned+list(execution_inputs)}
    profile=dict(schema=PROFILE_SCHEMA,created_utc=now(),root=str(root),namespace=str(namespace),
        GPU=hardware_observation(run),physical_SM_count=170,partition_SM_count=96,chunks=12,
        maximum_project_CUDA_SM_fraction=.6,actual_fraction=96/170,static_partitioning=True,
        active_thread_percentage_used=False,all_CUDA_descendants_share_same_partition=True,
        scope='CUDA_SM_resource_budget_only',replaces_policy_model_data_or_release_gates=False,
        foreign_CUDA_process_policy='observe_only_no_mutation',exclusive_SM_or_performance_guarantee=False,
        graphics_memory_bandwidth_NVENC_or_utilization_cap=False,allowed_command_argv=list(argv),
        context_probe_python=str(python_executable),inputs=list(inputs.values()),prepared_only=True,
        actual_MPS_or_workload_executed=False)
    validate_profile(profile,root)
    require(not namespace.exists(),'new execution namespace already exists')
    output.mkdir(parents=True)
    write(output/'profile.json',profile)
    contract=dict(schema=CONTRACT_SCHEMA,budget_profile=pointer(output/'profile.json'),
        exact_environment_template=dict(CUDA_VISIBLE_DEVICES='0',CUDA_MPS_PIPE_DIRECTORY=str(namespace/'p'),
            CUDA_MPS_LOG_DIRECTORY=str(namespace/'l'),CUDA_MPS_SM_PARTITION='derived exact owned partition ID'),
        actual_required_evidence=['server_PID_start_ticks_executable','partition_ID_same_GPU_UUID_12chunks_96SM',
            'context_SM_count96','all_actual_MPS_client_PIDs_are_owned_descendants',
            'each_actual_client_exact_environment','no_bypass_CUDA_compute_PIDs','raw_control_events',
            'workload_argv_exitcode','partition_remove_and_daemon_quit','source_inputs_before_after'],
        original_policy_model_data_source_split_clock_reset_collision_and_TEST_release_gates_retained=True,
        legacy_baseline_contract_without_MPS_witness_is_compatible=False,
        prepared_only=True,actual_MPS_or_workload_executed=False)
    write(output/'execution_environment_contract.json',contract)
    return dict(profile=pointer(output/'profile.json'),execution_environment_contract=pointer(output/'execution_environment_contract.json'))


def verify_approval(root,profile_path,approval_path):
    profile_path,approval_path=Path(profile_path).resolve(),Path(approval_path).resolve()
    profile=json.loads(profile_path.read_text());validate_profile(profile,root)
    approval=json.loads(approval_path.read_text())
    require(approval.get('schema')==APPROVAL_SCHEMA and approval.get('approved') is True
        and approval.get('budget_profile')==pointer(profile_path)
        and approval.get('allowed_command_argv')==profile['allowed_command_argv'], 'exact ROOT budget/argv approval required')
    contract_path=bound(approval['execution_environment_contract']);contract=json.loads(contract_path.read_text())
    require(contract.get('schema')==CONTRACT_SCHEMA and contract.get('budget_profile')==pointer(profile_path)
        and contract.get('legacy_baseline_contract_without_MPS_witness_is_compatible') is False,
        'new exact resource environment contract required; old baseline contract is insufficient')
    require(isinstance(approval.get('execution_inputs'),list) and approval['execution_inputs'],
        'approval must bind final workload plan/config inputs separately to avoid profile SHA cycles')
    for ref in approval['execution_inputs']:bound(ref)
    return profile,dict(approval=pointer(approval_path),execution_environment_contract=pointer(contract_path),
        execution_inputs=approval['execution_inputs'])


class MPSBudgetController:
    """Injectable lifecycle, with no policy action and no global compute changes."""
    def __init__(self,root,profile,receipt_dir,*,run=command,process_probe=process_identity,daemon_discovery=discover_daemon,
                 existing_MPS=existing_MPS_processes):
        self.root=Path(root).resolve();self.profile=profile
        self.namespace=validate_profile(profile,self.root)
        self.receipt_dir=canonical_inside(self.root,Path(receipt_dir).resolve())
        require(not self.receipt_dir.exists(),'preserve previous budget execution evidence')
        self.receipt_dir.mkdir(parents=True)
        self.run,self.process_probe,self.daemon_discovery=run,process_probe,daemon_discovery
        self.existing_MPS=existing_MPS
        self.env={key:value for key,value in os.environ.items() if not key.startswith('CUDA_MPS_')}
        self.env.update(CUDA_VISIBLE_DEVICES='0',CUDA_MPS_PIPE_DIRECTORY=str(self.namespace/'p'),
                        CUDA_MPS_LOG_DIRECTORY=str(self.namespace/'l'))
        self.events=[];self.partition=None;self.server=None;self.daemon_started=False;self.closed=False

    def event(self,kind,value):
        row=dict(kind=kind,utc=now(),value=value);write(self.receipt_dir/'events'/f'{len(self.events):05d}.json',row)
        self.events.append(row);return value

    def control(self,text,*,must_succeed=True):
        value=self.event('control_command',self.run([str(CONTROL)],env=self.env,input=text+'\n'))
        if must_succeed:require(value['returncode']==0,'MPS control command failed: '+text)
        return value

    def start(self):
        require(not self.daemon_started and not self.namespace.exists(),'MPS lifecycle can initialize only once into fresh namespace')
        require(hardware_observation(self.run)==self.profile['GPU'],'actual GPU identity/driver/compute mode changed')
        apps=self.event('GPU_compute_apps_before',self.run(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader']))
        require(apps['returncode']==0,'cannot observe existing CUDA workloads')
        # No existing daemon is stopped or reused, including another project.
        require(not self.existing_MPS(),'existing MPS daemon; no launch')
        self.namespace.mkdir(parents=True)
        (self.namespace/'p').mkdir();(self.namespace/'l').mkdir()
        result=self.event('daemon_start',self.run([str(CONTROL),'-d','-S'],env=self.env))
        require(result['returncode']==0,'owned static MPS daemon failed to start')
        self.daemon_started=True
        daemons=self.daemon_discovery(self.namespace)
        require(len(daemons)==1,'exact one own control daemon identity required')
        self.daemon=daemons[0];self.event('actual_daemon_identity',self.daemon)
        require(Path(self.daemon['executable']).resolve()==CONTROL.resolve()
            and self.daemon['MPS_environment'].get('CUDA_MPS_PIPE_DIRECTORY')==self.env['CUDA_MPS_PIPE_DIRECTORY'],
            'control daemon executable/namespace identity differs')
        self.control('start_server -uid '+str(os.getuid()))
        servers=self.control('get_server_list')['stdout'].strip().splitlines()
        require(len(servers)==1 and servers[0].isdigit(),'exact one owned MPS server required')
        self.server=self.process_probe(int(servers[0]));self.event('actual_server_identity',self.server)
        require(Path(self.server['executable']).resolve()==SERVER.resolve()
            and self.server['MPS_environment'].get('CUDA_MPS_PIPE_DIRECTORY')==self.env['CUDA_MPS_PIPE_DIRECTORY'],
            'MPS server executable/namespace identity differs')
        value=self.control('sm_partition add '+self.profile['GPU']['uuid']+' 12')['stdout'].strip()
        match=re.fullmatch(r'Partition ('+re.escape(self.profile['GPU']['uuid'])+r'/[A-Za-z0-9+/=]+) created',value)
        require(match,'actual exact fixed12 partition ID missing')
        self.partition=match.group(1);self.env['CUDA_MPS_SM_PARTITION']=self.partition
        self.event('actual_client_environment',{key:self.env[key] for key in ('CUDA_VISIBLE_DEVICES',)+MPS_KEYS})
        return dict(self.env)

    def clients(self):
        text=self.control('ps')['stdout'];rows=[]
        for line in text.splitlines()[1:]:
            values=line.split()
            if not values:continue
            require(len(values)>=5 and values[0].isdigit() and values[2].isdigit(),'MPS ps schema differs')
            require(self.server is not None and int(values[2])==self.server['pid'],'unexpected server in own MPS namespace')
            rows.append(int(values[0]))
        require(len(rows)==len(set(rows)),'duplicate client process rows')
        return rows

    def validate_clients(self,allowed_pids):
        identities=[]
        for pid in self.clients():
            require(pid in allowed_pids,'foreign client in owned namespace; do not terminate it')
            actual=self.process_probe(pid)
            require(all(actual['MPS_environment'].get(key)==self.env[key] for key in MPS_KEYS)
                and actual['MPS_environment'].get('CUDA_VISIBLE_DEVICES')=='0',
                'CUDA client did not inherit exact shared partition environment')
            identities.append(actual)
        return self.event('actual_owned_client_identities',identities)

    def verify_compute_processes(self,client_pids,*,owned_pids=None):
        owned=set(client_pids if owned_pids is None else owned_pids)
        value=self.run(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'])
        require(value['returncode']==0,'cannot verify actual CUDA compute processes')
        pids=[]
        for line in value['stdout'].strip().splitlines():
            token=line.split(',')[0].strip();require(token.isdigit(),'GPU compute process query schema differs')
            pids.append(int(token))
        bypass=(set(pids)&owned)-set(client_pids)-{self.server['pid']}
        foreign=set(pids)-owned-{self.server['pid']}
        value.update(owned_project_pids=sorted(owned),verified_MPS_client_pids=sorted(client_pids),
            foreign_observed_pids=sorted(foreign),foreign_processes_untouched=True,
            exclusive_SM_or_performance_guarantee=False)
        self.event('actual_GPU_compute_processes',value)
        require(not bypass,'owned GPU process bypasses the reviewed shared partition')
        return pids

    def close(self):
        if self.closed:return
        if self.daemon_started:
            require(not self.clients(),'clients remain: preserve server instead of unsafe shutdown')
            if self.partition:
                result=self.control('sm_partition rm '+self.partition)['stdout'].strip()
                require(result in ('Partition '+self.partition+' removed',
                    'Partition '+self.partition.split('/',1)[1]+' removed'),'partition removal was not confirmed')
            self.control('quit')
        self.closed=True
        self.event('cleanup',dict(partition_removed=self.partition is not None,
            own_daemon_quit=self.daemon_started,foreign_process_terminated=False,global_compute_mode_changed=False))
        for ref in self.profile['inputs']:bound(ref)


def context_probe():
    """Future launch handshake only. No CUDA kernel or model is executed."""
    library=ctypes.CDLL('libcuda.so.1');context=ctypes.c_void_p()
    def call(name,*args):
        result=getattr(library,name)(*args);require(result==0,name+' failed: '+str(result))
    try:
        call('cuInit',0);device=ctypes.c_int();call('cuDeviceGet',ctypes.byref(device),0)
        call('cuCtxCreate_v2',ctypes.byref(context),0,device)
        count=ctypes.c_int();call('cuDeviceGetAttribute',ctypes.byref(count),16,device)
        call('cuCtxSynchronize')
        print(json.dumps(dict(status='ready',identity=process_identity(os.getpid()),actual_SM_count=count.value,
            no_kernels_models_or_simulator=True)),flush=True)
        sys.stdin.readline()
        call('cuCtxSynchronize');call('cuCtxDestroy_v2',context);context=ctypes.c_void_p()
        print(json.dumps(dict(status='completed',context_destroyed=True,kernel_launches=0)),flush=True)
    finally:
        if context.value:library.cuCtxSynchronize();library.cuCtxDestroy_v2(context)
