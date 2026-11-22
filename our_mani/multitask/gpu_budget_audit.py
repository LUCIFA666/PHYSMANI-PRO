"""Pure historical audit of the independent owned CUDA resource witness."""
import json
from pathlib import Path
import re

from .gpu_budget import (APPROVAL_SCHEMA,CONTRACT_SCHEMA,CONTROL,MPS_KEYS,PROFILE_SCHEMA,
    RECEIPT_SCHEMA,SERVER,pointer,require,sha,validate_profile,verify_approval)


def read_bound(root,ref):
    path=(Path(root)/ref['path']).resolve()
    require(path.is_relative_to(Path(root).resolve()) and path.is_file() and sha(path)==ref['sha256'],
        'historical budget file missing/changed/outside project')
    return json.loads(path.read_text()),path


def process(value,*,environment=None,executable=None):
    require(isinstance(value,dict) and type(value.get('pid')) is int and value['pid']>0
        and type(value.get('start_ticks')) is int and value['start_ticks']>0
        and type(value.get('parent_pid')) is int and value['parent_pid']>=0
        and isinstance(value.get('argv'),list) and value['argv'],'raw budget PID/starttick/argv missing')
    if environment is not None:
        require(value.get('MPS_environment')==environment,'actual CUDA client environment differs from exact common partition')
    if executable is not None:
        require(value.get('executable')==str(Path(executable).resolve()),'actual owned process binary differs')
    return value['pid']


def validate_budget_execution(root,receipt_pointer,profile_pointer,*,required_case_process_ids=()):
    root=Path(root).resolve();receipt,rpath=read_bound(root,receipt_pointer)
    profile,ppath=read_bound(root,profile_pointer);validate_profile(profile,root)
    require(receipt.get('schema')==RECEIPT_SCHEMA and receipt.get('complete') is True
        and receipt.get('budget_profile')==pointer(ppath)
        and receipt.get('infrastructure_error') is None and receipt.get('cleanup_error') is None
        and receipt.get('workload_returncode')==0 and receipt.get('automatic_retry') is False
        and receipt.get('policy_failure_claimed') is False
        and receipt.get('original_policy_data_model_clock_physics_and_release_gates_replaced') is False,
        'budget scope incomplete/errored or altered original gates')
    approvals=receipt['approval'];approval,apath=read_bound(root,approvals['approval'])
    _,verified=verify_approval(root,ppath,apath)
    require(approvals==verified,'budget actual approval/contract/final input pointers differ')
    events=[]
    for index,ref in enumerate(receipt['runtime_events']):
        event,path=read_bound(root,ref)
        require(path==rpath.parent/'events'/f'{index:05d}.json','budget raw event sequence/path differs')
        events.append(event)
    kinds={}
    for event in events:kinds.setdefault(event['kind'],[]).append(event['value'])
    def one(kind):
        require(len(kinds.get(kind,[]))==1,'exact one budget '+kind+' witness required')
        return kinds[kind][0]
    require(not any(key in kinds for key in ('execution_error','cleanup_error','awaiting_owned_workload_cleanup')),
        'budget error or interrupted cleanup cannot qualify')
    start=one('daemon_start')
    require(start['argv']==[str(CONTROL),'-d','-S'] and start['returncode']==0,'stock owned static daemon not started')
    daemon=one('actual_daemon_identity');server=one('actual_server_identity')
    process(daemon,executable=CONTROL);server_pid=process(server,executable=SERVER)
    env=one('actual_client_environment')
    require(set(env)==set(('CUDA_VISIBLE_DEVICES',)+MPS_KEYS) and env['CUDA_VISIBLE_DEVICES']=='0'
        and env['CUDA_MPS_PIPE_DIRECTORY']==profile['namespace']+'/p'
        and env['CUDA_MPS_LOG_DIRECTORY']==profile['namespace']+'/l'
        and re.fullmatch(re.escape(profile['GPU']['uuid'])+r'/[A-Za-z0-9+/=]+',env['CUDA_MPS_SM_PARTITION']),
        'actual budget partition/environment is not the reviewed fixed namespace/GPU')
    for identity in (daemon,server):
        require(identity['MPS_environment'].get('CUDA_MPS_PIPE_DIRECTORY')==env['CUDA_MPS_PIPE_DIRECTORY'],
                'own daemon/server namespace differs')
    controls=kinds.get('control_command',[])
    require(all(value['argv']==[str(CONTROL)] and value['returncode']==0 for value in controls),
        'control process/exit differed')
    allowed_controls={'get_server_list\n','ps\n','lspart\n','quit\n',
        'sm_partition add '+profile['GPU']['uuid']+' 12\n',
        'sm_partition rm '+env['CUDA_MPS_SM_PARTITION']+'\n'}
    require(all(value['input'] in allowed_controls or re.fullmatch(r'start_server -uid [0-9]+\n',value['input'])
        for value in controls),'unknown resource-changing MPS command in execution')
    def control(text):
        rows=[row for row in controls if row['input']==text+'\n']
        require(len(rows)==1,'exact control witness missing: '+text);return rows[0]
    add=control('sm_partition add '+profile['GPU']['uuid']+' 12')
    require(add['stdout'].strip()=='Partition '+env['CUDA_MPS_SM_PARTITION']+' created','actual partition creation differs')
    require(control('get_server_list')['stdout'].strip()==str(server_pid),'actual server registry differs')
    removal=control('sm_partition rm '+env['CUDA_MPS_SM_PARTITION'])['stdout'].strip()
    require(removal in ('Partition '+env['CUDA_MPS_SM_PARTITION']+' removed',
        'Partition '+env['CUDA_MPS_SM_PARTITION'].split('/',1)[1]+' removed'),'partition cleanup not confirmed')
    control('quit')
    probe=one('actual_context_probe');probe_process=one('context_probe_process')
    require(probe.get('status')=='ready' and probe.get('actual_SM_count')==96
        and probe.get('no_kernels_models_or_simulator') is True
        and probe['identity']==probe_process['identity'],'actual context96 handshake differs')
    probe_pid=process(probe['identity'],environment=env)
    cleanup=one('context_probe_cleanup')
    require(cleanup.get('returncode')==0 and cleanup.get('identity_no_longer_live') is True,
        'handshake context process cleanup unproved')
    destroyed=json.loads(cleanup['stdout'].strip())
    require(destroyed.get('status')=='completed' and destroyed.get('context_destroyed') is True
        and destroyed.get('kernel_launches')==0,'handshake context destroy witness missing')
    seen={}
    for rows in kinds.get('actual_owned_client_identities',[]):
        for identity in rows:
            pid=process(identity,environment=env)
            require(pid not in seen or seen[pid]['start_ticks']==identity['start_ticks'],'CUDA client PID reused')
            seen[pid]=identity
    require(probe_pid in seen,'driver context was not observed as actual MPS client')
    require(all(type(pid) is int and pid>0 for pid in required_case_process_ids)
        and set(required_case_process_ids)<=set(seen),'required baseline native case PID lacks resource witness')
    child=receipt['actual_workload_process'];process(child,environment=env)
    workload=one('actual_workload_process')
    require(workload['identity']==child and workload['command']==profile['allowed_command_argv']
        and workload.get('original_workload_arguments_preserved') is True,'actual workload argv/identity differs')
    exit=one('actual_workload_exit')
    require(exit['returncode']==0 and exit['identity']==child and exit['identity_no_longer_live'] is True,
        'approved workload process did not finish cleanly')
    final=one('cleanup')
    require(final==dict(partition_removed=True,own_daemon_quit=True,foreign_process_terminated=False,
        global_compute_mode_changed=False),'owned lifecycle cleanup is incomplete/overbroad')
    for value in kinds.get('actual_GPU_compute_processes',[]):
        require(value['returncode']==0,'CUDA bypass query failed')
        pids={int(line.split(',')[0].strip()) for line in value['stdout'].strip().splitlines()}
        owned=set(value.get('owned_project_pids',[]));clients=set(value.get('verified_MPS_client_pids',[]))
        require(clients<=set(seen) and (pids&owned)<=(clients|{server_pid}),
            'owned GPU compute process bypassed common partition')
        require(set(value.get('foreign_observed_pids',[]))==pids-owned-{server_pid}
            and value.get('foreign_processes_untouched') is True
            and value.get('exclusive_SM_or_performance_guarantee') is False,
            'foreign CUDA classification/intervention/exclusivity claim differs')
    require(receipt.get('actual_MPS_log_files'),'complete budget execution must preserve actual MPS logs')
    for ref in receipt['actual_MPS_log_files']:
        path=(root/ref['path']).resolve()
        require(path.is_relative_to(Path(profile['namespace'])/'l') and sha(path)==ref['sha256'],
            'actual MPS log changed/outside owned namespace')
    return dict(schema='r10_verified_actual_shared_project_MPS96_resource_component_v1',
        valid=True,receipt=pointer(rpath),profile=pointer(ppath),approval=pointer(apath),
        environment_contract=dict(path=str(Path(verified['execution_environment_contract']['path']).relative_to(root)),
            sha256=verified['execution_environment_contract']['sha256']),
        required_case_process_ids=list(required_case_process_ids),actual_client_process_ids=sorted(seen),
        CUDA_SM_count=96,physical_SM_count=170,maximum_common_SM_fraction=96/170,
        complete_policy_model_data_or_TEST_release_claimed=False)
