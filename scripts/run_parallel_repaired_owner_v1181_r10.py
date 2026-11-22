"""Explicit, bounded parallel TRAIN suffix or admitted VAL owner; inert by default.

The old TRAIN prefix keeps its old graph and identities. Only the untouched
source-order suffix runs under the new control graph. NativeFalse/fatal/motion
diagnostics never affect admission, fairness or resource-level promotion.
"""
import argparse
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts import run_repaired_policy_owner_v1169_r10 as serial

graphs=serial.graphs
readiness=serial.readiness
SOURCE='scripts/run_parallel_repaired_owner_v1181_r10.py'
TEST_SOURCE='tests/test_parallel_repaired_owner_v1181.py'
WORKER='scripts/run_parallel_repaired_case_v1180_r10.py'
PREPARER='our_mani/multitask/parallel_repaired_case_v1180.py'
CONSUMER=serial.CONSUMER
WRAPPER=serial.WRAPPER
GUARD='our_mani/multitask/parallel_repaired_resource_guard_v1179.py'
GATE='our_mani/multitask/parallel_repaired_admission_v1183.py'
WRITER='our_mani/multitask/parallel_repaired_validation_inputs_v1184.py'
BASE='runtime/framework/parallel_repaired_owner_v1181'
ART='artifacts/framework/multitask_r10/parallel_repaired_policy_v1180'
RUNTIME='runtime/framework/parallel_repaired_policy_v1180'
SCHEMA='r10_parallel_graph_bound_owner_v1181'
TRAIN_PLAN_SCHEMA='r10_parallel_fixed_TRAIN_suffix_plan_v1181'
VAL_PLAN_SCHEMA='r10_parallel_validation_execution_plan_v1181'
VAL_COHORT_SCHEMA='r10_parallel_validation_execution_cohort_v1185'
REVIEW_SCHEMA='r10_parallel_owner_technical_review_v1181'
BUNDLE_SCHEMA='r10_post_admission_parallel_repaired_validation_inputs_v1184'
BUNDLE_PREFIX='artifacts/framework/multitask_r10/parallel_repaired_validation_inputs_v1184/'
CASE_SCHEMAS={s:'r10_parallel_graph_bound_repaired_policy_case_v1180_'+('TRAIN' if s=='train' else 'validation')+'_case' for s in ('train','validation')}
OLD_BASE='runtime/framework/repaired_policy_owner_v1169/train/native_release_v1168_train192'
OLD_PLAN_SHA='6f4cb58bedf96056b5b4299ddc6b5c56f233991ceb61d7ec59fe36d34c3b3db7'
OLD_GRAPH_SHA='3e4208721ccbc64963c4aa0f80b86f91155933dfe7abe344d307a015f146d23f'
OLD_PROCESS=dict(pid=1080979,start_ticks=13365749)
PHYSICAL_SHA='12c96b9b7818cc2cc76e2d185233b47042a3ef5063c5ddb25ecb20f63f21e470'
TRANSITION_SCHEMA='r10_actual_serial_to_parallel_transition_exit_v1173'
TRANSITION_PATH=OLD_BASE+'/parallel_transition_exit_v1173.json'
SUPERSEDED_BASE='runtime/framework/parallel_repaired_owner_v1173/train/repaired_parallel_v1178_TRAIN_suffix118'
SUPERSEDED_EXIT_PATH=SUPERSEDED_BASE+'/superseded_CPU_batch_exit_v1181.json'
SUPERSEDED_EXIT_SCHEMA='r10_actual_superseded_CPU_control_batch_exit_v1181'
SUPERSEDED_PLAN_SHA='58306b72e8c152379552f17706d00f16068142d2d32d16a545218feb64d8f27f'
SUPERSEDED_GRAPH_SHA='aeefcaa6f4de26821e20f6d98e2e433388bb004f367d6dcb17cf31ba62dd759e'
SUPERSEDED_PROCESS=dict(pid=32008,start_ticks=179637)
PROFILE=dict(max_native_slots=12,max_GPU_slots=5,max_CPU_preflight_slots=6,ready_limit=24)
LEVELS=((2,1,2),(4,2,2),(8,3,4),(12,5,6))
JOB_KEYS={'job_id','task','method','raw_episode_id','split','configuration','out','runtime','source_row_sha256'}
THREAD_CONTROL=dict(launcher_defaults_preserved=True,effective_launcher_environment=dict(
    OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4'),
    physical_source_or_device_selection_changed=False,bitwise_equivalence_claimed=False,
    note='Frozen runtime.sh unconditionally sets4/4/4; actual process threads and CPU busy are sampled.')
AFFINITY_PATH='artifacts/framework/multitask_r10/parallel_repaired_control_v1186_setup/parallel_CPU_affinity_profile.json'
AFFINITY_SCHEMA='r10_parallel_repaired_CPU_affinity_v1186'
path=serial.path
pointer=serial.pointer
read=serial.read
same=serial.same
fresh=serial.fresh
atomic=serial.atomic
artifact_ref=serial.artifact_ref
exact_ref=serial.exact_ref
process_identity=serial.process_identity
utc=serial.utc
stamp=serial.stamp


def require(value,message):
    if not value:raise ValueError('parallel1181: '+message)


def _module(name):return importlib.import_module(name[:-3].replace('/','.'))


def _source(root,name,expected):
    require(same(pointer(root,name),readiness.pointer(expected)),'actual frozen source differs: '+name)


def _cpulist(value):
    result=set()
    for part in value.strip().split(','):
        if '-' in part:
            first,last=map(int,part.split('-',1));result.update(range(first,last+1))
        elif part:result.add(int(part))
    return sorted(result)


def _hardware_snapshot():
    """Startup only, before narrowing the controller's own affinity."""
    allowed=sorted(os.sched_getaffinity(0));groups={}
    for cpu in allowed:
        base=Path('/sys/devices/system/cpu/cpu'+str(cpu));top=base/'topology'
        package=int((top/'physical_package_id').read_text());core=int((top/'core_id').read_text())
        siblings=_cpulist((top/'thread_siblings_list').read_text());L3=None
        for index in (base/'cache').glob('index*'):
            if (index/'level').read_text().strip()=='3':
                size=(index/'size').read_text().strip().upper();L3=int(size[:-1])*{'K':1024,'M':2**20,'G':2**30}[size[-1]];break
        require(L3 is not None,'actual L3 topology required')
        row=groups.setdefault((package,core),dict(package=package,core=core,cpus=[],L3_bytes=L3))
        require(row['L3_bytes']==L3,'SMT siblings cannot disagree on L3')
        row['cpus'].append(cpu);row.setdefault('_siblings',siblings)
        require(row['_siblings']==siblings,'inconsistent actual SMT siblings')
    result=[]
    for key in sorted(groups):
        row=groups[key];siblings=row.pop('_siblings');row['cpus'].sort()
        require(row['cpus']==[c for c in siblings if c in allowed],'actual complete core grouping differs')
        result.append(row)
    return dict(allowed_logical_cpus=allowed,core_groups=result)


def read_CPU_affinity(root,graph):
    """Bind control JSON, host taskset bytes and the actual startup topology."""
    static=graph['control']['static_inputs']
    reference=exact_ref(root,static['parallel_CPU_affinity_profile'],AFFINITY_PATH)
    data,_=read(root,reference);actual=_hardware_snapshot()
    require(data.get('schema')==AFFINITY_SCHEMA and data.get('physical_core_count')==16
        and data.get('max_native_slots')==PROFILE['max_native_slots']
        and data.get('bitwise_equivalence_claimed') is False
        and same(data.get('allowed_logical_cpus'),actual['allowed_logical_cpus'])
        and same(data.get('core_groups'),actual['core_groups'])
        and len(actual['core_groups'])==16 and all(len(g['cpus'])==2 for g in actual['core_groups']),
        'actual allowed CPUs/sysfs SMT/L3 topology must equal graph-frozen control metadata')
    native=data.get('native_slot_cpus');pool=data.get('control_and_CPU_preflight_cpus')
    cores=[g['cpus'] for g in actual['core_groups']]
    require(type(native) is list and len(native)==12 and all(c in cores for c in native)
        and len({tuple(c) for c in native})==12 and type(pool) is list and len(pool)==8 and pool==sorted(set(pool))
        and all(type(c) is int for c in pool) and not (set(pool)&{c for pair in native for c in pair})
        and set(pool)|{c for pair in native for c in pair}==set(actual['allowed_logical_cpus'])
        and sum(set(c)<=set(pool) for c in cores)==4,'12 whole Native cores and a disjoint whole4-core controller/CPU pool required')
    binary=data.get('taskset_binary');require(type(binary) is dict and binary.get('root')=='host_tools' and binary.get('path')=='taskset'
        and data.get('taskset_root')=='/usr/bin' and graph['roots'].get('host_tools')=='/usr/bin'
        and same(binary,static.get('parallel_taskset_binary'))
        and data.get('taskset_runs_outside820_as_control_tool') is True,'actual external control taskset declaration required')
    executable=Path(data['taskset_root'])/binary['path']
    require(executable.resolve()==executable and executable.is_file() and os.access(executable,os.X_OK)
        and hashlib.sha256(executable.read_bytes()).hexdigest()==binary['sha256'],'actual host taskset binary changed')
    return dict(data,profile_ref=reference)


def expected_CPU_cpus(plan,assignment):
    affinity=plan['CPU_affinity']
    return list(affinity['control_and_CPU_preflight_cpus'] if assignment['phase']=='CPU_check'
        else affinity['native_slot_cpus'][assignment['slot']])


def _self_thread_ids():return [int(p.name) for p in Path('/proc/self/task').iterdir()]


@contextmanager
def parent_CPU_affinity(cpus):
    """Pin every existing controller TID; later threads inherit the pool."""
    original={};default=set(os.sched_getaffinity(0));target=set(cpus)
    try:
        for _ in range(3):
            for tid in _self_thread_ids():
                try:
                    original.setdefault(tid,set(os.sched_getaffinity(tid)));os.sched_setaffinity(tid,target)
                except ProcessLookupError:continue
            try:
                if all(set(os.sched_getaffinity(tid))==target for tid in _self_thread_ids()):break
            except ProcessLookupError:continue
        else:raise ValueError('parallel1181: controller thread affinity did not converge')
        yield
    finally:
        for tid in _self_thread_ids():
            try:os.sched_setaffinity(tid,original.get(tid,default))
            except ProcessLookupError:pass


def graph_for_owner(root,graph_ref,split):
    require(split in ('train','validation'),'TEST refused before metadata/source reads')
    graph_ref=artifact_ref(root,graph_ref,prefix='artifacts/framework/multitask_r10/',basename='policy_graph.json')
    graph=graphs.read_policy_graph(root,graph_ref)
    role='train_owner' if split=='train' else 'evaluation_owner'
    _source(root,SOURCE,graph['control']['components'][role])
    _source(root,GUARD,graph['control']['components']['resource_guard'])
    require(graph['physical_policy_sha256']==PHYSICAL_SHA,'physical policy must equal the frozen serial parent')
    require(graph_ref['sha256']!=OLD_GRAPH_SHA,'new parallel control graph required')
    for route in graph['routes']:
        d=route['train' if split=='train' else 'evaluation']
        for name,key in ((WORKER,'worker'),(PREPARER,'preparer'),(CONSUMER,'consumer')):_source(root,name,d[key])
        require(d['case_schema']==CASE_SCHEMAS[split],'actual1180 case schema required')
    return graph,readiness.pointer(graph_ref)


def physical_equivalence(root,old_graph_ref,new_graph):
    old=graphs.read_policy_graph(root,old_graph_ref)
    require(old['physical_policy_sha256']==new_graph['physical_policy_sha256']==PHYSICAL_SHA
        and same(old['physical'],new_graph['physical']),'entire physical graph must be unchanged')
    require(all(new_graph['roots'].get(k)==v for k,v in old['roots'].items()),'every original root must keep its original path')
    before={(r['task'],r['method']):r for r in old['routes']};after={(r['task'],r['method']):r for r in new_graph['routes']}
    require(set(before)==set(after) and len(after)==48,'same original48 routes required')
    for key in before:
        a,b=before[key],after[key]
        require(same({k:v for k,v in a.items() if k not in ('train','evaluation')},
            {k:v for k,v in b.items() if k not in ('train','evaluation')}),'nonworker route fields changed')
        for split in ('train','evaluation'):
            allowed={'worker','preparer','case_schema'}
            require(same({k:v for k,v in a[split].items() if k not in allowed},
                {k:v for k,v in b[split].items() if k not in allowed}),'original route consumer/purpose fields changed')
    return dict(physical_unchanged=True,original_roots_unchanged=True,
        added_control_only_roots=sorted(set(new_graph['roots'])-set(old['roots'])))


def inspect_predecessor(root,exit_ref):
    """Read actual administrative drain; never relabel its preserved prefix."""
    exit_ref=exact_ref(root,exit_ref,TRANSITION_PATH);e,exit_ref=read(root,exit_ref)
    require(e.get('schema')==TRANSITION_SCHEMA and e.get('actual_tool_exit_observed') is True
        and e.get('controlled_drain') is True and e.get('natural_completion') is False
        and type(e.get('returncode')) is int and e['returncode']==1
        and e.get('native_children_all_dead') is True and e.get('Native_killed') is False,
        'Root actual administrative exit1/drain, never a fake natural192 exit')
    plan,pr=read(root,exact_ref(root,e['old_plan'],OLD_BASE+'/plan.json'))
    launch,lr=read(root,exact_ref(root,e['old_launch'],OLD_BASE+'/launch.json'))
    status,sr=read(root,exact_ref(root,e['old_status'],OLD_BASE+'/status.json'))
    failure,fr=read(root,exact_ref(root,e['terminal_or_failure'],OLD_BASE+'/failure.json'))
    require(pr['sha256']==OLD_PLAN_SHA and plan['graph']['sha256']==OLD_GRAPH_SHA
        and plan['physical_policy_sha256']==PHYSICAL_SHA and plan.get('split')=='train'
        and plan.get('expected_cases')==192 and len(plan.get('jobs',[]))==192,
        'actual original192 serial source plan required')
    p=readiness._process(launch.get('process'))
    require(same(p,OLD_PROCESS) and same(readiness._process(e.get('process')),p)
        and same(readiness.pointer(launch.get('plan')),pr) and readiness._not_live(p),
        'old actual owner PID:start must be dead')
    k=e.get('completed_prefix_count');active=status.get('active')
    require(type(k) is int and 0<=k<192 and k==len(status.get('completed',[]))
        and status.get('stage')=='failed' and same(status,failure)
        and (active is None or (type(active) is dict and active.get('phase')=='CPU_check')),
        'closed original prefix and at most a CPU-only orphan required')
    errors=status.get('failed',[])
    require(len(errors)==1 and errors[0].get('native_failure_inferred') is False
        and errors[0].get('exception_type')=='ValueError' and errors[0].get('automatic_retry') is False
        and same(errors[0].get('active'),active)
        and errors[0].get('message') in ('owner1169: STOP before next fixed source','owner1169: STOP after CPU preflight')
        and status.get('error',{}).get('message')==errors[0]['message'],
        'the only original owner error must be administrative STOP')
    tool=e.get('actual_tool_output');require(type(tool) is dict,'Root actual tool output pointer required')
    artifact_ref(root,tool,prefix=OLD_BASE+'/')
    old_graph=graphs.read_policy_graph(root,plan['graph'])
    records=status['completed']
    require([readiness._key(r) for r in records]==[readiness._key(j) for j in plan['jobs'][:k]],'original completed prefix order required')
    for job,row in zip(plan['jobs'],records):
        require(same(serial.completed_case(root,plan,old_graph,job,row['CPU_check'],row['native']),row),'old prefix changed')
        for phase in ('CPU_check','native'):
            l,_=read(root,row[phase]['launch']);s,_=read(root,row[phase]['sidecar'])
            require(readiness._not_live(l['process']) and readiness._not_live(s['process']),'old Native/CPU process identity still live')
    if active is not None:
        job=plan['jobs'][k]
        require(readiness._key(active)==readiness._key(job),'CPU orphan must be next original job')
        outer_name=OLD_BASE+'/children/'+job['task']+'/'+job['raw_episode_id']+'/'+job['method']+'/CPU_check/'
        l,_=read(root,pointer(root,outer_name+'launch.json'));t,_=read(root,pointer(root,outer_name+'terminal.json'))
        s,_=read(root,pointer(root,job['runtime']+'/CPU_check/worker_sidecar.json'))
        require(type(t.get('returncode')) is int and t['returncode']==0 and readiness._not_live(l['process'])
            and readiness._not_live(s['process']),'CPU orphan must have exited without Native')
        require(not path(root,job['runtime']+'/native').exists(),'orphan Native must never have started')
    return dict(old_graph=plan['graph'],original_plan=pr,launch=lr,status=sr,failure=fr,exit=exit_ref,
        completed_prefix_count=k,records=records,physical_policy_sha256=PHYSICAL_SHA,
        case_graphs_relabelled=False,mixed_control_lineage=True)


def inspect_superseded_CPU_batch(root,exit_ref):
    """Close the unqualified control failure without retrying a Native result."""
    exit_ref=exact_ref(root,exit_ref,SUPERSEDED_EXIT_PATH);e,exit_ref=read(root,exit_ref)
    require(e.get('schema')==SUPERSEDED_EXIT_SCHEMA and e.get('actual_tool_exit_observed') is True
        and e.get('natural_completion') is False and type(e.get('returncode')) is int and e['returncode']==1
        and e.get('Native_started') is False and e.get('all_child_processes_dead') is True
        and type(e.get('completed')) is int and e['completed']==0
        and type(e.get('attempted')) is int and e['attempted']==2,
        'actual unqualified CPU-only control failure exit1 is mandatory')
    plan,pr=read(root,exact_ref(root,e['plan'],SUPERSEDED_BASE+'/plan.json'))
    launch,lr=read(root,exact_ref(root,e['launch'],SUPERSEDED_BASE+'/launch.json'))
    status,sr=read(root,exact_ref(root,e['status'],SUPERSEDED_BASE+'/status.json'))
    terminal,tr=read(root,exact_ref(root,e['terminal'],SUPERSEDED_BASE+'/terminal.json'))
    actual,ar=read(root,exact_ref(root,e['actual_tool_output'],SUPERSEDED_BASE+'/failed_CPU_exec_completion_v1181.json'))
    process=readiness._process(launch.get('process'))
    require(pr['sha256']==SUPERSEDED_PLAN_SHA and plan['graph']['sha256']==SUPERSEDED_GRAPH_SHA
        and plan.get('schema')=='r10_parallel_fixed_TRAIN_suffix_plan_v1173'
        and plan.get('split')=='train' and plan.get('owner_runtime')==SUPERSEDED_BASE
        and plan.get('preserved_prefix_count')==74 and plan.get('global_expected_cases')==192
        and plan.get('expected_cases')==118 and len(plan.get('jobs',[]))==118
        and [j.get('job_id') for j in plan['jobs']]==[f'j{i:04d}' for i in range(74,192)]
        and plan.get('automatic_retry') is False and plan.get('source_outcome_filtering') is False
        and plan.get('physical_policy_sha256')==PHYSICAL_SHA,
        'exact failed118 suffix plan and unchanged old74 identity required')
    require(same(process,SUPERSEDED_PROCESS) and same(readiness._process(e.get('process')),process)
        and same(launch.get('plan'),pr) and readiness._not_live(process),
        'superseded actual owner identity must be dead')
    for value in (status,terminal):
        require(same(value.get('plan'),pr) and value.get('active') is None
            and type(value.get('attempted')) is int and value['attempted']==2
            and value.get('automatic_retry') is False and value.get('drain_reason')=='infrastructure_failure_drain',
            'superseded control failure must have drained both actual CPU workers')
    require(status.get('stage')=='drained' and status.get('completed')==[]
        and len(status.get('failed',[]))==2
        and {f.get('job_id') for f in status['failed']}=={'j0075','j0076'}
        and all(f.get('phase')=='CPU_check' and f.get('Native_failure_inferred') is False for f in status['failed'])
        and terminal.get('complete') is False and type(terminal.get('completed')) is int and terminal['completed']==0
        and type(terminal.get('infrastructure_failed')) is int and terminal['infrastructure_failed']==2
        and same(terminal.get('process'),process)
        and terminal.get('remaining_job_ids')==[j['job_id'] for j in plan['jobs']],
        'zero completed scientific cases; only the two CPU infrastructure failures are superseded')
    require(actual.get('schema')=='r10_actual_unqualified_CPU_parallel_owner_exit_v1178'
        and actual.get('actual_tool_exit_observed') is True and actual.get('natural_completion') is False
        and type(actual.get('returncode')) is int and actual['returncode']==1
        and type(actual.get('actual_exec_session')) is int and actual['actual_exec_session']==28165
        and actual.get('Native_started') is False and actual.get('Native_killed') is False
        and actual.get('owner_dead') is True and actual.get('all_children_dead') is True
        and actual.get('source_outcomes_created') is False and actual.get('scientific_qualification_granted') is False
        and type(actual.get('completed_native_cases')) is int and actual['completed_native_cases']==0
        and type(actual.get('attempted_CPU_cases')) is int and actual['attempted_CPU_cases']==2
        and actual.get('automatic_retry') is False and same(actual.get('graph'),plan['graph'])
        and type(actual.get('preserved_original_prefix_count')) is int and actual['preserved_original_prefix_count']==74
        and same(actual.get('process'),process)
        and all(same(actual.get(k),v) for k,v in (('plan',pr),('launch',lr),('status',sr),('terminal',tr))),
        'Root actual session28165 exit proof must bind the unqualified failed control batch')
    phases=actual.get('CPU_phases');require(type(phases) is list and len(phases)==2,'two actual closed CPU launches required')
    jobs_by_id={j['job_id']:j for j in plan['jobs']};seen=set();child_processes=set();launch_paths=[]
    for phase in phases:
        l,lref=read(root,artifact_ref(root,phase['launch'],prefix=SUPERSEDED_BASE+'/children/',basename='launch.json'))
        assignment=l.get('resource_assignment',{});job_id=assignment.get('job_id')
        require(job_id in {'j0075','j0076'} and job_id not in seen,'exact unique failed CPU job IDs required')
        job=jobs_by_id[job_id];base=SUPERSEDED_BASE+'/children/'+job['task']+'/'+job['raw_episode_id']+'/'+job['method']+'/CPU_check/'
        exact_ref(root,lref,base+'launch.json');t,tref=read(root,exact_ref(root,phase['terminal'],base+'terminal.json'))
        stderr=exact_ref(root,phase['stderr'],base+'stderr.log');text=path(root,stderr['path']).read_text()
        child=readiness._process(l.get('process'))
        require(l.get('check_only') is True and assignment.get('phase')=='CPU_check'
            and all(l.get(k)==job[k] for k in ('task','raw_episode_id','method'))
            and same(l.get('graph'),plan['graph']) and same(t.get('process'),child) and same(phase.get('process'),child)
            and type(t.get('returncode')) is int and t['returncode']==1 and t.get('automatic_retry') is False
            and readiness._not_live(child) and 'PermissionError' in text and '/proc/32008/fd/4' in text,
            'actual dead CPU-only cross-namespace FD permission failure required')
        token=(child['pid'],child['start_ticks'])
        require(token not in child_processes and not same(child,process),'fresh distinct CPU process identities required')
        child_processes.add(token)
        seen.add(job_id);launch_paths.append(lref['path'])
    children=path(root,SUPERSEDED_BASE+'/children')
    require(seen=={'j0075','j0076'} and sorted(p.relative_to(root).as_posix() for p in children.rglob('launch.json'))==sorted(launch_paths)
        and not any(p.is_dir() and p.name=='native' for p in children.rglob('*'))
        and all(not path(root,j['runtime']+'/native').exists() and not path(root,j['out']+'/case.json').exists() for j in plan['jobs']),
        'failed control batch must have created no Native phase or scientific outcome')
    return dict(exit=exit_ref,plan=pr,launch=lr,status=sr,terminal=tr,actual_tool_output=ar,
        process=process,completed_native_cases=0,attempted_CPU_cases=2,all_child_processes_dead=True,
        preserved_original_prefix_count=74,Native_retried=False,scientific_qualification_granted=False)


def validation_rows(root,graph_ref,graph,bundle_ref):
    bundle_ref=artifact_ref(root,bundle_ref,prefix=BUNDLE_PREFIX,basename='bundle.json')
    bundle,bundle_ref=read(root,bundle_ref)
    require(bundle.get('schema')==BUNDLE_SCHEMA and type(bundle.get('source_count')) is int and bundle['source_count']==315
        and bundle.get('Native_started') is False and bundle.get('TEST_read') is False,'actual1184 admitted bundle required')
    readiness._stamp(bundle,graph_ref,graph)
    directory=Path(bundle_ref['path']).parent.as_posix()
    for key,name in (('suite','suite.json'),('suite_lock','suite_lock.json'),('source_inputs','source_inputs.json')):exact_ref(root,bundle[key],directory+'/'+name)
    for name,key in ((GATE,'admission_guard'),(GATE,'formal_source_reader'),(WRITER,'split_source_lock_writer')):_source(root,name,graph['control']['components'][key])
    suite,_=read(root,bundle['suite']);lock,_=read(root,bundle['suite_lock'])
    guard=_module(GUARD);gate=_module(GATE)
    with guard.metadata_scope(root,enabled=True) as permit:
        with gate.admission_scope(root,graph_ref,bundle['admission'],permit):
            _,rows,_=gate.read_formal_source(root,suite,lock['files'],'validation')
    require(len(rows)==315 and len({(r['task'],r['raw_episode_id']) for r in rows})==315,'exact315 original admitted rows required')
    return rows,bundle_ref


def build_jobs(graph,sources,split,namespace,start=0):
    require(type(namespace) is str and namespace and all(c.isalnum() or c in '_.-' for c in namespace),'simple fresh namespace required')
    routes={(r['task'],r['method']):r for r in graph['routes']};jobs=[]
    for index,(source,method) in enumerate((s,m) for s in sources for m in graphs.METHODS):
        task,raw=source['task'],source['raw_episode_id'];route=routes[(task,method)]
        require(type(raw) is str and raw and all(c.isalnum() or c in '_.-' for c in raw),'simple original source ID required')
        suffix=f'cohorts/{namespace}/{task}/{raw}/{method}'
        jobs.append(dict(job_id=f'j{index:04d}',task=task,raw_episode_id=raw,method=method,split=split,
            configuration=route['configuration'],source_row_sha256=readiness.digest(source),
            out=f'{ART}/{split}/'+('development_demo_only/' if split=='train' else '')+suffix,
            runtime=f'{RUNTIME}/{split}/{suffix}'))
    return jobs[start:]


def validate_review(root,reference,graph_ref,graph,split,namespace,count,predecessor,bundle_ref,superseded):
    reference=artifact_ref(root,reference,prefix='artifacts/framework/multitask_r10/')
    review,reference=read(root,reference);readiness._stamp(review,graph_ref,graph)
    require(review.get('schema')==REVIEW_SCHEMA and review.get('reviewer')=='root' and review.get('review_complete') is True
        and review.get('release_fixed_cohort') is True and review.get('split')==split and review.get('namespace')==namespace
        and type(review.get('expected_cases')) is int and review['expected_cases']==count
        and review.get('automatic_retry') is False and review.get('source_outcome_filtering') is False,'Root exact parallel cohort review required')
    require(same(readiness.pointer(review.get('superseded_CPU_batch_exit')),superseded['exit']),
        'review must bind the actual superseded CPU-only control failure')
    if split=='train':require(same(readiness.pointer(review.get('predecessor_exit')),predecessor['exit']),'review must bind actual prefix drain')
    else:require(same(readiness.pointer(review.get('validation_bundle')),bundle_ref),'review must bind actual admitted bundle')
    return reference


def prepare_parallel_plan(root,graph_ref,*,split='train',namespace,review_ref,superseded_CPU_exit_ref,predecessor_exit_ref=None,bundle_ref=None):
    root=Path(root).resolve();graph,graph_ref=graph_for_owner(root,graph_ref,split)
    superseded=inspect_superseded_CPU_batch(root,superseded_CPU_exit_ref)
    affinity=read_CPU_affinity(root,graph)
    runtime=f'{BASE}/{split}/{namespace}';require(not path(root,runtime).exists(),'fresh owner namespace required')
    predecessor=None;start=0
    if split=='train':
        require(bundle_ref is None,'TRAIN cannot carry VAL metadata')
        predecessor=inspect_predecessor(root,predecessor_exit_ref);old_plan,_=read(root,predecessor['original_plan'])
        physical_equivalence(root,predecessor['old_graph'],graph)
        sources=old_plan['sources'];start=predecessor['completed_prefix_count']
        require(len(sources)==64,'all64 original TRAIN rows must remain in plan')
    else:
        require(predecessor_exit_ref is None,'VAL qualification comes from admitted mixed192, not a development prefix')
        sources,bundle_ref=validation_rows(root,graph_ref,graph,bundle_ref)
    jobs=build_jobs(graph,sources,split,namespace,start)
    review_ref=validate_review(root,review_ref,graph_ref,graph,split,namespace,len(jobs),predecessor,bundle_ref,superseded)
    require(jobs and all(not path(root,j['out']).exists() and not path(root,j['runtime']).exists() for j in jobs),'all suffix case namespaces must be fresh')
    plan=dict(schema=TRAIN_PLAN_SCHEMA if split=='train' else VAL_PLAN_SCHEMA,split=split,namespace=namespace,owner_runtime=runtime,
        global_expected_cases=192 if split=='train' else 945,expected_cases=len(jobs),preserved_prefix_count=start,
        sources=sources,jobs=jobs,predecessor=predecessor,validation_bundle=bundle_ref,review=review_ref,
        superseded_CPU_batch=superseded,
        profile=PROFILE,initial_limits=list(LEVELS[0]),permitted_limits=[list(x) for x in LEVELS],resource_target_percent=90,
        thread_control_policy=THREAD_CONTROL,
        CPU_affinity=affinity,
        automatic_retry=False,source_outcome_filtering=False,reused_cases=[],case_graphs_relabelled=False,
        mixed_control_lineage=split=='train',CPU_check_before_every_native_case=True,created_utc=utc(),
        formal_TEST_permitted=False,qualification_granted_by_owner=False,**stamp(graph_ref,graph))
    fresh(path(root,runtime+'/plan.json'),plan);return pointer(root,runtime+'/plan.json')


def load_plan(root,plan_ref,*,execute=False):
    plan_ref=artifact_ref(root,plan_ref,prefix=BASE+'/',basename='plan.json');plan,plan_ref=read(root,plan_ref)
    split=plan.get('split');graph,gr=graph_for_owner(root,plan['graph'],split);readiness._stamp(plan,gr,graph)
    superseded=inspect_superseded_CPU_batch(root,plan['superseded_CPU_batch']['exit'])
    require(same(superseded,plan['superseded_CPU_batch']),'immutable superseded CPU control failure evidence differs')
    require(same(plan.get('CPU_affinity'),read_CPU_affinity(root,graph)),'graph-bound CPU affinity profile or host tool changed')
    require(plan.get('schema')==(TRAIN_PLAN_SCHEMA if split=='train' else VAL_PLAN_SCHEMA)
        and plan.get('profile')==PROFILE and plan.get('initial_limits')==list(LEVELS[0])
        and plan.get('permitted_limits')==[list(x) for x in LEVELS] and plan.get('resource_target_percent')==90
        and plan.get('thread_control_policy')==THREAD_CONTROL
        and plan.get('automatic_retry') is False and plan.get('source_outcome_filtering') is False
        and plan.get('reused_cases')==[] and plan.get('case_graphs_relabelled') is False
        and plan.get('CPU_check_before_every_native_case') is True and plan.get('formal_TEST_permitted') is False
        and plan.get('qualification_granted_by_owner') is False,'exact bounded, unfiltered parallel protocol required')
    require(plan['owner_runtime']==f'{BASE}/{split}/{plan["namespace"]}' and plan_ref['path']==plan['owner_runtime']+'/plan.json','plan namespace differs')
    start=plan.get('preserved_prefix_count');require(type(start) is int and 0<=start<192,'explicit original prefix count required')
    if split=='train':
        predecessor=inspect_predecessor(root,plan['predecessor']['exit'])
        physical_equivalence(root,predecessor['old_graph'],graph)
        require(same(predecessor,plan['predecessor']) and predecessor['completed_prefix_count']==start,'immutable old prefix evidence differs')
        old,_=read(root,predecessor['original_plan']);require(same(plan['sources'],old['sources']),'all64 original sources required')
    else:
        require(start==0 and plan.get('predecessor') is None,'VAL cannot reuse TRAIN cases')
        if execute:
            rows,bref=validation_rows(root,gr,graph,plan['validation_bundle']);require(same(rows,plan['sources']),'admitted315 rows changed')
    count=192 if split=='train' else 945
    require(type(plan.get('global_expected_cases')) is int and plan['global_expected_cases']==count
        and type(plan.get('expected_cases')) is int and plan['expected_cases']==count-start
        and len(plan['sources'])==(64 if split=='train' else 315) and plan.get('mixed_control_lineage') is (split=='train')
        and same(plan['jobs'],build_jobs(graph,plan['sources'],split,plan['namespace'],start)),'exact original-order suffix/grid required')
    require(all(set(j)==JOB_KEYS for j in plan['jobs']),'exact guard job declarations required')
    validate_review(root,plan['review'],gr,graph,split,plan['namespace'],count-start,plan.get('predecessor'),plan.get('validation_bundle'),superseded)
    return plan,plan_ref,graph


class ParallelScheduler:
    """Pure state machine; process launches and physical observations are external."""
    def __init__(self,jobs):
        require(len({j['job_id'] for j in jobs})==len(jobs),'unique jobs required')
        self.jobs=deepcopy(jobs);self.by_id={j['job_id']:j for j in self.jobs}
        self.states={j['job_id']:'queued' for j in jobs};self.running={};self.completed={};self.failures=[]
        self.closed_reason=None;self.level=0;self.cursors={'CPU_check':0,'native':0};self.intervals=[];self.samples=[];self.promotions=[]

    @property
    def limits(self):return LEVELS[self.level]

    def close_admission(self,reason):
        if self.closed_reason is None:self.closed_reason=reason

    def sample(self,value):
        require(type(value) is dict and all(type(value.get(k)) in (int,float) and math.isfinite(value[k]) for k in ('at','CPU_percent','RAM_percent','VRAM_percent','GPU_percent')),'finite actual resource sample required')
        self.samples.append(deepcopy(value))
        if value['RAM_percent']>90 or value['VRAM_percent']>90:self.close_admission('memory_above90_drain')

    def _pick(self,phase,eligible):
        for offset in range(len(graphs.METHODS)):
            index=(self.cursors[phase]+offset)%len(graphs.METHODS);method=graphs.METHODS[index]
            for job in self.jobs:
                if job['method']==method and eligible(job):
                    self.cursors[phase]=(index+1)%len(graphs.METHODS);return job
        return None

    def dispatch(self):
        if self.closed_reason is not None:return []
        launches=[];native_max,gpu_max,cpu_max=self.limits
        for phase,limit in (('native',native_max),('CPU_check',cpu_max)):
            while True:
                occupied={x['slot'] for x in self.running.values() if x['phase']==phase}
                slots=[s for s in range(limit) if s not in occupied]
                gpu_used={x['gpu_slot'] for x in self.running.values() if x['gpu_slot'] is not None}
                gpu_free=[s for s in range(gpu_max) if s not in gpu_used]
                waiting=sum(v in ('CPU_check','ready') for v in self.states.values())
                if not slots or (phase=='CPU_check' and waiting>=PROFILE['ready_limit']):break
                target='ready' if phase=='native' else 'queued'
                job=self._pick(phase,lambda j:self.states[j['job_id']]==target and (phase!='native' or j['method']!='physmani' or bool(gpu_free)))
                if job is None:break
                item=dict(job=deepcopy(job),phase=phase,slot=slots[0],gpu_slot=gpu_free[0] if phase=='native' and job['method']=='physmani' else None)
                self.states[job['job_id']]=phase;self.running[job['job_id']]=item;launches.append(item)
        return launches

    def finish(self,job_id,phase,record,*,infrastructure_error=None):
        require(job_id in self.running and self.running[job_id]['phase']==phase,'one actual admitted phase completion required')
        if infrastructure_error is not None:
            self.running.pop(job_id)
            self.states[job_id]='infra_failed';self.failures.append(dict(job_id=job_id,phase=phase,error=str(infrastructure_error),Native_failure_inferred=False));self.close_admission('infrastructure_failure_drain');return
        if phase=='CPU_check':self.running.pop(job_id);self.states[job_id]='ready';return
        require(type(record.get('native_success')) is bool and type(record.get('native_terminal_failure')) is bool and type(record.get('motion_pass')) is bool,'original outcome diagnostics required')
        interval=record.get('native_interval')
        require(type(interval) is dict and all(type(interval.get(k)) in (int,float) and math.isfinite(interval[k]) for k in ('start_monotonic','end_monotonic'))
            and interval['end_monotonic']>=interval['start_monotonic'],'actual physical interval required')
        self.running.pop(job_id)
        self.intervals.append(deepcopy(interval));self.states[job_id]='completed';self.completed[job_id]=deepcopy(record)

    def cancel_unlaunched(self,job_id):
        assignment=self.running.pop(job_id)
        self.states[job_id]='queued' if assignment['phase']=='CPU_check' else 'ready'

    def maybe_promote(self):
        if self.closed_reason or self.failures or self.level>=len(LEVELS)-1:return False
        required=self.limits[0]
        good=[s for s in self.samples if sum(i['start_monotonic']<=s['at']<=i['end_monotonic'] for i in self.intervals)>=required
            and all(0<=s[k]<90 for k in ('CPU_percent','RAM_percent','VRAM_percent','GPU_percent'))]
        if len(good)<3 or len(self.completed)<required:return False
        previous=self.level;self.level+=1;self.promotions.append(dict(from_level=previous,to_level=self.level,
            actual_overlap_required=required,headroom_samples=good[-3:],completed_for_throughput=len(self.completed),success_rate_used=False))
        return True

    def ordered_records(self):return [self.completed[j['job_id']] for j in self.jobs if j['job_id'] in self.completed]


def command(root,plan,graph,job,*,check,assignment=None,session_ref=None):
    require(assignment is not None and session_ref is not None,'actual per-phase resource assignment required')
    route=next(r for r in graph['routes'] if (r['task'],r['method'])==(job['task'],job['method']))
    phase='CPU_check' if check else 'native';affinity_assignment=dict(assignment,phase=phase)
    cpus=expected_CPU_cpus(plan,affinity_assignment);profile=plan['CPU_affinity']
    argv=[str(Path(profile['taskset_root'])/profile['taskset_binary']['path']),'--cpu-list',','.join(map(str,cpus)),
        str(Path(root)/'.conda-env/bin/python'),'-B',WRAPPER,'--manifest-sha256',graph['physical']['environment_manifest']['sha256'],
        '--','bash','scripts/run_aligned_gpu.sh' if route['GPU_required'] else 'scripts/run_local_sim.sh',WORKER,
        '--graph',plan['graph']['path'],'--config',job['configuration']['path'],'--split',plan['split'],
        '--task',job['task'],'--raw-episode-id',job['raw_episode_id'],'--method',job['method'],'--out',job['out'],'--runtime',job['runtime'],
        '--resource-session',session_ref['path'],'--resource-session-sha256',session_ref['sha256'],
        '--resource-slot',str(assignment['slot']),'--resource-job-id',job['job_id']]
    if assignment.get('gpu_slot') is not None:argv+=['--resource-gpu-slot',str(assignment['gpu_slot'])]
    if plan['split']=='validation':
        bundle,_=read(root,plan['validation_bundle'])
        for flag,key in (('--suite','suite'),('--suite-lock','suite_lock'),('--source-inputs','source_inputs'),('--admission','admission')):argv.extend((flag,bundle[key]['path']))
    return argv+(['--check'] if check else [])


def validate_live_parent_proofs(root,resource,sidecar,session,session_ref,claim_ref,claim,graph_ref,graph):
    """Reuse the frozen offline broker protocol for both live parent checks."""
    auditor_source='our_mani/multitask/parallel_repaired_readiness_v1182.py'
    _source(root,auditor_source,graph['control']['components']['train_auditor'])
    auditor=_module(auditor_source)
    auditor.validate_broker_session(session,session_ref,graph_ref,graph)
    admission=auditor.validate_broker_verification(resource.get('parent_broker_admission'),session,session_ref,claim_ref,claim)
    final=auditor.validate_broker_verification(resource.get('parent_broker_final'),session,session_ref,claim_ref,claim)
    initial=sidecar.get('resource_proof',{})
    require(same(initial.get('parent_broker_admission'),admission)
        and 'parent_broker_final' not in initial and admission['nonce']!=final['nonce'],
        'distinct actual admission/final broker exchanges required; no final proof at admission')
    for field in ('schema','purpose','phase','slot','gpu_slot','job_id','process','session','claim',
            'parent_global_locks_verified','child_empty_inventory_claimed'):
        require(same(initial.get(field),resource.get(field)),'initial/final resource evidence mismatch: '+field)


def validate_phase(root,plan,graph,job,refs,*,check):
    require(type(refs) is dict and set(refs)=={'launch','terminal','sidecar'},'six actual phase references required')
    phase='CPU_check' if check else 'native';base=plan['owner_runtime']+'/children/'+job['task']+'/'+job['raw_episode_id']+'/'+job['method']+'/'+phase
    launch,_=read(root,exact_ref(root,refs['launch'],base+'/launch.json'))
    terminal,_=read(root,exact_ref(root,refs['terminal'],base+'/terminal.json'))
    sidecar,_=read(root,exact_ref(root,refs['sidecar'],job['runtime']+'/'+phase+'/worker_sidecar.json'))
    owner,_=read(root,pointer(root,plan['owner_runtime']+'/launch.json'))
    outer=readiness._process(launch.get('process'));inner=readiness._process(sidecar.get('process'))
    assignment=launch['resource_assignment'];session=exact_ref(root,launch['resource_session'],plan['owner_runtime']+'/resources/session.json')
    session_value,_=read(root,session)
    require(assignment.get('phase')==phase and assignment.get('job_id')==job['job_id']
        and same(readiness._process(terminal.get('process')),outer)
        and same(readiness.pointer(owner['resource_session']),session)
        and type(terminal.get('returncode')) is int and terminal['returncode']==0
        and terminal.get('automatic_retry') is False and launch.get('check_only') is check
        and same(launch.get('argv'),command(root,plan,graph,job,check=check,assignment=assignment,session_ref=session))
        and readiness._not_live(outer) and readiness._not_live(inner),'actual argv/strict exit0/dead child identity required')
    expected_cpus=expected_CPU_cpus(plan,assignment)
    require(launch.get('expected_CPU_cpus')==expected_cpus,'actual outer command affinity declaration required')
    uses_GPU=not check and job['method']=='physmani'
    require(type(assignment.get('slot')) is int and 0<=assignment['slot']<plan['profile']['max_CPU_preflight_slots' if check else 'max_native_slots']
        and ((type(assignment.get('gpu_slot')) is int and 0<=assignment['gpu_slot']<plan['profile']['max_GPU_slots']) if uses_GPU else assignment.get('gpu_slot') is None)
        and session_value.get('schema')=='r10_parent_owned_parallel_resource_session_v1179'
        and same(session_value.get('owner'),owner['process']) and same(session_value.get('graph'),plan['graph'])
        and session_value.get('profile')==plan['profile'] and same(session_value.get('jobs'),plan['jobs']),
        'exact bounded actual session/phase/GPU slot required')
    route=next(r for r in graph['routes'] if (r['task'],r['method'])==(job['task'],job['method']))
    d=route['train' if plan['split']=='train' else 'evaluation']
    readiness._stamp(launch,plan['graph'],graph);readiness._stamp(sidecar,plan['graph'],graph)
    ns=sidecar.get('frozen_OS_environment',{});e=sidecar.get('execution',{})
    require(sidecar.get('schema')=='r10_parallel_graph_bound_repaired_policy_case_v1180'
        and sidecar.get('complete') is True and e.get('complete') is True
        and same(readiness.pointer(sidecar.get('actual_worker')),d['worker'])
        and same(readiness.pointer(sidecar.get('actual_preparer')),d['preparer'])
        and same(readiness.pointer(launch.get('actual_worker')),d['worker'])
        and same(readiness.pointer(launch.get('actual_preparer')),d['preparer'])
        and ns.get('complete') is True
        and all(type(ns.get(k)) is int for k in ('outer_wrapper_pid','outer_wrapper_start_ticks','observer_pid','observer_start_ticks'))
        and ns['outer_wrapper_pid']==outer['pid'] and ns['outer_wrapper_start_ticks']==outer['start_ticks']
        and ns['observer_pid']==inner['pid'] and ns['observer_start_ticks']==inner['start_ticks'],
        'actual1180/820/process headers required')
    require(e.get('native_started') is (not check) and e.get('actual_split')==plan['split'],'CPU/Native actual purpose differs')
    affinity_receipt=sidecar.get('CPU_affinity_receipt',{})
    for label in ('before','after'):
        checked=affinity_receipt.get(label,{})
        threads=checked.get('thread_cpus')
        require(checked.get('expected_cpus')==expected_cpus and checked.get('actual_main_cpus')==expected_cpus
            and same(readiness.pointer(checked.get('profile')),plan['CPU_affinity']['profile_ref'])
            and checked.get('all_threads_checked') is True and type(threads) is list and bool(threads)
            and all(type(t) is dict and type(t.get('tid')) is int and t['tid']>0 and t['cpus']==expected_cpus for t in threads)
            and len({t['tid'] for t in threads})==len(threads) and inner['pid'] in {t['tid'] for t in threads}
            and type(checked.get('vanished_thread_ids')) is list and all(type(t) is int and t>0 for t in checked['vanished_thread_ids'])
            and not (set(checked['vanished_thread_ids'])&{t['tid'] for t in threads})
            and type(checked.get('additional_native_calls')) is int and checked['additional_native_calls']==0,
            'actual all-thread CPU mask must match phase slot before and after execution')
    if check:require(e.get('mode')=='CPU_check' and e.get('GPU_started') is False,'original complete CPU preflight required')
    else:
        require(e.get('mode')=='graph_bound_native_'+('TRAIN' if plan['split']=='train' else 'validation')
            and e.get('native_false_preserved') is True,'honest actual Native result required')
        interval=e.get('actual_native_interval',{})
        require(all(type(interval.get(k)) in (int,float) and math.isfinite(interval[k]) for k in ('start_monotonic','end_monotonic'))
            and launch['started_monotonic']<=interval['start_monotonic']<=interval['end_monotonic']<=terminal['finished_monotonic'],
            'bounded actual physical runtime interval required')
    resource=sidecar.get('resource_scope_receipt',{})
    require(resource.get('schema')=='r10_parallel_child_resource_receipt_v1179'
        and resource.get('purpose')=='child_slot' and resource.get('Native_authorized') is (not check)
        and resource.get('child_empty_inventory_claimed') is False
        and resource.get('body_completed') is True and resource.get('restored') is True
        and resource.get('leases_removed') is True and resource.get('errors')==[] and resource.get('test_only') is False
        and same(readiness.pointer(resource.get('session')),session)
        and same(resource.get('process'),inner) and same(resource.get('actual_parent'),owner['process'])
        and resource.get('phase')==phase and resource.get('job_id')==job['job_id']
        and resource.get('slot')==assignment['slot'] and resource.get('gpu_slot')==assignment.get('gpu_slot')
        and resource.get('parent_global_locks_verified') is True,'clean actual resource scope/slot closure required')
    claim_name=plan['owner_runtime']+'/resources/claims/'+job['job_id']+'_'+phase+'/claim.json'
    claim,claim_ref=read(root,exact_ref(root,resource['claim'],claim_name))
    claim_terminal=pointer(root,str(Path(claim_ref['path']).parent/'terminal.json'))
    closed,_=read(root,claim_terminal)
    require(claim.get('schema')=='r10_once_only_parallel_resource_claim_v1179'
        and same(readiness.pointer(claim.get('session')),session) and claim.get('phase')==phase
        and same(claim.get('process'),inner) and type(claim.get('slot')) is int and claim['slot']==assignment['slot']
        and claim.get('gpu_slot')==assignment.get('gpu_slot') and claim.get('automatic_retry') is False
        and claim.get('test_only') is False and same(claim.get('job'),job) and same(closed,resource),
        'original exact-session once-only claim/terminal differs')
    validate_live_parent_proofs(root,resource,sidecar,session_value,session,claim_ref,claim,plan['graph'],graph)
    return sidecar


def _completed_case_base(root,plan,graph,job,cpu,native):
    """Compact closure; the post-run1182/1185 auditor remains authoritative."""
    CPU=validate_phase(root,plan,graph,job,cpu,check=True)
    Native=validate_phase(root,plan,graph,job,native,check=False)
    case,case_ref=read(root,pointer(root,job['out']+'/case.json'))
    split=plan['split'];route=next(r for r in graph['routes'] if (r['task'],r['method'])==(job['task'],job['method']))
    declaration=route['train' if split=='train' else 'evaluation']
    source=next(r for r in plan['sources'] if (r['task'],r['raw_episode_id'])==(job['task'],job['raw_episode_id']))
    witness=readiness.make_graph_witness(plan['graph'],graph,route,source,split=split)
    identity=case.get('identity',{});unhashed=dict(identity);unhashed.pop('sha256',None)
    inner=readiness._process(Native.get('process'));cpu_inner=readiness._process(CPU.get('process'))
    require(not same(inner,cpu_inner) and case.get('version')==declaration['case_schema'] and case.get('complete') is True
        and case.get('cleanup_completed') is True and case.get('infrastructure_error') is None
        and type(case.get('process_id')) is int and case['process_id']==inner['pid']
        and (case.get('task'),case.get('source_raw_episode_id'),case.get('method'),case.get('split'))==
            (job['task'],job['raw_episode_id'],job['method'],split)
        and same(case.get('graph_witness'),witness) and same(CPU.get('graph_witness'),witness) and same(Native.get('graph_witness'),witness)
        and same(CPU['execution'].get('graph_witness'),witness) and same(Native['execution'].get('graph_witness'),witness)
        and identity.get('sha256')==readiness.digest(unhashed)
        and identity.get('config_sha256')==route['configuration']['sha256']
        and identity.get('frozen_inputs_sha256')==route['frozen_inputs']['sha256']
        and identity.get('source_row_sha256')==readiness.digest(source)
        and identity.get('graph_sha256')==plan['graph']['sha256'] and all(identity.get(k)==graph[k] for k in readiness.HASHES)
        and same(CPU['execution'].get('identity'),identity) and same(Native['execution'].get('identity'),identity),
        'clean actual CPU/native/case graph/config/model/source identity binding required')
    require(same(case.get('execution_protocol'),CPU['execution'].get('execution_protocol'))
        and same(case.get('execution_protocol'),Native['execution'].get('execution_protocol'))
        and identity.get('execution_protocol_schema')==case.get('execution_protocol',{}).get('schema')
        and identity.get('execution_protocol_sha256')==readiness.digest(case.get('execution_protocol'))
        and case.get('original_745_execution_acceptance_granted') is False
        and case.get('qualification_granted_by_worker') is False
        and case.get('formal_score_contribution') is (split=='validation'),'honest actual new protocol boundary required')
    require(same(readiness.pointer(Native['execution'].get('case')),case_ref)
        and Native['execution'].get('native_success') is case.get('official_task_success')
        and Native['execution'].get('native_failure') is case.get('native_terminal_failure')
        and Native['execution'].get('runtime_scope_restoration',{}).get('complete') is True
        and Native['execution'].get('runtime_scope_restoration',{}).get('restored') is True,'actual Native outcomes/restoration differ')
    profile,protocol=readiness._profile(graph,root)
    receipt=readiness.validate_sampling_case(case,protocol,block=job['task'] in readiness.BLOCKS,case_version=declaration['case_schema'])
    readiness._attestation(case,profile);motion_pass=readiness._motion(case)
    hooks,hook_ref=read(root,pointer(root,job['out']+'/owned_hook_receipt.json'))
    require(hooks.get('actual_split')==split and type(hooks.get('reset')) is list and type(hooks.get('paths')) is list
        and all(type(r) is dict and r.get('complete') is True for r in hooks['reset'])
        and all(type(r) is dict and r.get('close_complete') is True for r in hooks['paths']), 'actual owned reset/RML cleanup required')
    if job['task'] in readiness.BLOCKS:require(hooks['reset']==[],'original block reset interface required')
    else:require(len(hooks['reset'])==1,'one actual core reset receipt required')
    return dict(task=job['task'],raw_episode_id=job['raw_episode_id'],method=job['method'],attempt=1,
        native_success=case['official_task_success'],native_terminal_failure=case['native_terminal_failure'],
        counted_ticks=case['counted_clock']['ticks'],counted_seconds=case['counted_clock']['seconds'],motion_pass=motion_pass,
        completed_actions=len(receipt['completed_actions']),case=case_ref,owned_hooks=hook_ref,CPU_check=cpu,native=native)


def completed_case(root,plan,graph,job,cpu,native):
    record=_completed_case_base(root,plan,graph,job,cpu,native)
    value,_=read(root,native['sidecar'])
    record.update(job_id=job['job_id'],native_interval=value['execution']['actual_native_interval'])
    return record

def sample_resources(previous=None):
    """Host/proc and nvidia-smi metadata only; no simulation or policy query."""
    now=time.monotonic();cpu=[int(x) for x in Path('/proc/stat').read_text().splitlines()[0].split()[1:9]]
    mem={line.split(':',1)[0]:int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith(('MemTotal:','MemAvailable:'))}
    busy=0.0
    if previous is not None:
        delta=[x-y for x,y in zip(cpu,previous['raw_cpu'])];total=sum(delta)
        if total>0:busy=100*(1-(delta[3]+delta[4])/total)
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,memory.total','--format=csv,noheader,nounits'],text=True,timeout=10).strip().splitlines()
    require(len(raw)==1,'this resource profile is for the measured single GPU')
    util,used,total=[float(x.strip()) for x in raw[0].split(',')]
    return dict(at=now,CPU_percent=busy,RAM_percent=100*(1-mem['MemAvailable']/mem['MemTotal']),
        VRAM_percent=100*used/total,GPU_percent=util,raw_cpu=cpu,VRAM_used_MiB=used,VRAM_total_MiB=total)


class ProcessDriver:
    """No kill/terminate operation exists: every admitted child is reaped."""
    def __init__(self,root,plan,graph,session):self.root=Path(root);self.plan=plan;self.graph=graph;self.session=session;self.cpu={}

    def launch(self,assignment):
        job=assignment['job'];phase=assignment['phase'];base=self.plan['owner_runtime']+'/children/'+job['task']+'/'+job['raw_episode_id']+'/'+job['method']+'/'+phase
        directory=path(self.root,base);directory.mkdir(parents=True,exist_ok=False)
        out=(directory/'stdout.log').open('x');err=(directory/'stderr.log').open('x')
        resource=dict(job_id=job['job_id'],phase=phase,slot=assignment['slot'],gpu_slot=assignment['gpu_slot'])
        argv=command(self.root,self.plan,self.graph,job,check=phase=='CPU_check',assignment=resource,session_ref=self.session)
        started=time.monotonic();handle=dict(assignment=assignment,base=base,started=started,out=out,err=err,launch_error=None)
        try:
            child=subprocess.Popen(argv,cwd=self.root,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),stdin=subprocess.DEVNULL,stdout=out,stderr=err)
        except BaseException:
            out.close();err.close();raise
        handle['child']=child
        try:
            process=process_identity(child.pid);handle['process']=process
            route=next(r for r in self.graph['routes'] if (r['task'],r['method'])==(job['task'],job['method']))
            d=route['train' if self.plan['split']=='train' else 'evaluation']
            fresh(directory/'launch.json',dict(process=process,argv=argv,check_only=phase=='CPU_check',
                task=job['task'],raw_episode_id=job['raw_episode_id'],method=job['method'],automatic_retry=False,
                actual_worker=d['worker'],actual_preparer=d['preparer'],resource_assignment=resource,resource_session=self.session,
                thread_control_policy=THREAD_CONTROL,
                expected_CPU_cpus=expected_CPU_cpus(self.plan,resource),
                started_monotonic=started,created_utc=utc(),**stamp(self.plan['graph'],self.graph)))
        except BaseException as error:handle['launch_error']=error
        return handle

    def poll(self,handle):return handle['child'].poll()

    def wait(self,handle):return handle['child'].wait()

    def process_metadata(self,handles):
        roots={h['child'].pid for h in handles.values()};rows={}
        for file in Path('/proc').glob('[0-9]*/stat'):
            try:
                fields=file.read_text().rsplit(')',1)[1].split();pid=int(file.parent.name)
                rows[pid]=(int(fields[1]),int(fields[17]),int(fields[21])*os.sysconf('SC_PAGE_SIZE'))
            except (OSError,ValueError,IndexError):continue
        selected=set(roots)
        for _ in range(12):selected|={p for p,(parent,_,_) in rows.items() if parent in selected}
        output=[]
        for p in sorted(selected):
            if p not in rows:continue
            try:mask=sorted(os.sched_getaffinity(p))
            except ProcessLookupError:continue
            output.append(dict(pid=p,parent_pid=rows[p][0],threads=rows[p][1],RSS_bytes=rows[p][2],CPU_cpus=mask))
        return output

    def collect(self,handle,code):
        handle['child'].wait();handle['out'].close();handle['err'].close();finished=time.monotonic()
        fresh(path(self.root,handle['base']+'/terminal.json'),dict(process=handle.get('process'),returncode=code,
            automatic_retry=False,finished_monotonic=finished,wall_seconds=finished-handle['started'],finished_utc=utc()))
        if handle['launch_error'] is not None:raise handle['launch_error']
        require(type(code) is int and code==0,'child infrastructure exit; original NativeFalse is an exit0 result')
        assignment=handle['assignment'];job=assignment['job'];phase=assignment['phase']
        refs={key:pointer(self.root,handle['base']+'/'+key+'.json') for key in ('launch','terminal')}
        refs['sidecar']=pointer(self.root,job['runtime']+'/'+phase+'/worker_sidecar.json')
        validate_phase(self.root,self.plan,self.graph,job,refs,check=phase=='CPU_check')
        if phase=='CPU_check':self.cpu[job['job_id']]=refs;return refs
        return completed_case(self.root,self.plan,self.graph,job,self.cpu[job['job_id']],refs)


def run_scheduler(plan,driver,*,sampler=sample_resources,clock=time.monotonic,sleep=time.sleep,stop_requested=lambda:False,on_update=lambda state:None):
    scheduler=ParallelScheduler(plan['jobs']);handles={};previous=None;last_sample=-math.inf;started=clock()
    def collect_one(job_id,handle,code):
        assignment=scheduler.running[job_id]
        try:record=driver.collect(handle,code);scheduler.finish(job_id,assignment['phase'],record)
        except BaseException as error:scheduler.finish(job_id,assignment['phase'],{},infrastructure_error=error)
        handles.pop(job_id)
    try:
        while True:
            if stop_requested():scheduler.close_admission('administrative_STOP_drain')
            if clock()-last_sample>=1:
                try:
                    previous=sampler(previous)
                    if hasattr(driver,'process_metadata'):previous['owned_processes']=driver.process_metadata(handles)
                    scheduler.sample(previous)
                except BaseException as error:scheduler.close_admission('resource_monitor_unavailable_drain:'+type(error).__name__)
                last_sample=clock()
            for job_id,handle in list(handles.items()):
                code=driver.poll(handle)
                if code is not None:collect_one(job_id,handle,code)
            scheduler.maybe_promote()
            for assignment in scheduler.dispatch():
                job_id=assignment['job']['job_id']
                if stop_requested():scheduler.close_admission('administrative_STOP_drain')
                if scheduler.closed_reason is not None:
                    scheduler.cancel_unlaunched(job_id);continue
                try:
                    handles[job_id]=driver.launch(assignment)
                    if type(handles[job_id]) is dict and handles[job_id].get('launch_error') is not None:
                        scheduler.close_admission('launch_metadata_failure_drain')
                except BaseException as error:scheduler.finish(job_id,assignment['phase'],{},infrastructure_error=error)
            on_update(scheduler)
            if not handles and (scheduler.closed_reason is not None or len(scheduler.completed)==len(plan['jobs'])):break
            sleep(.1)
    except BaseException as error:
        scheduler.close_admission('controller_exception_drain:'+type(error).__name__)
        # Even a reporting exception must not release the global lease while
        # a previously admitted child is alive. wait() never kills a child.
        for job_id,handle in list(handles.items()):collect_one(job_id,handle,driver.wait(handle))
    return dict(complete=len(scheduler.completed)==len(plan['jobs']) and not scheduler.failures and scheduler.closed_reason is None,
        records=scheduler.ordered_records(),infrastructure_failures=scheduler.failures,drain_reason=scheduler.closed_reason,
        remaining_job_ids=[j['job_id'] for j in scheduler.jobs if j['job_id'] not in scheduler.completed],
        resource_samples=scheduler.samples,promotions=scheduler.promotions,elapsed_seconds=clock()-started,
        throughput_completed_per_second=len(scheduler.completed)/max(clock()-started,1e-9),
        attempted=sum(v!='queued' for v in scheduler.states.values()),success_rate_used_for_scheduling=False,automatic_retry=False)


def execute_owner(root,plan_ref,plan_sha256):
    root=Path(root).resolve();plan_ref=artifact_ref(root,plan_ref,prefix=BASE+'/',basename='plan.json')
    require(plan_ref['sha256']==plan_sha256,'explicit exact new plan pin required')
    lock=path(root,BASE+'/owner.lock');lock.parent.mkdir(parents=True,exist_ok=True)
    with lock.open('a+') as owner_lock:
        fcntl.flock(owner_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        header,_=read(root,plan_ref);base=header['owner_runtime'];require(not path(root,base+'/launch.json').exists(),'no resume/overwrite/retry')
        plan,plan_ref,graph=load_plan(root,plan_ref,execute=True)
        require(all(not path(root,j['out']).exists() and not path(root,j['runtime']).exists() for j in plan['jobs']),'all case namespaces must still be fresh')
        guard=_module(GUARD)
        with parent_CPU_affinity(plan['CPU_affinity']['control_and_CPU_preflight_cpus']), guard.owner_scope(root,graph_ref=plan['graph'],plan_ref=plan_ref,profile=plan['profile'],runtime=base,enabled=True) as scope:
            process=process_identity();session=scope['session']
            fresh(path(root,base+'/launch.json'),dict(schema=SCHEMA,process=process,source=pointer(root,SOURCE),plan=plan_ref,
                resource_session=session,automatic_retry=False,created_utc=utc(),**stamp(plan['graph'],graph)))
            def update(state):
                atomic(path(root,base+'/status.json'),dict(schema=SCHEMA,stage='draining' if state.closed_reason else 'running',
                    expected_cases=plan['expected_cases'],preserved_prefix_count=plan['preserved_prefix_count'],
                    completed=state.ordered_records(),failed=state.failures,active=[dict(job_id=k,phase=v['phase'],slot=v['slot'],gpu_slot=v['gpu_slot']) for k,v in state.running.items()],
                    admission_closed_reason=state.closed_reason,level=state.level,limits=list(state.limits),automatic_retry=False,
                    plan=plan_ref,resource_session=session,updated_utc=utc()))
            result=run_scheduler(plan,ProcessDriver(root,plan,graph,session),
                stop_requested=lambda:any(path(root,n).exists() for n in (base+'/STOP',BASE+'/STOP','runtime/framework/multitask_r10_STOP')),on_update=update)
            fresh(path(root,base+'/resource_metrics.json'),dict(schema=SCHEMA+'_resource_metrics',**result))
            # Global lease remains held until every real child was reaped.
            terminal=dict(schema=SCHEMA,complete=result['complete'],process=process,plan=plan_ref,resource_session=session,
                expected_cases=plan['expected_cases'],attempted=result['attempted'],completed=len(result['records']),infrastructure_failed=len(result['infrastructure_failures']),
                remaining_job_ids=result['remaining_job_ids'],active=None,automatic_retry=False,drain_reason=result['drain_reason'],
                actual_tool_exit_observed_by_owner=False,finished_utc=utc(),**stamp(plan['graph'],graph))
        fresh(path(root,base+'/terminal.json'),terminal)
        atomic(path(root,base+'/status.json'),dict(schema=SCHEMA,stage='complete' if result['complete'] else 'drained',
            completed=result['records'],failed=result['infrastructure_failures'],active=None,plan=plan_ref,
            attempted=result['attempted'],automatic_retry=False,resource_session=session,expected_cases=plan['expected_cases'],drain_reason=result['drain_reason']))
    return dict(complete=result['complete'],completed=len(result['records']),terminal=pointer(root,base+'/terminal.json'),
        qualification_granted=False,actual_tool_exit_observed_by_owner=False)


def _closed_owner(root,plan_ref,actual_exec_completion_ref):
    plan,pr,graph=load_plan(root,plan_ref,execute=False);base=plan['owner_runtime']
    launch,lr=read(root,pointer(root,base+'/launch.json'));status,_=read(root,pointer(root,base+'/status.json'))
    terminal,tr=read(root,pointer(root,base+'/terminal.json'))
    actual,ar=read(root,exact_ref(root,actual_exec_completion_ref,base+'/exec_completion.json'))
    process=readiness._process(launch.get('process'));count=plan['expected_cases']
    readiness._stamp(launch,plan['graph'],graph);readiness._stamp(terminal,plan['graph'],graph)
    require(same(readiness.pointer(launch['source']),pointer(root,SOURCE)) and same(readiness.pointer(launch['plan']),pr)
        and status.get('stage')=='complete' and status.get('failed')==[] and status.get('active') is None
        and type(status.get('attempted')) is int and status['attempted']==count and len(status.get('completed',[]))==count
        and terminal.get('complete') is True and type(terminal.get('completed')) is int and terminal['completed']==count
        and type(terminal.get('attempted')) is int and terminal['attempted']==count
        and type(terminal.get('infrastructure_failed')) is int and terminal['infrastructure_failed']==0
        and terminal.get('automatic_retry') is False and terminal.get('active') is None
        and terminal.get('drain_reason') is None and terminal.get('remaining_job_ids')==[]
        and same(terminal.get('process'),process),'all actual suffix/evaluation attempts must finish cleanly')
    require(actual.get('actual_tool_exit_observed') is True and actual.get('natural_completion') is True
        and type(actual.get('returncode')) is int and actual['returncode']==0
        and same(actual.get('process'),process) and same(readiness.pointer(actual['plan']),pr)
        and same(readiness.pointer(actual['terminal']),tr) and readiness._not_live(process),
        'Root actual exit0/dead owner proof required; owner cannot issue it')
    records=status['completed']
    require([r.get('job_id') for r in records]==[j['job_id'] for j in plan['jobs']],'all results must be restored to original plan order')
    for job,row in zip(plan['jobs'],records):require(same(completed_case(root,plan,graph,job,row['CPU_check'],row['native']),row),'post-exit case closure differs')
    return plan,pr,graph,records,dict(launch=lr,terminal=tr,exec_completion=ar)


def assemble_cohort(root,plan_ref,actual_exec_completion_ref):
    plan,pr,graph,records,owner=_closed_owner(root,plan_ref,actual_exec_completion_ref)
    require(plan['split']=='train','mixed TRAIN lineage cannot relabel VAL')
    value=dict(schema='r10_mixed_control_TRAIN192_lineage_v1182',complete=True,**stamp(plan['graph'],graph),
        predecessor_exit=plan['predecessor']['exit'],suffix_plan=pr,suffix_owner=owner,records=records,
        preserved_prefix_count=plan['preserved_prefix_count'],case_graphs_relabelled=False,mixed_control_lineage=True,
        automatic_retry=False,source_outcome_filtering=False,qualification_granted=False)
    name=plan['owner_runtime']+'/lineage.json';fresh(path(root,name),value);return pointer(root,name)


def assemble_validation_cohort(root,plan_ref,actual_exec_completion_ref):
    plan,pr,graph,records,owner=_closed_owner(root,plan_ref,actual_exec_completion_ref)
    require(plan['split']=='validation','VAL945 collector cannot relabel TRAIN')
    value=dict(schema=VAL_COHORT_SCHEMA,complete=True,split='validation',expected_cases=945,records=records,plan=pr,
        validation_bundle=plan['validation_bundle'],owner=owner,automatic_retry=False,source_outcome_filtering=False,
        reused_cases=[],readiness_granted_by_owner=False,**stamp(plan['graph'],graph))
    name=plan['owner_runtime']+'/cohort.json';fresh(path(root,name),value);return pointer(root,name)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);mode=parser.add_mutually_exclusive_group()
    for flag in ('prepare','execute','assemble-cohort','assemble-validation-cohort'):mode.add_argument('--'+flag,action='store_true')
    for key in ('graph','split','namespace','review','predecessor-exit','superseded-CPU-exit','validation-bundle','plan','plan-sha256','actual-exec-completion'):parser.add_argument('--'+key)
    args=parser.parse_args(argv)
    if args.prepare:
        require(all(getattr(args,k) for k in ('graph','split','namespace','review','superseded_CPU_exit')),'explicit graph/split/namespace/review/superseded-CPU-exit required')
        value=prepare_parallel_plan(ROOT,args.graph,split=args.split,namespace=args.namespace,review_ref=args.review,
            superseded_CPU_exit_ref=args.superseded_CPU_exit,predecessor_exit_ref=args.predecessor_exit,bundle_ref=args.validation_bundle)
    elif args.execute:
        require(args.plan and args.plan_sha256,'explicit new plan and SHA required')
        value=execute_owner(ROOT,args.plan,args.plan_sha256)
    elif args.assemble_cohort or args.assemble_validation_cohort:
        require(args.plan and args.actual_exec_completion,'actual Root post-exit receipt required')
        value=(assemble_validation_cohort if args.assemble_validation_cohort else assemble_cohort)(ROOT,args.plan,args.actual_exec_completion)
    else:value=dict(enabled=False,Native_started=False,GPU_started=False,VAL_read=False,TEST_read=False)
    print(json.dumps(value,ensure_ascii=False),flush=True);return value


if __name__=='__main__':
    outcome=main()
    if '--execute' in sys.argv and outcome.get('complete') is not True:raise SystemExit(1)
