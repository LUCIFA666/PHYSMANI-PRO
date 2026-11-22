"""Read-only fresh full TRAIN192 qualification under one new physical graph.

All192 are new executions. Earlier closed TRAIN and retired development cases
are retained solely as history and never count toward this qualification.
"""
from copy import deepcopy
import hashlib
import importlib
import math
import re
import errno
import os
from datetime import datetime
from pathlib import Path

from . import repaired_readiness_v1144 as base
from . import repaired_policy_graph_v1143 as graphs
from .repaired_readiness_v1144 import (require, same, pointer, read_json, digest,
    make_graph_witness, validate_sampling_case, _phase, _stamp, _key,
    _attestation, _motion, TASKS, METHODS, BLOCKS, HASHES)

SOURCE='our_mani/multitask/public_held_repaired_readiness_v1208.py'
READINESS_SCHEMA='r10_fresh_public_held_TRAIN192_readiness_v1208'
LINEAGE_SCHEMA='r10_fresh_public_held_TRAIN192_lineage_v1208'
SUFFIX_PLAN_SCHEMA='r10_public_held_fixed_TRAIN192_plan_v1207'
OWNER_SOURCE='scripts/run_public_held_repaired_owner_v1207_r10.py'
INHERITED_ENTRY=dict(root='project',path='scripts/run_repaired_policy_case_v1167_r10.py',
    sha256='0c01bf0091bc8c4728581b36ff8630a961f7cdb9b0b889b79c31ceb3b9a25175')
JOB_FIELDS=('job_id','task','method','raw_episode_id','split','configuration','out','runtime','source_row_sha256')
GUARD_SOURCE='our_mani/multitask/public_held_repaired_resource_guard_v1213.py'
PARENT_GUARD_SOURCE='our_mani/multitask/post_completed_TEST_resource_guard_v1106.py'


def _job(value):
    require(type(value) is dict and all(k in value for k in JOB_FIELDS),'complete nine-field resource job required')
    return {k:value[k] for k in JOB_FIELDS}


def _typed(ref, prefix, basename=None):
    ref=pointer(ref);p=Path(ref['path'])
    require(ref['root']=='project' and ref['path'].startswith(prefix)
        and not p.is_absolute() and '..' not in p.parts and p.as_posix()==ref['path']
        and (basename is None or p.name==basename), 'typed project qualification metadata required')
    return ref


def validate_previous_closed_TRAIN(value):
    """Validate disclosure shape only; the owner reader performs full reaudit."""
    require(type(value) is dict and value.get('schema')=='r10_closed_previous_TRAIN192_history_v1207'
        and type(value.get('completed')) is int and value['completed']==192
        and type(value.get('source_count')) is int and value['source_count']==64
        and value.get('all_child_processes_dead') is True
        and value.get('actual_owner_natural_exit0') is True
        and value.get('original_results_retained') is True
        and value.get('historical_results_counted_in_accepted192') is False
        and value.get('prior_TRAIN_results_reused_for_new_qualification') is False
        and value.get('prior_TRAIN_source_reexecution_declared_for_new_physical_qualification') is True
        and value.get('historical_Native_reexecution_disclosed') is True
        and type(value.get('repeated_development_Native_cases')) is int
        and value['repeated_development_Native_cases']==2
        and value.get('retired_development_records_not_counted_in_accepted192') is True,
        'complete dead previous TRAIN history and honest new-physical reexecution disclosure required')
    _typed(value.get('readiness'),'runtime/framework/','readiness.json')
    _typed(value.get('graph'),'artifacts/framework/multitask_r10/','policy_graph.json')
    _typed(value.get('lineage'),'runtime/framework/','lineage.json')
    _typed(value.get('owner_plan'),'runtime/framework/','plan.json')
    require(type(value.get('owner')) is dict and set(value['owner'])=={'launch','terminal','exec_completion'},
        'actual previous owner closure pointers required')
    for key,ref in value['owner'].items():_typed(ref,'runtime/framework/',key+'.json')
    from .parallel_repaired_readiness_v1198 import validate_retired_development_batch
    validate_retired_development_batch(value.get('retired_development_batch'))
    return value


def _previous_closed_TRAIN(root,graph,proof_ref):
    module=importlib.import_module('scripts.run_public_held_repaired_owner_v1207_r10')
    source=dict(root='project',path=OWNER_SOURCE,sha256=hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest())
    require(Path(module.__file__).resolve()==root/OWNER_SOURCE
        and same(pointer(graph['control']['components']['train_owner']),source),
        'actual frozen1207 historical closure reader required')
    return module.inspect_previous_closed_TRAIN(root,proof_ref,new_graph=graph)


def _adoption_fields():
    return dict(TRAIN_development_results_used_for_repair_adoption=True,
        fixed_TRAIN192_sources_selected_using_outcomes=False,full192_uniform_requalification=True,
        new_independent_dataset_claimed=False)


def _development_diagnostic(root,graph,reference):
    module=importlib.import_module('scripts.run_public_held_repaired_owner_v1207_r10')
    return module.inspect_development_diagnostic(root,reference,new_graph=graph)


def _fresh_fields():
    return dict(fresh_TRAIN192=True,preserved_prefix_count=0,preserved_parallel_count=0,
        control_segment_count=1,historical_results_counted_in_accepted192=False)


def _require_fresh(value):
    require(all(type(value.get(k)) is type(v) and value[k]==v for k,v in _fresh_fields().items())
        and value.get('mixed_control_lineage') is False and value.get('case_graphs_relabelled') is False
        and value.get('reused_cases')==[] and value.get('automatic_retry') is False
        and value.get('source_outcome_filtering') is False
        and all(value.get(k) is v for k,v in _adoption_fields().items()),
        'single fresh graph with zero inherited scientific results required')


def _suffix_owner(root, refs, plan_ref, graph_ref, graph, count):
    require(type(refs) is dict and set(refs)=={'launch','terminal','exec_completion'},
        'actual complete new suffix owner evidence required')
    directory=str(Path(plan_ref['path']).parent)+'/'
    values={}
    for key,ref in refs.items():
        ref=_typed(ref,directory)
        require(Path(ref['path']).parent==Path(directory), 'suffix owner evidence escaped owner namespace')
        if key in ('launch','terminal'):
            require(ref['path']==directory+key+'.json','actual owner metadata filename required')
        values[key]=read_json(root,ref)[0]
    launch,terminal,exit=values['launch'],values['terminal'],values['exec_completion']
    proc=base._process(launch.get('process'))
    require(same(pointer(launch.get('source')),pointer(graph['control']['components']['train_owner']))
        and same(pointer(launch.get('plan')),plan_ref), 'actual suffix owner/source/plan mismatch')
    _stamp(launch,graph_ref,graph);_stamp(terminal,graph_ref,graph)
    require(terminal.get('complete') is True and type(terminal.get('completed')) is int
        and terminal['completed']==count and type(terminal.get('attempted')) is int
        and terminal['attempted']==count and type(terminal.get('infrastructure_failed')) is int
        and terminal['infrastructure_failed']==0
        and terminal.get('active') is None and terminal.get('automatic_retry') is False
        and terminal.get('drain_reason') is None and terminal.get('remaining_job_ids')==[]
        and same(base._process(terminal.get('process')),proc)
        and same(pointer(terminal.get('plan')),plan_ref), 'suffix must naturally complete every remaining attempt')
    require(same(base._process(exit.get('process')),proc)
        and exit.get('actual_tool_exit_observed') is True and exit.get('natural_completion') is True
        and type(exit.get('returncode')) is int and exit['returncode']==0
        and same(pointer(exit.get('plan')),plan_ref)
        and same(pointer(exit.get('terminal')),pointer(refs['terminal'])) and base._not_live(proc),
        'Root actual suffix natural exit0/dead master required')
    return values


def validate_broker_session(session, session_ref, graph_ref, graph):
    """Audit saved parent-owned lock evidence; never reopen a dead parent's FDs."""
    from .post_completed_TEST_resource_guard_v1106 import LOCKS
    require(session.get('schema')=='r10_parent_owned_parallel_resource_session_v1213'
        and same(pointer(session.get('graph')),pointer(graph_ref))
        and session.get('test_only') is False, 'actual graph-bound1213 resource session required')
    guard=pointer(session.get('guard'));parent_guard=pointer(session.get('parent_guard'))
    require(guard['root']=='project' and guard['path']==GUARD_SOURCE
        and same(guard,pointer(graph['control']['components']['resource_guard']))
        and parent_guard['root']=='project' and parent_guard['path']==PARENT_GUARD_SOURCE
        and any(same(parent_guard,pointer(ref)) for ref in graph['control']['sources_map'].values()),
        'actual graph-frozen1213 guard and inherited1106 source required')
    locks=session.get('parent_global_locks')
    require(type(locks) is list and [row.get('path') for row in locks if type(row) is dict]==list(LOCKS),
        'complete original1106 parent FD/kernel FLOCK evidence required')
    require(all(type(row) is dict and all(type(row.get(key)) is int and row[key]>=0
        for key in ('device','inode','fd')) for row in locks), 'strict saved parent lock identities required')
    boot=session.get('boot_id');broker=session.get('parent_broker',{})
    require(type(boot) is str and re.fullmatch(r'[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}',boot)
        and type(broker) is dict
        and broker.get('schema')=='r10_live_parent_lock_broker_v1213'
        and broker.get('transport')=='AF_UNIX_abstract'
        and type(broker.get('address')) is str and re.fullmatch(r'our-mani-1213-[0-9a-f]{48}',broker['address'])
        and same(base._process(broker.get('owner')),base._process(session.get('owner')))
        and broker.get('boot_id')==boot
        and type(broker.get('uid')) is int and broker['uid']>=0
        and type(broker.get('gid')) is int and broker['gid']>=0
        and broker.get('verification')=='parent_namespace_FD_and_kernel_FLOCK'
        and broker.get('inherited_lock_descriptors_passed_to_children') is False,
        'saved authenticated parent-namespace broker declaration required')
    require(pointer(session_ref)['path']==session.get('runtime','')+'/resources/session.json',
        'broker belongs to this exact master session')
    return broker


def validate_broker_verification(proof, session, session_ref, claim_ref, claim):
    """Validate an offline broker response, without claiming direct child FD access."""
    broker=session['parent_broker']
    require(type(proof) is dict and proof.get('schema')=='r10_parent_lock_broker_verification_v1213'
        and same(pointer(proof.get('session')),session_ref)
        and same(pointer(proof.get('claim')),claim_ref)
        and same(base._process(proof.get('owner')),base._process(session['owner']))
        and same(base._process(proof.get('client')),base._process(claim['process']))
        and proof.get('boot_id')==session['boot_id']
        and type(proof.get('nonce')) is str and re.fullmatch(r'[0-9a-f]{64}',proof['nonce'])
        and proof.get('parent_FD_verified') is True and proof.get('kernel_FLOCK_verified') is True
        and proof.get('client_direct_parent_FD_read') is False
        and proof.get('server_authenticated_by_SO_PEERCRED') is True
        and proof.get('client_authenticated_by_SO_PEERCRED') is True
        and same(proof.get('lock_proof'),session['parent_global_locks'])
        and same(pointer(proof.get('guard')),pointer(session['guard']))
        and same(pointer(proof.get('parent_guard')),pointer(session['parent_guard'])),
        'actual authenticated parent FD/kernel proof bound to this exact child claim required')
    for field in ('job_id','phase','slot','gpu_slot'):
        expected=claim['job']['job_id'] if field=='job_id' else claim.get(field)
        require(type(proof.get(field)) is type(expected) and proof.get(field)==expected,
            'broker proof changed actual job/phase/slot: '+field)
    for who in ('server','client'):
        for key in ('uid','gid'):
            value=proof.get(who+'_'+key)
            require(type(value) is int and value==broker[key], 'broker peer credential mismatch')
    validate_per_FD_kernel_evidence(proof,session)
    try:
        checked=datetime.fromisoformat(proof.get('checked_utc'))
    except (TypeError,ValueError):
        require(False,'actual broker verification UTC time required')
    require(checked.tzinfo is not None and checked.utcoffset() is not None,
        'broker verification time must include its UTC offset')
    return proof


def validate_per_FD_kernel_evidence(proof,session):
    """Check recorded per-descriptor kernel locks and separate conflict probes."""
    require(proof.get('parent_kernel_verification')=='per_FD_fdinfo',
        'parent kernel evidence must use the actual per-FD fdinfo interface')
    locks=session['parent_global_locks'];rows=proof.get('per_FD_lock_evidence')
    conflicts=proof.get('independent_conflict_evidence')
    require(type(rows) is list and type(conflicts) is list and len(rows)==len(conflicts)==len(locks),
        'complete ordered per-FD and independent conflict evidence required')
    for lock,row,conflict in zip(locks,rows,conflicts):
        require(type(row) is dict and type(conflict) is dict,
            'per-FD and conflict evidence rows must be objects')
        for field in ('path','device','inode','fd'):
            require(type(row.get(field)) is type(lock[field]) and row.get(field)==lock[field],
                'per-FD evidence changed the actual parent descriptor identity')
        require(type(row.get('owner_pid')) is int and row['owner_pid']==session['owner']['pid']
            and row.get('lock_type')=='FLOCK' and row.get('advisory')=='ADVISORY'
            and row.get('mode')=='WRITE' and type(row.get('start')) is int and row['start']==0
            and row.get('end')=='EOF' and row.get('FD_stat_before_after_verified') is True,
            'actual parent per-FD exclusive full-file lock and before/after stat proof required')
        raw=row.get('raw')
        match=re.fullmatch(r'lock:\s+\d+:\s+FLOCK\s+ADVISORY\s+WRITE\s+(\d+)\s+([0-9a-fA-F]+):([0-9a-fA-F]+):(\d+)\s+0\s+EOF\s*',raw) if type(raw) is str else None
        require(match is not None and int(match[1])==session['owner']['pid']
            and (int(match[2],16),int(match[3],16),int(match[4]))==
                (os.major(lock['device']),os.minor(lock['device']),lock['inode']),
            'raw actual fdinfo row must bind owner, file identity and exclusive range')
        for field in ('path','device','inode'):
            require(type(conflict.get(field)) is type(lock[field]) and conflict.get(field)==lock[field],
                'independent conflict proof changed the actual lock file')
        require(type(conflict.get('conflict_errno')) is int
            and conflict['conflict_errno'] in (errno.EAGAIN,errno.EWOULDBLOCK)
            and conflict.get('confirmed') is True,
            'separate child open/flock EX|NB must actually fail with EAGAIN/EWOULDBLOCK')


def validate_master_session(root, refs, plan_ref, graph_ref, graph, jobs):
    """All case leases must descend from this one actual master/plan/session."""
    plan_ref=_typed(plan_ref,'runtime/framework/public_held_repaired_owner_v1207/','plan.json')
    plan,_=read_json(root,plan_ref)
    directory=str(Path(plan_ref['path']).parent)+'/'
    launch,_=read_json(root,_typed(refs['launch'],directory,'launch.json'))
    terminal,_=read_json(root,_typed(refs['terminal'],directory,'terminal.json'))
    runtime=str(Path(plan_ref['path']).parent)
    session_ref=_typed(launch.get('resource_session'),runtime+'/resources/','session.json')
    require(session_ref['path']==runtime+'/resources/session.json'
        and same(pointer(terminal.get('resource_session')),session_ref),
        'one exact master session is required')
    session,_=read_json(root,session_ref)
    _stamp(session,graph_ref,graph)
    validate_broker_session(session,session_ref,graph_ref,graph)
    require(same(pointer(session.get('plan')),plan_ref)
        and same(base._process(session.get('owner')),base._process(launch.get('process')))
        and session.get('runtime')==runtime and session.get('split')==plan.get('split')
        and same(session.get('jobs'),[_job(j) for j in jobs])
        and same(session.get('profile'),plan.get('profile')),
        'session owner/plan/exact source jobs/profile mismatch')
    from .public_held_repaired_resource_guard_v1213 import _profile
    _profile(session['profile'])
    return session_ref


def validate_parallel_phase(root, sidecar, launch, job, graph_ref, graph, *, check):
    """Offline receipts only; do not require an already-exited session to be live."""
    graph_ref=pointer(graph_ref)
    phase='CPU_check' if check else 'native'
    receipt=sidecar.get('resource_scope_receipt',{})
    require(receipt.get('schema')=='r10_parallel_child_resource_receipt_v1213'
        and receipt.get('purpose')=='child_slot' and receipt.get('phase')==phase
        and receipt.get('Native_authorized') is (not check)
        and receipt.get('test_only') is False and receipt.get('body_completed') is True
        and receipt.get('restored') is True and receipt.get('leases_removed') is True
        and receipt.get('errors')==[] and receipt.get('parent_global_locks_verified') is True
        and receipt.get('child_empty_inventory_claimed') is False
        and same(base._process(receipt.get('process')),base._process(sidecar.get('process'))),
        'real completed1213 child resource receipt required')
    _stamp(receipt,graph_ref,graph)
    session_ref=_typed(sidecar.get('resource_session'),'runtime/framework/public_held_repaired_owner_v1207/','session.json')
    session,_=read_json(root,session_ref)
    validate_broker_session(session,session_ref,graph_ref,graph)
    require(session_ref['path']==session.get('runtime','')+'/resources/session.json',
        'exact session runtime path required')
    expected_claim=session.get('runtime','')+'/resources/claims/'+str(job.get('job_id'))+'_'+phase+'/claim.json'
    claim_pointer=_typed(receipt.get('claim'),'runtime/framework/public_held_repaired_owner_v1207/','claim.json')
    require(claim_pointer['path']==expected_claim,'exact same-session job-phase claim required')
    claim,claim_ref=read_json(root,claim_pointer)
    require(session.get('schema')=='r10_parent_owned_parallel_resource_session_v1213'
        and same(pointer(receipt.get('session')),session_ref)
        and same(pointer(claim.get('session')),session_ref)
        and claim.get('schema')=='r10_once_only_parallel_resource_claim_v1213'
        and claim.get('phase')==phase and claim.get('automatic_retry') is False
        and claim.get('test_only') is False
        and same(base._process(claim.get('process')),base._process(sidecar['process']))
        and same(base._process(session.get('owner')),base._process(receipt.get('actual_parent'))),
        'actual session/claim/parent/process mismatch')
    _stamp(session,graph_ref,graph)
    require(same(pointer(session.get('guard')),pointer(graph['control']['components']['resource_guard']))
        and session.get('test_only') is False, 'graph-frozen real resource session required')
    require(same(pointer(session.get('graph')),graph_ref), 'resource session graph mismatch')
    # Empty inventory was checked by the parent BEFORE it launched this cohort.
    # A parallel child must never assert that its live siblings do not exist.
    inventory=session.get('parent_resource_proof',{}).get('host_inventory',{})
    require(inventory.get('host_visibility_available') is True
        and inventory.get('ancestor_chain_complete') is True
        and inventory.get('same_uid_inventory_incomplete') is False
        and inventory.get('live_project_native_processes')==[]
        and receipt.get('parent1106_empty_inventory_checked_at_session_start_only') is True,
        'actual parent initial inventory is required, not an empty child host')
    initial=sidecar.get('resource_proof',{})
    for field in ('schema','purpose','phase','slot','gpu_slot','job_id','process','session','claim',
            'parent_global_locks_verified','child_empty_inventory_claimed'):
        require(same(initial.get(field),receipt.get(field)), 'initial/final resource evidence mismatch: '+field)
    admission=validate_broker_verification(receipt.get('parent_broker_admission'),session,session_ref,claim_ref,claim)
    final=validate_broker_verification(receipt.get('parent_broker_final'),session,session_ref,claim_ref,claim)
    require(same(initial.get('parent_broker_admission'),admission)
        and 'parent_broker_final' not in initial and admission['nonce']!=final['nonce'],
        'distinct actual admission/final broker exchanges required; no final proof at admission')
    require(same(claim.get('job'),_job(job)) and any(same(item,_job(job)) for item in session.get('jobs',[])),
        'exact same-session nine-field resource job required')
    slot=receipt.get('slot');gpu=receipt.get('gpu_slot')
    bound=session['profile']['max_CPU_preflight_slots' if check else 'max_native_slots']
    require(type(bound) is int and type(slot) is int and 0<=slot<bound
        and type(claim.get('slot')) is int and claim['slot']==slot
        and receipt.get('job_id')==job.get('job_id') and sidecar.get('resource_job_id')==job.get('job_id')
        and type(sidecar.get('resource_slot')) is int and sidecar.get('resource_slot')==slot
        and type(sidecar.get('resource_gpu_slot')) is type(gpu) and sidecar.get('resource_gpu_slot')==gpu,
        'actual bounded case slot mismatch')
    if not check and job['method']=='physmani':
        require(type(gpu) is int and 0<=gpu<session['profile']['max_GPU_slots'], 'bounded actual GPU slot required')
    else:
        require(gpu is None, 'CPU/geometry must not claim a GPU slot')
    require(type(claim.get('gpu_slot')) is type(gpu) and claim.get('gpu_slot')==gpu
        and sidecar.get('private_process_paths_restored') is True,
        'actual GPU claim/private path restoration mismatch')
    terminal_name=str(Path(claim_ref['path']).with_name('terminal.json'))
    terminal_ref=dict(root='project',path=terminal_name,
        sha256=hashlib.sha256((Path(root)/terminal_name).read_bytes()).hexdigest())
    terminal,_=read_json(root,terminal_ref)
    require(same(terminal,receipt), 'slot terminal differs from final sidecar receipt')
    argv=launch.get('argv')
    for flag,value in (('--resource-session',session_ref['path']),('--resource-session-sha256',session_ref['sha256']),
            ('--resource-job-id',job['job_id']),('--resource-slot',str(slot))):
        base._option(argv,flag,value)
    if gpu is not None:base._option(argv,'--resource-gpu-slot',str(gpu))
    else:require('--resource-gpu-slot' not in argv,'unassigned GPU argv forbidden')
    validate_CPU_affinity(root,sidecar,launch,session,graph,slot=slot,check=check)
    return dict(session=session_ref,claim=claim_ref,terminal=terminal_ref,phase=phase,slot=slot,gpu_slot=gpu)


def validate_CPU_affinity(root,sidecar,launch,session,graph,*,slot,check):
    """Only stored OS receipts/metadata; never query or change an OS mask here."""
    plan,_=read_json(root,_typed(session.get('plan'),'runtime/framework/public_held_repaired_owner_v1207/','plan.json'))
    affinity=plan.get('CPU_affinity',{})
    ref=_typed(graph['control']['static_inputs']['parallel_CPU_affinity_profile'],
        'artifacts/framework/multitask_r10/','parallel_CPU_affinity_profile.json')
    data,_=read_json(root,ref)
    require(data.get('schema')=='r10_parallel_repaired_CPU_affinity_v1212'
        and same(pointer(affinity.get('profile_ref')),ref)
        and same({k:v for k,v in affinity.items() if k!='profile_ref'},data)
        and data.get('physical_core_count')==16 and data.get('max_native_slots')==12
        and data.get('bitwise_equivalence_claimed') is False,
        'exact graph-bound16-core CPU affinity declaration required')
    groups=data.get('core_groups');native=data.get('native_slot_cpus');pool=data.get('control_and_CPU_preflight_cpus')
    allowed=data.get('allowed_logical_cpus')
    require(type(groups) is list and len(groups)==16 and type(native) is list and len(native)==12
        and type(pool) is list and len(pool)==8 and pool==sorted(set(pool))
        and all(type(v) is int and v>=0 for v in pool)
        and all(type(c) is list and len(c)==2 and c==sorted(set(c))
            and all(type(v) is int and v>=0 for v in c) for c in native)
        and len({v for c in native for v in c})==24
        and not (set(pool)&{v for c in native for v in c})
        and allowed==sorted(set(pool)|{v for c in native for v in c})
        and all(c in [g.get('cpus') for g in groups] for c in native)
        and sum(set(g.get('cpus',[]))<=set(pool) for g in groups)==4,
        '12 separate whole Native cores and disjoint4-core CPU pool required')
    binary=pointer(data.get('taskset_binary'))
    require(binary['root']=='host_tools' and binary['path']=='taskset'
        and same(binary,pointer(graph['control']['static_inputs']['parallel_taskset_binary']))
        and graph['roots'].get('host_tools')==data.get('taskset_root')
        and data.get('taskset_runs_outside820_as_control_tool') is True,
        'actual taskset control binary must be graph-pinned')
    tool=Path(data['taskset_root'])/'taskset'
    require(tool.resolve()==tool and tool.is_file()
        and hashlib.sha256(tool.read_bytes()).hexdigest()==binary['sha256'],'taskset binary changed')
    cpus=pool if check else native[slot]
    argv=launch.get('argv')
    require(type(argv) is list and argv[:3]==[str(tool),'--cpu-list',','.join(map(str,cpus))]
        and launch.get('expected_CPU_cpus')==cpus,
        'graph-pinned taskset and exact phase mask must be the outermost argv')
    proof=sidecar.get('CPU_affinity_receipt',{})
    for label in ('before','after'):
        snap=proof.get(label,{});threads=snap.get('thread_cpus');vanished=snap.get('vanished_thread_ids')
        require(same(pointer(snap.get('profile')),ref) and snap.get('expected_cpus')==cpus
            and snap.get('actual_main_cpus')==cpus and snap.get('all_threads_checked') is True
            and type(snap.get('additional_native_calls')) is int and snap['additional_native_calls']==0
            and type(threads) is list and bool(threads)
            and all(type(t) is dict and type(t.get('tid')) is int and t['tid']>0 and t.get('cpus')==cpus for t in threads)
            and len({t['tid'] for t in threads})==len(threads) and sidecar['process']['pid'] in {t['tid'] for t in threads}
            and type(vanished) is list and all(type(t) is int and t>0 for t in vanished),
            'actual main/all-thread mask must match before and after execution')
    return ref


def _ref_equal(left,right):
    return same(pointer(left),pointer(right))


def _models(root,case,cpu,route,graph):
    """Check the actual CPU loader and recorded model branch, without loading it."""
    proof = cpu['execution'].get('source_and_model_preflight',{})
    config, _ = read_json(root,route['configuration'])
    require(proof.get('GPU_started') is False
        and proof.get('split_source_input_lock') is None,
        'actual TRAIN CPU loader must remain GPU-free and cannot carry VAL source admission')
    if route['method'] == 'physmani':
        checkpoint = proof.get('original_author_checkpoint',{})
        expected = Path(config.get('baseline_checkpoint',''))
        candidates = [dict(path=str(Path(graph['roots'][p['root']])/p['path']),sha256=p['sha256'])
            for p in route['checkpoints'].values()]
        require(expected.is_absolute() and expected.name=='epoch_99999.pth' and config.get('baseline_reference_filter')=='author_direct_setpoint' and checkpoint == dict(path=str(expected),sha256=checkpoint.get('sha256'))
            and checkpoint in candidates and proof.get('original_neural_policy_and_online_world_model') is True
            and case.get('trained_component_identity') is None
            and case.get('actor_metrics',{}).get('original_baseline_algorithm_unchanged') is True,
            'actual released PhysMani checkpoint and unchanged baseline branch required')
        return
    require(_ref_equal(case.get('shared_context_checkpoint'),graph['physical']['shared_context_checkpoint'])
        and _ref_equal(config.get('context_checkpoint'),graph['physical']['shared_context_checkpoint']),
        'actual shared context checkpoint differs')
    component = case.get('trained_component_identity',{})
    if route['task'] in BLOCKS:
        binding = proof.get('actual_shared16_block_adapter',{})
        require(binding.get('task') == route['task']
            and binding.get('checkpoint_sha256') == graph['physical']['shared_context_checkpoint']['sha256']
            and same(component.get('actual_shared16_block_adapter'),binding),
            'actual shared block adapter/checkpoint branch differs')
    else:
        identity = proof.get('component_identity',{})
        require(proof.get('actual_model_loader_validation') is True and proof.get('native_started') is False
            and proof.get('old_TRAIN_actor_constructor_executed') is False
            and identity.get('schema') == 'r10_direct_public_RML_component_identity_v643'
            and identity.get('task') == route['task'] and identity.get('method') == route['method'],
            'actual643 CPU model loader identity required')
        unhashed = dict(identity); unhashed.pop('sha256',None)
        expected = {k:v for k,v in config.get('learned_artifacts',{}).get(route['task'],{}).items()
            if k in ('geometry','interaction','rigid_layout','colour')}
        require(all(expected.get(k) is not None for k in ('geometry','interaction','rigid_layout'))
            and same(identity.get('learned_operation_artifacts'),expected)
            and same(component.get('learned_operation_artifacts'),expected)
            and identity.get('sha256') == base.digest(unhashed)
            and component.get('sha256') == identity.get('sha256')
            and same(proof.get('component_domain'),case['component_domain'])
            and _ref_equal(identity.get('shared_context_checkpoint'),graph['physical']['shared_context_checkpoint'])
            and _ref_equal(component.get('shared_context_checkpoint'),graph['physical']['shared_context_checkpoint'])
            and all(any(_ref_equal(ref,p) for p in route['models'].values()) for ref in expected.values() if ref is not None),
            'actual configured operation artifacts must match frozen graph models')


def _audit_records(root, graph_ref, graph, jobs, records, sources, seen, *, parallel=False, master_session=None):
    profile,protocol=base._profile(graph,root)
    routes = {(r['task'], r['method']):r for r in graph['routes']}; record_map = {_key(r):r for r in records}
    result = []
    resource_inputs=[]
    processes, case_paths, evidence_paths, all_processes = seen
    for job in jobs:
        key = _key(job); row = record_map[key]; route = routes[(key[0], key[2])]
        require(type(row.get('attempt')) is int and row['attempt'] == 1, 'exactly one actual attempt per source required')
        for field, prefix in (('out','artifacts/framework/multitask_r10/'), ('runtime','runtime/framework/')):
            name = job.get(field)
            require(type(name) is str and name.startswith(prefix) and Path(name).as_posix() == name
                and '..' not in Path(name).parts, 'explicit fresh project case namespaces required')
        source = sources[key[:2]]
        witness = make_graph_witness(graph_ref, graph, route, source, split='train')
        # Classify every addressed JSON role before opening any supplied bytes.
        for field,name in (('case','case.json'),('owned_hooks','owned_hook_receipt.json')):
            ref=_typed(row.get(field),job['out']+'/',name)
            require(ref['path']==job['out']+'/'+name,'case evidence escaped its exact planned path')
        for phase in ('CPU_check','native'):
            refs=row.get(phase)
            require(type(refs) is dict and set(refs)=={'launch','terminal','sidecar'},'exact phase evidence required')
            for kind in ('launch','terminal'):
                _typed(refs[kind],'runtime/framework/',kind+'.json')
                if parallel:
                    runtime=str(Path(master_session['path']).parent.parent)
                    name=runtime+'/children/'+key[0]+'/'+key[1]+'/'+key[2]+'/'+phase+'/'+kind+'.json'
                    require(pointer(refs[kind])['path']==name,'new phase owner path mismatch')
            sidecar_ref=_typed(refs['sidecar'],job['runtime']+'/'+phase+'/','worker_sidecar.json')
            require(sidecar_ref['path']==job['runtime']+'/'+phase+'/worker_sidecar.json','exact sidecar path required')
        cpu, cpu_refs, cpu_process = _phase(root, row.get('CPU_check'), job, route, graph_ref, graph, witness, check=True)
        native, native_refs, native_process = _phase(root, row.get('native'), job, route, graph_ref, graph, witness, check=False)
        current=[]
        for sidecar,refs,check in ((cpu,cpu_refs,True),(native,native_refs,False)):
            launch,_=read_json(root,refs['launch'])
            for process in (base._process(launch['process']),base._process(sidecar['process'])):
                token=(process['pid'],process['start_ticks'])
                require(token not in all_processes,'actual outer/inner process reused between cases/phases')
                all_processes.add(token);current.append(token)
                require(base._not_live(process),'completed case process still live')
            if parallel:
                resource=validate_parallel_phase(root,sidecar,launch,job,graph_ref,graph,check=check)
                require(master_session is not None and same(resource['session'],master_session),
                    'every parallel case must use the actual master session')
                resource_inputs.extend(resource[k] for k in ('session','claim','terminal'))
        require(cpu_process not in processes and native_process not in processes and cpu_process != native_process,
            'a fresh actual CPU/native worker process is required for every case')
        processes.update((cpu_process, native_process))
        for phase_refs in (cpu_refs, native_refs):
            for reference in phase_refs.values():
                require(reference['path'] not in evidence_paths, 'per-case execution evidence cannot be reused')
                evidence_paths.add(reference['path'])
        case, case_ref = read_json(root, row.get('case'))
        require(case_ref['path'] == job['out']+'/case.json' and case_ref['path'] not in case_paths, 'fresh planned native case required')
        case_paths.add(case_ref['path'])
        require(case.get('task') == key[0] and case.get('method') == key[2] and case.get('split') == 'train'
            and case.get('source_raw_episode_id') == key[1] and type(case.get('process_id')) is int and case['process_id'] == native_process[0]
            and case.get('automatic_retry') is False and case.get('TEST_used_for_fitting') is False
            and same(case.get('graph_witness'), witness), 'actual case source/worker/graph branch differs')
        identity = case.get('identity', {}); no_sha = dict(identity); no_sha.pop('sha256', None)
        require(identity.get('version') == 'r10_fresh_case_identity_v1'
            and identity.get('split') == 'train' and identity.get('task') == key[0] and identity.get('method') == key[2]
            and identity.get('source_raw_episode_id') == key[1] and identity.get('parent_raw_episode_id') == key[1]
            and identity.get('raw_episode_id') == key[1]+'__policy_aug_r0'
            and type(identity.get('replica')) is int and identity['replica'] == 0
            and identity.get('exploration_seed') is None and identity.get('demo_only') is True
            and identity.get('development_demo_only') is True and identity.get('fitting_source') is False
            and identity.get('formal_validation_metric') is False
            and case.get('raw_episode_id') == identity['raw_episode_id'] and case.get('parent_raw_episode_id') == key[1]
            and case.get('case_isolation') == 'fresh_simulator_process_per_episode'
            and case.get('formal_test_used') is False and case.get('original_source_motion_replayed') is False
            and case.get('recorded_initial_conditions',{}).get('metric_only') is True
            and case.get('recorded_initial_conditions',{}).get('policy_use_allowed') is False,
            'actual unaugmented first-parent original TRAIN identity and metric-only reset required')
        execution_protocol = case.get('execution_protocol', {})
        if parallel:
            require(same(pointer(execution_protocol.get('inherited742_preflight_entry')),INHERITED_ENTRY)
                and execution_protocol.get('actual_worker_verified_by_new_control_graph') is True
                and execution_protocol.get('preflight_entry_is_actual_outer_worker') is False,
                'actual1206 worker and inherited1167 preflight must not be conflated')
        require(type(execution_protocol.get('schema')) is str and bool(execution_protocol['schema'])
            and identity.get('execution_protocol_schema') == execution_protocol['schema']
            and identity.get('execution_protocol_sha256') == digest(execution_protocol)
            and same(cpu['execution'].get('execution_protocol'),execution_protocol)
            and same(native['execution'].get('execution_protocol'),execution_protocol),
            'CPU/native/case actual new execution protocol identity differs')
        for field,expected_value in (('actual_worker',route['train']['worker']),
                ('actual_preparer',route['train']['preparer']),('inherited_consumer',route['train']['consumer']),
                ('source_overlay_lock',graph['physical']['source_overlay_lock'])):
            require(same(pointer(execution_protocol.get(field)),pointer(expected_value)), 'new execution source differs: '+field)
        require(execution_protocol.get('selected_physical_factory') == route['physical_factory']
            and same(execution_protocol.get('factory_binding'),witness['factory_binding'])
            and execution_protocol.get('execution_branch') == witness['model_branch']
            and execution_protocol.get('actual_split') == 'train'
            and execution_protocol.get('component_purpose') == 'train_qualification'
            and execution_protocol.get('automatic_retry') is False
            and execution_protocol.get('original_745_execution_acceptance_granted') is False
            and execution_protocol.get('qualification_granted_by_worker') is False
            and execution_protocol.get('formal_score_contribution') is False
            and type(execution_protocol.get('additional_native_getter_observe_predict_or_success_calls')) is int
            and execution_protocol['additional_native_getter_observe_predict_or_success_calls'] == 0
            and same(execution_protocol.get('unchanged_action_outcome_sampling'),protocol),
            'new declared factory/purpose cannot claim legacy acceptance, change sampling or add Native observations')
        _stamp(execution_protocol, graph_ref, graph)
        require(same(pointer(case.get('source_overlay_lock')),graph['physical']['source_overlay_lock'])
            and case.get('original_745_execution_acceptance_granted') is False
            and case.get('qualification_granted_by_worker') is False
            and case.get('formal_score_contribution') is False,
            'honest new case namespace/protocol cannot impersonate previous acceptance')
        require(identity.get('sha256') == digest(no_sha)
            and identity.get('config_sha256') == route['configuration']['sha256']
            and identity.get('frozen_inputs_sha256') == route['frozen_inputs']['sha256']
            and identity.get('source_row_sha256') == digest(source)
            and identity.get('graph_sha256') == graph_ref['sha256']
            and all(identity.get(k) == graph[k] for k in HASHES)
            and same(cpu['execution'].get('identity'), identity) and same(native['execution'].get('identity'), identity),
            'CPU/native/case actual graph/config/source identity differs')
        domain = case.get('component_domain', {})
        require(domain.get('schema') == 'r10_source_bound_RML_component_purpose_v640'
            and domain.get('purpose') == 'train_qualification' and domain.get('split') == 'train'
            and domain.get('task') == key[0] and domain.get('method') == key[2] and domain.get('raw_episode_id') == key[1],
            'genuine fixed-TRAIN component purpose required')
        require(domain.get('VAL_TEST_relabelled_as_TRAIN') is False and domain.get('policy_sha256') is None
            and same(cpu['execution'].get('component_domain'), domain)
            and same(native['execution'].get('component_domain'), domain), 'CPU/native genuine640 TRAIN domain differs')
        require(same(pointer(native['execution'].get('case')), case_ref)
            and native['execution'].get('native_success') is case.get('official_task_success')
            and native['execution'].get('native_failure') is case.get('native_terminal_failure'), 'native sidecar/outcome differs')
        require(same(pointer(protocol.get('runtime_consumer')), pointer(route['train']['consumer'])), 'actual original consumer differs from741 protocol')
        _models(root,case,cpu,route,graph)
        validate_sampling_case(case, protocol, block=key[0] in BLOCKS, case_version=route['train']['case_schema'])
        _attestation(case, profile); motion_pass = _motion(case)
        hooks, hook_ref = read_json(root, row.get('owned_hooks'))
        require(hook_ref['path'] == job['out']+'/owned_hook_receipt.json' and hooks.get('actual_split') == 'train'
            and type(hooks.get('reset')) is list and type(hooks.get('paths')) is list
            and all(v.get('complete') is True for v in hooks['reset'])
            and all(v.get('close_complete') is True for v in hooks['paths']), 'actual reset/RML cleanup evidence required')
        if key[0] not in BLOCKS:
            require(len(hooks['reset']) == 1, 'one original captured core reset required')
        if parallel:
            require(same(hooks.get('graph_witness'),witness)
                and same(hooks.get('execution_protocol'),execution_protocol),
                'actual new hooks must retain the same graph/protocol identity')
        result.append(dict(task=key[0], raw=key[1], method=key[2], status='completed',
            native_success=case['official_task_success'], native_terminal_failure=case['native_terminal_failure'],
            motion_pass=motion_pass, counted_ticks=case['counted_clock']['ticks'], case=case_ref,
            owned_hooks=hook_ref, CPU_check=cpu_refs, native=native_refs, old_protocol_case_reused=False,
            case_graph=graph_ref,control_segment='fresh_full_parallel' if parallel else 'nonparallel_unqualified'))
    for reference in resource_inputs:read_json(root,reference)
    return result


def audit_train192(root,graph_ref,lineage_ref):
    root=Path(root).resolve()
    graph_ref=_typed(graph_ref,'artifacts/framework/multitask_r10/','policy_graph.json')
    lineage_ref=_typed(lineage_ref,'runtime/framework/public_held_repaired_owner_v1207/train/','lineage.json')
    graph=graphs.read_policy_graph(root,graph_ref)
    actual=dict(root='project',path=SOURCE,sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    require(same(pointer(graph['control']['components']['train_auditor']),actual)
        and graph['control']['contracts']['readiness_schema']==READINESS_SCHEMA,
        'actual1208 full new physical graph auditor must precede execution')
    lineage,lineage_ref=read_json(root,lineage_ref)
    require(lineage.get('schema')==LINEAGE_SCHEMA and lineage.get('complete') is True,
        'complete new single-graph TRAIN lineage required')
    _require_fresh(lineage);_stamp(lineage,graph_ref,graph)
    history=validate_previous_closed_TRAIN(lineage.get('previous_closed_TRAIN'))
    require(same(history,_previous_closed_TRAIN(root,graph,history['readiness'])),
        'closed previous TRAIN history must survive full independent reaudit')
    development=_development_diagnostic(root,graph,lineage.get('development_diagnostic'))
    plan_ref=_typed(lineage.get('plan'),'runtime/framework/public_held_repaired_owner_v1207/train/','plan.json')
    plan,plan_ref=read_json(root,plan_ref);_stamp(plan,graph_ref,graph);_require_fresh(plan)
    require(plan.get('schema')==SUFFIX_PLAN_SCHEMA and plan.get('split')=='train'
        and type(plan.get('expected_cases')) is int and plan['expected_cases']==192
        and type(plan.get('global_expected_cases')) is int and plan['global_expected_cases']==192
        and type(plan.get('new_suffix_count')) is int and plan['new_suffix_count']==192
        and plan.get('predecessor') is None and plan.get('validation_bundle') is None
        and same(plan.get('previous_closed_TRAIN'),history)
        and same(pointer(plan.get('development_diagnostic')),development),
        'all192 must be newly executed; prior74/35/83 are history only')
    manifest,manifest_ref=read_json(root,graph['train_source_manifest']);sources=base._sources(manifest)
    require(same(plan.get('sources'),list(sources.values())), 'exact64 original first4 TRAIN rows required')
    jobs=plan.get('jobs');records=lineage.get('records')
    expected=[(t,raw,m) for t,raw in sources for m in METHODS]
    require(type(jobs) is list and type(records) is list and len(jobs)==len(records)==192
        and [_key(v) for v in jobs]==expected and [_key(v) for v in records]==expected
        and [v.get('job_id') for v in jobs]==[f'j{i:04d}' for i in range(192)]
        and [v.get('job_id') for v in records]==[v['job_id'] for v in jobs],
        'complete original-order j0000..j0191 grid, exactly one new record each required')
    namespace=plan.get('namespace')
    require(type(namespace) is str and namespace and all(c.isalnum() or c in '_.-' for c in namespace)
        and plan.get('owner_runtime')==f'runtime/framework/public_held_repaired_owner_v1207/train/{namespace}'
        and plan_ref['path']==plan['owner_runtime']+'/plan.json'
        and lineage_ref['path']==plan['owner_runtime']+'/lineage.json', 'exact fresh owner namespace required')
    routes={(r['task'],r['method']):r for r in graph['routes']}
    for job,key in zip(jobs,expected):
        task,raw,method=key;suffix=f'cohorts/{namespace}/{task}/{raw}/{method}'
        require(set(job)==set(JOB_FIELDS) and job['split']=='train'
            and same(pointer(job['configuration']),pointer(routes[(task,method)]['configuration']))
            and job['source_row_sha256']==digest(sources[(task,raw)])
            and job['out']=='artifacts/framework/multitask_r10/public_held_repaired_policy_v1206/train/development_demo_only/'+suffix
            and job['runtime']=='runtime/framework/public_held_repaired_policy_v1206/train/'+suffix,
            'current1206 source/config/fresh path declaration required')
    values=_suffix_owner(root,lineage.get('owner'),plan_ref,graph_ref,graph,192)
    master_session=validate_master_session(root,lineage['owner'],plan_ref,graph_ref,graph,jobs)
    status_ref=dict(root='project',path=plan['owner_runtime']+'/status.json')
    status_ref['sha256']=hashlib.sha256((root/status_ref['path']).read_bytes()).hexdigest()
    status,_=read_json(root,status_ref)
    require(status.get('stage')=='complete' and status.get('active') is None and status.get('failed')==[]
        and type(status.get('attempted')) is int and status['attempted']==192
        and same(status.get('completed'),records),'actual complete unfiltered owner status required')
    seen=(set(),set(),set(),set())
    result=_audit_records(root,graph_ref,graph,jobs,records,sources,seen,parallel=True,master_session=master_session)
    require(len(result)==192 and len({j['out'] for j in jobs})==len({j['runtime'] for j in jobs})==192,
        'all192 new outcomes and distinct case namespaces must survive')
    require(same(graphs.read_policy_graph(root,graph_ref),graph),'source graph changed during audit')
    refs=[lineage_ref,plan_ref,status_ref,*lineage['owner'].values()]
    for row in records:
        refs.extend((row['case'],row['owned_hooks']))
        for phase in ('CPU_check','native'):refs.extend(row[phase].values())
    for reference in refs:read_json(root,reference)
    result.sort(key=lambda x:(x['task'],x['raw'],x['method']))
    return dict(schema=READINESS_SCHEMA,graph=graph_ref,lineage=lineage_ref,
        **{key:graph[key] for key in HASHES},TRAIN_manifest=manifest_ref,
        complete=True,completed=192,method_case_count=192,source_count=64,task_ready_count=16,
        previous_closed_TRAIN=history,development_diagnostic=development,new_suffix_count=192,qualification_kind='fresh_single_graph',
        mixed_control_lineage=False,case_graphs_relabelled=False,reused_cases=[],
        old1144_full_same_graph_readiness_claimed=False,new_graph_executed_all192=True,cases=result,
        native_success_is_not_an_admission_threshold=True,motion_is_diagnostic_only=True,
        all_failures_retained=True,automatic_retry=False,source_outcome_filtering=False,
        VAL_read=False,TEST_read=False,Native_started_by_auditor=False,GPU_started_by_auditor=False,
        **_adoption_fields(),**_fresh_fields())


def audit_mixed_train192(root,graph_ref,lineage_ref):
    """Compatibility API only; this version forbids any mixed scientific cohort."""
    return audit_train192(root,graph_ref,lineage_ref)


def require_readiness_proof(root,graph_ref,proof_ref):
    saved,_=read_json(root,_typed(proof_ref,'runtime/framework/','readiness.json'))
    require(saved.get('schema')==READINESS_SCHEMA,'new full single-graph readiness schema required')
    actual=audit_train192(root,graph_ref,saved.get('lineage'))
    require(same(saved,actual),'saved readiness differs from full192 plus closed-history reaudit')
    return actual
