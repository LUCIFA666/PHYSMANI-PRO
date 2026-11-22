"""Read-only mixed-control TRAIN192 qualification; original identities survive.

A controlled old serial prefix plus a new parallel suffix is one explicitly
segmented qualification, never a relabelled1144 same-graph fresh192 proof.
"""
from copy import deepcopy
import hashlib
import importlib
import math
from pathlib import Path

from . import repaired_readiness_v1144 as base
from . import repaired_policy_graph_v1143 as graphs
from .repaired_readiness_v1144 import (require, same, pointer, read_json, digest,
    make_graph_witness, validate_sampling_case, _phase, _stamp, _key,
    _attestation, _motion, TASKS, METHODS, BLOCKS, HASHES)

SOURCE='our_mani/multitask/parallel_repaired_readiness_v1174.py'
READINESS_SCHEMA='r10_mixed_control_TRAIN192_readiness_v1174'
LINEAGE_SCHEMA='r10_mixed_control_TRAIN192_lineage_v1174'
SUFFIX_PLAN_SCHEMA='r10_parallel_fixed_TRAIN_suffix_plan_v1173'
OWNER_SOURCE='scripts/run_parallel_repaired_owner_v1173_r10.py'
INHERITED_ENTRY=dict(root='project',path='scripts/run_repaired_policy_case_v1167_r10.py',
    sha256='0c01bf0091bc8c4728581b36ff8630a961f7cdb9b0b889b79c31ceb3b9a25175')
JOB_FIELDS=('job_id','task','method','raw_episode_id','split','configuration','out','runtime','source_row_sha256')


def _job(value):
    require(type(value) is dict and all(k in value for k in JOB_FIELDS),'complete nine-field resource job required')
    return {k:value[k] for k in JOB_FIELDS}


def _typed(ref, prefix, basename=None):
    ref=pointer(ref);p=Path(ref['path'])
    require(ref['root']=='project' and ref['path'].startswith(prefix)
        and not p.is_absolute() and '..' not in p.parts and p.as_posix()==ref['path']
        and (basename is None or p.name==basename), 'typed project qualification metadata required')
    return ref


def compare_physical(old, new):
    added=set(new['roots'])-set(old['roots'])
    require(added<= {'host_tools'},'only the explicit control-only host_tools root may be added')
    if added:
        binary=pointer(new['control']['static_inputs'].get('parallel_taskset_binary'))
        require(binary['root']=='host_tools' and binary['path']=='taskset',
            'new host_tools root must name the graph-pinned control taskset')
    require(same(old['physical'],new['physical'])
        and all(new['roots'].get(k)==v for k,v in old['roots'].items())
        and old['tasks']==new['tasks'] and old['methods']==new['methods']
        and same(pointer(old['train_source_manifest']),pointer(new['train_source_manifest'])),
        'entire physical graph, roots and original TRAIN manifest must remain unchanged')
    def physical_rows(g):
        return [{k:v for k,v in row.items() if k not in ('train','evaluation')} for row in g['routes']]
    require(same(physical_rows(old),physical_rows(new))
        and old['physical_policy_sha256']==new['physical_policy_sha256'],
        'all48 physical/config/model/checkpoint/frozen-input rows and physical fingerprint must match')
    def used_roots(value):
        if type(value) is dict:
            found={value.get('root','project')} if {'path','sha256'}<=set(value) else set()
            return found|set().union(*(used_roots(v) for v in value.values()))
        if type(value) is list:return set().union(*(used_roots(v) for v in value))
        return set()
    require(not (added & used_roots([new['physical'],*physical_rows(new)])),
        'additional host control roots cannot alter physical dependencies')
    return new['physical_policy_sha256']


def _predecessor(root, graph, exit_ref):
    module=importlib.import_module('scripts.run_parallel_repaired_owner_v1173_r10')
    source=dict(root='project',path=OWNER_SOURCE,sha256=hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest())
    require(Path(module.__file__).resolve()==root/OWNER_SOURCE
        and same(pointer(graph['control']['components']['train_owner']),source),
        'actual new1173 prefix inspector must be frozen in the control graph')
    return module.inspect_predecessor(root,exit_ref)


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
        and same(base._process(terminal.get('process')),proc)
        and same(pointer(terminal.get('plan')),plan_ref), 'suffix must naturally complete every remaining attempt')
    require(same(base._process(exit.get('process')),proc)
        and exit.get('actual_tool_exit_observed') is True and exit.get('natural_completion') is True
        and type(exit.get('returncode')) is int and exit['returncode']==0
        and same(pointer(exit.get('plan')),plan_ref)
        and same(pointer(exit.get('terminal')),pointer(refs['terminal'])) and base._not_live(proc),
        'Root actual suffix natural exit0/dead master required')
    return values


def validate_master_session(root, refs, plan_ref, graph_ref, graph, jobs):
    """All case leases must descend from this one actual master/plan/session."""
    plan_ref=_typed(plan_ref,'runtime/framework/parallel_repaired_owner_v1173/','plan.json')
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
    require(same(pointer(session.get('plan')),plan_ref)
        and same(base._process(session.get('owner')),base._process(launch.get('process')))
        and session.get('runtime')==runtime and session.get('split')==plan.get('split')
        and same(session.get('jobs'),[_job(j) for j in jobs])
        and same(session.get('profile'),plan.get('profile')),
        'session owner/plan/exact source jobs/profile mismatch')
    from .parallel_repaired_resource_guard_v1171 import _profile
    _profile(session['profile'])
    return session_ref


def validate_parallel_phase(root, sidecar, launch, job, graph_ref, graph, *, check):
    """Offline receipts only; do not require an already-exited session to be live."""
    phase='CPU_check' if check else 'native'
    receipt=sidecar.get('resource_scope_receipt',{})
    require(receipt.get('schema')=='r10_parallel_child_resource_receipt_v1171'
        and receipt.get('purpose')=='child_slot' and receipt.get('phase')==phase
        and receipt.get('Native_authorized') is (not check)
        and receipt.get('test_only') is False and receipt.get('body_completed') is True
        and receipt.get('restored') is True and receipt.get('leases_removed') is True
        and receipt.get('errors')==[] and receipt.get('parent_global_locks_verified') is True
        and receipt.get('child_empty_inventory_claimed') is False
        and same(base._process(receipt.get('process')),base._process(sidecar.get('process'))),
        'real completed1171 child resource receipt required')
    _stamp(receipt,graph_ref,graph)
    session_ref=_typed(sidecar.get('resource_session'),'runtime/framework/parallel_repaired_owner_v1173/','session.json')
    session,_=read_json(root,session_ref)
    require(session_ref['path']==session.get('runtime','')+'/resources/session.json',
        'exact session runtime path required')
    expected_claim=session.get('runtime','')+'/resources/claims/'+str(job.get('job_id'))+'_'+phase+'/claim.json'
    claim_pointer=_typed(receipt.get('claim'),'runtime/framework/parallel_repaired_owner_v1173/','claim.json')
    require(claim_pointer['path']==expected_claim,'exact same-session job-phase claim required')
    claim,claim_ref=read_json(root,claim_pointer)
    require(session.get('schema')=='r10_parent_owned_parallel_resource_session_v1171'
        and same(pointer(receipt.get('session')),session_ref)
        and same(pointer(claim.get('session')),session_ref)
        and claim.get('schema')=='r10_once_only_parallel_resource_claim_v1171'
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
    require(same(claim.get('job'),_job(job)) and any(same(item,_job(job)) for item in session.get('jobs',[])),
        'exact same-session nine-field resource job required')
    slot=receipt.get('slot');gpu=receipt.get('gpu_slot')
    bound=session['profile']['max_CPU_preflight_slots' if check else 'max_native_slots']
    require(type(bound) is int and type(slot) is int and 0<=slot<bound
        and type(claim.get('slot')) is int and claim['slot']==slot
        and receipt.get('job_id')==job.get('job_id') and sidecar.get('resource_job_id')==job.get('job_id')
        and sidecar.get('resource_slot')==slot and sidecar.get('resource_gpu_slot')==gpu,
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
    plan,_=read_json(root,_typed(session.get('plan'),'runtime/framework/parallel_repaired_owner_v1173/','plan.json'))
    affinity=plan.get('CPU_affinity',{})
    ref=_typed(graph['control']['static_inputs']['parallel_CPU_affinity_profile'],
        'artifacts/framework/multitask_r10/','parallel_CPU_affinity_profile.json')
    data,_=read_json(root,ref)
    require(data.get('schema')=='r10_parallel_repaired_CPU_affinity_v1178'
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
                'actual1172 worker and inherited1167 preflight must not be conflated')
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
            case_graph=graph_ref,control_segment='new_parallel_suffix' if parallel else 'old_serial_prefix'))
    for reference in resource_inputs:read_json(root,reference)
    return result


def audit_mixed_train192(root, graph_ref, lineage_ref):
    root=Path(root).resolve()
    graph_ref=_typed(graph_ref,'artifacts/framework/multitask_r10/','policy_graph.json')
    lineage_ref=_typed(lineage_ref,'runtime/framework/','lineage.json')
    graph=graphs.read_policy_graph(root,graph_ref)
    actual=dict(root='project',path=SOURCE,sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    require(same(pointer(graph['control']['components']['train_auditor']),actual)
        and graph['control']['contracts']['readiness_schema']==READINESS_SCHEMA,
        'actual1174 mixed-control auditor must precede execution')
    lineage,lineage_ref=read_json(root,lineage_ref)
    require(lineage.get('schema')==LINEAGE_SCHEMA and lineage.get('complete') is True
        and lineage.get('mixed_control_lineage') is True
        and lineage.get('automatic_retry') is False and lineage.get('source_outcome_filtering') is False
        and lineage.get('case_graphs_relabelled') is False, 'honest complete mixed-control lineage required')
    _stamp(lineage,graph_ref,graph)
    predecessor=_predecessor(root,graph,lineage.get('predecessor_exit'))
    old_graph_ref=pointer(predecessor['old_graph'])
    old=graphs.read_policy_graph(root,old_graph_ref);compare_physical(old,graph)
    original,original_ref=read_json(root,predecessor['original_plan'])
    base._stamp(original,old_graph_ref,old)
    jobs=original.get('jobs');prefix=predecessor['records'];k=predecessor['completed_prefix_count']
    require(type(k) is int and 0<=k<192 and type(jobs) is list and len(jobs)==192
        and len(prefix)==k and [_key(v) for v in prefix]==[_key(v) for v in jobs[:k]],
        'actual closed prefix must be the original192 plan prefix')
    manifest,manifest_ref=read_json(root,graph['train_source_manifest']);sources=base._sources(manifest)
    expected={(t,raw,m) for t,raw in sources for m in METHODS}
    require(len({_key(v) for v in jobs})==192 and {_key(v) for v in jobs}==expected,
        'the exact original first4 by16 by3 grid must remain')
    plan,plan_ref=read_json(root,_typed(lineage.get('suffix_plan'),'runtime/framework/parallel_repaired_owner_v1173/train/','plan.json'))
    require(plan.get('schema')==SUFFIX_PLAN_SCHEMA and plan.get('split')=='train'
        and type(plan.get('global_expected_cases')) is int and plan['global_expected_cases']==192
        and type(plan.get('preserved_prefix_count')) is int and plan['preserved_prefix_count']==k
        and type(plan.get('expected_cases')) is int and plan['expected_cases']==192-k
        and same(plan.get('predecessor'),predecessor) and plan.get('automatic_retry') is False
        and plan.get('source_outcome_filtering') is False, 'mechanical fixed suffix plan required')
    _stamp(plan,graph_ref,graph)
    suffix_jobs=plan.get('jobs');suffix=lineage.get('records')
    require(type(suffix_jobs) is list and len(suffix_jobs)==192-k and type(suffix) is list and len(suffix)==192-k
        and [_key(v) for v in suffix_jobs]==[_key(v) for v in jobs[k:]]
        and {_key(v) for v in suffix}=={_key(v) for v in suffix_jobs},
        'new controller may execute only original unexecuted suffix, once')
    for previous,current in zip(jobs[k:],suffix_jobs):
        require(current.get('source_row_sha256')==previous.get('source_row_sha256')
            and same(pointer(current.get('configuration')),pointer(previous.get('configuration'))),
            'suffix source/config changed')
        for field,prefix_name in (('out','artifacts/framework/multitask_r10/parallel_repaired_policy_v1172/train/'),
                ('runtime','runtime/framework/parallel_repaired_policy_v1172/train/')):
            name=current.get(field)
            require(type(name) is str and name.startswith(prefix_name)
                and Path(name).as_posix()==name and '..' not in Path(name).parts,
                'new suffix evidence must remain in actual1172 TRAIN namespaces')
    _suffix_owner(root,lineage.get('suffix_owner'),plan_ref,graph_ref,graph,192-k)
    master_session=validate_master_session(root,lineage['suffix_owner'],plan_ref,graph_ref,graph,suffix_jobs)
    all_jobs=jobs[:k]+suffix_jobs
    require(len({j['out'] for j in all_jobs})==len({j['runtime'] for j in all_jobs})==192,
        'all old/new case outputs and runtimes must be distinct')
    seen=(set(),set(),set(),set())
    result=_audit_records(root,old_graph_ref,old,jobs[:k],prefix,sources,seen)
    result+=_audit_records(root,graph_ref,graph,suffix_jobs,suffix,sources,seen,parallel=True,master_session=master_session)
    require(len(result)==192, 'all original192 outcomes must survive')
    require(same(graphs.read_policy_graph(root,graph_ref),graph)
        and same(graphs.read_policy_graph(root,old_graph_ref),old), 'source graph changed during audit')
    # Recheck every immutable reference after all per-case reads.
    refs=[lineage_ref,original_ref,plan_ref,lineage['predecessor_exit'],*lineage['suffix_owner'].values()]
    refs.extend(predecessor[k] for k in ('launch','status','failure','exit') if k in predecessor)
    for row in [*prefix,*suffix]:
        refs.extend((row['case'],row['owned_hooks']))
        for phase in ('CPU_check','native'):refs.extend(row[phase].values())
    for reference in refs:read_json(root,reference)
    result.sort(key=lambda x:(x['task'],x['raw'],x['method']))
    return dict(schema=READINESS_SCHEMA,graph=graph_ref,lineage=lineage_ref,
        **{key:graph[key] for key in HASHES},old_graph=old_graph_ref,original_plan=original_ref,
        TRAIN_manifest=manifest_ref,complete=True,completed=192,method_case_count=192,
        source_count=64,task_ready_count=16,preserved_prefix_count=k,new_suffix_count=192-k,
        qualification_kind='mixed_control_lineage',mixed_control_lineage=True,
        case_graphs_relabelled=False,old1144_full_same_graph_readiness_claimed=False,
        new_graph_executed_all192=False,same_physical_policy=True,cases=result,
        native_success_is_not_an_admission_threshold=True,motion_is_diagnostic_only=True,
        all_failures_retained=True,automatic_retry=False,source_outcome_filtering=False,
        VAL_read=False,TEST_read=False,Native_started_by_auditor=False,GPU_started_by_auditor=False)


def require_readiness_proof(root,graph_ref,proof_ref):
    saved,_=read_json(root,_typed(proof_ref,'runtime/framework/','readiness.json'))
    require(saved.get('schema')==READINESS_SCHEMA,'new mixed-control readiness schema required')
    actual=audit_mixed_train192(root,graph_ref,saved.get('lineage'))
    require(same(saved,actual),'mixed-control saved readiness differs from complete reaudit')
    return actual
