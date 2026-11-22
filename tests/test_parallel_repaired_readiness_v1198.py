"""Temporary synthetic mixed192 receipts; no real data or simulator.

The prefix inspector is tested by1197 separately. Full per-case auditing here is
real: its192 JSON receipts, sampling, identities and resources are not mocked.
"""
from copy import deepcopy
import hashlib
import os
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from our_mani.multitask import parallel_repaired_readiness_v1198 as subject
from our_mani.multitask import repaired_policy_graph_v1143 as graphs
from tests.test_repaired_readiness_v1144 import FreshFixture,write
from tests.test_repaired_policy_graph_v1143 import put
from tests.test_parallel_repaired_readiness_v1190 import MixedFixture as OriginalParallelFixture
from our_mani.multitask import parallel_repaired_readiness_v1190 as middle_audit


def make_retired_development_fixture(root):
    """Synthetic failed batch catalog; its successes never enter accepted192."""
    base='runtime/framework/parallel_repaired_owner_v1181/train/synthetic_retired'
    refs={key:write(root,base+'/'+key+'.json',dict(synthetic=True,role=key))
        for key in ('plan','launch','status','terminal','graph')}
    refs['exit']=write(root,base+'/actual_failure_exit_v1186.json',dict(actual_tool_exit_observed=True,returncode=1))
    records=[dict(job_id=job,case=write(root,base+'/'+job+'_case.json',dict(native_success=True)),
        owned_hooks=write(root,base+'/'+job+'_hooks.json',{}),native_success=True,accepted_into_new_cohort=False)
        for job in ('j0075','j0076')]
    return dict(refs,phases=[dict(synthetic_phase=i) for i in range(6)],completed_records=[deepcopy(records[1])],
        Native_development_records=records,attempted_CPU_cases=4,attempted_Native_cases=2,
        completed_accepted_cases=1,infrastructure_failed=3,all_child_processes_dead=True,
        original_results_retained=True,accepted_into_new_cohort=False,
        source_outcomes_used_for_rerun_or_version_selection=False,historical_Native_reexecution_disclosed=True,
        repeated_development_Native_cases=2,scientific_qualification_granted=False)


def retired_disclosure(value):
    return dict(retired_development_batch=deepcopy(value),historical_Native_reexecution_disclosed=True,
        repeated_development_Native_cases=2,retired_development_records_not_counted_in_accepted192=True)


def make_handoff_fixture(root):
    base='runtime/framework/parallel_repaired_owner_v1189/train/synthetic_handoff'
    refs={key:write(root,base+'/'+key+'.json',dict(synthetic=True,role=key))
        for key in ('plan','launch','status','terminal','session','old1194_graph')}
    refs['exit']=write(root,base+'/parallel_handoff_exit_v1197.json',dict(actual_tool_exit_observed=True,returncode=1))
    ids=['j0075','j0079','j0082']
    return dict(refs,completed_records=[dict(job_id=jid) for jid in ids],completed_job_ids=ids,preserved_count=3,
        CPU_only_records=[],all_child_processes_dead=True,Native_killed=False,case_graphs_relabelled=False,
        completed_records_reused_without_reexecution=True)


def handoff_disclosure(value):
    return dict(parallel_handoff=deepcopy(value),preserved_parallel_count=value['preserved_count'],
        control_segment_count=3,preserved_parallel_results_reused_without_reexecution=True)


def make_kernel_lock_evidence(session):
    """Temporary receipts with strict real kernel-row grammar; no actual locks."""
    return dict(parent_kernel_verification='per_FD_fdinfo',
        per_FD_lock_evidence=[dict(lock,owner_pid=session['owner']['pid'],lock_type='FLOCK',
            advisory='ADVISORY',mode='WRITE',start=0,end='EOF',FD_stat_before_after_verified=True,
            raw='lock:\t1: FLOCK ADVISORY WRITE '+str(session['owner']['pid'])+' '+
                format(os.major(lock['device']),'x')+':'+format(os.minor(lock['device']),'x')+':'+str(lock['inode'])+' 0 EOF')
            for lock in session['parent_global_locks']],
        independent_conflict_evidence=[dict(path=lock['path'],device=lock['device'],inode=lock['inode'],
            conflict_errno=11,confirmed=True) for lock in session['parent_global_locks']])


def add_broker_sources(root,spec):
    """Only temporary source declarations; no broker, native app or source data."""
    root=Path(root)
    for name,label in ((subject.GUARD_SOURCE,'resource_guard'),(subject.PARENT_GUARD_SOURCE,'broker_parent_guard')):
        file=root/name
        ref=put(root,name,file.read_bytes() if file.exists() else b'synthetic frozen broker source')
        spec['control']['sources_map'][label]=ref
        if label=='resource_guard':spec['control']['components']['resource_guard']=ref


def make_CPU_affinity_fixture(root,spec):
    """Synthetic control metadata/tool bytes only; never runs taskset/sysfs."""
    root=Path(root);tools=root/'synthetic_host_tools';tools.mkdir(exist_ok=True)
    binary=tools/'taskset';binary.write_bytes(b'synthetic control taskset; never executed')
    import hashlib
    tool=dict(root='host_tools',path='taskset',sha256=hashlib.sha256(binary.read_bytes()).hexdigest())
    data=dict(schema='r10_parallel_repaired_CPU_affinity_v1202',allowed_logical_cpus=list(range(32)),
        physical_core_count=16,core_groups=[dict(package=0,core=i,cpus=[i,i+16],L3_bytes=1024) for i in range(16)],
        native_slot_cpus=[[i,i+16] for i in range(12)],control_and_CPU_preflight_cpus=[12,13,14,15,28,29,30,31],
        max_native_slots=12,taskset_root=str(tools),taskset_binary=tool,
        taskset_runs_outside820_as_control_tool=True,bitwise_equivalence_claimed=False)
    ref=write(root,'artifacts/framework/multitask_r10/parallel_repaired_control_v1202_setup/parallel_CPU_affinity_profile.json',data)
    spec.setdefault('external_roots',{})['host_tools']=str(tools)
    spec['control']['static_inputs'].update(parallel_CPU_affinity_profile=ref,parallel_taskset_binary=tool)
    return dict(profile_ref=ref,tool_ref=tool,CPU_affinity=dict(data,profile_ref=ref))


def make_session(root,owner_runtime,graph_ref,graph,*,owner=None,plan_ref=None,jobs=None):
    from our_mani.multitask.post_completed_TEST_resource_guard_v1106 import LOCKS
    owner=owner or dict(pid=2147482000,start_ticks=80000)
    parent_guard=next(ref for ref in graph['control']['sources_map'].values() if ref['path']==subject.PARENT_GUARD_SOURCE)
    boot_id='00000000-0000-0000-0000-000000000001'
    session=dict(schema='r10_parent_owned_parallel_resource_session_v1195',owner=owner,
        graph=graph_ref,**{k:graph[k] for k in subject.HASHES},
        guard=graph['control']['components']['resource_guard'],parent_guard=parent_guard,test_only=False,
        boot_id=boot_id,parent_global_locks=[dict(path=name,device=1,inode=i+1,fd=i+3) for i,name in enumerate(LOCKS)],
        parent_broker=dict(schema='r10_live_parent_lock_broker_v1195',transport='AF_UNIX_abstract',
            address='our-mani-1195-'+'a'*48,owner=owner,uid=1000,gid=1000,boot_id=boot_id,
            verification='parent_namespace_FD_and_kernel_FLOCK',inherited_lock_descriptors_passed_to_children=False),
        profile=dict(max_native_slots=12,max_GPU_slots=5,max_CPU_preflight_slots=6,ready_limit=24),
        parent_resource_proof=dict(host_inventory=dict(host_visibility_available=True,
            ancestor_chain_complete=True,same_uid_inventory_incomplete=False,live_project_native_processes=[])),
        plan=plan_ref,jobs=[subject._job(j) for j in (jobs or [])],runtime=owner_runtime,
        split=(jobs[0]['split'] if jobs else 'train'))
    return write(root,owner_runtime+'/resources/session.json',session)


def add_resource_phase(root,owner_runtime,session_ref,graph_ref,graph,job,sidecar,launch,*,check):
    phase='CPU_check' if check else 'native';gpu=0 if not check and job['method']=='physmani' else None
    session=json.loads((Path(root)/session_ref['path']).read_text())
    claim=dict(schema='r10_once_only_parallel_resource_claim_v1195',session=session_ref,job=deepcopy(job),
        phase=phase,slot=0,gpu_slot=gpu,process=sidecar['process'],automatic_retry=False,test_only=False)
    directory=owner_runtime+'/resources/claims/'+job['job_id']+'_'+phase
    claim_ref=write(root,directory+'/claim.json',claim)
    receipt=dict(schema='r10_parallel_child_resource_receipt_v1195',purpose='child_slot',
        phase=phase,slot=0,gpu_slot=gpu,job_id=job['job_id'],process=sidecar['process'],session=session_ref,
        claim=claim_ref,actual_parent=session['owner'],parent_global_locks_verified=True,
        parent1106_empty_inventory_checked_at_session_start_only=True,child_empty_inventory_claimed=False,
        test_only=False,Native_authorized=not check,body_completed=True,restored=True,leases_removed=True,
        errors=[],graph=graph_ref,**{k:graph[k] for k in subject.HASHES})
    for label in ('admission','final'):
        receipt['parent_broker_'+label]=dict(schema='r10_parent_lock_broker_verification_v1195',session=session_ref,
            claim=claim_ref,owner=session['owner'],client=sidecar['process'],boot_id=session['boot_id'],
            nonce=hashlib.sha256((claim_ref['sha256']+label).encode()).hexdigest(),checked_utc='2026-10-24T00:00:00+00:00',
            parent_FD_verified=True,kernel_FLOCK_verified=True,client_direct_parent_FD_read=False,
            server_authenticated_by_SO_PEERCRED=True,client_authenticated_by_SO_PEERCRED=True,
            server_uid=1000,server_gid=1000,client_uid=1000,client_gid=1000,
            lock_proof=session['parent_global_locks'],guard=session['guard'],parent_guard=session['parent_guard'],
            job_id=job['job_id'],phase=phase,slot=0,gpu_slot=gpu)
        receipt['parent_broker_'+label].update(make_kernel_lock_evidence(session))
    write(root,directory+'/terminal.json',receipt)
    initial=deepcopy(receipt);initial.update(body_completed=False,restored=False);initial.pop('leases_removed')
    initial.pop('parent_broker_final')
    sidecar.update(resource_scope_receipt=receipt,resource_proof=initial,resource_session=session_ref,
        resource_job_id=job['job_id'],resource_slot=0,resource_gpu_slot=gpu,private_process_paths_restored=True)
    for flag,value in (('--resource-session',session_ref['path']),('--resource-session-sha256',session_ref['sha256']),
            ('--resource-job-id',job['job_id']),('--resource-slot','0')):launch['argv'] += [flag,value]
    if gpu is not None:launch['argv'] += ['--resource-gpu-slot','0']
    plan=json.loads((Path(root)/session['plan']['path']).read_text());affinity=plan['CPU_affinity']
    cpus=affinity['control_and_CPU_preflight_cpus'] if check else affinity['native_slot_cpus'][0]
    launch['argv']=[str(Path(affinity['taskset_root'])/'taskset'),'--cpu-list',','.join(map(str,cpus)),*launch['argv']]
    launch['expected_CPU_cpus']=cpus
    snap=dict(profile=affinity['profile_ref'],expected_cpus=cpus,actual_main_cpus=cpus,
        all_threads_checked=True,thread_cpus=[dict(tid=sidecar['process']['pid'],cpus=cpus)],
        vanished_thread_ids=[],additional_native_calls=0)
    sidecar['CPU_affinity_receipt']=dict(before=deepcopy(snap),after=deepcopy(snap))
    return sidecar,launch


class MixedFixture:
    def __init__(self,root,k=2):
        self.root=Path(root);middle=OriginalParallelFixture(root,k=k);old=middle.old
        self.old=old;self.middle=middle;self.k=k
        oldg=old.graph
        spec={n:deepcopy(middle.graph[n]) for n in ('tasks','methods','train_source_manifest','physical','control','routes')}
        spec.update(schema=graphs.SPEC_SCHEMA,external_roots={n:v for n,v in middle.graph['roots'].items() if n!='project'})
        spec['control']['sources_map']['middle_guard']=middle.graph['control']['components']['resource_guard']
        a=put(root,subject.SOURCE,Path(subject.__file__).read_bytes())
        spec['control']['components']['train_auditor']=a;spec['control']['sources_map']['auditor1198']=a
        owner=put(root,'scripts/run_parallel_repaired_owner_v1197_r10.py',b'synthetic owner')
        spec['control']['components']['train_owner']=owner;spec['control']['sources_map']['new_owner']=owner
        worker=put(root,'scripts/run_parallel_repaired_case_v1196_r10.py',b'synthetic actual new worker')
        preparer=put(root,'our_mani/multitask/parallel_repaired_case_v1196.py',b'synthetic new preparer')
        spec['control']['sources_map'].update(new_worker=worker,new_preparer=preparer)
        spec['control']['contracts']['readiness_schema']=subject.READINESS_SCHEMA
        for row in spec['routes']:
            for purpose in ('train','evaluation'):
                row[purpose].update(worker=worker,preparer=preparer,case_schema='new_actual_parallel_case')
        self.affinity=make_CPU_affinity_fixture(root,spec)['CPU_affinity']
        add_broker_sources(root,spec)
        self.graph_ref=graphs.write_policy_graph(root,spec,namespace='artifacts/framework/multitask_r10/third_graph')
        self.graph=graphs.read_policy_graph(root,self.graph_ref)
        self.stamp=dict(graph=self.graph_ref,**{key:self.graph[key] for key in subject.HASHES})
        self.owner_runtime='runtime/framework/parallel_repaired_owner_v1197/train/synthetic'
        manifest,_=subject.read_json(root,oldg['train_source_manifest']);sources=subject.base._sources(manifest)
        old_routes={(row['task'],row['method']):row for row in oldg['routes']}
        for job in old.jobs:
            key=subject._key(job)
            job.update(configuration=old_routes[(key[0],key[2])]['configuration'],
                source_row_sha256=subject.digest(sources[key[:2]]))
        old.plan_ref=write(root,old.plan_ref['path'],old.plan)
        self.predecessor=deepcopy(middle.predecessor);self.exit_ref=middle.exit_ref
        self.retired=deepcopy(middle.retired)
        middle_indices=[1,5,9];self.middle_records=[deepcopy(middle.records[i]) for i in middle_indices]
        self.middle_ids=[row['job_id'] for row in self.middle_records]
        middle_refs=middle.lineage['suffix_owner'];mt=json.loads((self.root/middle_refs['terminal']['path']).read_text())
        mt.update(complete=False,completed=len(self.middle_records),attempted=len(self.middle_records),drain_reason='STOP',
            remaining_job_ids=[job['job_id'] for job in middle.jobs if job['job_id'] not in self.middle_ids])
        middle_terminal=write(root,middle_refs['terminal']['path'],mt)
        middle_status=write(root,middle.owner_runtime+'/status.json',dict(stage='drained',active=None,failed=[],completed=self.middle_records))
        middle_exit=write(root,middle.owner_runtime+'/parallel_handoff_exit_v1197.json',dict(actual_tool_exit_observed=True,returncode=1,controlled_drain=True))
        self.handoff=dict(exit=middle_exit,old1194_graph=middle.graph_ref,plan=middle.plan_ref,launch=middle_refs['launch'],
            status=middle_status,terminal=middle_terminal,session=middle.session_ref,completed_records=self.middle_records,
            completed_job_ids=self.middle_ids,preserved_count=len(self.middle_ids),CPU_only_records=[],all_child_processes_dead=True,
            Native_killed=False,case_graphs_relabelled=False,completed_records_reused_without_reexecution=True)
        self.original_indices=[i for i in range(k,192) if f'j{i:04d}' not in self.middle_ids]
        self.jobs=[];self.records=[]
        manifest,_=subject.read_json(root,oldg['train_source_manifest']);sources=subject.base._sources(manifest)
        routes={(row['task'],row['method']):row for row in self.graph['routes']}
        for index in self.original_indices:
            previous=old.jobs[index];key=subject._key(previous);route=routes[(key[0],key[2])]
            job=deepcopy(previous);job.update(job_id=f'j{index:04d}',split='train',configuration=route['configuration'],
                source_row_sha256=subject.digest(sources[key[:2]]),
                out='artifacts/framework/multitask_r10/parallel_repaired_policy_v1196/train/development_demo_only/'+str(index),
                runtime='runtime/framework/parallel_repaired_policy_v1196/train/'+str(index))
            self.jobs.append(job)
        self.profile=dict(max_native_slots=12,max_GPU_slots=5,max_CPU_preflight_slots=6,ready_limit=24)
        self.plan=dict(schema=subject.SUFFIX_PLAN_SCHEMA,split='train',global_expected_cases=192,preserved_prefix_count=k,
            expected_cases=len(self.jobs),preserved_parallel_count=len(self.middle_ids),preserved_parallel_segment=deepcopy(self.handoff),
            automatic_retry=False,source_outcome_filtering=False,jobs=self.jobs,profile=self.profile,
            predecessor=deepcopy(self.predecessor),CPU_affinity=self.affinity,retired_partial_batch=deepcopy(self.retired),**self.stamp)
        self.plan_ref=write(root,self.owner_runtime+'/plan.json',self.plan)
        self.owner_process=dict(pid=2147482100,start_ticks=81000)
        self.session_ref=make_session(root,self.owner_runtime,self.graph_ref,self.graph,owner=self.owner_process,plan_ref=self.plan_ref,jobs=self.jobs)
        for order,index in enumerate(self.original_indices):
            previous=old.jobs[index];oldrow=old.records[index];key=subject._key(previous);route=routes[(key[0],key[2])]
            job=self.jobs[order]
            case=json.loads((self.root/oldrow['case']['path']).read_text())
            witness=subject.make_graph_witness(self.graph_ref,self.graph,route,sources[key[:2]],split='train')
            ep=case['execution_protocol'];ep.update(**self.stamp,actual_worker=worker,actual_preparer=preparer,
                inherited742_preflight_entry=subject.INHERITED_ENTRY,actual_worker_verified_by_new_control_graph=True,
                preflight_entry_is_actual_outer_worker=False)
            case.update(version=route['train']['case_schema'],graph_witness=witness)
            identity=case['identity'];identity.update(execution_protocol_sha256=subject.digest(ep))
            case['identity']=subject.base.bind_graph_identity(identity,witness)
            case_ref=write(root,job['out']+'/case.json',case)
            hooks=json.loads((self.root/oldrow['owned_hooks']['path']).read_text())
            hooks.update(graph_witness=witness,execution_protocol=ep)
            record=dict(task=key[0],raw_episode_id=key[1],method=key[2],attempt=1,job_id=job['job_id'],case=case_ref,
                owned_hooks=write(root,job['out']+'/owned_hook_receipt.json',hooks))
            for check in (True,False):
                phase='CPU_check' if check else 'native'
                launch=json.loads((self.root/oldrow[phase]['launch']['path']).read_text())
                launch.update(**self.stamp,actual_worker=worker,actual_preparer=preparer)
                for flag,value in (('--graph',self.graph_ref['path']),('--out',job['out']),('--runtime',job['runtime'])):
                    launch['argv'][launch['argv'].index(flag)+1]=value
                oldworker=oldg['routes'][0]['train']['worker']['path']
                launch['argv']=[worker['path'] if x==oldworker else x for x in launch['argv']]
                sidecar=json.loads((self.root/oldrow[phase]['sidecar']['path']).read_text())
                sidecar.update(**self.stamp,actual_worker=worker,actual_preparer=preparer,graph_witness=witness)
                sidecar['execution'].update(identity=case['identity'],graph_witness=witness,execution_protocol=ep,case=case_ref)
                sidecar,launch=add_resource_phase(root,self.owner_runtime,self.session_ref,self.graph_ref,self.graph,job,sidecar,launch,check=check)
                outer=self.owner_runtime+'/children/'+key[0]+'/'+key[1]+'/'+key[2]+'/'+phase
                record[phase]=dict(launch=write(root,outer+'/launch.json',launch),
                    terminal=write(root,outer+'/terminal.json',json.loads((self.root/oldrow[phase]['terminal']['path']).read_text())),
                    sidecar=write(root,job['runtime']+'/'+phase+'/worker_sidecar.json',sidecar))
            self.records.append(record)
        self.plan=dict(schema=subject.SUFFIX_PLAN_SCHEMA,split='train',global_expected_cases=192,preserved_prefix_count=k,
            expected_cases=len(self.jobs),preserved_parallel_count=len(self.middle_ids),preserved_parallel_segment=deepcopy(self.handoff),
            automatic_retry=False,source_outcome_filtering=False,jobs=self.jobs,
            predecessor=deepcopy(self.predecessor),profile=self.profile,CPU_affinity=self.affinity,
            retired_partial_batch=deepcopy(self.retired),**self.stamp)
        self.plan_ref=write(root,self.owner_runtime+'/plan.json',self.plan)
        process=self.owner_process
        launch=write(root,self.owner_runtime+'/launch.json',dict(process=process,
            source=self.graph['control']['components']['train_owner'],plan=self.plan_ref,resource_session=self.session_ref,**self.stamp))
        terminal=write(root,self.owner_runtime+'/terminal.json',dict(complete=True,completed=len(self.jobs),attempted=len(self.jobs),
            infrastructure_failed=0,active=None,automatic_retry=False,process=process,plan=self.plan_ref,resource_session=self.session_ref,**self.stamp))
        exit=write(root,self.owner_runtime+'/exec_completion.json',dict(process=process,actual_tool_exit_observed=True,
            natural_completion=True,returncode=0,plan=self.plan_ref,terminal=terminal))
        self.lineage=dict(schema=subject.LINEAGE_SCHEMA,complete=True,mixed_control_lineage=True,automatic_retry=False,
            source_outcome_filtering=False,case_graphs_relabelled=False,predecessor_exit=self.exit_ref,suffix_plan=self.plan_ref,
            suffix_owner=dict(launch=launch,terminal=terminal,exec_completion=exit),records=self.records,
            **retired_disclosure(self.retired),parallel_handoff=deepcopy(self.handoff),**self.stamp)
        self.refresh()
    def refresh(self):self.lineage_ref=write(self.root,self.owner_runtime+'/lineage.json',self.lineage)
    def audit(self):
        with patch.object(subject,'_predecessor',return_value=deepcopy(self.predecessor)),\
             patch.object(subject,'_retired_development_batch',return_value=deepcopy(self.retired)),\
             patch.object(subject,'_parallel_handoff',return_value=deepcopy(self.handoff)),\
             patch.object(subject,'_middle_auditor',return_value=middle_audit):
            return subject.audit_mixed_train192(self.root,self.graph_ref,self.lineage_ref)


class MixedContracts(unittest.TestCase):
    def setUp(self):
        self.temp=TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
    def test_all192_three_graphs_noncontiguous_middle_all_false_pass(self):
        f=MixedFixture(self.root);proof=f.audit()
        self.assertEqual(proof['completed'],192);self.assertTrue(proof['mixed_control_lineage'])
        self.assertFalse(any(row['native_success'] for row in proof['cases']))
        self.assertEqual(proof['preserved_prefix_count'],2)
        self.assertEqual(f.old.graph['physical_policy_sha256'],f.graph['physical_policy_sha256'])
        self.assertNotEqual(f.old.graph['control_graph_sha256'],f.graph['control_graph_sha256'])
        self.assertEqual(proof['control_segment_count'],3)
        self.assertEqual(proof['preserved_parallel_count'],3)
        self.assertEqual(proof['new_suffix_count'],192-f.k-3)
        self.assertEqual(len({r['case_graph']['sha256'] for r in proof['cases']}),3)
        middle=[r for r in proof['cases'] if r['control_segment']=='preserved_parallel_middle']
        self.assertEqual(len(middle),3)
        self.assertTrue(all(r['preserved_without_reexecution'] for r in middle))
        self.assertTrue(set(f.middle_ids).isdisjoint(j['job_id'] for j in f.jobs))
        self.assertGreater(int(f.middle_ids[1][1:])-int(f.middle_ids[0][1:]),1)
        self.assertTrue(proof['historical_Native_reexecution_disclosed'])
        self.assertTrue(proof['retired_development_records_not_counted_in_accepted192'])
        self.assertEqual(len(proof['retired_development_batch']['Native_development_records']),2)
        self.assertTrue(all(r['native_success'] for r in proof['retired_development_batch']['Native_development_records']))

    def test_retired_partial_batch_cannot_be_dropped_filtered_or_counted_as_new_acceptance(self):
        f=MixedFixture(self.root)
        for bad in ('drop_batch','drop_CV','count_accepted','hide_repeat','source_selection','different_exit','qualify'):
            old=deepcopy(f.lineage)
            if bad=='drop_batch':f.lineage.pop('retired_development_batch')
            elif bad=='drop_CV':f.lineage['retired_development_batch']['Native_development_records'].pop(0)
            elif bad=='count_accepted':f.lineage['retired_development_batch']['accepted_into_new_cohort']=True
            elif bad=='hide_repeat':f.lineage['historical_Native_reexecution_disclosed']=False
            elif bad=='source_selection':f.lineage['retired_development_batch']['source_outcomes_used_for_rerun_or_version_selection']=True
            elif bad=='different_exit':f.lineage['retired_development_batch']['exit']=f.exit_ref
            else:f.lineage['retired_development_batch']['scientific_qualification_granted']=True
            f.refresh()
            with self.subTest(bad=bad),self.assertRaises(ValueError):f.audit()
            f.lineage=old

    def test_preserved_middle_set_cannot_be_dropped_relabelled_or_reexecuted(self):
        f=MixedFixture(self.root)
        for bad in ('missing','drop_record','changed_graph','permit_reexecution','changed_count','duplicate_id'):
            original=deepcopy(f.lineage)
            if bad=='missing':f.lineage.pop('parallel_handoff')
            elif bad=='drop_record':f.lineage['parallel_handoff']['completed_records'].pop()
            elif bad=='changed_graph':f.lineage['parallel_handoff']['old1194_graph']=f.graph_ref
            elif bad=='permit_reexecution':f.lineage['parallel_handoff']['completed_records_reused_without_reexecution']=False
            elif bad=='changed_count':f.lineage['parallel_handoff']['preserved_count']=False
            else:f.lineage['parallel_handoff']['completed_job_ids'][1]=f.middle_ids[0]
            f.refresh()
            with self.subTest(bad=bad),self.assertRaises(ValueError):f.audit()
            f.lineage=original
        original=deepcopy(f.plan)
        f.plan['jobs'][0]=deepcopy(f.middle.jobs[1])
        f.plan_ref=write(self.root,f.plan_ref['path'],f.plan);f.lineage['suffix_plan']=f.plan_ref;f.refresh()
        with self.assertRaises(ValueError):f.audit()
        f.plan=original

    def test_original_middle_auditor_rejects_corrupt_original_resource_evidence(self):
        f=MixedFixture(self.root);row=deepcopy(f.middle_records[0]);job=next(j for j in f.middle.jobs if j['job_id']==row['job_id'])
        ref=row['native']['sidecar'];sidecar=json.loads((self.root/ref['path']).read_text())
        sidecar['resource_scope_receipt']['parent_broker_final']['per_FD_lock_evidence'][0]['FD_stat_before_after_verified']=False
        row['native']['sidecar']=write(self.root,ref['path'],sidecar)
        manifest,_=subject.read_json(self.root,f.middle.graph['train_source_manifest']);sources=subject.base._sources(manifest)
        with self.assertRaises(ValueError):middle_audit._audit_records(self.root,f.middle.graph_ref,f.middle.graph,
            [job],[row],sources,(set(),set(),set(),set()),parallel=True,master_session=f.middle.session_ref)

    def test_per_FD_raw_lock_identity_and_independent_conflict_are_both_required(self):
        session=dict(owner=dict(pid=1234,start_ticks=10),parent_global_locks=[
            dict(path='runtime/framework/lock_a.lock',device=1,inode=41,fd=7),
            dict(path='runtime/framework/lock_b.lock',device=1,inode=42,fd=8)])
        original=make_kernel_lock_evidence(session)
        with patch.object(Path,'read_text',side_effect=AssertionError('offline audit must not read /proc')):
            subject.validate_per_FD_kernel_evidence(original,session)
        for bad in ('global_proc_interface','missing_perFD','missing_conflict','short_rows','reordered',
                'wrong_fd','wrong_owner','read_lock','partial_range','stat_unverified','raw_PID','raw_inode','raw_device',
                'raw_mode','raw_not_kernel_row','conflict_success','conflict_permission_error','conflict_bool','not_confirmed','conflict_file'):
            changed=deepcopy(original)
            if bad=='global_proc_interface':changed['parent_kernel_verification']='global_proc_locks'
            elif bad=='missing_perFD':changed.pop('per_FD_lock_evidence')
            elif bad=='missing_conflict':changed.pop('independent_conflict_evidence')
            elif bad=='short_rows':changed['per_FD_lock_evidence'].pop()
            elif bad=='reordered':changed['per_FD_lock_evidence'].reverse()
            elif bad=='wrong_fd':changed['per_FD_lock_evidence'][0]['fd']=8
            elif bad=='wrong_owner':changed['per_FD_lock_evidence'][0]['owner_pid']=True
            elif bad=='read_lock':changed['per_FD_lock_evidence'][0]['mode']='READ'
            elif bad=='partial_range':changed['per_FD_lock_evidence'][0]['start']=1
            elif bad=='stat_unverified':changed['per_FD_lock_evidence'][0]['FD_stat_before_after_verified']=False
            elif bad=='raw_PID':changed['per_FD_lock_evidence'][0]['raw']=changed['per_FD_lock_evidence'][0]['raw'].replace('WRITE 1234','WRITE 1235')
            elif bad=='raw_inode':changed['per_FD_lock_evidence'][0]['raw']=changed['per_FD_lock_evidence'][0]['raw'].replace(':41 ',':42 ')
            elif bad=='raw_device':changed['per_FD_lock_evidence'][0]['raw']=changed['per_FD_lock_evidence'][0]['raw'].replace('0:1:', '0:2:')
            elif bad=='raw_mode':changed['per_FD_lock_evidence'][0]['raw']=changed['per_FD_lock_evidence'][0]['raw'].replace('WRITE','READ')
            elif bad=='raw_not_kernel_row':changed['per_FD_lock_evidence'][0]['raw']='held=True'
            elif bad=='conflict_success':changed['independent_conflict_evidence'][0]['conflict_errno']=0
            elif bad=='conflict_permission_error':changed['independent_conflict_evidence'][0]['conflict_errno']=13
            elif bad=='conflict_bool':changed['independent_conflict_evidence'][0]['conflict_errno']=True
            elif bad=='not_confirmed':changed['independent_conflict_evidence'][0]['confirmed']=False
            else:changed['independent_conflict_evidence'][0]['inode']=42
            with self.subTest(bad=bad),self.assertRaises(ValueError):
                subject.validate_per_FD_kernel_evidence(changed,session)
    def test_drifted_physical_or_merged_inputs_rejected(self):
        f=MixedFixture(self.root)
        for key in ('physical','routes','roots'):
            g=deepcopy(f.graph)
            if key=='physical':g[key]['complete_source_closure_declared']=False
            elif key=='routes':g[key][0]['frozen_inputs']['sha256']='0'*64
            else:g[key]['project']+='/changed'
            with self.subTest(key=key),self.assertRaises(ValueError):subject.compare_physical(f.old.graph,g)
    def test_missing_duplicate_retry_or_relabelled_record_rejected(self):
        f=MixedFixture(self.root)
        for bad in ('missing','duplicate','retry','relabel'):
            original=deepcopy(f.lineage)
            if bad=='missing':f.lineage['records'].pop()
            if bad=='duplicate':f.lineage['records'][0]=deepcopy(f.lineage['records'][1])
            if bad=='retry':f.lineage['automatic_retry']=True
            if bad=='relabel':f.lineage['case_graphs_relabelled']=True
            f.refresh()
            with self.subTest(bad=bad),self.assertRaises(ValueError):f.audit()
            f.lineage=original
    def test_unrestored_parallel_resource_is_not_native_false(self):
        f=MixedFixture(self.root);row=f.records[0];ref=row['native']['sidecar']
        value=json.loads((self.root/ref['path']).read_text());value['resource_scope_receipt']['restored']=False
        row['native']['sidecar']=write(self.root,ref['path'],value);f.refresh()
        with self.assertRaises(ValueError):f.audit()
    def test_wrong_metadata_domain_refused_before_open(self):
        with patch.object(subject,'read_json',side_effect=AssertionError('must classify first')):
            with self.assertRaises(ValueError):subject.require_readiness_proof(self.root,{},dict(root='project',path='val/data.h5',sha256='a'*64))

    def test_saved_mixed_proof_must_equal_full_reaudit_not_just_complete(self):
        value=dict(schema=subject.READINESS_SCHEMA,lineage=dict(root='project',path='runtime/framework/x/lineage.json',sha256='a'*64),complete=True)
        ref=write(self.root,'runtime/framework/x/readiness.json',value)
        with patch.object(subject,'audit_mixed_train192',return_value=deepcopy(value)):
            self.assertEqual(subject.require_readiness_proof(self.root,{},ref),value)
            ref=write(self.root,ref['path'],dict(value,unreviewed_extra_field=True))
            with self.assertRaises(ValueError):subject.require_readiness_proof(self.root,{},ref)

    def test_owner_bool_exit_or_wrong_plan_is_not_natural_completion(self):
        f=MixedFixture(self.root)
        ref=f.lineage['suffix_owner']['exec_completion'];original=json.loads((self.root/ref['path']).read_text())
        for field,value in [('returncode',False),('natural_completion',False),('plan',f.old.plan_ref)]:
            changed=deepcopy(original);changed[field]=value
            refs=deepcopy(f.lineage['suffix_owner']);refs['exec_completion']=write(self.root,ref['path'],changed)
            with self.subTest(field=field),self.assertRaises(ValueError):
                subject._suffix_owner(self.root,refs,f.plan_ref,f.graph_ref,f.graph,192-f.k)

    def test_wrong_same_graph_master_or_source_job_rejected(self):
        f=MixedFixture(self.root);refs=f.lineage['suffix_owner']
        session=json.loads((self.root/f.session_ref['path']).read_text())
        launch=json.loads((self.root/refs['launch']['path']).read_text());terminal=json.loads((self.root/refs['terminal']['path']).read_text())
        for field in ('owner','jobs','plan'):
            changed=deepcopy(session)
            if field=='owner':changed['owner']['start_ticks']+=1
            elif field=='jobs':changed['jobs'][0]['split']='validation'
            else:changed['plan']=f.old.plan_ref
            new_session=write(self.root,f.session_ref['path'],changed)
            launch['resource_session']=new_session;terminal['resource_session']=new_session
            rr=dict(launch=write(self.root,refs['launch']['path'],launch),terminal=write(self.root,refs['terminal']['path'],terminal))
            with self.subTest(field=field),self.assertRaises(ValueError):
                subject.validate_master_session(self.root,rr,f.plan_ref,f.graph_ref,f.graph,f.jobs)

    def test_claim_outside_exact_job_phase_is_rejected(self):
        f=MixedFixture(self.root);job=f.jobs[0];row=f.records[0]
        sidecar=json.loads((self.root/row['CPU_check']['sidecar']['path']).read_text())
        launch=json.loads((self.root/row['CPU_check']['launch']['path']).read_text())
        receipt=sidecar['resource_scope_receipt'];original=deepcopy(receipt['claim'])
        claim=json.loads((self.root/original['path']).read_text())
        wrong=write(self.root,f.owner_runtime+'/resources/claims/other_CPU_check/claim.json',claim)
        receipt['claim']=wrong
        with self.assertRaises(ValueError):subject.validate_parallel_phase(self.root,sidecar,launch,job,f.graph_ref,f.graph,check=True)

    def test_broker_admission_and_final_are_exact_authenticated_claim_proofs(self):
        f=MixedFixture(self.root);job=f.jobs[0];row=f.records[0]
        sidecar=json.loads((self.root/row['CPU_check']['sidecar']['path']).read_text())
        launch=json.loads((self.root/row['CPU_check']['launch']['path']).read_text())
        subject.validate_parallel_phase(self.root,sidecar,launch,job,f.graph_ref,f.graph,check=True)
        changes=[('parent_FD_verified',False),('kernel_FLOCK_verified',False),
            ('client_direct_parent_FD_read',True),('server_authenticated_by_SO_PEERCRED',False),
            ('client_authenticated_by_SO_PEERCRED',False),('server_uid',False),('client_gid',1001),
            ('nonce','not-a-fresh-nonce'),('checked_utc','2026-10-24T00:00:00'),('boot_id','other-boot'),
            ('owner',dict(pid=1,start_ticks=2)),('client',dict(pid=3,start_ticks=4)),
            ('session',f.old.plan_ref),('claim',f.old.plan_ref),('guard',subject.INHERITED_ENTRY),
            ('parent_guard',subject.INHERITED_ENTRY),('lock_proof',[]),('job_id','j_other'),
            ('phase','native'),('slot',True),('gpu_slot',0)]
        for which in ('parent_broker_admission','parent_broker_final'):
            for field,value in changes:
                changed=deepcopy(sidecar);changed['resource_scope_receipt'][which][field]=value
                with self.subTest(which=which,field=field),self.assertRaises(ValueError):
                    subject.validate_parallel_phase(self.root,changed,launch,job,f.graph_ref,f.graph,check=True)
        for bad in ('same_nonce','final_at_admission','missing_admission','missing_final'):
            changed=deepcopy(sidecar);receipt=changed['resource_scope_receipt']
            if bad=='same_nonce':receipt['parent_broker_final']['nonce']=receipt['parent_broker_admission']['nonce']
            elif bad=='final_at_admission':changed['resource_proof']['parent_broker_final']=receipt['parent_broker_final']
            elif bad=='missing_admission':receipt.pop('parent_broker_admission')
            else:receipt.pop('parent_broker_final')
            with self.subTest(bad=bad),self.assertRaises(ValueError):
                subject.validate_parallel_phase(self.root,changed,launch,job,f.graph_ref,f.graph,check=True)
        changed=deepcopy(sidecar);changed['resource_slot']=False
        with self.assertRaises(ValueError):
            subject.validate_parallel_phase(self.root,changed,launch,job,f.graph_ref,f.graph,check=True)

    def test_master_requires_actual_parent_namespace_broker_and_all_frozen_lock_rows(self):
        f=MixedFixture(self.root);refs=f.lineage['suffix_owner']
        session=json.loads((self.root/f.session_ref['path']).read_text())
        launch=json.loads((self.root/refs['launch']['path']).read_text())
        terminal=json.loads((self.root/refs['terminal']['path']).read_text())
        for bad in ('schema','transport','address','owner','boot','uid_bool','fd_inheritance','verification','missing_locks','bool_fd','guard'):
            changed=deepcopy(session);broker=changed['parent_broker']
            if bad=='schema':broker['schema']='r10_live_parent_lock_broker_v1171'
            elif bad=='transport':broker['transport']='TCP'
            elif bad=='address':broker['address']='our-mani-1195-fixed'
            elif bad=='owner':broker['owner']['start_ticks']+=1
            elif bad=='boot':broker['boot_id']='other-boot'
            elif bad=='uid_bool':broker['uid']=False
            elif bad=='fd_inheritance':broker['inherited_lock_descriptors_passed_to_children']=True
            elif bad=='verification':broker['verification']='child_direct_parent_FD_read'
            elif bad=='missing_locks':changed['parent_global_locks'].pop()
            elif bad=='bool_fd':changed['parent_global_locks'][0]['fd']=True
            else:changed['guard']=subject.INHERITED_ENTRY
            new_session=write(self.root,f.session_ref['path'],changed)
            launch['resource_session']=new_session;terminal['resource_session']=new_session
            rr=dict(launch=write(self.root,refs['launch']['path'],launch),terminal=write(self.root,refs['terminal']['path'],terminal))
            with self.subTest(bad=bad),self.assertRaises(ValueError):
                subject.validate_master_session(self.root,rr,f.plan_ref,f.graph_ref,f.graph,f.jobs)


if __name__=='__main__':unittest.main()
