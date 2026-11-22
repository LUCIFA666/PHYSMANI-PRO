"""Synthetic full192 new-graph receipts; no real source, lease or simulation."""
from copy import deepcopy
import hashlib
import os
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from our_mani.multitask import public_held_repaired_readiness_v1208 as subject
from our_mani.multitask import repaired_policy_graph_v1143 as graphs
from tests.test_repaired_readiness_v1144 import FreshFixture,write
from tests.test_repaired_policy_graph_v1143 import put


def make_previous_closed_TRAIN_fixture(root):
    from tests.test_parallel_repaired_readiness_v1198 import make_retired_development_fixture
    base='runtime/framework/parallel_repaired_owner_v1197/train/synthetic_closed_previous'
    refs={key:write(root,base+'/'+name,dict(synthetic=True,role=key)) for key,name in
        (('readiness','readiness.json'),('lineage','lineage.json'),('owner_plan','plan.json'))}
    refs['graph']=write(root,'artifacts/framework/multitask_r10/synthetic_previous/policy_graph.json',{})
    refs['owner']={key:write(root,base+'/'+key+'.json',dict(synthetic=True))
        for key in ('launch','terminal','exec_completion')}
    return dict(schema='r10_closed_previous_TRAIN192_history_v1207',**refs,
        completed=192,source_count=64,all_child_processes_dead=True,actual_owner_natural_exit0=True,
        original_results_retained=True,historical_results_counted_in_accepted192=False,
        prior_TRAIN_results_reused_for_new_qualification=False,
        prior_TRAIN_source_reexecution_declared_for_new_physical_qualification=True,
        historical_Native_reexecution_disclosed=True,
        retired_development_batch=make_retired_development_fixture(root),repeated_development_Native_cases=2,
        retired_development_records_not_counted_in_accepted192=True,
        source_outcomes_used_for_rerun_or_version_selection=False)


def make_development_diagnostic_fixture(root):
    return write(root,'runtime/framework/parallel_isolation_source_review_20261024/pilot1204_fixed6_20261024_a/development_summary.json',
        dict(synthetic_only=True,never_actual_qualification=True))


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
    data=dict(schema='r10_parallel_repaired_CPU_affinity_v1212',allowed_logical_cpus=list(range(32)),
        physical_core_count=16,core_groups=[dict(package=0,core=i,cpus=[i,i+16],L3_bytes=1024) for i in range(16)],
        native_slot_cpus=[[i,i+16] for i in range(12)],control_and_CPU_preflight_cpus=[12,13,14,15,28,29,30,31],
        max_native_slots=12,taskset_root=str(tools),taskset_binary=tool,
        taskset_runs_outside820_as_control_tool=True,bitwise_equivalence_claimed=False)
    ref=write(root,'artifacts/framework/multitask_r10/public_held_repaired_control_v1212_setup/parallel_CPU_affinity_profile.json',data)
    spec.setdefault('external_roots',{})['host_tools']=str(tools)
    spec['control']['static_inputs'].update(parallel_CPU_affinity_profile=ref,parallel_taskset_binary=tool)
    return dict(profile_ref=ref,tool_ref=tool,CPU_affinity=dict(data,profile_ref=ref))


def make_session(root,owner_runtime,graph_ref,graph,*,owner=None,plan_ref=None,jobs=None):
    from our_mani.multitask.post_completed_TEST_resource_guard_v1106 import LOCKS
    owner=owner or dict(pid=2147482000,start_ticks=80000)
    parent_guard=next(ref for ref in graph['control']['sources_map'].values() if ref['path']==subject.PARENT_GUARD_SOURCE)
    boot_id='00000000-0000-0000-0000-000000000001'
    session=dict(schema='r10_parent_owned_parallel_resource_session_v1213',owner=owner,
        graph=graph_ref,**{k:graph[k] for k in subject.HASHES},
        guard=graph['control']['components']['resource_guard'],parent_guard=parent_guard,test_only=False,
        boot_id=boot_id,parent_global_locks=[dict(path=name,device=1,inode=i+1,fd=i+3) for i,name in enumerate(LOCKS)],
        parent_broker=dict(schema='r10_live_parent_lock_broker_v1213',transport='AF_UNIX_abstract',
            address='our-mani-1213-'+'a'*48,owner=owner,uid=1000,gid=1000,boot_id=boot_id,
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
    claim=dict(schema='r10_once_only_parallel_resource_claim_v1213',session=session_ref,job=deepcopy(job),
        phase=phase,slot=0,gpu_slot=gpu,process=sidecar['process'],automatic_retry=False,test_only=False)
    directory=owner_runtime+'/resources/claims/'+job['job_id']+'_'+phase
    claim_ref=write(root,directory+'/claim.json',claim)
    receipt=dict(schema='r10_parallel_child_resource_receipt_v1213',purpose='child_slot',
        phase=phase,slot=0,gpu_slot=gpu,job_id=job['job_id'],process=sidecar['process'],session=session_ref,
        claim=claim_ref,actual_parent=session['owner'],parent_global_locks_verified=True,
        parent1106_empty_inventory_checked_at_session_start_only=True,child_empty_inventory_claimed=False,
        test_only=False,Native_authorized=not check,body_completed=True,restored=True,leases_removed=True,
        errors=[],graph=graph_ref,**{k:graph[k] for k in subject.HASHES})
    for label in ('admission','final'):
        receipt['parent_broker_'+label]=dict(schema='r10_parent_lock_broker_verification_v1213',session=session_ref,
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


class FreshParallelFixture:
    def __init__(self,root):
        self.root=Path(root);old=FreshFixture(root);self.old=old
        spec={n:deepcopy(old.graph[n]) for n in ('tasks','methods','train_source_manifest','physical','control','routes')}
        spec.update(schema=graphs.SPEC_SCHEMA,external_roots={n:v for n,v in old.graph['roots'].items() if n!='project'})
        auditor=put(root,subject.SOURCE,Path(subject.__file__).read_bytes())
        owner=put(root,subject.OWNER_SOURCE,b'synthetic new owner')
        worker=put(root,'scripts/run_public_held_repaired_case_v1206_r10.py',b'synthetic new worker')
        preparer=put(root,'our_mani/multitask/public_held_repaired_case_v1206.py',b'synthetic new preparer')
        spec['control']['components'].update(train_auditor=auditor,train_owner=owner)
        spec['control']['sources_map'].update(new_auditor=auditor,new_owner=owner,new_worker=worker,new_preparer=preparer)
        spec['control']['contracts']['readiness_schema']=subject.READINESS_SCHEMA
        # New synthetic overlay physically differs, without importing any real factory.
        spec['physical']['source_overlay_lock']=put(root,'metadata/new_public_held_source_lock.json',b'{"synthetic_new_physical":true}')
        for row in spec['routes']:
            for purpose in ('train','evaluation'):
                row[purpose].update(worker=worker,preparer=preparer,case_schema='new_actual_parallel_case')
        self.baseline=put(root,'metadata/epoch_99999.pth',b'synthetic baseline checkpoint')
        checkpoint=spec['physical']['shared_context_checkpoint']
        model=spec['routes'][0]['models']['trained_model']
        self.config=dict(context_checkpoint=checkpoint,baseline_checkpoint=str(self.root/self.baseline['path']),
            baseline_reference_filter='author_direct_setpoint',learned_artifacts={t:dict(geometry=model,interaction=model,rigid_layout=model)
                for t in subject.TASKS if t not in subject.BLOCKS})
        config_ref=write(root,'metadata/new_configuration.json',self.config)
        for route in spec['routes']:
            route['configuration']=config_ref
            if route['method']=='physmani':route['checkpoints']={'baseline':self.baseline}
        self.affinity=make_CPU_affinity_fixture(root,spec)['CPU_affinity'];add_broker_sources(root,spec)
        self.graph_ref=graphs.write_policy_graph(root,spec,namespace='artifacts/framework/multitask_r10/new_fresh_graph')
        self.graph=graphs.read_policy_graph(root,self.graph_ref)
        self.stamp=dict(graph=self.graph_ref,**{key:self.graph[key] for key in subject.HASHES})
        self.owner_runtime='runtime/framework/public_held_repaired_owner_v1207/train/synthetic'
        manifest,_=subject.read_json(root,self.graph['train_source_manifest']);sources=subject.base._sources(manifest)
        routes={(row['task'],row['method']):row for row in self.graph['routes']}
        self.jobs=[];self.records=[];self.history=make_previous_closed_TRAIN_fixture(root)
        self.development=make_development_diagnostic_fixture(root)
        for index,previous in enumerate(old.jobs):
            key=subject._key(previous);route=routes[(key[0],key[2])]
            suffix='cohorts/synthetic/'+key[0]+'/'+key[1]+'/'+key[2]
            self.jobs.append(dict(job_id=f'j{index:04d}',task=key[0],raw_episode_id=key[1],method=key[2],split='train',
                configuration=route['configuration'],source_row_sha256=subject.digest(sources[key[:2]]),
                out='artifacts/framework/multitask_r10/public_held_repaired_policy_v1206/train/development_demo_only/'+suffix,
                runtime='runtime/framework/public_held_repaired_policy_v1206/train/'+suffix))
        self.profile=dict(max_native_slots=12,max_GPU_slots=5,max_CPU_preflight_slots=6,ready_limit=24)
        self.plan=dict(schema=subject.SUFFIX_PLAN_SCHEMA,split='train',namespace='synthetic',owner_runtime=self.owner_runtime,
            global_expected_cases=192,expected_cases=192,new_suffix_count=192,predecessor=None,validation_bundle=None,
            sources=list(sources.values()),jobs=self.jobs,profile=self.profile,CPU_affinity=self.affinity,
            previous_closed_TRAIN=self.history,development_diagnostic=self.development,
            mixed_control_lineage=False,case_graphs_relabelled=False,reused_cases=[],
            automatic_retry=False,source_outcome_filtering=False,**subject._adoption_fields(),**subject._fresh_fields(),**self.stamp)
        self.plan_ref=write(root,self.owner_runtime+'/plan.json',self.plan)
        self.owner_process=dict(pid=2147482100,start_ticks=81000)
        self.session_ref=make_session(root,self.owner_runtime,self.graph_ref,self.graph,
            owner=self.owner_process,plan_ref=self.plan_ref,jobs=self.jobs)
        for index,oldrow in enumerate(old.records):
            job=self.jobs[index];key=subject._key(job);route=routes[(key[0],key[2])]
            case=json.loads((self.root/oldrow['case']['path']).read_text())
            witness=subject.make_graph_witness(self.graph_ref,self.graph,route,sources[key[:2]],split='train')
            ep=case['execution_protocol'];ep.update(**self.stamp,actual_worker=worker,actual_preparer=preparer,
                source_overlay_lock=self.graph['physical']['source_overlay_lock'],factory_binding=witness['factory_binding'],
                inherited742_preflight_entry=subject.INHERITED_ENTRY,actual_worker_verified_by_new_control_graph=True,
                preflight_entry_is_actual_outer_worker=False)
            case.update(version=route['train']['case_schema'],graph_witness=witness,
                source_overlay_lock=self.graph['physical']['source_overlay_lock'])
            identity=case['identity'];identity.update(execution_protocol_sha256=subject.digest(ep),config_sha256=route['configuration']['sha256'])
            case['identity']=subject.base.bind_graph_identity(identity,witness)
            model_proof=dict(GPU_started=False)
            if key[2]=='physmani':
                model_proof.update(original_author_checkpoint=dict(path=self.config['baseline_checkpoint'],sha256=self.baseline['sha256']),
                    original_neural_policy_and_online_world_model=True)
                case.update(trained_component_identity=None,actor_metrics=dict(original_baseline_algorithm_unchanged=True))
            else:
                case['shared_context_checkpoint']=self.graph['physical']['shared_context_checkpoint']
                if key[0] in subject.BLOCKS:
                    binding=dict(task=key[0],checkpoint_sha256=self.graph['physical']['shared_context_checkpoint']['sha256'])
                    model_proof['actual_shared16_block_adapter']=binding
                    case['trained_component_identity']=dict(actual_shared16_block_adapter=deepcopy(binding))
                else:
                    component=dict(schema='r10_direct_public_RML_component_identity_v643',task=key[0],method=key[2],
                        learned_operation_artifacts=deepcopy(self.config['learned_artifacts'][key[0]]),
                        shared_context_checkpoint=self.graph['physical']['shared_context_checkpoint'])
                    component['sha256']=subject.digest(component)
                    model_proof.update(actual_model_loader_validation=True,native_started=False,
                        old_TRAIN_actor_constructor_executed=False,component_identity=component,component_domain=case['component_domain'])
                    case['trained_component_identity']=deepcopy(component)
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
                oldworker=old.graph['routes'][0]['train']['worker']['path']
                launch['argv']=[worker['path'] if x==oldworker else x for x in launch['argv']]
                sidecar=json.loads((self.root/oldrow[phase]['sidecar']['path']).read_text())
                sidecar.update(**self.stamp,actual_worker=worker,actual_preparer=preparer,graph_witness=witness)
                sidecar['execution'].update(identity=case['identity'],graph_witness=witness,execution_protocol=ep,case=case_ref)
                if check:sidecar['execution']['source_and_model_preflight']=deepcopy(model_proof)
                sidecar,launch=add_resource_phase(root,self.owner_runtime,self.session_ref,self.graph_ref,self.graph,
                    job,sidecar,launch,check=check)
                outer=self.owner_runtime+'/children/'+key[0]+'/'+key[1]+'/'+key[2]+'/'+phase
                record[phase]=dict(launch=write(root,outer+'/launch.json',launch),
                    terminal=write(root,outer+'/terminal.json',json.loads((self.root/oldrow[phase]['terminal']['path']).read_text())),
                    sidecar=write(root,job['runtime']+'/'+phase+'/worker_sidecar.json',sidecar))
            self.records.append(record)
        self.owner_refs=dict(
            launch=write(root,self.owner_runtime+'/launch.json',dict(process=self.owner_process,
                source=self.graph['control']['components']['train_owner'],plan=self.plan_ref,resource_session=self.session_ref,**self.stamp)),
            terminal=write(root,self.owner_runtime+'/terminal.json',dict(complete=True,completed=192,attempted=192,infrastructure_failed=0,
                active=None,automatic_retry=False,process=self.owner_process,plan=self.plan_ref,resource_session=self.session_ref,
                drain_reason=None,remaining_job_ids=[],**self.stamp)))
        self.owner_refs['exec_completion']=write(root,self.owner_runtime+'/exec_completion.json',dict(process=self.owner_process,
            actual_tool_exit_observed=True,natural_completion=True,returncode=0,plan=self.plan_ref,terminal=self.owner_refs['terminal']))
        self.lineage=dict(schema=subject.LINEAGE_SCHEMA,complete=True,plan=self.plan_ref,owner=self.owner_refs,
            records=self.records,previous_closed_TRAIN=self.history,development_diagnostic=self.development,mixed_control_lineage=False,case_graphs_relabelled=False,
            reused_cases=[],automatic_retry=False,source_outcome_filtering=False,**subject._adoption_fields(),**subject._fresh_fields(),**self.stamp)
        self.refresh()
    def refresh(self):
        self.status_ref=write(self.root,self.owner_runtime+'/status.json',dict(stage='complete',active=None,failed=[],
            attempted=192,completed=self.records))
        self.lineage_ref=write(self.root,self.owner_runtime+'/lineage.json',self.lineage)
    def audit(self):
        with patch.object(subject,'_previous_closed_TRAIN',return_value=deepcopy(self.history)),\
                patch.object(subject,'_development_diagnostic',return_value=deepcopy(self.development)):
            return subject.audit_train192(self.root,self.graph_ref,self.lineage_ref)


class FreshContracts(unittest.TestCase):
    def setUp(self):
        self.temp=TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
    def test_full_new_graph192_all_false_pass_without_old_results(self):
        f=FreshParallelFixture(self.root);result=f.audit()
        self.assertEqual(result['completed'],192);self.assertEqual(result['preserved_prefix_count'],0)
        self.assertEqual(result['preserved_parallel_count'],0);self.assertEqual(result['control_segment_count'],1)
        self.assertTrue(result['fresh_TRAIN192']);self.assertTrue(result['new_graph_executed_all192'])
        self.assertFalse(result['mixed_control_lineage']);self.assertFalse(result['historical_results_counted_in_accepted192'])
        self.assertTrue(all(not r['native_success'] and not r['motion_pass'] for r in result['cases']))
        self.assertTrue(all(subject.same(subject.pointer(r['case_graph']),subject.pointer(f.graph_ref)) for r in result['cases']))
        self.assertNotEqual(f.graph['physical_policy_sha256'],f.old.graph['physical_policy_sha256'])
    def test_prior_prefix_parallel_or_historical_results_never_count(self):
        f=FreshParallelFixture(self.root)
        for key,value in (('preserved_prefix_count',74),('preserved_parallel_count',35),('control_segment_count',3),
                ('historical_results_counted_in_accepted192',True),('mixed_control_lineage',True),('fresh_TRAIN192',False)):
            old=f.lineage[key];f.lineage[key]=value;f.refresh()
            with self.subTest(key=key),self.assertRaises(ValueError):f.audit()
            f.lineage[key]=old
    def test_missing_duplicate_reorder_retry_or_foreign_graph_case_rejected(self):
        f=FreshParallelFixture(self.root);original=deepcopy(f.records)
        for change in ('missing','duplicate','reordered','retry','old_case'):
            f.records[:]=deepcopy(original)
            if change=='missing':f.records.pop()
            elif change=='duplicate':f.records[1]=deepcopy(f.records[0])
            elif change=='reordered':f.records[0],f.records[1]=f.records[1],f.records[0]
            elif change=='retry':f.records[0]['attempt']=2
            else:f.records[0]['case']=f.old.records[0]['case']
            f.refresh()
            with self.subTest(change=change),self.assertRaises(ValueError):f.audit()
    def test_previous_history_missing_filtered_live_or_wrong_roles_rejected(self):
        value=make_previous_closed_TRAIN_fixture(self.root);subject.validate_previous_closed_TRAIN(value)
        for key,changed in (('all_child_processes_dead',False),('actual_owner_natural_exit0',False),
                ('historical_results_counted_in_accepted192',True),('completed',True),
                ('repeated_development_Native_cases',0),('prior_TRAIN_results_reused_for_new_qualification',True)):
            v=deepcopy(value);v[key]=changed
            with self.subTest(key=key),self.assertRaises(ValueError):subject.validate_previous_closed_TRAIN(v)
        v=deepcopy(value);v['readiness']=dict(root='project',path='val/data.h5',sha256='a'*64)
        with self.assertRaises(ValueError):subject.validate_previous_closed_TRAIN(v)
    def test_owner_requires_actual_strict_exit0_no_drain_and_all_dead(self):
        f=FreshParallelFixture(self.root);subject._suffix_owner(self.root,f.owner_refs,f.plan_ref,f.graph_ref,f.graph,192)
        original=json.loads((self.root/f.owner_refs['exec_completion']['path']).read_text())
        for key,value in (('returncode',False),('natural_completion',False),('actual_tool_exit_observed',False)):
            changed=dict(original);changed[key]=value;rr=deepcopy(f.owner_refs)
            rr['exec_completion']=write(self.root,rr['exec_completion']['path'],changed)
            with self.subTest(key=key),self.assertRaises(ValueError):subject._suffix_owner(self.root,rr,f.plan_ref,f.graph_ref,f.graph,192)
        f.owner_refs['exec_completion']=write(self.root,f.owner_refs['exec_completion']['path'],original)
        with patch.object(subject.base,'_not_live',return_value=False),self.assertRaises(ValueError):
            subject._suffix_owner(self.root,f.owner_refs,f.plan_ref,f.graph_ref,f.graph,192)
    def test_broker_protocol_old_guard_or_failed_independent_conflict_refused(self):
        f=FreshParallelFixture(self.root);job=f.jobs[0];row=f.records[0]
        sidecar,_=subject.read_json(self.root,row['CPU_check']['sidecar']);launch,_=subject.read_json(self.root,row['CPU_check']['launch'])
        subject.validate_parallel_phase(self.root,sidecar,launch,job,f.graph_ref,f.graph,check=True)
        for key,value in (('schema','r10_parallel_child_resource_receipt_v1195'),('restored',False)):
            changed=deepcopy(sidecar);changed['resource_scope_receipt'][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):
                subject.validate_parallel_phase(self.root,changed,launch,job,f.graph_ref,f.graph,check=True)
        changed=deepcopy(sidecar)
        changed['resource_scope_receipt']['parent_broker_final']['independent_conflict_evidence'][0]['conflict_errno']=0
        with self.assertRaises(ValueError):subject.validate_parallel_phase(self.root,changed,launch,job,f.graph_ref,f.graph,check=True)
    def test_saved_readiness_must_equal_full_reaudit(self):
        f=FreshParallelFixture(self.root);value=f.audit();ref=write(self.root,f.owner_runtime+'/readiness.json',value)
        with patch.object(subject,'_previous_closed_TRAIN',return_value=f.history),\
                patch.object(subject,'_development_diagnostic',return_value=f.development):
            self.assertEqual(subject.require_readiness_proof(self.root,f.graph_ref,ref),value)
            changed=dict(value,completed=191);ref=write(self.root,ref['path'],changed)
            with self.assertRaises(ValueError):subject.require_readiness_proof(self.root,f.graph_ref,ref)

    def test_actual_CPU_and_Native_model_receipts_bind_each_registered_branch(self):
        f=FreshParallelFixture(self.root)
        chosen={}
        for job,row in zip(f.jobs,f.records):
            branch='physmani' if job['method']=='physmani' else ('block' if job['task'] in subject.BLOCKS else 'core')
            chosen.setdefault(branch,(job,row))
        routes={(r['task'],r['method']):r for r in f.graph['routes']}
        for branch,(job,row) in chosen.items():
            case,_=subject.read_json(self.root,row['case']);cpu,_=subject.read_json(self.root,row['CPU_check']['sidecar'])
            route=routes[(job['task'],job['method'])]
            subject._models(self.root,case,cpu,route,f.graph)
            bad=deepcopy(cpu);bad['execution']['source_and_model_preflight']['GPU_started']=True
            with self.subTest(branch=branch,kind='GPU'),self.assertRaises(ValueError):subject._models(self.root,case,bad,route,f.graph)
            bad=deepcopy(cpu);proof=bad['execution']['source_and_model_preflight']
            if branch=='physmani':proof['original_author_checkpoint']['sha256']='0'*64
            elif branch=='block':proof['actual_shared16_block_adapter']['checkpoint_sha256']='0'*64
            else:proof['component_identity']['sha256']='0'*64
            with self.subTest(branch=branch,kind='CPU_model'),self.assertRaises(ValueError):subject._models(self.root,case,bad,route,f.graph)
            bad_case=deepcopy(case)
            if branch=='physmani':bad_case['actor_metrics']['original_baseline_algorithm_unchanged']=False
            elif branch=='block':bad_case['trained_component_identity']['actual_shared16_block_adapter']['task']='wrong'
            else:bad_case['trained_component_identity']['learned_operation_artifacts']={}
            with self.subTest(branch=branch,kind='Native_model'),self.assertRaises(ValueError):subject._models(self.root,bad_case,cpu,route,f.graph)

    def test_repair_adoption_disclosure_does_not_claim_outcome_blind_version_selection(self):
        f=FreshParallelFixture(self.root)
        for key,value in subject._adoption_fields().items():
            old=f.lineage[key];f.lineage[key]=not value;f.refresh()
            with self.subTest(key=key),self.assertRaises(ValueError):f.audit()
            f.lineage[key]=old


if __name__=='__main__':unittest.main()
