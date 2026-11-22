"""Temporary synthetic mixed192 receipts; no real data or simulator.

The prefix inspector is tested by1173 separately. Full per-case auditing here is
real: its192 JSON receipts, sampling, identities and resources are not mocked.
"""
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from our_mani.multitask import parallel_repaired_readiness_v1174 as subject
from our_mani.multitask import repaired_policy_graph_v1143 as graphs
from tests.test_repaired_readiness_v1144 import FreshFixture,write
from tests.test_repaired_policy_graph_v1143 import put


def make_CPU_affinity_fixture(root,spec):
    """Synthetic control metadata/tool bytes only; never runs taskset/sysfs."""
    root=Path(root);tools=root/'synthetic_host_tools';tools.mkdir(exist_ok=True)
    binary=tools/'taskset';binary.write_bytes(b'synthetic control taskset; never executed')
    import hashlib
    tool=dict(root='host_tools',path='taskset',sha256=hashlib.sha256(binary.read_bytes()).hexdigest())
    data=dict(schema='r10_parallel_repaired_CPU_affinity_v1178',allowed_logical_cpus=list(range(32)),
        physical_core_count=16,core_groups=[dict(package=0,core=i,cpus=[i,i+16],L3_bytes=1024) for i in range(16)],
        native_slot_cpus=[[i,i+16] for i in range(12)],control_and_CPU_preflight_cpus=[12,13,14,15,28,29,30,31],
        max_native_slots=12,taskset_root=str(tools),taskset_binary=tool,
        taskset_runs_outside820_as_control_tool=True,bitwise_equivalence_claimed=False)
    ref=write(root,'artifacts/framework/multitask_r10/parallel_repaired_control_v1178_setup/parallel_CPU_affinity_profile.json',data)
    spec.setdefault('external_roots',{})['host_tools']=str(tools)
    spec['control']['static_inputs'].update(parallel_CPU_affinity_profile=ref,parallel_taskset_binary=tool)
    return dict(profile_ref=ref,tool_ref=tool,CPU_affinity=dict(data,profile_ref=ref))


def make_session(root,owner_runtime,graph_ref,graph,*,owner=None,plan_ref=None,jobs=None):
    owner=owner or dict(pid=2147482000,start_ticks=80000)
    session=dict(schema='r10_parent_owned_parallel_resource_session_v1171',owner=owner,
        graph=graph_ref,**{k:graph[k] for k in subject.HASHES},
        guard=graph['control']['components']['resource_guard'],test_only=False,
        profile=dict(max_native_slots=12,max_GPU_slots=5,max_CPU_preflight_slots=6,ready_limit=24),
        parent_resource_proof=dict(host_inventory=dict(host_visibility_available=True,
            ancestor_chain_complete=True,same_uid_inventory_incomplete=False,live_project_native_processes=[])),
        plan=plan_ref,jobs=[subject._job(j) for j in (jobs or [])],runtime=owner_runtime,
        split=(jobs[0]['split'] if jobs else 'train'))
    return write(root,owner_runtime+'/resources/session.json',session)


def add_resource_phase(root,owner_runtime,session_ref,graph_ref,graph,job,sidecar,launch,*,check):
    phase='CPU_check' if check else 'native';gpu=0 if not check and job['method']=='physmani' else None
    session=json.loads((Path(root)/session_ref['path']).read_text())
    claim=dict(schema='r10_once_only_parallel_resource_claim_v1171',session=session_ref,job=deepcopy(job),
        phase=phase,slot=0,gpu_slot=gpu,process=sidecar['process'],automatic_retry=False,test_only=False)
    directory=owner_runtime+'/resources/claims/'+job['job_id']+'_'+phase
    claim_ref=write(root,directory+'/claim.json',claim)
    receipt=dict(schema='r10_parallel_child_resource_receipt_v1171',purpose='child_slot',
        phase=phase,slot=0,gpu_slot=gpu,job_id=job['job_id'],process=sidecar['process'],session=session_ref,
        claim=claim_ref,actual_parent=session['owner'],parent_global_locks_verified=True,
        parent1106_empty_inventory_checked_at_session_start_only=True,child_empty_inventory_claimed=False,
        test_only=False,Native_authorized=not check,body_completed=True,restored=True,leases_removed=True,
        errors=[],graph=graph_ref,**{k:graph[k] for k in subject.HASHES})
    write(root,directory+'/terminal.json',receipt)
    initial=deepcopy(receipt);initial.update(body_completed=False,restored=False);initial.pop('leases_removed')
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
        self.root=Path(root);old=FreshFixture(root);self.old=old;self.k=k
        oldg=old.graph
        spec={n:deepcopy(oldg[n]) for n in ('tasks','methods','train_source_manifest','physical','control','routes')}
        spec.update(schema=graphs.SPEC_SCHEMA,external_roots={n:v for n,v in oldg['roots'].items() if n!='project'})
        a=put(root,subject.SOURCE,Path(subject.__file__).read_bytes())
        spec['control']['components']['train_auditor']=a;spec['control']['sources_map']['auditor1174']=a
        owner=put(root,'scripts/run_parallel_repaired_owner_v1173_r10.py',b'synthetic owner')
        spec['control']['components']['train_owner']=owner;spec['control']['sources_map']['new_owner']=owner
        worker=put(root,'scripts/run_parallel_repaired_case_v1172_r10.py',b'synthetic actual new worker')
        preparer=put(root,'our_mani/multitask/parallel_repaired_case_v1172.py',b'synthetic new preparer')
        spec['control']['sources_map'].update(new_worker=worker,new_preparer=preparer)
        spec['control']['contracts']['readiness_schema']=subject.READINESS_SCHEMA
        for row in spec['routes']:
            for purpose in ('train','evaluation'):
                row[purpose].update(worker=worker,preparer=preparer,case_schema='new_actual_parallel_case')
        self.affinity=make_CPU_affinity_fixture(root,spec)['CPU_affinity']
        self.graph_ref=graphs.write_policy_graph(root,spec,namespace='artifacts/framework/multitask_r10/new_graph')
        self.graph=graphs.read_policy_graph(root,self.graph_ref)
        self.stamp=dict(graph=self.graph_ref,**{key:self.graph[key] for key in subject.HASHES})
        self.owner_runtime='runtime/framework/parallel_repaired_owner_v1173/train/synthetic'
        manifest,_=subject.read_json(root,oldg['train_source_manifest']);sources=subject.base._sources(manifest)
        old_routes={(row['task'],row['method']):row for row in oldg['routes']}
        for job in old.jobs:
            key=subject._key(job)
            job.update(configuration=old_routes[(key[0],key[2])]['configuration'],
                source_row_sha256=subject.digest(sources[key[:2]]))
        old.plan_ref=write(root,old.plan_ref['path'],old.plan)
        self.predecessor=dict(old_graph=old.graph_ref,original_plan=old.plan_ref,
            records=deepcopy(old.records[:k]),completed_prefix_count=k)
        self.exit_ref=write(root,'runtime/framework/old_controlled_exit.json',dict(actual=True))
        self.jobs=[];self.records=[]
        manifest,_=subject.read_json(root,oldg['train_source_manifest']);sources=subject.base._sources(manifest)
        routes={(row['task'],row['method']):row for row in self.graph['routes']}
        for index in range(k,192):
            previous=old.jobs[index];key=subject._key(previous);route=routes[(key[0],key[2])]
            job=deepcopy(previous);job.update(job_id=f'j{index:04d}',split='train',configuration=route['configuration'],
                source_row_sha256=subject.digest(sources[key[:2]]),
                out='artifacts/framework/multitask_r10/parallel_repaired_policy_v1172/train/development_demo_only/'+str(index),
                runtime='runtime/framework/parallel_repaired_policy_v1172/train/'+str(index))
            self.jobs.append(job)
        self.profile=dict(max_native_slots=12,max_GPU_slots=5,max_CPU_preflight_slots=6,ready_limit=24)
        self.plan=dict(schema=subject.SUFFIX_PLAN_SCHEMA,split='train',global_expected_cases=192,preserved_prefix_count=k,
            expected_cases=192-k,automatic_retry=False,source_outcome_filtering=False,jobs=self.jobs,profile=self.profile,
            predecessor=deepcopy(self.predecessor),CPU_affinity=self.affinity,**self.stamp)
        self.plan_ref=write(root,self.owner_runtime+'/plan.json',self.plan)
        self.session_ref=make_session(root,self.owner_runtime,self.graph_ref,self.graph,plan_ref=self.plan_ref,jobs=self.jobs)
        for index in range(k,192):
            previous=old.jobs[index];oldrow=old.records[index];key=subject._key(previous);route=routes[(key[0],key[2])]
            job=self.jobs[index-k]
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
            expected_cases=192-k,automatic_retry=False,source_outcome_filtering=False,jobs=self.jobs,
            predecessor=deepcopy(self.predecessor),profile=self.profile,CPU_affinity=self.affinity,**self.stamp)
        self.plan_ref=write(root,self.owner_runtime+'/plan.json',self.plan)
        process=dict(pid=2147482000,start_ticks=80000)
        launch=write(root,self.owner_runtime+'/launch.json',dict(process=process,
            source=self.graph['control']['components']['train_owner'],plan=self.plan_ref,resource_session=self.session_ref,**self.stamp))
        terminal=write(root,self.owner_runtime+'/terminal.json',dict(complete=True,completed=192-k,attempted=192-k,
            infrastructure_failed=0,active=None,automatic_retry=False,process=process,plan=self.plan_ref,resource_session=self.session_ref,**self.stamp))
        exit=write(root,self.owner_runtime+'/exec_completion.json',dict(process=process,actual_tool_exit_observed=True,
            natural_completion=True,returncode=0,plan=self.plan_ref,terminal=terminal))
        self.lineage=dict(schema=subject.LINEAGE_SCHEMA,complete=True,mixed_control_lineage=True,automatic_retry=False,
            source_outcome_filtering=False,case_graphs_relabelled=False,predecessor_exit=self.exit_ref,suffix_plan=self.plan_ref,
            suffix_owner=dict(launch=launch,terminal=terminal,exec_completion=exit),records=self.records,**self.stamp)
        self.refresh()
    def refresh(self):self.lineage_ref=write(self.root,self.owner_runtime+'/lineage.json',self.lineage)
    def audit(self):
        with patch.object(subject,'_predecessor',return_value=deepcopy(self.predecessor)):
            return subject.audit_mixed_train192(self.root,self.graph_ref,self.lineage_ref)


class MixedContracts(unittest.TestCase):
    def setUp(self):
        self.temp=TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
    def test_all192_real_json_cases_two_graphs_and_all_false_pass(self):
        f=MixedFixture(self.root);proof=f.audit()
        self.assertEqual(proof['completed'],192);self.assertTrue(proof['mixed_control_lineage'])
        self.assertFalse(any(row['native_success'] for row in proof['cases']))
        self.assertEqual(proof['preserved_prefix_count'],2)
        self.assertEqual(f.old.graph['physical_policy_sha256'],f.graph['physical_policy_sha256'])
        self.assertNotEqual(f.old.graph['control_graph_sha256'],f.graph['control_graph_sha256'])
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


if __name__=='__main__':unittest.main()
