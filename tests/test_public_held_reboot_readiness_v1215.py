"""Synthetic CPU proof negatives; these dictionaries are never scientific evidence."""
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from our_mani.multitask import public_held_reboot_readiness_v1215 as r
from scripts import run_public_held_reboot_owner_v1214_r10 as o


def write(root,name,value):
    o.fresh(Path(root)/name,value);return o.pointer(root,name)


def proof_fixture(root):
    base='runtime/framework/public_held_repaired_owner_v1207/train/synthetic_reboot'
    plan=write(root,base+'/plan.json',{'synthetic':True})
    graph_ref=dict(root='project',path='artifacts/framework/multitask_r10/synthetic/policy_graph.json',sha256='a'*64)
    owner=dict(root='project',path=o.SOURCE,sha256='b'*64)
    graph=dict(control={'components':{'train_owner':owner}},**{k:str(i+1)*64 for i,k in enumerate(r.HASHES)})
    stamp=dict(graph=graph_ref,**{k:graph[k] for k in r.HASHES})
    process=dict(pid=2147483000,start_ticks=99999999999)
    launch=write(root,base+'/launch.json',dict(source=owner,plan=plan,process=process,**stamp))
    terminal=write(root,base+'/terminal.json',dict(complete=True,completed=107,attempted=107,infrastructure_failed=0,
        active=None,automatic_retry=False,drain_reason=None,remaining_job_ids=[],process=process,plan=plan,**stamp))
    actual=dict(actual_tool_exit_observed=True,natural_completion=True,returncode=0,process=process,plan=plan,terminal=terminal)
    return base,plan,graph_ref,graph,dict(launch=launch,terminal=terminal),actual


def run_proof(root,changes=None,terminal_changes=None):
    b,p,gref,g,refs,actual=proof_fixture(root)
    if terminal_changes:
        tpath=Path(root)/refs['terminal']['path'];t=json.loads(tpath.read_text());t.update(terminal_changes)
        tpath.write_text(json.dumps(t));refs['terminal']=o.pointer(root,refs['terminal']['path']);actual['terminal']=refs['terminal']
    actual.update(changes or {})
    refs['exec_completion']=write(root,b+'/exec_completion.json',actual)
    return r._suffix_owner(root,refs,p,gref,g,107)


class NaturalOwnerProofTests(unittest.TestCase):
    def test_current_suffix_requires_own_actual_natural_exit0(self):
        with TemporaryDirectory() as d:self.assertEqual(run_proof(d)['exec_completion']['returncode'],0)

    def test_absent_tool_observation_or_invented_completion_rejected(self):
        for changed in ({'actual_tool_exit_observed':False},{'natural_completion':False},{'returncode':None},{'returncode':1},{'returncode':False}):
            with TemporaryDirectory() as d,self.subTest(changed=changed),self.assertRaises(ValueError):run_proof(d,changed)

    def test_borrowing_old_owner_process_rejected(self):
        with TemporaryDirectory() as d,self.assertRaises(ValueError):run_proof(d,{'process':dict(pid=294392,start_ticks=1290342)})

    def test_incomplete_suffix_or_infrastructure_failures_rejected(self):
        for changed in ({'completed':106},{'attempted':106},{'complete':False},{'infrastructure_failed':1},
                        {'remaining_job_ids':['j0191']},{'active':[]},{'drain_reason':'reboot'}):
            with TemporaryDirectory() as d,self.subTest(changed=changed),self.assertRaises(ValueError):run_proof(d,terminal_changes=changed)

    def test_terminal_pointer_cannot_be_replaced_with_old_owner_terminal(self):
        with TemporaryDirectory() as d:
            b,p,gref,g,refs,actual=proof_fixture(d)
            actual['terminal']=dict(root='project',path='runtime/framework/old/terminal.json',sha256='a'*64)
            refs['exec_completion']=write(d,b+'/exec_completion.json',actual)
            with self.assertRaises(ValueError):r._suffix_owner(d,refs,p,gref,g,107)


class RebootDisclosureTests(unittest.TestCase):
    def valid(self):
        return dict(**r._fresh_fields(),**r._adoption_fields(),mixed_control_lineage=True,
            case_graphs_relabelled=False,reused_cases=[],automatic_retry=False,source_outcome_filtering=False)

    def test_exact_reboot_disclosure_accepted(self):r._require_fresh(self.valid())

    def test_false_single_graph_claim_or_different_counts_rejected(self):
        for key,value in (('mixed_control_lineage',False),('preserved_parallel_count',0),('control_segment_count',1),
                          ('preserved_current_physical_records',86),('explicit_host_reboot_recovery',False),
                          ('infrastructure_reexecution',False),('historical_results_counted_in_accepted192',True),
                          ('validation_control_segment_count',2),('validation_mixed_control_lineage',True)):
            data=self.valid();data[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):r._require_fresh(data)

    def test_false_retries_relabelling_or_unknown_outcome_inference_rejected(self):
        for key,value in (('NativeFalse_retried',True),('interrupted_outcomes_inferred_as_False',True),
                          ('case_graphs_relabelled',True),('automatic_retry',True),('source_outcome_filtering',True),
                          ('new_independent_dataset_claimed',True),('reused_cases',['j0000'])):
            data=self.valid();data[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):r._require_fresh(data)

    def test_boolean_does_not_impersonate_integer_counts(self):
        for key in ('control_segment_count','preserved_parallel_count','validation_control_segment_count'):
            data=self.valid();data[key]=True
            with self.subTest(key=key),self.assertRaises(ValueError):r._require_fresh(data)

    def test_readiness_saved_result_requires_literal_reaudit_equality(self):
        with TemporaryDirectory() as d:
            saved=dict(schema=r.READINESS_SCHEMA,lineage={'synthetic':True},completed=192)
            ref=write(d,'runtime/framework/synthetic/readiness.json',saved)
            with patch.object(r,'audit_train192',return_value=deepcopy(saved)):
                self.assertEqual(r.require_readiness_proof(d,{'synthetic':True},ref),saved)
            changed=dict(saved,completed=191)
            with patch.object(r,'audit_train192',return_value=changed),self.assertRaises(ValueError):r.require_readiness_proof(d,{},ref)




# Reuse the original full192 synthetic receipt fixture and its actual offline
# model/phase builders. Only the new control entry, namespace and job complement
# differ below; no generic scientific auditor is mocked by these tests.
from tests.test_public_held_repaired_readiness_v1208 import (FreshParallelFixture,
    make_previous_closed_TRAIN_fixture,make_development_diagnostic_fixture,
    make_CPU_affinity_fixture,add_broker_sources,add_resource_phase,make_session as old_make_session)
from tests.test_repaired_readiness_v1144 import FreshFixture,write
from tests.test_repaired_policy_graph_v1143 import put
from our_mani.multitask import repaired_policy_graph_v1143 as graphs
subject=r


def make_session(*args,**kwargs):
    ref=old_make_session(*args,**kwargs);root=args[0]
    value=json.loads((Path(root)/ref['path']).read_text())
    value['boot_id']='00000000-0000-0000-0000-000000000002'
    value['parent_broker']['boot_id']=value['boot_id']
    return write(root,ref['path'],value)

class RebootSuffixFixture:
    def __init__(self,root,accepted_ids,underlying_fixture):
        self.root=Path(root);self.accepted_ids=accepted_ids;old=underlying_fixture;self.old=old
        spec={n:deepcopy(old.graph[n]) for n in ('tasks','methods','train_source_manifest','physical','control','routes')}
        spec.update(schema=graphs.SPEC_SCHEMA,external_roots={n:v for n,v in old.graph['roots'].items() if n!='project'})
        auditor=put(root,subject.SOURCE,Path(subject.__file__).read_bytes())
        owner=put(root,subject.OWNER_SOURCE,b'synthetic new owner')
        worker=put(root,'scripts/run_public_held_reboot_case_v1220_r10.py',b'synthetic new worker')
        preparer=put(root,'our_mani/multitask/public_held_reboot_case_v1220.py',b'synthetic new preparer')
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
        self.graph_ref=graphs.write_policy_graph(root,spec,namespace='artifacts/framework/multitask_r10/new_reboot_graph')
        self.graph=graphs.read_policy_graph(root,self.graph_ref)
        self.stamp=dict(graph=self.graph_ref,**{key:self.graph[key] for key in subject.HASHES})
        self.owner_runtime='runtime/framework/public_held_repaired_owner_v1207/train/synthetic_reboot'
        manifest,_=subject.read_json(root,self.graph['train_source_manifest']);sources=subject.base._sources(manifest)
        routes={(row['task'],row['method']):row for row in self.graph['routes']}
        self.jobs=[];self.records=[];self.history=make_previous_closed_TRAIN_fixture(root)
        self.development=make_development_diagnostic_fixture(root)
        for index,previous in enumerate(old.jobs):
            key=subject._key(previous);route=routes[(key[0],key[2])]
            suffix='cohorts/synthetic_reboot/'+key[0]+'/'+key[1]+'/'+key[2]
            self.jobs.append(dict(job_id=f'j{index:04d}',task=key[0],raw_episode_id=key[1],method=key[2],split='train',
                configuration=route['configuration'],source_row_sha256=subject.digest(sources[key[:2]]),
                out='artifacts/framework/multitask_r10/public_held_repaired_policy_v1206/train/development_demo_only/'+suffix,
                runtime='runtime/framework/public_held_repaired_policy_v1206/train/'+suffix))
        self.jobs=[j for j in self.jobs if j['job_id'] not in accepted_ids]
        self.profile=dict(max_native_slots=12,max_GPU_slots=5,max_CPU_preflight_slots=6,ready_limit=24)
        self.plan=dict(schema=subject.SUFFIX_PLAN_SCHEMA,split='train',namespace='synthetic_reboot',owner_runtime=self.owner_runtime,
            global_expected_cases=192,expected_cases=107,new_suffix_count=107,predecessor=None,validation_bundle=None,
            sources=list(sources.values()),jobs=self.jobs,profile=self.profile,CPU_affinity=self.affinity,
            previous_closed_TRAIN=self.history,development_diagnostic=self.development,
            mixed_control_lineage=True,case_graphs_relabelled=False,reused_cases=[],
            automatic_retry=False,source_outcome_filtering=False,**subject._adoption_fields(),**subject._fresh_fields(),**self.stamp)
        self.plan_ref=write(root,self.owner_runtime+'/plan.json',self.plan)
        self.owner_process=dict(pid=2147482100,start_ticks=81000)
        self.session_ref=make_session(root,self.owner_runtime,self.graph_ref,self.graph,
            owner=self.owner_process,plan_ref=self.plan_ref,jobs=self.jobs)
        for job in self.jobs:
            oldrow=old.records[int(job['job_id'][1:])];key=subject._key(job);route=routes[(key[0],key[2])]
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
            terminal=write(root,self.owner_runtime+'/terminal.json',dict(complete=True,completed=107,attempted=107,infrastructure_failed=0,
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
            attempted=107,completed=self.records))
        self.lineage_ref=write(self.root,self.owner_runtime+'/lineage.json',self.lineage)
    def audit(self):
        with patch.object(subject,'_previous_closed_TRAIN',return_value=deepcopy(self.history)),\
                patch.object(subject,'_development_diagnostic',return_value=deepcopy(self.development)):
            return subject.audit_train192(self.root,self.graph_ref,self.lineage_ref)


class FullCompositeFixture:
    def __init__(self,root):
        self.root=Path(root);self.old=FreshParallelFixture(root)
        # Select by a fixed ID pattern, not synthetic success (all are False).
        self.preserved=[row for i,row in enumerate(self.old.records) if i<53 or 54<=i<86]
        ids={row['job_id'] for row in self.preserved}
        self.new=RebootSuffixFixture(root,ids,self.old.old)
        f=self.new
        self.recovery_ref=write(root,'runtime/framework/reboot_recovery_20261024/interrupted_receipts_audit.json',{'synthetic':True})
        self.recovery=dict(audit=self.recovery_ref,accepted_records=self.preserved,accepted_job_ids=sorted(ids),
            remaining_job_ids=[j['job_id'] for j in f.jobs],old_plan=self.old.plan_ref,old_graph=self.old.graph_ref,
            old_session=self.old.session_ref,old_boot_id='00000000-0000-0000-0000-000000000001',
            current_boot_id='00000000-0000-0000-0000-000000000002')
        f.plan.update(recovery_audit=self.recovery_ref,mixed_control_lineage=True)
        # The plan pointer is already source-bound into the synthetic session.
        # Keep its bytes stable: the union entry receives this parsed loader result.
        merged={row['job_id']:row for row in self.preserved+f.records}
        self.lineage=dict(schema=r.LINEAGE_SCHEMA,complete=True,plan=f.plan_ref,owner=f.owner_refs,
            records=[merged[f'j{i:04d}'] for i in range(192)],suffix_records=f.records,
            preserved_records=self.preserved,previous_closed_TRAIN=f.history,development_diagnostic=f.development,
            recovery_audit=self.recovery_ref,mixed_control_lineage=True,case_graphs_relabelled=False,reused_cases=[],
            automatic_retry=False,source_outcome_filtering=False,**r._adoption_fields(),**r._fresh_fields(),**f.stamp)
        self.refresh()

    def refresh(self):self.lineage_ref=write(self.root,self.new.owner_runtime+'/lineage.json',self.lineage)

    def audit(self):
        f=self.new
        with patch.object(o,'load_plan',return_value=(f.plan,f.plan_ref,f.graph)),\
                patch.object(o,'inspect_recovery_audit',return_value=deepcopy(self.recovery)),\
                patch.object(r,'_previous_closed_TRAIN',return_value=deepcopy(f.history)),\
                patch.object(r,'_development_diagnostic',return_value=deepcopy(f.development)):
            return r.audit_train192(self.root,f.graph_ref,self.lineage_ref)


class FullCompositeContracts(unittest.TestCase):
    def test_full192_union_reaudits_all_model_and_phase_receipts_and_rejects_damage(self):
        with TemporaryDirectory() as d:
            f=FullCompositeFixture(d);result=f.audit()
            self.assertEqual(result['completed'],192);self.assertEqual(result['new_suffix_count'],107)
            self.assertEqual(result['preserved_current_physical_records'],85)
            self.assertTrue(result['mixed_control_lineage']);self.assertFalse(result['old_owner_natural_exit0_claimed'])
            self.assertTrue(all(row['native_success'] is False for row in result['cases']))
            self.assertEqual(sum(r.same(r.pointer(row['case_graph']),r.pointer(f.old.graph_ref)) for row in result['cases']),85)
            self.assertEqual(sum(r.same(r.pointer(row['case_graph']),r.pointer(f.new.graph_ref)) for row in result['cases']),107)
            original=deepcopy(f.lineage)
            for kind in ('missing','duplicate','relabel'):
                f.lineage=deepcopy(original)
                if kind=='missing':f.lineage['records'].pop()
                elif kind=='duplicate':f.lineage['records'][1]=deepcopy(f.lineage['records'][0])
                else:f.lineage['preserved_records'][0]['case']=f.new.records[0]['case']
                f.refresh()
                with self.subTest(kind=kind),self.assertRaises(ValueError):f.audit()
            f.lineage=deepcopy(original);f.refresh()
            # Fail a genuine generic model receipt, rather than its outer loader.
            row=f.new.records[0];sidecar,_=r.read_json(f.root,row['CPU_check']['sidecar'])
            sidecar['execution']['source_and_model_preflight']['GPU_started']=True
            new_ref=write(f.root,row['CPU_check']['sidecar']['path'],sidecar)
            for collection in (f.new.records,f.lineage['suffix_records'],f.lineage['records']):
                for target in collection:
                    if target['job_id']==row['job_id']:target['CPU_check']['sidecar']=new_ref
            write(f.root,f.new.owner_runtime+'/status.json',dict(stage='complete',active=None,failed=[],attempted=107,completed=f.new.records))
            f.refresh()
            with self.assertRaisesRegex(ValueError,'actual TRAIN CPU loader'):f.audit()


if __name__=='__main__':unittest.main()
