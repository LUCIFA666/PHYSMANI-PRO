"""CPU source-gate contracts; capabilities and TRAIN proofs are synthetic."""
from contextlib import ExitStack
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

from our_mani.multitask import parallel_repaired_admission_v1199 as subject
from tests.test_parallel_repaired_readiness_v1198 import (make_retired_development_fixture,retired_disclosure,
    make_handoff_fixture,handoff_disclosure)


class AdmissionContracts(unittest.TestCase):
    def setUp(self):
        self.directory=TemporaryDirectory();self.addCleanup(self.directory.cleanup)
        self.root=Path(self.directory.name)
        def write(name,value):
            path=self.root/name;path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text(json.dumps(value));return subject.point(self.root,path)
        self.write=write
        source=self.root/subject.SOURCE;source.parent.mkdir(parents=True);source.write_text('synthetic frozen source')
        current=subject.point(self.root,source)
        self.graph_ref=write('artifacts/framework/multitask_r10/test/policy_graph.json',{'schema':'synthetic graph'})
        self.proof_ref=write('runtime/framework/test/readiness.json',{'schema':'synthetic proof'})
        self.manifest_ref=write('artifacts/manifest.json',{'episodes':[]})
        original_root=Path(__file__).resolve().parents[1]
        inherited=self.root/subject.INHERITED_PREFLIGHT_WORKER['path'];inherited.parent.mkdir(parents=True,exist_ok=True)
        inherited.write_bytes((original_root/subject.INHERITED_PREFLIGHT_WORKER['path']).read_bytes())
        source=original_root/'artifacts/framework/multitask_r10/author_outcome_qualification_v745_setup/routes.json'
        profile=write('artifacts/profile.json',json.loads(source.read_text())['common_author_RML_profile'])
        self.graph=dict(policy_sha256='policy',physical_policy_sha256='physical',control_graph_sha256='control',
            control=dict(contracts=dict(readiness_schema=subject.READINESS_SCHEMA,
                admission_schema=subject.SCHEMA,split_source_lock_schema=subject.SPLIT_LOCK_SCHEMA),
                components=dict(admission_guard=current,formal_source_reader=current)),
            physical=dict(common_RML=dict(profile=profile),shared_context_checkpoint=dict(sha256='checkpoint')),
            routes=[])
        guard_name='our_mani/multitask/parallel_repaired_resource_guard_v1195.py'
        guard=self.root/guard_name;guard.write_text('synthetic frozen capability provider')
        self.graph['control']['components']['resource_guard']=subject.point(self.root,guard)
        self.proof=dict(schema=subject.READINESS_SCHEMA,graph=self.graph_ref,mixed_control_lineage=True,case_graphs_relabelled=False,old1144_full_same_graph_readiness_claimed=False,
            **retired_disclosure(make_retired_development_fixture(self.root)),
            **handoff_disclosure(make_handoff_fixture(self.root)),
            policy_sha256='policy',physical_policy_sha256='physical',control_graph_sha256='control',
            complete=True,completed=192,method_case_count=192,source_count=64,task_ready_count=16,
            all_failures_retained=True,native_success_is_not_an_admission_threshold=True,
            automatic_retry=False,source_outcome_filtering=False,VAL_read=False,TEST_read=False)
        self.rows=[dict(task='synthetic',raw_episode_id=str(i)) for i in range(315)]
        stack=ExitStack();self.addCleanup(stack.close)
        stack.enter_context(patch.object(subject,'VALIDATION_MANIFEST',self.manifest_ref))
        self.permit=stack.enter_context(patch.object(subject,'_permit',return_value={'actual':'synthetic permit'}))
        stack.enter_context(patch.object(subject.graphs,'read_policy_graph',side_effect=lambda *args:deepcopy(self.graph)))
        self.readiness=stack.enter_context(patch.object(subject,'_readiness',side_effect=lambda *args:deepcopy(self.proof)))
        stack.enter_context(patch.object(subject,'validate_roster',return_value=self.rows))
        self.parent=self.root/'artifacts/framework/multitask_r10';self.parent.mkdir(parents=True,exist_ok=True)

    def create(self,name='new_admission'):
        return subject.create_admission(self.root,self.graph_ref,self.proof_ref,self.manifest_ref,
            namespace='artifacts/framework/multitask_r10/'+name,permit=object())

    def suite(self,reference):
        admission=json.loads((self.root/reference['path']).read_text())
        suite=dict(schema=subject.SUITE_SCHEMA,graph=self.graph_ref,admission=reference,
            source_manifests=admission['source_manifests'],routes=admission['routes'],actual_routes=admission['actual_routes'],preflight_routes=admission['preflight_routes'],inherited742_preflight_entry=subject.INHERITED_PREFLIGHT_WORKER)
        refs=[self.graph_ref,reference,admission['actual_routes'],admission['preflight_routes'],subject.INHERITED_PREFLIGHT_WORKER,self.manifest_ref]
        return suite,{str(self.root/ref['path']):ref['sha256'] for ref in refs}

    def test_capability_then_full_train_proof_precede_first_manifest_read(self):
        order=[]
        self.permit.side_effect=lambda *args:order.append('permit')
        self.readiness.side_effect=lambda *args:(order.append('train192') or deepcopy(self.proof))
        original=subject._manifest
        with patch.object(subject,'_manifest',side_effect=lambda *args:(order.append('manifest') or original(*args))):
            reference=self.create()
        self.assertLess(order.index('permit'),order.index('train192'))
        self.assertLess(order.index('train192'),order.index('manifest'))
        value=json.loads((self.root/reference['path']).read_text())
        self.assertFalse(value['formal_TEST_permitted']);self.assertFalse(value['new_independent_validation_dataset'])

    def test_retired_development_evidence_is_required_before_first_VAL_read(self):
        for bad in ('missing','filtered','counted','hidden_repeat','wrong_count'):
            proof=deepcopy(self.proof)
            if bad=='missing':proof.pop('retired_development_batch')
            elif bad=='filtered':proof['retired_development_batch']['Native_development_records'].pop()
            elif bad=='counted':proof['retired_development_batch']['accepted_into_new_cohort']=True
            elif bad=='hidden_repeat':proof['historical_Native_reexecution_disclosed']=False
            else:proof['repeated_development_Native_cases']=True
            with self.subTest(bad=bad),patch.object(subject,'_readiness',return_value=proof),\
                 patch.object(subject,'_manifest',side_effect=AssertionError('VAL before honest retired history')):
                with self.assertRaises(ValueError):self.create(bad)

    def test_preserved_parallel_handoff_is_required_before_first_VAL_read(self):
        for bad in ('missing','dropped_middle','reexecution','wrong_count','wrong_segments'):
            proof=deepcopy(self.proof)
            if bad=='missing':proof.pop('parallel_handoff')
            elif bad=='dropped_middle':proof['parallel_handoff']['completed_records'].pop()
            elif bad=='reexecution':proof['parallel_handoff']['completed_records_reused_without_reexecution']=False
            elif bad=='wrong_count':proof['preserved_parallel_count']=True
            else:proof['control_segment_count']=2
            with self.subTest(bad=bad),patch.object(subject,'_readiness',return_value=proof),\
                 patch.object(subject,'_manifest',side_effect=AssertionError('VAL before full three-segment proof')):
                with self.assertRaises(ValueError):self.create(bad)

    def test_missing_capability_or_incomplete_or_different_graph_never_reads_val(self):
        for bad in ('permit','missing192','policy','graph','filter','retry','test_read'):
            with self.subTest(bad=bad),patch.object(subject,'_manifest',side_effect=AssertionError('VAL read before admission')):
                proof=deepcopy(self.proof)
                if bad=='missing192':proof['completed']=191
                if bad=='policy':proof['policy_sha256']='different'
                if bad=='graph':proof['graph']=dict(self.graph_ref,sha256='different')
                if bad=='filter':proof['source_outcome_filtering']=True
                if bad=='retry':proof['automatic_retry']=True
                if bad=='test_read':proof['TEST_read']=True
                with patch.object(subject,'_readiness',return_value=proof),patch.object(subject,'_permit',
                    side_effect=ValueError('no capability') if bad=='permit' else None):
                    with self.assertRaises(ValueError):self.create(bad)
                self.assertFalse((self.parent/bad).exists())

    def test_no_scope_or_test_or_wrong_suite_or_lock_cannot_read(self):
        reference=self.create();suite,files=self.suite(reference)
        with self.assertRaises(ValueError):subject.read_formal_source(self.root,suite,files,'validation')
        with subject.admission_scope(self.root,self.graph_ref,reference,object()):
            for split in ('test','train'):
                with self.assertRaises(ValueError):subject.read_formal_source(self.root,suite,files,split)
            with self.assertRaises(ValueError):subject.read_formal_source(self.root,dict(suite,schema='old532'),files,'validation')
            with self.assertRaises(ValueError):subject.read_formal_source(self.root,suite,{},'validation')
            with self.assertRaises(ValueError):subject.read_formal_source(self.root,suite,files,'validation',validation_release={})

    def test_source_read_has_real_eval_purpose_and_scope_restores_after_exception(self):
        reference=self.create();suite,files=self.suite(reference)
        error=RuntimeError('original policy exception')
        with self.assertRaises(RuntimeError) as raised:
            with subject.admission_scope(self.root,self.graph_ref,reference,object()):
                manifest,rows,proof=subject.read_formal_source(self.root,suite,files,'validation')
                self.assertEqual(len(rows),315);self.assertEqual(proof['policy_sha256'],'policy')
                self.assertEqual(proof['component_purpose'],'evaluation')
                self.assertEqual(subject.admission_metrics()['admitted_manifest_reads'],1)
                raise error
        self.assertIs(raised.exception,error);self.assertFalse(subject.admission_metrics()['active'])

    def test_changed_routes_reader_or_manifest_and_repeated_namespace_reject(self):
        reference=self.create();suite,files=self.suite(reference)
        with self.assertRaises(FileExistsError):self.create()
        saved=deepcopy(self.graph)
        self.graph['control']['components']['formal_source_reader']['sha256']='changed'
        with self.assertRaises(ValueError):
            with subject.admission_scope(self.root,self.graph_ref,reference,object()):pass
        self.graph.clear();self.graph.update(saved)
        (self.root/self.manifest_ref['path']).write_text('{}')
        with subject.admission_scope(self.root,self.graph_ref,reference,object()):
            with self.assertRaises(ValueError):subject.read_formal_source(self.root,suite,files,'validation')

    def test_nested_scope_rejected_without_disturbing_original_scope(self):
        reference=self.create()
        with subject.admission_scope(self.root,self.graph_ref,reference,object()):
            with self.assertRaises(ValueError):
                with subject.admission_scope(self.root,self.graph_ref,reference,object()):pass
            self.assertTrue(subject.admission_metrics()['active'])
        self.assertFalse(subject.admission_metrics()['active'])

    def test_wrong_manifest_pointer_is_rejected_before_reading_its_bytes(self):
        for reference in (dict(path='artifacts/framework/protocol/test_manifest.json',sha256='0'*64),
                          dict(self.manifest_ref,sha256='1'*64)):
            with self.subTest(reference=reference),patch.object(subject,'_read',side_effect=AssertionError('wrong source opened')):
                with self.assertRaises(ValueError):subject._manifest(self.root,reference)

    def test_graph_source_change_after_scope_entry_precedes_and_blocks_manifest_read(self):
        reference=self.create();suite,files=self.suite(reference)
        with subject.admission_scope(self.root,self.graph_ref,reference,object()):
            self.graph['control_graph_sha256']='changed'
            with patch.object(subject,'_manifest',side_effect=AssertionError('VAL read after graph changed')):
                with self.assertRaises(ValueError):subject.read_formal_source(self.root,suite,files,'validation')

    def test_actual_and_preflight_projection_differ_only_worker(self):
        p=dict(path='src/actual1196.py',sha256='a'*64)
        self.graph['routes']=[dict(task='synthetic',method='cv_geometry',GPU_required=False,
            configuration=p,frozen_inputs=p,physical_factory='core',evaluation=dict(worker=p,preparer=p,consumer=p))]
        actual=subject.routes_from_graph(self.root,self.graph)
        projected=subject.preflight_routes_from_graph(self.root,self.graph)
        self.assertEqual(projected['routes'][0]['worker'],subject.INHERITED_PREFLIGHT_WORKER)
        normalized=deepcopy(projected);normalized['routes'][0]['worker']=p
        self.assertEqual(normalized,actual)
        reference=self.create();admission=json.loads((self.root/reference['path']).read_text())
        route=json.loads((self.root/admission['preflight_routes']['path']).read_text())
        route['routes'][0]['configuration']['sha256']='b'*64
        admission['preflight_routes']=self.write(admission['preflight_routes']['path'],route)
        admission['routes']=admission['preflight_routes'];reference=self.write(reference['path'],admission)
        with self.assertRaises(ValueError):
            with subject.admission_scope(self.root,self.graph_ref,reference,object()):pass

    def test_relabelled_or_old_full192_claim_never_reads_manifest(self):
        for field,value in [('mixed_control_lineage',False),('case_graphs_relabelled',True),
                ('old1144_full_same_graph_readiness_claimed',True)]:
            proof=deepcopy(self.proof);proof[field]=value
            with patch.object(subject,'_readiness',return_value=proof),patch.object(subject,'_manifest',side_effect=AssertionError('VAL read')):
                with self.subTest(field=field),self.assertRaises(ValueError):self.create(field)

    def test_metadata_roles_reject_dataset_before_hashing(self):
        bad=dict(path='artifacts/framework/protocol/validation_manifest.json',sha256='a'*64)
        with patch.object(subject.graphs,'read_policy_graph',side_effect=AssertionError('must classify first')):
            with self.assertRaises(ValueError):subject._graph(self.root,bad)


if __name__=='__main__':
    unittest.main()
