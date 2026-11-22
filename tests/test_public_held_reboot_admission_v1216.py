"""CPU source-gate contracts; capabilities and TRAIN proofs are synthetic."""
from contextlib import ExitStack
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

from our_mani.multitask import public_held_reboot_admission_v1216 as subject
from tests.test_public_held_repaired_readiness_v1208 import make_previous_closed_TRAIN_fixture


def make_development_ref(root):
    # These gate tests mock the complete readiness reaudit. This is explicitly
    # a temporary metadata reference, not a fabricated actual pilot summary.
    file=Path(root)/'runtime/framework/synthetic_diagnostic/development_summary.json'
    file.parent.mkdir(parents=True,exist_ok=True)
    file.write_text(json.dumps(dict(synthetic_only=True,actual_pilot_or_qualification_evidence=False)))
    return subject.point(root,file)


def fresh_disclosure(previous,development):
    # The complete1215 audit is mocked in these temporary gate contracts.
    recovery=dict(path='runtime/framework/reboot_recovery_20261024/interrupted_receipts_audit.json',sha256='a'*64)
    return dict(previous_closed_TRAIN=deepcopy(previous),**subject.RECOVERY_FIELDS,
        recovery_audit=recovery,development_diagnostic=deepcopy(development),
        TRAIN_development_results_used_for_repair_adoption=True,
        fixed_TRAIN192_sources_selected_using_outcomes=False,
        full192_uniform_requalification=True,new_independent_dataset_claimed=False)


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
        guard_name='our_mani/multitask/public_held_repaired_resource_guard_v1213.py'
        guard=self.root/guard_name;guard.write_text('synthetic frozen capability provider')
        self.graph['control']['components']['resource_guard']=subject.point(self.root,guard)
        self.proof=dict(schema=subject.READINESS_SCHEMA,graph=self.graph_ref,mixed_control_lineage=True,case_graphs_relabelled=False,reused_cases=[],
            **fresh_disclosure(make_previous_closed_TRAIN_fixture(self.root),make_development_ref(self.root)),
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
            **{key:admission[key] for key in subject.DISCLOSURE_FIELDS},mixed_control_lineage=True,
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

    def test_previous_history_is_required_but_cannot_replace_new_TRAIN192(self):
        for bad in ('missing','wrong_shape','missing_new192'):
            proof=deepcopy(self.proof)
            if bad=='missing':proof.pop('previous_closed_TRAIN')
            elif bad=='wrong_shape':proof['previous_closed_TRAIN']={}
            else:proof['completed']=0
            with self.subTest(bad=bad),patch.object(subject,'_readiness',return_value=proof),\
                 patch.object(subject,'_manifest',side_effect=AssertionError('VAL before fresh proof and closed history')):
                with self.assertRaises(ValueError):self.create(bad)

    def test_old_prefix_or_mixed_handoff_cannot_qualify_new_physical_policy(self):
        for field,bad in (('fresh_TRAIN192',False),('preserved_prefix_count',74),
                ('preserved_parallel_count',35),('preserved_parallel_count',False),
                ('control_segment_count',3),('control_segment_count',True),
                ('historical_results_counted_in_accepted192',True)):
            proof=deepcopy(self.proof)
            proof[field]=bad
            with self.subTest(field=field,bad=bad),patch.object(subject,'_readiness',return_value=proof),\
                 patch.object(subject,'_manifest',side_effect=AssertionError('VAL before fresh single-graph proof')):
                with self.assertRaises(ValueError):self.create(field)

    def test_development_adoption_must_be_disclosed_without_source_or_dataset_selection(self):
        for field,value in (('development_diagnostic',None),
                ('TRAIN_development_results_used_for_repair_adoption',False),
                ('fixed_TRAIN192_sources_selected_using_outcomes',True),
                ('full192_uniform_requalification',False),('new_independent_dataset_claimed',True),
                ('development_diagnostic',dict(path='artifacts/framework/protocol/validation_manifest.json',sha256='a'*64))):
            proof=deepcopy(self.proof);proof[field]=value
            with self.subTest(field=field,value=value),patch.object(subject,'_readiness',return_value=proof),\
                 patch.object(subject,'_manifest',side_effect=AssertionError('VAL before honest TRAIN diagnostic disclosure')):
                with self.assertRaises(ValueError):self.create(field)

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
        p=dict(path='src/actual1206.py',sha256='a'*64)
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
                ('reused_cases',[{'case':'old'}])]:
            proof=deepcopy(self.proof);proof[field]=value
            with patch.object(subject,'_readiness',return_value=proof),patch.object(subject,'_manifest',side_effect=AssertionError('VAL read')):
                with self.subTest(field=field),self.assertRaises(ValueError):self.create(field)

    def test_suite_cannot_drop_fresh_qualification_or_borrow_history_as_new_cases(self):
        reference=self.create();suite,files=self.suite(reference)
        with subject.admission_scope(self.root,self.graph_ref,reference,object()):
            for field in (*subject.DISCLOSURE_FIELDS,'mixed_control_lineage'):
                bad=deepcopy(suite);bad.pop(field)
                with self.subTest(field=field),patch.object(subject,'_manifest',side_effect=AssertionError('source read after dropped fresh contract')):
                    with self.assertRaises(ValueError):subject.read_formal_source(self.root,bad,files,'validation')

    def test_metadata_roles_reject_dataset_before_hashing(self):
        bad=dict(path='artifacts/framework/protocol/validation_manifest.json',sha256='a'*64)
        with patch.object(subject.graphs,'read_policy_graph',side_effect=AssertionError('must classify first')):
            with self.assertRaises(ValueError):subject._graph(self.root,bad)

    def test_reboot_recovery_counts_and_no_outcome_retry_are_strict_before_VAL(self):
        for field,expected in subject.RECOVERY_FIELDS.items():
            values=(not expected,) if type(expected) is bool else (expected+1,True)
            for bad in values:
                proof=deepcopy(self.proof);proof[field]=bad
                with self.subTest(field=field,bad=bad),patch.object(subject,'_readiness',return_value=proof),\
                     patch.object(subject,'_manifest',side_effect=AssertionError('VAL before correct reboot disclosure')):
                    with self.assertRaises(ValueError):self.create(field+str(bad))
        for value in (None,dict(path='artifacts/framework/protocol/test_manifest.json',sha256='a'*64),
                dict(path='runtime/framework/reboot_recovery_20261024/status.json',sha256='a'*64)):
            proof=deepcopy(self.proof);proof['recovery_audit']=value
            with patch.object(subject,'_readiness',return_value=proof),\
                 patch.object(subject,'_manifest',side_effect=AssertionError('VAL before typed recovery audit')):
                with self.assertRaises(ValueError):self.create('bad_recovery')

    def test_admission_and_reader_retain_TRAIN_mixed_and_VAL_single_control_roles(self):
        ref=self.create();admission=json.loads((self.root/ref['path']).read_text())
        self.assertTrue(admission['mixed_control_lineage'])
        self.assertEqual(admission['preserved_parallel_count'],85)
        self.assertEqual(admission['control_segment_count'],2)
        self.assertFalse(admission['validation_mixed_control_lineage'])
        self.assertEqual(admission['validation_control_segment_count'],1)
        suite,files=self.suite(ref)
        with subject.admission_scope(self.root,self.graph_ref,ref,object()):
            _,_,proof=subject.read_formal_source(self.root,suite,files,'validation')
        self.assertTrue(proof['mixed_control_lineage'])
        self.assertFalse(proof['NativeFalse_retried'])
        self.assertFalse(proof['interrupted_outcomes_inferred_as_False'])

    def test_new_metadata_writer_fsyncs_content_and_directory_without_overwriting(self):
        import os,stat
        file=self.root/'durable_admission.json';kinds=[];real=os.fsync
        def observe(fd):
            mode=os.fstat(fd).st_mode
            kinds.append('file' if stat.S_ISREG(mode) else 'directory' if stat.S_ISDIR(mode) else 'other')
            real(fd)
        with patch.object(subject.os,'fsync',side_effect=observe):
            subject._fresh(file,{'complete':True})
        self.assertEqual(kinds,['file','directory'])
        before=file.read_bytes();self.assertEqual(json.loads(before),{'complete':True})
        with self.assertRaises(FileExistsError):subject._fresh(file,{'complete':False})
        self.assertEqual(file.read_bytes(),before)
        for failing_call in (1,2):
            calls=[];error=OSError('durability test failure')
            def fail(fd):
                calls.append(fd)
                if len(calls)==failing_call:raise error
            with patch.object(subject.os,'fsync',side_effect=fail),self.assertRaises(OSError) as caught:
                subject._fresh(self.root/f'failed_fsync{failing_call}.json',{'synthetic':True})
            self.assertIs(caught.exception,error)


if __name__=='__main__':
    unittest.main()
