"""CPU/temp assertions for1041; production scheduler/resources never run."""
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch
import ast
import importlib.util
import json
import tempfile
import threading
import unittest

from scripts import run_reboot_TEST_owner_v1056_r10 as owner
from scripts import run_composite_TEST_owner_v1041_r10 as base

spec=importlib.util.spec_from_file_location('_composite956fixture',Path(__file__).with_name('test_composite_TEST_resume_v1040.py'))
fixtures=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixtures)


class OwnerTests(unittest.TestCase):
    def test_corrupt_saved_success_and_failure_both_excluded_without_changing_old_status(self):
        status={'completed':{'valid_false':{'native_success':False},'corrupt_true':{'native_success':True},'corrupt_false':{'native_success':False}}}
        before=deepcopy(status)
        control={'completed_job_ids':['valid_false'],'rejected_saved_completed_job_ids':['corrupt_true','corrupt_false']}
        adopted=owner.adopt_completed(control,status)
        self.assertEqual(adopted,{'valid_false':{'native_success':False}})
        adopted['valid_false']['native_success']=True
        self.assertEqual(status,before)
        for bad in (['valid_false','corrupt_true'],['valid_false','corrupt_false'],['valid_false','valid_false'],['missing']):
            with self.subTest(bad=bad),self.assertRaises(ValueError):owner.adopt_completed(dict(control,completed_job_ids=bad),status)

    def test_native_evidence_is_fsynced_without_rewriting_its_bytes(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);runtime=root/'run'
            row={'job_id':'test','physical_argv':['bash','p',owner.ORIGINAL_ENTRY,'--out','output']}
            leaves=[root/'output',runtime/'CPU/test',runtime/'native/test']
            for leaf in leaves:
                leaf.mkdir(parents=True);(leaf/'proof.json').write_text('{"native":false}')
            before={p:p.read_bytes() for leaf in leaves for p in leaf.iterdir()}
            seen=[]
            original=owner.fsync_path
            def sync(path):seen.append(Path(path));original(path)
            with patch.object(owner,'ROOT',root),patch.object(owner,'fsync_path',side_effect=sync):
                owner.persist_native_evidence(row,runtime)
            self.assertTrue(set(before)<=set(seen))
            self.assertTrue(set(leaves)<=set(seen))
            for leaf in leaves:
                parent=leaf.parent
                while True:
                    self.assertIn(parent,seen)
                    if parent==root:break
                    parent=parent.parent
            self.assertEqual({p:p.read_bytes() for p in before},before)
            (leaves[0]/'alias').symlink_to(leaves[1]/'proof.json')
            with patch.object(owner,'ROOT',root),self.assertRaises(ValueError):owner.persist_native_evidence(row,runtime)

    def test_atomic_status_sync_failure_propagates_as_infrastructure(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'status.json'
            path.write_text('{"completed":1}')
            with patch.object(owner,'fsync_path',side_effect=OSError('simulated disk sync failure')),self.assertRaises(OSError):
                owner.atomic_json(path,{'completed':2},overwrite=True)
            self.assertEqual(json.loads(path.read_text()),{'completed':1})

    def test_physical_child_quota_and_receipt_functions_remain_original902_AST(self):
        def bodies(module):
            return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(Path(module.__file__).read_text()).body
                if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
        old,new=bodies(base),bodies(owner)
        for name in ('child','select_native_jobs','future_result','terminal_is_complete','require_batch_execution',
                'require_batch_metrics','execution_implementation','accept_CPU','accept_native'):
            self.assertEqual(new[name],old[name],name)

    def test_invalid_external_lock_SHA_rejected_before_parse(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'inputs.json';path.write_bytes(b'{not JSON')
            with patch.object(owner.json,'loads',side_effect=AssertionError('no parse')):
                with self.assertRaises(ValueError):owner.read_pinned_resume_lock(path,'0'*64)

    def test_bound_JSON_parses_exact_matching_bytes_and_rejects_change(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);path=root/'record.json';path.write_bytes(b'{"literal":false}')
            pointer=dict(path='record.json',sha256=owner.file_sha(path))
            with patch.object(owner,'ROOT',root),patch.object(owner,'locked_pointer'):
                self.assertEqual(owner.read_bound_json({},pointer),{'literal':False})
                path.write_bytes(b'{"literal":true}')
                with patch.object(owner.json,'loads',side_effect=AssertionError('must not parse changed bytes')):
                    with self.assertRaises(ValueError):owner.read_bound_json({},pointer)

    def test_dispatch_global_or_historical_STOP_cancels_but_DRAIN_only_drains(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);runtime=root/owner.RUNTIME;runtime.mkdir(parents=True)
            with patch.object(owner,'ROOT',root):
                event=threading.Event();(runtime/'DRAIN').touch()
                self.assertTrue(owner.dispatch_control(runtime,event,False));self.assertFalse(event.is_set())
                old=root/owner.PREVIOUS/'STOP';old.parent.mkdir(parents=True);old.touch()
                owner.dispatch_control(runtime,event,False);self.assertTrue(event.is_set())

    def test_nativeFalse_and_unknown_motion_preserved_with_original_case_bytes(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);out=root/'caseout';out.mkdir()
            case=dict(official_task_success=False,cleanup_completed=True,
                component_domain=dict(purpose='evaluation',split='test'),
                full16_frozen_execution=dict(policy_sha256='e'*64),counted_clock=dict(ticks=0))
            (out/'case.json').write_text(json.dumps(case));(out/'admission_driver_receipt.json').write_text('{}')
            row=dict(physical_argv=['bash','p',owner.ORIGINAL_ENTRY,'--out','caseout'],
                execution_entry=dict(kind='original751',pointer=dict(path=owner.ORIGINAL_ENTRY,sha256='a'*64)))
            with patch.object(owner,'ROOT',root):
                result=dict(complete=True,case=owner.pointer(out/'case.json'),admission_driver_receipt=owner.pointer(out/'admission_driver_receipt.json'))
                before=(out/'case.json').read_bytes()
                with patch.object(owner.original_owner,'require_native_driver'),patch.object(owner.original_owner,'require_native_protocol'),\
                        patch.object(owner,'validate_completed_case'),patch.object(owner,'require_attestation'),\
                        patch.object(owner,'measured_motion',return_value=None):
                    record=owner.accept_native(row,result,{}, {},'e'*64,'f'*64,{})
                    self.assertIs(record['native_success'],False);self.assertIsNone(record['motion_pass'])
                    self.assertEqual((out/'case.json').read_bytes(),before)
                    changed=dict(result,case=dict(result['case'],sha256='0'*64))
                    with self.assertRaises(ValueError):owner.accept_native(row,changed,{}, {},'e'*64,'f'*64,{})

    def test_twelve_native_five_GPU_actual_quota_without_hardware_probe(self):
        profile=dict(schema=owner.PROFILE_SCHEMA,max_native_slots=12,max_GPU_slots=5,max_CPU_preflight_slots=6,ready_limit=24)
        ready=[dict(job_id='p'+str(i),method='physmani') for i in range(6)]
        ready += [dict(job_id='g'+str(i),method='cv_geometry' if i%2 else 'context_geometry') for i in range(9)]
        selected=owner.select_native_jobs(ready,[],profile)
        self.assertEqual(len(selected),12);self.assertEqual(sum(r['method']=='physmani' for r in selected),5)
        self.assertNotIn('p5',[r['job_id'] for r in selected])
        for name,value in (('max_native_slots',13),('max_GPU_slots',6),('max_CPU_preflight_slots',7),('ready_limit',25),('max_native_slots',True)):
            with self.subTest(name=name,value=value),self.assertRaises(ValueError):owner.validate_resource_profile(dict(profile,**{name:value}))

    def test_real_capacity_can_exceed_actual_quota_but_never_the_reverse_or_fake_fallback(self):
        high=dict(schema=owner.PROFILE_SCHEMA,max_native_slots=12,max_GPU_slots=5,max_CPU_preflight_slots=6,ready_limit=24)
        low=dict(high,max_native_slots=8,max_GPU_slots=3,max_CPU_preflight_slots=4,ready_limit=16)
        control=dict(resource_profile=low,resource_capacity_envelope=high,resource_profile_within_capacity=True,resource_capacity_source='qualified1031',
            execution_implementation={'actual':'same901'},policy_sha256='e'*64,author_outcome_protocol={'same':'741'},
            resource_qualifier=dict(path='scripts/audit_concurrent_TRAIN_resource_canary_v1031_r10.py',sha256='f'*64),
            resource_qualification=dict(path='runtime/framework/concurrent_TRAIN_resource_qualification_v1031.json',sha256='a'*64))
        q=dict(schema=owner.QUALIFICATION_SCHEMA,complete=True,split='train',resource_profile=high,capacity_envelope=high,capacity_envelope_only=True,source_count=6,
            execution_entry=control['execution_implementation'],policy_sha256='e'*64,author_outcome_protocol=control['author_outcome_protocol'],
            case_count=18,infrastructure_failed=0,actual_CPU_preflight_peak=6,cleanup_completed=True,resource_bounds_verified=True,
            overlap_verified=True,independent_audit_complete=True,formal_score_contribution=False,TEST_read=False,joint_supply_overlap=dict(complete=True),
            observed_native_peak=12,observed_GPU_peak=5,source_sha256='f'*64,native_successes_diagnostic=0)
        self.assertIs(owner.require_resource_qualification(q,control),q)
        for field,value in (('case_count',12),('actual_CPU_preflight_peak',4),('independent_audit_complete',False),('joint_supply_overlap',{}),('observed_GPU_peak',3),('resource_profile',low),('capacity_envelope_only',False)):
            with self.subTest(field=field),self.assertRaises(ValueError):owner.require_resource_qualification(dict(q,**{field:value}),control)
        for field,value in (('resource_profile',dict(high,max_native_slots=13)),('resource_capacity_envelope',low),('resource_capacity_source','qualified1005_fallback')):
            with self.subTest(field=field),self.assertRaises(ValueError):owner.require_resource_qualification(q,dict(control,**{field:value}))
        fallback=dict(schema=owner.FALLBACK_PROFILE_SCHEMA,max_native_slots=8,max_GPU_slots=3,max_CPU_preflight_slots=4,ready_limit=16)
        fc=dict(control,resource_profile=fallback,resource_capacity_envelope=fallback,resource_capacity_source='qualified1005_fallback',
            resource_qualifier=dict(path='scripts/audit_concurrent_TRAIN_resource_canary_v1005_r10.py',sha256='f'*64),
            resource_qualification=dict(path='runtime/framework/concurrent_TRAIN_resource_qualification_v1005.json',sha256='a'*64))
        fq=dict(q,schema='r10_independent_concurrent_TRAIN_resource_qualification_v1005',resource_profile=fallback,
            case_count=12,actual_CPU_preflight_peak=4,observed_native_peak=8,observed_GPU_peak=3)
        fq.pop('capacity_envelope');fq.pop('capacity_envelope_only');fq.pop('source_count')
        self.assertIs(owner.require_resource_qualification(fq,fc),fq)
        with self.assertRaises(ValueError):owner.require_resource_qualification(q,fc)


if __name__=='__main__':unittest.main()
