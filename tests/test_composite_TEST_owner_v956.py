"""CPU/temp assertions for956; production scheduler/resources never run."""
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch
import ast
import importlib.util
import json
import tempfile
import threading
import unittest

from scripts import run_composite_TEST_owner_v956_r10 as owner
from scripts import run_concurrent_resumed_formal_TEST_owner_v902_r10 as base

spec=importlib.util.spec_from_file_location('_composite956fixture',Path(__file__).with_name('test_composite_TEST_resume_v955.py'))
fixtures=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixtures)


class OwnerTests(unittest.TestCase):
    def test_physical_child_quota_and_receipt_functions_remain_original902_AST(self):
        def bodies(module):
            return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(Path(module.__file__).read_text()).body
                if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
        old,new=bodies(base),bodies(owner)
        for name in ('child','select_native_jobs','future_result','terminal_is_complete','require_batch_execution',
                'require_batch_metrics','require_resource_qualification','execution_implementation'):
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

    def test_fresh_partition_has_three_evidence_owners_and_only_remaining_scheduled(self):
        c,p,s,t,*_=fixtures.fixture();rows=owner.resumed_jobs(p,c,c['execution_entry'])
        owner.require_partition(c,p,s,t)
        pending=[r for r in rows if not r['imported_completed']]
        self.assertEqual(len(pending),1336)
        self.assertTrue(set(c['failed_job_ids'])<=set(r['job_id'] for r in pending))
        self.assertTrue(all(r['evidence_runtime']==owner.RUNTIME for r in pending))
        self.assertTrue(all(r['job_id'] not in s['completed'] for r in pending))

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

    def test_same_profile_preserves_four_native_two_GPU_bounds_without_hardware_probe(self):
        profile=dict(schema=owner.PROFILE_SCHEMA,max_native_slots=4,max_GPU_slots=2,max_CPU_preflight_slots=2,ready_limit=8)
        ready=[dict(job_id='p1',method='physmani'),dict(job_id='p2',method='physmani'),dict(job_id='p3',method='physmani'),
            dict(job_id='cv',method='cv_geometry'),dict(job_id='ctx',method='context_geometry')]
        selected=owner.select_native_jobs(ready,[],profile)
        self.assertEqual([r['job_id'] for r in selected],['p1','p2','cv','ctx'])


if __name__=='__main__':unittest.main()
