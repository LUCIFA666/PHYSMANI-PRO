"""CPU/temp assertions for1011; production scheduler/resources never run."""
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch
import ast
import importlib.util
import json
import tempfile
import threading
import unittest

from scripts import run_composite_TEST_owner_v1011_r10 as owner
from scripts import run_composite_TEST_owner_v956_r10 as base

spec=importlib.util.spec_from_file_location('_composite956fixture',Path(__file__).with_name('test_composite_TEST_resume_v1010.py'))
fixtures=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixtures)


class OwnerTests(unittest.TestCase):
    def test_history_closes_literal753_902_956_without_flattening_or_resource_rewrite(self):
        from scripts import run_composite_TEST_owner_v956_r10 as parent_owner
        from types import SimpleNamespace
        c,p,s,t,middle,prior,original,ancestor,middle_status,middle_terminal=fixtures.fixture()
        pointer=lambda n:dict(path=n,sha256=owner.digest(n))
        old_profile=dict(schema='r10_frozen_concurrent_native_resource_profile_v902',max_native_slots=4,max_GPU_slots=2,max_CPU_preflight_slots=2,ready_limit=8)
        keys=('suite','suite_lock','source_inputs','validation_release','author_outcome_protocol','execution_implementation',
            'environment_manifest','environment_wrapper','environment_namespace_proof','standing_continuation_record')
        for key in keys:prior[key]=c[key]=pointer('CPU/'+key)
        prior.update(resource_profile=old_profile,resource_qualification=pointer('CPU/old904'))
        c.update(resource_profile_changed=True,previous_resource_profile=deepcopy(old_profile),previous_resource_qualification=deepcopy(prior['resource_qualification']),
            previous_resume_control=pointer('CPU/955control'),previous_resume_control_lock=pointer('CPU/955lock'))
        records=dict(launch=dict(pid=753,start_ticks=1),plan=original,status={},terminal={})
        mid=dict(launch=dict(pid=902,start_ticks=2),plan=middle,status=middle_status,terminal=middle_terminal)
        parent=dict(launch=dict(pid=956,start_ticks=3),plan=p,status=s,terminal=t)
        values={'CPU/955control':prior,'CPU/955lock':dict(schema=parent_owner.LOCK_SCHEMA,approved_for_execution=True,files={'old':'a'*64})}
        for key in records:
            prior['original_'+key]=c['original_'+key]=pointer('CPU/753/'+key);values['CPU/753/'+key]=records[key]
            c['previous_'+key]=pointer('CPU/956/'+key);values['CPU/956/'+key]=parent[key]
        with patch.object(owner,'read_bound_json',side_effect=lambda files,p:deepcopy(values[p['path']])),patch.object(owner,'verify_inputs'),\
                patch.object(owner,'alive',return_value=False),patch.object(parent_owner,'require_history',return_value=(ancestor,records,mid)):
            out=owner.require_history(c,{'old':'a'*64});self.assertEqual(out,(prior,records,parent))
            changed=deepcopy(p);changed['jobs'][600]['resume_lineage']['parent_row_sha256']='0'*64
            values['CPU/956/plan']=changed
            with self.assertRaises(ValueError):owner.require_history(c,{'old':'a'*64})

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

    def test_fresh_partition_has_four_evidence_owners_and_only_remaining_scheduled(self):
        c,p,s,t,*_=fixtures.fixture();rows=owner.resumed_jobs(p,c,c['execution_entry'])
        owner.require_partition(c,p,s,t)
        pending=[r for r in rows if not r['imported_completed']]
        self.assertEqual(len(pending),720)
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

    def test_eight_native_three_GPU_quota_without_hardware_probe(self):
        profile=dict(schema=owner.PROFILE_SCHEMA,max_native_slots=8,max_GPU_slots=3,max_CPU_preflight_slots=4,ready_limit=16)
        ready=[dict(job_id='p'+str(i),method='physmani') for i in range(4)]
        ready += [dict(job_id='g'+str(i),method='cv_geometry' if i%2 else 'context_geometry') for i in range(6)]
        selected=owner.select_native_jobs(ready,[],profile)
        self.assertEqual(len(selected),8);self.assertEqual(sum(r['method']=='physmani' for r in selected),3)
        self.assertNotIn('p3',[r['job_id'] for r in selected])
        for name,value in (('max_native_slots',9),('max_GPU_slots',4),('max_CPU_preflight_slots',5),('max_native_slots',True)):
            changed=dict(profile,**{name:value})
            with self.subTest(name=name,value=value),self.assertRaises(ValueError):owner.validate_resource_profile(changed)

    def test_new12_case_qualified_profile_and_actual_CPUpeak4_required(self):
        profile=dict(schema=owner.PROFILE_SCHEMA,max_native_slots=8,max_GPU_slots=3,max_CPU_preflight_slots=4,ready_limit=16)
        control=dict(resource_profile=profile,execution_implementation={'actual':'same901'},policy_sha256='e'*64,
            author_outcome_protocol={'same':'741'},resource_qualifier=dict(path='scripts/audit_concurrent_TRAIN_resource_canary_v1005_r10.py',sha256='f'*64))
        q=dict(schema=owner.QUALIFICATION_SCHEMA,complete=True,split='train',resource_profile=profile,
            execution_entry=control['execution_implementation'],policy_sha256='e'*64,author_outcome_protocol=control['author_outcome_protocol'],
            case_count=12,infrastructure_failed=0,actual_CPU_preflight_peak=4,cleanup_completed=True,resource_bounds_verified=True,
            overlap_verified=True,independent_audit_complete=True,formal_score_contribution=False,TEST_read=False,
            joint_supply_overlap=dict(complete=True),
            observed_native_peak=8,observed_GPU_peak=3,source_sha256='f'*64,native_successes_diagnostic=0)
        self.assertIs(owner.require_resource_qualification(q,control),q)
        for field,value in (('case_count',6),('actual_CPU_preflight_peak',2),('independent_audit_complete',False),('joint_supply_overlap',{}),('observed_GPU_peak',2),('resource_profile',dict(profile,max_native_slots=4))):
            with self.subTest(field=field),self.assertRaises(ValueError):owner.require_resource_qualification(dict(q,**{field:value}),control)


if __name__=='__main__':unittest.main()
