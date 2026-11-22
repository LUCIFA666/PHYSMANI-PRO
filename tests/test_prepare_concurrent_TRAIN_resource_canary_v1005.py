"""Temporary synthetic small controls only; all resource/namespace gates mocked."""
from contextlib import contextmanager
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('prep1005_cpu',ROOT/'scripts/prepare_concurrent_TRAIN_resource_canary_v1005_r10.py')
p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p);m=p.runner
spec=importlib.util.spec_from_file_location('fixtures1005_cpu',ROOT/'tests/test_concurrent_TRAIN_resource_canary_v1005.py')
f=importlib.util.module_from_spec(spec);spec.loader.exec_module(f)


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        self.addCleanup(patch.stopall)
        patch.object(p,'ROOT',self.root).start();patch.object(m,'ROOT',self.root).start()
        for name in m.SOURCES:
            path=self.root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('# synthetic frozen source\n')
        inventory,cpus,c,plan=f.fixture(m)
        implementation=m.implementation_from_sources({name:m.point(name) for name in c['execution_implementation']['sources']})
        env=dict(environment_wrapper=m.point('scripts/run_frozen_system_environment_v820_r10.py'),
            environment_manifest=self.write('artifacts/fake_environment/manifest.json',{}),
            environment_namespace_proof=self.write('artifacts/fake_environment/qualification.json',{}))
        self.base=dict(policy_sha256='a'*64,author_outcome_protocol=c['author_outcome_protocol'],execution_implementation=implementation,**env)
        base=self.write(m.BASE_CONTROL,self.base);patch.object(p,'BASE_CONTROL_SHA',base['sha256']).start()
        self.write(m.STANDING,dict(continue_until_user_stop=True,user_stop_received=False))
        self.auth=self.write('runtime/framework/fake_concurrency_authorization.json',dict(user_authorized_concurrency_change=True))
        for q in c['previous_CPU_receipts']:q.update(self.write(q['path'],cpus[q['path']]))
        routes=dict(routes=[dict(task=m.TASK,method=method,common_RML_profile_sha256='d'*64,
            author_outcome_protocol=c['author_outcome_protocol'],worker=implementation['original_case_worker'],
            consumer=implementation['original_native_runtime']) for method in m.METHODS])
        rp=self.write(m.TRAIN_ROUTES,routes);plan['routes']=rp
        inventory.update(plan=self.write(m.TRAIN_PLAN,plan),routes=rp)
        ip=self.write(m.INVENTORY,inventory);patch.object(m,'INVENTORY_SHA',ip['sha256']).start()
        for key,name in m.PREVIOUS_PATHS.items():self.write(name,{})
        self.calls=[]
        @contextmanager
        def guard(control,event):
            self.calls.append(('gate',deepcopy(control)))
            self.assertFalse((self.root/m.SETUP).exists())
            yield {'fake_exclusive_resource_proof':True}
        self.gate=patch.object(m,'resource_guard',guard);self.gate.start()

    def write(self,name,value):
        path=self.root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))
        return m.point(name)

    def prepare(self):return p.prepare(self.auth['path'],self.auth['sha256'])

    def test_prepare_freezes_exact_small_sources_and_original_four_sources_only(self):
        result=self.prepare();self.assertTrue(result['complete']);self.assertFalse(result['native_started'])
        self.assertEqual(result['case_count'],12);self.assertEqual(len(self.calls),1)
        control,binding=m.load_control(m.CONTROL,m.INPUT_LOCK,result['lock']['sha256'])
        self.assertEqual(control['execution_implementation'],self.base['execution_implementation'])
        self.assertEqual(control['resource_profile'],dict(schema='r10_frozen_concurrent_native_resource_profile_v1005',
            max_native_slots=8,max_GPU_slots=3,max_CPU_preflight_slots=4,ready_limit=16))
        self.assertEqual(len(control['previous_CPU_receipts']),12)
        self.assertFalse((self.root/m.RUNTIME).exists());self.assertFalse((self.root/m.OUTPUT).exists())
        self.assertFalse(any('raw.hdf5' in name or name.endswith('.pt') for name in binding['files']))
        with self.assertRaises(ValueError):self.prepare()

    def test_default_main_inert_without_any_read_or_resource_callback(self):
        with patch.object(p,'prepare',side_effect=AssertionError('activation')),patch.object(Path,'read_bytes',side_effect=AssertionError('IO')):
            result=p.main([])
        self.assertFalse(result['enabled'])

    def test_STOP_and_resource_failure_before_setup_commit(self):
        with patch.object(m,'stopped',return_value=True),self.assertRaises(ValueError):self.prepare()
        self.assertFalse((self.root/m.SETUP).exists())
        with patch.object(m,'resource_guard',side_effect=RuntimeError('busy')),self.assertRaisesRegex(RuntimeError,'busy'):self.prepare()
        self.assertFalse((self.root/m.SETUP).exists())

    def test_wrong_authorization_or_original_inventory_refuses_before_resource_gate(self):
        for kind in ('authorization','inventory'):
            if kind=='authorization':
                self.write(self.auth['path'],{'user_authorized_concurrency_change':False})
            else:self.write(m.INVENTORY,{'cases':[]})
            with self.subTest(kind=kind),self.assertRaises(ValueError):self.prepare()
            self.assertEqual(self.calls,[])
            if kind=='authorization':self.write(self.auth['path'],{'user_authorized_concurrency_change':True})

    def test_lock_SHA_code_and_foreign_file_are_rejected_without_reading_foreign_data(self):
        result=self.prepare()
        with self.assertRaises(ValueError):m.load_control(m.CONTROL,m.INPUT_LOCK,'0'*64)
        source=self.root/m.SOURCE;source.write_text('# changed')
        with self.assertRaisesRegex(ValueError,'changed'):m.load_control(m.CONTROL,m.INPUT_LOCK,result['lock']['sha256'])

    def test_extra_raw_or_model_declaration_rejected_before_opening_its_bytes(self):
        self.prepare();lock=json.loads((self.root/m.INPUT_LOCK).read_text())
        model=self.root/'never_read.pt';model.write_bytes(b'forbidden model payload')
        lock['files'][str(model)]='a'*64;self.write(m.INPUT_LOCK,lock)
        original=Path.read_bytes
        def read(path):
            self.assertNotEqual(path,model,'extra model opened')
            return original(path)
        with patch.object(Path,'read_bytes',read),self.assertRaisesRegex(ValueError,'exactly declared'):
            m.load_control(m.CONTROL,m.INPUT_LOCK,m.sha((self.root/m.INPUT_LOCK).read_bytes()))

    def test_old956_must_exit_and_have_genuine_clean_drain_not_final_or_user_stop(self):
        control=dict(policy_sha256='a'*64,author_outcome_protocol={'protocol':1},base_formal_control=m.point(m.BASE_CONTROL),
            **{key:m.point(name) for key,name in m.PREVIOUS_PATHS.items()})
        metadata=dict(previous_launch=dict(pid=100,start_ticks=101,split='test',policy_sha256='a'*64,
            author_outcome_protocol={'protocol':1},source_sha256=m.point('scripts/run_composite_TEST_owner_v956_r10.py')['sha256'],
            resume_control=control['base_formal_control']),
            previous_controller_launch=dict(pid=200,start_ticks=201,
                source_sha256=m.point('scripts/continue_composite_TEST_v958_r10.py')['sha256'],resume_control=control['base_formal_control']),
            previous_status=dict(split='test',expected=1920,failed={},active=[],checking=[],completed={'a':{}},
                native_slots=0,GPU_slots=0,owner_error=None,drain_requested=True,stop_requested=False),
            previous_terminal=dict(complete=False,drained=True,drain_requested=True,stop_requested=False,expected=1920,
                completed=1,failed=0,automatic_retry=False,policy_sha256='a'*64),previous_DRAIN={},
            previous_controller_status=dict(stage='requires_concrete_repair',automatic_retry=False),
            previous_controller_failure=dict(complete=False,user_stop=False,automatic_retry=False))
        self.assertEqual(m.require_previous_exit(control,metadata,lambda _:False)['completed'],1)
        for kind in ('live','active','infra','userstop','final','source'):
            v=deepcopy(metadata)
            if kind=='active':v['previous_status']['active']=['x']
            elif kind=='infra':v['previous_status']['failed']={'x':{}}
            elif kind=='userstop':v['previous_terminal']['stop_requested']=True
            elif kind=='final':v['previous_terminal']['complete']=True
            elif kind=='source':v['previous_launch']['source_sha256']='0'*64
            with self.subTest(kind=kind),self.assertRaises(ValueError):m.require_previous_exit(control,v,lambda _:kind=='live')


if __name__=='__main__':unittest.main()
