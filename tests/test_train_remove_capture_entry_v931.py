"""CPU/temp-only931 entry tests; no actual guard, worker, model or native."""
import ast
from contextlib import contextmanager, ExitStack, redirect_stdout
from copy import deepcopy
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch


PROJECT=Path(__file__).resolve().parents[1]
SOURCE=PROJECT/'scripts/diagnose_TRAIN_remove_capture_v931_r10.py'
spec=importlib.util.spec_from_file_location('CPU_train_remove_entry931',SOURCE)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def encode(value):return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()


class Fixture:
    def __init__(self,root,task=None,method='cv_geometry'):
        self.root=Path(root);self.events=[];self.source_sha={};self.predicate_pins={};self.bodies=[];self.owners={}
        groups={}
        for base,cls,name,_ in m.EXPECTED_BODIES:groups.setdefault(base,{}).setdefault(cls,[]).append(name)
        for base,classes in groups.items():
            text='\n\n'.join('class '+cls+':\n'+'\n'.join('    def '+name+'(self):\n        return None\n' for name in names)
                for cls,names in classes.items())+'\n'
            name='our_mani/multitask/'+base;self.write(name,text.encode());self.predicate_pins[base]=m.sha(text.encode())
            for cls in classes:self.owners[cls]=type(cls,(),{})
            for node in ast.parse(text).body:
                for function in node.body:
                    self.bodies.append((base,node.name,function.name,m.sha(ast.dump(function,include_attributes=False).encode())))
        for name in m.SOURCES:
            if not (self.root/name).exists():self.write(name,('CPU931 source stub '+name).encode())
            self.source_sha[name]=self.pointer(name)['sha256']
        self.pinned={name:self.source_sha[name] for name in m.PINNED_SOURCES}
        self.environment=deepcopy(m.ENVIRONMENT)
        for key,value in self.environment.items():
            if key!='wrapper':self.write(value['path'],dict(CPU_only_environment_metadata=key))
            value['sha256']=self.pointer(value['path'])['sha256']
        task=task or m.TASKS[0];raw='CPU_episode0';suffix='/'+task+'/'+raw+'/'+method
        config=m.CONFIG_BASE+'/'+task+'_ours_configuration.json'
        self.write(config,dict(context_checkpoint=deepcopy(m.CHECKPOINT),CPU_fake_configuration=True))
        self.write(m.TRAIN_INPUTS,dict(files={'/UNOPENED/input.hdf5':'a'*64}))
        self.write(m.TRAIN_MANIFEST,dict(split='train',episodes=[dict(task=task,raw_episode_id=raw,path='/UNOPENED/TRAIN')]))
        self.control=dict(schema=m.CONTROL_SCHEMA,execution_enabled=True,
            selection=dict(task=task,method=method,raw_episode_id=raw),configuration=self.pointer(config),
            original_frozen_inputs=self.pointer(m.TRAIN_INPUTS),train_manifest=self.pointer(m.TRAIN_MANIFEST),
            checkpoint=deepcopy(m.CHECKPOINT),completed_TEST_results=dict(path=m.RESULT,sha256='a'*64),
            formal_resume_lock=dict(path=m.FORMAL_LOCK,sha256='b'*64),expected_body_pins=[list(row) for row in self.bodies],
            diagnostic_sources=dict(self.source_sha),max_calls=100,runtime=m.RUNTIME+suffix,output=m.OUTPUT+suffix,
            formal_score_contribution=False,TEST_used_for_fitting=False,environment=self.environment)
        self.reseal()

    def write(self,name,value):
        path=self.root/name;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(value if isinstance(value,bytes) else encode(value))
    def pointer(self,name):return dict(path=name,sha256=m.sha((self.root/name).read_bytes()))
    def reseal(self):
        self.write(m.CONTROL,self.control)
        files={str(self.root/name):checksum for name,checksum in self.control['diagnostic_sources'].items()}
        for value in (self.control['configuration'],self.control['original_frozen_inputs'],self.control['train_manifest'],*self.control['environment'].values()):
            files[str(self.root/value['path'])]=value['sha256']
        files[str(self.root/m.CONTROL)]=self.pointer(m.CONTROL)['sha256']
        self.lock=dict(schema=m.LOCK_SCHEMA,approved_for_execution=True,files=files);self.write(m.INPUTS,self.lock)
    def args(self,check=False):
        return m.parser().parse_args(['--execute',*(['--check'] if check else []),'--lock-sha256',self.pointer(m.INPUTS)['sha256'],
            '--test-results-sha256','a'*64,'--resume-lock-sha256','b'*64])
    @contextmanager
    def pins(self):
        with patch.object(m,'PREDICATE_PINS',self.predicate_pins),patch.object(m,'EXPECTED_BODIES',tuple(self.bodies)), \
                patch.object(m,'PINNED_SOURCES',self.pinned),patch.object(m,'ENVIRONMENT',self.environment):yield
    def binding(self):
        with self.pins():return m.load_control(self.root,self.args())
    def observer(self):
        fixture=self
        def plan(policy):
            return [(policy.frontend if cls=='PublicModelFrontend' else policy,self.owners[cls],name,
                str(self.root/'our_mani/multitask'/base),self.predicate_pins[base]) for base,cls,name,_ in self.bodies]
        class Observer:
            def __init__(self,policy,**kwargs):
                self.policy=policy;self.kwargs=kwargs;self.restored=False
                fixture.events.append('observer_construct')
            def __enter__(self):
                fixture.events.append('observer_enter');self.previous=self.policy.marker;self.policy.marker='CPU observing';return self
            def __exit__(self,*exc):
                self.policy.marker=self.previous;self.restored=True;fixture.events.append('observer_exit')
            def receipt(self):return dict(restored=self.restored,native_validated=False,unqualified=True,CPU_fake=True)
        return SimpleNamespace(PINS=self.predicate_pins,default_plan=plan,TrainRemoveCaptureValueObserver=Observer)
    def prepared(self):
        s=self.control['selection'];domain=SimpleNamespace(split='train',raw=s['raw_episode_id'])
        domain.require=lambda task,method:'train_qualification' if (task,method)==(s['task'],s['method']) else 'refused'
        domain.receipt=lambda:dict(split='train',purpose='train_qualification',raw=s['raw_episode_id'])
        return dict(args=SimpleNamespace(split='train',task=s['task'],method=s['method'],raw_episode_id=s['raw_episode_id']),
            component_domain=domain,config=dict(context_checkpoint=deepcopy(m.CHECKPOINT)),identity={'CPU_fake_identity':True},row={'CPU_fake_original_row':True})
    def runtime(self):
        policy=SimpleNamespace(frontend=SimpleNamespace(),marker='original')
        result=(object(),{'CPU_component':True},policy.frontend,policy,object());calls=[]
        def geometry(*args):calls.append(args);self.events.append('original_geometry');return result
        return SimpleNamespace(_geometry=geometry),result,calls


class FakeGuard:
    def __init__(self,fixture):self.fixture=fixture;self.held=False;self.permit=object();self.permit_calls=0;self.on_enter=None
    @contextmanager
    def reserved_resources_after_TEST(self,root,result_sha,**kwargs):
        assert result_sha=='a'*64 and kwargs==dict(resume_lock_sha256='b'*64,enabled=True)
        self.fixture.events.append('lease_enter');self.held=True
        try:
            if self.on_enter:self.on_enter()
            yield self.permit
        finally:self.held=False;self.fixture.events.append('lease_exit')
    def require_permit(self,permit,root):
        assert self.held and permit is self.permit
        self.permit_calls+=1;self.fixture.events.append('permit')
        return dict(complete=True,expected_cases=1920,CPU_fake_resource_proof=True)


class EntryTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.fixture=Fixture(self.root)
        self.denied=Mock(side_effect=AssertionError('CPU test reached production admission/resource/native/proc/model helper'))
        for name in ('_load_guard','_load_worker','_load_observer','_require_checkpoint','_environment','_process_identity','_validate_native_case'):
            scope=patch.object(m,name,self.denied);scope.start();self.addCleanup(scope.stop)

    def test_import_default_main_and_check_without_execute_are_inert(self):
        source=SOURCE.read_bytes();module=ModuleType('CPU931_inert');module.__file__=str(SOURCE)
        denied=Mock(side_effect=AssertionError('disabled entry touched project'))
        with patch.object(Path,'resolve',denied),patch.object(Path,'read_bytes',denied),patch.object(Path,'exists',denied):
            exec(compile(source,str(SOURCE),'exec'),module.__dict__)
            self.assertFalse(module.main([])['enabled']);self.assertFalse(module.main(['--check'])['enabled'])
        self.denied.assert_not_called()

    def test_production_pins_and_exact17_bodies_are_source_only(self):
        self.assertEqual(len(m.EXPECTED_BODIES),17);self.assertEqual(len(m.PREDICATE_PINS),7)
        self.assertEqual(m.PINNED_SOURCES[m.ENV_SOURCE],m.ENVIRONMENT['wrapper']['sha256'])
        self.assertIn(m.ORIGINAL_RUNTIME,m.PINNED_SOURCES);self.assertIn(m.OUTCOME_SOURCE,m.PINNED_SOURCES)
        self.assertEqual(m.CHECKPOINT['sha256'],'ebec3d1b3a83ceb0d4fb3bc889f8b62376cb4581071f53de29e0b6bd71c162ef')
        for base,cls,name,checksum in m.EXPECTED_BODIES:
            text=(PROJECT/'our_mani/multitask'/base).read_bytes()
            self.assertEqual(m.sha(text),m.PREDICATE_PINS[base])
            owner=next(n for n in ast.parse(text).body if isinstance(n,ast.ClassDef) and n.name==cls)
            method=next(n for n in owner.body if isinstance(n,ast.FunctionDef) and n.name==name)
            self.assertEqual(m.sha(ast.dump(method,include_attributes=False).encode()),checksum)

    def test_three_external_sha_pins_and_execute_required_before_resource(self):
        for key in ('execute','lock_sha256','test_results_sha256','resume_lock_sha256'):
            args=self.fixture.args();setattr(args,key,False if key=='execute' else None)
            with self.fixture.pins(),self.subTest(key=key),self.assertRaises(ValueError):m.run(self.root,args)
        self.denied.assert_not_called()

    def test_exact_small_map_and_checkpoint_declaration_without_reading_model(self):
        binding=self.fixture.binding()
        self.assertEqual(binding['config']['context_checkpoint'],m.CHECKPOINT)
        self.assertFalse((self.root/m.CHECKPOINT['path']).exists())
        self.assertNotIn(str(self.root/m.CHECKPOINT['path']),self.fixture.lock['files'])
        self.assertNotIn(str(self.root/m.RESULT),self.fixture.lock['files'])
        self.assertNotIn(str(self.root/m.FORMAL_LOCK),self.fixture.lock['files'])
        self.assertEqual(len(self.fixture.lock['files']),len(m.SOURCES)+6)
        self.denied.assert_not_called()

    def test_only_two_remove_tasks_and_two_geometry_methods(self):
        for task in m.TASKS:
            for method in m.METHODS:
                with tempfile.TemporaryDirectory() as name:
                    fixture=Fixture(Path(name),task,method);self.assertEqual(fixture.binding()['control']['selection']['task'],task)
        for key,value in (('task','moving_basketball_in_hoop'),('method','physmani'),('raw_episode_id','../TEST')):
            saved=deepcopy(self.fixture.control);self.fixture.control['selection'][key]=value;self.fixture.reseal()
            with self.fixture.pins(),self.subTest(key=key),self.assertRaises(ValueError):m.load_control(self.root,self.fixture.args())
            self.fixture.control=saved;self.fixture.reseal()

    def test_changed_body_source_checkpoint_environment_or_small_map_refused(self):
        for mode in ('body','pin','checkpoint','environment','extra','alias'):
            with tempfile.TemporaryDirectory() as name:
                fixture=Fixture(Path(name))
                if mode=='body':fixture.control['expected_body_pins'].pop()
                elif mode=='pin':fixture.control['diagnostic_sources'][m.WORKER]='f'*64
                elif mode=='checkpoint':fixture.control['checkpoint']['sha256']='f'*64
                elif mode=='environment':fixture.control['environment']=deepcopy(fixture.environment);fixture.control['environment']['manifest']['sha256']='f'*64
                fixture.reseal()
                if mode=='extra':fixture.lock['files'][str(fixture.root/'models/never-open.pt')]='a'*64;fixture.write(m.INPUTS,fixture.lock)
                if mode=='alias':fixture.lock['files'][m.SOURCE]=fixture.source_sha[m.SOURCE];fixture.write(m.INPUTS,fixture.lock)
                with fixture.pins(),self.subTest(mode=mode),self.assertRaises(ValueError):m.load_control(fixture.root,fixture.args())

    def test_actual_body_plan_17_exact_and_missing_duplicate_or_wrong_owner_rejected(self):
        observer=self.fixture.observer();policy=SimpleNamespace(frontend=object())
        with self.fixture.pins():
            self.assertEqual(len(m.require_body_plan(self.root,observer,policy)),17)
            plan=observer.default_plan(policy)
            for changed in (plan[:-1],plan[:-1]+[plan[0]]):
                with patch.object(observer,'default_plan',return_value=changed),self.assertRaises(ValueError):
                    m.require_body_plan(self.root,observer,policy)

    def test_original_geometry_called_once_and_tuple_identity_preserved(self):
        binding=self.fixture.binding();runtime,result,calls=self.fixture.runtime();saved=runtime._geometry;events={};p=self.fixture.prepared()
        with self.fixture.pins(),m.scoped_geometry(self.root,runtime,self.fixture.observer(),binding,events):
            self.assertIs(runtime._geometry(p,'CPU_env','CPU_descriptions'),result)
            with self.assertRaisesRegex(ValueError,'exactly one'):runtime._geometry(p,'CPU_env','CPU_descriptions')
            self.assertEqual(len(calls),1)
        self.assertIs(runtime._geometry,saved);self.assertEqual(result[3].marker,'original')
        self.assertTrue(events['geometry_hook_restored']);self.assertTrue(events['observer_receipts'][0]['restored'])
        self.assertEqual(self.fixture.events[-1],'observer_exit')

    def test_exception_restores_hook_and_observer_in_reverse_order(self):
        binding=self.fixture.binding();runtime,result,_=self.fixture.runtime();saved=runtime._geometry;events={};error=RuntimeError('CPU exact original error')
        with self.fixture.pins(),self.assertRaises(RuntimeError) as caught:
            with m.scoped_geometry(self.root,runtime,self.fixture.observer(),binding,events):
                runtime._geometry(self.fixture.prepared(),None,None);raise error
        self.assertIs(caught.exception,error);self.assertIs(runtime._geometry,saved);self.assertTrue(events['geometry_hook_restored'])
        self.assertEqual(result[3].marker,'original');self.assertTrue(events['observer_receipts'][0]['restored'])
        self.assertTrue(m._ACTIVE_LOCK.acquire(blocking=False));m._ACTIVE_LOCK.release()

    def install_worker(self,*,check=False,error=None):
        fixture=self.fixture;guard=FakeGuard(fixture);runtime,result,calls=fixture.runtime();observer=fixture.observer();evaluator=object()
        state=dict(main_calls=0,simulated_prepare_calls=0,simulated_cpu_preflight_calls=0,entry_check_modes=[],
            case_bytes=b'{ "official_task_success" : false, "CPU_marker":"unchanged" }\n')
        @contextmanager
        def declared():
            fixture.events.append('edge_enter')
            try:yield
            finally:fixture.events.append('edge_exit')
        @contextmanager
        def evaluating(actual):
            assert actual is evaluator;fixture.events.append('evaluator_enter')
            try:yield
            finally:fixture.events.append('evaluator_exit')
        edge=SimpleNamespace(declared_edge_scope=declared,evaluator_scope=evaluating)
        def worker_main():
            state['main_calls']+=1;fixture.events.append('worker_main');state['argv']=list(sys.argv)
            assert guard.held and '--development-demo' in sys.argv and sys.argv[sys.argv.index('--split')+1]=='train'
            # These counters model the original743 entry's preparation on
            # every invocation; they are not real admission/preflight proofs.
            state['simulated_prepare_calls']+=1;state['simulated_cpu_preflight_calls']+=1
            state['entry_check_modes'].append('--check' in sys.argv)
            if '--check' in sys.argv:
                print(json.dumps(dict(complete=True,native_started=False,GPU_started=False,actual_split='train')));return
            assert runtime._geometry(fixture.prepared(),None,None) is result
            if error is not None:raise error
            output=fixture.control['output'];(fixture.root/output).mkdir(parents=True)
            (fixture.root/output/'case.json').write_bytes(state['case_bytes'])
        worker=SimpleNamespace(runtime=runtime,main=worker_main)
        def checkpoint(root,binding):
            assert guard.held and binding['config']['context_checkpoint']==m.CHECKPOINT
            fixture.events.append('checkpoint')
        def environment(root,binding):
            assert guard.held;fixture.events.append('environment');return dict(complete=True,CPU_fake_namespace=True)
        def validate(case,events,binding):
            assert guard.held and case['official_task_success'] is False and events['expected_identity']=={'CPU_fake_identity':True}
            fixture.events.append('validate_case')
        context=ExitStack();context.enter_context(fixture.pins())
        for name,value in (('_load_guard',lambda root:guard),('_load_worker',lambda root:(worker,edge,evaluator)),
                ('_load_observer',lambda root:observer),('_require_checkpoint',checkpoint),('_environment',environment),
                ('_process_identity',lambda:dict(pid=101,start_ticks=102,CPU_fake_process=True)),('_validate_native_case',validate)):
            context.enter_context(patch.object(m,name,value))
        return context,guard,state,runtime,result,calls

    def test_fake_CPU_check_uses_lease_outer_scopes_without_installing_observer(self):
        context,guard,state,_,_,calls=self.install_worker(check=True)
        with context,patch.object(m,'_load_observer',self.denied):receipt=m.run(self.root,self.fixture.args(check=True))
        self.assertEqual(state['main_calls'],1);self.assertEqual(calls,[]);self.assertFalse(receipt['execution']['observer_installed'])
        self.assertFalse(receipt['execution']['native_started']);self.assertFalse(guard.held)
        self.assertLess(self.fixture.events.index('lease_enter'),self.fixture.events.index('checkpoint'))
        self.assertEqual(self.fixture.events[-4:],['evaluator_exit','edge_exit','permit','lease_exit'])
        self.assertLess(self.fixture.events.index('evaluator_exit'),self.fixture.events.index('edge_exit'))
        self.assertEqual(self.fixture.events[-1],'lease_exit');self.assertFalse((self.root/self.fixture.control['output']).exists())
        self.assertTrue((self.root/(self.fixture.control['runtime']+'/CPU_check/diagnostic_sidecar.json')).exists())

    def test_same_supplement_CPU_then_native_each_calls_original_entry_fresh(self):
        context,guard,state,_,_,calls=self.install_worker()
        with context:
            first=m.run(self.root,self.fixture.args(check=True))
            self.assertEqual(first['execution']['mode'],'CPU_check')
            self.assertFalse((self.root/self.fixture.control['output']).exists())
            self.assertEqual(state['entry_check_modes'],[True]);self.assertEqual(calls,[])
            reads=[];original=m.raw
            def raw(root,name,*args,**kwargs):reads.append(name);return original(root,name,*args,**kwargs)
            with patch.object(m,'raw',side_effect=raw):second=m.run(self.root,self.fixture.args())
        self.assertEqual(second['execution']['mode'],'instrumented_native_TRAIN')
        self.assertEqual(state['main_calls'],2);self.assertEqual(state['entry_check_modes'],[True,False])
        self.assertEqual(state['simulated_prepare_calls'],2);self.assertEqual(state['simulated_cpu_preflight_calls'],2)
        self.assertEqual(len(calls),1);self.assertFalse(guard.held)
        self.assertFalse(any('/CPU_check/' in name for name in reads))
        for mode in ('CPU_check','native'):
            self.assertTrue((self.root/(self.fixture.control['runtime']+'/'+mode+'/diagnostic_sidecar.json')).exists())
        self.assertEqual((self.root/self.fixture.control['output']/'case.json').read_bytes(),state['case_bytes'])

    def test_existing_original_output_blocks_before_any_resource_or_worker(self):
        self.fixture.write(self.fixture.control['output']+'/case.json',b'CPU preserved old case')
        with self.fixture.pins(),self.assertRaisesRegex(ValueError,'never overwrite or retry'):
            m.run(self.root,self.fixture.args())
        self.denied.assert_not_called()
        self.assertEqual((self.root/self.fixture.control['output']/'case.json').read_bytes(),b'CPU preserved old case')

    def test_fake_native_preserves_original_case_bytes_and_unqualified_sidecar(self):
        context,guard,state,runtime,result,calls=self.install_worker();saved=runtime._geometry;argv_before=sys.argv
        with context:receipt=m.run(self.root,self.fixture.args())
        self.assertEqual(len(calls),1);self.assertEqual(state['main_calls'],1);self.assertIs(runtime._geometry,saved);self.assertIs(sys.argv,argv_before)
        self.assertEqual((self.root/self.fixture.control['output']/'case.json').read_bytes(),state['case_bytes'])
        self.assertFalse(receipt['execution']['native_success']);self.assertTrue(receipt['execution']['case_original_bytes_unchanged'])
        for key in ('native_validated','physical_equivalence_claimed','formal_score_contribution','participates_in_VAL_TEST',
                    'TEST_used_for_fitting','policy_fix_claimed','original_751_or_901_entry_claimed','automatic_retry'):
            self.assertIs(receipt[key],False)
        self.assertTrue(receipt['unqualified']);self.assertEqual(receipt['actual_checkpoint'],m.CHECKPOINT)
        self.assertLess(self.fixture.events.index('observer_exit'),self.fixture.events.index('evaluator_exit'))
        self.assertLess(self.fixture.events.index('evaluator_exit'),self.fixture.events.index('edge_exit'))
        self.assertEqual(self.fixture.events[-1],'lease_exit');self.assertFalse(guard.held)

    def test_original_exception_survives_failure_recording_error_and_all_scopes_restore(self):
        problem=RuntimeError('CPU original geometry execution error');context,guard,state,runtime,_,_=self.install_worker(error=problem)
        saved=runtime._geometry;write=m.write_fresh_json
        def failing(path,value):
            if Path(path).name=='failure.json':raise OSError('CPU failure sidecar unavailable')
            return write(path,value)
        with context,patch.object(m,'write_fresh_json',side_effect=failing),self.assertRaises(RuntimeError) as caught:
            m.run(self.root,self.fixture.args())
        self.assertIs(caught.exception,problem);self.assertIs(runtime._geometry,saved);self.assertFalse(guard.held)
        self.assertLess(self.fixture.events.index('observer_exit'),self.fixture.events.index('evaluator_exit'))
        self.assertEqual(self.fixture.events[-1],'lease_exit')
        self.assertFalse((self.root/(self.fixture.control['runtime']+'/native/diagnostic_sidecar.json')).exists())

    def test_changed_supplement_before_original_entry_rejected(self):
        context,guard,state,_,_,_=self.install_worker()
        guard.on_enter=lambda:self.fixture.write(m.SOURCE,b'CPU source changed during fake lease acquisition')
        with context,self.assertRaisesRegex(ValueError,'SHA changed'):m.run(self.root,self.fixture.args())
        self.assertEqual(state['main_calls'],0);self.assertNotIn('checkpoint',self.fixture.events);self.assertFalse(guard.held)

    def test_forwarded_argv_is_only_original743_train_development_entry(self):
        binding=self.fixture.binding();argv=m.forwarded_argv(binding,check=True)
        self.assertEqual(argv[0],m.WORKER);self.assertEqual(argv[-1],'--check')
        self.assertEqual(argv[argv.index('--split')+1],'train');self.assertIn('--development-demo',argv)
        self.assertNotIn('--suite',argv);self.assertNotIn('--validation-release',argv)
        self.assertNotIn('physmani',argv);self.assertEqual(argv[argv.index('--out')+1],self.fixture.control['output'])


if __name__=='__main__':unittest.main()
