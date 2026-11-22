"""CPU/temp-only966 tests; all production resources and entry calls are fakes."""
from collections import Counter
from contextlib import ExitStack, contextmanager, redirect_stdout
from copy import deepcopy
import ast
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import threading
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch


PROJECT=Path(__file__).resolve().parents[1]
SOURCE=PROJECT/'scripts/diagnose_TRAIN_public_collision_composite_v966_r10.py'


def module_at(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module);return module


m=module_at('CPU_train_public_collision_composite966',SOURCE)
post=module_at('CPU_train_public_collision_composite966_post929',PROJECT/m.POST_SOURCE)


def encode(value):return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()


class Fixture:
    """Synthetic code/completion/TRAIN metadata; no model, case or source data."""
    def __init__(self,root,task=None,method='cv_geometry'):
        self.root=Path(root);self.events=[];self.source_sha={}
        for name in m.SOURCES:
            self.write(name,('CPU966 fake source '+name).encode())
            self.source_sha[name]=self.pointer(name)['sha256']
        self.pinned={name:self.source_sha[name] for name in m.PINNED_SOURCES}
        self.guard_sha=self.source_sha[m.GUARD]
        self.post_pins={name:self.source_sha[name] for name in m.POST_PINS}
        self.environment=deepcopy(m.ENVIRONMENT)
        for key,value in self.environment.items():
            if key!='wrapper':self.write(value['path'],dict(CPU_fake_environment=key))
            value['sha256']=self.pointer(value['path'])['sha256']
        self.result=dict(path=m.RESULT,sha256='a'*64)
        self.formal=dict(path=m.FORMAL_LOCK,sha256='b'*64)
        self.resume=dict(path=m.FORMAL_CONTROL,sha256='c'*64)
        self.controller=dict(path='runtime/framework/ordered_composite_TEST_v958/launch.json',sha256='d'*64)
        self.postprocessing={key:dict(path=name,sha256=m.sha(('CPU966 unread metadata '+name).encode()))
            for key,name in m.POST_PATHS.items()}
        self.summary_pointer=dict(path=m.SUMMARY,sha256='e'*64)
        #966 delegates the complete metadata gate to pinned965. These declared
        #pointers have no backing postprocessing/result bytes in this fixture.
        task=task or m.TASKS[0];raw='CPU_episode0';suffix='/'+task+'/'+raw+'/'+method
        config=m.CONFIG_BASE+'/'+task+'_ours_configuration.json'
        self.write(config,dict(context_checkpoint=deepcopy(m.CHECKPOINT),CPU_fake_configuration=True))
        self.write(m.TRAIN_INPUTS,dict(files={'/UNOPENED_TRAIN/never-read.hdf5':'e'*64}))
        self.roster=[dict(task=t,raw_episode_id='CPU_episode'+str(i),split='train',path='/UNOPENED_TRAIN/'+t+'/'+str(i)+'.hdf5')
            for t in post.TASKS for i in range(100)]
        self.write(m.TRAIN_MANIFEST,dict(split='train',episodes=self.roster))
        self.control=dict(schema=m.CONTROL_SCHEMA,execution_enabled=True,formal_score_contribution=False,TEST_used_for_fitting=False,
            selection=dict(task=task,method=method,raw_episode_id=raw),configuration=self.pointer(config),
            original_frozen_inputs=self.pointer(m.TRAIN_INPUTS),train_manifest=self.pointer(m.TRAIN_MANIFEST),
            checkpoint=deepcopy(m.CHECKPOINT),environment=self.environment,completed_TEST_results=self.result,
            formal_resume_lock=self.formal,saved_motion_summary=self.summary_pointer,
            postprocessing=deepcopy(self.postprocessing),
            runtime=m.RUNTIME+suffix,output=m.OUTPUT+suffix,max_events=100,diagnostic_sources=dict(self.source_sha))
        self.reseal()

    def write(self,name,value):
        path=self.root/name;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(value if type(value) is bytes else encode(value))
    def json(self,name):return json.loads((self.root/name).read_bytes())
    def pointer(self,name):return dict(path=name,sha256=m.sha((self.root/name).read_bytes()))
    def reseal(self):
        self.write(m.CONTROL,self.control)
        files={str(self.root/name):checksum for name,checksum in self.control['diagnostic_sources'].items()}
        for value in (self.control['configuration'],self.control['original_frozen_inputs'],self.control['train_manifest'],
                *self.control['environment'].values(),*self.control['postprocessing'].values()):files[str(self.root/value['path'])]=value['sha256']
        files[str(self.root/m.CONTROL)]=self.pointer(m.CONTROL)['sha256']
        self.lock=dict(schema=m.LOCK_SCHEMA,approved_for_execution=True,files=files);self.write(m.INPUTS,self.lock)
    def args(self,check=False):
        return m.parser().parse_args(['--execute',*(['--check'] if check else []),'--lock-sha256',self.pointer(m.INPUTS)['sha256'],
            '--test-results-sha256',self.result['sha256'],'--resume-lock-sha256',self.formal['sha256'],
            '--motion-summary-sha256',self.control['saved_motion_summary']['sha256'],
            '--postprocessing-lock-sha256',self.control['postprocessing']['input_lock']['sha256']])
    @contextmanager
    def pins(self):
        with patch.object(m,'PINNED_SOURCES',self.pinned),patch.object(m,'POST_PINS',self.post_pins),patch.object(m,'GUARD_SHA',self.guard_sha),patch.object(m,'ENVIRONMENT',self.environment):yield
    def binding(self):
        with self.pins():return m.load_control(self.root,self.args())
    def prepared(self):
        s=self.control['selection'];domain=SimpleNamespace(split='train',raw=s['raw_episode_id'])
        domain.require=Mock(return_value='train_qualification');domain.receipt=lambda:dict(split='train',purpose='train_qualification',raw=domain.raw)
        return dict(args=SimpleNamespace(split='train',task=s['task'],method=s['method'],raw_episode_id=s['raw_episode_id']),
            component_domain=domain,config=dict(context_checkpoint=deepcopy(m.CHECKPOINT)),identity={'CPU_fake_identity':True},row={'CPU_original_row':True})
    def assembly(self,guard=None):
        env=SimpleNamespace(map_service=SimpleNamespace(),robot_model=object())
        port=SimpleNamespace(map=env.map_service,model=env.robot_model)
        actor=SimpleNamespace(collision_port=port);result=(actor,object(),object(),object(),object());calls=[]
        runtime=SimpleNamespace(factory='original')
        def geometry(*args):
            if guard is not None:assert guard.held and runtime.factory=='outer_factory'
            self.events.append('geometry');calls.append(args);return result
        runtime._geometry=geometry
        return runtime,env,result,calls
    def tap(self,guard=None,fail_exit=None,fail_receipt=None):
        fixture=self
        class Tap:
            def __init__(self,map_service,**kwargs):
                if guard is not None:assert guard.held and kwargs['permit'] is guard.permit
                fixture.events.append('tap_construct');self.map=map_service;self.kwargs=kwargs;self.restored=False
                fixture.taps.append(self)
            def __enter__(self):fixture.events.append('tap_enter');return self
            def __exit__(self,*exc):
                fixture.events.append('tap_exit');self.restored=True
                if fail_exit:raise fail_exit
            def receipt(self):
                if fail_receipt:raise fail_receipt
                return dict(restored=self.restored,native_validated=False,unqualified=True,events=[],CPU_fake=True)
        self.taps=[];return SimpleNamespace(TrainPublicCollisionTap=Tap)


class FakeGuard:
    def __init__(self,fixture):
        self.fixture=fixture;self.held=False;self.permit=object();self.enter_calls=0;self.on_enter=None
        self.proof=dict(complete=True,expected_cases=1920,source_cases=640,TEST_results=fixture.result,
            resume_control_lock=fixture.formal,resume_control=fixture.resume,actual_controller=fixture.controller,policy_sha256=m.POLICY,
            task_count=16,motion_summary=fixture.summary_pointer,postprocessing_input_lock=fixture.postprocessing['input_lock'],
            postprocessing=deepcopy(fixture.postprocessing))
        self.require_complete_TEST_metadata=Mock(side_effect=self.full)
    def full(self,root,result_sha,**kwargs):
        assert root==self.fixture.root and result_sha==self.fixture.result['sha256']
        assert kwargs==dict(resume_lock_sha256=self.fixture.formal['sha256'],
            motion_summary_sha256=self.fixture.summary_pointer['sha256'],postprocessing_lock_sha256=self.fixture.postprocessing['input_lock']['sha256'])
        self.fixture.events.append('direct_full_TEST');return deepcopy(self.proof)
    @contextmanager
    def reserved_resources_after_TEST(self,root,result_sha,**kwargs):
        assert result_sha==self.fixture.result['sha256'] and kwargs==dict(resume_lock_sha256=self.fixture.formal['sha256'],
            motion_summary_sha256=self.fixture.summary_pointer['sha256'],postprocessing_lock_sha256=self.fixture.postprocessing['input_lock']['sha256'],enabled=True)
        self.enter_calls+=1;self.fixture.events.append('lease_enter');self.held=True
        try:
            if self.on_enter:self.on_enter()
            yield self.permit
        finally:self.held=False;self.fixture.events.append('lease_exit')
    def require_permit(self,permit,root,*,allow_test_callbacks=False):
        assert allow_test_callbacks is False
        assert self.held and permit is self.permit;self.fixture.events.append('permit')
        return dict(complete=True,CPU_fake_resource_proof=True)


class EntryTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.fixture=Fixture(self.root)
        self.denied=Mock(side_effect=AssertionError('CPU966 test reached production resource/admission/Native/model/proc'))
        for name in ('_load_guard','_load_tap','_load_worker','_require_checkpoint','_environment','_process_identity','_validate_native_case','_import'):
            scope=patch.object(m,name,self.denied);scope.start();self.addCleanup(scope.stop)

    @contextmanager
    def gate(self,guard=None):
        guard=guard or FakeGuard(self.fixture)
        with self.fixture.pins(),patch.object(m,'_load_guard',return_value=guard):yield guard

    def motion(self):return m.require_complete_motion_metadata(self.root,self.fixture.binding(),self.fixture.args())

    def test_default_import_and_check_without_execute_are_zero_IO(self):
        source=SOURCE.read_bytes();module=ModuleType('CPU966_inert');module.__file__=str(SOURCE)
        denied=Mock(side_effect=AssertionError('disabled936 performed IO'))
        with patch.object(Path,'resolve',denied),patch.object(Path,'read_bytes',denied),patch.object(Path,'exists',denied):
            exec(compile(source,str(SOURCE),'exec'),module.__dict__)
            for argv in ([],['--check']):
                result=module.main(argv);self.assertFalse(result['enabled']);self.assertFalse(result['materials_created'])
        self.denied.assert_not_called()

    def test_five_external_pins_and_execute_are_mandatory_before_any_gate(self):
        for key in ('execute','lock_sha256','test_results_sha256','resume_lock_sha256','motion_summary_sha256','postprocessing_lock_sha256'):
            args=self.fixture.args();setattr(args,key,False if key=='execute' else None)
            with self.fixture.pins(),self.subTest(key=key),self.assertRaises(ValueError):m.run(self.root,args)
        self.denied.assert_not_called()

    def test_declaration_load_reads_only_small_code_control_and_lock(self):
        read=m.raw;seen=[]
        def record(root,name,*args):seen.append(name);return read(root,name,*args)
        with self.fixture.pins(),patch.object(m,'raw',side_effect=record):binding=m.load_control(self.root,self.fixture.args())
        self.assertEqual(set(seen),{m.INPUTS,m.CONTROL,*m.SOURCES})
        self.assertEqual(binding['control']['checkpoint'],m.CHECKPOINT)
        self.assertFalse((self.root/m.CHECKPOINT['path']).exists());self.assertFalse((self.root/m.RESULT).exists())
        self.assertNotIn(str(self.root/m.SUMMARY),self.fixture.lock['files'])
        self.assertEqual(set(self.fixture.lock['files']),{str(self.root/name) for name in {*m.SOURCES,m.CONTROL,
            self.fixture.control['configuration']['path'],m.TRAIN_INPUTS,m.TRAIN_MANIFEST,
            *(p['path'] for p in self.fixture.environment.values()),*m.POST_PATHS.values()}})
        self.denied.assert_not_called()

    def test_original_frozen_source_pins_are_exact_source_bytes_only(self):
        for name,checksum in m.PINNED_SOURCES.items():self.assertEqual(m.sha((PROJECT/name).read_bytes()),checksum,name)
        self.assertEqual(m.PINNED_SOURCES[m.ENV_SOURCE],m.ENVIRONMENT['wrapper']['sha256'])
        self.assertEqual(set(m.TASKS),{'insert_onto_rotating_peg','insert_onto_rotating_peg_high_speed'})
        self.assertEqual(set(m.METHODS),{'cv_geometry','context_geometry'})

    def test_geometry_training_assembly_and_case_checks_keep_original936_bodies(self):
        def bodies(path):
            return {node.name:ast.dump(node,include_attributes=False) for node in ast.parse(path.read_bytes()).body
                if isinstance(node,ast.FunctionDef)}
        old=bodies(PROJECT/'scripts/diagnose_TRAIN_public_collision_v936_r10.py');new=bodies(SOURCE)
        for name in ('canonical','raw','verify_training_bindings','_require_checkpoint','_environment',
                '_load_worker','forwarded_argv','scoped_geometry','_process_identity','write_fresh_json','_validate_native_case'):
            self.assertEqual(new[name],old[name],name)

    def test_wrong_task_method_raw_checkpoint_source_or_small_alias_refused(self):
        original=deepcopy(self.fixture.control)
        for key,value in (('task','remove_cups_from_rotating_frame'),('method','physmani'),('raw_episode_id','../TEST')):
            self.fixture.control=deepcopy(original);self.fixture.control['selection'][key]=value;self.fixture.reseal()
            with self.fixture.pins(),self.subTest(key=key),self.assertRaises(ValueError):m.load_control(self.root,self.fixture.args())
        for mode in ('checkpoint','source','extra','alias','environment'):
            self.fixture.control=deepcopy(original)
            if mode=='checkpoint':self.fixture.control['checkpoint']['sha256']='f'*64
            if mode=='source':self.fixture.control['diagnostic_sources'][m.WORKER]='f'*64
            if mode=='environment':self.fixture.control['environment']=deepcopy(self.fixture.environment);self.fixture.control['environment']['manifest']['sha256']='f'*64
            self.fixture.reseal()
            if mode in ('extra','alias'):
                self.fixture.lock['files'][m.SOURCE if mode=='alias' else str(self.root/'unopened.pt')]='e'*64
                self.fixture.write(m.INPUTS,self.fixture.lock)
            with self.fixture.pins(),self.subTest(mode=mode),self.assertRaises(ValueError):m.load_control(self.root,self.fixture.args())
        self.denied.assert_not_called()

    def test_direct_full_TEST_missing_failed_or_old_canary_precedes_all_post_reads(self):
        for change in (dict(complete=False),dict(expected_cases=6,source_cases=2),dict(policy_sha256='e'*64)):
            guard=FakeGuard(self.fixture);guard.proof.update(change);binding=self.fixture.binding();seen=[];read=m.raw
            def record(root,name,*args):seen.append(name);return read(root,name,*args)
            with self.gate(guard),patch.object(m,'raw',side_effect=record),self.subTest(change=change),self.assertRaises(ValueError):
                m.require_complete_motion_metadata(self.root,binding,self.fixture.args())
            self.assertEqual(seen,[]);self.assertEqual(guard.enter_calls,0)
        guard=FakeGuard(self.fixture);guard.require_complete_TEST_metadata.side_effect=ValueError('incomplete907');binding=self.fixture.binding()
        with self.gate(guard),patch.object(m,'raw') as read,self.assertRaisesRegex(ValueError,'incomplete907'):
            m.require_complete_motion_metadata(self.root,binding,self.fixture.args())
        read.assert_not_called()

    def test_complete_delegated965_proof_never_reads_postprocessing_summary_or_TRAIN(self):
        binding=self.fixture.binding();seen=[];read=m.raw
        def record(root,name,*args):seen.append(name);return read(root,name,*args)
        with self.gate(),patch.object(m,'raw',side_effect=record):proof=m.require_complete_motion_metadata(self.root,binding,self.fixture.args())
        self.assertTrue(proof['complete']);self.assertFalse(proof['score_or_motion_values_used'])
        self.assertEqual(seen,[]);self.assertEqual(self.fixture.events,['direct_full_TEST'])
        self.assertFalse((self.root/m.SUMMARY).exists())
        self.assertFalse(any((self.root/name).exists() for name in m.POST_PATHS.values()))

    def test_delegated_result_motion_lock_policy_and_full_grid_are_bound(self):
        for key,value in (('TEST_results',dict(path=m.RESULT,sha256='0'*64)),
                ('resume_control_lock',dict(path=m.FORMAL_LOCK,sha256='0'*64)),
                ('motion_summary',dict(path=m.SUMMARY,sha256='0'*64)),
                ('postprocessing_input_lock',dict(path=m.POST_PATHS['input_lock'],sha256='0'*64)),
                ('task_count',15),('postprocessing',{})):
            guard=FakeGuard(self.fixture);guard.proof[key]=value
            with self.gate(guard),self.subTest(key=key),self.assertRaises(ValueError):self.motion()
            self.assertEqual(guard.enter_calls,0)

    def test_unfinished965_pin_or_external_postprocessing_lock_cannot_enter_gate(self):
        with self.fixture.pins(),patch.object(m,'GUARD_SHA',None),self.assertRaisesRegex(ValueError,'reviewed965'):
            m.load_control(self.root,self.fixture.args())
        args=self.fixture.args();args.postprocessing_lock_sha256='0'*64
        with self.fixture.pins(),self.assertRaisesRegex(ValueError,'activation-lock SHA'):
            m.load_control(self.root,args)
        self.denied.assert_not_called()

    def test_training_first_four_full_roster_and_actual671_configuration(self):
        binding=self.fixture.binding();seen=[]
        def validate(manifest,split):
            self.assertEqual(split,'train');self.assertEqual(manifest['split'],'train')
            self.assertEqual(Counter(row['task'] for row in manifest['episodes']),{task:100 for task in post.TASKS})
            seen.append('original_roster_validator');return manifest['episodes']
        with patch.object(m,'_import',return_value=SimpleNamespace(validate_roster=validate)):
            self.assertEqual(m.verify_training_bindings(self.root,binding)['context_checkpoint'],m.CHECKPOINT)
            binding['control']['selection']['raw_episode_id']='CPU_episode4'
            with self.assertRaisesRegex(ValueError,'first four'):m.verify_training_bindings(self.root,binding)
        self.assertEqual(len(seen),2);self.assertFalse((self.root/m.CHECKPOINT['path']).exists())
        config=binding['control']['configuration'];changed=self.fixture.json(config['path']);changed['context_checkpoint']['sha256']='e'*64
        self.fixture.write(config['path'],changed);binding['control']['configuration']=self.fixture.pointer(config['path'])
        with self.assertRaisesRegex(ValueError,'actual671'):m.verify_training_bindings(self.root,binding)

    def test_scoped_original_geometry_once_tuple_identity_and_same_production_permit(self):
        binding=self.fixture.binding();runtime,env,result,calls=self.fixture.assembly();old=runtime._geometry;events={};p=self.fixture.prepared();permit=object()
        tap=self.fixture.tap()
        with m.scoped_geometry(self.root,runtime,tap,binding,permit,events):
            self.assertIs(runtime._geometry(p,env,'CPU descriptions'),result)
            with self.assertRaisesRegex(ValueError,'single case'):runtime._geometry(p,env,'again')
        self.assertIs(runtime._geometry,old);self.assertEqual(len(calls),1);self.assertTrue(events['geometry_hook_restored'])
        observed=self.fixture.taps[0];self.assertIs(observed.kwargs['permit'],permit);self.assertIs(observed.kwargs['domain'],p['component_domain'])
        self.assertIs(observed.kwargs['port'],result[0].collision_port);self.assertIs(observed.map,env.map_service)
        self.assertEqual(observed.kwargs['max_events'],100);self.assertTrue(observed.restored)

    def test_wrong_geometry_domain_or_public_assembly_never_constructs_tap(self):
        binding=self.fixture.binding()
        for mode in ('split','source','model','domain','checkpoint'):
            runtime,env,result,_=self.fixture.assembly();p=self.fixture.prepared();tap=self.fixture.tap();events={}
            if mode=='split':p['args'].split='test'
            if mode=='source':p['args'].raw_episode_id='other'
            if mode=='model':result[0].collision_port.model=object()
            if mode=='domain':p['component_domain'].require.return_value='formal'
            if mode=='checkpoint':p['config']['context_checkpoint']['sha256']='e'*64
            with self.subTest(mode=mode),self.assertRaises(ValueError):
                with m.scoped_geometry(self.root,runtime,tap,binding,object(),events):runtime._geometry(p,env,None)
            self.assertEqual(self.fixture.taps,[]);self.assertTrue(events['geometry_hook_restored'])

    def test_scope_thread_overlap_and_original_exception_restore(self):
        binding=self.fixture.binding();runtime,env,result,_=self.fixture.assembly();old=runtime._geometry;p=self.fixture.prepared();events={};errors=[]
        original=RuntimeError('CPU original native error')
        with self.assertRaises(RuntimeError) as caught:
            with m.scoped_geometry(self.root,runtime,self.fixture.tap(fail_receipt=ValueError('CPU logging error')),binding,object(),events):
                def invoke():
                    try:runtime._geometry(p,env,None)
                    except BaseException as error:errors.append(error)
                thread=threading.Thread(target=invoke);thread.start();thread.join();self.assertEqual(len(errors),1)
                with self.assertRaisesRegex(ValueError,'one isolated'):
                    with m.scoped_geometry(self.root,runtime,self.fixture.tap(),binding,object(),{}):pass
                runtime._geometry(p,env,None);raise original
        self.assertIs(caught.exception,original);self.assertIs(runtime._geometry,old);self.assertTrue(self.fixture.taps[0].restored)
        self.assertEqual(events['tap_receipt_recording_error'],'ValueError')
        with m.scoped_geometry(self.root,runtime,self.fixture.tap(),binding,object(),{}):pass

    @contextmanager
    def integration(self,*,worker_error=None):
        fixture=self.fixture;guard=FakeGuard(fixture);runtime,env,result,calls=fixture.assembly(guard);original_geometry=runtime._geometry
        counts=dict(prepare=0,cpu_preflight=0,native=0);evaluator=object();p=fixture.prepared()
        @contextmanager
        def declared():
            self.assertTrue(guard.held);fixture.events.append('declared_enter')
            try:yield
            finally:fixture.events.append('declared_exit')
        @contextmanager
        def evaluate(actual):
            self.assertIs(actual,evaluator);self.assertEqual(runtime.factory,'original');runtime.factory='outer_factory';fixture.events.append('factory_enter')
            try:yield
            finally:runtime.factory='original';fixture.events.append('factory_exit')
        def worker_main():
            self.assertTrue(guard.held);self.assertEqual(runtime.factory,'outer_factory')
            self.assertEqual(sys.argv,m.forwarded_argv(dict(control=fixture.control),check='--check' in sys.argv))
            counts['prepare']+=1;counts['cpu_preflight']+=1;fixture.events.append('worker_prepare')
            if '--check' in sys.argv:
                self.assertIs(runtime._geometry,original_geometry)
                print(json.dumps(dict(complete=True,native_started=False,GPU_started=False,actual_split='train')));return
            counts['native']+=1;self.assertIs(runtime._geometry(p,env,'descriptions'),result)
            if worker_error:raise worker_error
            fixture.events.append('worker_cleanup');output=fixture.control['output']
            fixture.write(output+'/case.json',b'{"official_task_success":false,"CPU_original_case_bytes":"preserved"}\n')
            print('CPU fake original Native stdout')
        def roster_import(root,name):
            self.assertTrue(guard.held);self.assertEqual(name,'our_mani.multitask.evaluation_roster16_v283')
            return SimpleNamespace(validate_roster=lambda value,split:value['episodes'])
        def checkpoint(*args):self.assertTrue(guard.held);fixture.events.append('checkpoint')
        def environment(*args):self.assertTrue(guard.held);fixture.events.append('environment');return dict(complete=True,CPU_fake=True)
        def native_validator(case,events,binding):
            self.assertTrue(guard.held);self.assertTrue(events['geometry_hook_restored']);self.assertTrue(events['tap_receipts'][0]['restored'])
            self.assertFalse(case['official_task_success']);fixture.events.append('validate_original_case')
        write=m.write_fresh_json
        def fresh(path,value):self.assertTrue(guard.held);return write(path,value)
        with self.gate(guard),patch.object(m,'_import',side_effect=roster_import),patch.object(m,'_require_checkpoint',side_effect=checkpoint), \
                patch.object(m,'_environment',side_effect=environment),patch.object(m,'_process_identity',return_value=dict(pid=444,start_ticks=445,argv=['CPU966'])), \
                patch.object(m,'_load_worker',return_value=(SimpleNamespace(main=worker_main,runtime=runtime),
                    SimpleNamespace(declared_edge_scope=declared,evaluator_scope=evaluate),evaluator)), \
                patch.object(m,'_load_tap',return_value=fixture.tap(guard)),patch.object(m,'_validate_native_case',side_effect=native_validator), \
                patch.object(m,'write_fresh_json',side_effect=fresh):yield guard,counts,runtime,calls

    def test_CPU_then_native_fresh_leaves_original_prepare_runs_both_and_stdout_passthrough(self):
        with self.integration() as (guard,counts,runtime,calls):
            with redirect_stdout(io.StringIO()) as cpu_out:cpu=m.run(self.root,self.fixture.args(check=True))
            self.assertEqual(cpu_out.getvalue(),'');self.assertFalse(cpu['execution']['tap_installed']);self.assertEqual(self.fixture.taps,[])
            self.assertFalse((self.root/self.fixture.control['output']).exists())
            with redirect_stdout(io.StringIO()) as native_out:native=m.run(self.root,self.fixture.args())
            self.assertEqual(native_out.getvalue(),'CPU fake original Native stdout\n');self.assertEqual(counts,dict(prepare=2,cpu_preflight=2,native=1))
            self.assertEqual(guard.enter_calls,2);self.assertEqual(len(calls),1);self.assertEqual(runtime.factory,'original')
            self.assertFalse(native['execution']['native_success']);self.assertTrue(native['execution']['case_original_bytes_unchanged'])
            self.assertEqual((self.root/self.fixture.control['output']/'case.json').read_bytes(),b'{"official_task_success":false,"CPU_original_case_bytes":"preserved"}\n')
            for phase in ('CPU_check','native'):self.assertTrue((self.root/self.fixture.control['runtime']/phase/'diagnostic_sidecar.json').exists())
            for field in ('native_validated','formal_qualification','formal_score_contribution','participates_in_VAL_TEST','TEST_used_for_fitting',
                    'policy_fix_claimed','predicates_rewritten','author_stages_observed','saved_CPU_receipt_used_as_native_permit','physical_equivalence_claimed'):
                self.assertIs(native[field],False,field)
            self.assertEqual(native['forwarded_worker_argv'],m.forwarded_argv(self.fixture.binding()))
            self.assertEqual(native['actual_entry'],self.fixture.pointer(m.SOURCE));self.assertEqual(native['base_worker'],self.fixture.pointer(m.WORKER))
            self.assertLess(self.fixture.events.index('tap_exit'),self.fixture.events.index('factory_exit',self.fixture.events.index('tap_exit')))
            with self.assertRaisesRegex(ValueError,'fresh outputs'):m.run(self.root,self.fixture.args())
            self.assertEqual(counts['native'],1)

    def test_changed_readiness_on_lease_enter_blocks_TRAIN_and_worker(self):
        with self.integration() as (guard,counts,_,__):
            guard.on_enter=lambda:guard.proof.update(source_cases=639)
            with self.assertRaisesRegex(ValueError,'full1920'):m.run(self.root,self.fixture.args(check=True))
            self.assertEqual(counts['prepare'],0);self.assertNotIn('checkpoint',self.fixture.events)
            self.assertFalse((self.root/self.fixture.control['runtime']).exists());self.assertFalse(guard.held)

    def test_native_original_exception_failure_write_does_not_mask_restore_or_release(self):
        original=RuntimeError('CPU exact original exception');write=m.write_fresh_json
        with self.integration(worker_error=original) as (guard,counts,runtime,calls):
            def fail_record(path,value):
                if Path(path).name=='failure.json':raise OSError('CPU sidecar write failed')
                return write(path,value)
            with patch.object(m,'write_fresh_json',side_effect=fail_record),self.assertRaises(RuntimeError) as caught:
                m.run(self.root,self.fixture.args())
            self.assertIs(caught.exception,original);self.assertFalse(guard.held);self.assertEqual(runtime.factory,'original')
            self.assertTrue(self.fixture.taps[0].restored);self.assertEqual(counts['native'],1)
            self.assertFalse((self.root/self.fixture.control['runtime']/'native/diagnostic_sidecar.json').exists())
            self.assertEqual(self.fixture.events[-1],'lease_exit')

    def test_scope_error_normal_recording_failure_still_releases_process_lock(self):
        binding=self.fixture.binding();runtime,env,result,_=self.fixture.assembly();events={};old=runtime._geometry
        with self.assertRaisesRegex(ValueError,'CPU receipt'):
            with m.scoped_geometry(self.root,runtime,self.fixture.tap(fail_receipt=ValueError('CPU receipt')),binding,object(),events):
                runtime._geometry(self.fixture.prepared(),env,None)
        self.assertIs(runtime._geometry,old)
        with m.scoped_geometry(self.root,runtime,self.fixture.tap(),binding,object(),{}):pass


if __name__=='__main__':unittest.main()
