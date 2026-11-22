"""Temp/in-memory CPU controller fakes. Never real metadata, proc or children."""
from contextlib import contextmanager, redirect_stdout
from copy import deepcopy
import importlib.util
import io
import json
from pathlib import Path
from types import ModuleType
import tempfile
import unittest
from unittest.mock import Mock, patch


PROJECT=Path(__file__).resolve().parents[1]
SOURCE=PROJECT/'scripts/continue_post_TEST_motion_v929_r10.py'


def module_at(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module);return module


m=module_at('post_TEST_motion_controller929_subject',SOURCE)
fakes=module_at('post_TEST_motion_controller929_metadata_fakes',PROJECT/'tests/test_post_TEST_resource_guard_v924.py')


def encode(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()


class Fixture:
    """Only temp source stubs and completion metadata. No case/source bytes."""
    def __init__(self,root):
        self.root=Path(root)
        with patch.object(fakes,'ROOT',self.root):self.memory=fakes.Fixture()
        for name,value in self.memory.data.items():
            self.write(name,value)
        for name in m.SOURCES:
            if not (self.root/name).exists():self.write(name,('CPU929 fake source: '+name).encode())
        self.control=dict(schema=m.SCHEMA,execution_enabled=True,read_only_analysis=True,automatic_retry=False,
            runtime=m.RUNTIME,controller_runtime=m.CONTROLLER,owner_runtime=m.OWNER_RUNTIME,
            test_results=m.RESULT,motion_summary=m.SUMMARY_OUTPUT,poll_seconds=1,policy_sha256=m.POLICY,
            sources={name:self.pointer(name) for name in m.SOURCES},resume_control=self.pointer(m.RESUME_CONTROL),
            resume_control_lock=self.pointer(m.RESUME_LOCK),controller_launch=self.pointer(m.CONTROLLER+'/launch.json'))
        self.write(m.CONTROL,self.control)
        names={*m.SOURCES,m.CONTROL,m.RESUME_CONTROL,m.RESUME_LOCK,m.CONTROLLER+'/launch.json'}
        self.lock=dict(schema=m.LOCK_SCHEMA,execution_enabled=True,files={name:self.pointer(name)['sha256'] for name in names})
        self.write(m.LOCK,self.lock)

    def write(self,name,value):
        path=self.root/name;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(value if isinstance(value,bytes) else encode(value))

    def json(self,name):return json.loads((self.root/name).read_bytes())
    def pointer(self,name):return dict(path=name,sha256=m.sha((self.root/name).read_bytes()))
    def load(self):return m.load_control(m.CONTROL,m.LOCK,self.pointer(m.LOCK)['sha256'],root=self.root)
    def mutate(self,name,key,value):
        data=self.json(name);data[key]=value;self.write(name,data)
    def reseal(self):
        self.write(m.CONTROL,self.control)
        self.lock['files'][m.CONTROL]=self.pointer(m.CONTROL)['sha256'];self.write(m.LOCK,self.lock)
    def completion(self):
        control,binding=self.load();return control,binding,m.completion_snapshot(self.root,control,binding)


class Child:
    def __init__(self,code=0):
        self.pid=990;self.returncode=code;self.terminated=False;self.killed=False
    def poll(self):return self.returncode
    def wait(self,timeout=None):
        if self.returncode is None:raise AssertionError('CPU fake waited for an uncompleted child')
        return self.returncode
    def terminate(self):self.terminated=True;self.returncode=-15
    def kill(self):self.killed=True;self.returncode=-9


class ControllerTests(unittest.TestCase):
    def setUp(self):
        m.EVENT.clear();self.addCleanup(m.EVENT.clear)
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.fixture=Fixture(self.root);self.lock_events=[]
        @contextmanager
        def lock(root):
            self.lock_events.append('enter')
            try:yield
            finally:self.lock_events.append('exit')
        for name,value in (('observer_lock',lock),('ticks',lambda pid:pid+1),('_wait',Mock(return_value=False)),
                ('_popen',Mock(side_effect=AssertionError('CPU test attempted an unconfigured child')))):
            scope=patch.object(m,name,value);scope.start();self.addCleanup(scope.stop)
        scope=patch.object(m.os,'getpid',return_value=777);scope.start();self.addCleanup(scope.stop)

    def execute(self):
        return m.execute(m.CONTROL,m.LOCK,self.fixture.pointer(m.LOCK)['sha256'],root=self.root)

    def summary(self,control,binding,completion):
        return dict(schema=m.SUMMARY_SCHEMA,complete=True,read_only_analysis=True,split='test',task_count=16,
            source_count=640,case_count=1920,methods=list(m.METHODS),task_source_counts={task:40 for task in m.TASKS},
            policy_sha256=m.POLICY,source_sha256=binding['sources'][m.SUMMARY_SOURCE]['sha256'],
            test_results=completion['result'],resume_control=control['resume_control'],resume_control_lock=control['resume_control_lock'],
            actual_test_auditor=binding['sources'][m.AUDITOR_SOURCE],full1920_composite_TEST_results_and_complete907_observed=True,
            new_release_granted_by_this_summary=False,native_false_retained=True,success_rate_or_acceleration_filter_used=False,
            native_GPU_or_jobs_started=False,inputs_sha256={
                str(self.root/m.RESULT):completion['result_sha256'],str(self.root/m.RESUME_CONTROL):control['resume_control']['sha256'],
                str(self.root/m.RESUME_LOCK):control['resume_control_lock']['sha256'],
                str(self.root/m.SUMMARY_SOURCE):binding['sources'][m.SUMMARY_SOURCE]['sha256']})

    def popen(self,*,code=0,wrong_output=False,pending=False):
        calls=[];child=Child(None if pending else code)
        control,binding,completion=self.fixture.completion()
        def launch(argv,**kwargs):
            calls.append((deepcopy(argv),kwargs))
            if code==0 and not pending:
                output=self.summary(control,binding,completion)
                if wrong_output:output['resume_control_lock']={'path':m.RESUME_LOCK,'sha256':'f'*64}
                self.fixture.write(m.SUMMARY_OUTPUT,output)
                kwargs['stdout'].write(encode(dict(complete=True,output=m.SUMMARY_OUTPUT,
                    sha256=self.fixture.pointer(m.SUMMARY_OUTPUT)['sha256'],case_count=1920)))
                kwargs['stdout'].flush()
            return child
        return launch,calls,child

    def test_import_and_default_main_are_inert(self):
        raw=SOURCE.read_bytes();module=ModuleType('CPU929_inert');module.__file__=str(SOURCE)
        denied=Mock(side_effect=AssertionError('inert929 performed I/O'))
        with patch.object(Path,'resolve',denied),patch.object(Path,'read_bytes',denied),patch.object(Path,'exists',denied), \
                patch.object(m.fcntl,'flock',denied),patch.object(m.subprocess,'Popen',denied),redirect_stdout(io.StringIO()) as out:
            exec(compile(raw,str(SOURCE),'exec'),module.__dict__);module.main([])
        value=json.loads(out.getvalue());self.assertFalse(value['execution_enabled']);self.assertFalse(value['jobs_started'])

    def test_exact11_file_small_lock_binds_actual907_batch(self):
        control,binding=self.fixture.load()
        self.assertEqual(len(binding['canonical']),11)
        self.assertEqual(set(binding['sources']),set(m.SOURCES))
        self.assertEqual(binding['controller_launch'],control['controller_launch'])
        self.assertEqual(binding['original_control']['expected_cases'],1920)

    def test_pending_actual907_does_not_read_result_or_any_case(self):
        self.fixture.mutate(m.CONTROLLER+'/status.json','stage','formal_TEST_1920')
        control,binding=self.fixture.load()
        original=m.read_bytes;seen=[]
        def read(root,name,*args,**kwargs):
            seen.append(name);return original(root,name,*args,**kwargs)
        with patch.object(m,'read_bytes',side_effect=read):
            self.assertIsNone(m.completion_snapshot(self.root,control,binding))
        self.assertNotIn(m.RESULT,seen);self.assertFalse(any(p.endswith('/case.json') for p in seen))

    def test_incomplete902_or_unclean_stage_cannot_read_result(self):
        for path,key,value in ((m.OWNER_RUNTIME+'/terminal.json','completed',1919),
                (m.CONTROLLER+'/stages/independent_TEST_results/terminal.json','returncode',1)):
            with self.subTest(path=path):
                previous=(self.root/path).read_bytes();self.fixture.mutate(path,key,value)
                control,binding=self.fixture.load();original=m.read_bytes;seen=[]
                def read(root,name,*args,**kwargs):seen.append(name);return original(root,name,*args,**kwargs)
                with patch.object(m,'read_bytes',side_effect=read),self.assertRaises(ValueError):
                    m.completion_snapshot(self.root,control,binding)
                self.assertNotIn(m.RESULT,seen);self.fixture.write(path,previous)

    def test_complete_actual907_902_906_snapshot_has_no_score_gate(self):
        _,_,completion=self.fixture.completion()
        self.assertEqual(completion['result'],self.fixture.pointer(m.RESULT))
        self.assertEqual(self.fixture.json(m.RESULT)['native_successes_by_method'],{method:0 for method in m.METHODS})
        for name in completion.values():
            if isinstance(name,dict) and 'path' in name:self.assertNotIn('/case.json',name['path'])

    def test_wrong_lock_source_or_batch_controller_pointer_refused(self):
        with self.assertRaisesRegex(ValueError,'SHA changed'):
            m.load_control(m.CONTROL,m.LOCK,'0'*64,root=self.root)
        self.fixture.write(m.SUMMARY_SOURCE,b'CPU fake altered920')
        with self.assertRaisesRegex(ValueError,'SHA changed'):self.fixture.load()

    def test_wrong_selected907_launch_and_STOP_precede_any_analysis(self):
        self.fixture.control['controller_launch']['sha256']='f'*64;self.fixture.reseal()
        with self.assertRaisesRegex(ValueError,'batch pointer differs'):self.fixture.load()
        m.EVENT.set()
        with patch.object(m,'observer_lock') as lock,self.assertRaises(InterruptedError):self.execute()
        lock.assert_not_called()
        self.assertFalse((self.root/m.RUNTIME).exists())

    def test_duplicate_alias_extra_external_or_missing_small_lock_refused(self):
        original=deepcopy(self.fixture.lock)
        for mode in ('duplicate','extra','external','missing'):
            lock=deepcopy(original)
            if mode=='duplicate':lock['files'][str(self.root/m.SOURCE)]=lock['files'][m.SOURCE]
            elif mode=='extra':lock['files']['models/never-open.pt']='a'*64
            elif mode=='external':value=lock['files'].pop(m.SOURCE);lock['files']['/outside/fake.py']=value
            else:lock['files'].pop(m.SUMMARY_TEST)
            self.fixture.write(m.LOCK,lock)
            with self.subTest(mode=mode),self.assertRaises(ValueError):self.fixture.load()

    def test_waits_then_launches_only_one920_and_records_host_scope(self):
        _,_,ready=self.fixture.completion();snapshots=iter((None,ready,ready))
        launch,calls,child=self.popen()
        with patch.object(m,'completion_snapshot',side_effect=lambda *a:next(snapshots)),patch.object(m,'_popen',side_effect=launch):
            terminal=self.execute()
        self.assertTrue(terminal['complete']);self.assertEqual(len(calls),1);self.assertEqual(self.lock_events,['enter','exit'])
        argv=calls[0][0];self.assertEqual(argv[:4],[str(self.root/'.conda-env/bin/python'),'-B','-u',m.SUMMARY_SOURCE])
        self.assertEqual(set(p.name for p in (self.root/m.RUNTIME/'stages').iterdir()),{m.STAGE})
        for filename in ('launch.json','stages/'+m.STAGE+'/launch.json','terminal.json'):
            record=self.fixture.json(m.RUNTIME+'/'+filename)
            self.assertTrue(record['standard_host_analysis']);self.assertFalse(record['frozen_environment_inherited_claimed'])
            self.assertNotIn('frozen_environment',record)
        self.assertFalse(terminal['Native_GPU_TRAIN_started']);self.assertFalse(terminal['TXT_modified'])
        with self.assertRaisesRegex(ValueError,'no restart'):self.execute()
        self.assertEqual(len(calls),1)

    def test_existing_summary_never_starts_or_overwrites(self):
        self.fixture.write(m.SUMMARY_OUTPUT,b'CPU existing saved report')
        with self.assertRaisesRegex(ValueError,'no duplicate child'):self.execute()
        self.assertEqual((self.root/m.SUMMARY_OUTPUT).read_bytes(),b'CPU existing saved report')
        self.assertFalse((self.root/m.RUNTIME).exists())

    def test_child_failure_or_wrong_final_binding_never_retries(self):
        for code,wrong in ((1,False),(0,True)):
            with self.subTest(code=code,wrong=wrong),tempfile.TemporaryDirectory() as name:
                old_root,old_fixture=self.root,self.fixture
                try:
                    self.root=Path(name);self.fixture=Fixture(self.root)
                    launch,calls,_=self.popen(code=code,wrong_output=wrong)
                    with patch.object(m,'_popen',side_effect=launch),self.assertRaises(ValueError):self.execute()
                    self.assertEqual(len(calls),1);self.assertFalse((self.root/m.RUNTIME/'terminal.json').exists())
                    failure=self.fixture.json(m.RUNTIME+'/failure.json');self.assertFalse(failure['automatic_retry'])
                    self.assertEqual(self.fixture.json(m.RUNTIME+'/status.json')['stage'],'requires_concrete_repair')
                finally:self.root,self.fixture=old_root,old_fixture

    def test_STOP_waiting_and_active_only_cancels_owned920(self):
        _,_,ready=self.fixture.completion()
        with patch.object(m,'completion_snapshot',return_value=None),patch.object(m,'_wait',side_effect=lambda seconds:m.EVENT.set()):
            with self.assertRaises(InterruptedError):self.execute()
        self.assertTrue(self.fixture.json(m.RUNTIME+'/failure.json')['user_stop'])
        self.assertFalse((self.root/m.RUNTIME/'stages').exists())
        m.EVENT.clear()
        with tempfile.TemporaryDirectory() as name:
            self.root=Path(name);self.fixture=Fixture(self.root);launch,calls,child=self.popen(pending=True)
            with patch.object(m,'_popen',side_effect=launch),patch.object(m,'_wait',side_effect=lambda seconds:m.EVENT.set()):
                with self.assertRaises(InterruptedError):self.execute()
            self.assertEqual(len(calls),1);self.assertTrue(child.terminated);self.assertFalse(child.killed)
            self.assertTrue(self.fixture.json(m.RUNTIME+'/stages/'+m.STAGE+'/terminal.json')['interrupted'])
            self.assertEqual(self.fixture.json(m.CONTROLLER+'/status.json')['stage'],'complete')

    def test_source_change_during_final_snapshot_rejected_before_popen(self):
        original=m.completion_snapshot;count=0
        def snapshot(*args):
            nonlocal count
            count+=1;value=original(*args)
            if count==2:self.fixture.write(m.SUMMARY_SOURCE,b'CPU fake source swapped during final snapshot')
            return value
        with patch.object(m,'completion_snapshot',side_effect=snapshot),patch.object(m,'_popen') as popen:
            with self.assertRaises(ValueError):self.execute()
            popen.assert_not_called()

    def test_pid_reuse_never_signals_an_unowned_child(self):
        child=Child(None)
        with patch.object(m,'ticks',return_value=99999),self.assertRaisesRegex(ValueError,'changed child PID/start'):
            m._stop_child(child,991)
        self.assertFalse(child.terminated);self.assertFalse(child.killed)

    def test_failed_or_stopped907_raises_without_reading_result(self):
        for value,kind in ((dict(complete=False,user_stop=False),RuntimeError),(dict(complete=False,user_stop=True),InterruptedError)):
            self.fixture.write(m.CONTROLLER+'/failure.json',value)
            control,binding=self.fixture.load();original=m.read_bytes;seen=[]
            def read(root,name,*args,**kwargs):seen.append(name);return original(root,name,*args,**kwargs)
            with patch.object(m,'read_bytes',side_effect=read),self.assertRaises(kind):m.completion_snapshot(self.root,control,binding)
            self.assertNotIn(m.RESULT,seen)


if __name__=='__main__':unittest.main()
