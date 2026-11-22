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
SOURCE=PROJECT/'scripts/continue_post_composite_TEST_motion_v964_r10.py'


def module_at(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module);return module


m=module_at('post_TEST_motion_controller964_subject',SOURCE)


def encode(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()


class Fixture:
    """Only temp source stubs and completion metadata. No case/source bytes."""
    def __init__(self,root):
        self.root=Path(root)
        for name in m.SOURCES:self.write(name,('CPU964 source stub: '+name).encode())
        fixed=lambda path:dict(path=path,sha256=m.sha(encode({'invented_pointer':path})))
        ids=['invented_job_'+str(index) for index in range(1920)]
        protocol={'schema':'CPU fake original author outcome','runtime_consumer':fixed('scripts/runtime744.py')}
        profile=dict(schema='r10_frozen_concurrent_native_resource_profile_v902',max_native_slots=4,
            max_GPU_slots=2,max_CPU_preflight_slots=2,ready_limit=8)
        original=dict(schema='r10_composite_formal_TEST_resume_control_v955',split='test',expected_cases=1920,
            source_count=640,policy_sha256=m.POLICY,physical_policy_changed=False,author_outcome_protocol=protocol,
            source_inputs=fixed('artifacts/framework/multitask_r10/fixture826/test_source_inputs.json'),
            validation_release=fixed('artifacts/framework/multitask_r10/fixture823/VAL_release.json'),
            resource_profile=profile,resource_qualification=fixed('runtime/framework/fixture904/qualification.json'),
            execution_implementation={'schema':'CPU fake original implementation'},
            resume_owner=self.pointer(m.OWNER_SOURCE),resume_auditor=self.pointer(m.AUDITOR_SOURCE),
            result_auditor=self.pointer(m.AUDITOR_SOURCE),resume_controller=self.pointer(m.CONTROLLER_SOURCE),
            completed_job_ids=ids[:584],failed_job_ids=ids[584:590],
            original_launch=fixed(m.HISTORICAL_ORIGINAL+'/launch.json'),
            original_terminal=fixed(m.HISTORICAL_ORIGINAL+'/terminal.json'),
            previous_launch=fixed(m.HISTORICAL_PREVIOUS+'/launch.json'),
            previous_status=fixed(m.HISTORICAL_PREVIOUS+'/status.json'),
            previous_terminal=fixed(m.HISTORICAL_PREVIOUS+'/terminal.json'))
        self.write(m.RESUME_CONTROL,original)
        self.write(m.RESUME_LOCK,dict(schema='r10_composite_formal_TEST_resume_inputs_v955',approved_for_execution=True,
            control=self.pointer(m.RESUME_CONTROL),files={name:self.pointer(name)['sha256'] for name in
                (m.RESUME_CONTROL,m.OWNER_SOURCE,m.AUDITOR_SOURCE,m.CONTROLLER_SOURCE)}))
        cp,lp=self.pointer(m.RESUME_CONTROL),self.pointer(m.RESUME_LOCK)
        common=dict(split='test',policy_sha256=m.POLICY,author_outcome_protocol=protocol,
            resume_control=cp,resume_control_lock=lp,automatic_retry=False)
        argv=[m.OWNER_SOURCE,'--suite','fixture/suite.json','--suite-lock','fixture/suite_lock.json',
            '--source-inputs',original['source_inputs']['path'],'--validation-release',original['validation_release']['path'],
            '--runtime',m.OWNER_RUNTIME,'--out','artifacts/framework/multitask_r10/formal_TEST_composite_resume_v956/test',
            '--resume-control',m.RESUME_CONTROL,'--resume-lock',m.RESUME_LOCK,'--resume-lock-sha256',lp['sha256']]
        self.write(m.OWNER_RUNTIME+'/launch.json',dict(common,pid=956,start_ticks=95600,
            source_sha256=self.pointer(m.OWNER_SOURCE)['sha256'],argv=argv))
        completed={};tiers=[];cases=[];drivers=[];batches=[]
        specs=((m.HISTORICAL_ORIGINAL,753,75300,134),(m.HISTORICAL_PREVIOUS,902,90200,450),
            (m.OWNER_RUNTIME,956,95600,1336))
        index=0
        for runtime,pid,start,count in specs:
            for _ in range(count):
                key=ids[index];case=fixed('artifacts/framework/multitask_r10/invented_cases/'+key+'/case.json')
                driver=fixed('artifacts/framework/multitask_r10/invented_cases/'+key+'/admission_driver_receipt.json')
                batch=None if runtime==m.HISTORICAL_ORIGINAL else fixed('artifacts/framework/multitask_r10/invented_cases/'+key+'/batch_driver_receipt.json')
                completed[key]=dict(complete=True,returncode=0,interrupted=False,timed_out=False,automatic_retry=False,
                    native_success=False,motion_pass=None if index%7==0 else False,case=case,admission_driver_receipt=driver)
                if batch is not None:completed[key]['batch_driver_receipt']=batch;batches.append(batch)
                tiers.append(dict(job_id=key,evidence_runtime=runtime,imported_completed=runtime!=m.OWNER_RUNTIME,
                    actual_stage_owner_pid=pid,actual_stage_owner_start_ticks=start,case=case,admission_driver_receipt=driver,
                    batch_driver_receipt=batch,execution_entry=dict(kind='original751' if batch is None else 'batched901')))
                cases.append(case);drivers.append(driver);index+=1
        self.write(m.OWNER_RUNTIME+'/status.json',dict(common,expected=1920,completed=completed,failed={},active=[],checking=[],
            ready_jobs=[],CPU_ready=0,native_slots=0,GPU_slots=0,stop_requested=False,drain_requested=False,owner_error=None,
            imported_completed=584,CPU_launched=1336))
        self.write(m.OWNER_RUNTIME+'/terminal.json',dict(common,complete=True,expected=1920,completed=1920,failed=0,
            owner_error=None,stop_requested=False,drain_requested=False,drained=False,CPU_ready_unexecuted=[],CPU_not_launched=0,
            formal_release_requires_independent_audit=True,imported_completed_cases=584,resumed_new_cases=1336))
        env={'current_mount_namespace':'mnt:[CPU fake only]'}
        self.write(m.CONTROLLER+'/launch.json',dict(common,pid=958,start_ticks=95800,
            source_sha256=self.pointer(m.CONTROLLER_SOURCE)['sha256'],frozen_environment=env))
        self.write(m.CONTROLLER+'/status.json',dict(stage='complete',automatic_retry=False))
        audit_argv=[m.AUDITOR_SOURCE,'--resume-control',m.RESUME_CONTROL,'--resume-lock',m.RESUME_LOCK,'--resume-lock-sha256',lp['sha256']]
        for stage,source,pid,start,args in (('formal_TEST_1920',m.OWNER_SOURCE,956,95600,argv),
                ('independent_TEST_results',m.AUDITOR_SOURCE,957,95700,audit_argv)):
            leaf=m.CONTROLLER+'/stages/'+stage
            self.write(leaf+'/launch.json',dict(argv=[str(self.root/'.conda-env/bin/python'),'-B','-u',*args],
                pid=pid,start_ticks=start,source_sha256=self.pointer(source)['sha256'],automatic_retry=False,
                inherited_frozen_environment=env,current_mount_namespace=env['current_mount_namespace']))
            self.write(leaf+'/terminal.json',dict(complete=True,returncode=0,automatic_retry=False))
        stage_owners=[dict(runtime=runtime,pid=pid,start_ticks=start,completed=count,
            launch=original['original_launch'] if runtime==m.HISTORICAL_ORIGINAL else original['previous_launch'] if runtime==m.HISTORICAL_PREVIOUS else self.pointer(m.OWNER_RUNTIME+'/launch.json'),
            terminal=original['original_terminal'] if runtime==m.HISTORICAL_ORIGINAL else original['previous_terminal'] if runtime==m.HISTORICAL_PREVIOUS else self.pointer(m.OWNER_RUNTIME+'/terminal.json'))
            for runtime,pid,start,count in specs]
        result=dict(common,schema=m.RESULT_SCHEMA,complete=True,source_cases=640,method_cases=1920,completed_method_cases=1920,
            methods=list(m.METHODS),tasks=list(m.TASKS),source_sha256=self.pointer(m.AUDITOR_SOURCE)['sha256'],
            source_input_lock=original['source_inputs'],validation_release=original['validation_release'],
            independent_composite_audit=True,resource_profile=profile,resource_qualification=original['resource_qualification'],
            execution_implementation=original['execution_implementation'],native_failures_retained=True,source_outcome_filtering=False,
            Value_NGFF_FASTER_used=False,used_evidence_rechecked_at_close=True,original754_or906_single_result_claimed=False,
            cases=cases,admission_driver_receipts=drivers,evidence_tiers=tiers,batch_driver_receipts=batches,actual_stage_owners=stage_owners,
            native_successes_by_method={method:0 for method in m.METHODS},
            auditor_invocation=dict(pid=957,start_ticks=95700,source=self.pointer(m.AUDITOR_SOURCE),argv=audit_argv),
            historical_infrastructure_attempts=dict(job_ids=original['failed_job_ids'],previous_status=original['previous_status'],
                previous_terminal=original['previous_terminal'],counted_as_native_task_failures=False,
                all_keys_reexecuted_once_in_new_namespace=True))
        self.write(m.RESULT,result)
        self.write(m.CONTROLLER+'/terminal.json',dict(complete=True,automatic_retry=False,TEST_results=self.pointer(m.RESULT)))
        self.control=dict(schema=m.SCHEMA,execution_enabled=True,read_only_analysis=True,automatic_retry=False,
            runtime=m.RUNTIME,controller_runtime=m.CONTROLLER,owner_runtime=m.OWNER_RUNTIME,
            test_results=m.RESULT,motion_summary=m.SUMMARY_OUTPUT,poll_seconds=1,policy_sha256=m.POLICY,
            sources={name:self.pointer(name) for name in m.SOURCES},resume_control=cp,resume_control_lock=lp,
            controller_launch=self.pointer(m.CONTROLLER+'/launch.json'))
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
            actual_test_auditor=binding['sources'][m.AUDITOR_SOURCE],full1920_three_tier_TEST_results_and_complete958_observed=True,
            provenance_runtime_counts=completion['provenance_runtime_counts'],actual_stage_owners=completion['actual_stage_owners'],
            TEST_used_for_fitting=False,TXT_or_handoff_written=False,
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
        raw=SOURCE.read_bytes();module=ModuleType('CPU964_inert');module.__file__=str(SOURCE)
        denied=Mock(side_effect=AssertionError('inert964 performed I/O'))
        with patch.object(Path,'resolve',denied),patch.object(Path,'read_bytes',denied),patch.object(Path,'exists',denied), \
                patch.object(m.fcntl,'flock',denied),patch.object(m.subprocess,'Popen',denied),redirect_stdout(io.StringIO()) as out:
            exec(compile(raw,str(SOURCE),'exec'),module.__dict__);module.main([])
        value=json.loads(out.getvalue());self.assertFalse(value['execution_enabled']);self.assertFalse(value['jobs_started'])

    def test_exact11_file_small_lock_binds_actual958_batch(self):
        control,binding=self.fixture.load()
        self.assertEqual(len(binding['canonical']),11)
        self.assertEqual(set(binding['sources']),set(m.SOURCES))
        self.assertEqual(binding['controller_launch'],control['controller_launch'])
        self.assertEqual(binding['original_control']['expected_cases'],1920)

    def test_pending_actual958_does_not_read_result_or_any_case(self):
        self.fixture.mutate(m.CONTROLLER+'/status.json','stage','formal_TEST_1920')
        control,binding=self.fixture.load()
        original=m.read_bytes;seen=[]
        def read(root,name,*args,**kwargs):
            seen.append(name);return original(root,name,*args,**kwargs)
        with patch.object(m,'read_bytes',side_effect=read):
            self.assertIsNone(m.completion_snapshot(self.root,control,binding))
        self.assertNotIn(m.RESULT,seen);self.assertFalse(any(p.endswith('/case.json') for p in seen))

    def test_incomplete956_or_unclean_stage_cannot_read_result(self):
        for path,key,value in ((m.OWNER_RUNTIME+'/terminal.json','completed',1919),
                (m.CONTROLLER+'/stages/independent_TEST_results/terminal.json','returncode',1)):
            with self.subTest(path=path):
                previous=(self.root/path).read_bytes();self.fixture.mutate(path,key,value)
                control,binding=self.fixture.load();original=m.read_bytes;seen=[]
                def read(root,name,*args,**kwargs):seen.append(name);return original(root,name,*args,**kwargs)
                with patch.object(m,'read_bytes',side_effect=read),self.assertRaises(ValueError):
                    m.completion_snapshot(self.root,control,binding)
                self.assertNotIn(m.RESULT,seen);self.fixture.write(path,previous)

    def test_complete_actual958_956_957_snapshot_has_no_score_gate(self):
        _,_,completion=self.fixture.completion()
        self.assertEqual(completion['result'],self.fixture.pointer(m.RESULT))
        self.assertEqual(self.fixture.json(m.RESULT)['native_successes_by_method'],{method:0 for method in m.METHODS})
        for name in completion.values():
            if isinstance(name,dict) and 'path' in name:self.assertNotIn('/case.json',name['path'])
        self.assertEqual(completion['provenance_runtime_counts'],{m.HISTORICAL_ORIGINAL:134,m.HISTORICAL_PREVIOUS:450,m.OWNER_RUNTIME:1336})
        self.assertEqual(len(self.fixture.json(m.RESULT)['batch_driver_receipts']),1786)

    def test_three_tier_duplicate_wrong_owner_and_skipped_historical_infra_are_refused(self):
        original=self.fixture.json(m.RESULT)
        for mode in ('duplicate','tier','owner','batchcount','infra','nativeint','motionint','unclean'):
            result=deepcopy(original)
            if mode=='duplicate':result['evidence_tiers'][-1]['job_id']=result['evidence_tiers'][0]['job_id']
            elif mode=='tier':result['actual_stage_owners'][1]['runtime']=m.HISTORICAL_ORIGINAL
            elif mode=='owner':
                result['actual_stage_owners'][-1]['pid']=999
                for row in result['evidence_tiers'][584:]:row['actual_stage_owner_pid']=999
            elif mode=='batchcount':result['batch_driver_receipts']=result['batch_driver_receipts'][:1336]
            elif mode=='infra':result['historical_infrastructure_attempts']['counted_as_native_task_failures']=True
            else:
                status=self.fixture.json(m.OWNER_RUNTIME+'/status.json');first=next(iter(status['completed'].values()))
                first['native_success' if mode=='nativeint' else 'motion_pass' if mode=='motionint' else 'returncode']=0 if mode!='unclean' else 1
                self.fixture.write(m.OWNER_RUNTIME+'/status.json',status)
            self.fixture.write(m.RESULT,result)
            final=self.fixture.json(m.CONTROLLER+'/terminal.json');final['TEST_results']=self.fixture.pointer(m.RESULT)
            self.fixture.write(m.CONTROLLER+'/terminal.json',final)
            with self.subTest(mode=mode),self.assertRaises(ValueError):self.fixture.completion()
            if mode in ('nativeint','motionint','unclean'):
                status=self.fixture.json(m.OWNER_RUNTIME+'/status.json');first=next(iter(status['completed'].values()))
                first.update(native_success=False,motion_pass=None,returncode=0);self.fixture.write(m.OWNER_RUNTIME+'/status.json',status)
                # Rebind the new owner terminal/launch pointers if needed: status
                # snapshots themselves are not advertised as immutable pointers.
        self.fixture.write(m.RESULT,original)

    def test_old906_result_header_never_qualifies_the_new_batch(self):
        result=self.fixture.json(m.RESULT);result['schema']='r10_composite_concurrent_resumed_formal_TEST_results_v906'
        self.fixture.write(m.RESULT,result)
        final=self.fixture.json(m.CONTROLLER+'/terminal.json');final['TEST_results']=self.fixture.pointer(m.RESULT)
        self.fixture.write(m.CONTROLLER+'/terminal.json',final)
        with self.assertRaises(ValueError):self.fixture.completion()

    def test_reader_refuses_raw_model_manifest_case_or_unlisted_sidecar_before_open(self):
        denied=Mock(side_effect=AssertionError('not a metadata read'))
        with patch.object(Path,'read_bytes',denied):
            for name in ('models/checkpoint.pt','artifacts/framework/multitask_r10/fake/TEST_manifest.json',
                    'artifacts/framework/multitask_r10/fake/case.json','raw/episode0.hdf5','/etc/passwd'):
                with self.subTest(name=name),self.assertRaises(ValueError):m.read_bytes(self.root,name)
        denied.assert_not_called()

    def test_summary_wrong_provenance_or_unknown_motion_filter_claim_is_refused(self):
        control,binding,completion=self.fixture.completion();summary=self.summary(control,binding,completion)
        for field,value in (('provenance_runtime_counts',{m.OWNER_RUNTIME:1920}),
                ('actual_stage_owners',[]),('success_rate_or_acceleration_filter_used',True)):
            changed=deepcopy(summary);changed[field]=value
            sp=dict(path=m.SUMMARY_OUTPUT,sha256='a'*64)
            stdout=dict(complete=True,output=m.SUMMARY_OUTPUT,sha256='a'*64,case_count=1920)
            with self.subTest(field=field),self.assertRaises(ValueError):m.validate_summary(changed,sp,stdout,control,binding,completion,self.root)

    def test_wrong_lock_source_or_batch_controller_pointer_refused(self):
        with self.assertRaisesRegex(ValueError,'SHA changed'):
            m.load_control(m.CONTROL,m.LOCK,'0'*64,root=self.root)
        self.fixture.write(m.SUMMARY_SOURCE,b'CPU fake altered963')
        with self.assertRaisesRegex(ValueError,'SHA changed'):self.fixture.load()

    def test_wrong_selected958_launch_and_STOP_precede_any_analysis(self):
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

    def test_waits_then_launches_only_one963_and_records_host_scope(self):
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

    def test_STOP_waiting_and_active_only_cancels_owned963(self):
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

    def test_failed_or_stopped958_raises_without_reading_result(self):
        for value,kind in ((dict(complete=False,user_stop=False),RuntimeError),(dict(complete=False,user_stop=True),InterruptedError)):
            self.fixture.write(m.CONTROLLER+'/failure.json',value)
            control,binding=self.fixture.load();original=m.read_bytes;seen=[]
            def read(root,name,*args,**kwargs):seen.append(name);return original(root,name,*args,**kwargs)
            with patch.object(m,'read_bytes',side_effect=read),self.assertRaises(kind):m.completion_snapshot(self.root,control,binding)
            self.assertNotIn(m.RESULT,seen)


if __name__=='__main__':unittest.main()
