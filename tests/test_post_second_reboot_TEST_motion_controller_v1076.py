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
SOURCE=PROJECT/'scripts/continue_post_second_reboot_TEST_motion_v1076_r10.py'


def module_at(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module);return module


m=module_at('post_TEST_motion_controller1076_subject',SOURCE)


def encode(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()


class Fixture:
    """Seven fabricated metadata tiers; never actual cases, models or children."""
    def __init__(self,root,*,parent_change=None,fallback=False,actual_quota=None):
        self.root=Path(root)
        for name in m.SOURCES:self.write(name,('CPU1076 source stub: '+name).encode())
        fixed=lambda path:dict(path=path,sha256=m.sha(encode({'invented_pointer':path})))
        ids=['invented_job_'+str(index) for index in range(1920)]
        sorted_ids=lambda values:sorted(values)
        protocol={'schema':'CPU fake original author outcome','runtime_consumer':fixed('scripts/runtime744.py')}
        implementation={'schema':'CPU fake original implementation'}
        science=dict(split='test',expected_cases=1920,source_count=640,policy_sha256=m.POLICY,physical_policy_changed=False,
            author_outcome_protocol=protocol,execution_implementation=implementation)
        old_profile=dict(schema='r10_frozen_concurrent_native_resource_profile_v902',max_native_slots=4,max_GPU_slots=2,max_CPU_preflight_slots=2,ready_limit=8)
        baseline=dict(schema='r10_frozen_concurrent_native_resource_profile_v1005',max_native_slots=8,max_GPU_slots=3,max_CPU_preflight_slots=4,ready_limit=16)
        high=dict(schema='r10_frozen_concurrent_native_resource_profile_v1031',max_native_slots=12,max_GPU_slots=5,max_CPU_preflight_slots=6,ready_limit=24)
        low=dict(schema='r10_frozen_concurrent_native_resource_profile_v1031',max_native_slots=8,max_GPU_slots=3,max_CPU_preflight_slots=4,ready_limit=16)
        profile=deepcopy(baseline if fallback else low if actual_quota is None else actual_quota)
        oldq=fixed('runtime/framework/concurrent_TRAIN_resource_qualification_v1005.json')
        highq=fixed('runtime/framework/concurrent_TRAIN_resource_qualification_v1031.json')
        def resources(actual,prior,priorq):
            return dict(resource_profile=deepcopy(actual),resource_capacity_envelope=deepcopy(baseline if fallback and actual==profile else high),
                resource_profile_within_capacity=True,resource_capacity_source='qualified1005_fallback' if fallback and actual==profile else 'qualified1031',
                resource_qualification=oldq if fallback and actual==profile else highq,
                resource_qualifier=fixed('scripts/audit_concurrent_TRAIN_resource_canary_v'+('1005' if fallback and actual==profile else '1031')+'_r10.py'),
                resource_profile_changed=actual!=prior,previous_resource_profile=deepcopy(prior),previous_resource_qualification=priorq)
        great=dict(science,schema='r10_composite_formal_TEST_resume_control_v955',resource_profile=old_profile,
            resource_qualification=fixed('runtime/framework/fixture904/qualification.json'),completed_job_ids=ids[:584],failed_job_ids=ids[584:590],
            previous_launch=fixed(m.HISTORICAL_PREVIOUS+'/launch.json'),previous_status=fixed(m.HISTORICAL_PREVIOUS+'/status.json'),previous_terminal=fixed(m.HISTORICAL_PREVIOUS+'/terminal.json'))
        self.write(m.GREATGRANDPARENT_CONTROL,great)
        grand=dict(science,schema='r10_four_tier_composite_formal_TEST_resume_control_v1010',resource_profile=baseline,resource_qualification=oldq,
            completed_job_ids=ids[:1266],failed_job_ids=[],previous_resume_control=self.pointer(m.GREATGRANDPARENT_CONTROL),resource_profile_changed=True,
            previous_resource_profile=old_profile,previous_resource_qualification=great['resource_qualification'],
            previous_launch=fixed(m.HISTORICAL_PARENT+'/launch.json'),previous_status=fixed(m.HISTORICAL_PARENT+'/status.json'),previous_terminal=fixed(m.HISTORICAL_PARENT+'/terminal.json'))
        grand.update(parent_change or {});self.write(m.GRANDPARENT_CONTROL,grand)
        parent=dict(science,**resources(high,baseline,oldq),schema='r10_five_tier_composite_formal_TEST_resume_control_v1040',
            completed_job_ids=ids[:1409],failed_job_ids=[],previous_resume_control=self.pointer(m.GRANDPARENT_CONTROL),
            previous_launch=fixed(m.HISTORICAL_LATEST+'/launch.json'),previous_status=fixed(m.HISTORICAL_LATEST+'/status.json'),previous_terminal=fixed(m.HISTORICAL_LATEST+'/terminal.json'))
        self.write(m.PARENT_CONTROL,parent)
        def reboot_record(schema,boot,accepted,rejected,remaining,saved,previous_boot=None):
            obj=dict(schema=schema,audit_complete=True,formal_full_TEST_complete=False,old_outputs_modified=False,
                missing_evidence_reconstructed=False,unknown_counted_as_NativeFalse=False,success_based_case_selection=False,
                new_terminal_claimed_for_interrupted_owner=False,unregistered_clean_native_terminals=[],current_boot_id=boot,unknown_interrupted_cases=[])
            if previous_boot:obj['previous_boot_id']=previous_boot
            for key,value in dict(accepted_completed_job_ids=accepted,rejected_saved_completed_job_ids=rejected,remaining_job_ids=remaining,saved_completed_job_ids=saved).items():
                obj[key]=sorted(value);obj[key+'_sha256']=m.sha(encode(sorted(value)))
            return obj
        first_boot='00000000-0000-0000-0000-000000001054';boot='00000000-0000-0000-0000-000000001067'
        first=reboot_record('r10_readonly_host_reboot_TEST_evidence_observation_v1054',first_boot,ids[:1411],ids[1411:1413],ids[1411:],ids[:1413])
        second=reboot_record('r10_readonly_host_second_reboot_TEST_evidence_observation_v1067',boot,ids[:1419],[],ids[1419:],ids[:1419],first_boot)
        self.write(m.FIRST_REBOOT_AUDIT,first);self.write(m.REBOOT_AUDIT,second)
        for name,version,bid in ((m.FIRST_HARDWARE_OBSERVATION,'1059',first_boot),(m.HARDWARE_OBSERVATION,'1068',boot)):
            self.write(name,dict(schema='reboot_hardware_boot_point_observation_v'+version,boot_id=bid,old_owner_created_before_current_boot=True,
                old_start_ticks_impossible_in_current_boot=True,Native_started=False,historical_1031_capacity_reused=True,capacity_canary_rerun_after_reboot=False))
        previous=dict(science,**resources(high,high,highq),schema='r10_six_tier_reboot_formal_TEST_resume_control_v1055',explicit_host_reboot_recovery=True,
            previous_resume_control=self.pointer(m.PARENT_CONTROL),completed_job_ids=sorted(ids[:1411]),remaining_job_ids=sorted(ids[1411:]),
            saved_completed_job_ids=sorted(ids[:1413]),rejected_saved_completed_job_ids=sorted(ids[1411:1413]),failed_job_ids=[],
            previous_launch=fixed(m.HISTORICAL_FIFTH+'/launch.json'),previous_status=fixed(m.HISTORICAL_FIFTH+'/status.json'),
            reboot_audit=self.pointer(m.FIRST_REBOOT_AUDIT),host_reboot_observation=self.pointer(m.FIRST_HARDWARE_OBSERVATION))
        self.write(m.PREVIOUS_CONTROL,previous)
        actual_resources=resources(profile,high,highq)
        original=dict(science,**actual_resources,schema='r10_seven_tier_reboot_formal_TEST_resume_control_v1070',explicit_host_reboot_recovery=True,
            source_inputs=fixed('artifacts/framework/multitask_r10/fixture826/test_source_inputs.json'),validation_release=fixed('artifacts/framework/multitask_r10/fixture823/VAL_release.json'),
            resume_owner=self.pointer(m.OWNER_SOURCE),resume_auditor=self.pointer(m.AUDITOR_SOURCE),result_auditor=self.pointer(m.AUDITOR_SOURCE),resume_controller=self.pointer(m.CONTROLLER_SOURCE),
            previous_resume_control=self.pointer(m.PREVIOUS_CONTROL),completed_job_ids=sorted(ids[:1419]),remaining_job_ids=sorted(ids[1419:]),
            saved_completed_job_ids=sorted(ids[:1419]),rejected_saved_completed_job_ids=[],failed_job_ids=[],inherited_completed_job_ids=sorted(ids[:1411]),
            fresh_completed_job_ids=sorted(ids[1411:1419]),old_semantic_case_evidence_inherited=True,full_original_input_maps_reverified=True,
            original_launch=fixed(m.HISTORICAL_ORIGINAL+'/launch.json'),original_terminal=fixed(m.HISTORICAL_ORIGINAL+'/terminal.json'),
            previous_launch=fixed(m.HISTORICAL_SIXTH+'/launch.json'),previous_status=fixed(m.HISTORICAL_SIXTH+'/status.json'),
            reboot_audit=self.pointer(m.REBOOT_AUDIT),host_reboot_observation=self.pointer(m.HARDWARE_OBSERVATION))
        self.write(m.RESUME_CONTROL,original)
        names=(m.RESUME_CONTROL,m.PREVIOUS_CONTROL,m.PARENT_CONTROL,m.GRANDPARENT_CONTROL,m.GREATGRANDPARENT_CONTROL,
            m.REBOOT_AUDIT,m.HARDWARE_OBSERVATION,m.FIRST_REBOOT_AUDIT,m.FIRST_HARDWARE_OBSERVATION,m.OWNER_SOURCE,m.AUDITOR_SOURCE,m.CONTROLLER_SOURCE)
        files={name:self.pointer(name)['sha256'] for name in names}
        for key in ('resource_qualification','resource_qualifier'):files[actual_resources[key]['path']]=actual_resources[key]['sha256']
        self.write(m.RESUME_LOCK,dict(schema='r10_seven_tier_reboot_formal_TEST_resume_inputs_v1070',approved_for_execution=True,control=self.pointer(m.RESUME_CONTROL),files=files))
        cp,lp=self.pointer(m.RESUME_CONTROL),self.pointer(m.RESUME_LOCK)
        common=dict(actual_resources,split='test',policy_sha256=m.POLICY,author_outcome_protocol=protocol,resume_control=cp,resume_control_lock=lp,
            automatic_retry=False,infrastructure_recovery=True,reboot_audit=self.pointer(m.REBOOT_AUDIT),boot_id=boot)
        argv=[m.OWNER_SOURCE,'--suite','fixture/suite.json','--suite-lock','fixture/suite_lock.json','--source-inputs',original['source_inputs']['path'],
            '--validation-release',original['validation_release']['path'],'--runtime',m.OWNER_RUNTIME,'--out','artifacts/framework/multitask_r10/formal_TEST_reboot_resume_v1071/test',
            '--resume-control',m.RESUME_CONTROL,'--resume-lock',m.RESUME_LOCK,'--resume-lock-sha256',lp['sha256']]
        self.write(m.OWNER_RUNTIME+'/launch.json',dict(common,pid=1071,start_ticks=107100,source_sha256=self.pointer(m.OWNER_SOURCE)['sha256'],argv=argv))
        completed={};tiers=[];cases=[];drivers=[];batches=[]
        specs=((m.HISTORICAL_ORIGINAL,753,75300,134),(m.HISTORICAL_PREVIOUS,902,90200,450),(m.HISTORICAL_PARENT,956,95600,682),
            (m.HISTORICAL_LATEST,1011,101100,143),(m.HISTORICAL_FIFTH,1041,104100,2),(m.HISTORICAL_SIXTH,1056,105600,8),(m.OWNER_RUNTIME,1071,107100,501))
        index=0
        for runtime,pid,start,count in specs:
            for _ in range(count):
                key=ids[index];case=fixed('artifacts/framework/multitask_r10/invented_cases/'+key+'/case.json');driver=fixed('artifacts/framework/multitask_r10/invented_cases/'+key+'/admission_driver_receipt.json')
                batch=None if runtime==m.HISTORICAL_ORIGINAL else fixed('artifacts/framework/multitask_r10/invented_cases/'+key+'/batch_driver_receipt.json')
                completed[key]=dict(complete=True,returncode=0,interrupted=False,timed_out=False,automatic_retry=False,native_success=False,
                    motion_pass=None if index%7==0 else False,case=case,admission_driver_receipt=driver)
                if batch is not None:completed[key]['batch_driver_receipt']=batch;batches.append(batch)
                tiers.append(dict(job_id=key,evidence_runtime=runtime,imported_completed=runtime!=m.OWNER_RUNTIME,actual_stage_owner_pid=pid,
                    actual_stage_owner_start_ticks=start,case=case,admission_driver_receipt=driver,batch_driver_receipt=batch,execution_entry=dict(kind='original751' if batch is None else 'batched901')))
                cases.append(case);drivers.append(driver);index+=1
        self.write(m.OWNER_RUNTIME+'/status.json',dict(common,expected=1920,completed=completed,failed={},active=[],checking=[],ready_jobs=[],CPU_ready=0,native_slots=0,GPU_slots=0,
            stop_requested=False,drain_requested=False,owner_error=None,imported_completed=1419,CPU_launched=501))
        self.write(m.OWNER_RUNTIME+'/terminal.json',dict(common,complete=True,expected=1920,completed=1920,failed=0,owner_error=None,stop_requested=False,drain_requested=False,drained=False,
            CPU_ready_unexecuted=[],CPU_not_launched=0,formal_release_requires_independent_audit=True,imported_completed_cases=1419,resumed_new_cases=501))
        env={'current_mount_namespace':'mnt:[CPU fake only]'}
        self.write(m.CONTROLLER+'/launch.json',dict(common,pid=1073,start_ticks=107300,source_sha256=self.pointer(m.CONTROLLER_SOURCE)['sha256'],frozen_environment=env))
        self.write(m.CONTROLLER+'/status.json',dict(stage='complete',automatic_retry=False))
        audit_argv=[m.AUDITOR_SOURCE,'--resume-control',m.RESUME_CONTROL,'--resume-lock',m.RESUME_LOCK,'--resume-lock-sha256',lp['sha256']]
        for stage,source,pid,start,args in (('formal_TEST_1920',m.OWNER_SOURCE,1071,107100,argv),('independent_TEST_results',m.AUDITOR_SOURCE,1072,107200,audit_argv)):
            leaf=m.CONTROLLER+'/stages/'+stage
            self.write(leaf+'/launch.json',dict(argv=[str(self.root/'.conda-env/bin/python'),'-B','-u',*args],pid=pid,start_ticks=start,source_sha256=self.pointer(source)['sha256'],
                automatic_retry=False,inherited_frozen_environment=env,current_mount_namespace=env['current_mount_namespace']))
            self.write(leaf+'/terminal.json',dict(complete=True,returncode=0,automatic_retry=False))
        pointers={m.HISTORICAL_ORIGINAL:(original['original_launch'],original['original_terminal']),m.HISTORICAL_PREVIOUS:(great['previous_launch'],great['previous_terminal']),
            m.HISTORICAL_PARENT:(grand['previous_launch'],grand['previous_terminal']),m.HISTORICAL_LATEST:(parent['previous_launch'],parent['previous_terminal']),
            m.HISTORICAL_FIFTH:(previous['previous_launch'],None),m.HISTORICAL_SIXTH:(original['previous_launch'],None),m.OWNER_RUNTIME:(self.pointer(m.OWNER_RUNTIME+'/launch.json'),self.pointer(m.OWNER_RUNTIME+'/terminal.json'))}
        stage_owners=[dict(runtime=runtime,pid=pid,start_ticks=start,completed=count,launch=pointers[runtime][0],terminal=pointers[runtime][1]) for runtime,pid,start,count in specs]
        current_interruption=dict(runtime=m.HISTORICAL_SIXTH,reboot_audit=original['reboot_audit'],host_reboot_observation=original['host_reboot_observation'],
            saved_completed_job_ids=original['saved_completed_job_ids'],rejected_saved_completed_job_ids=[],unknown_interrupted_cases=second['unknown_interrupted_cases'],
            unknown_counted_as_native_task_failures=False,previous_terminal=None,previous_terminal_invented=False,damaged_outputs_retained=True,all_remaining_keys_executed_once_in_new_namespace=True)
        prior_interruption=dict(runtime=m.HISTORICAL_FIFTH,reboot_audit=previous['reboot_audit'],host_reboot_observation=previous['host_reboot_observation'],
            saved_completed_job_ids=previous['saved_completed_job_ids'],rejected_saved_completed_job_ids=previous['rejected_saved_completed_job_ids'],unknown_interrupted_cases=first['unknown_interrupted_cases'],
            unknown_counted_as_native_task_failures=False,previous_terminal=None,previous_terminal_invented=False,damaged_outputs_retained=True,
            first_recovery_owner_completed=False,first_recovery_clean_completions=8,second_recovery_clean_completions=501,all_remaining_keys_eventually_completed_with_clean_evidence=True)
        result=dict(common,schema=m.RESULT_SCHEMA,complete=True,source_cases=640,method_cases=1920,completed_method_cases=1920,methods=list(m.METHODS),tasks=list(m.TASKS),
            source_sha256=self.pointer(m.AUDITOR_SOURCE)['sha256'],source_input_lock=original['source_inputs'],validation_release=original['validation_release'],independent_composite_audit=True,
            execution_implementation=implementation,native_failures_retained=True,source_outcome_filtering=False,Value_NGFF_FASTER_used=False,used_evidence_rechecked_at_close=True,
            original754_906_957_1012_1042_or1057_complete_result_claimed=False,old907_or958_or1013_or1043_or1058_successful_terminal_claimed=False,
            cases=cases,admission_driver_receipts=drivers,evidence_tiers=tiers,batch_driver_receipts=batches,actual_stage_owners=stage_owners,native_successes_by_method={method:0 for method in m.METHODS},
            auditor_invocation=dict(pid=1072,start_ticks=107200,source=self.pointer(m.AUDITOR_SOURCE),argv=audit_argv),
            historical_infrastructure_attempts=[dict(runtime=runtime,job_ids=ctl['failed_job_ids'],previous_status=ctl['previous_status'],previous_terminal=ctl['previous_terminal'],
                counted_as_native_task_failures=False,all_keys_reexecuted_once_in_new_namespace=True) for runtime,ctl in ((m.HISTORICAL_PREVIOUS,great),(m.HISTORICAL_PARENT,grand),(m.HISTORICAL_LATEST,parent))],
            host_reboot_interruption=current_interruption,prior_host_reboot_interruption=prior_interruption,host_reboot_interruptions=[prior_interruption,current_interruption],
            prior_reboot_audit=previous['reboot_audit'],prior_host_reboot_observation=previous['host_reboot_observation'],historical_saved_registered_but_damaged_cases_rerun=2)
        self.write(m.RESULT,result);self.write(m.CONTROLLER+'/terminal.json',dict(complete=True,automatic_retry=False,TEST_results=self.pointer(m.RESULT)))
        self.control=dict(schema=m.SCHEMA,execution_enabled=True,read_only_analysis=True,automatic_retry=False,runtime=m.RUNTIME,controller_runtime=m.CONTROLLER,owner_runtime=m.OWNER_RUNTIME,
            test_results=m.RESULT,motion_summary=m.SUMMARY_OUTPUT,poll_seconds=1,policy_sha256=m.POLICY,sources={name:self.pointer(name) for name in m.SOURCES},resume_control=cp,resume_control_lock=lp,
            controller_launch=self.pointer(m.CONTROLLER+'/launch.json'),reboot_audit=self.pointer(m.REBOOT_AUDIT),host_reboot_observation=self.pointer(m.HARDWARE_OBSERVATION),
            prior_reboot_audit=self.pointer(m.FIRST_REBOOT_AUDIT),prior_host_reboot_observation=self.pointer(m.FIRST_HARDWARE_OBSERVATION))
        self.write(m.CONTROL,self.control)
        names={*m.SOURCES,m.CONTROL,m.RESUME_CONTROL,m.RESUME_LOCK,m.PREVIOUS_CONTROL,m.PARENT_CONTROL,m.GRANDPARENT_CONTROL,m.GREATGRANDPARENT_CONTROL,
            m.REBOOT_AUDIT,m.HARDWARE_OBSERVATION,m.FIRST_REBOOT_AUDIT,m.FIRST_HARDWARE_OBSERVATION,m.CONTROLLER+'/launch.json'}
        self.lock=dict(schema=m.LOCK_SCHEMA,execution_enabled=True,files={name:self.pointer(name)['sha256'] for name in names});self.write(m.LOCK,self.lock)

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
        self.pid=990;self.returncode=code;self.terminated=False;self.killed=False;self.wait_calls=0
    def poll(self):return self.returncode
    def wait(self,timeout=None):
        self.wait_calls+=1
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
        for name,value in (('observer_lock',lock),('current_boot_id',lambda:'00000000-0000-0000-0000-000000001067'),('ticks',lambda pid:pid+1),('_wait',Mock(return_value=False)),
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
            actual_test_auditor=binding['sources'][m.AUDITOR_SOURCE],infrastructure_recovery=True,
            reboot_audit=control['reboot_audit'],host_reboot_observation=control['host_reboot_observation'],
            adopted_completed_cases=1419,saved_registered_completed_cases=1419,damaged_saved_completed_cases_reexecuted=0,
            historical_damaged_saved_completed_cases_reexecuted=2,
            prior_reboot_audit=control['prior_reboot_audit'],prior_host_reboot_observation=control['prior_host_reboot_observation'],
            **{k:completion[k] for k in ('host_reboot_interruption','prior_host_reboot_interruption','host_reboot_interruptions')},
            unknown_motion_not_filled_as_zero_or_boolean=True,full1920_seven_tier_TEST_results_and_complete1073_observed=True,
            **{k:binding['original_control'][k] for k in ('resource_profile','resource_qualification','resource_qualifier',
                'resource_capacity_envelope','resource_profile_within_capacity','resource_capacity_source','resource_profile_changed',
                'previous_resource_profile','previous_resource_qualification')},
            provenance_runtime_counts=completion['provenance_runtime_counts'],actual_stage_owners=completion['actual_stage_owners'],
            TEST_used_for_fitting=False,TXT_or_handoff_written=False,
            new_release_granted_by_this_summary=False,native_false_retained=True,success_rate_or_acceleration_filter_used=False,
            native_GPU_or_jobs_started=False,inputs_sha256={
                str(self.root/m.RESULT):completion['result_sha256'],str(self.root/m.RESUME_CONTROL):control['resume_control']['sha256'],
                str(self.root/m.RESUME_LOCK):control['resume_control_lock']['sha256'],
                str(self.root/m.SUMMARY_SOURCE):binding['sources'][m.SUMMARY_SOURCE]['sha256'],
                str(self.root/m.REBOOT_AUDIT):control['reboot_audit']['sha256'],
                str(self.root/m.HARDWARE_OBSERVATION):control['host_reboot_observation']['sha256'],
                str(self.root/m.FIRST_REBOOT_AUDIT):control['prior_reboot_audit']['sha256'],
                str(self.root/m.FIRST_HARDWARE_OBSERVATION):control['prior_host_reboot_observation']['sha256']})

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
        raw=SOURCE.read_bytes();module=ModuleType('CPU1076_inert');module.__file__=str(SOURCE)
        denied=Mock(side_effect=AssertionError('inert1076 performed I/O'))
        with patch.object(Path,'resolve',denied),patch.object(Path,'read_bytes',denied),patch.object(Path,'exists',denied), \
                patch.object(m.fcntl,'flock',denied),patch.object(m.subprocess,'Popen',denied),redirect_stdout(io.StringIO()) as out:
            exec(compile(raw,str(SOURCE),'exec'),module.__dict__);module.main([])
        value=json.loads(out.getvalue());self.assertFalse(value['execution_enabled']);self.assertFalse(value['jobs_started'])

    def test_exact19_file_small_lock_binds_actual1073_batch_and_four_ancestors(self):
        control,binding=self.fixture.load()
        self.assertEqual(len(binding['canonical']),19)
        self.assertEqual(set(binding['sources']),set(m.SOURCES))
        self.assertEqual(binding['controller_launch'],control['controller_launch'])
        self.assertEqual(binding['original_control']['expected_cases'],1920)
        self.assertEqual(binding['parent_control']['policy_sha256'],m.POLICY)
        self.assertIn(m.PARENT_CONTROL,binding['canonical'])

    def test_parent_control_requires_both_new_and_inherited_declarations(self):
        for missing in ('new1076','inherited1040'):
            with self.subTest(missing=missing),tempfile.TemporaryDirectory() as d:
                fixture=Fixture(Path(d))
                if missing=='new1076':
                    fixture.lock['files'].pop(m.PARENT_CONTROL);fixture.write(m.LOCK,fixture.lock)
                else:
                    inherited=fixture.json(m.RESUME_LOCK);inherited['files'].pop(m.PARENT_CONTROL)
                    fixture.write(m.RESUME_LOCK,inherited)
                    fixture.control['resume_control_lock']=fixture.pointer(m.RESUME_LOCK)
                    fixture.lock['files'][m.RESUME_LOCK]=fixture.pointer(m.RESUME_LOCK)['sha256'];fixture.reseal()
                with self.assertRaises(ValueError):fixture.load()

    def test_bound_parent_schema_policy_protocol_and_resource_transition_are_literal(self):
        for field,value in (('schema','r10_wrong_parent'),('policy_sha256','0'*64),('resource_profile',{}),
                ('physical_policy_changed',True),('author_outcome_protocol',{}),('execution_implementation',{}),
                ('resource_qualification',{'path':'wrong','sha256':'a'*64})):
            with self.subTest(field=field),tempfile.TemporaryDirectory() as d:
                fixture=Fixture(Path(d),parent_change={field:value})
                with self.assertRaises(ValueError):fixture.load()

    def test_pending_actual1043_does_not_read_result_or_any_case(self):
        self.fixture.mutate(m.CONTROLLER+'/status.json','stage','formal_TEST_1920')
        control,binding=self.fixture.load()
        original=m.read_bytes;seen=[]
        def read(root,name,*args,**kwargs):
            seen.append(name);return original(root,name,*args,**kwargs)
        with patch.object(m,'read_bytes',side_effect=read):
            self.assertIsNone(m.completion_snapshot(self.root,control,binding))
        self.assertNotIn(m.RESULT,seen);self.assertFalse(any(p.endswith('/case.json') for p in seen))

    def test_incomplete1041_or_unclean_stage_cannot_read_result(self):
        for path,key,value in ((m.OWNER_RUNTIME+'/terminal.json','completed',1919),
                (m.CONTROLLER+'/stages/independent_TEST_results/terminal.json','returncode',1)):
            with self.subTest(path=path):
                previous=(self.root/path).read_bytes();self.fixture.mutate(path,key,value)
                control,binding=self.fixture.load();original=m.read_bytes;seen=[]
                def read(root,name,*args,**kwargs):seen.append(name);return original(root,name,*args,**kwargs)
                with patch.object(m,'read_bytes',side_effect=read),self.assertRaises(ValueError):
                    m.completion_snapshot(self.root,control,binding)
                self.assertNotIn(m.RESULT,seen);self.fixture.write(path,previous)

    def test_complete_actual1043_1041_1042_snapshot_has_no_score_gate(self):
        _,_,completion=self.fixture.completion()
        self.assertEqual(completion['result'],self.fixture.pointer(m.RESULT))
        self.assertEqual(self.fixture.json(m.RESULT)['native_successes_by_method'],{method:0 for method in m.METHODS})
        for name in completion.values():
            if isinstance(name,dict) and 'path' in name:self.assertNotIn('/case.json',name['path'])
        self.assertEqual(completion['provenance_runtime_counts'],{m.HISTORICAL_ORIGINAL:134,m.HISTORICAL_PREVIOUS:450,m.HISTORICAL_PARENT:682,m.HISTORICAL_LATEST:143,m.HISTORICAL_FIFTH:2,m.HISTORICAL_SIXTH:8,m.OWNER_RUNTIME:501})
        self.assertEqual(len(self.fixture.json(m.RESULT)['batch_driver_receipts']),1786)

    def test_seven_tier_duplicate_wrong_owner_and_skipped_historical_infra_are_refused(self):
        original=self.fixture.json(m.RESULT)
        for mode in ('duplicate','tier','owner','batchcount','infra','nativeint','motionint','unclean'):
            result=deepcopy(original)
            if mode=='duplicate':result['evidence_tiers'][-1]['job_id']=result['evidence_tiers'][0]['job_id']
            elif mode=='tier':result['actual_stage_owners'][1]['runtime']=m.HISTORICAL_ORIGINAL
            elif mode=='owner':
                result['actual_stage_owners'][-1]['pid']=999
                for row in result['evidence_tiers'][584:]:row['actual_stage_owner_pid']=999
            elif mode=='batchcount':result['batch_driver_receipts']=result['batch_driver_receipts'][:1336]
            elif mode=='infra':result['historical_infrastructure_attempts'][0]['counted_as_native_task_failures']=True
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
        self.fixture.write(m.SUMMARY_SOURCE,b'CPU fake altered1045')
        with self.assertRaisesRegex(ValueError,'SHA changed'):self.fixture.load()

    def test_wrong_selected1043_launch_and_STOP_precede_any_analysis(self):
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

    def test_waits_then_launches_only_one1045_and_records_host_scope(self):
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

    def test_STOP_waiting_and_active_only_cancels_owned1045(self):
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

    def test_forked_child_start_capture_failure_is_terminated_reaped_and_original_error_preserved(self):
        launch,calls,child=self.popen(pending=True);failure=OSError('CPU fake proc identity unavailable')
        def read_ticks(pid):
            if pid==child.pid:raise failure
            return pid+1
        with patch.object(m,'_popen',side_effect=launch),patch.object(m,'ticks',side_effect=read_ticks):
            with self.assertRaises(OSError) as caught:self.execute()
        self.assertIs(caught.exception,failure);self.assertEqual(len(calls),1)
        self.assertTrue(child.terminated);self.assertGreaterEqual(child.wait_calls,1)
        terminal=self.fixture.json(m.RUNTIME+'/stages/'+m.STAGE+'/terminal.json')
        self.assertFalse(terminal['complete']);self.assertFalse(terminal['child_start_identity_captured'])
        self.assertIsNone(terminal['child_start_ticks']);self.assertTrue(terminal['owned_analysis_child_reaped'])

    def test_exited_child_start_capture_failure_still_reaps_without_signalling(self):
        launch,calls,child=self.popen(code=1);failure=OSError('CPU fake short lived child')
        with patch.object(m,'_popen',side_effect=launch),patch.object(m,'ticks',side_effect=lambda pid:(_ for _ in ()).throw(failure) if pid==child.pid else pid+1):
            with self.assertRaises(OSError) as caught:self.execute()
        self.assertIs(caught.exception,failure);self.assertFalse(child.terminated);self.assertFalse(child.killed)
        self.assertGreaterEqual(child.wait_calls,1)

    def test_stage_launch_receipt_failure_cleans_owned_child_and_preserves_exception(self):
        launch,calls,child=self.popen(pending=True);failure=OSError('CPU fake receipt disk failure');original=m._write
        def write(root,name,value,**kwargs):
            if name==m.RUNTIME+'/stages/'+m.STAGE+'/launch.json':raise failure
            return original(root,name,value,**kwargs)
        with patch.object(m,'_popen',side_effect=launch),patch.object(m,'_write',side_effect=write):
            with self.assertRaises(OSError) as caught:self.execute()
        self.assertIs(caught.exception,failure);self.assertTrue(child.terminated);self.assertGreaterEqual(child.wait_calls,1)
        self.assertTrue(self.fixture.json(m.RUNTIME+'/stages/'+m.STAGE+'/terminal.json')['owned_analysis_child_reaped'])

    def test_failure_receipt_error_does_not_mask_original_after_child_is_reaped(self):
        launch,calls,child=self.popen(pending=True);failure=OSError('CPU fake no proc');original=m._write
        def write(root,name,value,**kwargs):
            if name==m.RUNTIME+'/stages/'+m.STAGE+'/terminal.json':raise ValueError('CPU fake secondary receipt failure')
            return original(root,name,value,**kwargs)
        with patch.object(m,'_popen',side_effect=launch),patch.object(m,'ticks',side_effect=lambda pid:(_ for _ in ()).throw(failure) if pid==child.pid else pid+1), \
                patch.object(m,'_write',side_effect=write):
            with self.assertRaises(OSError) as caught:self.execute()
        self.assertIs(caught.exception,failure);self.assertTrue(child.terminated);self.assertGreaterEqual(child.wait_calls,1)

    def test_pid_reuse_never_signals_an_unowned_child(self):
        child=Child(None)
        with patch.object(m,'ticks',return_value=99999),self.assertRaisesRegex(ValueError,'changed child PID/start'):
            m._stop_child(child,991)
        self.assertFalse(child.terminated);self.assertFalse(child.killed)

    def test_failed_or_stopped1043_raises_without_reading_result(self):
        for value,kind in ((dict(complete=False,user_stop=False),RuntimeError),(dict(complete=False,user_stop=True),InterruptedError)):
            self.fixture.write(m.CONTROLLER+'/failure.json',value)
            control,binding=self.fixture.load();original=m.read_bytes;seen=[]
            def read(root,name,*args,**kwargs):seen.append(name);return original(root,name,*args,**kwargs)
            with patch.object(m,'read_bytes',side_effect=read),self.assertRaises(kind):m.completion_snapshot(self.root,control,binding)
            self.assertNotIn(m.RESULT,seen)

    def test_actual_lower1031_capacity_is_changed_and_unselected_fallback_rejected(self):
        for fallback,quota in ((True,None),(False,dict(schema='r10_frozen_concurrent_native_resource_profile_v1031',max_native_slots=8,max_GPU_slots=3,max_CPU_preflight_slots=4,ready_limit=16))):
            with self.subTest(fallback=fallback),tempfile.TemporaryDirectory() as d:
                fixture=Fixture(Path(d),fallback=fallback,actual_quota=quota)
                if fallback:
                    with self.assertRaises(ValueError):fixture.completion()
                    continue
                control,binding,completion=fixture.completion()
                self.assertEqual(sum(completion['provenance_runtime_counts'].values()),1920)
                self.assertIs(binding['original_control']['resource_profile_changed'],True)
                self.assertIs(binding['previous_control']['resource_profile_changed'],False)
                self.assertIs(binding['original_control']['resource_profile_within_capacity'],True)
                if not fallback:self.assertNotEqual(binding['original_control']['resource_profile'],binding['original_control']['resource_capacity_envelope'])

    def test_grandparent_control_requires_both_current_and_small_lock_pins(self):
        for missing in ('small','current'):
            with self.subTest(missing=missing),tempfile.TemporaryDirectory() as d:
                fixture=Fixture(Path(d))
                if missing=='small':fixture.lock['files'].pop(m.GRANDPARENT_CONTROL);fixture.write(m.LOCK,fixture.lock)
                else:
                    original=fixture.json(m.RESUME_LOCK);original['files'].pop(m.GRANDPARENT_CONTROL);fixture.write(m.RESUME_LOCK,original)
                    fixture.control['resume_control_lock']=fixture.pointer(m.RESUME_LOCK)
                    fixture.lock['files'][m.RESUME_LOCK]=fixture.pointer(m.RESUME_LOCK)['sha256'];fixture.reseal()
                with self.assertRaises(ValueError):fixture.load()

    def test_capacity_mislabel_schema_quota_and_fake_fallback_refused(self):
        original=self.fixture.json(m.RESUME_CONTROL)
        for mode in ('envelope','schema','quota','bool','source','fallback','changed'):
            value=deepcopy(original)
            if mode=='envelope':value['resource_capacity_envelope']['max_GPU_slots']=4
            elif mode=='schema':value['resource_profile']['schema']='r10_frozen_concurrent_native_resource_profile_v1005'
            elif mode=='quota':value['resource_profile']['max_native_slots']=13
            elif mode=='bool':value['resource_profile']['max_GPU_slots']=True
            elif mode=='source':value['resource_capacity_source']='qualified1005_fallback'
            elif mode=='fallback':
                value.update(resource_profile=dict(schema='r10_frozen_concurrent_native_resource_profile_v1005',max_native_slots=7,max_GPU_slots=3,max_CPU_preflight_slots=4,ready_limit=16),
                    resource_capacity_envelope=dict(schema='r10_frozen_concurrent_native_resource_profile_v1005',max_native_slots=8,max_GPU_slots=3,max_CPU_preflight_slots=4,ready_limit=16),resource_capacity_source='qualified1005_fallback')
            else:value['resource_profile_changed']=False
            with self.subTest(mode=mode),self.assertRaises(ValueError):
                if mode=='changed':
                    self.fixture.write(m.RESUME_CONTROL,value)
                    locked=self.fixture.json(m.RESUME_LOCK);locked['files'][m.RESUME_CONTROL]=self.fixture.pointer(m.RESUME_CONTROL)['sha256'];self.fixture.write(m.RESUME_LOCK,locked)
                    self.fixture.control['resume_control']=self.fixture.pointer(m.RESUME_CONTROL);self.fixture.control['resume_control_lock']=self.fixture.pointer(m.RESUME_LOCK)
                    self.fixture.lock['files'][m.RESUME_CONTROL]=self.fixture.pointer(m.RESUME_CONTROL)['sha256'];self.fixture.lock['files'][m.RESUME_LOCK]=self.fixture.pointer(m.RESUME_LOCK)['sha256'];self.fixture.reseal();self.fixture.load()
                else:m._recorded_resources(value)


    def test_only_private_and_global_STOP_are_owned_by_this_waiter(self):
        self.assertEqual(m.STOP_PATHS,('runtime/framework/multitask_r10_STOP',m.RUNTIME+'/STOP'))
        old=self.root/m.HISTORICAL_FIFTH/'STOP';old.parent.mkdir(parents=True);old.touch()
        m.guard(self.root)
        for name in m.STOP_PATHS:
            p=self.root/name;p.parent.mkdir(parents=True,exist_ok=True);p.touch()
            with self.assertRaises(InterruptedError):m.guard(self.root)
            p.unlink()

    def test_both_interrupted_owner_terminal_and_science_payloads_are_never_readable(self):
        for name in (m.HISTORICAL_FIFTH+'/terminal.json',m.HISTORICAL_FIFTH+'/failure.json',
                m.HISTORICAL_FIFTH+'/DRAIN',m.HISTORICAL_SIXTH+'/terminal.json',m.HISTORICAL_SIXTH+'/failure.json',
                m.HISTORICAL_SIXTH+'/DRAIN','models/unopened.pt','data/unopened.hdf5',
                'artifacts/framework/multitask_r10/case.json'):
            with self.subTest(name=name),self.assertRaises(ValueError):m.read_bytes(self.root,name,missing=True)

    def test_seventh_owner_partition_rejects_fake_old_terminal_and_reboot_native_failure(self):
        control,binding,complete=self.fixture.completion()
        for mode in ('old_terminal','unknown_as_false','damaged_dropped','missing_proof'):
            result=self.fixture.json(m.RESULT)
            if mode=='old_terminal':result['actual_stage_owners'][4]['terminal']={'path':'fake.json','sha256':'a'*64}
            elif mode=='unknown_as_false':result['host_reboot_interruption']['unknown_counted_as_native_task_failures']=True
            elif mode=='damaged_dropped':result['prior_host_reboot_interruption']['rejected_saved_completed_job_ids']=[]
            else:result.pop('host_reboot_interruption')
            with self.subTest(mode=mode),self.assertRaises(ValueError):
                m.require_seven_tier_evidence(result,binding['original_control'],self.fixture.json(m.OWNER_RUNTIME+'/status.json'),
                    self.fixture.pointer(m.OWNER_RUNTIME+'/launch.json'),self.fixture.pointer(m.OWNER_RUNTIME+'/terminal.json'),
                    binding['previous_control'],binding['parent_control'],binding['grandparent_control'],
                    binding['greatgrandparent_control'],binding['prior_reboot_audit'],binding['reboot_audit'])

    def test_same_boot_and_exact_reboot_audit_required_by_all_new_owner_metadata(self):
        control,binding=self.fixture.load()
        for name in ('launch.json','status.json','terminal.json'):
            path=m.OWNER_RUNTIME+'/'+name;before=self.fixture.json(path)
            for key,value in (('boot_id','another-boot'),('reboot_audit',{}),('infrastructure_recovery',False)):
                bad=deepcopy(before);bad[key]=value;self.fixture.write(path,bad)
                with self.subTest(name=name,key=key),self.assertRaises(ValueError):m.completion_snapshot(self.root,control,binding)
                self.fixture.write(path,before)

    def test_current_machine_boot_must_match1067_and1068_before_activation(self):
        with patch.object(m,'current_boot_id',return_value='different-actual-boot'),self.assertRaises(ValueError):
            self.fixture.load()
        m._popen.assert_not_called()

    def test_both_reboot_evidence_pairs_and_greatgrandparent_are_in_small_map(self):
        for name in (m.FIRST_REBOOT_AUDIT,m.FIRST_HARDWARE_OBSERVATION,m.REBOOT_AUDIT,
                m.HARDWARE_OBSERVATION,m.GREATGRANDPARENT_CONTROL):
            with self.subTest(name=name),tempfile.TemporaryDirectory() as d:
                fixture=Fixture(Path(d));fixture.lock['files'].pop(name);fixture.write(m.LOCK,fixture.lock)
                with self.assertRaises(ValueError):fixture.load()

    def test_first509_is_split_eight_plus501_without_claiming_first_owner_complete(self):
        control,binding,complete=self.fixture.completion()
        for mode in ('first_complete','single_namespace','first_count','second_count','old1041_terminal','old1056_terminal'):
            result=self.fixture.json(m.RESULT)
            if mode=='first_complete':result['prior_host_reboot_interruption']['first_recovery_owner_completed']=True
            elif mode=='single_namespace':result['prior_host_reboot_interruption']['all_remaining_keys_executed_once_in_new_namespace']=True
            elif mode=='first_count':result['prior_host_reboot_interruption']['first_recovery_clean_completions']=509
            elif mode=='second_count':result['prior_host_reboot_interruption']['second_recovery_clean_completions']=0
            else:result['actual_stage_owners'][4 if mode=='old1041_terminal' else 5]['terminal']={'path':'invented.json','sha256':'a'*64}
            with self.subTest(mode=mode),self.assertRaises(ValueError):
                m.require_seven_tier_evidence(result,binding['original_control'],self.fixture.json(m.OWNER_RUNTIME+'/status.json'),
                    self.fixture.pointer(m.OWNER_RUNTIME+'/launch.json'),self.fixture.pointer(m.OWNER_RUNTIME+'/terminal.json'),
                    binding['previous_control'],binding['parent_control'],binding['grandparent_control'],
                    binding['greatgrandparent_control'],binding['prior_reboot_audit'],binding['reboot_audit'])

    def test_summary_must_retain_both_reboots_and_two_historical_damaged_cases(self):
        control,binding,complete=self.fixture.completion()
        summary=self.summary(control,binding,complete)
        for key,value in (('historical_damaged_saved_completed_cases_reexecuted',0),
                ('prior_reboot_audit',control['reboot_audit']),('prior_host_reboot_interruption',{}),
                ('host_reboot_interruptions',[])):
            bad=deepcopy(summary);bad[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):
                m.validate_summary(bad,{'path':m.SUMMARY_OUTPUT,'sha256':'a'*64},
                    {'complete':True,'output':m.SUMMARY_OUTPUT,'sha256':'a'*64,'case_count':1920},
                    control,binding,complete,self.root)

    def test_private_metadata_syncs_temporary_before_replace_and_all_ancestors(self):
        leaf=self.root/m.RUNTIME/'nested';leaf.mkdir(parents=True)
        name=m.RUNTIME+'/nested/status.json';events=[];replace=m.os.replace;sync=m.os.fsync
        def tracked_sync(fd):events.append('sync');return sync(fd)
        def tracked_replace(a,b):events.append('replace');return replace(a,b)
        with patch.object(m.os,'fsync',side_effect=tracked_sync),patch.object(m.os,'replace',side_effect=tracked_replace):
            m._write(self.root,name,{'completed':1})
        self.assertEqual(events[:2],['sync','replace'])
        self.assertGreater(len(events),len((leaf.relative_to(self.root)).parts))
        with patch.object(m.os,'fsync',side_effect=OSError('simulated sync failure')),self.assertRaises(OSError):
            m._write(self.root,name,{'completed':2},overwrite=True)
        self.assertEqual(self.fixture.json(name),{'completed':1})


if __name__=='__main__':unittest.main()
