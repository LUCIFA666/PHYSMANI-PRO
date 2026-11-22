"""CPU-only injected metadata/resources; never real TEST, /proc or kernel locks."""
from contextlib import contextmanager
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from types import ModuleType
import unittest
from unittest.mock import Mock, patch

from our_mani.multitask import post_TEST_resource_guard_v924 as guard


ROOT = Path('/synthetic/our-mani-v924')
SOURCE = Path(__file__).resolve().parents[1] / 'our_mani/multitask/post_TEST_resource_guard_v924.py'


def encode(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()


def environment(number):
    return dict(schema='r10_actual_frozen_OS_namespace_receipt_v820', complete=True,
        checked_OS_files=6418, all_current_OS_targets_full_byte_SHA_verified=True,
        host_PID_network_namespace_preserved=True, proc_or_dev_replaced=False,
        task_parameters_or_strategy_changed=False, observer_pid=number, pid=number,
        observer_start_ticks=number+1, current_mount_namespace='CPU-private-mount',
        mount_namespace='CPU-private-mount', reference_mount_namespace='CPU-host-mount',
        **deepcopy(guard.ENVIRONMENT_POINTERS))


class Fixture:
    """Real907/902/906 metadata shapes with dynamic134 or7 imported cases.

    Only small completion objects and source byte stubs exist. Case/driver/901
    sidecar, suite, source manifest, source-lock, raw and model bytes do not.
    Sources are independently hashed fake strings, patched only in this test.
    """
    def __init__(self, count=134):
        self.data = {name: ('CPU fake pinned source924: '+name).encode() for name in guard.SOURCE_SHA}
        self.source_sha = {name: guard.sha(raw) for name, raw in self.data.items()}
        self.reads = []
        self.count = count
        self.sources = {}
        source_pointer = lambda name: dict(path=name, sha256=self.source_sha[name])
        self.batch_control = dict(path=guard.BATCH_CONTROL, sha256='1'*64)
        self.batch_lock = dict(path=guard.BATCH_LOCK, sha256='2'*64)
        entry = dict(kind='batched901', pointer=source_pointer(guard.BATCH_SOURCE),
            batch_control=self.batch_control, batch_lock=self.batch_lock)
        shared = {
            guard.BATCH_SOURCE: source_pointer(guard.BATCH_SOURCE),
            guard.LOOKUP_SOURCE: source_pointer(guard.LOOKUP_SOURCE),
            guard.DRIVER['path']: deepcopy(guard.DRIVER),
            guard.GUARD['path']: deepcopy(guard.GUARD),
            'our_mani/multitask/rml_case_preflight_v742.py': dict(path='our_mani/multitask/rml_case_preflight_v742.py',sha256='3'*64),
            guard.WORKER['path']: deepcopy(guard.WORKER),
            guard.PROTOCOL['runtime_consumer']['path']: deepcopy(guard.PROTOCOL['runtime_consumer']),
        }
        implementation = dict(schema='r10_resource_execution_implementation_v902', kind='batched901',
            entrypoint=entry['pointer'], lookup_batch=shared[guard.LOOKUP_SOURCE],
            original_formal_driver=deepcopy(guard.DRIVER), original_case_worker=deepcopy(guard.WORKER),
            original_native_runtime=deepcopy(guard.PROTOCOL['runtime_consumer']), sources=shared)
        self.old_rows = []
        old_records = {}
        for task in guard.TASKS:
            for index in range(40):
                raw='CPU_fake_'+str(index)
                for method in guard.METHODS:
                    out=guard.OLD_ART+'/test/'+task+'/'+raw+'/'+method
                    argv=['bash','scripts/run_aligned_gpu.sh' if method=='physmani' else 'scripts/run_local_sim.sh',guard.DRIVER['path'],
                        '--split','test','--task',task,'--raw-episode-id',raw,'--method',method,
                        '--suite',guard.SUITE,'--suite-lock',guard.SUITE_LOCK,'--source-inputs',guard.SOURCE_LOCK,
                        '--manifest',guard.MANIFEST,'--validation-release',guard.VAL_RELEASE,'--out',out]
                    self.old_rows.append(dict(task=task,raw=raw,method=method,job_id='_'.join((task,raw,method)),
                        source_row_sha256=guard.sha((task+'/'+raw).encode()),worker_sha256=guard.DRIVER['sha256'],argv=argv,
                        admission_driver=deepcopy(guard.DRIVER),admission_guard=deepcopy(guard.GUARD),case_worker=deepcopy(guard.WORKER),
                        native_runtime=deepcopy(guard.PROTOCOL['runtime_consumer']),original_native_runtime=deepcopy(guard.PROTOCOL['original_runtime']),
                        author_outcome_protocol=deepcopy(guard.PROTOCOL)))
        done=sorted(r['job_id'] for r in self.old_rows[:count])
        done_set=set(done)
        common=dict(split='test',policy_sha256=guard.POLICY,author_outcome_protocol=deepcopy(guard.PROTOCOL),
            execution_protocol_changed=True,automatic_retry=False,admission_driver=deepcopy(guard.DRIVER),
            admission_guard=deepcopy(guard.GUARD),native_runtime=deepcopy(guard.PROTOCOL['runtime_consumer']),
            original_native_runtime=deepcopy(guard.PROTOCOL['original_runtime']))
        self.data[guard.OLD_RUNTIME+'/launch.json']=dict(common,pid=510,start_ticks=511,
            source_sha256=self.source_sha[guard.OLD_OWNER_SOURCE],max_native_slots=2,max_GPU_slots=1,max_CPU_preflight_slots=2)
        # Original753 plan really lacks author_outcome_protocol; don't invent it.
        self.data[guard.OLD_RUNTIME+'/plan.json']=dict(split='test',expected_cases=1920,policy_sha256=guard.POLICY,
            automatic_retry=False,source_outcome_filtering=False,jobs=self.old_rows)
        for index,row in enumerate(self.old_rows[:count]):
            out=guard.option(row['argv'],'--out')
            old_records[row['job_id']]=self.completed(out)
            if index==0:old_records[row['job_id']]['motion_pass']=None
        self.data[guard.OLD_RUNTIME+'/status.json']=dict(split='test',expected=1920,completed=old_records,
            failed={},active=[],checking=[],drain_requested=True,stop_requested=False,automatic_retry=False,
            native_slots=0,GPU_slots=0)
        self.data[guard.OLD_RUNTIME+'/terminal.json']=dict(complete=False,drained=True,drain_requested=True,
            stop_requested=False,expected=1920,completed=count,failed=0,automatic_retry=False,policy_sha256=guard.POLICY)
        self.data[guard.OLD_RUNTIME+'/DRAIN']=dict(reason='CPU fake planned natural drain')
        self.data[guard.OLD_CONTROLLER+'/launch.json']=dict(pid=410,start_ticks=411,
            source_sha256=self.source_sha[guard.OLD_CONTROLLER_SOURCE])
        self.data[guard.OLD_CONTROLLER+'/status.json']=dict(stage='requires_concrete_repair',automatic_retry=False)
        self.data[guard.OLD_CONTROLLER+'/failure.json']=dict(complete=False,user_stop=False,automatic_retry=False)
        self.data[guard.VAL_OWNER_STAGE+'/launch.json']=dict(pid=450,start_ticks=451,
            source_sha256=self.source_sha[guard.VAL_OWNER_SOURCE],
            argv=[str(ROOT/'.conda-env/bin/python'),'-u',guard.VAL_OWNER_SOURCE,'--split','validation'])
        self.data[guard.QUALIFICATION_LAUNCH]=dict(pid=950,start_ticks=951,source_sha256=self.source_sha[guard.QUALIFIER_SOURCE])
        self.data[guard.QUALIFICATION]=dict(schema='r10_concurrent_TRAIN_resource_qualification_v902',complete=True,
            split='train',case_count=6,infrastructure_failed=0,resource_profile=deepcopy(guard.PROFILE_LIMITS),
            execution_entry=implementation,policy_sha256=guard.POLICY,author_outcome_protocol=deepcopy(guard.PROTOCOL),
            source_sha256=self.source_sha[guard.QUALIFIER_SOURCE],cleanup_completed=True,resource_bounds_verified=True,
            overlap_verified=True,independent_audit_complete=True,observed_native_peak=4,observed_GPU_peak=2,
            formal_score_contribution=False,TEST_read=False,actual_independent_audit=self.pointer(guard.QUALIFICATION_LAUNCH))
        control=dict(schema='r10_concurrent_resumed_formal_TEST_control_v902',split='test',expected_cases=1920,
            source_count=640,policy_sha256=guard.POLICY,author_outcome_protocol=deepcopy(guard.PROTOCOL),
            resource_profile=deepcopy(guard.PROFILE_LIMITS),user_authorized_resume=True,standing_authorization_reused=True,
            new_user_event_invented=False,physical_policy_changed=False,execution_entry=entry,execution_implementation=implementation,
            completed_job_ids=done,remaining_job_ids=sorted(r['job_id'] for r in self.old_rows if r['job_id'] not in done_set),
            completed_key_sha256=guard.digest(done),remaining_key_sha256=guard.digest(sorted(r['job_id'] for r in self.old_rows if r['job_id'] not in done_set)),
            resume_controller=source_pointer(guard.CONTROLLER_SOURCE),resume_owner=source_pointer(guard.OWNER_SOURCE),
            resume_auditor=source_pointer(guard.AUDITOR_SOURCE),resource_qualifier=source_pointer(guard.QUALIFIER_SOURCE),
            resource_qualification=self.pointer(guard.QUALIFICATION),
            suite=dict(path=guard.SUITE,sha256='4'*64),suite_lock=dict(path=guard.SUITE_LOCK,sha256='5'*64),
            source_inputs=dict(path=guard.SOURCE_LOCK,sha256='6'*64),validation_release=dict(path=guard.VAL_RELEASE,sha256='7'*64),
            environment_wrapper=deepcopy(guard.ENVIRONMENT_POINTERS['wrapper']),
            environment_manifest=deepcopy(guard.ENVIRONMENT_POINTERS['manifest']),
            environment_namespace_proof=deepcopy(guard.ENVIRONMENT_POINTERS['namespace_qualification']))
        for key,name in (('previous_launch','launch.json'),('previous_plan','plan.json'),('previous_status','status.json'),
                         ('previous_terminal','terminal.json'),('previous_DRAIN','DRAIN')):
            control[key]=self.pointer(guard.OLD_RUNTIME+'/'+name)
        for key,name in (('previous_controller_launch','launch.json'),('previous_controller_status','status.json'),('previous_controller_failure','failure.json')):
            control[key]=self.pointer(guard.OLD_CONTROLLER+'/'+name)
        self.data[guard.CONTROL]=control
        files={name:value for name,value in self.source_sha.items()}
        files.update({p['path']:p['sha256'] for p in shared.values()})
        files.update({p['path']:p['sha256'] for p in control.values() if isinstance(p,dict) and set(p)=={'path','sha256'}})
        files[guard.CONTROL]=self.pointer(guard.CONTROL)['sha256']
        files[guard.BATCH_CONTROL]=self.batch_control['sha256'];files[guard.BATCH_LOCK]=self.batch_lock['sha256']
        self.data[guard.CONTROL_LOCK]=dict(schema='r10_concurrent_resumed_formal_TEST_inputs_v902',
            approved_for_execution=True,control=self.pointer(guard.CONTROL),files=files)
        cp,lp=self.pointer(guard.CONTROL),self.pointer(guard.CONTROL_LOCK)
        controller=dict(pid=710,start_ticks=711,source_sha256=self.source_sha[guard.CONTROLLER_SOURCE],
            resume_control=cp,resume_control_lock=lp,automatic_retry=False,frozen_environment=environment(710))
        self.data[guard.CONTROLLER+'/launch.json']=controller
        self.data[guard.CONTROLLER+'/status.json']=dict(stage='complete',automatic_retry=False,
            frozen_environment=controller['frozen_environment'])
        owner_args=[]
        for key,flag in (('suite','--suite'),('suite_lock','--suite-lock'),('source_inputs','--source-inputs'),('validation_release','--validation-release')):
            owner_args.extend((flag,control[key]['path']))
        owner_args+=['--runtime',guard.TEST_RUNTIME,'--out',guard.ART+'/test','--resume-control',guard.CONTROL,
            '--resume-lock',guard.CONTROL_LOCK,'--resume-lock-sha256',lp['sha256']]
        audit_args=['--resume-control',guard.CONTROL,'--resume-lock',guard.CONTROL_LOCK,'--resume-lock-sha256',lp['sha256']]
        for stage,name,number,args in ((guard.OWNER_STAGE,guard.OWNER_SOURCE,810,owner_args),(guard.AUDIT_STAGE,guard.AUDITOR_SOURCE,910,audit_args)):
            self.data[stage+'/launch.json']=dict(pid=number,start_ticks=number+1,source_sha256=self.source_sha[name],
                argv=[str(ROOT/'.conda-env/bin/python'),'-B','-u',name,*args],automatic_retry=False,
                inherited_frozen_environment=controller['frozen_environment'],current_mount_namespace='CPU-private-mount')
            self.data[stage+'/terminal.json']=dict(complete=True,returncode=0,automatic_retry=False)
        new_common=dict(common,resume_control=cp,resume_control_lock=lp,resource_profile=deepcopy(guard.PROFILE_LIMITS),
            resource_qualification=control['resource_qualification'],execution_implementation=implementation,frozen_environment=environment(810))
        launch=dict(new_common,pid=810,start_ticks=811,source_sha256=self.source_sha[guard.OWNER_SOURCE],
            argv=self.data[guard.OWNER_STAGE+'/launch.json']['argv'][3:],execution_entry=entry,
            base_stage_owner=source_pointer(guard.OLD_OWNER_SOURCE),imported_completed_cases=count,
            **{k:guard.PROFILE_LIMITS[k] for k in ('max_native_slots','max_GPU_slots','max_CPU_preflight_slots')})
        self.data[guard.TEST_RUNTIME+'/launch.json']=launch
        new_rows=[];records={};cases=[];drivers=[];tiers=[];sidecars=[]
        for index,previous in enumerate(self.old_rows):
            row=deepcopy(previous);adopt=row['job_id'] in done_set
            row.update(evidence_runtime=guard.OLD_RUNTIME if adopt else guard.TEST_RUNTIME,imported_completed=adopt,
                previous_row_sha256=guard.digest(previous),previous_worker_sha256=previous['worker_sha256'])
            if adopt:
                row.update(physical_argv=deepcopy(previous['argv']),execution_entry=dict(kind='original751',pointer=deepcopy(guard.DRIVER)))
                record=deepcopy(old_records[row['job_id']])
            else:
                inner=deepcopy(previous['argv']);out=guard.ART+'/test/'+row['task']+'/'+row['raw']+'/'+row['method']
                inner[inner.index('--out')+1]=out
                row.update(physical_argv=inner,argv=[*inner[:2],guard.BATCH_SOURCE,'--batch-control',guard.BATCH_CONTROL,
                    '--batch-lock',guard.BATCH_LOCK,'--batch-lock-sha256',self.batch_lock['sha256'],'--entry','formal','--',*inner[3:]],
                    worker_sha256=entry['pointer']['sha256'],execution_entry=deepcopy(entry))
                record=self.completed(out);record['batch_driver_receipt']=self.unopened(out+'/batch_driver_receipt.json')
                if index==count:record['motion_pass']=None
                sidecars.append(record['batch_driver_receipt'])
            cases.append(record['case']);drivers.append(record['admission_driver_receipt']);records[row['job_id']]=record
            owner= self.data[guard.OLD_RUNTIME+'/launch.json'] if adopt else launch
            tiers.append(dict(job_id=row['job_id'],evidence_runtime=row['evidence_runtime'],imported_completed=adopt,
                actual_stage_owner_pid=owner['pid'],actual_stage_owner_start_ticks=owner['start_ticks'],
                previous_row_sha256=row['previous_row_sha256'],execution_entry=row['execution_entry'],
                case=record['case'],admission_driver_receipt=record['admission_driver_receipt']))
            new_rows.append(row)
        self.data[guard.TEST_RUNTIME+'/plan.json']=dict(new_common,expected_cases=1920,source_outcome_filtering=False,
            imported_completed_cases=count,jobs=new_rows)
        self.data[guard.TEST_RUNTIME+'/status.json']=dict(new_common,expected=1920,completed=records,failed={},active=[],checking=[],ready_jobs=[],
            CPU_ready=0,native_slots=0,GPU_slots=0,owner_error=None,stop_requested=False,drain_requested=False,
            imported_completed=count,CPU_launched=1920-count)
        self.data[guard.TEST_RUNTIME+'/terminal.json']=dict(new_common,complete=True,expected=1920,completed=1920,failed=0,
            owner_error=None,stop_requested=False,drain_requested=False,drained=False,CPU_ready_unexecuted=[],CPU_not_launched=0,
            formal_release_requires_independent_audit=True,imported_completed_cases=count,resumed_new_cases=1920-count,native_successes=0)
        result=dict(schema='r10_composite_concurrent_resumed_formal_TEST_results_v906',complete=True,split='test',source_cases=640,
            method_cases=1920,completed_method_cases=1920,methods=list(guard.METHODS),tasks=list(guard.TASKS),policy_sha256=guard.POLICY,
            common_RML_profile_sha256=guard.PROFILE,source_sha256=self.source_sha[guard.AUDITOR_SOURCE],resume_control=cp,resume_control_lock=lp,
            source_input_lock=control['source_inputs'],validation_release=control['validation_release'],resource_profile=deepcopy(guard.PROFILE_LIMITS),
            resource_qualification=control['resource_qualification'],execution_implementation=implementation,
            admission_driver=deepcopy(guard.DRIVER),admission_guard=deepcopy(guard.GUARD),native_runtime=deepcopy(guard.PROTOCOL['runtime_consumer']),
            original_native_runtime=deepcopy(guard.PROTOCOL['original_runtime']),author_outcome_protocol=deepcopy(guard.PROTOCOL),
            base_result_auditor=source_pointer(guard.BASE_AUDITOR_SOURCE),TEST_used_for_fitting=False,motion_is_diagnostic_only=True,
            motion_metrics_required_for_all_methods=True,native_task_success_unchanged=True,native_success_is_not_a_release_threshold=True,
            source_outcome_filtering=False,native_failures_retained=True,Value_NGFF_FASTER_used=False,execution_protocol_changed=True,
            used_evidence_rechecked_at_close=True,old753_results_not_rerun=True,independent_composite_audit=True,
            original754_single_owner_result_claimed=False,automatic_retry=False,cases=cases,admission_driver_receipts=drivers,
            batch_driver_receipts=sidecars,evidence_tiers=tiers,native_successes_by_method={k:0 for k in guard.METHODS},
            auditor_invocation=dict(pid=910,start_ticks=911,argv=self.data[guard.AUDIT_STAGE+'/launch.json']['argv'][3:],
                source=source_pointer(guard.AUDITOR_SOURCE),inherited_physical_stage_identity_claimed=False),frozen_environment=environment(910),
            actual_stage_owners=[dict(runtime=guard.OLD_RUNTIME,launch=control['previous_launch'],terminal=control['previous_terminal'],
                pid=510,start_ticks=511,completed=count,resource_limits=dict(max_native_slots=2,max_GPU_slots=1,max_CPU_preflight_slots=2)),
                dict(runtime=guard.TEST_RUNTIME,launch=self.pointer(guard.TEST_RUNTIME+'/launch.json'),terminal=self.pointer(guard.TEST_RUNTIME+'/terminal.json'),
                    pid=810,start_ticks=811,completed=1920-count,resource_profile=deepcopy(guard.PROFILE_LIMITS))])
        required=[guard.CONTROL,guard.CONTROL_LOCK,guard.OWNER_SOURCE,guard.AUDITOR_SOURCE,guard.OLD_OWNER_SOURCE,
            guard.BASE_AUDITOR_SOURCE,guard.QUALIFIER_SOURCE,guard.QUALIFICATION,
            *[guard.TEST_RUNTIME+'/'+n for n in ('launch.json','plan.json','status.json','terminal.json')],
            *[guard.OLD_RUNTIME+'/'+n for n in ('launch.json','plan.json','status.json','terminal.json','DRAIN')],
            *[guard.OLD_CONTROLLER+'/'+n for n in ('launch.json','status.json','failure.json')]]
        result['inputs_sha256']={str(ROOT/name):self.pointer(name)['sha256'] for name in required}
        result['inputs_sha256'][str(ROOT/guard.SOURCE_LOCK)]=control['source_inputs']['sha256']
        self.data[guard.RESULT]=result
        self.data[guard.CONTROLLER+'/terminal.json']=dict(complete=True,automatic_retry=False,TEST_results=self.pointer(guard.RESULT))
        self.data[guard.AUDIT_STAGE+'/stdout.log']=dict(complete=True,schema=result['schema'],split='test',cases=1920,
            out=guard.RESULT,motion_is_diagnostic_only=True,native_successes_by_method={k:0 for k in guard.METHODS})

    @staticmethod
    def unopened(name):
        return dict(path=name,sha256=guard.sha(('unopened CPU fake pointer '+name).encode()))

    def completed(self,out):
        return dict(complete=True,returncode=0,timed_out=False,interrupted=False,automatic_retry=False,
            native_success=False,motion_pass=False,case=self.unopened(out+'/case.json'),
            admission_driver_receipt=self.unopened(out+'/admission_driver_receipt.json'))

    def raw(self,name):
        value=self.data[name]
        return value if isinstance(value,bytes) else encode(value)

    def pointer(self,name):
        return dict(path=name,sha256=guard.sha(self.raw(name)))

    def read(self,name):
        self.reads.append(name)
        if name not in self.data:
            raise AssertionError('forbidden/unexpected non-metadata read: '+name)
        return self.raw(name)

    def refresh_result(self):
        self.data[guard.CONTROLLER+'/terminal.json']['TEST_results']=self.pointer(guard.RESULT)

    def proof(self):
        with patch.object(guard,'SOURCE_SHA',self.source_sha):
            return guard.require_complete_TEST_metadata(ROOT,self.pointer(guard.RESULT)['sha256'],
                resume_lock_sha256=self.pointer(guard.CONTROL_LOCK)['sha256'],read_bytes=self.read)


class Resources:
    def __init__(self):
        self.events=[];self.stop_calls=0;self.stop_at=None;self.busy=False;self.live=False;self.held=False
        self.visibility=dict(host_visibility_available=True,ancestor_chain_complete=True,
            same_uid_inventory_incomplete=False,live_project_native_processes=[])
    def stop(self,root):
        self.events.append('stop');self.stop_calls+=1
        if self.stop_calls==self.stop_at:raise InterruptedError('CPU fake STOP')
    @contextmanager
    def reserve(self,root):
        self.events.append(('reserve_enter',tuple(guard.LOCKS)))
        if self.busy:raise RuntimeError('CPU fake busy lock')
        self.held=True
        try:yield
        finally:self.held=False;self.events.append('reserve_exit')
    def inventory(self,root):
        self.events.append('inventory');return deepcopy(self.visibility)
    def is_alive(self,record):
        self.events.append(('is_alive',record['pid'],record['start_ticks']));return self.live
    def kwargs(self,fixture):
        return dict(resume_lock_sha256=fixture.pointer(guard.CONTROL_LOCK)['sha256'],enabled=True,
            read_bytes=fixture.read,stop_check=self.stop,reserve=self.reserve,inventory=self.inventory,is_alive=self.is_alive)


class GuardTests(unittest.TestCase):
    def setUp(self):
        denied=Mock(side_effect=AssertionError('CPU test reached a production OS helper'))
        for name in ('_stop_guard','_reserve','_inventory','_is_alive'):
            scope=patch.object(guard,name,denied);scope.start();self.addCleanup(scope.stop)
        scope=patch.object(guard.os,'getpid',return_value=4242);scope.start();self.addCleanup(scope.stop)

    def test_import_and_disabled_context_construction_are_inert(self):
        raw=SOURCE.read_bytes();module=ModuleType('CPU_guard924_inert')
        denied=Mock(side_effect=AssertionError('inert import performed I/O'))
        with patch.object(Path,'read_bytes',denied),patch.object(Path,'read_text',denied),patch.object(Path,'exists',denied), \
                patch.object(guard.os,'readlink',denied):
            exec(compile(raw,str(SOURCE),'exec'),module.__dict__)
            context=module.reserved_resources_after_TEST(ROOT,'a'*64,resume_lock_sha256='b'*64)
            with self.assertRaisesRegex(ValueError,'default remains inert'):
                with context: self.fail('disabled scope entered')
        self.assertEqual(len(module._LEASES),0)

    def test_complete_dynamic134_and7_all_false_metadata_only(self):
        for count in (134,7):
            fixture=Fixture(count);proof=fixture.proof()
            with self.subTest(count=count):
                self.assertTrue(proof['complete']);self.assertEqual(proof['imported_completed'],count)
                self.assertEqual(proof['resumed_completed'],1920-count)
                self.assertFalse(proof['success_rate_threshold_used']);self.assertFalse(proof['motion6_used_for_gate'])
                self.assertFalse(proof['new_release_granted']);self.assertFalse(proof['source_case_bytes_read'])
                self.assertFalse(proof['source_manifest_or_raw_bytes_read'])
                self.assertNotIn('author_outcome_protocol',fixture.data[guard.OLD_RUNTIME+'/plan.json'])
                self.assertFalse(any(p.endswith('/case.json') or p in (guard.MANIFEST,guard.SOURCE_LOCK,guard.SUITE) for p in fixture.reads))
                self.assertNotIn(guard.OLD_CONTROLLER+'/terminal.json',fixture.reads)
                rows=fixture.data[guard.TEST_RUNTIME+'/plan.json']['jobs']
                records=fixture.data[guard.TEST_RUNTIME+'/status.json']['completed']
                self.assertIsNone(records[rows[0]['job_id']]['motion_pass'])
                self.assertIsNone(records[rows[count]['job_id']]['motion_pass'])
                self.assertEqual(len({(p['pid'],p['start_ticks']) for p in proof['known_processes']}),7)

    def test_incomplete_controller_stage_and_owner_never_read_result(self):
        for path,key,value in ((guard.CONTROLLER+'/status.json','stage','independent_TEST_results'),
                (guard.CONTROLLER+'/terminal.json','complete',False),(guard.OWNER_STAGE+'/terminal.json','returncode',1),
                (guard.AUDIT_STAGE+'/terminal.json','complete',False),(guard.TEST_RUNTIME+'/terminal.json','completed',1919),
                (guard.TEST_RUNTIME+'/status.json','active',['fake-live'])):
            fixture=Fixture();fixture.data[path][key]=value
            with self.subTest(path=path,key=key),self.assertRaises(ValueError):fixture.proof()
            self.assertNotIn(guard.RESULT,fixture.reads)

    def test_external_lock_and_result_sha_pins_are_mandatory(self):
        fixture=Fixture()
        with self.assertRaises(TypeError):guard.require_complete_TEST_metadata(ROOT,'a'*64,read_bytes=fixture.read)
        self.assertEqual(fixture.reads,[])
        with patch.object(guard,'SOURCE_SHA',fixture.source_sha):
            with self.assertRaisesRegex(ValueError,'external final906'):
                guard.require_complete_TEST_metadata(ROOT,None,resume_lock_sha256='b'*64,read_bytes=fixture.read)
            with self.assertRaisesRegex(ValueError,'metadata SHA changed'):
                guard.require_complete_TEST_metadata(ROOT,'a'*64,resume_lock_sha256='b'*64,read_bytes=fixture.read)
            with self.assertRaisesRegex(ValueError,'actual907 complete final result pointer'):
                guard.require_complete_TEST_metadata(ROOT,'a'*64,
                    resume_lock_sha256=fixture.pointer(guard.CONTROL_LOCK)['sha256'],read_bytes=fixture.read)
        self.assertNotIn(guard.RESULT,fixture.reads)

    def test_motion_unknown_retained_but_numeric_string_or_missing_not_bool(self):
        for tier in ('old','new'):
            for bad in (0,1,'False','missing'):
                fixture=Fixture();rows=fixture.data[guard.TEST_RUNTIME+'/plan.json']['jobs']
                row=rows[0 if tier=='old' else fixture.count]
                record=fixture.data[guard.TEST_RUNTIME+'/status.json']['completed'][row['job_id']]
                if bad=='missing':record.pop('motion_pass')
                else:record['motion_pass']=bad
                with self.subTest(tier=tier,bad=bad),self.assertRaisesRegex(ValueError,'only clean native outcomes'):
                    fixture.proof()

    def test_source_code_and_control_protocol_mismatches_refused(self):
        fixture=Fixture();fixture.data[guard.OWNER_SOURCE]=b'CPU fake altered source'
        with self.assertRaisesRegex(ValueError,'metadata SHA changed'):fixture.proof()
        fixture=Fixture();fixture.data[guard.TEST_RUNTIME+'/status.json']['author_outcome_protocol']={}
        with self.assertRaisesRegex(ValueError,'scientific TEST owner binding'):fixture.proof()
        self.assertNotIn(guard.RESULT,fixture.reads)

    def test_partial906_wrong_protocol_or_single_owner_claim_refused(self):
        for key,value in (('completed_method_cases',1919),('author_outcome_protocol',{}),('original754_single_owner_result_claimed',True)):
            fixture=Fixture();fixture.data[guard.RESULT][key]=value;fixture.refresh_result()
            with self.subTest(key=key),self.assertRaises(ValueError):fixture.proof()

    def test_new901_sidecar_tier_and_old_false_must_remain_literal(self):
        for fault in ('sidecar','tier','old_false'):
            fixture=Fixture();result=fixture.data[guard.RESULT];rows=fixture.data[guard.TEST_RUNTIME+'/plan.json']['jobs']
            if fault=='sidecar':result['batch_driver_receipts'].pop()
            elif fault=='tier':result['evidence_tiers'][134]['evidence_runtime']=guard.OLD_RUNTIME
            else:fixture.data[guard.TEST_RUNTIME+'/status.json']['completed'][rows[0]['job_id']]['native_success']=True
            fixture.refresh_result()
            with self.subTest(fault=fault),self.assertRaises(ValueError):fixture.proof()

    def test_original_grid_duplicate_source_or_changed_method_rejected(self):
        fixture=Fixture();plan=deepcopy(fixture.data[guard.OLD_RUNTIME+'/plan.json'])
        self.assertEqual(len(guard._row_grid(plan)),1920)
        for fault in ('duplicate','wrong_method','source_sha'):
            changed=deepcopy(plan)
            if fault=='duplicate':changed['jobs'][-1]=deepcopy(changed['jobs'][-2])
            elif fault=='wrong_method':changed['jobs'][0]['method']='Value'
            else:changed['jobs'][1]['source_row_sha256']='f'*64
            with self.subTest(fault=fault),self.assertRaises(ValueError):guard._row_grid(changed)

    def test_changed_C_partition_or_existing_control_lock_binding_refused(self):
        fixture=Fixture();fixture.data[guard.CONTROL]['completed_job_ids'].pop()
        with self.assertRaisesRegex(ValueError,'metadata SHA changed'):fixture.proof()
        fixture=Fixture();fixture.data[guard.CONTROL_LOCK]['files'][guard.CONTROL]='f'*64
        with self.assertRaisesRegex(ValueError,'metadata SHA changed'):fixture.proof()

    def test_metadata_allowlist_never_opens_case_raw_manifest_model(self):
        reader=guard._Metadata(ROOT,lambda name:self.fail('forbidden read callback '+name))
        for name in (guard.MANIFEST,guard.SOURCE_LOCK,guard.SUITE,'data/test.hdf5','models/weights.pt',
                guard.ART+'/test/'+guard.TASKS[0]+'/CPU_fake_0/physmani/case.json'):
            with self.subTest(name=name),self.assertRaisesRegex(ValueError,'non-metadata'):reader.raw(name)

    def test_lock_order_prefixes_and_default_no_activation(self):
        self.assertEqual(guard.LOCKS,(guard.CONTROLLER+'.lock','runtime/framework/ordered_resume_VAL_TEST_v775.lock',
            guard.OLD_CONTROLLER+'.lock','runtime/framework/fixed_TRAIN104_geometry_native.lock'))
        for source in (guard.OWNER_SOURCE,guard.CONTROLLER_SOURCE,guard.BATCH_SOURCE,guard.AUDITOR_SOURCE):
            self.assertTrue(Path(source).name.startswith(guard.NATIVE_PREFIXES))
        resources=Resources();fixture=Fixture()
        context=guard.reserved_resources_after_TEST(ROOT,fixture.pointer(guard.RESULT)['sha256'],
            resume_lock_sha256=fixture.pointer(guard.CONTROL_LOCK)['sha256'],read_bytes=fixture.read,
            stop_check=resources.stop,reserve=resources.reserve,inventory=resources.inventory,is_alive=resources.is_alive)
        self.assertEqual(resources.events,[]);self.assertEqual(fixture.reads,[])
        with self.assertRaises(ValueError):
            with context:self.fail('default disabled entered')
        self.assertEqual(resources.events,[]);self.assertEqual(fixture.reads,[])

    def lease(self,fixture,resources):
        return guard.reserved_resources_after_TEST(ROOT,fixture.pointer(guard.RESULT)['sha256'],**resources.kwargs(fixture))

    def test_live_permit_requires_explicit_test_only_and_dies_at_exit(self):
        fixture=Fixture();resources=Resources()
        with patch.object(guard,'SOURCE_SHA',fixture.source_sha):
            with self.lease(fixture,resources) as permit:
                self.assertTrue(resources.held)
                with self.assertRaisesRegex(ValueError,'cannot authorize native'):guard.require_permit(permit,ROOT)
                proof=guard.require_permit(permit,ROOT,allow_test_callbacks=True)
                self.assertEqual(proof['expected_cases'],1920)
                proof['complete']=False
                self.assertTrue(guard.require_permit(permit,ROOT,allow_test_callbacks=True)['complete'])
            self.assertFalse(resources.held)
            with self.assertRaisesRegex(ValueError,'live owned'):guard.require_permit(permit,ROOT,allow_test_callbacks=True)
        self.assertEqual(resources.events[-1],'reserve_exit')

    def test_STOP_before_reserve_inside_after_inventory_and_during_permit(self):
        for point in (1,2,3,4):
            fixture=Fixture();resources=Resources();resources.stop_at=point
            with patch.object(guard,'SOURCE_SHA',fixture.source_sha):
                with self.subTest(point=point),self.assertRaisesRegex(InterruptedError,'CPU fake STOP'):
                    with self.lease(fixture,resources) as permit:
                        guard.require_permit(permit,ROOT,allow_test_callbacks=True)
            self.assertFalse(resources.held)
            if point==1:self.assertNotIn('reserve_exit',resources.events)
            else:self.assertEqual(resources.events[-1],'reserve_exit')
            if point in (1,2):self.assertEqual(fixture.reads,[])

    def test_busy_knownlive_or_incomplete_inventory_cannot_grant_permit(self):
        for fault in ('busy','knownlive','host_hidden','ancestor','uid_unknown','native_live'):
            fixture=Fixture();resources=Resources()
            if fault=='busy':resources.busy=True
            elif fault=='knownlive':resources.live=True
            elif fault=='host_hidden':resources.visibility['host_visibility_available']=False
            elif fault=='ancestor':resources.visibility['ancestor_chain_complete']=False
            elif fault=='uid_unknown':resources.visibility['same_uid_inventory_incomplete']=True
            else:resources.visibility['live_project_native_processes']=[dict(pid=999,start_ticks=1)]
            with patch.object(guard,'SOURCE_SHA',fixture.source_sha):
                with self.subTest(fault=fault),self.assertRaises((ValueError,RuntimeError)):
                    with self.lease(fixture,resources):self.fail('unsafe inventory entered')
            self.assertFalse(resources.held)
            if fault!='busy':self.assertEqual(resources.events[-1],'reserve_exit')

    def test_metadata_failure_after_lock_releases_without_inventory_or_lease(self):
        fixture=Fixture();resources=Resources()
        fixture.data[guard.TEST_RUNTIME+'/terminal.json']['complete']=False
        with patch.object(guard,'SOURCE_SHA',fixture.source_sha):
            with self.assertRaises(ValueError):
                with self.lease(fixture,resources):self.fail('incomplete metadata entered')
        self.assertFalse(resources.held);self.assertNotIn('inventory',resources.events)
        self.assertEqual(resources.events[-1],'reserve_exit');self.assertEqual(len(guard._LEASES),0)

    def test_exception_finally_releases_and_permit_cannot_cross_pid_or_root(self):
        fixture=Fixture();resources=Resources();permit=None
        with patch.object(guard,'SOURCE_SHA',fixture.source_sha):
            with self.assertRaisesRegex(RuntimeError,'CPU fake caller failure'):
                with self.lease(fixture,resources) as permit:
                    with patch.object(guard.os,'getpid',return_value=4243):
                        with self.assertRaisesRegex(ValueError,'same-process'):guard.require_permit(permit,ROOT,allow_test_callbacks=True)
                    with self.assertRaisesRegex(ValueError,'canonical-root'):guard.require_permit(permit,ROOT/'other',allow_test_callbacks=True)
                    with self.assertRaisesRegex(ValueError,'live owned'):guard.require_permit(guard._Permit(),ROOT,allow_test_callbacks=True)
                    raise RuntimeError('CPU fake caller failure')
        self.assertFalse(resources.held);self.assertEqual(resources.events[-1],'reserve_exit')
        with self.assertRaisesRegex(ValueError,'live owned'):guard.require_permit(permit,ROOT,allow_test_callbacks=True)
        self.assertEqual(len(guard._LEASES),0)


if __name__=='__main__':
    unittest.main()
