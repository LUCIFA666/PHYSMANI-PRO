"""Source-only and memory/temp963 tests. No real experiment metadata is read."""
from collections import Counter
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock,patch

PROJECT=Path(__file__).resolve().parents[1]


def module_at(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module);return module


m=module_at('CPU_composite_motion963',PROJECT/'scripts/summarize_composite_TEST_motion_v963_r10.py')
old=module_at('CPU_composite_motion963_old920fixture',PROJECT/'tests/test_full_TEST_motion_v920.py')


def source(name):return dict(path=name,sha256=m.PINS.get(name,m.sha_bytes(('CPU-fake-source:'+name).encode())))


def memory_fixture(reducer):
    with patch.object(old,'source',source),patch.object(old.fakes,'source',source):initial=old.memory_fixture(reducer)
    data=initial.data
    def checksum(name):
        return m.PINS[name] if name in m.PINS else m.digest(data[name]) if name in data else source(name)['sha256']
    def ptr(name):return dict(path=name,sha256=checksum(name))
    parent_plan=data[m.PARENT+'/plan.json'];ids=[r['job_id'] for r in parent_plan['jobs']]
    parent_status=data[m.PARENT+'/status.json'];parent_terminal=data[m.PARENT+'/terminal.json']
    parent_status.update(completed={k:deepcopy(parent_status['completed'][k]) for k in ids[:584]},
        failed={k:dict(complete=False,automatic_retry=False,returncode=1,error='CPU synthetic environmental attempt') for k in ids[584:590]},
        active=[],checking=[],ready_jobs=[],CPU_ready=0,native_slots=0,GPU_slots=0,drain_requested=True,stop_requested=False)
    parent_terminal.update(complete=False,completed=584,failed=6,drained=True,drain_requested=True,stop_requested=False)
    data[m.PARENT+'/DRAIN']=dict(reason='CPU synthetic natural environmental drain')
    data[m.OLD_CONTROLLER+'/status.json']=dict(stage='requires_concrete_repair',automatic_retry=False)
    data[m.OLD_CONTROLLER+'/failure.json']=dict(complete=False,user_stop=False,automatic_retry=False)
    data['runtime/framework/frozen_TEST_environment_recheck_v954.json']=dict(schema='r10_fresh_frozen_TEST_environment_CPU_observation_v954',
        complete=True,source=source(m.CHECK_SOURCE),Native_started=False,GPU_started=False,admission_or_TEST_started=False,
        declared_model_case_raw_bytes_read=False,scientific_inputs_or_old_locks_modified=False,
        original_v900_topology_constructor_passed=True,original_full_topology_final_check_passed=True,
        parsed_suite_lock_is_exact_hashed_payload=True,topology_is_point_observation_not_continuous=True,
        declared_files=100,unique_canonical_paths=100,watched_directories=10,
        suite_lock=ptr(m.BASE.SUITE_LOCK),actual_environment=dict(schema='r10_actual_frozen_OS_namespace_receipt_v820',complete=True,
            checked_OS_files=6418,all_current_OS_targets_full_byte_SHA_verified=True,
            **{field:data[m.PARENT_CONTROL][key] for key,field in (('environment_manifest','manifest'),
                ('environment_wrapper','wrapper'),('environment_namespace_proof','namespace_qualification'))}))
    control=deepcopy(data[m.PARENT_CONTROL]);control.update(schema=m.C.CONTROL_SCHEMA,
        completed_job_ids=ids[:584],failed_job_ids=ids[584:590],remaining_job_ids=ids[584:],
        completed_key_sha256=m.digest(sorted(ids[:584])),failed_key_sha256=m.digest(sorted(ids[584:590])),
        remaining_key_sha256=m.digest(sorted(ids[584:])),previous_runtime=m.PARENT,previous_controller_runtime=m.OLD_CONTROLLER,
        previous_resume_control=ptr(m.PARENT_CONTROL),previous_resume_control_lock=ptr(m.PARENT_LOCK),
        resume_owner=source(m.OWNER),resume_auditor=source(m.AUDITOR),result_auditor=source(m.AUDITOR),resume_controller=source(m.CONTROLLER_SOURCE),
        environment_recheck=ptr('runtime/framework/frozen_TEST_environment_recheck_v954.json'))
    for key in ('launch','plan','status','terminal'):
        control['original_'+key]=data[m.PARENT_CONTROL]['previous_'+key]
        control['previous_'+key]=ptr(m.PARENT+'/'+key+'.json')
    for key in ('launch','status','failure'):control['previous_controller_'+key]=ptr(m.OLD_CONTROLLER+'/'+key+'.json')
    control['previous_DRAIN']=ptr(m.PARENT+'/DRAIN');data[m.CONTROL]=control
    files=deepcopy(data[m.PARENT_LOCK]['files'])
    for name in m.SOURCES:files[str(PROJECT/name)]=source(name)['sha256']
    for key,p in control.items():
        if isinstance(p,dict) and set(p)=={'path','sha256'} and (key.startswith('previous_') or key.startswith('original_') or key=='environment_recheck'):
            files[str(PROJECT/p['path'])]=p['sha256']
    for row in parent_plan['jobs'][:584]:
        runtime=row['evidence_runtime'];out=m.option(row['physical_argv'],'--out')
        for kind,leaves in (('CPU',('launch.json','terminal.json','stdout.log','CPU_preflight.json')),('native',('launch.json','terminal.json'))):
            for leaf in leaves:
                name=runtime+'/'+kind+'/'+row['job_id']+'/'+leaf;files[str(PROJECT/name)]=checksum(name)
        for leaf in ('case.json','admission_driver_receipt.json',*(['batch_driver_receipt.json'] if runtime==m.PARENT else [])):
            name=out+'/'+leaf;files[str(PROJECT/name)]=checksum(name)
    files[str(PROJECT/m.CONTROL)]=checksum(m.CONTROL)
    data[m.LOCK]=dict(schema=m.C.LOCK_SCHEMA,approved_for_execution=True,control=ptr(m.CONTROL),files=files)
    cp,lp=ptr(m.CONTROL),ptr(m.LOCK);jobs=m.C.resumed_jobs(parent_plan,control,control['execution_entry'])
    common={k:deepcopy(data[m.PARENT+'/launch.json'][k]) for k in ('split','policy_sha256','author_outcome_protocol','execution_protocol_changed',
        'resource_profile','resource_qualification','execution_implementation','admission_driver','admission_guard','native_runtime','original_native_runtime','automatic_retry')}
    common.update(resume_control=cp,resume_control_lock=lp,previous_resume_control=control['previous_resume_control'],
        previous_resume_control_lock=control['previous_resume_control_lock'],environment_recheck=control['environment_recheck'])
    env=old.fakes.environment(1002,control);parent_owner=data[m.PARENT+'/launch.json'];original_owner=data[m.ORIGINAL+'/launch.json']
    launch=dict(common,pid=1002,start_ticks=1003,source_sha256=source(m.OWNER)['sha256'],imported_completed_cases=584,
        execution_entry=control['execution_entry'],frozen_environment=env,max_native_slots=4,max_GPU_slots=2,max_CPU_preflight_slots=2,
        suite_lock_sha256=checksum(m.BASE.SUITE_LOCK),split_source_lock_sha256=checksum(m.BASE.SOURCE_LOCK),source_manifest=ptr(m.BASE.MANIFEST),
        base_stage_owner=source(m.BASE.OWNER),parent_stage_owner=source(m.LEGACY.OWNER),batch_sources=control['execution_implementation']['sources'])
    data[m.RUNTIME+'/launch.json']=launch
    data[m.RUNTIME+'/plan.json']=dict(common,jobs=jobs,expected_cases=1920,imported_completed_cases=584,source_outcome_filtering=False,frozen_environment=env)
    completed={};cases=[];drivers=[];tiers=[];batches=[];clean=dict(complete=True,returncode=0,timed_out=False,interrupted=False,automatic_retry=False)
    shared=control['execution_implementation']['sources']
    for i,row in enumerate(jobs):
        imported=row['imported_completed'];runtime=row['evidence_runtime'];out=m.option(row['physical_argv'],'--out')
        oldrow=parent_plan['jobs'][i];oldout=m.option(oldrow['physical_argv'],'--out')
        if not imported:
            case=deepcopy(data[oldout+'/case.json'])
            if i==584:
                case['all_stage_counted_joint_motion']['all_stage_actual_window_checks_passed']=None
                mirror=case['alignment_receipt'].get('all_stage_counted_joint_motion_v191')
                if isinstance(mirror,dict):mirror['all_stage_actual_window_checks_passed']=None
            data[out+'/case.json']=case;data[out+'/admission_driver_receipt.json']=deepcopy(data[oldout+'/admission_driver_receipt.json'])
            data[out+'/admission_driver_receipt.json']['case']=ptr(out+'/case.json')
        case=data[out+'/case.json'];casep,driverp=ptr(out+'/case.json'),ptr(out+'/admission_driver_receipt.json')
        if not imported:
            bindings={k:row[k] for k in m.H.BINDINGS}
            for kind in ('CPU','native'):
                check=kind=='CPU';leaf=runtime+'/'+kind+'/'+row['job_id']
                data[leaf+'/launch.json']=dict(argv=row['argv']+(['--check'] if check else []),physical_argv=row['physical_argv']+(['--check'] if check else []),
                    check_only=check,owner_pid=1002,native_slots=0 if check else 1,GPU_slots=0 if check else int(row['method']=='physmani'),
                    execution_protocol_changed=True,expected_identity_sha256=None if check else case['identity']['sha256'],
                    launcher_pid=30000+2*i+int(not check),launcher_start_ticks=40000+2*i+int(not check),execution_entry=row['execution_entry'],
                    inherited_frozen_environment=env,current_mount_namespace=env['current_mount_namespace'],**bindings)
                data[leaf+'/terminal.json']=dict(clean)
                if check:
                    data[leaf+'/CPU_preflight.json']=dict(complete=True,native_started=False,GPU_started=False,actual_split='test',
                        execution_protocol_changed=True,identity=case['identity'],component_domain=case['component_domain'],
                        batch_execution=old.batch_receipt(row,shared,check=True),**bindings)
                    data[leaf+'/stdout.log']=deepcopy(data[leaf+'/CPU_preflight.json'])
                else:
                    bp=out+'/batch_driver_receipt.json';data[bp]=old.batch_receipt(row,shared,check=False,case=casep,driver=driverp)
                    data[leaf+'/terminal.json'].update(case=casep,admission_driver_receipt=driverp,batch_driver_receipt=ptr(bp),
                        native_success=False,motion_pass=m.recorded_motion_flag(case))
        record=deepcopy(data[runtime+'/native/'+row['job_id']+'/terminal.json']);completed[row['job_id']]=record
        bp=record.get('batch_driver_receipt')
        if bp:batches.append(bp)
        owner={m.ORIGINAL:original_owner,m.PARENT:parent_owner,m.RUNTIME:launch}[runtime]
        cases.append(casep);drivers.append(driverp)
        tiers.append(dict(job_id=row['job_id'],evidence_runtime=runtime,imported_completed=imported,actual_stage_owner_pid=owner['pid'],
            actual_stage_owner_start_ticks=owner['start_ticks'],resume_lineage=row['resume_lineage'],previous_row_sha256=row['previous_row_sha256'],
            execution_entry=row['execution_entry'],actual_outer_argv=row['argv'],physical_argv=row['physical_argv'],case=casep,
            admission_driver_receipt=driverp,batch_driver_receipt=bp))
    data[m.RUNTIME+'/status.json']=dict(common,expected=1920,completed=completed,failed={},active=[],checking=[],ready_jobs=[],
        CPU_ready=0,native_slots=0,GPU_slots=0,CPU_launched=1336,imported_completed=584,owner_error=None,stop_requested=False,drain_requested=False,frozen_environment=env)
    data[m.RUNTIME+'/terminal.json']=dict(common,complete=True,expected=1920,completed=1920,failed=0,owner_error=None,stop_requested=False,
        drain_requested=False,drained=False,CPU_ready_unexecuted=[],CPU_not_launched=0,imported_completed_cases=584,resumed_new_cases=1336,
        formal_release_requires_independent_audit=True,frozen_environment=env,native_successes=0)
    controller=dict(pid=1102,start_ticks=1103,source_sha256=source(m.CONTROLLER_SOURCE)['sha256'],resume_control=cp,resume_control_lock=lp,
        automatic_retry=False,frozen_environment=old.fakes.environment(1102,control))
    data[m.CONTROLLER+'/launch.json']=controller;data[m.CONTROLLER+'/status.json']=dict(stage='complete',automatic_retry=False,frozen_environment=controller['frozen_environment'])
    args=[]
    for key,flag in (('suite','--suite'),('suite_lock','--suite-lock'),('source_inputs','--source-inputs'),('validation_release','--validation-release')):args.extend((flag,control[key]['path']))
    args+=['--runtime',m.RUNTIME,'--out',m.ART+'/test','--resume-control',m.CONTROL,'--resume-lock',m.LOCK,'--resume-lock-sha256',lp['sha256']]
    auditargs=['--resume-control',m.CONTROL,'--resume-lock',m.LOCK,'--resume-lock-sha256',lp['sha256']]
    for stage,script,number,tail in ((m.OWNER_STAGE,m.OWNER,1002,args),(m.AUDIT_STAGE,m.AUDITOR,1202,auditargs)):
        data[stage+'/launch.json']=dict(pid=number,start_ticks=number+1,argv=[str(PROJECT/'.conda-env/bin/python'),'-B','-u',script,*tail],
            source_sha256=source(script)['sha256'],automatic_retry=False,inherited_frozen_environment=controller['frozen_environment'],current_mount_namespace=env['current_mount_namespace'])
        data[stage+'/terminal.json']=dict(complete=True,returncode=0,automatic_retry=False)
    launch['argv']=data[m.OWNER_STAGE+'/launch.json']['argv'][3:]
    result=deepcopy(data[old.m.RESULT]);result.update(schema=m.RESULT_SCHEMA,source_sha256=source(m.AUDITOR)['sha256'],resume_control=cp,resume_control_lock=lp,
        environment_recheck=control['environment_recheck'],old584_results_not_rerun=True,motion_unknown_retained=True,zero_tick_cases_retained=True,
        old907_successful_terminal_claimed=False,original754_or906_single_result_claimed=False,cases=cases,admission_driver_receipts=drivers,
        evidence_tiers=tiers,batch_driver_receipts=batches,frozen_environment=old.fakes.environment(1202,control),
        auditor_invocation=dict(pid=1202,start_ticks=1203,argv=data[m.AUDIT_STAGE+'/launch.json']['argv'][3:],source=source(m.AUDITOR),inherited_physical_stage_identity_claimed=False),
        historical_infrastructure_attempts=dict(job_ids=control['failed_job_ids'],previous_status=control['previous_status'],previous_terminal=control['previous_terminal'],
            counted_as_native_task_failures=False,all_keys_reexecuted_once_in_new_namespace=True))
    result['actual_stage_owners']=[dict(runtime=runtime,launch=ptr(runtime+'/launch.json'),terminal=ptr(runtime+'/terminal.json'),
        pid=owner['pid'],start_ticks=owner['start_ticks'],completed=count,resource_limits={k:owner[k] for k in ('max_native_slots','max_GPU_slots','max_CPU_preflight_slots')})
        for runtime,owner,count in ((m.ORIGINAL,original_owner,134),(m.PARENT,parent_owner,450),(m.RUNTIME,launch,1336))]
    excluded={m.BASE.MANIFEST,m.BASE.ORIGINAL_MANIFEST,m.BASE.ROUTES,m.RESULT,old.m.RESULT,m.BASE.RESULT}
    #957 does not consume its parent958 completion/stages: those happen after
    # the audit. Never pad its ledger with impossible future evidence.
    result['inputs_sha256']={str(PROJECT/name):checksum(name) for name in data
        if name not in excluded and not name.startswith(m.CONTROLLER+'/')}
    result['inputs_sha256'].update({str(PROJECT/name):source(name)['sha256'] for name in m.SOURCES})
    data[m.RESULT]=result;data[m.CONTROLLER+'/terminal.json']=dict(complete=True,automatic_retry=False,TEST_results=ptr(m.RESULT))
    data[m.AUDIT_STAGE+'/stdout.log']=dict(complete=True,schema=m.RESULT_SCHEMA,split='test',cases=1920,out=m.RESULT,
        motion_is_diagnostic_only=True,native_successes_by_method=result['native_successes_by_method'])

    class MemoryReader:
        root=PROJECT;lookup=m.Reader.lookup
        def __init__(self):
            self.inputs={};self.evidence={};self.frozen={};self.original={};self.resumed={};self.opened=[];self.data=data;self.closed=False
        def check(self,name,actual,kwargs):
            wanted=[kwargs['expected']] if kwargs.get('expected') is not None else []
            for key in ('evidence','frozen','original','resumed'):
                if kwargs.get(key):wanted.append(self.lookup(getattr(self,key),name))
            m.require(all(v==actual for v in wanted),'CPU fake input SHA changed: '+name)
            m.require(name not in self.inputs or self.inputs[name]==actual,'CPU fake input changed during read')
            self.inputs[name]=actual
        def raw(self,name,**kwargs):
            self.opened.append(name)
            value=(PROJECT/name).read_bytes() if name in m.PINS else ('CPU-fake-source:'+name).encode()
            self.check(name,m.sha_bytes(value),kwargs);return value
        def json(self,name,**kwargs):self.opened.append(name);self.check(name,checksum(name),kwargs);return data[name]
        def pointer(self,name):return dict(path=name,sha256=self.inputs[name])
        def pointed(self,p,**kwargs):m.pointer(p);return self.json(p['path'],expected=p['sha256'],**kwargs)
        def close(self):self.closed=True
    return MemoryReader()


class MotionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.reducer=m.H.load_reducer(m.Reader(PROJECT))
    def fixture(self):return memory_fixture(self.reducer)
    def collect(self,reader):return m.collect(reader,m.digest(reader.data[m.RESULT]),resume_lock_sha=m.digest(reader.data[m.LOCK]))

    def test_complete_three_tier_saved_cohort_retains_False_None_zero_ticks(self):
        reader=self.fixture();rows,result,subset,reducer=self.collect(reader)
        self.assertEqual(Counter(r['evidence_runtime'] for r in rows),{m.ORIGINAL:134,m.PARENT:450,m.RUNTIME:1336})
        self.assertEqual(Counter(r['method'] for r in rows),{method:640 for method in m.METHODS});self.assertEqual(len(result['batch_driver_receipts']),1786)
        self.assertTrue(all(r['official_task_success'] is False for r in rows));self.assertTrue(all(r['metrics']['peak_window_joint_acceleration_rad_s2'] is None for r in rows))
        row=reader.data[m.RUNTIME+'/plan.json']['jobs'][584]
        self.assertIsNone(reader.data[m.RUNTIME+'/status.json']['completed'][row['job_id']]['motion_pass'])
        self.assertTrue(reader.closed)
        for method in m.METHODS:self.assertEqual(reducer.summarize([r for r in rows if r['method']==method])['case_count'],640)

    def test_no958_complete_means_no_result_manifest_or_case_read(self):
        for path,field,value in ((m.CONTROLLER+'/terminal.json','complete',False),(m.CONTROLLER+'/status.json','stage','independent_TEST_results'),
                (m.RUNTIME+'/terminal.json','completed',1919),(m.OWNER_STAGE+'/terminal.json','returncode',1)):
            reader=self.fixture();reader.data[path][field]=value
            with self.subTest(path=path),self.assertRaises(ValueError):self.collect(reader)
            self.assertNotIn(m.RESULT,reader.opened);self.assertNotIn(m.BASE.MANIFEST,reader.opened)
            self.assertFalse(any(p.endswith('/case.json') for p in reader.opened))

    def test_missing_external_pins_refuses_before_any_read(self):
        reader=self.fixture()
        for result_sha,lock_sha in ((None,'a'*64),('a'*64,None),('notSHA','b'*64)):
            with self.assertRaises(ValueError):m.collect(reader,result_sha,resume_lock_sha=lock_sha)
        self.assertEqual(reader.opened,[])

    def test_three_owner_or_1920_pointer_mismatch_precedes_manifest_and_cases(self):
        for mode in ('owner','tierpid','missingcase','sidecar','old_native_rewrite'):
            reader=self.fixture();result=reader.data[m.RESULT]
            if mode=='owner':result['actual_stage_owners'][1]['pid']=1002
            elif mode=='tierpid':result['evidence_tiers'][134]['actual_stage_owner_pid']=1002
            elif mode=='missingcase':result['cases'].pop()
            elif mode=='sidecar':result['batch_driver_receipts'].pop()
            else:
                row=reader.data[m.RUNTIME+'/plan.json']['jobs'][0];reader.data[m.RUNTIME+'/status.json']['completed'][row['job_id']]['native_success']=True
            if mode!='old_native_rewrite':
                reader.data[m.CONTROLLER+'/terminal.json']['TEST_results']=dict(path=m.RESULT,sha256=m.digest(result))
            with self.subTest(mode=mode),self.assertRaises(ValueError):self.collect(reader)
            self.assertNotIn(m.BASE.MANIFEST,reader.opened);self.assertFalse(any(p.endswith('/case.json') for p in reader.opened))

    def test_saved_case_unknown_bool_and_wrong_native_types(self):
        reader=self.fixture();row=reader.data[m.RUNTIME+'/plan.json']['jobs'][584];case=reader.data[m.option(row['physical_argv'],'--out')+'/case.json']
        args=(row,row['source_row_sha256'],reader.data[m.CONTROL]['author_outcome_protocol'],row and case['identity']['source_input_lock_sha256'])
        for value in (None,False,True):
            changed=deepcopy(case);changed['all_stage_counted_joint_motion']['all_stage_actual_window_checks_passed']=value
            m.require_saved_case(changed,*args)
        for value in (0,1,'false',None):
            changed=deepcopy(case);changed['official_task_success']=value
            with self.subTest(value=value),self.assertRaises(ValueError):m.require_saved_case(changed,*args)
        for value in (0,1,'false'):
            changed=deepcopy(case);changed['all_stage_counted_joint_motion']['all_stage_actual_window_checks_passed']=value
            with self.subTest(motion=value),self.assertRaises(ValueError):m.require_saved_case(changed,*args)

    def test_saved_case_clock_cleanup_identity_and_RML_remain_strict(self):
        reader=self.fixture();row=reader.data[m.RUNTIME+'/plan.json']['jobs'][584]
        case=reader.data[m.option(row['physical_argv'],'--out')+'/case.json']
        args=(row,row['source_row_sha256'],reader.data[m.CONTROL]['author_outcome_protocol'],case['identity']['source_input_lock_sha256'])
        for mode in ('clock','cleanup','identity','RML','scope','coverage'):
            changed=deepcopy(case)
            if mode=='clock':changed['counted_clock']['seconds']=.05
            elif mode=='cleanup':changed['cleanup_completed']=False
            elif mode=='identity':changed['identity']['config_sha256']='f'*64
            elif mode=='RML':changed['common_author_RML_parameters']['actual_parameters']=dict(m.PARAMETERS,max_acceleration=6)
            elif mode=='scope':changed['TEST_used_for_fitting']=True
            else:changed['all_stage_counted_joint_motion']['events']=[{}]
            with self.subTest(mode=mode),self.assertRaises(ValueError):m.require_saved_case(changed,*args)

    def test_model_raw_and_unlisted_paths_are_refused_before_open(self):
        with tempfile.TemporaryDirectory() as name:
            reader=m.Reader(Path(name))
            for path in ('model.pt','raw.hdf5','../outside.json','/absolute/case.json','runtime/unknown/case.json'):
                with self.subTest(path=path),patch.object(Path,'read_bytes',side_effect=AssertionError('forbidden path was opened')),self.assertRaises(ValueError):
                    reader.raw(path)

    def test_existing_output_never_instantiates_reader_or_overwrites(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);output=root/m.OUTPUT;output.parent.mkdir(parents=True);output.write_bytes(b'CPU old summary')
            with patch.object(m,'ROOT',root),patch.object(m,'Reader') as reader,self.assertRaisesRegex(ValueError,'preserve existing963'):
                m.main(['--test-results-sha256','a'*64,'--resume-lock-sha256','b'*64])
            reader.assert_not_called();self.assertEqual(output.read_bytes(),b'CPU old summary')

    def test_reporting_helper_is_supplementary_not_invented_old_membership(self):
        reader=self.fixture();self.collect(reader);record=reader.reporting_dependencies[0]
        self.assertFalse(record['original_physical_input_lock_member']);self.assertFalse(record['original_TEST_suite_input_lock_member'])
        self.assertFalse(record['old_input_lock_amended']);self.assertFalse(record['proof_of_original_physical_freeze_membership_claimed'])
        self.assertTrue(record['bound_by_actual955_supplement_and957_evidence'])

    def test_new_report_is_only_postprocessing_and_algorithms_not_replaced(self):
        reader=self.fixture();rows,result,subset,reducer=self.collect(reader);reader.inputs[m.SOURCE]='f'*64
        report=m.build_report(reader,rows,result,subset,reducer)
        self.assertEqual(report['schema'],m.SCHEMA);self.assertEqual(report['case_count'],1920)
        self.assertEqual(report['actual_test_auditor'],dict(path=m.AUDITOR,sha256=m.AUDITOR_SHA))
        self.assertEqual(report['provenance_runtime_counts'],{m.ORIGINAL:134,m.PARENT:450,m.RUNTIME:1336})
        self.assertTrue(report['full1920_three_tier_TEST_results_and_complete958_observed'])
        for flag in ('new_release_granted_by_this_summary','success_rate_or_acceleration_filter_used','native_GPU_or_jobs_started','TEST_used_for_fitting','TXT_or_handoff_written'):
            self.assertIs(report[flag],False)
        self.assertEqual(report['saved_motion_algorithm_source'],dict(path=m.LEGACY_SOURCE,sha256=m.LEGACY_SHA))

    def test_failure_main_never_creates_output_or_parent_directory(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);denied=Mock(side_effect=ValueError('CPU incomplete958'))
            with patch.object(m,'ROOT',root),patch.object(m,'Reader',return_value=object()),patch.object(m,'collect',denied),self.assertRaisesRegex(ValueError,'CPU incomplete958'):
                m.main(['--test-results-sha256','a'*64,'--resume-lock-sha256','b'*64])
            self.assertFalse((root/m.OUTPUT).exists());self.assertFalse((root/m.OUTPUT).parent.exists())


if __name__=='__main__':unittest.main()
