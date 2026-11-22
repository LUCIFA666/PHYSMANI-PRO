"""CPU-only source/fake tests. No real TEST metadata, case or851 main is read."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import tempfile
import unittest

PROJECT=Path(__file__).resolve().parents[1]


def module_at(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module);return module


report=module_at('TEST_motion_v851_test_subject',PROJECT/'scripts/summarize_formal_TEST_motion_v851_r10.py')
old_fakes=module_at('VAL_motion_v847_CPU_fakes',PROJECT/'tests/test_formal_VAL_motion_v847.py')


def source(name):return dict(path=name,sha256=report.sha_bytes(('CPU-fake-source:'+name).encode()))


def environment(pid,control):
    return dict(schema='r10_actual_frozen_OS_namespace_receipt_v820',complete=True,checked_OS_files=6418,
        all_current_OS_targets_full_byte_SHA_verified=True,host_PID_network_namespace_preserved=True,proc_or_dev_replaced=False,
        task_parameters_or_strategy_changed=False,current_mount_namespace='mnt:[2]',mount_namespace='mnt:[2]',reference_mount_namespace='mnt:[1]',
        manifest=control['environment_manifest'],wrapper=control['environment_wrapper'],namespace_qualification=control['environment_namespace_proof'],
        observer_pid=pid,pid=pid,observer_start_ticks=pid+1)


def source_fixture(val_pointer,preparer_sha):
    rows=[];original=[];files={}
    for task in report.TASKS:
        for i in range(40):
            raw='episode'+str(i);name='/UNOPENED_TEST/'+task+'/'+raw+'/first.pkl'
            row=dict(task=task,raw_episode_id=raw,path='/UNOPENED_TEST/'+task+'/'+raw,split='test',mapping_status='verified',
                variation=0,official_episode_index=i,source_required_files={'first.pkl':dict(path=name,sha256='a'*64)})
            rows.append(row);original.append(dict(task=task,raw_episode_id=raw,path='/UNOPENED_original/'+task+'/'+raw));files[name]='a'*64
    manifest=dict(split='test',mapping_status='verified',episode_count=640,episodes=rows,training_calibration_or_parameter_selection_used=False,
        complete_unchanged_VAL_release_observed_before_source_selection=val_pointer)
    lock=dict(schema='r10_full16_post_admission_split_source_lock_v1',split='test',source_count=640,policy_sha256=report.POLICY,
        complete_VAL_release_observed=True,full16_admission_observed_before_source_selection=True,source_content_used_for_policy_fitting=False,
        TEST_used_for_fitting=False,native_started=False,GPU_started=False,source_lock_preparer_sha256=preparer_sha,files=files)
    return manifest,dict(episodes=original),lock


def result_fixture():
    return dict(complete=True,split='test',source_cases=640,method_cases=1920,completed_method_cases=1920,
        methods=list(report.METHODS),tasks=list(report.TASKS),policy_sha256=report.POLICY,common_RML_profile_sha256=report.PROFILE,
        TEST_used_for_fitting=False,motion_is_diagnostic_only=True,motion_metrics_required_for_all_methods=True,native_task_success_unchanged=True,
        native_success_is_not_a_release_threshold=True,source_outcome_filtering=False,native_failures_retained=True,Value_NGFF_FASTER_used=False,
        execution_protocol_changed=True,ours_measured_motion_passed=False,native_successes_by_method={m:0 for m in report.METHODS},
        cases=[],admission_driver_receipts=[],inputs_sha256={'fake-only':'a'*64})


def test_case(row,protocol,source_lock_sha):
    case,_,_,_=old_fakes.zero_tick_case();task=row['task'];raw=row['raw'];method=row['method']
    case.update(split='test',task=task,method=method,source_raw_episode_id=raw,parent_raw_episode_id=None,
        formal_validation_metric=False,formal_TEST_metric=True,formal_test_used=True,author_outcome_protocol=protocol)
    case['identity'].update(split='test',task=task,method=method,raw_episode_id=raw,source_raw_episode_id=raw,parent_raw_episode_id=None,
        source_row_sha256=row['source_row_sha256'],config_sha256=row['configuration_sha256'],frozen_inputs_sha256=row['input_lock_sha256'],
        source_input_lock_sha256=source_lock_sha)
    case['identity'].pop('sha256',None);case['identity']['sha256']=report.digest(case['identity'])
    case['component_domain'].update(split='test',task=task,method=method,raw_episode_id=raw)
    case['full16_frozen_execution'].update(actual_split='test',source_row_sha256=row['source_row_sha256'])
    if method=='physmani':case['actor_metrics']=dict(baseline_steps=[])
    return case


def memory_fixture(reducer):
    """All structures below remain in memory; no fake release/artifact is written."""
    data={}
    def pointer(name):return dict(path=name,sha256=report.PINS.get(name,report.digest(data[name])))
    protocol=dict(implementation=source('our_mani/multitask/native_outcome_sampling_v741.py'),
        runtime_consumer=source('scripts/run_task_executor_smoke_author_suite16_v744_r10.py'),
        original_runtime=source('scripts/run_task_executor_smoke_rml_suite16_v664_r10.py'))
    control=dict(policy_sha256=report.POLICY,author_outcome_protocol=protocol,
        resume_auditor=source('scripts/audit_environment_resumed_formal_VAL_v824_r10.py'),
        environment_manifest=dict(path='fake-env-manifest',sha256='a'*64),environment_wrapper=dict(path='fake-wrapper',sha256='b'*64),
        environment_namespace_proof=dict(path='fake-qualification',sha256='c'*64))
    data[report.CONTROL]=control
    original_files={str(PROJECT/name):source(name)['sha256'] for name in report.SOURCES}
    data[report.LOCK]=dict(approved_for_execution=True,files=original_files)
    data[report.VAL_RELEASE]=old_fakes.release_fixture()
    data[report.VAL_RELEASE].update(source_sha256=control['resume_auditor']['sha256'],resume_control=pointer(report.CONTROL),resume_control_lock=pointer(report.LOCK))
    val_pointer=pointer(report.VAL_RELEASE)
    manifest,original,source_lock=source_fixture(val_pointer,source(report.SOURCE_PREPARER)['sha256'])
    data[report.ORIGINAL_MANIFEST]=original;manifest['original_source']=pointer(report.ORIGINAL_MANIFEST)
    data[report.MANIFEST]=manifest;source_lock['source_manifest_sha256']=pointer(report.MANIFEST)['sha256'];data[report.SOURCE_LOCK]=source_lock
    sources=report.source_grid(manifest,original,source_lock,val_pointer,source(report.SOURCE_PREPARER)['sha256'])
    driver=source('scripts/run_formal_case_author_outcome_v751_r10.py');guard=source('our_mani/multitask/formal_suite_admission_v728.py')
    worker=source('scripts/run_task_executor_smoke_author_outcome_v743_r10.py')
    routes=dict(routes=[dict(task=t,method=m,worker=worker,consumer=protocol['runtime_consumer'],
        configuration=dict(path='UNOPENED-config/'+t+'/'+m,sha256='c'*64),frozen_inputs=dict(path='UNOPENED-lock/'+t+'/'+m,sha256='d'*64))
        for t in report.TASKS for m in report.METHODS])
    data[report.ROUTES]=routes;original_files[str(PROJECT/report.ROUTES)]=pointer(report.ROUTES)['sha256']
    data[report.SUITE]=dict(validation_release_path=report.VAL_RELEASE,source_manifests={'test':pointer(report.MANIFEST)},routes=pointer(report.ROUTES),
        author_outcome_protocol=protocol,formal_case_driver=driver,formal_admission_guard=guard,formal_result_auditor=source(report.AUDITOR))
    frozen=dict(original_files)
    for name in (report.SUITE,report.MANIFEST,report.ORIGINAL_MANIFEST,report.VAL_RELEASE):frozen[str(PROJECT/name)]=pointer(name)['sha256']
    data[report.SUITE_LOCK]=dict(approved_for_execution=True,actual_formal_split='test',policy_sha256=report.POLICY,
        model_or_policy_parameters_changed=False,original_945_VAL_release=val_pointer,files=frozen)
    data[report.CONTROLLER+'/launch.json']=dict(source_sha256=source(report.CONTROLLER_SOURCE)['sha256'],pid=400,
        resume_control=pointer(report.CONTROL),resume_control_lock=pointer(report.LOCK),frozen_environment=environment(400,control))
    data[report.CONTROLLER+'/terminal.json']=dict(complete=True,automatic_retry=False,TEST_results=report.RESULT)
    common=['--suite',report.SUITE,'--suite-lock',report.SUITE_LOCK,'--source-inputs',report.SOURCE_LOCK,'--split','test',
        '--validation-release',report.VAL_RELEASE,'--runtime',report.RUNTIME]
    for stage,script,pid,out in ((report.OWNER_STAGE,report.OWNER,700,report.ART+'/test'),(report.AUDIT_STAGE,report.AUDITOR,800,report.RESULT)):
        data[stage+'/launch.json']=dict(argv=[str(PROJECT/'.conda-env/bin/python'),'-u',script,*common,'--out',out],
            source_sha256=source(script)['sha256'],pid=pid,start_ticks=pid+1,automatic_retry=False,
            inherited_frozen_environment=environment(400,control),current_mount_namespace='mnt:[2]')
        data[stage+'/terminal.json']=dict(complete=True,returncode=0,automatic_retry=False)
    data[report.RUNTIME+'/launch.json']=dict(source_sha256=source(report.OWNER)['sha256'],split='test',pid=700,start_ticks=701,
        policy_sha256=report.POLICY,suite_lock_sha256=pointer(report.SUITE_LOCK)['sha256'],split_source_lock_sha256=pointer(report.SOURCE_LOCK)['sha256'],
        source_manifest=pointer(report.MANIFEST),author_outcome_protocol=protocol,execution_protocol_changed=True,automatic_retry=False)
    data[report.RUNTIME+'/terminal.json']=dict(complete=True,split='test',expected=1920,completed=1920,failed=0,stop_requested=False,
        automatic_retry=False,policy_sha256=report.POLICY,author_outcome_protocol=protocol,execution_protocol_changed=True)
    result=result_fixture();result.update(source_sha256=source(report.AUDITOR)['sha256'],author_outcome_protocol=protocol,
        admission_driver=driver,admission_guard=guard,source_input_lock=pointer(report.SOURCE_LOCK),actual_stage_owner=pointer(report.RUNTIME+'/terminal.json'))
    jobs=[];clean=dict(complete=True,returncode=0,timed_out=False,interrupted=False,automatic_retry=False)
    for (task,raw),row_sha in sources.items():
        for method in report.METHODS:
            bindings=dict(admission_driver=driver,admission_guard=guard,case_worker=worker,native_runtime=protocol['runtime_consumer'],
                original_native_runtime=protocol['original_runtime'],author_outcome_protocol=protocol)
            route=next(r for r in routes['routes'] if (r['task'],r['method'])==(task,method));out=report.ART+'/test/'+'/'.join((task,raw,method))
            argv=['bash','scripts/run_aligned_gpu.sh' if method=='physmani' else 'scripts/run_local_sim.sh',driver['path'],
                '--config',route['configuration']['path'],'--frozen-inputs',route['frozen_inputs']['path'],
                '--manifest',report.MANIFEST,*common,'--task',task,'--raw-episode-id',raw,'--method',method,'--out',out]
            row=dict(task=task,raw=raw,method=method,source_row_sha256=row_sha,job_id='_'.join((task,raw,method)),argv=argv,
                worker_sha256=driver['sha256'],configuration_sha256='c'*64,input_lock_sha256='d'*64,**bindings);jobs.append(row)
            case=test_case(row,protocol,pointer(report.SOURCE_LOCK)['sha256']);case_name=out+'/case.json';driver_name=out+'/admission_driver_receipt.json'
            data[case_name]=case;case_pointer=pointer(case_name)
            data[driver_name]=dict(schema='r10_explicit_author_outcome_formal_driver_v751',complete=True,actual_split='test',
                execution_protocol_changed=True,case=case_pointer,**bindings)
            driver_pointer=pointer(driver_name);result['cases'].append(case_pointer);result['admission_driver_receipts'].append(driver_pointer)
            for kind in ('CPU','native'):
                check=kind=='CPU';leaf=report.RUNTIME+'/'+kind+'/'+row['job_id']
                data[leaf+'/launch.json']=dict(argv=argv+(['--check'] if check else []),check_only=check,owner_pid=700,
                    native_slots=0 if check else 1,GPU_slots=0 if check else int(method=='physmani'),execution_protocol_changed=True,
                    expected_identity_sha256=None if check else case['identity']['sha256'],**bindings)
                data[leaf+'/terminal.json']=dict(clean,**({} if check else dict(case=case_pointer,admission_driver_receipt=driver_pointer)))
                if check:
                    cpu=dict(complete=True,native_started=False,GPU_started=False,actual_split='test',execution_protocol_changed=True,
                        identity=case['identity'],component_domain=case['component_domain'],**bindings)
                    data[leaf+'/CPU_preflight.json']=cpu;data[leaf+'/stdout.log']=deepcopy(cpu)
    data[report.RUNTIME+'/plan.json']=dict(split='test',expected_cases=1920,policy_sha256=report.POLICY,
        source_outcome_filtering=False,automatic_retry=False,jobs=jobs)
    data[report.RESULT]=result
    data[report.AUDIT_STAGE+'/stdout.log']=dict(complete=True,split='test',cases=1920,TEST_permitted=False,out=report.RESULT,
        native_successes_by_method=result['native_successes_by_method'])
    data[report.SUBSET]=dict(metric_definitions={},statistical_definition={},diagnostic_definition={},hard_budget_disclosure={})
    evidence_names=[report.SUITE,report.SUITE_LOCK,report.SOURCE_LOCK,report.RUNTIME+'/launch.json',
        report.RUNTIME+'/plan.json',report.RUNTIME+'/terminal.json']
    for row in jobs:
        name=row['job_id']
        evidence_names.extend([report.RUNTIME+'/CPU/'+name+'/launch.json',report.RUNTIME+'/native/'+name+'/launch.json',
            report.RUNTIME+'/CPU/'+name+'/CPU_preflight.json',report.RUNTIME+'/native/'+name+'/terminal.json',
            report.option(row['argv'],'--out')+'/admission_driver_receipt.json'])
    result['inputs_sha256']={str(PROJECT/name):pointer(name)['sha256'] for name in evidence_names}
    result['inputs_sha256'].update({str(PROJECT/name):source(name)['sha256'] for name in
        (driver['path'],guard['path'],report.AUDITOR,report.OWNER)})
    class FakeReader:
        root=PROJECT
        def __init__(self):self.inputs={};self.evidence={};self.frozen={};self.original={};self.opened=[];self.data=data
        def raw(self,name,**kwargs):
            self.opened.append(name)
            if name==report.REDUCER:
                raw=(PROJECT/name).read_bytes();self.inputs[name]=report.sha_bytes(raw);return raw
            self.inputs[name]=source(name)['sha256'];return ('CPU-fake-source:'+name).encode()
        def json(self,name,**kwargs):
            self.opened.append(name);actual=pointer(name)['sha256']
            if kwargs.get('evidence'):
                report.require(self.lookup(self.evidence,name)==actual,'input SHA changed: '+name)
            self.inputs[name]=actual;return self.data[name]
        def pointer(self,name):return dict(path=name,sha256=self.inputs[name])
        def pointed(self,p,**kwargs):return self.json(p['path'],**kwargs)
        lookup=report.Reader.lookup
    return FakeReader()


class FormalTESTMotionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.reducer=report.H.load_reducer(report.Reader(PROJECT))

    def test_header_matches754_shape_without_VAL_schema_or_TEST_used(self):
        result=result_fixture();result['cases']=[{}]*1920;result['admission_driver_receipts']=[{}]*1920
        report.require_result_header(result)
        result['completed_method_cases']=1919
        with self.assertRaisesRegex(ValueError,'complete16/640/1920'):report.require_result_header(result)

    def test_incomplete_results_refused_before_other_inputs(self):
        class OnlyHeader:
            def __init__(self):self.calls=[]
            def json(self,name,**kwargs):self.calls.append(name);return dict(complete=False)
        reader=OnlyHeader()
        with self.assertRaises(ValueError):report.collect(reader,'a'*64)
        self.assertEqual(reader.calls,[report.RESULT])

    def test_unfinished827_refused_before_any_TEST_setup_or_source_metadata(self):
        reader=memory_fixture(self.reducer);reader.data[report.CONTROLLER+'/terminal.json']['complete']=False
        with self.assertRaisesRegex(ValueError,'full827 TEST completion'):report.collect(reader,'a'*64)
        self.assertNotIn(report.SUITE,reader.opened);self.assertNotIn(report.SOURCE_LOCK,reader.opened);self.assertNotIn(report.MANIFEST,reader.opened)

    def test_original640_metadata_without_split_and_variation_local_indices(self):
        pointer=dict(path='fake-VAL-release',sha256='a'*64);manifest,original,lock=source_fixture(pointer,'b'*64)
        self.assertEqual(len(report.source_grid(manifest,original,lock,pointer,'b'*64)),640)
        manifest['episodes'][1]['official_episode_index']=0
        with self.assertRaisesRegex(ValueError,'variation-local'):report.source_grid(manifest,original,lock,pointer,'b'*64)

    def test_incomplete753_owner_refused_before_TEST_metadata_even_with_clean_stage(self):
        reader=memory_fixture(self.reducer);name=report.RUNTIME+'/terminal.json';reader.data[name]['complete']=False
        checksum=report.digest(reader.data[name]);reader.data[report.RESULT]['inputs_sha256'][str(PROJECT/name)]=checksum
        reader.data[report.RESULT]['actual_stage_owner']=dict(path=name,sha256=checksum)
        with self.assertRaisesRegex(ValueError,'owner required before TEST metadata'):report.collect(reader,'a'*64)
        self.assertNotIn(report.SUITE,reader.opened);self.assertNotIn(report.SOURCE_LOCK,reader.opened);self.assertNotIn(report.MANIFEST,reader.opened)

    def test_full_plan_rejects_duplicate_method_or_VAL_import(self):
        reader=memory_fixture(self.reducer);d=reader.data;plan=d[report.RUNTIME+'/plan.json']
        sources=report.source_grid(d[report.MANIFEST],d[report.ORIGINAL_MANIFEST],d[report.SOURCE_LOCK],
            d[report.MANIFEST]['complete_unchanged_VAL_release_observed_before_source_selection'],source(report.SOURCE_PREPARER)['sha256'])
        report.require_plan(plan,sources,d[report.ROUTES],d[report.SUITE]['author_outcome_protocol'])
        changed=deepcopy(plan);changed['jobs'][-1]=changed['jobs'][-2]
        with self.assertRaisesRegex(ValueError,'unique original'):report.require_plan(changed,sources,d[report.ROUTES],d[report.SUITE]['author_outcome_protocol'])
        plan['jobs'][0]['imported_completed']=True
        with self.assertRaisesRegex(ValueError,'fresh753'):report.require_plan(plan,sources,d[report.ROUTES],d[report.SUITE]['author_outcome_protocol'])

    def test_original_TEST_identity_native_false_motion_false_and_zero_ticks(self):
        reader=memory_fixture(self.reducer);row=reader.data[report.RUNTIME+'/plan.json']['jobs'][0]
        case=reader.data[report.option(row['argv'],'--out')+'/case.json'];protocol=row['author_outcome_protocol'];lock_sha=case['identity']['source_input_lock_sha256']
        report.require_case(case,row,row['source_row_sha256'],protocol,lock_sha)
        metrics,_=self.reducer.reduce_saved_case(case);self.assertIsNone(metrics['peak_window_joint_acceleration_rad_s2'])
        case['identity']['raw_episode_id']+='__policy_aug_r0'
        with self.assertRaisesRegex(ValueError,'original TEST source'):report.require_case(case,row,row['source_row_sha256'],protocol,lock_sha)

    def test1920_case_driver_bijection_rejects_duplicates(self):
        reader=memory_fixture(self.reducer);d=reader.data;result=d[report.RESULT];rows=d[report.RUNTIME+'/plan.json']['jobs']
        report.require_pointers(rows,result['cases'],result['admission_driver_receipts'])
        result['cases'][-1]=result['cases'][-2]
        with self.assertRaisesRegex(ValueError,'unique ordered'):report.require_pointers(rows,result['cases'],result['admission_driver_receipts'])

    def test_memory_only_full_collect_retains_all1920_native_false(self):
        reader=memory_fixture(self.reducer);rows,_,_,_=report.collect(reader,'a'*64)
        self.assertEqual(len(rows),1920);self.assertTrue(all(r['official_task_success'] is False for r in rows))
        self.assertTrue(all(r['recorded_window_diagnostics']['combined_window_checks_passed'] is False for r in rows))
        self.assertTrue(all(r['metrics']['peak_window_joint_acceleration_rad_s2'] is None for r in rows))
        self.assertTrue(any(p.endswith('/CPU_preflight.json') for p in reader.opened))

    def test_mismatched_stage_namespace_or_native_CPU_identity_rejected(self):
        reader=memory_fixture(self.reducer);reader.data[report.AUDIT_STAGE+'/launch.json']['current_mount_namespace']='mnt:[different]'
        with self.assertRaisesRegex(ValueError,'stage argv/PID/inherited'):report.collect(reader,'a'*64)
        reader=memory_fixture(self.reducer);row=reader.data[report.RUNTIME+'/plan.json']['jobs'][0]
        reader.data[report.RUNTIME+'/native/'+row['job_id']+'/launch.json']['expected_identity_sha256']='f'*64
        with self.assertRaisesRegex(ValueError,'input SHA changed'):report.collect(reader,'a'*64)

    def test_raw_reset_TRAIN_and_model_reads_refused_before_open(self):
        with tempfile.TemporaryDirectory() as name:
            reader=report.Reader(Path(name))
            for path in ('/UNOPENED_TEST/a/first.pkl','artifacts/framework/protocol/train_manifest.json','artifacts/models/weights.pt','images/test.png'):
                with self.subTest(path=path),self.assertRaisesRegex(ValueError,'non-evidence'):reader.raw(path)
            self.assertEqual(reader.inputs,{report.HELPER:report.HELPER_SHA})

    def test_symlink_alias_refused_before_open(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);target=root/'unopened-reset.pkl';target.write_text('CPU fake only')
            alias=root/report.MANIFEST;alias.parent.mkdir(parents=True);alias.symlink_to(target)
            with self.assertRaisesRegex(ValueError,'symlink aliases'):report.Reader(root).raw(report.MANIFEST)


if __name__=='__main__':unittest.main()
