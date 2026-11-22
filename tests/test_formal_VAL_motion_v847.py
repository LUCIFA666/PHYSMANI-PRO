"""Pure fake full-grid and release boundaries; never run847 main or real cases."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import tempfile
import unittest

PROJECT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('formal_VAL_motion_v847_test_subject',PROJECT/'scripts/summarize_formal_VAL_motion_v847_r10.py')
report=importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


def metadata_fixture():
    rows=[dict(task=task,raw_episode_id='episode'+str(i),path='/unopened/VAL/'+task+'/'+str(i)+'.h5')
          for task in report.TASKS for i in range(report.COUNTS[task])]
    manifest=dict(split='validation',episodes=rows)
    source_lock=dict(split='validation',source_count=315,policy_sha256=report.POLICY,
        source_manifest_sha256=report.PINS[report.SOURCE_MANIFEST],TEST_used_for_fitting=False,
        files={row['path']:'a'*64 for row in rows})
    return manifest,source_lock


def release_fixture():
    value=dict(schema='r10_full16_common_author_RML_validation_release_v531',complete=True,split='validation',
        source_cases=315,method_cases=945,completed_method_cases=945,methods=list(report.METHODS),tasks=list(report.TASKS),
        policy_sha256=report.POLICY,common_RML_profile_sha256=report.PROFILE,
        formal_test_permitted=True,TEST_used=False,TEST_used_for_fitting=False,motion_is_diagnostic_only=True,
        motion_metrics_required_for_all_methods=True,native_task_success_unchanged=True,
        native_success_is_not_a_release_threshold=True,source_outcome_filtering=False,native_failures_retained=True,
        Value_NGFF_FASTER_used=False,physical_policy_changed=False,ours_measured_motion_passed=False,
        native_successes_by_method={method:0 for method in report.METHODS},
        cases=[dict(path='fake/'+str(i)+'/case.json',sha256='a'*64) for i in range(945)],
        admission_driver_receipts=[dict(path='fake/'+str(i)+'/admission_driver_receipt.json',sha256='b'*64) for i in range(945)],
        inputs_sha256={'fake':'a'*64})
    return value


def plan_fixture():
    manifest,lock=metadata_fixture();sources=report.source_grid(manifest,lock)
    previous=[];current=[];completed=[];remaining=[]
    stages=list(report.STAGES)
    for task,raw in sources:
        for method in report.METHODS:
            index=len(previous);key=(task,raw,method);name='_'.join(key)
            stage=stages[0] if index<95 else stages[1]
            out=report.STAGES[stage][0]+'/validation/'+'/'.join(key)
            old=dict(task=task,raw=raw,method=method,job_id=name,source_row_sha256=sources[key[:2]],
                evidence_runtime=stage,imported_completed=index<95,argv=['fake-only','--out',out])
            row=deepcopy(old)
            if index<591:
                completed.append(name);row['imported_completed']=True
            else:
                remaining.append(name);row['evidence_runtime']=report.CURRENT;row['imported_completed']=False
                row['argv'][-1]=report.STAGES[report.CURRENT][0]+'/validation/'+'/'.join(key)
            row.update(previous_row_sha256=report.digest(old),previous_imported_completed=old['imported_completed'])
            previous.append(old);current.append(row)
    plan=dict(split='validation',expected_cases=945,policy_sha256=report.POLICY,source_outcome_filtering=False,
              automatic_retry=False,jobs=current)
    return plan,dict(jobs=previous),dict(completed_job_ids=completed,remaining_job_ids=remaining),sources


def zero_tick_case():
    task=report.TASKS[0];raw='episode0';method='cv_geometry';sha='e'*64;protocol=dict(schema='CPU-fake-protocol')
    row=dict(task=task,raw=raw,method=method,configuration_sha256='c'*64,input_lock_sha256='d'*64)
    identity=dict(split='validation',task=task,method=method,source_raw_episode_id=raw,parent_raw_episode_id=raw,
        raw_episode_id=raw+'__policy_aug_r0',replica=0,source_row_sha256=sha,config_sha256='c'*64,
        frozen_inputs_sha256='d'*64,source_input_lock_sha256=report.PINS[report.SOURCE_LOCK])
    case=dict(complete=True,cleanup_completed=True,infrastructure_error=None,official_task_success=False,native_terminal_failure=True,
        split='validation',task=task,method=method,source_raw_episode_id=raw,identity=identity,
        demo_only=False,development_demo_only=False,formal_validation_metric=True,formal_TEST_metric=False,
        TEST_used_for_fitting=False,Value_NGFF_FASTER_used=False,author_outcome_protocol=protocol,
        component_domain=dict(purpose='evaluation',split='validation',task=task,method=method,raw_episode_id=raw),
        full16_frozen_execution=dict(policy_sha256=report.POLICY,source_row_sha256=sha,actual_split='validation',VAL_TEST_relabelled_as_TRAIN=False),
        common_author_RML_parameters=dict(complete=True,native_parameters_mutated=False,actual_motion_bound_claimed=False,
            profile_sha256=report.PROFILE,actual_parameters=report.PARAMETERS),
        all_stage_counted_joint_motion=dict(all_stage_actual_window_checks_passed=False,actual_window_acceleration_criterion_rad_s2=6.,
            planning_metadata_used_as_actual_acceleration_cap=False,policy_or_physics_modified=False,events=[],peak_actual_window_acceleration=0.),
        counted_clock=dict(ticks=0,dt=.05,seconds=0.),alignment_receipt=dict(actual_physics_seconds=0.,actual_task_seconds=0.,
            clock=dict(ticks=0,dt=.05,budget_seconds=10.,simulation_seconds=0.,phases={})),
        actor_metrics=dict(physmani_geometry_executor=dict(actions=[])))
    return case,row,sha,protocol


def in_memory_collect_fixture(reducer):
    """Shape integration only: fake storage, no files or real release are created."""
    data={};plan,previous,partition,sources=plan_fixture()
    def code(name):return b'CPU-only-fake-source:'+name.encode()
    def source(name):return dict(path=name,sha256=report.sha_bytes(code(name)))
    def pointer(name):
        checksum=report.PINS.get(name,report.digest(data[name]))
        return dict(path=name,sha256=checksum)
    protocol=dict(schema='in-memory-CPU-protocol',implementation=source('our_mani/multitask/native_outcome_sampling_v741.py'),
        runtime_consumer=source('scripts/run_task_executor_smoke_author_suite16_v744_r10.py'),
        original_runtime=source('scripts/run_task_executor_smoke_rml_suite16_v664_r10.py'))
    control=dict(schema='r10_frozen_environment_formal_VAL_resume_control_v823',expected_cases=945,policy_sha256=report.POLICY,
        physical_policy_changed=False,completed_results_selected_by_success=False,author_outcome_protocol=protocol,
        resume_auditor=source(report.AUDITOR),**partition,
        environment_manifest=dict(path='fake-environment-manifest',sha256='a'*64),
        environment_wrapper=dict(path='fake-environment-wrapper',sha256='b'*64),
        environment_namespace_proof=dict(path='fake-environment-proof',sha256='c'*64))
    def environment(pid):
        return dict(schema='r10_actual_frozen_OS_namespace_receipt_v820',complete=True,checked_OS_files=6418,
            all_current_OS_targets_full_byte_SHA_verified=True,host_PID_network_namespace_preserved=True,proc_or_dev_replaced=False,
            task_parameters_or_strategy_changed=False,current_mount_namespace='mnt:[2]',mount_namespace='mnt:[2]',reference_mount_namespace='mnt:[1]',
            manifest=control['environment_manifest'],wrapper=control['environment_wrapper'],namespace_qualification=control['environment_namespace_proof'],
            observer_pid=pid,pid=pid,observer_start_ticks=pid+1,outer_wrapper_pid=10,outer_wrapper_start_ticks=11)
    bindings=dict(admission_driver=source('scripts/run_formal_case_author_outcome_v751_r10.py'),
        admission_guard=source('our_mani/multitask/formal_suite_admission_v728.py'),
        case_worker=source('scripts/run_task_executor_smoke_author_outcome_v743_r10.py'),native_runtime=protocol['runtime_consumer'],
        original_native_runtime=protocol['original_runtime'],author_outcome_protocol=protocol)
    owner_pids=dict(zip(report.STAGES,(500,600,700)))
    for row,old in zip(plan['jobs'],previous['jobs']):
        for item in (row,old):
            out=report.option(item['argv'],'--out')
            item.update(bindings,configuration_sha256='c'*64,input_lock_sha256='d'*64)
            item['argv']=['bash','scripts/run_aligned_gpu.sh' if item['method']=='physmani' else 'scripts/run_local_sim.sh',
                bindings['admission_driver']['path'],'--config','fake-VAL-config','--frozen-inputs','fake-frozen-inputs',
                '--split','validation','--task',item['task'],'--raw-episode-id',item['raw'],'--method',item['method'],
                '--manifest',report.SOURCE_MANIFEST,'--suite',report.SUITE,'--suite-lock',report.SUITE_LOCK,
                '--source-inputs',report.SOURCE_LOCK,'--out',out]
        row['previous_row_sha256']=report.digest(old)
    for stage,(_,script) in report.STAGES.items():
        data[stage+'/launch.json']=dict(source_sha256=source(script)['sha256'],split='validation',policy_sha256=report.POLICY,
            author_outcome_protocol=protocol,pid=owner_pids[stage])
        if stage==report.CURRENT:data[stage+'/launch.json']['frozen_environment']=environment(owner_pids[stage])
    old_runtime=list(report.STAGES)[1]
    data[old_runtime+'/plan.json']=previous
    control['previous_plan']=pointer(old_runtime+'/plan.json')
    data[old_runtime+'/terminal.json']=dict(complete=False)
    control['previous_terminal']=pointer(old_runtime+'/terminal.json')
    data[report.CONTROL]=control
    data[report.LOCK]=dict(schema='r10_frozen_environment_formal_VAL_resume_inputs_v823',approved_for_execution=True,files={})
    data[report.SUITE]=dict(formal_result_auditor=source('scripts/audit_formal_suite16_results_v754_r10.py'),
        formal_case_driver=bindings['admission_driver'],formal_admission_guard=bindings['admission_guard'],author_outcome_protocol=protocol)
    data[report.SOURCE_MANIFEST],data[report.SOURCE_LOCK]=metadata_fixture()
    data[report.CURRENT+'/plan.json']=plan
    data[report.CURRENT+'/terminal.json']=dict(complete=True,completed=945,expected=945,failed=0,split='validation',
        policy_sha256=report.POLICY,stop_requested=False,automatic_retry=False,frozen_environment=environment(owner_pids[report.CURRENT]))
    data[report.CONTROLLER+'/launch.json']=dict(source_sha256=source(report.CONTROLLER_SOURCE)['sha256'],pid=400,
        resume_control=pointer(report.CONTROL),resume_control_lock=pointer(report.LOCK),frozen_environment=environment(400))
    data[report.AUDIT_STAGE+'/launch.json']=dict(source_sha256=source(report.AUDITOR)['sha256'],pid=800,start_ticks=801,automatic_retry=False,
        current_mount_namespace='mnt:[2]',inherited_frozen_environment=environment(400),
        argv=[str(PROJECT/'.conda-env/bin/python'),'-u',report.AUDITOR,'--suite',report.SUITE,'--suite-lock',report.SUITE_LOCK,
            '--source-inputs',report.SOURCE_LOCK,'--split','validation','--runtime',report.CURRENT,
            '--resume-control',report.CONTROL,'--resume-lock',report.LOCK,'--out',report.RELEASE])
    data[report.AUDIT_STAGE+'/terminal.json']=dict(complete=True,returncode=0,automatic_retry=False)
    data[report.AUDIT_STAGE+'/stdout.log']=dict(complete=True,split='validation',cases=945,imported=591,new_native=354,TEST_permitted=True,out=report.RELEASE)
    release=release_fixture()
    release.update(source_sha256=source(report.AUDITOR)['sha256'],resume_control=pointer(report.CONTROL),resume_control_lock=pointer(report.LOCK),
        author_outcome_protocol=protocol,base_result_auditor=data[report.SUITE]['formal_result_auditor'],
        admission_driver=bindings['admission_driver'],admission_guard=bindings['admission_guard'],actual_auditor_environment=environment(800),
        source_input_lock=pointer(report.SOURCE_LOCK),provenance_runtime_counts={str(PROJECT/k):v for k,v in report.EXPECTED_PROVENANCE.items()},
        imported_completed_cases=591,newly_completed_cases=354,actual_stage_owner=pointer(report.CURRENT+'/terminal.json'),
        previous_stage_owner=pointer(old_runtime+'/terminal.json'),cases=[],admission_driver_receipts=[])
    subset=[]
    clean=dict(complete=True,returncode=0,automatic_retry=False,timed_out=False,interrupted=False)
    for row in plan['jobs']:
        case,_,_,_=zero_tick_case();task,raw,method=row['task'],row['raw'],row['method']
        case.update(task=task,method=method,source_raw_episode_id=raw,author_outcome_protocol=protocol)
        case['identity'].update(task=task,method=method,source_raw_episode_id=raw,parent_raw_episode_id=raw,
            raw_episode_id=raw+'__policy_aug_r0',source_row_sha256=row['source_row_sha256'])
        case['identity']['sha256']=report.digest(case['identity'])
        case['component_domain'].update(task=task,method=method,raw_episode_id=raw)
        case['full16_frozen_execution']['source_row_sha256']=row['source_row_sha256']
        if method=='physmani':case['actor_metrics']=dict(baseline_steps=[])
        out=report.option(row['argv'],'--out');case_name=out+'/case.json';driver_name=out+'/admission_driver_receipt.json'
        data[case_name]=case;case_pointer=pointer(case_name)
        data[driver_name]=dict(schema='r10_explicit_author_outcome_formal_driver_v751',complete=True,actual_split='validation',
            execution_protocol_changed=True,case=case_pointer,**bindings)
        driver_pointer=pointer(driver_name);release['cases'].append(case_pointer);release['admission_driver_receipts'].append(driver_pointer)
        stage=row['evidence_runtime'];name=row['job_id']
        for kind in ('CPU','native'):
            leaf=stage+'/'+kind+'/'+name;check=kind=='CPU'
            launch=dict(argv=row['argv']+(['--check'] if check else []),check_only=check,owner_pid=owner_pids[stage],
                native_slots=0 if check else 1,GPU_slots=0 if check else int(method=='physmani'),
                expected_identity_sha256=None if check else case['identity']['sha256'],**bindings)
            if stage==report.CURRENT:launch.update(inherited_frozen_environment=environment(owner_pids[stage]),current_mount_namespace='mnt:[2]')
            data[leaf+'/launch.json']=launch
            data[leaf+'/terminal.json']=dict(clean,**({} if check else dict(case=case_pointer,admission_driver_receipt=driver_pointer)))
            if check:data[leaf+'/CPU_preflight.json']=dict(complete=True,native_started=False,GPU_started=False,actual_split='validation',
                execution_protocol_changed=True,identity=case['identity'],component_domain=case['component_domain'],**bindings)
        metrics,diagnostics=reducer.reduce_saved_case(case)
        subset.append(dict(case=case_pointer,metrics=metrics,recorded_window_diagnostics=diagnostics))
    data[report.SUBSET]=dict(case_evidence=subset[:591])
    data[report.RELEASE]=release
    class FakeReader:
        root=PROJECT
        def __init__(self):self.inputs={};self.evidence={};self.frozen={}
        def json(self,name,**kwargs):
            self.inputs[name]=pointer(name)['sha256']
            return deepcopy(data[name])
        def raw(self,name,**kwargs):
            value=(PROJECT/name).read_bytes() if name==report.REDUCER else code(name)
            self.inputs[name]=report.sha_bytes(value)
            return value
        def pointer(self,name):return dict(path=name,sha256=self.inputs[name])
        def pointed(self,value,**kwargs):return self.json(value['path'],**kwargs)
    return FakeReader()


class FullVALMotionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Opens only the SHA-pinned Python reducer source, never any real data.
        cls.reducer=report.load_reducer(report.Reader(PROJECT))

    def test_complete_header_accepts_zero_success_and_motion_false(self):
        report.require_release_header(release_fixture())

    def test_incomplete_release_refused_before_other_evidence_reads(self):
        release=release_fixture();release['complete']=False
        class FakeReader:
            calls=[]
            def json(self,name,**kwargs):
                self.calls.append(name)
                if name!=report.RELEASE:raise AssertionError('must not read another input')
                return release
        reader=FakeReader()
        with self.assertRaisesRegex(ValueError,'real complete16'):
            report.collect(reader,'a'*64)
        self.assertEqual(reader.calls,[report.RELEASE])

    def test_original315_metadata_and_full945_three_tier_plan_pass(self):
        plan,old,control,sources=plan_fixture()
        rows,provenance=report.require_plan(plan,old,control,sources)
        self.assertEqual(len(sources),315)
        self.assertEqual(len(rows),945)
        self.assertEqual(provenance,report.EXPECTED_PROVENANCE)

    def test_duplicate_source_or_method_is_not_a_full_grid(self):
        manifest,lock=metadata_fixture();manifest['episodes'][-1]=manifest['episodes'][-2]
        with self.assertRaisesRegex(ValueError,'unique canonical'):
            report.source_grid(manifest,lock)
        plan,old,control,sources=plan_fixture()
        plan['jobs'][-1]=deepcopy(plan['jobs'][-2])
        with self.assertRaisesRegex(ValueError,'unique original'):
            report.require_plan(plan,old,control,sources)

    def test_changed_historical_provenance_is_rejected(self):
        plan,old,control,sources=plan_fixture()
        plan['jobs'][0]['previous_row_sha256']='f'*64
        with self.assertRaisesRegex(ValueError,'previous-row provenance'):
            report.require_plan(plan,old,control,sources)

    def test_release_case_driver_bijection_rejects_duplicate_or_wrong_tier(self):
        plan,_,_,_=plan_fixture();rows=plan['jobs']
        cases=[dict(path=report.option(r['argv'],'--out')+'/case.json',sha256='a'*64) for r in rows]
        drivers=[dict(path=report.option(r['argv'],'--out')+'/admission_driver_receipt.json',sha256='b'*64) for r in rows]
        report.require_case_pointers(rows,cases,drivers)
        cases[-1]=cases[-2]
        with self.assertRaisesRegex(ValueError,'unique case/driver'):
            report.require_case_pointers(rows,cases,drivers)

    def test827_terminal_uses_actual_writer_shape_without_extra_flags(self):
        terminal=dict(complete=True,returncode=0,automatic_retry=False,finished_utc='CPU-fixture')
        report.require_clean_terminal(terminal,controller_stage=True)
        with self.assertRaisesRegex(ValueError,'uninterrupted'):
            report.require_clean_terminal(terminal)
        terminal['returncode']=False
        with self.assertRaisesRegex(ValueError,'clean one-time'):
            report.require_clean_terminal(terminal,controller_stage=True)

    def test_zero_ticks_native_false_and_unmeasured_peaks_are_retained(self):
        case,row,source,protocol=zero_tick_case()
        report.require_case(case,row,source,protocol)
        metrics,diagnostics=self.reducer.reduce_saved_case(case)
        self.assertIsNone(metrics['peak_window_joint_speed_rad_s'])
        self.assertIsNone(metrics['peak_window_joint_acceleration_rad_s2'])
        records=[dict(task=t,method='cv_geometry',source_raw_episode_id='episode'+str(i),source_row_sha256='a'*64,
            official_task_success=False,metrics=metrics,recorded_window_diagnostics=diagnostics)
            for t in report.TASKS for i in range(report.COUNTS[t])]
        result=self.reducer.summarize(records)
        self.assertEqual(result['case_count'],315)
        self.assertEqual(result['native_false_cases'],315)
        self.assertEqual(result['native_success_cases'],0)
        self.assertEqual(result['metrics']['peak_window_joint_acceleration_rad_s2']['unknown_or_nonfinite_cases'],315)

    def test_changed_case_protocol_or_source_is_rejected(self):
        case,row,source,protocol=zero_tick_case();case['identity']['source_row_sha256']='wrong'
        with self.assertRaisesRegex(ValueError,'source/method/CPU'):
            report.require_case(case,row,source,protocol)
        case,row,source,protocol=zero_tick_case();case['common_author_RML_parameters']['actual_parameters']=dict(report.PARAMETERS,max_acceleration=6.)
        with self.assertRaisesRegex(ValueError,'common RML'):
            report.require_case(case,row,source,protocol)

    def test_TEST_TRAIN_or_model_paths_refused_before_open(self):
        with tempfile.TemporaryDirectory() as d:
            reader=report.Reader(Path(d))
            for name in ('artifacts/framework/multitask_r10/formal_VAL_env_resume_v823/test/a/case.json',
                         'artifacts/framework/protocol/train_manifest.json','artifacts/models/dynamics.pt'):
                with self.subTest(name=name),self.assertRaisesRegex(ValueError,'non-VAL or undeclared'):
                    reader.raw(name)
            self.assertEqual(reader.inputs,{})

    def test_memory_only_collect_keeps_all945_native_false_and_motion_false(self):
        reader=in_memory_collect_fixture(self.reducer)
        rows,_,_,_,provenance=report.collect(reader,'a'*64)
        self.assertEqual(len(rows),945)
        self.assertTrue(all(row['official_task_success'] is False for row in rows))
        self.assertTrue(all(row['recorded_window_diagnostics']['combined_window_checks_passed'] is False for row in rows))
        self.assertTrue(all(row['metrics']['peak_window_joint_acceleration_rad_s2'] is None for row in rows))
        self.assertEqual(provenance,report.EXPECTED_PROVENANCE)

    def test_symlink_cannot_alias_a_disallowed_file_into_allowed_input(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);target=root/'unopened_TEST_contents';target.write_text('CPU fixture only')
            alias=root/report.REDUCER;alias.parent.mkdir(parents=True);alias.symlink_to(target)
            reader=report.Reader(root)
            with self.assertRaisesRegex(ValueError,'symlink aliases are refused before open'):
                reader.raw(report.REDUCER)
            self.assertEqual(reader.inputs,{})


if __name__=='__main__': unittest.main()
