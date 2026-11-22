"""Controller unit boundaries only; no actual stage launches or process probes."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

path = Path(__file__).resolve().parents[1] / 'scripts/continue_composite_TEST_v1043_r10.py'
spec = importlib.util.spec_from_file_location('controller958_fake_tests', path)
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for name in (m.SOURCE, m.OWNER, m.AUDITOR):
            p = self.root / name; p.parent.mkdir(parents=True, exist_ok=True); p.write_text('# fake immutable source\n')
        self.root_patch = patch.object(m, 'ROOT', self.root); self.root_patch.start(); self.addCleanup(self.root_patch.stop)
        for name, value in (('FROZEN_CONTROL', None), ('CONTROL_BINDING', None), ('OWNED_OWNER', None)):
            p = patch.object(m, name, value); p.start(); self.addCleanup(p.stop)
        m.EVENT.clear()
        self.setup = self.root / 'artifacts/framework/multitask_r10/formal_TEST_composite_resume_v1041_setup'; self.setup.mkdir(parents=True)
        self.c = self.setup / 'resume_control.json'; self.l = self.setup / 'resume_inputs.json'
        self.control = dict(schema='r10_five_tier_composite_formal_TEST_resume_control_v1040', split='test', physical_policy_changed=False,
            resume_controller=m.point(m.SOURCE), resume_owner=m.point(m.OWNER), resume_auditor=m.point(m.AUDITOR),
            policy_sha256='a' * 64, source_inputs=dict(path='sources.json', sha256='b'*64),
            validation_release=dict(path='release.json', sha256='c'*64), resource_qualification=dict(path='qualification.json',sha256='d'*64),
            author_outcome_protocol={'original': True},resource_qualifier=dict(path='qualifier.py',sha256='f'*64),
            resource_profile={'schema':'CPU_actual12/5/6/24'},previous_resource_profile={'schema':'CPU_prior8/3/4/16'},
            resource_capacity_envelope={'schema':'CPU_capacity12/5/6/24'},resource_profile_within_capacity=True,resource_capacity_source='qualified1031',resource_profile_changed=True,
            previous_resource_qualification=dict(path='old904.json',sha256='e'*64),execution_implementation={'same901':True})
        self.control['result_auditor']=self.control['resume_auditor']
        m.CHILDREN=[]
        self.addCleanup(self.close_children)

    def close_children(self):
        for _,out,err in m.CHILDREN:out.close();err.close()

    def freeze(self):
        self.c.write_text(json.dumps(self.control))
        files = {str(self.c): m.point(self.c)['sha256']}
        files.update({str(self.root/name): m.point(name)['sha256'] for name in (m.SOURCE,m.OWNER,m.AUDITOR)})
        self.l.write_text(json.dumps(dict(schema='r10_five_tier_composite_formal_TEST_resume_inputs_v1040',approved_for_execution=True,control=m.point(self.c),files=files)))
        self.args=SimpleNamespace(resume_control=str(self.c),resume_lock=str(self.l),resume_lock_sha256=m.point(self.l)['sha256'])
        return m.load_small_control(self.args)

    def test_entry_is_inert_without_explicit_frozen_control_and_source(self):
        control,binding=self.freeze()
        self.assertEqual(control,self.control)
        self.assertEqual(binding['lock']['sha256'],self.args.resume_lock_sha256)
        self.assertFalse((self.root/m.RUNTIME).exists())

    def test_external_pin_rejects_before_JSON_parse(self):
        self.freeze();self.args.resume_lock_sha256='0'*64
        with patch.object(m.json,'loads',side_effect=AssertionError('must not parse mismatched bytes')):
            with self.assertRaises(ValueError):m.load_small_control(self.args)

    def test_changed_stage_source_is_rejected(self):
        self.freeze();(self.root/m.OWNER).write_text('# changed\n')
        with self.assertRaises(ValueError):m.load_small_control(self.args)

    def test_old902_control_and_lock_cannot_activate_new958(self):
        self.control['schema']='r10_concurrent_resumed_formal_TEST_control_v902'
        with self.assertRaises(ValueError):self.freeze()
        self.control['schema']='r10_five_tier_composite_formal_TEST_resume_control_v1040'
        self.freeze()
        old=json.loads(self.l.read_text())
        old['schema']='r10_concurrent_resumed_formal_TEST_inputs_v902'
        self.l.write_text(json.dumps(old));self.args.resume_lock_sha256=m.point(self.l)['sha256']
        with self.assertRaises(ValueError):m.load_small_control(self.args)

    def test_alias_declarations_cannot_hide_duplicate_stage_source(self):
        self.freeze();record=json.loads(self.l.read_text())
        record['files'][m.SOURCE]=m.point(m.SOURCE)['sha256']
        self.l.write_text(json.dumps(record));self.args.resume_lock_sha256=m.point(self.l)['sha256']
        with self.assertRaises(ValueError):m.load_small_control(self.args)

    def test_stage_drift_is_rejected_before_Popen_or_new_attempt_directory(self):
        control,binding=self.freeze();m.FROZEN_CONTROL=control;m.CONTROL_BINDING=binding
        (self.root/m.OWNER).write_text('# changed after control admission\n')
        with patch.object(m.subprocess,'Popen',side_effect=AssertionError('must not launch')):
            with self.assertRaises(ValueError):m.launch('formal_TEST_1920',m.OWNER,[])
        self.assertFalse((self.root/m.RUNTIME).exists())

    def test_old906_result_is_not_accepted_as_new_three_tier_release(self):
        control,binding=self.freeze()
        with self.assertRaises(ValueError):
            m.require_final_results({'schema':'r10_composite_concurrent_resumed_formal_TEST_results_v906'},control,binding)

    def test_stage_boundary_rechecks_source_control_and_lock(self):
        control,binding=self.freeze();m.FROZEN_CONTROL=control;m.CONTROL_BINDING=binding
        m.recheck_stage_inputs(m.OWNER)
        for name in (m.OWNER,m.AUDITOR,m.SOURCE):
            original=(self.root/name).read_text();(self.root/name).write_text('# drift\n')
            with self.subTest(name=name),self.assertRaises(ValueError):m.recheck_stage_inputs(m.AUDITOR)
            (self.root/name).write_text(original)
        self.c.write_text('{}')
        with self.assertRaises(ValueError):m.recheck_stage_inputs(m.AUDITOR)

    def test_child_scope_paths_are_canonical_relative_even_for_absolute_entry_args(self):
        control,binding=self.freeze()
        for key in ('suite','suite_lock'):control[key]=dict(path=key+'.json',sha256='e'*64)
        argv=m.owner_arguments(control,self.args)
        self.assertEqual(argv[argv.index('--resume-control')+1],binding['control']['path'])
        self.assertEqual(argv[argv.index('--resume-lock')+1],binding['lock']['path'])
        self.assertEqual(m.auditor_arguments(self.args)[1],binding['control']['path'])

    def test_foreign_or_unstarted_owner_never_receives_DRAIN(self):
        control,binding=self.freeze();m.FROZEN_CONTROL=control
        leaf=self.root/m.OWNER_RUNTIME;leaf.mkdir(parents=True)
        record=dict(pid=123,start_ticks=456,source_sha256=control['resume_owner']['sha256'])
        (leaf/'launch.json').write_text(json.dumps(record))
        self.assertFalse(m.request_drain('test'))
        m.OWNED_OWNER=dict(pid=999,start_ticks=456)
        self.assertFalse(m.request_drain('test'))
        self.assertFalse((leaf/'DRAIN').exists())

    def test_exact_owned_PID_start_receives_only_one_natural_DRAIN(self):
        control,binding=self.freeze();m.FROZEN_CONTROL=control
        leaf=self.root/m.OWNER_RUNTIME;leaf.mkdir(parents=True)
        m.OWNED_OWNER=dict(pid=123,start_ticks=456)
        (leaf/'launch.json').write_text(json.dumps(dict(m.OWNED_OWNER,source_sha256=control['resume_owner']['sha256'])))
        self.assertTrue(m.request_drain('concrete failure'));before=(leaf/'DRAIN').read_bytes()
        self.assertTrue(m.request_drain('second failure'))
        self.assertEqual((leaf/'DRAIN').read_bytes(),before)
        self.assertFalse((leaf/'STOP').exists())

    def test_owner_errors_trigger_concrete_failure_without_task_success_threshold(self):
        self.assertIsNone(m.failure_from_status(dict(failed={},owner_error=None,native_successes=0)))
        self.assertIsNotNone(m.failure_from_status(dict(failed={'case':{'error':'bad'}},owner_error=None)))
        self.assertIsNotNone(m.failure_from_status(dict(failed={},owner_error='source changed')))

    def test_all_STOP_paths_are_checked_without_writing_to_other_work(self):
        for name in m.STOP_PATHS:
            p=self.root/name;p.parent.mkdir(parents=True,exist_ok=True);p.touch()
            with self.subTest(name=name),self.assertRaises(InterruptedError):m.guard()
            p.unlink()

    def test_final_result_must_bind_exact_independent_full_cohort(self):
        from our_mani.multitask.evaluation_roster16_v283 import TASKS,METHODS
        control,binding=self.freeze()
        result=self.complete_result(control,binding)
        m.require_final_results(result,control,binding)
        for key in ('source_sha256','resume_control','resume_control_lock','policy_sha256','source_input_lock',
                    'author_outcome_protocol','validation_release','resource_qualification','resource_qualifier','completed_method_cases','methods','tasks','split',
                    'resource_capacity_envelope','resource_profile_within_capacity','resource_capacity_source',
                    'resource_profile_changed','previous_resource_profile','actual_stage_owners','evidence_tiers'):
            bad=deepcopy(result);bad[key]=None
            with self.subTest(key=key),self.assertRaises(ValueError):m.require_final_results(bad,control,binding)

    def test_real_fallback_same_actual_quota_is_not_mislabeled_as_resource_change(self):
        control,binding=self.freeze()
        control['resource_profile']=deepcopy(control['previous_resource_profile'])
        control['resource_capacity_envelope']=deepcopy(control['resource_profile'])
        control['resource_capacity_source']='qualified1005_fallback'
        control['resource_profile_changed']=False
        control['resource_qualification']=dict(path='runtime/framework/concurrent_TRAIN_resource_qualification_v1005.json',sha256='c'*64)
        result=self.complete_result(control,binding)
        m.require_final_results(result,control,binding)
        bad=deepcopy(result);bad['resource_profile_changed']=True
        with self.assertRaises(ValueError):m.require_final_results(bad,control,binding)
        bad=deepcopy(result);bad['schema']='r10_four_tier_composite_formal_TEST_results_v1012'
        with self.assertRaises(ValueError):m.require_final_results(bad,control,binding)

    def complete_result(self,control,binding):
        from our_mani.multitask.evaluation_roster16_v283 import TASKS,METHODS
        result=dict(schema='r10_five_tier_composite_formal_TEST_results_v1042',complete=True,split='test',
            source_cases=640,method_cases=1920,completed_method_cases=1920,methods=list(METHODS),tasks=list(TASKS),
            source_sha256=control['resume_auditor']['sha256'],resume_control=binding['control'],resume_control_lock=binding['lock'],
            policy_sha256=control['policy_sha256'],source_input_lock=control['source_inputs'],author_outcome_protocol=control['author_outcome_protocol'],
            validation_release=control['validation_release'],resource_qualification=control['resource_qualification'],resource_qualifier=control['resource_qualifier'],
            native_failures_retained=True,Value_NGFF_FASTER_used=False,native_successes_by_method={x:0 for x in METHODS})
        result.update(resource_profile=control['resource_profile'],previous_resource_profile=control['previous_resource_profile'],
            previous_resource_qualification=control['previous_resource_qualification'],resource_profile_changed=control['resource_profile_changed'],
            resource_capacity_envelope=control['resource_capacity_envelope'],resource_profile_within_capacity=True,resource_capacity_source=control['resource_capacity_source'],
            execution_implementation=control['execution_implementation'],independent_composite_audit=True,used_evidence_rechecked_at_close=True,
            source_outcome_filtering=False,motion_unknown_retained=True,zero_tick_cases_retained=True,motion_is_diagnostic_only=True,automatic_retry=False)
        runtimes=('runtime/framework/formal_TEST_after_environment_resume_v753','runtime/framework/formal_TEST_concurrent_resume_v902',
            'runtime/framework/formal_TEST_composite_resume_v956','runtime/framework/formal_TEST_composite_resume_v1011',m.OWNER_RUNTIME)
        counts=(134,450,616,200,520);owners=[];tiers=[]
        for runtime,count in zip(runtimes,counts):
            owners.append(dict(runtime=runtime,completed=count))
            for i in range(count):tiers.append(dict(job_id='CPU_'+str(len(tiers)),evidence_runtime=runtime))
        result.update(actual_stage_owners=owners,evidence_tiers=tiers,cases=[{}]*1920,admission_driver_receipts=[{}]*1920)
        return result

    def child(self,*,poll=None,after_wait=None):
        from unittest.mock import Mock
        child=Mock(pid=50001);child.poll.side_effect=poll or [0]
        def wait(**_):
            if after_wait:after_wait()
            return 0
        child.wait.side_effect=wait
        return child

    def test_Popen_is_never_called_before_STOP_or_small_source_gate(self):
        self.freeze();stop=self.root/m.STOP_PATHS[0];stop.parent.mkdir(parents=True,exist_ok=True);stop.touch()
        with patch.object(m.subprocess,'Popen') as popen,self.assertRaises(InterruptedError):m.launch('formal_TEST_1920',m.OWNER,[])
        popen.assert_not_called();self.assertFalse((self.root/m.RUNTIME).exists())

    def test_Popen_success_then_start_tick_or_launch_receipt_error_reaps_exact_own_child(self):
        from unittest.mock import Mock
        control,binding=self.freeze();m.FROZEN_CONTROL=control;m.CONTROL_BINDING=binding;m.ENVIRONMENT={'current_mount_namespace':'mnt:[fake]'}
        child=self.child(poll=[None]);child.terminate=Mock()
        with patch.object(m.os,'readlink',return_value='mnt:[fake]'),patch.object(m.subprocess,'Popen',return_value=child),patch.object(m,'ticks',side_effect=OSError('CPU stat capture failure')),self.assertRaises(OSError):
            m.launch('formal_TEST_1920',m.OWNER,[])
        child.terminate.assert_called_once();child.wait.assert_called_once()
        self.assertFalse((self.root/m.OWNER_RUNTIME/'STOP').exists())

    def test_STOP_after_child_wait_cannot_write_clean_terminal_or_advance_stage(self):
        child=self.child(after_wait=m.EVENT.set);leaf=self.root/'CPU/ownstage';leaf.mkdir(parents=True)
        try:
            with patch.object(m,'launch',return_value=(child,leaf)),self.assertRaises(InterruptedError):m.run('independent_TEST_results',m.AUDITOR,[])
            self.assertFalse((leaf/'terminal.json').exists())
        finally:m.EVENT.clear()

    def test_nonzero_child_or_infrastructure_failure_has_no_retry(self):
        child=self.child();child.wait.side_effect=None;child.wait.return_value=1
        leaf=self.root/'CPU/failedstage';leaf.mkdir(parents=True)
        with patch.object(m,'launch',return_value=(child,leaf)) as launch,self.assertRaises(ValueError):m.run('independent_TEST_results',m.AUDITOR,[])
        launch.assert_called_once();self.assertFalse(json.loads((leaf/'terminal.json').read_bytes())['complete'])

    def test_final_result_all_False_is_allowed_but_old_three_tier_or_wrong_owner_partition_rejected(self):
        control,binding=self.freeze();result=self.complete_result(control,binding);m.require_final_results(result,control,binding)
        for mode in ('oldschema','missingowner','duplicatejob','misassigned','count','motion_unknown'):
            bad=deepcopy(result)
            if mode=='oldschema':bad['schema']='r10_three_tier_composite_formal_TEST_results_v957'
            elif mode=='missingowner':bad['actual_stage_owners'].pop()
            elif mode=='duplicatejob':bad['evidence_tiers'][-1]=deepcopy(bad['evidence_tiers'][0])
            elif mode=='misassigned':bad['evidence_tiers'][0]['evidence_runtime']=m.OWNER_RUNTIME
            elif mode=='count':bad['actual_stage_owners'][0]['completed']+=1
            else:bad['motion_unknown_retained']=False
            with self.subTest(mode=mode),self.assertRaises(ValueError):m.require_final_results(bad,control,binding)

    def test_control_paths_and_lock_control_pointer_cannot_select_other_namespace(self):
        self.freeze();wrong=deepcopy(self.args);wrong.resume_control=str(self.c.parent/'other.json');Path(wrong.resume_control).write_text('{}')
        with self.assertRaises(ValueError):m.load_small_control(wrong)
        lock=json.loads(self.l.read_bytes());lock['control']['sha256']='f'*64;self.l.write_text(json.dumps(lock));self.args.resume_lock_sha256=m.point(self.l)['sha256']
        with self.assertRaises(ValueError):m.load_small_control(self.args)


if __name__=='__main__':unittest.main()
