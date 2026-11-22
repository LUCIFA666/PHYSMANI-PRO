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

path = Path(__file__).resolve().parents[1] / 'scripts/continue_composite_TEST_v958_r10.py'
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
        self.setup = self.root / 'artifacts/framework/multitask_r10/fake_setup'; self.setup.mkdir(parents=True)
        self.c = self.setup / 'control.json'; self.l = self.setup / 'lock.json'
        self.control = dict(schema='r10_composite_formal_TEST_resume_control_v955', split='test', physical_policy_changed=False,
            resume_controller=m.point(m.SOURCE), resume_owner=m.point(m.OWNER), resume_auditor=m.point(m.AUDITOR),
            policy_sha256='a' * 64, source_inputs=dict(path='sources.json', sha256='b'*64),
            validation_release=dict(path='release.json', sha256='c'*64), resource_qualification=dict(path='qualification.json',sha256='d'*64),
            author_outcome_protocol={'original': True})

    def freeze(self):
        self.c.write_text(json.dumps(self.control))
        files = {str(self.c): m.point(self.c)['sha256']}
        files.update({str(self.root/name): m.point(name)['sha256'] for name in (m.SOURCE,m.OWNER,m.AUDITOR)})
        self.l.write_text(json.dumps(dict(schema='r10_composite_formal_TEST_resume_inputs_v955',approved_for_execution=True,files=files)))
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
        self.control['schema']='r10_composite_formal_TEST_resume_control_v955'
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
        result=dict(schema='r10_three_tier_composite_formal_TEST_results_v957',complete=True,split='test',
            source_cases=640,method_cases=1920,completed_method_cases=1920,methods=list(METHODS),tasks=list(TASKS),
            source_sha256=control['resume_auditor']['sha256'],resume_control=binding['control'],resume_control_lock=binding['lock'],
            policy_sha256=control['policy_sha256'],source_input_lock=control['source_inputs'],author_outcome_protocol=control['author_outcome_protocol'],
            validation_release=control['validation_release'],resource_qualification=control['resource_qualification'],
            native_failures_retained=True,Value_NGFF_FASTER_used=False,native_successes_by_method={x:0 for x in METHODS})
        m.require_final_results(result,control,binding)
        for key in ('source_sha256','resume_control','resume_control_lock','policy_sha256','source_input_lock',
                    'author_outcome_protocol','validation_release','resource_qualification','completed_method_cases','methods','tasks','split'):
            bad=deepcopy(result);bad[key]=None
            with self.subTest(key=key),self.assertRaises(ValueError):m.require_final_results(bad,control,binding)


if __name__=='__main__':unittest.main()
