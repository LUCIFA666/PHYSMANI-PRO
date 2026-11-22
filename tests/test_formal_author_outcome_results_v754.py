"""Independent binding/launch checks and full synthetic 945-case release gate."""
from contextlib import ExitStack
from copy import deepcopy
import importlib.util,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from scripts import audit_formal_suite16_results_v754_r10 as auditor
from scripts import run_formal_case_author_outcome_v751_r10 as driver
from our_mani.multitask import formal_suite_admission_v728 as admission
from our_mani.multitask import three_method_release_v531 as release

def fixture_module(name):
    spec=importlib.util.spec_from_file_location('_'+name,Path(__file__).with_name(name+'.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

_fixture=fixture_module('test_formal_author_outcome_driver_v751')
_science=fixture_module('test_common_rml_release_v531')


class FormalAuthorResultsTests(unittest.TestCase):
    def setUp(self):
        tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup)
        self.fixture=_fixture.FrozenAuthorSuite(tmp.name)
        project=Path(auditor.ROOT)
        for name in ('scripts/run_formal_suite16_owner_v753_r10.py','scripts/audit_formal_suite16_results_v754_r10.py'):
            pointer=self.fixture.write(name,(project/name).read_bytes())
            if 'auditor' in name:self.auditor_pointer=pointer
        self.fixture.suite['formal_result_auditor']=self.fixture.pointer(self.fixture.root/'scripts/audit_formal_suite16_results_v754_r10.py')
        self.fixture.refresh()
        self.scope=ExitStack();self.addCleanup(self.scope.close)
        self.scope.enter_context(patch.object(auditor,'ROOT',self.fixture.root))

    def expected(self):
        return dict(admission_driver=self.fixture.suite['formal_case_driver'],
            admission_guard=self.fixture.suite['formal_admission_guard'],case_worker=self.fixture.route['worker'],
            native_runtime=self.fixture.route['consumer'],original_native_runtime=self.fixture.protocol['original_runtime'],
            author_outcome_protocol=self.fixture.protocol)

    def test_auditor_requires_its_exact_frozen_driver_owner_and_new_protocol(self):
        driver_pointer,guard_pointer=auditor.driver_bindings(self.fixture.suite,self.fixture.files)
        self.assertEqual(driver_pointer,self.fixture.suite['formal_case_driver'])
        self.assertEqual(guard_pointer,self.fixture.suite['formal_admission_guard'])
        changed=deepcopy(self.fixture.suite);changed['author_outcome_protocol']['public_hold_max_ticks']=10
        with self.assertRaises(ValueError):auditor.driver_bindings(changed,self.fixture.files)

    def test_external_driver_receipt_binds_743_744_664_protocol_and_exact_case_bytes(self):
        path=self.fixture.root/'native_case.json';path.write_text('{}')
        expected=self.expected();receipt=dict(expected,schema=driver.DRIVER_SCHEMA,complete=True,
            execution_protocol_changed=True,actual_split='validation',case=auditor.pointer(path))
        result=auditor.require_driver_evidence(receipt,expected['admission_driver'],expected['admission_guard'],
            self.fixture.route,path,'validation')
        self.assertEqual(result,expected)
        for key in driver.BINDING_KEYS:
            bad=dict(receipt);bad[key]={}
            with self.subTest(key=key),self.assertRaises(ValueError):
                auditor.require_driver_evidence(bad,expected['admission_driver'],expected['admission_guard'],self.fixture.route,path,'validation')
        path.write_text('{"changed":true}')
        with self.assertRaises(ValueError):
            auditor.require_driver_evidence(receipt,expected['admission_driver'],expected['admission_guard'],self.fixture.route,path,'validation')

    def test_actual_launch_cannot_substitute_argv_runtime_or_CPU_GPU_scope(self):
        expected=self.expected();row=dict(method='physmani',argv=['bash','scripts/run_aligned_gpu.sh','driver751','--split','validation'])
        identity={'sha256':'actualCPUidentity'}
        for check in (True,False):
            launch=dict(expected,argv=row['argv']+(['--check'] if check else []),check_only=check,
                execution_protocol_changed=True,native_slots=0 if check else 1,GPU_slots=0 if check else 1,
                expected_identity_sha256=None if check else identity['sha256'])
            auditor.require_actual_launch(launch,row,expected,check=check,identity=identity)
            changes=[dict(argv=launch['argv']+['--check']),dict(native_runtime={}),dict(author_outcome_protocol={}),
                dict(execution_protocol_changed=False),dict(expected_identity_sha256='wrong'),
                dict(GPU_slots=1 if check else 0),dict(native_slots=1 if check else 0)]
            for values in changes:
                with self.subTest(check=check,values=values),self.assertRaises(ValueError):
                    auditor.require_actual_launch(dict(launch,**values),row,expected,check=check,identity=identity)

    def test_auditor_uses_shared_sampling_verifier_and_rejects_uncovered_native_ticks(self):
        case=self.fixture.case()
        auditor.require_native_protocol(case,self.fixture.protocol)
        case['native_outcome_sampling']['completed_actions']=[]
        with self.assertRaises(ValueError):auditor.require_native_protocol(case,self.fixture.protocol)

    def test_full945_gate_keeps_native_failures_and_motion_diagnostic_but_rejects_partial_or_wrong_RML(self):
        routes,manifest,cases=_science.Tests().fixture()
        proof=auditor.validation_release(cases,manifest,routes)
        self.assertEqual(proof['method_cases'],945)
        self.assertEqual(proof['source_cases'],315)
        self.assertTrue(release.require_test_release(proof,routes))
        self.assertEqual(sum(proof['native_successes_by_method'].values()),0)
        for altered in (cases[:-1],cases[:-1]+[cases[0]]):
            with self.assertRaises(ValueError):auditor.validation_release(altered,manifest,routes)
        wrong=deepcopy(cases);wrong[0]['common_author_RML_parameters']['actual_parameters']['max_acceleration']=6.
        with self.assertRaises(ValueError):auditor.validation_release(wrong,manifest,routes)

    def test_new_TEST_source_gate_cannot_select_any_manifest_before_complete945_VAL(self):
        routes,manifest,cases=_science.Tests().fixture()
        completed=auditor.validation_release(cases,manifest,routes)
        incomplete=dict(completed,completed_method_cases=944)
        with patch.object(admission,'admit',return_value=dict(routes=routes)),patch.object(admission,'locked_json') as read:
            with self.assertRaisesRegex(ValueError,'VAL release'):
                auditor.read_formal_source(self.fixture.root,{}, {},'test',validation_release=incomplete)
            read.assert_not_called()


if __name__=='__main__':unittest.main()
