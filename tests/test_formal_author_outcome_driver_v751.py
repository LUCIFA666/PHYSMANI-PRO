"""Explicit new formal bindings and original worker CPU branch, no data/native."""
from contextlib import ExitStack,redirect_stdout
from copy import deepcopy
import io,json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from scripts import run_formal_case_author_outcome_v751_r10 as driver
from our_mani.multitask.evaluation import file_sha
from our_mani.multitask.native_outcome_sampling_v741 import ActionOutcomeSampling,NONTERMINAL


class FrozenAuthorSuite:
    def __init__(self,root):
        self.root=Path(root);self.files={};self.module_paths={}
        modules={'driver':driver,'guard':driver.guard,'worker':driver.worker,
            'runtime':driver.worker.runtime,'contract':driver.worker.contract}
        project=Path(driver.ROOT)
        for key,module in modules.items():
            path=Path(module.__file__);relative=path.relative_to(project)
            self.module_paths[key]=self.root/relative
            self.write(str(relative),path.read_bytes())
        for relative in ('our_mani/multitask/native_outcome_sampling_v741.py',
                         'scripts/run_task_executor_smoke_rml_suite16_v664_r10.py'):
            self.write(relative,(project/relative).read_bytes())
        self.protocol=driver.worker.contract.outcome_protocol_binding(self.root)
        self.route=dict(task='push_moving_button',method='context_geometry',
            worker=self.pointer(self.module_paths['worker']),consumer=self.pointer(self.module_paths['runtime']))
        self.routes={'routes':[self.route]}
        self.suite=dict(formal_case_driver=self.pointer(self.module_paths['driver']),
            formal_admission_guard=self.pointer(self.module_paths['guard']),author_outcome_protocol=self.protocol)
        self.refresh()

    def pointer(self,path):
        path=Path(path);return dict(path=str(path.relative_to(self.root)),sha256=file_sha(path))

    def write(self,name,value):
        path=self.root/name;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(value if isinstance(value,bytes) else json.dumps(value).encode())
        item=self.pointer(path);self.files[str(path)]=item['sha256'];return item

    def refresh(self):
        self.suite['routes']=self.write('routes.json',self.routes)
        self.write('suite.json',self.suite)
        (self.root/'suite_lock.json').write_text(json.dumps(dict(files=self.files)))

    def argv(self,check=False):
        args=['--split','validation','--suite','suite.json','--suite-lock','suite_lock.json',
            '--task',self.route['task'],'--method',self.route['method'],'--raw-episode-id','fixed0',
            '--config','unused.json','--manifest','unused_VAL.json','--frozen-inputs','unused_lock.json',
            '--source-inputs','unused_source_lock.json','--out','result']
        return args+(['--check'] if check else [])

    def scope(self):
        stack=ExitStack()
        stack.enter_context(patch.object(driver,'ROOT',self.root))
        stack.enter_context(patch.object(driver.worker,'ROOT',self.root))
        modules={'driver':driver,'guard':driver.guard,'worker':driver.worker,
            'runtime':driver.worker.runtime,'contract':driver.worker.contract}
        for name,module in modules.items():
            stack.enter_context(patch.object(module,'__file__',str(self.module_paths[name])))
        return stack

    def case(self):
        sampling=ActionOutcomeSampling();sampling.initialize(NONTERMINAL)
        sampling.begin_action(tick=0,label='geometry_execute')
        sampling.observe_tick(NONTERMINAL,tick=1);sampling.end_action(NONTERMINAL,tick=1)
        transcript=sampling.receipt()
        return dict(version='r10_full16_explicit_author_action_outcome_v744',complete=True,cleanup_completed=True,
            split='validation',task=self.route['task'],method=self.route['method'],official_task_success=False,
            native_terminal_failure=False,counted_clock=dict(ticks=1,dt=.05,seconds=.05),
            author_outcome_protocol=self.protocol,native_outcome_sampling=transcript,
            alignment_receipt=dict(native_outcome_sampling=deepcopy(transcript),native_success_latched=False,
                native_terminal_failure_latched=False,native_terminal_latched=False,native_terminal_tick=None,
                native_outcome=dict(success=False,terminate=False,terminal_failure=False)))


class FormalAuthorDriverTests(unittest.TestCase):
    def setUp(self):
        tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup)
        self.fixture=FrozenAuthorSuite(tmp.name);scope=self.fixture.scope();self.addCleanup(scope.close)

    def run_driver(self,check=False):
        stream=io.StringIO()
        with patch.object(driver.sys,'argv',['v751']+self.fixture.argv(check)),redirect_stdout(stream):driver.main()
        return stream.getvalue()

    def test_actual_bindings_select751_728_743_744_and_exact_protocol(self):
        expected,split=driver.bindings(self.fixture.argv())
        self.assertEqual(split,'validation')
        self.assertEqual(set(expected),set(driver.BINDING_KEYS))
        self.assertEqual(expected['author_outcome_protocol'],self.fixture.protocol)
        self.assertEqual(expected['original_native_runtime'],self.fixture.protocol['original_runtime'])
        self.assertEqual(expected['case_worker'],self.fixture.route['worker'])

    def test_old_runtime_or_different_protocol_cannot_enter_worker(self):
        for mutate in (
            lambda:self.fixture.route.update(consumer=self.fixture.protocol['original_runtime']),
            lambda:self.fixture.suite.update(author_outcome_protocol=dict(self.fixture.protocol,public_hold_max_ticks=10))):
            with self.subTest(mutation=mutate):
                original_route=deepcopy(self.fixture.route);original_suite=deepcopy(self.fixture.suite)
                mutate();self.fixture.refresh()
                with patch.object(driver.worker,'main') as worker:
                    with self.assertRaises(ValueError):self.run_driver(True)
                    worker.assert_not_called()
                self.fixture.route.clear();self.fixture.route.update(original_route)
                self.fixture.suite.clear();self.fixture.suite.update(original_suite);self.fixture.refresh()

    def test_original743_CPU_branch_is_CPU_only_and_truthfully_bound(self):
        prepared=dict(identity={'sha256':'identity'},component_domain=SimpleNamespace(receipt=lambda:{'purpose':'evaluation'}),
            model_preflight={},source_lock_sha256='a'*64)
        previous_admit=driver.original_guard.admit;previous_read=driver.original_guard.read_formal_source
        def preflight(*_):
            self.assertIs(driver.original_guard.admit,driver.guard.admit)
            self.assertIs(driver.original_guard.read_formal_source,driver.guard.read_formal_source)
            return prepared
        with patch.object(driver.worker.contract,'prepare',side_effect=preflight),\
             patch.object(driver.worker.runtime,'run',side_effect=AssertionError('native/GPU launched')) as native:
            proof=json.loads(self.run_driver(True))
        native.assert_not_called()
        self.assertIs(proof['native_started'],False);self.assertIs(proof['GPU_started'],False)
        self.assertIs(proof['execution_protocol_changed'],True)
        self.assertEqual(proof['native_runtime'],self.fixture.route['consumer'])
        self.assertEqual(proof['author_outcome_protocol'],self.fixture.protocol)
        self.assertNotIn('unchanged_case_worker',proof);self.assertNotIn('physical_policy_changed',proof)
        self.assertIs(driver.original_guard.admit,previous_admit)
        self.assertIs(driver.original_guard.read_formal_source,previous_read)

    def test_CPU_cannot_replace_protocol_and_scopes_restore_on_rejection(self):
        previous_admit=driver.original_guard.admit;previous_read=driver.original_guard.read_formal_source
        proof=dict(native_started=False,GPU_started=False,native_runtime=self.fixture.route['consumer'],
            original_native_runtime=self.fixture.protocol['original_runtime'],author_outcome_protocol={})
        with patch.object(driver.worker,'main',side_effect=lambda:print(json.dumps(proof))):
            with self.assertRaisesRegex(ValueError,'CPU author-outcome'):self.run_driver(True)
        self.assertIs(driver.original_guard.admit,previous_admit)
        self.assertIs(driver.original_guard.read_formal_source,previous_read)

    def test_native_receipt_binds_new_policy_and_never_rewrites_original_case(self):
        case=self.fixture.case();path=self.fixture.root/'result/case.json'
        payload=json.dumps(case).encode()
        def worker():path.parent.mkdir();path.write_bytes(payload)
        with patch.object(driver.worker,'main',side_effect=worker):self.run_driver()
        receipt=json.loads((path.parent/'admission_driver_receipt.json').read_text())
        self.assertEqual(path.read_bytes(),payload)
        self.assertEqual(receipt['schema'],driver.DRIVER_SCHEMA)
        self.assertEqual(receipt['case'],self.fixture.pointer(path))
        self.assertEqual(receipt['author_outcome_protocol'],self.fixture.protocol)
        self.assertIs(receipt['execution_protocol_changed'],True)
        self.assertNotIn('physical_policy_changed',receipt)

    def test_native_stale_policy_or_partial_transcript_cannot_receive_receipt(self):
        for change in ('old_version','protocol','incomplete','missing_tick','aborted'):
            with self.subTest(change=change):
                case=self.fixture.case()
                if change=='old_version':case['version']='r10_full16_actual_split_common_author_RML_v664'
                elif change=='protocol':case['author_outcome_protocol']={}
                elif change=='incomplete':case['cleanup_completed']=False
                elif change=='missing_tick':case['counted_clock']['ticks']=2
                else:case['native_outcome_sampling']['aborted']=True
                path=self.fixture.root/'result/case.json';path.parent.mkdir(exist_ok=True);path.write_text(json.dumps(case))
                before=path.read_bytes()
                with patch.object(driver.worker,'main'):
                    with self.assertRaises(ValueError):self.run_driver()
                self.assertEqual(path.read_bytes(),before)
                self.assertFalse((path.parent/'admission_driver_receipt.json').exists())

    def test_actual_options_require_exactly_one_value(self):
        for args in (['--split'],['--split','--check'],['--split','validation','--split','test']):
            with self.subTest(args=args),self.assertRaises(ValueError):driver.option(args,'--split')


if __name__=='__main__':unittest.main()
