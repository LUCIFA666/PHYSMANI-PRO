"""Owner evidence and real dispatch-loop draining with CPU-only child doubles."""
from contextlib import ExitStack,nullcontext,redirect_stdout
from copy import deepcopy
import importlib.util,io,json,tempfile,threading,unittest
from pathlib import Path
from unittest.mock import patch
from scripts import run_formal_suite16_owner_v753_r10 as owner
from scripts import run_formal_case_author_outcome_v751_r10 as driver

_spec=importlib.util.spec_from_file_location('_author_driver_fixture',Path(__file__).with_name('test_formal_author_outcome_driver_v751.py'))
_fixture=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(_fixture)


class FormalAuthorOwnerTests(unittest.TestCase):
    def setUp(self):
        tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup)
        self.fixture=_fixture.FrozenAuthorSuite(tmp.name)
        self.owner_file=self.fixture.write('scripts/run_formal_suite16_owner_v753_r10.py',Path(owner.__file__).read_bytes())
        self.fixture.refresh()
        self.scope=ExitStack();self.addCleanup(self.scope.close)
        self.scope.enter_context(patch.object(owner,'ROOT',self.fixture.root))
        self.scope.enter_context(patch.object(owner,'__file__',str(self.fixture.root/self.owner_file['path'])))

    def bindings(self):
        return dict(admission_driver=self.fixture.suite['formal_case_driver'],
            admission_guard=self.fixture.suite['formal_admission_guard'],case_worker=self.fixture.route['worker'],
            native_runtime=self.fixture.route['consumer'],original_native_runtime=self.fixture.protocol['original_runtime'],
            author_outcome_protocol=self.fixture.protocol)

    def test_CPU_requires_every_new_runtime_protocol_binding(self):
        row=self.bindings();proof=dict(row,execution_protocol_changed=True)
        owner.require_CPU_driver(proof,row)
        for key in driver.BINDING_KEYS:
            with self.subTest(key=key):
                bad=dict(proof);bad.pop(key)
                with self.assertRaises(ValueError):owner.require_CPU_driver(bad,row)
        with self.assertRaises(ValueError):owner.require_CPU_driver(dict(row,physical_policy_changed=False),row)

    def test_native_receipt_requires_exact_case_SHA_and_truthful_new_schema(self):
        path=self.fixture.root/'case.json';path.write_text('{}')
        row=self.bindings();proof=dict(row,schema=driver.DRIVER_SCHEMA,complete=True,
            execution_protocol_changed=True,actual_split='validation',case=owner.pointer(path))
        owner.require_native_driver(proof,row,path,'validation')
        for change in ('old_schema','old_claim','wrong_case','wrong_protocol'):
            bad=deepcopy(proof)
            if change=='old_schema':bad['schema']='r10_guard_only_formal_driver_v729'
            elif change=='old_claim':bad.pop('execution_protocol_changed');bad['physical_policy_changed']=False
            elif change=='wrong_case':bad['case']['sha256']='0'*64
            else:bad['author_outcome_protocol']['public_hold_max_ticks']=10
            with self.subTest(change=change),self.assertRaises(ValueError):owner.require_native_driver(bad,row,path,'validation')

    def test_DRAIN_never_sets_cancellation_but_STOP_and_global_STOP_do(self):
        runtime=self.fixture.root/'runtime/framework/check';runtime.mkdir(parents=True)
        cancelled=threading.Event();(runtime/'DRAIN').touch()
        self.assertTrue(owner.dispatch_control(runtime,cancelled,False));self.assertFalse(cancelled.is_set())
        (runtime/'DRAIN').unlink()
        self.assertTrue(owner.dispatch_control(runtime,cancelled,True)) # latched; no implicit resume
        self.assertFalse(owner.drained(True,{'checking':object()},{}))
        self.assertFalse(owner.drained(True,{}, {'active':object()}))
        self.assertTrue(owner.drained(True,{},{}))
        (runtime/'STOP').touch();owner.dispatch_control(runtime,cancelled,False)
        self.assertTrue(cancelled.is_set())
        (runtime/'STOP').unlink();cancelled.clear()
        (self.fixture.root/'runtime/framework/multitask_r10_STOP').touch()
        owner.dispatch_control(runtime,cancelled,False);self.assertTrue(cancelled.is_set())

    def run_draining_owner(self,when):
        root=self.fixture.root;runtime=root/'runtime/framework/fake_owner';output=root/'artifacts/framework/multitask_r10/fake_VAL'
        runtime.parent.mkdir(parents=True,exist_ok=True)
        manifest=dict(path='validation.json',sha256='a'*64)
        suite=dict(self.fixture.suite,source_manifests={'validation':manifest})
        (root/'suite.json').write_text(json.dumps(suite))
        source_lock=dict(schema='r10_full16_post_admission_split_source_lock_v1',split='validation',policy_sha256='policy',
            source_manifest_sha256=manifest['sha256'],source_count=315,TEST_used_for_fitting=False,files={})
        (root/'source.json').write_text(json.dumps(source_lock))
        routes=[]
        for method in owner.METHODS:
            routes.append(dict(self.fixture.route,method=method,configuration=dict(path='config.json',sha256='b'*64),
                frozen_inputs=dict(path='input.json',sha256='c'*64)))
        rows=[dict(task=self.fixture.route['task'],raw_episode_id=str(index),path='synthetic_VAL') for index in range(315)]
        proof=dict(policy_sha256='policy',routes=dict(routes=routes,common_author_RML_profile={}))
        calls=[];finished=[]

        def child(row,leaf,cancelled,*,check):
            self.assertFalse(cancelled.is_set())
            calls.append(('CPU' if check else 'native',row['job_id']))
            leaf.mkdir(parents=True)
            if check:
                receipt={key:row[key] for key in driver.BINDING_KEYS}
                receipt.update(complete=True,native_started=False,GPU_started=False,execution_protocol_changed=True,
                    actual_split='validation',component_domain={'purpose':'evaluation'},identity=dict(
                    sha256='identity-'+row['job_id'],source_row_sha256=row['source_row_sha256'],method=row['method'],
                    config_sha256=row['configuration_sha256'],frozen_inputs_sha256=row['input_lock_sha256'],
                    source_input_lock_sha256=owner.file_sha(root/'source.json')))
                (leaf/'stdout.log').write_text(json.dumps(receipt))
                if when=='checking':(runtime/'DRAIN').touch()
                result=dict(complete=True,returncode=0,automatic_retry=False)
            else:
                path=root/driver.option(row['argv'],'--out')/'case.json';path.parent.mkdir(parents=True)
                case=self.fixture.case();case.update(method=row['method'],component_domain={'purpose':'evaluation'},
                    full16_frozen_execution={'policy_sha256':'policy'})
                if row['method']=='physmani':
                    for sampling in (case['native_outcome_sampling'],case['alignment_receipt']['native_outcome_sampling']):
                        sampling['completed_actions'][0]['label']='baseline_execute'
                path.write_text(json.dumps(case))
                external=dict({key:row[key] for key in driver.BINDING_KEYS},schema=driver.DRIVER_SCHEMA,complete=True,
                    execution_protocol_changed=True,actual_split='validation',case=owner.pointer(path))
                driver_path=path.parent/'admission_driver_receipt.json';driver_path.write_text(json.dumps(external))
                result=dict(complete=True,case=owner.pointer(path),admission_driver_receipt=owner.pointer(driver_path))
                (runtime/'DRAIN').touch()
            self.assertFalse(cancelled.is_set())
            finished.append(('CPU' if check else 'native',row['job_id']))
            return result

        args=['v753','--suite','suite.json','--suite-lock','suite_lock.json','--source-inputs','source.json',
            '--split','validation','--runtime',str(runtime.relative_to(root)),'--out',str(output.relative_to(root))]
        with ExitStack() as stack,redirect_stdout(io.StringIO()):
            for name,value in (
                ('verify_inputs',lambda *_:None),('locked_pointer',lambda root,files,pointer:root/pointer['path']),
                ('driver_bindings',lambda *_:(suite['formal_case_driver'],suite['formal_admission_guard'])),
                ('read_formal_source',lambda *_a,**_k:({},rows,proof)),('fingerprint',lambda *_:'profile'),
                ('process_visibility',lambda:dict(host_visibility_available=True,live_project_native_processes=[],same_uid_inventory_incomplete=False)),
                ('exclusive_pair',lambda:nullcontext()),('child',child),('validate_completed_case',lambda *_:None),
                ('require_attestation',lambda *_:None),('measured_motion',lambda *_:False)):
                stack.enter_context(patch.object(owner,name,value))
            stack.enter_context(patch.object(owner.sys,'argv',args))
            owner.main()
        terminal=json.loads((runtime/'terminal.json').read_text())
        status=json.loads((runtime/'status.json').read_text())
        self.assertEqual(sorted(calls),sorted(finished))
        self.assertTrue(terminal['drained']);self.assertTrue(terminal['drain_requested'])
        self.assertFalse(terminal['stop_requested']);self.assertFalse(terminal['complete'])
        self.assertEqual(terminal['failed'],0);self.assertEqual(status['checking'],[]);self.assertEqual(status['active'],[])
        self.assertEqual(terminal['expected'],945)
        self.assertEqual(terminal['CPU_not_launched']+len([c for c in calls if c[0]=='CPU']),945)
        return calls,terminal

    def test_DRAIN_finishes_already_checking_CPU_without_starting_native(self):
        calls,terminal=self.run_draining_owner('checking')
        self.assertTrue(calls);self.assertTrue(all(stage=='CPU' for stage,_ in calls))
        self.assertEqual(len(terminal['CPU_ready_unexecuted']),len(calls))

    def test_DRAIN_finishes_active_native_and_checking_without_cancelling_them(self):
        calls,terminal=self.run_draining_owner('active')
        native=[job for stage,job in calls if stage=='native']
        self.assertTrue(native);self.assertLessEqual(len(native),2)
        self.assertEqual(terminal['completed'],len(native))


if __name__=='__main__':unittest.main()
