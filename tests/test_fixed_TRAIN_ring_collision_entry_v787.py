"""CPU mocks only: no source data, native environment, GPU or queued work."""
from contextlib import nullcontext, redirect_stdout, redirect_stderr
import fcntl
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from scripts import diagnose_fixed_TRAIN_ring_collision_v787_r10 as entry
from tests.test_train_public_collision_diagnostics_v783 import Map, Port, domain


class Fixture:
    def __init__(self, root):
        self.root = Path(root)
        (self.root/'runtime/framework').mkdir(parents=True)
        # Copy code only; never load a dataset/checkpoint/source manifest.
        for name in (*entry.SOURCES, entry.WORKER):
            target = self.root/name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((entry.ROOT/name).read_bytes())
        self.out = self.root/entry.OUTPUT/'synthetic_one_case'
        self.map = Map();self.port = Port(self.map)
        self.attachments = []
        self.map.env.map_service = self.map
        self.map.env.gripper.get_grasped_objects = lambda:self.attachments[:]
        self.actor = SimpleNamespace(map=self.map, collision_port=self.port)
        self.prepared = dict(component_domain=domain(), formal=False, out=self.out/'case', model_preflight={},
            args=SimpleNamespace(task='insert_onto_rotating_peg',method='cv_geometry',split='train'))
        self.value = (self.actor, {'original':True}, object(), object(), object())
        def geometry(*args, **kwargs):return self.value
        self.worker = SimpleNamespace(__file__=str(self.root/entry.WORKER),
            runtime=SimpleNamespace(_geometry=geometry),
            base_factory=SimpleNamespace(create_actor=object()),
            extension=SimpleNamespace(create_actor=object()),
            base_policy=SimpleNamespace(construct=object()),latest_policy=SimpleNamespace(construct=object()))
        self.result = dict(official_task_success=False, synthetic=True)
        self.error = None
        def run(p):
            assert self.worker.base_factory.create_actor is self.worker.extension.create_actor
            assert self.worker.base_policy.construct is self.worker.latest_policy.construct
            assert self.worker.runtime._geometry(p,self.map.env,['synthetic']) is self.value
            # A real frozen maps.check branch, with exclusively CPU ghosts.
            self.attachments.append(SimpleNamespace(get_handle=lambda:9001))
            self.map.env.arm.aggregate_result = True
            self.map.env.arm.positive_handles = {30}
            assert self.port.model.collision_query(0.) is True
            if self.error is not None:raise self.error
            return self.result
        self.worker.runtime.run=Mock(side_effect=run)
        self.lock_path = self.root/entry.DIAGNOSTIC_LOCK
        self.lock_path.parent.mkdir(parents=True)
        self.write_lock()

    def write_lock(self):
        self.lock_path.write_text(json.dumps(dict(schema=entry.INPUT_SCHEMA, split='train',
            approved_for_development_only=True,files=entry.source_fingerprints(self.root))))

    def argv(self):
        return ['--config',entry.CONFIG_BASE+'/insert_onto_rotating_peg_ours_configuration.json',
            '--frozen-inputs',entry.TRAIN_LOCK,
            '--manifest','artifacts/framework/protocol/train_manifest.json',
            '--task','insert_onto_rotating_peg','--raw-episode-id','synthetic_train_only',
            '--method','cv_geometry','--split','train','--development-only',
            '--out',str(self.out.relative_to(self.root))]

    def run(self, permit):
        return entry.run_prepared(self.root,self.prepared,self.worker,self.out,permit,
            diagnostic_lock=entry.verify_diagnostic_inputs(self.root,self.lock_path))


def proc_entry(proc,pid,start,*,state='S',script=None,cwd=None):
    directory=proc/str(pid);directory.mkdir(parents=True,exist_ok=True)
    fields=[state,'1']+['0']*17+[str(start)]+['0']*8
    (directory/'stat').write_text(str(pid)+' (synthetic python) '+' '.join(fields))
    (directory/'cmdline').write_bytes(('python\0'+(script or 'unrelated.py')+'\0').encode())
    if cwd is not None:(directory/'cwd').symlink_to(cwd)


class EntryTests(unittest.TestCase):
    def setUp(self):
        temporary=tempfile.TemporaryDirectory();self.addCleanup(temporary.cleanup)
        self.fixture=Fixture(temporary.name)
        self.free=patch.object(entry,'resource_snapshot',return_value={'synthetic_CPU_inventory':True})

    def test_default_main_is_CPU_only_and_creates_no_output_or_queue(self):
        fixture=self.fixture;stream=io.StringIO()
        with patch.object(entry,'ROOT',fixture.root),self.free,\
             patch.object(entry,'original_cpu_scopes',return_value=nullcontext()),\
             patch.object(entry,'prepare',return_value=(fixture.prepared,fixture.worker)),redirect_stdout(stream):
            entry.main(fixture.argv())
        result=json.loads(stream.getvalue())
        fixture.worker.runtime.run.assert_not_called()
        self.assertFalse(result['native_started']);self.assertFalse(result['GPU_started'])
        self.assertFalse(result['participates_in_formal_qualification'])
        self.assertFalse(fixture.out.exists())
        self.assertEqual(set(result['future_diagnostic_input_template']['files']),set(entry.source_fingerprints(fixture.root)))

    def test_request_rejects_non_train_non_ring_non_geometry_and_reused_output(self):
        fixture=self.fixture
        for option,value in [('--split','validation'),('--split','test'),('--method','physmani'),('--task','push_moving_button')]:
            argv=fixture.argv();argv[argv.index(option)+1]=value
            with self.subTest(option=option,value=value),redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
                entry.parser().parse_args(argv)
        args=entry.parser().parse_args(fixture.argv())
        fixture.out.mkdir(parents=True)
        with self.assertRaisesRegex(ValueError,'fresh isolated'):entry.validate_request(fixture.root,args)

    def test_native_requires_explicit_source_lock_before_any_prepare(self):
        args=entry.parser().parse_args(self.fixture.argv()+['--execute-native'])
        with self.assertRaisesRegex(ValueError,'source lock'):entry.validate_request(self.fixture.root,args)

    def test_non_train_file_paths_rejected_before_reading_any_input(self):
        fixture=self.fixture
        for option in ('--manifest','--config','--frozen-inputs'):
            argv=fixture.argv();argv[argv.index(option)+1]='forbidden_TEST_data.json'
            args=entry.parser().parse_args(argv)
            with self.subTest(option=option),patch.object(entry,'sha') as read_bytes:
                with self.assertRaises(ValueError):entry.validate_request(fixture.root,args)
                read_bytes.assert_not_called()
        with self.assertRaisesRegex(ValueError,'isolated exact'):
            entry.verify_diagnostic_inputs(fixture.root,'forbidden_TEST_data.json')

    def test_native_cpu_display_gate_failure_never_calls_runtime(self):
        fixture=self.fixture
        argv=fixture.argv()+['--execute-native','--diagnostic-inputs',entry.DIAGNOSTIC_LOCK]
        with patch.object(entry,'ROOT',fixture.root),self.free,\
             patch.object(entry,'original_cpu_scopes',return_value=nullcontext()),\
             patch.object(entry,'prepare',return_value=(fixture.prepared,fixture.worker)),\
             patch('scripts.native_display_guard_r10.readiness',return_value={'ready':False,'reason':'synthetic CUDA/X11 rejection'}),\
             patch.object(entry,'run_prepared') as native:
            with self.assertRaisesRegex(RuntimeError,'environment not ready'):entry.main(argv)
        native.assert_not_called();fixture.worker.runtime.run.assert_not_called()
        self.assertFalse(fixture.out.exists())

    def test_same_kernel_native_lock_busy_fails_closed_before_data_or_worker(self):
        fixture=self.fixture
        for lock in entry.LOCKS:
            with self.subTest(lock=lock), (fixture.root/lock).open('a+') as stream:
                fcntl.flock(stream.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
                with patch.object(entry,'ROOT',fixture.root),patch.object(entry,'prepare') as prepare:
                    with self.assertRaisesRegex(RuntimeError,'lock busy'):entry.main(fixture.argv())
                prepare.assert_not_called()

    def test_global_stop_and_uncertain_visibility_fail_closed(self):
        fixture=self.fixture
        stop=fixture.root/'runtime/framework/multitask_r10_STOP';stop.touch()
        with self.assertRaises(InterruptedError):
            with entry.reserved_resources(fixture.root):pass
        stop.unlink()
        with patch.object(entry,'resource_snapshot',side_effect=RuntimeError('incomplete inventory')):
            with self.assertRaisesRegex(RuntimeError,'incomplete inventory'):
                with entry.reserved_resources(fixture.root):pass
        # Every partially acquired lock was released after rejection.
        with self.free,entry.reserved_resources(fixture.root) as permit:self.assertTrue(permit.active)
        self.assertFalse(permit.active)
        with self.assertRaises(RuntimeError):entry.require_permit(permit,fixture.root)

    def test_exact_pid_start_records_block_stopped775_and_other_native_workers(self):
        fixture=self.fixture;proc=fixture.root/'synthetic_proc';(proc/'1').mkdir(parents=True)
        (proc/'1/comm').write_text('systemd\n')
        for index,name in enumerate(entry.RUNTIMES[:2]):
            runtime=fixture.root/name;runtime.mkdir(parents=True)
            (runtime/'launch.json').write_text(json.dumps(dict(pid=100+index,start_ticks=200+index)))
            (runtime/'terminal.json').write_text('{}')
        proc_entry(proc,100,200,state='T')  # SIGSTOP is live, not released.
        with self.assertRaisesRegex(RuntimeError,'still live'):entry.resource_snapshot(fixture.root,proc)
        (proc/'100/stat').write_text((proc/'100/stat').read_text().replace(') T ',') Z '))
        entry.resource_snapshot(fixture.root,proc)
        # The actual orphan native launcher may retain751 in argv, even
        # though its code internally invokes743; detect both identities.
        proc_entry(proc,303,404,script='scripts/run_formal_case_author_outcome_v751_r10.py',cwd=fixture.root)
        with self.assertRaisesRegex(RuntimeError,'existing project'):entry.resource_snapshot(fixture.root,proc)

    def test_sealed_original_train_prepare_is_used_with_frozen743_identity(self):
        fixture=self.fixture;worker=SimpleNamespace(__file__=str(fixture.root/entry.WORKER),
            contract=SimpleNamespace(prepare=Mock(return_value=fixture.prepared)))
        args=entry.parser().parse_args(fixture.argv())
        prepared,_=entry.prepare(fixture.root,args,fixture.out,worker)
        called=worker.contract.prepare.call_args.args
        self.assertIs(prepared,fixture.prepared)
        self.assertEqual(called[2],Path(worker.__file__))
        self.assertEqual(called[1].split,'train');self.assertTrue(called[1].development_demo)
        self.assertIsNone(called[1].suite);self.assertIsNone(called[1].source_inputs)
        self.assertEqual(called[1].out,str((fixture.out/'case').relative_to(fixture.root)))
        fixture.prepared['component_domain']=domain('validation')
        with self.assertRaisesRegex(ValueError,'sealed capability'):entry.prepare(fixture.root,args,fixture.out,worker)

    def test_execution_scope_forwards_exact_return_restores_hooks_and_records_unknown_collection(self):
        fixture=self.fixture;runtime=fixture.worker.runtime
        originals=(runtime._geometry,fixture.worker.base_factory.create_actor,
            fixture.worker.base_policy.construct,fixture.port.model.collision_query)
        with self.free,entry.reserved_resources(fixture.root) as permit:
            self.assertIs(fixture.run(permit),fixture.result)
        self.assertEqual((runtime._geometry,fixture.worker.base_factory.create_actor,
            fixture.worker.base_policy.construct,fixture.port.model.collision_query),originals)
        sidecar=json.loads((fixture.out/'public_collision_diagnostics_v787.json').read_text())
        self.assertTrue(sidecar['complete']);self.assertFalse(sidecar['participates_in_VAL_TEST'])
        self.assertEqual(len(sidecar['public_query_traces'][0]['events']),1)
        self.assertTrue(sidecar['public_query_traces'][0]['restored'])
        self.assertFalse(sidecar['public_query_traces'][0]['native_validated'])
        observations=sidecar['collection_observations']
        self.assertEqual([x['label'] for x in observations],[
            'after_actor_creation_before_first_policy_observation_or_action',
            'first_actual_lift_public_map_rejection'])
        self.assertEqual(observations[0]['collection']['cached_native_grasped_handles'],[])
        self.assertEqual(observations[1]['collection']['cached_native_grasped_handles'],[9001])
        for observation in observations:
            inventory=observation['collection']
            self.assertIsNone(inventory['collection_members']);self.assertIsNone(inventory['overrideProperties'])
            self.assertTrue(inventory['arm_tree_is_not_assumed_collection'])
            self.assertFalse(inventory['native_target_pose_read'])

    def test_original_runtime_exception_is_not_replaced_and_scope_always_restored(self):
        fixture=self.fixture;error=RuntimeError('original synthetic744 error');fixture.error=error
        original=fixture.worker.runtime._geometry
        with self.free,entry.reserved_resources(fixture.root) as permit:
            with self.assertRaises(RuntimeError) as caught:fixture.run(permit)
        self.assertIs(caught.exception,error)
        self.assertIs(fixture.worker.runtime._geometry,original)
        self.assertIs(fixture.port.model.collision_query,fixture.port.query_callable)
        value=json.loads((fixture.out/'public_collision_diagnostics_v787.json').read_text())
        self.assertFalse(value['complete']);self.assertTrue(value['public_query_traces'][0]['restored'])

    def test_sidecar_write_error_cannot_mask_original_runtime_exception(self):
        fixture=self.fixture;error=RuntimeError('original failure');fixture.error=error
        original=entry._atomic
        def write(path,value):
            if Path(path).name=='public_collision_diagnostics_v787.json':raise OSError('sidecar write unavailable')
            return original(path,value)
        with self.free,entry.reserved_resources(fixture.root) as permit,patch.object(entry,'_atomic',side_effect=write):
            with self.assertRaises(RuntimeError) as caught:fixture.run(permit)
        self.assertIs(caught.exception,error)
        self.assertIs(fixture.port.model.collision_query,fixture.port.query_callable)

    def test_stale_diagnostic_sources_or_missing_reservation_cannot_execute(self):
        fixture=self.fixture
        with self.assertRaisesRegex(RuntimeError,'resource reservation'):
            fixture.run(None)
        locked=entry.verify_diagnostic_inputs(fixture.root,fixture.lock_path)
        (fixture.root/entry.SOURCES[1]).write_text('changed diagnostic source')
        with self.free,entry.reserved_resources(fixture.root) as permit:
            with self.assertRaisesRegex(ValueError,'four-source'):
                entry.run_prepared(fixture.root,fixture.prepared,fixture.worker,fixture.out,permit,diagnostic_lock=locked)
        self.assertFalse(fixture.out.exists());fixture.worker.runtime.run.assert_not_called()


if __name__=='__main__':unittest.main()
