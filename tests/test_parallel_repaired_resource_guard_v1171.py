"""CPU-only capability/locking fixtures. No simulator or GPU is imported."""
from contextlib import contextmanager, ExitStack
from copy import deepcopy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from weakref import WeakKeyDictionary

from our_mani.multitask import parallel_repaired_resource_guard_v1171 as subject


class FakePermit:
    pass


class FakeLegacy:
    _Permit = FakePermit

    def __init__(self):
        self._LEASES = WeakKeyDictionary()

    def require_permit(self, permit, root, *, allow_test_callbacks=False):
        subject.require(permit in self._LEASES, 'missing test registry entry')
        lease = self._LEASES[permit]
        subject.require(lease['pid'] == os.getpid() and lease['root'] == root,
            'same process test capability required')
        subject.require(not lease['test_only'] or allow_test_callbacks is True,
            'CPU test capability cannot authorize Native')
        lease['stop'](root)
        return deepcopy(lease['proof'])


class FakeParent:
    """Fake historical proof explicitly marked test_only; real tempfile flock."""
    LOCKS = ('runtime/framework/test_only_global.lock',)

    def __init__(self):
        self.legacy = FakeLegacy()
        self.entries = 0
        self.exits = 0

    def _legacy(self, root):
        return self.legacy

    def _stop(self, root):
        subject.require(not (root / 'runtime/framework/multitask_r10_STOP').exists(), 'test global STOP')

    def require_permit(self, permit, root):
        return dict(complete=True, test_only=True, scientific_admission_granted=False)

    @contextmanager
    def reserved_resources_after_TEST(self, root, *, enabled):
        subject.require(enabled is True, 'explicit fake scope')
        self.entries += 1
        file = root / self.LOCKS[0]
        file.parent.mkdir(parents=True, exist_ok=True)
        with file.open('a+b') as stream:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                yield FakePermit()
            finally:
                self.exits += 1


class Fixture:
    def __init__(self, root):
        self.root = root
        self.parent = FakeParent()
        self.runtime = subject.BASE + '/train/cpu_fixture'
        self.graph_path = 'artifacts/framework/multitask_r10/cpu_fixture/graph/policy_graph.json'
        self.config = dict(root='project', path='configs/cpu_fixture.json', sha256='4' * 64)
        self.job = dict(job_id='j0000', task='insert_onto_rotating_peg', method='cv_geometry',
            raw_episode_id='episode101', split='train', configuration=self.config,
            out=subject.ART + '/train/development_demo_only/cpu_fixture/ring/episode101/cv_geometry',
            runtime=subject.CASE_RUNTIME + '/train/cpu_fixture/ring/episode101/cv_geometry',
            source_row_sha256='5' * 64)
        self.profile = dict(subject.PROFILE_CEILING)
        for name in (subject.SOURCE, subject.PARENT_SOURCE):
            p = root / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text('# CPU-only fake source identity\n')
        self.graph = dict(schema='r10_frozen_repaired_policy_graph_v1143', roots=dict(project=str(root)),
            tasks=['insert_onto_rotating_peg'], routes=[dict(task=self.job['task'],
                method=self.job['method'], configuration=self.config)],
            physical_policy_sha256='1' * 64, control_graph_sha256='2' * 64,
            policy_sha256='3' * 64)
        guard_source = subject._source(root, subject.SOURCE)
        parent_source = subject._source(root, subject.PARENT_SOURCE)
        self.graph['control'] = dict(components=dict(resource_guard=guard_source),
            sources_map=dict(resource_guard=guard_source, serial_parent_guard=parent_source))
        self.graph_ref = self.write(self.graph_path, self.graph)
        self.plan = dict(split='train', graph=self.graph_ref, profile=self.profile,
            jobs=[self.job], **{k: self.graph[k] for k in subject.HASHES})
        self.plan_ref = self.write(self.runtime + '/plan.json', self.plan)

    def write(self, name, value):
        file = self.root / name
        file.parent.mkdir(parents=True, exist_ok=True)
        data = subject._bytes(value) + b'\n'
        file.write_bytes(data)
        return dict(root='project', path=name, sha256=hashlib.sha256(data).hexdigest())

    @contextmanager
    def patched(self):
        with ExitStack() as stack:
            stack.enter_context(patch.object(subject, '_parent', return_value=self.parent))
            stack.enter_context(patch.object(subject, '_graph', return_value=deepcopy(self.graph)))
            stack.enter_context(patch.object(subject, '_other_controllers', return_value=[]))
            # Same-process fixture only; its proof remains test_only and refuses Native.
            stack.enter_context(patch.object(subject, '_descendant', return_value=True))
            yield

    def owner(self, **kwargs):
        return subject.owner_scope(self.root, graph_ref=self.graph_ref, plan_ref=self.plan_ref,
            profile=self.profile, runtime=self.runtime, **kwargs)

    def child(self, context, **kwargs):
        defaults = dict(session_ref=context['session'], job=self.job,
            phase='CPU_check', slot=0, enabled=True)
        defaults.update(kwargs)
        return subject.child_scope(self.root, **defaults)


class ParallelResourceGuardTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        self.f = Fixture(self.root)

    def tearDown(self):
        self.temp.cleanup()

    def test_import_is_inert_and_requires_explicit_opt_in(self):
        with self.f.patched(), self.assertRaisesRegex(ValueError, 'enabled=True'):
            with self.f.owner():
                self.fail('inert default must not enter')
        self.assertEqual(self.f.parent.entries, 0)
        self.assertFalse((self.root / self.f.runtime / 'resources').exists())

    def test_real_tempfile_parent_and_child_flocks_and_receipt_after_exit(self):
        with self.f.patched(), self.f.owner(enabled=True) as context:
            session = subject.read_session(self.root, context['session'])
            self.assertTrue(session['test_only'])
            self.assertEqual(session['owner'], subject._process())
            with self.f.child(context) as permit:
                receipt = subject.require_permit(permit, self.root, allow_test_callbacks=True)
                self.assertFalse(receipt['child_empty_inventory_claimed'])
                self.assertIn(permit, self.f.parent.legacy._LEASES)
                with self.assertRaises(ValueError):
                    subject.require_permit(permit, self.root)
            receipt = subject.permit_receipt(permit)
            self.assertTrue(receipt['restored'])
            self.assertTrue(receipt['body_completed'])
            self.assertTrue(receipt['leases_removed'])
            self.assertEqual(receipt['errors'], [])
            self.assertNotIn(permit, self.f.parent.legacy._LEASES)
            with self.assertRaises(ValueError):
                subject.require_permit(permit, self.root, allow_test_callbacks=True)
        self.assertEqual(self.f.parent.exits, 1)

    def test_fake_cpu_proof_cannot_authorize_native(self):
        with self.f.patched(), self.f.owner(enabled=True) as context:
            with self.f.child(context):
                pass
            with self.assertRaisesRegex(ValueError, 'cannot authorize Native'):
                with self.f.child(context, phase='native'):
                    self.fail('fake Native capability issued')

    def test_profile_rejects_bool_float_extra_fields_and_ceiling(self):
        for key, value in (('max_native_slots', True), ('max_GPU_slots', 1.0),
                ('max_CPU_preflight_slots', 7), ('ready_limit', 0)):
            profile = dict(self.f.profile, **{key: value})
            with self.assertRaises(ValueError):
                subject._profile(profile)
        with self.assertRaises(ValueError):
            subject._profile(dict(self.f.profile, unbound=1))

    def test_foreign_controller_blocks_before_global_resource_acquisition(self):
        with self.f.patched(), patch.object(subject, '_other_controllers', return_value=[dict(pid=123)]):
            with self.assertRaisesRegex(ValueError, 'actually exit'):
                with self.f.owner(enabled=True):
                    pass
        self.assertEqual(self.f.parent.entries, 0)

    def test_second_controller_recheck_inside_parent_scope(self):
        with self.f.patched(), patch.object(subject, '_other_controllers', side_effect=[[], [dict(pid=123)]]):
            with self.assertRaisesRegex(ValueError, 'still live'):
                with self.f.owner(enabled=True):
                    pass
        self.assertEqual(self.f.parent.exits, 1)

    def test_exact_plan_graph_profile_and_route_binding(self):
        for key, value in (('policy_sha256', '9' * 64), ('profile', dict(self.f.profile, max_native_slots=2))):
            plan = deepcopy(self.f.plan)
            plan[key] = value
            ref = self.f.write(self.f.runtime + '/plan.json', plan)
            with self.f.patched(), self.assertRaises(ValueError):
                with subject.owner_scope(self.root, graph_ref=self.f.graph_ref, plan_ref=ref,
                        profile=self.f.profile, runtime=self.f.runtime, enabled=True):
                    pass
        bad = deepcopy(self.f.job)
        bad['configuration']['sha256'] = 'a' * 64
        with self.assertRaisesRegex(ValueError, 'graph route'):
            subject._job(self.root, bad, 'train', self.f.graph)

    def test_duplicate_jobs_and_outputs_refused(self):
        self.f.plan['jobs'] = [self.f.job, dict(self.f.job, job_id='j0001')]
        self.f.plan_ref = self.f.write(self.f.runtime + '/plan.json', self.f.plan)
        with self.f.patched(), self.assertRaisesRegex(ValueError, 'unique'):
            with self.f.owner(enabled=True):
                pass

    def test_session_bytes_immutable_and_typed_before_read(self):
        with self.f.patched(), self.f.owner(enabled=True) as context:
            p = self.root / context['session']['path']
            original = p.read_bytes()
            p.write_bytes(original + b' ')
            with self.assertRaisesRegex(ValueError, 'immutable'):
                subject.read_session(self.root, context['session'])
            p.write_bytes(original)
            forbidden = dict(root='project', path='artifacts/validation_manifest.json', sha256='0' * 64)
            with patch.object(Path, 'read_bytes', side_effect=AssertionError('forbidden read')):
                with self.assertRaisesRegex(ValueError, 'typed metadata'):
                    subject.read_session(self.root, forbidden)

    def test_owner_identity_boot_ancestry_and_actual_flock_are_required(self):
        with self.f.patched(), self.f.owner(enabled=True) as context:
            with patch.object(subject, '_boot', return_value='another-boot'):
                with self.assertRaisesRegex(ValueError, 'boot'):
                    subject.read_session(self.root, context['session'])
            with patch.object(subject, '_descendant', return_value=False):
                with self.assertRaisesRegex(ValueError, 'ancestry'):
                    subject.read_session(self.root, context['session'])
            real = subject._flock_rows
            with patch.object(subject, '_flock_rows', return_value=[]):
                with self.assertRaisesRegex(ValueError, 'FLOCK'):
                    subject.read_session(self.root, context['session'])
            self.assertTrue(real(os.getpid(), self.root / self.f.parent.LOCKS[0]))

    def test_actual_graph_guard_source_contract_refuses_old_graph_and_missing_parent(self):
        with patch.object(subject, '__file__', str(self.root / subject.SOURCE)):
            self.assertEqual(subject._graph(self.root, self.f.graph_ref), self.f.graph)
            graph = deepcopy(self.f.graph)
            graph['control']['components']['resource_guard'] = subject._source(self.root, subject.PARENT_SOURCE)
            ref = self.f.write(self.f.graph_path, graph)
            with self.assertRaisesRegex(ValueError, 'new1171'):
                subject._graph(self.root, ref)
            graph = deepcopy(self.f.graph)
            del graph['control']['sources_map']['serial_parent_guard']
            ref = self.f.write(self.f.graph_path, graph)
            with self.assertRaisesRegex(ValueError, 'both be source-bound'):
                subject._graph(self.root, ref)

    def test_slot_lock_contention_and_phase_claim_prevent_duplicate(self):
        with self.f.patched(), self.f.owner(enabled=True) as context:
            with self.f.child(context):
                with self.assertRaises(BlockingIOError):
                    with self.f.child(context):
                        pass
            with self.assertRaises(FileExistsError):
                with self.f.child(context, slot=1):
                    pass

    def test_exact_job_and_slots_reject_bool_or_unbound_gpu(self):
        with self.f.patched(), self.f.owner(enabled=True) as context:
            wrong = dict(self.f.job, raw_episode_id='episode102')
            with self.assertRaisesRegex(ValueError, 'exactly match'):
                with self.f.child(context, job=wrong):
                    pass
            for values in (dict(slot=True), dict(slot=6), dict(gpu_slot=0), dict(phase='TEST')):
                with self.assertRaises(ValueError):
                    with self.f.child(context, **values):
                        pass

    def test_original_body_exception_preserved_and_terminal_still_written(self):
        marker = RuntimeError('original worker error')
        with self.f.patched(), self.f.owner(enabled=True) as context:
            with self.assertRaises(RuntimeError) as caught:
                with self.f.child(context) as permit:
                    raise marker
            self.assertIs(caught.exception, marker)
            receipt = subject.permit_receipt(permit)
            self.assertFalse(receipt['body_completed'])
            self.assertTrue(receipt['restored'])
            self.assertEqual(receipt['original_exception_type'], 'RuntimeError')
            self.assertNotIn(permit, self.f.parent.legacy._LEASES)

    def test_lost_parent_on_exit_is_resource_failure_and_lease_cleanup(self):
        original = subject.read_session
        with self.f.patched(), self.f.owner(enabled=True) as context:
            final_patch = patch.object(subject, 'read_session', side_effect=ValueError('parent lost'))
            try:
                with self.assertRaisesRegex(RuntimeError, 'finalization'):
                    with self.f.child(context) as permit:
                        final_patch.start()
            finally:
                final_patch.stop()
            receipt = subject.permit_receipt(permit)
            self.assertTrue(receipt['restored'])
            self.assertFalse(receipt['body_completed'])
            self.assertEqual(receipt['errors'][0]['stage'], 'parent_or_slot_final_check')
            self.assertNotIn(permit, self.f.parent.legacy._LEASES)
        self.assertIs(subject.read_session, original)

    def test_stop_and_closed_session_cannot_be_used_as_live_permit(self):
        with self.f.patched(), self.f.owner(enabled=True) as context:
            stop = self.root / self.f.runtime / 'STOP'
            stop.write_text('stop')
            with self.assertRaisesRegex(ValueError, 'STOP'):
                subject.read_session(self.root, context['session'])
            stop.unlink()
            with self.f.child(context) as permit:
                pass
        with self.f.patched(), self.assertRaisesRegex(ValueError, 'FLOCK'):
            subject.read_session(self.root, context['session'])

    def test_resource_namespace_cannot_be_reused(self):
        with self.f.patched():
            with self.f.owner(enabled=True):
                pass
            with self.assertRaisesRegex(ValueError, 'already used'):
                with self.f.owner(enabled=True):
                    pass

    def test_canonical_924_registry_shape_and_no_legacy_empty_inventory_claim(self):
        with self.f.patched(), self.f.owner(enabled=True) as context:
            with self.f.child(context) as permit:
                lease = self.f.parent.legacy._LEASES[permit]
                self.assertEqual(lease['pid'], os.getpid())
                self.assertEqual(lease['root'], self.root)
                self.assertTrue(lease['test_only'])
                self.assertTrue(lease['proof']['parent1106_empty_inventory_checked_at_session_start_only'])
                self.assertFalse(lease['proof']['scientific_admission_granted'])

    def test_administrative_stop_drains_existing_claim_but_refuses_new_claim(self):
        with self.f.patched(), self.f.owner(enabled=True) as context:
            with self.f.child(context) as permit:
                stop = self.root / self.f.runtime / 'STOP'
                stop.write_text('administrative drain')
                subject.require_permit(permit, self.root, allow_test_callbacks=True)
                with self.assertRaisesRegex(ValueError, 'STOP'):
                    with self.f.child(context, slot=1):
                        pass
            self.assertTrue(subject.permit_receipt(permit)['body_completed'])
            stop.unlink()

    def test_global_stop_also_drains_claim_but_blocks_new_metadata_or_child(self):
        with self.f.patched(), self.f.owner(enabled=True) as context:
            with self.f.child(context) as permit:
                stop = self.root / 'runtime/framework/multitask_r10_STOP'
                stop.write_text('administrative drain')
                subject.require_permit(permit, self.root, allow_test_callbacks=True)
                with self.assertRaisesRegex(ValueError, 'STOP'):
                    with self.f.child(context, slot=1):
                        pass
            self.assertTrue(subject.permit_receipt(permit)['body_completed'])
            stop.unlink()

    def test_original_body_exception_not_masked_by_parent_loss(self):
        marker = RuntimeError('original worker failure')
        with self.f.patched(), self.f.owner(enabled=True) as context:
            final_patch = patch.object(subject, 'read_session', side_effect=ValueError('parent lost'))
            try:
                with self.assertRaises(RuntimeError) as caught:
                    with self.f.child(context) as permit:
                        final_patch.start()
                        raise marker
                self.assertIs(caught.exception, marker)
            finally:
                final_patch.stop()
            receipt = subject.permit_receipt(permit)
            self.assertTrue(receipt['restored'])
            self.assertEqual(receipt['original_exception_type'], 'RuntimeError')
            self.assertTrue(receipt['errors'])

    def test_metadata_scope_is_inert_exclusive_same_process_and_not_native(self):
        with self.f.patched():
            with self.assertRaisesRegex(ValueError, 'enabled=True'):
                with subject.metadata_scope(self.root):
                    pass
            with subject.metadata_scope(self.root, enabled=True) as permit:
                proof = subject.require_permit(permit, self.root, allow_test_callbacks=True)
                self.assertEqual(proof['purpose'], 'metadata_only')
                self.assertFalse(proof['Native_authorized'])
                self.assertFalse(proof['child_slots_authorized'])
                with self.assertRaises(ValueError):
                    subject.require_native_permit(permit, self.root)
            receipt = subject.permit_receipt(permit)
            self.assertTrue(receipt['restored'])
            self.assertTrue(receipt['body_completed'])
            self.assertNotIn(permit, self.f.parent.legacy._LEASES)

    def test_metadata_scope_blocks_old_owner_before_resource_acquisition(self):
        with self.f.patched(), patch.object(subject, '_other_controllers', return_value=[dict(pid=1)]):
            with self.assertRaisesRegex(ValueError, 'actually exit'):
                with subject.metadata_scope(self.root, enabled=True):
                    pass
        self.assertEqual(self.f.parent.entries, 0)

    def test_metadata_original_exception_preserved_and_capability_removed(self):
        marker = RuntimeError('metadata caller failed')
        with self.f.patched(), self.assertRaises(RuntimeError) as caught:
            with subject.metadata_scope(self.root, enabled=True) as permit:
                raise marker
        self.assertIs(caught.exception, marker)
        self.assertFalse(subject.permit_receipt(permit)['body_completed'])
        self.assertTrue(subject.permit_receipt(permit)['restored'])
        self.assertNotIn(permit, self.f.parent.legacy._LEASES)


if __name__ == '__main__':
    unittest.main()
