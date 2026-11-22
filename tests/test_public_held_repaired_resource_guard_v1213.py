"""CPU-only capability/locking fixtures. No simulator or GPU is imported."""
from contextlib import contextmanager, ExitStack
from copy import deepcopy
import fcntl
import errno
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from weakref import WeakKeyDictionary

from our_mani.multitask import public_held_repaired_resource_guard_v1213 as subject


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
            self.assertTrue(receipt['parent_broker_admission']['parent_FD_verified'])
            self.assertEqual(receipt['parent_broker_admission']['parent_kernel_verification'], 'per_FD_fdinfo')
            self.assertTrue(receipt['parent_broker_admission']['per_FD_lock_evidence'][0]['FD_stat_before_after_verified'])
            self.assertEqual(receipt['parent_broker_admission']['independent_conflict_evidence'][0]['conflict_errno'], errno.EAGAIN)
            self.assertFalse(receipt['parent_broker_admission']['client_direct_parent_FD_read'])
            self.assertEqual(receipt['parent_broker_admission']['claim'], receipt['claim'])
            self.assertNotEqual(receipt['parent_broker_admission']['nonce'], receipt['parent_broker_final']['nonce'])
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
            session = json.loads((self.root / context['session']['path']).read_bytes())
            fd = session['parent_global_locks'][0]['fd']
            fcntl.flock(fd, fcntl.LOCK_UN)
            try:
                with self.assertRaisesRegex(ValueError, 'FLOCK'):
                    subject.read_session(self.root, context['session'])
            finally:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def test_global_table_visibility_is_not_an_authorization_dependency(self):
        with self.f.patched(), patch.object(subject, '_flock_rows', side_effect=AssertionError('global iterator forbidden')):
            with self.f.owner(enabled=True) as context:
                with self.f.child(context) as permit:
                    proof = subject.require_permit(permit, self.root, allow_test_callbacks=True)
                    self.assertEqual(proof['parent_broker_admission']['parent_kernel_verification'], 'per_FD_fdinfo')
                self.assertTrue(subject.permit_receipt(permit)['body_completed'])

    def test_six_actual_bwrap_CPU_slots_overlap_and_close_without_native_capability(self):
        jobs=[]
        for index in range(6):
            jobs.append(dict(self.f.job, job_id='j'+str(index).zfill(4),
                out=self.f.job['out']+'/slot'+str(index), runtime=self.f.job['runtime']+'/slot'+str(index)))
        self.f.plan['jobs']=jobs
        self.f.plan_ref=self.f.write(self.f.runtime+'/plan.json',self.f.plan)
        code=r'''
import json,os,sys,time
from pathlib import Path
from unittest.mock import patch
from our_mani.multitask import public_held_repaired_resource_guard_v1213 as g
from tests.test_public_held_repaired_resource_guard_v1213 import FakeParent
root=Path(sys.argv[1]);ref=json.loads(sys.argv[2]);slot=int(sys.argv[3])
session=json.loads((root/ref['path']).read_bytes());graph=json.loads((root/session['graph']['path']).read_bytes())
with patch.object(g,'_parent',return_value=FakeParent()),patch.object(g,'_graph',return_value=graph):
    with g.child_scope(root,session_ref=ref,job=session['jobs'][slot],phase='CPU_check',slot=slot,enabled=True) as permit:
        proof=g.require_permit(permit,root,allow_test_callbacks=True)
        (root/('entered_'+str(slot))).write_text(str(os.getpid()))
        deadline=time.monotonic()+15
        while not all((root/('entered_'+str(i))).exists() for i in range(6)):
            if time.monotonic()>deadline:raise RuntimeError('six CPU-only slot rendezvous did not complete')
            time.sleep(.01)
        assert proof['test_only'] is True and proof['Native_authorized'] is False
    receipt=g.permit_receipt(permit)
    print(json.dumps(dict(process=receipt['process'],slot=receipt['slot'],test_only=receipt['test_only'],
        Native_authorized=receipt['Native_authorized'],body_completed=receipt['body_completed'],
        restored=receipt['restored'],leases_removed=receipt['leases_removed'],errors=receipt['errors'],
        nonce_distinct=receipt['parent_broker_admission']['nonce']!=receipt['parent_broker_final']['nonce'])))
'''
        project=Path(__file__).resolve().parents[1]
        with self.f.patched(), self.f.owner(enabled=True) as context:
            processes=[subprocess.Popen(['/usr/bin/bwrap','--bind','/','/','--dev-bind','/dev','/dev','--',
                sys.executable,'-B','-c',code,str(self.root),json.dumps(context['session']),str(i)],
                cwd=project,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True) for i in range(6)]
            results=[]
            for process in processes:
                out,err=process.communicate(timeout=30)
                self.assertEqual(process.returncode,0,err)
                results.append(json.loads(out))
            self.assertEqual({v['slot'] for v in results},set(range(6)))
            self.assertEqual(len({v['process']['pid'] for v in results}),6)
            for row in results:
                self.assertTrue(row['test_only'])
                self.assertFalse(row['Native_authorized'])
                self.assertTrue(row['body_completed'] and row['restored'] and row['leases_removed'] and row['nonce_distinct'])
                self.assertEqual(row['errors'],[])

    def test_independent_conflict_never_reacquires_or_leaks_a_missing_lock(self):
        file = self.root / 'only_probe.lock'
        with file.open('x+b') as stream:
            stat = os.fstat(stream.fileno())
            rows = [dict(path='only_probe.lock', device=stat.st_dev, inode=stat.st_ino, fd=stream.fileno())]
            with self.assertRaisesRegex(ValueError, 'unexpectedly acquired'):
                subject._independent_conflicts(self.root, rows)
            # The failed check must have immediately unlocked its independent descriptor.
            with file.open('r+b') as other:
                fcntl.flock(other.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                fcntl.flock(other.fileno(), fcntl.LOCK_UN)
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            proof = subject._independent_conflicts(self.root, rows)
            self.assertEqual(proof[0]['conflict_errno'], errno.EAGAIN)
            self.assertTrue(proof[0]['confirmed'])

    def test_per_FD_record_rejects_wrong_owner_mode_range_or_file(self):
        row = dict(path='f.lock',device=os.makedev(8,1),inode=7,fd=3)
        good = 'lock:\t1: FLOCK ADVISORY WRITE '+str(os.getpid())+' 08:01:7 0 EOF'
        self.assertEqual(subject._parse_fdinfo_lock(good,subject._process(),row)['inode'],7)
        for bad in (good.replace('WRITE','READ'),good.replace('0 EOF','1 EOF'),
                good.replace('08:01:7','08:01:8'),good.replace('WRITE '+str(os.getpid()),'WRITE 1')):
            with self.assertRaises(ValueError):subject._parse_fdinfo_lock(bad,subject._process(),row)

    def test_actual_graph_guard_source_contract_refuses_old_graph_and_missing_parent(self):
        with patch.object(subject, '__file__', str(self.root / subject.SOURCE)):
            self.assertEqual(subject._graph(self.root, self.f.graph_ref), self.f.graph)
            graph = deepcopy(self.f.graph)
            graph['control']['components']['resource_guard'] = subject._source(self.root, subject.PARENT_SOURCE)
            ref = self.f.write(self.f.graph_path, graph)
            with self.assertRaisesRegex(ValueError, 'new1213'):
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
        with self.f.patched(), self.assertRaisesRegex(ValueError, 'broker unavailable'):
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

    def test_broker_rejects_forged_session_or_claim_and_keeps_slot_check_strict(self):
        with self.f.patched(), self.f.owner(enabled=True) as context:
            session = json.loads((self.root / context['session']['path']).read_bytes())
            peer = dict(process=subject._process(), uid=os.getuid(), gid=os.getgid())
            request = dict(schema=subject.BROKER_REQUEST_SCHEMA, session=context['session'],
                process=subject._process(), boot_id=subject._boot(), nonce='1' * 64, claim=None)
            forged = deepcopy(request)
            forged['session']['sha256'] = '0' * 64
            with self.assertRaisesRegex(ValueError, 'request binding'):
                subject._broker_response(self.root, session, context['session'], forged, peer)
            with self.f.child(context) as permit:
                receipt = subject.permit_receipt(permit)
                claim_ref = receipt['claim']
                with self.assertRaisesRegex(ValueError, 'immutable'):
                    subject._verify_parent_locks(self.root, session, context['session'],
                        claim=dict(claim_ref, sha256='0' * 64))
                with patch.object(subject, '_check_locks', side_effect=ValueError('actual slot FLOCK lost')):
                    with self.assertRaises(ValueError):
                        subject.require_permit(permit, self.root, allow_test_callbacks=True)


class MechanicalGuardMappingContracts(unittest.TestCase):
    def test_guard_and_probe_only_apply_declared_version_and_namespace_mapping(self):
        root=Path(__file__).resolve().parents[1]
        mappings=(('1195','1213'),('1196','1206'),('1197','1207'),
            ('parallel_repaired_resource_guard_v1213','public_held_repaired_resource_guard_v1213'),
            ('parallel_repaired_owner_v1207','public_held_repaired_owner_v1207'),
            ('parallel_repaired_policy_v1206','public_held_repaired_policy_v1206'),
            ('probe_parallel_parent_locks_v1213','probe_public_held_parent_locks_v1213'))
        import ast
        for old,new in (('our_mani/multitask/parallel_repaired_resource_guard_v1195.py',subject.SOURCE),
                ('scripts/probe_parallel_parent_locks_v1195_r10.py','scripts/probe_public_held_parent_locks_v1213_r10.py')):
            expected=(root/old).read_text()
            for before,after in mappings:expected=expected.replace(before,after)
            actual=(root/new).read_text()
            with self.subTest(source=new):
                self.assertEqual(ast.dump(ast.parse(actual)),ast.dump(ast.parse(expected)))
                self.assertEqual(actual,expected)
        self.assertEqual(subject.BASE,'runtime/framework/public_held_repaired_owner_v1207')
        self.assertEqual(subject.ART,'artifacts/framework/multitask_r10/public_held_repaired_policy_v1206')
        self.assertTrue(subject.BROKER_SCHEMA.endswith('_v1213'))


class ActualRootlessBrokerProbes(unittest.TestCase):
    """Real OS processes and kernel locks; only fabricated scenarios are explicit negatives."""
    def probe(self, scenario, *, frozen820=False):
        from scripts.probe_public_held_parent_locks_v1213_r10 import run_probe
        return run_probe(frozen820=frozen820, scenario=scenario)

    def test_actual_frozen820_cross_user_namespace_EACCES_and_live_broker_success(self):
        result = self.probe('success', frozen820=True)
        self.assertEqual(result['direct_parent_FD'], 'EACCES')
        self.assertTrue(result['broker_passed'])
        self.assertTrue(result['fresh_distinct_nonce'])
        self.assertTrue(result['per_FD_records_all_valid'])
        self.assertTrue(result['independent_conflicts_all_confirmed'])
        self.assertEqual(len(result['admission']['per_FD_lock_evidence']),11)
        self.assertTrue(result['namespace']['actual820_namespace_verified'])
        self.assertNotEqual(result['namespace']['current_namespaces']['mnt'],
            result['namespace']['reference_namespaces']['mnt'])
        self.assertEqual(result['namespace']['current_namespaces']['pid'],
            result['namespace']['reference_namespaces']['pid'])
        self.assertEqual(result['namespace']['current_namespaces']['net'],
            result['namespace']['reference_namespaces']['net'])
        self.assertFalse(result['Native_started'])
        self.assertFalse(result['GPU_started'])

    def test_actual_bwrap_parent_unlock_fails_closed(self):
        result = self.probe('parent_unlock')
        self.assertEqual(result['direct_parent_FD'], 'EACCES')
        self.assertFalse(result['broker_passed'])
        self.assertIn('FLOCK', result['error'])

    def test_actual_bwrap_wrong_server_PID_fails_peer_authentication(self):
        result = self.probe('wrong_server_PID')
        self.assertEqual(result['direct_parent_FD'], 'EACCES')
        self.assertFalse(result['broker_passed'])
        self.assertIn('SO_PEERCRED', result['error'])

    def test_actual_bwrap_nonce_mismatch_fails_live_proof(self):
        result = self.probe('nonce_mismatch')
        self.assertEqual(result['direct_parent_FD'], 'EACCES')
        self.assertFalse(result['broker_passed'])
        self.assertIn('nonce', result['error'])

    def test_actual_bwrap_churn_preserves_per_FD_and_independent_conflicts(self):
        result=self.probe('churn')
        self.assertTrue(result['broker_passed'])
        self.assertTrue(result['per_FD_records_all_valid'])
        self.assertTrue(result['independent_conflicts_all_confirmed'])
        self.assertEqual(len(result['churn_children']),4)
        # Whole-table misses are diagnostic; their occurrence is scheduling-dependent.
        self.assertGreaterEqual(result['global_table_diagnostic_missing_rows'],0)


if __name__ == '__main__':
    unittest.main()
