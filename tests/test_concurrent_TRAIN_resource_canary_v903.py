"""Pure fake scheduling/metadata and temporary files; no native/process/GPU probe."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import threading
import tempfile
import unittest
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[1] / 'scripts/run_concurrent_TRAIN_resource_canary_v903_r10.py'
spec = importlib.util.spec_from_file_location('canary903_test', SOURCE)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def pointer(name):
    return dict(path=name, sha256='a' * 64)


def fixture():
    control = dict(author_outcome_protocol={'schema': 'fake_frozen_protocol'},
        previous_CPU_receipts=[], training_bootstrap=dict(control=pointer('train_control.json'), lock=pointer('train_lock.json')))
    source_names = (m.BOOTSTRAP, 'our_mani/multitask/frozen_lookup_batch_v900.py',
        'scripts/run_formal_case_author_outcome_v751_r10.py', 'our_mani/multitask/formal_suite_admission_v728.py',
        'our_mani/multitask/rml_case_preflight_v742.py', m.WORKER, 'scripts/run_task_executor_smoke_author_suite16_v744_r10.py')
    control['execution_implementation'] = m.implementation_from_sources({name: pointer(name) for name in source_names})
    cases, cpus = [], {}
    for raw in m.RAWS:
        for method in m.METHODS:
            p = pointer('artifacts/framework/multitask_r10/author_outcome_qualification_v745_setup/' + raw + '_' + method + '_CPU_preflight.json')
            control['previous_CPU_receipts'].append(p)
            identity = dict(task=m.TASK, split='train', method=method, source_raw_episode_id=raw,
                source_row_sha256='b' * 64, sha256='c' * 64)
            cpus[p['path']] = dict(complete=True, native_started=False, GPU_started=False, actual_split='train',
                identity=identity, author_outcome_protocol=control['author_outcome_protocol'])
            argv = ['bash', 'scripts/run_aligned_gpu.sh' if method == 'physmani' else 'scripts/run_local_sim.sh', m.WORKER,
                '--config', 'frozen_config.json', '--manifest', m.MANIFEST, '--frozen-inputs', m.BASE_LOCK,
                '--task', m.TASK, '--raw-episode-id', raw, '--split', 'train', '--method', method,
                '--development-demo', '--out', 'old_frozen/development_demo_only/' + raw + '/' + method]
            cases.append(dict(task=m.TASK, raw_episode_id=raw, method=method, official_task_success=False,
                source_row_sha256='b' * 64, case=pointer('old/' + raw + '/' + method + '/case.json'),
                provenance=dict(CPU_receipt=p, actual_argv=argv)))
    return dict(cases=cases), cpus, control


def selected():
    inventory, cpu, control = fixture()
    return m.selected_rows(inventory, cpu, control), control


def cpu_receipt(row, control):
    batch = dict(schema='r10_scoped_frozen_lookup_bootstrap_receipt_v901', complete=True, entry='train', check_only=True,
        actual_outer_argv=row['argv'][2:] + ['--check'], forwarded_original_argv=row['physical_argv'][2:] + ['--check'],
        control=control['training_bootstrap']['control'], lock=control['training_bootstrap']['lock'],
        sources=control['execution_implementation']['sources'], scientific_policy_changed=False,
        full_verify_inputs_calls_retained=True, phases=[dict(phase='original742_prepare', boundary_recheck_complete=True,
            metrics=dict(complete=True, closed=True))])
    return dict(complete=True, native_started=False, GPU_started=False, actual_split='train', identity=row['expected_identity'],
        author_outcome_protocol=control['author_outcome_protocol'], component_domain=dict(purpose='train_qualification',
            split='train', method=row['method'], raw_episode_id=row['raw'], task=m.TASK), batch_execution=batch)


class SelectionTests(unittest.TestCase):
    def test_fixed_six_sources_preserve_all_prior_false_and_use_real901_train(self):
        inventory, cpu, control = fixture()
        rows = m.selected_rows(inventory, cpu, control)
        self.assertEqual(len(rows), 6)
        self.assertTrue(all(case['official_task_success'] is False for case in inventory['cases']))
        for row in rows:
            self.assertEqual(row['argv'][2], m.BOOTSTRAP)
            self.assertEqual(m.option(row['argv'], '--entry'), 'train')
            self.assertEqual(row['physical_argv'][2], m.WORKER)
            self.assertIn('development_demo_only', row['out'])
            self.assertEqual(m.option(row['physical_argv'], '--frozen-inputs'), m.BASE_LOCK)
        self.assertEqual({(r['raw'], r['method']) for r in rows}, {(r, method) for r in m.RAWS for method in m.METHODS})

    def test_missing_duplicate_and_wrong_source_rejected_without_padding(self):
        for mutation in ('missing', 'duplicate', 'wrong_raw'):
            inventory, cpu, control = fixture()
            if mutation == 'missing': inventory['cases'].pop()
            elif mutation == 'duplicate': inventory['cases'][-1] = deepcopy(inventory['cases'][0])
            else: inventory['cases'][-1]['raw_episode_id'] = 'episode2'
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                m.selected_rows(inventory, cpu, control)

    def test_changed_original_split_manifest_worker_or_demo_rejected(self):
        for field in ('--split', '--manifest', '--frozen-inputs', 'worker', 'demo'):
            inventory, cpu, control = fixture()
            argv = inventory['cases'][0]['provenance']['actual_argv']
            if field == 'worker': argv[2] = 'other.py'
            elif field == 'demo': argv.remove('--development-demo')
            else: argv[argv.index(field) + 1] = 'changed'
            with self.subTest(field=field), self.assertRaises(ValueError):
                m.selected_rows(inventory, cpu, control)

    def test_original_cpu_identity_and_protocol_must_match(self):
        for field in ('identity', 'author_outcome_protocol', 'native_started'):
            inventory, cpus, control = fixture()
            key = next(iter(cpus))
            if field == 'identity': cpus[key]['identity']['source_raw_episode_id'] = 'episode999'
            elif field == 'native_started': cpus[key][field] = True
            else: cpus[key][field] = {}
            with self.subTest(field=field), self.assertRaises(ValueError):
                m.selected_rows(inventory, cpus, control)

    def test_cpu_binding_requires_exact901_control_argv_domain_and_closed_scope(self):
        rows, control = selected()
        valid = cpu_receipt(rows[0], control)
        m.require_CPU_receipt(rows[0], valid, control)
        for mutation in ('GPU', 'identity', 'domain', 'control', 'outer', 'phase'):
            value = deepcopy(valid)
            if mutation == 'GPU': value['GPU_started'] = True
            elif mutation == 'identity': value['identity'] = {}
            elif mutation == 'domain': value['component_domain']['split'] = 'test'
            elif mutation == 'control': value['batch_execution']['control'] = pointer('formal_control.json')
            elif mutation == 'outer': value['batch_execution']['actual_outer_argv'] = []
            else: value['batch_execution']['phases'][0]['metrics']['closed'] = False
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                m.require_CPU_receipt(rows[0], value, control)

    def test_shared_implementation_has_no_future_formal_or_train_control(self):
        _, control = selected()
        implementation = control['execution_implementation']
        self.assertEqual(implementation['schema'], 'r10_resource_execution_implementation_v902')
        self.assertEqual(len(implementation['sources']), 7)
        self.assertNotIn('batch_control', implementation)
        self.assertNotIn('batch_lock', implementation)
        changed = dict(implementation['sources'], extra=pointer('extra'))
        with self.assertRaises(ValueError): m.implementation_from_sources(changed)


class SchedulingTests(unittest.TestCase):
    def run_fake(self, *, failure=False, stop_after_four=False, launch_error=False):
        rows, _ = selected()
        launched, interrupted, handled = [], [], []
        timeline = [0]
        def launch(row):
            if launch_error and len(launched) == 2: raise RuntimeError('launch failed')
            launched.append(row)
            return dict(pid=100 + len(launched), start_ticks=900 + len(launched), done=False,
                age=0, row=row, interrupted=False)
        live = {}
        def remember(row):
            value = launch(row); live[row['job_id']] = value; return value
        def pause():
            timeline[0] += 1
            for handle in live.values():
                handle['age'] += 1
                if handle['age'] >= (3 if handle['row']['method'] == 'physmani' else 1): handle['done'] = True
        def finish(row, handle):
            handled.append(row['job_id'])
            return dict(complete=not handle['interrupted'] and not (failure and row['method'] == 'cv_geometry'), native_success=False)
        def interrupt(handle):
            interrupted.append(handle['pid']);handle.update(done=True, interrupted=True)
        self.last = dict(launched=launched, interrupted=interrupted, handled=handled)
        result = m.schedule_native(rows, launch=remember, poll=lambda h: 0 if h['done'] else None,
            finish=finish, identify=lambda h: dict(pid=h['pid'], start_ticks=h['start_ticks']),
            interrupt=interrupt, pause=pause, should_stop=lambda: stop_after_four and len(launched) >= 4,
            clock=lambda: timeline[0])
        return result, launched, interrupted

    def test_real_four_way_request_overlap_and_two_phys_before_later_geometries(self):
        result, launched, interrupted = self.run_fake()
        self.assertTrue(result['complete'])
        self.assertEqual([(r['raw'], r['method']) for r in launched[:4]],
            [('episode0', 'physmani'), ('episode1', 'physmani'), ('episode0', 'cv_geometry'), ('episode0', 'context_geometry')])
        self.assertEqual([(r['raw'], r['method']) for r in launched[4:]],
            [('episode1', 'cv_geometry'), ('episode1', 'context_geometry')])
        self.assertEqual((result['observed_native_peak'], result['observed_GPU_peak']), (4, 2))
        self.assertTrue(all(v['native_success'] is False for v in result['results'].values()))
        self.assertTrue(all(e['native_slots'] <= 4 and e['GPU_slots'] <= 2 for e in result['events']))
        self.assertEqual(interrupted, [])

    def test_infrastructure_failure_drains_active_without_starting_remaining_or_retry(self):
        result, launched, interrupted = self.run_fake(failure=True)
        self.assertFalse(result['complete'])
        self.assertEqual(len(launched), 4)
        self.assertEqual(len(result['results']), 4)
        self.assertEqual(interrupted, [])
        self.assertEqual(len({r['job_id'] for r in launched}), 4)

    def test_STOP_interrupts_owned_active_and_never_launches_later_cases(self):
        result, launched, interrupted = self.run_fake(stop_after_four=True)
        self.assertFalse(result['complete'])
        self.assertEqual(len(launched), 4)
        self.assertEqual(len(interrupted), 4)

    def test_launch_exception_interrupts_only_its_owned_existing_children(self):
        with self.assertRaisesRegex(RuntimeError, 'launch failed'):
            self.run_fake(launch_error=True)
        self.assertEqual(len(self.last['launched']), 2)
        self.assertEqual(len(self.last['interrupted']), 2)


class HistoricalGuardTests(unittest.TestCase):
    def metadata(self):
        control = dict(policy_sha256='policy', author_outcome_protocol={'protocol': True})
        control.update({key: pointer(path) for key, path in m.PREVIOUS_PATHS.items()})
        metadata = dict(previous_launch=dict(pid=1, start_ticks=10, split='test', policy_sha256='policy', author_outcome_protocol={'protocol': True}),
            previous_controller_launch=dict(pid=2, start_ticks=20),
            previous_status=dict(split='test', expected=1920, failed={}, active=[], checking=[], completed={'a': {}}, drain_requested=True, stop_requested=False),
            previous_terminal=dict(complete=False, drained=True, drain_requested=True, stop_requested=False,
                expected=1920, completed=1, failed=0, automatic_retry=False, policy_sha256='policy'), previous_DRAIN={},
            previous_controller_status=dict(stage='requires_concrete_repair', automatic_retry=False),
            previous_controller_failure=dict(complete=False, user_stop=False, automatic_retry=False))
        return control, metadata

    def test_accepts_genuine_partial_drain_and827_failure_without775_or827_terminal(self):
        control, metadata = self.metadata()
        result = m.require_previous_exit(control, metadata, lambda pid: False)
        self.assertEqual(result['completed'], 1)
        self.assertFalse(any('775' in p['path'] for p in result['pointers'].values()))
        self.assertNotIn('previous_controller_terminal', result['pointers'])

    def test_active_previous_pid_or_interruption_or_failure_refuses(self):
        for mutation in ('alive', 'STOP', 'infra', 'active', 'controller_success'):
            control, metadata = self.metadata()
            if mutation == 'STOP': metadata['previous_status']['stop_requested'] = True
            elif mutation == 'infra': metadata['previous_status']['failed'] = {'x': {}}
            elif mutation == 'active': metadata['previous_status']['active'] = ['x']
            elif mutation == 'controller_success': metadata['previous_controller_failure']['complete'] = True
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                m.require_previous_exit(control, metadata, lambda pid: mutation == 'alive')

    def test_incomplete_inventory_or_any_remaining_native_refuses(self):
        valid = dict(host_visibility_available=True, same_uid_inventory_incomplete=False,
            live_project_native_processes=[], ancestor_chain_complete=True)
        m.require_empty_inventory(valid)
        for field, value in [('host_visibility_available', False), ('same_uid_inventory_incomplete', True),
                             ('ancestor_chain_complete', False), ('live_project_native_processes', [{'pid': 3}])]:
            with self.subTest(field=field), self.assertRaises(ValueError):
                m.require_empty_inventory(dict(valid, **{field: value}))


class ObservationTests(unittest.TestCase):
    def jobs(self):
        return [dict(job_id='phy0', method='physmani', pid=10, start_ticks=100),
                dict(job_id='phy1', method='physmani', pid=20, start_ticks=200)]

    def stats(self):
        return {11: dict(pid=11, ppid=10, start_ticks=110, state='R'), 10: dict(pid=10, ppid=1, start_ticks=100, state='S'),
                21: dict(pid=21, ppid=20, start_ticks=210, state='R'), 20: dict(pid=20, ppid=1, start_ticks=200, state='S'),
                99: dict(pid=99, ppid=0, start_ticks=990, state='R')}

    def test_two_actual_compute_processes_associate_by_real_pid_start_ancestry(self):
        def query(args):
            return 'GPU-actual, 4000, 16000, 95\n' if '--query-gpu=' in args[0] else '11, 1800, GPU-actual\n21, 1900, GPU-actual\n99, 1, GPU-actual\n'
        stats = self.stats()
        value = m.sample_resources(self.jobs(), query=query, stat_reader=lambda pid: stats[pid], clock=lambda: 44)
        self.assertTrue(value['GPU_observation_complete'])
        self.assertEqual(value['monotonic_ns'], 44)
        self.assertEqual([p['job_id'] for p in value['compute_apps']], ['phy0', 'phy1', None])
        self.assertEqual(value['compute_apps'][0]['pid_start_ticks'], 110)
        self.assertEqual(value['gpu']['memory_total_mib'], 16000)

    def test_pid_reuse_cannot_be_attributed_to_planned_launcher(self):
        stats = self.stats();stats[10]['start_ticks'] = 101;stats[1] = dict(pid=1, ppid=0, start_ticks=1, state='S')
        value = m.attribute_compute_pid(11, self.jobs(), lambda pid: stats[pid])
        self.assertIsNone(value['job_id'])

    def test_normal_running_sleeping_transition_preserves_real_process_identity(self):
        stats=self.stats();calls={}
        def read(pid):
            calls[pid]=calls.get(pid,0)+1
            value=dict(stats[pid])
            if pid==11 and calls[pid]>1:value['state']='S'
            return value
        value=m.attribute_compute_pid(11,self.jobs(),read)
        self.assertEqual(value['job_id'],'phy0')
        self.assertNotIn('ancestry_error',value)

    def test_unavailable_GPU_is_unknown_not_zero_or_verified(self):
        def unavailable(args): raise OSError('nvidia unavailable')
        value = m.sample_resources([], query=unavailable, clock=lambda: 55)
        self.assertFalse(value['GPU_observation_complete'])
        self.assertIsNone(value['gpu'])
        self.assertIn('GPU_observation_error', value)

    def test_unreadable_ancestry_keeps_unknown_job(self):
        def missing(pid): raise FileNotFoundError(pid)
        value = m.attribute_compute_pid(777, self.jobs(), missing)
        self.assertIsNone(value['job_id'])
        self.assertIsNone(value['pid_start_ticks'])

    def test_CPU_RAM_reader_is_injected_without_real_proc_read(self):
        def query(args): return 'GPU, 15000, 16000, 50\n' if '--query-gpu=' in args[0] else ''
        value = m.sample_resources([], query=query, cpu_reader=lambda: {'MemAvailable_kib': 1024})
        self.assertEqual(value['host_CPU_RAM'], {'MemAvailable_kib': 1024})
        # Low headroom remains an actual measurement. This runner signs no qualification.
        self.assertEqual(value['gpu']['memory_total_mib'] - value['gpu']['memory_used_mib'], 1000)
        self.assertNotIn('resource_bounds_verified', value)

    def test_sampler_owns_only_fake_observer_thread_and_create_only_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'samples.jsonl'
            fake=dict(monotonic_ns=1,gpu=None,compute_apps=[],GPU_observation_complete=False)
            with patch.object(m,'sample_resources',return_value=fake):
                with m.ResourceSampler(path) as sampler:
                    sampler.observe(self.jobs())
                self.assertFalse(sampler.thread.is_alive())
                self.assertIsNone(sampler.failure)
            self.assertGreaterEqual(len(path.read_text().splitlines()),1)
            self.assertEqual(json.loads(path.read_text().splitlines()[0]),fake)
            with self.assertRaises(FileExistsError):
                with m.ResourceSampler(path): pass

    def test_unexpected_sampler_thread_exception_cannot_be_silent_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(m,'sample_resources',side_effect=RuntimeError('observer failed')):
                with self.assertRaisesRegex(ValueError,'observer failed'):
                    with m.ResourceSampler(Path(tmp)/'samples.jsonl'): pass


class WriteSafetyTests(unittest.TestCase):
    def test_evidence_is_create_only_and_import_does_not_load_native(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'record.json'
            m.atomic_new(path, {'one': 1})
            with self.assertRaises(FileExistsError): m.atomic_new(path, {'one': 2})
            self.assertEqual(json.loads(path.read_bytes()), {'one': 1})
        # Every production side effect remains in an explicitly called function.
        self.assertFalse(hasattr(m, 'ENVIRONMENT_RECEIPT'))


class ControlLockTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.patch = patch.object(m, 'ROOT', self.root); self.patch.start(); self.addCleanup(self.patch.stop)
        self.setup = self.root / 'artifacts/framework/multitask_r10/canary_fake_setup'
        self.setup.mkdir(parents=True)
        self.control_path, self.lock_path = self.setup / 'control.json', self.setup / 'lock.json'
        for name in m.SOURCES:
            path = self.root / name; path.parent.mkdir(parents=True, exist_ok=True);path.write_text('# frozen CPU fake\n')
        inventory, cpu, control = fixture()
        control.update(schema=m.CONTROL_SCHEMA, execution_enabled=True, split='train', task=m.TASK,
            raw_episode_ids=list(m.RAWS), resource_profile=deepcopy(m.PROFILE), training_weights_updated=False,
            formal_score_contribution=False, TEST_read=False, common_RML_profile_sha256='d' * 64,
            inventory=pointer(m.INVENTORY), resource_qualifier=m.point(m.AUDITOR))
        def write(name, value):
            path=self.root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value));return m.point(path)
        control['inventory'] = write(m.INVENTORY, inventory)
        for p in control['previous_CPU_receipts']:
            actual = write(p['path'], cpu[p['path']]);p.update(actual)
        for key, name in m.PREVIOUS_PATHS.items(): control[key] = write(name, {})
        control['training_bootstrap'] = dict(control=write('artifacts/framework/multitask_r10/fake_train_bootstrap/control.json', {}),
            lock=write('artifacts/framework/multitask_r10/fake_train_bootstrap/lock.json', {}))
        control['environment_manifest'] = write('artifacts/framework/multitask_r10/fake_environment/manifest.json', {})
        control['environment_namespace_proof'] = write('artifacts/framework/multitask_r10/fake_environment/proof.json', {})
        control['environment_wrapper'] = m.point('scripts/run_frozen_system_environment_v820_r10.py')
        control['execution_implementation'] = m.implementation_from_sources({name:m.point(name) for name in control['execution_implementation']['sources']})
        self.control = control

    def freeze(self, extra=None):
        self.control_path.write_text(json.dumps(self.control))
        names = [self.control_path, *(self.root/name for name in m.SOURCES), self.root/m.INVENTORY]
        pointers = [*self.control['training_bootstrap'].values(), *(self.control[k] for k in m.PREVIOUS_PATHS),
            *(self.control[k] for k in ('environment_manifest','environment_wrapper','environment_namespace_proof')),
            *self.control['previous_CPU_receipts']]
        names.extend(self.root/p['path'] for p in pointers)
        files = {str(path.resolve()):m.sha(path.read_bytes()) for path in names}
        files.update(extra or {})
        self.lock_path.write_text(json.dumps(dict(schema=m.LOCK_SCHEMA,execution_enabled=True,files=files)))
        return m.sha(self.lock_path.read_bytes())

    def load(self, expected=None):
        return m.load_control(str(self.control_path),str(self.lock_path),expected or m.sha(self.lock_path.read_bytes()))

    def test_small_explicit_source_metadata_lock_accepts_without_native_import(self):
        expected=self.freeze();control,binding=self.load(expected)
        self.assertEqual(control,self.control)
        self.assertEqual(binding['lock']['sha256'],expected)
        self.assertEqual(control['resource_qualifier']['path'],m.AUDITOR)

    def test_unpinned_disabled_or_changed_control_source_refuses(self):
        self.freeze()
        with self.assertRaises(ValueError): self.load('0'*64)
        self.control['execution_enabled']=False;self.freeze()
        with self.assertRaises(ValueError): self.load()
        self.control['execution_enabled']=True;self.freeze()
        (self.root/m.SOURCE).write_text('# changed\n')
        with self.assertRaisesRegex(ValueError,'changed'): self.load()

    def test_auditor_and_common_profile_are_required_before_run(self):
        for field,value in [('resource_qualifier',pointer('unrelated.py')),('common_RML_profile_sha256',None)]:
            old=self.control[field];self.control[field]=value;self.freeze()
            with self.subTest(field=field),self.assertRaises(ValueError): self.load()
            self.control[field]=old

    def test_unexpected_raw_or_model_bytes_are_never_read(self):
        raw=self.root/'fake_model.hdf5';raw.write_bytes(b'no raw reads')
        self.freeze({str(raw):m.sha(raw.read_bytes())})
        original=Path.read_bytes
        def guarded(path):
            self.assertNotEqual(path,raw,'raw file was read')
            return original(path)
        with patch.object(Path,'read_bytes',guarded),self.assertRaisesRegex(ValueError,'no raw/model/TEST'):
            self.load()

    def test_external_control_dependency_is_rejected_before_bytes(self):
        external=Path(self.temp.name).parent/'nonexistent_outside_903_test'
        # Resolving a missing dependency itself refuses; no external data is read.
        self.freeze({str(external):'0'*64})
        with self.assertRaises((ValueError,FileNotFoundError)):self.load()


class CPUQuotaTests(unittest.TestCase):
    def test_all_six_CPU_checks_use_at_most_two_workers(self):
        rows,control=selected()
        barrier=threading.Barrier(2)
        lock=threading.Lock();state=dict(active=0,peak=0,finished=0)
        def launch(row,check,event,environment):
            self.assertTrue(check)
            with lock:state['active']+=1;state['peak']=max(state['peak'],state['active'])
            barrier.wait(timeout=2)
            return {'row':row}
        def finish(row,handle,*args):
            with lock:state['active']-=1;state['finished']+=1
            return {'complete':True}
        with patch.object(m,'launch_child',launch),patch.object(m,'poll_child',return_value=0), \
                patch.object(m,'finish_child',finish),patch.object(m,'interrupt_child'),patch.object(m,'stopped',return_value=False):
            results=m.run_CPU(rows,control,threading.Event(),{})
        self.assertEqual((state['peak'],state['finished'],len(results)),(2,6,6))


if __name__ == '__main__':
    unittest.main()
