"""CPU/temp-only synthetic metadata; no real TEST, process, lock or Native API."""
from copy import deepcopy
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

PROJECT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('report959_fake', PROJECT/'scripts/report_composite_TEST_halfhour_v959_r10.py')
m = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(m)
CORE_BYTES = (PROJECT/m.CORE).read_bytes()  # Source only, never a dataset/lock.


def ptr(name, value='a'):
    return dict(path=name, sha256=value*64)


def original_plan():
    rows = []
    driver = ptr('scripts/run_formal_case_author_outcome_v751_r10.py', 'b')
    for task in m.TASK_LABELS:
        for index in range(40):
            for method in m.METHODS:
                raw = 'variation3_episode'+str(index+70)
                out = 'artifacts/framework/multitask_r10/synthetic753/test/'+task+'/'+raw+'/'+method
                rows.append(dict(job_id=task+'--'+raw+'--'+method, task=task, raw=raw, method=method,
                    worker_sha256=driver['sha256'], admission_driver=driver,
                    argv=['bash', 'scripts/run_aligned_gpu.sh' if method == 'physmani' else 'scripts/run_local_sim.sh',
                        driver['path'], '--split', 'test', '--out', out]))
    return dict(split='test', expected_cases=1920, source_outcome_filtering=False, automatic_retry=False, jobs=rows)


def make_parent(original, entry):
    result = deepcopy(original)
    for index, row in enumerate(result['jobs']):
        previous = original['jobs'][index]; imported = index < 134
        row.update(evidence_runtime=m.ORIGINAL if imported else m.PARENT, imported_completed=imported,
            previous_row_sha256=m.digest(previous), previous_worker_sha256=previous['worker_sha256'])
        if imported:
            row.update(physical_argv=deepcopy(previous['argv']), execution_entry=dict(kind='original751', pointer=previous['admission_driver']))
        else:
            inner = deepcopy(previous['argv'])
            inner[-1] = str(Path('artifacts/framework/multitask_r10/formal_TEST_concurrent_resume_v902/test')/row['task']/row['raw']/row['method'])
            outer = [*inner[:2], entry['pointer']['path'], '--batch-control', entry['batch_control']['path'],
                '--batch-lock', entry['batch_lock']['path'], '--batch-lock-sha256', entry['batch_lock']['sha256'],
                '--entry', 'formal', '--', *inner[3:]]
            row.update(physical_argv=inner, argv=outer, execution_entry=entry, worker_sha256=entry['pointer']['sha256'])
    return result


def outcome(row, native=False, motion=None):
    argv = row.get('physical_argv', row['argv']); out = argv[argv.index('--out')+1]
    value = dict(complete=True, native_success=native, motion_pass=motion, returncode=0, timed_out=False,
        interrupted=False, automatic_retry=False, case=ptr(str(Path(out)/'case.json')),
        admission_driver_receipt=ptr(str(Path(out)/'admission_driver_receipt.json')))
    if row.get('evidence_runtime') not in (None, m.ORIGINAL):
        value['batch_driver_receipt'] = ptr(str(Path(out)/'batch_driver_receipt.json'))
    return value


class Fixture:
    def __init__(self, *, full=False, extra=2):
        self.store, self.reads = {}, []
        for name in m.SOURCES: self.store[name] = CORE_BYTES if name == m.CORE else ('# synthetic '+name+'\n').encode()
        self.bindings = {name: self.point(name) for name in m.SOURCES}
        self.policy = 'e'*64
        self.protocol = dict(schema='r10_author_action_outcome_protocol_v741', native_RML_parameters_unchanged=True,
            pose_action_max_ticks=10, public_hold_max_ticks=1)
        self.base = dict(policy_sha256=self.policy, author_outcome_protocol=self.protocol, automatic_retry=False)
        self.entry = dict(kind='batched901', pointer=ptr('scripts/run_batched_formal_driver_v901_r10.py', 'c'),
            batch_control=ptr('artifacts/framework/multitask_r10/synthetic901/control.json'),
            batch_lock=ptr('artifacts/framework/multitask_r10/synthetic901/lock.json'))
        original = original_plan(); original['policy_sha256'] = self.policy
        parent = make_parent(original, self.entry)
        original_records = {r['job_id']: outcome(r) for r in original['jobs'][:134]}
        parent_records = {r['job_id']: outcome(r) for r in parent['jobs'][:584]}
        parent_records.update(deepcopy(original_records))
        oldfailed = {r['job_id']: dict(complete=False, automatic_retry=False, returncode=1, phase='CPU_preflight') for r in parent['jobs'][584:590]}
        self.original = dict(launch=dict(self.base, pid=101, start_ticks=1001, source_sha256=self.bindings[m.ORIGINAL_SOURCE]['sha256']),
            plan=original, status=dict(split='test', completed=original_records), terminal=dict(complete=False, completed=134, failed=0, drained=True))
        self.parent = dict(launch=dict(self.base, pid=102, start_ticks=1002, source_sha256=self.bindings[m.PARENT_SOURCE]['sha256']), plan=parent,
            status=dict(split='test', expected=1920, completed=parent_records, failed=oldfailed, active=[], checking=[], native_slots=0,
                GPU_slots=0, drain_requested=True, stop_requested=False, automatic_retry=False),
            terminal=dict(complete=False, expected=1920, completed=584, failed=6, drained=True, drain_requested=True,
                stop_requested=False, automatic_retry=False, policy_sha256=self.policy))
        for runtime, values in ((m.ORIGINAL, self.original), (m.PARENT, self.parent)):
            for key, value in values.items(): self.put(runtime+'/'+key+'.json', value)
        old_control = dict(policy_sha256=self.policy, completed_job_ids=list(original_records),
            remaining_job_ids=[r['job_id'] for r in original['jobs'][134:]], execution_entry=self.entry)
        for key in self.original: old_control['previous_'+key] = self.point(m.ORIGINAL+'/'+key+'.json')
        self.put(m.OLD_CONTROL, old_control)
        self.put(m.OLD_LOCK, dict(files={p['path']: p['sha256'] for p in [self.point(m.OLD_CONTROL), *(old_control['previous_'+k] for k in self.original)]}))
        self.control = dict(self.base, schema='r10_composite_formal_TEST_resume_control_v955', split='test', expected_cases=1920,
            source_count=640, user_authorized_resume=True, standing_authorization_reused=True, new_user_event_invented=False,
            physical_policy_changed=False, completed_job_ids=list(parent_records), failed_job_ids=list(oldfailed),
            remaining_job_ids=[r['job_id'] for r in parent['jobs'][584:]], resource_profile=m.PROFILE,
            execution_entry=self.entry, execution_implementation=dict(kind='batched901', pointer=self.entry['pointer']),
            source_inputs=ptr('artifacts/framework/multitask_r10/synthetic/test_source_inputs.json'),
            validation_release=ptr('artifacts/framework/multitask_r10/synthetic/VAL_release.json'),
            resource_qualification=ptr('runtime/framework/synthetic904.json'),
            environment_manifest=ptr('artifacts/framework/multitask_r10/synthetic820/manifest.json'),
            environment_wrapper=ptr('scripts/run_frozen_system_environment_v820_r10.py'),
            environment_namespace_proof=ptr('artifacts/framework/multitask_r10/synthetic820/proof.json'),
            previous_resume_control=self.point(m.OLD_CONTROL), previous_resume_control_lock=self.point(m.OLD_LOCK),
            resume_owner=self.bindings[m.OWNER_SOURCE], resume_controller=self.bindings[m.CONTROLLER_SOURCE], resume_auditor=self.bindings[m.AUDITOR_SOURCE])
        for prefix, runtime, values in (('original_', m.ORIGINAL, self.original), ('previous_', m.PARENT, self.parent)):
            for key in values: self.control[prefix+key] = self.point(runtime+'/'+key+'.json')
        for key in ('completed', 'failed', 'remaining'):
            self.control[key+'_key_sha256'] = m.digest(sorted(self.control[key+'_job_ids']))
        self.env = self.environment(201, 2001)
        recheck = dict(schema='r10_fresh_frozen_TEST_environment_CPU_observation_v954', complete=True,
            source=self.bindings[m.CHECK_SOURCE], Native_started=False, GPU_started=False, admission_or_TEST_started=False,
            declared_model_case_raw_bytes_read=False, scientific_inputs_or_old_locks_modified=False,
            original_v900_topology_constructor_passed=True, original_full_topology_final_check_passed=True,
            parsed_suite_lock_is_exact_hashed_payload=True, topology_is_point_observation_not_continuous=True,
            suite_lock=ptr('artifacts/framework/multitask_r10/synthetic/suite_lock.json'), declared_files=130292,
            unique_canonical_paths=130292, watched_directories=10, actual_environment=self.env)
        self.control['suite_lock'] = recheck['suite_lock']
        self.put(m.RECHECK, recheck); self.control['environment_recheck'] = self.point(m.RECHECK)
        self.put(m.CONTROL, self.control)
        selected = [self.point(m.CONTROL), *self.bindings.values(), self.point(m.OLD_CONTROL), self.point(m.OLD_LOCK),
            *(self.control['previous_'+key] for key in self.parent)]
        self.put(m.CONTROL_LOCK, dict(schema='r10_composite_formal_TEST_resume_inputs_v955', approved_for_execution=True,
            files={p['path']: p['sha256'] for p in selected}))
        self.retirement()
        self.config = dict(schema=m.SCHEMA, protocol_status='formal_TEST_composite', runtime=m.RUNTIME, controller=m.CONTROLLER,
            test_results=m.RESULTS, source_bindings=self.bindings, policy_sha256=self.policy, author_outcome_protocol=self.protocol,
            resume_control=self.point(m.CONTROL), resume_control_lock=self.point(m.CONTROL_LOCK),
            legacy_retirement=self.point(m.RETIREMENT), environment_recheck=self.point(m.RECHECK))
        self.freeze_config()
        core = m.load_core(m.Metadata(read_bytes=self.read), self.config)
        self.plan = dict(self.base, split='test', expected_cases=1920, imported_completed_cases=584,
            source_outcome_filtering=False, jobs=core.resumed_jobs(parent, self.control, self.entry))
        self.common = dict(self.base, split='test', resume_control=self.config['resume_control'], resume_control_lock=self.config['resume_control_lock'],
            resource_profile=m.PROFILE, frozen_environment=self.environment(202, 2002),
            **{key:self.control[key] for key in ('previous_resume_control', 'previous_resume_control_lock', 'environment_recheck')})
        self.plan.update(self.common)
        self.owner = dict(self.common, pid=202, start_ticks=2002, source_sha256=self.bindings[m.OWNER_SOURCE]['sha256'],
            argv=[m.OWNER_SOURCE, '--resume-control', m.CONTROL], base_stage_owner=self.bindings[m.ORIGINAL_SOURCE], parent_stage_owner=self.bindings[m.PARENT_SOURCE])
        count = 1920 if full else 584+extra
        records = {r['job_id']: outcome(r) for r in self.plan['jobs'][:count]}; records.update(deepcopy(parent_records))
        self.status = dict(self.common, expected=1920, completed=records, failed={}, active=[], checking=[], ready_jobs=[],
            CPU_ready=0, native_slots=0, GPU_slots=0, owner_error=None, drain_requested=False, stop_requested=False, updated_utc='synthetic')
        self.put(m.RUNTIME+'/launch.json', self.owner); self.put(m.RUNTIME+'/plan.json', self.plan); self.put(m.RUNTIME+'/status.json', self.status)
        self.put(m.CONTROLLER+'/launch.json', dict(pid=201, start_ticks=2001, source_sha256=self.bindings[m.CONTROLLER_SOURCE]['sha256'],
            resume_control=self.config['resume_control'], resume_control_lock=self.config['resume_control_lock'], frozen_environment=self.env, automatic_retry=False))
        self.put(m.CONTROLLER+'/status.json', dict(stage='complete' if full else 'formal_TEST_1920', frozen_environment=self.env, automatic_retry=False))
        self.put(m.CONTROLLER+'/stages/formal_TEST_1920/launch.json', self.stage(m.OWNER_SOURCE, 202, 2002, self.owner['argv']))
        if full: self.complete_result()
        self.reads.clear()

    def point(self, name): return dict(path=name, sha256=m.sha(self.store[name]))
    def put(self, name, value): self.store[name] = m.encoded(value); return self.point(name)
    def read(self, name):
        self.reads.append(name)
        if name not in self.store: raise FileNotFoundError(name)
        return self.store[name]
    def environment(self, pid, start):
        return dict(schema='r10_actual_frozen_OS_namespace_receipt_v820', complete=True, checked_OS_files=6418,
            all_current_OS_targets_full_byte_SHA_verified=True, observer_pid=pid, observer_start_ticks=start,
            current_mount_namespace='mnt:[synthetic]', manifest=self.control['environment_manifest'],
            wrapper=self.control['environment_wrapper'], namespace_qualification=self.control['environment_namespace_proof'])
    def stage(self, source, pid, start, argv):
        return dict(pid=pid, start_ticks=start, source_sha256=self.bindings[source]['sha256'],
            argv=[str(m.ROOT/'.conda-env/bin/python'), '-B', '-u', *argv], inherited_frozen_environment=self.env,
            current_mount_namespace=self.env['current_mount_namespace'], automatic_retry=False)
    def retirement(self):
        for key, name in m.RETIREMENT_FILES.items():
            if name not in self.store:
                self.store[name] = b'{"summary":"actual922 historical infra6"}\n' if key == 'history' else b'{}\n'
        self.put(m.RETIREMENT_FILES['launch'], dict(pid=99, start_ticks=999, source_sha256=self.point(m.LEGACY_SOURCE)['sha256'],
            source_config_sha256=self.point(m.RETIREMENT_FILES['source_config'])['sha256'],
            source_lock_sha256=self.point(m.RETIREMENT_FILES['source_lock'])['sha256']))
        value = dict(schema=m.RETIREMENT_SCHEMA, complete=True, observer_only=True, private_stop_only=True, exit_observed=True,
            experiment_STOP_written=False, session_id=123, exit_returncode=0, observed_exit_utc='2026-10-21T23:10:00Z', pid=99, start_ticks=999,
            **{key: self.point(name) for key, name in m.RETIREMENT_FILES.items()})
        self.put(m.RETIREMENT, value)
    def freeze_config(self):
        self.put(m.STATE+'/source.json', self.config)
        pointers = [self.point(m.STATE+'/source.json'), *self.bindings.values(), *(self.config[k] for k in
            ('resume_control', 'resume_control_lock', 'legacy_retirement', 'environment_recheck'))]
        self.put(m.STATE+'/source_inputs.json', dict(schema=m.LOCK_SCHEMA, approved_for_observation=True,
            files={p['path']: p['sha256'] for p in pointers}))
    def bundle(self): return m.load_source(self.point(m.STATE+'/source_inputs.json')['sha256'], read_bytes=self.read)
    def collect(self): return m.collect(self.bundle(), read_bytes=self.read)
    def complete_result(self):
        terminal = dict(self.common, complete=True, expected=1920, completed=1920, failed=0, owner_error=None,
            stop_requested=False, drain_requested=False, drained=False, CPU_ready_unexecuted=[], CPU_not_launched=0,
            imported_completed_cases=584, resumed_new_cases=1336)
        self.put(m.RUNTIME+'/terminal.json', terminal)
        clean = dict(complete=True, returncode=0, finished_utc='synthetic', automatic_retry=False)
        for stage in ('formal_TEST_1920', 'independent_TEST_results'):
            self.put(m.CONTROLLER+'/stages/'+stage+'/terminal.json', clean)
        argv = [m.AUDITOR_SOURCE, '--resume-control', m.CONTROL]
        self.put(m.CONTROLLER+'/stages/independent_TEST_results/launch.json', self.stage(m.AUDITOR_SOURCE, 203, 2003, argv))
        tiers, cases, drivers = [], [], []
        for row in self.plan['jobs']:
            record = self.status['completed'][row['job_id']]; runtime = row['evidence_runtime']
            pid, start = (101, 1001) if runtime == m.ORIGINAL else (102, 1002) if runtime == m.PARENT else (202, 2002)
            tiers.append(dict(job_id=row['job_id'], evidence_runtime=runtime, imported_completed=row['imported_completed'],
                actual_stage_owner_pid=pid, actual_stage_owner_start_ticks=start, resume_lineage=row['resume_lineage'],
                previous_row_sha256=row['previous_row_sha256'], execution_entry=row['execution_entry'],
                actual_outer_argv=row['argv'], physical_argv=row['physical_argv'], case=record['case'],
                admission_driver_receipt=record['admission_driver_receipt'], batch_driver_receipt=record.get('batch_driver_receipt')))
            cases.append(record['case']); drivers.append(record['admission_driver_receipt'])
        result = dict(self.base, schema='r10_three_tier_composite_formal_TEST_results_v957', complete=True, split='test',
            source_cases=640, method_cases=1920, completed_method_cases=1920, methods=list(m.METHODS), tasks=list(m.TASK_LABELS),
            source_sha256=self.bindings[m.AUDITOR_SOURCE]['sha256'], resume_control=self.config['resume_control'], resume_control_lock=self.config['resume_control_lock'],
            source_input_lock=self.control['source_inputs'], validation_release=self.control['validation_release'], resource_profile=m.PROFILE,
            resource_qualification=self.control['resource_qualification'], execution_implementation=self.control['execution_implementation'],
            environment_recheck=self.control['environment_recheck'], independent_composite_audit=True, used_evidence_rechecked_at_close=True,
            native_failures_retained=True, motion_unknown_retained=True, zero_tick_cases_retained=True, motion_is_diagnostic_only=True,
            old584_results_not_rerun=True, source_outcome_filtering=False, Value_NGFF_FASTER_used=False,
            old907_successful_terminal_claimed=False, original754_or906_single_result_claimed=False,
            auditor_invocation=dict(pid=203, start_ticks=2003, source=self.bindings[m.AUDITOR_SOURCE], argv=argv), frozen_environment=self.environment(203, 2003),
            actual_stage_owners=[dict(runtime=runtime, launch=self.point(runtime+'/launch.json'), terminal=self.point(runtime+'/terminal.json'),
                pid=pid, start_ticks=start, completed=count) for runtime, pid, start, count in
                ((m.ORIGINAL, 101, 1001, 134), (m.PARENT, 102, 1002, 450), (m.RUNTIME, 202, 2002, 1336))],
            cases=cases, admission_driver_receipts=drivers, evidence_tiers=tiers, native_successes_by_method={method: 0 for method in m.METHODS},
            per_task=[dict(task=task, methods={method: dict(episodes=40, successes=0) for method in m.METHODS}) for task in m.TASK_LABELS],
            historical_infrastructure_attempts=dict(job_ids=self.control['failed_job_ids'], previous_status=self.control['previous_status'],
                previous_terminal=self.control['previous_terminal'], counted_as_native_task_failures=False, all_keys_reexecuted_once_in_new_namespace=True),
            inputs_sha256={name: m.sha(raw) for name, raw in self.store.items()})
        self.put(m.RESULTS, result)
        self.put(m.CONTROLLER+'/terminal.json', dict(complete=True, automatic_retry=False, TEST_results=self.point(m.RESULTS)))
        self.put(m.CONTROLLER+'/stages/independent_TEST_results/stdout.log', dict(complete=True, schema=result['schema'], split='test', cases=1920,
            native_successes_by_method=result['native_successes_by_method'], motion_is_diagnostic_only=True, out=m.RESULTS))


class ReporterTests(unittest.TestCase):
    def test_partial_three_tiers_preserved_once_and_history_six_separate(self):
        f = Fixture(); data = f.collect()
        self.assertEqual(sum(data['completed'].values()), 586)
        self.assertEqual((data['imported_completed'], data['new_completed']), (584, 2))
        self.assertEqual(dict(data['evidence_tiers']), {m.ORIGINAL:134, m.PARENT:450, m.RUNTIME:2})
        self.assertEqual((data['failed'], data['historical_infrastructure_failed']), (0, 6))
        self.assertEqual(sum(data['success'].values()), 0)
        self.assertEqual(sum(data['unknown_motion'].values()), 586)
        self.assertFalse(data['release_complete']); self.assertNotIn(m.RESULTS, f.reads)
        self.assertFalse(any('/case.json' in name or 'manifest' in name for name in f.reads))

    def test_full1920_all_native_false_motion_unknown_is_real_release(self):
        f = Fixture(full=True); data = f.collect()
        self.assertTrue(data['release_complete']); self.assertEqual(sum(data['completed'].values()), 1920)
        self.assertEqual(sum(data['success'].values()), 0); self.assertEqual(sum(data['unknown_motion'].values()), 1920)
        self.assertIn(m.RESULTS, f.reads)
        self.assertNotIn(m.LEGACY+'/terminal.json', f.reads)

    def test_current_native_none_motion_false_remain_unknown_and_diagnostic(self):
        f = Fixture(); row = f.plan['jobs'][584]; key = row['job_id']
        f.status['completed'][key].update(native_success=None, motion_pass=False)
        f.put(m.RUNTIME+'/status.json', f.status); data = f.collect()
        self.assertEqual(sum(data['unknown_native'].values()), 1)
        self.assertEqual(sum(data['motion_false'].values()), 1)
        self.assertEqual(sum(data['completed'].values()), 586)

    def test_current_infra_one_does_not_include_old_six_or_native_false(self):
        f = Fixture(); key = f.plan['jobs'][590]['job_id']
        f.status['failed'][key] = dict(complete=False, automatic_retry=False, returncode=1)
        f.status['drain_requested'] = True; f.put(m.RUNTIME+'/status.json', f.status)
        data = f.collect(); self.assertEqual((data['failed'], data['historical_infrastructure_failed']), (1, 6))
        self.assertEqual(sum(data['completed'].values()), 586); self.assertTrue(data['errors'])

    def test_incomplete_owner_or_stage_never_reads_final_result(self):
        for kind in ('owner', 'audit', 'controller'):
            f = Fixture(full=True)
            name = m.RUNTIME+'/terminal.json' if kind == 'owner' else m.CONTROLLER+'/stages/independent_TEST_results/terminal.json' if kind == 'audit' else m.CONTROLLER+'/terminal.json'
            value = json.loads(f.store[name]); value['complete'] = False; f.put(name, value)
            with self.subTest(kind=kind), self.assertRaises(ValueError): f.collect()
            self.assertNotIn(m.RESULTS, f.reads)

    def test_retirement_is_required_but_legacy_terminal_is_not_invented(self):
        f = Fixture(); f.store.pop(m.RETIREMENT)
        with self.assertRaises(FileNotFoundError): f.collect()
        self.assertNotIn(m.RUNTIME+'/status.json', f.reads)
        f = Fixture(); value = json.loads(f.store[m.RETIREMENT]); value['exit_observed'] = False
        f.config['legacy_retirement'] = f.put(m.RETIREMENT, value); f.freeze_config()
        with self.assertRaises(ValueError): f.collect()
        self.assertNotIn(m.RUNTIME+'/status.json', f.reads)

    def test_plan_import_or_lineage_or_source_grid_change_refused(self):
        for kind in ('lineage', 'tier', 'duplicate', 'source'):
            f = Fixture(); row = f.plan['jobs'][0]
            if kind == 'lineage': row['resume_lineage']['parent_row_sha256'] = '0'*64
            elif kind == 'tier': row['evidence_runtime'] = m.RUNTIME
            elif kind == 'duplicate': f.plan['jobs'][1]['job_id'] = row['job_id']
            else: row['raw'] = 'wrong_original_raw'
            f.put(m.RUNTIME+'/plan.json', f.plan)
            with self.subTest(kind=kind), self.assertRaises(ValueError): f.collect()

    def test_rewritten_imported_outcome_refused(self):
        f = Fixture(); key = f.plan['jobs'][0]['job_id']; f.status['completed'][key]['native_success'] = True
        f.put(m.RUNTIME+'/status.json', f.status)
        with self.assertRaises(ValueError): f.collect()

    def test_956_historical_owner_fields_are_source_not_runtime_launch_pointers(self):
        f = Fixture()
        self.assertEqual(f.owner['base_stage_owner'], f.bindings[m.ORIGINAL_SOURCE])
        self.assertEqual(f.owner['parent_stage_owner'], f.bindings[m.PARENT_SOURCE])
        f.collect()
        for field, wrong in (('base_stage_owner', f.control['original_launch']), ('parent_stage_owner', f.control['previous_launch'])):
            value = deepcopy(f.owner); value[field] = wrong; f.put(m.RUNTIME+'/launch.json', value)
            with self.subTest(field=field), self.assertRaises(ValueError): f.collect()

    def test_owner_environment_has_own_observer_and_check_time_in_same_namespace(self):
        f = Fixture()
        actual = deepcopy(f.owner['frozen_environment']); reference = deepcopy(f.env)
        actual['checked_utc'] = '2026-10-21T23:12:00Z'; reference['checked_utc'] = '2026-10-21T23:11:00Z'
        self.assertNotEqual(actual, reference)
        self.assertEqual(m.recorded_environment(actual, f.control, pid=202, start=2002, expected=reference), actual)
        actual['current_mount_namespace'] = 'mnt:[wrong]'
        with self.assertRaises(ValueError): m.recorded_environment(actual, f.control, pid=202, start=2002, expected=reference)

    def test_final_tier_or_audit_pid_or_header_or_sha_mismatch_refused(self):
        for kind in ('tier', 'pid', 'header', 'sha'):
            f = Fixture(full=True); value = json.loads(f.store[m.RESULTS])
            if kind == 'tier': value['evidence_tiers'][0]['evidence_runtime'] = m.RUNTIME
            elif kind == 'pid': value['auditor_invocation']['start_ticks'] += 1
            elif kind == 'header': value['old907_successful_terminal_claimed'] = True
            else: value['inputs_sha256'][m.RUNTIME+'/status.json'] = '0'*64
            pointer = f.put(m.RESULTS, value)
            f.put(m.CONTROLLER+'/terminal.json', dict(complete=True, automatic_retry=False, TEST_results=pointer))
            with self.subTest(kind=kind), self.assertRaises(ValueError): f.collect()

    def test_changed_source_is_detected(self):
        f = Fixture(); bundle = f.bundle(); f.store[m.OWNER_SOURCE] += b'# changed'
        with self.assertRaises(ValueError): m.collect(bundle, read_bytes=f.read)

    def test_small_reporter_lock_cannot_expand_into_scientific_map(self):
        f = Fixture(); name = m.STATE+'/source_inputs.json'; value = json.loads(f.store[name])
        value['files']['models/hidden.pt'] = '0'*64; pointer = f.put(name, value)
        with self.assertRaises(ValueError): m.load_source(pointer['sha256'], read_bytes=f.read)
        self.assertNotIn('models/hidden.pt', f.reads)

    def test_status_resource_boolean_is_not_measured_zero(self):
        f = Fixture(); f.status['native_slots'] = False; f.put(m.RUNTIME+'/status.json', f.status)
        with self.assertRaises(ValueError): f.collect()

    def test_forbidden_path_alias_and_STOP_reject_before_callback(self):
        def no_read(name): self.fail('forbidden callback: '+name)
        reader = m.Metadata(read_bytes=no_read)
        for name in ('data/raw.pkl', 'outputs/case.json', 'artifacts/a/manifest.json', 'models/checkpoint.pt',
                     '/etc/shadow', m.RUNTIME+'/../status.json'):
            with self.subTest(name=name), self.assertRaises(ValueError): reader.bytes(name)
        with self.assertRaises(m.ObservationStopped): m.Metadata(stop=lambda:True, read_bytes=no_read).json(m.CONTROL)

    def test_one_snapshot_does_not_reread_mutating_status(self):
        calls = []
        def read(name): calls.append(name); return m.encoded(dict(value=len(calls)))
        reader = m.Metadata(read_bytes=read)
        self.assertEqual(reader.json(m.RUNTIME+'/status.json'), reader.json(m.RUNTIME+'/status.json'))
        self.assertEqual(len(calls), 1)

    def test_fixed_microsecond_phase_never_activation_plus1800(self):
        for stamp in ('2026-10-21T23:10:00+00:00', '2026-10-22T00:59:59+00:00'):
            now = datetime.fromisoformat(stamp).timestamp(); due = m.phase_deadline(now)
            self.assertTrue(m.is_phase(due)); self.assertGreater(due, now)
            actual = datetime.fromtimestamp(due, timezone.utc)
            self.assertIn(actual.minute, (8, 38)); self.assertEqual((actual.second, actual.microsecond), (21, 384754))
            self.assertNotEqual(due, now+1800)
            self.assertEqual(m.phase_deadline(due), due+1800)

    def test_unadmitted_counts_are_none_and_txt_not_zero(self):
        data = m.empty_data('gate failed'); now = 1792624200.
        item = m.snapshot(data, now=now, next_due=m.phase_deadline(now), reason='fake')
        self.assertIsNone(item['total_completed']); self.assertIsNone(item['current956_infrastructure_failed'])
        self.assertTrue(all(v['completed'] is None for v in item['methods'].values()))
        text, _ = m.render(data, item, []); self.assertIn('未知／未获准读取', text)

    def test_later_read_failure_keeps_frozen_history_but_never_stale_current_counts(self):
        f = Fixture(); previous = f.collect(); data = m.observation_error(previous, 'later source SHA mismatch')
        self.assertIsNone(data['completed']); self.assertIsNone(data['failed'])
        self.assertEqual(data['historical_infrastructure_failed'], 6)
        self.assertEqual(data['legacy_history'], previous['legacy_history'])
        now = 1792624200.; item = m.snapshot(data, now=now, next_due=m.phase_deadline(now), reason='failed')
        text, _ = m.render(data, item, [])
        self.assertIn('actual922 historical infra6', text); self.assertIn('不代表当前读取成功', text)
        self.assertIsNone(item['total_completed'])

    def test_temp_output_is_only_new_reporter_and_never_ack(self):
        f = Fixture(); data = f.collect(); bundle = f.bundle(); now = 1792624200.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); old = root/'TEST_结果更新_并发修正版.txt'; old.write_text('untouched922')
            old_history = root/m.LEGACY/'history.jsonl'; old_history.parent.mkdir(parents=True); old_history.write_text('frozen922')
            history = m.save_report(data, bundle, [], next_due=m.phase_deadline(now), reason='CPU fake', now=now, root=root)
            self.assertEqual(len(history), 1); self.assertEqual(old.read_text(), 'untouched922'); self.assertEqual(old_history.read_text(), 'frozen922')
            self.assertIn('[922真实历史]', (root/m.OUT).read_text()); self.assertNotIn('[909', (root/m.OUT).read_text())
            names = {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()}
            self.assertFalse(any('ack' in name.lower() or name.endswith('/STOP') for name in names))
            chat = json.loads((root/m.STATE/'chat_ready.json').read_bytes())
            self.assertFalse(chat['chat_sent']); self.assertFalse(chat['delivery_acknowledged'])

    def test_default_main_performs_no_read_or_activation(self):
        with patch.object(m, 'load_source', side_effect=AssertionError('activated')), patch.object(m, 'collect', side_effect=AssertionError('collected')):
            self.assertEqual(m.main([]), 0)

    def test_prepared_controls_are_allowed_but_existing_runtime_outputs_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); state = root/m.STATE; state.mkdir(parents=True)
            for name in ('source.json', 'source_inputs.json', 'legacy922_retirement.json'): (state/name).write_text('{}')
            m.require_fresh_observer(root)
            for name in ('launch.json', 'status.json', 'history.jsonl', 'terminal.json', 'chat_ready.json', 'snapshots'):
                (state/name).write_text('{}')
                with self.subTest(name=name), self.assertRaises(ValueError): m.require_fresh_observer(root)
                (state/name).unlink()
            (root/m.OUT).write_text('preserve959')
            with self.assertRaises(ValueError): m.require_fresh_observer(root)


if __name__ == '__main__': unittest.main()
