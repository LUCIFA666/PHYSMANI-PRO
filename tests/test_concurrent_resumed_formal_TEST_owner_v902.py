"""Pure CPU fakes for the inactive902 resource and continuation contracts.

All rows are invented in memory.  STOP/DRAIN files exist only in temporary
directories; these tests never load experiment inputs or run an owner/child.
"""
from copy import deepcopy
from concurrent.futures import Future
from pathlib import Path
from unittest.mock import patch
import tempfile
import threading
import unittest

from scripts import run_concurrent_resumed_formal_TEST_owner_v902_r10 as owner


class ConcurrentTESTOwnerTests(unittest.TestCase):
    def test_external_resume_lock_pin_checks_exact_parsed_bytes(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'lock.json';path.write_text('{"fixed":true}')
            self.assertEqual(owner.read_pinned_resume_lock(path,owner.file_sha(path)),{'fixed':True})

    def test_external_resume_lock_pin_rejects_missing_malformed_or_changed_bytes(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'lock.json';path.write_text('{"fixed":true}');expected=owner.file_sha(path)
            for invalid in (None,True,'','A'*64,'z'*64,'0'*64):
                with self.subTest(invalid=invalid),self.assertRaises(ValueError):
                    owner.read_pinned_resume_lock(path,invalid)
            path.write_text('{"fixed":false}')
            with self.assertRaises(ValueError):owner.read_pinned_resume_lock(path,expected)

    def test_invalid_resume_pin_rejects_before_JSON_parse(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'lock.json';path.write_text('{broken JSON}')
            with patch.object(owner.json,'loads',side_effect=AssertionError('must not parse mismatched bytes')):
                with self.assertRaises(ValueError):owner.read_pinned_resume_lock(path,'0'*64)

    def profile(self, **changes):
        value = dict(schema=owner.PROFILE_SCHEMA, max_native_slots=4,
                     max_GPU_slots=2, max_CPU_preflight_slots=4, ready_limit=32)
        value.update(changes)
        return value

    def row(self, name, method):
        return dict(job_id=name, method=method)

    def entry(self, kind='original751'):
        result = dict(kind=kind, pointer=dict(
            path=owner.ORIGINAL_ENTRY if kind == 'original751' else owner.BATCH_ENTRY,
            sha256='a' * 64 if kind == 'original751' else 'b' * 64))
        if kind == 'batched901':
            result.update(batch_control=dict(path='fixture/batch_control.json', sha256='c' * 64),
                          batch_lock=dict(path='fixture/batch_inputs.json', sha256='d' * 64))
        return result

    def grid(self):
        policy = 'e' * 64
        jobs = []
        for task in owner.TASKS:
            for source in range(40):
                raw = 'invented_episode_' + str(source)
                for method in owner.METHODS:
                    name = task + '/' + raw + '/' + method
                    jobs.append(dict(job_id=name, task=task, raw=raw, method=method,
                        argv=['bash', 'fixture/physical_python.sh', owner.ORIGINAL_ENTRY,
                              '--suite', 'fixture/suite.json', '--split', 'test',
                              '--task', task, '--raw', raw, '--method', method,
                              '--source-inputs', 'fixture/source_inputs.json',
                              '--validation-release', 'fixture/VAL_release.json',
                              '--out', 'fixture/old_output/' + name],
                        worker_sha256='a' * 64,
                        admission_driver=deepcopy(self.entry()['pointer']),
                        source_row_sha256='f' * 64,
                        preserved_metadata=dict(source_index=source, nested=['unchanged'])))
        completed = [row['job_id'] for row in jobs[:7]]
        remaining = [row['job_id'] for row in jobs[7:]]
        control = dict(schema=owner.CONTROL_SCHEMA, split='test', expected_cases=1920,
            source_count=640, user_authorized_resume=True, standing_authorization_reused=True,
            new_user_event_invented=False, physical_policy_changed=False,
            completed_job_ids=completed, remaining_job_ids=remaining,
            completed_key_sha256=owner.digest(sorted(completed)),
            remaining_key_sha256=owner.digest(sorted(remaining)), policy_sha256=policy)
        plan = dict(jobs=jobs, split='test', expected_cases=1920, policy_sha256=policy)
        status = dict(split='test', expected=1920,
            completed={name: dict(complete=True, native_success=bool(index % 2), returncode=0,
                                  timed_out=False, interrupted=False, automatic_retry=False)
                       for index, name in enumerate(completed)}, failed={}, active=[], checking=[],
            drain_requested=True, stop_requested=False)
        terminal = dict(complete=False, drained=True, drain_requested=True, stop_requested=False,
            expected=1920, completed=len(completed), failed=0, automatic_retry=False,
            policy_sha256=policy)
        return control, plan, status, terminal

    def test_resource_profile_is_bounded_and_copied(self):
        value = self.profile()
        checked = owner.validate_resource_profile(value)
        self.assertEqual(checked, value)
        checked['ready_limit'] = 1
        self.assertEqual(value['ready_limit'], 32)
        self.assertEqual(owner.validate_resource_profile(self.profile(
            max_native_slots=1, max_GPU_slots=1, max_CPU_preflight_slots=1, ready_limit=1))['max_native_slots'], 1)

    def test_resource_profile_requires_exact_schema_and_strict_integer_quotas(self):
        bad_values = [None, {}, dict(self.profile(), schema='other'), dict(self.profile(), extra=1)]
        for field, maximum in [('max_native_slots', 4), ('max_GPU_slots', 2),
                               ('max_CPU_preflight_slots', 4), ('ready_limit', 32)]:
            for bad in (True, False, 0, -1, maximum + 1, 1.0, '1', None):
                bad_values.append(self.profile(**{field: bad}))
        bad_values.append(self.profile(max_native_slots=1, max_GPU_slots=2))
        for value in bad_values:
            with self.subTest(value=value), self.assertRaises(ValueError):
                owner.validate_resource_profile(value)

    def test_selection_fills_four_native_slots_with_at_most_two_GPU_jobs(self):
        ready = [self.row('p1', 'physmani'), self.row('p2', 'physmani'),
                 self.row('p3', 'physmani'), self.row('cv', 'cv_geometry'),
                 self.row('ctx', 'context_geometry'), self.row('extra', 'cv_geometry')]
        selected = owner.select_native_jobs(ready, [], self.profile())
        self.assertEqual([r['job_id'] for r in selected], ['p1', 'p2', 'cv', 'ctx'])
        self.assertEqual(len(selected), 4)
        self.assertEqual(sum(r['method'] == 'physmani' for r in selected), 2)

    def test_selection_skips_GPU_full_rows_and_keeps_later_CPU_order(self):
        active = [self.row('running-p1', 'physmani'), self.row('running-p2', 'physmani')]
        ready = [self.row('wait-p', 'physmani'), self.row('ctx', 'context_geometry'),
                 self.row('cv', 'cv_geometry'), self.row('later', 'cv_geometry')]
        before = deepcopy((ready, active))
        selected = owner.select_native_jobs(ready, active, self.profile())
        self.assertEqual([r['job_id'] for r in selected], ['ctx', 'cv'])
        self.assertEqual((ready, active), before)
        still_ready = [row for row in ready if row not in selected]
        self.assertEqual(owner.select_native_jobs(still_ready, active + selected, self.profile()), [])

    def test_selection_rejects_duplicates_overlap_and_unauthorized_methods(self):
        cv = self.row('same', 'cv_geometry')
        cases = [([cv, deepcopy(cv)], []), ([], [cv, deepcopy(cv)]), ([cv], [deepcopy(cv)]),
                 ([self.row('bad', 'Value')], []), ([], [self.row('bad', 'NGFF')]),
                 ([self.row('bad', 'FASTER')], [])]
        for ready, active in cases:
            with self.subTest(ready=ready, active=active), self.assertRaises(ValueError):
                owner.select_native_jobs(ready, active, self.profile())

    def test_selection_rejects_existing_overquota_before_dispatch(self):
        for active in ([self.row(str(i), 'cv_geometry') for i in range(5)],
                       [self.row(str(i), 'physmani') for i in range(3)]):
            with self.subTest(active=active), self.assertRaises(ValueError):
                owner.select_native_jobs([], active, self.profile())

    def test_each_STOP_scope_sets_cancellation_even_with_DRAIN(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(owner, 'ROOT', Path(temp)):
            root = Path(temp)
            runtime = root / owner.RUNTIME
            runtime.mkdir(parents=True)
            (runtime / 'DRAIN').touch()
            for relative in ('runtime/framework/multitask_r10_STOP', owner.RUNTIME + '/STOP',
                             owner.PREVIOUS + '/STOP', owner.PREVIOUS_CONTROLLER + '/STOP'):
                with self.subTest(relative=relative):
                    stop = root / relative
                    stop.parent.mkdir(parents=True, exist_ok=True)
                    stop.touch()
                    cancelled = threading.Event()
                    self.assertTrue(owner.dispatch_control(runtime, cancelled, False))
                    self.assertTrue(cancelled.is_set())
                    stop.unlink()
                    owner.dispatch_control(runtime, cancelled, False)
                    self.assertTrue(cancelled.is_set())

    def test_DRAIN_is_latched_and_waits_for_both_checking_and_native_work(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(owner, 'ROOT', Path(temp)):
            runtime = Path(temp) / owner.RUNTIME
            runtime.mkdir(parents=True)
            cancelled = threading.Event()
            self.assertFalse(owner.dispatch_control(runtime, cancelled, False))
            (runtime / 'DRAIN').touch()
            self.assertTrue(owner.dispatch_control(runtime, cancelled, False))
            self.assertFalse(cancelled.is_set())
            (runtime / 'DRAIN').unlink()
            self.assertTrue(owner.dispatch_control(runtime, cancelled, True))
        self.assertFalse(owner.drained(False, {}, {}))
        self.assertFalse(owner.drained(True, {'cpu': object()}, {}))
        self.assertFalse(owner.drained(True, {}, {'native': object()}))
        self.assertTrue(owner.drained(True, {}, {}))

    def test_partition_accepts_full_same_source_grid_and_retains_native_False(self):
        values = self.grid()
        before = deepcopy(values)
        by_id = owner.require_partition(*values)
        self.assertEqual(len(by_id), 1920)
        self.assertFalse(values[2]['completed'][values[0]['completed_job_ids'][0]]['native_success'])
        self.assertEqual(values, before)

    def test_partition_requires_all_unique_1920_rows_and_exact_three_methods(self):
        for change in ('missing', 'duplicate_id', 'missing_source', 'unknown_method', 'unknown_task'):
            control, plan, status, terminal = self.grid()
            if change == 'missing':
                plan['jobs'].pop()
            elif change == 'duplicate_id':
                plan['jobs'][-1]['job_id'] = plan['jobs'][0]['job_id']
            elif change == 'missing_source':
                plan['jobs'][-1]['raw'] = plan['jobs'][-4]['raw']
            else:
                plan['jobs'][-1]['method' if change == 'unknown_method' else 'task'] = 'invented'
            with self.subTest(change=change), self.assertRaises(ValueError):
                owner.require_partition(control, plan, status, terminal)

    def test_partition_requires_identical_forty_sources_for_each_method(self):
        control, plan, status, terminal = self.grid()
        plan['jobs'][-1]['raw'] = 'different_but_still_unique_fortieth_source'
        with self.assertRaisesRegex(ValueError, 'identical source'):
            owner.require_partition(control, plan, status, terminal)

    def test_partition_requires_exact_disjoint_keys_and_frozen_key_digests(self):
        for change in ('duplicate_completed', 'duplicate_remaining', 'overlap', 'omitted_key',
                       'unknown_key', 'status_missing', 'status_extra', 'completed_digest', 'remaining_digest'):
            control, plan, status, terminal = self.grid()
            if change == 'duplicate_completed':
                control['completed_job_ids'].append(control['completed_job_ids'][0])
            elif change == 'duplicate_remaining':
                control['remaining_job_ids'].append(control['remaining_job_ids'][0])
            elif change == 'overlap':
                control['remaining_job_ids'].append(control['completed_job_ids'][0])
            elif change == 'omitted_key':
                control['remaining_job_ids'].pop()
            elif change == 'unknown_key':
                control['remaining_job_ids'][-1] = 'not_in_the_original_plan'
            elif change == 'status_missing':
                status['completed'].pop(control['completed_job_ids'][0])
            elif change == 'status_extra':
                status['completed']['not_in_the_original_plan'] = dict(complete=True, native_success=False)
            else:
                control[change.replace('_digest', '_key_sha256')] = '0' * 64
            with self.subTest(change=change), self.assertRaises(ValueError):
                owner.require_partition(control, plan, status, terminal)

    def test_partition_requires_explicit_authorization_same_policy_and_TEST_split(self):
        changes = [('control', 'schema', 'wrong'), ('control', 'split', 'validation'),
                   ('control', 'expected_cases', 1919), ('control', 'source_count', 639),
                   ('control', 'user_authorized_resume', 1), ('control', 'standing_authorization_reused', 1),
                   ('control', 'new_user_event_invented', 0), ('control', 'physical_policy_changed', 0),
                   ('plan', 'split', 'train'), ('plan', 'policy_sha256', '0' * 64),
                   ('terminal', 'policy_sha256', '0' * 64)]
        for target, field, value in changes:
            control, plan, status, terminal = self.grid()
            dict(control=control, plan=plan, terminal=terminal)[target][field] = value
            with self.subTest(target=target, field=field), self.assertRaises(ValueError):
                owner.require_partition(control, plan, status, terminal)

    def test_partition_rejects_STOP_infrastructure_failures_and_unfinished_drain(self):
        changes = [('status', 'failed', {'infra': {'complete': False}}),
                   ('status', 'active', ['live']), ('status', 'checking', ['live']),
                   ('status', 'split', 'validation'), ('status', 'expected', 1919),
                   ('status', 'drain_requested', False), ('status', 'stop_requested', True),
                   ('terminal', 'complete', True), ('terminal', 'drained', False),
                   ('terminal', 'drain_requested', False), ('terminal', 'stop_requested', True),
                   ('terminal', 'completed', 6), ('terminal', 'failed', 1),
                   ('terminal', 'automatic_retry', True)]
        for target, field, value in changes:
            control, plan, status, terminal = self.grid()
            dict(status=status, terminal=terminal)[target][field] = value
            with self.subTest(target=target, field=field), self.assertRaises(ValueError):
                owner.require_partition(control, plan, status, terminal)

    def test_partition_rejects_nonboolean_or_interrupted_completed_outcomes(self):
        for field, value in [('complete', False), ('complete', 1), ('native_success', 0),
                             ('native_success', 1), ('native_success', None),
                             ('interrupted', True), ('timed_out', True)]:
            control, plan, status, terminal = self.grid()
            status['completed'][control['completed_job_ids'][0]][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                owner.require_partition(control, plan, status, terminal)

    def test_original751_rows_keep_imported_bytes_and_only_pending_output_changes(self):
        control, plan, _, _ = self.grid()
        before = deepcopy(plan)
        rows = owner.resumed_jobs(plan, control, self.entry())
        for row, old in zip(rows, plan['jobs']):
            self.assertEqual(row['previous_row_sha256'], owner.digest(old))
            self.assertEqual(row['previous_worker_sha256'], old['worker_sha256'])
            restored = {key: value for key, value in row.items() if key not in owner.ROW_PROVENANCE}
            if old['job_id'] in control['completed_job_ids']:
                self.assertTrue(row['imported_completed'])
                self.assertEqual(row['evidence_runtime'], owner.PREVIOUS)
                self.assertEqual(restored, old)
                self.assertEqual(row['physical_argv'], old['argv'])
            else:
                self.assertFalse(row['imported_completed'])
                self.assertEqual(row['evidence_runtime'], owner.RUNTIME)
                expected = deepcopy(old)
                out = str(Path(owner.OUTPUT) / old['task'] / old['raw'] / old['method'])
                expected['argv'][expected['argv'].index('--out') + 1] = out
                self.assertEqual(restored, expected)
                self.assertEqual(row['physical_argv'], row['argv'])
            self.assertEqual(row['execution_entry'], self.entry())
        rows[0]['preserved_metadata']['nested'].append('changed_copy')
        self.assertEqual(plan, before)

    def test_batched901_is_real_outer_entry_with_exact_forwarded751_arguments(self):
        control, plan, _, _ = self.grid()
        rows = owner.resumed_jobs(plan, control, self.entry('batched901'))
        imported = rows[0]
        self.assertEqual(imported['argv'], plan['jobs'][0]['argv'])
        self.assertEqual(imported['execution_entry'], self.entry())
        pending = rows[len(control['completed_job_ids'])]
        inner = pending['physical_argv']
        outer = pending['argv']
        self.assertEqual(inner[:3], plan['jobs'][len(control['completed_job_ids'])]['argv'][:3])
        self.assertEqual(outer[:3], [*inner[:2], owner.BATCH_ENTRY])
        self.assertEqual(outer[3:12], ['--batch-control', 'fixture/batch_control.json',
            '--batch-lock', 'fixture/batch_inputs.json', '--batch-lock-sha256', 'd' * 64,
            '--entry', 'formal', '--'])
        self.assertEqual(outer[12:], inner[3:])
        self.assertEqual(pending['execution_entry'], self.entry('batched901'))
        self.assertEqual(pending['worker_sha256'], 'b' * 64)
        self.assertEqual(pending['previous_worker_sha256'], 'a' * 64)

    def test_resume_refuses_any_prior_provenance_tier_instead_of_flattening(self):
        control, plan, _, _ = self.grid()
        for field in owner.ROW_PROVENANCE:
            changed = deepcopy(plan)
            changed['jobs'][0][field] = 'prior_tier'
            with self.subTest(field=field), self.assertRaises(ValueError):
                owner.resumed_jobs(changed, control, self.entry())

    def test_entry_requires_exact_valid_project_only_immutable_pointers(self):
        invalid = [None, {}, dict(self.entry(), kind='fake'),
                   dict(self.entry(), batch_control={}),
                   dict(self.entry('batched901'), missing=True)]
        wrong_target = self.entry()
        wrong_target['pointer']['path'] = owner.BATCH_ENTRY
        invalid.append(wrong_target)
        for kind in ('original751', 'batched901'):
            for bad_sha in ('', 'g' * 64, 'a' * 63, 1, None):
                value = self.entry(kind)
                value['pointer']['sha256'] = bad_sha
                invalid.append(value)
        for name in ('batch_control', 'batch_lock'):
            for bad_pointer in ({}, {'path': 'fixture/file.json'},
                                dict(path='fixture/file.json', sha256='c' * 64, extra=True),
                                dict(path='fixture/file.json', sha256='not_sha'),
                                dict(path='/outside/file.json', sha256='c' * 64),
                                dict(path='../outside.json', sha256='c' * 64),
                                dict(path='fixture/../../outside.json', sha256='c' * 64),
                                dict(path='', sha256='c' * 64), dict(path=1, sha256='c' * 64)):
                value = self.entry('batched901')
                value[name] = bad_pointer
                invalid.append(value)
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(ValueError):
                owner.validate_entry(value)

    def test_physical_argv_requires_one_out_value_and_the_original751_entry(self):
        _, plan, _, _ = self.grid()
        row = plan['jobs'][0]
        before = deepcopy(row)
        changed = owner.physical_argv(row, Path('fixture/new_output'))
        expected = deepcopy(row['argv'])
        expected[expected.index('--out') + 1] = 'fixture/new_output'
        self.assertEqual(changed, expected)
        self.assertEqual(row, before)
        invalid = [row['argv'] + ['--out', 'duplicate'], row['argv'][:-2], row['argv'][:-1],
                   ['bash', 'fixture/python.sh', owner.BATCH_ENTRY, '--out', 'somewhere']]
        for argv in invalid:
            with self.subTest(argv=argv), self.assertRaises(ValueError):
                owner.physical_argv(dict(row, argv=argv), Path('fixture/new_output'))

    def source_bindings(self, kind='batched901'):
        names = list(owner.COMMON_SOURCES)
        if kind == 'batched901':
            names += [owner.BATCH_ENTRY, 'our_mani/multitask/frozen_lookup_batch_v900.py']
        sources = {name: dict(path=name, sha256=owner.digest(name)) for name in names}
        target = owner.ORIGINAL_ENTRY if kind == 'original751' else owner.BATCH_ENTRY
        sources[target] = deepcopy(self.entry(kind)['pointer'])
        return sources

    def qualification(self):
        entry = self.entry('batched901')
        implementation = owner.execution_implementation(entry, self.source_bindings())
        control = dict(resource_profile=self.profile(), execution_entry=entry,
            execution_implementation=implementation, policy_sha256='e' * 64,
            author_outcome_protocol=dict(schema='invented_unchanged_protocol', max_ticks=200),
            resource_qualifier=dict(path='fixture/resource_qualifier.py', sha256='f' * 64))
        proof = dict(schema=owner.QUALIFICATION_SCHEMA, complete=True, split='train',
            resource_profile=deepcopy(control['resource_profile']),
            execution_entry=deepcopy(implementation), policy_sha256=control['policy_sha256'],
            author_outcome_protocol=deepcopy(control['author_outcome_protocol']), case_count=6,
            infrastructure_failed=0, cleanup_completed=True, resource_bounds_verified=True,
            overlap_verified=True, independent_audit_complete=True, formal_score_contribution=False,
            TEST_read=False, observed_native_peak=4, observed_GPU_peak=2, source_sha256='f' * 64,
            native_successes=0, native_outcomes=[False] * 6)
        return control, proof

    def batch_metrics(self):
        return dict(schema='r10_scoped_frozen_lookup_metrics_v900', complete=True, closed=True,
            maps=2, declared_files=10, watched_directories=8, pointer_checks=3,
            full_graph_boundary_checks=4, used_path_end_checks=2, used_external=1,
            actual_SHA_checks=5, strong_stat_reuses=2, exact_JSON_payloads=1)

    def batch_execution(self, check=True):
        entry = self.entry('batched901')
        old = self.grid()[1]['jobs'][7]
        row = owner.resumed_jobs(dict(jobs=[old]), dict(completed_job_ids=[]), entry)[0]
        sources = self.source_bindings()
        tail = ['--check'] if check else []
        value = dict(schema=owner.BATCH_SCHEMA, complete=True, check_only=check, entry='formal',
            original_entry=owner.ORIGINAL_ENTRY, actual_outer_argv=row['argv'][2:] + tail,
            forwarded_original_argv=row['physical_argv'][2:] + tail,
            control=deepcopy(entry['batch_control']), lock=deepcopy(entry['batch_lock']),
            sources=deepcopy(sources), scientific_policy_changed=False,
            full_verify_inputs_calls_retained=True,
            native_entry_claim='original_function_called_under_explicit_v901_bootstrap',
            phases=[dict(phase=name, wall_seconds=.25, boundary_recheck_complete=True,
                         metrics=self.batch_metrics())
                    for name in ('original751_bindings', 'original742_prepare')])
        case = dict(path='fixture/new_output/case.json', sha256='1' * 64)
        receipt = dict(path='fixture/new_output/admission_driver_receipt.json', sha256='2' * 64)
        if not check:
            value.update(case=deepcopy(case), original_admission_driver_receipt=deepcopy(receipt))
        return row, sources, value, case, receipt

    def test_execution_implementation_is_shared_across_TRAIN_and_formal_controls(self):
        for kind in ('original751', 'batched901'):
            sources = self.source_bindings(kind)
            entry = self.entry(kind)
            before = deepcopy((entry, sources))
            descriptor = owner.execution_implementation(entry, sources)
            self.assertEqual(descriptor['schema'], 'r10_resource_execution_implementation_v902')
            self.assertEqual(descriptor['kind'], kind)
            self.assertEqual(descriptor['sources'], sources)
            self.assertEqual(descriptor['entrypoint'], entry['pointer'])
            self.assertEqual(descriptor['original_formal_driver'], sources[owner.ORIGINAL_ENTRY])
            self.assertNotIn('batch_control', descriptor)
            self.assertNotIn('batch_lock', descriptor)
            if kind == 'batched901':
                changed = deepcopy(entry)
                changed['batch_control'] = dict(path='fixture/TRAIN_control.json', sha256='3' * 64)
                changed['batch_lock'] = dict(path='fixture/TRAIN_lock.json', sha256='4' * 64)
                self.assertEqual(owner.execution_implementation(changed, sources), descriptor)
                self.assertEqual(len(descriptor['sources']), 7)
            else:
                self.assertIsNone(descriptor['lookup_batch'])
                self.assertEqual(len(descriptor['sources']), 5)
            descriptor['sources'][owner.ORIGINAL_ENTRY]['sha256'] = '0' * 64
            self.assertEqual((entry, sources), before)

    def test_execution_implementation_requires_exact_actual_source_pointers(self):
        for kind in ('original751', 'batched901'):
            for change in ('missing', 'extra', 'wrong_path', 'bad_sha', 'wrong_entry'):
                entry = self.entry(kind)
                sources = self.source_bindings(kind)
                name = owner.COMMON_SOURCES[-1]
                if change == 'missing':
                    sources.pop(name)
                elif change == 'extra':
                    sources['fixture/unrelated.py'] = dict(path='fixture/unrelated.py', sha256='0' * 64)
                elif change == 'wrong_path':
                    sources[name]['path'] = 'fixture/different_runtime.py'
                elif change == 'bad_sha':
                    sources[name]['sha256'] = 'invalid'
                else:
                    entry['pointer']['sha256'] = '0' * 64
                with self.subTest(kind=kind, change=change), self.assertRaises(ValueError):
                    owner.execution_implementation(entry, sources)

    def test_resource_qualification_allows_six_native_False_outcomes(self):
        control, proof = self.qualification()
        before = deepcopy((control, proof))
        self.assertEqual(owner.require_resource_qualification(proof, control), proof)
        self.assertEqual((control, proof), before)
        self.assertEqual(proof['native_outcomes'], [False] * 6)
        proof['execution_entry'] = deepcopy(control['execution_entry'])
        with self.assertRaises(ValueError):
            owner.require_resource_qualification(proof, control)

    def test_resource_qualification_requires_actual_six_TRAIN_resource_and_audit_gates(self):
        changes = [('schema', 'other'), ('complete', 1), ('split', 'test'), ('case_count', 5),
            ('case_count', True), ('infrastructure_failed', False), ('infrastructure_failed', 1),
            ('cleanup_completed', False), ('resource_bounds_verified', False),
            ('overlap_verified', False), ('independent_audit_complete', False),
            ('formal_score_contribution', 0), ('TEST_read', 0), ('observed_native_peak', 3),
            ('observed_native_peak', True), ('observed_GPU_peak', 1), ('observed_GPU_peak', True),
            ('source_sha256', '0' * 64), ('policy_sha256', '0' * 64),
            ('resource_profile', self.profile(ready_limit=1)), ('author_outcome_protocol', {})]
        for field, value in changes:
            control, proof = self.qualification()
            proof[field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                owner.require_resource_qualification(proof, control)
        control, proof = self.qualification()
        proof['execution_entry']['sources'][owner.ORIGINAL_ENTRY]['sha256'] = '0' * 64
        with self.assertRaises(ValueError):
            owner.require_resource_qualification(proof, control)

    def test_batch_metrics_require_closed_complete_boundaries_and_real_SHA_rechecks(self):
        self.assertIsNone(owner.require_batch_metrics(self.batch_metrics()))
        counts = ('maps', 'declared_files', 'watched_directories', 'pointer_checks',
                  'full_graph_boundary_checks', 'used_path_end_checks', 'used_external',
                  'actual_SHA_checks', 'strong_stat_reuses', 'exact_JSON_payloads')
        for field in counts:
            for bad in (True, -1, 1.0):
                value = self.batch_metrics()
                value[field] = bad
                with self.subTest(field=field, bad=bad), self.assertRaises(ValueError):
                    owner.require_batch_metrics(value)
        for field, bad in [('schema', 'old'), ('closed', 1), ('complete', 1), ('maps', 0),
            ('declared_files', 1), ('watched_directories', 0), ('pointer_checks', 0),
            ('used_path_end_checks', 0), ('used_path_end_checks', 4),
            ('full_graph_boundary_checks', 2), ('actual_SHA_checks', 4),
            ('strong_stat_reuses', 6), ('used_external', 4)]:
            value = self.batch_metrics()
            value[field] = bad
            with self.subTest(field=field, bad=bad), self.assertRaises(ValueError):
                owner.require_batch_metrics(value)
        for malformed in (None, [], 'metrics'):
            with self.subTest(malformed=malformed), self.assertRaises(ValueError):
                owner.require_batch_metrics(malformed)

    def test_batch_execution_accepts_exact_CPU_and_native_sidecar_bindings(self):
        for check in (True, False):
            row, sources, value, case, receipt = self.batch_execution(check)
            before = deepcopy((row, sources, value))
            kwargs = {} if check else dict(case=case, original_receipt=receipt)
            self.assertEqual(owner.require_batch_execution(value, row, check=check,
                expected_sources=sources, **kwargs), value)
            self.assertEqual((row, sources, value), before)
            self.assertEqual(value['actual_outer_argv'][0], owner.BATCH_ENTRY)
            self.assertEqual(value['forwarded_original_argv'][0], owner.ORIGINAL_ENTRY)
            self.assertEqual(len(value['sources']), 7)

    def test_batch_execution_rejects_forged_argv_sources_boundaries_or_native_claims(self):
        changes = ('schema', 'complete', 'check_bool', 'entry', 'original_entry', 'outer', 'forwarded',
            'control', 'lock', 'sources', 'missing_expected_source', 'changed_policy', 'verify_calls',
            'native_claim', 'phase_order', 'phase_missing', 'phase_nondict', 'boundary',
            'phase_negative_time', 'phase_bool_time', 'phase_nan_time', 'metrics', 'CPU_case',
            'CPU_receipt', 'native_case', 'native_receipt', 'native_none_pointers')
        for change in changes:
            check = not change.startswith('native_') or change == 'native_claim'
            row, sources, value, case, receipt = self.batch_execution(check)
            if change == 'schema': value['schema'] = 'old'
            elif change == 'complete': value['complete'] = 1
            elif change == 'check_bool': value['check_only'] = int(check)
            elif change == 'entry': value['entry'] = 'development'
            elif change == 'original_entry': value['original_entry'] = owner.BATCH_ENTRY
            elif change == 'outer': value['actual_outer_argv'] = value['forwarded_original_argv']
            elif change == 'forwarded': value['forwarded_original_argv'] = value['actual_outer_argv']
            elif change in ('control', 'lock'): value[change]['sha256'] = '0' * 64
            elif change == 'sources': value['sources'][owner.BATCH_ENTRY]['sha256'] = '0' * 64
            elif change == 'missing_expected_source':
                sources.pop(owner.COMMON_SOURCES[-1])
                value['sources'] = deepcopy(sources)
            elif change == 'changed_policy': value['scientific_policy_changed'] = 0
            elif change == 'verify_calls': value['full_verify_inputs_calls_retained'] = False
            elif change == 'native_claim': value['native_entry_claim'] = 'renamed751'
            elif change == 'phase_order': value['phases'].reverse()
            elif change == 'phase_missing': value['phases'].pop()
            elif change == 'phase_nondict': value['phases'].insert(1, 'invalid phase')
            elif change == 'boundary': value['phases'][0]['boundary_recheck_complete'] = 1
            elif change == 'phase_negative_time': value['phases'][0]['wall_seconds'] = -1
            elif change == 'phase_bool_time': value['phases'][0]['wall_seconds'] = True
            elif change == 'phase_nan_time': value['phases'][0]['wall_seconds'] = float('nan')
            elif change == 'metrics': value['phases'][0]['metrics']['actual_SHA_checks'] = 0
            elif change == 'CPU_case': value['case'] = case
            elif change == 'CPU_receipt': value['original_admission_driver_receipt'] = receipt
            elif change == 'native_case': value['case']['sha256'] = '0' * 64
            elif change == 'native_receipt': value['original_admission_driver_receipt']['sha256'] = '0' * 64
            elif change == 'native_none_pointers':
                case = receipt = None
                value['case'] = value['original_admission_driver_receipt'] = None
            kwargs = {} if check else dict(case=case, original_receipt=receipt)
            with self.subTest(change=change), self.assertRaises(ValueError):
                owner.require_batch_execution(value, row, check=check, expected_sources=sources, **kwargs)

    def test_original751_cannot_claim_a901_execution_sidecar(self):
        row = dict(execution_entry=self.entry())
        for check in (False, True):
            self.assertIsNone(owner.require_batch_execution(None, row, check=check, expected_sources={}))
            with self.assertRaises(ValueError):
                owner.require_batch_execution({}, row, check=check, expected_sources={})

    def test_child_STOP_preempts_Popen_including_immediate_final_dispatch_check(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(owner, 'ROOT', Path(temp)):
            root = Path(temp)
            row = dict(argv=['never_spawn'], method='cv_geometry')
            for check in (True, False):
                cancelled = threading.Event()
                cancelled.set()
                with patch.object(owner.subprocess, 'Popen') as popen:
                    result = owner.child(row, root / ('stopped_' + str(check)), cancelled, check=check)
                    popen.assert_not_called()
                self.assertFalse(result['complete'])
                self.assertTrue(result['interrupted'])
                self.assertFalse(result['automatic_retry'])
            dispatches = []
            def stop_on_final_dispatch(runtime, cancelled, draining):
                dispatches.append(runtime)
                if len(dispatches) == 2:
                    cancelled.set()
                return False
            with (patch.object(owner, 'ENVIRONMENT_RECEIPT', dict(current_mount_namespace='fake_namespace')),
                  patch.object(owner.os, 'readlink', return_value='fake_namespace'),
                  patch.object(owner, 'dispatch_control', side_effect=stop_on_final_dispatch),
                  patch.object(owner.subprocess, 'Popen') as popen):
                result = owner.child(row, root / 'final_STOP', threading.Event(), check=True)
                popen.assert_not_called()
            self.assertEqual(len(dispatches), 2)
            self.assertTrue(result['interrupted'])

    def test_child_missing_or_wrong_namespace_never_calls_Popen(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(owner, 'ROOT', Path(temp)):
            for index, receipt in enumerate((None, dict(current_mount_namespace='required_namespace'))):
                with (patch.object(owner, 'ENVIRONMENT_RECEIPT', receipt),
                      patch.object(owner.os, 'readlink', return_value='different_namespace'),
                      patch.object(owner.subprocess, 'Popen') as popen):
                    result = owner.child(dict(argv=['never_spawn']), Path(temp) / str(index),
                                         threading.Event(), check=True)
                    popen.assert_not_called()
                self.assertFalse(result['complete'])
                self.assertFalse(result['interrupted'])
                self.assertIn('namespace', result['error'])
                self.assertFalse(result['automatic_retry'])

    def test_final_completion_requires_full_grid_clean_results_and_no_STOP(self):
        cancelled = threading.Event()
        self.assertTrue(owner.terminal_is_complete(1920, {}, None, cancelled))
        self.assertFalse(owner.terminal_is_complete(1919, {}, None, cancelled))
        self.assertFalse(owner.terminal_is_complete(1921, {}, None, cancelled))
        self.assertFalse(owner.terminal_is_complete(1920, {'infra': {}}, None, cancelled))
        self.assertFalse(owner.terminal_is_complete(1920, {}, 'invented owner error', cancelled))
        cancelled.set()
        self.assertFalse(owner.terminal_is_complete(1920, {}, None, cancelled))

    def test_future_exception_becomes_nonretry_infrastructure_record(self):
        successful = Future()
        original = dict(complete=True, returncode=0, timed_out=False, interrupted=False,
                        automatic_retry=False)
        successful.set_result(original)
        self.assertEqual(owner.future_result(successful), original)
        for error in (OSError('invented leaf creation failure'), RuntimeError('invented child failure')):
            failed = Future()
            failed.set_exception(error)
            result = owner.future_result(failed)
            self.assertFalse(result['complete'])
            self.assertFalse(result['automatic_retry'])
            self.assertIn(str(error), result['error'])
            self.assertNotIn('native_success', result)

    def test_partition_rejects_unclean_or_missing_real753_completion_fields(self):
        changes = [('returncode', 1), ('returncode', False), ('returncode', 0.0),
                   ('returncode', '0'), ('timed_out', True), ('timed_out', 0),
                   ('interrupted', True), ('interrupted', 0),
                   ('automatic_retry', True), ('automatic_retry', 0)]
        for field, bad in changes:
            control, plan, status, terminal = self.grid()
            status['completed'][control['completed_job_ids'][0]][field] = bad
            with self.subTest(field=field, bad=bad), self.assertRaises(ValueError):
                owner.require_partition(control, plan, status, terminal)
        for field in ('returncode', 'timed_out', 'interrupted', 'automatic_retry'):
            control, plan, status, terminal = self.grid()
            status['completed'][control['completed_job_ids'][0]].pop(field)
            with self.subTest(missing=field), self.assertRaises(ValueError):
                owner.require_partition(control, plan, status, terminal)

    def test_child_leaf_creation_failure_returns_infrastructure_without_Popen(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(owner, 'ROOT', Path(temp)):
            leaf = Path(temp) / 'invented_leaf'
            actual_mkdir = Path.mkdir
            first_attempt = True
            def fail_initial_leaf(path, *args, **kwargs):
                nonlocal first_attempt
                if path == leaf and first_attempt:
                    first_attempt = False
                    raise OSError('invented first leaf mkdir failure')
                return actual_mkdir(path, *args, **kwargs)
            with (patch.object(Path, 'mkdir', autospec=True, side_effect=fail_initial_leaf),
                  patch.object(owner.subprocess, 'Popen') as popen):
                result = owner.child(dict(argv=['never_spawn']), leaf, threading.Event(), check=True)
                popen.assert_not_called()
            self.assertFalse(result['complete'])
            self.assertFalse(result['automatic_retry'])
            self.assertIn('invented first leaf mkdir failure', result['error'])
            self.assertNotIn('native_success', result)


if __name__ == '__main__':
    unittest.main()
