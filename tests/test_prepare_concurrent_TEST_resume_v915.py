"""Pure CPU and temporary-file tests for the inactive915 setup preparer."""
from copy import deepcopy
from pathlib import Path
from contextlib import redirect_stdout
from unittest.mock import patch
import fcntl
import importlib.util
import io
import tempfile
import unittest

from our_mani.multitask.frozen_lookup_batch_v900 import LookupBatch
from our_mani.multitask.protocol import MultitaskProtocolError
from scripts import prepare_concurrent_TEST_resume_v915_r10 as prepare


def fixture_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_owner_fixture = fixture_module('_owner902_fixture_for_prepare915', 'test_concurrent_resumed_formal_TEST_owner_v902.py')
_audit_fixture = fixture_module('_audit906_fixture_for_prepare915', 'test_concurrent_resumed_formal_TEST_audit_v906.py')


class PrepareConcurrentTESTTests(unittest.TestCase):
    def grid(self, count=7):
        _, plan, status, terminal = _owner_fixture.ConcurrentTESTOwnerTests().grid()
        plan.update(source_outcome_filtering=False, automatic_retry=False)
        status.update(native_slots=0, GPU_slots=0, automatic_retry=False,
            completed={row['job_id']: dict(complete=True, native_success=False, returncode=0,
                timed_out=False, interrupted=False, automatic_retry=False)
                for row in plan['jobs'][:count]})
        terminal['completed'] = count
        return plan, status, terminal

    def previous(self, count=7):
        plan, status, terminal = self.grid(count)
        return dict(plan=plan, status=status, terminal=terminal,
            launch=dict(pid=7530, start_ticks=75300),
            controller_launch=dict(pid=8270, start_ticks=82700),
            controller_status=dict(stage='requires_concrete_repair', automatic_retry=False),
            controller_failure=dict(complete=False, user_stop=False, automatic_retry=False))

    def resource_proof(self):
        fixture = _owner_fixture.ConcurrentTESTOwnerTests()
        control, value = fixture.qualification()
        env_control, environment = _audit_fixture.ConcurrentTESTAuditTests().environment(pid=9040)
        control.update(env_control, resource_profile=deepcopy(prepare.PROFILE),
            resource_qualifier=dict(path=prepare.QUALIFIER, sha256='f' * 64))
        point = lambda name: dict(path=name, sha256=prepare.digest(name))
        keys = ['case_' + str(index) + '_' + method for index in range(2) for method in prepare.owner.METHODS]
        physmani = [name for name in keys if name.endswith('_physmani')]
        value.update(resource_profile=deepcopy(prepare.PROFILE), native_successes_diagnostic=0,
            actual_independent_audit=point(prepare.QUALIFICATION_LAUNCH),
            control=point('artifacts/framework/multitask_r10/fake_TRAIN/control.json'),
            lock=point('artifacts/framework/multitask_r10/fake_TRAIN/inputs.json'),
            launcher_overlap=dict(independently_swept_all_child_intervals=True, overlap_verified=True,
                observed_native_peak=4, observed_GPU_peak=2,
                intervals={name: [1, 100, 1000 + index, 10000 + index] for index, name in enumerate(keys)}),
            hardware_observation=dict(actual_two_compute_contexts_observed=True, valid_samples=1,
                required_headroom_mib=2048, minimum_sampled_headroom_mib=4096,
                continuous_hardware_bound_claimed=False,
                witness=dict(headroom_mib=4096, jobs={
                    name: dict(pid=2000 + index, start_ticks=20000 + index)
                    for index, name in enumerate(physmani)})),
            inputs={prepare.canary.RUNTIME + '/plan.json': 'a' * 64}, native_failures_retained=True,
            raw_manifest_model_contents_read=False, wall_clock_contention_invariance_claimed=False)
        audit_launch = dict(source_sha256='f' * 64, pid=9040, start_ticks=environment['observer_start_ticks'],
            frozen_environment=environment, control=deepcopy(value['control']), lock=deepcopy(value['lock']),
            argv=[prepare.QUALIFIER, '--control', value['control']['path'], '--lock', value['lock']['path'],
                  '--lock-sha256', value['lock']['sha256'], '--out', prepare.QUALIFICATION])
        return value, control, audit_launch

    def test_merge_retains_original_826_and_source_keys_and_is_idempotent(self):
        suite = {'fixture/suite.py': 'a' * 64, '/invented/native/source.py': 'b' * 64}
        source = {'/invented/TEST/source_0': 'c' * 64, 'fixture/suite.py': 'a' * 64}
        before = deepcopy((suite, source))
        merged = prepare.merge_maps(suite, source, suite)
        self.assertEqual(merged, {**suite, **source})
        self.assertEqual((suite, source), before)

    def test_merge_refuses_conflicting_keys_or_malformed_SHA_maps(self):
        with self.assertRaises(ValueError):
            prepare.merge_maps({'same': 'a' * 64}, {'same': 'b' * 64})
        for value in ({}, [], {'': 'a' * 64}, {'key': 'g' * 64}, {'key': True}, {'key': 'a' * 63}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                prepare.merge_maps(value)

    def test_supplement_checks_temp_bytes_and_absolute_relative_alias_conflicts(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / 'source.py'
            path.write_text('# invented source\n')
            checksum = prepare.file_sha(path)
            files = {'source.py': checksum}
            self.assertEqual(prepare.add_frozen_file(root, files, 'source.py'), dict(path='source.py', sha256=checksum))
            self.assertEqual(files, {'source.py': checksum})
            for mapping, expected in (({'source.py': '0' * 64}, None),
                    ({'source.py': checksum, str(path): checksum}, None), ({}, '0' * 64)):
                with self.subTest(mapping=mapping), self.assertRaises(ValueError):
                    prepare.add_frozen_file(root, mapping, path, expected=expected)
            (root / 'alias.py').symlink_to(path)
            with self.assertRaises(ValueError):
                prepare.add_frozen_file(root, {}, 'alias.py')

    def test_partition_uses_dynamic_C_and_preserves_every_native_False(self):
        for count in (1, 7, 135, 1919):
            plan, status, terminal = self.grid(count)
            before = deepcopy((plan, status, terminal))
            result = prepare.partition(plan, status, terminal)
            self.assertEqual(len(result['completed']), count)
            self.assertEqual(len(result['remaining']), 1920-count)
            self.assertEqual(set(result['completed']) | set(result['remaining']), set(result['jobs']))
            self.assertEqual((plan, status, terminal), before)
            self.assertTrue(all(row['native_success'] is False for row in status['completed'].values()))

    def test_partition_rejects_other_resume_tiers_unclean_results_and_incomplete_grids(self):
        for change in ('provenance', 'missing', 'duplicate', 'interrupted', 'failure', 'filter', 'retry', 'live'):
            plan, status, terminal = self.grid()
            if change == 'provenance': plan['jobs'][0]['imported_completed'] = True
            elif change == 'missing': plan['jobs'].pop()
            elif change == 'duplicate': plan['jobs'][-1]['job_id'] = plan['jobs'][0]['job_id']
            elif change == 'interrupted': next(iter(status['completed'].values()))['interrupted'] = True
            elif change == 'failure': status['failed'] = {'infra': {}}
            elif change == 'filter': plan['source_outcome_filtering'] = True
            elif change == 'retry': status['automatic_retry'] = True
            else: status['native_slots'] = 1
            with self.subTest(change=change), self.assertRaises(ValueError):
                prepare.partition(plan, status, terminal)

    def test_old_evidence_paths_are_exactly_eleven_and_keep_real_old_output(self):
        row = self.grid()[0]['jobs'][0]
        paths = prepare.old_evidence_paths(row)
        cpu = Path(prepare.PREVIOUS) / 'CPU' / row['job_id']
        native = Path(prepare.PREVIOUS) / 'native' / row['job_id']
        out = Path(prepare.owner.option(row['argv'], '--out'))
        expected = {str(cpu / name) for name in ('launch.json', 'stdout.log', 'stderr.log', 'terminal.json', 'CPU_preflight.json')}
        expected |= {str(native / name) for name in ('launch.json', 'stdout.log', 'stderr.log', 'terminal.json')}
        expected |= {str(out / 'case.json'), str(out / 'admission_driver_receipt.json')}
        self.assertEqual(len(paths), 11)
        self.assertEqual(set(paths), expected)
        self.assertTrue(all(not Path(name).is_absolute() for name in paths))

    def test_formal_bootstrap_has_exact_six_original_scope_bindings(self):
        suite = dict(source_manifests=dict(test=dict(path='fixture/original_TEST_manifest.json')))
        result = prepare.formal_bootstrap_control(suite)
        self.assertEqual(result, dict(schema=prepare.bootstrap.SCHEMA, execution_enabled=True, entry='formal',
            scientific_policy_changed=False, models_or_training_changed=False,
            argv_bindings={'--split': 'test', '--manifest': 'fixture/original_TEST_manifest.json',
                '--suite': prepare.SUITE, '--suite-lock': prepare.SUITE_LOCK,
                '--source-inputs': prepare.SOURCE_INPUTS, '--validation-release': prepare.VAL_RELEASE}))

    def test_previous_owner_and_controller_need_clean_exit_without_invented827_terminal(self):
        previous = self.previous(135)
        proof = prepare.require_previous_exit(previous, lambda _: False)
        self.assertEqual(proof['completed'], 135)
        self.assertEqual(proof['remaining'], 1785)
        self.assertNotIn('controller_terminal', previous)
        for change in ('alive', 'pid_bool', 'STOP', 'controller_success', 'controller_retry'):
            changed = deepcopy(previous)
            if change == 'pid_bool': changed['launch']['pid'] = True
            elif change == 'STOP': changed['controller_failure']['user_stop'] = True
            elif change == 'controller_success': changed['controller_status']['stage'] = 'complete'
            elif change == 'controller_retry': changed['controller_failure']['automatic_retry'] = True
            with self.subTest(change=change), self.assertRaises(ValueError):
                prepare.require_previous_exit(changed, lambda _: change == 'alive')

    def test_actual_independent904_six_TRAIN_resource_proof_accepts_native_False(self):
        value, control, launch = self.resource_proof()
        before = deepcopy((value, control, launch))
        self.assertEqual(prepare.require_resource_proof(value, control, audit_launch=launch), value)
        self.assertEqual((value, control, launch), before)
        self.assertEqual(value['native_successes_diagnostic'], 0)
        launch['argv'][0] = str(prepare.ROOT / prepare.QUALIFIER)
        for flag in ('--control', '--lock'):
            index = launch['argv'].index(flag) + 1
            launch['argv'][index] = str(prepare.ROOT / launch['argv'][index])
        prepare.require_resource_proof(value, control, audit_launch=launch)

    def test_resource_proof_rejects_selfsigning_changed_sources_or_weak_qualification(self):
        for change in ('runner_source', 'source_SHA', 'five', 'not_independent', 'no_overlap', 'no_GPU',
                       'headroom', 'continuous_claim', 'same_main_PID', 'unread_manifest', 'input_SHA',
                       'wrong_audit_path', 'environment_start', 'argv_lock_SHA', 'TEST_read'):
            value, control, launch = self.resource_proof()
            if change == 'runner_source': control['resource_qualifier']['path'] = prepare.canary.SOURCE
            elif change == 'source_SHA': value['source_sha256'] = '0' * 64
            elif change == 'five': value['case_count'] = 5
            elif change == 'not_independent': value['independent_audit_complete'] = False
            elif change == 'no_overlap': value['launcher_overlap']['independently_swept_all_child_intervals'] = False
            elif change == 'no_GPU': value['hardware_observation']['actual_two_compute_contexts_observed'] = False
            elif change == 'headroom': value['hardware_observation']['minimum_sampled_headroom_mib'] = 2047
            elif change == 'continuous_claim': value['hardware_observation']['continuous_hardware_bound_claimed'] = True
            elif change == 'same_main_PID':
                jobs = value['hardware_observation']['witness']['jobs']
                keys = list(jobs)
                jobs[keys[1]] = deepcopy(jobs[keys[0]])
            elif change == 'unread_manifest': value['inputs'] = {'artifacts/framework/protocol/train_manifest.json': 'a' * 64}
            elif change == 'input_SHA': value['inputs'][prepare.canary.RUNTIME + '/plan.json'] = 'invalid'
            elif change == 'wrong_audit_path': value['actual_independent_audit']['path'] = prepare.canary.RUNTIME + '/native_owner_launch.json'
            elif change == 'environment_start': launch['frozen_environment']['observer_start_ticks'] += 1
            elif change == 'argv_lock_SHA': launch['argv'][launch['argv'].index('--lock-sha256') + 1] = '0' * 64
            else: value['TEST_read'] = True
            with self.subTest(change=change), self.assertRaises(ValueError):
                prepare.require_resource_proof(value, control, audit_launch=launch)

    def test_STOP_scopes_exclude_private_observers_and_each_actual_scope_blocks(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name in ('runtime/framework/formal_TEST_report_v830/STOP',
                         'runtime/framework/concurrent_TEST_report_v909/STOP'):
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
            self.assertFalse(prepare.stopped(root))
            for name in prepare.STOP_PATHS:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
                self.assertTrue(prepare.stopped(root))
                path.unlink()

    def test_idle_guard_holds_only_three_temp_resource_locks_and_releases_them(self):
        previous = self.previous()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name in prepare.canary.LOCKS:
                (root / name).parent.mkdir(parents=True, exist_ok=True)
            inventory = dict(host_visibility_available=True, same_uid_inventory_incomplete=False,
                live_project_native_processes=[], ancestor_chain_complete=True)
            with (patch.object(prepare.owner, 'alive', return_value=False),
                  patch.object(prepare.owner, 'process_visibility', return_value=inventory),
                  patch.object(prepare.owner.environment_owner, 'require_frozen_environment', return_value={'invented': True})):
                with prepare.idle_resources({}, previous, root=root):
                    for name in prepare.canary.LOCKS:
                        with (root / name).open('a+') as stream, self.assertRaises(BlockingIOError):
                            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            for name in prepare.canary.LOCKS:
                with (root / name).open('a+') as stream:
                    fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def test_default_main_is_inactive_without_prepare_or_any_data_reads(self):
        output = io.StringIO()
        with (patch.object(prepare, 'prepare') as actual_prepare,
              patch.object(Path, 'read_bytes', side_effect=AssertionError('unexpected data read')),
              patch.object(Path, 'read_text', side_effect=AssertionError('unexpected data read')),
              redirect_stdout(output)):
            prepare.main([])
        actual_prepare.assert_not_called()
        self.assertIn('inactive', output.getvalue())

    def test_same_SHA_symlink_aliases_keep_all_original_keys_without_lookup_authorization(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'real.py').write_text('# invented source\n')
            (root / 'alias.py').symlink_to(root / 'real.py')
            checksum = prepare.file_sha(root / 'real.py')
            files = {'real.py': checksum, 'alias.py': checksum}
            before = dict(files)
            graph = prepare.FrozenInputIndex(root, files)
            self.assertTrue(prepare.require_canonical_consistency(graph, files))
            self.assertEqual(files, before)
            self.assertEqual(graph.keys, set(before))
            self.assertTrue(graph.valid())

    def test_different_SHA_symlink_aliases_fail_canonical_consistency(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'real.py').write_text('# invented source\n')
            (root / 'alias.py').symlink_to(root / 'real.py')
            files = {'real.py': prepare.file_sha(root / 'real.py'), 'alias.py': '0' * 64}
            graph = prepare.FrozenInputIndex(root, files)
            with self.assertRaises(ValueError):
                prepare.require_canonical_consistency(graph, files)

    def test_used_same_SHA_alias_still_fails_original900_unique_canonical_gate(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'real.py').write_text('# invented source\n')
            (root / 'alias.py').symlink_to(root / 'real.py')
            checksum = prepare.file_sha(root / 'real.py')
            files = {'real.py': checksum, 'alias.py': checksum}
            before = dict(files)
            graph = prepare.FrozenInputIndex(root, files)
            self.assertTrue(prepare.require_canonical_consistency(graph, files))
            with self.assertRaises(MultitaskProtocolError):
                with LookupBatch(root, cache=root / 'cache') as batch:
                    batch.locked_pointer(root, files, dict(path='real.py', sha256=checksum))
            self.assertEqual(files, before)


if __name__ == '__main__':
    unittest.main()
