"""CPU-only synthetic mixed753/902 provenance and native-outcome audit tests.

Fixtures are invented in memory or imported from other pure fixture modules.
No test runs main, captures a namespace, or reads experiment data or outputs.
"""
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch
import hashlib
import importlib.util
import tempfile
import unittest

from our_mani.multitask.common_rml_protocol_v530 import PARAMETERS
from our_mani.multitask.evaluation import case_identity, digest
from our_mani.multitask.joint16_tracks_v31 import BLOCKS
from our_mani.multitask.protocol import MultitaskProtocolError
from scripts import run_concurrent_resumed_formal_TEST_owner_v902_r10 as owner
from scripts import audit_concurrent_resumed_formal_TEST_v906_r10 as audit


def fixture_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_owner_fixture = fixture_module('_owner902_fakes_for_audit906',
                               'test_concurrent_resumed_formal_TEST_owner_v902.py')
_outcome_fixture = fixture_module('_outcome749_fakes_for_audit906',
                                 'test_author_outcome_receipt_v749.py')


class ConcurrentTESTAuditTests(unittest.TestCase):
    POLICY = 'e' * 64
    PROFILE = '4' * 64

    def point(self, name):
        return dict(path=name, sha256=digest(name))

    def bindings(self):
        protocol = deepcopy(_outcome_fixture.PROTOCOL)
        protocol.update(runtime_consumer=self.point('scripts/run_task_executor_smoke_author_suite16_v744_r10.py'),
                        original_runtime=self.point('scripts/run_task_executor_smoke_author_suite16_v664_r10.py'))
        return dict(admission_driver=dict(path=owner.ORIGINAL_ENTRY, sha256='a' * 64),
            admission_guard=self.point('our_mani/multitask/formal_suite_admission_v728.py'),
            case_worker=self.point('scripts/run_task_executor_smoke_author_outcome_v743_r10.py'),
            native_runtime=deepcopy(protocol['runtime_consumer']),
            original_native_runtime=deepcopy(protocol['original_runtime']), author_outcome_protocol=protocol)

    def grid(self, kind='batched901'):
        fixture = _owner_fixture.ConcurrentTESTOwnerTests()
        control, old_plan, old_status, old_terminal = fixture.grid()
        bindings = self.bindings()
        for row in old_plan['jobs']:
            row.update(deepcopy(bindings))
            row['argv'][1] = ('scripts/run_aligned_gpu.sh' if row['method'] == 'physmani'
                              else 'scripts/run_local_sim.sh')
            row['argv'][row['argv'].index('--raw')] = '--raw-episode-id'
            row['argv'] += ['--config', 'fixture/config.json', '--frozen-inputs', 'fixture/inputs.json',
                            '--manifest', 'fixture/manifest.json', '--suite-lock', 'fixture/suite_lock.json']
            row['configuration_sha256'] = '1' * 64
            row['input_lock_sha256'] = '2' * 64
        entry = fixture.entry(kind)
        control.update(previous_runtime=owner.PREVIOUS, previous_controller_runtime=owner.PREVIOUS_CONTROLLER,
            execution_entry=entry, resource_profile=fixture.profile(max_CPU_preflight_slots=2),
            author_outcome_protocol=deepcopy(bindings['author_outcome_protocol']))
        old_terminal.update(split='test', execution_protocol_changed=True,
                            author_outcome_protocol=deepcopy(bindings['author_outcome_protocol']))
        new_plan = dict(split='test', expected_cases=1920, policy_sha256=self.POLICY,
            author_outcome_protocol=deepcopy(bindings['author_outcome_protocol']),
            resource_profile=deepcopy(control['resource_profile']), execution_protocol_changed=True,
            imported_completed_cases=len(control['completed_job_ids']), source_outcome_filtering=False,
            jobs=owner.resumed_jobs(old_plan, control, entry))
        return control, old_plan, old_status, old_terminal, new_plan

    def environment(self, pid=9020):
        control = dict(environment_wrapper=self.point('scripts/run_frozen_system_environment_v820_r10.py'),
            environment_manifest=self.point('fixture/environment_manifest.json'),
            environment_namespace_proof=self.point('fixture/namespace_proof.json'))
        restored = ('/usr/bin/git', '/usr/bin/git-shell', '/usr/bin/scalar',
                    '/usr/lib/x86_64-linux-gnu/libXtst.so.6.1.0')
        receipt = dict(schema='r10_actual_frozen_OS_namespace_receipt_v820', complete=True,
            checked_OS_files=6418, all_current_OS_targets_full_byte_SHA_verified=True,
            host_PID_network_namespace_preserved=True, proc_or_dev_replaced=False,
            task_parameters_or_strategy_changed=False, project_writable=True,
            wrapper=deepcopy(control['environment_wrapper']), manifest=deepcopy(control['environment_manifest']),
            namespace_qualification=deepcopy(control['environment_namespace_proof']),
            current_mount_namespace='mnt:[invented902]', mount_namespace='mnt:[invented902]',
            reference_mount_namespace='mnt:[invented_host]', qualified_host_mount_namespace='mnt:[invented_host]',
            observer_pid=pid, pid=pid, observer_start_ticks=90200,
            outer_wrapper_pid=9000, outer_wrapper_start_ticks=90000,
            current_OS_target_SHA_map_sha256='3' * 64,
            restored_four_target_SHA256={name: digest(name) for name in restored},
            source_locks={'fixture/old_inputs.json': '5' * 64})
        return control, receipt

    def mixed_launch(self, *, imported=False, kind='batched901', check=True, method='physmani'):
        values = self.grid(kind)
        control, _, _, _, plan = values
        candidates = [row for row in plan['jobs'] if row['imported_completed'] is imported
                      and row['method'] == method]
        row = candidates[0]
        bindings = self.bindings()
        identity = dict(sha256='6' * 64)
        pid = 7530 if imported else 9020
        stage_owner = dict(pid=pid, start_ticks=pid * 10)
        actual = dict(argv=deepcopy(row['argv']) + (['--check'] if check else []),
            check_only=check, owner_pid=pid, launcher_pid=pid + 1, launcher_start_ticks=pid * 10 + 1,
            execution_protocol_changed=True, native_slots=0 if check else 1,
            GPU_slots=0 if check else int(method == 'physmani'),
            expected_identity_sha256=None if check else identity['sha256'], **deepcopy(bindings))
        if not imported:
            environment_control, receipt = self.environment(pid)
            control.update(environment_control)
            stage_owner['frozen_environment'] = receipt
            actual.update(execution_entry=deepcopy(row['execution_entry']),
                physical_argv=deepcopy(row['physical_argv']) + (['--check'] if check else []),
                inherited_frozen_environment=deepcopy(receipt),
                current_mount_namespace=receipt['current_mount_namespace'])
        return actual, row, bindings, identity, stage_owner, control

    def case(self, *, task='beat_the_rotating_buzz', method='context_geometry', raw='invented_episode_0',
             ticks=0, success=False, fatal=False, motion=True):
        case = _outcome_fixture.fixture(block=task in BLOCKS, method=method, ticks=ticks,
                                        success=success, fatal=fatal)
        case['task'] = task
        source = dict(task=task, raw_episode_id=raw, path='fixture/invented_source')
        identity = case_identity(split='test', task=task, raw_episode_id=raw, method=method,
            source_row=source, config_sha256='1' * 64, inputs_sha256='2' * 64)
        identity['source_input_lock_sha256'] = '7' * 64
        identity['sha256'] = digest({key: value for key, value in identity.items() if key != 'sha256'})
        protocol = self.bindings()['author_outcome_protocol']
        case.update(identity=deepcopy(identity), raw_episode_id=raw, parent_raw_episode_id=None,
            source_raw_episode_id=raw, split='test', process_id=12345,
            case_isolation='fresh_simulator_process_per_episode', formal_test_used=True,
            TEST_used_for_fitting=False, demo_only=False, development_demo_only=False,
            recorded_initial_conditions=dict(metric_only=True, policy_use_allowed=False),
            author_outcome_protocol=deepcopy(protocol),
            counted_clock=dict(ticks=ticks, dt=.05, seconds=ticks * .05),
            component_domain=dict(purpose='evaluation', split='test'),
            full16_frozen_execution=dict(policy_sha256=self.POLICY, source_row_sha256=digest(source)),
            common_author_RML_parameters=dict(schema='r10_observed_common_author_RML_parameters_v530',
                profile_sha256=self.PROFILE, actual_parameters=deepcopy(PARAMETERS), complete=True,
                native_parameters_mutated=False, actual_motion_bound_claimed=False),
            all_stage_counted_joint_motion=dict(events=[dict(tick=i + 1) for i in range(ticks)],
                all_stage_actual_window_checks_passed=motion,
                peak_actual_window_velocity=0. if ticks == 0 else 1.,
                peak_actual_window_acceleration=0. if ticks == 0 else 9.))
        case['alignment_receipt'].update(clock=dict(ticks=ticks, dt=.05, budget_seconds=10.,
            simulation_seconds=ticks * .05), actual_physics_seconds=ticks * .05,
            actual_task_seconds=ticks * .05)
        return case, identity, protocol

    def test_exact_mixed_1920_plan_keeps_old_False_and_new_entry_provenance(self):
        for kind in ('original751', 'batched901'):
            values = self.grid(kind)
            before = deepcopy(values)
            audit.require_resume_plan(*values)
            self.assertEqual(values, before)
            self.assertFalse(values[2]['completed'][values[0]['completed_job_ids'][0]]['native_success'])

    def test_mixed_plan_rejects_flattened_runtime_or_changed_rows_and_worker(self):
        for change in ('old_runtime', 'new_runtime', 'row_sha', 'previous_worker', 'old_argv',
                       'new_out', 'new_outer', 'new_inner', 'new_worker', 'old_entry'):
            values = list(self.grid())
            old = values[4]['jobs'][0]
            new = values[4]['jobs'][len(values[0]['completed_job_ids'])]
            if change == 'old_runtime': old['evidence_runtime'] = owner.RUNTIME
            elif change == 'new_runtime': new['evidence_runtime'] = owner.PREVIOUS
            elif change == 'row_sha': old['previous_row_sha256'] = '0' * 64
            elif change == 'previous_worker': new['previous_worker_sha256'] = '0' * 64
            elif change == 'old_argv': old['argv'].append('--invented')
            elif change == 'new_out':
                new['physical_argv'][new['physical_argv'].index('--out') + 1] = 'fixture/old_output'
            elif change == 'new_outer': new['argv'][2] = owner.ORIGINAL_ENTRY
            elif change == 'new_inner': new['physical_argv'][2] = owner.BATCH_ENTRY
            elif change == 'new_worker': new['worker_sha256'] = 'a' * 64
            else: old['execution_entry'] = deepcopy(values[0]['execution_entry'])
            with self.subTest(change=change), self.assertRaises(ValueError):
                audit.require_resume_plan(*values)

    def test_mixed_plan_rejects_missing_duplicate_or_nonidentical_source_grid(self):
        for change in ('missing_new', 'duplicate_new', 'duplicate_old', 'asymmetric_source', 'False_dropped'):
            control, old_plan, old_status, old_terminal, new_plan = self.grid()
            if change == 'missing_new': new_plan['jobs'].pop()
            elif change == 'duplicate_new': new_plan['jobs'][-1] = deepcopy(new_plan['jobs'][0])
            elif change == 'duplicate_old': old_plan['jobs'][-1]['job_id'] = old_plan['jobs'][0]['job_id']
            elif change == 'asymmetric_source': old_plan['jobs'][-1]['raw'] = 'different_unique_source'
            else: old_status['completed'].pop(control['completed_job_ids'][0])
            with self.subTest(change=change), self.assertRaises(ValueError):
                audit.require_resume_plan(control, old_plan, old_status, old_terminal, new_plan)

    def test_each_CPU_and_native_terminal_requires_clean_one_time_completion(self):
        clean = dict(complete=True, returncode=0, timed_out=False, interrupted=False, automatic_retry=False)
        audit.require_clean_terminal(clean)
        for field, bad in [('complete', 1), ('returncode', False), ('returncode', 1),
                           ('timed_out', 0), ('timed_out', True), ('interrupted', 0),
                           ('interrupted', True), ('automatic_retry', 0), ('automatic_retry', True)]:
            with self.subTest(field=field, bad=bad), self.assertRaises(ValueError):
                audit.require_clean_terminal(dict(clean, **{field: bad}))
        for missing in clean:
            value = deepcopy(clean)
            value.pop(missing)
            with self.subTest(missing=missing), self.assertRaises(ValueError):
                audit.require_clean_terminal(value)

    def test_old753_launch_binds_its_real_owner_without_inventing_new_fields(self):
        for check in (True, False):
            actual, row, bindings, identity, stage_owner, control = self.mixed_launch(imported=True, check=check)
            self.assertNotIn('execution_entry', actual)
            self.assertNotIn('inherited_frozen_environment', actual)
            audit.require_mixed_launch(actual, row, bindings, check=check, identity=identity,
                                       stage_owner=stage_owner, control=control)
            for change in ('owner', 'argv', 'check', 'identity'):
                changed = deepcopy(actual)
                if change == 'owner': changed['owner_pid'] = 9020
                elif change == 'argv': changed['argv'][2] = owner.BATCH_ENTRY
                elif change == 'check': changed['check_only'] = int(check)
                else: changed['expected_identity_sha256'] = '0' * 64
                with self.subTest(check=check, change=change), self.assertRaises(ValueError):
                    audit.require_mixed_launch(changed, row, bindings, check=check, identity=identity,
                                               stage_owner=stage_owner, control=control)

    def test_new902_launch_binds_actual_outer_inner_owner_and_inherited_namespace(self):
        for kind in ('original751', 'batched901'):
            for check in (True, False):
                actual, row, bindings, identity, stage_owner, control = self.mixed_launch(kind=kind, check=check)
                audit.require_mixed_launch(actual, row, bindings, check=check, identity=identity,
                                           stage_owner=stage_owner, control=control)
                for change in ('owner', 'physical', 'entry', 'namespace', 'receipt', 'GPU', 'binding'):
                    changed = deepcopy(actual)
                    if change == 'owner': changed['owner_pid'] = 7530
                    elif change == 'physical': changed['physical_argv'].append('--invented')
                    elif change == 'entry': changed['execution_entry']['pointer']['sha256'] = '0' * 64
                    elif change == 'namespace': changed['current_mount_namespace'] = 'mnt:[auditor_not_owner]'
                    elif change == 'receipt': changed['inherited_frozen_environment']['observer_pid'] = 9999
                    elif change == 'GPU': changed['GPU_slots'] = 1 if check else 0
                    else: changed['admission_driver'] = deepcopy(row['execution_entry']['pointer'])
                    if change == 'binding' and kind == 'original751':
                        changed['admission_driver']['sha256'] = '0' * 64
                    with self.subTest(kind=kind, check=check, change=change), self.assertRaises(ValueError):
                        audit.require_mixed_launch(changed, row, bindings, check=check, identity=identity,
                                                   stage_owner=stage_owner, control=control)

    def test_zero_tick_policy_negative_and_reset_success_are_valid_core_and_block_cases(self):
        for task in ('beat_the_rotating_buzz', next(iter(BLOCKS))):
            for method in owner.METHODS:
                for success in (False, True):
                    case, identity, protocol = self.case(task=task, method=method, success=success)
                    before = deepcopy(case)
                    audit.require_case_outcome(case, identity, protocol, self.POLICY, self.PROFILE)
                    self.assertEqual(case, before)
                    self.assertEqual(case['counted_clock']['ticks'], 0)
                    self.assertEqual(case['all_stage_counted_joint_motion']['events'], [])

    def test_native_False_fatal_or_policy_negative_and_acceleration9_are_not_release_thresholds(self):
        for fatal in (False, True):
            for method in owner.METHODS:
                case, identity, protocol = self.case(method=method, ticks=10, fatal=fatal, motion=False)
                audit.require_case_outcome(case, identity, protocol, self.POLICY, self.PROFILE)
                self.assertFalse(case['official_task_success'])
                self.assertFalse(case['all_stage_counted_joint_motion']['all_stage_actual_window_checks_passed'])
                self.assertGreater(case['all_stage_counted_joint_motion']['peak_actual_window_acceleration'], 6.)

    def test_aborted_actions_uncovered_ticks_or_zero_tick_fatal_cannot_be_completed_native_cases(self):
        for change in ('aborted', 'uncovered', 'zero_fatal', 'extra_zero_motion'):
            case, identity, protocol = self.case(ticks=0 if change.startswith('zero') or change == 'extra_zero_motion' else 10)
            if change == 'aborted':
                case['native_outcome_sampling']['aborted'] = True
                _outcome_fixture.sync(case)
            elif change == 'uncovered':
                case['native_outcome_sampling']['completed_actions'][0]['start_tick'] = 1
                _outcome_fixture.sync(case)
            elif change == 'zero_fatal':
                case['native_terminal_failure'] = True
                fatal = deepcopy(_outcome_fixture.FATAL)
                case['native_outcome_sampling']['sampled_outcome'] = deepcopy(fatal)
                case['alignment_receipt'].update(native_terminal_failure_latched=True,
                    native_terminal_latched=True, native_outcome=fatal, native_terminal_tick=0)
                _outcome_fixture.sync(case)
            else: case['all_stage_counted_joint_motion']['events'] = [dict(tick=1)]
            with self.subTest(change=change), self.assertRaises((ValueError, MultitaskProtocolError)):
                audit.require_case_outcome(case, identity, protocol, self.POLICY, self.PROFILE)

    def test_case_identity_policy_RML_and_full_motion_measurements_are_required(self):
        for change in ('identity', 'forged_identity_hash', 'policy', 'RML', 'clock', 'event_count', 'motion_missing',
                       'cleanup', 'split', 'domain', 'native_bool'):
            case, identity, protocol = self.case(ticks=10)
            if change == 'identity': case['identity']['sha256'] = '0' * 64
            elif change == 'forged_identity_hash':
                case['identity']['sha256'] = identity['sha256'] = '0' * 64
            elif change == 'policy': case['full16_frozen_execution']['policy_sha256'] = '0' * 64
            elif change == 'RML': case['common_author_RML_parameters']['actual_parameters']['max_acceleration'] = 6.
            elif change == 'clock': case['counted_clock']['seconds'] = 100.
            elif change == 'event_count': case['all_stage_counted_joint_motion']['events'].pop()
            elif change == 'motion_missing': case['all_stage_counted_joint_motion'].pop('all_stage_actual_window_checks_passed')
            elif change == 'cleanup': case['cleanup_completed'] = False
            elif change == 'split': case['split'] = 'validation'
            elif change == 'domain': case['component_domain']['purpose'] = 'development'
            else: case['official_task_success'] = 0
            with self.subTest(change=change), self.assertRaises((ValueError, MultitaskProtocolError)):
                audit.require_case_outcome(case, identity, protocol, self.POLICY, self.PROFILE)

    def test_summary_keeps_all_1920_native_False_including_zero_ticks_and_motion_violations(self):
        cases = []
        for task in owner.TASKS:
            for method in owner.METHODS:
                for index in range(40):
                    case, _, _ = self.case(task=task, method=method, raw='invented_episode_' + str(index),
                                           ticks=0 if index == 0 else 10, motion=index == 0)
                    cases.append(case)
        result = audit.summarize(cases)
        self.assertEqual({row['task'] for row in result}, set(owner.TASKS))
        self.assertEqual(len(result), 16)
        episodes = 0
        for row in result:
            self.assertEqual(set(row['methods']), set(owner.METHODS))
            for method in owner.METHODS:
                report = row['methods'][method]
                self.assertEqual(report['episodes'], 40)
                self.assertEqual(report['successes'], 0)
                self.assertEqual(report['success_rate'], 0.)
                self.assertEqual(report['native_terminal_failures'], 0)
                self.assertEqual(report['actual_window_acceleration_peak'], 9.)
                self.assertEqual(report['measured_motion_violation_cases'], 39)
                self.assertEqual(report['zero_tick_cases'], 1)
                self.assertEqual(report['other_nonterminal_stops'], 40)
                episodes += report['episodes']
        self.assertEqual(episodes, 1920)

    def test_evidence_ledger_rechecks_temp_bytes_allows_unrelated_outputs_and_closes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            payload = b'{"invented": true}\n'
            (root / 'evidence.json').write_bytes(payload)
            expected = hashlib.sha256(payload).hexdigest()
            ledger = audit.EvidenceLedger(root, cache=root / 'cache')
            self.assertEqual(ledger.json('evidence.json', expected=expected), dict(invented=True))
            self.assertEqual(ledger.pointer(root / 'evidence.json'), dict(path='evidence.json', sha256=expected))
            (root / 'unrelated_output').mkdir()
            (root / 'unrelated_output/result.txt').write_text('unrelated')
            self.assertEqual(ledger.close(), {str(root / 'evidence.json'): expected})
            self.assertTrue(ledger.closed)
            with self.assertRaises(ValueError):
                ledger.pointer('evidence.json')

    def test_evidence_ledger_rejects_changed_bytes_replaced_inode_and_symlink_names(self):
        for change in ('bytes', 'inode', 'symlink', 'wrong_SHA'):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                path = root / 'evidence.json'
                path.write_bytes(b'{"invented": 1}\n')
                ledger = audit.EvidenceLedger(root, cache=root / 'cache')
                if change == 'symlink':
                    (root / 'alias.json').symlink_to(path)
                    with self.assertRaises(ValueError):
                        ledger.pointer('alias.json')
                    continue
                if change == 'wrong_SHA':
                    with self.assertRaises(ValueError):
                        ledger.json('evidence.json', expected='0' * 64)
                    continue
                ledger.pointer('evidence.json')
                if change == 'bytes':
                    path.write_bytes(b'{"invented": 2}\n')
                else:
                    replacement = root / 'replacement.json'
                    replacement.write_bytes(path.read_bytes())
                    replacement.replace(path)
                with self.assertRaises((ValueError, MultitaskProtocolError)):
                    ledger.close()
                self.assertFalse(ledger.closed)

    def owner_records(self, root):
        control, _, _, _, plan = self.grid()
        environment_control, receipt = self.environment()
        control.update(environment_control,
            resource_qualification=self.point('fixture/TRAIN_resource_qualification.json'),
            execution_implementation=owner.execution_implementation(control['execution_entry'],
                _owner_fixture.ConcurrentTESTOwnerTests().source_bindings()))
        control.update({key: self.point('fixture/' + name) for key, name in (
            ('suite', 'suite.json'), ('suite_lock', 'suite_lock.json'),
            ('source_inputs', 'source_inputs.json'), ('validation_release', 'VAL_release.json'))})
        resume_control = self.point('fixture/resume_control.json')
        resume_lock = self.point('fixture/resume_inputs.json')
        bindings = self.bindings()
        old_source = root / 'scripts/run_formal_suite16_owner_v753_r10.py'
        new_source = root / 'scripts/run_concurrent_resumed_formal_TEST_owner_v902_r10.py'
        old_source.parent.mkdir(parents=True)
        old_source.write_text('# invented old source\n')
        new_source.write_text('# invented new source\n')
        common_bindings = dict(split='test', policy_sha256=self.POLICY, automatic_retry=False,
            execution_protocol_changed=True, **deepcopy(bindings))
        old_launch = dict(common_bindings, pid=7530, start_ticks=75300,
            source_sha256=audit.file_sha(old_source), suite_lock_sha256='8' * 64,
            split_source_lock_sha256='7' * 64, max_native_slots=2, max_GPU_slots=1,
            max_CPU_preflight_slots=2)
        common = dict(common_bindings, resume_control=resume_control, resume_control_lock=resume_lock,
            resource_profile=deepcopy(control['resource_profile']),
            resource_qualification=deepcopy(control['resource_qualification']),
            execution_implementation=deepcopy(control['execution_implementation']), frozen_environment=deepcopy(receipt))
        count = len(control['completed_job_ids'])
        launch = dict(common, pid=9020, start_ticks=90200,
            source_sha256=audit.file_sha(new_source), suite_lock_sha256='8' * 64,
            split_source_lock_sha256='7' * 64, execution_entry=deepcopy(control['execution_entry']),
            base_stage_owner=owner.pointer(old_source), imported_completed_cases=count)
        launch.update({key: value for key, value in control['resource_profile'].items()
                       if key.startswith('max_')})
        launch['argv'] = [str(new_source)]
        for flag, value in (
            ('--suite', control['suite']['path']), ('--suite-lock', control['suite_lock']['path']),
            ('--source-inputs', control['source_inputs']['path']),
            ('--validation-release', control['validation_release']['path']),
            ('--runtime', owner.RUNTIME), ('--out', owner.OUTPUT),
            ('--resume-control', resume_control['path']), ('--resume-lock', resume_lock['path']),
            ('--resume-lock-sha256', resume_lock['sha256'])):
            launch['argv'] += [flag, value]
        plan.update(deepcopy(common), imported_completed_cases=count)
        clean = dict(complete=True, returncode=0, timed_out=False, interrupted=False,
                     automatic_retry=False, native_success=False, motion_pass=False)
        status = dict(common, imported_completed=count, expected=1920,
            completed={row['job_id']: deepcopy(clean) for row in plan['jobs']}, failed={},
            active=[], checking=[], ready_jobs=[], CPU_ready=0, native_slots=0, GPU_slots=0,
            CPU_launched=1920-count, owner_error=None, stop_requested=False, drain_requested=False)
        terminal = dict(common, imported_completed_cases=count, resumed_new_cases=1920-count,
            complete=True, expected=1920, completed=1920, failed=0, owner_error=None,
            stop_requested=False, drain_requested=False, drained=False, CPU_not_launched=0,
            CPU_ready_unexecuted=[], formal_release_requires_independent_audit=True)
        _, auditor_environment = self.environment(pid=9060)
        auditor_environment.update(current_mount_namespace='mnt:[invented906]', mount_namespace='mnt:[invented906]',
                                   observer_start_ticks=90600)
        kwargs = dict(source_sha='7' * 64, suite_lock_sha='8' * 64,
            driver=bindings['admission_driver'], guard=bindings['admission_guard'],
            resume_control=resume_control, resume_lock=resume_lock, expected_environment=auditor_environment)
        return [control, old_launch, launch, plan, status, terminal], kwargs

    def test_owner_records_accept_real_tiers_and_separate_auditor_namespace(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with (patch.object(audit, 'ROOT', root), patch.object(owner, 'ROOT', root),
                  patch.object(owner, '__file__', str(root / 'scripts/run_concurrent_resumed_formal_TEST_owner_v902_r10.py'))):
                args, kwargs = self.owner_records(root)
                result = audit.require_owner_records(*args, **kwargs)
                self.assertEqual(result, args[2]['frozen_environment'])
                self.assertNotEqual(result['current_mount_namespace'],
                                    kwargs['expected_environment']['current_mount_namespace'])

    def test_owner_records_reject_STOP_drain_unfinished_grid_or_changed_sources(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with (patch.object(audit, 'ROOT', root), patch.object(owner, 'ROOT', root),
                  patch.object(owner, '__file__', str(root / 'scripts/run_concurrent_resumed_formal_TEST_owner_v902_r10.py'))):
                original, kwargs = self.owner_records(root)
                for change in ('old_source', 'new_source', 'new_start', 'resource', 'entry',
                               'env_start', 'STOP', 'DRAIN', 'unfinished', 'missing_completed', 'active',
                               'argv_SHA', 'argv_duplicate', 'argv_scope', 'old_quota', 'new_quota'):
                    args = deepcopy(original)
                    if change == 'old_source': args[1]['source_sha256'] = '0' * 64
                    elif change == 'new_source': args[2]['source_sha256'] = '0' * 64
                    elif change == 'new_start': args[2]['start_ticks'] = True
                    elif change == 'resource': args[4]['resource_profile']['max_GPU_slots'] = 1
                    elif change == 'entry': args[2]['execution_entry']['pointer']['sha256'] = '0' * 64
                    elif change == 'env_start':
                        for record in args[2:]: record['frozen_environment']['observer_start_ticks'] = 90201
                    elif change == 'STOP': args[5]['stop_requested'] = True
                    elif change == 'DRAIN': args[5]['drain_requested'] = True
                    elif change == 'unfinished': args[5]['CPU_ready_unexecuted'] = ['not_executed']
                    elif change == 'missing_completed': args[4]['completed'].pop(next(iter(args[4]['completed'])))
                    elif change == 'active': args[4]['active'] = ['still_active']
                    elif change == 'argv_SHA':
                        argv = args[2]['argv']
                        argv[argv.index('--resume-lock-sha256') + 1] = '0' * 64
                    elif change == 'argv_duplicate': args[2]['argv'] += ['--out', owner.OUTPUT]
                    elif change == 'old_quota': args[1]['max_native_slots'] = 4
                    elif change == 'new_quota': args[2]['max_GPU_slots'] = 1
                    else:
                        argv = args[2]['argv']
                        argv[argv.index('--source-inputs') + 1] = 'fixture/different_source_inputs.json'
                    with self.subTest(change=change), self.assertRaises(ValueError):
                        audit.require_owner_records(*args, **kwargs)

    def test_independent_audit_respects_all_four_existing_STOP_scopes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with patch.object(audit, 'ROOT', root), patch.object(owner, 'ROOT', root):
                audit.require_not_stopped()
                for relative in ('runtime/framework/multitask_r10_STOP', owner.RUNTIME + '/STOP',
                                 owner.PREVIOUS + '/STOP', owner.PREVIOUS_CONTROLLER + '/STOP'):
                    path = root / relative
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.touch()
                    with self.subTest(relative=relative), self.assertRaises((ValueError, InterruptedError)):
                        audit.require_not_stopped()
                    path.unlink()
                    audit.require_not_stopped()

    def test_original_plan_uses_real827_ART823_output_and_unchanged751_route(self):
        control, old_plan, _, _, _ = self.grid()
        for key, name in (('suite', 'suite.json'), ('suite_lock', 'suite_lock.json'),
                          ('source_inputs', 'source_inputs.json'), ('validation_release', 'VAL_release.json')):
            control[key] = self.point('fixture/' + name)
        row = deepcopy(old_plan['jobs'][0])
        source = dict(task=row['task'], raw_episode_id=row['raw'], path='fixture/original_source')
        bindings = self.bindings()
        route = dict(task=row['task'], method=row['method'], worker=bindings['case_worker'],
            consumer=bindings['native_runtime'], configuration=self.point('fixture/config.json'),
            frozen_inputs=self.point('fixture/inputs.json'))
        suite = dict(source_manifests=dict(test=self.point('fixture/test_manifest.json')))
        row.update(source_row_sha256=digest(source), configuration_sha256=route['configuration']['sha256'],
                   input_lock_sha256=route['frozen_inputs']['sha256'])
        original_out = str(Path(audit.ORIGINAL_OUTPUT) / row['task'] / row['raw'] / row['method'])
        row['argv'] = ['bash', 'scripts/run_local_sim.sh', owner.ORIGINAL_ENTRY,
            '--config', route['configuration']['path'], '--manifest', suite['source_manifests']['test']['path'],
            '--frozen-inputs', route['frozen_inputs']['path'], '--suite', control['suite']['path'],
            '--suite-lock', control['suite_lock']['path'], '--source-inputs', control['source_inputs']['path'],
            '--task', row['task'], '--raw-episode-id', row['raw'], '--method', row['method'], '--split', 'test',
            '--out', original_out, '--validation-release', control['validation_release']['path']]
        plan = dict(jobs=[row])
        args = (control, suite, [source], dict(routes=dict(routes=[route])),
                bindings['admission_driver'], bindings['admission_guard'])
        audit.require_original_plan(plan, *args)
        self.assertTrue(original_out.startswith('artifacts/framework/multitask_r10/formal_VAL_env_resume_v823/test/'))
        for change in ('runtime_out', 'source_SHA', 'route_SHA'):
            changed = deepcopy(plan)
            if change == 'runtime_out':
                argv = changed['jobs'][0]['argv']
                argv[argv.index('--out') + 1] = str(Path(owner.PREVIOUS) / 'test' / row['job_id'])
            elif change == 'source_SHA': changed['jobs'][0]['source_row_sha256'] = '0' * 64
            else: changed['jobs'][0]['configuration_sha256'] = '0' * 64
            with self.subTest(change=change), self.assertRaises(ValueError):
                audit.require_original_plan(changed, *args)


if __name__ == '__main__':
    unittest.main()
