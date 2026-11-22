"""Synthetic CPU checks for immutable evidence and explicit resume control."""
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import hashlib
import json
import unittest

from scripts import prepare_formal_resume_v777_r10 as prepare
from our_mani.multitask.evaluation import digest, file_sha


def stopped_fixture():
    policy = 'a' * 64
    names = ['original_case_%04d' % i for i in range(943)] + sorted(prepare.INTERRUPTED)
    rows = [dict(job_id=name, task='synthetic', raw=name, method='cv_geometry') for name in names]
    plan = dict(split='validation', expected_cases=945, jobs=rows, policy_sha256=policy,
                source_outcome_filtering=False, automatic_retry=False)
    interrupted = {key: dict(stage='native', complete=False, interrupted=True, timed_out=False,
        returncode=-2, automatic_retry=False) for key in prepare.INTERRUPTED}
    status = dict(split='validation', expected=945, completed={key: {} for key in names[:95]},
        failed=interrupted, stop_requested=True, active=[], checking=[], native_slots=0, GPU_slots=0)
    terminal = dict(split='validation', expected=945, complete=False, stop_requested=True,
                    completed=95, failed=2, policy_sha256=policy, native_successes=0)
    return plan, status, terminal


class ResumePreparationTests(unittest.TestCase):
    def test_partition_preserves_all_native_failures_and_exact_two_interruptions(self):
        plan, status, terminal = stopped_fixture()
        jobs, completed, interrupted, remaining, policy = prepare.partition(plan, status, terminal)
        self.assertEqual((len(jobs), len(completed), len(interrupted), len(remaining)), (945, 95, 2, 850))
        self.assertEqual(set(completed) | set(remaining), set(jobs))
        self.assertFalse(set(completed) & set(remaining))
        self.assertTrue(set(interrupted) <= set(remaining))
        self.assertEqual(policy, plan['policy_sha256'])

    def test_partition_rejects_duplicates_foreign_results_and_censored_imports(self):
        for mutation in ('duplicate', 'foreign', 'partial', 'non_user_failure', 'active', 'policy', 'wrong_split'):
            plan, status, terminal = stopped_fixture()
            if mutation == 'duplicate': plan['jobs'][-1] = deepcopy(plan['jobs'][0])
            elif mutation == 'foreign': status['completed']['foreign'] = status['completed'].pop(next(iter(status['completed'])))
            elif mutation == 'partial': terminal['completed'] = 94
            elif mutation == 'non_user_failure': next(iter(status['failed'].values()))['interrupted'] = False
            elif mutation == 'active': status['active'] = ['still_running']
            elif mutation == 'policy': terminal['policy_sha256'] = 'b' * 64
            else: plan['split'] = 'test'
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                prepare.partition(plan, status, terminal)

    def test_process_identity_check_rejects_live_old_process_but_allows_pid_reuse(self):
        with TemporaryDirectory() as directory:
            proc = Path(directory); leaf = proc / '123'; leaf.mkdir()
            fields = ['S'] + ['0'] * 19; fields[19] = '456'
            (leaf / 'stat').write_text('123 (worker with spaces) ' + ' '.join(fields))
            with self.assertRaisesRegex(ValueError, 'still owns'):
                prepare.process_exited(123, 456, proc=proc)
            self.assertTrue(prepare.process_exited(123, 455, proc=proc))
            self.assertTrue(prepare.process_exited(999, 456, proc=proc))
            fields[0] = 'Z'; (leaf / 'stat').write_text('123 (worker) ' + ' '.join(fields))
            self.assertTrue(prepare.process_exited(123, 456, proc=proc))

    def test_added_evidence_rejects_changed_bytes_conflicting_sha_and_project_escape(self):
        with TemporaryDirectory() as directory, TemporaryDirectory() as outside:
            root = Path(directory); path = root / 'record.json'; path.write_text('{}')
            original = file_sha(path); files = {str(path): original}
            self.assertEqual(prepare.freeze_file(root, files, path)['sha256'], original)
            path.write_text('{"changed":true}')
            with self.assertRaisesRegex(ValueError, 'conflicting'):
                prepare.freeze_file(root, files, path)
            with self.assertRaisesRegex(ValueError, 'bytes differ'):
                prepare.freeze_file(root, {}, path, expected=original)
            other = Path(outside) / 'external'; other.write_text('external')
            with self.assertRaisesRegex(ValueError, 'project evidence'):
                prepare.freeze_file(root, {}, other)

    def test_all_new_control_sources_and_tests_must_exist_before_setup(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            for name in prepare.REQUIRED_SOURCES[:-1]:
                path = root / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_text('source')
            with self.assertRaisesRegex(ValueError, 'not ready'):
                prepare.require_new_sources(root)
            self.assertFalse((root / prepare.SETUP).exists())

    def test_real_CPU_helper_checks_bindings_actual_argv_and_source_scope_without_starting_native(self):
        with TemporaryDirectory() as directory:
            root = Path(directory); source = root / prepare.SOURCE_INPUTS
            source.parent.mkdir(parents=True); source.write_text('{"original_source_scope":true}')
            runtime = root / prepare.PREVIOUS; leaf = runtime / 'CPU' / 'case'; leaf.mkdir(parents=True)
            bindings = {key: {'path': key, 'sha256': 'a' * 64} for key in prepare.BINDING_KEYS}
            row = dict(bindings, job_id='case', argv=['bash', 'run_local_sim.sh', 'driver751'],
                       method='cv_geometry', configuration_sha256='b' * 64, input_lock_sha256='c' * 64)
            identity = dict(sha256='d' * 64, source_row_sha256=digest({}), config_sha256='b' * 64,
                            frozen_inputs_sha256='c' * 64, source_input_lock_sha256=file_sha(source))
            cpu = dict(bindings, complete=True, native_started=False, GPU_started=False, actual_split='validation',
                       component_domain={'purpose': 'evaluation'}, execution_protocol_changed=True, identity=identity)
            launch = dict(bindings, argv=row['argv'] + ['--check'], check_only=True,
                          execution_protocol_changed=True, native_slots=0, GPU_slots=0, expected_identity_sha256=None)
            terminal = dict(complete=True, returncode=0, interrupted=False, timed_out=False)
            for mutation in (None, 'native_started', 'source_lock', 'argv'):
                value = deepcopy(cpu); actual_launch = deepcopy(launch)
                if mutation == 'native_started': value['native_started'] = True
                elif mutation == 'source_lock': value['identity']['source_input_lock_sha256'] = '0' * 64
                elif mutation == 'argv': actual_launch['argv'] += ['--hidden-option', 'changed']
                for name, data in [('CPU_preflight.json', value), ('stdout.log', value),
                                   ('launch.json', actual_launch), ('terminal.json', terminal)]:
                    (leaf / name).write_text(json.dumps(data))
                (leaf / 'stderr.log').write_text('')
                if mutation is None:
                    self.assertEqual(prepare.check_CPU(root, {}, runtime, row, {}), identity)
                else:
                    with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                        prepare.check_CPU(root, {}, runtime, row, {})

    def make_setup_fixture(self, root, external):
        def write(name, value):
            path = root / name; path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(value) if not isinstance(value, str) else value)
            return path
        for name in prepare.REQUIRED_SOURCES:
            write(name, 'frozen source ' + name)
        write('scripts/run_formal_suite16_owner_v753_r10.py', 'original owner')
        plan, status, terminal = stopped_fixture()
        protocol = {'schema': 'synthetic_original_author_protocol',
                    'runtime_consumer': {'path': 'runtime744', 'sha256': 'd' * 64},
                    'original_runtime': {'path': 'runtime664', 'sha256': 'e' * 64}}
        suite = {'author_outcome_protocol': protocol,
                 'source_manifests': {'validation': {'path': 'validation_manifest', 'sha256': 'f' * 64}},
                 'formal_case_driver': {'path': 'driver751', 'sha256': '1' * 64},
                 'formal_admission_guard': {'path': 'guard728', 'sha256': '2' * 64}}
        write(prepare.SUITE, suite)
        write(prepare.SUITE_LOCK, dict(files={str(external): file_sha(external)}))
        write(prepare.SOURCE_INPUTS, {'original_validation_lock': True})
        terminal['author_outcome_protocol'] = protocol
        terminal.update(execution_protocol_changed=True, automatic_retry=False)
        launch = dict(pid=123, start_ticks=456, split='validation', policy_sha256=plan['policy_sha256'],
            source_sha256=file_sha(root / 'scripts/run_formal_suite16_owner_v753_r10.py'),
            suite_lock_sha256=file_sha(root / prepare.SUITE_LOCK),
            split_source_lock_sha256=file_sha(root / prepare.SOURCE_INPUTS), author_outcome_protocol=protocol,
            source_manifest=suite['source_manifests']['validation'], admission_driver=suite['formal_case_driver'],
            admission_guard=suite['formal_admission_guard'], execution_protocol_changed=True,
            native_runtime=protocol['runtime_consumer'], original_native_runtime=protocol['original_runtime'],
            max_native_slots=2, max_GPU_slots=1, max_CPU_preflight_slots=2, automatic_retry=False)
        for name, value in [('launch', launch), ('plan', plan), ('status', status), ('terminal', terminal)]:
            write(prepare.PREVIOUS + '/' + name + '.json', value)
        stop = dict(created_utc='2026-10-20T10:33:05+00:00', user_message='先停下', user_stop_received=True)
        write(prepare.GLOBAL_STOP, stop); write(prepare.STOP_RECORD, stop)
        resume = dict(schema='r10_explicit_user_resume_v772', created_utc='2026-10-20T10:38:52+00:00',
            user_authorized_resume=True, revokes_previous_stop=True,
            previous_user_stop=prepare.pointer(root, prepare.STOP_RECORD),
            previous_global_stop_sha256=file_sha(root / prepare.GLOBAL_STOP),
            preserve_completed95=True, resume_remaining850_in_new_namespace=True,
            training_parameters_unchanged=True, TEST_requires_full_VAL_release=True)
        write(prepare.RESUME_RECORD, resume)
        proof = dict(policy_sha256=plan['policy_sha256'], routes={'common_author_RML_profile': {}})
        source_rows = {(row['task'], row['raw']): {} for row in plan['jobs']}
        return suite, proof, source_rows

    def test_fresh_control_archives_stop_preserves_complete_base_map_and_uses_json_key_digest(self):
        with TemporaryDirectory() as directory, TemporaryDirectory() as outside:
            root = Path(directory); external = Path(outside) / 'original_native_model'
            external.write_text('external original immutable input')
            suite, proof, source_rows = self.make_setup_fixture(root, external)
            before = {str(p): p.read_bytes() for p in root.rglob('*') if p.is_file()}
            def verify(root, files):
                for name, sha in files.items(): self.assertEqual(file_sha(Path(root) / name), sha)
            with patch.object(prepare, 'admit_validation', return_value=(suite, [], proof)), \
                 patch.object(prepare, 'require_previous_exited', return_value={'previous_owner_and_controller_exited': True}), \
                 patch.object(prepare, 'check_plan', return_value=({}, source_rows)), \
                 patch.object(prepare, 'fingerprint', return_value='original_profile'), \
                 patch.object(prepare, 'check_CPU', return_value={'sha256': 'identity'}) as cpu, \
                 patch.object(prepare, 'check_completed', return_value=False) as completed, \
                 patch.object(prepare, 'freeze_interrupted') as interrupted, \
                 patch.object(prepare, 'verify_inputs', side_effect=verify):
                result = prepare.prepare(root)
            self.assertEqual((cpu.call_count, completed.call_count, interrupted.call_count), (97, 95, 2))
            self.assertEqual((result['completed'], result['remaining']), (95, 850))
            setup = root / prepare.SETUP
            control = json.loads((setup / 'resume_control.json').read_text())
            locked = json.loads((setup / 'resume_inputs.json').read_text())
            self.assertEqual(control['schema'], prepare.CONTROL_SCHEMA)
            self.assertEqual(locked['schema'], prepare.LOCK_SCHEMA)
            self.assertIs(locked['approved_for_execution'], True)
            self.assertEqual(locked['files'][str(external)], file_sha(external))
            self.assertEqual(control['completed_key_sha256'], digest(sorted(control['completed_job_ids'])))
            self.assertEqual(control['remaining_key_sha256'], digest(sorted(control['remaining_job_ids'])))
            line_hash = hashlib.sha256(('\n'.join(control['remaining_job_ids']) + '\n').encode()).hexdigest()
            self.assertNotEqual(control['remaining_key_sha256'], line_hash)
            self.assertEqual((setup / 'previous_user_STOP.json').read_bytes(), (root / prepare.GLOBAL_STOP).read_bytes())
            self.assertNotIn(str(root / prepare.GLOBAL_STOP), locked['files'])
            self.assertIn(str(setup / 'previous_user_STOP.json'), locked['files'])
            self.assertEqual(locked['files'][str(setup / 'resume_control.json')], file_sha(setup / 'resume_control.json'))
            for name in prepare.REQUIRED_SOURCES:
                self.assertEqual(locked['files'][str(root / name)], file_sha(root / name))
            for path, original in before.items(): self.assertEqual(Path(path).read_bytes(), original)
            self.assertFalse(result['native_started']); self.assertFalse(result['GPU_started']); self.assertFalse(result['TEST_read'])
            with self.assertRaisesRegex(ValueError, 'one-time'):
                prepare.prepare(root)

    def test_resume_must_revoke_the_exact_preserved_stop_before_any_setup_is_created(self):
        with TemporaryDirectory() as directory, TemporaryDirectory() as outside:
            root = Path(directory); external = Path(outside) / 'native'; external.write_text('original')
            suite, proof, source_rows = self.make_setup_fixture(root, external)
            resume_path = root / prepare.RESUME_RECORD
            resume = json.loads(resume_path.read_text()); resume['previous_global_stop_sha256'] = '0' * 64
            resume_path.write_text(json.dumps(resume))
            with patch.object(prepare, 'admit_validation', return_value=(suite, [], proof)), \
                 patch.object(prepare, 'require_previous_exited', return_value={}), \
                 patch.object(prepare, 'check_plan', return_value=({}, source_rows)), \
                 patch.object(prepare, 'fingerprint', return_value='profile'), \
                 patch.object(prepare, 'check_CPU', return_value={}), \
                 patch.object(prepare, 'check_completed', return_value=False), \
                 patch.object(prepare, 'freeze_interrupted'):
                with self.assertRaisesRegex(ValueError, 'explicit user-authorized resume'):
                    prepare.prepare(root)
            self.assertFalse((root / prepare.SETUP).exists())


if __name__ == '__main__':
    unittest.main()
