"""Mocked CPU-only gates: no real TEST manifest, simulator or GPU is opened."""
import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import prepare_formal_TEST_after_environment_resume_v826_r10 as module
from scripts import prepare_formal_TEST_author_outcome_v755_r10 as original

SCRIPT = Path(module.__file__)
ROOT = SCRIPT.parents[1]


class ReachedFirstTESTRead(RuntimeError):
    pass


class Tests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.root = self.directory / 'project'
        self.root.mkdir()
        self.addCleanup(patch.stopall)
        patch.object(module, 'ROOT', self.root).start()
        self.events = []
        self.proof = dict(policy_sha256='a' * 64, routes={})
        self.owner = self.write('scripts/run_environment_resumed_formal_VAL_owner_v823_r10.py', b'owner823')
        self.auditor = self.write('scripts/audit_environment_resumed_formal_VAL_v824_r10.py', b'auditor824')
        self.controller = self.write('scripts/continue_environment_resumed_VAL_TEST_v827_r10.py', b'controller827')
        self.this_script = self.write('scripts/prepare_formal_TEST_after_environment_resume_v826_r10.py', b'prepare826')
        patch.object(module, '__file__', str(self.root / self.this_script['path'])).start()
        self.base_auditor = self.write('scripts/audit_formal_suite16_results_v754_r10.py', b'base754')
        self.suite = dict(formal_result_auditor=self.base_auditor,
            author_outcome_protocol={'schema': 'same741'},
            validation_release_path='artifacts/framework/multitask_r10/formal_suite16_v752/VAL_release.json')
        self.base = 'artifacts/framework/multitask_r10/formal_suite16_v752_setup/'
        self.suite_pointer = self.write(self.base + 'suite.json', self.suite)
        # Simulate an external author/model dependency retained by the base map.
        self.external = self.directory / 'author_model.bin'
        self.external.write_bytes(b'unchanged author weights')
        self.files = {str(self.root / self.base_auditor['path']): self.base_auditor['sha256'],
                      str(self.root / self.suite_pointer['path']): self.suite_pointer['sha256'],
                      str(self.external): self.sha(self.external)}
        self.suite_lock = self.write(self.base + 'frozen_inputs.json', dict(files=self.files))
        self.source_lock = self.write(self.base + 'validation_source_inputs.json', dict(split='validation', source_count=315))
        self.user_resume = self.write('runtime/framework/user_resume_v772.json', {'actual_old_user_resume': True})
        self.standing = self.write('runtime/framework/latest_executor_continuation_v817.json',
            dict(continue_until_user_stop=True, user_stop_received=False))
        self.environment_wrapper = self.write('scripts/run_frozen_system_environment_v820_r10.py', b'wrapper820')
        self.environment_manifest = self.write('snapshot/manifest.json', {'fake_CPU_snapshot': True})
        self.environment_proof = self.write('snapshot/cpu_namespace_proof_v820.json', {'fake_CPU_namespace_proof': True})
        self.previous_input = self.write('runtime/framework/formal_VAL_v753/imported_original.json', {'original': True})
        self.previous_control = self.write('old772/resume_control.json',
            dict(schema=module.environment_audit.legacy.CONTROL_SCHEMA, suite=self.suite_pointer,
                 suite_lock=self.suite_lock, source_inputs=self.source_lock, user_resume_record=self.user_resume,
                 author_outcome_protocol=self.suite['author_outcome_protocol']))
        old_files = dict(self.files)
        old_files[str(self.root / self.previous_input['path'])] = self.previous_input['sha256']
        self.previous_lock = self.write('old772/resume_inputs.json',
            dict(schema=module.environment_audit.legacy.LOCK_SCHEMA, approved_for_execution=True, files=old_files))
        self.actual_environment = dict(fake_CPU_namespace=True, manifest=self.environment_manifest,
            wrapper=self.environment_wrapper, qualification=self.environment_proof)
        self.control = dict(schema='r10_frozen_environment_formal_VAL_resume_control_v823',
            user_authorized_resume=True, split='validation', expected_cases=945,
            policy_sha256=self.proof['policy_sha256'], suite=self.suite_pointer,
            suite_lock=self.suite_lock, source_inputs=self.source_lock,
            resume_owner=self.owner, resume_auditor=self.auditor,
            author_outcome_protocol=self.suite['author_outcome_protocol'],
            standing_authorization_reused=True, new_user_event_invented=False, physical_policy_changed=False,
            user_resume_record=self.user_resume, standing_continuation_record=self.standing,
            previous_resume_control=self.previous_control, previous_resume_control_lock=self.previous_lock,
            environment_wrapper=self.environment_wrapper, environment_manifest=self.environment_manifest,
            environment_namespace_proof=self.environment_proof)
        self.control_pointer = self.write('artifacts/framework/multitask_r10/formal_VAL_env_resume_v823_setup/resume_control.json', self.control)
        self.lock_files = dict(self.files)
        for pointer in (self.owner, self.auditor, self.controller, self.this_script,
                        self.suite_lock, self.source_lock, self.control_pointer,
                        self.user_resume, self.standing, self.environment_wrapper, self.environment_manifest,
                        self.environment_proof, self.previous_input, self.previous_control, self.previous_lock):
            self.lock_files[str(self.root / pointer['path'])] = pointer['sha256']
        self.lock = dict(schema='r10_frozen_environment_formal_VAL_resume_inputs_v823',
                         approved_for_execution=True, files=self.lock_files)
        self.lock_pointer = self.write('artifacts/framework/multitask_r10/formal_VAL_env_resume_v823_setup/resume_inputs.json', self.lock)
        evidence = {name: value for name, value in self.lock_files.items() if name not in self.files}
        evidence[str(self.root / self.lock_pointer['path'])] = self.lock_pointer['sha256']
        self.release = dict(complete=True, method_cases=945, policy_sha256=self.proof['policy_sha256'],
            source_sha256=self.auditor['sha256'], base_result_auditor=self.base_auditor,
            resume_control=self.control_pointer, resume_control_lock=self.lock_pointer,
            inputs_sha256=evidence,
            environment_wrapper=self.environment_wrapper, environment_manifest=self.environment_manifest,
            environment_namespace_proof=self.environment_proof, actual_auditor_environment=self.actual_environment,
            cases=[dict(path=f'VAL/{index}/case.json', sha256=f'{index:064x}') for index in range(945)])
        self.release_pointer = self.write('artifacts/framework/multitask_r10/formal_VAL_env_resume_v823/VAL_release.json', self.release)
        self.case_shas = {str(self.root / pointer['path']): pointer['sha256'] for pointer in self.release['cases']}
        self.actual_read = Path.read_text

    @staticmethod
    def sha(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    def write(self, name, value):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value if isinstance(value, bytes) else json.dumps(value).encode())
        return dict(path=name, sha256=self.sha(path))

    def install_mock_gates(self, *, control=None, lock=None):
        def require_release(release, routes):
            self.events.append('complete_VAL_gate')
            if (release.get('complete') is not True or release.get('method_cases') != 945
                    or release.get('policy_sha256') != self.proof['policy_sha256']):
                raise ValueError('complete unchanged945 VAL required')
        def locked_pointer(root, files, pointer):
            if not isinstance(pointer, dict) or set(pointer) != {'path', 'sha256'}:
                raise ValueError('explicit pointer required')
            path = (root / pointer['path']).resolve()
            if files.get(str(path)) != pointer['sha256']:
                raise ValueError('pointer absent from frozen evidence')
            return path
        def locked_json(root, files, pointer):
            locked_pointer(root, files, pointer)
            if pointer == self.control_pointer:
                self.events.append('resume_control_checked')
                return deepcopy(self.control if control is None else control)
            if pointer == self.lock_pointer:
                self.events.append('resume_lock_checked')
                return deepcopy(self.lock if lock is None else lock)
            return json.loads((self.root / pointer['path']).read_text())
        def case_hash(path):
            name = str(Path(path))
            if name in self.case_shas:
                self.events.append('case_SHA_checked')
                return self.case_shas[name]
            return self.sha(path)
        def driver_evidence(suite, release, files):
            self.events.append('original755_full_driver_receipt_gate')
            return release['inputs_sha256']
        def guarded_read(path, *args, **kwargs):
            if path == self.root / 'artifacts/framework/protocol/test_manifest_all16.json':
                self.events.append('first_TEST_read_attempt')
                raise ReachedFirstTESTRead('mock stopped before any TEST bytes')
            return self.actual_read(path, *args, **kwargs)
        patch.object(module, 'require_test_release', side_effect=require_release).start()
        patch.object(module, 'locked_pointer', side_effect=locked_pointer).start()
        patch.object(module, 'locked_json', side_effect=locked_json).start()
        patch.object(module, 'verify_inputs', side_effect=lambda *args: self.events.append('complete_map_verified')).start()
        patch.object(module, 'file_sha', side_effect=case_hash).start()
        patch.object(module, 'require_driver_release_evidence', side_effect=driver_evidence).start()
        def environment_capture(control):
            self.events.append('actual_environment_capture')
            return deepcopy(self.actual_environment)
        def environment_receipt(receipt, control, *, expected):
            self.events.append('actual_auditor_environment_checked')
            if receipt != self.actual_environment or expected != self.actual_environment:
                raise ValueError('actual independently frozen environment required')
            return receipt
        patch.object(module.environment_audit, 'capture_environment', side_effect=environment_capture).start()
        patch.object(module.environment_audit, 'require_environment_receipt', side_effect=environment_receipt).start()

        self.seen_suite = []
        patch.object(module, 'admit', side_effect=lambda root, suite, files: self.seen_suite.append(suite) or self.proof).start()
        patch.object(Path, 'read_text', guarded_read).start()

    def run_until_gate(self, expected=ReachedFirstTESTRead):
        with patch('sys.argv', ['prepare826']), self.assertRaises(expected):
            module.main()

    def test_complete_source_resume_case_and_original755_gates_precede_TEST(self):
        self.install_mock_gates()
        self.run_until_gate()
        self.assertLess(self.events.index('complete_VAL_gate'), self.events.index('resume_control_checked'))
        self.assertLess(self.events.index('resume_lock_checked'), self.events.index('original755_full_driver_receipt_gate'))
        self.assertLess(self.events.index('original755_full_driver_receipt_gate'), self.events.index('first_TEST_read_attempt'))
        self.assertEqual(self.events.count('case_SHA_checked'), 945)
        self.assertLess(self.events.index('resume_control_checked'), self.events.index('actual_environment_capture'))
        self.assertLess(self.events.index('actual_auditor_environment_checked'), self.events.index('first_TEST_read_attempt'))
        extended = self.seen_suite[0]
        self.assertEqual(extended['formal_result_auditor'], self.base_auditor)
        self.assertEqual(extended['resumed_validation_auditor'], self.auditor)
        self.assertEqual(extended['validation_release_path'], self.release_pointer['path'])

    def test_incomplete_or_changed_policy_release_cannot_touch_TEST(self):
        self.release['complete'] = False
        self.write(self.release_pointer['path'], self.release)
        self.install_mock_gates()
        self.run_until_gate(ValueError)
        self.assertNotIn('resume_control_checked', self.events)
        self.assertNotIn('actual_environment_capture', self.events)
        self.assertNotIn('first_TEST_read_attempt', self.events)

    def test_actual824_source_cannot_be_relabelled_as754(self):
        self.release['source_sha256'] = self.base_auditor['sha256']
        self.write(self.release_pointer['path'], self.release)
        self.install_mock_gates()
        self.run_until_gate(ValueError)
        self.assertNotIn('first_TEST_read_attempt', self.events)

    def test_user_authorization_and_approved_control_lock_are_required(self):
        control = deepcopy(self.control)
        control['user_authorized_resume'] = False
        self.install_mock_gates(control=control)
        self.run_until_gate(ValueError)
        self.assertNotIn('first_TEST_read_attempt', self.events)

    def test_no_alias_for_missing_resume_control_lock_pointer(self):
        self.release['resume_lock'] = self.release.pop('resume_control_lock')
        self.write(self.release_pointer['path'], self.release)
        self.install_mock_gates()
        self.run_until_gate(ValueError)
        self.assertNotIn('first_TEST_read_attempt', self.events)

    def test_unapproved_resume_input_lock_is_rejected(self):
        lock = deepcopy(self.lock)
        lock['approved_for_execution'] = False
        self.install_mock_gates(lock=lock)
        self.run_until_gate(ValueError)
        self.assertNotIn('first_TEST_read_attempt', self.events)

    def test_complete_external_base_graph_is_preserved_without_release_expansion(self):
        self.assertNotIn(str(self.external), self.release['inputs_sha256'])
        self.install_mock_gates()
        self.run_until_gate()
        self.assertIn('original755_full_driver_receipt_gate', self.events)

    def test_missing_base_entry_or_new_control_evidence_is_rejected(self):
        broken = deepcopy(self.lock)
        broken['files'].pop(str(self.external))
        self.install_mock_gates(lock=broken)
        self.run_until_gate(ValueError)
        self.assertNotIn('first_TEST_read_attempt', self.events)

    def test_uncovered_new_control_input_is_rejected(self):
        pointer = self.write('new_uncovered_control.json', {'complete': True})
        broken = deepcopy(self.lock)
        broken['files'][str(self.root / pointer['path'])] = pointer['sha256']
        self.install_mock_gates(lock=broken)
        self.run_until_gate(ValueError)
        self.assertNotIn('first_TEST_read_attempt', self.events)


    def test_standing_authorization_and_unchanged_author_protocol_cannot_be_relabelled(self):
        for key, value in [('standing_authorization_reused', False), ('new_user_event_invented', True),
                           ('physical_policy_changed', True), ('author_outcome_protocol', {})]:
            bad = deepcopy(self.control); bad[key] = value
            self.install_mock_gates(control=bad)
            self.run_until_gate(ValueError)
            self.assertNotIn('first_TEST_read_attempt', self.events)
            patch.stopall(); patch.object(module, 'ROOT', self.root).start()
            patch.object(module, '__file__', str(self.root / self.this_script['path'])).start()
            self.events.clear()

    def test_release_must_bind_exact_snapshot_wrapper_and_namespace_proof(self):
        self.release['environment_manifest'] = dict(path='different_snapshot.json', sha256='0' * 64)
        self.write(self.release_pointer['path'], self.release)
        self.install_mock_gates(); self.run_until_gate(ValueError)
        self.assertNotIn('first_TEST_read_attempt', self.events)

    def test_actual_auditor_environment_receipt_required_before_TEST(self):
        self.release['actual_auditor_environment'] = {'storage_only': True}
        self.write(self.release_pointer['path'], self.release)
        self.install_mock_gates(); self.run_until_gate(ValueError)
        self.assertNotIn('first_TEST_read_attempt', self.events)

    def test_nested772_original_project_input_must_be_retained(self):
        broken = deepcopy(self.lock)
        broken['files'].pop(str(self.root / self.previous_input['path']))
        self.install_mock_gates(lock=broken); self.run_until_gate(ValueError)
        self.assertNotIn('first_TEST_read_attempt', self.events)

    def test_live_user_STOP_prevents_first_TEST_source_selection(self):
        self.write('runtime/framework/multitask_r10_STOP', b'new user STOP')
        self.install_mock_gates(); self.run_until_gate(RuntimeError)
        self.assertNotIn('first_TEST_read_attempt', self.events)

    def test_TEST_reset_alias_source_body_remains_identical_to755(self):
        def source_body(path):
            main = next(node for node in ast.parse(path.read_text()).body
                        if isinstance(node, ast.FunctionDef) and node.name == 'main')
            start = next(index for index, node in enumerate(main.body) if isinstance(node, ast.Assign)
                         and any(isinstance(target, ast.Name) and target.id == 'original' for target in node.targets))
            end = next(index for index, node in enumerate(main.body[start:], start)
                       if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                       and isinstance(node.value.func, ast.Name) and node.value.func.id == 'validate_roster')
            return ast.dump(ast.Module(body=main.body[start:end + 1], type_ignores=[]))
        self.assertEqual(source_body(SCRIPT), source_body(Path(original.__file__)))
        def source_text(path):
            text = path.read_text()
            start = text.index("    original=ROOT/'artifacts/framework/protocol/test_manifest_all16.json'")
            end = text.index("    setup.mkdir();manifest_path=setup/'official_test_manifest16.json'", start)
            return text[start:end]
        self.assertEqual(source_text(SCRIPT), source_text(Path(original.__file__)))
        self.assertIs(module.require_driver_release_evidence, original.require_driver_release_evidence)


if __name__ == '__main__':
    unittest.main()
