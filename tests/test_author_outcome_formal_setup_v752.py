"""Pure CPU checks for explicit new-protocol setup and pre-TEST evidence."""
import ast
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import prepare_formal_VAL_author_outcome_v752_r10 as val
from scripts import prepare_formal_TEST_author_outcome_v755_r10 as test_setup
from scripts import check_formal_admission_v757_r10 as cpu
from our_mani.multitask.evaluation import file_sha

ROOT = Path(__file__).resolve().parents[1]


class Tests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.addCleanup(patch.stopall)
        patch.object(val, 'ROOT', self.root).start()
        patch.object(test_setup, 'ROOT', self.root).start()
        self.worker = self.write('scripts/run_task_executor_smoke_author_outcome_v743_r10.py', b'# worker743')
        self.protocol = dict(schema='r10_author_action_outcome_protocol_v741',
            implementation=self.write('implementation741.py', b'# action sampling'),
            runtime_consumer=self.write('runtime744.py', b'# explicit native runtime'),
            original_runtime=self.write('runtime664.py', b'# original native runtime'),
            failure_sampling='current_native_result_at_explicit_action_end')

    def write(self, name, value):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value if isinstance(value, bytes) else json.dumps(value).encode())
        return dict(path=name, sha256=file_sha(path))

    def qualification(self):
        return dict(author_outcome_protocol=self.protocol, routes=[dict(
            task=t, method=m, worker=self.worker, consumer=self.protocol['runtime_consumer'],
            author_outcome_protocol=self.protocol) for t in val.TASKS for m in val.METHODS])

    def test_new_qualification_binds_every743744_route_and_same_protocol(self):
        routes = self.qualification()
        readiness = dict(author_outcome_protocol=self.protocol)
        val.require_author_qualification(routes, readiness, self.protocol)
        for key, value in (('worker', {'path': 'old709', 'sha256': 'a' * 64}),
                           ('consumer', self.protocol['original_runtime']),
                           ('author_outcome_protocol', {})):
            changed = deepcopy(routes)
            changed['routes'][-1][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                val.require_author_qualification(changed, readiness, self.protocol)
        with self.assertRaises(ValueError):
            val.require_author_qualification(routes, {}, self.protocol)

    def CPU_fixture(self):
        row = dict(task='place_cups_on_rotating_frame', raw='episodeA', method='context_geometry',
            source_row_sha256='a' * 64, configuration_sha256='b' * 64, input_lock_sha256='c' * 64,
            admission_driver={'path': 'driver751', 'sha256': 'd' * 64},
            admission_guard={'path': 'guard728', 'sha256': 'e' * 64}, case_worker=self.worker,
            native_runtime=self.protocol['runtime_consumer'],
            original_native_runtime=self.protocol['original_runtime'], author_outcome_protocol=self.protocol)
        source_sha = 'f' * 64
        receipt = {key: row[key] for key in ('admission_driver', 'admission_guard', 'case_worker',
                   'native_runtime', 'original_native_runtime', 'author_outcome_protocol')}
        receipt.update(complete=True, execution_protocol_changed=True, native_started=False,
            GPU_started=False, actual_split='validation',
            identity=dict(split='validation', method=row['method'], task=row['task'],
                source_raw_episode_id=row['raw'], source_row_sha256=row['source_row_sha256'],
                config_sha256=row['configuration_sha256'], frozen_inputs_sha256=row['input_lock_sha256'],
                source_input_lock_sha256=source_sha, demo_only=False),
            component_domain=dict(purpose='evaluation', split='validation', task=row['task'],
                                  method=row['method'], raw_episode_id=row['raw']))
        return row, receipt, source_sha

    def test_actual_CPU_receipt_requires_new_native_and_protocol_bindings(self):
        row, receipt, sha = self.CPU_fixture()
        cpu.require_receipt(receipt, row, sha)
        for key in ('case_worker', 'native_runtime', 'original_native_runtime', 'author_outcome_protocol'):
            broken = deepcopy(receipt)
            broken.pop(key)
            if key == 'native_runtime':
                broken['unchanged_native_runtime'] = row['native_runtime']
            with self.subTest(key=key), self.assertRaises(ValueError):
                cpu.require_receipt(broken, row, sha)
        receipt['identity']['split'] = 'train'
        with self.assertRaises(ValueError):
            cpu.require_receipt(receipt, row, sha)

    def evidence_fixture(self, *, count=945, legacy=False):
        suite = dict(author_outcome_protocol=self.protocol,
            formal_case_driver=dict(path='driver751', sha256='a' * 64),
            formal_admission_guard=dict(path='guard728', sha256='b' * 64))
        cases, pointers, inputs = [], [], {}
        for index in range(count):
            case = dict(path=f'VAL/{index}/case.json', sha256=f'{index:064x}')
            cases.append(case)
            value = dict(schema='r10_guard_only_formal_driver_v729' if legacy else
                'r10_explicit_author_outcome_formal_driver_v751', complete=True, actual_split='validation',
                execution_protocol_changed=True, admission_driver=suite['formal_case_driver'],
                admission_guard=suite['formal_admission_guard'], case_worker=self.worker,
                native_runtime=self.protocol['runtime_consumer'],
                original_native_runtime=self.protocol['original_runtime'],
                author_outcome_protocol=self.protocol, case=case)
            pointer = self.write(f'receipts/{index}.json', value)
            pointers.append(pointer)
            inputs[str(self.root / pointer['path'])] = pointer['sha256']
        return suite, dict(author_outcome_protocol=self.protocol, cases=cases,
                           admission_driver_receipts=pointers, inputs_sha256=inputs)

    def test_all945_new_receipts_required_and_old20_cannot_release_TEST(self):
        suite, release = self.evidence_fixture()
        self.assertEqual(test_setup.require_driver_release_evidence(suite, release, {}), release['inputs_sha256'])
        release['admission_driver_receipts'] = release['admission_driver_receipts'][:20]
        release['cases'] = release['cases'][:20]
        with self.assertRaisesRegex(ValueError, 'all945'):
            test_setup.require_driver_release_evidence(suite, release, {})

    def test_old_driver_schema_and_changed_protocol_are_rejected(self):
        suite, release = self.evidence_fixture(legacy=True)
        with self.assertRaisesRegex(ValueError, 'author-outcome protocol receipts'):
            test_setup.require_driver_release_evidence(suite, release, {})
        release['author_outcome_protocol'] = {}
        with self.assertRaisesRegex(ValueError, 'same new author-outcome protocol'):
            test_setup.require_driver_release_evidence(suite, release, {})

    def test_frozen_release_input_conflict_rejected_before_TEST_selection(self):
        suite, release = self.evidence_fixture(count=1)
        name = next(iter(release['inputs_sha256']))
        with self.assertRaisesRegex(ValueError, 'conflicts'):
            test_setup.require_driver_release_evidence(suite, release, {name: 'f' * 64})

    def test_complete_VAL_gate_and_receipt_audit_precede_TEST_manifest_read(self):
        tree = ast.parse(Path(test_setup.__file__).read_text())
        main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'main')
        first_test = next(node.lineno for node in main.body if isinstance(node, ast.Assign)
                          and any(isinstance(target, ast.Name) and target.id == 'original' for target in node.targets))
        calls = {name: min(node.lineno for node in ast.walk(main) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Name) and node.func.id == name)
                 for name in ('require_test_release', 'require_driver_release_evidence')}
        self.assertLess(calls['require_test_release'], calls['require_driver_release_evidence'])
        self.assertLess(calls['require_driver_release_evidence'], first_test)

    def test_source_lock_semantics_identical_and_CPU_selection_retained(self):
        original = ROOT / 'scripts/prepare_formal_split_source_lock_v734_r10.py'
        current = ROOT / 'scripts/prepare_formal_split_source_lock_v756_r10.py'
        self.assertEqual(ast.dump(ast.parse(original.read_text())), ast.dump(ast.parse(current.read_text())))
        self.assertEqual(cpu.NATIVE_OUT, Path('artifacts/framework/multitask_r10/formal_suite16_v752/validation'))
        self.assertEqual(len(cpu.CASES), 5)
        self.assertEqual(cpu.CASES[0], ('beat_the_rotating_buzz', 'cv_geometry'))
        self.assertNotIn('unchanged_case_worker=', Path(cpu.__file__).read_text())


if __name__ == '__main__':
    unittest.main()
