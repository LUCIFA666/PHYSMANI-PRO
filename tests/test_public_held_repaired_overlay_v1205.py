"""Synthetic source/CPU-evidence contracts; no simulator, model or case run."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from our_mani.multitask import public_held_repaired_overlay_v1205 as subject

ROOT = Path(__file__).resolve().parents[1]


def put(root, name, data):
    path = root/name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return dict(path=name, sha256=hashlib.sha256(data).hexdigest())


def write(root, name, value):
    return put(root, name, json.dumps(value, sort_keys=True).encode())


class PhysicalOverlayContracts(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for name in subject.OVERLAY_SOURCES:
            data = ((ROOT/name).read_bytes() if name in (subject.BLOCK_SOURCE, subject.BLOCK_TEST)
                    else ('CPU fixture source:' + name).encode())
            put(self.root, name, data)
        self.inherited = dict(integration_complete=True, files={name: subject.pointer(self.root, name)['sha256']
            for name in subject.parent.OVERLAY_SOURCES})
        self.parent_ref = dict(path=subject.PARENT_OVERLAY_LOCK, sha256=subject.PARENT_OVERLAY_LOCK_SHA)
        parent = patch.object(subject, '_parent_lock', return_value=(self.inherited, self.parent_ref))
        parent.start()
        self.addCleanup(parent.stop)
        self.files = subject.collect_source_hashes(self.root)
        self.evidence = dict(schema=subject.CPU_EVIDENCE_SCHEMA, complete=True,
            actual_tool_exit_observed=True, returncode=0, tests_exit_code=0, tests_run=22, skipped=0,
            native_started=False, GPU_started=False, source_hashes=deepcopy(self.files),
            source_hashes_before=deepcopy(self.files), source_hashes_after=deepcopy(self.files),
            parent_source_overlay_lock=self.parent_ref)
        self.name = subject.SETUP_PREFIX + 'source_overlay_lock.json'

    def cpu(self, value=None, suffix=''):
        return write(self.root, subject.SETUP_PREFIX + suffix + 'physical_CPU_evidence.json',
            self.evidence if value is None else value)

    def test_complete_independent_closure_and_only_block_factory_changes(self):
        self.assertTrue(set(subject.parent.OVERLAY_SOURCES) <= set(subject.OVERLAY_SOURCES))
        self.assertTrue(set(subject.block.SOURCES) <= set(subject.OVERLAY_SOURCES))
        self.assertIn(subject.TEST_SOURCE, subject.OVERLAY_SOURCES)
        self.assertEqual(len(subject.OVERLAY_SOURCES), len(set(subject.OVERLAY_SOURCES)))
        self.assertEqual(subject.CORE_INTEGRATION_SOURCE, subject.parent.CORE_INTEGRATION_SOURCE)
        self.assertEqual(subject.CHECKPOINT, subject.parent.CHECKPOINT)
        for field in ('ROUTES', 'ROUTES_SHA', 'MANIFEST', 'MANIFEST_SHA', 'TRAIN_INPUTS', 'TRAIN_INPUTS_SHA'):
            self.assertEqual(getattr(subject, field), getattr(subject.parent, field))
        for name in subject.OVERLAY_SOURCES:
            self.assertNotIn('v1206', name)
            self.assertNotIn('v1207', name)
            self.assertNotIn('v1212', name)
            self.assertNotIn('physical_CPU_evidence', name)
            self.assertNotIn('policy_graph.json', name)

    def test_actual_evidence_precedes_first_write_and_reader_binds_pointer(self):
        evidence = self.cpu()
        reference = subject.write_overlay_lock(self.root, self.name, integration_cpu_evidence=evidence)
        value, actual = subject.read_overlay_lock(self.root, dict(root='project', **reference))
        self.assertEqual(actual, reference)
        self.assertEqual(value['files'], self.files)
        self.assertEqual(value['core_factory']['source'], subject.parent.CORE_INTEGRATION_SOURCE)
        self.assertEqual(value['block_factory']['source'], subject.BLOCK_SOURCE)
        self.assertFalse(value['formal_VAL_TEST_qualification'])
        self.assertFalse(value['borrowed_frozen_physical_graph_identity'])
        with self.assertRaises(FileExistsError):
            subject.write_overlay_lock(self.root, self.name, integration_cpu_evidence=evidence)
        with self.assertRaises(ValueError):
            subject.read_overlay_lock(self.root, dict(reference, sha256='0'*64))

    def test_no_pending_or_fabricated_CPU_evidence_can_activate(self):
        for field, value in (('complete', False), ('actual_tool_exit_observed', False), ('returncode', 1),
                ('returncode', False), ('tests_exit_code', False), ('tests_run', 0), ('skipped', 1),
                ('native_started', True), ('GPU_started', True), ('source_hashes', {}),
                ('source_hashes_before', {}), ('source_hashes_after', {}), ('parent_source_overlay_lock', {})):
            changed = deepcopy(self.evidence)
            changed[field] = value
            reference = self.cpu(changed)
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                subject.write_overlay_lock(self.root, self.name, integration_cpu_evidence=reference)
            self.assertFalse((self.root/self.name).exists())
        with self.assertRaises(ValueError):
            subject.write_overlay_lock(self.root, self.name, integration_cpu_evidence=None)

    def test_CPU_evidence_namespace_and_basename_are_typed(self):
        for name in ('metadata/physical_CPU_evidence.json', subject.SETUP_PREFIX+'untyped.json'):
            evidence = write(self.root, name, self.evidence)
            with self.subTest(name=name), self.assertRaises(ValueError):
                subject.write_overlay_lock(self.root, self.name, integration_cpu_evidence=evidence)

    def test_frozen_parent_and1203_changes_fail_closed(self):
        for name in (subject.parent.CORE_INTEGRATION_SOURCE, subject.BLOCK_SOURCE, subject.BLOCK_TEST):
            path = self.root/name
            before = path.read_bytes()
            path.write_bytes(before + b'changed')
            with self.subTest(name=name), self.assertRaises(ValueError):
                subject.collect_source_hashes(self.root)
            path.write_bytes(before)

    def test_lock_metadata_cannot_borrow_old_identity_or_change_factory(self):
        reference = subject.write_overlay_lock(self.root, self.name, integration_cpu_evidence=self.cpu())
        original = json.loads((self.root/reference['path']).read_text())
        for field, value in (('block_factory', original['core_factory']), ('required_sources', []),
                ('parent_source_overlay_lock', {}), ('integration_complete', False),
                ('borrowed_frozen_physical_graph_identity', True), ('formal_VAL_TEST_qualification', True)):
            changed = deepcopy(original)
            changed[field] = value
            write(self.root, self.name, changed)
            with self.subTest(field=field), self.assertRaises(ValueError):
                subject.read_overlay_lock(self.root, self.name)

    def test_nonblock_binding_is_exact_parent_delegate(self):
        sentinel = object()
        with patch.object(subject.parent, 'phase_component_binding', return_value=sentinel) as inherited:
            for task, method in ((subject.BLOCKS[0], 'physmani'), ('beat_the_rotating_buzz', 'context_geometry')):
                self.assertIs(subject.phase_component_binding(self.root, {}, task, method), sentinel)
                inherited.assert_called_with(self.root, {}, task, method)
        module = type('Factory', (), {'binding': staticmethod(lambda root, native: {'new1203': True})})
        with patch.object(subject.importlib, 'import_module', return_value=module) as imported:
            self.assertEqual(subject.phase_component_binding(self.root, {'baseline_repo':'never-started'},
                subject.BLOCKS[1], 'cv_geometry'), {'block_integration': {'new1203': True}})
            imported.assert_called_once_with(subject.BLOCK_MODULE)


if __name__ == '__main__':
    unittest.main()
