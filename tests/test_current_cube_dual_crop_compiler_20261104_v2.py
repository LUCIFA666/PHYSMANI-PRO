"""Compiler gating/publication contracts; no real capture or model reads."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('dual_cf_compiler', ROOT/'scripts/prepare_current_cube_dual_crop_TRAIN_manifest_20261104_v2.py')
compiler = importlib.util.module_from_spec(spec); spec.loader.exec_module(compiler)


class StubRoleDataset:
    roles = []
    getters = []
    fail = False

    def __init__(self, _path, role='fit', **_kwargs):
        self.roles.append(role)
        if role not in ('fit', 'selection'):
            raise AssertionError('Reserved labels must never decode')
        self.role = role
        self.records = [dict(role=role)]*208

    def __len__(self): return 208

    def __getitem__(self, index):
        self.getters.append((self.role, index))
        if self.fail: raise ValueError('synthetic invalid label')
        return dict(role=self.role, targets=dict(presence_target=-1, centre_valid=False,
            reason='unverified_absence_UNKNOWN'))

    def revalidate(self): return dict(synthetic_source_digest='same')


class DualCFCompilerContracts(unittest.TestCase):
    def setUp(self):
        StubRoleDataset.roles = []; StubRoleDataset.getters = []; StubRoleDataset.fail = False

    def test_missing_cohort_refuses_before_output_or_any_label_getter(self):
        with tempfile.TemporaryDirectory(prefix='cfcompileunit_', dir=ROOT/'artifacts') as tmp:
            output = Path(tmp)/'aggregate'
            with (patch.object(compiler, 'audit_all_closed', side_effect=ValueError('39 of40')),
                 patch.object(compiler, 'candidate_manifest', side_effect=AssertionError('No label pointers')),
                 patch.object(compiler.data, 'DualCropCubeDataset', side_effect=AssertionError('No NPZ'))):
                with self.assertRaises(ValueError): compiler.build_manifest('plan', Path(tmp)/'captures', output)
            self.assertFalse(output.exists())

    def run_compiler(self, base):
        plan = base/'plan.json'; plan.write_text('{}')
        capture = base/'captures'; capture.mkdir()
        output = base/'aggregate'
        candidate = dict(capture_plan=compiler.pointer(plan), schema='synthetic_only')
        with (patch.object(compiler, 'audit_all_closed', return_value=(plan, {}, capture, [{'parent_id':str(i)} for i in range(40)])),
             patch.object(compiler, 'candidate_manifest', return_value=candidate),
             patch.object(compiler.data, 'DualCropCubeDataset', StubRoleDataset)):
            receipt = compiler.build_manifest(plan, capture, output)
        return receipt, output

    def test_only416_training_role_getters_and_atomic_fresh_publication(self):
        with tempfile.TemporaryDirectory(prefix='cfcompileunit_', dir=ROOT/'artifacts') as tmp:
            receipt, output = self.run_compiler(Path(tmp))
            self.assertEqual(StubRoleDataset.roles, ['fit', 'selection'])
            self.assertEqual(len(StubRoleDataset.getters), 416)
            self.assertEqual(receipt['decoded_reserved_samples'], 0)
            self.assertEqual(receipt['reserved_metadata_and_hash_only'], 104)
            self.assertEqual((output/'manifest.json').read_bytes(), (output/'candidate_manifest.json').read_bytes())
            self.assertTrue((output/'receipt.json').exists())
            with self.assertRaises(ValueError): compiler.build_manifest('plan', 'captures', output)

    def test_bad_training_label_never_leaves_final_manifest(self):
        with tempfile.TemporaryDirectory(prefix='cfcompileunit_', dir=ROOT/'artifacts') as tmp:
            StubRoleDataset.fail = True
            with self.assertRaises(ValueError): self.run_compiler(Path(tmp))
            output = Path(tmp)/'aggregate'
            self.assertTrue((output/'candidate_manifest.json').exists())
            self.assertTrue((output/'failure.json').exists())
            self.assertFalse((output/'manifest.json').exists())
            self.assertFalse((output/'receipt.json').exists())


if __name__ == '__main__': unittest.main()
