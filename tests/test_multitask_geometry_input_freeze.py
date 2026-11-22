"""Enabled TRAIN geometry proofs must enter the actual collector input lock."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts.collect_multitask_training_r10 import optional_geometry_input_files, CollectionError, ROOT


class GeometryInputFreeze(unittest.TestCase):
    def test_nested_text_proof_is_bound_without_treating_report_description_as_pointer(self):
        base = ROOT/'runtime/framework/tmp'
        base.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=base) as directory:
            directory = Path(directory)
            def pointer(path):
                return dict(path=str(path.relative_to(ROOT)), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            proof = directory/'proof.txt'; proof.write_text('immutable inclusion argument')
            source = directory/'source.py'; source.write_text('fixed construction')
            report = directory/'report.json'
            report.write_text(json.dumps(dict(analytic_proof=str(proof.relative_to(ROOT)),
                inputs_sha256={str(p.relative_to(ROOT)):pointer(p)['sha256'] for p in (proof, source)})))
            asset = directory/'asset.json'
            asset.write_text(json.dumps(dict(source_certificate=pointer(report),
                engineering_inflation_proof=pointer(proof))))
            protocol = dict(axis_swept_assets={'put_rubbish_in_moving_bin':{'bin':pointer(asset)}})
            self.assertEqual(set(optional_geometry_input_files(protocol)), {proof, source, report, asset})
            source.write_text('changed construction')
            with self.assertRaisesRegex(CollectionError, 'dependency changed'):
                optional_geometry_input_files(protocol)

    def test_permission_hash_and_task_are_checked_before_execution(self):
        with self.assertRaises(CollectionError):
            optional_geometry_input_files(dict(axis_swept_assets={'unknown_task':{}}))
        with self.assertRaises(CollectionError):
            optional_geometry_input_files(dict(finite_rotation_symmetries={
                'insert_onto_rotating_peg':{'ring':dict(path='missing.json',sha256='0'*64)}}))

    def test_disabled_optional_geometry_has_no_invented_dependencies(self):
        self.assertEqual(optional_geometry_input_files({}), ())


if __name__ == '__main__':
    unittest.main()
