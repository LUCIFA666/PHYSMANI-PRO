import json
from pathlib import Path
import tempfile
import unittest

from our_mani.hierarchy_20261030.freeze_builder import (
    BASE, EvaluationBlocked, _read_complete, _read_segmentation, _write_new, build_freeze,
)


class FreezeBuilderTests(unittest.TestCase):
    def test_incomplete_actual_fitting_evidence_cannot_create_freeze(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            data = root/BASE/"data_v1/manifest.json"
            skill = root/BASE/"skill_handoff_pretrain_v2/status.json"
            candidate = root/BASE/"candidate_pretrain_v1/status.json"
            for path in (data, skill, candidate):
                path.parent.mkdir(parents=True)
                path.write_text(json.dumps({"complete": path != candidate}))
            with self.assertRaises(EvaluationBlocked):
                build_freeze(root)
            self.assertFalse((root/BASE/"freeze_model_v1").exists())

    def test_readiness_requires_actual_complete_boolean(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = root/"state.json"
            for value in (False, "true", 1, None):
                path.write_text(json.dumps({"complete": value}))
                with self.assertRaises(EvaluationBlocked):
                    _read_complete(root, Path("state.json"))
            path.write_text('{"complete":true}')
            self.assertTrue(_read_complete(root, Path("state.json"))["complete"])

    def test_existing_evidence_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"evidence.json"
            _write_new(path, {"source": "original"})
            with self.assertRaises(FileExistsError):
                _write_new(path, {"source": "replacement"})
            self.assertEqual(json.loads(path.read_text())["source"], "original")

    def test_real_segmentation_completion_schema_and_bound_bytes(self):
        import hashlib
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root/"model.pt").write_bytes(b"model fixture")
            model_sha = hashlib.sha256(b"model fixture").hexdigest()
            document = dict(schema="hierarchy_public_rgbd_semantic_mlp_v1_20261030",
                            training_complete=True, stage="complete", TEST_read=False,
                            original_VAL_read=False, native_or_instance_inputs_used=False,
                            model={"sha256": model_sha}, code_sha256={})
            (root/"seg.json").write_text(json.dumps(document))
            self.assertTrue(_read_segmentation(root, Path("seg.json"), Path("model.pt"))["training_complete"])
            (root/"model.pt").write_bytes(b"changed weights")
            with self.assertRaises(EvaluationBlocked):
                _read_segmentation(root, Path("seg.json"), Path("model.pt"))


if __name__ == "__main__":
    unittest.main()
