"""Synthetic CPU freeze contracts, without GPU/Native/real TEST artifacts."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from our_mani.hierarchy_block_20261102 import freeze_builder as f


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def prepare(root):
    names = {*f.SOURCE_FILES, *f.OLD_RUNTIME_FILES,
             "our_mani/hierarchy_block_20261102/actor.py", "our_mani/hierarchy_block_20261102/freeze_builder.py",
             "our_mani/hierarchy_block_20261102/training.py", "scripts/train_hierarchy_block_20261102.py"}
    for name in names:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# synthetic source identity only\n")
    forecaster = root / f.FORECAST
    forecaster.parent.mkdir(parents=True, exist_ok=True)
    forecaster.write_bytes(b"synthetic retained GRU artifact")
    segment = root / f.SEGMENTATION
    segment.parent.mkdir(parents=True, exist_ok=True)
    segment.write_bytes(b"synthetic segmentation artifact")
    segment_sha, forecast_sha = f.file_sha256(segment), f.file_sha256(forecaster)
    rows, cases, files = [], [], {}
    for split in ("train", "validation"):
        for task in f.TASK_NAMES:
            directory = f.LEGACY / "synthetic" / split / task
            scene, motion, audit = directory / "scene.npz", directory / "motion.npz", directory / "audit.json"
            for path in (scene, motion, audit):
                target = root / path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(b"hash-bound synthetic content")
                files[str(path)] = f.file_sha256(target)
            row = dict(split=split, task=task, raw_episode_id="episode1", parent_source_group=f"{split}:{task}:episode1",
                       source_path=f"/unopened/{split}/{task}/data.h5", npz_path=str(scene), cache_sha256=files[str(scene)])
            rows.append(row)
            cases.append(dict(**{k: row[k] for k in ("split", "task", "raw_episode_id", "parent_source_group", "source_path")},
                              npz_path=str(motion), sha256=files[str(motion)], audit_path=str(audit), audit_sha256=files[str(audit)]))
    write_json(root / f.DATA, dict(complete=True, smoke_only=False, TEST_read=False,
                                  source_debug_target_positions_read=False, episodes=rows))
    write_json(root / f.MOTION, dict(complete=True, smoke_only=False, TEST_read=False,
        source_debug_target_positions_read=False, frontend="blind_parallel_fixed_common_TRAIN_priors", source_task_selects_frontend=False,
        identity=dict(data_manifest_sha256=f.file_sha256(root / f.DATA), motion_checkpoint_sha256=forecast_sha,
                      segmentation_checkpoint_sha256=segment_sha, code_sha256={}), cases=cases))
    write_json(root / f.SEGMENTATION_MANIFEST, dict(schema="hierarchy_public_rgbd_semantic_mlp_v1_20261030",
        training_complete=True, stage="complete", TEST_read=False, original_VAL_read=False, native_or_instance_inputs_used=False,
        model=dict(sha256=segment_sha), code_sha256={}))
    files.update({str(p): f.file_sha256(root / p) for p in (f.DATA, f.MOTION)})
    identity = dict(code_sha256={"our_mani/hierarchy_block_20261102/training.py": f.file_sha256(root / "our_mani/hierarchy_block_20261102/training.py")},
        data=dict(version=f.DATA_VERSION, files=files, TEST_read=False, other_task_arrays_read=False, success_filter=False,
                  loader_source_sha256=f.file_sha256(root / "our_mani/hierarchy_block_20261102/data.py"),
                  selected_sources=[{k: row[k] for k in ("split", "task", "raw_episode_id", "parent_source_group")} for row in rows]))
    training = root / f.BASE / "training_v1"
    training.mkdir(parents=True)
    metadata = dict(identity=identity, workspace_center=[0, 0, .8], workspace_half_width=[1, 1, 1], model_config=f.MODEL_CONFIG,
                    training_complete=True, completed_epochs=120, TEST_read=False, score_semantics=f.SCORE_SEMANTICS, precision_qualified=False)
    torch.save(dict(version=f.TRAINING_VERSION, model_state_dict={"synthetic_only": torch.zeros(1)}, metadata=metadata), training / "best.pt")
    write_json(training / "status.json", dict(version=f.TRAINING_VERSION, complete=True, completed_epochs=120,
        TEST_read=False, precision_qualified=False, identity=identity, best_checkpoint_sha256=f.file_sha256(training / "best.pt")))
    return forecast_sha


class TestBlockFreezeBuilder(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        forecast = prepare(self.root)
        self.forecast_patch = patch.object(f, "FORECAST_SHA", forecast)
        self.forecast_patch.start(); self.addCleanup(self.forecast_patch.stop)
        self.count_patch = patch.dict(f.EXPECTED_COUNTS, {(s, t): 1 for s in ("train", "validation") for t in f.TASK_NAMES}, clear=True)
        self.count_patch.start(); self.addCleanup(self.count_patch.stop)

    def build(self, **kwargs):
        with patch.object(f, "_construct_cpu_actor", return_value=dict(complete=True, device="cpu", actor_constructed=True,
                native_started=False, GPU_started=False, TEST_read=False)) as ready:
            result = f.build_freeze(self.root, **kwargs)
        return result, ready

    def test_complete_unqualified_training_can_freeze_with_cpu_readiness(self):
        path, ready = self.build()
        freeze = json.loads(path.read_text())
        self.assertFalse(freeze["precision_qualified"])
        self.assertTrue(freeze["evaluation_of_unqualified_model_allowed"])
        self.assertEqual(len({freeze["components"][n]["artifact"]["sha256"] for n in f.NETWORKS}), 1)
        ready.assert_called_once()
        self.assertEqual(freeze["protocol"]["inference_device"], "cpu")
        self.assertEqual(freeze["tasks"], list(f.TASK_NAMES))
        self.assertIn("cpu_actor_readiness", freeze)
        evidence = json.loads((path.parent / "training_evidence.json").read_text())
        self.assertFalse(evidence["native_success_certified"])
        self.assertEqual(evidence["source_count"], 4)
        self.assertEqual(evidence["completed_epochs"], 120)
        runtime = {p["path"] for p in freeze["runtime_sources"]}
        self.assertIn("our_mani/hierarchy_block_20261102/actor.py", runtime)
        self.assertIn("our_mani/hierarchy_20261030/tracking.py", runtime)
        self.assertIn("scripts/train_hierarchy_block_20261102.py", runtime)
        f.validate_freeze(self.root, freeze)
        with self.assertRaises(FileExistsError):
            f.build_freeze(self.root)

    def test_incomplete_training_rejected_before_cpu_readiness_or_output(self):
        path = self.root / f.BASE / "training_v1/status.json"
        status = json.loads(path.read_text()); status["complete"] = False; write_json(path, status)
        with patch.object(f, "_construct_cpu_actor") as ready:
            with self.assertRaisesRegex(f.EvaluationBlocked, "completed 120"):
                f.build_freeze(self.root)
            ready.assert_not_called()
        self.assertFalse((self.root / f.BASE / "freeze_model_v1").exists())

    def test_fitting_source_changed_rejected(self):
        (self.root / "our_mani/hierarchy_block_20261102/training.py").write_text("changed")
        with self.assertRaisesRegex(f.EvaluationBlocked, "SHA256 changed"):
            self.build()

    def test_cpu_construction_failure_has_no_usable_freeze(self):
        with patch.object(f, "_construct_cpu_actor", side_effect=RuntimeError("missing real weights")):
            with self.assertRaisesRegex(RuntimeError, "missing real weights"):
                f.build_freeze(self.root)
        directory = self.root / f.BASE / "freeze_model_v1"
        self.assertFalse((directory / "freeze.json").exists())
        failure = json.loads((directory / "cpu_readiness_failure.json").read_text())
        self.assertFalse(failure["ready"])
        self.assertFalse(failure["native_started"])

    def test_runtime_mutation_and_checkpoint_mutation_rejected(self):
        path, _ = self.build()
        freeze = json.loads(path.read_text())
        (self.root / "our_mani/hierarchy_block_20261102/actor.py").write_text("changed runtime")
        with self.assertRaisesRegex(f.EvaluationBlocked, "changed"):
            f.validate_freeze(self.root, freeze)
        with (self.root / f.BASE / "training_v1/best.pt").open("ab") as stream:
            stream.write(b"changed model")
        with self.assertRaisesRegex(f.EvaluationBlocked, "bind best.pt"):
            self.build(output_dir=f.BASE / "freeze_model_v2")

    def test_test_artifact_path_rejected_before_open(self):
        with self.assertRaisesRegex(f.EvaluationBlocked, "TEST input"):
            f.build_freeze(self.root, training_dir=f.BASE / "TEST/nonexistent")


if __name__ == "__main__":
    unittest.main()
