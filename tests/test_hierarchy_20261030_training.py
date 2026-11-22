"""Small CPU checks of the real pretraining data and action-loss contracts."""
import tempfile
import unittest
import argparse
import contextlib
import io
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from scipy.spatial.transform import Rotation
from torch.utils.data import DataLoader

from our_mani.hierarchy_20261030.data import SCHEMA, TASK_FAMILIES, file_sha256
from our_mani.hierarchy_20261030 import training as training_module
from our_mani.hierarchy_20261030.networks import CausalSceneSkillEncoder, PretrainedSkillPolicy
from our_mani.hierarchy_20261030.training import (
    ROOT, CausalSkillWindows, evaluate, quaternion_matrix, skill_loss,
)


class HierarchyTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(31)
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.manifest = {"episodes": []}
        for split in ("train", "validation"):
            for task_index, task in enumerate(TASK_FAMILIES):
                self._episode(split, task, "episode0", 6, 20 if split == "train" else 160)
        self._episode("train", next(iter(TASK_FAMILIES)), "episode1", 8, 35)

    def tearDown(self):
        self.temporary.cleanup()

    def _episode(self, split, task, episode, n, pixel_base):
        destination = self.root / split / task / (episode + ".npz")
        destination.parent.mkdir(parents=True, exist_ok=True)
        rgb = np.empty((n, 4, 64, 64, 3), np.uint8)
        for frame in range(n):
            rgb[frame] = pixel_base + frame
        robot = np.zeros((n, 15), np.float32)
        robot[:, 0] = np.arange(n) * .01
        robot[:, 6] = 1
        robot[:, 7] = 1
        action = np.zeros((n, 8), np.float32)
        # Repeated commands coexist with genuinely changing targets inside a
        # skill segment. Robot pose is deliberately different from supervision.
        command_position = np.arange(n, dtype=np.float32) // 2 * .1
        action[:, 0] = command_position + .3
        action[:, 2] = .8
        action[:, 6] = 1
        action[:, 7] = np.arange(n) < n // 2
        labels = np.full(n, 0 if task.startswith("pick_") else 4, np.int64)
        labels[n // 2:] = 3 if task.startswith("pick_") else 7
        starts = np.r_[True, np.any(action[1:] != action[:-1], axis=-1)]
        np.savez_compressed(destination, rgb=rgb,
                            depth=np.full((n, 4, 64, 64), 1.2, np.float32),
                            robot=robot, expert_actions=action, skill_labels=labels,
                            expert_command_run=np.cumsum(starts) - 1)
        self.manifest["episodes"].append(dict(
            split=split, task=task, raw_episode_id=episode,
            parent_source_group=f"{split}:{task}:{episode}",
            npz_path=str(destination.relative_to(self.root)), frames=n,
            cache_sha256=file_sha256(destination),
        ))

    def test_project_root_resolves_to_real_repository(self):
        self.assertTrue((ROOT / "our_mani/hierarchy_20261030/training.py").is_file())
        self.assertTrue((ROOT / "scripts/train_hierarchy_skills_20261030.py").is_file())

    def test_windows_use_only_causal_scene_and_measured_robot_inputs(self):
        dataset = CausalSkillWindows(self.root, self.manifest, "train", history_frames=5)
        original = dataset[2]
        self.assertEqual(original['lengths'], 3)
        self.assertEqual(original['frame'], 2)
        np.testing.assert_array_equal(original['robot_history'][2], dataset.episodes[0]['robot'][2])
        self.assertFalse(np.array_equal(original['robot_history'][2, :3], original['action'][:3]))
        self.assertTrue(np.all(original['rgbd'][3:] == 0))
        self.assertTrue(np.all(original['robot_history'][3:] == 0))
        arrays = dataset.episodes[0]
        arrays['rgb'][3:] = 255
        arrays['depth'][3:] = 200
        arrays['robot'][3:] = 200
        arrays['skill_labels'][:] = 10
        arrays['expert_actions'][:] = 123
        after = dataset[2]
        np.testing.assert_array_equal(after['rgbd'], original['rgbd'])
        np.testing.assert_array_equal(after['robot_history'], original['robot_history'])
        self.assertEqual(after['lengths'], original['lengths'])
        self.assertNotIn('task_id', original)
        self.assertNotIn('expert_stage', original)

    def test_skill_goal_is_segment_entry_not_each_current_action(self):
        dataset = CausalSkillWindows(self.root, self.manifest, "train")
        first, later, next_segment = dataset[0], dataset[2], dataset[3]
        np.testing.assert_array_equal(later['goal_xyz'], first['action'][:3])
        self.assertGreater(float(np.linalg.norm(later['goal_xyz'] - later['action'][:3])), .05)
        np.testing.assert_array_equal(next_segment['goal_xyz'], next_segment['action'][:3])
        self.assertEqual(float(later['goal_time']), .5)

    def test_sampler_mass_is_equal_per_task_parent_and_distinct_command(self):
        dataset = CausalSkillWindows(self.root, self.manifest, "train")
        task_mass, parent_mass = {}, []
        for episode_index, row in enumerate(dataset.rows):
            selected = np.asarray([ep == episode_index for ep, _ in dataset.anchors])
            weights = dataset.weights[selected]
            parent_mass.append(weights.sum())
            task_mass[row['task']] = task_mass.get(row['task'], 0.) + weights.sum()
            run_ids = dataset.episodes[episode_index]['expert_command_run']
            per_run_mass = [weights[run_ids == run].sum() for run in np.unique(run_ids)]
            np.testing.assert_allclose(per_run_mass, np.full(len(per_run_mass), per_run_mass[0]), rtol=1e-10)
        masses = list(task_mass.values())
        np.testing.assert_allclose(masses, np.full(len(masses), masses[0]), rtol=1e-10)
        self.assertAlmostEqual(parent_mass[0], parent_mass[-1])
        self.assertAlmostEqual(parent_mass[1], 2 * parent_mass[0])

    def test_split_selection_never_reads_other_split_or_test_paths(self):
        for row in self.manifest['episodes']:
            if row['split'] == 'validation':
                row['npz_path'] = 'validation/DO_NOT_OPEN.npz'
        self.manifest['episodes'].append(dict(split='test', task=next(iter(TASK_FAMILIES)), npz_path='TEST/DO_NOT_OPEN.npz'))
        training = CausalSkillWindows(self.root, self.manifest, 'train')
        self.assertTrue(all(row['split'] == 'train' for row in training.rows))
        self.assertEqual(training.summary['episodes'], 5)
        with self.assertRaises(ValueError):
            CausalSkillWindows(self.root, self.manifest, 'test')
        with self.assertRaises(FileNotFoundError):
            CausalSkillWindows(self.root, self.manifest, 'validation')

    def test_changed_cache_is_rejected_before_training(self):
        row = next(row for row in self.manifest['episodes'] if row['split'] == 'train')
        path = self.root / row['npz_path']
        with path.open('ab') as stream:
            stream.write(b'changed-cache')
        with self.assertRaisesRegex(ValueError, 'fingerprint'):
            CausalSkillWindows(self.root, self.manifest, 'train')

    def test_xyzw_target_rotations_match_scipy_and_sign_is_irrelevant(self):
        quaternion = Rotation.random(12, random_state=31).as_quat().astype(np.float32)
        actual = quaternion_matrix(torch.tensor(quaternion))
        expected = torch.tensor(Rotation.from_quat(quaternion).as_matrix(), dtype=torch.float32)
        torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-6)
        torch.testing.assert_close(actual, quaternion_matrix(-torch.tensor(quaternion)))

    def test_skill_loss_is_finite_with_frozen_scene_encoder_and_no_trajectory(self):
        dataset = CausalSkillWindows(self.root, self.manifest, 'train')
        batch = next(iter(DataLoader(dataset, batch_size=2, shuffle=False)))
        encoder = CausalSceneSkillEncoder().eval()
        encoder.requires_grad_(False)
        policy = PretrainedSkillPolicy((.4, 0., .8), (.5, .5, .5))
        with torch.no_grad():
            encoded = encoder(batch['rgbd'], batch['robot_history'], batch['lengths'])
        rows = torch.arange(len(batch['lengths']))
        robot = batch['robot_history'][rows, batch['lengths'] - 1]
        predicted = policy(encoded['scene_features'], encoded['skill_latent'], robot, batch['goal_xyz'], batch['goal_time'])
        self.assertFalse(predicted['trajectory_consumed'])
        losses = skill_loss(predicted, batch['action'], torch.tensor([.5, .5, .5]))
        self.assertEqual(losses.shape, (2,))
        self.assertTrue(bool(torch.isfinite(losses).all()))
        losses.mean().backward()
        self.assertTrue(all(parameter.grad is None for parameter in encoder.parameters()))
        for head in (policy.xyz_head, policy.rotation_head, policy.gripper_head):
            self.assertTrue(bool(torch.isfinite(head.weight.grad).all()))
        opposite = batch['action'].clone()
        opposite[:, 3:7] *= -1
        torch.testing.assert_close(losses, skill_loss(predicted, opposite, torch.tensor([.5, .5, .5])))

    def test_offline_evaluation_does_not_claim_native_or_candidate_success(self):
        dataset = CausalSkillWindows(self.root, self.manifest, 'validation')
        encoder = CausalSceneSkillEncoder()
        policy = PretrainedSkillPolicy((.4, 0., .8), (.5, .5, .5))
        result = evaluate(encoder, policy, DataLoader(dataset, batch_size=4), torch.device('cpu'), 'skill', torch.ones(3) * .5)
        self.assertFalse(result['native_success_measured'])
        self.assertFalse(result['candidate_network_evaluated'])
        self.assertFalse(result['trajectory_consumed'])
        self.assertEqual(set(result['per_task']), set(TASK_FAMILIES))
        self.assertTrue(np.isfinite(result['loss']))
        self.assertTrue(np.isfinite(result['position_error_m']))
        expected_copy = sum(
            dataset.weights[index] * np.linalg.norm(dataset[index]['goal_xyz'] - dataset[index]['action'][:3])
            for index in range(len(dataset))
        ) / dataset.weights.sum()
        self.assertAlmostEqual(result['copy_segment_entry_goal_position_error_m'], float(expected_copy), places=6)

    def test_real_two_stage_cpu_loop_learns_embedding_and_freezes_visual_classifier(self):
        # Exercise the actual training loop with a tiny four-task fixture. All
        # outputs remain in TemporaryDirectory; no real dataset, GPU or simulator
        # is used. This catches missing optimizer membership/no_grad mistakes.
        code_names = (
            'our_mani/hierarchy_20261030/training.py',
            'our_mani/hierarchy_20261030/networks.py',
            'our_mani/hierarchy_20261030/data.py',
            'scripts/train_hierarchy_skills_20261030.py',
        )
        for name in code_names:
            destination = self.root / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.symlink_to(ROOT / name)
        config_path = self.root / 'config.json'
        config_path.write_text(json.dumps(dict(
            seed=31, scene_history_frames=3,
            skill_pretraining=dict(batch_size=16, classification_epochs=1,
                                   skill_epochs=1, learning_rate=1e-3, weight_decay=0.),
        )))
        self.manifest.update(schema=SCHEMA, complete=True, smoke_only=False,
                             TEST_read=False, source_debug_target_positions_read=False)
        manifest_path = self.root / 'manifest.json'
        manifest_path.write_text(json.dumps(self.manifest))
        relative_output = 'artifacts/framework/multitask_r10/hierarchy_20261030/unit_cpu'
        args = argparse.Namespace(config='config.json', data_manifest='manifest.json',
                                  out=relative_output, device='cpu', batch_size=16,
                                  workers=0, smoke=False)
        try:
            with patch.object(training_module, 'ROOT', self.root), \
                 patch.object(training_module, '__file__', str(self.root / code_names[0])), \
                 contextlib.redirect_stdout(io.StringIO()):
                result = training_module.run(args)
        finally:
            torch.set_num_threads(2)
        output = self.root / relative_output
        classification = torch.load(output / 'best_classification.pt', map_location='cpu', weights_only=False)
        skill = torch.load(output / 'best_skill.pt', map_location='cpu', weights_only=False)
        before, after = classification['encoder_state_dict'], skill['encoder_state_dict']
        self.assertFalse(torch.equal(before['skill_embeddings.weight'], after['skill_embeddings.weight']))
        for name in before:
            if name != 'skill_embeddings.weight':
                torch.testing.assert_close(before[name], after[name], rtol=0, atol=0, msg=name)
        self.assertIsNone(classification['skill_state_dict'])
        self.assertIsNotNone(skill['skill_state_dict'])
        self.assertFalse(skill['metadata']['skill_policy_trajectory_consumed'])
        self.assertFalse(skill['metadata']['candidate_network_trained'])
        self.assertTrue(result['complete'])
        self.assertFalse(result['full_hierarchy_ready'])


if __name__ == '__main__':
    unittest.main()
