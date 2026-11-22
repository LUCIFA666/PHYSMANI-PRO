"""Synthetic fixed-cohort contracts and CPU gradients, no optimizer/Native."""
from copy import deepcopy
import json
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from our_mani.near_goal_depth_refiner_20261103 import data as d
from our_mani.near_goal_depth_refiner_20261103 import model as m
from our_mani.near_goal_depth_refiner_20261103 import evaluation as e
from our_mani.local_mpc_20261103.depth_owner_capture_sensors import _labels


def public():
    return dict(rgb=np.zeros((4, 256, 256, 3), np.uint8), depth_m=np.ones((4, 256, 256), np.float32),
        K=np.repeat(np.array([[-100., 0, 128], [0, -100., 128], [0, 0, 1]])[None], 4, 0),
        T=np.repeat(np.eye(4)[None], 4, 0), public_sphere_center=np.array([0., 0., 1.]),
        camera_names=np.array(d.CAMERAS), timestamp=np.array(0.))


def labels(inputs):
    base = dict(rgb=inputs['rgb'], depth_m=inputs['depth_m'])
    cube = {k: v.copy() for k, v in base.items()}
    goal = {k: v.copy() for k, v in base.items()}
    cube['depth_m'][:, 120:136, 120:136] = 1.1
    return _labels(base, dict(cube=cube, goal=goal))


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def write_npz(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **value)


def ptr(path):
    return dict(path=str(path), sha256=d.sha256(path))


def complete_pilot(tmp, *, cross_fold_q=False):
    """Fake evidence lives only in temporary unit-test files; never runtimeready."""
    partition = tmp/'partition.json'
    write_json(partition, dict(parents=[dict(parent_id=pid, seed=261103600+i,
        role='fit' if i < 24 else ('selection' if i < 28 else 'calibration'),
        pilot=pid in d.PILOT_IDS, shot_ids=list(d.SHOT_IDS)) for i, pid in enumerate(d.PARTITION_IDS)]))
    source = tmp/'source.txt'
    source.write_text('synthetic test source, no Native')
    repository = Path(d.__file__).resolve().parents[2]
    repo_map = {'synthetic_unit_source.py': d.sha256(source)}
    external = {str(source): d.sha256(source)}
    native = {'synthetic_task': '0'*64}
    native_cfg = dict(path='synthetic_config', sha256='1'*64)
    plan = tmp/'plan.json'
    plan_parents = [dict(parent_id=pid, seed=261103600+d.PARTITION_IDS.index(pid),
        fold='fit' if pid in d.PARTITION_IDS[:24] else ('selection' if pid in d.PARTITION_IDS[24:28] else 'calibration'),
        shots=[dict(shot_id=s, condition='sphere_only' if s == 'shot00' else 'near_goal',
                    condition_detail='synthetic_sphere_only' if s == 'shot00' else 'synthetic_near')
               for s in d.SHOT_IDS]) for pid in d.PILOT_IDS]
    for parent in plan_parents:
        source_q = tmp/(parent['parent_id']+'.q_source')
        source_q.write_text('synthetic q origin; no Native data')
        parent['TRAIN_q_source'] = dict(file=ptr(source_q), dataset='low_dim_obs/joint_positions', index=0)
    if cross_fold_q:
        plan_parents[-1]['TRAIN_q_source'] = deepcopy(plan_parents[0]['TRAIN_q_source'])
    write_json(plan, dict(split='train', label_method='counterfactual_current_depth_owner',
        native_instance_id_masks=False, material_or_alpha_edits=False, render_warmup_rounds=12,
        parents=plan_parents, source_sha256=repo_map, external_source_sha256=external, native_config=native_cfg))
    inputs = public()
    label_base = labels(inputs)
    manifest_parents = []
    for planned in plan_parents:
        pid, seed, role = planned['parent_id'], planned['seed'], planned['fold']
        case = tmp/pid
        initial = case/'initialization/public_rgbd.npz'
        write_npz(initial, inputs)
        fit = case/'public_initial_sphere.json'
        write_json(fit, dict(Native_pose_used=False, metadata=dict(simulator_pose_used=False),
                             sphere=dict(center=inputs['public_sphere_center'].tolist())))
        archive = case/'source_snapshot'
        archive.mkdir()
        (archive/'source.txt').write_bytes(source.read_bytes())
        (archive/'plan.json').write_bytes(plan.read_bytes())
        source_manifest = archive/'manifest.json'
        write_json(source_manifest, dict(files=[
            dict(original=str(repository/'synthetic_unit_source.py'), archive='source.txt', sha256=d.sha256(source)),
            dict(original=str(source), archive='source.txt', sha256=d.sha256(source)),
            dict(original=str(plan), archive='plan.json', sha256=d.sha256(plan))]))
        run = case/'run_inputs.json'
        write_json(run, dict(parent_id=pid, seed=seed, fold=role, plan_sha256=d.sha256(plan),
            TRAIN_q_source=planned['TRAIN_q_source'],
            source_sha256=repo_map, external_source_sha256=external,
            Native_sources_sha256=native, native_config=native_cfg,
            label_method='counterfactual_current_depth_owner'))
        owner = case/'owner.actual_exit.json'
        write_json(owner, dict(parent_id=pid, actual_process_returncode=0,
            actual_wait_completed=True, source_unchanged=True, stop_reason=None))
        proof = case/'labels/outside_frusta_proof.json'
        write_json(proof, dict(verified=True, all_four_camera_frusta_outside=True,
                              source='predeclared_analytic_four_camera_frusta'))
        frames, shots = [], []
        for shot_id in d.SHOT_IDS:
            scene = pid+'/'+shot_id
            condition = 'sphere_only' if shot_id == 'shot00' else 'near_goal'
            detail = 'synthetic_sphere_only' if shot_id == 'shot00' else 'synthetic_near'
            ip, lp, meta = case/'inputs'/(shot_id+'.npz'), case/'labels'/(shot_id+'.npz'), case/'labels'/(shot_id+'.json')
            write_npz(ip, inputs)
            base = dict(rgb=inputs['rgb'], depth_m=inputs['depth_m'])
            raw_labels = _labels(base, dict(cube=base, goal=base)) if condition == 'sphere_only' else label_base
            label = dict(raw_labels, cube_pose_world=np.array([0., 0., .975, 0., 0., 0., 1.]),
                         timestamp=np.array(0.), counted_tick=np.array(0), scene_id=np.array(scene))
            write_npz(lp, label)
            write_json(meta, dict(scene_id=scene, shot_id=shot_id, condition=condition, condition_detail=detail,
                cube_scene_present=True, absence_from_all_camera_frusta_by_construction=condition == 'sphere_only',
                absence_proof=dict(path='labels/outside_frusta_proof.json', sha256=d.sha256(proof))
                    if condition == 'sphere_only' else None))
            camera = dict(label_method='counterfactual_current_depth_owner', warmup_rounds_fixed=12,
                **{k: True for k in ('original_clone_equal', 'all_counterfactual_flags_restored',
                    'all_restore_RGBD_equal', 'clock_unchanged', 'handling_restored',
                    'global_vision_sensors_enabled', 'created_nonpassive')})
            frames.append(dict(shot_id=shot_id, scene_id=scene, timestamp=0.,
                input=dict(path=str(ip.relative_to(case)), sha256=d.sha256(ip)),
                labels=dict(path=str(lp.relative_to(case)), sha256=d.sha256(lp)),
                camera=camera, augmentation=dict(restored=True)))
            shots.append(dict(shot_id=shot_id, scene_id=scene, independent_scene=True, physics_ticks=0,
                condition=condition, condition_detail=detail, inputs=ptr(ip), labels=ptr(lp), annotation_metadata=ptr(meta)))
        result = case/'result.json'
        write_json(result, dict(complete=True, native_closed=True, source_unchanged=True,
            parent_id=pid, controlled_physics_ticks=0, Native_sources_sha256_after=native, shots=frames))
        manifest_parents.append(dict(parent_id=pid, seed=seed, role=role,
            capture=dict(result=ptr(result), owner_actual_exit=ptr(owner), run_inputs=ptr(run), source_manifest=ptr(source_manifest)),
            initial_public_rgbd=ptr(initial), public_sphere_fit=ptr(fit), shots=shots))
    manifest = tmp/'manifest.json'
    write_json(manifest, dict(schema=d.SCHEMA, dataset_split='TRAIN', label_protocol=d.LABEL_PROTOCOL,
        scope='pilot_smoke', complete=True, all_native_closed=True,
        all_process_exits_observed=True, source_unchanged=True,
        partition=ptr(partition), capture_plan=ptr(plan), parents=manifest_parents))
    return manifest


class DepthRefinerTests(unittest.TestCase):
    def test_current7_crop_is_exact_and_GT_does_not_change_raw_labels(self):
        inputs = public()
        annotations = labels(inputs)
        target = d.binary_depth_labels(inputs, annotations)
        self.assertTrue((target[:, 120, 125] == 1).all())  # raw inner edge remains positive
        self.assertTrue((target[:, 119, 125] == 255).all())  # zero-delta outer ring UNKNOWN
        annotations['cube_pose_world'] = np.full(7, np.nan)
        annotations['TCP_future_GT'] = np.full(100, 999.)
        np.testing.assert_array_equal(target, d.binary_depth_labels(inputs, annotations))
        from our_mani.near_goal_segmentation_20261103.data import make_near_goal_crops
        for k, value in d.make_near_goal_crops(inputs).items():
            np.testing.assert_array_equal(value, make_near_goal_crops(inputs)[k])
        with self.assertRaises(ValueError):
            d.make_near_goal_crops(dict(inputs, tcp_pose=np.zeros(7)))

    def test_CPU_forward_backward_no_optimizer_no_weight_update(self):
        torch.set_num_threads(2)
        network = m.TinyDepthInfluenceUNet()
        before = {k: v.detach().clone() for k, v in network.state_dict().items()}
        y = torch.zeros(4, 64, 64, dtype=torch.long)
        y[:, 20:30, 20:30] = 1
        y[:, 0] = 255
        output = network(torch.randn(4, 9, 64, 64))
        self.assertEqual(tuple(output.shape), (4, 2, 64, 64))
        loss = m.binary_segmentation_loss(output, y).mean()
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in network.parameters()))
        self.assertTrue(all(torch.equal(before[k], v) for k, v in network.state_dict().items()))
        zero = m.binary_segmentation_loss(output, torch.full_like(y, 255))
        self.assertTrue(torch.equal(zero, torch.zeros_like(zero)))
        with self.assertRaises(ValueError):
            network(dict(features=torch.zeros(4, 9, 64, 64), GT=np.ones(3)))

    def test_conservative_U_can_reject_perfect_known_precision(self):
        truth = np.array([1]*1000+[255]*6, np.uint8)
        metric = e.conservative_pixel_metrics(truth, np.ones(len(truth), bool))
        self.assertEqual(metric['known_precision'], 1.)
        self.assertEqual(metric['U'], 6)
        self.assertLess(metric['conservative_precision_lowerbound'], .995)

    def test_complete64_fixture_input_separation_and_tamper_failclosed(self):
        with tempfile.TemporaryDirectory() as td:
            manifest = complete_pilot(Path(td))
            dataset = d.DepthInfluenceDataset(manifest)
            self.assertEqual(len(dataset), 64)
            self.assertEqual(len(d.DepthInfluenceDataset(manifest, 'fit')), 48)
            self.assertEqual(len(d.DepthInfluenceDataset(manifest, 'calibration')), 8)
            sample = dataset[0]
            self.assertEqual(set(sample['inputs']), d.PUBLIC_KEYS)
            self.assertEqual(sample['features'].shape, (4, 9, 64, 64))
            self.assertNotIn('cube_pose_world', sample['inputs'])
            self.assertTrue(sample['absence_from_all_camera_frusta_by_construction'])
            self.assertFalse((sample['semantic'] == 1).any())
            dataset.revalidate()
            proof = dataset.all_records[0]['labels_path'].parent/'outside_frusta_proof.json'
            original_proof = proof.read_bytes()
            proof.write_bytes(b'tamperedanalyticproof')
            with self.assertRaises(ValueError):
                d.DepthInfluenceDataset(manifest)
            proof.write_bytes(original_proof)
            document = json.loads(manifest.read_text())
            removed = document['parents'].pop()
            write_json(manifest, document)
            with self.assertRaisesRegex(ValueError, 'whole-parent'):
                d.DepthInfluenceDataset(manifest)
            document['parents'].append(removed)
            write_json(manifest, document)
            all_shots = document['parents'][0]['shots']
            document['parents'][0]['shots'] = all_shots[:2]
            write_json(manifest, document)
            with self.assertRaisesRegex(ValueError, 'two-shot smoke'):
                d.DepthInfluenceDataset(manifest)
            document['parents'][0]['shots'] = all_shots
            write_json(manifest, document)
            pointer = document['parents'][0]['shots'][0]['labels']
            Path(pointer['path']).write_bytes(b'tamperedlabel')
            with self.assertRaises(ValueError):
                dataset[0]

    def test_default_CLI_cannot_fit_or_construct_optimizer(self):
        path = Path(__file__).resolve().parents[1]/'scripts/train_near_goal_depth_refiner_20261103.py'
        spec = importlib.util.spec_from_file_location('binary_refiner_CLI_test', path)
        cli = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cli)
        class FakeDataset:
            scope = 'pilot_smoke'
            def __init__(self, manifest):
                pass
            def __len__(self):
                return 1
            def __getitem__(self, index):
                if index > 0:
                    raise IndexError(index)
                return dict(features=np.zeros((4, 9, 64, 64), np.float32))
            def revalidate(self):
                pass
        with patch.object(cli, 'DepthInfluenceDataset', FakeDataset), \
                patch.object(cli, 'train_capture', side_effect=AssertionError('fit not authorized')), \
                patch('torch.optim.AdamW', side_effect=AssertionError('no optimizer in check')):
            self.assertEqual(cli.main(['--manifest', 'not-opened']), 0)

    def test_q_source_group_cannot_cross_fold_or_differ_from_actual_run(self):
        with tempfile.TemporaryDirectory() as td:
            manifest = complete_pilot(Path(td), cross_fold_q=True)
            with self.assertRaisesRegex(ValueError, 'q source group cannot cross'):
                d.DepthInfluenceDataset(manifest)
        with tempfile.TemporaryDirectory() as td:
            manifest = complete_pilot(Path(td))
            document = json.loads(manifest.read_text())
            pointer = document['parents'][0]['capture']['run_inputs']
            path = Path(pointer['path'])
            run = json.loads(path.read_text())
            run['TRAIN_q_source']['index'] = 99
            write_json(path, run)
            pointer['sha256'] = d.sha256(path)
            write_json(manifest, document)
            with self.assertRaisesRegex(ValueError, 'actual|Actual'):
                d.DepthInfluenceDataset(manifest)


if __name__ == '__main__':
    unittest.main()
