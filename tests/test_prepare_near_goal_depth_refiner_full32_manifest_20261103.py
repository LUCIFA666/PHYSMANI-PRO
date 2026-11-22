"""Synthetic256 whole-cohort tests. No real capture/fit files are read."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from tests.test_prepare_near_goal_depth_refiner_manifest_independent_20261103 import (
    complete_capture_fixture, altered_json, ROOT,
)
from tests.test_near_goal_depth_refiner_20261103 import write_json, write_npz, ptr
from our_mani.near_goal_depth_refiner_20261103 import data

SPEC = importlib.util.spec_from_file_location('new_full32_manifest_unit',
    ROOT/'scripts/prepare_near_goal_depth_refiner_TRAIN_full32_manifest_20261103.py')
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)


def full32_fixture(capture_root):
    """Extend fake metadata/files to256; never evidence for a real run."""
    plan_path = complete_capture_fixture(capture_root)
    plan = json.loads(plan_path.read_text())
    template = capture_root/data.PILOT_IDS[0]
    template_run = json.loads((template/'run_inputs.json').read_text())
    template_sources = json.loads((template/'source_snapshot/manifest.json').read_text())
    template_result = json.loads((template/'result.json').read_text())
    template_launch = json.loads((capture_root/(data.PILOT_IDS[0]+'.owner.json')).read_text())
    base_shots = deepcopy(plan['parents'][0]['shots'])
    plan.update(schema=builder.PLAN_SCHEMA, scope='full32')
    parents, rows = [], []
    for ordinal, pid in enumerate(data.PARTITION_IDS):
        role = 'fit' if ordinal < 24 else ('selection' if ordinal < 28 else 'calibration')
        qfile = capture_root/(pid+'.full_q_source')
        qfile.write_text('synthetic new full32 source '+pid)
        q = dict(parent_seed=261103600+ordinal, parent_role=role,
            file=ptr(qfile), dataset='low_dim_obs/joint_positions', index=0)
        rows.append(q)
        direction = (np.asarray(builder.OCTANTS[ordinal % 8])/np.sqrt(3.)).tolist()
        directions = dict(near_direction_ordinal=ordinal % 8, near_direction_unit_world=direction)
        shots = deepcopy(base_shots)
        for shot in shots:
            shot.update(directions)
            shot['condition_detail'] += '_dir'+str(ordinal % 8)
        parents.append(dict(parent_id=pid, seed=261103600+ordinal, role=role, fold=role,
                            TRAIN_q_source=q, shots=shots, **directions))
        if not (capture_root/pid).exists():
            shutil.copytree(template, capture_root/pid)
    registry = capture_root/'full32_q_registry.json'
    write_json(registry, dict(rows=rows, pilot_rows_exactly_preserved=True))
    plan.update(parents=parents, TRAIN_q_registry=ptr(registry))
    write_json(plan_path, plan)
    plan_sha = data.sha256(plan_path)
    for parent in parents:
        pid, case = parent['parent_id'], capture_root/parent['parent_id']
        run = deepcopy(template_run)
        run.update(parent_id=pid, seed=parent['seed'], fold=parent['role'],
            TRAIN_q_source=parent['TRAIN_q_source'], plan_sha256=plan_sha)
        write_json(case/'run_inputs.json', run)
        (case/'source_snapshot/plan.json').write_bytes(plan_path.read_bytes())
        sources = deepcopy(template_sources)
        for entry in sources['files']:
            if entry['original'] == str(plan_path):
                entry['sha256'] = plan_sha
        write_json(case/'source_snapshot/manifest.json', sources)
        result = deepcopy(template_result)
        result.update(parent_id=pid)
        for frame, planned in zip(result['shots'], parent['shots']):
            sid, scene = planned['shot_id'], pid+'/'+planned['shot_id']
            lp, meta = case/'labels'/(sid+'.npz'), case/'labels'/(sid+'.json')
            with np.load(lp, allow_pickle=False) as opened:
                labels = {k: opened[k].copy() for k in opened.files}
            labels['scene_id'] = np.array(scene)
            write_npz(lp, labels)
            metadata = json.loads(meta.read_text())
            metadata.update(scene_id=scene, condition_detail=planned['condition_detail'],
                near_direction_ordinal=parent['near_direction_ordinal'],
                near_direction_unit_world=parent['near_direction_unit_world'])
            write_json(meta, metadata)
            frame.update(scene_id=scene, condition_detail=planned['condition_detail'],
                near_direction_ordinal=parent['near_direction_ordinal'],
                near_direction_unit_world=parent['near_direction_unit_world'])
            frame['labels']['sha256'] = data.sha256(lp)
            frame['annotation_metadata']['sha256'] = data.sha256(meta)
        materialized = case/'labels/materialized_shot_plan.json'
        write_json(materialized, dict(parent_id=pid, complete=True, shots=parent['shots']))
        result['materialized_shot_plan_sha256'] = data.sha256(materialized)
        write_json(case/'result.json', result)
        owner = dict(parent_id=pid, output=str(case), actual_process_returncode=0,
            actual_wait_completed=True, source_unchanged=True, stop_reason=None,
            started_utc='2026-11-03T01:00:00+00:00', finished_utc='2026-11-03T01:01:00+00:00')
        write_json(capture_root/(pid+'.owner.actual_exit.json'), owner)
        launch = deepcopy(template_launch)
        launch.update(parent_id=pid)
        launch['argv'][5], launch['argv'][7] = pid, str(case)
        launch['source_sha256'][str(plan_path)] = plan_sha
        write_json(capture_root/(pid+'.owner.json'), launch)
        (capture_root/(pid+'.owner.log')).write_text('synthetic full32 only; no actual process started\n')
    return plan_path, registry


class Full32ManifestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix='synthetic_full_cohort_')
        cls.base = Path(cls.temp.name)
        cls.capture_root = cls.base/'captures'
        cls.plan, cls.registry = full32_fixture(cls.capture_root)
        cls.registry_patch = patch.object(builder, 'FULL_REGISTRY_SHA256', data.sha256(cls.registry))
        cls.registry_patch.start()  # synthetic trust anchor only; production constant is unchanged

    @classmethod
    def tearDownClass(cls):
        cls.registry_patch.stop()
        cls.temp.cleanup()

    def output(self, name):
        return self.base/name

    def test_new256_cohort_fully_validated_and_atomic_no_fit(self):
        out = self.output('complete')
        receipt = builder.build_manifest(self.plan, self.capture_root, out)
        self.assertEqual(receipt['shot_count'], 256)
        self.assertEqual(receipt['role_shots'], dict(fit=192, selection=32, calib=32))
        self.assertEqual(receipt['predeclared_direction_parent_counts'], {str(i): 4 for i in range(8)})
        self.assertFalse(receipt['old_pilot_DATA_reused'])
        self.assertFalse(receipt['fit_started'])
        self.assertEqual(len(data.DepthInfluenceDataset(out/'manifest.json')), 256)
        self.assertFalse((out/'.manifest.candidate.json').exists())
        with self.assertRaises(FileExistsError):
            builder.build_manifest(self.plan, self.capture_root, out)

    def test_missing_parent_refuses_before_NPZ_or_oracle(self):
        missing = self.capture_root/data.PARTITION_IDS[-1]
        saved = self.base/'saved_case'
        missing.rename(saved)
        try:
            with patch.object(builder, 'DepthInfluenceDataset', side_effect=AssertionError('no data before closure')):
                with self.assertRaises(ValueError):
                    builder.build_manifest(self.plan, self.capture_root, self.output('missing'))
            self.assertFalse(self.output('missing').exists())
        finally:
            saved.rename(missing)

    def test_old_pilot_plan_mix_cannot_supply_the_eight_existing_parent_ids(self):
        case = self.capture_root/data.PILOT_IDS[0]
        with altered_json(case/'run_inputs.json', lambda x: x.update(plan_sha256='old_pilot_plan_SHA')):
            out = self.output('mixed')
            with self.assertRaises(ValueError):
                builder.build_manifest(self.plan, self.capture_root, out)
            self.assertFalse((out/'manifest.json').exists())

    def test_old_scope_wrong_fold_and_direction_identity_refused(self):
        for name, mutate in (
            ('oldscope', lambda x: x.update(scope='pilot64')),
            ('fold', lambda x: x['parents'][24].update(role='fit', fold='fit')),
            ('direction', lambda x: x['parents'][0].update(near_direction_unit_world=[1., 0., 0.]))):
            with altered_json(self.plan, mutate):
                with self.assertRaises(ValueError):
                    builder.build_manifest(self.plan, self.capture_root, self.output(name))

    def test_actual_direction_metadata_cannot_differ_from_planned_direction(self):
        case = self.capture_root/data.PARTITION_IDS[0]
        result_path, metadata_path = case/'result.json', case/'labels/shot00.json'
        saved_result, saved_metadata = result_path.read_bytes(), metadata_path.read_bytes()
        try:
            value = json.loads(saved_metadata)
            value['near_direction_ordinal'] = 7
            write_json(metadata_path, value)
            result = json.loads(saved_result)
            result['shots'][0]['annotation_metadata']['sha256'] = data.sha256(metadata_path)
            write_json(result_path, result)
            with self.assertRaisesRegex(ValueError, 'direction'):
                builder.build_manifest(self.plan, self.capture_root, self.output('metadata_direction'))
            self.assertFalse(self.output('metadata_direction').exists())
        finally:
            metadata_path.write_bytes(saved_metadata)
            result_path.write_bytes(saved_result)


if __name__ == '__main__':
    unittest.main()
