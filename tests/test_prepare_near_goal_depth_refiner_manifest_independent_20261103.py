"""Synthetic closed-cohort manifest checks; never Native, fit or real TEST."""
from contextlib import contextmanager
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from tests.test_near_goal_depth_refiner_20261103 import complete_pilot, ptr, write_json
from our_mani.near_goal_depth_refiner_20261103 import data

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('independent_closed_manifest_builder',
    ROOT/'scripts/prepare_near_goal_depth_refiner_TRAIN_manifest_20261103.py')
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)


def complete_capture_fixture(root):
    """Temporary synthetic attestations, explicitly no trained/Native evidence."""
    original = json.loads(complete_pilot(root).read_text())
    plan_path = root/'plan.json'; plan = json.loads(plan_path.read_text())
    plan.update(scope='pilot64', perception_only=True, Native_started=False,
        fit_started=False, TEST_read=False, benchmark_episode=False, execution_transition=False,
        partition=original['partition'], collector_script='scripts/synthetic_unit_collector.py')
    source = root/'source.txt'; source_sha = data.sha256(source)
    shell = 'scripts/synthetic_unit_capture.sh'
    plan['source_sha256'].update({key: source_sha for key in (
        'scripts/launch_near_goal_perception_capture_20261103.py', shell, plan['collector_script'])})
    for parent in plan['parents']: parent['role'] = parent['fold']
    write_json(plan_path, plan)
    plan_sha = data.sha256(plan_path)
    for parent, old in zip(plan['parents'], original['parents']):
        pid = parent['parent_id']; case = root/pid
        run_path = case/'run_inputs.json'; run = json.loads(run_path.read_text())
        run.update(plan_sha256=plan_sha, source_sha256=plan['source_sha256'])
        write_json(run_path, run)
        archive = case/'source_snapshot'; (archive/'plan.json').write_bytes(plan_path.read_bytes())
        archive_manifest = archive/'manifest.json'; sources = json.loads(archive_manifest.read_text())
        sources['files'] = [x for x in sources['files'] if x['original'] != str(plan_path)]
        existing = {x['original'] for x in sources['files']}
        for key in plan['source_sha256']:
            name = str((ROOT/key).resolve())
            if name not in existing:
                sources['files'].append(dict(original=name, archive='source.txt', sha256=source_sha))
        sources['files'].append(dict(original=str(plan_path), archive='plan.json', sha256=plan_sha))
        write_json(archive_manifest, sources)
        result_path = case/'result.json'; result = json.loads(result_path.read_text())
        for frame, shot in zip(result['shots'], old['shots']):
            frame.update(condition=shot['condition'], condition_detail=shot['condition_detail'],
                independent_scene=True, physics_ticks=0,
                annotation_metadata=dict(path=str(Path(shot['annotation_metadata']['path']).relative_to(case)),
                    sha256=shot['annotation_metadata']['sha256']))
        write_json(result_path, result)
        started, finished = '2026-11-03T01:00:00+00:00', '2026-11-03T01:01:00+00:00'
        owner = json.loads((case/'owner.actual_exit.json').read_text())
        owner.update(output=str(case), started_utc=started, finished_utc=finished)
        write_json(root/(pid+'.owner.actual_exit.json'), owner)
        launch_sources = {key: plan['source_sha256'][key] for key in (
            'scripts/launch_near_goal_perception_capture_20261103.py', shell, plan['collector_script'])}
        launch_sources[str(plan_path)] = plan_sha
        write_json(root/(pid+'.owner.json'), dict(parent_id=pid, started_utc=started,
            child_pid=12345, process_group=12345, perception_only=True,
            benchmark_episode=False, old_runs_resumed=False, collector_script=plan['collector_script'],
            argv=['bash', str(ROOT/shell), '--plan', str(plan_path), '--parent-id', pid,
                  '--output', str(case)], source_sha256=launch_sources))
        (root/(pid+'.owner.log')).write_text('synthetic unit fixture, no process was started\n')
    return plan_path


@contextmanager
def altered_json(path, mutate):
    previous = path.read_bytes(); value = json.loads(previous); mutate(value)
    write_json(path, value)
    try: yield
    finally: path.write_bytes(previous)


class IndependentManifestBuilderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix='synthetic_manifest_review_')
        cls.base = Path(cls.temp.name); cls.root = cls.base/'captures'
        cls.plan = complete_capture_fixture(cls.root)
        cls.first = data.PILOT_IDS[0]

    @classmethod
    def tearDownClass(cls): cls.temp.cleanup()

    def output(self, name): return self.base/name

    def test_complete64_is_validated_then_exclusively_published_without_fit(self):
        out = self.output('complete')
        result = builder.build_manifest(self.plan, self.root, out)
        self.assertEqual(result['parent_order'], list(data.PILOT_IDS))
        self.assertEqual(result['shot_count'], 64)
        self.assertFalse(result['fit_started']); self.assertFalse(result['Native_started'])
        self.assertTrue(result['pilot_never_runtime_qualified'])
        self.assertEqual(len(data.DepthInfluenceDataset(out/'manifest.json')), 64)
        self.assertFalse((out/'.manifest.candidate.json').exists())
        before = (out/'manifest.json').read_bytes()
        with self.assertRaises(FileExistsError): builder.build_manifest(self.plan, self.root, out)
        self.assertEqual(before, (out/'manifest.json').read_bytes())

    def test_boolean_false_exit_and_wrong_case_refuse_before_dataset_or_output(self):
        owner = self.root/(self.first+'.owner.actual_exit.json')
        for name, mutation in (
            ('false_exit', lambda x: x.update(actual_process_returncode=False)),
            ('wrong_case', lambda x: x.update(output=str(self.root/data.PILOT_IDS[1])))):
            with altered_json(owner, mutation), patch.object(builder, 'DepthInfluenceDataset',
                    side_effect=AssertionError('must not open data until full closure')):
                out = self.output(name)
                with self.assertRaises(ValueError): builder.build_manifest(self.plan, self.root, out)
                self.assertFalse(out.exists())

    def test_duplicate_actual_shot_is_not_dict_deduplicated_or_published(self):
        case = self.root/self.first; result = case/'result.json'
        with altered_json(result, lambda x: x['shots'].__setitem__(-1, deepcopy(x['shots'][0]))):
            out = self.output('duplicate')
            with self.assertRaises(ValueError): builder.build_manifest(self.plan, self.root, out)
            self.assertFalse(out.exists())

    def test_same_count_reordered_parent_or_actual_launch_source_rejected(self):
        with altered_json(self.plan, lambda x: x['parents'].reverse()):
            with self.assertRaises(ValueError): builder.build_manifest(self.plan, self.root, self.output('reordered'))
        launch = self.root/(self.first+'.owner.json')
        with altered_json(launch, lambda x: x['source_sha256'].__setitem__('unexpected_source.py', '0'*64)):
            out = self.output('launch_source')
            with self.assertRaises(ValueError): builder.build_manifest(self.plan, self.root, out)
            self.assertFalse(out.exists())

    def test_public_GT_injection_reaches_frozen_loader_but_cannot_publish(self):
        case = self.root/self.first; result_path = case/'result.json'; input_path = case/'inputs/shot00.npz'
        before_input, before_result = input_path.read_bytes(), result_path.read_bytes()
        with np.load(input_path, allow_pickle=False) as opened: arrays = {k: opened[k] for k in opened.files}
        arrays['cube_pose_world'] = np.zeros(7)
        np.savez_compressed(input_path, **arrays)
        result = json.loads(before_result); result['shots'][0]['input']['sha256'] = data.sha256(input_path)
        write_json(result_path, result)
        out = self.output('poisoned_public')
        try:
            with self.assertRaises(ValueError): builder.build_manifest(self.plan, self.root, out)
            self.assertFalse((out/'manifest.json').exists())
            self.assertFalse(json.loads((out/'failure.json').read_text())['manifest_published'])
        finally:
            input_path.write_bytes(before_input); result_path.write_bytes(before_result)

    def test_output_under_capture_root_cannot_pollute_fixed_cohort(self):
        out = self.root/'aggregate'
        with self.assertRaises(ValueError): builder.build_manifest(self.plan, self.root, out)
        self.assertFalse(out.exists())


if __name__ == '__main__': unittest.main()
