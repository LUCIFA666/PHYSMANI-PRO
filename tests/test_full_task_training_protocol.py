"""Guard complete task coverage, matched collection, and split separation."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from collect_full_task_rollouts import TASKS, sampling


class FullTaskTrainingProtocolTests(unittest.TestCase):
    def test_matching_perturbations_do_not_depend_on_worker_order(self):
        original = {ep: sampling(TASKS[0], f'episode{ep}', 2, 'train') for ep in range(10)}
        for ep in reversed(range(10)):
            repeated = sampling(TASKS[0], f'episode{ep}', 2, 'train')
            self.assertEqual(original[ep][0], repeated[0])
            np.testing.assert_array_equal(original[ep][1], repeated[1])
        self.assertNotEqual(original[0][0], sampling(TASKS[0], 'episode0', 2, 'validation')[0])

    def test_all_train_sources_fit_and_val_groups_are_disjoint(self):
        parent = ROOT/'runtime/framework/tmp'
        parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='protocol-unit-', dir=parent) as tmp:
            base = Path(tmp)
            for split, episodes, first in [('train', 100, 0), ('validation', 20, 1000)]:
                folder = base/split
                folder.mkdir()
                rng = np.random.default_rng(910 if split == 'train' else 911)
                for task in TASKS:
                    for episode in range(first, first+episodes):
                        for replica in range(4):
                            key = f'{task}__episode{episode}__{replica}'
                            row = {'key': key, 'task': task, 'source_episode': f'episode{episode}',
                                'initial_features': rng.normal(size=18).tolist(),
                                'split': split, 'formal_test_used': False, 'full_task_executed': True,
                                'controller_sha256': 'synthetic-controller-unit-fixture',
                                'official_task_success_immediate': bool(replica % 2)}
                            (folder/(key+'.json')).write_text(json.dumps(row))
                (folder/'report.json').write_text(json.dumps({'full_task_executed': True, 'synthetic_unit_fixture': True}))
            out = base/'model'
            subprocess.run([sys.executable, str(ROOT/'scripts/train_success_predictor_full.py'),
                '--data', str(base/'train'), '--validation', str(base/'validation'), '--out', str(out),
                '--outcome', 'official_task_success_immediate', '--updates', '2'],
                cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
            report = json.loads((out/'report.json').read_text())
            parts = report['metadata']['group_partitions']
            fit, select, calibration = [set(map(tuple, parts[k])) for k in ('fit', 'select', 'calibration')]
            self.assertEqual(len(fit), 200)
            self.assertEqual(len(select), 20)
            self.assertEqual(len(calibration), 20)
            self.assertFalse(fit & (select | calibration))
            self.assertFalse(select & calibration)
            self.assertEqual(report['partitions']['fit']['records'], 800)
            self.assertTrue(report['metadata']['all_selected_task_TRAIN_used'])
            self.assertFalse(report['metadata']['formal_test_used'])
            self.assertEqual(report['independent_validation_records'], 0)


if __name__ == '__main__':
    unittest.main()
