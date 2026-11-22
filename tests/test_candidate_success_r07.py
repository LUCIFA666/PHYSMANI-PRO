import hashlib
import json
from pathlib import Path
import pickle
import tempfile
import unittest

import numpy as np
from sklearn.dummy import DummyClassifier
from our_mani.risk_v3.model import (CandidateProbability, FORMAT, FEATURE_SCHEMA_SHA256,
    TASKS, transform_features, calibrate, build_task_support)
from scripts.train_candidate_success_r07 import (balanced_group_folds,
    bayesian_source_weights, crossfit_calibration, make_model, fit_model)


class CandidateSuccessR07Tests(unittest.TestCase):
    def data(self, dimensions=71):
        x = np.random.default_rng(810).normal(0., .1, (20, dimensions))
        x[:, 61] = .5; x[:, 62] = 8.; x[:, 63] = 7.5
        return x

    def test_derived_features_preserve_all_context_coordinates(self):
        x = self.data(87); transformed = transform_features(x, 87)
        self.assertEqual(transformed.shape, (20, 106))
        np.testing.assert_array_equal(transformed[:, :87], x)
        self.assertTrue(np.isfinite(transformed).all())
        with self.assertRaises(ValueError): transform_features(x, 71)

    def test_sources_never_cross_folds_and_bootstrap_never_drops_one(self):
        rows = [dict(source_episode=str(i//2), official_task_success=bool(i%2)) for i in range(20)]
        folds, report = balanced_group_folds(rows, 5)
        for source in report['source_to_fold']:
            self.assertEqual(len(set(folds[[r['source_episode']==source for r in rows]])), 1)
        weights, audit = bayesian_source_weights(rows, set(report['source_to_fold']), 810)
        self.assertTrue((weights > 0).all())
        self.assertTrue(audit['all_source_weights_strictly_positive'])
        for source in report['source_to_fold']:
            self.assertEqual(len(set(weights[[r['source_episode']==source for r in rows]])), 1)

    def test_insufficient_positive_sources_cannot_pass_crossfit(self):
        rows = [dict(source_episode=str(i//2), official_task_success=(i==0)) for i in range(20)]
        y = np.array([r['official_task_success'] for r in rows], int)
        result = crossfit_calibration(np.full(20, .1), y, rows, 810)
        self.assertFalse(result['valid']); self.assertIsNone(result['metrics'])
        self.assertTrue(any('fewer than two positive' in reason for reason in result['reasons']))

    def test_native_model_interfaces_accept_positive_weights(self):
        x = transform_features(self.data()); y = np.arange(20)%2
        for candidate in (dict(kind='logreg', C=.1), dict(kind='hist_gradient_boosting', max_leaf_nodes=7)):
            model = fit_model(make_model(candidate, 810), candidate, x, y, np.ones(20))
            self.assertEqual(model.predict_proba(x).shape, (20, 2))

    def test_artifact_load_calibrated_spread_and_task_support(self):
        x = self.data(); transformed = transform_features(x)
        a = DummyClassifier(strategy='prior').fit(transformed[:4], [0, 0, 0, 1])
        b = DummyClassifier(strategy='prior').fit(transformed[:4], [0, 1, 1, 1])
        correction = dict(slope=.5, intercept=-1.)
        heads = {task: dict(models=[a, a, b, b, b], calibration=correction) for task in TASKS}
        metadata = dict(input_dim=71, feature_schema_sha256=FEATURE_SCHEMA_SHA256,
            controller_sha256='continuation', selection_sha256='selection',
            per_task={task: dict(ranking_enabled=(task==TASKS[0])) for task in TASKS})
        sources = [str(i//2) for i in range(20)]
        supports = {TASKS[0]: build_task_support(x, sources, 71),
            TASKS[1]: build_task_support(x+100., sources, 71)}
        self.assertGreater(supports[TASKS[0]]['different_source_distance_quantiles']['0.5'], 0.)
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=root/'runtime/framework/tmp') as temporary:
            path = Path(temporary)/'value.pt'
            with path.open('wb') as stream: pickle.dump(dict(format=FORMAT, metadata=metadata, tasks=heads), stream)
            support = dict(value_checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                feature_schema_sha256=FEATURE_SCHEMA_SHA256, selection_sha256='selection', tasks=supports)
            path.with_name('value_support.json').write_text(json.dumps(support))
            predictor = CandidateProbability(path, controller_sha256='continuation')
            result = predictor.predict(x[:2], TASKS[0])
            np.testing.assert_allclose(result['probability'], calibrate(np.full(2, .55), correction))
            expected_std = calibrate(np.array([.25, .25, .75, .75, .75]), correction).std()
            np.testing.assert_allclose(result['ensemble_std'], expected_std)
            self.assertTrue(result['supported'].all()); self.assertTrue(result['ranking_enabled'])
            self.assertFalse(predictor.predict(x[:2], TASKS[1])['supported'].any())
            with self.assertRaises(ValueError): CandidateProbability(path, controller_sha256='wrong')


if __name__ == '__main__':
    unittest.main()
