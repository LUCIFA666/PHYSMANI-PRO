import unittest

import numpy as np

from our_mani.success_prediction import feature_vector, group_partitions, outcome_label, SuccessProbability


class SuccessPredictionTests(unittest.TestCase):
    def test_lift_only_failure_is_censored_for_task_success(self):
        row = {'stable_lift': True, 'official_task_success': False, 'full_task_executed': False}
        self.assertEqual(outcome_label(row, 'stable_lift'), 1.)
        with self.assertRaisesRegex(ValueError, 'censored'):
            outcome_label(row, 'official_task_success', dataset_full_task=True)

    def test_success_conjunction_is_not_product_of_independent_predictions(self):
        row = {'stable_lift': True, 'official_task_success': False, 'full_task_executed': True}
        self.assertEqual(outcome_label(row, 'stable_lift_and_official_task_success'), 0.)
        row['official_task_success'] = True
        self.assertEqual(outcome_label(row, 'stable_lift_and_official_task_success'), 1.)

    def test_replicas_never_cross_fit_selection_and_calibration(self):
        rows = [{'task': task, 'source_episode': f'episode{ep}', 'replica': replica}
                for task in ('normal', 'fast') for ep in range(10) for replica in range(3)]
        parts = group_partitions(rows)
        self.assertFalse(parts['fit'] & parts['select'])
        self.assertFalse(parts['fit'] & parts['calibration'])
        self.assertFalse(parts['select'] & parts['calibration'])
        self.assertEqual(len(set.union(*parts.values())), 20)

    def test_future_context_and_missing_context_are_rejected(self):
        row = {'initial_features': [0.]*18, 'decision_timestamp': 1.,
               'dynamics_context': {'values': [0.5]*16, 'encoder_sha256': 'encoder-a',
                                    'source': 'past_observations', 'observed_until': 1.1}}
        with self.assertRaisesRegex(ValueError, 'future'):
            feature_vector(row, 16)
        row['dynamics_context']['observed_until'] = .95
        self.assertEqual(feature_vector(row, 16).shape, (34,))
        del row['dynamics_context']
        with self.assertRaisesRegex(ValueError, 'missing'):
            feature_vector(row, 16)

    def test_inference_requires_correct_controller_and_context_encoder(self):
        import torch
        from our_mani.local_learning import MLP
        torch.manual_seed(0)
        checkpoint = {'state_dict': MLP(34, 1, 8).state_dict(), 'hidden': 8,
            'feature_mean': np.zeros(34), 'feature_std': np.ones(34), 'temperature': 1.,
            'metadata': {'controller_sha256': 'controller-a', 'context_dim': 16,
                         'context_encoder_sha256': 'encoder-a'}}
        with self.assertRaisesRegex(ValueError, 'controller'):
            SuccessProbability(checkpoint, controller_sha256='controller-b')
        predictor = SuccessProbability(checkpoint, controller_sha256='controller-a')
        with self.assertRaisesRegex(ValueError, 'required'):
            predictor(np.zeros(18))
        with self.assertRaisesRegex(ValueError, 'encoder'):
            predictor(np.zeros(18), context=np.zeros(16), context_encoder_sha256='encoder-b')
        p = predictor(np.zeros((2, 18)), context=np.zeros((2, 16)), context_encoder_sha256='encoder-a')
        self.assertEqual(p.shape, (2,))
        self.assertTrue(((p >= 0.) & (p <= 1.)).all())


if __name__ == '__main__':
    unittest.main()
