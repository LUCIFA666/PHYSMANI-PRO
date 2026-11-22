import copy
import unittest
from unittest.mock import patch

import numpy as np
from our_mani.risk_v3.model import calibrate
from scripts.recalibrate_candidate_success_r08 import (
    identity, select_correction, nested_crossfit, fit_shrunk_affine)


class NestedCandidateCalibrationR08Tests(unittest.TestCase):
    def observations(self):
        rows = []
        for source in range(12):
            for attempt in range(6):
                rows.append(dict(source_episode=f's{source}', official_task_success=(attempt < 2+(source%2))))
        y = np.array([r['official_task_success'] for r in rows], int)
        # Deliberately biased but finite raw probabilities, entirely external to calibration.
        raw = .57 + .10*np.sin(np.arange(len(rows))*.37)
        grouping = dict(seed=810, source_to_fold={f's{s}': s%2 for s in range(12)})
        return raw, y, rows, grouping

    def test_outer_held_labels_cannot_change_its_selection_or_fit(self):
        raw, y, rows, grouping = self.observations()
        first = nested_crossfit(raw, y, rows, grouping)
        changed = y.copy(); changed_rows = copy.deepcopy(rows)
        for i, row in enumerate(rows):
            if grouping['source_to_fold'][row['source_episode']] == 0:
                changed[i] = 1-y[i]; changed_rows[i]['official_task_success'] = bool(changed[i])
        second = nested_crossfit(raw, changed, changed_rows, grouping)
        self.assertEqual(first['folds'][0]['correction'], second['folds'][0]['correction'])
        self.assertEqual(first['folds'][0]['inner_selection'], second['folds'][0]['inner_selection'])
        self.assertNotEqual(first['folds'][0]['held_positive'], second['folds'][0]['held_positive'])

    def test_every_inner_source_is_inside_its_outer_fit_and_disjoint(self):
        raw, y, rows, grouping = self.observations()
        result = nested_crossfit(raw, y, rows, grouping)
        for outer in result['folds']:
            fit, held = set(outer['fit_sources']), set(outer['held_sources'])
            self.assertFalse(fit & held)
            self.assertEqual(set(outer['inner_selection']['fit_sources']), fit)
            for candidate in outer['inner_selection']['candidates']:
                for inner in candidate.get('folds', []):
                    a, b = set(inner['fit_sources']), set(inner['held_sources'])
                    self.assertFalse(a & b); self.assertEqual(a | b, fit)
                    self.assertFalse((a | b) & held)

    def test_rare_positive_source_uses_identity_without_fitting(self):
        raw, _, rows, _ = self.observations()
        y = np.zeros(len(rows), int); y[0] = 1
        for row, label in zip(rows, y): row['official_task_success'] = bool(label)
        with patch('scripts.recalibrate_candidate_success_r08.fit_shrunk_affine', side_effect=AssertionError('must not fit')):
            correction, trace = select_correction(raw, y, rows, 810)
        self.assertEqual(correction, identity())
        self.assertEqual(trace['source_coverage']['positive_sources'], 1)
        self.assertIsNotNone(trace['fallback_reason'])

    def test_exact_ties_prefer_fixed_identity(self):
        rows = [dict(source_episode=f's{s}', official_task_success=bool(a)) for s in range(8) for a in (0, 1)]
        y = np.array([r['official_task_success'] for r in rows], int)
        correction, trace = select_correction(np.full(len(rows), .5), y, rows, 810)
        self.assertEqual(trace['selected'], 'identity')
        self.assertEqual(correction, identity())

    def test_stronger_regularization_shrinks_both_parameters(self):
        raw = np.full(100, .8); y = np.r_[np.ones(30), np.zeros(70)]
        weak, strong = fit_shrunk_affine(raw, y, .01), fit_shrunk_affine(raw, y, 1.)
        size = lambda c: np.hypot(c['slope']-1., c['intercept'])
        self.assertLess(size(strong), size(weak))
        self.assertGreater(strong['slope'], 0.)
        self.assertTrue(np.isfinite(calibrate(raw, strong)).all())

    def test_fixed_outer_source_assignment_is_required(self):
        raw, y, rows, grouping = self.observations()
        bad = copy.deepcopy(grouping); bad['source_to_fold'].pop('s0')
        with self.assertRaises(ValueError): nested_crossfit(raw, y, rows, bad)


if __name__ == '__main__':
    unittest.main()
