"""CPU synthetic critic checks; no native scenes or real training data."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from our_mani.multitask.candidate_success import (CandidateSuccessModel,
    CONTINUATION_POLICY,ContinuationAttempts,FEATURE_NAMES,FEATURE_SCHEMA_SHA256,LABEL_SCOPE)
from our_mani.multitask.candidate_training import (CriticFitConfig,MANIFEST_SCHEMA,
    _calibrate,fit_phase_head,load_executed_manifest,nested_calibration,
    parent_weights,select_calibration,sha)


def rows(split,count):
    result=[]
    for index in range(count):
        label=index % 2
        for replica in range(2):
            features=np.zeros(len(FEATURE_NAMES));features[0]=1. if label else -1.
            result.append(dict(x=features,y=label,group=f'{split}:push_moving_button:episode{index}',
                               phase='approach'))
    return result


class CriticTrainingChecks(unittest.TestCase):
    def test_bootstrap_covers_every_parent_and_VAL_cannot_fit_normalisation(self):
        train,validation=rows('train',8),rows('validation',8)
        config=CriticFitConfig(regularization_grid=(.1,),TRAIN_folds=2)
        head,report=fit_phase_head(train,validation,config)
        changed=deepcopy(validation)
        for row in changed:
            row['x'][0] *= 1000.
        other,other_report=fit_phase_head(train,changed,config)
        np.testing.assert_array_equal(head['TRAIN_mean'],other['TRAIN_mean'])
        np.testing.assert_array_equal(head['TRAIN_scale'],other['TRAIN_scale'])
        np.testing.assert_array_equal(head['ensemble_weights'],other['ensemble_weights'])
        for member in report['bootstrap']:
            self.assertEqual(len(member['parent_source_weights']),8)
            self.assertTrue(all(weight > 0 for weight in member['parent_source_weights'].values()))
        groups=np.array(['one']*100+['two'])
        weights=parent_weights(groups)
        self.assertAlmostEqual(weights[:100].sum(),weights[-1])
        self.assertTrue(report['TRAIN_ranking_guard_passed'])

    def test_outer_held_sources_never_enter_inner_calibration_selection(self):
        y=np.array([0,1]*8);p=np.where(y,.7,.3);groups=np.array(['val'+str(i) for i in range(16)])
        correction,report=nested_calibration(p,y,groups)
        self.assertTrue(report['valid'])
        for outer in report['folds']:
            self.assertFalse(set(outer['fit_sources']) & set(outer['held_sources']))
            self.assertEqual(set(outer['inner_selection']['fit_sources']),set(outer['fit_sources']))
            fit=np.isin(groups,outer['fit_sources'])
            independently_fit,trace=select_calibration(p[fit],y[fit],groups[fit])
            self.assertEqual(independently_fit,outer['correction'])
        rare_correction,rare_report=nested_calibration(np.full(4,.05),np.array([0,0,0,1]),np.array(['a','b','c','d']))
        self.assertEqual(rare_correction['method'],'identity')
        self.assertTrue(all(outer['correction']['method'] == 'identity' for outer in rare_report['folds']))

    def test_native_task_labels_not_skill_labels_and_TEST_manifest_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            directory=Path(directory)
            ledger=ContinuationAttempts(task_name='push_moving_button',predictor='cv',
                continuation_sha256='controller',library_sha256='library')
            ledger.begin('approach','observed_candidate',0.,np.zeros(len(FEATURE_NAMES)))
            ledger.skill_completed('approach',.1)
            attempts=ledger.finish(False,.2,'native_budget_failure')
            case=dict(task='push_moving_button',predictor='cv',split='train',raw_episode_id='episode0__aug',
                parent_raw_episode_id='episode0',complete=True,official_task_success=False,
                continuation_sha256='controller',interaction_library_sha256='library',
                continuation_selection_policy=CONTINUATION_POLICY,downstream_selection_policy='geometry',
                first_selection_policy='TRAIN_explore_once',unexecuted_candidates_labelled=False,
                continuation_attempts=attempts)
            case_path=directory/'case.json';case_path.write_text(json.dumps(case))
            entry=dict(task='push_moving_button',predictor='cv',split='train',raw_episode_id='episode0__aug',
                parent_raw_episode_id='episode0',path=str(case_path),sha256=sha(case_path))
            manifest=directory/'manifest.json';manifest.write_text(json.dumps(dict(schema=MANIFEST_SCHEMA,entries=[entry])))
            data,subset,audit=load_executed_manifest(manifest,split='train',task_name='push_moving_button',
                predictor='cv',continuation_sha256='controller',library_sha256='library')
            self.assertEqual(data[0]['y'],0)
            self.assertEqual(data[0]['group'],'train:push_moving_button:episode0')
            self.assertTrue(attempts[0]['skill_success'])
            manifest.write_text(json.dumps(dict(schema=MANIFEST_SCHEMA,entries=[dict(entry,split='test',path='never-read.json')])))
            with self.assertRaises(ValueError):
                load_executed_manifest(manifest,split='train',task_name='push_moving_button',
                    predictor='cv',continuation_sha256='controller',library_sha256='library')

    def test_saved_reference_model_matches_ensemble_and_constant_class_disables_ranking(self):
        train,validation=rows('train',8),rows('validation',8)
        head,report=fit_phase_head(train,validation,CriticFitConfig(regularization_grid=(.1,),TRAIN_folds=2))
        metadata=dict(training_complete=True,classifier_fit_split='train',classifier_selection_split='train',
            normalization_fit_split='train',TEST_used_for_fitting=False,label_scope=LABEL_SCOPE,
            continuation_selection_policy=CONTINUATION_POLICY,feature_schema_sha256=FEATURE_SCHEMA_SHA256,
            continuation_sha256='controller',interaction_library_sha256='library',
            TRAIN_source_manifest_sha256='synthetic-fixture-only',synthetic=True)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'value.json';path.write_text(json.dumps(dict(metadata=metadata,
                heads={'push_moving_button|cv|approach':head})))
            model=CandidateSuccessModel(path,continuation_sha256='controller',library_sha256='library',allow_synthetic=True)
            prediction=model.predict(np.stack([row['x'] for row in validation]),
                task_name='push_moving_button',predictor='cv',phase='approach')
            self.assertTrue(np.isfinite(prediction['probability']).all())
            self.assertTrue((prediction['ensemble_std'] >= 0).all())
        for row in train:
            row['y']=0
        constant,constant_report=fit_phase_head(train,validation,CriticFitConfig(regularization_grid=(.1,),TRAIN_folds=2))
        self.assertFalse(constant['ranking_enabled'])
        self.assertFalse(constant_report['TRAIN_ranking_guard_passed'])


if __name__ == '__main__':
    unittest.main()
