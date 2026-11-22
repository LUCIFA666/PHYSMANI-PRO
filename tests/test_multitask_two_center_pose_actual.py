"""Readonly actual TRAIN model loader/consumer boundary tests, no simulator."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np

from our_mani.multitask import two_center_pose as pose

ROOT=Path(__file__).resolve().parents[1]
ARTIFACT=ROOT/'artifacts/framework/multitask_r10/buzz_two_center_TRAIN_models_and_teacher_scope_v3/report.json'


class ActualTwoCenterModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not ARTIFACT.is_file():raise unittest.SkipTest('Actual TRAIN-only frozen model evidence is not installed')
        cls.report=json.loads(ARTIFACT.read_text())

    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='two_center_actual_CPU_',dir=ROOT/'artifacts/framework/multitask_r10')
        self.addCleanup(self.temp.cleanup);self.directory=Path(self.temp.name)
        self.task='beat_the_rotating_buzz';self.pointer=self.report['models'][self.task]
        self.model=json.loads((ROOT/self.pointer['path']).read_text())
        self.cad=self.model['canonical_basis']['TRAIN_CAD']

    def write(self,name,value):
        path=self.directory/(name+'.json');path.write_text(json.dumps(value))
        return dict(path=str(path.relative_to(ROOT)),sha256=hashlib.sha256(path.read_bytes()).hexdigest())

    def load(self,value=None):
        pointer=self.pointer if value is None else self.write('model',value)
        return pose.load_two_center_pose_prior(ROOT,pointer,canonical_asset_pointer=self.cad)

    def test_actual_models_load_same_frozen_CAD_but_grant_no_orientation_permission(self):
        for task,pointer in self.report['models'].items():
            model=json.loads((ROOT/pointer['path']).read_text())
            prior=pose.load_two_center_pose_prior(ROOT,pointer,canonical_asset_pointer=model['canonical_basis']['TRAIN_CAD'])
            self.assertEqual(prior.task,task);self.assertEqual(prior.model_sha256,pointer['sha256'])
            self.assertFalse(model['orientation_permission_granted'])
            self.assertFalse(self.report['public_none_orientation_permission_changed'])

    def test_wrong_CAD_helper_or_bias_scope_role_and_extra_parameters_rejected(self):
        changes=[lambda m:m['configuration'].update(public_centroid_bias_correction=True),
            lambda m:m.update(runtime_scope='unrestricted_full_pose'),
            lambda m:m.update(orientation_permission_granted=True),
            lambda m:m['roles'].update(wand='other_body'),
            lambda m:m['canonical_basis']['role_native_roots'].update(frame='oracle_waypoint'),
            lambda m:m['helper_source'].update(sha256='0'*64),
            lambda m:m['static_prior'].update(native_yaw_parameter=0.)]
        for index,change in enumerate(changes):
            model=deepcopy(self.model);change(model)
            with self.subTest(change=index),self.assertRaises(ValueError):self.load(model)

    def test_duplicate_actual_TRAIN_cases_and_parents_are_rejected(self):
        receipt=json.loads((ROOT/self.model['TRAIN_evidence']['training_model_receipt']['path']).read_text())
        for key in ('training_parent_source_groups','actual_case_provenance'):
            changed=deepcopy(receipt);changed[key][1]=deepcopy(changed[key][0])
            model=deepcopy(self.model);model['TRAIN_evidence']['training_model_receipt']=self.write('receipt_'+key,changed)
            with self.subTest(key=key),self.assertRaises(ValueError):self.load(model)

    def test_crossfit_overlap_or_outlier_removal_is_rejected(self):
        cv=json.loads((ROOT/self.model['TRAIN_evidence']['parent_crossvalidation']['path']).read_text())
        changed=deepcopy(cv);fold=changed['tasks'][self.task]['folds']['0'];fold['held_parents'][0]=fold['prior']['training_parents'][0]
        pointer=self.write('crossvalidation_overlap',changed)
        model=deepcopy(self.model)
        receipt=json.loads((ROOT/model['TRAIN_evidence']['training_model_receipt']['path']).read_text())
        tails=json.loads((ROOT/model['TRAIN_evidence']['tails_and_canonical_basis']['path']).read_text())
        model['TRAIN_evidence']['parent_crossvalidation']=pointer;receipt['parent_crossvalidation']=pointer;tails['parent_crossvalidation']=pointer
        tails_pointer=self.write('tails_overlap',tails);model['TRAIN_evidence']['tails_and_canonical_basis']=tails_pointer;receipt['tails_and_canonical_basis']=tails_pointer
        model['TRAIN_evidence']['training_model_receipt']=self.write('receipt_overlap',receipt)
        with self.assertRaises(ValueError):self.load(model)
        model=deepcopy(self.model);tails=json.loads((ROOT/model['TRAIN_evidence']['tails_and_canonical_basis']['path']).read_text());tails['outliers_removed']=True
        model['TRAIN_evidence']['tails_and_canonical_basis']=self.write('tails_outlier_removal',tails)
        with self.assertRaises(ValueError):self.load(model)

    def test_real_public_cache_positions_only_and_next_invalid_observation_clear_R(self):
        prior=self.load()
        case=next(c for c in self.report['actual_case_provenance'] if c['task']==self.task and c['behavior']=='TRAIN_hold_motion_identification')
        cache=ROOT/case['cache']['path'];self.assertEqual(hashlib.sha256(cache.read_bytes()).hexdigest(),case['cache']['sha256'])
        # Read only public keys, never target poses, into this consumer test.
        with np.load(cache,allow_pickle=False) as arrays:
            ids=list(map(str,arrays['body_ids']));timestamp=float(arrays['times'][0])
            centers={name:pose.PublicCenter(arrays['positions'][0,ids.index(name)],bool(arrays['valid'][0,ids.index(name)]),
                bool(arrays['mode_valid'][0,ids.index(name)]),'external' if arrays['modes'][0,ids.index(name)]==0 else 'unknown',timestamp)
                for name in ('frame','wand')}
        first=pose.estimate_two_center_pose(prior,centers,current_timestamp=timestamp)
        self.assertTrue(first.valid);self.assertFalse(first.grants_runtime_full_orientation)
        centers['wand']=pose.PublicCenter(centers['wand'].position,False,True,'external',timestamp)
        failed=pose.estimate_two_center_pose(prior,centers,current_timestamp=timestamp)
        self.assertFalse(failed.valid);self.assertIsNone(failed.frame_rotation);self.assertIsNone(failed.wand_rotation)


if __name__=='__main__':unittest.main()
