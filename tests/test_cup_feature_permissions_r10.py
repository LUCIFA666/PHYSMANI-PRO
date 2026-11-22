"""Production model permission boundary and frozen primitive AST parity."""
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import unittest

from our_mani.multitask.cup_features import (ROOT,FeatureConfig,load_frozen_cup_features,
    _json_sha256,validate_frozen_cup_cad,_same_feature_receipt)
from scripts.build_cup_feature_permissions_r10 import mathematical_ast_parity


class CupProductionParityTests(unittest.TestCase):
    def test_every_original_math_function_and_class_has_identical_AST(self):
        result=mathematical_ast_parity(ROOT/'scripts/experimental_cup_features_r10.py',ROOT/'our_mani/multitask/cup_features.py')
        self.assertTrue(result['all_original_functions_and_classes_AST_identical'])
        self.assertIn('estimate_cup_feature_pose',result['mathematical_function_and_class_AST_sha256'])
        self.assertIn('FeatureConfig',result['mathematical_function_and_class_AST_sha256'])

    def test_permission_never_enables_itself_or_changes_fixed_parameters(self):
        for permission in ({},dict(enabled=False),dict(enabled=True,roles=['cup0'],configuration=asdict(FeatureConfig())),
                dict(enabled=True,roles=['cup0','cup1','cup2'],configuration={**asdict(FeatureConfig()),'free_space_weight':3.})):
            with self.assertRaises(ValueError):load_frozen_cup_features(permission,{})

    def test_model_reconstruction_roundoff_tolerance_is_tiny_and_does_not_erase_changes(self):
        value=dict(role='cup0',half_height_m=.04,cylinder_centre_canonical=[0.,-.01585,0.],
            local_vertex_correspondence_max_error_m=1e-7,canonical_outside_handle_points=21)
        benign=deepcopy(value);benign['half_height_m']+=1e-14
        self.assertTrue(_same_feature_receipt(value,benign))
        changed=deepcopy(value);changed['half_height_m']+=1e-6
        self.assertFalse(_same_feature_receipt(value,changed))
        changed=deepcopy(value);changed['canonical_outside_handle_points']+=1
        self.assertFalse(_same_feature_receipt(value,changed))


class ActualFrozenCupPermissionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path=ROOT/'artifacts/framework/multitask_r10/cup_feature_permissions_v1/cup_feature_priors.json'
        if not path.exists():raise unittest.SkipTest('build frozen cup feature permissions first')
        cls.permissions=json.loads(path.read_text());cls.task='place_cups_on_rotating_frame'
        cls.permission=cls.permissions[cls.task]
        cls.asset=json.loads((ROOT/cls.permission['TRAIN_CAD']['path']).read_text())

    def test_all_four_same_task_TRAIN_models_pass_strict_loader(self):
        for task,permission in self.permissions.items():
            asset=json.loads((ROOT/permission['TRAIN_CAD']['path']).read_text())
            models=load_frozen_cup_features(permission,asset)
            self.assertEqual(set(models),{'cup0','cup1','cup2'})
            self.assertTrue(all(model.receipt['GT_or_world_pose_used'] is False for model in models.values()))

    def test_changed_permission_CAD_role_configuration_and_cross_task_source_rejected(self):
        mutations=('file_hash','canonical_pointer','roles','configuration','task','evidence_pointer')
        for mutation in mutations:
            permission=deepcopy(self.permission);asset=deepcopy(self.asset)
            if mutation=='file_hash':permission['sha256']='0'*64
            elif mutation=='canonical_pointer':permission['canonical_CAD']['sha256']='0'*64
            elif mutation=='roles':permission['roles'].append('frame')
            elif mutation=='configuration':permission['configuration']['minimum_handle_points']=1
            elif mutation=='evidence_pointer':permission['evidence_audit']['sha256']='0'*64
            else:
                asset['task']='put_rubbish_in_moving_bin'
                asset['asset_sha256']=_json_sha256({k:v for k,v in asset.items() if k!='asset_sha256'})
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):load_frozen_cup_features(permission,asset)

    def test_current_CAD_mutation_does_not_migrate_or_borrow_static_permission(self):
        asset=deepcopy(self.asset);asset['bodies']['cup0']['collision_vertices_body'][0][0]+=1e-8
        asset['asset_sha256']=_json_sha256({k:v for k,v in asset.items() if k!='asset_sha256'})
        with self.assertRaises(ValueError):load_frozen_cup_features(self.permission,asset)
        asset=deepcopy(self.asset);asset['split']='validation'
        asset['asset_sha256']=_json_sha256({k:v for k,v in asset.items() if k!='asset_sha256'})
        with self.assertRaises(ValueError):validate_frozen_cup_cad(asset,task=self.task)


if __name__=='__main__':unittest.main()
