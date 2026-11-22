import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from our_mani.multitask.colour_only_candidate import (
    SCHEMA, bound, confusion_and_parent_results, load_visual_patch, pointer, verify_native_colour_cases)
from our_mani.multitask.public_roles import PublicRoleResolver, PublicRoleResolutionError


ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'artifacts/framework/multitask_r10'


class ColourOnlyCandidateTests(unittest.TestCase):
    def test_exact_visual_patch_has_no_native_role_authority(self):
        sample,annotation=load_visual_patch(ROOT,
            BASE/'TRAIN_colour_visual_patch_annotation_candidate_v1/annotation.json',[],
            BASE/'TRAIN_colour_visual_patch_independent_peer_v1/report.json')
        self.assertEqual(sample.rgb_pixels_255.shape,(80,3))
        self.assertEqual(sample.colour_name,'orange')
        self.assertEqual(sample.component_names,('__no_runtime_component_authority__',))
        with self.assertRaisesRegex(ValueError,'independent'):
            load_visual_patch(ROOT,BASE/'TRAIN_colour_visual_patch_annotation_candidate_v1/annotation.json',
                [sample.source_group],BASE/'TRAIN_colour_visual_patch_independent_peer_v1/report.json')

    def test_candidate_schema_is_not_a_production_resolver(self):
        with self.assertRaises(PublicRoleResolutionError):
            PublicRoleResolver(None,dict(metadata=dict(schema=SCHEMA)),shared_semantic_permission=True)

    def test_bound_sha_and_root_are_strict(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'runtime/framework') as name:
            root=Path(name).resolve();p=root/'one.json';p.write_text('{}')
            ref=pointer(root,p);self.assertEqual(bound(root,ref),p)
            p.write_text('{"extra":true}')
            with self.assertRaises(ValueError): bound(root,ref)
            with self.assertRaises(ValueError): bound(root,dict(path='/etc/hostname',sha256='0'*64))

    def test_native_incomplete_or_TEST_proof_rejected(self):
        proof=json.loads((BASE/'TRAIN_primary3000_counted_highcup100_bundle_v1/bundle_audit.json').read_text())
        with tempfile.TemporaryDirectory(dir=ROOT/'runtime/framework') as name:
            p=Path(name)/'proof.json'
            for change in ('incomplete','TEST'):
                altered=copy.deepcopy(proof)
                if change=='incomplete': altered['complete']=False
                else: altered['acquisition_projection']['TEST_used']=True
                p.write_text(json.dumps(altered))
                with self.assertRaises(ValueError): verify_native_colour_cases(ROOT,p)

    def test_parent_macro_does_not_weight_replicas(self):
        class S:
            def __init__(self,g):self.source_group=g
        rows=[dict(source_group='a',true_public_colour='orange',predicted_colour='orange',
                   correct_and_supported=True)]*9+[dict(source_group='b',true_public_colour='orange',
                   predicted_colour='red',correct_and_supported=False)]
        result,confusion=confusion_and_parent_results(dict(held_predictions=rows,parent_to_fold={'a':0,'b':1}),[S('a'),S('b')])
        self.assertEqual([r['correct_and_supported_fraction'] for r in result],[1.,0.])
        self.assertEqual(confusion['orange'],{'orange':9,'red':1})


if __name__=='__main__':unittest.main()
