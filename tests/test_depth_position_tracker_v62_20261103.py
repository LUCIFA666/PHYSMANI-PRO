"""Pure V62 interfaces, rejected capability and one frozen public F cloud.

No passing V7 artifact is fabricated and no calibration cohort is decoded.
"""
import ast
from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from our_mani.local_mpc_control_20261103 import depth_position_goal_perception_v62 as p
from our_mani.local_mpc_control_20261103 import goal_aware_position_binding_v62 as b
from our_mani.local_mpc_control_20261103 import public_candidate_position_v62 as c

ROOT=Path(__file__).resolve().parents[1]


class Interfaces(unittest.TestCase):
    def test_flags_keep_model_failure_separate(self):
        base=dict(position_measurement_qualified=True,qualified=False,model_qualified=False,raw_mask_model_qualified=False)
        self.assertTrue(p._position_flags(SimpleNamespace(**base)))
        for key in base:
            changed=dict(base);changed[key]=not base[key]
            self.assertFalse(p._position_flags(SimpleNamespace(**changed)))
        self.assertFalse(p._position_flags(SimpleNamespace()))

    def test_no_diagnostic_or_testdouble_constructor(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'not_a_checkpoint';path.write_bytes(b'never torch loaded')
            digest=hashlib.sha256(path.read_bytes()).hexdigest()
            fake=SimpleNamespace(position_measurement_qualified=True,qualified=False,model_qualified=False,
                raw_mask_model_qualified=False,model=object(),metadata={})
            with patch.object(p,'CurrentCubePositionMeasurementV7',return_value=fake) as factory:
                with self.assertRaises(ValueError):p.CurrentDepthPositionTrackerV62(path,digest)
                self.assertEqual(factory.call_args.kwargs,{'allow_unqualified_diagnostic':False})
            with patch.object(p,'CurrentCubePositionMeasurementV7') as factory:
                with self.assertRaises(ValueError):p.CurrentDepthPositionTrackerV62(path,'')
                factory.assert_not_called()

    def test_selected_and_qualified_evidence_roots_do_not_merge(self):
        selected={'selected':{'path':'/selected/selected.pt','sha256':'a'*64},
            'fit_status':{'path':'/selected/fit_status.json','sha256':'b'*64},
            'evidence':{'identity':{'path':'/selected/identity.json','sha256':'c'*64}}}
        self.assertEqual([x['path'] for x in p._evidence_pointers(selected)],
            ['/selected/selected.pt','/selected/fit_status.json','/selected/identity.json'])
        self.assertEqual(list(p._evidence_pointers({'path':'not_pointer','extra':1})),[])
        with tempfile.TemporaryDirectory() as tmp:
            file=Path(tmp)/'evidence';file.write_bytes(b'first')
            before=p._file_stat(file);file.write_bytes(b'other')
            self.assertNotEqual(before,p._file_stat(file))

    def test_real_frozen_F_points_are_the_cloud_without_label_access(self):
        base=ROOT/'artifacts/framework/near_goal_segmentation_20261103/v5_fixed995_position_support_research_v7'
        self.assertEqual(hashlib.sha256((base/'prediction_freeze.json').read_bytes()).hexdigest(),
            '775d66d3066f95dda121d515a46db2907fcfd08a6895cb3757ce55683fae6ba9')
        # Only stored public measurement JSON is decoded, no inputs or labels.
        rows=[json.loads(line) for line in (base/'predictions.jsonl').read_text().splitlines()]
        row=next(r for r in rows if r['shot_id']=='shot02')
        path=Path(row['full_measurement']['path'])
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),row['full_measurement']['sha256'])
        with gzip.open(path,'rt') as file:measurement=json.load(file)
        cloud,views=p.current_position_depth_cloud(measurement)
        np.testing.assert_array_equal(cloud,np.asarray(measurement['current_points_world']))
        self.assertEqual((len(cloud),views),(measurement['measured_points'],measurement['visible_views']))
        changed=deepcopy(measurement);changed['current_points_world'][0][0]+=.001
        with self.assertRaises(ValueError):p.current_position_depth_cloud(changed)
        changed=deepcopy(measurement);changed['camera_point_ledger'][0]['accepted']=[1]*len(changed['camera_point_ledger'][0]['accepted'])
        with self.assertRaises(ValueError):p.current_position_depth_cloud(changed)

    def test_explicit_route_and_exact_binding_only(self):
        for bad in (False,1,'true',None):
            with self.assertRaises(ValueError):c.v62_enabled(goal_sphere_filter=bad)
            with self.assertRaises(ValueError):c.v62_enabled(postconfirmed_maintenance=bad)
        with self.assertRaises(TypeError):c.PublicCandidatePositionStateAssemblerV62(object())
        backend=SimpleNamespace(bridge=SimpleNamespace(goal_sphere_filter=True,postconfirmed_maintenance=True))
        with self.assertRaises(ValueError):c._validate_v62_runner(backend,{})
        c._validate_v62_runner(backend,dict(candidate_position_measurement_v62=True,
            candidate_goal_sphere_filter_v56=True,postconfirmed_position_maintenance_v56=True))
        for relative,expected in b.APPROVED_COMPONENT_SHA256.items():
            self.assertEqual(hashlib.sha256((ROOT/relative).read_bytes()).hexdigest(),expected)


if __name__=='__main__':unittest.main()
