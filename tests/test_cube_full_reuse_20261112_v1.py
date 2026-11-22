import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from our_mani.cube_full_reuse_20261112_v1.common import plan,parents,physical_plan,sha
from our_mani.cube_full_reuse_20261112_v1.weak import public_windows
from our_mani.cube_full_reuse_20261112_v1.backends import verify_archive_source
from our_mani.cube_motion_residual_20261104_v1.runtime import CubeResidualMotionDynamics

class ReuseBoundaryContracts(unittest.TestCase):
    def test_all200_recorded_sources_have_public_windows(self):
        p=plan();cases=[c for c in p['cases']if c['source_split']=='train']
        self.assertEqual(len(cases),200)
        for c in cases:
            self.assertTrue(public_windows(c));self.assertTrue(public_windows(c,False))
        self.assertEqual([len(parents(p,r))for r in('fit','selection','validation','test')],[160,40,40,80])
    def test_completed_actual_clock_sources_are_separate_and_closed(self):
        p=plan();q=physical_plan(p)
        self.assertEqual(len(q['cases']),19)
        self.assertEqual(len(q['heldout_parent_override']),120)
        self.assertFalse(set(parents(q,'fit'))&set(parents(q,'selection')))
        for c in q['cases']:
            r=json.loads(Path(c['owner_receipt']).read_text())
            self.assertTrue(r['complete']);self.assertTrue(r['actual_wait_completed']);self.assertFalse(r['unreaped_descendants'])
    def test_weak_schema_is_rejected_by_physical_runtime(self):
        with tempfile.TemporaryDirectory()as d:
            p=Path(d)/'weak.pt';torch.save(dict(schema='Cube_recorded_weak_pretraining_only_20261112_v1',model_state={}),p)
            with self.assertRaises(ValueError):CubeResidualMotionDynamics(p,expected_sha256=sha(p))
    def test_TEST_still_requires_full_VAL_release(self):
        with patch('our_mani.cube_full_reuse_20261112_v1.backends.read',return_value={'complete':False})as read:
            with self.assertRaisesRegex(ValueError,'Complete full40'):verify_archive_source({})
            self.assertEqual(read.call_count,1)
    def test_recorded_forward_never_reads_expert_labels_or_truth(self):
        p=plan();c=next(c for c in p['cases']if c['source_split']=='train')
        with patch('h5py.File',side_effect=AssertionError('private HDF labels opened')):
            a=public_windows(c)
        self.assertTrue(a)
        self.assertEqual(set(a[0]),{'positions','timestamps','confidence','lengths','anchor_index','target_positions','target_valid'})
        np.testing.assert_allclose(np.diff(a[0]['timestamps'][:a[0]['lengths']]),.05)

if __name__=='__main__':unittest.main()
