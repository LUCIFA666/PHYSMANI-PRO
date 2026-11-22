"""Stored mask fingerprints and controlled counterevidence are preserved."""
import importlib.util
from pathlib import Path
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('rigid_prior_v6_cpu',ROOT/'scripts/audit_rigid_priors_r10_v6.py')
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)


class RigidPriorAuditChecks(unittest.TestCase):
    def fixture(self):
        arrays=dict(body_ids=np.array(['frame','spoke0']),times=np.arange(6)*.05,
            target_positions=np.zeros((6,2,3)),target_rotations=np.tile(np.eye(3),(6,2,1,1)),
            target_valid=np.ones((6,2),bool))
        arrays['target_valid'][0]=False
        relation=dict(parent='frame',child='spoke0',samples=5,position_error_max_m=0.,orientation_error_max_rad=0.,
            child_to_parent_position=[0.,0.,0.],child_to_parent_rotation=np.eye(3).tolist())
        layout=dict(version='r10_TRAIN_observed_rigid_layout_v1',split='train',shared_with_baseline=True,
            thresholds=dict(minimum_samples=5,position_tolerance_m=.002,orientation_tolerance_rad=.02),
            arrays_sha256=audit.digest({k:v.tolist() for k,v in arrays.items()}),accepted=[relation],rejected=[])
        layout['evidence_sha256']=audit.digest(layout);return layout,arrays

    def test_false_boundary_is_fingerprinted_and_source_remains_usable(self):
        layout,arrays=self.fixture();result=audit.validate_layout(layout,arrays)
        self.assertFalse(result['first_frame_target_valid']['frame'])
        self.assertTrue(result['first_false_boundary_does_not_drop_source'])
        arrays['target_valid'][0]=True
        with self.assertRaisesRegex(ValueError,'fingerprint'):audit.validate_layout(layout,arrays)

    def test_teacher_deformation_cannot_be_hidden_by_short_hold(self):
        relation=dict(parent='button',child='wrap',samples=60,position_error_max_m=1e-7,orientation_error_max_rad=0.)
        accepted,rejected=audit.select_frame_relations({'accepted':[relation]},
            {'rejected':[dict(parent='button',child='wrap',position_error_max_m=.0037)]})
        self.assertEqual(accepted,[]);self.assertEqual(len(rejected),1)
        self.assertEqual(rejected[0]['teacher_counterevidence']['position_error_max_m'],.0037)

    def test_only_frame_relations_survive_independent_teacher_check(self):
        relation=dict(parent='frame',child='spoke0')
        self.assertEqual(audit.select_frame_relations({'accepted':[relation]},{'rejected':[]})[0],[relation])
        accepted,rejected=audit.select_frame_relations({'accepted':[relation]},
            {'rejected':[dict(parent='frame',child='spoke0',position_error_max_m=.003)]})
        self.assertEqual(accepted,[]);self.assertEqual(len(rejected),1)


if __name__=='__main__':unittest.main()
