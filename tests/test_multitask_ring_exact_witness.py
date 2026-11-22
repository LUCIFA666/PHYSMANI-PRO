"""Exact measured TCP orbit hashes do not depend on SciPy quaternion output."""
from copy import deepcopy
from fractions import Fraction
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np

from our_mani.multitask.evaluation import digest,file_sha
from our_mani.multitask.ring_affordance import exact_measured_action_orbit,VerifiedRingAffordance,EXACT_ORBIT_SCHEMA
from tests.test_multitask_rotation_equivalence import D4

ROOT=Path(__file__).resolve().parents[1]


class ExactOrbitArithmeticChecks(unittest.TestCase):
    def sample(self):return (np.array([.1,.2,.3]),np.eye(3),np.array([.4,.5,.6,0.,0.,1.,1.]))
    def test_identity_action_uses_exact_raw_difference_and_normalized_quaternion_products(self):
        sample=self.sample();orbit=exact_measured_action_orbit([sample],D4(),np.zeros(3));self.assertEqual(len(orbit),8)
        self.assertEqual(orbit[0]['curve_hash_schema'],EXACT_ORBIT_SCHEMA)
        pose=orbit[0]['start_relative_TCP_exact'];difference=Fraction.from_float(.4)-Fraction.from_float(.1)
        self.assertEqual(pose['position_m'][0],[difference.numerator,difference.denominator])
        self.assertEqual(pose['rotation'],[[[0,1],[-1,1],[0,1]],[[1,1],[0,1],[0,1]],[[0,1],[0,1],[1,1]]])
        self.assertEqual(orbit[0]['start_relative_TCP_display']['rotation'],[[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
    def test_quaternion_sign_and_binary_scaling_have_same_physical_curve_hash_but_distinct_raw_source_hash(self):
        sample=self.sample();base=exact_measured_action_orbit([sample],D4(),np.zeros(3))
        for scale in (-1.,2.):
            tcp=sample[2].copy();tcp[3:]*=scale
            changed=exact_measured_action_orbit([(sample[0],sample[1],tcp)],D4(),np.zeros(3))
            self.assertEqual([o['curve_sha256'] for o in base],[o['curve_sha256'] for o in changed])
            self.assertNotEqual(base[0]['source_raw_samples_sha256'],changed[0]['source_raw_samples_sha256'])
    def test_one_ulp_source_change_changes_exact_witness_and_invalid_input_is_rejected(self):
        sample=self.sample();base=exact_measured_action_orbit([sample],D4(),np.zeros(3))
        position=sample[0].copy();position[0]=np.nextafter(position[0],np.inf)
        changed=exact_measured_action_orbit([(position,sample[1],sample[2])],D4(),np.zeros(3))
        self.assertNotEqual(base[0]['curve_sha256'],changed[0]['curve_sha256'])
        tcp=sample[2].copy();tcp[3:]=0.
        with self.assertRaises(ValueError):exact_measured_action_orbit([(sample[0],sample[1],tcp)],D4(),np.zeros(3))
        with self.assertRaises(ValueError):exact_measured_action_orbit([(sample[0],sample[1]*2,sample[2])],D4(),np.zeros(3))


class ActualFrozenPermissionTamperChecks(unittest.TestCase):
    def test_actual_exact_proof_rejects_resigned_orbit_hash_rational_endpoint_source_and_authorization_tampering(self):
        report=json.loads((ROOT/'artifacts/framework/multitask_r10/ring_affordance_runtime_v4/report.json').read_text())
        pointer=report['entries'][0]['permission'];baseline=json.loads((ROOT/pointer['path']).read_text())
        verified=VerifiedRingAffordance.load(ROOT,pointer)
        self.assertFalse(verified.permission['policy_skill_authorized'])
        self.assertEqual(verified.permission['phase_support']['acquire']['completed_parent_count'],1)
        with tempfile.TemporaryDirectory(dir=ROOT/'runtime/framework/tmp') as directory:
            path=Path(directory)/'changed.json'
            for variant in ('rational_endpoint','curve_hash','raw_source_SHA','display_value','authorization'):
                value=deepcopy(baseline)
                orbit=value['actual_TRAIN_grasp_rows'][0]['phases']['acquire']['transformed_action_orbit'][0]
                if variant=='rational_endpoint':orbit['start_relative_TCP_exact']['position_m'][0][0]+=1
                elif variant=='curve_hash':orbit['curve_sha256']='0'*64
                elif variant=='display_value':orbit['start_relative_TCP_display']['position_m'][0]+=1e-12
                elif variant=='raw_source_SHA':
                    key=next(k for k in value['inputs_sha256'] if k.endswith('body_tracks.npz'));value['inputs_sha256'][key]='f'*64
                else:value['policy_skill_authorized']=True
                value['permission_sha256']=digest({k:v for k,v in value.items() if k!='permission_sha256'})
                path.write_text(json.dumps(value))
                with self.assertRaises(ValueError,msg=variant):
                    VerifiedRingAffordance.load(ROOT,dict(path=str(path.relative_to(ROOT)),sha256=file_sha(path)))


if __name__=='__main__':unittest.main()
