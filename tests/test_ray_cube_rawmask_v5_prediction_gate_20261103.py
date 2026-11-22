"""Pure public-row gate checks before this experiment ever reads labels."""
import importlib.util
from copy import deepcopy
from pathlib import Path
import unittest
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('rawmask_v5_experiment',ROOT/'artifacts/framework/local_mpc_20261102/ray_cube_current_depth_pilot_v5_rawmask_before_shell_v1/run_public_then_posthoc.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def rows():
    return [dict(case=name,timestamp=at,seed_actual_timestamp=at,seed_age_seconds=0.,GT_read=False,
        TCP_GRU_or_controlled_prediction_inputs_used=False,
        variants={v:dict(available=False,centre=None,mode='UNKNOWN') for v in m.VARIANTS}) for name,times in m.PLAN for at in times]

class PublicGateTests(unittest.TestCase):
    def test_all16_and_both_fixed_variants_required_no_UNKNOWN_promotion(self):
        rr=rows();m.validate_rows(rr)
        with self.assertRaisesRegex(ValueError,'entire ordered'):m.validate_rows(rr[:-1])
        bad=deepcopy(rr);bad[-1]['variants'].pop('v4_erode_after_shell')
        with self.assertRaisesRegex(ValueError,'declared public variants'):m.validate_rows(bad)
        bad=deepcopy(rr);bad[0]['variants']['v5_erode_raw_before_shell']['centre']=[0.,0.,1.]
        with self.assertRaisesRegex(ValueError,'UNKNOWN diagnostic'):m.validate_rows(bad)
    def test_seed_age_no_future_clock_and_no_robot_prediction_channel(self):
        bad=rows();bad[0]['seed_actual_timestamp']+=.05;bad[0]['seed_age_seconds']=-.05
        with self.assertRaisesRegex(ValueError,'causal initializer'):m.validate_rows(bad)
        bad=rows();bad[0]['TCP_GRU_or_controlled_prediction_inputs_used']=True
        with self.assertRaisesRegex(ValueError,'declared public variants'):m.validate_rows(bad)
        with self.assertRaises(ValueError):m.index(True)
        with self.assertRaises(ValueError):m.index(1.01)

if __name__=='__main__':unittest.main()
