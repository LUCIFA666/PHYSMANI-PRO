"""First-case vs retry denominators stay distinct; no recorded labels opened."""
import importlib.util
from copy import deepcopy
from pathlib import Path
import unittest
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('v59_accuracy_summary',ROOT/'artifacts/framework/local_mpc_20261102/P4_v59_actual_selected_point_accuracy_tool_v1/summarize_cases.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def report():
    names=m.base.COHORTS['pilot4'];attempts=[];cases=[]
    for name in names:
        attempts.append(dict(case=name,attempt_index=0,A_available=True,A=dict(error_l2_m=.02),
            A_dispatch_boundary_legacy_available=True,A_dispatch_boundary_legacy=dict(error_l2_m=.025)))
        cases.append(dict(case=name,expected_actual_close_attempts=1))
    retry=deepcopy(attempts[-1]);retry['attempt_index']=1;retry['A']['error_l2_m']=.001
    attempts.append(retry);cases[-1]['expected_actual_close_attempts']=2
    return dict(schema='V59_posthoc_actual_50ms_close_edge_selected_reference_alignment_v1',
        cohort='pilot4',attempts=attempts,cases=cases)

class SummaryTests(unittest.TestCase):
    def test_retry_not_used_to_improve_primary_case_average_or_create_full8(self):
        s=m.summarize(report())
        self.assertEqual(s['primary_first_per_case_A']['expected_samples'],4)
        self.assertEqual(s['primary_first_per_case_A']['mean_m'],.02)
        self.assertLess(s['all_attempt_statistics']['mean_m'],.02)
        self.assertEqual(len(s['extra_attempts']),1);self.assertTrue(s['all_attempts_preserved'])
        self.assertFalse(s['full8_first_edge_alignment_numerical_requirement_met']);self.assertFalse(s['admission_granted'])
    def test_missing_case_edge_fails_closed_and_filtered_retry_rejected(self):
        r=report();r['attempts']=r['attempts'][1:];r['cases'][0]['expected_actual_close_attempts']=0
        s=m.summarize(r);self.assertFalse(s['primary_first_per_case_A']['all_expected_samples_le_5cm'])
        self.assertEqual(s['primary_first_per_case_A']['missing'],1)
        r=report();r['attempts'].pop()
        with self.assertRaisesRegex(ValueError,'all actual close'):m.summarize(r)

if __name__=='__main__':unittest.main()
