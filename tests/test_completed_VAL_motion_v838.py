"""CPU-only checks of quantiles and missing/direct-command measurement limits."""
from copy import deepcopy
import importlib.util
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SOURCE=Path(__file__).resolve().parents[1]/'scripts/summarize_completed_VAL_motion_v838_r10.py'
spec=importlib.util.spec_from_file_location('motion_v838_test_subject',SOURCE)
summary=importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary)


def case_fixture():
    return dict(method='cv_geometry',counted_clock=dict(ticks=1,seconds=.05),alignment_receipt={},
        actor_metrics=dict(physmani_geometry_executor=dict(actions=[dict(original_executor_receipt=dict(actual_speed_receipt=dict(
            command_ticks=0,maximum_actual_native_reported_speed_by_joint_rad_per_s=[0.]*7,
            maximum_actual_average_speed_by_joint_rad_per_s=[0.]*7,
            maximum_actual_native_reported_speed_fraction=0.,maximum_actual_average_speed_fraction=0.)))])),
        all_stage_counted_joint_motion=dict(peak_actual_window_acceleration=0.,all_stage_actual_window_checks_passed=True,
            events=[dict(backward_window_velocity=[0.]*7,backward_window_acceleration=None,
                         acceleration_passed=None,speed_passed=True,position_passed=True)]))


class CompletedVALMotionTests(unittest.TestCase):
    def test_linear_quantiles_weight_cases_equally(self):
        self.assertEqual(summary.quantile([1,2,10,12],.5),6.)
        self.assertTrue(math.isclose(summary.quantile([1,2,10,12],.95),11.7))
        self.assertEqual(summary.quantile([8],.95),8.)
        self.assertIsNone(summary.quantile([],.95))

    def test_unknown_nonfinite_and_bool_do_not_become_zero(self):
        result=summary.statistics([None,False,float('nan'),0.,2.])
        self.assertEqual(result['n'],2)
        self.assertEqual(result['unknown_or_nonfinite_cases'],3)
        self.assertEqual(result['min'],0.)
        self.assertEqual(result['median'],1.)

    def test_unobserved_acceleration_and_zero_tick_native_qdot_stay_unknown(self):
        metrics,diagnostics=summary.reduce_saved_case(case_fixture())
        self.assertEqual(metrics['peak_window_joint_speed_rad_s'],0.)
        self.assertIsNone(metrics['peak_window_joint_acceleration_rad_s2'])
        self.assertIsNone(metrics['direct_command_native_qdot_peak_rad_s'])
        self.assertIsNone(metrics['direct_command_native_speed_fraction_peak'])
        self.assertEqual(metrics['direct_command_ticks'],0)
        self.assertEqual(diagnostics['original_flag_counts']['acceleration_passed']['unevaluated'],1)

    def test_acceleration_uses_saved_peak_and_flags_without_new_differencing(self):
        case=case_fixture()
        case['all_stage_counted_joint_motion']['peak_actual_window_acceleration']=12.
        case['all_stage_counted_joint_motion']['events'].append(dict(backward_window_velocity=[1.]*7,
            backward_window_acceleration=[100.]*7,acceleration_passed=False,speed_passed=True,position_passed=True))
        metrics,diagnostics=summary.reduce_saved_case(case)
        self.assertEqual(metrics['peak_window_joint_acceleration_rad_s2'],12.)
        self.assertEqual(diagnostics['original_flag_counts']['acceleration_passed']['failed'],1)

    def test_native_false_case_remains_in_equal_weight_cohort(self):
        metrics,diagnostics=summary.reduce_saved_case(case_fixture())
        rows=[dict(task='t',method='cv_geometry',source_raw_episode_id='one',source_row_sha256='a',
            official_task_success=True,metrics=metrics,recorded_window_diagnostics=diagnostics),
              dict(task='t',method='cv_geometry',source_raw_episode_id='two',source_row_sha256='b',
            official_task_success=False,metrics=deepcopy(metrics),recorded_window_diagnostics=deepcopy(diagnostics))]
        rows[1]['metrics']['peak_window_joint_speed_rad_s']=10.
        result=summary.summarize(rows)
        self.assertEqual(result['case_count'],2)
        self.assertEqual(result['native_false_cases'],1)
        self.assertEqual(result['metrics']['peak_window_joint_speed_rad_s']['median'],5.)

    def test_TRAIN_case_pointer_rejected_before_JSON_loading(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            relative='artifacts/framework/multitask_r10/fake/train/case.json'
            path=root/relative
            path.parent.mkdir(parents=True)
            path.write_text('not JSON and must never be parsed')
            with patch.object(summary,'ROOT',root):
                with self.assertRaisesRegex(ValueError,'VAL case evidence only'):
                    summary.load_checked(dict(path=relative,sha256='0'*64),{},case=True)


if __name__=='__main__':
    unittest.main()
