from copy import deepcopy
from pathlib import Path
import json,tempfile
import unittest
import numpy as np

from our_mani.local_mpc_control_20261103.public_shadow_evaluation import evaluate_rows,evaluate_shadow_directory


def fixture(*,contact_labels=True,final=.8,anchor=0.):
    times=anchor+np.arange(17)*.05
    obj=dict(object_id='target',timestamp=anchor,visible=True,confidence=1.,
             pose=dict(position=[0.,0.,.775]),linear_velocity=[.1,0.,0.])
    positions=np.c_[.1*(times-anchor),np.zeros(17),np.full(17,.775)]
    result=dict(timestamp=anchor,mode='FREE',available=True,free_forecast_available=True,object_observation=obj,
                endpoint_times=times[2::2].tolist(),positions=positions[2::2].tolist(),
                forecast=dict(object_ids=['target'],times=times.tolist(),positions=positions[:,None,:].tolist()))
    outputs=[dict(index=0,result=result,close_requested=False)]
    labels=[]
    for time in np.arange(0.,final+.00001,.05):
        label=dict(time=float(time),object_position=[.1*(time-anchor),0.,.775],held=False,native_success=False)
        if contact_labels:label['contact']=False
        labels.append(label)
    return outputs,labels


class ShadowEvaluationTests(unittest.TestCase):
    def test_exact_H8_full_horizon_and_same_observation_CV_statistics(self):
        public,labels=fixture();before=deepcopy(public);r=evaluate_rows(public,labels)
        self.assertEqual(r['emitted_free_forecast_anchors'],1)
        self.assertEqual(r['forecast']['GRU_H8']['aggregate']['all_raw']['expected_comparisons'],8)
        self.assertEqual(r['forecast']['GRU_full_future']['aggregate']['all_raw']['expected_comparisons'],16)
        self.assertEqual(r['H8_full_forecast_consistency']['max_difference_m'],0.)
        for head in ('GRU_H8','CV_H8','GRU_full_future','CV_full_future'):
            stats=r['forecast'][head]['aggregate']['strict_free_precontact']
            self.assertAlmostEqual(stats['mean_m'],0.)
            self.assertTrue(stats['position_scope_gate_passed'])
        self.assertEqual(public,before);self.assertFalse(r['Native_task_success_claimed'])

    def test_contact_crossing_censors_only_free_scope_and_raw_errors_remain(self):
        public,labels=fixture()
        labels[6]['contact']=True
        for row in labels:
            if row['time']>=.3:row['object_position'][0]+=.1
        r=evaluate_rows(public,labels);head=r['forecast']['GRU_H8']['aggregate']
        self.assertEqual(head['all_raw']['expected_comparisons'],8)
        self.assertEqual(head['strict_free_precontact']['expected_comparisons'],2)
        self.assertEqual(head['strict_free_precontact']['contact_censored_comparisons'],6)
        self.assertGreater(head['all_raw']['max_m'],.09)
        self.assertTrue(head['strict_free_precontact']['position_scope_gate_passed'])
        self.assertFalse(head['auxiliary_preheld_only_contact_unverified']['strict_free_motion_certified'])
        self.assertEqual(head['auxiliary_preheld_only_contact_unverified']['expected_comparisons'],8)

    def test_missing_contact_never_passes_strict_gate_but_unheld_errors_reported(self):
        public,labels=fixture(contact_labels=False);r=evaluate_rows(public,labels)
        group=r['forecast']['GRU_H8']['aggregate'];strict=group['strict_free_precontact']
        self.assertEqual(strict['expected_comparisons'],0)
        self.assertEqual(strict['eligibility_unknown_comparisons'],8)
        self.assertFalse(strict['position_scope_gate_passed'])
        self.assertEqual(group['auxiliary_preheld_only_contact_unverified']['observed_finite_comparisons'],8)
        self.assertEqual(group['all_raw']['max_m'],0.)

    def test_missing_endpoints_no_interpolation_and_missing_middle_tick_unknown(self):
        public,labels=fixture(final=.4);r=evaluate_rows(public,labels)
        stats=r['forecast']['GRU_H8']['aggregate']['all_raw']
        self.assertEqual(stats['expected_comparisons'],8)
        self.assertEqual(stats['observed_finite_comparisons'],4)
        self.assertEqual(stats['missing_or_invalid_comparisons'],4)
        self.assertFalse(stats['position_scope_gate_passed']);self.assertEqual(stats['fraction_le_5cm'],.5)
        public,labels=fixture();del labels[1]
        r=evaluate_rows(public,labels);strict=r['forecast']['GRU_H8']['aggregate']['strict_free_precontact']
        self.assertEqual(strict['eligibility_unknown_comparisons'],8)
        self.assertFalse(strict['position_scope_gate_passed'])

    def test_nonfinite_or_missing_prediction_retains_denominator_and_hard_max_gate(self):
        public,labels=fixture();public[0]['result']['positions'][0]=[float('nan'),0.,.775]
        public[0]['result']['positions'][1][0]+=.06
        r=evaluate_rows(public,labels);stats=r['forecast']['GRU_H8']['aggregate']['all_raw']
        self.assertEqual(stats['expected_comparisons'],8);self.assertEqual(stats['observed_finite_comparisons'],7)
        self.assertAlmostEqual(stats['max_m'],.06);self.assertFalse(stats['all_samples_le_5cm'])
        public,labels=fixture();public[0]['result']['object_observation']['timestamp']=-1.
        r=evaluate_rows(public,labels)
        self.assertEqual(r['forecast']['CV_H8']['aggregate']['all_raw']['observed_finite_comparisons'],0)

    def test_held_boundary_mode_unavailable_counts_and_timestamp_ambiguity(self):
        public,labels=fixture();labels[4]['held']=True
        extra=deepcopy(public[0]);extra['result']=dict(timestamp=.05,mode='STALE',free_forecast_available=False,reason='no_current_observation')
        public.append(extra);r=evaluate_rows(public,labels)
        self.assertEqual(r['unavailable_forecast_anchors'],1)
        self.assertEqual(r['forecast']['GRU_H8']['aggregate']['strict_free_precontact']['held_censored_comparisons'],7)
        with self.assertRaisesRegex(ValueError,'duplicate'):evaluate_rows(public+[deepcopy(public[0])],labels)

    def test_file_TEST_scope_guard_precedes_labels_and_artifacts_are_fresh(self):
        public,labels=fixture()
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory);shadow=base/'shadow';shadow.mkdir()
            (shadow/'scope.json').write_text(json.dumps(dict(formal_TEST_used=True)))
            (shadow/'evaluation_labels.jsonl').write_text('MUST_NOT_PARSE')
            with self.assertRaisesRegex(ValueError,'TEST'):evaluate_shadow_directory(shadow)
            (shadow/'scope.json').write_text(json.dumps(dict(formal_TEST_used=False)))
            (shadow/'public_outputs.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in public))
            (shadow/'evaluation_labels.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in labels))
            out=base/'new_artifact';result=evaluate_shadow_directory(shadow,output_dir=out)
            self.assertTrue((out/'report.json').is_file());self.assertTrue((out/'sample_ledger.jsonl').is_file())
            self.assertEqual(result['sample_ledger_file']['rows'],17+48)
            with self.assertRaises(FileExistsError):evaluate_shadow_directory(shadow,output_dir=out)


if __name__=='__main__':unittest.main()
