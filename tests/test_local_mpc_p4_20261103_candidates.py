"""CPU fixed-generation adapter contracts; no model fit or Native calls."""
from copy import deepcopy
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
import torch

from our_mani.local_mpc_p4_20261103.candidates import CandidateLedger
from our_mani.local_mpc_p4_20261103.model import CandidateTransformerPilot
from our_mani.local_mpc_20261102.mpc import validate_candidate
from tests.test_local_mpc_p4_20261103_model import features


def output():
    return dict(positions=np.c_[np.arange(8)*.001,np.zeros(8),np.full(8,-.015)],
                rotation_matrices=Rotation.from_euler('xyz',np.array([[.4,-.3,.2]])+np.c_[np.zeros(8),np.zeros(8),np.arange(8)*.1]).as_matrix(),
                times=np.linspace(.05,.8,8),proposal_logits=np.zeros(8))


class P4CandidateLedgerTests(unittest.TestCase):
    def test_object_relative_metres_rotations_and_aperture_units_match_old_schema(self):
        proposals=output();result=CandidateLedger().new_generation(10.,proposals)
        self.assertEqual(len(result['candidates']),8)
        for i,c in enumerate(result['candidates']):
            validate_candidate(c)
            np.testing.assert_array_equal(c['object_tcp_position'],proposals['positions'][i])
            np.testing.assert_allclose(Rotation.from_quat(c['object_tcp_quaternion']).as_matrix(),proposals['rotation_matrices'][i],atol=1e-12)
            self.assertAlmostEqual(np.linalg.norm(c['object_tcp_quaternion']),1.)
            self.assertAlmostEqual(c['deadline'],10.+proposals['times'][i])
            self.assertEqual(c['opening'],0.);self.assertEqual(c['required_opening'],1.)
            self.assertEqual(c['required_width_m'],.08);self.assertEqual(c['pregrasp_offset'],.07)
            self.assertEqual(c['approach_direction'],[0.,0.,1.])
            self.assertEqual(c['proposal_weight'],1./8.);self.assertEqual(c['proposal_logit'],0.)
            self.assertFalse(c['native_pose_qualified']);self.assertFalse(c['physical_execution_qualified'])

    def test_active_generation_cannot_chase_replans_or_mutating_input_and_output(self):
        ledger=CandidateLedger();proposals=output();initial=ledger.new_generation(1.,proposals)
        original=deepcopy(initial['candidates']);proposals['positions'][:]=.1
        initial['candidates'][0]['object_tcp_position'][0]=999.
        current=ledger.update_expiry(1.2)
        for before,after in zip(original,current['candidates']):
            self.assertEqual(before['id'],after['id']);self.assertEqual(before['deadline'],after['deadline'])
            self.assertEqual(before['object_tcp_position'],after['object_tcp_position'])
            self.assertEqual(before['object_tcp_quaternion'],after['object_tcp_quaternion'])
        with self.assertRaisesRegex(ValueError,'active'):ledger.update_expiry(1.3,output())
        with self.assertRaisesRegex(ValueError,'active'):ledger.new_generation(1.3,output())

    def test_expired_and_outside_forecast_slots_are_masked_without_time_changes(self):
        ledger=CandidateLedger();first=ledger.new_generation(1.,output());times=[c['deadline'] for c in first['candidates']]
        qualified=np.ones(8,dtype=bool);qualified[2]=False
        view=ledger.update_expiry(1.1,forecast_end=1.5,feasible_mask=qualified)
        self.assertFalse(view['available_mask'][0]);self.assertIn('fixed_deadline_expired',view['mask_reasons'][0])
        self.assertFalse(view['available_mask'][2]);self.assertIn('caller_unreachable_or_unqualified',view['mask_reasons'][2])
        self.assertFalse(view['available_mask'][-1]);self.assertIn('outside_current_forecast_domain',view['mask_reasons'][-1])
        self.assertEqual(times,[c['deadline'] for c in view['candidates']])
        self.assertTrue(view['available_mask'][1]);self.assertFalse(view['physical_execution_qualified'])

    def test_all_expired_requires_explicit_current_output_and_new_generation_ids(self):
        ledger=CandidateLedger();old=ledger.new_generation(2.,output());oldids=[c['id'] for c in old['candidates']]
        edge=ledger.update_expiry(2.8);self.assertFalse(edge['new_generation_required'])
        waiting=ledger.update_expiry(2.81);self.assertTrue(waiting['new_generation_required'])
        self.assertFalse(waiting['available_mask'].any());self.assertFalse(waiting['geometry_fallback_used'])
        new=ledger.update_expiry(2.9,output(),forecast_end=3.7)
        self.assertEqual(new['generation'],1);self.assertFalse(new['new_generation_required'])
        self.assertFalse(set(oldids)&{c['id'] for c in new['candidates']})
        self.assertEqual(new['events'][-1]['reason'],'all_fixed_deadlines_expired_new_model_generation')
        self.assertEqual(new['events'][-1]['retired'],[dict(id=c['id'],deadline=c['deadline']) for c in old['candidates']])
        for c,tau in zip(new['candidates'],output()['times']):self.assertAlmostEqual(c['deadline'],2.9+tau)

    def test_no_initial_weights_or_outputs_produces_no_hidden_geometry_fallback(self):
        ledger=CandidateLedger();waiting=ledger.update_expiry(.2)
        self.assertEqual(waiting['candidates'],[]);self.assertEqual(waiting['generation'],-1)
        self.assertTrue(waiting['new_generation_required']);self.assertFalse(waiting['geometry_fallback_used'])
        self.assertEqual(waiting['available_mask'].shape,(0,))

    def test_invalid_numeric_units_rotations_scores_and_extra_truth_channels_rejected(self):
        for variant in ('nanposition','inftime','milliseconds','belowtime','badrotation','reflection','metres','successlogits','truth'):
            proposals=output()
            if variant=='nanposition':proposals['positions'][3,1]=np.nan
            elif variant=='inftime':proposals['times'][3]=np.inf
            elif variant=='milliseconds':proposals['times']*=1000.
            elif variant=='belowtime':proposals['times'][0]=0.
            elif variant=='badrotation':proposals['rotation_matrices'][2,0,0]=2.
            elif variant=='reflection':proposals['rotation_matrices'][2,:,0]*=-1
            elif variant=='metres':proposals['positions']*=1000.
            elif variant=='successlogits':proposals['proposal_logits'][0]=.7
            else:proposals['Native_held']=True
            with self.subTest(variant=variant),self.assertRaises(ValueError):CandidateLedger().new_generation(0.,proposals)

    def test_model_batch_one_is_accepted_and_multiple_anchors_are_not_flattened(self):
        model=CandidateTransformerPilot(d_model=32,num_layers=1,num_heads=4).eval()
        out=model(**features(1));view=CandidateLedger().new_generation(5.,out)
        self.assertEqual(len(view['candidates']),8)
        for c in view['candidates']:self.assertAlmostEqual(c['deadline'],5.+.425,places=6)
        multiple=model(**features(2))
        with self.assertRaises(ValueError):CandidateLedger().new_generation(5.,multiple)

    def test_time_reversal_nonfinite_clock_and_nonboolean_qualification_are_rejected(self):
        ledger=CandidateLedger();ledger.new_generation(1.,output());ledger.update_expiry(1.1)
        with self.assertRaisesRegex(ValueError,'backwards'):ledger.update_expiry(1.05)
        for t in (float('nan'),float('inf'),True,-1.):
            with self.assertRaises(ValueError):CandidateLedger().new_generation(t,output())
        with self.assertRaisesRegex(ValueError,'bool'):ledger.update_expiry(1.2,feasible_mask=np.ones(8))
        with self.assertRaisesRegex(ValueError,'Forecast'):ledger.update_expiry(1.2,forecast_end=1.1)

    def test_float32_endpoint_roundoff_is_kept_and_masked_if_forecast_domain_is_stricter(self):
        proposals=output();proposals['times']=proposals['times'].astype(np.float32)
        ledger=CandidateLedger();initial=ledger.new_generation(0.,proposals)
        raw=float(proposals['times'][-1]);self.assertEqual(initial['candidates'][-1]['deadline'],raw)
        view=ledger.update_expiry(0.,forecast_end=.8)
        self.assertEqual(view['candidates'][-1]['deadline'],raw)
        if raw>.8+1e-9:
            self.assertFalse(view['available_mask'][-1])
            self.assertIn('outside_current_forecast_domain',view['mask_reasons'][-1])


if __name__=='__main__':unittest.main()
