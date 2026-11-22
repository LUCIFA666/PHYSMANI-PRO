"""Independent maintenance provenance/finite-proof regression examples."""
from copy import deepcopy
import unittest

from our_mani.local_mpc_control_20261103.postconfirmed_maintenance_candidate_20261103 import PostConfirmedPositionMaintenance
from tests.test_postconfirmed_maintenance_candidate_20261103 import proof,after,public_result
from tests.test_local_mpc_active_visual_attachment_phase_scope_20261103 import sensor_frame


class IndependentMaintenanceTests(unittest.TestCase):
    def start(self):
        result,robot,original=proof();model=PostConfirmedPositionMaintenance()
        self.assertTrue(model.start(result,robot,close_requested=True)['current_position_maintenance_passed'])
        return model,original

    def test_foreign_or_missing_original_protocol_latches_unknown_and_cannot_recover(self):
        for protocol in (None,'legacy_attachment','unrelated'):
            with self.subTest(protocol=protocol):
                model,_=self.start();result,robot=after(6)
                if protocol is None:result['attachment_estimate'].pop('protocol')
                else:result['attachment_estimate']['protocol']=protocol
                output=model.observe(result,robot,close_requested=True)
                self.assertFalse(output['current_position_maintenance_passed']);self.assertTrue(output['failed'])
                self.assertIsNone(output['evidence'])
                result,robot=after(7)
                self.assertFalse(model.observe(result,robot,close_requested=True)['current_position_maintenance_passed'])

    def test_nonfinite_or_untyped_confirmation_cannot_become_initial_proof(self):
        for field,value in (('duration_seconds',float('inf')),('duration_seconds',float('nan')),
                ('TCP_signed_vertical_m',float('inf')),('object_signed_vertical_m',float('inf')),
                ('real_probe_frames',6.),('real_probe_frames',True)):
            with self.subTest(field=field,value=value):
                result,robot,_=proof();result['attachment_estimate']['diagnostic']['confirmation'][field]=value
                model=PostConfirmedPositionMaintenance()
                with self.assertRaises(ValueError):model.start(result,robot,close_requested=True)
                self.assertFalse(model.started)

    def test_genuine_old6mm_failure_can_be_rescored_but_large_current_drift_never_rolls_anchor(self):
        model,original=self.start();r0=model.r0.copy()
        target,robot=sensor_frame(6,height=.04,relative_shift=(.007,0.,0.))
        old=original.observe(target,robot,close_requested=True)
        self.assertTrue(old['probe_failed']);self.assertEqual(old['mode'],'UNRESOLVED')
        output=model.observe(public_result(target,robot,old),robot,close_requested=True)
        self.assertTrue(output['current_position_maintenance_passed'])
        self.assertFalse(output['force_or_frictional_grasp_proof']);self.assertFalse(output['full_SE3_attachment_qualified'])
        target,robot=sensor_frame(7,height=.04,relative_shift=(.011,0.,0.))
        old=original.observe(target,robot,close_requested=True)
        output=model.observe(public_result(target,robot,old),robot,close_requested=True)
        self.assertFalse(output['current_position_maintenance_passed']);self.assertIsNone(output['evidence'])
        self.assertEqual(model.r0.tolist(),r0.tolist());self.assertTrue(model.failed)


if __name__=='__main__':unittest.main()
