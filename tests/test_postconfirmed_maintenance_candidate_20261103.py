"""Independent candidate only; genuine unchanged synthetic probe admission."""
from copy import deepcopy
import unittest
import numpy as np
from our_mani.local_mpc_control_20261103.active_visual_attachment import ActiveVisualAttachment,ActiveVisualConfig
from our_mani.local_mpc_control_20261103.postconfirmed_maintenance_candidate_20261103 import (
    PostConfirmedPositionMaintenance,PostConfirmedMaintenanceConfig)
from tests.test_local_mpc_active_visual_attachment_phase_scope_20261103 import sensor_frame,actual_h


def public_result(target,robot,attachment):
    return dict(timestamp=robot.timestamp,mode=attachment['mode'],available=attachment['mode']=='ATTACHED_PUBLIC',
        controlled_mode_available=attachment['mode']=='ATTACHED_PUBLIC',object_observation=target,
        attachment_estimate=attachment,Native_inputs_used=False,future_observations_used=False,observations_fabricated=False)


def proof():
    original=ActiveVisualAttachment(ActiveVisualConfig(enforce_probe_anchor_orientation_after_confirmation=False))
    obj,robot=sensor_frame(0);a=original.observe(obj,robot,close_requested=True);original.begin_probe(obj,robot,actual_h(robot))
    for i in range(1,6):
        obj,robot=sensor_frame(i,height=.012*i/5);a=original.observe(obj,robot,close_requested=True)
    assert a['mode']=='ATTACHED_PUBLIC'
    return public_result(obj,robot,a),robot,original


def after(i,*,shift=(0.,0.,0.),height=.04,attempt=1):
    obj,robot=sensor_frame(i,height=height,relative_shift=shift)
    return public_result(obj,robot,dict(timestamp=robot.timestamp,protocol='active_vertical_visual_comotion_v4',attempt_id=attempt,attempt_active=True,mode='UNRESOLVED')),robot


class PostConfirmedCandidateTests(unittest.TestCase):
    def start(self):
        result,robot,_=proof();model=PostConfirmedPositionMaintenance()
        self.assertTrue(model.start(result,robot,close_requested=True)['current_position_maintenance_passed'])
        return model

    def test_no_new_rule_can_replace_initial_probe6mm_or_angle_proof(self):
        r,rb,_=proof();model=PostConfirmedPositionMaintenance()
        r['attachment_estimate']['diagnostic']['confirmation']['maximum_fixed_r0_deviation_m']=.007
        with self.assertRaises(ValueError):model.start(r,rb,close_requested=True)
        r,rb,_=proof();r['attachment_estimate']['probe']['config']['maximum_rotation_rad']=.05
        with self.assertRaises(ValueError):PostConfirmedPositionMaintenance().start(r,rb,close_requested=True)
        r,rb,_=proof();r['attachment_estimate']['diagnostic']['confirmation']['real_probe_frames']=5
        with self.assertRaises(ValueError):PostConfirmedPositionMaintenance().start(r,rb,close_requested=True)
        r,rb,original=proof();self.assertEqual(original.config.maximum_relative_deviation_m,.006)

    def test_optional_phase_candidate10mm_uses_fixed_original_r0_without_claiming_force(self):
        model=self.start();frozen=model.r0.copy()
        r,rb=after(6,shift=(.009,0.,0.));out=model.observe(r,rb,close_requested=True)
        self.assertTrue(out['current_position_maintenance_passed'])
        self.assertAlmostEqual(out['current_fixed_r0_deviation_m'],.009)
        self.assertFalse(out['evidence'].orientation_observable)
        self.assertFalse(out['force_or_frictional_grasp_proof']);self.assertFalse(out['full_SE3_attachment_qualified'])
        r,rb=after(7,shift=(.0101,0.,0.));out=model.observe(r,rb,close_requested=True)
        self.assertTrue(out['failed']);np.testing.assert_array_equal(model.r0,frozen)
        r,rb=after(8);self.assertFalse(model.observe(r,rb,close_requested=True)['current_position_maintenance_passed'])

    def test_drop_back_to_table_fails_even_at_constant_relative_position(self):
        model=self.start();r,rb=after(6,height=0.)
        out=model.observe(r,rb,close_requested=True)
        self.assertAlmostEqual(out['current_fixed_r0_deviation_m'],0.)
        self.assertEqual(out['reason'],'current_visual_body_below_original_probe_anchor_plus8mm')
        self.assertTrue(out['failed'])

    def test_goal_marker_jump_not_a_grasped_target_and_failure_cannot_reconfirm(self):
        model=self.start();r,rb=after(6,shift=(.07,0.,0.))
        out=model.observe(r,rb,close_requested=True);self.assertTrue(out['failed'])
        self.assertIsNone(out['evidence'])
        fresh,robot,_=proof()
        with self.assertRaises(ValueError):model.start(fresh,robot,close_requested=True)

    def test_gap_open_and_attempt_mismatch_latch_failure(self):
        for fault in ['gap','open','attempt','stale_attempt']:
            with self.subTest(fault=fault):
                model=self.start();r,rb=after(7 if fault=='gap' else 6,attempt=2 if fault=='attempt' else 1)
                if fault=='stale_attempt':r['attachment_estimate']['timestamp']-=.05
                out=model.observe(r,rb,close_requested=fault!='open');self.assertTrue(out['failed'])
                self.assertFalse(out['current_position_maintenance_passed'])

    def test_model_height_or_stale_and_nonpublic_poses_do_not_supply_current_vision(self):
        for fault in ['missing','future_result','Native','future','fabricated']:
            with self.subTest(fault=fault):
                model=self.start();r,rb=after(6)
                if fault=='missing':r['object_observation']=None
                elif fault=='future_result':r['timestamp']+=.05
                else:r[{'Native':'Native_inputs_used','future':'future_observations_used','fabricated':'observations_fabricated'}[fault]]=True
                r['controlled']=dict(height=999.,held=True);r['forecast']=dict(position=[999,999,999])
                self.assertFalse(model.observe(r,rb,close_requested=True)['current_position_maintenance_passed'])

    def test_first_proof_must_be_fresh_and_current_closed(self):
        for fault in ['open','old_confirmation','inactive']:
            with self.subTest(fault=fault):
                r,rb,_=proof()
                if fault=='old_confirmation':r['attachment_estimate']['diagnostic']['confirmation']['timestamp']-=.05
                if fault=='inactive':r['attachment_estimate']['attempt_active']=False
                with self.assertRaises(ValueError):PostConfirmedPositionMaintenance().start(r,rb,close_requested=fault!='open')

    def test_no_rolling_anchor_can_hide_several_sub10mm_steps(self):
        model=self.start()
        for i,drift in [(6,.004),(7,.008),(8,.012)]:
            r,rb=after(i,shift=(drift,0.,0.));out=model.observe(r,rb,close_requested=True)
        self.assertFalse(out['current_position_maintenance_passed']);self.assertTrue(out['failed'])
        self.assertGreater(out['current_fixed_r0_deviation_m'],.010)
        with self.assertRaises(ValueError):PostConfirmedMaintenanceConfig(maximum_fixed_r0_deviation_m=.02)

    def test_unrelated_or_missing_original_estimator_protocol_is_not_same_attempt_authority(self):
        for protocol in [None,'legacy_attachment','unrelated']:
            with self.subTest(protocol=protocol):
                model=self.start();r,rb=after(6)
                if protocol is None:r['attachment_estimate'].pop('protocol')
                else:r['attachment_estimate']['protocol']=protocol
                out=model.observe(r,rb,close_requested=True)
                self.assertFalse(out['current_position_maintenance_passed']);self.assertTrue(out['failed'])

    def test_nonfinite_proof_values_config_or_untyped_frame_counts_cannot_admit(self):
        for section,field,value in [('confirmation','duration_seconds',float('inf')),
                ('confirmation','TCP_signed_vertical_m',float('inf')),
                ('confirmation','object_signed_vertical_m',float('nan')),
                ('confirmation','maximum_fixed_r0_deviation_m',-float('inf')),
                ('confirmation','real_probe_frames',6.),('confirmation','real_probe_frames',True),
                ('config','confirmation_seconds',float('inf')),('config','confirmation_frames',6.),
                ('probe_config','position_scale_m',float('inf'))]:
            with self.subTest(section=section,field=field,value=value):
                r,rb,_=proof()
                if section=='confirmation':target=r['attachment_estimate']['diagnostic']['confirmation']
                elif section=='config':target=r['attachment_estimate']['config']
                else:target=r['attachment_estimate']['probe']['config']
                target[field]=value
                with self.assertRaises(ValueError):PostConfirmedPositionMaintenance().start(r,rb,close_requested=True)


if __name__=='__main__':unittest.main()
