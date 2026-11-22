"""CPU operational visual-co-motion protocol, no Native/fit/force claims."""
from copy import deepcopy
import unittest
import numpy as np
from our_mani.contracts import ObjectState,Pose
from our_mani.local_mpc_control_20261103.public_motion_bridge import PublicRobotSensors
from our_mani.local_mpc_control_20261103.active_visual_attachment import ActiveVisualAttachment


def sample(t,dz=0.,object_dz=None,*,fingers=(.7,.3),tcp_dx=0.,object_dx=0.,object_id='target'):
    p=np.array([.2+tcp_dx,-.15,.8+dz]);body=np.array([.2+object_dx,-.15,.775+(dz if object_dz is None else object_dz)])
    robot=PublicRobotSensors(t,p,[0,0,0,1],np.zeros(7),np.zeros(7),fingers)
    obj=ObjectState(object_id,t,Pose(body),body+np.array([[0,0,0],[.01,0,0],[0,.01,0]]),np.zeros(3),np.zeros(3))
    return obj,robot


def h(t,*,done=True,target=0.):
    return dict(reference_valid=True,reference_positions=np.zeros(7),reference_velocities=np.zeros(7),reference_accelerations=np.zeros(7),
        target_joints=np.zeros(7),reference_clock=t,gripper_target=target,gripper_done=done,reference_is_measured_state=False)


def admitted():
    estimator=ActiveVisualAttachment();obj,robot=sample(0.)
    estimator.observe(obj,robot,close_requested=False)
    obj,robot=sample(.05);estimator.observe(obj,robot,close_requested=True)
    spec=estimator.begin_probe(obj,robot,h(.05))
    return estimator,spec


class ActiveVisualTests(unittest.TestCase):
    def test_done_is_probe_admission_not_attachment_and_must_be_current_closed_attempt(self):
        estimator=ActiveVisualAttachment();obj,robot=sample(0.)
        estimator.observe(obj,robot,close_requested=False)
        self.assertIsNone(estimator.begin_probe(obj,robot,h(0.)))
        obj,robot=sample(.05);result=estimator.observe(obj,robot,close_requested=True)
        self.assertEqual(result['mode'],'UNRESOLVED')
        self.assertIsNone(estimator.begin_probe(obj,robot,h(.05,done=False)))
        self.assertIsNone(estimator.begin_probe(obj,robot,h(.05,target=1.)))
        with self.assertRaisesRegex(ValueError,'fresh'):estimator.begin_probe(obj,robot,h(0.))
        spec=estimator.begin_probe(obj,robot,h(.05));self.assertIsNotNone(spec)
        self.assertFalse(spec['actuator_done_is_contact']);self.assertFalse(spec['held'])
        self.assertEqual(estimator.mode,'UNRESOLVED')

    def test_six_real_samples_and_signed_co_motion_confirm_asymmetric_fingers(self):
        estimator,spec=admitted()
        for step in range(1,6):
            obj,robot=sample(.05+.05*step,dz=.012*step/5)
            result=estimator.observe(obj,robot,close_requested=True)
            self.assertEqual(result['mode'],'ATTACHED_PUBLIC' if step==5 else 'UNRESOLVED')
        self.assertEqual(result['diagnostic']['real_probe_frames'],6)
        self.assertGreaterEqual(result['diagnostic']['TCP_signed_vertical_m'],.010)
        self.assertEqual(result['evidence'].event,'attachment_confirmed')
        self.assertFalse(result['two_pad_contact_proven']);self.assertFalse(result['force_or_frictional_grasp_proof'])
        self.assertFalse(result['external_synchronous_Z_motion_ruled_out'])

    def test_confirmed_maintenance_can_lift_past_probe_cap_but_missing_visual_removes_evidence(self):
        estimator,spec=admitted()
        for step in range(1,6):
            obj,robot=sample(.05+.05*step,dz=.012*step/5);result=estimator.observe(obj,robot,close_requested=True)
        obj,robot=sample(.35,dz=.05);result=estimator.observe(obj,robot,close_requested=True)
        self.assertEqual(result['mode'],'ATTACHED_PUBLIC')
        _,robot=sample(.4,dz=.05);missing=estimator.observe(None,robot,close_requested=True)
        self.assertNotEqual(missing['mode'],'ATTACHED_PUBLIC');self.assertTrue(missing['probe_failed'])
        obj,robot=sample(.45,dz=.05);recovered=estimator.observe(obj,robot,close_requested=True)
        self.assertEqual(recovered['mode'],'UNRESOLVED')

    def test_fixed_r0_all_trace_does_not_allow_midpoint_mean_or_last_window(self):
        estimator,spec=admitted()
        obj,robot=sample(.1,dz=.008,object_dz=0.)
        result=estimator.observe(obj,robot,close_requested=True)
        self.assertTrue(result['probe_failed']);self.assertIn('fixed_r0',result['reason'])
        for step in range(2,10):
            obj,robot=sample(.05+.05*step,dz=.012)
            result=estimator.observe(obj,robot,close_requested=True)
        self.assertEqual(result['mode'],'UNRESOLVED')
        self.assertGreater(result['diagnostic']['maximum_fixed_r0_deviation_m'],.006)

    def test_fixed_deadline_does_not_roll_and_safety_precedes_positive(self):
        estimator,spec=admitted();obj,robot=sample(.1,dz=.0024)
        estimator.observe(obj,robot,close_requested=True)
        again=estimator.begin_probe(obj,robot,h(.1));self.assertEqual(spec['deadline'],again['deadline'])
        self.assertEqual(spec['identity'],again['identity'])
        for step in range(2,5):
            obj,robot=sample(.05+.05*step,dz=.012*step/5);estimator.observe(obj,robot,close_requested=True)
        obj,robot=sample(.3,dz=.016)
        result=estimator.observe(obj,robot,close_requested=True)
        self.assertTrue(result['probe_failed']);self.assertNotEqual(result['mode'],'ATTACHED_PUBLIC')
        self.assertIn('bound',result['reason'])

    def test_horizontal_or_model_only_or_downward_or_empty_stationary_body_do_not_confirm(self):
        for variant in ('horizontal','body_only','downward','stationary_body'):
            estimator,spec=admitted()
            for step in range(1,6):
                frac=step/5
                if variant=='horizontal':obj,robot=sample(.05+.05*step,tcp_dx=.012*frac,object_dx=.012*frac)
                elif variant=='body_only':obj,robot=sample(.05+.05*step,dz=0.,object_dz=.012*frac)
                elif variant=='downward':obj,robot=sample(.05+.05*step,dz=-.012*frac)
                else:obj,robot=sample(.05+.05*step,dz=.012*frac,object_dz=0.,fingers=(0.,0.))
                result=estimator.observe(obj,robot,close_requested=True)
            self.assertNotEqual(result['mode'],'ATTACHED_PUBLIC',variant)

    def test_gap_and_identity_are_failed_unknown_never_FREE(self):
        for variant in ('gap','identity','missing'):
            estimator,spec=admitted();obj,robot=sample(.15 if variant=='gap' else .1,dz=.0024,object_id='goal' if variant=='identity' else 'target')
            result=estimator.observe(None if variant=='missing' else obj,robot,close_requested=True)
            self.assertTrue(result['probe_failed']);self.assertEqual(result['mode'],'UNRESOLVED')

    def test_reopen_requires_public_separation_then_new_close_has_new_identity(self):
        estimator,spec=admitted()
        for step in range(1,5):
            obj,robot=sample(.05+.05*step,dz=.08,object_dz=0.,fingers=(1.,1.))
            result=estimator.observe(obj,robot,close_requested=False)
        self.assertEqual(result['mode'],'FREE');self.assertEqual(result['evidence'].event,'release_confirmed')
        obj,robot=sample(.3,dz=.08);estimator.observe(obj,robot,close_requested=True)
        self.assertEqual(estimator.attempt_id,2)
        spec2=estimator.begin_probe(obj,robot,h(.3))
        self.assertIsNotNone(spec2);self.assertNotEqual(spec['identity'],spec2['identity'])


if __name__=='__main__':unittest.main()
