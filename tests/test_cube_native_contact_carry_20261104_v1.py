"""Pure geometry/provenance only. No Native, learned model or raw recordings."""
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from our_mani.contracts import Pose,ObjectState,Forecast
from our_mani.local_mpc_control_20261103.public_motion_bridge import PublicRobotSensors
from our_mani.cube_native_contact_feedback_live_20261104_v1.carry import NativeContactCarryEstimate


def robot(t,p=(0.,0.,0.),angle=0.):
    return PublicRobotSensors(t,np.asarray(p),Rotation.from_rotvec([0.,0.,angle]).as_quat(),np.zeros(7),np.zeros(7),np.ones(2))

def target(t,p=(.1,0.,0.)):
    return ObjectState('target',t,Pose(np.asarray(p)),np.asarray([p,np.asarray(p)+[.01,0,0],np.asarray(p)+[0,.01,0]]),np.zeros(3),np.zeros(3))

def forecast(t0=0.,horizon=.8):
    times=t0+np.linspace(0.,horizon,17)
    p=np.zeros((17,1,3));p[:,:,0]=.1
    return Forecast(('target',),times,p,np.broadcast_to(np.eye(3),(17,1,3,3)),np.zeros_like(p),np.zeros_like(p),metadata=dict(observed_until=float(t0),checkpoint_sha256='a'*64))


class CarryContracts(unittest.TestCase):
    def test_actual_visual_offset_then_occlusion_rigid_pose_and_lever_velocity(self):
        c=NativeContactCarryEstimate()
        a=c.observe(robot(0.),held=True,current_target=target(0.),last_public_forecast=None)
        self.assertEqual(a['status'],'HELD_CURRENT_VISUAL');self.assertFalse(a['velocity_estimate_valid'])
        b=c.observe(robot(.1,p=(0.,.02,0.),angle=.2),held=True,current_target=None,last_public_forecast=None)
        lever=Rotation.from_rotvec([0,0,.2]).apply([.1,0,0])
        np.testing.assert_allclose(b['estimated_pose'].position,np.asarray([0,.02,0])+lever)
        np.testing.assert_allclose(b['linear_velocity'],np.asarray([0,.2,0])+np.cross([0,0,2.],lever),atol=1e-12)
        np.testing.assert_allclose(b['angular_velocity'],[0,0,2.],atol=1e-12)
        self.assertAlmostEqual(b['offset_age'],.1);self.assertTrue(b['proprioceptive_propagation'])
        self.assertFalse(b['current_visual_measurement']);self.assertFalse(b['FREE_history_eligible'])
        self.assertFalse(b['Native_target_pose_used']);self.assertFalse(b['orientation_observed'])

    def test_missing_at_first_contact_forecast_initialization_is_explicit(self):
        c=NativeContactCarryEstimate()
        c.observe(robot(.1),held=False,current_target=None,last_public_forecast=forecast())
        a=c.observe(robot(.2,p=(.02,0,0)),held=True,current_target=None,last_public_forecast=forecast())
        np.testing.assert_allclose(a['estimated_pose'].position,[.1,0,0]);self.assertTrue(a['predicted_offset_initialization'])
        self.assertEqual(a['offset_source'],'in_horizon_public_forecast_at_actual_contact')
        b=c.observe(robot(.3,p=(.04,0,0)),held=True,current_target=None,last_public_forecast=None)
        np.testing.assert_allclose(b['estimated_pose'].position,[.12,0,0]);self.assertAlmostEqual(b['offset_age'],.1)
        self.assertFalse(b['current_visual_measurement']);self.assertFalse(b['task_success_claimed'])

    def test_expired_future_observed_or_long_forecast_never_extrapolated(self):
        for f,t in ((forecast(),.800000001),(forecast(.2),.1),(forecast(horizon=1.),.1)):
            c=NativeContactCarryEstimate();a=c.observe(robot(t),held=True,current_target=None,last_public_forecast=f)
            self.assertIsNone(a['estimated_pose']);self.assertEqual(a['status'],'HELD_OFFSET_UNKNOWN')
        c=NativeContactCarryEstimate();f=forecast();f.metadata['observed_until']=.01
        with self.assertRaises(ValueError):c.observe(robot(.1),held=True,current_target=None,last_public_forecast=f)

    def test_later_visual_refresh_and_release_clear_predicted_offset(self):
        c=NativeContactCarryEstimate()
        c.observe(robot(.1),held=True,current_target=None,last_public_forecast=forecast())
        a=c.observe(robot(.2),held=True,current_target=target(.2,(.2,0,0)),last_public_forecast=None)
        self.assertEqual(a['offset_source'],'current_visual_TCP_relative_estimate');self.assertFalse(a['predicted_offset_initialization'])
        self.assertEqual(a['offset_age'],0.)
        b=c.observe(robot(.3),held=False,current_target=target(.3),last_public_forecast=None)
        self.assertTrue(b['membership_loss_observed']);self.assertIsNone(b['estimated_pose']);self.assertIsNone(b['offset_source'])
        a=c.observe(robot(.4),held=True,current_target=None,last_public_forecast=None)
        self.assertIsNone(a['estimated_pose']);self.assertEqual(a['status'],'HELD_OFFSET_UNKNOWN')

    def test_same_frame_time_type_and_target_identity_fail_closed(self):
        c=NativeContactCarryEstimate();c.observe(robot(.1),held=False,current_target=None,last_public_forecast=None)
        with self.assertRaises(ValueError):c.observe(robot(.1),held=True,current_target=None,last_public_forecast=None)
        with self.assertRaises(ValueError):c.observe(robot(.2),held=True,current_target=target(.15),last_public_forecast=None)
        with self.assertRaises(TypeError):c.observe(robot(.2),held=1,current_target=None,last_public_forecast=None)
        with self.assertRaises(TypeError):c.observe({'timestamp':.2},held=True,current_target=None,last_public_forecast=None)
        with self.assertRaises(TypeError):c.observe(robot(.2),held=True,current_target=None,last_public_forecast={})
        # Rejected records do not consume the timestamp or forge a carry anchor.
        a=c.observe(robot(.2),held=True,current_target=None,last_public_forecast=None)
        self.assertIsNone(a['estimated_pose']);self.assertEqual(a['contact_observation_time'],.2)

    def test_no_late_forecast_after_contact_can_initialize_from_future_observations(self):
        c=NativeContactCarryEstimate()
        c.observe(robot(.1),held=True,current_target=None,last_public_forecast=None)
        a=c.observe(robot(.2),held=True,current_target=None,last_public_forecast=forecast(.15))
        self.assertIsNone(a['estimated_pose'])
        # A genuinely prior forecast can be evaluated at first contact, with
        # the original contact TCP rather than today's already moved TCP.
        b=c.observe(robot(.3,p=(.03,0,0)),held=True,current_target=None,last_public_forecast=forecast())
        np.testing.assert_allclose(b['estimated_pose'].position,[.13,0,0]);self.assertAlmostEqual(b['offset_age'],.2)

if __name__=='__main__':unittest.main()
