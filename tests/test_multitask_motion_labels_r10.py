"""CPU target-only finite-difference provenance and boundary checks."""
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from our_mani.multitask.motion_labels import backward_pose_supervision,MotionLabelError,RAW_LINEAR,RAW_ANGULAR


def sample(n=8,b=2):
    t=np.arange(n)*.05
    p=np.zeros((n,b,3));p[:,:,0]=t[:,None]*2.
    r=np.broadcast_to(Rotation.from_rotvec(t[:,None]*np.array([0.,0.,.4])).as_matrix()[:,None],(n,b,3,3)).copy()
    return dict(times=t,body_ids=np.asarray(['body'+str(i) for i in range(b)]),target_positions=p,
        target_rotations=r,target_linear_velocities=np.full((n,b,3),9.),
        target_angular_velocities=np.full((n,b,3),-7.),target_valid=np.ones((n,b),bool),
        native_pose_valid=np.ones((n,b),bool),mode_valid=np.ones((n,b),bool),dynamics_segment_valid=np.ones((n,b),bool),
        modes=np.zeros((n,b),np.int8),valid=np.ones((n,b),bool),positions=p+100.,
        linear_velocities=np.full((n,b,3),123.),angular_velocities=np.full((n,b,3),456.))


class MotionLabelChecks(unittest.TestCase):
    def test_constant_motion_preserves_raw_and_history_and_masks_first_frame(self):
        a=sample();before={k:v.copy() for k,v in a.items()}
        out,meta=backward_pose_supervision(a)
        np.testing.assert_allclose(out['target_linear_velocities'][1:,:,0],2.)
        np.testing.assert_allclose(out['target_linear_velocities'][1:,:,1:],0.)
        np.testing.assert_allclose(out['target_angular_velocities'][1:,:,2],.4,atol=1e-12)
        self.assertFalse(out['target_twist_valid'][0].any());self.assertTrue(out['target_twist_valid'][1:].all())
        np.testing.assert_array_equal(out[RAW_LINEAR],before['target_linear_velocities'])
        np.testing.assert_array_equal(out[RAW_ANGULAR],before['target_angular_velocities'])
        for key in a:np.testing.assert_array_equal(a[key],before[key])
        for key in ('positions','valid','linear_velocities','angular_velocities'):
            np.testing.assert_array_equal(out[key],before[key])
        self.assertFalse(meta['history_fields_modified']);self.assertFalse(meta['raw_native_velocity_modified'])

    def test_mode_and_invalid_mode_gaps_break_both_endpoint_derivatives(self):
        a=sample();a['modes'][3:,0]=1;a['mode_valid'][4,1]=False
        out,_=backward_pose_supervision(a)
        self.assertFalse(out['target_twist_valid'][3,0]);self.assertTrue(out['target_twist_valid'][4,0])
        self.assertFalse(out['target_twist_valid'][4,1]);self.assertFalse(out['target_twist_valid'][5,1]);self.assertTrue(out['target_twist_valid'][6,1])
        np.testing.assert_array_equal(out['target_linear_velocities'][4:6,1],np.zeros((2,3)))

    def test_exogenous_segment_boundaries_and_pose_invalidity_are_not_bridged(self):
        a=sample();a['dynamics_segment_valid'][3,0]=False;a['native_pose_valid'][4,1]=False
        out,_=backward_pose_supervision(a)
        self.assertFalse(out['target_valid'][3:5,0].any());self.assertFalse(out['target_valid'][4:6,1].any())
        self.assertTrue(out['target_pose_valid_before_twist_mask'].all());self.assertTrue(a['target_valid'].all())

    def test_unexpected_time_interval_is_masked_without_retiming_or_changing_raw(self):
        a=sample();a['times'][4:]+=.01
        out,meta=backward_pose_supervision(a)
        self.assertFalse(out['target_twist_valid'][4].any());self.assertTrue(out['target_twist_valid'][5].all())
        self.assertEqual(meta['invalid_intervals'],1);np.testing.assert_array_equal(out['times'],a['times'])

    def test_world_angular_increment_uses_current_times_previous_transpose(self):
        a=sample(n=2,b=1);base=Rotation.from_rotvec([.5,0,0]).as_matrix()
        inc=Rotation.from_rotvec([0.,0.,.1]).as_matrix()
        a['target_rotations'][:,0]=[base,inc@base]
        out,_=backward_pose_supervision(a)
        np.testing.assert_allclose(out['target_angular_velocities'][1,0],[0,0,2.],atol=1e-12)

    def test_invalid_declared_GT_pose_is_visible_in_audit_and_never_derived(self):
        a=sample();a['target_rotations'][3,0]=np.zeros((3,3))
        out,meta=backward_pose_supervision(a)
        self.assertEqual(meta['native_pose_marked_valid_but_nonfinite_or_non_SO3'],1)
        self.assertFalse(out['target_twist_valid'][3:5,0].any())

    def test_no_cross_body_differencing_and_second_derivation_rejected(self):
        a=sample();a['target_positions'][:,1,0]=a['times']*-3.
        out,_=backward_pose_supervision(a)
        np.testing.assert_allclose(out['target_linear_velocities'][1:,0,0],2.)
        np.testing.assert_allclose(out['target_linear_velocities'][1:,1,0],-3.)
        with self.assertRaises(MotionLabelError):backward_pose_supervision(out)
        a['body_ids']=np.asarray(['same','same'])
        with self.assertRaises(MotionLabelError):backward_pose_supervision(a)

    def test_existing_dataset_and_fitter_accept_single_conservative_mask_without_history_leak(self):
        import torch
        from tests.test_multitask_tracks_r10 import episode
        from our_mani.multitask.context_se3 import SE3ContextNetwork
        from our_mani.multitask.track_dataset import BodyTrackEpisode,examples
        from our_mani.multitask.fitting import batch_tensors,loss_per_window
        torch.set_num_threads(2)
        source=episode();a=source.arrays
        for name in ('native_pose_valid','mode_valid','dynamics_segment_valid'):a[name]=a['valid'].copy()
        a['target_linear_velocities'][:]=9.;model=SE3ContextNetwork()
        old,_=examples([source],model,steps=7)
        derived,_=backward_pose_supervision(a)
        new,_=examples([BodyTrackEpisode(dict(source.entry),derived)],model,steps=7)
        np.testing.assert_array_equal(old[0]['history'],new[0]['history'])
        np.testing.assert_array_equal(old[0]['state'],new[0]['state'])
        self.assertFalse(np.array_equal(old[0]['targets']['linear_velocities'],new[0]['targets']['linear_velocities']))
        batch=batch_tensors(new,[0,1],torch.device('cpu'))
        with torch.no_grad():loss=loss_per_window(model,batch,steps=7)
        self.assertTrue(torch.isfinite(loss).all())
if __name__=='__main__':unittest.main()
