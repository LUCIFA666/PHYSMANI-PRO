"""CPU-only two-centroid estimator boundaries; no tracker integration."""
from dataclasses import replace
import unittest
import numpy as np

from our_mani.multitask.two_center_pose import (PublicCenter,TwoCenterPosePrior,estimate_two_center_pose)


class TwoCenterPoseTests(unittest.TestCase):
    def setUp(self):
        local=np.array([0.,1.,0.]);world=np.array([1.,0.,0.])
        basis=np.column_stack(([1.,0.,0.],[0.,0.,-1.],local))
        self.R=np.column_stack(([0.,1.,0.],[0.,0.,1.],world))@basis.T
        self.prior=TwoCenterPosePrior('beat_the_rotating_buzz','a'*64,'b'*64,local,world,np.array([.2,.03,0.]),basis,np.eye(3))
        origin=np.array([.3,-.2,.8]);self.centers={
            'frame':PublicCenter(origin,True,True,'external',3.),
            'wand':PublicCenter(origin+self.R@self.prior.local_frame_to_wand_offset,True,True,'external',3.)}

    def test_non_world_Z_axis_two_vector_pose_is_exact_and_grants_no_permission(self):
        result=estimate_two_center_pose(self.prior,self.centers,current_timestamp=3.)
        self.assertTrue(result.valid);np.testing.assert_allclose(result.frame_rotation,self.R,atol=1e-12)
        np.testing.assert_allclose(result.wand_rotation,self.R,atol=1e-12)
        self.assertFalse(result.grants_runtime_full_orientation)
        with self.assertRaises(ValueError):result.frame_rotation[0,0]=0.
        with self.assertRaises(ValueError):self.prior.world_axis[0]=0.

    def test_translation_changes_do_not_change_orientation(self):
        offset=np.array([1.,-.5,.4])
        centers={k:replace(v,position=v.position+offset) for k,v in self.centers.items()}
        np.testing.assert_allclose(estimate_two_center_pose(self.prior,centers,current_timestamp=3.).frame_rotation,self.R,atol=1e-12)

    def test_attachment_release_unknown_mode_or_invalid_public_data_clear_pose(self):
        for field,value in (('mode','attached'),('mode','released'),('mode','unknown'),
                ('mode_valid',False),('valid',False),('position',np.array([np.nan,0.,0.]))):
            centers=dict(self.centers);centers['wand']=replace(centers['wand'],**{field:value})
            result=estimate_two_center_pose(self.prior,centers,current_timestamp=3.)
            with self.subTest(field=field,value=value):
                self.assertFalse(result.valid);self.assertIsNone(result.frame_rotation);self.assertIsNone(result.wand_rotation)
                self.assertIsNone(result.model_sha256)

    def test_stale_future_and_nonfinite_timestamps_reject_without_history_reuse(self):
        self.assertTrue(estimate_two_center_pose(self.prior,self.centers,current_timestamp=3.).valid)
        for timestamp in (2.95,3.05,np.nan):
            centers=dict(self.centers);centers['wand']=replace(centers['wand'],observed_at=timestamp)
            with self.subTest(timestamp=timestamp):
                result=estimate_two_center_pose(self.prior,centers,current_timestamp=3.)
                self.assertFalse(result.valid);self.assertIsNone(result.frame_rotation)

    def test_missing_extra_or_nonpublic_roles_rejected(self):
        for centers in ({'frame':self.centers['frame']},dict(self.centers,other=self.centers['wand']),
                {'frame':self.centers['frame'],'wand':{'native_pose':'forbidden'}}):
            with self.subTest(roles=list(centers)):
                result=estimate_two_center_pose(self.prior,centers,current_timestamp=3.)
                self.assertFalse(result.valid);self.assertIsNone(result.frame_rotation)

    def test_axial_or_coincident_centers_are_degenerate(self):
        for delta in (np.zeros(3),self.prior.world_axis*.2):
            centers=dict(self.centers);centers['wand']=replace(centers['wand'],position=centers['frame'].position+delta)
            with self.subTest(delta=delta):self.assertFalse(estimate_two_center_pose(self.prior,centers,current_timestamp=3.).valid)

    def test_static_axis_basis_rotation_and_asset_identity_are_strict(self):
        for updates in (dict(world_axis=[0.,0.,2.]),dict(local_basis=np.eye(3)),
                dict(frame_to_wand_rotation=np.diag([1.,1.,-1.])),dict(canonical_asset_sha256=''),
                dict(local_frame_to_wand_offset=[0.,1.,0.]),dict(task='unknown_task')):
            with self.subTest(updates=updates),self.assertRaises(ValueError):replace(self.prior,**updates)

    def test_no_hidden_native_or_future_arguments_exist(self):
        with self.assertRaises(TypeError):estimate_two_center_pose(self.prior,self.centers,current_timestamp=3.,native_yaw=1.)
        with self.assertRaises(TypeError):PublicCenter([0.,0.,0.],True,True,'external',3.,native_rotation=np.eye(3))


if __name__=='__main__':unittest.main()
