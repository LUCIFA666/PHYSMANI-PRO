"""Synthetic public crop contracts only; no actual model/data/Native."""
from copy import deepcopy
import unittest
import numpy as np
from our_mani.near_goal_segmentation_20261103.data import make_near_goal_crops
from our_mani.current_cube_rgbd_centre_20261103_v1.features import make_dual_goal_crops,public_model_arrays


def fixture():
    yy,xx=np.indices((256,256));rgb=np.zeros((4,256,256,3),np.uint8)
    rgb[...,0]=xx;rgb[...,1]=yy
    K=np.repeat(np.array([[[-200.,0,128.],[0,-210.,128.],[0,0,1.]]]),4,axis=0)
    T=np.repeat(np.eye(4)[None],4,axis=0)
    return dict(rgb=rgb,depth_m=np.ones((4,256,256),np.float32),K=K,T=T,
        public_sphere_center=np.array([0.,0.,1.]),camera_names=np.array(['front','8','16','36']),timestamp=np.array(0.))


class DualPublicCrop(unittest.TestCase):
    def test_near_all_arrays_bitexact_original_and_input_immutable(self):
        x=fixture();old=deepcopy(x);a=make_near_goal_crops(x);b=make_dual_goal_crops(x)
        for k,v in a.items():np.testing.assert_array_equal(v,b['near'][k])
        for k,v in x.items():np.testing.assert_array_equal(v,old[k])

    def test_wide_true_stride2_pixels_depth_and_signedK_pairing(self):
        x=fixture();w=make_dual_goal_crops(x)['wide']
        np.testing.assert_array_equal(w['origins_xy'],np.full((4,2),64))
        for j in range(4):
            uv=w['source_pixels_uv'][j];np.testing.assert_array_equal(w['rgb'][j],x['rgb'][j,uv[...,1],uv[...,0]])
            rays=np.concatenate((uv,np.ones((64,64,1))),axis=-1)@np.linalg.inv(x['K'][j]).T
            np.testing.assert_allclose(w['features'][j,4:7],np.moveaxis((rays-[0,0,1])/.1,-1,0),atol=2e-7,rtol=0)
        self.assertEqual(w['stride'],2);self.assertEqual(w['input_window_pixels'],128)

    def test_invalid_image_and_depth_remain_explicit_not_interpolated(self):
        x=fixture();x['K'][:,0,2]=5.;x['K'][:,1,2]=7.;x['depth_m'][:,7,5]=np.nan
        w=make_dual_goal_crops(x)['wide']
        self.assertTrue((~w['valid_crop']).any());self.assertTrue((~w['valid_depth']).any())
        self.assertTrue(np.isfinite(w['features']).all())
        self.assertTrue(np.all(w['features'][:,8]==w['valid_crop']))
        self.assertTrue(np.all(w['features'][:,7]==w['valid_depth']))

    def test_exact_public_whitelist_and_model_arrays_no_targets(self):
        x=fixture()
        with self.assertRaises(ValueError):make_dual_goal_crops(dict(x,cube_center_world=np.zeros(3)))
        values=public_model_arrays(x)
        self.assertEqual(set(values),{'features','wide_features','K','camera_rotations','initial_sphere'})
        self.assertFalse(make_dual_goal_crops(x)['scope_qualified'])


if __name__=='__main__':unittest.main()
