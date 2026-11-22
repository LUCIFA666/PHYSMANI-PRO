"""Small real-format HDF fixtures; no model, Native or TEST payload."""
from pathlib import Path
import tempfile
import unittest
import numpy as np
import h5py
from our_mani.multitask_data_preparation_20261104_v1.data import decode_depth,episode_window,check_stat,SemanticTaskHistoryDataset
from our_mani.multitask_data_preparation_20261104_v1.manifest import ROOT


def fixture(path):
 n=6;K=np.array([[-200.,0,128],[0,-200.,128],[0,0,1.]])
 with h5py.File(path,'w')as h:
  for cam in ('front','8','16','36'):
   group=h.create_group('cam_params/'+cam);group['intrinsics']=K;group['extrinsics']=np.eye(4);group['near']=.01;group['far']=4.5
   rgb=np.zeros((n,256,256,3),np.uint8);yy,xx=np.indices((256,256));rgb[...,0]=xx;rgb[...,1]=yy
   for t in range(n):rgb[t,...,2]=t
   h['observations/'+cam+'/rgb']=rgb;h['observations/'+cam+'/depth']=np.full_like(rgb,128)
  h['low_dim_obs/joint_positions']=np.arange(n*7,dtype=np.float32).reshape(n,7)/100
  pose=np.zeros((n,7),np.float32);pose[:,0]=np.arange(n)/10;pose[:,6]=1.;h['low_dim_obs/gripper_pose']=pose
  # Contaminated private scalar is deliberately unusable; public read must not touch it.
  h['low_dim_obs/gripper_open']=np.full(n,np.nan,np.float32)
 s=path.stat()
 return dict(source_path=str(path),split='train',offline_eligible=True,frame_count=n,task='beat_the_rotating_buzz',role='fit',source_parent_group='synthetic',source_stat=dict(size=s.st_size,mtime_ns=s.st_mtime_ns,inode=s.st_ino))


class OfficialStageDataContracts(unittest.TestCase):
 def test_depth24_exact_endpoints_no_guess_normalization(self):
  image=np.array([[[0,0,0],[255,255,255]]],np.uint8)
  np.testing.assert_allclose(decode_depth(image,.01,4.5),[[.01,4.5]],rtol=0,atol=1e-7)
 def test_original256_and_causal_prefix_mask_no_contact_or_future_input(self):
  with tempfile.TemporaryDirectory()as tmp:
   row=fixture(Path(tmp)/'data.h5');s=episode_window(row,2,image_size=256,include_labels=False)
   self.assertEqual(s['rgbd'].shape,(5,4,4,256,256));self.assertEqual(int(s['lengths']),3)
   np.testing.assert_array_equal(s['frame_indices'],[0,1,2,-1,-1]);self.assertTrue(np.all(s['robot_history'][:,-1]==0))
   self.assertFalse(s['robot_observed_mask'][:,-1].any());self.assertEqual(s['labels'],{});self.assertFalse(s['physical_clock_verified'])
   np.testing.assert_array_equal(s['rgbd'][2,0,0,0],np.arange(256,dtype=np.float32)/127.5-1.)
   self.assertTrue(np.all(s['rgbd'][3:]==0));self.assertEqual(s['calibration']['stride'],1)
   small=episode_window(row,2,image_size=64,include_labels=False)
   np.testing.assert_array_equal(small['rgbd'][:3],s['rgbd'][:3,...,::4,::4])
 def test_TEST_or_quarantine_payload_rejected_before_open(self):
  for row in (dict(split='test',offline_eligible=False),dict(split='train',offline_eligible=False)):
   with self.assertRaises(ValueError):check_stat(row)
 def test_frozen_manifest_preserves_groups_roles_all_native_denominators(self):
  path=ROOT/'artifacts/framework/local_mpc_20261102/current_network16_data_manifest_v1/manifest.json'
  fit=SemanticTaskHistoryDataset(path,'fit',resolution=256);val=SemanticTaskHistoryDataset(path,'selection',resolution=256)
  self.assertEqual((len(fit.episodes),len(fit)),(1537,3074));self.assertEqual((len(val.episodes),len(val)),(323,646))
  self.assertTrue(all(r['split']!='test'and r['role']=='fit'for r in fit.episodes))
  self.assertEqual(fit.metadata['Native_validation_denominator'],315);self.assertEqual(fit.metadata['Native_TEST_denominator'],640)
  self.assertFalse({r['source_parent_group']for r in fit.episodes}&{r['source_parent_group']for r in val.episodes})

if __name__=='__main__':unittest.main()
