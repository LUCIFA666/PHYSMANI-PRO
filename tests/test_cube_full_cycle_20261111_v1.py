"""Full source coverage, TEST admission, archive staging and public-frame evidence."""
import copy
import io
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from our_mani.cube_full_cycle_20261111_v1.common import plan,parents
from our_mani.cube_full_cycle_20261111_v1.datasets import capture
from our_mani.cube_full_cycle_20261111_v1.backends import ArchiveContactCubeBackend,verify_archive_source
from our_mani.cube_full_cycle_20261111_v1.frame_recording import capture_evidence

class FullCycleContracts(unittest.TestCase):
    def test_all_original_source_roles_and_variations(self):
        p=plan()
        self.assertEqual([len(parents(p,r))for r in('fit','selection','validation','test')],[160,40,40,80])
        self.assertEqual(len({(c['source_split'],c['task'],c['raw_episode_id'])for c in p['cases']}),320)
        groups=[set(parents(p,r))for r in('fit','selection','validation','test')]
        for i,a in enumerate(groups):
            for b in groups[:i]:self.assertFalse(a&b)
        for task in p['tasks']:
            for variation in (0,1):
                rows=[c for c in p['cases']if c['task']==task and c['variation']==variation]
                self.assertEqual([sum(c['role']==r for c in rows)for r in('fit','selection','validation','test')],[40,10,10,20])
    def test_training_compiler_rejects_VAL_and_TEST_before_open(self):
        p=plan()
        for role in('validation','test'):
            c=next(c for c in p['cases']if c['role']==role)
            with self.assertRaisesRegex(ValueError,'No validation/test'):capture(p,c)
    def test_TEST_payload_gate_requires_complete_VAL_first(self):
        with patch('our_mani.cube_full_cycle_20261111_v1.backends.read',return_value={'complete':False})as read:
            with self.assertRaisesRegex(ValueError,'Complete full40'):verify_archive_source({})
            self.assertEqual(read.call_count,1)
            self.assertTrue(str(read.call_args.args[0]).endswith('validation_release.json'))
    def test_archived_reset_has_no_added_home_staging_or_physics(self):
        b=ArchiveContactCubeBackend.__new__(ArchiveContactCubeBackend)
        b._closed=False;b.tick_records=[];b.motion_events=[];b.archive_row={'raw_episode_id':'episode1'}
        called=[]
        b.raw=SimpleNamespace(_requested_opening=1.,reset_receipt={'author_reset':True},reset_official=lambda row:called.append(row))
        b.task=SimpleNamespace(t=.15);b._physics_clock=lambda:1.25;b._sample_native_result=lambda:None
        b._geometry_snapshot=lambda:{};b.joint_limits=lambda:{};b.read_state=lambda:{'time':0.}
        b.source_case={'raw_episode_id':'episode1','config':{'task_time_limit_seconds':60.}};b.task_name='test'
        b.reset_scenario()
        self.assertEqual(called,[b.archive_row]);self.assertEqual(b._physics_origin,1.25)
        self.assertEqual(b._task_origin,.15);self.assertEqual(b.staging_receipt['extra_raw_settle_steps'],0)
        self.assertFalse(b.staging_receipt['additional_robot_repositioning'])
        self.assertEqual(b.initial_snapshot['task_budget']['budget_ticks'],1200)
    def test_saved_npz_roundtrip_preserves_actual_frame_digest(self):
        o={'timestamp':.05,'observation/images':{'front':np.arange(12,dtype=np.uint8).reshape(2,2,3)},'observation/depths':{'front':np.ones((2,2),np.float32)}}
        expected=capture_evidence(o,20,'calibration')['rgbd_sha256']
        arrays={'timestamp':np.asarray(.05),'images_front':o['observation/images']['front'],'depths_front':o['observation/depths']['front']}
        f=io.BytesIO();np.savez_compressed(f,**arrays);f.seek(0)
        with np.load(f,allow_pickle=False)as saved:
            reconstructed={'timestamp':float(saved['timestamp'])}
            for group in('images','depths'):
                reconstructed['observation/'+group]={name.removeprefix(group+'_'):saved[name]for name in saved.files if name.startswith(group+'_')}
            self.assertEqual(capture_evidence(reconstructed,20,'calibration')['rgbd_sha256'],expected)
    def test_every_actual_pixel_and_depth_changes_frame_digest(self):
        o={'timestamp':.05,'observation/images':{'front':np.zeros((2,2,3),np.uint8)},'observation/depths':{'front':np.ones((2,2),np.float32)}}
        a=capture_evidence(o,1,'calibration');rgb=copy.deepcopy(o);rgb['observation/images']['front'][0,0,0]=1
        dep=copy.deepcopy(o);dep['observation/depths']['front'][0,0]=2
        self.assertNotEqual(a['rgbd_sha256'],capture_evidence(rgb,1,'calibration')['rgbd_sha256'])
        self.assertNotEqual(a['rgbd_sha256'],capture_evidence(dep,1,'calibration')['rgbd_sha256'])
        self.assertFalse(a['raw_rgbd_saved']);self.assertTrue(capture_evidence(o,20,'calibration')['raw_rgbd_saved'])
        self.assertEqual(a['timestamp'],.05);self.assertFalse(a['Native_target_pose_or_future_used'])

if __name__=='__main__':unittest.main()
