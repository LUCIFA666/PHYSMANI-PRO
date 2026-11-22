"""Real V8/current public RGBD → original P4 packer, no Native or GT labels."""
from copy import deepcopy
from pathlib import Path
import hashlib,json,os,unittest
import numpy as np
import torch
from our_mani.contracts import Pose
from our_mani.perception import CameraCalibration
from our_mani.local_mpc_control_20261103.public_motion_bridge import PublicRobotSensors
from our_mani.local_mpc_p4_20261103.runtime import pack_public_row, _FORBIDDEN
from our_mani.local_mpc_control_20261103 import public_candidate_position_v63 as old
from our_mani.local_mpc_control_20261103 import public_candidate_position_v64 as new
from our_mani.local_mpc_control_20261103 import depth_position_goal_perception_v64 as tracker

ROOT=Path(__file__).resolve().parents[1]
PUBLIC=ROOT/'runtime/framework/local_mpc_20261102/public_candidate_position_v63_development_v1/00_val_static/public_shadow'
CHECKPOINT=ROOT/'artifacts/framework/near_goal_segmentation_20261103/position_measurement_adaptive_fresh8_v8/calibrated.pt'
CHECKPOINT_SHA='a3a56410e06b0150471500407b0f70d29bbe4dcf7aa540ee08039da95feb721a'


def load_real_public_packet():
    """Four exact public sources; no labels/initial_snapshot/native_ticks read."""
    row=json.loads((PUBLIC/'public_outputs.jsonl').read_text().splitlines()[0])
    raw=PUBLIC/row['rgbd']
    with np.load(raw,allow_pickle=False)as z:
        now=float(z['timestamp'])
        images={n:z['images_'+n].copy()for n in ('front','8','16','36')}
        depths={n:z['depths_'+n].copy()for n in ('front','8','16','36')}
    calibrations=json.loads((PUBLIC/'camera_calibration.json').read_text())
    observation={'timestamp':now,'observation/images':images,'observation/depths':depths,
        'observation/calibrations':{n:CameraCalibration(np.asarray(v['intrinsics']),Pose.from_xyzw(v['camera_to_world']))for n,v in calibrations.items()}}
    robot=PublicRobotSensors.from_mapping(row['public_robot'])
    if robot.timestamp!=now or type(row['close_requested'])is not bool:raise ValueError('actual sameframe packet required')
    return observation,robot,row['close_requested'],row


def forbidden_paths(value,path=()):
    found=[]
    if isinstance(value,dict):
        for key,item in value.items():
            if key.casefold()in _FORBIDDEN or key.casefold().startswith('gt_'):
                found.append('.'.join(path+(key,)))
            found.extend(forbidden_paths(item,path+(key,)))
    elif isinstance(value,(tuple,list)):
        for i,item in enumerate(value):found.extend(forbidden_paths(item,path+(str(i),)))
    return found


class V64ActualPublicPacketTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
            if os.environ.get(name)!='4':raise ValueError('Native policy environment4 required')
        torch.set_num_threads(2)
        cls.observation,cls.robot,cls.close,cls.logged=load_real_public_packet()
        kwargs=dict(refiner_checkpoint=CHECKPOINT,refiner_checkpoint_sha256=CHECKPOINT_SHA)
        cls.before=old.PublicCandidatePositionBridgeV63(**kwargs)
        cls.after=new.PublicCandidatePositionBridgeV64(**kwargs)
        cls.old_result=cls.before.observe(cls.observation,cls.robot,close_requested=cls.close)
        cls.new_result=cls.after.observe(cls.observation,cls.robot,close_requested=cls.close)

    def test_old_real_result_reproduces_exact_existing_forbidden_key_failure(self):
        paths=forbidden_paths(self.old_result)
        self.assertEqual(paths,['depth_refiner_trace.NN_current_measurement.source.GT_or_label_or_robot_or_history_input'])
        with self.assertRaisesRegex(ValueError,'GT_or_label_or_robot_or_history_input'):
            pack_public_row(self.old_result,self.robot,self.close,0,'rgbd/0000.npz')

    def test_whitelist_real_result_packs_with_identical_policy_state(self):
        packed=pack_public_row(self.new_result,self.robot,self.close,0,'rgbd/0000.npz')
        # Test-only removal of the entire private F tree shows pack projection
        # equality. Production V64's whitelist is independently exercised above.
        stripped=dict(self.old_result);stripped['depth_refiner_trace']=dict(self.old_result['depth_refiner_trace'])
        stripped['depth_refiner_trace'].pop('NN_current_measurement')
        reference=pack_public_row(stripped,self.robot,self.close,0,'rgbd/0000.npz')
        self.assertEqual(packed,reference)
        self.assertEqual(forbidden_paths(self.new_result),[])
        self.assertIs(self.after.binding.latest_result,self.new_result)
        self.assertIs(self.after.latest_result,self.new_result)
        self.assertIs(self.new_result['object_observation'],self.after.binding.latest_visual_target)
        for key in ('accepted_source','accepted_history_reset_for_route_change','measurement_available','packed_current_input_sha256'):
            self.assertEqual(self.new_result['depth_refiner_trace'][key],self.old_result['depth_refiner_trace'][key])
        for key in ('proposal_thresholds','threshold','selected_threshold','thresholds_tried','proposal_route','proposal_trace','threshold_change_is_same_position_source'):
            self.assertEqual(self.new_result['depth_refiner_trace']['position_proposals'][key],self.old_result['depth_refiner_trace']['position_proposals'][key])
        for key in ('goal_pose','public_preclose_visual_baseline','attachment_estimate','mode','available','free_forecast_available','controlled_mode_available'):
            if key=='goal_pose':
                np.testing.assert_array_equal(self.new_result[key].position,self.old_result[key].position)
                np.testing.assert_array_equal(self.new_result[key].rotation,self.old_result[key].rotation)
            else:self.assertEqual(self.new_result[key],self.old_result[key])

    def test_full_diagnostics_stay_private_and_unknown_projection_keys_failclosed(self):
        public=self.new_result['depth_refiner_trace']
        for key in ('NN_current_measurement','far_consistency','far_legacy_surface_filter','far_raw_error'):
            self.assertNotIn(key,public)
        private=self.after.tracker.last_measurement_metadata
        self.assertIn('NN_current_measurement',private)
        self.assertFalse(private['NN_current_measurement']['source']['GT_or_label_or_robot_or_history_input'])
        changed=deepcopy(private);changed['unexpected_diagnostic_tree']={'value':False}
        with self.assertRaises((ValueError,TypeError)):tracker.project_public_position_evidence(changed)
        changed=deepcopy(private);changed['position_proposals']['proposal_trace'][0]['unapproved_nested_key']=False
        with self.assertRaises((ValueError,TypeError)):tracker.project_public_position_evidence(changed)

    def test_same_result_identity_and_projection_digest_are_required(self):
        self.after.binding.approve(self.new_result)
        with self.assertRaises((ValueError,TypeError)):self.after.binding.approve(deepcopy(self.new_result))
        trace=self.new_result['depth_refiner_trace'];changed=deepcopy(trace);changed['unexpected_public_extra']=False
        self.new_result['depth_refiner_trace']=changed
        try:
            with self.assertRaises((ValueError,TypeError)):self.after.binding.approve(self.new_result)
        finally:self.new_result['depth_refiner_trace']=trace
        self.after.binding.approve(self.new_result)

if __name__=='__main__':unittest.main()
