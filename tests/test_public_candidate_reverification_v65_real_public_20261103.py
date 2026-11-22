"""Actual frozen V8/public RGBD -> V65 -> original P4 packer, no Native/GT/TEST.

The full replays keep old development robot trajectories and actual h records.
They are observer integration checks, not counterfactual closed-loop trials.
"""
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import hashlib,json,os,unittest
import numpy as np
import torch
from our_mani.contracts import Pose
from our_mani.perception import CameraCalibration
from our_mani.local_mpc_control_20261103.active_visual_attachment import ActiveVisualConfig
from our_mani.local_mpc_control_20261103.public_motion_bridge import PublicRobotSensors
from our_mani.local_mpc_control_20261103 import public_candidate_position_v64 as old
from our_mani.local_mpc_control_20261103 import public_candidate_reverification_v65 as new
from our_mani.local_mpc_p4_20261103.runtime import pack_public_row, PublicRowCache
from tests.test_public_candidate_position_v64_independent_20261103 import forbidden_paths

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'runtime/framework/local_mpc_20261102/public_candidate_position_v64_development_v1'
REFINER=ROOT/'artifacts/framework/near_goal_segmentation_20261103/position_measurement_adaptive_fresh8_v8/calibrated.pt'
REFINER_SHA='a3a56410e06b0150471500407b0f70d29bbe4dcf7aa540ee08039da95feb721a'
PILOT=ROOT/'artifacts/framework/local_mpc_20261102/P4_candidate_training_v1/final.pt'
PILOT_SHA='61e5827b4838649b8e477048bea56fcace18697f906fb610a3f963bba2437f60'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def bridge_kwargs():return dict(refiner_checkpoint=REFINER,refiner_checkpoint_sha256=REFINER_SHA,
    active_config=ActiveVisualConfig(enforce_probe_anchor_orientation_after_confirmation=False))
def load_case(name):
    case=BASE/name
    if json.loads((case/'worker_exit.json').read_text())['native_closed'] is not True:raise ValueError('Only already-closed development public records')
    public=case/'public_shadow';rp=public/'public_outputs.jsonl';dp=case/'public_candidate_position_v64_decisions.jsonl'
    rows=[json.loads(s)for s in rp.read_text().splitlines()]
    decisions={round(x['time'],8):x for x in map(json.loads,dp.read_text().splitlines())}
    cp=public/'camera_calibration.json';cal=json.loads(cp.read_text())
    cams={n:CameraCalibration(np.asarray(v['intrinsics']),Pose.from_xyzw(v['camera_to_world']))for n,v in cal.items()}
    return case,public,rows,decisions,cams,{str(p):sha(p)for p in (rp,dp,cp)}
def packet(public,row,cams):
    p=public/row['rgbd']
    with np.load(p,allow_pickle=False)as z:
        now=float(z['timestamp']);images={n:z['images_'+n].copy()for n in ('front','8','16','36')};depths={n:z['depths_'+n].copy()for n in ('front','8','16','36')}
    if now!=row['public_robot']['timestamp']:raise ValueError('Recorded public sameframe input required')
    return {'timestamp':now,'observation/images':images,'observation/depths':depths,'observation/calibrations':cams}


def replay(name):
    case,public,rows,decisions,cams,sources=load_case(name)
    bridge=new.PublicCandidateReverificationBridgeV65(**bridge_kwargs());cache=PublicRowCache()
    admissions=[];epoch_frames=[];failures=[];old_identity=None;fixed=None;epoch_identity=None;reappeared_after_failed=False
    for index,row in enumerate(rows):
        obs=packet(public,row,cams);inner=PublicRobotSensors.from_mapping(row['public_robot'])
        result=bridge.observe(obs,inner,close_requested=row['close_requested'])
        outer=PublicRobotSensors.from_mapping(row['public_robot'])
        assert outer is not inner and bridge._inner_robot is inner
        assert bridge.binding.latest_result is result and bridge.latest_result is result
        assert bridge._frame['object_observation'] is bridge.tracker.last_current_target
        assert (bridge.cache._pending is None or bridge.cache._pending['consumed'] is True)
        assert bridge.cache._last_staged==inner.timestamp and bridge.cache.discarded_unconsumed_frames==0
        bridge.binding.approve(result)
        packed=pack_public_row(result,outer,row['close_requested'],index,row['rgbd']);cache.observe(packed)
        assert cache.latest['index']==index and forbidden_paths(result)==[]
        assert all(key not in result for key in ('witness','epoch_owner','committed','trace','private_diagnostics'))
        if bridge.witness is not None:
            p=bridge.witness
            if old_identity is None:
                old_identity=id(p.old);fixed=(p.r0.copy(),p.object_anchor.copy(),p.original_preclose_baseline.pose.position.copy(),deepcopy(p.old.proof))
            assert id(p.old)==old_identity and p.old.failed is True
            assert np.array_equal(p.old.r0,fixed[0])and np.array_equal(p.old.object_anchor,fixed[1])
            assert np.array_equal(bridge.close_visual_baseline.pose.position,fixed[2])and p.old.proof==fixed[3]
            if p.witness_failed and result['object_observation'] is not None:reappeared_after_failed=True
        if bridge.epoch_owner is not None:
            owner=bridge.epoch_owner
            if epoch_identity is None:epoch_identity=id(owner)
            assert id(owner)==epoch_identity and owner.last_timestamp==inner.timestamp
            assert result['public_reverification']['current_reverified_position_maintenance_passed']==owner.latest['current_reverified_position_maintenance_passed']
            if owner.latest['evidence'] is not None:assert owner.latest['evidence'].timestamp==inner.timestamp
            epoch_frames.append(dict(time=inner.timestamp,passed=owner.latest['current_reverified_position_maintenance_passed'],failed=owner.failed))
        decision=decisions.get(round(inner.timestamp,8))
        if decision is not None:
            h=decision['actuator_state_before']
            admission=bridge.try_begin_natural_witness(result,outer,h)
            if admission is not None:admissions.append(admission)
            # Reproduce the original recorded initial probe admission only.
            # No new action, IK, optimizer or imagined state is generated.
            original_probe=decision['decision'].get('probe')
            if original_probe is not None and abs(original_probe['trigger_time']-inner.timestamp)<1e-8 and bridge.active.spec is None:
                spec=bridge.active.begin_probe(bridge.binding.latest_visual_target,inner,h)
                assert spec is not None and spec['identity']==original_probe['identity']
        if bridge.witness is not None and bridge.witness.first_failure is not None:failures.append(deepcopy(bridge.witness.first_failure))
    return dict(frames=len(rows),admissions=admissions,epoch_frames=epoch_frames,
        witness_failed=False if bridge.witness is None else bridge.witness.witness_failed,
        witness_failure=None if bridge.witness is None else bridge.witness.first_failure,
        final_epoch=bridge.verification_epoch,old_failed=False if bridge.witness is None else bridge.witness.old.failed,
        one_epoch_owner_identity=epoch_identity is not None,reappearance_after_failed_witness=reappeared_after_failed,
        cache_final=cache.receipt(),source_sha256=sources,
        public_packet_count=len(rows),stage_discard_count=bridge.cache.discarded_unconsumed_frames,
        final_private_diagnostics_present=bridge.private_diagnostics()['witness'] is not None)


class OneFrameBackend:
    """CPU public-frame source; placeholder label fields are never read as inputs."""
    def __init__(self,observation,robot,close):
        self.observation=observation;self.now=robot.timestamp
        self.state=dict(time=robot.timestamp,tcp_position=robot.tcp_position,tcp_quaternion=robot.tcp_quaternion,
            q=robot.q,dq=robot.dq,finger_open_amounts=robot.finger_open_amounts,requested_opening=0. if close else 1.,
            object_position=np.zeros(3),object_quaternion=np.array([0.,0.,0.,1.]),object_linear_velocity=np.zeros(3),
            object_angular_velocity=np.zeros(3),held=False,native_success=False)
    def clock(self):return self.now
    def read_state(self):return self.state
    def observe_rgbd(self):return self.observation


class V65RealPublicTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if any(os.environ.get(k)!='4'for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')):raise ValueError('Native policy env4 required')
        torch.set_num_threads(2)
        _,public,rows,_,cams,_=load_case('00_val_static');row=rows[0]
        obs=packet(public,row,cams);robot=PublicRobotSensors.from_mapping(row['public_robot'])
        cls.temp=TemporaryDirectory(prefix='v65_real_public_');cls.addClassCleanup(cls.temp.cleanup)
        cls.observer=new.ObservedPublicCandidateReverificationV65Backend(OneFrameBackend(obs,robot,row['close_requested']),
            Path(cls.temp.name)/'observer',checkpoint=PILOT,checkpoint_sha256=PILOT_SHA,save_frames=False,**bridge_kwargs())
        cls.first_result=cls.observer.sample()
        baseline=old.PublicCandidatePositionBridgeV64(**bridge_kwargs())
        cls.old_first=baseline.observe(obs,robot,close_requested=row['close_requested'])
        cls.original_robot=robot;cls.original_close=row['close_requested']
        cls.replays={name:replay(name)for name in ('00_val_static','uniform_002')}

    def test_full_sample_outer_inner_robot_and_original_pack_state_exact(self):
        outer=self.observer.latest_robot;inner=self.observer.bridge._inner_robot
        self.assertIsNot(outer,inner);self.assertTrue(new._same_robot_values(outer,inner))
        self.assertEqual(self.observer.samples,1)
        self.assertEqual(self.observer.candidate_cache.latest['index'],0)
        before=pack_public_row(self.old_first,self.original_robot,self.original_close,0,'rgbd/0000.npz')
        after=pack_public_row(self.first_result,outer,self.original_close,0,'rgbd/0000.npz')
        self.assertEqual(before,after)
        self.assertEqual(forbidden_paths(self.first_result),[])

    def test_all_real_frames_pack_bind_and_consume_once(self):
        for name,row in self.replays.items():
            with self.subTest(case=name):
                self.assertEqual(row['frames'],row['public_packet_count'])
                self.assertEqual(row['stage_discard_count'],0)
                self.assertEqual(len(row['admissions']),1)
                self.assertEqual(row['admissions'][0]['timestamp'],3.3000000000000003)
                self.assertTrue(row['old_failed'])
                self.assertTrue(row['final_private_diagnostics_present'])

    def test_natural_epoch_has_current50ms_owner_not_duplicate_commit_frame(self):
        row=self.replays['00_val_static'];self.assertEqual(row['final_epoch'],1)
        self.assertTrue(row['one_epoch_owner_identity'])
        self.assertGreater(len(row['epoch_frames']),1)
        self.assertTrue(row['epoch_frames'][0]['passed'])
        for before,after in zip(row['epoch_frames'],row['epoch_frames'][1:]):self.assertAlmostEqual(after['time']-before['time'],.05)
        failed=[i for i,x in enumerate(row['epoch_frames'])if x['failed']]
        if failed:self.assertTrue(all(not x['passed']for x in row['epoch_frames'][failed[0]:]))

    def test_failed_natural_witness_is_not_recreated_on_reappearance(self):
        row=self.replays['uniform_002'];self.assertEqual(row['final_epoch'],0)
        self.assertTrue(row['witness_failed']);self.assertTrue(row['reappearance_after_failed_witness'])
        self.assertEqual(row['witness_failure']['time'],3.45)
        self.assertEqual(len(row['admissions']),1)
        self.assertEqual(row['epoch_frames'],[])


if __name__=='__main__':unittest.main()
