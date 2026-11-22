"""Zero-physics capture contracts with fake hardware only; no Native import."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import h5py
import numpy as np

from our_mani.local_mpc_20261103 import perception_capture as c


class Node:
    def __init__(self, n, backend):
        self.q=np.arange(n,dtype=float)/100; self.target=self.q.copy()
        self.velocity=np.arange(n,dtype=float)/1000;self.target_velocity=np.zeros(n)
        self.pose=np.array([.1,.2,.775,0.,0.,0.,1.]);self.backend=backend;self.calls=[]
    def get_joint_positions(self):return self.q.copy()
    def get_joint_target_positions(self):return self.target.copy()
    def get_joint_velocities(self):return self.velocity.copy()
    def get_joint_target_velocities(self):return self.target_velocity.copy()
    def set_joint_positions(self,q,disable_dynamics):
        if disable_dynamics:raise AssertionError('hidden steps forbidden')
        self.calls.append(('q',False));self.q=np.array(q)
        if self.backend.advance_on_set:self.backend.ticks+=1;self.backend.advance_on_set=False
    def set_joint_target_positions(self,q):self.target=np.array(q)
    def set_joint_target_velocities(self,v):self.target_velocity=np.array(v)
    def get_pose(self):return self.pose.copy()
    def set_pose(self,p,reset_dynamics):
        if reset_dynamics:raise AssertionError('dynamic reset forbidden')
        self.calls.append(('pose',False));self.pose=np.array(p)


class Backend:
    def __init__(self):
        self.ticks=0;self.advance_on_set=False;self.closed=False
        self.arm=Node(7,self);self.arm._ik_target=Node(0,self)
        self.gripper=Node(2,self);self.task=SimpleNamespace(target_block=Node(0,self))
        self.raw=SimpleNamespace(ledger=SimpleNamespace(ticks=0));self.staging_receipt={'controlled_ticks':0,'extra_raw_settle_steps':3}
    def _clock_tuple(self):return (3.25+self.ticks*.05,0.,0,self.raw.ledger.ticks)
    def clock(self):return 0.
    def joint_limits(self):return {'lower':-np.ones(7)*3,'upper':np.ones(7)*3}
    def close(self):self.closed=True
    def step(self):raise AssertionError('step forbidden')
    def actuate_gripper(self,*a):raise AssertionError('actuate/grasp forbidden')


def shot(name='native',near=False):
    result=dict(shot_id=name,cube_mode='public_sphere_offset' if near else 'native_initial',
        joint_positions=np.arange(7).tolist(),finger_positions_m=[.04,.04],finger_source='source_defined_Native_reset_open')
    result['joint_positions']=(np.arange(7)/10).tolist()
    if near:result.update(offset_world_m=[.06,0.,0.],quaternion_xyzw=[0.,0.,0.,1.])
    return result


def arrays():
    return dict(rgb=np.zeros((4,256,256,3),np.uint8),depth_m=np.ones((4,256,256),np.float32),
      camera_intrinsics=np.tile(np.eye(3),(4,1,1)),camera_to_world=np.tile(np.eye(4),(4,1,1)),camera_names=np.array(c.CAMERAS))


class CollectorTests(unittest.TestCase):
    def test_zero_tick_pose_transaction_and_exception_restoration(self):
        b=Backend();before=c.snapshot_pose(b)
        with c.zero_clock_shot(b,shot('near',True),{'center':[.3,.1,.9]}) as receipt:
            np.testing.assert_array_equal(b.task.target_block.pose,[.36,.1,.9,0,0,0,1])
            self.assertEqual(c.clock(b),before['clock'])
        self.assertTrue(receipt['restored']);self.assertFalse(receipt['dynamics_restoration_claimed'])
        for key in before:self.assertEqual(np.asarray(c.snapshot_pose(b)[key],dtype=object).tolist(),np.asarray(before[key],dtype=object).tolist())
        with self.assertRaisesRegex(RuntimeError,'synthetic render'):
            with c.zero_clock_shot(b,shot(),None):raise RuntimeError('synthetic render')
        np.testing.assert_array_equal(b.arm.q,before['arm_q'])
        self.assertTrue(all(v[1] is False for n in (b.arm,b.gripper,b.task.target_block,b.arm._ik_target) for v in n.calls))

    def test_hidden_clock_advance_is_rejected_and_poses_restored(self):
        b=Backend();before=b.arm.q.copy();b.advance_on_set=True
        with self.assertRaisesRegex(c.CaptureInvariantError,'clock'):
            with c.zero_clock_shot(b,shot(),None):pass
        np.testing.assert_array_equal(b.arm.q,before)

    def test_public_npz_exact_loader_contract_and_labels_separate(self):
        from our_mani.near_goal_segmentation_20261103.data import validate_public
        p=c.public_payload(arrays(),{'center':[0.,0.,1.]},0.)
        self.assertEqual(set(p),c.INPUT_KEYS);validate_public(p)
        self.assertNotIn('cube_pose_world',p);self.assertNotIn('counted_tick',p)
        with self.assertRaises(c.CaptureInvariantError):c.public_payload(dict(arrays(),held=True),{'center':[0,0,1]},0)

    def test_camera_reject_preserves_raw_and_privileged_evidence(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);(p/'labels').mkdir()
            receipt={'original_clone_equal':False,'per_camera':{'front':{'depth_max_abs_diff_m':.01}}}
            c.preserve_capture_diagnostic(p,'bad',arrays(),np.zeros((4,256,256),np.uint32),{'cube_pose_world':np.ones(7)},receipt)
            self.assertEqual(json.loads((p/'bad.camera_diagnostic.json').read_text())['camera'],receipt)
            self.assertFalse(json.loads((p/'bad.camera_diagnostic.json').read_text())['accepted_for_training'])
            with np.load(p/'diagnostics_inputs/bad.npz') as f:self.assertEqual(set(f.files),c.PUBLIC_ARRAY_KEYS)
            with np.load(p/'labels/diagnostics_bad.npz') as f:self.assertIn('cube_pose_world',f)

    def test_mask_rgb_depth_mismatch_rejects_and_keeps_label_only_depth(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);(p/'labels').mkdir()
            receipt={k:True for k in('original_clone_equal','global_vision_sensors_enabled','created_nonpassive','handling_restored','mask_rgb_projection_equal')}
            receipt['mask_rgb_depth_equal']=False
            bundle=SimpleNamespace(last_annotation_arrays={'mask_depth_m':np.ones((4,256,256),np.float32)},
              capture=lambda:(arrays(),np.zeros((4,256,256),np.uint32),receipt))
            poses={'cube_pose_world':np.array([0,0,.7,0,0,0,1.]),'goal_mesh_pose_world':np.array([.3,0,.8,0,0,0,1.])}
            with patch.object(c,'annotation_labels',return_value=poses.copy()),self.assertRaisesRegex(c.CaptureInvariantError,'occlusion-depth'):
                c.capture_checked(bundle,{},p,'bad_occlusion')
            with np.load(p/'labels/diagnostics_bad_occlusion.npz') as f:self.assertIn('mask_depth_m',f)
            with np.load(p/'diagnostics_inputs/bad_occlusion.npz') as f:self.assertNotIn('mask_depth_m',f)
            self.assertFalse(json.loads((p/'bad_occlusion.camera_diagnostic.json').read_text())['camera']['mask_rgb_depth_equal'])

    def test_clock_boolean_and_nonfinite_rejected(self):
        b=Backend()
        for value in ((True,0.,0,0),(float('nan'),0.,0,0)):
            with patch.object(b,'_clock_tuple',return_value=value),self.assertRaises(c.CaptureInvariantError):c.clock(b)

    def test_settled_bootstrap_diff_is_diagnostic_but_each_shot_readback_stays_strict(self):
        b=Backend();source=np.arange(7)/10
        b.arm.q=source+.003;b.gripper.q=np.array([.0392,.0396])
        before=c.snapshot_pose(b)
        d=c.bootstrap_robot_diagnostics(b,source)
        self.assertFalse(d['equality_to_unapplied_shot_required'])
        np.testing.assert_allclose(d['measured_q_minus_TRAIN_q0_rad'],.003,atol=1e-16)
        self.assertAlmostEqual(d['max_abs_finger_minus_reset_open_m'],.0008)
        s=shot();s['joint_positions']=source.tolist()
        with c.zero_clock_shot(b,s,None):np.testing.assert_array_equal(b.arm.q,source)
        np.testing.assert_array_equal(b.arm.q,before['arm_q'])
        # The correction does not waive requested-q readback after a setter.
        original=b.arm.set_joint_positions
        calls=[0]
        def biased(q,disable_dynamics):
            original(q,disable_dynamics);calls[0]+=1
            if calls[0]==1:b.arm.q[0]+=.002
        with patch.object(b.arm,'set_joint_positions',side_effect=biased),self.assertRaisesRegex(c.CaptureInvariantError,'q readback'):
            with c.zero_clock_shot(b,s,None):pass
        np.testing.assert_array_equal(b.arm.q,before['arm_q'])

    def test_default_cli_is_cpu_check_only(self):
        path=Path(__file__).resolve().parents[1]/'scripts/collect_near_goal_perception_20261103.py'
        spec=importlib.util.spec_from_file_location('capture_cli_cpu',path);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
        with patch.object(mod,'check_capture_plan',return_value={'Native_started':False}) as check,patch.object(mod,'capture_parent',side_effect=AssertionError('Native forbidden')):
            self.assertEqual(mod.main(['--plan','unused.json']),0);check.assert_called_once()

    def test_strict_plan_pins_q_fingers_sources_and_goal(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);(root/'config/framework').mkdir(parents=True)
            for name in c.REQUIRED_SOURCES:
                p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('{}' if p.suffix=='.json' else '# source\n')
            qpath=root/'native/train_package_compressed/episode0/data.h5';qpath.parent.mkdir(parents=True)
            q=np.arange(7)/10
            with h5py.File(qpath,'w')as h:h['low_dim_obs/joint_positions']=q[None]
            native=root/'native'
            cfgpath=root/'config/framework/train.json';cfgpath.write_text(json.dumps(dict(dataset_use='train',task=c.TASK,preserve_native_height=True,calibration_train_episode=str(qpath),native_repo=str(native))))
            for p in c.REQUIRED_EXTERNAL:
                external=native/p;external.parent.mkdir(parents=True,exist_ok=True);external.write_text('# actual installed\n')
            shots=[shot(),shot('near',True)]
            plan=dict(schema=c.SCHEMA,split='train',perception_only=True,
              **{k:False for k in('benchmark_episode','execution_transition','task_success_supervision','Native_started','fit_started','TEST_read')},
              native_config={'path':str(cfgpath),'sha256':c.sha(cfgpath)},
              TRAIN_q_source={'file':{'path':str(qpath),'sha256':c.sha(qpath)},'dataset':'low_dim_obs/joint_positions','index':0,'joint_positions':q.tolist()},
              goal_calibration={'path':'artifacts/framework/local_policy/goal_marker_calibration.json','sha256':c.sha(root/'artifacts/framework/local_policy/goal_marker_calibration.json')},
              parents=[{'parent_id':'smoke','seed':261103600,'fold':'fit','shots':shots}],
              source_sha256={p:c.sha(root/p) for p in c.REQUIRED_SOURCES|{'config/framework/train.json'}},external_source_sha256={str(native/p):c.sha(native/p) for p in c.REQUIRED_EXTERNAL})
            planpath=root/'config/framework/smoke.json'
            def put(v):planpath.write_text(json.dumps(v))
            put(plan);self.assertFalse(c.check_capture_plan(root,planpath,'smoke')['Native_started'])
            for change in ('fingers','q','sources','test'):
                bad=deepcopy(plan)
                if change=='fingers':bad['parents'][0]['shots'][1]['finger_source']='fictional HDF fingers'
                if change=='q':bad['parents'][0]['shots'][0]['joint_positions'][0]+=.01
                if change=='sources':bad['source_sha256']={}
                if change=='test':bad['split']='test'
                put(bad)
                with self.assertRaises((ValueError,c.CaptureInvariantError)):c.check_capture_plan(root,planpath,'smoke')

    def test_bootstrap_materializes_all_poses_before_two_shots_and_closes(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);output=root/c.FAMILY/'canary';source=root/'source.py';source.write_text('# frozen\n')
            planpath=root/'plan.json';planpath.write_text('{}')
            b=Backend();q0=b.arm.q.copy();shots=[shot(),shot('near',True)]
            for s in shots:s['joint_positions']=q0.tolist()
            # Real initialization may have settled away from the requested q0;
            # bootstrap remains public evidence, then each shot applies q0.
            b.arm.q=q0+.002;b.gripper.q=np.array([.0392,.0396]);actual_bootstrap_q=b.arm.q.copy()
            parent={'parent_id':'smoke','seed':261103600,'fold':'fit','shots':shots}
            plan={'source_sha256':{'source.py':c.sha(source)},'external_source_sha256':{str(source):c.sha(source)},
              'TRAIN_q_source':{'joint_positions':q0.tolist(),'file':{'path':str(source),'sha256':c.sha(source)}},'frozen_environment_sha256':'fixed','goal_calibration':{'path':'goal.json'},'native_config':{}}
            b.gripper.target=np.array([.04,.04])
            bundle=SimpleNamespace(count=0,closed=False,last_annotation_arrays={'mask_depth_m':np.ones((4,256,256),np.float32)})
            def capture():
                if bundle.count==0:self.assertFalse((output/'labels/materialized_shot_plan.json').exists())
                else:
                    frozen=json.loads((output/'labels/materialized_shot_plan.json').read_text())
                    self.assertEqual(len(frozen['shots']),2);self.assertTrue(frozen['derived_before_any_kinematic_shot'])
                bundle.count+=1
                return arrays(),np.zeros((4,256,256),np.uint32),{k:True for k in('original_clone_equal','global_vision_sensors_enabled','created_nonpassive','handling_restored','mask_rgb_projection_equal','mask_rgb_depth_equal')}
            bundle.capture=capture
            def close():bundle.closed=True
            bundle.close=close
            def labels(roles):return {'cube_pose_world':b.task.target_block.get_pose(),'goal_mesh_pose_world':np.array([.3,.1,.9,0,0,0,1.])}
            with patch.object(c,'checked_plan',return_value=(plan,{},parent)),\
                 patch('scripts.run_frozen_system_environment_v820_r10.namespace_context',return_value={'CPU_only':True}),\
                 patch('our_mani.local_mpc_20261102.worker.native_source_snapshot',return_value={'task':'fixed'}),\
                 patch('our_mani.local_mpc_20261102.worker.make_backend',return_value=b),\
                 patch('our_mani.local_mpc_20261103.perception_capture_sensors.create_capture_cameras',return_value=bundle),\
                 patch('our_mani.local_mpc_20261103.perception_capture_sensors.resolve_annotation_roles',return_value=({},dict(cube=[1],goal=[2]),{})),\
                 patch.object(c,'annotation_labels',side_effect=labels),\
                 patch.object(c,'fit_public_sphere',return_value=({'center':[.3,.1,.9],'radius_m':.04},{'Native_pose_used':False})),\
                 patch.dict('os.environ',{'CUDA_VISIBLE_DEVICES':'','DISPLAY':':cpu'}):
                result=c.capture_parent(root,planpath,'smoke',output)
            self.assertEqual(bundle.count,3);self.assertTrue(bundle.closed and b.closed)
            self.assertTrue(result['complete']);self.assertTrue(result['source_unchanged'])
            self.assertEqual(result['controlled_physics_ticks'],0);self.assertFalse(result['data_admitted_for_training'])
            self.assertEqual(len(result['shots']),2)
            for s in shots:
                with np.load(output/'inputs'/(s['shot_id']+'.npz')) as f:self.assertEqual(set(f.files),c.INPUT_KEYS)
            self.assertTrue((output/'labels/staging_receipt.json').is_file())
            diag=json.loads((output/'labels/bootstrap_robot_diagnostics.json').read_text())
            self.assertAlmostEqual(diag['max_abs_q_minus_TRAIN_q0_rad'],.002)
            np.testing.assert_array_equal(b.arm.q,actual_bootstrap_q)


if __name__=='__main__':unittest.main()
