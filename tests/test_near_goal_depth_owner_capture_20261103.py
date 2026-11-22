"""New depth-owner worker contracts; fake sensors only, no Native or fitting."""
import ast
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys,tempfile,types
import unittest
from unittest.mock import patch
import numpy as np

from our_mani.local_mpc_20261103 import depth_owner_capture as c
from tests.test_near_goal_perception_capture_20261103 import Backend,arrays,shot

ROOT=Path(__file__).resolve().parents[1]


def annotation_fixture():
    shape=(4,256,256);without=np.ones(shape,np.float32);without[:,10,10]+=.02;without[:,30,30]-=.01
    owner=np.zeros(shape,bool);owner[:,10,10]=True
    unknown=np.zeros(shape,bool);unknown[:,30,30]=True
    response=np.zeros(shape,bool);response[:,10,10]=True;response[:,20,20]=True
    sem=np.zeros(shape,np.uint8);sem[owner]=1;sem[:,20,20]=2;sem[unknown]=255
    goal_depth=np.ones(shape,np.float32);goal_depth[:,20,20]+=.03
    return dict(semantic_role_ids=sem,cube_depth_owner=owner,cube_unknown=unknown,
        without_cube_depth_m=without,goal_rgb_response=response,without_goal_depth_m=goal_depth)


def receipt():
    r={k:True for k in('original_clone_equal','global_vision_sensors_enabled','created_nonpassive','handling_restored',
      'all_counterfactual_flags_restored','all_restore_RGBD_equal','clock_unchanged')}
    return dict(r,label_method=c.LABEL_METHOD,warmup_rounds=12,render_warmup_is_physics=False)


def native_clamping_fingers(backend):
    """Model Native float32 interval clipping, not a relaxed restoration check."""
    upper=float(np.float32(.04));original=backend.gripper.set_joint_positions
    backend.gripper.get_joint_intervals=lambda:([False,False],[[0.,upper],[0.,upper]])
    def set_positions(positions,disable_dynamics):
        original(positions,disable_dynamics)
        backend.gripper.q=np.clip(np.asarray(positions,dtype=np.float32).astype(float),0.,upper)
    backend.gripper.set_joint_positions=set_positions
    return upper


class DepthOwnerTests(unittest.TestCase):
    def test_q_readback_clock_bootstrap_source_functions_AST_unchanged(self):
        def functions(path):return {n.name:ast.dump(n,include_attributes=False)for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef)}
        old=functions(ROOT/'our_mani/local_mpc_20261103/perception_capture.py');new=functions(ROOT/'our_mani/local_mpc_20261103/depth_owner_capture.py')
        for name in('checked_plan','archive_capture_sources','sha','pointer','snapshot_pose','zero_clock_shot','assert_close','assert_pose_close','clock','materialize_shots','bootstrap_robot_diagnostics','public_payload','fit_public_sphere'):
            self.assertEqual(old[name],new[name],name)

    def test_owner_is_only_positive_normal_depth_and_goal_response_has_lower_priority(self):
        a=annotation_fixture();result=c.validate_depth_owner_annotations(arrays(),a)
        self.assertEqual(result['semantic_role_ids'][0,10,10],1)
        self.assertTrue(result['goal_rgb_response'][0,10,10])
        self.assertEqual(result['semantic_role_ids'][0,20,20],2)
        self.assertEqual(result['semantic_role_ids'][0,30,30],255)
        bad=deepcopy(a);bad['cube_depth_owner'][:,40,40]=True;bad['semantic_role_ids'][:,40,40]=1
        with self.assertRaisesRegex(c.CaptureInvariantError,'positive normal-renderer'):c.validate_depth_owner_annotations(arrays(),bad)
        bad=deepcopy(a);bad['cube_unknown'][:]=False;bad['semantic_role_ids'][:,30,30]=0
        with self.assertRaisesRegex(c.CaptureInvariantError,'negative/unknown'):c.validate_depth_owner_annotations(arrays(),bad)
        bad=dict(a,instances=np.zeros((4,256,256),np.uint32))
        with self.assertRaisesRegex(c.CaptureInvariantError,'masquerade'):c.validate_depth_owner_annotations(arrays(),bad)
        bad=deepcopy(a);bad['without_goal_depth_m'][:,20,20]=1.
        with self.assertRaisesRegex(c.CaptureInvariantError,'Goal class requires'):c.validate_depth_owner_annotations(arrays(),bad)

    def test_restore_rejection_preserves_private_arrays_but_no_counterfactual_input(self):
        with tempfile.TemporaryDirectory()as td:
            output=Path(td);(output/'labels').mkdir()
            r=receipt();r['all_restore_RGBD_equal']=False
            bundle=types.SimpleNamespace(capture=lambda:(arrays(),annotation_fixture(),r))
            poses={'cube_pose_world':np.array([0,0,.7,0,0,0,1.]),'goal_mesh_pose_world':np.array([.3,0,.8,0,0,0,1.])}
            with patch.object(c,'annotation_labels',side_effect=lambda roles:deepcopy(poses)),self.assertRaisesRegex(c.CaptureInvariantError,'render/restore'):
                c.capture_checked(bundle,{},output,'rejected')
            with np.load(output/'diagnostics_inputs/rejected.npz')as z:self.assertEqual(set(z.files),c.PUBLIC_ARRAY_KEYS)
            with np.load(output/'labels/diagnostics_rejected.npz')as z:
                self.assertIn('without_cube_depth_m',z);self.assertIn('cube_pose_world',z);self.assertNotIn('instances',z)
            self.assertFalse(json.loads((output/'rejected.camera_diagnostic.json').read_text())['native_instance_id_masks'])

    def test_explicit_protocol_fixed12_no_bool_or_material_fallback(self):
        good={'label_method':c.LABEL_METHOD,'native_instance_id_masks':False,'material_or_alpha_edits':False,'render_warmup_rounds':12,
              'baseline_finger_normalization':'source_reset_open_kinematic_before_snapshot'}
        with patch.object(c,'checked_plan',return_value=(good,{},{})):self.assertEqual(c.checked_depth_owner_plan(None,None)[0],good)
        for key,value in [('render_warmup_rounds',True),('render_warmup_rounds',13),('native_instance_id_masks',True),('material_or_alpha_edits',True),('label_method','native_colorcoded'),('baseline_finger_normalization',None)]:
            bad=dict(good,**{key:value})
            with patch.object(c,'checked_plan',return_value=(bad,{},{})),self.assertRaises(ValueError):c.checked_depth_owner_plan(None,None)

    def test_early_sensor_guard_keeps_original_error_with_empty_partial_arrays(self):
        with tempfile.TemporaryDirectory()as td:
            output=Path(td);(output/'labels').mkdir()
            def fail():raise RuntimeError('guard before any buffer')
            bundle=types.SimpleNamespace(capture=fail,last_public_arrays={},last_annotation_arrays={},last_receipt=None)
            with patch.object(c,'annotation_labels',return_value={}),self.assertRaisesRegex(RuntimeError,'guard before any buffer'):
                c.capture_checked(bundle,{},output,'guard')
            r=json.loads((output/'guard.camera_diagnostic.json').read_text())
            self.assertFalse(r['capture_returned']);self.assertFalse(r['accepted_for_training'])

    def test_dynamic_overshoot_would_fail_restore_then_legal_baseline_restores_strictly(self):
        b=Backend();upper=native_clamping_fingers(b);q0=b.arm.q.copy()
        overshoot=np.array([.040001653134822845,.039998043328523636]);b.gripper.q=overshoot.copy()
        s=shot();s['joint_positions']=q0.tolist()
        with self.assertRaisesRegex(c.CaptureInvariantError,'finger_q readback/restoration'):
            with c.zero_clock_shot(b,s,None):pass
        b.gripper.q=overshoot.copy();r=c.normalize_baseline_fingers(b,q0)
        self.assertTrue(r['complete']);self.assertEqual(r['simulation_steps'],0)
        np.testing.assert_array_equal(r['original_actual_state']['measured_finger_positions_m'],overshoot)
        np.testing.assert_array_equal(r['original_outside_native_interval'],[True,False])
        np.testing.assert_array_equal(b.gripper.q,[upper,upper]);self.assertFalse(r['readback_and_restore_tolerance_changed'])
        with c.zero_clock_shot(b,s,None)as actual:pass
        self.assertTrue(actual['restored']);np.testing.assert_array_equal(b.gripper.q,[upper,upper])

    def test_baseline_normalization_rejects_bad_readback_and_hidden_clock_advance(self):
        for failure in ['readback','clock']:
            b=Backend();native_clamping_fingers(b);q0=b.arm.q.copy();r={}
            if failure=='readback':b.gripper.set_joint_positions=lambda positions,disable_dynamics:setattr(b.gripper,'q',np.array([.03,.03]))
            else:b.advance_on_set=True
            with self.assertRaises(c.CaptureInvariantError):c.normalize_baseline_fingers(b,q0,r)
            self.assertFalse(r['complete']);self.assertIn('error',r)

    def test_default_CLI_no_Native(self):
        spec=importlib.util.spec_from_file_location('depth_owner_cli',ROOT/'scripts/collect_near_goal_depth_owner_20261103.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
        with patch.object(m,'check_capture_plan',return_value={'Native_started':False})as check,patch.object(m,'capture_parent',side_effect=AssertionError('Native forbidden')):
            self.assertEqual(m.main(['--plan','unused']),0);check.assert_called_once()

    def test_two_shots_public_inputs_and_cf_labels_fresh_close(self):
        with tempfile.TemporaryDirectory()as td:
            root=Path(td);output=root/c.FAMILY/'depth_canary';source=root/'source.py';source.write_text('# source\n')
            planpath=root/'plan.json';planpath.write_text('{}')
            b=Backend();q0=b.arm.q.copy();b.arm.q+=.002;initial=b.arm.q.copy();upper=native_clamping_fingers(b)
            b.gripper.q=np.array([.039,.04]);b.gripper.target=np.array([.04,.04])
            shots=[shot(),shot('near',True)]
            for s in shots:s['joint_positions']=q0.tolist()
            parent=dict(parent_id='depth_smoke',seed=261103600,fold='fit',shots=shots)
            plan=dict(source_sha256={'source.py':c.sha(source)},external_source_sha256={str(source):c.sha(source)},TRAIN_q_source={'joint_positions':q0.tolist(),'file':{'path':str(source),'sha256':c.sha(source)}},frozen_environment_sha256='frozen',goal_calibration={'path':'goal.json'},native_config={})
            bundle=types.SimpleNamespace(count=0,closed=False)
            def capture():
                if bundle.count:self.assertTrue((output/'labels/materialized_shot_plan.json').is_file())
                bundle.count+=1;return arrays(),annotation_fixture(),receipt()
            bundle.capture=capture
            bundle.close=lambda:setattr(bundle,'closed',True)
            helper=types.ModuleType('our_mani.local_mpc_20261103.depth_owner_capture_sensors')
            helper.create_depth_owner_cameras=lambda backend,warmup_rounds:bundle
            def poses(roles):return {'cube_pose_world':b.task.target_block.get_pose(),'goal_mesh_pose_world':np.array([.3,.1,.9,0,0,0,1.])}
            with patch.object(c,'checked_depth_owner_plan',return_value=(plan,{},parent)),\
                 patch('scripts.run_frozen_system_environment_v820_r10.namespace_context',return_value={'CPUonly':True}),\
                 patch('our_mani.local_mpc_20261102.worker.native_source_snapshot',return_value={'task':'exact'}),\
                 patch('our_mani.local_mpc_20261102.worker.make_backend',return_value=b),\
                 patch('our_mani.local_mpc_20261103.perception_capture_sensors.resolve_annotation_roles',return_value=({},dict(cube=[88],goal=[83]),{})),\
                 patch.object(c,'annotation_labels',side_effect=poses),\
                 patch.object(c,'fit_public_sphere',return_value=({'center':[.3,.1,.9],'radius_m':.04},{})),\
                 patch.dict(sys.modules,{helper.__name__:helper}),patch.dict('os.environ',{'CUDA_VISIBLE_DEVICES':'','DISPLAY':':fake'}):
                result=c.capture_parent(root,planpath,'depth_smoke',output)
            self.assertTrue(result['complete']);self.assertTrue(result['native_closed']);self.assertTrue(bundle.closed)
            self.assertEqual(result['label_method'],c.LABEL_METHOD);self.assertFalse(result['native_instance_id_masks'])
            self.assertEqual(result['controlled_physics_ticks'],0);self.assertFalse(result['data_admitted_for_training'])
            self.assertEqual(bundle.count,3);self.assertEqual(len(result['shots']),2)
            for s in shots:
                with np.load(output/'inputs'/(s['shot_id']+'.npz'))as z:self.assertEqual(set(z.files),c.INPUT_KEYS)
                with np.load(output/'labels'/(s['shot_id']+'.npz'))as z:
                    self.assertIn('without_cube_depth_m',z);self.assertNotIn('instances',z);self.assertEqual(z['semantic_role_ids'][0,10,10],1)
            self.assertTrue((output/'labels/role_bindings.json').is_file());np.testing.assert_array_equal(b.arm.q,initial)
            self.assertTrue((output/'labels/bootstrap_robot_diagnostics_raw.json').is_file())
            norm=json.loads((output/'labels/baseline_finger_normalization.json').read_text());self.assertTrue(norm['complete'])
            np.testing.assert_array_equal(b.gripper.q,[upper,upper])


if __name__=='__main__':unittest.main()
