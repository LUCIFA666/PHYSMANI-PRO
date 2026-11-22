import tempfile
from pathlib import Path
from types import SimpleNamespace,ModuleType
from unittest import mock
import unittest
import numpy as np

from our_mani.physmani_feedback_interface_20261104_v1.backend import PhysManiFeedbackBackendV1
from our_mani.physmani_feedback_interface_20261104_v1.observation import (
    validate_feedback_snapshot,legacy_policy_packet,pack_feedback_snapshot)
from our_mani.generic_reference_mpc_20261104_v1.public_observation import pack_public_observation


class Camera:
    def __init__(self,name='front',mutate=None):
        self.name=name;self.mutate=mutate;self.handles=0;self.matrix=np.eye(4);self.matrix[0,3]=.25
        self.mode=SimpleNamespace(name='OPENGL3'if name=='front'else'OPENGL')
    def handle_explicitly(self):
        self.handles+=1
        if self.mutate:self.mutate()
    def capture_rgb(self):return np.full((256,256,3),.5,dtype=np.float32)
    def capture_depth(self,in_meters=False):
        assert in_meters is True
        return np.full((256,256),.75,dtype=np.float32)
    def get_intrinsic_matrix(self):return np.array([[-256.,0,128],[0,-256.,128],[0,0,1.]])
    def get_matrix(self):return self.matrix.copy()
    def get_near_clipping_plane(self):return .01
    def get_far_clipping_plane(self):return 4.
    def get_render_mode(self):return self.mode
    def set_resolution(self,x):assert x==[256,256]
    def set_explicit_handling(self,x):assert x is True
    def set_render_mode(self,x):self.mode=x
    def set_matrix(self,x):
        assert np.asarray(x).shape==(4,4),'must preserve real PyRep T4'
        self.matrix=np.asarray(x).copy()
    def set_perspective_angle(self,x):assert np.isfinite(x)
    def set_near_clipping_plane(self,x):assert x==.01
    def set_far_clipping_plane(self,x):assert x==4.


def backend_fixture():
    state=SimpleNamespace(physics=7.2)
    b=PhysManiFeedbackBackendV1.__new__(PhysManiFeedbackBackendV1)
    b._closed=False;b._fault=None;b._ticks=0;b._origin=(7.2,1.3)
    b._physics_clock=lambda:state.physics;b.task=SimpleNamespace(t=1.3)
    b.cameras={name:Camera(name)for name in ('front','8','16','36')}
    b.arm=SimpleNamespace(get_joint_positions=lambda:np.arange(7)*.1,
        get_joint_velocities=lambda:np.arange(7)*.01,
        get_tip=lambda:SimpleNamespace(get_pose=lambda:[.1,.2,.8,0,0,0,1]))
    b.gripper=SimpleNamespace(get_joint_positions=lambda:[.040002,.03999],
        get_joint_velocities=lambda:[.002,-.001],get_open_amount=lambda:[1.00005,.9997])
    return b,state


class ObservationBackend(unittest.TestCase):
    def test_actual_T4_and_all_mechanical_fields_zero_physics_and_legacy_ABI(self):
        b,state=backend_fixture();p=b.read_snapshot();validate_feedback_snapshot(p)
        self.assertEqual(state.physics,7.2);self.assertEqual(b.task.t,1.3)
        self.assertEqual([cam.handles for cam in b.cameras.values()],[1]*4)
        self.assertEqual(p['T'].shape,(4,4,4));self.assertEqual(p['T'][0,0,3],.25)
        np.testing.assert_array_equal(p['finger_joint_positions'],[.040002,.03999])
        np.testing.assert_array_equal(p['finger_joint_velocities'],[.002,-.001])
        self.assertFalse(p['finger_open_amounts_within_nominal']);self.assertGreater(p['finger_open_amounts'][0],1.)
        self.assertFalse(p['q'].flags.writeable)
        old=legacy_policy_packet(p)
        original=pack_public_observation(timestamp=0,rgb=p['rgb'],depth_m=p['depth_m'],K=p['K'],T=p['T'],
            camera_clips=p['camera_clips'],q=p['q'],tcp_pose=p['tcp_pose'])
        self.assertEqual(set(old),set(original));self.assertEqual(old['robot15'][-1],0.)
        self.assertFalse(old['robot15_observed_mask'][-1]);self.assertNotIn('dq',old)
        for key,value in original.items():
            if isinstance(value,np.ndarray):np.testing.assert_array_equal(old[key],value)

    def test_camera_read_cannot_advance_any_clock_or_accept_T3(self):
        b,state=backend_fixture();b.cameras['16'].mutate=lambda:setattr(state,'physics',state.physics+.05)
        with self.assertRaises(RuntimeError):b.read_snapshot()
        b,state=backend_fixture();b.cameras['front'].matrix=np.eye(4)[:3]
        with self.assertRaises(ValueError):b.read_snapshot()

    def test_private_fields_and_mismatched_measured_clock_rejected(self):
        b,_=backend_fixture();p=b.read_snapshot()
        with self.assertRaises(ValueError):validate_feedback_snapshot(dict(p,Native_held=False))
        bad=dict(p,task_timestamp=99.)
        with self.assertRaises(ValueError):validate_feedback_snapshot(bad)
        with self.assertRaises(ValueError):validate_feedback_snapshot(dict(p,timestamp=True))

    def test_false_execute_refuses_before_files_or_native_access(self):
        class Forbidden:
            def __fspath__(self):raise AssertionError('path accessed before permission guard')
        with self.assertRaises(ValueError):
            PhysManiFeedbackBackendV1('not-a-task',Forbidden(),seed=1,variation=0,budget_ticks=2)

    def test_actual_camera_config_setter_getter_uses_full_T4(self):
        import h5py
        cams=[]
        class Vision:
            def __new__(cls,name):camera=Camera(name);cams.append(camera);return camera
            @classmethod
            def create(cls,*args,**kwargs):camera=Camera('side');cams.append(camera);return camera
        vision=ModuleType('pyrep.objects.vision_sensor');vision.VisionSensor=Vision
        const=ModuleType('pyrep.const');const.RenderMode=SimpleNamespace(
            OPENGL=SimpleNamespace(name='OPENGL'),OPENGL3=SimpleNamespace(name='OPENGL3'))
        with tempfile.TemporaryDirectory()as tmp:
            path=Path(tmp)/'synthetic.h5'
            with h5py.File(path,'w')as h:
                for name in ('front','8','16','36'):
                    g=h.create_group('cam_params/'+name);g['intrinsics']=Camera().get_intrinsic_matrix()
                    T=np.eye(4);T[2,3]=.3;g['extrinsics']=T;g['near']=.01;g['far']=4.
            b=PhysManiFeedbackBackendV1.__new__(PhysManiFeedbackBackendV1);b.renderer_profile='legacy_front_GL3_side_GL'
            with mock.patch.dict('sys.modules',{'pyrep.objects.vision_sensor':vision,'pyrep.const':const}):
                b._configure_cameras(path)
            self.assertEqual(len(cams),4)
            for cam in cams:self.assertEqual(cam.get_matrix().shape,(4,4));self.assertEqual(cam.get_matrix()[2,3],.3)


if __name__=='__main__':unittest.main()
