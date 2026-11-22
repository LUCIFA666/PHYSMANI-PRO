"""Candidate recorder tests use numeric observations and no simulator imports."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from our_mani.contracts import Pose, RobotState
from our_mani.perception import CameraCalibration
from our_mani.multitask import buzz_visual_capture as capture


def observation(timestamp=0.):
    images={name:np.full((256,256,3),17,np.uint8) for name in capture.CAMERAS}
    depths={name:np.zeros((256,256),np.float32) for name in capture.CAMERAS}
    depths['front'][0,0]=np.nan
    depths['front'][1,1]=.25
    raw={name:(images[name].copy() if name=='front' else images[name].astype(np.float32)+.25)
         for name in capture.CAMERAS}
    masks={name:np.full((256,256,3),.125,np.float32) for name in capture.CAMERAS}
    calib={name:CameraCalibration(np.array([[200.,0,128],[0,-200.,128],[0,0,1]]),
             Pose([.1,.2,.3]),depth_scale=1.,timestamp_offset=0.) for name in capture.CAMERAS}
    data={'timestamp':timestamp,'observation/images':images,'observation/depths':depths,
          'observation/raw_rgb_255':raw,'observation/instance_masks':masks,
          'observation/calibrations':calib,
          'observation/sensor_protocol':{'version':'fixture_public_sensor_metadata',
              'camera_render_modes':{name:'OPENGL' for name in capture.CAMERAS},
              'rgb_processing':{'front':'floor','others':'native float32*255'}}}
    robot=RobotState(timestamp,Pose([0,0,0]),np.zeros(3),np.zeros(3),np.zeros(7),np.zeros(7),task_success=True)
    return data,robot


class BuzzVisualCaptureTests(unittest.TestCase):
    def setUp(self):
        capture.NAMESPACE.mkdir(parents=True,exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(prefix='CPU_fixture_',dir=capture.NAMESPACE)
        self.base=Path(self.temp.name)
        cad=self.base/'fixture_CAD.json';cad.write_text('{}\n')
        self.cad=capture.pointer(cad)
        self.identity=dict(split='train',task=capture.TASKS[0],parent_source_group='train:'+capture.TASKS[0]+':0',
            source_raw_episode_id='0',replica=1,seed=3,variation=0)
        self.components={'Cuboid':[1],'wand_visual':[2],'wand_visual_sub':[3]}

    def tearDown(self):
        self.temp.cleanup()

    def writer(self,name='sequence',**kwargs):
        return capture.CandidateVisualCaptureWriter(self.base/name,identity=kwargs.get('identity',self.identity),
            component_instance_ids=kwargs.get('components',self.components),canonical_CAD=self.cad)

    def test_copy_preserves_actual_buffers_calibration_modes_empty_depth_and_digest(self):
        data,robot=observation()
        frame=capture.capture_existing_public_rgbd(data,robot)
        self.assertTrue(np.isnan(frame.arrays['depth_front'][0,0]))
        self.assertEqual(frame.metadata['cameras']['front']['finite_positive_depth_pixels'],1)
        self.assertEqual(frame.metadata['cameras']['front']['invalid_or_empty_depth_pixels'],65535)
        np.testing.assert_array_equal(frame.arrays['K_front'],data['observation/calibrations']['front'].intrinsics)
        self.assertEqual(frame.arrays['K_front'][1,1],-200.)
        self.assertEqual(frame.arrays['raw_rgb_255_8'][0,0,0],17.25)
        self.assertEqual(frame.metadata['sensor_protocol'],data['observation/sensor_protocol'])
        self.assertIn('fallback_not_exposure_measurement',frame.metadata['cameras']['front']['exposure_timestamp_source'])
        expected=hashlib.sha256()
        for name in sorted(capture.CAMERAS):
            for channel in ('observation/images','observation/depths','observation/instance_masks'):
                expected.update(channel.encode());expected.update(name.encode())
                expected.update(np.ascontiguousarray(data[channel][name]).tobytes())
        self.assertEqual(frame.metadata['original_collector_sensor_digest'],expected.hexdigest())
        data['observation/images']['front'][0,0,0]=99
        self.assertEqual(frame.arrays['rgb_front'][0,0,0],17)
        self.assertFalse(frame.arrays['rgb_front'].flags.writeable)
        self.assertNotIn('task_success',str(frame.metadata))
        self.assertNotIn('robot_task_success',frame.arrays)

    def test_only_whitelisted_existing_observation_fields_are_accessed(self):
        data,robot=observation()
        allowed=set(data)
        class PublicObservation(dict):
            def __getitem__(self,key):
                if key not in allowed: raise AssertionError('forbidden key '+key)
                return super().__getitem__(key)
            def get(self,key,default=None):
                if key not in allowed|{'observation/camera_timestamps'}: raise AssertionError('forbidden key '+key)
                return super().get(key,default)
        public=PublicObservation(data)
        public['native_task_handle']=object();public['future_ground_truth']=object()
        frame=capture.capture_existing_public_rgbd(public,robot)
        self.assertFalse(frame.metadata['task_outcome_or_native_pose_recorded'])
        with self.assertRaisesRegex(ValueError,'RobotState'):
            capture.capture_existing_public_rgbd(public,object())

    def test_roundtrip_and_preserved_structure_failure_do_not_assign_task_outcome(self):
        writer=self.writer();data,robot=observation();writer.append(data,robot)
        bad=deepcopy(data);del bad['observation/depths']['front']
        with self.assertRaisesRegex(ValueError,'four camera'):
            writer.append(bad,robot)
        data2,robot2=observation(.05);writer.append(data2,robot2)
        ref=writer.finish();report=capture.validate_candidate_capture(ref)
        self.assertEqual(report['captured'],2);self.assertEqual(report['structural_failures'],1)
        self.assertFalse(report['native_execution_authorized'])
        manifest=json.loads(capture.check_pointer(ref).read_text())
        self.assertIsNone(manifest['actual_acquisition_registration'])
        with self.assertRaisesRegex(ValueError,'closed'):writer.append(data2,robot2)

    def test_missing_camera_empty_timestamp_sensor_modes_and_repeated_clock(self):
        data,robot=observation()
        for key in ('observation/images','observation/depths','observation/instance_masks','observation/calibrations'):
            bad=deepcopy(data);del bad[key]['front']
            with self.subTest(key=key),self.assertRaises(ValueError):capture.capture_existing_public_rgbd(bad,robot)
        bad=deepcopy(data);bad['timestamp']=np.nan
        with self.assertRaises(ValueError):capture.capture_existing_public_rgbd(bad,robot)
        bad=deepcopy(data);bad['observation/sensor_protocol']['camera_render_modes']={}
        with self.assertRaises(ValueError):capture.capture_existing_public_rgbd(bad,robot)
        data['observation/camera_timestamps']={name:.01 for name in capture.CAMERAS}
        frame=capture.capture_existing_public_rgbd(data,robot)
        self.assertEqual(frame.metadata['cameras']['front']['exposure_timestamp_source'],'provided_public_camera_timestamp')
        writer=self.writer();writer.append(data,robot)
        with self.assertRaisesRegex(ValueError,'increase'):writer.append(data,robot)
        self.assertEqual(capture.validate_candidate_capture(writer.finish())['structural_failures'],1)

    def test_scope_and_component_associations_reject_foreign_or_overlapping_inputs(self):
        for key,value in [('split','test'),('replica',True),('task','other'),('parent_source_group','wrong')]:
            ident=deepcopy(self.identity);ident[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):self.writer(identity=ident)
        with self.assertRaisesRegex(ValueError,'overlap'):
            self.writer(components={'Cuboid':[1],'wand_visual':[1],'wand_visual_sub':[3]})
        with self.assertRaisesRegex(ValueError,'namespace'):
            capture.CandidateVisualCaptureWriter(capture.ROOT/'runtime/forbidden_capture',identity=self.identity,
                component_instance_ids=self.components,canonical_CAD=self.cad)

    def test_raw_bytes_and_scope_tampering_are_detected(self):
        writer=self.writer();writer.append(*observation());ref=writer.finish()
        path=capture.check_pointer(ref);manifest=json.loads(path.read_text())
        npz=self.base/'sequence/frame_000000.npz';npz.write_bytes(npz.read_bytes()+b'tamper')
        with self.assertRaisesRegex(ValueError,'SHA'):capture.validate_candidate_capture(ref)
        writer2=self.writer('sequence2');writer2.append(*observation());ref2=writer2.finish()
        path2=capture.check_pointer(ref2);value=json.loads(path2.read_text());value['orientation_none_promoted']=True
        path2.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError,'scope'):capture.validate_candidate_capture(capture.pointer(path2))


if __name__=='__main__':unittest.main()
