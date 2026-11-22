"""Actual public buffers through installed tracker; reject post-action prior reuse."""
from pathlib import Path
from types import MethodType,SimpleNamespace
import json,unittest
import numpy as np
from scipy.spatial.transform import Rotation
from our_mani.contracts import Pose
from our_mani.perception import CameraCalibration
from our_mani.multitask.perception import MultiBodySE3Tracker
from our_mani.multitask.training_geometry import body_templates_from_asset
from our_mani.multitask.buzz_branch_v63 import admissible_hypotheses
from our_mani.multitask.buzz_runtime_v64 import binding,install

ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'artifacts/framework/multitask_r10'


class Tests(unittest.TestCase):
    def actor(self,task,raw,*,contact=False):
        c=json.loads((BASE/'task_executor_smoke_v60_setup/configuration.json').read_text())
        c['body_orientation_observability'][task]={'frame':'full','wand':'full'}
        c['buzz_runtime_v64']={task:binding(ROOT,c,task)}
        capture=BASE/'buzz_visual_capture_candidates'/f'native_v39_{task}_{raw}_cv_geometry'
        m=json.loads((capture/'manifest.json').read_text());asset=json.loads((ROOT/m['canonical_CAD']['path']).read_text())
        bindings={role:{part['name']:tuple(m['component_instance_ids'][part['name']]) for part in body['components']}
            for role,body in asset['bodies'].items()}
        tracker=MultiBodySE3Tracker(body_templates_from_asset(asset,bindings,orientation_observability={'frame':'full','wand':'full'}))
        actor=SimpleNamespace(tracker=tracker)
        actor.act=MethodType(lambda self,observation,robot,remaining:self.tracker.observe(observation,modes={'frame':'external','wand':'external'}),actor)
        actor.metrics_report=lambda:dict()
        install(actor,c,ROOT,task)
        with np.load(capture/'frame_000000.npz',allow_pickle=False) as stored:z={k:stored[k] for k in stored.files}
        calibrations={}
        for camera in ('front','8','16','36'):
            t=z['camera_to_world_'+camera];calibrations[camera]=CameraCalibration(z['K_'+camera],Pose(t[:3,3],t[:3,:3]),
                depth_scale=float(z['depth_scale_'+camera]),timestamp_offset=float(z['timestamp_offset_'+camera]))
        observation=dict(timestamp=0.,**{'observation/depths':{c:z['depth_'+c] for c in calibrations},
            'observation/instance_masks':{c:z['mask_'+c] for c in calibrations},'observation/calibrations':calibrations,
            'observation/camera_timestamps':{c:float(z['timestamp_'+c]) for c in calibrations}})
        robot=SimpleNamespace(timestamp=0.,gripper_open=True,contact=contact)
        return actor,observation,robot

    def test_actual_normal_and_high_RGBD_initialization_supplies_conditional_full_orientation_and_provenance(self):
        for task,raw in [('beat_the_rotating_buzz','episode10'),('beat_the_rotating_buzz_high_speed','episode102')]:
            actor,observation,robot=self.actor(task,raw)
            states,metadata=actor.act(observation,robot,10.)
            self.assertEqual(len(states),2)
            for role,d in metadata['bodies'].items():
                self.assertTrue(d['orientation_observable'])
                self.assertIn('conditioned_on_TRAIN_initial_branch',d['pose_source'])
            self.assertTrue(all(e['qualified'] for e in actor.metrics_report()['buzz_branch_v63']['events']))

    def test_contact_or_noninitial_observation_cannot_initialize_or_reuse_pair_prior(self):
        for now,contact in ((0.,True),(.05,False)):
            actor,observation,robot=self.actor('beat_the_rotating_buzz','episode10',contact=contact)
            observation['timestamp']=now;robot.timestamp=now
            observation['observation/camera_timestamps']={k:now for k in observation['observation/camera_timestamps']}
            states,metadata=actor.act(observation,robot,10.)
            self.assertEqual(states,[])
            self.assertTrue(all(not d['orientation_observable'] for d in metadata['bodies'].values()))

    def test_conditional_likelihood_keeps_all_hypotheses_and_rejects_outside_prior(self):
        good=Pose([0.,0.,0.]);flipped=Pose([0.,0.,0.],Rotation.from_euler('y',np.pi).as_matrix())
        index,e=admissible_hypotheses([flipped,good],[.002,.003],np.eye(3),rotation_bound_rad=.15,rms_slack_m=.0005,rotation_ambiguity_rad=.3)
        self.assertEqual(index,1);self.assertEqual(e['all_retained_hypotheses'],2)
        self.assertTrue(e['absolute_registration_quality_gate_unchanged'])
        index,e=admissible_hypotheses([flipped],[.002],np.eye(3),rotation_bound_rad=.15,rms_slack_m=.0005,rotation_ambiguity_rad=.3)
        self.assertIsNone(index)


if __name__=='__main__':unittest.main()
