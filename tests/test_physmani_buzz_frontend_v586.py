from pathlib import Path
from types import SimpleNamespace as NS,MethodType
from unittest.mock import patch,Mock
import json
import pickle
import unittest
import numpy as np
from our_mani.contracts import Pose
from our_mani.perception import CameraCalibration
from our_mani.multitask.perception import MultiBodySE3Tracker
from our_mani.multitask.buzz_runtime_v64 import install
from our_mani.multitask.physmani_buzz_frontend_v586 import PublicBuzzFrontend,OriginalFrontend

ROOT=Path(__file__).resolve().parents[1]
REPLAY=ROOT/'artifacts/framework/multitask_r10/public_replay_v579/development_demo_only/beat_the_rotating_buzz/episode10/context_geometry/public_replay'


class Tests(unittest.TestCase):
    def fixture(self):
        payload=pickle.loads((REPLAY/'public_TRAIN_templates.pkl').read_bytes())
        tracker=MultiBodySE3Tracker(payload['templates'],rigid_children=payload['constraints'],config=payload['config'])
        c=json.loads((ROOT/'artifacts/framework/multitask_r10/physmani_skills_v536_setup/beat_the_rotating_buzz_configuration.json').read_text())
        actor=NS(predictor='context',ranking='geometry',tracker=tracker,act=MethodType(lambda *a:None,tracker),metrics_report=lambda:{})
        install(actor,c,ROOT,'beat_the_rotating_buzz')
        data=np.load(REPLAY/'public_frame_000.npz');info=json.loads((REPLAY/'public_frame_000.json').read_text())
        cams=info['cameras'];calibrations={k:CameraCalibration(np.asarray(v['intrinsics']),
            Pose.from_xyzw(v['camera_to_world_xyzw']),depth_scale=v['depth_scale']) for k,v in info['calibrations'].items()}
        observation=dict(timestamp=info['timestamp'])
        for key,prefix in (('observation/images','rgb_'),('observation/depths','depth_'),('observation/instance_masks','mask_')):
            observation[key]={k:data[prefix+k] for k in cams}
        observation['observation/calibrations']=calibrations
        return actor,observation

    def test_actual_saved_public_RGBD_initializes_when_public_flags_are_forwarded_without_old_act(self):
        actor,observation=self.fixture()
        states,_=actor.tracker.observe(observation,modes={'frame':'external','wand':'external'})
        self.assertEqual(states,[])
        actor,observation=self.fixture();actor.act=Mock(side_effect=AssertionError('old controller forbidden'))
        frontend=PublicBuzzFrontend(actor)
        # This fixture supplies the original declared public initial condition:
        # t=0, open empty gripper. Runtime586 reads these same flags from RobotState.
        robot=NS(timestamp=0.,gripper_open=True,contact=False)
        def observed(instance,observation,robot):
            return instance.actor.tracker.observe(observation,modes={'frame':'external','wand':'external'})
        with patch.object(OriginalFrontend,'observe',observed):states,metadata=frontend.observe(observation,robot)
        self.assertEqual({s.object_id for s in states},{'frame','wand'})
        self.assertTrue(all(metadata['bodies'][k]['registration_valid'] for k in ('frame','wand')))
        self.assertTrue(all(metadata['bodies'][k]['orientation_observable'] for k in ('frame','wand')))
        actor.act.assert_not_called()
        self.assertTrue(any(e.get('qualified') is True for e in frontend.conditional_registration.events))


if __name__=='__main__':unittest.main()
