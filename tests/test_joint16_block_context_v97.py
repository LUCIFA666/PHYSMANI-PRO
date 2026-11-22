"""Real shared-network inference with an explicitly untrained in-memory fixture."""
from pathlib import Path
from collections import deque
from threading import RLock
from types import SimpleNamespace
from unittest.mock import patch
from dataclasses import replace
import json,tempfile,unittest
import numpy as np
import torch

from our_mani.multitask import joint16_block_context_v97 as adapter_module
from our_mani.multitask.joint16_block_context_v97 import Joint16BlockContext
from our_mani.multitask.context_se3 import MultiBodyContextDynamics,SE3ContextNetwork
from our_mani.multitask.joint16_tracks_v31 import BLOCKS
from our_mani.multitask.specs import TASK_SPECS
from our_mani.contracts import ObjectState,Pose,RobotState,HandoverRegion
from our_mani.risk_v2.selector import candidate_features

ROOT=Path(__file__).resolve().parents[1]


class Tests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='UNIT_TEST_untrained_shared_block_',dir=ROOT/'runtime/framework/tmp')
        self.root=Path(self.tmp.name);self.source=self.root/'training_sources.json'
        rows=[dict(task=t,raw_episode_id=f'episode{i}',split='train') for t in sorted(set(TASK_SPECS)|set(BLOCKS)) for i in range(100)]
        self.source.write_text(json.dumps(dict(entries=rows)))
        roster=self.root/'artifacts/framework/protocol/train_manifest.json';roster.parent.mkdir(parents=True)
        roster.write_text(json.dumps(dict(episodes=rows)))
        # No fitted checkpoint or native task is created. Exercise the actual
        # observe/history/network/predict methods with zero residual weights.
        self.engine=MultiBodyContextDynamics.__new__(MultiBodyContextDynamics)
        self.engine.path=self.root/'UNTRAINED_UNIT_TEST_ONLY.pt'
        self.engine.model=SE3ContextNetwork().eval();self.engine.history={};self.engine.lock=RLock()
        self.engine.encoder_sha256=self.engine.model.encoder_sha256()
        self.engine.checkpoint_sha256='a'*64;self.engine.maximum_validated_horizon=1.1
        self.engine.metadata=dict(jointly_trained_tasks=sorted(set(TASK_SPECS)|set(BLOCKS)),
            training_source_manifest=dict(source_episode_count=1600,path_relative_to_checkpoint=self.source.name),
            unit_test_untrained_fixture=True)
        self.patch=patch.object(adapter_module,'MultiBodyContextDynamics',return_value=self.engine);self.patch.start()
        self.adapter=Joint16BlockContext(self.engine.path,task=BLOCKS[0],root=self.root)
    def tearDown(self):self.patch.stop();self.tmp.cleanup()
    def obj(self,t):
        p=np.array([.1+.2*t,0.,.77])
        return ObjectState('target',t,Pose(p),np.tile(p,(3,1)),np.array([.2,0.,0.]),np.zeros(3))
    def warm(self):
        for t in np.arange(5)*.05:
            self.adapter.set_public_motion_state(timestamp=t,gripper_open=True,contact=False)
            self.adapter.observe([self.obj(t)])
        return self.obj(.2)

    def test_shared_model_forecast_matches_actual_network_and_legacy_feature_interface(self):
        obj=self.warm();reference=self.engine.predict([obj],1.1,.05)
        forecast=self.adapter.predict([obj],1.1,.05)
        np.testing.assert_array_equal(forecast.positions,reference.positions)
        self.assertEqual(forecast.metadata['context'],reference.metadata['contexts']['target'])
        self.assertEqual(forecast.metadata['history_frames'],5)
        self.assertEqual(forecast.metadata['observed_until'],.2)
        self.assertIs(self.adapter.engine.model,self.engine.model)
        self.assertFalse(self.engine.history['target'][-1]['rotation_observable'])
        region=HandoverRegion('unit','target',Pose(np.zeros(3)),np.full(3,.01))
        robot=RobotState(.2,Pose([.2,0.,1.]),np.zeros(3),np.zeros(3),np.zeros(7),np.zeros(7))
        pose,v,w=forecast.at('target',.5)
        check=SimpleNamespace(joints=np.zeros(7),joint_velocities=np.zeros(7),
            evidence=dict(maximum_speed_fraction=.1,maximum_acceleration_fraction=.1))
        feature=candidate_features(forecast,region,robot,Pose([0.,0.,.8]),.5,pose,v,check,
            dict(simulation_seconds=10.),'context')
        self.assertTrue(np.isfinite(feature).all());np.testing.assert_allclose(feature[-16:],forecast.metadata['context'])

    def test_first_close_or_contact_irrevocably_stops_exogenous_prediction(self):
        obj=self.warm()
        self.adapter.set_public_motion_state(timestamp=.25,gripper_open=False,contact=False)
        self.assertFalse(self.adapter.observe([self.obj(.25)]));self.assertFalse(self.engine.history)
        self.adapter.set_public_motion_state(timestamp=.3,gripper_open=True,contact=False)
        self.assertFalse(self.adapter.observe([self.obj(.3)]))
        with self.assertRaisesRegex(ValueError,'after first'):self.adapter.predict([self.obj(.3)],1.1,.05)
        with self.assertRaisesRegex(ValueError,'pre-contact'):self.adapter.context_snapshot()
        self.adapter.reset();self.warm()
        self.adapter.set_public_motion_state(timestamp=.25,gripper_open=True,contact=True)
        attached=replace(self.obj(.25),angular_velocity=np.array([0.,0.,1.]))
        self.assertFalse(self.adapter.observe([attached]))
        with self.assertRaisesRegex(ValueError,'after first'):self.adapter.predict([attached],1.1,.05)

    def test_missing_public_state_stale_or_rotated_channel_is_rejected(self):
        with self.assertRaisesRegex(ValueError,'current public'):self.adapter.observe([self.obj(0.)])
        self.adapter.set_public_motion_state(timestamp=.05,gripper_open=True,contact=False)
        with self.assertRaises(ValueError):self.adapter.observe([self.obj(0.)])
        obj=replace(self.obj(.05),angular_velocity=np.array([0.,0.,1.]))
        with self.assertRaisesRegex(ValueError,'axis-aligned'):self.adapter.observe([obj])
        with self.assertRaises(TypeError):
            self.adapter.set_public_motion_state(timestamp=.05,gripper_open=True,contact=False,native_success=True)

    def test_insufficient_history_and_episode_reversal_do_not_fabricate_context(self):
        self.adapter.set_public_motion_state(timestamp=.1,gripper_open=True,contact=False)
        self.adapter.observe([self.obj(.1)])
        with self.assertRaisesRegex(ValueError,'five causal'):self.adapter.predict([self.obj(.1)],1.1,.05)
        with self.assertRaisesRegex(ValueError,'episode boundaries'):
            self.adapter.set_public_motion_state(timestamp=0.,gripper_open=True,contact=False)

    def test_old_checkpoint_and_incomplete_joint_roster_are_rejected(self):
        self.engine.metadata['training_source_manifest']['source_episode_count']=1400
        with self.assertRaisesRegex(ValueError,'joint16'):Joint16BlockContext(self.engine.path,task=BLOCKS[0],root=self.root)
        self.engine.metadata['training_source_manifest']['source_episode_count']=1600
        data=json.loads(self.source.read_text());data['entries'].pop();self.source.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError,'complete original TRAIN roster'):
            Joint16BlockContext(self.engine.path,task=BLOCKS[0],root=self.root)

    def test_other_task_cannot_use_the_block_contract(self):
        with self.assertRaisesRegex(ValueError,'two original moving-block'):
            Joint16BlockContext(self.engine.path,task='push_moving_button',root=self.root)


if __name__=='__main__':unittest.main()
