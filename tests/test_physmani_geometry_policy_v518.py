from pathlib import Path
from types import SimpleNamespace as NS
import json
import unittest
import numpy as np
from our_mani.contracts import Pose,RobotState,ObjectState
from our_mani.multitask.interaction import PolicyState
from our_mani.multitask.button_intent_library_v71 import ButtonIntentLibrary
from our_mani.multitask.physmani_geometry_policy_v518 import ButtonGeometryIntents
ROOT=Path(__file__).resolve().parents[1]


class Forecast:
    def __init__(self,body,speed):
        self.body=body;self.speed=np.asarray(speed,float)
        self.times=np.array([1.,2.1]);self.metadata={'observed_until':1.}
    def at(self,identity,time):
        return (Pose(self.body.pose.position+self.speed*(time-1.),self.body.pose.rotation),
                self.speed,np.zeros(3))


class Tests(unittest.TestCase):
    def fixture(self,speed=(.1,0,0)):
        task='push_moving_button'
        config=json.loads((ROOT/'artifacts/framework/multitask_r10/button_corridor_v414_setup/configuration.json').read_text())
        library=ButtonIntentLibrary(ROOT,ROOT/config['learned_artifacts'][task]['interaction']['path'],
                                    purpose='train_development')
        body=ObjectState('button',1.,Pose(np.array([.3,0.,.8])),np.zeros((3,3)),np.asarray(speed),np.zeros(3),1.,True)
        robot=RobotState(1.,Pose(np.array([1.,0.,1.])),np.zeros(3),np.zeros(3),np.zeros(7),np.zeros(7))
        state=PolicyState(robot,{'button':body},{'bodies':{'button':{'registration_valid':True}}},{'button':('button',)})
        model=NS(lower=np.full(7,-2.),upper=np.full(7,2.),collision=lambda q,t:False)
        c=NS(library=library,phase_index=0,origin=None,committed_roles={},
             generator=NS(observed_reference=lambda s,i,t,o:body,bind_candidate_scene=lambda *a:None))
        f=NS(actor=NS(controller=c,predictor='context',robot_model=model),state=state,
             forecast=Forecast(body,speed),ready=lambda phase:True,issued=lambda phase,opening:None)
        return ButtonGeometryIntents(f),f

    def test_original_TRAIN_models_supply_predicted_pose_at_half_second(self):
        moving,f=self.fixture((.1,0,0));static,_=self.fixture((0.,0,0))
        a,b=moving.select(),static.select()
        self.assertIsNotNone(a);self.assertEqual(a.shape,(8,))
        self.assertAlmostEqual(a[0]-b[0],.05,places=8)
        self.assertAlmostEqual(moving.selected['predicted_reference_time'],1.5)
        self.assertEqual(moving.phase_index,0)
        self.assertTrue(moving.admit_pose(a)[0])

    def test_cannot_extrapolate_past_public_forecast_or_use_stale_intent(self):
        p,f=self.fixture();f.forecast.times=np.array([1.,1.1])
        self.assertIsNone(p.select())
        p,f=self.fixture();action=p.select()
        changed=action.copy();changed[0]+=.01
        self.assertFalse(p.admit_pose(changed)[0])
        f.forecast.metadata['observed_until']=0.
        self.assertFalse(p.admit_joint(np.zeros(7))[0])

    def test_original_author_approach_guard_transitions_without_claiming_press_completion(self):
        p,f=self.fixture();first=p.select();target=p.selected['present_pose']
        f.state=PolicyState(RobotState(1.,target,np.zeros(3),np.zeros(3),np.zeros(7),np.zeros(7)),
                            f.state.bodies,f.state.perception_metadata,f.state.role_bindings)
        self.assertIsNotNone(p.select())
        self.assertEqual(p.phase_index,1)
        self.assertFalse(p.events[-2]['public_press_completion_claimed'])
        self.assertEqual(p.receipt()['events'][-1]['phase'],'press')

    def test_collision_sweep_and_native_joint_bounds_remain_hard_admission(self):
        p,f=self.fixture();p.select()
        self.assertTrue(p.admit_joint(np.zeros(7))[0])
        f.actor.robot_model.collision=lambda q,t:True
        self.assertFalse(p.admit_joint(np.zeros(7))[0])
        self.assertFalse(p.admit_joint(np.full(7,3.))[0])


if __name__=='__main__':unittest.main()
