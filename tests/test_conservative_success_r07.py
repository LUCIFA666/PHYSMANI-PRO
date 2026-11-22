import unittest
from types import SimpleNamespace
import numpy as np
from our_mani.contracts import Pose,RobotState,Forecast,HandoverRegion,Reachability
from our_mani.risk_v3.selector import AttemptSelector


class ConservativeSelectionTests(unittest.TestCase):
    def select(self,probability=(.2,.8),std=(.01,.01),support=(True,True),enabled=True,feasible=True):
        pose=Pose.from_xyzw([.2,0.,.775,0,0,0,1])
        robot=RobotState(0.,pose,np.zeros(3),np.zeros(3),np.zeros(7),np.zeros(7))
        forecast=Forecast(('target',),np.array([0.,.3,.5]),np.array([[pose.position]]*3),
            np.broadcast_to(np.eye(3),(3,1,3,3)),np.zeros((3,1,3)),np.zeros((3,1,3)),metadata={})
        region=HandoverRegion('r','target',Pose.from_xyzw([0,0,.07,0,1,0,0]),np.ones(3)*.04)
        check=Reachability(feasible,'valid' if feasible else 'collision',joints=np.zeros(7),joint_velocities=np.zeros(7),
            evidence=dict(maximum_speed_fraction=.5,maximum_acceleration_fraction=.6))
        original=SimpleNamespace(reachability=SimpleNamespace(check=lambda *args:check),
            relative_velocity_options=[np.zeros(3)],object_position_bounds={},latency_reserve=0.)
        calls=[]
        owner=SimpleNamespace(goal_pose=pose,predictor='cv',config=dict(simulation_seconds=10.,
            _episode_task='task',ranking_minimum_gain=.02,ranking_disagreement_multiplier=1.),
            start_attempt=lambda *args:calls.append(args))
        model=SimpleNamespace(predict=lambda x,t:dict(probability=np.array(probability),ensemble_std=np.array(std),
            supported=np.array(support),ranking_enabled=enabled))
        selector=AttemptSelector(owner,original,model=model)
        goal=selector.select(forecast,[region],robot,0.,stride=1)
        return goal,selector.last_diagnostics,calls

    def test_clear_supported_gain_overrides_geometry(self):
        goal,diagnostics,calls=self.select()
        self.assertEqual(goal.arrival_time,.5)
        self.assertEqual(goal.region.value,.8)
        self.assertEqual(calls[0][-1],.8)
        self.assertIsNone(diagnostics['fallback'])

    def test_disagreement_preserves_geometry_and_true_probability(self):
        goal,diagnostics,_=self.select(probability=(.3,.6),std=(.3,.3))
        self.assertEqual(goal.arrival_time,.3)
        self.assertEqual(goal.region.value,.3)
        self.assertEqual(diagnostics['fallback'],'gain_below_pre_registered_disagreement_margin')

    def test_train_guard_disables_overrides(self):
        goal,diagnostics,_=self.select(enabled=False)
        self.assertEqual(goal.arrival_time,.3)
        self.assertFalse(diagnostics['learned_ranking_enabled'])

    def test_unsupported_geometry_cannot_be_reference(self):
        goal,diagnostics,_=self.select(support=(False,True))
        self.assertEqual(goal.arrival_time,.3)
        self.assertEqual(diagnostics['fallback'],'geometry_reference_outside_TRAIN_support')
        self.assertIsNone(diagnostics['selected_success_probability'])
        self.assertEqual(goal.region.source,'geometry_template_no_probability_claim')

    def test_learning_cannot_override_hard_collision_rejection(self):
        goal,diagnostics,calls=self.select(feasible=False)
        self.assertIsNone(goal)
        self.assertEqual(diagnostics['rejections']['collision'],2)
        self.assertEqual(calls,[])

if __name__=='__main__':unittest.main()
