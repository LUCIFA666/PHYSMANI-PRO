import unittest
import numpy as np
from scipy.special import logit
from our_mani.contracts import Pose,RobotState,Forecast,HandoverRegion,Reachability
from our_mani.risk_v2.selector import candidate_features,AttemptInterception
from our_mani.risk_v2.model import fit_calibration,calibrated_probabilities,TASKS


class CandidateSuccessTests(unittest.TestCase):
    def features(self,now=2.,context=None):
        pose=Pose.from_xyzw([.2,0.,.775,0.,0.,0.,1.])
        robot=RobotState(now,pose,np.zeros(3),np.zeros(3),np.zeros(7),np.zeros(7))
        forecast=Forecast(('target',),np.array([now,now+.5]),np.array([[pose.position],[pose.position]]),
            np.broadcast_to(np.eye(3),(2,1,3,3)),np.zeros((2,1,3)),np.zeros((2,1,3)),
            metadata={} if context is None else dict(context=context,context_encoder_sha256='frozen',history_frames=20))
        region=HandoverRegion('r','target',Pose.from_xyzw([0,0,.07,0,1,0,0]),np.ones(3)*.04)
        check=Reachability(True,'valid',joints=np.ones(7)*.1,joint_velocities=np.zeros(7),
            evidence=dict(maximum_speed_fraction=.5,maximum_acceleration_fraction=.6))
        return candidate_features(forecast,region,robot,pose,now+.5,pose,np.zeros(3),check,
            {'simulation_seconds':10.},'cv' if context is None else 'context')

    def test_remaining_budget_is_current_not_constant(self):
        a,b=self.features(2.),self.features(5.)
        different=np.flatnonzero(a!=b)
        self.assertEqual(len(different),2)
        np.testing.assert_allclose(a[different]-b[different],3.)

    def test_causal_context_reaches_probability_features(self):
        a=self.features(context=np.ones(16)/8)
        b=self.features(context=np.r_[1.,np.zeros(15)])
        np.testing.assert_array_equal(a[:-16],b[:-16])
        self.assertFalse(np.array_equal(a[-16:],b[-16:]))

    def test_replaced_attempt_does_not_inherit_later_success(self):
        owner=object.__new__(AttemptInterception)
        failed=dict(complete=False);success=dict(complete=False)
        owner.active_attempt=failed;owner.close_attempt(False,'replaced',3.)
        owner.active_attempt=success;owner.close_attempt(True,'native_task_success',5.)
        self.assertFalse(failed['official_task_success']);self.assertTrue(success['official_task_success'])

    def test_temporary_occlusion_does_not_end_local_attempt(self):
        from unittest.mock import patch
        from types import SimpleNamespace
        from our_mani.intercept_planner import CandidateInterception
        owner=object.__new__(AttemptInterception)
        owner.stage='local';owner.goal=None
        attempt=dict(complete=False);owner.active_attempt=attempt
        with patch.object(CandidateInterception,'step',return_value={'command':'hold'}):
            owner.step(SimpleNamespace(timestamp=3.),None,7.)
        self.assertIs(owner.active_attempt,attempt)
        self.assertFalse(attempt['complete'])
        owner.close_attempt(True,'native_task_success',5.)
        self.assertTrue(attempt['official_task_success'])

    def test_intercept_calibration_can_correct_overconfidence(self):
        # Temperature alone cannot make a constant p>.5 become p<.5.
        p=np.full(200,.75);labels=np.r_[np.ones(20),np.zeros(80),np.ones(80),np.zeros(20)]
        tasks=[TASKS[1]]*100+[TASKS[0]]*100
        calibration=fit_calibration(logit(p),labels,tasks)
        corrected=calibrated_probabilities(p,tasks,calibration)
        self.assertLess(corrected[:100].mean(),.35)
        self.assertGreater(corrected[100:].mean(),.65)
        self.assertLess(np.mean((corrected-labels)**2),np.mean((p-labels)**2))

if __name__=='__main__':unittest.main()
