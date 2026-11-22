import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from our_mani.executor_posture_lead_20261113_v15.pose_executor import PoseGoalExecutor,DropExecutor

class PostureLeadTest(unittest.TestCase):
    def test_primary_posture_branch_receives_the_declared_control_lead(self):
        driver=PoseGoalExecutor.__new__(PoseGoalExecutor);received=[]
        driver.live_unload=None;driver.control_lead_seconds=.05
        driver._sample_target=lambda:(np.array([.3,0.,.9]),np.array([0.,0.,0.,1.]),np.array([.1,0.,0.,0.,0.,0.]),dict(valid=True))
        driver.clearance=SimpleNamespace(fk=lambda q:np.eye(4))
        driver.feedback_mode='actual';driver.posture_activation='always';driver.posture_mode='joint_goal'
        def update(clearance,limits,current,p,q,clock):
            received.append((p.copy(),q.copy()));return np.zeros(7),dict(available=True)
        driver.posture_prior=SimpleNamespace(update=update)
        driver.executor=SimpleNamespace(config=SimpleNamespace(reference_frequency=6.),limits=object(),gripper_done=True)
        driver.backend=SimpleNamespace(clock=lambda:1.)
        driver.speed_norm=3.;driver.joint_margin=0.
        with patch('our_mani.executor_posture_lead_20261113_v15.pose_executor.step',return_value=('bounded_reference',{})):
            _,receipt=driver._next(SimpleNamespace(position=np.zeros(7)),None,np.zeros(7),None,None,retained=False)
        horizon=1./6.+.025+.05
        np.testing.assert_allclose(received[0][0],[.3+.1*horizon,0.,.9])
        self.assertAlmostEqual(receipt['intercept_horizon_seconds'],horizon)
        self.assertEqual(receipt['control_target_lead_seconds'],.05)

    def test_retained_braking_uses_existing_bounded_recovery(self):
        driver=PoseGoalExecutor.__new__(PoseGoalExecutor)
        with patch.object(DropExecutor,'_next',return_value=('bounded_brake',{}))as method:
            result=driver._next(None,None,None,None,None,retained=True)
        self.assertEqual(result[0],'bounded_brake');method.assert_called_once()

if __name__=='__main__':unittest.main()
