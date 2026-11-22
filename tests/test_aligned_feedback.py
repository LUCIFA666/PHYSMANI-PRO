import unittest
import numpy as np
from our_mani.aligned.feedback import CachedVisualDynamics
from our_mani.contracts import ObjectState,Pose
from our_mani.dynamics import ConstantVelocityDynamics

def obj(time,x):
    return ObjectState('target',time,Pose([x,0,.775]),np.array([[x,0,.775],[x,.01,.775],[x,0,.785]]),[.1,0,0],[0,0,0])

class FeedbackTests(unittest.TestCase):
    def test_fresh_background_observation_cannot_change_cached_action_forecast(self):
        model=CachedVisualDynamics(ConstantVelocityDynamics())
        model.refresh(obj(0.,0.))
        model.observe([obj(.2,100.)])
        prediction=model.predict([obj(.2,100.)],.3,.05)
        self.assertAlmostEqual(prediction.positions[0,0,0],.02)
        self.assertAlmostEqual(prediction.positions[-1,0,0],.05)
        self.assertEqual(prediction.metadata['visual_feedback_frozen_at'],0.)
        model.refresh(obj(.5,.4))
        self.assertAlmostEqual(model.object_at(.5).pose.position[0],.4)

    def test_forecast_does_not_silently_extrapolate_beyond_cached_horizon(self):
        model=CachedVisualDynamics(ConstantVelocityDynamics());model.refresh(obj(0.,0.))
        with self.assertRaises(ValueError):model.predict([obj(1.,0.)],.3,.05)

if __name__=='__main__':unittest.main()
