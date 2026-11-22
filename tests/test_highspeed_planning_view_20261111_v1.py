"""A fused planning forecast cannot rewrite frozen neural candidate inputs."""
import unittest
from types import SimpleNamespace
import numpy as np
from our_mani.contracts import Pose
from our_mani.highspeed_control_20261111_v1.coordinator_v8 import PublicMotionPlanningView
from tests.test_highspeed_public_motion_20261111_v1 import forecast


class PlanningViewContracts(unittest.TestCase):
    def test_planning_view_does_not_mutate_original_gru_or_nn_cache(self):
        p=np.array([.2,0.,.8]);f=forecast(0.,p,np.array([.1,0.,0.]))
        original=dict(timestamp=0.,object_observation=SimpleNamespace(pose=Pose(p)),forecast=f,frozen_checkpoint_sha256='703',Native_inputs_used=False)
        cache=object();backend=SimpleNamespace(latest=original,latest_contact={'held':False},candidate_cache=cache)
        view=PublicMotionPlanningView(backend);out=view.latest
        self.assertIs(original['forecast'],f);self.assertEqual(original['frozen_checkpoint_sha256'],'703')
        self.assertIs(view.candidate_cache,cache);self.assertIsNot(out,original)
        self.assertFalse(out['neural_candidate_cache_modified']);self.assertIs(view.latest,out)
        np.testing.assert_array_equal(out['forecast'].positions,f.positions)


if __name__=='__main__':unittest.main()
