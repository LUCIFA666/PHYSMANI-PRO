"""An ambiguous directed-axis fit must not create a later angular impulse."""
from dataclasses import replace
import unittest
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation

from our_mani.contracts import Pose
from our_mani.multitask.perception import MultiBodySE3Tracker
from test_multitask_perception_r10 import multi_component_body, render


class AxisHistory(unittest.TestCase):
    def test_rotation_history_resets_without_discarding_observed_translation(self):
        template = replace(multi_component_body('bin'), orientation_observability='axis',
            symmetry_axis=np.array([1.,0.,0.]), axis_polarity='directed')
        tracker = MultiBodySE3Tracker([template])
        states = []
        for tick, (angle, observable) in enumerate(zip([0.,.02,.8,.04,.06], [True,True,False,True,True])):
            pose = Pose(np.array([tick*.002,0.,1.]),Rotation.from_euler('z',angle).as_matrix())
            fit = dict(pose=pose,valid=True,rotation_ambiguous=not observable,rms_m=0.,
                per_role={'bin':dict(rms_m=0.,inlier_fraction=1.,valid=True)},
                axis_evidence_by_role={'bin':dict(axis_observable=observable)})
            with patch.object(tracker,'_registration',return_value=fit):
                objects, _ = tracker.observe(render([template], {'bin':pose},tick*.05), modes={'bin':'external'})
            states.append(objects[0])
        self.assertAlmostEqual(states[1].angular_velocity[2], .4, places=10)
        np.testing.assert_allclose(states[2].angular_velocity, 0., atol=1e-10)
        np.testing.assert_allclose(states[3].angular_velocity, 0., atol=1e-10)
        self.assertAlmostEqual(states[4].angular_velocity[2], .4, places=10)
        for state in states[1:]:
            np.testing.assert_allclose(state.linear_velocity, [.04,0.,0.], atol=1e-10)


if __name__ == '__main__':
    unittest.main()
