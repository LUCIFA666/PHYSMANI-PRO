import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from our_mani.multitask.axis_evidence import near_optimal_axis_evidence


class AxisEvidenceTests(unittest.TestCase):
    def evidence(self,rotations,directed=True):
        return near_optimal_axis_evidence([(0.,r) for r in rotations],np.eye(3),[1.,0.,0.],
            directed=directed,best_rms_m=0.,rms_slack_m=.001,ambiguity_radians=.1)

    def test_axial_yaw_is_free_but_polarity_flip_is_not_directed_evidence(self):
        yaws=Rotation.from_euler('x',[0.,.5,2.,np.pi]).as_matrix()
        self.assertTrue(self.evidence(yaws)['axis_observable'])
        flip=Rotation.from_euler('y',np.pi).as_matrix()
        self.assertFalse(self.evidence([np.eye(3),flip])['axis_observable'])
        self.assertTrue(self.evidence([np.eye(3),flip],False)['axis_observable'])

    def test_tilt_or_missing_hypotheses_cannot_claim_a_known_axis(self):
        self.assertFalse(self.evidence([Rotation.from_euler('z',.3).as_matrix()])['axis_observable'])
        self.assertFalse(self.evidence([])['axis_observable'])
        with self.assertRaises(ValueError):self.evidence([np.zeros((3,3))])


if __name__=='__main__':unittest.main()
