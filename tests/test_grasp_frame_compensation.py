from pathlib import Path
import json
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from our_mani.contracts import Pose
from our_mani.multitask.interaction import InteractionLibrary
from our_mani.multitask.grasp_frame_compensation import compose,inverse,correction,transform,binding
ROOT=Path(__file__).resolve().parents[1]


class Tests(unittest.TestCase):
    def setUp(self):
        self.prior=json.loads((ROOT/'artifacts/framework/multitask_r10/ring_grasp_prior_v419/insert_onto_rotating_peg/model.json').read_text())
        self.config=json.loads((ROOT/'artifacts/framework/multitask_r10/registered_tasks_v408_setup/insert_onto_rotating_peg_configuration.json').read_text())
        self.library=InteractionLibrary.load(ROOT/self.config['learned_artifacts']['insert_onto_rotating_peg']['interaction']['path'])

    def test_current_grasp_compensation_preserves_the_learned_ring_target_at_every_curve_sample(self):
        expected=Pose.from_xyzw(self.prior['tool_to_ring_xyzw'])
        measured=Pose(expected.position+[.006,-.003,.004],Rotation.from_rotvec([.02,-.03,.01]).as_matrix()@expected.rotation)
        delta,actual=correction(self.prior,measured)
        for phase in ('align','insert'):
            original=self.library.for_phase(phase)[0];corrected=transform(original,delta)
            for progress in (0.,.25,.5,.75,1.):
                target=compose(original.at(progress)[0],expected)
                result=compose(corrected.at(progress)[0],actual)
                np.testing.assert_allclose(result.position,target.position,atol=1e-10)
                np.testing.assert_allclose(result.rotation,target.rotation,atol=1e-10)
            self.assertEqual((corrected.position_tolerance_m,corrected.orientation_tolerance_rad,
                corrected.duration_seconds,corrected.duration_budget_seconds),(original.position_tolerance_m,
                original.orientation_tolerance_rad,original.duration_seconds,original.duration_budget_seconds))
            target=compose(Pose.from_xyzw(original.completion_geometry_xyzw),expected)
            result=compose(Pose.from_xyzw(corrected.completion_geometry_xyzw),actual)
            np.testing.assert_allclose(result.position,target.position,atol=1e-10)

    def test_D4_equivalent_grasp_does_not_request_a_quarter_turn_of_the_robot(self):
        expected=Pose.from_xyzw(self.prior['tool_to_ring_xyzw']);center=np.asarray(self.prior['D4_center_body'])
        for s in np.asarray(self.prior['D4_group_rotations_body']):
            measured=compose(expected,Pose(center-s@center,s))
            delta,_=correction(self.prior,measured)
            np.testing.assert_allclose(delta.position,0.,atol=1e-10)
            np.testing.assert_allclose(delta.rotation,np.eye(3),atol=1e-10)

    def test_binding_uses_all100_TRAIN_parents_and_no_success_selection(self):
        self.assertEqual(self.prior['captured_parent_count'],100)
        self.assertFalse(self.prior['native_success_used_for_selection'])
        self.assertTrue(binding(ROOT,self.config,'insert_onto_rotating_peg')['completion_geometry_transformed_consistently'])


if __name__=='__main__':unittest.main()
