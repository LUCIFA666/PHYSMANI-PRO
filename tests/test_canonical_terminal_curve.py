from pathlib import Path
import json
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from our_mani.contracts import Pose
from our_mani.multitask.interaction import InteractionLibrary,_world_target
from our_mani.multitask.canonical_terminal_curve import terminal_curve
ROOT=Path(__file__).resolve().parents[1]


class Tests(unittest.TestCase):
    def setUp(self):
        c=json.loads((ROOT/'artifacts/framework/multitask_r10/registered_tasks_v408_setup/insert_onto_rotating_peg_configuration.json').read_text())
        self.library=InteractionLibrary.load(ROOT/c['learned_artifacts']['insert_onto_rotating_peg']['interaction']['path'])

    def test_real_ring_endpoints_hold_the_original_canonical_pose_and_zero_relative_velocity(self):
        for phase in ('align','insert'):
            source=self.library.for_phase(phase)[0];control=terminal_curve(source)
            endpoint=Pose.from_xyzw(source.completion_geometry_xyzw)
            for progress in (1.,1.5,2.):
                pose,v,w,_=control.at(progress)
                np.testing.assert_allclose(pose.position,endpoint.position,atol=1e-10)
                np.testing.assert_allclose(pose.rotation,endpoint.rotation,atol=1e-10)
                np.testing.assert_allclose(v,0.,atol=1e-10);np.testing.assert_allclose(w,0.,atol=1e-10)
            self.assertEqual(control.duration_seconds,source.duration_seconds)
            self.assertEqual(control.position_tolerance_m,source.position_tolerance_m)

    def test_control_velocity_matches_finite_differences_and_external_motion_is_not_zeroed(self):
        source=self.library.for_phase('align')[0];control=terminal_curve(source)
        for progress in (.13,.37,.71,.93):
            eps=1e-6;a=control.at(progress-eps)[0];b=control.at(progress+eps)[0]
            _,v,w,_=control.at(progress)
            np.testing.assert_allclose(v,(b.position-a.position)/(2*eps*control.duration_seconds),atol=2e-5)
            numerical=Rotation.from_matrix(b.rotation@a.rotation.T).as_rotvec()/(2*eps*control.duration_seconds)
            np.testing.assert_allclose(w,numerical,atol=2e-4)
        ref=Pose([.1,.2,.8]);linear=np.array([.1,.2,.0]);angular=np.array([0.,0.,.3])
        pose,v,w,_=_world_target(ref,linear,angular,control,1.)
        np.testing.assert_allclose(v,linear+np.cross(angular,pose.position-ref.position),atol=1e-10)
        np.testing.assert_allclose(w,angular,atol=1e-10)


if __name__=='__main__':unittest.main()
