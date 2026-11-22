"""Pure metrics/observability-boundary tests; no native or GPU execution."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
from scipy.spatial.transform import Rotation
from our_mani.contracts import Pose

path=Path(__file__).resolve().parents[1]/'scripts/pilot_orientation_registration_r10.py'
spec=importlib.util.spec_from_file_location('orientation_pilot_tested',path)
pilot=importlib.util.module_from_spec(spec);spec.loader.exec_module(pilot)


class OrientationPilotTests(unittest.TestCase):
    def test_axis_symmetry_does_not_turn_pi_rotation_into_full_orientation_agreement(self):
        r=Rotation.from_euler('z',np.pi).as_matrix();metrics=pilot.rotation_metrics(r,np.eye(3))
        assert abs(metrics['raw_SO3_error_rad']-np.pi)<1e-12
        assert metrics['axes']['local_z']['undirected_error_rad']==0
        assert metrics['axes']['local_x']['undirected_error_rad']==0
        assert abs(metrics['axes']['local_x']['directed_error_rad']-np.pi)<1e-12

    def test_invalid_rotation_is_rejected_instead_of_silently_projected(self):
        with self.assertRaisesRegex(ValueError,'physical SO3'):pilot.rotation_metrics(np.eye(3)*2,np.eye(3))

    def test_none_and_ball_do_not_claim_full_truth_from_emitted_identity_gauge(self):
        pose=Pose(np.zeros(3));states=[SimpleNamespace(object_id='ball',pose=pose),SimpleNamespace(object_id='cup0',pose=pose)]
        metadata={'bodies':{role:{'orientation_observability':'none','orientation_observable':False} for role in ('ball','cup0')}}
        gt={'positions':np.zeros((2,3)),'rotations':np.tile(Rotation.from_euler('z',.7).as_matrix(),(2,1,1))}
        call=dict(raw_body_poses_xyzw={'ball':pose.xyzw().tolist(),'cup0':pose.xyzw().tolist()},
            rms_m=.002,valid=True,rotation_ambiguous=True)
        result=pilot.metrics_for_variant(states,metadata,[call],gt,('ball','cup0'))
        assert result['ball']['raw_fit_metric_only']['rotation_errors'] is None
        assert result['cup0']['full_orientation_verified'] is False
        assert result['cup0']['raw_fit_metric_only']['physical_orientation_claimed'] is False
        assert abs(result['cup0']['raw_fit_metric_only']['rotation_errors']['raw_SO3_error_rad']-.7)<1e-12

    def test_missing_registration_stays_missing_with_no_pseudo_rotation_error(self):
        gt={'positions':np.zeros((1,3)),'rotations':np.eye(3)[None]}
        result=pilot.metrics_for_variant([],{'bodies':{'frame':{'registration_valid':False}}},[],gt,('frame',))
        assert result['frame']['candidate_state_present'] is False
        assert result['frame']['raw_fit_metric_only'] is None

    def test_CAD_symmetry_retains_raw_curves_without_automatic_annotation(self):
        points=np.array([[x,y,0.] for x,y in ((1,0),(0,1),(-1,0),(0,-1))])
        result=pilot.symmetry_residuals({'bodies':{'ring':{'collision_vertices_body':points.tolist()}}})['ring']
        assert len(result['curves']['local_z'])==11
        assert result['curves']['local_z'][2]['symmetric_nn_rms_m']<1e-12
        assert result['absence_of_detected_symmetry_is_not_proof_of_full_observability'] is True


if __name__=='__main__':unittest.main()
