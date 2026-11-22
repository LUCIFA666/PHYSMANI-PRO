"""Current public orientation selects fixed primary/enclosure meshes."""
from types import SimpleNamespace
import unittest
import numpy as np
from our_mani.contracts import Pose,ObjectState
from our_mani.dynamics import ConstantVelocityDynamics
from our_mani.multitask.interaction import PolicyState
from our_mani.multitask.pipeline import PhaseCollisionPort
from our_mani.multitask.specs import get_task_spec
from tests.test_multitask_pipeline import SceneMap,ProbeRobot,robot


class VariantMap(SceneMap):
    def __init__(self):
        super().__init__()
        self.estimated_bodies={body:SimpleNamespace(fallback_ghost=object(),source_sha256='a'*64,
            collision_geometry_receipt=dict(collision_geometry_kind='exact_TRAIN_triangles'))
            for body in self.required_dynamic_role_ids}
    def check(self,poses=None,*,ignored_object_ids=(),allowed_robot_contact_ids=(),geometry_modes=None):
        super().check(poses,ignored_object_ids=ignored_object_ids,allowed_robot_contact_ids=allowed_robot_contact_ids)
        self.checks[-1]['geometry_modes']=dict(geometry_modes);return False


def frame(scene):
    bodies={body:ObjectState(body,1.,Pose(np.zeros(3)),np.zeros((3,3)),np.zeros(3),np.zeros(3))
        for body in scene.required_dynamic_role_ids}
    metadata={body:dict(registration_valid=True,orientation_observable=True,pose_evidence_until=1.,
        directly_visible=True) for body in bodies}
    state=PolicyState(robot(1.),bodies,dict(bodies=metadata),{})
    forecast=ConstantVelocityDynamics().predict(list(bodies.values()),1.,.05)
    port=PhaseCollisionPort(get_task_spec('push_moving_button'),scene,ProbeRobot(),maximum_observation_age=.1)
    port.configure_frame(state,forecast);return state,forecast,port,metadata


class GeometryModeChecks(unittest.TestCase):
    def test_unobservable_other_body_keeps_enclosure_and_does_not_stop_ready_scene(self):
        scene=VariantMap();state,forecast,port,meta=frame(scene)
        meta['wrap']['orientation_observable']=False
        self.assertTrue(port.readiness(state,forecast,'approach')['ready'])
        self.assertFalse(port.check_future(.15))
        self.assertEqual(scene.checks[-1]['geometry_modes'],dict(button='primary',plate='primary',wrap='rotation_uncertainty'))
        self.assertEqual(set(scene.checks[-1]['body_ids']),set(scene.required_dynamic_role_ids))
        self.assertEqual(scene.checks[-1]['ignored'],())
        self.assertEqual(port.receipt()['geometry_mode_queries:wrap:rotation_uncertainty'],1)

    def test_stale_pose_clock_and_json_quotient_flag_do_not_grant_exact_geometry(self):
        scene=VariantMap();state,forecast,port,meta=frame(scene)
        meta['button']['pose_evidence_until']=.95
        meta['wrap'].update(orientation_observable=False,orientation_equivalence_observable=True,
            rotation_equivalence_evidence=dict(orientation_equivalence_observable=True))
        port.readiness(state,forecast,'approach');self.assertFalse(port.check_future(.1))
        self.assertEqual(scene.checks[-1]['geometry_modes']['button'],'rotation_uncertainty')
        self.assertEqual(scene.checks[-1]['geometry_modes']['wrap'],'rotation_uncertainty')

    def test_directed_axis_sweep_needs_current_direct_axis_and_explicit_inflation_assumptions(self):
        scene=VariantMap();state,forecast,port,meta=frame(scene)
        receipt=dict(collision_geometry_kind='directed_axis_yaw_uncertainty_sweep',
            engineering_pose_error_assumption=True,engineering_tilt_bound_degrees=10.,
            engineering_translation_bound_m=.005,calibrated_safety_guarantee=False,
            current_directed_axis_readiness_required_for_every_collision_query=True)
        scene.estimated_bodies['button'].collision_geometry_receipt=receipt
        meta['button'].update(orientation_observable=False,axis_observable=True,axis_polarity='directed')
        port.readiness(state,forecast,'approach');self.assertFalse(port.check_future(.1))
        self.assertEqual(scene.checks[-1]['geometry_modes']['button'],'primary')
        meta['button']['axis_polarity']='undirected';self.assertFalse(port.check_future(.1))
        self.assertEqual(scene.checks[-1]['geometry_modes']['button'],'rotation_uncertainty')
        meta['button']['axis_polarity']='directed';receipt.pop('engineering_translation_bound_m')
        self.assertFalse(port.check_future(.1));self.assertEqual(scene.checks[-1]['geometry_modes']['button'],'rotation_uncertainty')

    def test_missing_forecast_still_fails_and_missing_enclosure_cannot_skip_unknown_body(self):
        scene=VariantMap();state,forecast,port,meta=frame(scene)
        meta['wrap']['orientation_observable']=False;scene.estimated_bodies['wrap'].fallback_ghost=None
        self.assertFalse(port.readiness(state,forecast,'approach')['ready'])
        self.assertTrue(port.check_future(.1));self.assertEqual(scene.checks,[])


if __name__=='__main__':unittest.main()
