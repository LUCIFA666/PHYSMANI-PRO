"""CPU unit fixtures only; no genuine qualified model or live capability.

Branch tests deliberately bypass a missing loader on an unbound math fixture.
The production constructors expose no such injection and the exact binding
rejects these fixtures.  Passing these tests is not deployment qualification.
"""
from collections import deque
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np

from our_mani.benchmark_perception import BoxPrior,RedBoxTracker,box_surface
from our_mani.contracts import ObjectState,Pose
from our_mani.perception import CameraCalibration
from our_mani.local_mpc_control_20261103 import depth_refiner_goal_perception_v60 as v60
from our_mani.local_mpc_control_20261103.goal_aware_perception import GoalAwareRedBoxTracker,VisualGoalSphere
from our_mani.local_mpc_control_20261103.goal_aware_binding_v60 import ApprovedDepthRefinerTrackerCacheBindingV60
from our_mani.local_mpc_control_20261103.public_attachment import PublicTargetFrameCache
from our_mani.local_mpc_control_20261103.public_motion_bridge import PublicMotionBridge
from our_mani.local_mpc_control_20261103.depth_refiner_public_inputs_v60 import pack_current_public_refiner_inputs
from our_mani.near_goal_depth_refiner_20261103.data import make_measurement,make_near_goal_crops


def unbound_math_fixture():
    t=object.__new__(v60.CurrentDepthRefinerTrackerV60)
    GoalAwareRedBoxTracker.__init__(t,tracker=RedBoxTracker(prior=BoxPrior(minimum_red=100),complete_geometry=False))
    t._sphere=VisualGoalSphere([0,0,1.],.04,0.,0.,.004,t.marker.offset)
    t._initial_fit_attempted=True;t.expected_refiner_sha256='a'*64
    t._accepted_source=None;t.last_measurement_metadata={};t.last_measured_cloud=None
    t.last_current_target=None;t._last_observed_timestamp=None
    return t


def public_observation(now=0.):
    K=np.array([[-200.,0,128],[0,-200.,128],[0,0,1.]])
    names=('front','8','16','36')
    return dict(timestamp=now,**{'observation/images':{n:np.zeros((256,256,3),np.uint8) for n in names},
        'observation/depths':{n:np.full((256,256),1.025,np.float32) for n in names},
        'observation/calibrations':{n:CameraCalibration(K,Pose(np.zeros(3))) for n in names}})


def nn_prediction(t,obs):
    inputs=pack_current_public_refiner_inputs(obs,t.sphere);crops=make_near_goal_crops(inputs)
    mask=np.zeros((4,64,64),bool);mask[:,27:37,27:37]=True
    m=make_measurement(inputs,crops,mask)
    assert m['refiner_available']
    return dict(checkpoint_sha256=t.expected_refiner_sha256,refiner_available=True,
                centre=m['centre'],measurement=m,raw_Cube_mask=mask,completion=False)


def component(center):
    p=np.array([[x,y,z] for x in (-.025,.025) for y in (-.025,.025) for z in (-.025,.025)])
    return np.repeat(p+np.asarray(center),3,axis=0)


class V60CandidateUnitTests(unittest.TestCase):
    def test_public_constructor_has_no_diag_or_runtime_injection(self):
        for kwargs in [dict(refiner=object()),dict(allow_unqualified_diagnostic=True)]:
            with self.assertRaises(TypeError):v60.CurrentDepthRefinerTrackerV60('missing.pt','a'*64,**kwargs)
        for bad in [True,None,'bad','A'*64]:
            with self.assertRaises(ValueError):v60.CurrentDepthRefinerTrackerV60('missing.pt',bad)
        with self.assertRaises(FileNotFoundError):v60.CurrentDepthRefinerTrackerV60('/nonexistent/V60.pt','a'*64)

    def test_loader_testdouble_is_rejected_before_history_or_bridge(self):
        with patch.object(Path,'resolve',return_value=Path('/fixture/not_a_weight.pt')),\
             patch.object(v60,'CurrentCubeDepthRefinerV2',return_value=SimpleNamespace(qualified=True)):
            with self.assertRaisesRegex(ValueError,'Real fully loaded'):
                v60.CurrentDepthRefinerTrackerV60('/fixture/not_a_weight.pt','a'*64)

    def test_unbound_math_fixture_is_never_a_live_capability(self):
        t=unbound_math_fixture()
        with self.assertRaisesRegex(ValueError,'genuine'):t.validate_refiner()
        cache=PublicTargetFrameCache(t);bridge=object.__new__(PublicMotionBridge);bridge.tracker=cache
        with self.assertRaises(ValueError):ApprovedDepthRefinerTrackerCacheBindingV60(bridge,t,cache)
        with patch.object(t,'validate_refiner',return_value=True):
            with self.assertRaisesRegex(ValueError,'real methods'):
                ApprovedDepthRefinerTrackerCacheBindingV60(bridge,t,cache)

    def test_NN_first_does_not_call_raw_or_append_raw_bias(self):
        t=unbound_math_fixture();obs=public_observation();p=nn_prediction(t,obs)
        t.refiner=SimpleNamespace(predict=lambda _:p)
        before=obs['observation/images']['front'].copy()
        with patch.object(t,'validate_refiner',return_value=True),\
             patch.object(t.tracker,'observe',side_effect=AssertionError('raw must not run when NN available')):
            target,metadata=t.observe(obs)
        self.assertEqual(len(t.history),1);self.assertEqual(t.last_measurement_metadata['accepted_source'],'NN_CURRENT_DEPTH')
        self.assertFalse(t.last_measurement_metadata['velocity_observed'])
        np.testing.assert_array_equal(target.pose.position,p['centre'])
        np.testing.assert_array_equal(target.points_world,box_surface(target.pose,(.05,.05,.05),256))
        self.assertEqual(len(t.last_measured_cloud),metadata['measured_points'])
        self.assertFalse(np.array_equal(target.points_world,t.last_measured_cloud))
        np.testing.assert_array_equal(obs['observation/images']['front'],before)

    def test_raw_preview_does_not_pollute_final_history_or_velocity(self):
        t=unbound_math_fixture();previous=np.array([.20,0,1.]);current=np.array([.205,0,1.])
        t.history.append((0.,previous));t.last_timestamp=0.;t._accepted_source='FAR_R100_DEVELOPMENT_CONSISTENCY'
        t.refiner=SimpleNamespace(predict=lambda _:dict(checkpoint_sha256=t.expected_refiner_sha256,
            refiner_available=False,measurement=dict(refiner_available=False)))
        def provisional(obs):
            t.history.append((obs['timestamp'],np.array([99.,99.,99.])))
            return ObjectState('target',obs['timestamp'],Pose(current),component(current),np.full(3,99.),np.zeros(3),1.),dict(geometry_completion=False,visible_views=4)
        with patch.object(t,'validate_refiner',return_value=True),patch.object(t.tracker,'observe',side_effect=provisional):
            state,_=t.observe(public_observation(.05))
        self.assertEqual(len(t.history),2)
        np.testing.assert_allclose(state.linear_velocity,[.1,0,0],atol=1e-14)
        self.assertFalse(any(np.max(np.abs(row[1]))>2 for row in t.history))

    def test_NN_unknown_cannot_escape_via_biased_raw_centre(self):
        t=unbound_math_fixture();t.refiner=SimpleNamespace(predict=lambda _:dict(
            checkpoint_sha256=t.expected_refiner_sha256,refiner_available=False,measurement=dict(refiner_available=False)))
        raw=ObjectState('target',0.,Pose([.2,0,1.]),component([.04,0,1.]),np.zeros(3),np.zeros(3),1.)
        with patch.object(t,'validate_refiner',return_value=True),\
             patch.object(t.tracker,'observe',return_value=(raw,dict(geometry_completion=False,visible_views=4))):
            with self.assertRaisesRegex(ValueError,'no_near_shell_fallback'):t.observe(public_observation())
        self.assertIsNone(t.last_current_target);self.assertEqual(len(t.history),0)
        self.assertFalse(t.last_measurement_metadata['measurement_available'])

    def test_route_change_keeps_current_pose_but_velocity_is_unobserved(self):
        t=unbound_math_fixture();t.history.extend([(0.,np.array([.2,0,1.])),(.05,np.array([.205,0,1.]))])
        t.last_velocity=np.array([.1,0,0]);t._accepted_source='FAR_R100_DEVELOPMENT_CONSISTENCY'
        target,_=t._accept([.21,0,1.],1.,component([.21,0,1.]),4,.1,'NN_CURRENT_DEPTH')
        self.assertEqual(len(t.history),1);self.assertTrue(target.visible)
        np.testing.assert_array_equal(target.pose.position,[.21,0,1.])
        np.testing.assert_array_equal(target.linear_velocity,[0,0,0])
        self.assertTrue(t.last_measurement_metadata['accepted_history_reset_for_route_change'])
        self.assertFalse(t.last_measurement_metadata['velocity_observed'])
        self.assertEqual(t.last_measurement_metadata['velocity_source'],'unobserved_zero_initialization')

    def test_source_and_measurement_contracts_remain_distinct(self):
        rng=np.random.default_rng(261103700);t=unbound_math_fixture()
        for i in range(5):
            center=np.array([.20+.001*i,0,1.]);cloud=component(center)+rng.uniform(-1e-4,1e-4,(24,3))
            state,raw=t._accept(center,.8,cloud,4,.05*i,'NN_CURRENT_DEPTH')
            np.testing.assert_array_equal(np.ptp(state.points_world,axis=0),np.ptp(box_surface(state.pose,(.05,.05,.05),256),axis=0))
            self.assertEqual(raw['measured_points'],24);self.assertTrue(raw['geometry_completion'])
        self.assertTrue(t.last_measurement_metadata['velocity_observed']);self.assertEqual(len(t.history),5)


if __name__=='__main__':unittest.main()
