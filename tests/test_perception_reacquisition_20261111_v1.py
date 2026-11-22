"""A new measurement producer cannot invent missing frames or NN authority."""
from collections import deque,Counter
from types import SimpleNamespace
from unittest.mock import patch
import unittest
import numpy as np
from our_mani.contracts import Pose,ObjectState
from our_mani.benchmark_perception import BoxPrior,box_surface
from our_mani.perception_repair_20261111_v1.tracker import ReacquiringPublicCubeTracker,SOURCE
from tests.test_perception_yaw_cube_20261111_v1 import component as surface_component


def component(**kwargs):
    return dict(surface_component(**kwargs),n=600,visible_views=4,initial_height_gate=True)


class FakeBase:
    def __init__(self,error=None):
        self.history=deque(maxlen=5);self.last_velocity=np.zeros(3);self.error=error;self.prior=BoxPrior()
        self.sphere=SimpleNamespace(center=np.array([.2,0.,1.]));self.last_measured_cloud=np.ones((64,3))
        self.last_measurement_metadata={};self.calls=0
    def observe(self,observation):
        self.calls+=1
        self.last_measurement_metadata=dict(accepted_source='NN' if self.calls%2 else 'FAR',position_measurement_qualified=True)
        if self.error:raise ValueError(self.error)
        now=observation['timestamp'];pose=Pose(np.array([.2+.2*now,.2,.775]))
        return ObjectState('target',now,pose,box_surface(pose,(.05,)*3),np.zeros(3),np.zeros(3),1.),{}


def tracker(base):
    t=ReacquiringPublicCubeTracker.__new__(ReacquiringPublicCubeTracker)
    t.base=base;t.history=deque(maxlen=5);t.last_velocity=np.zeros(3);t.counts=Counter()
    t.last_measurement_metadata=t.last_measured_cloud=None
    return t


class ReacquisitionContracts(unittest.TestCase):
    def test_source_switch_keeps_actual_history_and_measured_velocity(self):
        t=tracker(FakeBase())
        for i in range(5):o,_=t.observe({'timestamp':i*.05})
        self.assertEqual(len(t.history),5);np.testing.assert_allclose(o.linear_velocity,[.2,0,0],atol=1e-12)
        self.assertEqual(t.last_measurement_metadata['accepted_source'],SOURCE)
        o,_=t.observe({'timestamp':.35})
        self.assertEqual(len(t.history),1);np.testing.assert_array_equal(o.linear_velocity,[0,0,0])
    def test_reacquisition_needs_a_recent_real_identity_and_current_cloud(self):
        t=tracker(FakeBase('NN_unavailable_and_no_current_far_component'))
        with self.assertRaises(ValueError):t.observe({'timestamp':0.})
        t.history.append((0.,np.array([.2,.2,.775])))
        with patch('our_mani.perception_repair_20261111_v1.tracker.public_red_components',return_value=([component()],{})):
            o,_=t.observe({'timestamp':.05})
        np.testing.assert_allclose(o.pose.position,[.2,.2,.775],atol=.001)
        self.assertEqual(o.timestamp,.05);self.assertFalse(t.last_measurement_metadata['position_measurement_qualified'])
        self.assertFalse(t.last_measurement_metadata['observations_fabricated'])
    def test_ambiguous_components_and_nonmeasurement_errors_never_bypass_checks(self):
        t=tracker(FakeBase('NN_unavailable_and_no_current_far_component'));t.history.append((0.,np.array([.2,.2,.775])))
        with patch('our_mani.perception_repair_20261111_v1.tracker.public_red_components',return_value=([component(),component(center=(.24,.2,.775))],{})):
            with self.assertRaises(ValueError):t.observe({'timestamp':.05})
        self.assertEqual(len(t.history),1)
        t.base.error='checkpoint identity changed'
        with patch('our_mani.perception_repair_20261111_v1.tracker.public_red_components') as components:
            with self.assertRaisesRegex(ValueError,'checkpoint identity'):t.observe({'timestamp':.1})
            components.assert_not_called()


if __name__=='__main__':unittest.main()
