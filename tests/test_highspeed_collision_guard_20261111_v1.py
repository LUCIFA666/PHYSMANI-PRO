"""An immediately safe reference can still make bounded braking collide."""
import unittest
from types import SimpleNamespace
import numpy as np
from our_mani.execution_interface_20261111_v1.executor import JointLimits
from our_mani.highspeed_control_20261111_v1.collision_guard import CollisionStoppingGuard
from our_mani.highspeed_control_20261111_v1.tracking_v4 import StoppableTrackingReferencePlanner


class CollisionGuardContracts(unittest.TestCase):
    def test_rejects_unsafe_future_brake_without_advancing_physics(self):
        def fk(q):
            T=np.eye(4);T[:3,3]=np.asarray(q)[:3]+[0,0,.8];return T
        class Response:
            def predict_joint(self,q,dq,qref,vref,aref):return dict(q=qref.copy(),dq=(qref-q)/.05)
        class Helper:
            def qualify_action(self,backend,state,action,**kw):
                collision=action['joint_goal'][0]>.145
                return (None,'action_arm_static_collision',{'physical_steps':0}) if collision else (action['joint_goal'],None,{'physical_steps':0})
        limits=JointLimits([-3.]*7,[3.]*7,[3.]*7)
        planner=StoppableTrackingReferencePlanner(fk,Response(),limits)
        q=np.zeros(7);v=q.copy();v[0]=1.8
        state=dict(time=0.,q=q,dq=q,actuator_state=dict(reference_positions=q,reference_velocities=v,reference_accelerations=q,reference_clock=0.))
        guard=CollisionStoppingGuard(planner,Helper(),SimpleNamespace())
        ref=q.copy();ref[0]=.12
        proposed=guard.action(ref,1.)
        ok,receipt=guard.certify(state,ref,1.)
        self.assertFalse(ok);self.assertGreater(receipt['step'],0)
        action,receipt=guard.select(state,proposed,1.)
        self.assertLess(receipt['proposal_weight'],1.)
        self.assertTrue(receipt['stopping_continuation_qualified'])
        self.assertEqual(receipt['physical_steps'],0)
        self.assertLess(action['joint_goal'][0],ref[0])
        self.assertIsNone(planner.last_applied)


if __name__=='__main__':unittest.main()
