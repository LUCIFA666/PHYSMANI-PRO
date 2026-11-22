from copy import deepcopy
from types import SimpleNamespace
import unittest
import numpy as np

from our_mani.local_mpc_control_20261103.hold_orientation_guard_prototype_20261103 import (
    HoldMicroOrientationGuard,bounded_object_nominal_destination)


def state():
    return dict(time=1.,q=np.zeros(7),dq=np.zeros(7),
        actuator_state=dict(reference_clock=1.,reference_valid=False),
        phase_state=dict(phase='hold',hold_tcp_quaternion=np.array([0.,0.,0.,1.])))


class Execution:
    requires_joint_state=requires_joint_goal=True
    config=SimpleNamespace(dt=.1)
    def __init__(self,micro,feasible=None):
        self.micro=np.asarray(micro,float);n=len(self.micro)
        self.out=(dict(tcp_quaternion=np.tile([0.,0.,0.,1.],(n,1))),{'unchanged_h':True},
            dict(feasible=np.ones(n,bool)if feasible is None else feasible,microsteps=2,predicted_actual_q=self.micro))
    def initialize_batch(self,actual,n):return actual,{'same_actual':True}
    def step_batch(self,x,u,h,dt):return self.out
    def fk_batch(self,q):
        angle=q[:,0]
        return dict(tcp_quaternion=np.c_[np.sin(angle/2),np.zeros((len(q),2)),np.cos(angle/2)])


class HoldGuardTests(unittest.TestCase):
    def test_catches_interior_excursion_despite_perfect_macroendpoint(self):
        micro=np.zeros((1,2,7));micro[0,0,0]=.08
        base=Execution(micro);guard=HoldMicroOrientationGuard(base,state(),maximum_rotation_rad=.04)
        action=dict(phase_index=np.array([3]));before=deepcopy(action)
        out=guard.step_batch({'time':np.array([1.])},action,{},.1)
        self.assertIs(out[0],base.out[0]);self.assertIs(out[1],base.out[1])
        self.assertFalse(out[2]['feasible'][0]);self.assertTrue(base.out[2]['feasible'][0])
        np.testing.assert_array_equal(action['phase_index'],before['phase_index'])

    def test_other_phases_exact_passthrough_and_mixed_hold_rows_only(self):
        micro=np.zeros((2,2,7));micro[:,:,0]=.1
        base=Execution(micro);guard=HoldMicroOrientationGuard(base,state(),maximum_rotation_rad=.04)
        x={'time':np.array([1.,1.])}
        self.assertIs(guard.step_batch(x,dict(phase_index=np.array([0,2])),{},.1),base.out)
        out=guard.step_batch(x,dict(phase_index=np.array([3,2])),{},.1)
        np.testing.assert_array_equal(out[2]['feasible'],[False,True])

    def test_valid_microsteps_do_not_restore_original_infeasible(self):
        micro=np.zeros((2,2,7));micro[:,:,0]=.02
        base=Execution(micro,np.array([False,True]));guard=HoldMicroOrientationGuard(base,state(),maximum_rotation_rad=.04)
        out=guard.step_batch({'time':np.ones(2)},dict(phase_index=np.array([3,3])),{},.1)
        np.testing.assert_array_equal(out[2]['feasible'],[False,True])
        np.testing.assert_array_equal(out[0]['tcp_quaternion'],base.out[0]['tcp_quaternion'])

    def test_missing_microsteps_or_phase_never_get_interpolated(self):
        base=Execution(np.zeros((1,2,7)));guard=HoldMicroOrientationGuard(base,state(),maximum_rotation_rad=.04)
        with self.assertRaises(ValueError):guard.step_batch({'time':np.ones(1)},{},{},.1)
        del base.out[2]['predicted_actual_q']
        with self.assertRaises(ValueError):guard.step_batch({'time':np.ones(1)},dict(phase_index=np.array([3])),{},.1)

    def test_fresh_actual_q_h_and_existing_hold_reference_required(self):
        base=Execution(np.zeros((1,2,7)));actual=state();guard=HoldMicroOrientationGuard(base,actual,maximum_rotation_rad=.04)
        self.assertIs(guard.initialize_batch(actual,1)[0],actual)
        for alter in ('q','clock','phase','holdR'):
            value=state()
            if alter=='q':value['q'][0]=.001
            if alter=='clock':value['actuator_state']['reference_clock']=.9
            if alter=='phase':value['phase_state']['phase']='close'
            if alter=='holdR':value['phase_state']['hold_tcp_quaternion']=np.array([.01,0,0,np.sqrt(1-.0001)])
            with self.subTest(alter=alter),self.assertRaises(ValueError):guard.initialize_batch(value,1)
        with self.assertRaises(ValueError):HoldMicroOrientationGuard(base,state(),maximum_rotation_rad=True)

    def test_bounded_nominal_is_only_explicit_destination_proposal(self):
        start=np.array([0.,0.,1.]);goal=np.array([0.,.3,1.4]);saved=goal.copy()
        point=bounded_object_nominal_destination(start,goal,maximum_step_m=.01)
        self.assertAlmostEqual(np.linalg.norm(point-start),.01)
        np.testing.assert_allclose((point-start)/.01,(goal-start)/.5)
        np.testing.assert_array_equal(goal,saved)
        np.testing.assert_array_equal(bounded_object_nominal_destination(start,start,maximum_step_m=.01),start)
        with self.assertRaises(ValueError):bounded_object_nominal_destination(start,goal,maximum_step_m=0.)


if __name__=='__main__':unittest.main()
