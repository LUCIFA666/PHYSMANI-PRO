import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np

from our_mani.multitask.direct_counted_setpoint import apply_direct_setpoint, CountedDirectDispatcher
from our_mani.multitask.protocol import MultitaskProtocolError


class Arm:
    def __init__(self):
        self.q = np.zeros(2)
        self.target = self.q.copy()
        self.velocity = self.q.copy()
    def get_joint_positions(self): return self.q.tolist()
    def get_joint_velocities(self): return self.velocity.tolist()
    def get_joint_target_positions(self): return self.target.tolist()
    def set_joint_target_positions(self, values): self.target = np.asarray(values, float)


class Environment:
    def __init__(self):
        self.arm = Arm()
        self.ledger = SimpleNamespace(ticks=0, dt=.05, release_requests=0, grasp_requests=0)
        self.robot_model = SimpleNamespace(lower=np.full(2,-2.), upper=np.full(2,2.),
            velocity_limits=np.full(2,1.), acceleration_limits=np.full(2,6.))
        self.protocol = dict(gripper_velocity=.2)
        self._requested_opening = 1.
        self.calls = []
        self.gripper = SimpleNamespace(release=lambda:self.calls.append('release'),
            grasp=lambda obj:self.calls.append(('grasp',obj)),
            actuate=lambda opening,velocity:self.calls.append(('actuate',opening,velocity)))
        self.task = SimpleNamespace(get_graspable_objects=lambda:['ball'])
        self.response = 0.
    def _before_action(self): self.calls.append('guard')
    def _tick(self, phase):
        before = self.arm.q.copy()
        self.arm.q += self.response*(self.arm.target-self.arm.q)
        self.arm.velocity = (self.arm.q-before)/self.ledger.dt
        self.ledger.ticks += 1
        self.calls.append(('tick',phase))
    def command_joints(self,joints,*,opening,phase):
        current = np.asarray(self.arm.get_joint_positions())
        target = current+np.clip(np.asarray(joints)-current,-.05,.05)
        apply_direct_setpoint(self,target,opening=opening,phase=phase)
    def hold_tick(self): self.command_joints(self.arm.q,opening=self._requested_opening,phase='hold')


class DirectCountedTests(unittest.TestCase):
    def test_direct_reference_preserves_position_bound_gripper_and_single_tick(self):
        env=Environment()
        result=apply_direct_setpoint(env,[1.,3.],opening=0.,phase='approach')
        np.testing.assert_array_equal(result,[1.,2.])
        self.assertEqual(env.ledger.ticks,1)
        self.assertEqual(env.ledger.grasp_requests,1)
        self.assertEqual(env.calls[0],'guard')
        self.assertEqual(sum(isinstance(row,tuple) and row[0]=='tick' for row in env.calls),1)
        apply_direct_setpoint(env,[1.,2.],opening=0.,phase='approach')
        self.assertEqual(env.ledger.grasp_requests,1)
        apply_direct_setpoint(env,[1.,2.],opening=1.,phase='approach')
        self.assertEqual(env.ledger.release_requests,1)

    def test_invalid_reference_never_steps_or_changes_target(self):
        for joints,opening in (([np.nan,0.],1.),([0.],1.),([0.,0.],None)):
            env=Environment()
            with self.assertRaises(MultitaskProtocolError):
                apply_direct_setpoint(env,joints,opening=opening,phase='approach')
            self.assertEqual(env.ledger.ticks,0)
            np.testing.assert_array_equal(env.arm.target,[0.,0.])

    def _command(self, joints, tick=0):
        from our_mani.multitask.persistent_reference_feedforward import SCHEMA as COMMAND_SCHEMA
        return dict(command='joint_trajectory',joints=joints,opening=1.,phase='approach',
            feedforward_execution=dict(schema=COMMAND_SCHEMA,reference_gate_passed=True,
                issued=True,reference_to_joints=joints,candidate_id='fixed_candidate',time=tick*.05))

    def test_only_explicit_verified_stream_gets_direct_filter(self):
        env=Environment();dispatcher=CountedDirectDispatcher()
        result=dispatcher.dispatch(env,self._command([1.,1.]))
        np.testing.assert_array_equal(env.arm.target,[1.,1.])
        self.assertTrue(result['persistent_reference_execution']['persistent_reference'])
        dispatcher.dispatch(env,dict(command='joint_trajectory',joints=[1.,1.],opening=1.,phase='approach'))
        np.testing.assert_array_equal(env.arm.target,[.05,.05])
        bad=self._command([1.,1.],2);bad['feedforward_execution']['reference_gate_passed']=False
        before=env.ledger.ticks
        with self.assertRaises(MultitaskProtocolError): dispatcher.dispatch(env,bad)
        self.assertEqual(env.ledger.ticks,before)

    def test_actual_acceleration_violation_is_retained_not_relabelled(self):
        env=Environment();env.response=1.;dispatcher=CountedDirectDispatcher()
        dispatcher.dispatch(env,self._command([.01,.01]))
        dispatcher.dispatch(env,self._command([.05,.05],1))
        receipt=dispatcher.receipt()
        self.assertIsNone(receipt['events'][0]['backward_window_acceleration'])
        self.assertAlmostEqual(receipt['events'][1]['backward_window_acceleration'][0],12.)
        self.assertFalse(receipt['actual_observed_window_limits_passed'])
        self.assertEqual(receipt['violations'][0]['tick'],2)
        self.assertFalse(receipt['hardware_acceleration_limit_claimed'])

    def test_reference_certificate_cannot_authorize_other_joints_or_tick(self):
        for variant in ('joints','time','issued'):
            env=Environment();dispatcher=CountedDirectDispatcher()
            command=self._command([.01,.01])
            if variant=='joints': command['joints']=[.02,.02]
            elif variant=='time': command['feedforward_execution']['time']=.05
            else: command['feedforward_execution']['issued']=False
            with self.assertRaises(MultitaskProtocolError): dispatcher.dispatch(env,command)
            self.assertEqual(env.ledger.ticks,0)


if __name__=='__main__': unittest.main()
