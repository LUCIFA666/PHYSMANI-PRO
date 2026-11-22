import unittest
from copy import deepcopy
from types import SimpleNamespace
import numpy as np

from tests.test_direct_counted_setpoint import Environment
from our_mani.multitask.continuous_direct_counted_setpoint_v3 import CountedDirectDispatcher, SCHEMA
from our_mani.multitask.protocol import MultitaskProtocolError


class TimedEnvironment(Environment):
    @property
    def time(self): return self.ledger.ticks*self.ledger.dt


class IssuedManager:
    def __init__(self): self.confirmations=[]
    def confirm_actual_issued(self,**kwargs): self.confirmations.append(deepcopy(kwargs))


class ContinuousDirectTests(unittest.TestCase):
    def test_execution_receipt_version_matches_real_v3_helper(self):
        self.assertEqual(SCHEMA, 'r10_continuous_reference_direct_counted_setpoint_v3')

    def command(self,q):
        from our_mani.multitask.continuous_reference_feedforward_v3 import SCHEMA
        return dict(command='joint_trajectory',joints=q,opening=1.,phase='approach',
            feedforward_execution=dict(schema=SCHEMA,issued=False,dispatch_authorized=True,reference_gate_passed=True,
                reference_to_joints=q,reference_to_velocity=[.1,.2],candidate_id='planned',time=0.))

    def test_intent_is_not_actual_issue_until_successful_counted_dispatch(self):
        env=TimedEnvironment();d=CountedDirectDispatcher();m=IssuedManager();d.bind_manager(m)
        command=self.command([.01,.02])
        self.assertEqual(m.confirmations,[])
        d.dispatch(env,command)
        self.assertEqual(env.ledger.ticks,1)
        self.assertEqual(len(m.confirmations),1)
        row=m.confirmations[0]
        self.assertEqual(row['time'],.05)
        np.testing.assert_array_equal(row['joints'],env.arm.target)
        np.testing.assert_array_equal(row['nominal_velocity'],[.1,.2])
        self.assertEqual(row['execution']['tick_after'],1)

    def test_failed_native_tick_does_not_confirm_unexecuted_intent(self):
        env=TimedEnvironment();d=CountedDirectDispatcher();m=IssuedManager();d.bind_manager(m)
        def fail(phase): raise RuntimeError('native counted step failed')
        env._tick=fail
        with self.assertRaises(RuntimeError): d.dispatch(env,self.command([.01,.02]))
        self.assertEqual(m.confirmations,[])
        self.assertEqual(d.events,[])

    def test_unbound_or_foreign_certificate_is_rejected_before_setter(self):
        env=TimedEnvironment();d=CountedDirectDispatcher()
        with self.assertRaises(MultitaskProtocolError): d.dispatch(env,self.command([.01,.02]))
        np.testing.assert_array_equal(env.arm.target,[0.,0.])
        m=IssuedManager();d.bind_manager(m)
        command=self.command([.01,.02]);command['joints']=[.5,.5]
        with self.assertRaises(MultitaskProtocolError): d.dispatch(env,command)
        self.assertEqual(env.ledger.ticks,0);self.assertEqual(m.confirmations,[])

    def test_ordinary_hold_confirmation_does_not_invent_nominal_velocity(self):
        env=TimedEnvironment();d=CountedDirectDispatcher();m=IssuedManager();d.bind_manager(m)
        d.dispatch(env,dict(command='hold',phase='warmup'))
        self.assertIsNone(m.confirmations[0]['nominal_velocity'])
        self.assertFalse(m.confirmations[0]['execution']['persistent_reference'])


if __name__=='__main__': unittest.main()
