"""Exercise the real current helper certificate through its actual dispatcher."""
from pathlib import Path
import sys,types
package=types.ModuleType('tests');package.__path__=[str(Path(__file__).resolve().parent)];sys.modules['tests']=package
from copy import deepcopy
import unittest
from tests import test_task_reference_v37 as fixture
from tests.test_continuous_reference_feedforward_v3_peer import environment
from our_mani.multitask.task_dispatch_v43 import CountedDirectDispatcher
from our_mani.multitask.protocol import MultitaskProtocolError


class Tests(unittest.TestCase):
    def setup_command(self):
        actor,model,measured=fixture.PartialApproachTests().setup_scene()
        m=actor.continuous_reference_manager
        self.assertTrue(m.approach_transit.try_approach(measured,9.5));m.pending_dispatch=m.pending
        certificate=deepcopy(m.pending);certificate.update(issued=False,dispatch_authorized=True)
        command=dict(command='joint_trajectory',phase='approach',opening=1.,public_command_kind='joint_trajectory',
            joints=m.pending['reference_to_joints'],joint_velocities=m.pending['reference_to_velocity'],feedforward_execution=certificate)
        env=environment(measured);dispatcher=CountedDirectDispatcher();dispatcher.bind_manager(m)
        return m,env,dispatcher,command

    def test_current_certificate_is_actually_dispatched_and_acknowledged(self):
        m,env,dispatcher,command=self.setup_command();before=env.ledger.ticks
        result=dispatcher.dispatch(env,command)
        self.assertEqual(env.ledger.ticks,before+1);self.assertEqual(m.last_confirmed['time'],env.time)
        self.assertTrue(result['persistent_reference_execution']['persistent_reference'])
        self.assertTrue(m.reference_events[-1]['dispatch_confirmed'])

    def test_wrong_helper_certificate_fails_before_physics(self):
        m,env,dispatcher,command=self.setup_command();before=env.ledger.ticks
        command['feedforward_execution']['schema']='wrong_source_version'
        with self.assertRaises(MultitaskProtocolError):dispatcher.dispatch(env,command)
        self.assertEqual(env.ledger.ticks,before)


if __name__=='__main__':unittest.main()
