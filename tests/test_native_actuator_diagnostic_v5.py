"""CPU observer interfaces; no native simulator or policy execution."""
import unittest
from types import SimpleNamespace, MethodType
from unittest.mock import patch
import numpy as np
from our_mani.multitask.native_actuator_diagnostic_v5 import (
    snapshot, NativeActuatorDiagnosticBackend, diagnostic_binding,
)
from our_mani.multitask.protocol import MultitaskProtocolError
from pathlib import Path


class Joint:
    def is_control_loop_enabled(self): return True
    def is_motor_enabled(self): return True


class Object:
    def get_name(self): return 'diagnostic_ball'
    def get_handle(self): return 10
    def get_contact(self, *, get_contact_normal):
        assert get_contact_normal is True
        return [dict(contact_handles=[10,20],contact=list(range(9)))]


class Env:
    def __init__(self):
        self.ledger=SimpleNamespace(ticks=0,dt=.05)
        self.task=SimpleNamespace(t=0.,get_graspable_objects=lambda:[Object()])
        self.arm=SimpleNamespace(joints=[Joint(),Joint()],
            get_joint_positions=lambda:[0.,0.],get_joint_velocities=lambda:[.1,.2],
            get_joint_target_positions=lambda:[.01,.02],get_joint_target_velocities=lambda:[2.,2.],
            get_joint_forces=lambda:[3.,4.],get_joint_modes=lambda:[SimpleNamespace(name='FORCE',value=5)]*2)
    @property
    def time(self): return self.ledger.ticks*.05
    def physics_time(self): return self.task.t


class DiagnosticTests(unittest.TestCase):
    def test_snapshot_reads_only_and_captures_actual_API_payload(self):
        env=Env();row=snapshot(env,'before_policy')
        self.assertTrue(row['clock_unchanged']);self.assertEqual(env.ledger.ticks,0)
        self.assertEqual(row['queries']['joint_forces_or_torques']['value'],[3.,4.])
        self.assertEqual(row['queries']['graspable_object_last_physics_step_contacts']['value'][0]['contacts'][0]['contact_handles'],[10,20])
        self.assertTrue(all(x['available'] for x in row['queries'].values()))

    def test_unavailable_query_is_honest_and_clock_mutation_rejected(self):
        env=Env();env.arm.get_joint_forces=lambda:(_ for _ in ()).throw(RuntimeError('unsupported'))
        row=snapshot(env,'before_policy');self.assertFalse(row['queries']['joint_forces_or_torques']['available'])
        def changed(): env.task.t+=.05;return [1.,2.]
        env.arm.get_joint_forces=changed
        with self.assertRaises(MultitaskProtocolError): snapshot(env,'before_policy')

    def test_actor_receives_original_inputs_and_exact_command_object(self):
        env=Env();command={'command':'hold'};state=object();calls=[]
        actor=SimpleNamespace()
        def act(instance,value): calls.append(value);return command
        actor.act=MethodType(act,actor);actor.metrics_report=lambda: {'original':True}
        backend=NativeActuatorDiagnosticBackend();backend.instrument_actor(actor,env)
        self.assertIs(actor.act(state),command);self.assertEqual(calls,[state])
        self.assertEqual([r['stage'] for r in backend.diagnostic_events],['before_policy','after_policy'])
        self.assertFalse(actor.metrics_report()['native_metric_diagnostics']['diagnostic_values_supplied_to_policy'])

    def test_dispatch_reuses_original_dispatch_once_and_preserves_return(self):
        env=Env();backend=NativeActuatorDiagnosticBackend();command={'x':object()};result=object();calls=[]
        def dispatch(instance,actual_env,actual_command):
            self.assertIs(actual_env,env);self.assertIs(actual_command,command);calls.append(1)
            env.ledger.ticks+=1;env.task.t+=.05;return result
        with patch('our_mani.multitask.native_actuator_diagnostic_v5.ContinuousReferenceNativeBackend.dispatch',dispatch):
            self.assertIs(backend.dispatch(env,command),result)
        self.assertEqual(calls,[1]);self.assertEqual(env.ledger.ticks,1)
        self.assertEqual([r['stage'] for r in backend.diagnostic_events],['before_dispatch','after_dispatch'])

    def test_parent_command_uses_real_new_worker_and_no_GPU_or_TEST(self):
        from scripts import run_native_actuator_diagnostic_pair_v5_r10 as parent
        root=Path(__file__).resolve().parents[1]
        for method in ('cv_geometry','context_geometry'):
            argv=parent.command(method)
            self.assertTrue((root/argv[2]).is_file())
            self.assertEqual(argv[2],'scripts/run_geometry_native_actuator_diagnostic_v5_r10.py')
            self.assertEqual(argv[argv.index('--split')+1],'train')
            self.assertEqual(argv[argv.index('--raw-episode-id')+1],'episode104')
        self.assertLessEqual(parent.WALL_TIMEOUT+10.,300.)


if __name__=='__main__': unittest.main()
