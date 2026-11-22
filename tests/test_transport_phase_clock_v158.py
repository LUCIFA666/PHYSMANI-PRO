from pathlib import Path
from types import SimpleNamespace as NS
import json,unittest
import numpy as np
from our_mani.contracts import Pose,ObjectState,RobotState,Reachability
from our_mani.dynamics import ConstantVelocityDynamics
from our_mani.multitask.interaction import CandidateConfig,PolicyState
from our_mani.multitask.normal_cup_primary_library_v109 import PrimarySequenceLibrary
from our_mani.multitask.normal_cup_native_v111 import library_scope
from our_mani.multitask.transport_phase_clock_v158 import TransportPhaseClock
from our_mani.multitask.transport_endpoint_candidates_v160 import TransportEndpointCandidateGenerator
from our_mani.multitask.terminal_candidate_generator_v57 import TerminalCandidateGenerator
from our_mani.multitask.cup_transport_native_v161 import install_confirmation

ROOT=Path(__file__).resolve().parents[1];TASK='place_cups_on_rotating_frame'


class Collision:
    def __init__(self,feasible=True):self.feasible=feasible;self.bindings=[]
    def bind_candidate(self,**kwargs):self.bindings.append(kwargs)
    def __call__(self,**kwargs):return dict(feasible=self.feasible,reason='blocked_fixture',evidence={})


class Tests(unittest.TestCase):
    def setUp(self):
        self.protocol=json.loads((ROOT/'artifacts/framework/multitask_r10/cup_public_support_v151_setup/configuration.json').read_text())
        self.library=PrimarySequenceLibrary.load(ROOT/self.protocol['learned_artifacts'][TASK]['interaction']['path'])
        self.config=CandidateConfig(**self.protocol['candidate_config']);self.index=self.library.phase_order.index('transport')
        self.clock=TransportPhaseClock(self.library,self.config)
    def start(self):self.clock.confirm(self.index,6.55,6.6,8.85)
    def generator(self,cls=TransportEndpointCandidateGenerator,collision=True):
        reach=NS(check=lambda *args:Reachability(True,'fixture',joints=np.zeros(7),joint_velocities=np.zeros(7)))
        port=Collision(collision)
        with library_scope(ROOT,self.protocol,TASK):g=cls(self.library,reach,port,self.config)
        if isinstance(g,TransportEndpointCandidateGenerator):g.transport_clock=self.clock
        return g,port
    def scene(self,now):
        robot=RobotState(now,Pose([.2,0.,.9]),np.zeros(3),np.zeros(3),np.zeros(7),np.zeros(7),gripper_open=False,contact=True)
        body=ObjectState('spoke0',now,Pose([.4,.1,.9]),np.array([[.4,.1,.9],[.41,.1,.9],[.4,.11,.9]]),np.zeros(3),np.zeros(3))
        meta={'bodies':{'spoke0':dict(directly_visible=True,registration_valid=True,orientation_observable=True,motion_mode='external')}}
        state=PolicyState(robot,{'spoke0':body},meta,{'selected_spoke':('spoke0',),'cup0':('cup0',)})
        return state,ConstantVelocityDynamics().predict([body],1.1,.05)
    def test_unissued_proposals_and_budget_queries_do_not_start_clock(self):
        self.assertIsNone(self.clock.remaining(self.index,6.55))
        self.assertAlmostEqual(self.clock.provisional_deadline(self.index,6.55,9.),8.85)
        self.assertFalse(self.clock.epochs)
    def test_one_actual_counted_dispatch_starts_original_maximum_deadline(self):
        with self.assertRaises(ValueError):self.clock.confirm(self.index,6.55,6.56,8.85)
        self.assertFalse(self.clock.epochs);self.start()
        self.assertAlmostEqual(self.clock.epochs[self.index]['deadline'],8.85)
        self.assertAlmostEqual(self.clock.epochs[self.index]['original_maximum_phase_budget'],2.3)
    def test_later_reference_reissues_cannot_restart_or_extend_clock(self):
        self.start();self.clock.confirm(self.index,7.55,7.6,9.9)
        self.assertAlmostEqual(self.clock.epochs[self.index]['issued_at'],6.55)
        self.assertAlmostEqual(self.clock.epochs[self.index]['deadline'],8.85)
        self.assertAlmostEqual(self.clock.remaining(self.index,7.6),1.25)
    def test_global_budget_can_only_shorten_phase_and_expiration_is_zero(self):
        end=self.clock.provisional_deadline(self.index,6.55,7.)
        self.clock.confirm(self.index,6.55,6.6,end)
        self.assertEqual(self.clock.remaining(self.index,7.05),0.)
        with self.assertRaises(ValueError):self.clock.confirm(self.index,7.05,7.1,9.)
    def test_elapsed_credit_preserves_real_selection_time_and_original_deadline(self):
        self.start();state,forecast=self.scene(7.55);old,_=self.generator(TerminalCandidateGenerator)
        self.assertFalse(old.generate(state,forecast,self.index,2.45,Pose([.2,0.,.9])))
        g,port=self.generator();candidates=g.generate(state,forecast,self.index,2.45,Pose([.2,0.,.9]))
        self.assertTrue(candidates);self.assertTrue(port.bindings)
        for c in candidates:
            self.assertEqual(c.selected_at,7.55);self.assertLessEqual(c.selected_at+c.phase_budget,8.85+1e-8)
            self.assertLessEqual(c.arrival_time+self.config.dt,8.85+1e-8)
            self.assertLessEqual(c.phase_budget+c.suffix_budget,2.45+1e-8)
    def test_original_collision_gate_still_rejects_every_endpoint_after_credit(self):
        self.start();state,forecast=self.scene(7.55);g,port=self.generator(collision=False)
        self.assertFalse(g.generate(state,forecast,self.index,2.45,Pose([.2,0.,.9])))
        self.assertTrue(port.bindings);self.assertIn('dynamic_collision_blocked_fixture',g.last_diagnostics['rejections'])
    def test_endpoint_arrival_cannot_cross_original_phase_deadline(self):
        self.start();state,forecast=self.scene(8.7);g,_=self.generator()
        self.assertFalse(g.generate(state,forecast,self.index,1.3,Pose([.2,0.,.9])))
        self.assertIn('original_transport_deadline_before_arrival',g.last_diagnostics['rejections'])
    def test_confirmation_hook_waits_for_original_backend_acknowledgement(self):
        def original(**kwargs):
            if not kwargs['execution']['validated']:raise ValueError('original backend rejected')
        manager=NS(confirm_actual_issued=original,approach_transit=NS(active=dict(phase='transport',phase_index=self.index,deadline=8.85)),
            plan_kind='approach_transit',plan_candidate_id='partial_approach:fixture',qualification_failed=False)
        install_confirmation(manager,self.clock)
        command=dict(feedforward_execution=dict(candidate_id='partial_approach:fixture'))
        e=dict(validated=False,actual_issued_at=6.55,actual_window_velocity_within_limits=True,actual_window_acceleration_within_limits=True)
        with self.assertRaises(ValueError):manager.confirm_actual_issued(command=command,execution=e,time=6.6)
        self.assertFalse(self.clock.epochs);e['validated']=True
        manager.confirm_actual_issued(command=command,execution=e,time=6.6)
        self.assertAlmostEqual(self.clock.epochs[self.index]['issued_at'],6.55)


if __name__=='__main__':unittest.main()
