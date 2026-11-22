"""Regression for short-delay prefixes being mistaken for complete skills."""
from pathlib import Path
import sys,types
package=types.ModuleType('tests');package.__path__=[str(Path(__file__).resolve().parent)];sys.modules['tests']=package
from types import SimpleNamespace
import unittest
import numpy as np
from our_mani.contracts import Reachability
from our_mani.intercept_planner import JointTrajectory
from our_mani.multitask.interaction import CandidateGenerator,_reference_at,_world_target
from our_mani.multitask.terminal_candidate_generator_v22 import TerminalCandidateGenerator
import tests.test_task_reference_v22 as helpers
from tests.test_feedback_interaction import robot


class RecordingGate:
    def __init__(self):self.poses=[]
    def check(self,robot,pose,*args):
        self.poses.append(pose)
        return Reachability(False,'original_gate_sentinel_rejection')


class Tests(unittest.TestCase):
    def test_short_arrival_targets_complete_TRAIN_endpoint_and_retains_hard_rejection(self):
        actor,_,measured=helpers.PartialApproachTests().setup_scene(.3)
        c=actor.controller;m=actor.continuous_reference_manager;old=c.generator
        gate=RecordingGate();generator=TerminalCandidateGenerator(c.library,gate,old.collision_check,old.config)
        result=generator.generate(m.current_state,m.current_forecast,0,9.5,c.origin)
        self.assertEqual(result,[]);self.assertGreater(len(gate.poses),0)
        template=c.library.for_phase('approach')[0]
        reference,v,w=_reference_at(m.current_forecast,'ball',template,measured.timestamp+.5+template.lag_seconds,c.origin)
        expected=_world_target(reference,np.zeros(3),np.zeros(3),template,1.)[0]
        for pose in gate.poses:
            np.testing.assert_allclose(pose.position,expected.position)
            np.testing.assert_allclose(pose.rotation,expected.rotation)
        legacy_gate=RecordingGate();legacy=CandidateGenerator(c.library,legacy_gate,old.collision_check,old.config)
        legacy.generate(m.current_state,m.current_forecast,0,9.5,c.origin)
        self.assertTrue(any(np.linalg.norm(p.position-expected.position)>1e-5 for p in legacy_gate.poses))

    def test_current_approach_endpoint_requires_fresh_roll_without_resetting_phase_epoch(self):
        _,c,m=helpers.fixture();c.goal=SimpleNamespace(phase='approach',selected_at=.1,phase_budget=1.5,
            template=SimpleNamespace(reference_time_basis='current'))
        m.plan=JointTrajectory(0.,.5,np.zeros(7),np.zeros(7),np.zeros(7),np.zeros(7));m.invalidated=False
        m.capture_at_current_endpoint=lambda observed:False
        called=[];m.roll_current_reference=lambda observed:called.append(observed.timestamp) or False
        self.assertIsNone(m.check_stream(robot(np.zeros(7),np.zeros(7),.5),.05))
        self.assertEqual(called,[.5]);self.assertEqual(c.goal.selected_at,.1);self.assertEqual(c.goal.phase_budget,1.5)


if __name__=='__main__':unittest.main()
