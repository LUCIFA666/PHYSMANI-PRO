from pathlib import Path
import sys,types
package=types.ModuleType('tests');package.__path__=[str(Path(__file__).resolve().parent)];sys.modules['tests']=package
from dataclasses import FrozenInstanceError
from types import SimpleNamespace as NS
import unittest
import numpy as np
from our_mani.contracts import Pose,Reachability
from our_mani.multitask.terminal_candidate_generator_v57 import TerminalCandidateGenerator,ClearanceCandidate
from our_mani.multitask.placement_clearance_v50 import PlacementClearance
from tests import test_task_reference_v37 as fixture


class Tests(unittest.TestCase):
    def test_clearance_receipt_is_constructed_inside_the_frozen_actual_candidate(self):
        actor,_,measured=fixture.PartialApproachTests().setup_scene(.3);c=actor.controller;m=actor.continuous_reference_manager;old=c.generator
        class Gate:
            def check(self,robot,pose,*args):return Reachability(True,'CPU_fixture_only',joints=robot.joints.copy(),joint_velocities=np.zeros_like(robot.joints))
        g=TerminalCandidateGenerator(c.library,Gate(),old.collision_check,old.config)
        class Clearance:
            def adjust(self,state,forecast,template,reference,target,arrival):return Pose(reference.position+[0,0,.01],reference.rotation),{'vertical_shift_m':.01}
        g.placement_clearance=Clearance();values=g.generate(m.current_state,m.current_forecast,0,9.5,c.origin)
        self.assertGreater(len(values),0)
        for value in values:
            self.assertIsInstance(value,ClearanceCandidate);self.assertEqual(value.placement_clearance,{'vertical_shift_m':.01})
            with self.assertRaises(FrozenInstanceError):value.placement_clearance={}

    def test_completion_wrapper_runs_the_final_feedback_controller_instead_of_abandoned_original(self):
        actor,_,measured=fixture.PartialApproachTests().setup_scene(.3);m=actor.continuous_reference_manager
        clear=object.__new__(PlacementClearance);clear.actor=actor;clear.events=[];clear.binding={'CPU_fixture':True}
        clear.install_completion(actor.controller)
        self.assertIs(actor.controller._phase_done.__self__,actor.controller)
        template=actor.controller.library.for_phase('deposit')[0]
        actor.controller.goal=NS(phase='deposit',template=template,selected_at=0.,placement_clearance={'vertical_shift_m':.02})
        # A far endpoint simply fails the original phase tolerance gate.
        self.assertFalse(actor.controller._phase_done(m.current_state,Pose([9,0,1]),public_geometry=True))
        self.assertEqual(len(clear.events),1);self.assertTrue(clear.events[0]['original_phase_verifier_called'])


if __name__=='__main__':unittest.main()
