from pathlib import Path
from types import SimpleNamespace as NS,MethodType
from dataclasses import replace
import json
import unittest
import numpy as np
from our_mani.contracts import Pose
from our_mani.multitask.interaction import InteractionLibrary,InteractionController,COMMAND_GOAL
from our_mani.multitask.bin_capture_region import capture_template,binding,TASK
import test_bin_physical_acquire_v234 as actual
ROOT=Path(__file__).resolve().parents[1]


class Tests(unittest.TestCase):
    def test_actual_high_bin_template_changes_only_control_endpoint_and_keeps_physical_canonical(self):
        config=json.loads((ROOT/'artifacts/framework/multitask_r10/bin_author_v428_setup'/
            (TASK+'_configuration.json')).read_text())
        library=InteractionLibrary.load(ROOT/config['learned_artifacts'][TASK]['interaction']['path'])
        source=library.for_phase('acquire')[0];policy=binding(ROOT,config,TASK)
        changed=capture_template(source,policy['control_vertical_shift_m'],policy['source_sha256'])
        self.assertEqual(changed.reference_time_basis,COMMAND_GOAL)
        np.testing.assert_allclose(changed.completion_geometry_xyzw,source.at(1.)[0].xyzw())
        self.assertAlmostEqual(changed.at(1.)[0].position[2]-source.at(1.)[0].position[2],.01)
        self.assertEqual(changed.position_tolerance_m,source.position_tolerance_m)
        self.assertEqual(changed.orientation_tolerance_rad,source.orientation_tolerance_rad)

    def test_real_gripper_gate_still_rejects_points_outside_original_region_even_inside_shifted_region(self):
        helper=actual.Tests();helper.setUp();c=helper.c;state=helper.state
        source=c.goal.template;changed=capture_template(source,.01,'a'*64)
        c.goal=NS(**dict(c.goal.__dict__,template=changed))
        c.attempts=NS(records=[]);c._gripper_goal_token=None;c.gripper_timing_events=[];c.last_diagnostics={}
        c._close_edge_issued_at=None
        c._gripper_opening=MethodType(InteractionController._gripper_opening,c)
        target=helper.target
        for offset,expected in ((.015,0.),(.025,1.)):
            c._close_edge_issued_at=None;c._gripper_goal_token=None
            state.robot=replace(state.robot,pose=Pose(target.position+[0.,0.,offset],target.rotation),contact=False,gripper_open=True)
            opening=c._gripper_opening(state,0.,target)
            self.assertEqual(opening,expected)
        self.assertEqual(c.gripper_timing_config.close_position_tolerance_m,.02)


if __name__=='__main__':unittest.main()
