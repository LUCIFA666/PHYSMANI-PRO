from pathlib import Path
from dataclasses import replace
import unittest
import numpy as np
from our_mani.contracts import Pose
from our_mani.multitask.interaction import PolicyState
from our_mani.multitask.physmani_stage_lookahead_v594 import binding as delay_binding
from our_mani.multitask.physmani_author_stages_v574 import binding as stage_binding
from our_mani.multitask.physmani_bin_filter_progress_v631 import NormalBinProgressIntents,binding
import test_physmani_skill_v548 as previous
ROOT=Path(__file__).resolve().parents[1];NATIVE=ROOT.parent/'phsimani/PhysMani';TASK='put_rubbish_in_moving_bin'


class Tests(unittest.TestCase):
    def fixture(self):
        _,f=previous.Tests().fixture('acquire')
        p=NormalBinProgressIntents(f,stage_binding(ROOT,NATIVE,TASK),delay_binding(ROOT,NATIVE,TASK))
        p.phase_index=p.phase_order.index('acquire');p.select()
        target=p._completion_target(p.selected)
        f.state=PolicyState(replace(f.state.robot,pose=Pose(target.position+[.0112,0.,0.],target.rotation)),
            f.state.bodies,f.state.perception_metadata,f.state.role_bindings)
        return p,f,target

    def test_actual_own_short_path_can_extend_intent_but_never_relaxes_original_close(self):
        p,f,target=self.fixture();first=p.candidates('acquire')[0]
        np.testing.assert_allclose(first['pose'].position,target.position)
        p.observe_owned_RML_path_result(dict(error='released_modified_path_too_short',
            waypoint_modification=dict(selected_waypoints=1,cartesian_spacing_m=.01)))
        row=p.candidates('acquire')[0]
        self.assertLessEqual(row['public_filter_resolution_bias_m'],.01)
        self.assertAlmostEqual(np.linalg.norm(row['pose'].position-f.state.robot.pose.position),.02)
        self.assertEqual(p.qualify_opening(0.,1.)[0],1.)
        np.testing.assert_allclose(p._completion_target(row).position,target.position)
        self.assertTrue(binding(ROOT)['original_canonical_close_geometry_and_native_RML_unchanged'])

    def test_unknown_or_successful_path_and_actual_close_cannot_activate_bias(self):
        p,f,target=self.fixture()
        with self.assertRaises(ValueError):p.observe_owned_RML_path_result(dict(error=None,native_success=True))
        for receipt in [dict(error=None),dict(error='IKError'),dict(error='released_modified_path_too_short',
            waypoint_modification=dict(selected_waypoints=2,cartesian_spacing_m=.01))]:
            p.observe_owned_RML_path_result(receipt);self.assertNotIn('public_filter_resolution_bias_m',p.candidates('acquire')[0])
        p.observe_owned_RML_path_result(dict(error='released_modified_path_too_short',
            waypoint_modification=dict(selected_waypoints=1,cartesian_spacing_m=.01)))
        p.close_witness=dict(confirmed_after=1.)
        self.assertNotIn('public_filter_resolution_bias_m',p.candidates('acquire')[0])


if __name__=='__main__':unittest.main()
