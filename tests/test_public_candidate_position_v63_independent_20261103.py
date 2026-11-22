"""V63 delta only: pure public route guards and unchanged control AST."""
import ast
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock
import numpy as np

from our_mani.cube_position_measurement_20261103_v8 import prediction
from our_mani.local_mpc_control_20261103 import depth_position_goal_perception_v63 as tracker
from our_mani.local_mpc_control_20261103 import public_candidate_position_v63 as producer
from our_mani.local_mpc_control_20261103 import public_candidate_position_v62 as old


def route(availability):
    probabilities=np.zeros((4,2,64,64),np.float32);probabilities[:,0]=1
    crops={'valid_crop':np.ones((4,64,64),bool)}
    proposals=[]
    for threshold,flag in zip((.995,.9),availability):
        proposals.append((threshold,dict(geometry_profile=prediction.GEOMETRY_VERSION,refiner_available=flag,
            centre=[0.,0.,1.] if flag else None,reasons=[] if flag else ['unknown'],
            measured_points=24 if flag else 0,visible_views=3 if flag else 0,geometry_confidence=1. if flag else 0.)))
    return prediction.route_current_public_proposals(crops,probabilities,proposals)


class V63Independent(unittest.TestCase):
    def test_all_three_public_routes_validate_without_granting_generic_qualification(self):
        for flags,expected in [([True],'first_0p995'),([False,True],'fallback_0p9'),([False,False],'UNKNOWN')]:
            result=route(flags);checked=tracker.validate_position_proposal_trace(result)
            self.assertEqual(checked['proposal_route'],expected)
            self.assertTrue(checked['threshold_change_is_same_position_source'])
            self.assertFalse(checked['route_uses_GT_or_future'])
            self.assertFalse(result['raw_mask_qualified'])

    def test_false_trace_cutoff_shortcircuit_GT_or_unknown_centre_rejected(self):
        for change in (
            lambda x:x.update(selected_threshold=.995),
            lambda x:x.update(threshold=.999),
            lambda x:x.update(route_uses_GT_or_future=True),
            lambda x:x['proposal_trace'][0].update(geometry_available=True),
            lambda x:x['proposal_trace'][-1].update(measured_points=99),
        ):
            value=route([False,True]);change(value)
            with self.assertRaises(ValueError):tracker.validate_position_proposal_trace(value)
        unknown=route([False,False]);unknown['centre']=[0.,0.,1.]
        with self.assertRaises(ValueError):tracker.validate_position_proposal_trace(unknown)
        primary=route([True]);primary['thresholds_tried']=[.995,.9]
        with self.assertRaises(ValueError):tracker.validate_position_proposal_trace(primary)

    def test_threshold_change_does_not_reset_motion_or_attachment_anchor(self):
        bridge=SimpleNamespace(_clear_motion=Mock(),anchor=object(),attachment=object())
        before=(bridge.anchor,bridge.attachment)
        for flags in ([True],[False,True]):
            trace=tracker.validate_position_proposal_trace(route(flags))
            producer._reset_motion_history_for_route_change(bridge,trace)
        bridge._clear_motion.assert_not_called()
        self.assertEqual(before,(bridge.anchor,bridge.attachment))

    def test_no_diagnostic_injection_or_unbound_live_capability(self):
        for kwargs in ({'allow_unqualified_diagnostic':True},{'refiner':object()}):
            with self.assertRaises(TypeError):tracker.CurrentDepthPositionTrackerV63('missing','a'*64,**kwargs)
        with self.assertRaisesRegex(ValueError,'genuine'):
            object.__new__(tracker.CurrentDepthPositionTrackerV63).validate_refiner()
        with self.assertRaises(TypeError):producer.PublicCandidatePositionStateAssemblerV63(object())

    def test_runner_observer_and_coordinator_control_AST_equal_V62(self):
        new=ast.parse(Path(producer.__file__).read_text().replace('V63','V62').replace('v63','v62'))
        oldtree=ast.parse(Path(old.__file__).read_text())
        def get(tree,name,parent=None):
            nodes=tree.body if parent is None else next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==parent).body
            return next(n for n in nodes if isinstance(n,ast.FunctionDef) and n.name==name)
        for name,parent in [('run_public_candidate_position_v62',None),('sample','ObservedPublicCandidatePositionV62Backend'),('decide','CandidatePositionV62Coordinator')]:
            self.assertEqual(ast.dump(get(new,name,parent),include_attributes=False),ast.dump(get(oldtree,name,parent),include_attributes=False))


if __name__=='__main__':unittest.main()
