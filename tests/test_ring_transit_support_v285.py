from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
import json
import unittest
import numpy as np
from our_mani.contracts import Pose
from our_mani.multitask import ring_transit_support_v285 as policy
from our_mani.multitask.high_peg_ring_runtime_v243 import RuntimeRingAffordance
from our_mani.multitask.high_peg_operational_gauge_v244 import certified_gauge_evidence
from our_mani.multitask.perception import RegistrationConfig
from our_mani.multitask.bounded_transit_v180 import BoundedApproach

ROOT=Path(__file__).resolve().parents[1]


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol=json.loads((ROOT/'artifacts/framework/multitask_r10/high_peg_depth_stop_v281_setup/configuration.json').read_text())
        cls.task=policy.TASK
        settings=cls.protocol['high_peg_ring_runtime_v243'][cls.task]
        cls.capability=RuntimeRingAffordance.load(ROOT,cls.protocol['ring_affordances'][cls.task],replay_report=settings['replay_report'])

    def fixture(self):
        c=deepcopy(self.protocol);c['ring_transit_support_v285']={self.task:policy.binding(ROOT,c,self.task)}
        body=json.loads((ROOT/c['learned_artifacts'][self.task]['geometry']['path']).read_text())['bodies']['ring']
        vertices=np.asarray(body['collision_vertices_body'],float)
        raw=Pose([0.,0.,-vertices[:,2].min()-.001])
        components=[NS(name=p['name'],points_body=np.asarray(p['points_body'],float)) for p in body['components']]
        clouds={'ring':{p.name:raw.apply(p.points_body)+[0.,0.,.0015] for p in components}}
        symmetry=self.capability.symmetry;group=np.asarray(symmetry.certificate['group_rotations_body']);center=np.asarray(symmetry.certificate['center_body'])
        poses=[Pose(raw.position+raw.rotation@(center-g@center),raw.rotation@g) for g in group]
        evidence=certified_gauge_evidence(symmetry,representative_body_pose=raw,body_id='ring',timestamp=1.,
            root_candidate_poses=poses,candidate_rms_m=[.001]*len(poses),body_in_root=Pose([0.,0.,0.]),
            registration_valid=True,directly_visible=True,all_retained_hypotheses_provided=True)
        meta=dict(directly_visible=True,registration_valid=True,mode_provenance='caller_controller_touch_action_proprioception',rotation_equivalence_evidence=evidence)
        state=NS(robot=NS(contact=True,timestamp=1.,pose=Pose([0.,0.,.2])),bodies={'ring':NS(visible=True,timestamp=1.,pose=raw)},
            perception_metadata={'bodies':{'ring':meta}})
        result=dict(valid=True,rotation_ambiguous=True,pose=raw)
        tracker=NS(_registration=lambda *args:result,_current_timestamp=1.,_current_views=None,
            templates={'ring':NS(components=components)},config=RegistrationConfig())
        raw_config=deepcopy(c['policy_overlay']['configurations']['persistent_reference']['approach_transit'])
        m=NS(confirmations=[],approach_transit=NS(normalized=raw_config),check_stream=object())
        ghost=NS(get_matrix=lambda:np.eye(4))
        actor=NS(map=NS(map=[{'name':'workspace'}],static_ghosts=[ghost]),tracker=tracker,
            controller=NS(goal=None,feedback_step_calls=0,generator=NS(config=NS(maximum_observation_age=.1))),
            continuous_reference_manager=m,collision_port=NS(ring_affordance=self.capability),attachments={},metrics_report=lambda:{})
        def original_attach(s,modes,invalid):
            if not s.robot.contact or modes.get('ring')!='attached':actor.attachments.pop('ring',None)
        actor._attachments=original_attach
        with patch.object(policy,'read_native_mesh',return_value=(vertices,[],None)),patch.object(policy,'horizontal_rectangle',return_value=dict(height=0.,xy_min=[-1.,-1.],xy_max=[1.,1.])):
            policy.install(actor,c,ROOT,self.task)
        return actor,state,clouds,result,vertices

    def test_real_TRAIN_ring_CAD_support_preserves_raw_tracker_and_next_stop_hook(self):
        actor,state,clouds,result,vertices=self.fixture();hook=actor.continuous_reference_manager.check_stream
        self.assertIsInstance(actor.continuous_reference_manager.approach_transit,BoundedApproach)
        self.assertEqual(actor.continuous_reference_manager.approach_transit.config.maximum_TCP_step_m,.2)
        original=state.bodies['ring'].pose.position.copy()
        self.assertIs(actor.tracker._registration({'ring':Pose([0.,0.,0.])},clouds),result)
        actor._attachments(state,{'ring':'attached'},set())
        attachment=actor.attachments['ring'];world=state.robot.pose.position+state.robot.pose.rotation@attachment.tool_to_body.position
        self.assertGreater((vertices+world)[:,2].min(),0.)
        np.testing.assert_array_equal(state.bodies['ring'].pose.position,original)
        self.assertIs(actor.continuous_reference_manager.check_stream,hook)

    def test_missing_D4_or_ambiguous_identity_cannot_create_supported_attachment(self):
        actor,state,clouds,_,_=self.fixture();actor.tracker._registration({'ring':Pose([0.,0.,0.])},clouds)
        evidence=state.perception_metadata['bodies']['ring'].pop('rotation_equivalence_evidence')
        actor._attachments(state,{'ring':'attached'},set());self.assertNotIn('ring',actor.attachments)
        state.perception_metadata['bodies']['ring']['rotation_equivalence_evidence']=evidence
        actor._attachments(state,{'ring':'attached','other':'attached'},set());self.assertNotIn('ring',actor.attachments)

    def test_depth_contradiction_rejects_otherwise_close_support_fit(self):
        actor,state,clouds,_,_=self.fixture();actor.tracker.config=replace(actor.tracker.config,free_space_weight=1.)
        actor.tracker._registration({'ring':Pose([0.,0.,0.])},clouds)
        with patch.object(policy,'current_depth_pose_cost',return_value={'cost_m2':1.}):
            actor._attachments(state,{'ring':'attached'},set())
        self.assertNotIn('ring',actor.attachments)

    def test_release_clears_supported_transform_and_cannot_resurrect_it_without_new_evidence(self):
        actor,state,clouds,_,_=self.fixture();actor.tracker._registration({'ring':Pose([0.,0.,0.])},clouds)
        actor._attachments(state,{'ring':'attached'},set());self.assertIn('ring',actor.attachments)
        state.robot.contact=False;actor._attachments(state,{'ring':'released'},set());self.assertNotIn('ring',actor.attachments)
        state.robot.contact=True;state.perception_metadata['bodies']['ring']['rotation_equivalence_evidence']=None
        actor._attachments(state,{'ring':'attached'},set());self.assertNotIn('ring',actor.attachments)


if __name__=='__main__':unittest.main()
