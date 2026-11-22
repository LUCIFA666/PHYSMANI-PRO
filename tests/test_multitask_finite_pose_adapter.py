"""Wire actual frozen TRAIN ring CAD into the current RGBD adapter on CPU.

Real rendered ICP checks retain every hypothesis and may correctly reject a
quotient. The controlled accepted-hypothesis case tests only the adapter, not
registration accuracy or closed-loop grasp success.
"""
from dataclasses import replace
import json
from pathlib import Path
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
from scipy.spatial.transform import Rotation

from our_mani.contracts import Pose
from our_mani.multitask.rotation_equivalence import VerifiedFiniteSymmetry,CurrentRotationEquivalence
from our_mani.multitask.training_geometry import body_templates_from_asset
from our_mani.multitask.perception import MultiBodySE3Tracker
from our_mani.multitask.ring_affordance import VerifiedRingAffordance
from our_mani.multitask.pipeline import PhaseCollisionPort
from our_mani.multitask.specs import get_task_spec
from tests.test_multitask_perception_r10 import render
from tests.test_multitask_geometry_mode_pipeline import VariantMap,frame

ROOT=Path(__file__).resolve().parents[1]
REPORT=ROOT/'artifacts/framework/multitask_r10/rotation_equivalence_runtime_v4/report.json'


class FinitePoseAdapterChecks(unittest.TestCase):
    def templates(self):
        report=json.loads(REPORT.read_text())
        for row in report['entries']:
            symmetry=VerifiedFiniteSymmetry.load(ROOT,row['certificate'])
            asset=json.loads((ROOT/symmetry.certificate['CAD_file']['path']).read_text())
            bindings={name:{c['name']:(i+1,) for c in body['components']}
                for i,(name,body) in enumerate(asset['bodies'].items())}
            templates=body_templates_from_asset(asset,bindings,
                orientation_observability={name:'none' for name in asset['bodies']},
                finite_rotation_symmetries={'ring':symmetry})
            yield next(t for t in templates if t.role_id=='ring'),symmetry

    def test_real_current_rendered_ICP_keeps_representative_R_all_retained_hypotheses_and_zero_spin(self):
        for template,symmetry in self.templates():
            tracker=MultiBodySE3Tracker([template])
            for tick in range(2):
                now=tick*.05
                pose=Pose([0.,0.,1.],Rotation.from_rotvec([.3,.2,.4+tick*.3]).as_matrix())
                states,meta=tracker.observe(render([template],{'ring':pose},now),modes={'ring':'external'})
                body=states[0];diagnostic=meta['bodies']['ring'];evidence=diagnostic['rotation_equivalence_evidence']
                self.assertIsInstance(evidence,CurrentRotationEquivalence)
                self.assertGreaterEqual(evidence.evidence['root_hypothesis_count'],4)
                self.assertGreater(np.linalg.norm(body.pose.rotation-np.eye(3)),.1)
                self.assertFalse(diagnostic['orientation_observable'])
                self.assertEqual(diagnostic['angular_velocity_observability'],'none')
                np.testing.assert_array_equal(body.angular_velocity,np.zeros(3))
                self.assertEqual(evidence.evidence['tested_hypotheses'],evidence.evidence['root_hypothesis_count'])
                self.assertEqual(diagnostic['rotation_equivalence_receipt'],evidence.receipt())
                self.assertFalse(evidence.evidence['absolute_registration_accuracy_proven'])

    def test_controlled_D4_retained_hypotheses_bind_observed_pose_without_promoting_full_R(self):
        template,symmetry=next(self.templates());tracker=MultiBodySE3Tracker([template])
        pose=Pose([0.,0.,1.],Rotation.from_rotvec([.3,.2,.4]).as_matrix())
        group=np.asarray(symmetry.certificate['group_rotations_body']);center=np.asarray(symmetry.certificate['center_body'])
        second=Pose(pose.position+pose.rotation@(center-group[1]@center),pose.rotation@group[1])
        fitted=dict(pose=pose,rms_m=.002,per_role=dict(ring=dict(rms_m=.002,inlier_fraction=1.,valid=True)),
            valid=True,rotation_ambiguous=True,axis_evidence_by_role={},registration_selection_cost_m2=.002**2,
            retained_root_candidate_poses=(pose,second),retained_candidate_rms_m=(.002,.0021),current_depth_pose_cost=None)
        with patch.object(tracker,'_registration',return_value=fitted):
            states,meta=tracker.observe(render([template],{'ring':pose},0.),modes={'ring':'external'})
        evidence=meta['bodies']['ring']['rotation_equivalence_evidence'];body=states[0]
        self.assertTrue(evidence.matches(body_id='ring',timestamp=0.,pose=body.pose))
        self.assertTrue(meta['bodies']['ring']['orientation_equivalence_observable'])
        self.assertFalse(meta['bodies']['ring']['orientation_observable'])
        self.assertFalse(evidence.receipt()['policy_skill_authorized'])

    def test_same_instance_identity_mask_without_visible_depth_cannot_unlock_quotient(self):
        template,symmetry=next(self.templates());tracker=MultiBodySE3Tracker([template]);pose=Pose([0.,0.,1.])
        observation=render([template],{'ring':pose},0.)
        for depth in observation['observation/depths'].values():depth[:]=0.
        states,meta=tracker.observe(observation,modes={'ring':'external'})
        self.assertEqual(states,[]);self.assertFalse(meta['bodies']['ring']['registration_valid'])
        self.assertFalse(meta['bodies']['ring'].get('orientation_equivalence_observable',False))
        self.assertNotIn('rotation_equivalence_evidence',meta['bodies']['ring'])
        with self.assertRaisesRegex(ValueError,'same certified TRAIN ring CAD'):
            replace(template,orientation_observability='full')

    def test_actual_pending_TRAIN_permission_grants_only_same_CAD_geometry_with_typed_current_evidence(self):
        report=json.loads((ROOT/'artifacts/framework/multitask_r10/ring_affordance_runtime_v4/report.json').read_text())
        permission=VerifiedRingAffordance.load(ROOT,report['entries'][0]['permission'])
        self.assertFalse(permission.permission['policy_skill_authorized'])
        self.assertEqual(permission.permission['phase_support']['acquire']['completed_parent_count'],1)
        scene=VariantMap();scene.required_dynamic_role_ids=('ring','frame','peg0','peg1','peg2')
        scene.estimated_bodies={name:SimpleNamespace(fallback_ghost=object(),source_sha256='a'*64,
            collision_geometry_receipt=dict(collision_geometry_kind='exact_TRAIN_triangles'))
            for name in scene.required_dynamic_role_ids}
        state,forecast,_,meta=frame(scene);pose=state.bodies['ring'].pose;symmetry=permission.symmetry
        group=np.asarray(symmetry.certificate['group_rotations_body']);center=np.asarray(symmetry.certificate['center_body'])
        second=Pose(pose.position+pose.rotation@(center-group[1]@center),pose.rotation@group[1])
        evidence=CurrentRotationEquivalence.from_root_hypotheses(symmetry,body_id='ring',timestamp=1.,
            root_candidate_poses=(pose,second),candidate_rms_m=(.002,.0021),body_in_root=Pose(np.zeros(3)),
            registration_valid=True,directly_visible=True,all_retained_hypotheses_provided=True,
            representative_body_pose=pose)
        scene.estimated_bodies['ring'].source_sha256=symmetry.certificate['CAD_asset_sha256']
        meta['ring'].update(orientation_observable=False,rotation_equivalence_evidence=evidence)
        port=PhaseCollisionPort(get_task_spec('insert_onto_rotating_peg'),scene,SimpleNamespace(),
            maximum_observation_age=.1,ring_affordance=permission)
        port.configure_frame(state,forecast)
        port.bind_candidate(state=state,forecast=forecast,phase='acquire',reference_id='ring',manipulated_ids=('ring',),arrival_time=1.1)
        self.assertFalse(port.check_future(.1))
        self.assertEqual(scene.checks[-1]['geometry_modes']['ring'],'primary')
        self.assertEqual(scene.checks[-1]['permitted'],('ring',))
        self.assertEqual(scene.checks[-1]['ignored'],())
        self.assertTrue(port.receipt()['held_vs_world_contact_targets_are_checked'])
        scene.estimated_bodies['ring'].source_sha256='f'*64
        self.assertFalse(port.check_future(.1));self.assertEqual(scene.checks[-1]['geometry_modes']['ring'],'rotation_uncertainty')
        scene.estimated_bodies['ring'].source_sha256=symmetry.certificate['CAD_asset_sha256']
        meta['ring']['rotation_equivalence_evidence']=evidence.receipt()
        self.assertFalse(port.check_future(.1));self.assertEqual(scene.checks[-1]['geometry_modes']['ring'],'rotation_uncertainty')


if __name__=='__main__':unittest.main()
