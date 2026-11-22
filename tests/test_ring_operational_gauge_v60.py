"""Exercise the installed gauge through actual tracker.observe and typed gates."""
import json
from pathlib import Path
from types import SimpleNamespace,MethodType
import unittest
from unittest.mock import patch
import numpy as np
from scipy.spatial.transform import Rotation
from our_mani.contracts import Pose
from our_mani.multitask.perception import MultiBodySE3Tracker
from our_mani.multitask.rotation_equivalence import CurrentRotationEquivalence
from our_mani.multitask.ring_operational_gauge_v60 import binding,install,certified_gauge_evidence,evidence_scope
from our_mani.multitask.interaction import InteractionLibrary
from our_mani.multitask.rotation_equivalence import VerifiedFiniteSymmetry
from our_mani.multitask.training_geometry import body_templates_from_asset
import importlib.util

ROOT=Path(__file__).resolve().parents[1]
_spec=importlib.util.spec_from_file_location('_r10_rgbd_fixture',ROOT/'tests/test_multitask_perception_r10.py')
_fixture=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(_fixture)
render=_fixture.render


class Tests(unittest.TestCase):
    def setup_case(self):
        report=json.loads((ROOT/'artifacts/framework/multitask_r10/rotation_equivalence_runtime_v4/report.json').read_text())
        symmetry=VerifiedFiniteSymmetry.load(ROOT,report['entries'][0]['certificate'])
        asset=json.loads((ROOT/symmetry.certificate['CAD_file']['path']).read_text())
        bindings={name:{component['name']:(i+1,) for component in body['components']}
            for i,(name,body) in enumerate(asset['bodies'].items())}
        templates=body_templates_from_asset(asset,bindings,orientation_observability={name:'none' for name in asset['bodies']},
            finite_rotation_symmetries={'ring':symmetry})
        template=next(t for t in templates if t.role_id=='ring')
        pose=Pose([0.,0.,1.],Rotation.from_rotvec([.3,.2,.4]).as_matrix())
        group=np.asarray(symmetry.certificate['group_rotations_body']);center=np.asarray(symmetry.certificate['center_body'])
        second=Pose(pose.position+pose.rotation@(center-group[1]@center),pose.rotation@group[1])
        fit=dict(pose=pose,rms_m=.002,per_role=dict(ring=dict(rms_m=.002,inlier_fraction=1.,valid=True)),
            valid=True,rotation_ambiguous=True,axis_evidence_by_role={},registration_selection_cost_m2=.002**2,
            retained_root_candidate_poses=(pose,second),retained_candidate_rms_m=(.002,.0021),current_depth_pose_cost=None)
        return template,symmetry,pose,group,center,fit

    def test_actual_installed_tracker_selects_nonidentity_gauge_and_supplies_matching_typed_evidence(self):
        template,symmetry,pose,group,center,fit=self.setup_case()
        c=json.loads((ROOT/'artifacts/framework/multitask_r10/task_executor_smoke_v58_setup/configuration.json').read_text())
        task='insert_onto_rotating_peg';c['ring_operational_gauge_v60']={task:binding(ROOT,c,task)}
        library=InteractionLibrary.load(ROOT/c['learned_artifacts'][task]['interaction']['path'])
        tracker=MultiBodySE3Tracker([template]);actor=SimpleNamespace(tracker=tracker,controller=SimpleNamespace(library=library))
        actor.act=MethodType(lambda self,observation,robot,remaining: self.tracker.observe(observation,modes={'ring':'external'}),actor)
        actor.metrics_report=lambda:dict()
        install(actor,c,ROOT,task)
        grasp=library.for_phase('approach')[0].at(1.)[0].rotation
        robot=SimpleNamespace(pose=Pose(np.zeros(3),pose.rotation@group[4]@grasp))
        with patch.object(tracker,'_registration',return_value=fit):
            states,meta=actor.act(render([template],{'ring':pose},0.),robot,10.)
        body=states[0];e=meta['bodies']['ring']['rotation_equivalence_evidence']
        self.assertTrue(e.matches(body_id='ring',timestamp=0.,pose=body.pose))
        self.assertGreater(np.linalg.norm(body.pose.rotation-pose.rotation),1.)
        self.assertFalse(meta['bodies']['ring']['orientation_observable'])
        np.testing.assert_array_equal(body.angular_velocity,np.zeros(3))
        self.assertEqual(e.evidence['root_hypothesis_count'],len(fit['retained_root_candidate_poses']))
        self.assertTrue(e.evidence['operational_gauge_v60']['all_raw_hypotheses_preserved'])
        from our_mani.multitask import perception
        self.assertIs(perception.CurrentRotationEquivalence,CurrentRotationEquivalence)

    def test_non_group_pose_is_rejected_and_scope_restores_on_exception(self):
        template,symmetry,pose,group,center,fit=self.setup_case()
        bad=Pose(pose.position,pose.rotation@Rotation.from_rotvec([.01,0,0]).as_matrix())
        from our_mani.multitask import perception
        with self.assertRaisesRegex(ValueError,'exact certified D4'):
            with evidence_scope():
                certified_gauge_evidence(symmetry,body_id='ring',timestamp=0.,root_candidate_poses=fit['retained_root_candidate_poses'],
                    candidate_rms_m=fit['retained_candidate_rms_m'],body_in_root=Pose(np.zeros(3)),registration_valid=True,
                    directly_visible=True,all_retained_hypotheses_provided=True,representative_body_pose=bad)
        self.assertIs(perception.CurrentRotationEquivalence,CurrentRotationEquivalence)

    def test_coordinate_change_does_not_qualify_a_missing_or_out_of_group_hypothesis(self):
        template,symmetry,pose,group,center,fit=self.setup_case()
        bad=Pose(pose.position,pose.rotation@Rotation.from_rotvec([.5,0,0]).as_matrix())
        selected=Pose(pose.position+pose.rotation@(center-group[4]@center),pose.rotation@group[4])
        for all_present,other in ((False,fit['retained_root_candidate_poses'][1]),(True,bad)):
            value=certified_gauge_evidence(symmetry,body_id='ring',timestamp=0.,root_candidate_poses=(pose,other),
                candidate_rms_m=(.002,.0021),body_in_root=Pose(np.zeros(3)),registration_valid=True,
                directly_visible=True,all_retained_hypotheses_provided=all_present,representative_body_pose=selected)
            self.assertFalse(value.observable)


if __name__=='__main__':unittest.main()
