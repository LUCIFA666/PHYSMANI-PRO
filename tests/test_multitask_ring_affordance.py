"""Finite ring permissions remain phase/library/current-pose bound."""
from dataclasses import replace
import json
import unittest
import numpy as np

from our_mani.contracts import Pose,ObjectState,RobotState
from our_mani.multitask.evaluation import digest
from our_mani.multitask.interaction import PhaseTemplate,PolicyState,_observed_reference
from our_mani.multitask import ring_affordance as permission
from our_mani.multitask.rotation_equivalence import CurrentRotationEquivalence
from tests import test_multitask_rotation_equivalence as helper
from tests.test_multitask_interaction import candidates_config


class RingAffordanceChecks(unittest.TestCase):
    def setUp(self):
        self.helper=helper.RotationEquivalenceTests();self.helper.setUp()
        self.verified,self.path,_=self.helper.loaded()
        self.template=PhaseTemplate('approach','ring_approach','ring','observed_se3','ring',0.,
            np.array([0.,1.]),np.zeros((2,3)),np.tile(np.eye(3),(2,1,1)),np.ones(2),
            .1,.1,.01,.05,'external')
        self.pose=Pose(np.array([.3,.2,.5]))
        self.evidence=CurrentRotationEquivalence.from_root_hypotheses(self.verified,body_id='ring',timestamp=1.,
            root_candidate_poses=[self.pose,Pose(self.pose.position,self.pose.rotation@helper.D4()[1])],
            candidate_rms_m=[.002,.0021],body_in_root=Pose(np.zeros(3)),registration_valid=True,
            directly_visible=True,all_retained_hypotheses_provided=True,representative_body_pose=self.pose)
    def tearDown(self):self.helper.tearDown()
    def fixture_permission(self,*,development=False,authorized=True):
        # This is an explicit CPU fixture capability, not a fabricated TRAIN
        # coverage result. Real source/receipt reproduction is exercised by the
        # standalone TRAIN evidence builder and its saved diagnostic report.
        p=dict(schema=permission.SCHEMA,certificate_sha256=self.verified.certificate['certificate_sha256'],
            task='insert_onto_rotating_peg',body_id='ring',policy_skill_authorized=authorized,
            development_only=development,library=dict(payload_sha256='b'*64),
            templates=[dict(phase='approach',anchor_id=self.template.anchor_id,
                template_sha256=permission.template_sha256(self.template))])
        p['permission_sha256']=digest(p)
        return permission.VerifiedRingAffordance(p,self.verified,permission._TOKEN)
    def arguments(self):
        return dict(task='insert_onto_rotating_peg',body_id='ring',phase='approach',template=self.template,
            library_sha256='b'*64,evidence=self.evidence,observed_pose=self.pose,observed_timestamp=1.,
            now=1.05,maximum_age=.1)
    def test_valid_permission_is_bound_to_phase_template_and_library(self):
        grant=self.fixture_permission();kwargs=self.arguments()
        self.assertTrue(grant.permits_reference(**kwargs))
        for field,value in [('phase','insert'),('task','insert_onto_rotating_peg_high_speed'),
                ('body_id','peg0'),('library_sha256','c'*64),('observed_timestamp',1.01),('now',1.2),
                ('template',replace(self.template,anchor_id='different')),
                ('evidence',self.evidence.receipt())]:
            self.assertFalse(grant.permits_reference(**{**kwargs,field:value}),field)
        self.assertFalse(self.evidence.receipt()['full_orientation_observable'])
    def test_pending_one_source_and_development_cannot_silently_be_production(self):
        self.assertFalse(self.fixture_permission(authorized=False).permits_reference(**self.arguments()))
        dev=self.fixture_permission(development=True)
        self.assertFalse(dev.permits_reference(**self.arguments()))
        self.assertTrue(dev.permits_reference(**self.arguments(),allow_development=True))
    def test_interaction_uses_typed_ring_permission_without_changing_full_R_metadata(self):
        body=ObjectState('ring',1.,self.pose,np.zeros((3,3)),np.zeros(3),np.zeros(3))
        robot=RobotState(1.05,self.pose,np.zeros(3),np.zeros(3),np.zeros(7),np.zeros(7))
        meta=dict(directly_visible=True,registration_valid=True,orientation_observable=False,
            rotation_equivalence_evidence=self.evidence)
        state=PolicyState(robot,dict(ring=body),dict(bodies=dict(ring=meta)),dict(ring=('ring',)))
        with self.assertRaisesRegex(ValueError,'unobservable'):
            _observed_reference(state,'ring',self.template,self.pose,candidates_config())
        observed=_observed_reference(state,'ring',self.template,self.pose,candidates_config(),
            ring_affordance=self.fixture_permission(),library_sha256='b'*64,task_name='insert_onto_rotating_peg')
        self.assertIs(observed,body);self.assertFalse(meta['orientation_observable'])
        with self.assertRaisesRegex(ValueError,'unobservable'):
            _observed_reference(state,'ring',replace(self.template,phase='insert'),self.pose,candidates_config(),
                ring_affordance=self.fixture_permission(),library_sha256='b'*64,task_name='insert_onto_rotating_peg')
    def test_geometry_permission_does_not_require_or_authorize_unfitted_skill(self):
        pending=self.fixture_permission(authorized=False)
        self.assertTrue(pending.permits_geometry(self.evidence,body_id='ring',observed_pose=self.pose,
            observed_timestamp=1.,now=1.05,maximum_age=.1))
        self.assertFalse(pending.permits_reference(**self.arguments()))
        pending.permission['policy_skill_authorized']=True
        with self.assertRaisesRegex(ValueError,'mutated'):pending.permits_reference(**self.arguments())
    def test_native_rule_audits_proximity_only_and_real_acquire_grasp_transition(self):
        root=self.helper.root
        grip=root/'gripper.py';task=root/'insert_onto_rotating_peg.py'
        grip.write_text('''class Gripper:
 def grasp(self,obj):
  detected = self._proximity_sensor.is_detected(obj)
  if detected and obj not in self._grasped_objects:
   self._grasped_objects.append(obj)
   self._old_parents.append(obj.get_parent())
   obj.set_parent(self._attach_point, keep_in_place=True)
  return detected
''')
        task.write_text('''def get_expert_info(task):
 is_grasp = len(task.robot.gripper.get_grasped_objects()) > 0
 if stage == 'wp1-grasp' and is_grasp:
  stage = 'wp2'
class Task:
 def init_task(self):
  self.register_graspable_objects([self._square_ring])
''')
        result=permission._native_rule(grip,task)
        self.assertTrue(result['acquire_to_lift_requires_native_grasped_object'])
        task.write_text(task.read_text().replace("stage == 'wp1-grasp' and is_grasp","stage == 'wp1-grasp'"))
        with self.assertRaisesRegex(ValueError,'no longer proves a grasp'):permission._native_rule(grip,task)


if __name__=='__main__':unittest.main()
