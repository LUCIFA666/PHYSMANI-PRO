from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
import json,unittest
import numpy as np
from our_mani.contracts import Pose
from our_mani.multitask import rml_cup_support_v695 as m
from our_mani.multitask.perception import RegistrationConfig

ROOT=Path(__file__).resolve().parents[1]


class Tests(unittest.TestCase):
    def state(self):
        body=NS(visible=True,timestamp=1.,pose=Pose([0.,0.,.1]))
        meta=dict(directly_visible=True,registration_valid=True,orientation_observable=True)
        return NS(robot=NS(contact=True,timestamp=1.,pose=Pose([0.,0.,.2])),bodies={'cup0':body},
            perception_metadata={'bodies':{'cup0':meta}})
    def evidence(self):return dict(timestamp=1.,pose=Pose([0.,0.,.1]),support_quality_passed=True)
    def test_current_fullR_visible_contact_admits_only_a_measured_transform(self):
        state=self.state();self.assertTrue(m.admissible(state,{'cup0':'attached'},set(),self.evidence(),.1))
        active={};attachments={};item=m.update_attachment(state,{'cup0':'attached'},set(),self.evidence(),active,attachments,.1)
        self.assertIs(item,attachments['cup0']);self.assertEqual(item.provenance,'current_rgbd_robot_proprioception')
    def test_stale_invisible_or_unqualified_fullR_cannot_create_attachment(self):
        state=self.state();e=self.evidence();e['timestamp']=.5
        self.assertFalse(m.admissible(state,{'cup0':'attached'},set(),e,.1))
        state.bodies['cup0'].visible=False
        self.assertFalse(m.admissible(state,{'cup0':'attached'},set(),self.evidence(),.1))
        state.bodies['cup0'].visible=True;state.perception_metadata['bodies']['cup0']['orientation_observable']=False
        self.assertFalse(m.admissible(state,{'cup0':'attached'},set(),self.evidence(),.1))
    def test_ambiguous_association_or_no_contact_cannot_enable_the_support_transform(self):
        state=self.state();self.assertFalse(m.admissible(state,{'cup0':'attached'},{'cup0'},self.evidence(),.1))
        state.robot.contact=False
        self.assertFalse(m.admissible(state,{'cup0':'attached'},set(),self.evidence(),.1))
    def test_prior_transform_persists_only_during_continuous_unambiguous_contact(self):
        state=self.state();active={};attachments={}
        first=m.update_attachment(state,{'cup0':'attached'},set(),self.evidence(),active,attachments,.1)
        state.bodies['cup0'].visible=False
        self.assertIsNone(m.update_attachment(state,{'cup0':'attached'},set(),None,active,attachments,.1))
        self.assertIs(attachments['cup0'],first)
        state.robot.contact=False;attachments.clear()
        self.assertIsNone(m.update_attachment(state,{'cup0':'released'},set(),None,active,attachments,.1))
        self.assertFalse(active);self.assertFalse(attachments)
    def test_binding_requires_existing_fullR_placement_capability(self):
        c=json.loads((ROOT/'artifacts/framework/multitask_r10/rml_suite16_smoke_v677_setup/place_cups_on_rotating_frame_high_speed_ours_configuration.json').read_text())
        self.assertFalse(m.binding(ROOT,c,m.TASKS[1])['original_public_measurement']['collision_meshes_or_permissions_modified'])
        c['body_orientation_observability'][m.TASKS[1]]['cup0']='none'
        with self.assertRaisesRegex(ValueError,'full-R'):m.binding(ROOT,c,m.TASKS[1])
    def test_direct_RML_without_old_reference_manager_restores_measured_held_transform_and_preserves_tracker(self):
        c=json.loads((ROOT/'artifacts/framework/multitask_r10/rml_suite16_smoke_v677_setup/place_cups_on_rotating_frame_high_speed_ours_configuration.json').read_text());task=m.TASKS[1]
        b=m.binding(ROOT,c,task);c['RML_cup_public_support_v695']={task:b}
        body=json.loads((ROOT/b['original_public_measurement']['TRAIN_geometry']['path']).read_text())['bodies']['cup0']
        vertices=np.asarray(body['collision_vertices_body'],float);raw=Pose([0.,0.,-vertices[:,2].min()-.001])
        components=[NS(name=p['name'],points_body=np.asarray(p['points_body'],float)) for p in body['components']]
        clouds={'cup0':{p.name:raw.apply(p.points_body)+[0.,0.,.001] for p in components}}
        result=dict(valid=True,rotation_ambiguous=False,per_role={'cup0':dict(valid=True)},pose=raw)
        tracker=NS(_registration=lambda members,clouds,previous=None:result,_current_timestamp=1.,
            templates={'cup0':NS(components=components)},config=RegistrationConfig())
        ghost=NS(get_matrix=lambda:np.c_[np.eye(3),np.zeros(3)])
        actor=NS(map=NS(map=[dict(name='workspace')],static_ghosts=[ghost]),tracker=tracker,
            controller=NS(feedback_step_calls=0,generator=NS(config=NS(maximum_observation_age=.1))),
            last_time=None,previous_states=[],attachments={},_attachments=lambda state,modes,invalid:None,metrics_report=lambda:{})
        with patch.object(m,'read_native_mesh',return_value=(vertices,[],None)),patch.object(m,'horizontal_rectangle',return_value=dict(height=0.,xy_min=[-1.,-1.],xy_max=[1.,1.])):
            m.install(actor,c,ROOT,task)
        returned=tracker._registration({'cup0':Pose([0.,0.,0.])},clouds)
        self.assertIs(returned,result);np.testing.assert_array_equal(raw.position,[0.,0.,-vertices[:,2].min()-.001])
        state=self.state();state.bodies['cup0'].pose=raw
        actor._attachments(state,{'cup0':'attached'},set())
        held=actor.attachments['cup0'];held_world=state.robot.pose.apply(held.tool_to_body.position)
        self.assertGreater((vertices+held_world)[:,2].min(),0.)
        metrics=actor.metrics_report()['RML_cup_public_support_v695']
        self.assertTrue(metrics['registration_events'][0]['valid']);self.assertEqual(len(metrics['attachment_events']),1)


if __name__=='__main__':unittest.main()
