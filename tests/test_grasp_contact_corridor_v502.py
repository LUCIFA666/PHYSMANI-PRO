from pathlib import Path
from contextlib import contextmanager
from types import SimpleNamespace as NS
from unittest.mock import patch
import unittest
import numpy as np
from our_mani.contracts import Pose
from our_mani.multitask.interaction import InteractionLibrary,GripperTimingConfig,_world_target
from our_mani.multitask.grasp_contact_corridor_v502 import install
ROOT=Path(__file__).resolve().parents[1]
class Shape:
 def __init__(self,h,name,colliding=()):self.h=h;self.name=name;self.colliding=set(colliding)
 def get_handle(self):return self.h
 def get_name(self):return self.name
 def is_collidable(self):return True
 def is_respondable(self):return True
 def check_collision(self,other):return other.h in self.colliding
class Tests(unittest.TestCase):
 def fixture(self,distance=.01,collision=(2,)):
  library=InteractionLibrary.load(ROOT/'artifacts/framework/multitask_r10/high_cup_all300_issue_operation_v125/interaction.json')
  t=library.for_phase('acquire')[0];reference=Pose([.2,.1,.8]);grasp=_world_target(reference,np.zeros(3),np.zeros(3),t,1.)[0]
  tip=Pose(grasp.position+[distance,0.,0.],grasp.rotation)
  palm,left,right=Shape(1,'palm'),Shape(2,'left_finger'),Shape(3,'right_finger')
  cup=Shape(4,'estimated_cup',collision);table=Shape(5,'static_table');active={'cup0':cup};entered=[False];calls=[]
  @contextmanager
  def scene(*a,**kw):
   entered[0]=True
   try:yield active
   finally:entered[0]=False
  service=NS(check=lambda *a,**kw:calls.append(kw) or False,public_scene=scene,estimated_attachments={},static_ghosts=[table])
  state=NS(robot=NS(timestamp=0.),bodies={'cup0':NS(timestamp=0.)},perception_metadata={'bodies':{'cup0':dict(registration_valid=True,orientation_observable=True,pose_evidence_until=0.)}})
  port=NS(map=service,bound=dict(phase='acquire',manipulated_ids=('cup0',)),frame=(state,None),maximum_observation_age=.1)
  env=NS(gripper=NS(joints=[NS(get_objects_in_tree=lambda **kw:[left]),NS(get_objects_in_tree=lambda **kw:[right])],
   get_objects_in_tree=lambda **kw:[palm,left,right]),arm=NS(get_tip=lambda:NS(get_pose=lambda:tip.xyzw())))
  actor=NS(controller=NS(feedback_step_calls=0,gripper_timing_config=GripperTimingConfig(.02,.1),library=library),collision_port=port,metrics_report=lambda:{})
  counts,events=install(actor,env,shape_type='CPU_shape')
  return actor,service,cup,table,active,state,calls,entered,counts,events,{'cup0':reference}
 def check(self,fixture,**kw):
  actor,service,*_=fixture;poses=fixture[-1]
  return service.check(poses,allowed_robot_contact_ids=('cup0',),geometry_modes={'cup0':'primary'},**kw)
 def test_real_TRAIN_grasp_region_allows_fingers_but_rejects_early_contact_and_palm(self):
  for distance,collision,expected in [(.01,(2,),False),(.025,(2,),False),(.05,(2,),True),(.01,(1,),True)]:
   fixture=self.fixture(distance,collision);self.assertEqual(self.check(fixture),expected)
   self.assertFalse(fixture[7][0]);self.assertEqual(len(fixture[6]),1)
   self.assertEqual(fixture[0].controller.gripper_timing_config.close_position_tolerance_m,.02)
   self.assertEqual(fixture[0].controller.gripper_timing_config.close_orientation_tolerance_rad,.1)
 def test_unknown_rotation_stale_evidence_and_static_world_cannot_gain_contact_permission(self):
  fixture=self.fixture();fixture[5].perception_metadata['bodies']['cup0']['orientation_observable']=False
  self.assertTrue(self.check(fixture))
  fixture=self.fixture();fixture[5].robot.timestamp=.2;self.assertTrue(self.check(fixture))
  fixture=self.fixture(collision=());fixture[3].colliding={2};self.assertTrue(self.check(fixture))
 def test_existing_held_fingers_only_and_unselected_body_contacts_are_solid(self):
  fixture=self.fixture(.2);fixture[1].estimated_attachments={'cup0':Pose([0.,0.,0.])};self.assertFalse(self.check(fixture))
  fixture[2].colliding={1};self.assertTrue(self.check(fixture))
  fixture=self.fixture(collision=());fixture[4]['cup1']=Shape(6,'unselected_cup',(2,));fixture[-1]['cup1']=Pose([0.,0.,0.])
  self.assertTrue(self.check(fixture))
 def test_native_collision_errors_propagate_and_metrics_do_not_query_closed_shapes(self):
  fixture=self.fixture();fixture[2].check_collision=lambda other:(_ for _ in ()).throw(RuntimeError('native pair query'))
  # Palm is never exempt, even when fingers are qualified.
  with self.assertRaises(RuntimeError):self.check(fixture)
  self.assertFalse(fixture[7][0])
  with patch.object(Shape,'get_name',side_effect=RuntimeError('closed native shape')):
   self.assertEqual(fixture[0].metrics_report()['grasp_contact_corridor_v502']['physical_gripper_shapes'],['palm','left_finger','right_finger'])
if __name__=='__main__':unittest.main()
