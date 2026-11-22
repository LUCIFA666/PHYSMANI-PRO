from types import SimpleNamespace
import unittest
import numpy as np
from our_mani.contracts import Pose
from our_mani.multitask.perception import RegistrationConfig
from our_mani.multitask.supported_sphere_attachment_v9 import constrained_fit,horizontal_rectangle,install,binding
from our_mani.multitask.protocol import MultitaskProtocolError
class TestSupportFit(unittest.TestCase):
 def setUp(self):
  self.plane=dict(height=.752,xy_min=[-.075,-.455],xy_max=[.575,.455]);self.config=RegistrationConfig()
  center=np.array([.2,.1,.781]);self.points=np.array([center+.03*np.array([np.cos(a)*np.cos(b),np.sin(a)*np.cos(b),np.sin(b)]) for a in np.linspace(-1.,1.,10) for b in np.linspace(.1,1.1,12)])
 def test_current_points_and_original_quality_constrain_center(self):
  pose,row=constrained_fit(self.points,Pose([.2,.1,.7804]),.03,.030137,self.plane,self.config)
  assert row['accepted'] and pose.position[2]>=row['constraint_floor'] and row['rms_m']<self.config.maximum_registration_rms_m
  assert row['correction_m']<.003
 def test_observed_below_table_or_outside_footprint_never_corrected(self):
  for points,pose in [(self.points-[0,0,.1],Pose([.2,.1,.7804])),(self.points,Pose([.8,.1,.7804]))]:
   result,row=constrained_fit(points,pose,.03,.030137,self.plane,self.config)
   assert result is pose and not row['accepted']
 def test_original_registration_gate_still_rejects_bad_surface(self):
  result,row=constrained_fit(self.points+[0,0,.2],Pose([.2,.1,.7804]),.03,.030137,self.plane,self.config)
  # These points can fit an elevated sphere; the support constraint does not
  # force contact or make a detached sphere remain on the table.
  assert row['accepted'] and result.position[2]>.9
  bad=self.points.copy();bad[::2,0]+=.2
  result,row=constrained_fit(bad,Pose([.2,.1,.7804]),.03,.030137,self.plane,self.config)
  assert not row['accepted']
 def test_static_plane_requires_full_rectangular_mesh(self):
  v=np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0.]])
  f=np.array([[0,1,2],[0,2,3]]);m=np.eye(4);m[2,3]=.752
  plane=horizontal_rectangle(v,f,m);assert plane['height']==.752
  with self.assertRaises(MultitaskProtocolError):horizontal_rectangle(v,f[:1],m)
  v[0,2]=.001
  with self.assertRaises(MultitaskProtocolError):horizontal_rectangle(v,f,m)
if __name__=='__main__':unittest.main()
