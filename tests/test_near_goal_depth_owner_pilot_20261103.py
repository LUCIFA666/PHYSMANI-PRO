"""Pure CPU materialization fixtures; no Native or parent capture is launched."""
from pathlib import Path
import json,tempfile,unittest
from copy import deepcopy
import numpy as np
from our_mani.local_mpc_20261103 import depth_owner_pilot as p

ROOT=Path(__file__).resolve().parents[1]


def fixture():
 case=ROOT/'runtime/framework/near_goal_perception_20261103/smoke2_depth_owner_v2/p3_near_goal_render_smoke_seed261103600'
 sphere=json.loads((case/'public_initial_sphere.json').read_text())['sphere']
 with np.load(case/'initialization/public_rgbd.npz',allow_pickle=False)as z:
  cameras={n:dict(K=z['K'][i],T=z['T'][i],near=.009999999776482582,far=4.5 if i==0 else 10.,width=256,height=256)for i,n in enumerate(p.b.CAMERAS)}
 rows=json.loads((ROOT/'artifacts/framework/local_mpc_20261102/P3_pilot64_TRAIN_q_FK_source_audit_v1/q_sources.json').read_text())['rows']
 limits=json.loads((ROOT/'artifacts/framework/local_mpc_20261102/P3_pilot64_TRAIN_q_FK_source_audit_v1/native_joint_limits.json').read_text())['joint_limits']
 kin=p.PoEKinematics(json.loads((ROOT/'artifacts/framework/local_mpc_20261102/P2_coupled_response_v1/source_snapshot/kinematics.json').read_text()))
 off=json.loads((ROOT/'artifacts/framework/local_policy/goal_marker_calibration.json').read_text())['offset_train_calibrated_m']
 plan=dict(marker_offset_train_m=off,DLS=p.DLS,tool_quaternion_xyzw=[1.,0.,0.,0.],world_outside_position_m=[100.,100.,-100.])
 return plan,rows,sphere,cameras,kin,limits


class PilotTests(unittest.TestCase):
 def test_eight_specs_conditions_and_deterministic_world_arrays(self):
  plan,rows,s,c,k,l=fixture();parent=dict(parent_id='CPU_example_only',seed=rows[0]['parent_seed'],TRAIN_q_source=rows[0],shots=p.SHOT_SPECS)
  a=p.materialize_pilot_shots(plan,parent,s,[.1,-.12,.775,0,0,0,1],[-.025,.025]*3,c,k,l)
  z=p.materialize_pilot_shots(plan,parent,s,[.1,-.12,.775,0,0,0,1],[-.025,.025]*3,c,k,l)
  self.assertTrue(a['complete']);self.assertEqual(a,z);self.assertEqual(len(a['shots']),8)
  self.assertEqual([x['condition']for x in a['shots']],['sphere_only','cube_far']+['near_goal']*5+['sphere_only'])
  np.testing.assert_allclose(a['public_goal_reference'],np.array(s['center'])+plan['marker_offset_train_m'],atol=0,rtol=0)
  for i in range(5):self.assertEqual(a['shots'][i]['joint_positions'],rows[0]['joint_positions'])
  self.assertEqual(a['shots'][6]['joint_positions'],a['shots'][7]['joint_positions'])

 def test_frustum_is_eight_corner_signedK_near_far_proof(self):
  _,_,s,c,_,_=fixture();proof=p.outside_frusta_proof(np.array([100.,100.,-100.]),[0,0,0,1],[-.025,.025]*3,c)
  self.assertTrue(proof['verified']);self.assertEqual(len(proof['world_corners']),8)
  for row in proof['views'].values():self.assertEqual(len(row['camera_corners']),8);self.assertTrue(row['all8corners_outside_planes']);self.assertIn('near',row['halfspace_values'])
  self.assertFalse(p.outside_frusta_proof(np.array(s['center']),[0,0,0,1],[-.025,.025]*3,c)['verified'])

 def test_infeasible_DLS_keeps_original_source_and_all_declared_shots(self):
  plan,rows,s,c,k,l=fixture();parent=dict(parent_id='CPU_infeasible',seed=rows[2]['parent_seed'],TRAIN_q_source=rows[2],shots=p.SHOT_SPECS)
  a=p.materialize_pilot_shots(plan,parent,s,[.1,-.12,.775,0,0,0,1],[-.025,.025]*3,c,k,l)
  self.assertFalse(a['complete']);self.assertIn('predeclared_DLS_infeasible_no_source_or_target_retry',a['failures']);self.assertEqual(len(a['shots']),8)
  self.assertEqual(parent['TRAIN_q_source'],rows[2]);self.assertEqual(a['DLS']['config'],p.DLS)

 def test_pilot_reuses_immutable_clock_q_and_normalization_implementations(self):
  for name in ('clock','zero_clock_shot','capture_checked','normalize_baseline_fingers','archive_capture_sources'):
   self.assertIs(getattr(p,name),getattr(p.b,name))


if __name__=='__main__':unittest.main()
