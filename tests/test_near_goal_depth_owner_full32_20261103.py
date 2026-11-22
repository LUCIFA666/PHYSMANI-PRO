"""New octantfull32 only; no Native/fit or oldproducer modification."""
from pathlib import Path
import ast,json,unittest
import numpy as np
from our_mani.local_mpc_20261103 import depth_owner_full32 as p
from tests.test_near_goal_depth_owner_pilot_v2_20261103 import fixture

ROOT=Path(__file__).resolve().parents[1]

class Full32Tests(unittest.TestCase):
 def test_predeclared_octant_schedule_and_three_canonical_conditions(self):
  directions=[]
  for i in range(32):
   shots=p.shot_specs_for_parent(i);directions.append(shots[0]['near_direction_ordinal'])
   self.assertEqual([x['shot_id']for x in shots],[f'shot{j:02d}'for j in range(8)])
   for shot in shots:
    self.assertEqual(shot['near_direction_ordinal'],i%8);np.testing.assert_array_equal(shot['near_direction_unit_world'],p.DIRECTIONS[i%8])
    self.assertIn(shot['condition'],['sphere_only','cube_far','near_goal']);self.assertIn(f'dir{i%8:02d}',shot['condition_detail'])
  self.assertEqual(np.bincount(directions).tolist(),[4]*8)
  self.assertEqual(np.bincount(directions[:24]).tolist(),[3]*8)
  for cohort in [p.DIRECTIONS[:4],p.DIRECTIONS[4:]]:self.assertTrue(np.all(np.min(cohort,axis=0)<0)&np.all(np.max(cohort,axis=0)>0))

 def test_near_goals_follow_xyz_direction_with_same_radii_and_hand_offsets(self):
  plan,rows,s,c,k,l=fixture()
  for i in [0,1,4,7]:
   parent=dict(parent_id='CPU_example',seed=rows[i]['parent_seed'],TRAIN_q_source=rows[i],shots=p.shot_specs_for_parent(i),near_direction_ordinal=i,near_direction_unit_world=p.DIRECTIONS[i].tolist())
   m=p.materialize_pilot_shots(plan,parent,s,[.1,-.12,.775,0,0,0,1],[-.025,.025]*3,c,k,l)
   G=np.array(m['public_goal_reference']);d=p.DIRECTIONS[i]
   for j,radius in [(2,.06),(3,.03),(4,0.),(5,.06),(6,.06)]:np.testing.assert_allclose(m['shots'][j]['materialized_cube_pose_world'][:3],G+radius*d,rtol=0,atol=1e-14)
   np.testing.assert_allclose(np.array(m['DLS']['target_positions']),[G+.06*d+[0,-.075,.025],G+.06*d],rtol=0,atol=1e-14)
   self.assertEqual(m['shots'][6]['joint_positions'],m['shots'][7]['joint_positions'])

 def test_fixed_solver_frustum_and_snapshot_functions_unchanged(self):
  def defs(path):return {n.name:ast.dump(n,include_attributes=False)for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef)}
  old=defs(ROOT/'our_mani/local_mpc_20261103/depth_owner_pilot_v2.py');new=defs(ROOT/'our_mani/local_mpc_20261103/depth_owner_full32.py')
  for name in ['solve_fixed_starts','outside_frusta_proof']:self.assertEqual(old[name],new[name])
  for name in ['clock','zero_clock_shot','normalize_baseline_fingers','capture_checked']:self.assertIs(getattr(p,name),getattr(p.b,name))

if __name__=='__main__':unittest.main()
