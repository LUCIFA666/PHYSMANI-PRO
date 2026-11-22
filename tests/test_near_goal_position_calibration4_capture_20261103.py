"""Independent position calibration capture contracts; no Native/images/fit."""
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import ast,importlib.util,io,json,unittest
import numpy as np
from our_mani.local_mpc_20261103 import depth_owner_position_calibration4 as p
from our_mani.local_mpc_20261103 import depth_owner_fresh4_calibration as old
from scripts import collect_near_goal_position_calibration4_20261103 as cli
ROOT=Path(__file__).resolve().parents[1]
PLAN=ROOT/'config/framework/P3_near_goal_depth_owner_position_calibration4_20261103_v1.json'

class PositionCaptureTests(unittest.TestCase):
 def test_capture_branch_serializes_real_array_shape_result_without_Native(self):
  result={'complete':True,'q':np.arange(7,dtype=np.float64),'camera':{'value':np.array([.04,.04])},'counter':np.int64(32)}
  stream=io.StringIO()
  with patch.object(cli.producer,'capture_parent',return_value=result)as capture,\
   patch.object(cli.producer,'check_capture_plan',side_effect=AssertionError('capture branch did not run')),redirect_stdout(stream):
   self.assertEqual(cli.main(['--capture','--plan','unused','--parent-id','fixed','--output','newcase']),0)
  captured=json.loads(stream.getvalue());self.assertEqual(captured['q'],list(range(7)));self.assertEqual(captured['counter'],32)
  capture.assert_called_once()

 def test_default_check_does_not_capture(self):
  with patch.object(cli.producer,'check_capture_plan',return_value={'complete':True}),\
   patch.object(cli.producer,'capture_parent',side_effect=AssertionError('Native forbidden'))as capture:
   self.assertEqual(cli.main(['--plan','unused']),0);capture.assert_not_called()

 def test_source_roles_ordered_shots_and_direction_assignment(self):
  plan=json.loads(PLAN.read_text());partition=json.loads((ROOT/plan['partition']['path']).read_text())
  self.assertEqual([r['seed']for r in plan['parents']],list(range(261103636,261103640)))
  self.assertEqual([r['TRAIN_q_source']['raw_episode_id']for r in plan['parents']],list(p.RAW_IDS))
  for i,(r,d)in enumerate(zip(plan['parents'],partition['parents'])):
   self.assertEqual(r['role'],'calibration');self.assertEqual(r['fold'],'calibration');self.assertIs(r['pilot'],False)
   self.assertEqual(r['near_direction_ordinal'],4+i);self.assertEqual(r['shots'],p.old.shot_specs_for_parent(4+i))
   self.assertEqual(d['shot_ids'],['shot'+str(j).zfill(2)for j in range(8)])
  self.assertEqual(plan['scope'],'position_calibration4');self.assertTrue(plan['position_measurement_calibration_only'])

 def test_old_schema_cannot_accept_position_parents(self):
  with self.assertRaises(ValueError):old.checked_pilot_plan(ROOT,PLAN)

 def test_renderer_materialization_clock_restore_mechanics_are_old_functions(self):
  for name in ('materialize_pilot_shots','solve_fixed_starts','outside_frusta_proof','clock','zero_clock_shot','normalize_baseline_fingers','capture_checked','archive_capture_sources'):
   self.assertIs(getattr(p,name),getattr(old,name))
  def fn(path):return next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef)and n.name=='capture_parent')
  class Metadata(ast.NodeTransformer):
   def visit_Call(self,node):
    node=self.generic_visit(node);node.keywords=[k for k in node.keywords if k.arg!='position_measurement_calibration_only'];return node
   def visit_Constant(self,node):
    if isinstance(node.value,str):
     value=node.value.replace('position_calibration4 calibration-only','fresh4 calibration-only').replace('position_calibration4','fresh4_calibration')
     return ast.copy_location(ast.Constant(value),node)
    return node
  a=fn(ROOT/'our_mani/local_mpc_20261103/depth_owner_fresh4_calibration.py');b=Metadata().visit(fn(ROOT/'our_mani/local_mpc_20261103/depth_owner_position_calibration4.py'))
  self.assertEqual(ast.dump(a,include_attributes=False),ast.dump(b,include_attributes=False))

 def test_host_default_never_spawns_or_creates_output(self):
  path=ROOT/'scripts/launch_near_goal_position_calibration4_20261103.py';spec=importlib.util.spec_from_file_location('positionhost',path);host=importlib.util.module_from_spec(spec);spec.loader.exec_module(host)
  with TemporaryDirectory()as t:
   out=Path(t)/'newcase'
   with patch.object(p,'check_capture_plan',return_value={'complete':True}),patch.object(host.subprocess,'Popen',side_effect=AssertionError('Native forbidden'))as spawn:
    self.assertEqual(host.main(['--plan','unused','--output',str(out)]),0)
    self.assertFalse(out.exists());spawn.assert_not_called()

if __name__=='__main__':unittest.main()
