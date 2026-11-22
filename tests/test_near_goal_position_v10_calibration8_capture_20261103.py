"""V10 static collector CPU contracts only; no Native/model/calibration forward."""
from contextlib import redirect_stdout
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import ast,importlib.util,inspect,io,json,unittest
import numpy as np
from our_mani.local_mpc_20261103 import depth_owner_position_v10_calibration8 as p
from scripts import collect_near_goal_position_v10_calibration8_20261103 as cli
from tests.test_near_goal_perception_capture_20261103 import Backend,shot
ROOT=Path(__file__).resolve().parents[1]


class V10CaptureContracts(unittest.TestCase):
    def test_outside_bootstrap_then_inner_scene_restores_both_legal_states(self):
        b=Backend();native=p.snapshot_pose(b);bootstrap=dict(cube_mode='fixed_world_pose',fixed_cube_pose_world=[100.,100.,-100.,0.,0.,0.,1.],joint_positions=[.1]*7,finger_positions_m=[.04,.04])
        with p.zero_clock_shot(b,bootstrap,None)as outer:
            np.testing.assert_array_equal(b.task.target_block.pose[:3],[100.,100.,-100.]);np.testing.assert_array_equal(b.arm.q,[.1]*7)
            legal=p.snapshot_pose(b)
            with p.zero_clock_shot(b,shot('near',True),{'center':[.3,.1,.9]})as inner:
                np.testing.assert_array_equal(b.task.target_block.pose[:3],[.36,.1,.9])
            self.assertTrue(inner['restored'])
            for k in('arm_q','arm_target','finger_q','finger_target','cube_pose','ik_pose'):np.testing.assert_array_equal(p.snapshot_pose(b)[k],legal[k])
        self.assertTrue(outer['restored']);self.assertFalse(outer['dynamics_restoration_claimed'])
        for k in('arm_q','arm_target','finger_q','finger_target','cube_pose','ik_pose'):np.testing.assert_array_equal(p.snapshot_pose(b)[k],native[k])
        self.assertTrue(all(v[1]is False for node in(b.arm,b.gripper,b.task.target_block,b.arm._ik_target)for v in node.calls))
    def test_exception_restores_outer_and_false_setter_clock_violation_rejects(self):
        b=Backend();saved=p.snapshot_pose(b);s=dict(cube_mode='fixed_world_pose',fixed_cube_pose_world=[100.,100.,-100.,0.,0.,0.,1.],joint_positions=[.1]*7,finger_positions_m=[.04,.04])
        with self.assertRaisesRegex(RuntimeError,'synthetic image failure'):
            with p.zero_clock_shot(b,s,None):raise RuntimeError('synthetic image failure')
        np.testing.assert_array_equal(p.snapshot_pose(b)['cube_pose'],saved['cube_pose'])
        b.advance_on_set=True
        with self.assertRaisesRegex(p.CaptureInvariantError,'clock'):
            with p.zero_clock_shot(b,s,None):pass
    def test_original_transaction_AST_except_explicit_world_branch_is_exact(self):
        class RemoveWorld(ast.NodeTransformer):
            def visit_If(self,node):
                node=self.generic_visit(node)
                if isinstance(node.test,ast.Compare)and any(isinstance(x,ast.Constant)and x.value=='fixed_world_pose'for x in node.test.comparators):return node.orelse
                return node
        a=ast.parse(inspect.getsource(p.b.zero_clock_shot));b=RemoveWorld().visit(ast.parse(inspect.getsource(p.zero_clock_shot)))
        self.assertEqual(ast.dump(a),ast.dump(b))
        for name in('clock','normalize_baseline_fingers','capture_checked','archive_capture_sources','public_payload','fit_public_sphere'):
            self.assertIs(getattr(p,name),getattr(p.b,name))
    def test_capture_and_host_preparation_only_refuse_before_creation(self):
        with TemporaryDirectory()as t:
            out=Path(t)/'case'
            with patch.object(p,'checked_pilot_plan',return_value=({'preparation_only':True},{},{})):
                with self.assertRaisesRegex(ValueError,'Preparation-only'):p.capture_parent(ROOT,'unused','parent',out)
            self.assertFalse(out.exists())
            file=Path(t)/'plan.json';file.write_text(json.dumps({'preparation_only':True}))
            spec=importlib.util.spec_from_file_location('new_v9_host',ROOT/'scripts/launch_near_goal_position_v10_calibration8_20261103.py');host=importlib.util.module_from_spec(spec);spec.loader.exec_module(host)
            with patch.object(p,'check_capture_plan',return_value={'complete':True,'capture_ready':False}),patch.object(host.subprocess,'Popen',side_effect=AssertionError('Native forbidden'))as spawn,redirect_stdout(io.StringIO()):
                with self.assertRaises(SystemExit):host.main(['--execute','--plan',str(file),'--parent-id','parent','--output',str(out)])
                spawn.assert_not_called()
            self.assertFalse(out.exists())
    def test_actual_capture_CLI_branch_serializes_numpy_and_default_only_checks(self):
        result={'complete':True,'q':np.arange(7),'n':np.int64(10),'camera':{'normal':np.eye(3)}};stream=io.StringIO()
        with patch.object(cli.producer,'capture_parent',return_value=result)as capture,patch.object(cli.producer,'check_capture_plan',side_effect=AssertionError('not capture'))as check,redirect_stdout(stream):
            self.assertEqual(cli.main(['--capture','--plan','unused','--parent-id','fixed','--output','fresh']),0);capture.assert_called_once();check.assert_not_called()
        self.assertEqual(json.loads(stream.getvalue())['n'],10)
        with patch.object(cli.producer,'check_capture_plan',return_value={'complete':True}),patch.object(cli.producer,'capture_parent',side_effect=AssertionError('Native forbidden'))as capture,redirect_stdout(io.StringIO()):
            self.assertEqual(cli.main(['--plan','unused']),0);capture.assert_not_called()
    def test_queue_parent_STOP_is_preserved_by_core_and_host_without_spawn(self):
        with TemporaryDirectory()as t:
            root=Path(t);family=root/'family';family.mkdir();output=family/'case';marker=family/'STOP';marker.write_text('requested')
            self.assertTrue(p.stop_requested(ROOT,output));self.assertEqual(p.stop_paths(ROOT,output)[1],family/'case.STOP')
            self.assertEqual(p.stop_paths(ROOT,output)[-1],marker)
            plan=root/'plan.json';plan.write_text(json.dumps({'preparation_only':False}))
            spec=importlib.util.spec_from_file_location('stop_v9_host',ROOT/'scripts/launch_near_goal_position_v10_calibration8_20261103.py');host=importlib.util.module_from_spec(spec);spec.loader.exec_module(host)
            with patch.object(host,'ROOT',root),patch.object(host,'FAMILY',family),patch.object(p,'check_capture_plan',return_value={'complete':True}),patch.object(host.subprocess,'Popen',side_effect=AssertionError('Native forbidden'))as spawn:
                with self.assertRaises(SystemExit):host.main(['--execute','--plan',str(plan),'--parent-id','fixed','--output',str(output)])
                spawn.assert_not_called()
            self.assertTrue(marker.exists());self.assertEqual(marker.read_text(),'requested');self.assertFalse(output.exists())
            self.assertEqual(list(family.glob('*.owner*')),[])
    def test_stale_collector_metadata_rejects_before_other_file_or_HDF_reads(self):
        plan_path=ROOT/'config/framework/P3_near_goal_depth_owner_position_calibration8_v10_20261103_v1.json'
        value={'collector_script':'scripts/old_capture.py','owner_launcher':'old','owner_shell':'old','preparation_only':True}
        with patch.object(Path,'read_text',return_value=json.dumps(value)),patch.object(p.h5py,'File',side_effect=AssertionError('No HDF beforemetadata'))as hdf:
            with self.assertRaisesRegex(ValueError,'collector/owner/shell metadata'):p.checked_pilot_plan(ROOT,plan_path)
            hdf.assert_not_called()
    def test_source_order_public_inputs_and_10shot_metadata_are_explicit(self):
        source=inspect.getsource(p.capture_parent)
        self.assertLess(source.index('with zero_clock_shot(backend,bootstrap_shot,None)'),source.index("capture_checked(bundle,roles,output,'initialization')"))
        self.assertLess(source.index('fit_public_sphere('),source.index('materialize_v10_shots('));self.assertLess(source.index('materialize_v10_shots('),source.index("for shot in materialized['shots']"))
        self.assertEqual(source.count('fit_public_sphere('),1)
        self.assertIn("position_band=shot['position_band']",source)
        self.assertEqual(p.SEEDS,tuple(range(261103656,261103664)))
        self.assertEqual(p.SCOPE,'position_calibration8_v10')
        self.assertEqual([x['shot_id']for x in p.shot_specs_for_parent(0)],['shot'+str(i).zfill(2)for i in range(10)])

if __name__=='__main__':unittest.main()
