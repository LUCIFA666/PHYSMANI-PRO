"""Fresh4 source/role/host contracts only; no Native/fit/calibration image read."""
from copy import deepcopy
from pathlib import Path
import ast,importlib.util,json,tempfile,unittest
from unittest.mock import patch
import numpy as np
from our_mani.local_mpc_20261103 import depth_owner_fresh4_calibration as fresh
from our_mani.local_mpc_20261103 import depth_owner_full32 as old
from scripts import collect_near_goal_fresh4_calibration_20261103 as collector

ROOT=Path(__file__).resolve().parents[1]
PLAN=ROOT/'config/framework/P3_near_goal_depth_owner_fresh4_calibration_20261103_v1.json'


class FreshCalibrationTests(unittest.TestCase):
    def test_partition_matches_actual_fresh_loader_ordered_shot_contract(self):
        plan=json.loads(PLAN.read_text())
        partition=json.loads((ROOT/plan['partition']['path']).read_text())
        expected=['shot'+str(i).zfill(2) for i in range(8)]
        for parent in partition['parents']:
            self.assertEqual(parent['shot_ids'],expected)
            self.assertEqual(parent['shot_ids'],[shot['shot_id'] for shot in parent['shots']])
            self.assertIs(parent['pilot'],False)
            self.assertEqual(parent['role'],'calibration')

    def test_fixed_seeds_first_unused_sources_roles_and_balanced_layout(self):
        plan=json.loads(PLAN.read_text());parents=plan['parents']
        self.assertEqual([p['seed'] for p in parents],list(fresh.SEEDS))
        self.assertEqual([p['TRAIN_q_source']['raw_episode_id'] for p in parents],list(fresh.RAW_IDS))
        for i,p in enumerate(parents):
            self.assertEqual(p['role'],'calibration');self.assertEqual(p['fold'],'calibration');self.assertFalse(p['pilot'])
            self.assertEqual(p['shots'],old.shot_specs_for_parent(i))
        d=np.asarray([p['near_direction_unit_world'] for p in parents])
        np.testing.assert_array_equal((d>0).sum(0),[2,2,2]);np.testing.assert_array_equal((d<0).sum(0),[2,2,2])
        self.assertEqual([p['near_direction_ordinal'] for p in parents],[0,1,2,3])

    def test_capture_solver_render_restore_mechanics_are_reused(self):
        for name in ('materialize_pilot_shots','solve_fixed_starts','outside_frusta_proof'):
            self.assertIs(getattr(fresh,name),getattr(old,name))
        for name in ('clock','zero_clock_shot','normalize_baseline_fingers','capture_checked','archive_capture_sources'):
            self.assertIs(getattr(fresh,name),getattr(old,name))
        def body(path):
            return next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='capture_parent')
        class MetadataOnly(ast.NodeTransformer):
            def visit_Call(self,n):
                n=self.generic_visit(n);n.keywords=[k for k in n.keywords if k.arg not in {'calibration_only','calibration_cohort','excluded_from_fit_and_selection'}];return n
            def visit_Constant(self,n):
                if isinstance(n.value,str) and ('full32 per-parent8 newshots' in n.value or 'fresh4 calibration-only per-parent8newshots' in n.value):
                    return ast.copy_location(ast.Constant(value='declared_capture_scope_metadata'),n)
                return n
        a=MetadataOnly().visit(body(ROOT/'our_mani/local_mpc_20261103/depth_owner_full32.py'))
        b=MetadataOnly().visit(body(ROOT/'our_mani/local_mpc_20261103/depth_owner_fresh4_calibration.py'))
        self.assertEqual(ast.dump(a,include_attributes=False),ast.dump(b,include_attributes=False))

    def test_old_collector_does_not_silently_accept_new_schema(self):
        plan=json.loads(PLAN.read_text());self.assertNotEqual(plan['schema'],old.SCHEMA)
        self.assertEqual(plan['scope'],'fresh4_calibration')
        with self.assertRaises(ValueError):old.checked_pilot_plan(ROOT,PLAN)

    def test_role_and_pilot_corruption_rejected_before_capture(self):
        original=json.loads(PLAN.read_text())
        for key,value in [('role','fit'),('pilot',True),('seed',261103628)]:
            changed=deepcopy(original);changed['parents'][0][key]=value
            with tempfile.NamedTemporaryFile(mode='w',suffix='.json',dir=ROOT/'config/framework',delete=False) as stream:
                path=Path(stream.name);json.dump(changed,stream)
            try:
                with self.assertRaises(ValueError):fresh.checked_pilot_plan(ROOT,path)
            finally:path.unlink()

    def test_worker_default_never_enters_capture(self):
        with patch.object(collector.producer,'check_capture_plan',return_value={'complete':True}) as check,\
             patch.object(collector.producer,'capture_parent',side_effect=AssertionError('Native forbidden')) as capture:
            self.assertEqual(collector.main(['--plan','unused.json']),0)
            check.assert_called_once();capture.assert_not_called()

    def test_host_default_preflight_never_spawns_or_creates_output(self):
        path=ROOT/'scripts/launch_near_goal_fresh4_calibration_20261103.py'
        spec=importlib.util.spec_from_file_location('fresh4_host_test',path);host=importlib.util.module_from_spec(spec);spec.loader.exec_module(host)
        with tempfile.TemporaryDirectory() as t:
            out=Path(t)/'not_created'
            with patch.object(fresh,'check_capture_plan',return_value={'complete':True}),\
                 patch.object(host.subprocess,'Popen',side_effect=AssertionError('Native forbidden')) as popen:
                self.assertEqual(host.main(['--plan','unused.json','--output',str(out)]),0)
                popen.assert_not_called();self.assertFalse(out.exists())


if __name__=='__main__':unittest.main()
