"""The active entries share60s, while historical16s receipts stay reproducible."""
import json
from pathlib import Path
import unittest
from our_mani.task_time_limits_20261111_v1 import TaskBudget
from our_mani.highspeed_control_20261111_v1.task_budget import TaskBudget as HistoricalBudget
from our_mani.multitask_execution_pipeline_20261104_v1.registry import TASKS
from scripts import run_physmani_cube as cube
from scripts.validate_perception_repair_20261111_v1 import source_manifest


class Active60sContracts(unittest.TestCase):
    def test_all_active_tasks_use60seconds_1200_ticks(self):
        b=TaskBudget.from_config({});self.assertEqual(b.seconds,60.);self.assertEqual(b.ticks,1200)
        active=json.loads((cube.ROOT/'config/framework/task_time_limits_20261111_v1/active_runtime.json').read_text())
        self.assertEqual(set(active['tasks']),set(TASKS))
        for row in active['tasks'].values():self.assertEqual(row,dict(simulation_seconds=60.,budget_ticks=1200))
    def test_historical16s_budget_and_explicit_overrides_remain_distinct(self):
        self.assertEqual(HistoricalBudget().seconds,16.)
        self.assertEqual(TaskBudget.from_config(dict(budget_ticks=320)).seconds,16.)
        with self.assertRaises(ValueError):TaskBudget.from_config(dict(task_time_limit_seconds=60.,budget_ticks=320))
    def test_both_cube_entry_tasks_prepare_full60s_without_launching_native(self):
        for task in ['ordinary','high_speed']:
            output=cube.ROOT/'artifacts/framework/task_time_limits_20261111_v1'/('unexecuted_prepare_'+task)
            path,_=cube.prepare(task,2611116100,output)
            plan=json.loads(path.read_text());config=plan['cases'][0]['config']
            self.assertEqual(config['budget_ticks'],1200);self.assertEqual(config['task_time_limit_seconds'],60.)
            self.assertEqual(plan['worker_module'],'our_mani.perception_repair_20261111_v1.native_worker')
            self.assertEqual(plan['scope'],'perception_grasp_repair_development')
            self.assertFalse(output.exists())
            manifest=source_manifest(path)
            self.assertIn('our_mani/task_time_limits_20261111_v1/budget.py',manifest['repository'])
            self.assertIn('scripts/run_physmani_cube.py',manifest['repository'])
            self.assertIn('our_mani/perception_repair_20261111_v1/tracker.py',manifest['repository'])


if __name__=='__main__':unittest.main()
