"""Fixed complete development cohort and unchanged physical execution."""
import ast
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from our_mani.local_mpc_20261103 import position_v64_devval8_worker as worker
from our_mani.local_mpc_20261103 import position_v64_worker as old
from scripts import launch_local_mpc_position_v64_devval8_20261103 as host
from scripts import run_position_v64_full_devval8_20261103 as queue

ROOT=Path(__file__).resolve().parents[1]
PLAN=ROOT/'config/framework/P4_student_closedloop_devVAL_v64_v1/plan.json'


class FullDevelopmentPreparation(unittest.TestCase):
    def test_fixed8_V56_parents_numeric_configs_and_failed_public_gate_preserved(self):
        plan=json.loads(PLAN.read_text());prior=json.loads(worker.REFERENCE_PLAN.read_text())
        self.assertEqual(tuple(j['id']for j in plan['jobs']),worker.CASE_IDS)
        self.assertEqual([j['seed']for j in plan['jobs']],list(range(261103100,261103108)))
        self.assertFalse(plan['public_hold_canary_gate']['passed']);self.assertTrue(plan['reuse_development'])
        self.assertTrue(plan['all8_regardless_of_task_outcomes']);self.assertFalse(plan['task_failure_retry'])
        self.assertFalse(plan['formal_TEST_used']);self.assertFalse(plan['new_TEST_500_515_read_or_materialized'])
        for row,original in zip(plan['jobs'],prior['jobs']):
            value=json.loads((ROOT/row['config']).read_text());base=json.loads((ROOT/original['config']).read_text())
            report=worker.check_config(ROOT/row['config'],row['scene'])
            self.assertFalse(report['Native_started']);self.assertFalse(report['ready'])
            self.assertEqual(value['seed'],base['seed']);self.assertEqual(row['parent_id'],original['parent_id'])
            self.assertEqual(value['P4_collection_metadata'],base['P4_collection_metadata'])
            self.assertTrue(value['candidate_goal_sphere_filter_v56']);self.assertTrue(value['finger_response_fastpath'])

    def test_execute_AST_and_shell_physical_environment_exact_fourcase_V64(self):
        def function(path,name):return next(x for x in ast.parse(Path(path).read_text()).body if isinstance(x,ast.FunctionDef)and x.name==name)
        self.assertEqual(ast.dump(function(worker.__file__,'execute'),include_attributes=False),ast.dump(function(old.__file__,'execute'),include_attributes=False))
        shell=(ROOT/'scripts/run_local_mpc_position_v64_devval8_20261103.sh').read_text().replace('local_mpc_position_v64_devval8_20261103','local_mpc_position_v64_20261103')
        self.assertEqual(shell,(ROOT/'scripts/run_local_mpc_position_v64_20261103.sh').read_text())
        self.assertIs(worker.original.make_backend,old.original.make_backend)
        self.assertIs(worker.original.make_executor,old.original.make_executor)

    def test_check_only_host_worker_queue_never_construct_Native_or_child(self):
        row=json.loads(PLAN.read_text())['jobs'][0];args=['--config',str(ROOT/row['config']),'--scene',row['scene']]
        with patch.object(worker.original,'make_backend',side_effect=AssertionError('no Native')),\
             patch.object(host.subprocess,'Popen',side_effect=AssertionError('no process')),\
             patch.object(queue.subprocess,'run',side_effect=AssertionError('no queue')),redirect_stdout(io.StringIO()):
            self.assertEqual(worker.main(args),0);self.assertEqual(host.main(args),0);self.assertEqual(queue.main([]),0)

    def test_missing_freeze_rejects_execute_before_mkdir(self):
        row=json.loads(PLAN.read_text())['jobs'][0]
        with TemporaryDirectory(dir=ROOT/'runtime/framework/local_mpc_20261102')as tmp:
            output=Path(tmp)/'fresh_parent'/'case';args=['--config',str(ROOT/row['config']),'--scene',row['scene'],'--output',str(output),'--execute']
            with patch.object(worker.original,'make_backend',side_effect=AssertionError('no Native')):
                with self.assertRaises(ValueError):worker.main(args)
            self.assertFalse(output.parent.exists())

    def test_physical_fields_or_frozen_compute_flag_cannot_change(self):
        row=json.loads(PLAN.read_text())['jobs'][0];base=json.loads((ROOT/row['config']).read_text())
        with TemporaryDirectory(dir=ROOT/'runtime/framework/local_mpc_20261102')as tmp:
            for name,value in [('seed',123),('physics_dt',.1),('finger_response_fastpath',False),('candidate_goal_sphere_filter_v56',False)]:
                altered=dict(base);altered[name]=value;path=Path(tmp)/(name+'.json');path.write_text(json.dumps(altered))
                with self.assertRaises(ValueError):worker.check_config(path,row['scene'])
        frozen=json.loads((ROOT/'artifacts/framework/local_mpc_20261102/V64_independent_worker_preparation_v1/execution_source_freeze.json').read_text())
        for name,expected in frozen['files_sha256'].items():self.assertEqual(worker.sha(ROOT/name),expected)

    def test_explicit_failed_public_gate_acknowledgement_is_required(self):
        plan=json.loads(PLAN.read_text());reports=[{'ready':True}]*8
        with patch.object(queue,'check_plan',return_value=(plan,reports)),\
             patch.object(queue,'verify_sources',side_effect=AssertionError('no source/Native execution'))as verify,\
             redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit):queue.main(['--execute'])
            verify.assert_not_called()

    def test_all8_task_false_complete_once_without_retry(self):
        plan=json.loads(PLAN.read_text());reports=[{'ready':True}]*8
        with TemporaryDirectory()as tmp:
            root=Path(tmp)
            with patch.object(queue,'ROOT',root),patch.object(queue,'check_plan',return_value=(plan,reports)),\
                 patch.object(queue,'verify_sources',return_value={}),\
                 patch.object(queue,'audit_closed',side_effect=lambda j,s:dict(case_id=j['id'],public_hold_final=False)),\
                 patch.object(queue.subprocess,'run',return_value=SimpleNamespace(returncode=0))as run:
                self.assertEqual(queue.main(['--execute','--source-freeze',str(root/'freeze'),'--source-freeze-sha256','a'*64,'--acknowledge-public-hold-canary-failed']),0)
                self.assertEqual(run.call_count,8)
            rows=[json.loads(x)for x in next(root.rglob('serial_queue_v1.jsonl')).read_text().splitlines()]
            self.assertEqual([r['case_id']for r in rows if r['event']=='actual_closed'],list(queue.EXPECTED))
            self.assertFalse(rows[-1]['public_hold_canary_gate_passed'])

    def test_infrastructure_failure_stops_queue_without_task_retry(self):
        plan=json.loads(PLAN.read_text());reports=[{'ready':True}]*8
        with TemporaryDirectory()as tmp:
            root=Path(tmp)
            with patch.object(queue,'ROOT',root),patch.object(queue,'check_plan',return_value=(plan,reports)),\
                 patch.object(queue,'verify_sources',return_value={}),\
                 patch.object(queue,'audit_closed',side_effect=AssertionError('no audit nonzero host'))as audit,\
                 patch.object(queue.subprocess,'run',return_value=SimpleNamespace(returncode=2))as run:
                with self.assertRaisesRegex(RuntimeError,'stop without retry'):
                    queue.main(['--execute','--source-freeze',str(root/'freeze'),'--source-freeze-sha256','a'*64,'--acknowledge-public-hold-canary-failed'])
                self.assertEqual(run.call_count,1);audit.assert_not_called()


if __name__=='__main__':unittest.main()
