"""Preparation-only guards and original mechanics; no Native/certificate fake."""
import ast
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from our_mani.local_mpc_20261103 import position_v63_worker as worker
from our_mani.local_mpc_20261102 import worker as original
from scripts import launch_local_mpc_position_v63_20261103 as host

ROOT=Path(__file__).resolve().parents[1]
PLAN=ROOT/'config/framework/P4_position_v63_development_canaries_v1/plan.json'


class WorkerPreparation(unittest.TestCase):
    def test_fixed_four_configs_preserve_parent_motion_and_control(self):
        plan=json.loads(PLAN.read_text())
        self.assertEqual(tuple(j['id'] for j in plan['jobs']),worker.CASE_IDS)
        self.assertFalse(plan['execution_authorized']);self.assertFalse(plan['Native_started'])
        for job in plan['jobs']:
            report=worker.check_config(ROOT/job['config'],job['scene'])
            self.assertFalse(report['ready']);self.assertFalse(report['Native_started'])
            self.assertIn('final_source_freeze_and_external_SHA256',report['missing_admission'])

    def test_default_worker_and_host_check_never_construct_or_spawn(self):
        job=json.loads(PLAN.read_text())['jobs'][0]
        argv=['--config',str(ROOT/job['config']),'--scene',job['scene']]
        with patch.object(original,'make_backend',side_effect=AssertionError('no Native')),\
             patch.object(host.subprocess,'Popen',side_effect=AssertionError('no child')),redirect_stdout(io.StringIO()):
            self.assertEqual(worker.main(argv),0)
            self.assertEqual(host.main(argv),0)

    def test_execute_missing_admission_rejects_before_mkdir_or_Native(self):
        job=json.loads(PLAN.read_text())['jobs'][0]
        with TemporaryDirectory(dir=ROOT/'runtime/framework/local_mpc_20261102') as tmp:
            output=Path(tmp)/'never_created'/'case'
            argv=['--config',str(ROOT/job['config']),'--scene',job['scene'],'--output',str(output),'--execute']
            with patch.object(original,'make_backend',side_effect=AssertionError('no Native')),\
                 patch.object(host.subprocess,'Popen',side_effect=AssertionError('no child')):
                with self.assertRaisesRegex(ValueError,'blocked'):worker.main(argv)
                with self.assertRaisesRegex(ValueError,'blocked'):host.main(argv)
            self.assertFalse(output.parent.exists())

    def test_changed_seed_budget_or_physics_rejected_before_admission(self):
        job=json.loads(PLAN.read_text())['jobs'][0]
        value=json.loads((ROOT/job['config']).read_text())
        with TemporaryDirectory(dir=ROOT/'runtime/framework/local_mpc_20261102') as tmp:
            for key,new in [('seed',123),('physics_dt',.1),('cem_sequences_total',128)]:
                changed=dict(value);changed[key]=new;path=Path(tmp)/(key+'.json');path.write_text(json.dumps(changed))
                with self.assertRaisesRegex(ValueError,'preserve'):
                    worker.check_config(path,job['scene'])
        with self.assertRaises(ValueError):worker.check_config(ROOT/job['config'],'uniform')

    def test_old_native_mechanics_and_judge_source_untouched(self):
        protected=json.loads((ROOT/'config/framework/P3_near_goal_depth_owner_position_calibration4_20261103_v1.json').read_text())['source_sha256']
        names=('our_mani/local_mpc_20261102/worker.py','our_mani/local_mpc_20261102/records.py',
               'our_mani/local_mpc_20261102/native.py','our_mani/local_mpc_20261102/executor.py')
        for name in names:self.assertEqual(worker.sha(ROOT/name),protected[name])
        self.assertIs(worker.original.make_backend,original.make_backend)
        self.assertIs(worker.original.make_executor,original.make_executor)
        self.assertIs(worker.original.append_ticks,original.append_ticks)
        source=ast.parse(Path(worker.__file__).read_text())
        calls=[n for n in ast.walk(source) if isinstance(n,ast.Call)]
        self.assertEqual(sum(isinstance(n.func,ast.Attribute) and n.func.attr=='make_backend' for n in calls),1)
        self.assertFalse(any(isinstance(n.func,ast.Name) and n.func.id in ('run_p1','evaluate_case','evaluate_states') for n in calls))

    def test_shell_reuses_same_locks_environment_and_frozen_wrapper(self):
        before=(ROOT/'scripts/run_local_mpc_20261102.sh').read_text()
        after=(ROOT/'scripts/run_local_mpc_position_v63_20261103.sh').read_text().replace(
            'scripts/run_local_mpc_position_v63_20261103.py','scripts/run_local_mpc_20261102.py')
        lines=lambda text:[line for line in text.splitlines() if line.strip() and not line.lstrip().startswith('#')]
        self.assertEqual(lines(before),lines(after))
        self.assertIn('process.wait()',Path(host.__file__).read_text())
        self.assertIn('start_new_session=True',Path(host.__file__).read_text())

    def test_common_closure_includes_negative_only_dependency_for_all_cases(self):
        plan=json.loads(PLAN.read_text());fake=SimpleNamespace(model_sources=lambda:{})
        producer=SimpleNamespace(position_producer_source_map=lambda:{})
        sources=worker._source_closure(ROOT/plan['jobs'][0]['config'],producer,fake)
        self.assertIn('our_mani/local_mpc_control_20261103/no_proxy_attachment.py',sources)
        self.assertIn('our_mani/local_mpc_control_20261103/native_task_evaluation.py',sources)
        for row in plan['jobs']:self.assertIn(row['config'],sources)


if __name__=='__main__':unittest.main()
