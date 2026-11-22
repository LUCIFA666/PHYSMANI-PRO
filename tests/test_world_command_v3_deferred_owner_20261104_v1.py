"""Toy stdlib child lifecycles only; no Native/GPU/model/cache operations."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('world_v3_owner_test',ROOT/'scripts/run_world_command_v3_deferred_20261104_v1.py')
owner=importlib.util.module_from_spec(spec);spec.loader.exec_module(owner)


class Lifecycle(unittest.TestCase):
    def config(self,root,body,timeout=1.):
        script=root/'toy.py';script.write_text(body)
        mapping=root/'sources.json';mapping.write_text(json.dumps({'source_sha256':{str(script):owner.sha(script)}}))
        out=root/'out';out.mkdir();(out/'owners').mkdir()
        return dict(python=sys.executable,output_root=str(out),method_sources=owner.pointer(mapping),
            CF_output_root=str(root/'no_native'),stage_timeout_seconds={'toy':timeout},poll_seconds=.01,stop_grace_seconds=.02),script

    def test_actual_child_zero_wait_and_no_outer_resource_ownership(self):
        with tempfile.TemporaryDirectory()as d:
            cfg,script=self.config(Path(d),'print("toy only")\n')
            result=owner.base.run_stage(cfg,'toy',[str(script)],cpu=True)
            self.assertEqual(result['actual_process_returncode'],0)
            self.assertTrue(result['actual_wait_completed']);self.assertTrue(result['source_unchanged'])
            request=owner.read(Path(cfg['output_root'])/'owners/toy/request.json')
            self.assertFalse(request['driver_holds_resource_locks']);self.assertEqual(result['unreaped_descendants'],[])

    def test_timeout_preserves_failure_waits_and_does_not_retry(self):
        with tempfile.TemporaryDirectory()as d:
            cfg,script=self.config(Path(d),'import time\ntime.sleep(10)\n',timeout=.03)
            with self.assertRaises(RuntimeError):owner.base.run_stage(cfg,'toy',[str(script)],cpu=True)
            actual=owner.read(Path(cfg['output_root'])/'owners/toy/actual_exit.json')
            self.assertTrue(actual['actual_wait_completed']);self.assertNotEqual(actual['actual_process_returncode'],0)
            self.assertEqual(actual['stop_reason'],'stage_timeout');self.assertEqual(actual['unreaped_descendants'],[])
            with self.assertRaises(FileExistsError):owner.base.run_stage(cfg,'toy',[str(script)],cpu=True)

    def test_new_family_STOP_is_seen_inside_original_helper_loop(self):
        with tempfile.TemporaryDirectory()as d:
            root=Path(d);cfg,script=self.config(root,'pass\n')
            newplan=root/'newfamily/plan.json';newplan.parent.mkdir();(newplan.parent/'STOP').write_text('stop')
            with patch.object(owner,'DEFAULT_PLAN',newplan):
                with self.assertRaises(InterruptedError):owner.base.assert_running(cfg)
            self.assertEqual(len(owner.STAGES),6)
            self.assertEqual(owner.STAGES[0],'world_GPU_canary4')
            self.assertEqual(owner.STAGES[-1],'world_CPU_selection_score5168')

    def test_partial_e2_stage_set_denied_before_selected_or_authorization(self):
        with tempfile.TemporaryDirectory()as d:
            root=Path(d);planfile=root/'priorplan.json';planfile.write_text('{}')
            stages=('e2_GPU_preflight','e2_fit10','e2_selected646_evaluation')
            partial=dict(complete=True,completed_stages=list(stages[:2]),
                current_network_end_to_end_ready=False,Native_VAL_TEST_started=False)
            for name in ('completion.json','terminal_status.json'):(root/name).write_text(json.dumps(partial))
            helper=SimpleNamespace(STAGES=stages,load_plan=lambda *a:{},validate_prior=lambda p:{})
            cfg=dict(e2_plan=owner.pointer(planfile),e2_output_root=str(root))
            with patch.object(owner,'prior_helper',return_value=helper):
                with self.assertRaisesRegex(ValueError,'All three completed'):
                    owner.validate_prior(cfg)
            self.assertFalse((root/'Root_GPU_authorization.json').exists())

    def test_prior_failure_stops_wait_without_successor_or_resample(self):
        with tempfile.TemporaryDirectory()as d:
            root=Path(d);(root/'failure.json').write_text('{}')
            cfg=dict(e2_output_root=str(root),wait_timeout_seconds=1,poll_seconds=.01)
            with patch.object(owner,'assert_running'),patch.object(owner.base,'verify_sources'):
                with self.assertRaisesRegex(ValueError,'e2 failed'):
                    owner.wait_prior(cfg)


if __name__=='__main__':unittest.main()
