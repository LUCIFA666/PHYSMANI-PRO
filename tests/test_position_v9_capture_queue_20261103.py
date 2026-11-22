"""Queue admission/STOP/typed-closure tests without Native or model input."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location('v9_capture_queue',ROOT/'scripts/run_position_v9_calibration8_capture_queue_20261103.py')
queue=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(queue)


class V9CaptureQueueContracts(unittest.TestCase):
    def test_preparation_execute_rejected_before_output_or_child(self):
        with tempfile.TemporaryDirectory(prefix='v9_queue_') as temp:
            root=Path(temp);plan=root/'plan.json';plan.write_text('{"preparation_only":true}')
            output=root/'never_created'
            with patch.object(queue.subprocess,'run')as run,contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):queue.main(['--plan',str(plan),'--output-root',str(output),'--execute','--plan-sha256','unused'])
                run.assert_not_called()
            self.assertFalse(output.exists())

    def test_current_capture_metadata_requires_typed_clean_zero_tick_closure(self):
        with tempfile.TemporaryDirectory(prefix='v9_queue_')as temp:
            root=Path(temp);ident=queue.EXPECTED[0];case=root/ident;case.mkdir()
            shots=[dict(shot_id=s,condition='near_goal',condition_detail='fixed',position_band='near_0_08')for s in queue.SHOTS]
            parent=dict(parent_id=ident,shots=shots)
            frames=[dict(s,scene_id=ident+'/'+s['shot_id'],independent_scene=True,physics_ticks=0)for s in shots]
            plan=dict(source_sha256={'capture.py':'fixed'},external_source_sha256={})
            owner=dict(actual_process_returncode=0,actual_wait_completed=True,stop_reason=None,
                source_unchanged=True,parent_id=ident,output=str(case))
            result=dict(complete=True,native_closed=True,source_unchanged=True,parent_id=ident,controlled_physics_ticks=0,shots=frames)
            run=dict(parent_id=ident,plan_sha256='fixed_plan',**plan)
            op=root/(ident+'.owner.actual_exit.json');op.write_text(json.dumps(owner))
            (case/'result.json').write_text(json.dumps(result));(case/'run_inputs.json').write_text(json.dumps(run))
            self.assertEqual(queue.audit_closed(root,parent,plan,'fixed_plan')['shots'],10)
            owner['actual_process_returncode']=False;op.write_text(json.dumps(owner))
            with self.assertRaises(ValueError):queue.audit_closed(root,parent,plan,'fixed_plan')
            owner['actual_process_returncode']=0;op.write_text(json.dumps(owner))
            frames[-1]['position_band']='wrong';(case/'result.json').write_text(json.dumps(result))
            with self.assertRaises(ValueError):queue.audit_closed(root,parent,plan,'fixed_plan')

    def test_parent_queue_stop_prevents_any_second_child(self):
        with tempfile.TemporaryDirectory(prefix='v9_queue_')as temp:
            family=Path(temp);output=family/'capture';path=family/'plan.json'
            plan=dict(preparation_only=False,parents=[dict(parent_id=i)for i in queue.EXPECTED])
            path.write_text(json.dumps(plan))
            def closed(*args):
                (output/'STOP').write_text('user stop\n')
                return dict(parent_id=queue.EXPECTED[0],native_closed=True)
            with patch.object(queue,'FAMILY',family),patch.object(queue,'verify_plan',return_value=plan),\
                    patch('our_mani.local_mpc_20261103.depth_owner_position_v9_calibration8.check_capture_plan',return_value={'capture_ready':True}),\
                    patch.object(queue,'audit_closed',side_effect=closed),\
                    patch.object(queue.subprocess,'run',return_value=SimpleNamespace(returncode=0))as run:
                with self.assertRaises(ValueError):queue.main(['--plan',str(path),'--output-root',str(output),'--plan-sha256','fixed','--execute'])
                self.assertEqual(run.call_count,1)
            self.assertFalse((output/queue.EXPECTED[1]).exists())
            self.assertTrue((output/'STOP').is_file())


if __name__=='__main__':unittest.main()
