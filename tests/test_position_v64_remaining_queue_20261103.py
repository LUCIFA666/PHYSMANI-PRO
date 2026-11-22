"""Synthetic task-false versus infrastructure-stop queue behavior."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from scripts import run_position_v64_remaining_development_20261103 as queue


class RemainingQueueTests(unittest.TestCase):
    def run_queue(self, root, returncodes):
        jobs=[dict(id=name,config=name+'.json',scene='static' if name=='static_no_proxy' else 'uniform',path=name)
              for name in queue.EXPECTED[1:]]
        (root/'runtime/framework/local_mpc_20261102/public_candidate_position_v64_development_v1').mkdir(parents=True)
        patches=[patch.object(queue,'ROOT',root),patch.object(queue,'check_queue',return_value=({},jobs,{'case_id':'00_val_static'})),
                 patch.object(queue,'verify_sources',return_value={}),
                 patch.object(queue,'audit_closed',side_effect=lambda job,s:dict(case_id=job['id'],public_hold_final=False)),
                 patch.object(queue.subprocess,'run',side_effect=[SimpleNamespace(returncode=x)for x in returncodes])]
        return patches

    def test_task_false_does_not_skip_retry_or_replace_any_case(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);patches=self.run_queue(root,[0,0,0])
            with patches[0],patches[1],patches[2],patches[3],patches[4] as run:
                self.assertEqual(queue.main(['--execute']),0)
                self.assertEqual(run.call_count,3)
                ids=[call.args[0][call.args[0].index('--config')+1]for call in run.call_args_list]
                self.assertEqual([Path(x).stem for x in ids],list(queue.EXPECTED[1:]))
            rows=[json.loads(x)for x in next(root.rglob('remaining3_queue_v1.jsonl')).read_text().splitlines()]
            self.assertEqual([r['case_id']for r in rows if r['event']=='actual_closed'],list(queue.EXPECTED[1:]))
            self.assertTrue(rows[-1]['all3_actual_closed'])

    def test_first_infrastructure_failure_stops_without_retry(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);patches=self.run_queue(root,[2])
            with patches[0],patches[1],patches[2],patches[3] as audit,patches[4] as run:
                with self.assertRaisesRegex(RuntimeError,'no next case'):queue.main(['--execute'])
                self.assertEqual(run.call_count,1);audit.assert_not_called()
            rows=[json.loads(x)for x in next(root.rglob('remaining3_queue_v1.jsonl')).read_text().splitlines()]
            self.assertEqual(rows[-1]['event'],'infrastructure_stop');self.assertTrue(rows[-1]['no_retry'])


if __name__=='__main__':unittest.main()
