from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import json,unittest
from scripts import evaluate_frozen_position_TEST16_closed_20261103 as e

class EvaluationOnlyTests(unittest.TestCase):
 def test_exact_preassigned16_identity_order_and_count(self):
  assigned=e.common.read(e.PARTITION)['jobs'];jobs=[dict(x,path='/not_materialized/'+x['id'])for x in assigned]
  e.validate_jobs(jobs,assigned)
  for altered in (jobs[:-1],list(reversed(jobs))):
   with self.assertRaises(ValueError):e.validate_jobs(altered,assigned)
  changed=deepcopy(jobs);changed[0]['seed']=261103999
  with self.assertRaises(ValueError):e.validate_jobs(changed,assigned)

 def test_unapproved_plan_cannot_read_GT_or_create_report(self):
  with TemporaryDirectory(prefix='frozen_eval_')as t:
   root=Path(t);plan=root/'plan.json';plan.write_text(json.dumps({'schema':e.SCHEMA,'evaluation_only':True,'public_hold_canary_gate':{'passed':True}}));out=root/'output'
   with patch.object(e,'evaluate_states',side_effect=AssertionError('GT must wait'))as judge:
    with self.assertRaises(ValueError):e.evaluate_plan(plan,out)
    judge.assert_not_called();self.assertFalse(out.exists())

 def test_original_judge_TEST_guard_remains_unchanged(self):
  from our_mani.local_mpc_control_20261103.native_task_evaluation import evaluate_case
  with TemporaryDirectory(prefix='frozen_eval_')as t:
   forbidden=Path(t)/'test';forbidden.mkdir()
   with self.assertRaisesRegex(ValueError,'non-TEST'):evaluate_case(forbidden)

if __name__=='__main__':unittest.main()
