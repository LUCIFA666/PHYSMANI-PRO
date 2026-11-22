"""After-exit all-case guard and pooled wall denominators, no real GT read."""
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import json,unittest
from scripts import evaluate_public_position_closed_cases_20261103 as e

class ClosedEvaluatorTests(unittest.TestCase):
 def metadata_case(self,base,name,*,source_corrupt=False):
  case=base/name;case.mkdir();source='our_mani/local_mpc_control_20261103/native_task_evaluation.py';digest=e.sha(e.ROOT/source)
  archived=case/'source_snapshot'/source;archived.parent.mkdir(parents=True);archived.write_bytes(b'changed'if source_corrupt else(e.ROOT/source).read_bytes())
  values={'worker_exit.json':{'exit_code':0,'native_closed':True},'result.json':{'complete':True,'source_unchanged':True},
   'run_inputs.json':{'TEST_read':False,'config':{'formal_evaluation':False},'source_sha256':{source:digest}},'execution_source_freeze.json':{'files_sha256':{source:digest}}}
  for n,v in values.items():(case/n).write_text(json.dumps(v))
  # Deliberately invalid truth JSON must never be decoded before cohort gate.
  (case/'initial_snapshot.json').write_text('not truth JSON');(case/'native_ticks.jsonl').write_text('not truth JSON')
  case.with_name(name+'.owner.actual_exit.json').write_text(json.dumps(dict(actual_process_returncode=0,actual_wait_completed=True,source_unchanged=True,stop_reason=None,output=str(case))))
  return case

 def test_missing_second_actual_exit_prevents_first_case_GT_and_output(self):
  with TemporaryDirectory(prefix='closed_eval_')as t:
   base=Path(t);first=self.metadata_case(base,'first');second=base/'second';second.mkdir();plan=base/'plan.json';plan.write_text(json.dumps({'jobs':[{'id':'first','path':str(first)},{'id':'second','path':str(second)}]}));out=base/'new'
   with patch.object(e,'evaluate_case',side_effect=AssertionError('GT must wait for wholecohort'))as judge:
    with self.assertRaises(FileNotFoundError):e.evaluate_plan(plan,out)
    judge.assert_not_called();self.assertFalse(out.exists())

 def test_source_corruption_rejected_before_GT_or_output(self):
  with TemporaryDirectory(prefix='closed_eval_')as t:
   base=Path(t);case=self.metadata_case(base,'first',source_corrupt=True);plan=base/'plan.json';plan.write_text(json.dumps({'jobs':[{'id':'first','path':str(case)}]}));out=base/'new'
   with patch.object(e,'evaluate_case',side_effect=AssertionError('GT forbidden'))as judge:
    with self.assertRaisesRegex(ValueError,'Archived source changed'):e.evaluate_plan(plan,out)
    judge.assert_not_called();self.assertFalse(out.exists())

 def test_quantiles_pool_all_actual_cycle_samples(self):
  q=e.quantiles([.05,.15]);self.assertEqual(q['samples'],2);self.assertAlmostEqual(q['p50_seconds'],.1);self.assertAlmostEqual(q['p95_seconds'],.145)
  self.assertEqual(q['over100ms_count'],1);self.assertEqual(q['over100ms_fraction'],.5);self.assertFalse(q['real_time_qualified'])

if __name__=='__main__':unittest.main()
