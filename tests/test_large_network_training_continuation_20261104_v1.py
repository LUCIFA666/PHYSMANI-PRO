"""Pure admission/owner contracts. Toy child processes are stdlib-only, no model/Native."""
import contextlib
from copy import deepcopy
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('continuation',ROOT/'scripts/run_large_network_training_continuation_20261104_v1.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
CONFIG=m.DEFAULT_CONFIG


def save(path,value):
 path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))


class ContinuationContracts(unittest.TestCase):
 def test_default_no_child_no_NN_and_only_pending_admissions(self):
  with patch.object(m.subprocess,'Popen',side_effect=AssertionError('No child in default')),contextlib.redirect_stdout(io.StringIO())as out:
   self.assertEqual(m.main(['--config',str(CONFIG)]),0)
  value=json.loads(out.getvalue());self.assertTrue(value['preparation_only']);self.assertEqual(value['child_processes_started'],0)
  self.assertFalse(value['all_task_Native_VAL_TEST_authorized']);self.assertNotIn('torch',sys.modules)

 def test_wait_requires_typed_actual_zero_not_pid_disappearance_or_false(self):
  with tempfile.TemporaryDirectory()as temp:
   path=Path(temp)/'exit.json'
   for bad in ({'actual_process_returncode':False,'actual_wait_completed':True},{'actual_process_returncode':0,'actual_wait_completed':False},{'pid_absent':True},{'actual_process_returncode':-15,'actual_wait_completed':True}):
    save(path,bad)
    with self.assertRaises(ValueError):m.validate_e_exit(path)
   save(path,{'actual_process_returncode':0,'actual_wait_completed':True});self.assertEqual(m.validate_e_exit(path)['actual_process_returncode'],0)

 def test_CF_final_admission_only_flags_can_change_not_cube_source_or_hand(self):
  config=m.load_config(CONFIG)
  with tempfile.TemporaryDirectory(dir=ROOT/'config/framework')as temp:
   temp=Path(temp);plan=m.read(m.checked_pointer(config['CF_prepared_snapshot']))
   plan.update(preparation_only=False,execution_authorized=True,capture_ready=True,required_pending=[])
   pp=temp/'final.json';save(pp,plan)
   admission=dict(schema=m.CF_ADMISSION_SCHEMA,execution_authorized=True,capture_plan=m.pointer(pp),prepared_plan=config['CF_prepared_snapshot'],output_root=config['CF_output_root'])
   ap=temp/'admission.json';save(ap,admission)
   self.assertEqual(m.validate_CF_admission(config,ap)[1]['hand_pad_radius_cap_m'],.12)
   for mutation in ('seed','hand','recipe','role'):
    wrong=deepcopy(plan)
    if mutation=='seed':wrong['parents'][0]['seed']+=1
    elif mutation=='hand':wrong['hand_pad_radius_cap_m']=.16
    elif mutation=='recipe':wrong['parents'][0]['shots'][1]['requested_distance_from_public_sphere_m']=.24
    else:wrong['parents'][0]['role']='reserved_fresh_centre_VAL'
    save(pp,wrong);save(ap,dict(admission,capture_plan=m.pointer(pp)))
    with self.assertRaises(ValueError):m.validate_CF_admission(config,ap)

 def test_QA_needs_real_visual_approval_and_exact_report_figure_plan(self):
  with tempfile.TemporaryDirectory()as temp:
   root=Path(temp);qa=root/'QA';qa.mkdir();config=dict(CF_prepared_snapshot={})
   prepared={'parents':[{'parent_id':'fixed_first_fit'}]};save(root/'prepared.json',prepared);config['CF_prepared_snapshot']=m.pointer(root/'prepared.json')
   save(qa/'report.json',dict(complete=True,role='fit',parent_id='fixed_first_fit',frames=[{}]*13));(qa/'RGBD_owner_dual_crop_QA.png').write_bytes(b'explicit synthetic PNG pointer only')
   final={'path':'fixed-final','sha256':'a'*64};good=dict(schema=m.QA_ADMISSION_SCHEMA,approved=True,visually_inspected=True,capture_plan=final,report=m.pointer(qa/'report.json'),figure=m.pointer(qa/'RGBD_owner_dual_crop_QA.png'))
   ap=root/'QA_admission.json';save(ap,good);m.validate_QA_admission(config,ap,qa,final)
   for key in ('approved','visually_inspected'):
    save(ap,dict(good,**{key:False}))
    with self.assertRaises(ValueError):m.validate_QA_admission(config,ap,qa,final)
   save(ap,dict(good,capture_plan={'path':'different','sha256':'b'*64}))
   with self.assertRaises(ValueError):m.validate_QA_admission(config,ap,qa,final)

 def test_STOP_wait_refuses_and_driver_lock_is_unique_not_resource_lock(self):
  with tempfile.TemporaryDirectory()as temp:
   root=Path(temp);(root/'STOP').write_text('Root STOP')
   config=dict(output_root=str(root),poll_seconds=.001,wait_timeout_seconds=.01)
   with self.assertRaises(InterruptedError):m.wait_file(config,root/'never')
   lock=root/'independent.lock'
   with m.driver_lock(lock):
    with self.assertRaises(BlockingIOError):
     with m.driver_lock(lock):pass
  text=(ROOT/'scripts/run_large_network_training_continuation_20261104_v1.py').read_text()
  self.assertNotIn('--pipeline-owned-locks',text)
  self.assertNotIn("open('fixed_TRAIN104",text)
  self.assertEqual(len(m.STAGES),12);self.assertEqual(m.STAGES[-2:],('centre_reserved_predict104','centre_reserved_score104'))

 def toy_config(self,root,timeout=2):
  return dict(python=sys.executable,output_root=str(root),CF_output_root=str(root/'CFnever'),poll_seconds=.01,stop_grace_seconds=.02,stage_timeout_seconds={'toy':timeout})

 def test_actual_stdlib_child_wait_receipt_success_and_fresh_owner(self):
  with tempfile.TemporaryDirectory()as temp:
   root=Path(temp);config=self.toy_config(root)
   with patch.object(m,'verify_sources',return_value={'explicit':'synthetic'}),patch.object(m,'assert_running'):
    value=m.run_stage(config,'toy',['-c','print("toy-public-only")'],cpu=True)
    self.assertEqual(value['actual_process_returncode'],0);self.assertTrue(value['actual_wait_completed']);self.assertTrue(value['source_unchanged']);self.assertEqual(value['unreaped_descendants'],[])
    with self.assertRaises(FileExistsError):m.run_stage(config,'toy',['-c','pass'],cpu=True)
   self.assertTrue((root/'owners/toy/stdout.log').is_file())

 def test_child_failure_and_timeout_always_actual_wait_no_retry(self):
  for timeout,program,expected in ((2,'raise SystemExit(7)',7),(.01,'import time;time.sleep(10)',-15)):
   with tempfile.TemporaryDirectory()as temp:
    root=Path(temp);config=self.toy_config(root,timeout)
    with patch.object(m,'verify_sources',return_value={'explicit':'synthetic'}),patch.object(m,'assert_running'):
     with self.assertRaises(RuntimeError):m.run_stage(config,'toy',['-c',program],cpu=True)
    value=m.read(root/'owners/toy/actual_exit.json');self.assertTrue(value['actual_wait_completed']);self.assertEqual(value['actual_process_returncode'],expected);self.assertEqual(value['unreaped_descendants'],[])
    self.assertEqual(len(list((root/'owners').iterdir())),1)

 def test_adopted_grandchild_is_terminated_and_actually_reaped_on_abort(self):
  m.enable_subreaper()
  with tempfile.TemporaryDirectory()as temp:
   root=Path(temp);config=self.toy_config(root)
   program='import os,time;pid=os.fork();time.sleep(10) if pid==0 else None'
   with patch.object(m,'verify_sources',return_value={'explicit':'synthetic'}),patch.object(m,'assert_running'):
    with self.assertRaisesRegex(RuntimeError,'live descendants'):m.run_stage(config,'toy',['-c',program],cpu=True)
   value=m.read(root/'owners/toy/actual_exit.json');self.assertTrue(value['actual_wait_completed']);self.assertEqual(value['actual_process_returncode'],0);self.assertEqual(value['unreaped_descendants'],[]);self.assertGreaterEqual(len(value['adopted_descendants_reaped']),1)


if __name__=='__main__':unittest.main()
