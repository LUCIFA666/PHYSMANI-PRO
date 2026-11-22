"""Pure two-part predecessor metadata plus real temp-path mutex contracts; no NN/GT/Native."""
import ast
import contextlib
from copy import deepcopy
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import subprocess
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('e2deferred3',ROOT/'scripts/run_semantic_task_e_v2_deferred_20261104_v3.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
def save(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x))
def done(stage,rc=0):return dict(stage=stage,actual_process_returncode=rc,actual_wait_completed=True,source_unchanged=True,error=None,stop_reason=None,unreaped_descendants=[])

class TwoPartProofContracts(unittest.TestCase):
 def test_default_no_model_or_child(self):
  with patch.object(m.subprocess,'Popen',side_effect=AssertionError('no child')),contextlib.redirect_stdout(io.StringIO())as output:self.assertEqual(m.main([]),0)
  result=json.loads(output.getvalue());self.assertTrue(result['preparation_only']);self.assertEqual(result['child_processes_started'],0);self.assertNotIn('torch',sys.modules)

 def fixture(self,root):
  old=root/'old';new=root/'new';cf=root/'CF';source={'explicit.py':'a'*64};mp=root/'map.json';save(mp,{'source_sha256':source});mapref=m.pointer(mp)
  centreplan=root/'centreplan.json';save(centreplan,{});oldplan=root/'oldplan.json';save(oldplan,{})
  oldrows=[]
  for name in m.base.STAGES[:6]:
   row=done(name);oldrows.append(row);save(old/'owners'/name/'actual_exit.json',row);save(old/'owners'/name/'request.json',dict(stage=name,source_sha256=source,driver_holds_resource_locks=False))
  ledger=old/'ledger.jsonl';ledger.parent.mkdir(exist_ok=True);ledger.write_text(''.join(json.dumps(r)+'\n'for r in oldrows))
  failed=old/'owners/centre_GPU_preflight/actual_exit.json';save(failed,done('centre_GPU_preflight',1))
  previous=root/'Rootprevious.json';save(previous,dict(actual_process_returncode=1,actual_wait_completed=True,all40_capture_completed=True,all520_compiled=True,fit_optimizer_steps=0,stage_ledger=m.pointer(ledger),failed_child=m.pointer(failed)))
  admission={'capture_plan':{'path':'explicit-final','sha256':'b'*64}};parents=[]
  for i in range(40):
   ident='fixed%02d'%i;parents.append({'parent_id':ident});save(cf/(ident+'.owner.actual_exit.json'),dict(actual_process_returncode=0,actual_wait_completed=True,source_unchanged=True,parent_id=ident));save(cf/ident/'result.json',dict(native_closed=True,source_unchanged=True))
  manifest=root/'manifest.json';save(manifest,{'explicit_fixture':True})
  selected=new/'artifacts/centre_fit/selected.pt';selected.parent.mkdir(parents=True);selected.write_bytes(b'non-model fixture bytes')
  save(new/'artifacts/centre_fit/fit_status.json',dict(training_complete=True,epochs_completed=40,actual_optimizer_steps=520,selected_checkpoint=m.pointer(selected)))
  completion=dict(complete=True,completed_stages=list(m.CENTRE_STAGES),previous_root_exit=m.pointer(previous),current_network_end_to_end_ready=False,Native_VAL_TEST_started=False,compiled_manifest=m.pointer(manifest),selected_centre=m.pointer(selected),finished_utc='first')
  save(new/'completion.json',completion);save(new/'terminal_status.json',dict(completion,finished_utc='later'))
  save(new/'request.json',dict(config=m.pointer(centreplan),source_map=mapref));newrows=[]
  for name in m.CENTRE_STAGES:
   row=done(name);newrows.append(row);save(new/'owners'/name/'actual_exit.json',row);save(new/'owners'/name/'request.json',dict(stage=name,source_sha256=source,driver_holds_resource_locks=False))
  (new/'stage_ledger.jsonl').write_text(''.join(json.dumps(r)+'\n'for r in newrows))
  e1=root/'e1.json';save(e1,dict(actual_process_returncode=0,actual_wait_completed=True,source_unchanged=True))
  plan=dict(centre_config=m.pointer(centreplan),previous_root_exit=m.pointer(previous),prior_config=m.pointer(oldplan),centre_method_sources=mapref,centre_continuation_root=str(new),prior_continuation_root=str(old),CF_output_root=str(cf),e1_actual_exit=m.pointer(e1))
  cfg=dict(manifest=m.pointer(manifest));helper=SimpleNamespace(load_config=lambda *a:cfg,validate_previous=lambda c:dict(manifest=m.pointer(manifest),all40_closed=True))
  return plan,helper,admission,{'parents':parents},source

 def validate(self,fixture):
  plan,helper,admission,cfplan,source=fixture
  with patch.object(m,'centre_helper',return_value=helper),patch.object(m.base,'verify_sources',return_value=source),patch.object(m.base,'load_config',return_value={'CF_final_admission':'explicit'}),patch.object(m.base,'validate_CF_admission',return_value=(admission,cfplan)):
   return m.validate_prior(plan)

 def test_old_failure_preserved_then_exact_new6_success_and_native40(self):
  with tempfile.TemporaryDirectory()as temp:
   fixture=self.fixture(Path(temp));facts=self.validate(fixture)
   self.assertEqual(len(facts['actual_stage_owners']),12);self.assertEqual(len(facts['CF_native_owners']),40);self.assertFalse(facts['successor_Root6691_parent_actual_wait_claimed'])
   failed=m.read(facts['preserved_failed_child_not_success']['path']);self.assertEqual(failed['actual_process_returncode'],1)
   self.assertEqual(len(facts['old_six_success_stages']),6);self.assertEqual(len(facts['new_six_success_stages']),6)

 def test_wrong_old_failure_or_missing_new_success_never_authorizes(self):
  with tempfile.TemporaryDirectory()as temp:
   fixture=self.fixture(Path(temp));plan=fixture[0];prev=Path(plan['previous_root_exit']['path']);original=m.read(prev)
   for field,value in [('actual_process_returncode',0),('actual_process_returncode',True),('fit_optimizer_steps',1)]:
    save(prev,dict(original,**{field:value}));plan['previous_root_exit']=m.pointer(prev)
    with self.assertRaises(ValueError):self.validate(fixture)
   save(prev,original);plan['previous_root_exit']=m.pointer(prev)
   # The new chain still has six owners but one is not actually waited.
   owner=Path(plan['centre_continuation_root'])/'owners/centre_fit40/actual_exit.json';r=m.read(owner);save(owner,dict(r,actual_wait_completed=False))
   with self.assertRaises(ValueError):self.validate(fixture)

 def test_no_new_driver_wait_inference_and_no_capture_relaunch(self):
  source=(ROOT/'scripts/run_semantic_task_e_v2_deferred_20261104_v3.py').read_text()
  self.assertNotIn('collect_current_cube',source);self.assertNotIn('base.execute(',source)
  with self.assertRaises(ValueError):m.execute(m.DEFAULT_PLAN,'0'*64)


class CorrectOuterResourceTests(unittest.TestCase):
 def config(self,root):return dict(output_root=str(root/'run'),CF_output_root=str(root/'CF'),python=sys.executable,poll_seconds=.01,stop_grace_seconds=.02,resource_wait_timeout_seconds=1,stage_timeout_seconds={'toy':2})

 def probe(self,paths):
  program="import sys,json,fcntl;blocked=0;handles=[]\nfor p in json.loads(sys.argv[1]):\n h=open(p,'a+');handles.append(h)\n try:fcntl.flock(h.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)\n except BlockingIOError:blocked+=1\nprint(blocked)"
  r=subprocess.run([sys.executable,'-B','-c',program,json.dumps([str(p)for p in paths])],capture_output=True,text=True)
  self.assertEqual(r.returncode,0,r.stderr);return int(r.stdout)

 def test_real_same_path_blocks_wrong_old_directory_does_not_and_release_works(self):
  with tempfile.TemporaryDirectory()as temp:
   root=Path(temp);cfg=self.config(root)
   with patch.object(m,'ROOT',root),patch.object(m,'assert_running'):
    correct=m.resource_paths();wrong=tuple(root/'runtime/framework/local_mpc_20261102'/n for n in m.RESOURCE_NAMES)
    for p in wrong:p.parent.mkdir(parents=True,exist_ok=True)
    with m.correct_stage_resource_locks(cfg,'toy')as token:
     self.assertTrue(token['actual_outer_locks_held']);self.assertEqual(self.probe(correct),2);self.assertEqual(self.probe(wrong),0)
     self.assertEqual([p.parent for p in correct],[root/'runtime/framework']*2)
    self.assertEqual(self.probe(correct),0)
   self.assertTrue(m.read(root/'run/resource_ownership/toy/released.json')['locks_released'])

 def test_actual_child_sees_locks_and_bypass_receipts_true_until_after_wait(self):
  with tempfile.TemporaryDirectory()as temp:
   root=Path(temp);cfg=self.config(root);program="import sys,json,fcntl;assert '--pipeline-owned-locks' in sys.argv;blocked=0;handles=[]\nfor p in json.loads(sys.argv[1]):\n h=open(p,'a+');handles.append(h)\n try:fcntl.flock(h.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)\n except BlockingIOError:blocked+=1\nassert blocked==2;print('actual-correct-outer-locks-held-and-bypass-argv')"
   with patch.object(m,'ROOT',root),patch.object(m,'assert_running'),patch.object(m,'verify_sources',return_value={'explicit':'toy'}):
    with m.correct_stage_resource_locks(cfg,'toy')as token:
     result=m.run_gpu_stage(cfg,'toy',['-c',program,json.dumps([str(p)for p in m.resource_paths()]),'--pipeline-owned-locks'],token)
     self.assertEqual(result['actual_process_returncode'],0);self.assertTrue(result['actual_wait_completed']);self.assertTrue(result['correct_outer_resource_locks_held_through_actual_wait']);self.assertEqual(self.probe(m.resource_paths()),2)
    self.assertEqual(self.probe(m.resource_paths()),0)
   request=m.read(root/'run/owners/toy/request.json');released=m.read(root/'run/resource_ownership/toy/released.json')
   self.assertTrue(request['driver_holds_resource_locks']);self.assertTrue(request['child_pipeline_owned_locks']);self.assertTrue(released['actual_child_wait_completed']);self.assertEqual(released['unreaped_descendants'],[])

 def test_child_failure_waited_before_resource_release(self):
  with tempfile.TemporaryDirectory()as temp:
   root=Path(temp);cfg=self.config(root)
   with patch.object(m,'ROOT',root),patch.object(m,'assert_running'),patch.object(m,'verify_sources',return_value={'explicit':'toy'}):
    with self.assertRaises(RuntimeError):
     with m.correct_stage_resource_locks(cfg,'toy')as token:m.run_gpu_stage(cfg,'toy',['-c','raise SystemExit(7)','--pipeline-owned-locks'],token)
    self.assertEqual(self.probe(m.resource_paths()),0)
   value=m.read(root/'run/owners/toy/actual_exit.json');self.assertEqual(value['actual_process_returncode'],7);self.assertTrue(value['actual_wait_completed']);self.assertTrue(m.read(root/'run/resource_ownership/toy/released.json')['actual_child_wait_completed'])

 def test_fabricated_ownership_token_rejected_before_child(self):
  with patch.object(m.subprocess,'Popen',side_effect=AssertionError('no child')):
   with self.assertRaisesRegex(ValueError,'Live exact'):m.run_gpu_stage({},'toy',[],{})

 def test_waiting_for_prior_never_acquires_GPU_resource_locks(self):
  with tempfile.TemporaryDirectory()as temp:
   root=Path(temp);cfg=dict(output_root=str(root/'newrun'),method_sources={'path':'synthetic','sha256':'x'},prior_continuation_root=str(root/'prior'))
   with patch.object(m,'load_plan',return_value=cfg),patch.object(m,'LOCK',root/'private-driver.lock'),patch.object(m,'assert_running'),patch.object(m.base,'enable_subreaper'),patch.object(m,'pointer',return_value={'path':'explicit','sha256':'a'*64}),patch.object(m,'wait_for_prior',side_effect=TimeoutError('prior-not-complete')),patch.object(m,'correct_stage_resource_locks',side_effect=AssertionError('must not lock while waiting'))as locks:
    with self.assertRaises(TimeoutError):m.execute('explicit','a'*64)
    locks.assert_not_called()
   self.assertTrue((root/'newrun/failure.json').exists());self.assertFalse((root/'newrun/resource_ownership').exists())


if __name__=='__main__':unittest.main()
