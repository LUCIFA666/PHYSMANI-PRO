"""Stdlib metadata gates only; no actual model, pixels, Native or GPU calls."""
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
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('deferred_e2',ROOT/'scripts/run_semantic_task_e_v2_deferred_20261104_v2.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def save(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x))
def terminal(stage):return dict(stage=stage,actual_process_returncode=0,actual_wait_completed=True,source_unchanged=True,error=None,stop_reason=None,unreaped_descendants=[])


class DeferredOwnerTests(unittest.TestCase):
 def test_default_starts_nothing_and_does_not_import_NN(self):
  with patch.object(m.base.subprocess,'Popen',side_effect=AssertionError('default cannot spawn')),contextlib.redirect_stdout(io.StringIO())as out:
   self.assertEqual(m.main([]),0)
  value=json.loads(out.getvalue());self.assertEqual(value['child_processes_started'],0);self.assertFalse(value['all_task_Native_VAL_TEST_authorized']);self.assertTrue(value['Root_parent_actual_wait_not_claimed']);self.assertNotIn('torch',sys.modules)

 def test_all12_actual_source_waits_no_boolean_returncode_or_unreaped(self):
  rows=[terminal(s)for s in m.base.STAGES];sources={'fixture.py':'a'*64}
  requests=[dict(stage=s,source_sha256=sources,driver_holds_resource_locks=False)for s in m.base.STAGES]
  m.validate_prior_stage_rows(rows,m.base.STAGES,owner_rows=deepcopy(rows),request_rows=requests,expected_sources=sources)
  for wrong in (rows[:-1],list(reversed(rows))):
   with self.assertRaises(ValueError):m.validate_prior_stage_rows(wrong,m.base.STAGES)
  for key,value in [('actual_process_returncode',False),('actual_process_returncode',1),('actual_wait_completed',False),('source_unchanged',False),('unreaped_descendants',[1])]:
   wrong=deepcopy(rows);wrong[0][key]=value
   with self.assertRaises(ValueError):m.validate_prior_stage_rows(wrong,m.base.STAGES)
  wrong=deepcopy(rows);wrong[1]['seconds']=99
  with self.assertRaises(ValueError):m.validate_prior_stage_rows(rows,m.base.STAGES,owner_rows=wrong)
  requests[0]['source_sha256']={}
  with self.assertRaises(ValueError):m.validate_prior_stage_rows(rows,m.base.STAGES,request_rows=requests,expected_sources=sources)

 def test_prior_failure_and_STOP_prevent_waiting_to_GPU(self):
  with tempfile.TemporaryDirectory()as temp:
   root=Path(temp);save(root/'failure.json',{'error':'oldCFfailed'})
   plan=dict(prior_continuation_root=str(root),wait_timeout_seconds=.1,poll_seconds=.01)
   with patch.object(m,'assert_running'):
    with self.assertRaisesRegex(ValueError,'failed'):m.wait_for_prior(plan)
   with patch.object(m.base,'assert_running',side_effect=InterruptedError('STOP')):
    with self.assertRaises(InterruptedError):m.assert_running(plan)

 def fixture(self,root):
  chain=root/'chain';cohort=root/'CF';owners=[];results=[];parents=[];manifestparents=[]
  for i in range(40):
   ident=f'fixed{i:02d}';parent={'parent_id':ident};parents.append(parent);case=cohort/ident
   owner=cohort/(ident+'.owner.actual_exit.json');result=case/'result.json'
   save(owner,dict(actual_process_returncode=0,actual_wait_completed=True,source_unchanged=True,parent_id=ident,output=str(case)))
   save(result,dict(parent_id=ident,complete=True,native_closed=True,source_unchanged=True))
   owners.append(m.pointer(owner));results.append(m.pointer(result));manifestparents.append(dict(parent_id=ident,capture=dict(owner_actual_exit=m.pointer(owner),result=m.pointer(result))))
  oldconfig=root/'oldconfig.json';save(oldconfig,{})
  sourcefile=root/'source.json';save(sourcefile,dict(source_sha256={'fixture.py':'a'*64}));sourcemap=m.pointer(sourcefile)
  e1=root/'e1.actual_exit.json';save(e1,dict(actual_process_returncode=0,actual_wait_completed=True,source_unchanged=True))
  old=dict(method_sources=sourcemap,CF_final_admission=str(root/'admission.json'),CF_output_root=str(cohort))
  cfg=dict(prior_config=m.pointer(oldconfig),prior_continuation_root=str(chain),CF_output_root=str(cohort),e1_actual_exit=m.pointer(e1))
  admission=dict(capture_plan=dict(path=str(root/'finalplan.json'),sha256='b'*64))
  manifest=root/'manifest.json';save(manifest,dict(capture_plan=admission['capture_plan'],complete=True,all_native_closed=True,all_process_exits_observed=True,source_unchanged=True,parents=manifestparents))
  selectede=root/'e1selected.pt';selectede.write_bytes(b'explicit non-model test bytes')
  selectedc=chain/'artifacts/centre_fit/selected.pt';selectedc.parent.mkdir(parents=True);selectedc.write_bytes(b'explicit non-model test bytes')
  save(chain/'artifacts/centre_fit/fit_status.json',dict(training_complete=True,epochs_completed=40,actual_optimizer_steps=520,selected_checkpoint=m.pointer(selectedc)))
  completion=dict(complete=True,completed_stages=list(m.base.STAGES),current_network_end_to_end_ready=False,model_or_runtime_qualified=False,Native_VAL_TEST_started=False,CF_admission=admission,compiled_manifest=m.pointer(manifest),e_actual_exit=m.pointer(e1),selected_e=m.pointer(selectede),selected_centre=m.pointer(selectedc))
  save(chain/'completion.json',completion);save(chain/'terminal_status.json',dict(completion,finished_utc='synthetic'))
  save(chain/'request.json',dict(config=m.pointer(oldconfig),source_map=sourcemap))
  rows=[]
  for stage in m.base.STAGES:
   row=terminal(stage);rows.append(row)
   save(chain/'owners'/stage/'actual_exit.json',row)
   save(chain/'owners'/stage/'request.json',dict(stage=stage,source_sha256={'fixture.py':'a'*64},driver_holds_resource_locks=False))
  (chain/'stage_ledger.jsonl').write_text(''.join(json.dumps(r)+'\n'for r in rows))
  return cfg,old,admission,dict(parents=parents),owners,results

 def validate_fixture(self,cfg,old,admission,cfplan):
  with patch.object(m.base,'verify_sources',return_value={'fixture.py':'a'*64}),patch.object(m.base,'load_config',return_value=old),patch.object(m.base,'validate_CF_admission',return_value=(admission,cfplan)),patch.object(m.base,'queue_audit',return_value=[{'explicit_archive_audit_stub':True}]*40):
   return m.validate_prior(cfg)

 def test_full40_parent_results_manifest_and_centre_child_wait_required(self):
  with tempfile.TemporaryDirectory()as temp:
   cfg,old,admission,cfplan,owners,results=self.fixture(Path(temp));facts=self.validate_fixture(cfg,old,admission,cfplan)
   self.assertEqual(len(facts['CF_native_owners']),40);self.assertEqual(len(facts['prior_training_owners']),2);self.assertFalse(facts['Root_parent_process_actual_wait_claimed'])
   path=Path(results[0]['path']);original=m.read(path);save(path,dict(original,native_closed=False))
   with self.assertRaises(ValueError):self.validate_fixture(cfg,old,admission,cfplan)
   save(path,original)
   exitpath=Path(cfg['prior_continuation_root'])/'owners/centre_fit40/actual_exit.json';row=m.read(exitpath);save(exitpath,dict(row,actual_wait_completed=False))
   with self.assertRaises(ValueError):self.validate_fixture(cfg,old,admission,cfplan)

 def test_authorization_factory_matches_actual_frozen_common_pure_function(self):
  # Compile ONLY the frozen stdlib metadata validator AST, not the NN module.
  with tempfile.TemporaryDirectory()as temp:
   root=Path(temp);cfg,old,admission,cfplan,owners,results=self.fixture(root);facts=self.validate_fixture(cfg,old,admission,cfplan)
   modelconfig=root/'v2.json';save(modelconfig,{})
   sourcefile=root/'v2sources.json';save(sourcefile,{'explicit_test_source':'c'*64})
   split=root/'split.json';save(split,{})
   plan=dict(v2_method_sources=m.pointer(sourcefile),split_manifest=m.pointer(split),v2_config=m.pointer(modelconfig),_config_path='explicit-test-plan',_config_sha256='d'*64)
   value=m.build_authorization(plan,facts);self.assertFalse(value['Root_parent_process_actual_wait_claimed']);self.assertEqual(len(value['native_lifecycle']['owners']),40)
   auth=root/'auth.json';save(auth,value)
   tree=ast.parse((ROOT/'our_mani/semantic_task_e_large_20261104_v2/common.py').read_text());function=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='root_authorization')
   namespace=dict(Path=Path,sha=m.sha,read=m.read,pointer=m.pointer,checked_pointer=m.checked_pointer,COMPONENT='semantic_task_e8_only',CONFIG=modelconfig)
   exec(compile(ast.Module(body=[function],type_ignores=[]),'frozen_v2_metadata_only','exec'),namespace)
   self.assertEqual(namespace['root_authorization'](auth,m.sha(auth),split,{'explicit_test_source':'c'*64}),m.pointer(auth))
   falsewait=Path(owners[0]['path']);record=m.read(falsewait);save(falsewait,dict(record,actual_process_returncode=False))
   value['native_lifecycle']['owners'][0]=m.pointer(falsewait);save(auth,value)
   with self.assertRaises(ValueError):namespace['root_authorization'](auth,m.sha(auth),split,{'explicit_test_source':'c'*64})

 def test_external_plan_SHA_before_output_and_only_isolated_driver_lock(self):
  with self.assertRaises(ValueError):m.execute(m.DEFAULT_PLAN,'0'*64)
  source=(ROOT/'scripts/run_semantic_task_e_v2_deferred_20261104_v2.py').read_text()
  self.assertIn('--pipeline-owned-locks',source);self.assertNotIn('native=True',source);self.assertNotIn('base.execute(',source)
  self.assertNotEqual(m.LOCK,m.base.DRIVER_LOCK);self.assertEqual(m.STAGES,('e2_GPU_preflight','e2_fit10','e2_selected646_evaluation'))


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
