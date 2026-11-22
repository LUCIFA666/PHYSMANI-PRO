"""Independent six-reference worker preparation; zero Native/model forwards."""
import ast
from contextlib import redirect_stdout
from copy import deepcopy
import importlib.util,io,json,signal,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from our_mani.local_mpc_20261103 import position_lazy_v71_worker as w
ROOT=w.ROOT
SPEC=importlib.util.spec_from_file_location('position_lazy_v71_host',ROOT/'scripts/launch_local_mpc_position_lazy_v71_20261103.py')
host=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(host)
QUEUE_SPEC=importlib.util.spec_from_file_location('position_lazy_v71_queue',ROOT/'scripts/run_position_lazy_v71_fixed_development_20261103.py')
queue=importlib.util.module_from_spec(QUEUE_SPEC);QUEUE_SPEC.loader.exec_module(queue)


class MockChild:
 def __init__(self,live=False):self.pid=123456789;self.live=live;self.wait_calls=[]
 def poll(self):return None if self.live else 0
 def wait(self,timeout=None):self.wait_calls.append(timeout);self.live=False;return 0


class V71Preparation(unittest.TestCase):
 def test_default_all_six_declared_references_are_not_ready_and_never_native(self):
  plan=json.loads(w.CANARY_PLAN.read_text());self.assertEqual(tuple(j['id']for j in plan['jobs']),w.CASE_IDS);self.assertEqual(len(w.CASE_IDS),6)
  with patch.object(w.original,'make_backend',side_effect=AssertionError('No Native')):
   for job in plan['jobs']:
    c=json.loads((ROOT/job['config']).read_text());r=w.check_config(ROOT/job['config'],job['scene']);self.assertFalse(r['ready']);self.assertEqual(r['model_forward_calls'],0)
    self.assertEqual(c['position_refiner_checkpoint_sha256'],'ee079d1f0ea302f65ed4123c5137328005f582f5347e282497dc2d2e07dc0884')
    oldjob,oldpath=w._reference(c);self.assertEqual(oldjob['id'],job['id']);expected='fixed6'
    self.assertEqual(c['v71_development_reference']['plan'],str(w.REFERENCE_PLANS[expected].relative_to(ROOT)))
    self.assertIs(c['v71_preparation_only'],True)

 def test_wrong_newflag_and_behavior_changes_reject_before_native_or_mkdir(self):
  job=json.loads(w.CANARY_PLAN.read_text())['jobs'][0];base=json.loads((ROOT/job['config']).read_text())
  with tempfile.TemporaryDirectory(prefix='v71_contract_',dir=ROOT/'artifacts')as tmp:
   path=Path(tmp)/'config.json'
   for key,value in [('candidate_lazy_free_v71',False),('candidate_lazy_free_v71',1),('candidate_lazy_free_v71','true'),('candidate_response_seed_v70',False),('candidate_response_seed_v70',1),('candidate_response_seed_v70','true'),('candidate_far_pixel_centre_v70',False),('candidate_far_pixel_centre_v70',1),('candidate_response_seed_v69',False),('candidate_response_seed_v68',False),('candidate_response_seed_v68',1),('physics_dt',.1),('cem_sequences_total',257),('probe_terminal_at_fixed_deadline',False),('position_refiner_checkpoint_sha256','0'*64)]:
    bad=deepcopy(base);bad[key]=value;path.write_text(json.dumps(bad))
    with self.subTest(key=key,value=value),patch.object(w.original,'make_backend',side_effect=AssertionError('No Native')):
     with self.assertRaises(ValueError):w.check_config(path,job['scene'],require_ready=True)

 def test_common_source_builder_unions_all_six_and_negative_specific_source(self):
  plan=json.loads(w.CANARY_PLAN.read_text());negative='our_mani/local_mpc_control_20261103/no_proxy_attachment.py'
  calls=[]
  def sources(path,mode):
   calls.append(Path(path));return {negative:w.sha(ROOT/negative)}if Path(path).stem=='static_no_proxy'else{}
  fake=SimpleNamespace(lazy_free_producer_source_map=lambda:{})
  with patch.object(w,'freeze_sources',side_effect=sources):
   result=w._source_closure(ROOT/plan['jobs'][0]['config'],fake,SimpleNamespace(model_sources=lambda:{}))
  self.assertIn(negative,result);self.assertTrue(all(j['config']in result for j in plan['jobs']))
  self.assertTrue(all(str(p.relative_to(ROOT))in result for p in w.REFERENCE_PLANS.values()))
  self.assertTrue(all(name in result for name in w.OWN_FILES));self.assertEqual(len(calls),7)

 def test_actual_execute_import_route_is_the_new_producer_and_old_execution_math(self):
  tree=ast.parse(Path(w.__file__).read_text());old=ast.parse((ROOT/'our_mani/local_mpc_20261103/position_response_v70_worker.py').read_text())
  fn=lambda t:next(n for n in t.body if isinstance(n,ast.FunctionDef)and n.name=='execute')
  new=fn(tree);imports=[n for n in ast.walk(new)if isinstance(n,ast.ImportFrom)]
  selected=next(n for n in imports if n.module=='local_mpc_control_20261103.public_candidate_lazy_free_v71')
  self.assertEqual([a.name for a in selected.names],['ObservedPublicCandidateLazyFreeV71Backend','run_public_candidate_lazy_free_v71'])
  mapping={'public_candidate_lazy_free_v71':'public_candidate_response_seed_v70',
   'ObservedPublicCandidateLazyFreeV71Backend':'ObservedPublicCandidateResponseSeedV70Backend',
   'run_public_candidate_lazy_free_v71':'run_public_candidate_response_seed_v70',
   'p1_public_lazy_free_v71':'p1_public_response_seed_v70'}
  def normalize(value):
   for a,b in mapping.items():value=value.replace(a,b)
   return value
  class Normalize(ast.NodeTransformer):
   def visit_ImportFrom(self,node):
    node.module=normalize(node.module)
    for a in node.names:a.name=normalize(a.name)
    return node
   def visit_Name(self,node):node.id=normalize(node.id);return node
   def visit_Constant(self,node):
    if isinstance(node.value,str):node.value=normalize(node.value)
    return node
  self.assertEqual(ast.dump(Normalize().visit(new),include_attributes=False),ast.dump(fn(old),include_attributes=False))



class V71HostLifecycle(unittest.TestCase):
 """Mock admission and child only; this never fabricates a runtime certificate."""
 def lifecycle(self,tmp,child,writer=None):
  base=Path(tmp);case=base/'queue'/'case';freeze=base/'freeze.json';freeze.write_text('{}\n')
  job=json.loads(w.CANARY_PLAN.read_text())['jobs'][0]
  admission={'ready':True,'case_id':job['id'],'source_freeze':{'path':str(freeze.resolve()),'sha256':host.sha(freeze)}}
  argv=['--config',str(ROOT/job['config']),'--scene',job['scene'],'--output',str(case),
        '--source-freeze',str(freeze),'--source-freeze-sha256',host.sha(freeze),'--execute']
  return case,admission,argv

 def test_host_execute_serializes_check_pointer_and_actually_waits_for_mock_child(self):
  with tempfile.TemporaryDirectory(prefix='v71_host_cpu_',dir=ROOT/'artifacts')as tmp:
   child=MockChild();case,admission,argv=self.lifecycle(tmp,child)
   with patch.object(host,'FAMILY',Path(tmp)),patch.object(w,'check_config',return_value=admission)as check,patch.object(host.subprocess,'Popen',return_value=child)as popen,patch.object(host.os,'killpg',side_effect=AssertionError('No signal on success')),redirect_stdout(io.StringIO()):
    self.assertEqual(host.main(argv),0)
   self.assertIs(check.call_args.kwargs['require_ready'],True);self.assertEqual(popen.call_count,1);self.assertTrue(child.wait_calls)
   owner=json.loads(case.with_name('case.owner.json').read_text());closed=json.loads(case.with_name('case.owner.actual_exit.json').read_text())
   self.assertEqual(owner['source_freeze'],admission['source_freeze']);self.assertEqual(owner['case_id'],admission['case_id'])
   self.assertIs(closed['actual_wait_completed'],True);self.assertEqual(closed['actual_process_returncode'],0);self.assertIs(closed['source_unchanged'],True)
   self.assertFalse(case.exists())

 def test_post_popen_metadata_failure_terminates_and_reaps_mock_child(self):
  with tempfile.TemporaryDirectory(prefix='v71_host_cleanup_',dir=ROOT/'artifacts')as tmp:
   child=MockChild(live=True);case,admission,argv=self.lifecycle(tmp,child);original=host.write_new
   def writing(path,value):
    if str(path).endswith('.owner.json'):raise KeyError('synthetic metadata failure')
    return original(path,value)
   def killed(pid,sig):self.assertEqual(pid,child.pid);self.assertEqual(sig,signal.SIGTERM);child.live=False
   with patch.object(host,'FAMILY',Path(tmp)),patch.object(w,'check_config',return_value=admission),patch.object(host.subprocess,'Popen',return_value=child),patch.object(host,'write_new',side_effect=writing),patch.object(host.os,'killpg',side_effect=killed)as kill,redirect_stdout(io.StringIO()):
    with self.assertRaises(KeyError):host.main(argv)
   self.assertEqual(kill.call_count,1);self.assertTrue(child.wait_calls)
   closed=json.loads(case.with_name('case.owner.actual_exit.json').read_text());self.assertIs(closed['actual_wait_completed'],True);self.assertIs(closed['source_unchanged'],False);self.assertIs(closed['terminated_during_owner_cleanup'],True)


class V71QueuePreparation(unittest.TestCase):
 def test_plan_preparation_and_authorization_block_before_sources_or_child(self):
  plan=json.loads(w.CANARY_PLAN.read_text());reports=[{'ready':True}for _ in plan['jobs']]
  with patch.object(queue,'check_plan',return_value=(plan,reports)),patch.object(queue,'verify_sources',side_effect=AssertionError('No execution source admission')),patch.object(queue.subprocess,'run',side_effect=AssertionError('No child')),redirect_stdout(io.StringIO()):
   with self.assertRaises(SystemExit):queue.main(['--execute','--source-freeze','not-opened.json','--source-freeze-sha256','0'*64])

 def test_parent_STOP_is_seen_before_any_case_admission_or_child(self):
  plan=json.loads(w.CANARY_PLAN.read_text())
  with tempfile.TemporaryDirectory(prefix='v71_queue_stop_',dir=ROOT/'artifacts')as tmp:
   base=Path(tmp);queue_parent=base/'runtime'/'fresh';queue_parent.mkdir(parents=True);(queue_parent/'STOP').write_text('administrative\n')
   for job in plan['jobs']:job['path']='runtime/fresh/'+job['id']
   path=base/'plan.json';path.write_text(json.dumps(plan))
   with patch.object(queue,'ROOT',base),patch.object(queue,'PLAN',path),patch.object(w,'check_config',side_effect=AssertionError('STOP precedes certificate admission')):
    with self.assertRaises(ValueError):queue.check_plan(SimpleNamespace(source_freeze=None,source_freeze_sha256=None))

if __name__=='__main__':unittest.main()
