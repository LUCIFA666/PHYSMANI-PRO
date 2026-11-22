"""Actual v2 check -> host --execute lifecycle with Popen mocked, no Native."""
import ast
from contextlib import redirect_stdout
import importlib.util
import io,json,signal,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from our_mani.local_mpc_20261103 import public_prefix_static_collision_replay_20261103_v2 as r
ROOT=r.ROOT;REQ=ROOT/'artifacts/framework/local_mpc_20261102/V67_hold_C_first_action_static_collision_request_v1/request.json'
SPEC=importlib.util.spec_from_file_location('static_replay_host_v2',ROOT/'scripts/launch_public_prefix_C_static_collision_20261103_v2.py');host=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(host)


class Child:
 def __init__(self,live=False):self.pid=123456789;self.live=live;self.wait_calls=[]
 def poll(self):return None if self.live else 0
 def wait(self,timeout=None):self.wait_calls.append(timeout);self.live=False;return 0


class HostLifecycleV2(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.temp=tempfile.TemporaryDirectory(prefix='replay_v2_host_cpu_',dir=ROOT/'artifacts')
  cls.base=Path(cls.temp.name);cls.freeze=cls.base/'execution_source_freeze.json'
  r.write(cls.freeze,r.check(REQ,collect_sources=True));cls.digest=r.sha(cls.freeze)
 @classmethod
 def tearDownClass(cls):cls.temp.cleanup()
 def arguments(self,case):
  return['--request',str(REQ),'--anchor-id',r.ANCHORS[0],'--output',str(case),'--source-freeze',str(self.freeze),'--source-freeze-sha256',self.digest,'--execute']

 def test_real_check_returns_pointer_and_real_host_execute_metadata_then_wait(self):
  admission=r.check(REQ,r.ANCHORS[0],self.freeze,self.digest)
  self.assertEqual(admission['source_freeze'],dict(path=str(self.freeze.resolve()),sha256=self.digest))
  family=self.base/'success';case=family/'case';child=Child()
  with patch.object(host,'FAMILY',family),patch.object(host.subprocess,'Popen',return_value=child)as popen,patch.object(host.os,'killpg',side_effect=AssertionError('No signal on success')),redirect_stdout(io.StringIO()):
   self.assertEqual(host.main(self.arguments(case)),0)
  self.assertEqual(popen.call_count,1);self.assertTrue(child.wait_calls)
  owner=json.loads(case.with_name('case.owner.json').read_text());closed=json.loads(case.with_name('case.owner.actual_exit.json').read_text())
  self.assertEqual(owner['source_freeze'],admission['source_freeze']);self.assertEqual(owner['anchor_id'],r.ANCHORS[0]);self.assertFalse(owner['C_command_executed'])
  self.assertIs(closed['actual_wait_completed'],True);self.assertEqual(closed['actual_process_returncode'],0);self.assertIs(closed['source_unchanged'],True)
  self.assertFalse(case.exists()) # Mock child never invokes worker or Native.

 def test_post_Popen_metadata_exception_still_terminates_and_reaps_mock_child(self):
  family=self.base/'cleanup';case=family/'case';child=Child(live=True);original=host.write_new
  def writing(path,value):
   if str(path).endswith('.owner.json'):raise KeyError('intentional owner metadata failure')
   return original(path,value)
  def killed(pid,sig):self.assertEqual(pid,child.pid);self.assertEqual(sig,signal.SIGTERM);child.live=False
  with patch.object(host,'FAMILY',family),patch.object(host.subprocess,'Popen',return_value=child),patch.object(host,'write_new',side_effect=writing),patch.object(host.os,'killpg',side_effect=killed)as kill,redirect_stdout(io.StringIO()):
   with self.assertRaises(KeyError):host.main(self.arguments(case))
  self.assertEqual(kill.call_count,1);self.assertTrue(child.wait_calls)
  closed=json.loads(case.with_name('case.owner.actual_exit.json').read_text());self.assertIs(closed['actual_wait_completed'],True);self.assertIs(closed['source_unchanged'],False);self.assertIs(closed['terminated_during_owner_cleanup'],True)

 def test_v2_replay_physics_helper_and_tolerances_AST_exact_to_frozen_v1(self):
  old=ast.parse((ROOT/'our_mani/local_mpc_20261103/public_prefix_static_collision_replay_20261103.py').read_text());new=ast.parse(Path(r.__file__).read_text())
  names=('robot','match_robot','match_h','stop_paths','stop_requested','load_request','execute','ReplayBackend')
  def item(tree,name):return next(n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))and n.name==name)
  for name in names:self.assertEqual(ast.dump(item(old,name),include_attributes=False),ast.dump(item(new,name),include_attributes=False),name)
  oldtol=next(n for n in old.body if isinstance(n,ast.Assign)and isinstance(n.targets[0],ast.Name)and n.targets[0].id=='TOL');newtol=next(n for n in new.body if isinstance(n,ast.Assign)and isinstance(n.targets[0],ast.Name)and n.targets[0].id=='TOL')
  self.assertEqual(ast.dump(oldtol,include_attributes=False),ast.dump(newtol,include_attributes=False))

if __name__=='__main__':unittest.main()
