from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
import tempfile,unittest,hashlib,os
from our_mani.multitask import frozen_input_edges_v463 as module
from our_mani.multitask.protocol import MultitaskProtocolError
from our_mani.multitask.input_stat_cache_v79 import ROOT

class Tests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory(dir=ROOT/'runtime/framework/tmp');self.root=Path(self.tmp.name)
  self.path=self.root/'data';self.path.write_bytes(b'data');self.sha=hashlib.sha256(b'data').hexdigest();self.files={'data':self.sha}
 def tearDown(self):self.tmp.cleanup()
 def test_unrelated_outputs_do_not_invalidate_declared_edges(self):
  index=module.FrozenInputIndex(self.root,self.files)
  for i in range(10):
   (self.root/str(i)).mkdir();self.assertEqual(index.checked(self.path),self.sha)
 def test_symlink_retarget_still_invalidates_and_rejects_duplicate(self):
  other=self.root/'other';other.write_bytes(b'else');link=self.root/'alias';link.symlink_to(other)
  files=dict(self.files,alias=hashlib.sha256(b'else').hexdigest());index=module.FrozenInputIndex(self.root,files)
  link.unlink();link.symlink_to(self.path);self.assertFalse(index.valid())
  with self.assertRaisesRegex(MultitaskProtocolError,'conflicting'):module.FrozenInputIndex(self.root,files).checked(self.path)
 def test_tracked_directory_replacement_invalidates(self):
  folder=self.root/'folder';folder.mkdir();p=folder/'entry';p.write_bytes(b'data')
  index=module.FrozenInputIndex(self.root,{'folder/entry':self.sha});folder.rename(self.root/'old');folder.mkdir();(folder/'entry').write_bytes(b'data')
  self.assertFalse(index.valid())
 def test_same_size_bytes_restored_mtime_and_atomic_file_replacement_still_detected(self):
  index=module.FrozenInputIndex(self.root,self.files);index.checked(self.path);s=self.path.stat()
  self.path.write_bytes(b'xxxx');os.utime(self.path,ns=(s.st_atime_ns,s.st_mtime_ns))
  with self.assertRaises(MultitaskProtocolError):index.checked(self.path)
  new=self.root/'replacement';new.write_bytes(b'data');new.replace(self.path);self.assertFalse(index.valid())
  self.assertEqual(module.FrozenInputIndex(self.root,self.files).checked(self.path),self.sha)
 def test_continuous_unrelated_creation_during_build_and_scoped_lookup(self):
  original=module.directory_stamp;calls=[]
  def concurrent(path):
   (self.root/('result'+str(len(calls)))).touch();calls.append(path);return original(path)
  e=NS(ROOT=self.root,locked_digest=None,verify_inputs=None)
  with patch.object(module,'directory_stamp',concurrent),module.evaluator_scope(e):
   self.assertEqual(e.locked_digest(self.path,self.files),self.sha)
  self.assertIsNone(e.locked_digest);self.assertIsNone(e.verify_inputs)
 def test_changed_expected_hash_and_added_declared_alias_are_not_hidden(self):
  index=module.FrozenInputIndex(self.root,self.files);self.files['data']='0'*64
  with self.assertRaises(MultitaskProtocolError):index.checked(self.path)
  self.files['data']=self.sha;alias=self.root/'alias';alias.symlink_to(self.path);self.files['alias']=self.sha
  self.assertFalse(index.valid())
  with self.assertRaises(MultitaskProtocolError):module.FrozenInputIndex(self.root,self.files).checked(self.path)
if __name__=='__main__':unittest.main()
