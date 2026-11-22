from pathlib import Path
import hashlib,tempfile,unittest
from our_mani.multitask.audit_sha_cache_v12 import AuditSHACache
class TestAuditSHA(unittest.TestCase):
 def test_unchanged_bytes_are_read_once_and_mutations_or_replacement_invalidate(self):
  root=Path(__file__).resolve().parents[1];directory=root/'runtime/framework/tmp';directory.mkdir(parents=True,exist_ok=True)
  with tempfile.TemporaryDirectory(dir=directory) as tmp:
   p=Path(tmp)/'data';p.write_bytes(b'abc');cache=AuditSHACache()
   self.assertEqual(cache(p),hashlib.sha256(b'abc').hexdigest());self.assertEqual(cache(p),cache(p));self.assertEqual(cache.reads,1)
   p.write_bytes(b'def');self.assertEqual(cache(p),hashlib.sha256(b'def').hexdigest());self.assertEqual(cache.reads,2)
   q=Path(tmp)/'replacement';q.write_bytes(b'xyz');q.replace(p);self.assertEqual(cache(p),hashlib.sha256(b'xyz').hexdigest());self.assertEqual(cache.reads,3)
if __name__=='__main__':unittest.main()
