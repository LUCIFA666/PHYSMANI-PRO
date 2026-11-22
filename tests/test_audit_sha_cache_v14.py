from pathlib import Path
import hashlib,tempfile,unittest,os
from our_mani.multitask.audit_sha_cache_v14 import AuditSHACache
class TestAuditSHA(unittest.TestCase):
 def test_mutation_replacement_and_symlink_retargeting_invalidate_even_same_size(self):
  directory=Path(__file__).resolve().parents[1]/'runtime/framework/tmp';directory.mkdir(parents=True,exist_ok=True)
  with tempfile.TemporaryDirectory(dir=directory) as tmp:
   p=Path(tmp)/'a';p.write_bytes(b'abc');q=Path(tmp)/'b';q.write_bytes(b'def');link=Path(tmp)/'link';link.symlink_to(p)
   cache=AuditSHACache();self.assertEqual(cache(p),hashlib.sha256(b'abc').hexdigest());self.assertEqual(cache(link),cache(p));self.assertEqual(cache.reads,1)
   link.unlink();link.symlink_to(q);self.assertEqual(cache(link),hashlib.sha256(b'def').hexdigest());self.assertEqual(cache.reads,2)
   before=p.stat();p.write_bytes(b'ghi');os.utime(p,ns=(before.st_atime_ns,before.st_mtime_ns));self.assertEqual(cache(p),hashlib.sha256(b'ghi').hexdigest())
   q.replace(p);self.assertEqual(cache(p),hashlib.sha256(b'def').hexdigest())
if __name__=='__main__':unittest.main()
