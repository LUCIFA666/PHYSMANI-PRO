"""Audit-only SHA deduplication with the same stat guards as _Inputs.

Every distinct unchanged file is read completely on its first use. No declared
SHA or historical receipt seeds this cache. Any inode/size/mtime/ctime change
invalidates it. Never imported or installed by a policy/native actor.
"""
from pathlib import Path
import hashlib,time
class AuditSHACache:
 def __init__(self):self.values={};self.hits=0;self.reads=0;self.bytes=0;self.started=time.monotonic()
 @staticmethod
 def signature(path):
  s=path.stat();return (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
 def __call__(self,path):
  p=Path(path).resolve();before=self.signature(p);entry=self.values.get(p)
  if entry is not None and entry[0]==before:self.hits+=1;return entry[1]
  h=hashlib.sha256()
  with p.open('rb') as stream:
   for block in iter(lambda:stream.read(1<<20),b''):h.update(block)
  if self.signature(p)!=before:raise RuntimeError('audit input changed while hashing: '+str(p))
  value=h.hexdigest();self.values[p]=(before,value);self.reads+=1;self.bytes+=before[2];return value
 def receipt(self):return dict(unique_file_reads=self.reads,bytes_read=self.bytes,cache_hits=self.hits,wall_seconds=time.monotonic()-self.started,every_unique_file_fully_read=True,declared_SHA_used_to_seed_cache=False,actor_or_native_process_cache_installed=False)
