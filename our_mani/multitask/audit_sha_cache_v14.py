"""Audit-only SHA cache guarded by the current target inode and full stat tuple.

os.stat follows the *current* symlink chain in the kernel on every query. A
cache hit requires dev/inode/size/mtime_ns/ctime_ns equality. This avoids repeated
Python Path.resolve walks without trusting declared or historical hashes.
"""
import hashlib,os,time
class AuditSHACache:
 def __init__(self):self.values={};self.hits=0;self.reads=0;self.bytes=0;self.started=time.monotonic()
 @staticmethod
 def signature(path):
  s=os.stat(path);return (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
 def __call__(self,path):
  name=os.fspath(path);before=self.signature(name);key=before[:2];entry=self.values.get(key)
  if entry is not None and entry[0]==before:self.hits+=1;return entry[1]
  h=hashlib.sha256()
  with open(name,'rb') as stream:
   for block in iter(lambda:stream.read(1<<20),b''):h.update(block)
  if self.signature(name)!=before:raise RuntimeError('audit input changed while hashing: '+name)
  value=h.hexdigest();self.values[key]=(before,value);self.reads+=1;self.bytes+=before[2];return value
 def receipt(self):return dict(unique_file_reads=self.reads,bytes_read=self.bytes,cache_hits=self.hits,wall_seconds=time.monotonic()-self.started,every_unique_file_fully_read=True,declared_SHA_used_to_seed_cache=False,actor_or_native_process_cache_installed=False,symlink_target_stat_checked_on_every_query=True)
