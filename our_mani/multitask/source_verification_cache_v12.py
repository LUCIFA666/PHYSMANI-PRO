"""Parent-side input verification cache; no policy state/command/physics caching."""
import hashlib,json,sys
from pathlib import Path
from .audit_sha_cache_v14 import AuditSHACache
from .protocol import MultitaskProtocolError
CACHE=None

def install(root,files):
 global CACHE
 from . import evaluation,acquisition_lineage,reference_time,reference_overrides
 if CACHE is not None:return CACHE
 root=Path(root).resolve()
 # Bind this optimization and its exact byte-hashing implementation before use.
 for p in (Path(__file__).resolve(),root/'our_mani/multitask/audit_sha_cache_v14.py'):
  declared={value for name,value in files.items() if (root/name).resolve()==p}
  actual=hashlib.sha256(p.read_bytes()).hexdigest()
  if declared!={actual}:raise MultitaskProtocolError('source-verification cache implementation is not in actual frozen lock')
 originals=(evaluation.file_sha,acquisition_lineage.file_sha,reference_time.sha,reference_overrides.sha)
 cache=AuditSHACache()
 for name,module in list(sys.modules.items()):
  if name=='__main__' or name.startswith(('our_mani','scripts')):
   for attribute in ('file_sha','sha'):
    function=getattr(module,attribute,None)
    if any(function is original for original in originals):setattr(module,attribute,cache)
 CACHE=cache;return cache

def receipt():
 if CACHE is None:return None
 return dict(schema='r10_parent_source_SHA_cache_v12',scope='current file byte/source verification only',
  unique_file_reads=CACHE.reads,cache_hits=CACHE.hits,every_unique_file_fully_read=True,
  current_target_inode_size_mtime_ctime_checked_each_lookup=True,historical_or_declared_SHA_seeds=False,
  policy_state_cached=False,policy_commands_changed=False,physics_or_camera_queries_added=False)
