"""Canonical lock index unaffected by unrelated output directory creation.

Directory timestamp changes refresh only their declared child edges. Any
changed input name, inode, symlink or ancestor still invalidates the complete
canonical index. Exactly-one-canonical-match and strong byte checks remain.
"""
from pathlib import Path
from contextlib import contextmanager
import stat
from .frozen_input_index_v93 import InputTopologyChanged,directory_stamp
from .input_stat_cache_v79 import checked_sha,verify_inputs
from .protocol import MultitaskProtocolError


def edge_stamp(path):
 s=path.lstat();basic=(s.st_dev,s.st_ino,s.st_mode)
 return (*basic,s.st_mtime_ns,s.st_ctime_ns,str(path.readlink())) if stat.S_ISLNK(s.st_mode) else basic


class FrozenInputIndex:
 def __init__(self,root,files):
  if not isinstance(files,dict) or not files:raise MultitaskProtocolError('explicit frozen map required')
  self.root=Path(root).resolve();self.files=files;self.keys=set(files)
  self.by_path={};self.watches={};self.edges={}
  for name in files:
   path=self.root/name;child=path
   while child!=child.parent:
    parent=child.parent
    if parent not in self.watches:self.watches[parent]=directory_stamp(parent)
    siblings=self.edges.setdefault(parent,{})
    if child not in siblings:siblings[child]=edge_stamp(child)
    child=parent
   self.by_path.setdefault(str(path.resolve()),[]).append(name)
  if not self.valid():raise InputTopologyChanged('declared input edge changed during index construction')
 def valid(self):
  if set(self.files)!=self.keys:return False
  try:
   for parent,previous in self.watches.items():
    current=directory_stamp(parent)
    if current==previous:continue
    if current[:3]!=previous[:3]:return False
    if any(edge_stamp(child)!=expected for child,expected in self.edges[parent].items()):return False
    # Parent entries outside the explicit input graph may be new result files.
    # They cannot create a duplicate key in the fixed declared map.
    after=directory_stamp(parent)
    if after[:3]!=current[:3]:return False
    if after!=current and any(edge_stamp(child)!=expected for child,expected in self.edges[parent].items()):return False
    self.watches[parent]=after
   return True
  except OSError:return False
 def checked(self,path):
  if not self.valid():raise InputTopologyChanged('declared canonical input edges need refresh')
  path=Path(path).resolve();names=self.by_path.get(str(path),())
  if not self.valid():raise InputTopologyChanged('declared edge changed during canonical resolution')
  if len(names)!=1:raise MultitaskProtocolError('used input missing or conflicting in explicit frozen lock: '+str(path))
  actual,_=checked_sha(path,self.files[names[0]])
  if not self.valid():raise InputTopologyChanged('declared edge changed during byte validation')
  return actual


@contextmanager
def evaluator_scope(evaluator):
 old_lookup,old_verify=evaluator.locked_digest,evaluator.verify_inputs;indices={}
 def lookup(path,files):
  key=id(files)
  for _ in range(3):
   try:
    index=indices.get(key)
    if index is None or index.files is not files or not index.valid():
     index=FrozenInputIndex(evaluator.ROOT,files);indices[key]=index
    return index.checked(path)
   except InputTopologyChanged:indices.pop(key,None)
  raise MultitaskProtocolError('declared canonical edges remained unstable across three index checks')
 evaluator.locked_digest=lookup;evaluator.verify_inputs=verify_inputs
 try:yield
 finally:evaluator.locked_digest=old_lookup;evaluator.verify_inputs=old_verify


@contextmanager
def declared_edge_scope():
 from . import frozen_input_index_v93 as original
 previous=original.evaluator_scope;original.evaluator_scope=evaluator_scope
 try:yield
 finally:original.evaluator_scope=previous
