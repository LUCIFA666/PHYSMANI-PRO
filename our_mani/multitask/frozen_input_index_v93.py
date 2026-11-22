"""Scoped canonical lookup index for the unchanged explicit input lock.

Preserves the original exactly-one-canonical-match rule. Directory/ancestor
identity stamps invalidate the index when names or symlinks change; current
expected values are read from the declared map and checked by actual-SHA plus
strong-stat reuse. This removes repeated path resolution of the entire lock
for every single used input, without changing model/policy input declarations.
"""
from pathlib import Path
from contextlib import contextmanager
from .input_stat_cache_v79 import checked_sha,verify_inputs
from .protocol import MultitaskProtocolError


class InputTopologyChanged(MultitaskProtocolError):
    """Retry only canonical-index validation, never a native action or rollout."""


def directory_stamp(path):
    s=path.lstat();return (s.st_dev,s.st_ino,s.st_mode,s.st_mtime_ns,s.st_ctime_ns)


class FrozenInputIndex:
    def __init__(self,root,files):
        if not isinstance(files,dict) or not files:raise MultitaskProtocolError('explicit frozen map required')
        self.root=Path(root).resolve();self.files=files;self.keys=set(files);self.by_path={};self.watches={}
        for name in files:
            path=self.root/name
            parent=path.parent
            while parent!=parent.parent:
                if parent not in self.watches:self.watches[parent]=directory_stamp(parent)
                parent=parent.parent
            canonical=str(path.resolve());self.by_path.setdefault(canonical,[]).append(name)
        if not self.valid():raise InputTopologyChanged('input path topology changed during index construction')

    def valid(self):
        if set(self.files)!=self.keys:return False
        try:return all(directory_stamp(p)==s for p,s in self.watches.items())
        except OSError:return False

    def checked(self,path):
        if not self.valid():raise InputTopologyChanged('canonical input index needs refresh')
        path=Path(path).resolve();names=self.by_path.get(str(path),())
        if not self.valid():raise InputTopologyChanged('input topology changed during canonical resolution')
        if len(names)!=1:raise MultitaskProtocolError('used input missing or conflicting in explicit frozen lock: '+str(path))
        expected=self.files[names[0]];actual,_=checked_sha(path,expected)
        if not self.valid():raise InputTopologyChanged('input topology changed during checked SHA')
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
            except InputTopologyChanged:
                indices.pop(key,None)
        raise MultitaskProtocolError('canonical input topology remained unstable across three index checks')

    evaluator.locked_digest=lookup;evaluator.verify_inputs=verify_inputs
    try:yield
    finally:evaluator.locked_digest=old_lookup;evaluator.verify_inputs=old_verify
