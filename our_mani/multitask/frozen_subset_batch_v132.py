"""Verify a declared subset with one canonical topology audit per batch.

The complete index is checked before and after every batch. Each used file
still needs exactly one canonical lock entry and the original strong-stat/SHA
check. No policy step or native retry occurs inside this CPU verification.
"""
from pathlib import Path
from .frozen_input_index_v93 import FrozenInputIndex,InputTopologyChanged
from .input_stat_cache_v79 import checked_sha
from .protocol import MultitaskProtocolError


def checked_subset(root,lock,subset):
    if not isinstance(subset,dict) or not subset:raise MultitaskProtocolError('nonempty explicit used-input subset required')
    for _ in range(3):
        try:
            index=FrozenInputIndex(root,lock)
            if not index.valid():raise InputTopologyChanged('batch index topology changed before verification')
            verified={}
            for name,declared_sha in subset.items():
                path=Path(name);path=path if path.is_absolute() else Path(root)/path
                canonical=path.resolve();names=index.by_path.get(str(canonical),())
                if len(names)!=1:raise MultitaskProtocolError('used input missing or conflicting in explicit frozen lock: '+str(path))
                expected=lock[names[0]]
                if expected!=declared_sha:raise MultitaskProtocolError('used-input subset differs from frozen SHA: '+str(path))
                actual,_=checked_sha(canonical,expected);verified[str(canonical)]=actual
            if not index.valid():raise InputTopologyChanged('batch index topology changed during verification')
            return verified
        except InputTopologyChanged:continue
    raise MultitaskProtocolError('canonical batch topology remained unstable across three index validations')
