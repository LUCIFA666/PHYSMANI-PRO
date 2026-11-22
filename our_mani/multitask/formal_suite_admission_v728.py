"""Batch only the topology checks around unchanged v718 scientific admission.

Every used pointer retains its canonical uniqueness, declared path ancestry
and strict SHA/strong-stat checks. Complete topology and input-map snapshots
are checked at both transaction boundaries. No physical policy or admission
condition changes, and public lookups outside a transaction remain v532.
"""
from pathlib import Path
from threading import RLock,local
from unittest.mock import patch
import hashlib,json
from . import formal_suite_admission_v718 as scientific
from .formal_suite_admission_v532 import locked_pointer,locked_json
from .frozen_input_edges_v463 import FrozenInputIndex,InputTopologyChanged,edge_stamp
from .input_stat_cache_v79 import checked_sha
from .protocol import MultitaskProtocolError
from .evaluation_roster16_v283 import TASKS,METHODS,validate_roster
from .three_method_release_v531 import require_test_release

_SCOPE_LOCK=RLock()
_STATE=local()


class AdmissionTransaction:
    """One immutable declared input graph for a batch of checked pointers."""

    def __init__(self,root,files):
        self.root=Path(root).resolve();self.files=files
        if not isinstance(files,dict) or not files:raise MultitaskProtocolError('explicit frozen map required')
        self.snapshot=dict(files)
        self.index=FrozenInputIndex(self.root,files)
        self.used={};self.active=False
        self.metrics=dict(pointer_checks=0,full_graph_boundary_checks=0,
            declared_files=len(files),watched_directories=len(self.index.watches))

    def _graph(self):
        if dict(self.files)!=self.snapshot:raise InputTopologyChanged('frozen input graph keys or expected hashes changed')
        self.metrics['full_graph_boundary_checks']+=1
        if not self.index.valid():raise InputTopologyChanged('declared canonical input topology changed during admission')
        if dict(self.files)!=self.snapshot:raise InputTopologyChanged('frozen input graph changed during topology validation')

    def _ancestry(self,raw):
        raw=Path(raw);stamps={}
        for path in (raw,*raw.parents):
            current=edge_stamp(path);stamps[path]=current
            expected=self.index.edges.get(path.parent,{}).get(path)
            if expected is not None and current!=expected:
                raise InputTopologyChanged('used declared path or ancestor changed: '+str(path))
        return stamps

    def __enter__(self):
        self._graph();self.active=True;return self

    def __exit__(self,kind,error,traceback):
        try:
            # Recheck every used input, including changes made after its last
            # individual lookup. Unused alias changes are caught by _graph.
            for (requested,declared),item in self.used.items():
                canonical,expected,before_requested,before_declared=item
                if (self._ancestry(requested)!=before_requested or self._ancestry(declared)!=before_declared
                    or requested.resolve(strict=True)!=canonical or declared.resolve(strict=True)!=canonical):
                    raise InputTopologyChanged('used canonical path changed before admission completed')
                actual,_=checked_sha(canonical,expected)
                if actual!=expected:raise MultitaskProtocolError('used input differs from the frozen SHA')
                if (self._ancestry(requested)!=before_requested or self._ancestry(declared)!=before_declared
                    or requested.resolve(strict=True)!=canonical or declared.resolve(strict=True)!=canonical):
                    raise InputTopologyChanged('used path changed during final byte validation')
            self._graph()
        finally:self.active=False
        return False

    def locked_pointer(self,root,files,pointer):
        if not self.active or Path(root).resolve()!=self.root or files is not self.files:
            raise ValueError('source-bound active admission transaction required')
        if not isinstance(pointer,dict) or set(pointer)!={'path','sha256'}:
            raise ValueError('explicit frozen pointer required')
        requested=self.root/pointer['path'];before_requested=self._ancestry(requested)
        canonical=requested.resolve(strict=True)
        if not canonical.is_relative_to(self.root):raise ValueError('suite evidence must be inside this project')
        names=self.index.by_path.get(str(canonical),())
        if len(names)!=1:raise MultitaskProtocolError('used input missing or conflicting in explicit frozen lock: '+str(canonical))
        name=names[0];declared=self.root/name;before_declared=self._ancestry(declared)
        expected=self.snapshot[name]
        if self.files.get(name)!=expected:raise InputTopologyChanged('used input expected SHA changed during admission')
        if declared.resolve(strict=True)!=canonical:raise InputTopologyChanged('declared input canonical mapping changed')
        if pointer['sha256']!=expected:raise ValueError('used pointer differs from declared frozen SHA')
        actual,_=checked_sha(canonical,expected)
        if actual!=expected:raise MultitaskProtocolError('used input differs from the frozen SHA')
        if (self._ancestry(requested)!=before_requested or self._ancestry(declared)!=before_declared
            or requested.resolve(strict=True)!=canonical or declared.resolve(strict=True)!=canonical):
            raise InputTopologyChanged('used input path changed during byte validation')
        self.used[(requested,declared)]=(canonical,expected,before_requested,before_declared)
        self.metrics['pointer_checks']+=1
        return canonical

    def locked_json(self,root,files,pointer):
        path=self.locked_pointer(root,files,pointer)
        payload=path.read_bytes()
        if hashlib.sha256(payload).hexdigest()!=pointer['sha256']:
            raise MultitaskProtocolError('parsed JSON bytes differ from the frozen SHA')
        result=json.loads(payload)
        # The JSON parse reads the same strictly checked bytes; a concurrent
        # write cannot substitute unchecked metadata between hash and parse.
        self.locked_pointer(root,files,pointer)
        return result


def admission_metrics():
    """Control-plane counters for the last admission in this thread."""
    return dict(getattr(_STATE,'metrics',{}))


def admit(root,suite,files):
    # Preserve the scientific gate's ordering for an invalid task/method set:
    # it must reject before touching any declared artifact or empty map.
    if suite.get('methods')!=list(METHODS) or suite.get('tasks')!=list(TASKS):
        return scientific.admit(root,suite,files)
    with _SCOPE_LOCK:
        transaction=AdmissionTransaction(root,files)
        try:
            with transaction,patch.object(scientific,'locked_pointer',transaction.locked_pointer),\
                 patch.object(scientific,'locked_json',transaction.locked_json):
                return scientific.admit(root,suite,files)
        finally:_STATE.metrics=dict(transaction.metrics)


def read_formal_source(root,suite,files,split,*,validation_release=None):
    if split not in ('validation','test'):raise ValueError('formal VAL/TEST only')
    proof=admit(root,suite,files)
    if split=='test':require_test_release(validation_release or {},proof['routes'])
    # No TEST manifest or reset file is selected before the complete VAL gate.
    manifest=locked_json(root,files,suite['source_manifests'][split])
    rows=validate_roster(manifest,split,completed_VAL_audit=split=='test')
    return manifest,rows,proof
