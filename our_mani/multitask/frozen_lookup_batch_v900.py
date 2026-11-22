"""Scoped used-input lookup acceleration for a future, separately frozen wrapper.

This does not replace verify_inputs, scientific admission, or any release gate.
Every declared map retains a complete canonical/edge index and key/value
snapshot at transaction boundaries. Each used path is independently checked
with the original actual-SHA/strong-stat cache and rechecked at close. An
unused subtree's transient ABA between boundaries is not continuously sampled.

No hooks or filesystem writes occur on import/construction. The caller must
limit the active scope to driver bindings/preflight preparation, before native
execution. Existing direct json.loads(path.read_text()) calls are unchanged;
only locked_json/read_json below attest the exact parsed bytes.
"""
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import re
from threading import RLock, get_ident, local

from .frozen_input_edges_v463 import FrozenInputIndex, InputTopologyChanged, edge_stamp
from .frozen_input_index_v93 import directory_stamp
from .input_stat_cache_v79 import checked_sha
from .protocol import MultitaskProtocolError


_HOOK_LOCK = RLock()
_HOOK_STATE = local()
_SHA = re.compile(r'[0-9a-f]{64}\Z')


def _pointer(pointer):
    if not isinstance(pointer, dict) or set(pointer) != {'path', 'sha256'}:
        raise MultitaskProtocolError('explicit frozen pointer required')
    if not isinstance(pointer['path'], str) or not pointer['path']:
        raise MultitaskProtocolError('explicit frozen pointer path required')
    if not isinstance(pointer['sha256'], str) or _SHA.fullmatch(pointer['sha256']) is None:
        raise MultitaskProtocolError('explicit lowercase SHA256 pointer required')
    return pointer


class _CompleteEdgeIndex(FrozenInputIndex):
    """Also watch resolved target ancestors when a declared path is a symlink."""

    def __init__(self, root, files):
        super().__init__(root, files)
        for canonical in self.by_path:
            child = Path(canonical)
            while child != child.parent:
                parent = child.parent
                if parent not in self.watches:
                    self.watches[parent] = directory_stamp(parent)
                siblings = self.edges.setdefault(parent, {})
                current = edge_stamp(child)
                if child in siblings and siblings[child] != current:
                    raise InputTopologyChanged('resolved input edge changed during construction')
                siblings[child] = current
                child = parent
        if not self.valid():
            raise InputTopologyChanged('complete input topology changed during construction')


class _MapTransaction:
    def __init__(self, batch, files):
        if not isinstance(files, dict) or not files:
            raise MultitaskProtocolError('explicit frozen map required')
        if any(not isinstance(name, str) or not name or not isinstance(sha, str)
               or _SHA.fullmatch(sha) is None for name, sha in files.items()):
            raise MultitaskProtocolError('explicit paths and lowercase SHA256 values required')
        self.batch, self.files = batch, files
        self.snapshot = dict(files)
        self.index = _CompleteEdgeIndex(batch.root, files)
        self.used = {}
        self.graph()
        batch.metrics['declared_files'] += len(self.snapshot)
        batch.metrics['watched_directories'] += len(self.index.watches)
        batch.metrics['maps'] += 1

    def graph(self):
        self.batch.metrics['full_graph_boundary_checks'] += 1
        if dict(self.files) != self.snapshot:
            raise InputTopologyChanged('declared input map changed in lookup batch')
        if not self.index.valid():
            raise InputTopologyChanged('declared input topology changed in lookup batch')
        if dict(self.files) != self.snapshot:
            raise InputTopologyChanged('declared input map changed during graph check')

    def ancestry(self, raw):
        stamps = []
        for path in (raw, *raw.parents):
            stamp = edge_stamp(path)
            expected = self.index.edges.get(path.parent, {}).get(path)
            if expected is not None and stamp != expected:
                raise InputTopologyChanged('declared input path identity changed: ' + str(path))
            stamps.append((str(path), stamp))
        return tuple(stamps)

    def signature(self, requested, declared, canonical, expected):
        if self.files.get(self.index.by_path[str(canonical)][0]) != expected:
            raise InputTopologyChanged('used declared SHA changed in lookup batch')
        result = (str(canonical), expected, self.ancestry(requested),
                  self.ancestry(declared), self.ancestry(canonical))
        if (requested.resolve(strict=True) != canonical
                or declared.resolve(strict=True) != canonical):
            raise InputTopologyChanged('used input canonical mapping changed')
        return result

    def lookup(self, path, *, project_only, pointer_sha=None):
        requested = self.batch.root / Path(path)
        self.ancestry(requested)
        canonical = requested.resolve(strict=True)
        if project_only and not canonical.is_relative_to(self.batch.root):
            raise MultitaskProtocolError('suite evidence must be inside this project')
        names = self.index.by_path.get(str(canonical), ())
        if len(names) != 1:
            raise MultitaskProtocolError('used input missing or conflicting in explicit frozen lock: ' + str(canonical))
        name = names[0]
        declared, expected = self.batch.root / name, self.snapshot[name]
        if pointer_sha is not None and pointer_sha != expected:
            raise MultitaskProtocolError('used pointer differs from declared frozen SHA')
        before = self.signature(requested, declared, canonical, expected)
        key = (str(requested), str(declared))
        if key in self.used and self.used[key] != before:
            raise InputTopologyChanged('previously used input identity changed')
        self.batch._sha(canonical, expected)
        if self.signature(requested, declared, canonical, expected) != before:
            raise InputTopologyChanged('used input changed during byte validation')
        self.used[key] = before
        self.batch.metrics['pointer_checks'] += 1
        if not canonical.is_relative_to(self.batch.root):
            self.batch.metrics['used_external'] += 1
        return canonical, expected

    def close(self):
        first_error = None
        for (requested, declared), before in self.used.items():
            canonical, expected = Path(before[0]), before[1]
            try:
                if self.signature(Path(requested), Path(declared), canonical, expected) != before:
                    raise InputTopologyChanged('used input identity changed before close')
                self.batch._sha(canonical, expected)
                if self.signature(Path(requested), Path(declared), canonical, expected) != before:
                    raise InputTopologyChanged('used input changed during final byte validation')
                self.batch.metrics['used_path_end_checks'] += 1
            except Exception as error:
                if first_error is None:
                    first_error = error
        try:
            self.graph()
        except Exception as error:
            if first_error is None:
                first_error = error
        if first_error is not None:
            raise first_error


class LookupBatch:
    """One thread-owned, single-use transaction containing distinct lock maps.

    ``cache=None`` uses the unchanged v79 cache; CPU tests should pass a private
    temporary cache. No entire-map byte verification is implied by this class.
    The original preflight's verify_inputs calls remain mandatory.
    """

    def __init__(self, root, *, cache=None):
        self.root = Path(root).resolve()
        self.cache = None if cache is None else Path(cache)
        self._maps, self._owner, self._failure = {}, None, None
        self._active, self._closed, self._hook_active = False, False, False
        self.metrics = dict(schema='r10_scoped_frozen_lookup_metrics_v900', maps=0,
                            declared_files=0, watched_directories=0, pointer_checks=0,
                            full_graph_boundary_checks=0, used_path_end_checks=0,
                            used_external=0, actual_SHA_checks=0, strong_stat_reuses=0,
                            exact_JSON_payloads=0, closed=False, complete=False)

    def __enter__(self):
        if self._active or self._closed:
            raise RuntimeError('lookup batch is single-use')
        self._owner, self._active = get_ident(), True
        return self

    def _require(self, root=None):
        if not self._active or self._owner != get_ident():
            raise RuntimeError('lookup batch requires its active owning thread')
        if root is not None and Path(root).resolve() != self.root:
            raise MultitaskProtocolError('lookup batch root differs from caller root')

    def _map(self, root, files):
        self._require(root)
        key = id(files)
        transaction = self._maps.get(key)
        if transaction is None:
            transaction = _MapTransaction(self, files)
            self._maps[key] = transaction
        elif transaction.files is not files:
            raise RuntimeError('lock map object identity reused')
        return transaction

    def _sha(self, path, expected):
        actual, reused = (checked_sha(path, expected) if self.cache is None
                          else checked_sha(path, expected, cache=self.cache))
        if actual != expected:
            raise MultitaskProtocolError('actual SHA differs from declared frozen input')
        self.metrics['actual_SHA_checks'] += 1
        self.metrics['strong_stat_reuses'] += int(reused)

    @contextmanager
    def _operation(self):
        self._require()
        if self._failure is not None:
            raise MultitaskProtocolError('lookup batch previously rejected an input') from self._failure
        try:
            yield
        except Exception as error:
            self._failure = error
            raise

    def checked_external(self, root, files, path):
        """742._checked compatible; explicitly declared external paths allowed."""
        with self._operation():
            return self._map(root, files).lookup(path, project_only=False)[0]

    def locked_pointer(self, root, files, pointer):
        with self._operation():
            _pointer(pointer)
            return self._map(root, files).lookup(pointer['path'], project_only=True,
                                                 pointer_sha=pointer['sha256'])[0]

    def _json(self, transaction, path, *, project_only, pointer_sha=None):
        canonical, expected = transaction.lookup(path, project_only=project_only, pointer_sha=pointer_sha)
        payload = canonical.read_bytes()
        if hashlib.sha256(payload).hexdigest() != expected:
            raise MultitaskProtocolError('parsed JSON bytes differ from frozen SHA')
        # Parse only the exact verified payload, then detect changes during parsing.
        result = json.loads(payload)
        transaction.lookup(path, project_only=project_only, pointer_sha=pointer_sha)
        self.metrics['exact_JSON_payloads'] += 1
        return result

    def locked_json(self, root, files, pointer):
        with self._operation():
            _pointer(pointer)
            return self._json(self._map(root, files), pointer['path'], project_only=True,
                              pointer_sha=pointer['sha256'])

    def read_json(self, root, files, path, *, project_only=False):
        """Explicit exact-payload reader; never implicitly declares an input."""
        with self._operation():
            return self._json(self._map(root, files), path, project_only=project_only)

    def close(self):
        self._require()
        if self._hook_active:
            raise RuntimeError('close lookup hooks before closing the batch')
        first_error = self._failure
        try:
            for transaction in self._maps.values():
                try:
                    transaction.close()
                except Exception as error:
                    if first_error is None:
                        first_error = error
            if first_error is not None:
                raise first_error
            self.metrics['complete'] = True
        finally:
            self._active, self._closed = False, True
            self.metrics['closed'] = True

    def __exit__(self, exc_type, exc, tb):
        self.close()
        # A clean close does not turn a failed scientific/preflight call into success.
        if exc_type is not None:
            self.metrics['complete'] = False
        return False

    @contextmanager
    def scoped_hooks(self, *, external=(), pointers=(), json_readers=()):
        """Optional explicit lookup-only hooks; all unrelated threads fall back.

        Targets are (module, attribute) pairs. Only _checked, locked_pointer and
        locked_json may be replaced. No verify_inputs/scientific function hooks.
        The future wrapper owns the duration of bindings/prepare execution.
        """
        self._require()
        with _HOOK_LOCK:
            if self._hook_active or getattr(_HOOK_STATE, 'batch', None) is not None:
                raise RuntimeError('nested lookup hook scopes are not supported')
            targets = []
            for entries, allowed, handler in ((external, '_checked', self.checked_external),
                                               (pointers, 'locked_pointer', self.locked_pointer),
                                               (json_readers, 'locked_json', self.locked_json)):
                for owner, attribute in entries:
                    if attribute != allowed:
                        raise ValueError('only explicit lookup entry points may be hooked')
                    if any(owner is item[0] and attribute == item[1] for item in targets):
                        continue
                    original = getattr(owner, attribute)
                    if not callable(original):
                        raise ValueError('lookup hook target must be callable')
                    targets.append((owner, attribute, original, handler))
            installed = []
            self._hook_active, _HOOK_STATE.batch = True, self
            try:
                for owner, attribute, original, handler in targets:
                    def dispatch(*args, _old=original, _handler=handler, **kwargs):
                        if getattr(_HOOK_STATE, 'batch', None) is self:
                            return _handler(*args, **kwargs)
                        return _old(*args, **kwargs)
                    setattr(owner, attribute, dispatch)
                    installed.append((owner, attribute, original))
                yield self
            finally:
                for owner, attribute, original in reversed(installed):
                    setattr(owner, attribute, original)
                self._hook_active, _HOOK_STATE.batch = False, None
