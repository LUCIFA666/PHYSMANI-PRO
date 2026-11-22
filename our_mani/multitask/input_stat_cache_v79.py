"""Reuse a byte-verified SHA only while strong filesystem identity is unchanged.

Each first/change read hashes all bytes and checks stat before and after. Cache
reuse requires matching canonical path, device, inode, size, nanosecond mtime
and ctime, and the exact expected SHA from the caller's frozen input lock.
This is an optimization for this trusted local workspace, not an attestation
against a privileged adversary able to forge filesystem metadata/cache records.
The policy's declared input set, expected SHAs and scientific data stay intact.
"""
from pathlib import Path
from contextlib import contextmanager
import hashlib,json,os,stat,fcntl
from .evaluation import file_sha
from .protocol import MultitaskProtocolError

ROOT=Path(__file__).resolve().parents[2]
CACHE=ROOT/'runtime/framework/source_input_stat_cache_v79'
SCHEMA='r10_actual_SHA_strong_stat_reuse_v79'


def stamp(path):
    s=path.stat()
    if not stat.S_ISREG(s.st_mode):raise MultitaskProtocolError('regular frozen input required: '+str(path))
    return dict(device=s.st_dev,inode=s.st_ino,size=s.st_size,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns)


def checked_sha(path,expected,*,cache=CACHE):
    if not isinstance(expected,str) or len(expected)!=64 or any(c not in '0123456789abcdef' for c in expected):raise MultitaskProtocolError('explicit SHA256 required')
    try:path=Path(path).resolve(strict=True)
    except OSError as e:raise MultitaskProtocolError('frozen input missing: '+str(path)) from e
    cache=Path(cache);cache.mkdir(parents=True,exist_ok=True)
    key=hashlib.sha256(str(path).encode()).hexdigest();receipt=cache/(key+'.json');lock=cache/(key+'.lock')
    with lock.open('a+b') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX)
        before=stamp(path);record=None
        try:record=json.loads(receipt.read_text())
        except (OSError,json.JSONDecodeError):pass
        if (isinstance(record,dict) and record.get('schema')==SCHEMA and record.get('canonical_path')==str(path)
            and record.get('actual_sha256')==expected and record.get('strong_stat')==before):return expected,True
        actual=file_sha(path);after=stamp(path)
        if before!=after:raise MultitaskProtocolError('frozen input changed during byte hashing: '+str(path))
        if actual!=expected:raise MultitaskProtocolError('frozen input SHA changed: '+str(path))
        value=dict(schema=SCHEMA,canonical_path=str(path),actual_sha256=actual,strong_stat=after,full_byte_hash_verified=True)
        temporary=receipt.with_suffix('.tmp.'+str(os.getpid()))
        temporary.write_text(json.dumps(value,sort_keys=True)+'\n');temporary.replace(receipt)
        return actual,False


def verify_inputs(root,declared,*,cache=CACHE,stats=None):
    if not isinstance(declared,dict) or not declared:raise MultitaskProtocolError('explicit frozen input lock required')
    verified={};hits=misses=0
    for name,expected in declared.items():
        actual,reused=checked_sha(Path(root)/name,expected,cache=cache);verified[name]=actual
        hits+=int(reused);misses+=int(not reused)
    if stats is not None:stats.update(files=len(declared),strong_stat_reuses=hits,full_byte_hashes=misses,expected_input_set_changed=False)
    return verified


@contextmanager
def evaluator_scope(evaluator):
    previous=evaluator.verify_inputs
    evaluator.verify_inputs=verify_inputs
    try:yield
    finally:evaluator.verify_inputs=previous
