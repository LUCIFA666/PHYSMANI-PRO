"""Read-only, exact-format review helper; it grants no execution permission.

No production inventory or wrapper imports this helper.  The restricted parser
is deliberately narrower than CoppeliaSim's general persistent-data container.
"""
import hashlib
from pathlib import Path
import struct


MARKER = b'SIMSETTINGS_SIM_CRASHED'


def parse_crash_marker(payload):
    """Accept exactly one known marker entry, with no extra bytes or settings."""
    def reject(message):
        raise ValueError('exact Coppelia crash-marker review: '+message)
    if not isinstance(payload, bytes):
        reject('bytes required')
    if len(payload) not in (39, 40):
        reject('unexpected length')
    cursor = 0
    def integer():
        nonlocal cursor
        if cursor+4 > len(payload):
            reject('truncated integer')
        value = struct.unpack_from('<i', payload, cursor)[0]
        cursor += 4
        return value
    if integer() != 1:
        reject('exactly one entry required')
    if payload[cursor:cursor+1] != b'\x01':
        reject('entry tag differs')
    cursor += 1
    if integer() != len(MARKER) or payload[cursor:cursor+len(MARKER)] != MARKER:
        reject('unknown persistent-data key')
    cursor += len(MARKER)
    length = integer()
    value = payload[cursor:cursor+length]
    cursor += length
    if value not in (b'No', b'Yes') or length != len(value):
        reject('unknown marker value')
    if payload[cursor:] != b'\x00':
        reject('extra persistent state or invalid terminator')
    return {'entry_count':1, 'key':MARKER.decode('ascii'), 'value':value.decode('ascii'),
            'size_bytes':len(payload), 'sha256':hashlib.sha256(payload).hexdigest(),
            'hex':payload.hex()}


def inspect_exact_file(path, expected_resolved_path):
    """Inspect only the declared path; never write, repair, or normalize it."""
    path = Path(path)
    if path.resolve() != Path(expected_resolved_path).resolve():
        raise ValueError('exact Coppelia crash-marker review: undeclared state path')
    before = path.stat()
    payload = path.read_bytes()
    after = path.stat()
    stat = lambda s: {'size':s.st_size, 'mtime_ns':s.st_mtime_ns, 'inode':s.st_ino}
    if stat(before) != stat(after):
        raise ValueError('exact Coppelia crash-marker review: concurrent unstable read')
    return {'path':str(path), 'resolved_path':str(path.resolve()),
            'before_stat':stat(before), 'after_stat':stat(after),
            'parsed':parse_crash_marker(payload), 'read_only':True,
            'execution_permission_granted':False}
