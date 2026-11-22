"""Restore exact frozen OS file bytes inside a private bwrap mount namespace.

This is file environment recovery, not a change to task parameters or policy.
The project, PID inventory, network and devices remain the host surfaces. No
simulation, model or job is selected by this standalone wrapper.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SETUP = ROOT / 'artifacts/framework/multitask_r10/frozen_environment_restore_v820_setup'
SNAPSHOT = SETUP / 'system_snapshot_v820'
BASE_MANIFEST = SNAPSHOT / 'manifest.json'
BASE_MANIFEST_SHA = '35ac1ac2bcc764855fb46704d0603d0262801e80c187531f9476a3b4e7b85191'
MANIFEST = SETUP / 'hybrid_environment_manifest_v820.json'
SCHEMA = 'r10_exact_frozen_OS_hybrid_snapshot_v820'
LIBRARY_TARGET = Path('/usr/lib/x86_64-linux-gnu')
LIBRARY_SNAPSHOT = SETUP/'complete_library_directory_v820'
LIBRARY_INVENTORY = SETUP/'complete_library_directory_inventory_v820.json'
LOCKS = (
    ('artifacts/framework/multitask_r10/author_outcome_qualification_v745_setup/frozen_inputs.json',
     'd497facb879abb961cc2d2812e9233bccf43f35c927d71e1a1caa6e3f1e9932e'),
    ('artifacts/framework/multitask_r10/formal_suite16_v752_setup/frozen_inputs.json',
     '65d6f816c31d209b66e9b7a23825e9572d3d038c520b5c4fbd64aac8f06bb354'),
    ('artifacts/framework/multitask_r10/formal_VAL_resume_v772_setup/resume_inputs.json',
     'a17aaa1be764bcc0b82955571dccb7db27c84fb8c47ee0a9192d0dfcfcc17011'),
)
OS_ROOTS = ('/usr/', '/etc/', '/lib/', '/lib64/', '/bin/', '/sbin/')
BWRAP = Path('/usr/bin/bwrap')
BWRAP_SHA = '523da3e7399044be5163aee6f57a77a6bef7454376e28f0a0627920bae1b76b6'
RESTORE_RECEIPT_SHA = '3fa79af0ee862a529229770d5a13943f1da00f439c924aab0303518f8c5b5d36'
RESTORED_TARGETS = frozenset((
    '/usr/bin/git', '/usr/bin/git-shell', '/usr/bin/scalar',
    '/usr/lib/x86_64-linux-gnu/libXtst.so.6.1.0',
))
MARKER = 'R10_FROZEN_SYSTEM_ENVIRONMENT_V820'


def utc():
    return datetime.now(timezone.utc).isoformat()


def project_pointer(path, expected_sha=None):
    path = Path(path).resolve(strict=True)
    return dict(path=str(path.relative_to(ROOT)), sha256=expected_sha or sha(path))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda: f.read(1024 * 1024), b''):
            h.update(data)
    return h.hexdigest()


def require_sha(value):
    if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError('explicit lowercase SHA256 required')
    return value


def os_target(name):
    """No relative paths, traversal, aliases or non-OS/project overrides."""
    if (not isinstance(name, str) or '\0' in name or not name.startswith(OS_ROOTS)
            or str(PurePosixPath(name)) != name or '..' in PurePosixPath(name).parts):
        raise ValueError('frozen OS absolute target required: ' + repr(name))
    path = Path(name)
    if path.is_symlink() or str(path.resolve(strict=True)) != name or not path.is_file():
        raise ValueError('canonical regular OS target required: ' + name)
    return path


def frozen_os_files():
    merged = {}
    sources = []
    for relative, expected in LOCKS:
        path = ROOT / relative
        if sha(path) != expected:
            raise ValueError('exact original frozen lock changed: ' + str(path))
        files = json.loads(path.read_text())['files']
        count = 0
        for name, expected_file in files.items():
            if not isinstance(name, str) or not name.startswith(OS_ROOTS):
                continue
            os_target(name)
            require_sha(expected_file)
            if name in merged and merged[name] != expected_file:
                raise ValueError('conflicting frozen OS SHA: ' + name)
            merged[name] = expected_file
            count += 1
        sources.append(dict(path=str(path), sha256=expected, OS_file_count=count))
    if len(merged) != 6418 or merged.get(str(BWRAP)) != BWRAP_SHA:
        raise ValueError('exact v745/v752/v772 frozen OS union required')
    return dict(sorted(merged.items())), sources


def stamp(s):
    return (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns, s.st_mode)


def file_identity(s):
    return dict(device=s.st_dev, inode=s.st_ino, bytes=s.st_size, mode=s.st_mode,
                mtime_ns=s.st_mtime_ns, ctime_ns=s.st_ctime_ns)


def inventory_tree(root):
    """Complete no-follow inventory of one explicitly authorized public library tree."""
    root = Path(root)
    if root not in (LIBRARY_TARGET, LIBRARY_SNAPSHOT) or root.is_symlink() or root.resolve(strict=True) != root:
        raise ValueError('only the authorized complete public library directory may be inventoried')
    entries = {}
    def visit(directory):
        before = directory.lstat()
        if not stat.S_ISDIR(before.st_mode):
            raise ValueError('regular directory required during library inventory')
        for path in sorted(directory.iterdir()):
            first = path.lstat()
            relative = str(path.relative_to(root))
            item = dict(identity=file_identity(first), mode=oct(stat.S_IMODE(first.st_mode)))
            if stat.S_ISLNK(first.st_mode):
                item.update(kind='symlink', target=os.readlink(path))
            elif stat.S_ISDIR(first.st_mode):
                item.update(kind='directory')
                visit(path)
            elif stat.S_ISREG(first.st_mode):
                digest = hashlib.sha256()
                fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
                with os.fdopen(fd, 'rb') as f:
                    if stamp(first) != stamp(os.fstat(f.fileno())):
                        raise ValueError('library file identity raced before inventory: '+str(path))
                    for data in iter(lambda: f.read(1024*1024), b''):
                        digest.update(data)
                    if stamp(first) != stamp(os.fstat(f.fileno())):
                        raise ValueError('library file changed during inventory: '+str(path))
                item.update(kind='file', bytes=first.st_size, sha256=digest.hexdigest())
            else:
                raise ValueError('unsupported special entry in public library tree: '+str(path))
            if stamp(first) != stamp(path.lstat()):
                raise ValueError('library entry changed during inventory: '+str(path))
            entries[relative] = item
        if stamp(before) != stamp(directory.lstat()):
            raise ValueError('library directory changed during inventory: '+str(directory))
    root_before = root.lstat()
    visit(root)
    return dict(root_identity=file_identity(root_before), root_mode=oct(stat.S_IMODE(root_before.st_mode)),
                entries=dict(sorted(entries.items())))


def content_inventory(inventory):
    return dict(root_mode=inventory['root_mode'], entries={name:{k:v for k,v in row.items() if k != 'identity'}
                for name,row in inventory['entries'].items()})


def write_json_new(path, value):
    with Path(path).open('x') as f:
        f.write(json.dumps(value, indent=2, sort_keys=True) + '\n')


def create_snapshot():
    """CPU-only, no install. Leave a failed partial snapshot for inspection."""
    expected, sources = frozen_os_files()
    receipt = SETUP / 'restore_materials_v820.json'
    if sha(receipt) != RESTORE_RECEIPT_SHA:
        raise ValueError('exact verified restore materials receipt required')
    materials = json.loads(receipt.read_text())
    old = {r['system_target']: r for r in materials['files']}
    if set(old) != RESTORED_TARGETS or not materials['all_four_files_exact_frozen_SHA']:
        raise ValueError('exact four previously verified restored files required')
    required_bytes = sum(Path(name).stat().st_size for name in expected)
    free = shutil.disk_usage(SETUP).free
    if free < required_bytes + 1024 ** 3:
        raise OSError('insufficient space for isolated snapshot plus 1 GiB margin')
    SNAPSHOT.mkdir(exist_ok=False)
    start = utc()
    entries = []
    try:
        for index, (name, digest) in enumerate(expected.items(), 1):
            src = Path(old[name]['staged_file']) if name in old else Path(name)
            if name in old and (src != SETUP/'overlay'/name.lstrip('/') or sha(src) != digest):
                raise ValueError('exact allowlisted old overlay required: ' + name)
            if src.is_symlink() or src.resolve(strict=True) != src:
                raise ValueError('canonical regular snapshot source required: ' + str(src))
            relative = 'files/' + name.lstrip('/')
            dest = SNAPSHOT / relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            actual = hashlib.sha256()
            with src.open('rb') as source, dest.open('xb') as target:
                before = source.fileno()
                first = os.fstat(before)
                if not stat.S_ISREG(first.st_mode):
                    raise ValueError('regular source file required: ' + str(src))
                for data in iter(lambda: source.read(1024 * 1024), b''):
                    actual.update(data)
                    target.write(data)
                if stamp(first) != stamp(os.fstat(before)):
                    raise ValueError('source changed while copying: ' + str(src))
            if actual.hexdigest() != digest or sha(dest) != digest:
                raise ValueError('frozen source/snapshot SHA mismatch: ' + name)
            # Read/execute permissions suffice for a read-only bind. The saved
            # bytes retain their frozen SHA; no special privilege bits are added.
            snapshot_mode = stat.S_IMODE(first.st_mode) & 0o555
            dest.chmod(snapshot_mode)
            entries.append(dict(target=name, snapshot_file=relative, sha256=digest,
                bytes=first.st_size, source=str(src), restored_from_old_package=name in old,
                source_mode=oct(stat.S_IMODE(first.st_mode)), snapshot_mode=oct(snapshot_mode)))
            if index % 1000 == 0:
                print(json.dumps(dict(copied=index, total=len(expected))), flush=True)
        manifest = dict(schema='r10_exact_frozen_OS_file_snapshot_v820', complete=True, created_utc=start, finished_utc=utc(),
            scope='exact frozen OS file bytes; file environment recovery, no parameter or strategy change',
            source_locks=sources, frozen_OS_file_count=len(entries), total_bytes=sum(e['bytes'] for e in entries),
            restore_materials=dict(path=str(receipt), sha256=RESTORE_RECEIPT_SHA),
            bwrap=dict(path=str(BWRAP), sha256=BWRAP_SHA), files=entries,
            pre_copy_free_bytes=free, mode_note='snapshot strips write/setuid/setgid bits; original modes recorded',
            namespace_contract=dict(root_bind='read-write host /', per_file_binds='read-only',
                unshare_PID=False, unshare_network=False, unshare_devices=False,
                proc_or_dev_replaced=False, system_or_locks_modified=False,
                native_GPU_model_or_job_started=False))
        write_json_new(BASE_MANIFEST, manifest)
        print(json.dumps(dict(complete=True, manifest=str(BASE_MANIFEST), manifest_sha256=sha(BASE_MANIFEST),
                              files=len(entries), bytes=manifest['total_bytes'])), flush=True)
        return manifest
    except BaseException as error:
        write_json_new(SNAPSHOT/'partial_failure.json', dict(complete=False, failed_utc=utc(),
            error=repr(error), copied_verified_files=len(entries), no_host_files_changed=True))
        raise


def validate_library_binding(m, expected):
    binding = m.get('library_binding', {})
    pointer = binding.get('inventory', {})
    if (binding.get('target') != str(LIBRARY_TARGET)
            or binding.get('snapshot_directory') != str(LIBRARY_SNAPSHOT)
            or binding.get('full_directory_copy') is not True
            or binding.get('locked_regular_files') != 4828
            or pointer.get('path') != str(LIBRARY_INVENTORY.relative_to(ROOT))
            or sha(LIBRARY_INVENTORY) != require_sha(pointer.get('sha256'))
            or m.get('individual_file_bind_count') != 1590 or m.get('directory_bind_count') != 1
            or m.get('total_read_only_bind_count') != 1591):
        raise ValueError('exact authorized complete public library binding required')
    inv = json.loads(LIBRARY_INVENTORY.read_text())
    if (inv.get('schema') != 'r10_complete_public_library_directory_inventory_v820'
            or inv.get('complete') is not True or inv.get('source_before') != inv.get('source_after')
            or inv.get('source_full_inventory_and_stat_stable') is not True
            or inv.get('symlinks_preserved_without_following') is not True
            or inv.get('authorized_source_directory') != str(LIBRARY_TARGET)
            or inv.get('snapshot_directory') != str(LIBRARY_SNAPSHOT)):
        raise ValueError('complete stable source/copy library inventory required')
    desired = content_inventory(inv['source_before'])
    old = 'libXtst.so.6.1.0'
    frozen_sha = expected[str(LIBRARY_TARGET/old)]
    desired['entries'][old]['sha256'] = frozen_sha
    desired['entries'][old]['bytes'] = (SNAPSHOT/'files/usr/lib/x86_64-linux-gnu'/old).stat().st_size
    actual = inventory_tree(LIBRARY_SNAPSHOT)
    if (content_inventory(actual) != desired or content_inventory(actual) != content_inventory(inv['snapshot'])):
        raise ValueError('complete library snapshot bytes, entries or symlink targets changed')
    locked = {name:checksum for name,checksum in expected.items() if name.startswith(str(LIBRARY_TARGET)+'/')}
    if len(locked) != 4828:
        raise ValueError('exact reviewed library partition required')
    for name,checksum in locked.items():
        relative = str(Path(name).relative_to(LIBRARY_TARGET))
        if actual['entries'][relative].get('sha256') != checksum:
            raise ValueError('frozen library target changed: '+name)


def validate_manifest(path, manifest_sha256):
    path = Path(path)
    if path != MANIFEST or path.is_symlink() or path.resolve(strict=True) != path:
        raise ValueError('exact isolated v820 snapshot manifest path required')
    if sha(path) != require_sha(manifest_sha256):
        raise ValueError('snapshot manifest SHA mismatch')
    expected, sources = frozen_os_files()
    m = json.loads(path.read_text())
    if (m.get('schema') != SCHEMA or m.get('complete') is not True or m.get('source_locks') != sources
            or m.get('frozen_OS_file_count') != len(expected)
            or m.get('bwrap') != dict(path=str(BWRAP), sha256=BWRAP_SHA)):
        raise ValueError('exact completed frozen OS snapshot manifest required')
    if sha(BWRAP) != BWRAP_SHA:
        raise ValueError('host bwrap SHA changed; fail closed')
    validate_library_binding(m, expected)
    entries = m.get('files')
    if not isinstance(entries, list) or len(entries) != len(expected):
        raise ValueError('complete frozen OS entry list required')
    seen = set()
    for row in entries:
        name = row['target']
        os_target(name)
        if name in seen or name not in expected or row['sha256'] != expected[name]:
            raise ValueError('duplicate, undeclared or changed OS target: ' + name)
        seen.add(name)
        relative = 'files/' + name.lstrip('/')
        snapshot = SNAPSHOT / relative
        if (row['snapshot_file'] != relative or snapshot.is_symlink()
                or snapshot.resolve(strict=True) != snapshot or not snapshot.is_file()):
            raise ValueError('canonical declared snapshot file required: ' + str(snapshot))
        before = snapshot.stat()
        if (before.st_size != row['bytes'] or sha(snapshot) != expected[name]
                or stamp(before) != stamp(snapshot.stat())):
            raise ValueError('snapshot bytes/identity changed: ' + name)
    if seen != set(expected):
        raise ValueError('exact frozen OS union required')
    return m


def bwrap_arguments(manifest, command):
    if not command or any(not isinstance(a, str) or '\0' in a for a in command):
        raise ValueError('explicit NUL-free command argv required; shell is never used')
    # --bind makes device nodes inaccessible (nodev). Rebind the same complete
    # host /dev with device access; do not create a replacement device tree.
    args = ['--bind', '/', '/', '--dev-bind', '/dev', '/dev']
    if manifest.get('library_binding'):
        args += ['--ro-bind', str(LIBRARY_SNAPSHOT), str(LIBRARY_TARGET)]
    for row in manifest['files']:
        if manifest.get('library_binding') and row['target'].startswith(str(LIBRARY_TARGET)+'/'):
            continue
        args += ['--ro-bind', str(SNAPSHOT/row['snapshot_file']), row['target']]
    args += ['--', *command]
    return args


def namespace_id(name, pid='self'):
    return os.readlink('/proc/' + str(pid) + '/ns/' + name)


def process_start_ticks(pid):
    return int(Path('/proc/' + str(pid) + '/stat').read_text().rsplit(')', 1)[1].split()[19])


def namespace_context(manifest_sha256):
    try:
        marker = json.loads(os.environ[MARKER])
    except (KeyError, ValueError) as error:
        raise ValueError('explicit v820 outer-wrapper namespace marker required') from error
    wrapper = project_pointer(__file__)
    if (marker.get('schema') != 'r10_frozen_OS_outer_marker_v820'
            or marker.get('manifest') != project_pointer(MANIFEST, manifest_sha256)
            or marker.get('wrapper') != wrapper):
        raise ValueError('exact v820 manifest/wrapper namespace marker required')
    outer_pid = marker['outer_wrapper_pid']
    if process_start_ticks(outer_pid) != marker['outer_wrapper_start_ticks']:
        raise ValueError('live original outer wrapper identity required')
    actual = {kind: namespace_id(kind) for kind in ('mnt', 'pid', 'net')}
    reference = marker['reference_namespaces']
    try:
        live_outer_mnt = namespace_id('mnt', outer_pid)
        outer_namespace_read = dict(available=True, mount_namespace=live_outer_mnt)
        if live_outer_mnt != reference['mnt']:
            raise ValueError('outer wrapper mount namespace changed')
    except PermissionError:
        # Rootless bwrap automatically enters a user namespace. Linux can deny
        # cross-user-namespace /proc/PID/ns readlink even while public stat is
        # visible. Retain the pre-entry marker; never claim a live read occurred.
        outer_namespace_read = dict(available=False, reason='EACCES reading live outer /proc/PID/ns/mnt',
            reference_source='outer-wrapper marker captured before bwrap')
    if (actual['mnt'] == reference['mnt']
            or any(actual[kind] != reference[kind] for kind in ('pid', 'net'))
            or os.getuid() != marker['uid'] or os.getgid() != marker['gid']):
        raise ValueError('private mount, shared host PID/network and original uid/gid required')
    return marker, wrapper, actual, reference, outer_namespace_read


def require_environment(manifest_path, manifest_sha256):
    """Read-only proof of the *current* restored namespace, never just storage.

    Callers separately freeze this module and the initial CPU proof in their new
    control inputs. The marker is checked against the live outer wrapper process;
    it is not treated as a substitute for reading all current OS target bytes.
    """
    m = validate_manifest(Path(manifest_path), manifest_sha256)
    marker, wrapper, actual, reference, outer_namespace_read = namespace_context(manifest_sha256)
    actual_hashes = {}
    for row in m['files']:
        target = os_target(row['target'])
        first = target.stat()
        actual_hashes[str(target)] = sha(target)
        if (actual_hashes[str(target)] != row['sha256']
                or stamp(first) != stamp(target.stat())):
            raise ValueError('actual namespace frozen OS target SHA changed: ' + str(target))
    union_sha = hashlib.sha256(json.dumps(actual_hashes, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return dict(schema='r10_actual_frozen_OS_namespace_receipt_v820', complete=True,
        checked_utc=utc(), manifest=marker['manifest'], wrapper=wrapper,
        source_locks=m['source_locks'], current_mount_namespace=actual['mnt'],
        mount_namespace=actual['mnt'], pid=os.getpid(),
        reference_mount_namespace=reference['mnt'], current_namespaces=actual,
        live_outer_namespace_read=outer_namespace_read,
        outer_wrapper_pid=marker['outer_wrapper_pid'], outer_wrapper_start_ticks=marker['outer_wrapper_start_ticks'],
        observer_pid=os.getpid(), observer_start_ticks=process_start_ticks(os.getpid()),
        uid=os.getuid(), gid=os.getgid(), checked_OS_files=len(actual_hashes),
        all_current_OS_targets_full_byte_SHA_verified=True,
        current_OS_target_SHA_map_sha256=union_sha,
        restored_four_target_SHA256={name: actual_hashes[name] for name in sorted(RESTORED_TARGETS)},
        project_writable=os.access(ROOT, os.W_OK), host_PID_network_namespace_preserved=True,
        proc_or_dev_replaced=False, same_host_dev_directory_rebound_with_device_access=True,
        task_parameters_or_strategy_changed=False)


# Stable spelling shared with new control/admission callers.
require_namespace = require_environment


def capture_namespace_evidence(manifest_path, expected_sha, qualification_path=None, qualification_sha=None):
    """Capture runtime evidence, optionally bind the independently frozen CPU proof."""
    evidence = require_environment(manifest_path, expected_sha)
    if (qualification_path is None) != (qualification_sha is None):
        raise ValueError('CPU qualification path and SHA must be supplied together')
    if qualification_path is not None:
        path = Path(qualification_path)
        if path != SETUP/'cpu_namespace_proof_v820.json' or sha(path) != require_sha(qualification_sha):
            raise ValueError('exact frozen v820 CPU namespace proof required')
        proof = json.loads(path.read_text())
        if (proof.get('schema') != 'r10_frozen_OS_CPU_namespace_proof_v820' or proof.get('complete') is not True
                or proof.get('manifest') != evidence['manifest'] or proof.get('wrapper') != evidence['wrapper']
                or proof.get('host_mount_namespace') != evidence['reference_mount_namespace']
                or evidence['current_mount_namespace'] == proof.get('host_mount_namespace')):
            raise ValueError('actual namespace must match the independently qualified frozen environment')
        evidence['namespace_qualification'] = project_pointer(path, qualification_sha)
        evidence['qualified_host_mount_namespace'] = proof['host_mount_namespace']
    return evidence


def argument_memfd():
    if hasattr(os, 'memfd_create'):
        return os.memfd_create('r10-v820-bwrap-args', flags=os.MFD_CLOEXEC)
    # The project's older CPython lacks os.memfd_create. Call only the Linux
    # libc file-descriptor API; no simulator/native binding is imported.
    import ctypes
    libc = ctypes.CDLL(None, use_errno=True)
    create = libc.memfd_create
    create.argtypes = (ctypes.c_char_p, ctypes.c_uint)
    create.restype = ctypes.c_int
    fd = create(b'r10-v820-bwrap-args', 1)  # Linux MFD_CLOEXEC
    if fd < 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    return fd


def run_namespace(manifest, command):
    args = bwrap_arguments(manifest, command)
    if len(args) >= 9000:
        raise ValueError('bwrap expanded argument count must stay below its 9000 limit')
    # bwrap --args consumes NUL-separated argv from an inherited file descriptor.
    # memfd avoids ARG_MAX and does not create or modify a system/cache file.
    fd = argument_memfd()
    try:
        # --args parses bwrap options only; the command must remain after the
        # outer argv's --. Keeping it there also preserves its literal argv.
        options = args[:args.index('--')]
        data = b'\0'.join(os.fsencode(a) for a in options) + b'\0'
        with os.fdopen(os.dup(fd), 'wb') as f:
            f.write(data)
        os.lseek(fd, 0, os.SEEK_SET)
        env = dict(os.environ)
        env[MARKER] = json.dumps(dict(schema='r10_frozen_OS_outer_marker_v820',
                manifest=project_pointer(MANIFEST), wrapper=project_pointer(__file__),
                outer_wrapper_pid=os.getpid(), outer_wrapper_start_ticks=process_start_ticks(os.getpid()),
                reference_namespaces={kind: namespace_id(kind) for kind in ('mnt', 'pid', 'net')},
                mount_stages=1, complete_library_directory_binds=1, individual_file_binds=1590,
                uid=os.getuid(), gid=os.getgid()), sort_keys=True)
        return subprocess.run([str(BWRAP), '--args', str(fd), '--', *command],
                              pass_fds=(fd,), shell=False, env=env).returncode
    finally:
        os.close(fd)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--create-snapshot', action='store_true', help='CPU-only creation; refuses existing snapshot')
    p.add_argument('--manifest', default=str(MANIFEST))
    p.add_argument('--manifest-sha256')
    p.add_argument('--check-only', action='store_true', help='validate bytes without launching any command')
    p.add_argument('command', nargs=argparse.REMAINDER)
    args = p.parse_args(argv)
    if args.create_snapshot:
        if args.command or args.check_only or args.manifest_sha256 or Path(args.manifest) != MANIFEST:
            p.error('--create-snapshot cannot be combined with execution or alternate manifest arguments')
        create_snapshot()
        return 0
    if not args.manifest_sha256:
        p.error('--manifest-sha256 is required')
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if args.check_only and command:
        p.error('--check-only cannot execute a command')
    if not args.check_only and not command:
        p.error('explicit command argv required after --')
    m = validate_manifest(Path(args.manifest), args.manifest_sha256)
    if args.check_only:
        print(json.dumps(dict(complete=True, checked_OS_files=len(m['files']),
            manifest=str(MANIFEST), manifest_sha256=args.manifest_sha256, command_started=False)))
        return 0
    return run_namespace(m, command)


if __name__ == '__main__':
    raise SystemExit(main())
