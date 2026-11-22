"""One new bwrap: original frozen OS bytes plus a private worker network.

The original v820 module, manifest, mount materials and physics remain unchanged.
This new contract explicitly records that network isolation differs from v820.
Serial --shared-network invokes the original run_namespace and original marker.
No failed namespace operation silently falls back to another environment.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_frozen_system_environment_v820_r10 as original


MARKER = "OUR_MANI_BLOCK_FROZEN_OS_NETWORK_20261102"
SCHEMA = "hierarchy_block_frozen_OS_private_network_20261102_v1"
SOURCE = Path("scripts/run_hierarchy_frozen_environment_block_20261102.py")


def _identity(pid):
    fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    if fields[0] in ("Z", "X"):
        raise ValueError("original environment wrapper process is not live")
    return dict(pid=pid, start_ticks=int(fields[19]), boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip())


def _pointer(path):
    path = Path(path).resolve(strict=True)
    return dict(path=str(path.relative_to(ROOT)), sha256=original.sha(path))


def _dev():
    directory, null = Path("/dev").stat(), Path("/dev/null").stat()
    return dict(device=directory.st_dev, inode=directory.st_ino, null_rdev=null.st_rdev)


def namespace_context(manifest_sha256):
    """New attestation: private mount AND network, same PID/uid/dev surfaces."""
    try:
        marker = json.loads(os.environ[MARKER])
    except (KeyError, ValueError) as error:
        raise ValueError("explicit new block private-network environment marker required") from error
    wrapper = _pointer(ROOT / SOURCE)
    manifest = dict(path=str(original.MANIFEST.relative_to(ROOT)), sha256=manifest_sha256)
    if (marker.get("schema") != SCHEMA or marker.get("manifest") != manifest
            or marker.get("wrapper") != wrapper or marker.get("original_wrapper") != _pointer(original.__file__)
            or original.sha(original.MANIFEST) != manifest_sha256):
        raise ValueError("new wrapper/original material SHA-bound attestation differs")
    owner = marker["outer_process"]
    if _identity(owner["pid"]) != owner:
        raise ValueError("original new environment wrapper process exited or PID was reused")
    actual = {kind: os.readlink("/proc/self/ns/"+kind) for kind in ("mnt", "pid", "net")}
    reference = marker["reference_namespaces"]
    if (actual["mnt"] == reference["mnt"] or actual["net"] == reference["net"]
            or actual["pid"] != reference["pid"] or os.getuid() != marker["uid"]
            or os.getgid() != marker["gid"] or _dev() != marker["dev"]):
        raise ValueError("private mount/network with unchanged PID/uid/gid/dev surfaces required")
    try:
        observed = {kind: os.readlink(f"/proc/{owner['pid']}/ns/{kind}") for kind in ("mnt", "pid", "net")}
        if observed != reference:
            raise ValueError("outer environment wrapper namespaces changed")
        outer_read = dict(available=True, namespaces=observed)
    except PermissionError:
        outer_read = dict(available=False, reason="EACCES reading outer namespaces across rootless user namespace",
                          reference_source="new wrapper capture before its sole bwrap")
    return marker, wrapper, actual, reference, outer_read


require_namespace = namespace_context


def require_environment(manifest_path, manifest_sha256):
    """Read-only full current OS-target-byte verification under the new contract."""
    manifest = original.validate_manifest(Path(manifest_path), manifest_sha256)
    marker, wrapper, actual, reference, outer_read = namespace_context(manifest_sha256)
    verified = {}
    for row in manifest["files"]:
        path = original.os_target(row["target"])
        before = path.stat()
        digest = original.sha(path)
        if digest != row["sha256"] or original.stamp(before) != original.stamp(path.stat()):
            raise ValueError("current frozen OS bytes changed: "+str(path))
        verified[str(path)] = digest
    return dict(schema=SCHEMA+"_actual_environment", complete=True, wrapper=wrapper, manifest=marker["manifest"],
        current_namespaces=actual, reference_namespaces=reference, live_outer_namespace_read=outer_read,
        observer_process=_identity(os.getpid()), checked_OS_files=len(verified),
        all_current_OS_targets_full_byte_SHA_verified=True, network_namespace_isolated=True,
        host_network_preserved=False, host_PID_namespace_preserved=True, original_uid_gid_dev_preserved=True,
        frozen_mount_materials_unchanged=True, physical_parameters_changed=False)


def run_namespace(manifest, command):
    arguments = original.bwrap_arguments(manifest, ["xvfb-run", "-a", *command])
    options = ["--unshare-net", *arguments[:arguments.index("--")]]
    fd = original.argument_memfd()
    try:
        with os.fdopen(os.dup(fd), "wb") as stream:
            stream.write(b"\0".join(os.fsencode(arg) for arg in options)+b"\0")
        os.lseek(fd, 0, os.SEEK_SET)
        env = dict(os.environ)
        env.pop(original.MARKER, None)
        env[MARKER] = json.dumps(dict(schema=SCHEMA, wrapper=_pointer(ROOT / SOURCE),
            original_wrapper=_pointer(original.__file__), manifest=_pointer(original.MANIFEST),
            outer_process=_identity(os.getpid()), reference_namespaces={kind: os.readlink("/proc/self/ns/"+kind) for kind in ("mnt", "pid", "net")},
            uid=os.getuid(), gid=os.getgid(), dev=_dev(), frozen_mount_materials_unchanged=True,
            host_network_preserved=False, network_isolation_requested=True), sort_keys=True)
        return subprocess.run([str(original.BWRAP), "--args", str(fd), "--", "xvfb-run", "-a", *command],
                              pass_fds=(fd,), env=env, shell=False).returncode
    finally:
        os.close(fd)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--shared-network", action="store_true", help="explicit serial original-v820 contract; never an automatic fallback")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("explicit command required")
    manifest = original.validate_manifest(original.MANIFEST, args.manifest_sha256)
    if args.shared_network:
        return original.run_namespace(manifest, ["xvfb-run", "-a", *command])
    return run_namespace(manifest, command)


if __name__ == "__main__":
    raise SystemExit(main())
