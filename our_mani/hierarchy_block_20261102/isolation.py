"""New-family process isolation and bounded Native resource leases.

No simulation, model, scientific admission, old STOP or physical parameter is
changed here. A new wrapper combines unchanged frozen OS mount materials and a
private network in one bwrap under its own explicit attestation. Every proof
below describes the actual current process/kernel resources.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import socket
import stat
import tempfile
import time


SCHEMA = "hierarchy_block_native_isolation_20261102_v1"
RUNTIME = Path("runtime/framework/hierarchy_block_20261102")
ARTIFACTS = Path("artifacts/framework/multitask_r10/hierarchy_block_20261102")
SHARED_LOCKS = ("runtime/framework/fixed_TRAIN104_geometry_native.lock",
                "runtime/framework/TRAIN_postcomplete_diagnostics_v1106.lock")
PRIVATE_NAMES = dict(TMPDIR="tmp", XDG_CACHE_HOME="cache", XDG_RUNTIME_DIR="xdg-runtime", XDG_CONFIG_HOME="config")
THREAD_ENV = dict(OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4")
MAX_SLOTS = 4
LOGICAL_CPUS_PER_SLOT = 8
_ACTIVE = {}


class IsolationError(RuntimeError):
    pass


def _require(condition, message):
    if not condition:
        raise IsolationError(message)


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def process_identity(pid=None):
    pid = os.getpid() if pid is None else pid
    _require(type(pid) is int and pid > 0, "actual positive PID required")
    fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    _require(fields[0] not in ("Z", "X"), "live process required")
    return dict(pid=pid, start_ticks=int(fields[19]), boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip())


def _path(root, value, family):
    raw = Path(value)
    path = root / raw if not raw.is_absolute() else raw
    _require(".." not in raw.parts and path.resolve(strict=False) == path, "canonical nonsymlink path required")
    _require(path.is_relative_to(root / family) and path != root / family, "new block namespace required")
    return path


def partition_cpus(allowed, slot):
    """Four disjoint eight-logical-CPU slots from the actual allowed CPU set."""
    _require(type(slot) is int and 0 <= slot < MAX_SLOTS, "slot must be an integer in 0..3")
    allowed = sorted(set(allowed))
    _require(len(allowed) >= MAX_SLOTS*LOGICAL_CPUS_PER_SLOT, "32 allowed logical CPUs required for the explicit four-slot profile")
    _require(all(type(cpu) is int and cpu >= 0 for cpu in allowed), "valid allowed CPU identifiers required")
    return allowed[slot*LOGICAL_CPUS_PER_SLOT:(slot+1)*LOGICAL_CPUS_PER_SLOT]


def _write_new(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, allow_nan=False)
        stream.write("\n")


def prepare_shell(root, case_runtime, slot, shell_pid, command):
    """CPU-only pre-import environment setup called by the new outer shell."""
    root = Path(root).resolve(strict=True)
    case_runtime = _path(root, case_runtime, RUNTIME)
    _require("cases" in case_runtime.relative_to(root / RUNTIME).parts, "explicit case runtime required")
    case_runtime.mkdir(parents=True, exist_ok=True)
    allowed = sorted(os.sched_getaffinity(0))
    assigned = partition_cpus(allowed, slot)
    paths = {key: str(case_runtime / name) for key, name in PRIVATE_NAMES.items()}
    for path in paths.values():
        Path(path).mkdir(mode=0o700, exist_ok=False)
    script = root / "scripts/run_hierarchy_isolated_block_20261102.sh"
    proof = dict(schema=SCHEMA+"_outer_shell", case_runtime=str(case_runtime), slot=slot,
        outer_process=process_identity(shell_pid), host_network_namespace=os.readlink("/proc/self/ns/net"),
        allowed_logical_cpus=allowed, assigned_cpus=assigned, private_paths=paths,
        thread_environment=THREAD_ENV, CUDA_VISIBLE_DEVICES="", command=list(command),
        outer_shell_sha256=_sha(script), prepared_monotonic=time.monotonic(), Native_started=False)
    _write_new(case_runtime / "shell_environment.json", proof)
    return ",".join(map(str, assigned))


def _ancestor(identity):
    current, seen = os.getpid(), set()
    while current > 1 and current not in seen:
        seen.add(current)
        if current == identity["pid"]:
            return process_identity(current) == identity
        fields = Path(f"/proc/{current}/stat").read_text().rsplit(")", 1)[1].split()
        current = int(fields[1])
    return False


def _validate_shell(root, case_runtime, slot, *, parallel):
    path = case_runtime / "shell_environment.json"
    before = _sha(path)
    proof = json.loads(path.read_text())
    _require(_sha(path) == before and proof.get("schema") == SCHEMA+"_outer_shell"
             and proof.get("case_runtime") == str(case_runtime) and proof.get("slot") == slot,
             "matching real outer-shell receipt required")
    _require(_ancestor(proof["outer_process"]), "live original outer shell/bwrap ancestor required")
    _require(proof.get("outer_shell_sha256") == _sha(root / "scripts/run_hierarchy_isolated_block_20261102.sh"), "outer shell source changed")
    _require(proof.get("assigned_cpus") == partition_cpus(proof["allowed_logical_cpus"], slot), "assigned CPU partition differs")
    current = os.readlink("/proc/self/ns/net")
    _require((current != proof["host_network_namespace"]) if parallel else (current == proof["host_network_namespace"]),
             "actual worker network namespace differs from its explicit parallel/serial contract")
    paths = {key: str(case_runtime / name) for key, name in PRIVATE_NAMES.items()}
    _require(proof.get("private_paths") == paths, "private path receipt differs")
    for key, value in paths.items():
        target = Path(value)
        _require(os.environ.get(key) == value and target.resolve(strict=True) == target and target.is_dir()
                 and target.stat().st_uid == os.getuid() and stat.S_IMODE(target.stat().st_mode) == 0o700,
                 "actual private environment path differs: " + key)
    _require(all(os.environ.get(key) == value for key, value in THREAD_ENV.items())
             and os.environ.get("CUDA_VISIBLE_DEVICES") == "", "preserved 4/4/4 CPU-only launcher environment required")
    _require(str(Path(tempfile.gettempdir()).resolve()) == paths["TMPDIR"], "Python temporary directory was cached before private setup")
    return proof, before


def _affinity_evidence(assigned):
    assigned = set(assigned)
    main = sorted(os.sched_getaffinity(0))
    _require(set(main) <= assigned and bool(main), "main thread escaped assigned CPU partition")
    threads, vanished = [], []
    for entry in Path("/proc/self/task").iterdir():
        try:
            mask = sorted(os.sched_getaffinity(int(entry.name)))
        except ProcessLookupError:
            vanished.append(int(entry.name)); continue
        _require(bool(mask) and set(mask) <= assigned, "worker thread escaped assigned CPU partition")
        threads.append(dict(tid=int(entry.name), cpus=mask))
    return dict(assigned_cpus=sorted(assigned), main_thread_cpus=main, threads=threads,
                vanished_thread_ids=vanished, all_observed_threads_within_partition=True)


def _display_evidence():
    display = os.environ.get("DISPLAY", "")
    match = re.fullmatch(r":([0-9]+)(?:\.0)?", display)
    _require(match is not None, "actual local independent Xvfb DISPLAY required")
    number = match.group(1)
    lock, unix = Path(f"/tmp/.X{number}-lock"), Path(f"/tmp/.X11-unix/X{number}")
    pid = int(lock.read_text().strip())
    identity = process_identity(pid)
    command = Path(f"/proc/{pid}/cmdline").read_bytes().decode(errors="replace").split("\0")
    _require(any(Path(arg).name == "Xvfb" for arg in command) and ":"+number in command,
             "DISPLAY is not backed by the corresponding live Xvfb process")
    _require(stat.S_ISSOCK(unix.stat().st_mode), "actual X11 UNIX socket missing")
    return dict(display=display, server=identity, command=command, unix_socket=str(unix),
                socket_device=unix.stat().st_dev, socket_inode=unix.stat().st_ino)


def _address(raw, family):
    address, port = raw.split(":")
    packed = bytes.fromhex(address)
    if family == "tcp":
        packed = packed[::-1]
        host = socket.inet_ntop(socket.AF_INET, packed)
    else:
        packed = b"".join(packed[i:i+4][::-1] for i in range(0, 16, 4))
        host = socket.inet_ntop(socket.AF_INET6, packed)
    return host, int(port, 16)


def listener_snapshot():
    """Join this process's actual socket FDs to kernel TCP/TCP6 LISTEN rows.

    /proc/self/net alone is namespace-wide; ownership is proven by FD inode.
    This records observed sockets, not a claim that plugins expose no ports.
    """
    owned = {}
    vanished = []
    for fd in Path("/proc/self/fd").iterdir():
        try:
            target = os.readlink(fd)
        except FileNotFoundError:
            vanished.append(int(fd.name)); continue
        match = re.fullmatch(r"socket:\[([0-9]+)\]", target)
        if match:
            owned.setdefault(int(match.group(1)), []).append(int(fd.name))
    listeners = []
    for family in ("tcp", "tcp6"):
        for line in Path("/proc/self/net/"+family).read_text().splitlines()[1:]:
            values = line.split()
            if len(values) < 10 or values[3] != "0A":
                continue
            inode = int(values[9])
            if inode not in owned:
                continue
            address, port = _address(values[1], family)
            listeners.append(dict(family=family, local_address=address, local_port=port, inode=inode,
                                  fds=sorted(owned[inode]), network_namespace=os.readlink("/proc/self/ns/net")))
    return dict(complete=True, process=process_identity(), network_namespace=os.readlink("/proc/self/ns/net"),
                listeners=sorted(listeners, key=lambda value: (value["family"], value["local_port"], value["inode"])),
                vanished_fd_ids=vanished, ownership="socket inode joined to /proc/self/fd", scope="current process TCP/TCP6 LISTEN only")


def _baseline_snapshot():
    repo = Path("/home/ubuntu/桌面/phsimani/PhysMani")
    simulator = Path(os.environ["COPPELIASIM_ROOT"])
    paths = [repo / "third_party/RLBench/rlbench/task_design.ttt",
             *(repo / ("third_party/RLBench/rlbench/task_ttms/"+task+".ttm") for task in
               ("pick_moving_target_on_the_table", "pick_moving_target_on_the_table_high_speed")),
             simulator / "remoteApiConnections.txt", simulator / "system/usrset.txt"]
    return {str(path.resolve(strict=True)): _sha(path) for path in paths}


def _lock_proof(fd, path, mode):
    before, actual = path.stat(), os.fstat(fd)
    _require((before.st_dev, before.st_ino) == (actual.st_dev, actual.st_ino), "lock path was replaced")
    expected = "READ" if mode == fcntl.LOCK_SH else "WRITE"
    lines = Path(f"/proc/self/fdinfo/{fd}").read_text().splitlines()
    matches = []
    for line in lines:
        parts = line.split()
        if len(parts) != 9 or parts[0] != "lock:" or parts[2:5] != ["FLOCK", "ADVISORY", expected]:
            continue
        device = parts[6].split(":")
        if len(device) == 3 and int(parts[5]) == os.getpid() and parts[7:] == ["0", "EOF"] and (
            int(device[0], 16), int(device[1], 16), int(device[2])) == (os.major(actual.st_dev), os.minor(actual.st_dev), actual.st_ino):
            matches.append(line)
    _require(len(matches) == 1, "actual per-FD kernel flock missing")
    return dict(path=str(path), fd=fd, device=actual.st_dev, inode=actual.st_ino, mode=expected, kernel_row=matches[0])


def assert_live(receipt):
    """Optional physics-observer check; no extra physics/model calls occur."""
    context = _ACTIVE.get(id(receipt))
    _require(context is not None and process_identity() == receipt["process"], "same live resource lease required")
    _require(os.readlink("/proc/self/ns/net") == receipt["network_namespace"], "worker network namespace changed")
    _require(_sha(context["runtime"] / "shell_environment.json") == context["shell_sha"], "outer-shell evidence changed")
    _require(_ancestor(context["shell"]["outer_process"]), "original outer launcher exited")
    _require(os.environ.get("DISPLAY") == receipt["display"], "worker DISPLAY changed")
    for key, value in receipt["private_paths"].items():
        _require(os.environ.get(key) == value, "private process path changed: " + key)
    receipt["affinity_latest"] = _affinity_evidence(context["shell"]["assigned_cpus"])
    return [_lock_proof(fd, path, mode) for fd, path, mode in context["locks"]]


def record_native_interval(receipt, phase):
    _require(phase in ("start", "end"), "interval phase must be start or end")
    assert_live(receipt)
    interval = receipt["native_interval"]
    key = phase+"_monotonic"
    _require(key not in interval, "native interval phase cannot be recorded twice")
    _require(phase == "start" or "start_monotonic" in interval, "native interval start required before end")
    interval[key] = time.monotonic()
    audit = listener_snapshot()
    receipt["listener_audit_complete"] &= audit["complete"]
    receipt["listener_"+phase] = audit
    if phase == "start":
        receipt["listeners_active"] = audit["listeners"]
        receipt["listener_active"] = audit
    return interval


def validate_receipt(receipt, *, require_native_interval=True):
    """Fail closed on resource/cleanup errors, independently of Native outcome."""
    _require(receipt.get("schema") == SCHEMA and receipt.get("complete") is True
             and receipt.get("restored") is True and receipt.get("errors") == [], "incomplete/failed resource receipt")
    _require(receipt.get("listener_audit_complete") is True and (receipt.get("netnamespace_isolated") is True or receipt.get("parallel") is False)
             and receipt.get("all_threads_within_assignment") is True, "kernel listener/network/CPU audit incomplete")
    _require(type(receipt.get("parallel")) is bool and
             ((receipt["parallel"] is True and receipt.get("netnamespace_isolated") is True and receipt.get("netnamespace_preserved") is False)
              or (receipt["parallel"] is False and receipt.get("netnamespace_isolated") is False and receipt.get("netnamespace_preserved") is True)),
             "explicit parallel-private/serial-shared network contract required")
    if require_native_interval:
        interval = receipt.get("native_interval", {})
        start, end = interval.get("start_monotonic"), interval.get("end_monotonic")
        _require(type(start) in (float, int) and type(end) in (float, int) and math.isfinite(start) and math.isfinite(end)
                 and end >= start, "actual native start/end interval required")
    return receipt


@contextmanager
def resource_scope(root, case_runtime, *, slot=0, parallel=False, case_output):
    """Whole-lifetime EX case/output/slot leases; new parallel workers share guards.

    Both historical global guards are SH only on this explicit new parallel
    path, so any old EX owner still excludes all new workers. Four globally
    shared slot files cap every new owner together. Serial mode retains old EX.
    """
    _require(type(parallel) is bool, "parallel must be an explicit boolean")
    _require(type(slot) is int and 0 <= slot < MAX_SLOTS, "slot must be integer 0..3")
    root = Path(root).resolve(strict=True)
    runtime, output = _path(root, case_runtime, RUNTIME), _path(root, case_output, ARTIFACTS)
    _require("cases" in runtime.relative_to(root / RUNTIME).parts and "cases" in output.relative_to(root / ARTIFACTS).parts,
             "independent case runtime and output namespaces required")
    receipt = dict(schema=SCHEMA, complete=False, restored=False, errors=[], slot=slot, parallel=parallel,
        process=process_identity(), case_runtime=str(runtime), case_output=str(output), private_paths={},
        display=os.environ.get("DISPLAY"), network_namespace=os.readlink("/proc/self/ns/net"),
        net_namespace_id=os.stat("/proc/self/ns/net").st_ino, native_interval={}, listeners_active=[], listener_audit_complete=False)
    locks = []
    old_tempdir = tempfile.tempdir
    context = dict(locks=locks, runtime=runtime)
    failure = None
    try:
        shell, shell_sha = _validate_shell(root, runtime, slot, parallel=parallel)
        context.update(shell=shell, shell_sha=shell_sha)
        receipt.update(private_paths=shell["private_paths"], outer_shell=shell,
            host_network_namespace=shell["host_network_namespace"], network_namespace_isolated=parallel,
            netnamespace_isolated=parallel, netnamespace_preserved=not parallel,
            assigned_cpus=shell["assigned_cpus"], all_threads_within_assignment=True,
            affinity_initial=_affinity_evidence(shell["assigned_cpus"]), display_evidence=_display_evidence(),
            baseline_sources_before=_baseline_snapshot(), baseline_audit_scope="two task TTM, RLBench scene, simulator remoteAPI and user defaults only; not all filesystem writes")
        gate = root / RUNTIME / "isolation_v1"
        output_key = hashlib.sha256(str(output.relative_to(root)).encode()).hexdigest()
        display_key = hashlib.sha256(receipt["display"].encode()).hexdigest()
        specs = [(root / name, fcntl.LOCK_SH if parallel else fcntl.LOCK_EX) for name in SHARED_LOCKS]
        specs += [(gate / "slots" / f"slot_{slot}.lock", fcntl.LOCK_EX),
                  (runtime / ".native_lease.lock", fcntl.LOCK_EX),
                  (gate / "outputs" / (output_key+".lock"), fcntl.LOCK_EX),
                  (gate / "displays" / (display_key+".lock"), fcntl.LOCK_EX)]
        for path, mode in specs:
            path.parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_CLOEXEC | os.O_NOFOLLOW, 0o600)
            try:
                fcntl.flock(fd, mode | fcntl.LOCK_NB)
            except BaseException:
                os.close(fd); raise
            locks.append((fd, path, mode))
        _ACTIVE[id(receipt)] = context
        receipt["lock_proofs"] = assert_live(receipt)
        before = listener_snapshot()
        receipt.update(listener_before=before, listener_audit_complete=before["complete"])
        yield receipt
        receipt["lock_proofs_final"] = assert_live(receipt)
        receipt["baseline_sources_after"] = _baseline_snapshot()
        _require(receipt["baseline_sources_after"] == receipt["baseline_sources_before"], "audited baseline or scene bytes changed during case")
    except BaseException as error:
        failure = error
        receipt["errors"].append(dict(type=type(error).__name__, message=str(error)))
        raise
    finally:
        try:
            after = listener_snapshot()
            receipt["listener_after"] = after
            receipt["listener_audit_complete"] = bool(receipt["listener_audit_complete"] and after["complete"])
        except BaseException as error:
            receipt["errors"].append(dict(type=type(error).__name__, message="final listener audit: "+str(error)))
        _ACTIVE.pop(id(receipt), None)
        released = True
        for fd, path, mode in reversed(locks):
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            except BaseException as error:
                released = False
                receipt["errors"].append(dict(type=type(error).__name__, message="lease cleanup: "+str(error)))
            finally:
                try:
                    os.close(fd)
                except BaseException as error:
                    released = False
                    receipt["errors"].append(dict(type=type(error).__name__, message="descriptor cleanup: "+str(error)))
        tempfile.tempdir = old_tempdir
        receipt["restored"] = released
        receipt["complete"] = not receipt["errors"] and receipt["listener_audit_complete"]
        if failure is None:
            validate_receipt(receipt, require_native_interval=False)


def main():
    parser = argparse.ArgumentParser(description="CPU-only outer-shell preparation")
    parser.add_argument("--prepare-shell", action="store_true", required=True)
    parser.add_argument("root")
    parser.add_argument("case_runtime")
    parser.add_argument("slot", type=int)
    parser.add_argument("shell_pid", type=int)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    print(prepare_shell(args.root, args.case_runtime, args.slot, args.shell_pid, command))


if __name__ == "__main__":
    main()
