#!/usr/bin/env python3
"""Observe one fresh GPU trainer process while Native orchestration is prepared."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / "runtime/framework/hierarchy_block_20261102/training_process_v1"
STATUS = ROOT / "artifacts/framework/multitask_r10/hierarchy_block_20261102/training_v1/status.json"
CHECKPOINT = STATUS.parent / "best.pt"


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def identity(pid):
    fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    return dict(pid=pid, start_ticks=int(fields[19]), boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip())


def write(name, payload):
    with (DIRECTORY / name).open("x") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def main():
    DIRECTORY.mkdir(parents=True, exist_ok=False)
    command = [sys.executable, "-B", "-u", "scripts/train_hierarchy_block_20261102.py", "--device", "cuda:0", "--workers", "4"]
    launch = dict(schema="hierarchy_block_external_training_process_20261102_v1", command=command,
                  started_utc=utc(), parent=identity(os.getpid()), automatic_retry=False,
                  launcher_source_sha256=sha(__file__))
    write("launch.json", launch)
    stopped = False
    with (DIRECTORY / "training.log").open("xb") as stream:
        process = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                   start_new_session=True, env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
        write("spawned.json", dict(process=identity(process.pid), observed_utc=utc()))

        def stop(*_):
            nonlocal stopped
            stopped = True
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass

        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        rc = process.wait()
    receipt = dict(schema="hierarchy_block_external_training_actual_exit_20261102_v1", command=command,
                   actual_process_exit_observed=True, actual_returncode=rc, returncode=rc,
                   timed_out=False, administrative_stop=stopped, finished_utc=utc(),
                   launch=dict(path=str((DIRECTORY / "launch.json").relative_to(ROOT)), sha256=sha(DIRECTORY / "launch.json")),
                   spawned_process=dict(path=str((DIRECTORY / "spawned.json").relative_to(ROOT)), sha256=sha(DIRECTORY / "spawned.json")),
                   log=dict(path=str((DIRECTORY / "training.log").relative_to(ROOT)), sha256=sha(DIRECTORY / "training.log")))
    if STATUS.exists():
        receipt["training_status"] = dict(path=str(STATUS.relative_to(ROOT)), sha256=sha(STATUS))
    if CHECKPOINT.exists():
        receipt["checkpoint"] = dict(path=str(CHECKPOINT.relative_to(ROOT)), sha256=sha(CHECKPOINT))
    write("exit.json", receipt)
    print(json.dumps(receipt), flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
