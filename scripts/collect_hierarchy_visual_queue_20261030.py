"""Finish the fixed four-case TRAIN-only visual annotation capture, serially.

Completed cases are verified and retained; failed attempts are never retried.
Each fresh case is a separate frozen-environment process with its own Xvfb.
"""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from our_mani.hierarchy_20261030.native_capture import (
    ARTIFACTS, RUNTIME, atomic_json, pointer, verify_plan,
)


def main():
    plan_path = ROOT / ARTIFACTS / "plan_v1.json"
    plan = verify_plan(ROOT, plan_path)
    output = ROOT / RUNTIME / "queue_v1"
    output.mkdir(exist_ok=False)
    state = dict(schema="hierarchy_fixed_visual_TRAIN_queue_20261030_v1",
                 plan=pointer(ROOT, plan_path), source=pointer(ROOT, Path(__file__)),
                 pid=os.getpid(), stage="running", complete=False, active=None,
                 completed=[], preserved=[], failed=[], automatic_retry=False,
                 TEST_read=False, validation_read=False)
    atomic_json(output/"status.json", state)
    for case in plan["cases"]:
        key = case["case_key"]
        terminal_path = ROOT / RUNTIME / "cases" / key / "terminal.json"
        if terminal_path.exists():
            terminal = json.loads(terminal_path.read_text())
            if not terminal.get("complete") or not terminal.get("natural_20_tick_capture"):
                raise RuntimeError("preserve and audit previous failed attempt: " + key)
            state["preserved"].append(dict(case_key=key, terminal=pointer(ROOT, terminal_path)))
            atomic_json(output/"status.json", state)
            continue
        if (ROOT/RUNTIME/"STOP").exists():
            state.update(stage="administrative_stop", active=None)
            atomic_json(output/"status.json", state)
            return 2
        command = [str(ROOT/".conda-env/bin/python"), "-B",
                   str(ROOT/"scripts/run_frozen_system_environment_v820_r10.py"),
                   "--manifest-sha256", plan["frozen_environment"]["sha256"], "--",
                   "bash", "scripts/run_local_sim.sh",
                   "scripts/collect_hierarchy_visual_train_20261030.py", "--run-case", key]
        state.update(active=key, updated_utc=datetime.now(timezone.utc).isoformat())
        atomic_json(output/"status.json", state)
        with (output/(key+".log")).open("x") as stream:
            result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode != 0 or not terminal_path.exists():
            state["failed"].append(dict(case_key=key, returncode=result.returncode))
            state.update(stage="failed", active=None)
            atomic_json(output/"status.json", state)
            return 1
        terminal = json.loads(terminal_path.read_text())
        if not terminal.get("complete") or not terminal.get("natural_20_tick_capture"):
            state["failed"].append(dict(case_key=key, terminal=pointer(ROOT, terminal_path)))
            state.update(stage="failed", active=None)
            atomic_json(output/"status.json", state)
            return 1
        state["completed"].append(dict(case_key=key, terminal=pointer(ROOT, terminal_path)))
        state.update(active=None)
        atomic_json(output/"status.json", state)
    state.update(stage="complete", complete=True, active=None,
                 finished_utc=datetime.now(timezone.utc).isoformat())
    atomic_json(output/"status.json", state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
