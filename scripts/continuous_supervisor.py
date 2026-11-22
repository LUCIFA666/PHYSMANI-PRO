"""Keep the independent data and training workers alive until explicitly stopped."""
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.continuous_common import DATA, STATE, atomic_json, event, read_json, setup, StopFlag


def main():
    setup()
    stop = StopFlag()
    env = dict(os.environ, PYTHONUNBUFFERED="1", OMP_NUM_THREADS="4",
               MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4",
               MPLCONFIGDIR=str(ROOT / ".cache/matplotlib"))
    workers = {
        "data": {"script": "continuous_data.py"},
        "trainer": {"script": "continuous_train.py"},
    }
    for worker in workers.values():
        worker.update(process=None, restarts=0, failures=0, next_start=0., started=0.)
    while not stop.requested():
        now = time.monotonic()
        for role, worker in workers.items():
            process = worker["process"]
            if process is not None and process.poll() is not None:
                code = process.returncode
                if code == 0 or now - worker["started"] > 300:
                    worker["failures"] = 0
                worker["failures"] += 1
                worker["restarts"] += 1
                delay = min(300, 5 * 2**min(worker["failures"], 6))
                worker.update(process=None, next_start=now + delay, last_exit=code)
                event("supervisor", "worker_exited", worker=role, exit_code=code, restart_in_seconds=delay)
            if worker["process"] is None and now >= worker["next_start"]:
                log = ROOT / "logs/continuous" / f"{role}.log"
                if log.exists() and log.stat().st_size > 64 * 1024**2:
                    os.replace(log, log.with_suffix(".previous.log"))
                with log.open("ab", buffering=0) as stream:
                    worker["process"] = subprocess.Popen(
                        [sys.executable, "-u", str(ROOT / "scripts" / worker["script"])],
                        cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                        env=env, start_new_session=True,
                    )
                worker["started"] = now
                event("supervisor", "worker_started", worker=role, pid=worker["process"].pid,
                      restarts=worker["restarts"])
        ready = [read_json(p) for p in (DATA / "cache").glob("*.json")]
        ready = [x for x in ready if x]
        plan = read_json(DATA / "plan.json", {})
        atomic_json(STATE / "status.json", {
            "status": "running", "time": time.time(), "supervisor_pid": os.getpid(),
            "workers": {role: {"pid": w["process"].pid if w["process"] else None,
                                "restarts": w["restarts"], "last_exit": w.get("last_exit")}
                        for role, w in workers.items()},
            "canonical_cache": {"train_ready": sum(x["split"] == "train" for x in ready),
                                "validation_ready": sum(x["split"] == "validation" for x in ready),
                                "planned_train": sum(x["split"] == "train" for x in plan.get("entries", [])),
                                "planned_validation": sum(x["split"] == "validation" for x in plan.get("entries", [])),
                                "missing_paper_groups": plan.get("missing_paper_training_groups", [])},
            "trainer": read_json(STATE / "trainer_status.json", {}),
            "data": read_json(STATE / "data_status.json", {}),
        })
        stop.sleep(15)
    event("supervisor", "stopping")
    for worker in workers.values():
        process = worker["process"]
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        if all(w["process"] is None or w["process"].poll() is not None for w in workers.values()):
            break
        time.sleep(.5)
    for worker in workers.values():
        process = worker["process"]
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)
    status = read_json(STATE / "status.json", {})
    status.update(status="stopped", time=time.time())
    atomic_json(STATE / "status.json", status)
    event("supervisor", "stopped")


if __name__ == "__main__":
    main()
