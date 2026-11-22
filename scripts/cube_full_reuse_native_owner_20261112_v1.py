"""Serial Native owner for the explicitly authorized new Cube training cycle."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path): return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def stop_identity(path):
    if not path.exists(): return None
    stat = path.stat()
    return dict(sha256=sha(path), mtime_ns=stat.st_mtime_ns, inode=stat.st_ino)


def stopped(plan, out):
    # The old pause remains in place for old queues. Only its exact historical
    # instance is waived for this separately requested repair validation. A new,
    # replaced or touched global STOP immediately stops this scoped run too.
    global_stop = ROOT / "runtime/framework/STOP"
    return ((out / "STOP").exists() or (ROOT/"artifacts/framework/cube_full_reuse_20261112_v1/STOP").exists() or
            stop_identity(global_stop) != plan["historical_global_stop"] or
            (ROOT / "runtime/framework/cube_full_reuse_20261112_v1/STOP").exists())


def verify(manifest):
    for name, digest in manifest["repository"].items():
        if sha(ROOT / name) != digest: raise RuntimeError("repository input changed: " + name)
    for name, digest in manifest["native"].items():
        if sha(name) != digest: raise RuntimeError("Native input changed: " + name)


def source_manifest(plan_path):
    from scripts.validate_perception_repair_20261111_v1 import source_manifest as baseline_manifest
    manifest = baseline_manifest(plan_path)
    frozen_native=read(ROOT/'artifacts/framework/cube_full_reuse_20261112_v1/native_sources.json')
    if manifest['native']!=frozen_native:raise RuntimeError('Native sources differ from full-cycle initial freeze')
    additions = list((ROOT/'our_mani/cube_full_reuse_20261112_v1').glob('*.py'))
    additions += [ROOT/"scripts/run_cube_full_reuse_cycle_20261112_v1.sh",ROOT/"tests/test_cube_full_reuse_20261112_v1.py",ROOT/"our_mani/official_simulation.py",ROOT/"our_mani/pyrep_env.py",Path(__file__), ROOT/'scripts/run_cube_full_reuse_native_20261112_v1.sh', ROOT/'config/framework/cube_full_reuse_20261112_v1/plan.json']
    for case in read(plan_path)['cases']:
        for key in ('candidate_transformer_checkpoint','corrected_forecast_checkpoint','position_refiner_checkpoint'):
            additions.append(ROOT/case['config'][key])
    additions=list(dict.fromkeys(additions))
    manifest['repository'].update({p.relative_to(ROOT).as_posix(): sha(p) for p in additions})
    return manifest


def worker(plan_path, out, case_id):
    plan = read(plan_path)
    manifest = read(out / "sources.json")
    verify(manifest)
    from scripts.feedback_current_environment_20261104_v2 import current_environment
    write(out / "environment.json", current_environment())
    module_name=plan.get('worker_module','our_mani.highspeed_control_20261111_v1.native_validation_v2')
    if not module_name.startswith('our_mani.cube_full_reuse_20261112_v1.'): raise ValueError('Scoped highspeed worker module required')
    import importlib
    run_case=importlib.import_module(module_name).run_case
    results = []
    for case in [c for c in plan["cases"] if c["id"]==case_id]:
        if stopped(plan, out): raise InterruptedError("repair validation STOP")
        verify(manifest)
        results.append(run_case(case, out / "episode", lambda: stopped(plan, out)))
    if len(results)!=1: raise ValueError("Exact single case required")
    verify(manifest)
    write(out / "native_result.json", dict(complete=all(r["complete"] for r in results), results=results,
        source_unchanged=True, formal_VAL_TEST=False, training_started=False,
        old_queues_resumed=False, global_STOP_preserved=True))
    return 0


def supervise(plan_path, out, case_id):
    plan = read(plan_path)
    if plan.get("scope") != "cube_full_cycle_native" or plan.get("training_authorized") is not True:
        raise ValueError("explicit executor repair scope required")
    if sum(c["id"]==case_id for c in plan["cases"])!=1: raise ValueError("Unknown or duplicate source")
    out.mkdir(parents=True, exist_ok=False)
    manifest = source_manifest(plan_path)
    write(out / "sources.json", manifest)
    write(out / "request.json", dict(user_request="全部TRAIN重新训练，再跑完整VAL/TEST；普通及高速方块；60秒", plan=dict(scope=plan["scope"],case=next(c for c in plan["cases"]if c["id"]==case_id)), plan_sha256=sha(plan_path),
        started_utc=datetime.now(timezone.utc).isoformat(), Native_at_most=1, training_started=False,
        historical_STOP_waiver_scope="this named repair validation only; old queues remain paused"))
    if stopped(plan, out): raise InterruptedError("repair validation STOP before launch")
    spec = importlib.util.spec_from_file_location("execution_repair_lifecycle", ROOT / "scripts/cube_revision_lifecycle_20261104_v2.py")
    lifecycle = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lifecycle)
    lifecycle.enable_subreaper()
    process, code, tracked, reaped, reason = None, None, {}, [], None
    signaled = []
    previous = {s: signal.signal(s, lambda n, f: signaled.append(n)) for s in (signal.SIGINT, signal.SIGTERM)}
    started = time.monotonic()
    try:
        command = ["bash", str(ROOT / "scripts/run_cube_full_reuse_native_20261112_v1.sh"),
                   "--plan", str(plan_path), "--output", str(out), "--case-id", case_id]
        with (out / "stdout.log").open("x") as log:
            process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            write(out / "child.json", dict(pid=process.pid, argv=command))
            stopping = None
            while process.poll() is None:
                tracked.update(lifecycle.descendants())
                if stopping is None and (signaled or stopped(plan, out) or time.monotonic() - started > plan["timeout_seconds"]):
                    reason, stopping = "STOP_signal_or_timeout", time.monotonic()
                    if not (out / "STOP").exists(): (out / "STOP").write_text(reason + "\n")
                if stopping is not None and time.monotonic() - stopping > 10:
                    lifecycle.signal_known(tracked, signal.SIGTERM)
                if stopping is not None and time.monotonic() - stopping > 20:
                    lifecycle.signal_known(tracked, signal.SIGKILL)
                time.sleep(.5)
            code = process.wait()
            tracked.update(lifecycle.descendants())
        # xvfb-run can leave an X server exiting a moment after its worker.
        remaining = tracked
        for _ in range(40):
            remaining, newly = lifecycle.reap_adopted(remaining, process.pid)
            reaped.extend(newly)
            if not remaining: break
            time.sleep(.05)
        if remaining:
            lifecycle.signal_known(remaining, signal.SIGTERM)
            for _ in range(40):
                remaining, newly = lifecycle.reap_adopted(remaining, process.pid)
                reaped.extend(newly)
                if not remaining: break
                time.sleep(.05)
        if remaining:
            lifecycle.signal_known(remaining, signal.SIGKILL)
            for _ in range(40):
                remaining, newly = lifecycle.reap_adopted(remaining, process.pid)
                reaped.extend(newly)
                if not remaining: break
                time.sleep(.05)
        verify(manifest)
        passed = code == 0 and not remaining and (out / "native_result.json").is_file() and read(out / "native_result.json")["complete"]
        receipt = dict(complete=passed, actual_process_returncode=code, actual_wait_completed=True,
            unreaped_descendants=remaining, adopted_reaped=reaped, source_unchanged=True,
            stop_reason=reason, elapsed_seconds=time.monotonic() - started,
            global_STOP_preserved=stop_identity(ROOT / "runtime/framework/STOP") == plan["historical_global_stop"],
            formal_VAL_TEST=False, training_started=False, old_queues_resumed=False)
        write(out / "actual_exit.json", receipt)
        print(json.dumps(receipt, ensure_ascii=False), flush=True)
        if not passed: raise RuntimeError("Native interface validation failed; inspect stdout.log and worker_exit.json")
        return 0
    finally:
        if process is not None and process.poll() is None:
            tracked.update(lifecycle.descendants())
            lifecycle.signal_known(tracked, signal.SIGKILL)
            process.wait()
            for _ in range(40):
                tracked, newly = lifecycle.reap_adopted(tracked, process.pid)
                if not tracked: break
                time.sleep(.05)
        for sig, handler in previous.items(): signal.signal(sig, handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=ROOT / "config/framework/highspeed_control_20261111_v1/plan.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--native-worker", action="store_true")
    parser.add_argument("--case-id", required=True)
    args = parser.parse_args()
    return worker(args.plan.resolve(strict=True), args.output.resolve(), args.case_id) if args.native_worker else supervise(args.plan.resolve(strict=True), args.output.resolve(), args.case_id)


if __name__ == "__main__":
    raise SystemExit(main())
