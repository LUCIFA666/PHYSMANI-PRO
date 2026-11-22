"""Prepared TRAIN-only truth-teacher collection, shadow public50ms, serial.

Default is CPU preflight only. --execute is required to launch the existing
Native owner. No fit, policy modifications, synthetic frames or TEST access.
The first four jobs are a fixed pilot; posthoc fidelity/positive-label gates
must pass before the remaining twelve preassigned TRAIN parents may run.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
FAMILY = ROOT / "runtime/framework/local_mpc_20261102"
DT = .05
CAMERAS = {"front", "8", "16", "36"}
ROBOT_KEYS = {"timestamp", "tcp_position", "tcp_quaternion", "q", "dq", "finger_open_amounts"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def relative_path(value, *, family=None):
    value = Path(value)
    if value.is_absolute():
        raise ValueError("prepared project-relative paths required")
    path = (ROOT / value).resolve()
    if ROOT not in path.parents or (family is not None and family.resolve() not in path.parents):
        raise ValueError("prepared path is outside the declared project family")
    return path


def checked_tick(time):
    if isinstance(time, bool) or not isinstance(time, (int, float)):
        raise ValueError("numeric actual time required")
    k = round(time / DT)
    if abs(k * DT - time) > 1e-8:
        raise ValueError("actual time is not on 50ms lattice")
    return k


def check_sources(expected):
    for name, digest in expected.items():
        if sha(relative_path(name)) != digest:
            raise RuntimeError("prepared source changed: " + name)


def validate_plan(plan_path):
    path = Path(plan_path).resolve()
    raw = path.read_bytes()
    plan = json.loads(raw)
    if plan.get("schema") != "P4_preassigned_TRAIN_truthB_shadow50_teacher_20261103_v1":
        raise ValueError("explicit P4 TRAIN teacher plan required")
    jobs = plan.get("jobs", [])
    if len(jobs) != 16 or len({j["parent_id"] for j in jobs}) != 16:
        raise ValueError("sixteen new independently assigned parent IDs required")
    if plan.get("formal_TEST_read") is not False or plan.get("training_started") is not False:
        raise ValueError("TRAIN collection only; no fit/TEST")
    if plan.get("pilot_gate", {}).get("pilot_jobs") != 4:
        raise ValueError("fixed first-four pilot gate required")
    check_sources(plan["source_sha256"])
    configs = []
    for index, job in enumerate(jobs):
        if job.get("split") != "train" or job.get("tranche") != ("pilot" if index < 4 else "continuation"):
            raise ValueError("all splits and fixed pilot/continuation must be assigned before collection")
        config_path = relative_path(job["config"])
        if sha(config_path) != job["config_sha256"]:
            raise RuntimeError("prepared config changed")
        config = json.loads(config_path.read_text())
        if config.get('seed') != job.get('seed') or config.get('P4_collection_metadata', {}).get('parent_id') != job['parent_id']:
            raise ValueError('exact preassigned parent and seed must reach worker config')
        required = dict(dataset_use="train", preserve_native_height=True, simulation_seconds=8.,
            physics_dt=.05, control_dt=.1, decision_ticks=2, horizon_steps=8, num_candidates=8,
            response_factory="coupled_joint_20261103_v1", response_controller="coupled", action_space="joint",
            use_joint_goal_planner=True, public_motion_shadow=True, public_shadow_save_frames=True,
            gripper_completion_latch=True, formal_evaluation=False)
        if any(config.get(k) != v for k, v in required.items()):
            raise ValueError("teacher/shadow50/B joint/protocol config differs from reviewed contract")
        if config.get("cem_iteration_sequences") != [88, 88, 80] or config.get("cem_sequences_total") != 256:
            raise ValueError("unchanged 256/H8 search budget required")
        if any(config.get(k) for k in ("public_grasp_prototype", "public_grasp_active_visual_v4",
            "public_grasp_active_visual_v41", "public_pregrasp_control", "diagnostic_no_proxy_attachment")):
            raise ValueError("truth teacher plus passive public shadow only")
        if job["scene"] not in ("static", "uniform", "speed_change"):
            raise ValueError("worker scene unsupported")
        if config["uniform_velocity"][2] != 0 or config["changed_velocity"][2] != 0:
            raise ValueError("this prepared pilot supports planar translation only")
        if not .05 <= config["initial_target_position"][0] <= .35 or not -.45 <= config["initial_target_position"][1] <= .45:
            raise ValueError("initial XY outside recorded Native task bounds")
        if "/train_package_compressed/" not in config["calibration_train_episode"]:
            raise ValueError("original TRAIN calibration only")
        if any("test" == p.lower() for p in Path(config["calibration_train_episode"]).parts):
            raise ValueError("TEST source forbidden")
        relative_path(job["path"], family=FAMILY)
        relative_path(job["evaluation"], family=ROOT / "artifacts/framework/local_mpc_20261102")
        configs.append(config)
    return plan, raw, configs


def read_lines(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def validate_archived_sources(case, source_sha256):
    """Validate the case's archived bytes, independent of today's working tree."""
    if not isinstance(source_sha256, dict) or not source_sha256:
        raise ValueError("nonempty run_inputs source identity required")
    archive = (Path(case) / "source_snapshot").resolve()
    for name, expected in source_sha256.items():
        relative = Path(name)
        path = (archive / relative).resolve()
        if relative.is_absolute() or archive not in path.parents:
            raise ValueError("archived source path escaped case snapshot")
        if sha(path) != expected:
            raise RuntimeError("archived source differs from run_inputs SHA: " + name)
    return dict(archived_source_files_verified=len(source_sha256),
                source_of_truth="case/source_snapshot versus recorded run_inputs.source_sha256",
                current_working_tree_substituted_for_archived_source=False)


def audit_public_capture(case, states, checkpoint_sha):
    """Read public logs only; no Native object/held enters this validation."""
    import numpy as np
    directory = Path(case) / "public_shadow"
    scope = json.loads((directory / "scope.json").read_text())
    if scope.get("controller_uses_public_forecast") is not False or scope.get("actual_sensor_dt") != DT:
        raise ValueError("passive public50ms observer scope required")
    if scope.get("object_GT_held_success_allowed_in_public_bridge") is not False:
        raise ValueError("public bridge must exclude Native object/held/success")
    rows = read_lines(directory / "public_outputs.jsonl")
    if len(rows) != len(states):
        raise ValueError("one initial plus one actual public sample per native tick required")
    capture, times, forecasts = {}, [], {}
    for index, row in enumerate(rows):
        result, robot = row["result"], row["public_robot"]
        now = float(result["timestamp"])
        if set(robot) != ROBOT_KEYS or abs(robot["timestamp"] - now) > 1e-8:
            raise ValueError("only fresh sanitized actual robot channels may accompany RGBD")
        if abs(now - states[index]["time"]) > 1e-8 or checked_tick(now) != index:
            raise ValueError("public and Native clock mismatch; no interpolated/fake samples")
        if any(result.get(k) is not False for k in
               ("Native_inputs_used", "future_observations_used", "observations_fabricated")):
            raise ValueError("public observer causality flags failed")
        path = (directory / row["rgbd"]).resolve()
        if directory.resolve() not in path.parents:
            raise ValueError("public frame path outside real capture directory")
        data = path.read_bytes()
        with np.load(io.BytesIO(data), allow_pickle=False) as arrays:
            if abs(float(arrays["timestamp"]) - now) > 1e-8:
                raise ValueError("saved real frame timestamp mismatch")
            if not all("images_" + c in arrays and "depths_" + c in arrays for c in CAMERAS):
                raise ValueError("four actual RGB-D view pairs required")
        capture[index] = dict(path=str(path.relative_to(ROOT)), sha256=hashlib.sha256(data).hexdigest())
        times.append(now)
        if result.get("free_forecast_available") is True:
            f = result.get("forecast")
            if (result.get("history_frames", 0) < 5 or result.get("frozen_checkpoint_sha256") != checkpoint_sha
                    or not isinstance(f, dict) or len(f.get("times", [])) != 17
                    or not np.allclose(f["times"], now + np.arange(17) * DT, atol=1e-8, rtol=0)):
                raise ValueError("frozen causal GRU forecast timing/provenance invalid")
            forecasts[index] = result
    label_path = directory / "evaluation_labels.jsonl"
    if len(read_lines(label_path)) != len(states):
        raise ValueError("separate offline labels must cover every actual frame")
    return dict(real_public_samples=len(rows), exactly_50ms=True, initial_zero_frame=True,
        camera_calibration_sha256=sha(directory / "camera_calibration.json"), frames_sha256=capture,
        public_outputs_sha256=sha(directory / "public_outputs.jsonl"),
        separate_evaluation_labels_sha256=sha(label_path), free_forecast_samples=len(forecasts),
        current_public_robot_whitelist_passed=True, truth_read_by_public_validator=False), forecasts


def contact_labels(case, states, forecasts, parent_id):
    """Future Native labels are consumed only after the worker actually exits."""
    import numpy as np
    from scipy.spatial.transform import Rotation
    from our_mani.local_mpc_20261102.records import HoldMetric
    by_tick = {checked_tick(s["time"]): s for s in states}
    if states[0]['held']:
        raise ValueError('initially free teacher state required')
    rows = read_lines(Path(case) / "transitions.jsonl")
    ticks = [t for row in rows for t in row["executor"].get("ticks", []) if t.get("tick_completed") is True]
    if len(ticks) != len(states) - 1:
        raise ValueError("complete dispatched-reference and Native tick ledger required")
    target = states[0]["requested_opening"]
    starts = []
    for t in ticks:
        if t["gripper_target"] == 0. and target != 0.:
            starts.append(t)
        target = t["gripper_target"]
    labels = []
    for number, start in enumerate(starts):
        begin = checked_tick(start["clock_before"])
        end = next((checked_tick(t["clock_before"]) for t in ticks
                    if checked_tick(t["clock_before"]) > begin and t["gripper_target"] == 1.), len(states))
        complete = next((t for t in ticks if begin <= checked_tick(t["clock_before"]) < end
                         and t["gripper_target"] == 0. and t.get("gripper_done") is True), None)
        segments, active = [], []
        for state in states:
            if not begin < checked_tick(state['time']) <= end:
                continue
            if state['held']:
                active.append(state)
            elif active:
                segments.append(active); active = []
        if active:
            segments.append(active)
        # A later held segment cannot lend its hold/goal evidence to an
        # earlier tau. Keep failed segments as labels with acceptance=False.
        for segment_id, segment in enumerate(segments or [None]):
            held = None if segment is None else segment[0]
            row = dict(parent_id=parent_id, attempt_id=number + 1, held_segment_id=segment_id + 1,
                dispatch_tBefore=start["clock_before"],
                first_actual_Native_held_time=None if held is None else held["time"],
                API_completion_call_time=None if complete is None else complete["clock_before"],
                API_completion_feedback_time=None if complete is None else complete["clock_after"],
                accepted_positive=False, causal_50ms_anchors=[], Native_labels_only_posthoc=True,
                Native_held_is_not_force_contact_proof=True)
            if held is None:
                labels.append(row)
                continue
            tau_k = checked_tick(held["time"])
            metric = HoldMetric(seconds=.5, height=.04, max_relative_drift=.02)
            metric.update(states[0])
            for state in segment:
                metric.update(state)
            row.update(accepted_positive=bool(metric.passed and any(s['native_success'] for s in segment)),
                continuous_held_segment_end=segment[-1]['time'], same_segment_hold=metric.receipt(),
                original_Native_goal_in_same_segment=any(s['native_success'] for s in segment))
            obj = Rotation.from_quat(held["object_quaternion"])
            row["label"] = dict(tau=held["time"], object_TCP_position_m=obj.inv().apply(
                np.asarray(held["tcp_position"]) - held["object_position"]).tolist(),
                object_TCP_quaternion_xyzw=(obj.inv() * Rotation.from_quat(held["tcp_quaternion"])).as_quat().tolist())
            if row["accepted_positive"]:
                for k in range(max(0, tau_k - 16), tau_k):
                    if k > begin or by_tick[k]["held"] or k not in forecasts or k < 4:
                        continue
                    row["causal_50ms_anchors"].append(dict(t0=by_tick[k]["time"], tau=held["time"],
                        tau_minus_t0_seconds=(tau_k-k)*DT, real_history_frames_available=min(20,k+1),
                        macro_100ms_anchor=(k % 2 == 0), features_emitted=False))
            labels.append(row)
    return labels


def audit_completed_job(job, config):
    """No return from the child is assumed before checking actual exit receipts."""
    case = relative_path(job["path"], family=FAMILY)
    exit_path = case.parent / (case.name + ".owner.actual_exit.json")
    owner = json.loads(exit_path.read_text())
    worker = json.loads((case / "worker_exit.json").read_text())
    result = json.loads((case / "result.json").read_text())
    inputs = json.loads((case / "run_inputs.json").read_text())
    if owner.get("actual_process_returncode") != 0 or worker.get("exit_code") != 0:
        raise RuntimeError("actual Native/worker exit was not successful")
    if result.get("complete") is not True or result.get("source_unchanged") is not True or inputs.get("TEST_read") is not False:
        raise RuntimeError("completed unchanged-source TRAIN case required")
    if inputs.get("config") != config:
        raise RuntimeError("worker did not use exact prepared config")
    snapshot_evidence = validate_archived_sources(case, inputs.get('source_sha256'))
    native = inputs.get("native_sources_sha256", {})
    expected = {"third_party/RLBench/rlbench/tasks/" + config["task"] + ".py": config["native_task_source_sha256"],
                "third_party/RLBench/rlbench/task_ttms/" + config["task"] + ".ttm": config["native_task_scene_sha256"]}
    if any(native.get(k) != v for k, v in expected.items()):
        raise RuntimeError("original Native task/scene source receipt failed")
    snapshot = json.loads((case / "initial_snapshot.json").read_text())
    if (snapshot.get("native_height_preserved") is not True or
            abs(snapshot["state"]["object_position"][2] - snapshot["native_default_target_pose_before_override"][2]) > 1e-8):
        raise RuntimeError("Native default table height was not preserved")
    states = [snapshot["state"], *read_lines(case / "native_ticks.jsonl")]
    sys.path.insert(0, str(ROOT))
    from our_mani.local_mpc_control_20261103.native_task_evaluation import write_case_evaluation, evaluate_states
    report = write_case_evaluation(case, relative_path(job["evaluation"]))
    evaluate_states(states[0], states[1:])
    public, forecasts = audit_public_capture(case, states, job["frozen_GRU_checkpoint_sha256"])
    labels = contact_labels(case, states, forecasts, job["parent_id"])
    positive = [label for label in labels if label["accepted_positive"]]
    return dict(episode_id=job["episode_id"], parent_id=job["parent_id"], split="train", tranche=job["tranche"],
        task_success=report["metrics"], actual_owner_exit=owner, actual_worker_exit=worker,
        source_snapshot_evidence=snapshot_evidence,
        original_Native_sources_passed=True, native_height_preserved=True, public_capture=public,
        contact_labels=labels, positive_contact_labels=len(positive),
        causal_macro_anchors=sum(a["macro_100ms_anchor"] for label in positive for a in label["causal_50ms_anchors"]),
        completion_latched_actual_protocol=True, old_NN_finger_response_requalified=False,
        success_failure_both_retained=True, finished_utc=datetime.now(timezone.utc).isoformat())


def pilot_gate(rows):
    positive_parents = {r["parent_id"] for r in rows if r["positive_contact_labels"] > 0}
    checks = dict(exactly_four_preassigned_pilot_cases=len(rows) == 4,
        all_original_Native_source_receipts_passed=all(r["original_Native_sources_passed"] for r in rows),
        all_real_public50ms_captures_passed=all(r["public_capture"]["exactly_50ms"] for r in rows),
        at_least_two_independent_positive_parents=len(positive_parents) >= 2,
        positive_causal_macro_anchor_count_nonzero=sum(r["causal_macro_anchors"] for r in rows) > 0)
    return dict(passed=all(checks.values()), checks=checks, positive_parent_count=len(positive_parents),
                future_labels_used_only_after_case_exit=True, continuation_parent_ids_remain_preassigned=True)


def stop_requested(owner, batch):
    return any(p.exists() for p in (owner / "STOP", batch / "STOP", FAMILY / "STOP"))


def run(plan_path, owner_path, *, execute=False):
    plan, plan_raw, configs = validate_plan(plan_path)
    if not execute:
        return dict(preflight_passed=True, Native_started=False, jobs=len(plan["jobs"]), pilot_jobs=4,
                    source_files_checked=len(plan["source_sha256"]), independent_TRAIN_parents=16,
                    execution_requires_explicit_execute=True)
    owner = Path(owner_path).resolve()
    if FAMILY not in owner.parents or owner.exists():
        raise ValueError("fresh runtime-family owner directory required")
    for job in plan["jobs"]:
        if relative_path(job["path"]).exists() or relative_path(job["evaluation"]).exists():
            raise FileExistsError("fresh case/evaluation outputs required; never silently resume or overwrite")
    batch = relative_path(plan["batch_output_parent"])
    batch.mkdir(parents=True, exist_ok=False)
    owner.mkdir(parents=True, exist_ok=False)
    write_new(owner / "inputs.json", dict(plan_path=str(Path(plan_path).resolve()),
        plan_sha256=hashlib.sha256(plan_raw).hexdigest(), runner_sha256=sha(__file__),
        exact_prepared_plan=plan, Native_serial_locks="existing original shell wrapper owns both shared exclusion locks"))
    results = []
    for index, (job, config) in enumerate(zip(plan["jobs"], configs)):
        if stop_requested(owner, batch):
            write_new(owner / "stopped.json", dict(complete=False, user_stop=True, results=results))
            return 2
        if Path(plan_path).read_bytes() != plan_raw:
            raise RuntimeError("prepared plan changed")
        check_sources(plan["source_sha256"])
        if sha(relative_path(job['config'])) != job['config_sha256']:
            raise RuntimeError('prepared job config changed before dispatch')
        if index == 4:
            gate = pilot_gate(results)
            write_new(owner / "pilot_gate.json", gate)
            if not gate["passed"]:
                write_new(owner / "pilot_gate_failed.json", dict(complete=False, all_pilot_cases_retained=True, results=results))
                return 3
        command = [sys.executable, str(ROOT / "scripts/launch_local_mpc_20261102.py"), "--mode", "p1",
                   "--scene", job["scene"], "--config", job["config"], "--output", job["path"]]
        print(json.dumps(dict(event="start", episode_id=job["episode_id"], parent_id=job["parent_id"], tranche=job["tranche"])), flush=True)
        process = subprocess.Popen(command, cwd=ROOT)
        while True:
            try:
                code = process.wait(timeout=.25)
                break
            except subprocess.TimeoutExpired:
                if stop_requested(owner, batch):
                    # Existing worker checks this file at its next100ms boundary;
                    # never kill its owner before its actual child exit receipt.
                    if not (batch / "STOP").exists():
                        with (batch / "STOP").open("x") as stream:
                            stream.write("P4 owner STOP propagated; reap actual Native owner\n")
        if code != 0:
            write_new(owner / "infrastructure_or_stop.json", dict(complete=False, episode_id=job["episode_id"],
                actual_launcher_returncode=code, user_stop_requested=stop_requested(owner,batch),
                case_retained=job["path"], previous_results=results))
            return 2 if stop_requested(owner,batch) else code
        check_sources(plan["source_sha256"])
        row = audit_completed_job(job, config)
        write_new(owner / (job["episode_id"] + ".json"), row)
        results.append(row)
        print(json.dumps(dict(event="finished", episode_id=job["episode_id"],
            positive_contact_labels=row["positive_contact_labels"], causal_macro_anchors=row["causal_macro_anchors"])), flush=True)
    if Path(plan_path).read_bytes() != plan_raw:
        raise RuntimeError("prepared plan changed")
    check_sources(plan["source_sha256"])
    write_new(owner / "result.json", dict(complete=True, episodes=16, independent_TRAIN_parents=16,
        pilot_gate=pilot_gate(results[:4]), results=results, training_started=False, TEST_read=False,
        failed_task_episodes_retained=True, original_sources_modified=False))
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--owner", type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.execute and args.owner is None:
        parser.error("--owner required for explicit --execute")
    result = run(args.plan, args.owner, execute=args.execute)
    if isinstance(result, dict):
        print(json.dumps(result))
        return 0
    return result


if __name__ == "__main__":
    raise SystemExit(main())
