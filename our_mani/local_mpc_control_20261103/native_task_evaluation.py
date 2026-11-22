"""Offline-only task diagnostics from completed, recorded development cases.

Never import this module in a policy or use its truth labels for decisions.
The Native task flag is reported verbatim.  P1 HoldMetric is an additional
diagnostic, with its sticky historical pass kept separate from current hold.
This evaluator neither runs a simulator nor grants benchmark qualification.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from our_mani.local_mpc_20261102.records import HoldMetric


SCHEMA = "offline_native_task_and_current_hold_diagnostics_20261103_v1"
PHYSICS_DT = .05
HOLD_SECONDS, LIFT_HEIGHT_M, MAX_RELATIVE_DRIFT_M = .5, .04, .02
_REQUIRED_FILES = ("initial_snapshot.json", "native_ticks.jsonl", "result.json", "run_inputs.json")
_PUBLIC_LOG_NAMES = ("public_grasp_v4_decisions.jsonl", "public_grasp_decisions.jsonl")


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _json_lines(data, name):
    rows = []
    for line_number, line in enumerate(data.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"{name}:{line_number}: record dictionary required")
        rows.append(row)
    return rows


def _test_path(path):
    return any(part.lower() in {"test", "test_package_compressed"} for part in Path(path).parts)


def _reject_test_source(case, run):
    if _test_path(case) or run.get("TEST_read") is True:
        raise ValueError("TEST-derived cases are outside this development-only evaluator")
    config = run.get("config")
    if not isinstance(config, dict):
        raise ValueError("run_inputs.config dictionary required")
    for key in ("split", "data_split", "evaluation_split"):
        if str(config.get(key, "")).lower() == "test":
            raise ValueError("declared TEST split is outside this evaluator")
    # Inspect only source declarations; never open the declared dataset paths.
    for key in ("calibration_train_episode", "calibration_episode", "data_path", "episode_path"):
        value = config.get(key)
        if isinstance(value, str) and _test_path(value):
            raise ValueError("declared TEST dataset source is outside this evaluator")
    return config


def _state(row, context):
    if not isinstance(row, dict):
        raise ValueError(context + ": state dictionary required")
    now = row.get("time")
    tick = row.get("counted_tick")
    if isinstance(now, bool) or not isinstance(now, (int, float)) or not np.isfinite(now) or now < 0:
        raise ValueError(context + ": finite nonnegative time required")
    if type(tick) is not int or tick < 0:
        raise ValueError(context + ": nonnegative integer counted_tick required")
    for key, size in (("object_position", 3), ("tcp_position", 3), ("tcp_quaternion", 4)):
        value = np.asarray(row.get(key), dtype=float)
        if value.shape != (size,) or not np.isfinite(value).all():
            raise ValueError(context + ": finite " + key + " required")
        if key.endswith("quaternion") and np.linalg.norm(value) < 1e-12:
            raise ValueError(context + ": nonzero quaternion required")
    for key in ("held", "native_success"):
        if type(row.get(key)) is not bool:
            raise ValueError(context + ": explicit bool " + key + " required")
    native_result = row.get("native_result")
    if isinstance(native_result, dict) and "success" in native_result:
        if type(native_result["success"]) is not bool or native_result["success"] != row["native_success"]:
            raise ValueError(context + ": contradictory Native success fields")
    return row


def evaluate_states(initial_state, native_ticks):
    """Pure CPU replay of the *original* P1 HoldMetric on every 50 ms state.

    Returns (summary, ledger). ``requested_joint_condition`` is exactly the
    requested ever-Native AND current-held AND sticky-P1-pass formula.  A
    stricter current-hold formula is separate, so a historical pass cannot
    certify a later low grasp or a final drop.
    """
    initial = _state(initial_state, "initial")
    if initial["counted_tick"] != 0 or abs(initial["time"]) > 1e-8:
        raise ValueError("initial snapshot must be the zero controlled-tick anchor")
    metric = HoldMetric(HOLD_SECONDS, LIFT_HEIGHT_M, MAX_RELATIVE_DRIFT_M)
    ledger, ever_native = [], False
    first_native = first_requested = first_current = None
    previous = None
    for index, row in enumerate([initial, *native_ticks]):
        row = _state(row, f"state[{index}]")
        now = float(row["time"])
        if previous is not None:
            if row["counted_tick"] != previous["counted_tick"] + 1:
                raise ValueError("each controlled tick must appear exactly once in order")
            if abs(now - previous["time"] - PHYSICS_DT) > 1e-8:
                raise ValueError("every recorded transition must be exactly 50 ms; no gap interpolation")
        metric.update(row)
        ever_native = ever_native or row["native_success"]
        if first_native is None and row["native_success"]:
            first_native = now
        lift = float(row["object_position"][2]) - metric.initial_height
        current_duration = 0. if metric.start_time is None else max(0., now - metric.start_time)
        current_passed = bool(row["held"] and lift >= LIFT_HEIGHT_M - 1e-9
                              and current_duration + 1e-9 >= HOLD_SECONDS)
        requested = bool(ever_native and row["held"] and metric.passed)
        current = bool(ever_native and current_passed)
        if first_requested is None and requested:
            first_requested = now
        if first_current is None and current:
            first_current = now
        ledger.append(dict(time=now, counted_tick=row["counted_tick"], native_success=row["native_success"],
            ever_native_success=ever_native, current_held=row["held"], current_lift_m=lift,
            current_lifted_hold_seconds=current_duration, current_lifted_hold_passed=current_passed,
            p1_hold_passed_ever=metric.passed, longest_lifted_hold_seconds=metric.longest_seconds,
            maximum_lift_m=metric.max_lift, drops=metric.drops,
            requested_joint_condition=requested, strict_current_joint_condition=current))
        previous = row
    final = ledger[-1]
    summary = dict(native_success_any=ever_native, native_success_first_time=first_native,
        native_success_final=final["native_success"],
        p1_hold_metric=metric.receipt(),
        requested_joint_condition=dict(formula="ever_native_success AND current_held AND p1_hold_passed_ever",
            any=first_requested is not None, first_time=first_requested, final=final["requested_joint_condition"],
            historical_hold_pass_is_sticky=True, final_does_not_alone_certify_current_lift=True),
        strict_current_joint_condition=dict(formula="ever_native_success AND current_lifted_hold_passed",
            any=first_current is not None, first_time=first_current, final=final["strict_current_joint_condition"]),
        final=dict(time=final["time"], counted_tick=final["counted_tick"], held=final["current_held"],
            lift_m=final["current_lift_m"], current_lifted_hold_seconds=final["current_lifted_hold_seconds"],
            current_lifted_hold_passed=final["current_lifted_hold_passed"]),
        drops_definition="recorded Native held True-to-False edges, including deliberate releases; not a force-sensor slip diagnosis",
        recorded_controlled_ticks=len(ledger)-1, initial_snapshot_included=True,
        temporal_coverage="all recorded real 50ms ticks; no resampling, interpolation or fabricated frames")
    return summary, ledger


def _public_logged_summary(rows):
    observations = []
    for row in rows:
        metric = row.get("public_lift_hold")
        if not isinstance(metric, dict) or not isinstance(metric.get("latest"), dict):
            continue
        latest = metric["latest"]
        duration = latest.get("public_hold_seconds")
        if isinstance(duration, bool) or not isinstance(duration, (int, float)) or not np.isfinite(duration) or duration < 0:
            raise ValueError("invalid logged public hold duration")
        observations.append(dict(timestamp=latest.get("timestamp"), public_hold_seconds=float(duration),
            public_lift_hold_observed=latest.get("public_lift_hold_observed") is True,
            current_visual_target=latest.get("current_visual_target") is True,
            current_public_attachment_confirmed=latest.get("current_public_attachment_confirmed") is True))
    return dict(available=bool(observations), observations=len(observations),
        longest_logged_public_hold_seconds=max((x["public_hold_seconds"] for x in observations), default=None),
        public_lift_hold_observed_any=any(x["public_lift_hold_observed"] for x in observations),
        final_logged=observations[-1] if observations else None,
        scope="existing public metric receipts at decision logging boundaries; not recomputed from GT and not assumed complete 50ms public coverage",
        used_in_Native_or_hold_diagnostic=False)


def evaluate_case(case_dir):
    """Read only completed case records; return ``(report, tick_ledger)``.

    Only files inside the given case are read. Declared calibration/dataset
    paths are never opened. Source SHA checks detect concurrent modification.
    """
    case = Path(case_dir).resolve()
    if not case.is_dir() or _test_path(case):
        raise ValueError("existing non-TEST case directory required")
    raw = {name: (case / name).read_bytes() for name in _REQUIRED_FILES}
    run = json.loads(raw["run_inputs.json"])
    config = _reject_test_source(case, run)
    result = json.loads(raw["result.json"])
    if result.get("complete") is not True:
        raise ValueError("completed result.json required; do not evaluate an active case")
    snapshot = json.loads(raw["initial_snapshot.json"])
    summary, ledger = evaluate_states(snapshot.get("state"), _json_lines(raw["native_ticks.jsonl"], "native_ticks"))
    public_logs = [name for name in _PUBLIC_LOG_NAMES if (case / name).is_file()]
    if len(public_logs) > 1:
        raise ValueError("ambiguous public decision logs; choose an unambiguous archived case")
    public = dict(available=False, used_in_Native_or_hold_diagnostic=False)
    if public_logs:
        name = public_logs[0]
        raw[name] = (case / name).read_bytes()
        public = _public_logged_summary(_json_lines(raw[name], name))
    for name, data in raw.items():
        if (case / name).read_bytes() != data:
            raise RuntimeError("case source changed during evaluation: " + name)
    no_proxy = any(config.get(key) is True for key in
                   ("counterfactual_no_proxy", "diagnostic_no_proxy_attachment"))
    reasons = ["offline development diagnostics do not grant standard benchmark qualification"]
    if no_proxy:
        reasons.append("counterfactual_no_proxy diagnostic cannot receive standard benchmark positive credit")
    if config.get("formal_evaluation") is not True:
        reasons.append("run config does not declare formal evaluation")
    return dict(schema=SCHEMA, case=str(case), metrics=summary, public_logged_metrics=public,
        criteria=dict(seconds=HOLD_SECONDS, lift_m=LIFT_HEIGHT_M, maximum_relative_drift_m=MAX_RELATIVE_DRIFT_M,
                      implementation="original our_mani.local_mpc_20261102.records.HoldMetric"),
        scope=dict(GT_used_only_posthoc=True, policy_import_permitted=False, policy_inputs_modified=False,
            Native_started=False, fitted_parameters=False, TEST_sources_read=False,
            original_task_success_replaced=False, formal_qualification=False,
            maximum_or_past_hold_is_final_grip_proof=False),
        benchmark=dict(counterfactual_no_proxy=no_proxy, standard_benchmark_positive_credited=False,
            reasons=reasons, Native_success_still_reported_verbatim=True),
        run=dict(mode=run.get("mode"), scene=run.get("scene"), information_setting=run.get("information_setting"),
            config_schema=config.get("schema"), physics_dt=config.get("physics_dt"),
            control_dt=config.get("control_dt"), result_stage=result.get("stage"),
            result_complete=True, result_source_unchanged=result.get("source_unchanged"),
            result_declared_Native_task_success_claimed=result.get("Native_task_success_claimed")),
        source_files={name: dict(path=str(case / name), sha256=_sha(data)) for name, data in raw.items()},
        evaluator_sha256=_sha(Path(__file__).read_bytes()),
        HoldMetric_source_sha256=_sha(Path(__file__).parents[1].joinpath("local_mpc_20261102/records.py").read_bytes())), ledger


def write_case_evaluation(case_dir, output_dir):
    """Create a new independent artifact; never overwrite or write into case."""
    case, output = Path(case_dir).resolve(), Path(output_dir).resolve()
    if output == case or case in output.parents:
        raise ValueError("evaluation artifact must be outside the original case")
    if output.exists():
        raise FileExistsError(output)
    report, ledger = evaluate_case(case)
    output.mkdir(parents=True, exist_ok=False)
    with (output / "report.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    with (output / "tick_metrics.jsonl").open("x", encoding="utf-8") as stream:
        for row in ledger:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True)
    parser.add_argument("--output", required=True, help="new artifact directory outside the case")
    args = parser.parse_args()
    report = write_case_evaluation(args.case, args.output)
    print(json.dumps(dict(case=report["case"], metrics=report["metrics"],
                         public_logged_metrics=report["public_logged_metrics"]), ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
