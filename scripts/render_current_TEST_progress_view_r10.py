"""Render one externally pinned TEST status observation; preview unless --write.

This is a display tool. It reads exactly one status JSON and the fixed1071 plan.
It never grants an audit/release, acknowledges a report, or controls any job.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
FRAMEWORK = ROOT / "runtime/framework"
CURRENT_STATUS = FRAMEWORK / "formal_TEST_reboot_resume_v1071/status.json"
PLAN = FRAMEWORK / "formal_TEST_reboot_resume_v1071/plan.json"
PLAN_SHA256 = "47da0b58c6fec8191c048d7bbe104ac4352d76a4034b552fc610f1b4e073aff2"
OUTPUT = ROOT / "TEST_16任务进度.txt"
METHODS = ("cv_geometry", "context_geometry", "physmani")
METHOD_LABELS = {"cv_geometry": "CV＋几何", "context_geometry": "Context＋几何", "physmani": "PhysMani"}
TASK_LABELS = {
    "beat_the_rotating_buzz": "旋转电线穿环·普通",
    "beat_the_rotating_buzz_high_speed": "旋转电线穿环·高速",
    "insert_onto_rotating_peg": "向旋转插杆套环·普通",
    "insert_onto_rotating_peg_high_speed": "向旋转插杆套环·高速",
    "moving_basketball_in_hoop": "向移动篮筐投球·普通",
    "moving_basketball_in_hoop_high_speed": "向移动篮筐投球·高速",
    "pick_moving_target_on_the_table": "抓取桌上移动方块·普通",
    "pick_moving_target_on_the_table_high_speed": "抓取桌上移动方块·高速",
    "push_moving_button": "按压移动按钮·普通",
    "push_moving_button_high_speed": "按压移动按钮·高速",
    "put_rubbish_in_moving_bin": "向移动垃圾桶投放垃圾·普通",
    "put_rubbish_in_moving_bin_high_speed": "向移动垃圾桶投放垃圾·高速",
    "place_cups_on_rotating_frame": "向旋转杯架挂杯子·普通",
    "place_cups_on_rotating_frame_high_speed": "向旋转杯架挂杯子·高速",
    "remove_cups_from_rotating_frame": "从旋转杯架取下杯子·普通",
    "remove_cups_from_rotating_frame_high_speed": "从旋转杯架取下杯子·高速",
}
DISCLAIMER = "运行状态统计，完整独审未确认"


def require(value, message):
    if not value:
        raise ValueError(message)


def valid_sha(value):
    return isinstance(value, str) and len(value) == 64 and not set(value) - set("0123456789abcdef")


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "JSON contains a duplicate key: " + key)
        result[key] = value
    return result


def invalid_constant(value):
    raise ValueError("non-finite JSON constant refused: " + value)


def read_pinned_json(path, expected):
    require(valid_sha(expected), "external lowercase SHA256 required")
    raw = path.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    require(actual == expected, "JSON byte SHA256 mismatch: " + str(path.relative_to(ROOT)))
    value = json.loads(raw, object_pairs_hook=unique_object, parse_constant=invalid_constant)
    require(isinstance(value, dict), "JSON object required")
    return value, dict(path=str(path.relative_to(ROOT)), sha256=actual)


def status_path(argument):
    value = Path(argument)
    path = value if value.is_absolute() else ROOT / value
    require(".." not in value.parts, "parent path components refused")
    require(path.resolve(strict=True) == path and path.is_file(), "canonical status file required; aliases refused")
    require(path.is_relative_to(FRAMEWORK), "status must be inside this project's runtime/framework")
    # A saved root/reporter observation is identified by its owner_status.json
    # name. Its bytes must still have the caller's external SHA and1071 binding.
    require(path == CURRENT_STATUS or path.name == "owner_status.json",
            "only current1071 status.json or a saved owner_status.json observation is admitted")
    return path


def observation_utc(value):
    if value is None:
        return datetime.now(timezone.utc).isoformat()
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    require(parsed.tzinfo is not None and parsed.utcoffset().total_seconds() == 0,
            "observation timestamp must be an explicit UTC ISO8601 time")
    return parsed.astimezone(timezone.utc).isoformat()


def summarize(status, plan, status_pointer, plan_pointer, observed, expected_completed):
    require(plan.get("split") == status.get("split") == "test"
            and plan.get("expected_cases") == status.get("expected") == 1920,
            "original full1920 TEST plan/status required")
    for field in ("policy_sha256", "resume_control", "resume_control_lock"):
        require(field in plan and status.get(field) == plan[field], "status/1071 plan binding differs: " + field)
    jobs = plan.get("jobs")
    require(isinstance(jobs, list) and len(jobs) == 1920, "exact1920 plan rows required")
    indexed, coordinates, raw_groups = {}, set(), {}
    for row in jobs:
        require(isinstance(row, dict), "plan row must be an object")
        key, task, raw, method = (row.get(k) for k in ("job_id", "task", "raw", "method"))
        require(isinstance(key, str) and key and key not in indexed, "unique nonempty plan job IDs required")
        require(task in TASK_LABELS and method in METHODS and isinstance(raw, str) and raw,
                "only original16 tasks, three methods and explicit raw IDs are admitted")
        coordinate = task, raw, method
        require(coordinate not in coordinates, "duplicate task/raw/method coordinate")
        coordinates.add(coordinate)
        indexed[key] = row
        raw_groups.setdefault((task, method), set()).add(raw)
    require(len(raw_groups) == 48
            and all(len(raw_groups.get((task, method), ())) == 40 for task in TASK_LABELS for method in METHODS)
            and all(raw_groups[task, method] == raw_groups[task, METHODS[0]] for task in TASK_LABELS for method in METHODS),
            "same original40 raw IDs per task across all three methods required")
    completed, failed = status.get("completed"), status.get("failed")
    require(isinstance(completed, dict) and isinstance(failed, dict), "completed and failed dictionaries required")
    require(set(completed) <= set(indexed) and set(failed) <= set(indexed)
            and not set(completed) & set(failed), "unknown or overlapping completed/failed job keys")
    if expected_completed is not None:
        require(0 <= expected_completed <= 1920 and len(completed) == expected_completed,
                "completed count differs from --expected-completed")
    counts, successes = Counter(), Counter()
    for key, record in completed.items():
        require(isinstance(record, dict) and record.get("complete") is True, "partial record inside completed: " + key)
        require(type(record.get("native_success")) is bool,
                "Native outcome must be bool; None/unknown is never counted as False: " + key)
        row = indexed[key]
        pair = row["task"], row["method"]
        counts[pair] += 1
        successes[pair] += int(record["native_success"])
        require(counts[pair] <= 40, "task/method completion count exceeds40")
    require("owner_error" in status, "explicit owner_error status key required")
    for field in ("stop_requested", "drain_requested"):
        require(type(status.get(field)) is bool, "explicit status flag required: " + field)
    resource = status.get("resource_profile")
    require(isinstance(resource, dict) and isinstance(resource.get("schema"), str)
            and all(type(resource.get(k)) is int and resource[k] >= 1 for k in
                    ("max_native_slots", "max_GPU_slots", "max_CPU_preflight_slots", "ready_limit")),
            "recorded actual resource profile required; capacity is not substituted")

    def metrics(won, done, expected):
        return dict(successes=won, completed=done, expected=expected,
                    success_rate=None if done == 0 else won / done,
                    success_percentage=None if done == 0 else 100 * won / done)

    tasks = [dict(task=task, label=label,
                  task_samples_complete=all(counts[task, method] == 40 for method in METHODS),
                  methods={method: metrics(successes[task, method], counts[task, method], 40) for method in METHODS})
             for task, label in TASK_LABELS.items()]
    per_method = {method: metrics(sum(successes[task, method] for task in TASK_LABELS),
                                  sum(counts[task, method] for task in TASK_LABELS), 640) for method in METHODS}
    require(sum(value["completed"] for value in per_method.values()) == len(completed), "completion accounting differs")
    queues = {}
    for field in ("active", "checking", "ready_jobs"):
        names = status.get(field)
        require(isinstance(names, list) and all(isinstance(k, str) and k in indexed for k in names),
                "known plan IDs required in queue: " + field)
        queues[field] = len(names)
    return dict(schema="r10_current_TEST_progress_display_v1", observation_utc=observed,
                disclaimer=DISCLAIMER, independent_audit_confirmed=False, pure_display=True,
                status=status_pointer, plan=plan_pointer, expected_method_cases=1920,
                completed_method_cases=len(completed), progress_percentage=100 * len(completed) / 1920,
                task_count=16, tasks_samples_complete=sum(row["task_samples_complete"] for row in tasks),
                per_method=per_method, per_task=tasks, infrastructure_failed_count=len(failed),
                owner_error=status.get("owner_error"), stop_requested=status["stop_requested"],
                drain_requested=status["drain_requested"], actual_resource_profile=resource,
                resource_capacity_envelope=status.get("resource_capacity_envelope"), queue_counts=queues,
                status_updated_utc=status.get("updated_utc", status.get("created_utc")))


def cell(value):
    if value["completed"] == 0:
        return "待评（0/40）"
    return f'{value["successes"]}/{value["completed"]}（{value["success_percentage"]:.1f}%）'


def render_txt(view):
    observed = datetime.fromisoformat(view["observation_utc"])
    local = observed.astimezone(timezone(timedelta(hours=8)))
    profile = view["actual_resource_profile"]
    lines = ["TEST 16任务运行进度", view["disclaimer"],
             f"观察时刻：{observed.isoformat()}（北京时间{local.strftime('%Y-%m-%d %H:%M:%S')}）",
             f"已完成：{view['completed_method_cases']}/1920（{view['progress_percentage']:.2f}%）；"
             f"三组样本均跑齐的任务：{view['tasks_samples_complete']}/16", ""]
    for method in METHODS:
        value = view["per_method"][method]
        rate = "待评" if value["completed"] == 0 else f"{value['success_percentage']:.2f}%"
        lines.append(f"{METHOD_LABELS[method]}：{value['successes']}/{value['completed']}，{rate}；进度{value['completed']}/640")
    lines += ["", "下表为成功/已完成（成功率）；每个任务每组目标40次，待评不计为0%成功率。",
              "任务 | CV＋几何 | Context＋几何 | PhysMani | 三组样本跑齐",
              "--- | --- | --- | --- | ---"]
    for row in view["per_task"]:
        lines.append(" | ".join([row["label"], *(cell(row["methods"][m]) for m in METHODS),
                                 "是" if row["task_samples_complete"] else "否"]))
    error = "无" if view["owner_error"] is None else json.dumps(view["owner_error"], ensure_ascii=False)
    lines += ["", f"基础设施failed：{view['infrastructure_failed_count']}；owner_error：{error}",
              f"stop_requested：{view['stop_requested']}；drain_requested：{view['drain_requested']}",
              f"实际配额：Native{profile['max_native_slots']} / GPU{profile['max_GPU_slots']} / "
              f"CPU预检{profile['max_CPU_preflight_slots']} / ready{profile['ready_limit']}；schema：{profile['schema']}",
              "状态文件：" + view["status"]["path"], "状态SHA256：" + view["status"]["sha256"],
              "固定计划：" + view["plan"]["path"], "计划SHA256：" + view["plan"]["sha256"],
              "即使1920条跑齐，本展示也不确认完整独审完成；正式结果以独立审计产物为准。"]
    return "\n".join(lines) + "\n"


def write_txt(value):
    require(OUTPUT.parent.resolve(strict=True) == ROOT and not OUTPUT.is_symlink(), "fixed TXT target must be canonical")
    require(not OUTPUT.exists() or OUTPUT.is_file(), "fixed TXT target is not a regular file")
    mode = stat.S_IMODE(OUTPUT.stat().st_mode) if OUTPUT.exists() else 0o644
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=ROOT,
                                         prefix=".TEST_16任务进度.", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            os.fchmod(stream.fileno(), mode)
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        require(not OUTPUT.is_symlink(), "fixed TXT alias appeared before replacement")
        os.replace(temporary, OUTPUT)
        temporary = None
        descriptor = os.open(ROOT, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--status-file", required=True)
    parser.add_argument("--status-sha256", required=True)
    parser.add_argument("--observation-utc")
    parser.add_argument("--expected-completed", type=int)
    parser.add_argument("--write", action="store_true", help="atomically fsync only fixed TEST_16任务进度.txt")
    args = parser.parse_args(argv)
    observed = observation_utc(args.observation_utc)
    path = status_path(args.status_file)
    require(PLAN.resolve(strict=True) == PLAN, "fixed1071 plan alias refused")
    status, status_pointer = read_pinned_json(path, args.status_sha256)
    plan, plan_pointer = read_pinned_json(PLAN, PLAN_SHA256)
    view = summarize(status, plan, status_pointer, plan_pointer, observed, args.expected_completed)
    preview = render_txt(view)
    if args.write:
        write_txt(preview)
    result = dict(view, written=args.write, output_path=str(OUTPUT.relative_to(ROOT)), txt_preview=preview)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return result


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, TypeError, KeyError) as error:
        print(json.dumps(dict(error=str(error), written=False, disclaimer=DISCLAIMER), ensure_ascii=False), file=sys.stderr)
        raise SystemExit(2)
