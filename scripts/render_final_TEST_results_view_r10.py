#!/usr/bin/env python3
"""Display the existing complete1072/1075 products; never run their producers.

Default: JSON preview only.  --write atomically updates exactly one display TXT.
Only fixed metadata/result files are opened; case pointers are never followed.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]
OWNER = "runtime/framework/formal_TEST_reboot_resume_v1071"
CONTROLLER = "runtime/framework/ordered_second_reboot_TEST_v1073"
POST = "runtime/framework/post_complete_second_reboot_TEST_motion_v1076"
RESULT = "artifacts/framework/multitask_r10/formal_TEST_reboot_resume_v1071/TEST_results.json"
MOTION = OWNER + "/full_sixteen_task_motion_summary_v1075.json"
OUTPUT = "TEST_16任务最终结果.txt"
RESULT_SCHEMA = "r10_seven_tier_second_reboot_formal_TEST_results_v1072"
MOTION_SCHEMA = "r10_full1920_seven_tier_TEST_saved_motion_summary_v1075"
POST_SCHEMA = "r10_readonly_post_complete_reboot_TEST_motion_terminal_v1076"
AUDITOR_SHA = "2fc1064e4a9ded840f5e3543f56e2f39e49cedfd3ffe2a3183e561ed1155c1e3"
MOTION_SOURCE_SHA = "6c29b7fdcaddd25549130ea4f3164d68e98b1ac685d580978f9bbf5373710ffe"
POLICY = "eb8f768d99975a7c2de2c36de5283d64898c2296f33fb342e332310eb32db6ef"
METHODS = ("cv_geometry", "context_geometry", "physmani")
LABELS = {
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
FILES = (
    OWNER + "/plan.json", OWNER + "/status.json", OWNER + "/terminal.json",
    CONTROLLER + "/status.json", CONTROLLER + "/terminal.json", RESULT, MOTION,
    POST + "/status.json", POST + "/terminal.json",
)
TIERS = {
    "runtime/framework/formal_TEST_after_environment_resume_v753": 134,
    "runtime/framework/formal_TEST_concurrent_resume_v902": 450,
    "runtime/framework/formal_TEST_composite_resume_v956": 682,
    "runtime/framework/formal_TEST_composite_resume_v1011": 143,
    "runtime/framework/formal_TEST_composite_resume_v1041": 2,
    "runtime/framework/formal_TEST_reboot_resume_v1056": 8,
    OWNER: 501,
}


class NotReady(Exception):
    pass


def require(value, message):
    if not value:
        raise ValueError(message)


def checksum(raw):
    return hashlib.sha256(raw).hexdigest()


def read_products():
    missing = [name for name in FILES if not (ROOT / name).is_file()]
    if missing:
        raise NotReady("缺少完整产物：" + "；".join(missing))
    data, pointers = {}, {}
    for name in FILES:
        path = ROOT / name
        require(path.resolve(strict=True) == path, "固定产物路径不能是别名：" + name)
        raw = path.read_bytes()
        value = json.loads(raw)
        require(isinstance(value, dict), "JSON对象格式错误：" + name)
        data[name] = value
        pointers[name] = {"path": name, "sha256": checksum(raw)}
    return data, pointers


def declared_sha(mapping, name):
    require(isinstance(mapping, dict), "缺少已保存输入SHA映射")
    values = [mapping[key] for key in (name, str(ROOT / name)) if key in mapping]
    require(len(values) == 1, "产物输入必须唯一声明：" + name)
    return values[0]


def pointer(value, expected_path):
    require(isinstance(value, dict) and set(value) == {"path", "sha256"}
            and value.get("path") == expected_path and isinstance(value.get("sha256"), str)
            and len(value["sha256"]) == 64 and not set(value["sha256"]) - set("0123456789abcdef"),
            "缺少固定产物SHA指针：" + expected_path)
    return value


def flag(value, expected, label):
    require(type(value) is bool and value is expected, label)


def check_products(data, ptr):
    plan, status, terminal = (data[OWNER + "/" + n + ".json"] for n in ("plan", "status", "terminal"))
    ctl, post = data[CONTROLLER + "/terminal.json"], data[POST + "/terminal.json"]
    result, motion = data[RESULT], data[MOTION]
    if (terminal.get("complete") is not True or ctl.get("complete") is not True
            or post.get("complete") is not True or result.get("complete") is not True
            or motion.get("complete") is not True
            or data[CONTROLLER + "/status.json"].get("stage") != "complete"
            or data[POST + "/status.json"].get("stage") != "complete"):
        raise NotReady("原执行、独立审计或运动汇总尚未全部完成")
    require(result.get("schema") == RESULT_SCHEMA and motion.get("schema") == MOTION_SCHEMA
            and post.get("schema") == POST_SCHEMA, "完整产物schema不一致")
    require(terminal.get("expected") == terminal.get("completed") == 1920
            and terminal.get("failed") == 0 and terminal.get("owner_error") is None
            and terminal.get("imported_completed_cases") == 1419
            and terminal.get("resumed_new_cases") == 501
            and terminal.get("CPU_not_launched") == 0 and terminal.get("CPU_ready_unexecuted") == [],
            "1071终态不是完整1920/保留1419/新增501")
    for key in ("stop_requested", "drain_requested", "drained", "automatic_retry"):
        flag(terminal.get(key), False, "1071终态标志异常：" + key)
    for obj, label in ((ctl, "1073"), (post, "1076"), (result, "1072")):
        flag(obj.get("automatic_retry"), False, label + "不是原单次执行产物")
    flag(post.get("Native_GPU_TRAIN_started"), False, "1076不是原保存结果展示流程")
    require(status.get("expected") == 1920 and status.get("failed") == {}
            and status.get("owner_error") is None and status.get("imported_completed") == 1419
            and status.get("active") == status.get("checking") == status.get("ready_jobs") == []
            and status.get("CPU_ready") == status.get("native_slots") == status.get("GPU_slots") == 0,
            "1071最后状态不是完整空闲状态")
    for key in ("stop_requested", "drain_requested", "automatic_retry"):
        flag(status.get(key), False, "1071状态标志异常：" + key)
    require(result.get("split") == motion.get("split") == plan.get("split") == status.get("split") == "test"
            and result.get("source_cases") == 640
            and result.get("method_cases") == result.get("completed_method_cases") == 1920
            and motion.get("task_count") == 16 and motion.get("source_count") == 640 and motion.get("case_count") == 1920
            and result.get("methods") == motion.get("methods") == list(METHODS)
            and isinstance(result.get("tasks"), list) and len(result["tasks"]) == 16
            and set(result["tasks"]) == set(LABELS)
            and motion.get("task_source_counts") == {task: 40 for task in LABELS},
            "产物不是同一个16任务/640来源/1920三组网格")
    require(result.get("source_sha256") == AUDITOR_SHA and motion.get("source_sha256") == MOTION_SOURCE_SHA
            and motion.get("actual_test_auditor") == {"path": "scripts/audit_second_reboot_TEST_results_v1072_r10.py", "sha256": AUDITOR_SHA},
            "不是本轮已固定的1072/1075产物")
    require(ctl.get("TEST_results") == post.get("TEST_results") == motion.get("test_results") == ptr[RESULT]
            and post.get("motion_summary") == ptr[MOTION], "既有终态与真实结果/运动文件SHA指针不一致")
    for key in ("resume_control", "resume_control_lock"):
        expected = "artifacts/framework/multitask_r10/formal_TEST_reboot_resume_v1071_setup/" + (
            "resume_control.json" if key == "resume_control" else "resume_inputs.json")
        pointer(result.get(key), expected)
        require(result.get(key) == motion.get(key) == post.get(key) == plan.get(key) == status.get(key) == terminal.get(key),
                "产物所属冻结批次不一致：" + key)
    require(result.get("policy_sha256") == motion.get("policy_sha256") == plan.get("policy_sha256") == POLICY,
            "产物所属策略不一致")
    for key in ("independent_composite_audit", "used_evidence_rechecked_at_close", "native_failures_retained",
                "motion_unknown_retained", "zero_tick_cases_retained", "motion_is_diagnostic_only", "adopted_completed_results_not_rerun"):
        flag(result.get(key), True, "正式结果保留声明不一致：" + key)
    for key in ("source_outcome_filtering", "Value_NGFF_FASTER_used"):
        flag(result.get(key), False, "正式结果范围不一致：" + key)
    for key in ("read_only_analysis", "native_false_retained", "unknown_motion_not_filled_as_zero_or_boolean",
                "full1920_seven_tier_TEST_results_and_complete1073_observed"):
        flag(motion.get(key), True, "运动汇总保留声明不一致：" + key)
    for key in ("success_rate_or_acceleration_filter_used", "native_GPU_or_jobs_started", "TEST_used_for_fitting",
                "new_release_granted_by_this_summary", "TXT_or_handoff_written"):
        flag(motion.get(key), False, "运动汇总用途不一致：" + key)
    require(motion.get("adopted_completed_cases") == motion.get("saved_registered_completed_cases") == 1419
            and motion.get("damaged_saved_completed_cases_reexecuted") == 0
            and motion.get("historical_damaged_saved_completed_cases_reexecuted") == 2,
            "运动汇总中的恢复分区不一致")
    for key in ("host_reboot_interruption", "prior_host_reboot_interruption", "host_reboot_interruptions",
                "reboot_audit", "host_reboot_observation", "prior_reboot_audit", "prior_host_reboot_observation",
                "resource_profile", "resource_qualification", "resource_qualifier", "resource_capacity_envelope",
                "resource_capacity_source", "resource_profile_within_capacity", "resource_profile_changed",
                "previous_resource_profile", "previous_resource_qualification"):
        require(key in result and motion.get(key) == result[key], "汇总没有保留原结果字段：" + key)
    owners = result.get("actual_stage_owners")
    require(isinstance(owners, list) and len(owners) == 7
            and {v.get("runtime"): v.get("completed") for v in owners} == TIERS
            and motion.get("actual_stage_owners") == owners and motion.get("provenance_runtime_counts") == TIERS,
            "七层实际结果数量不一致")
    for owner in owners:
        if owner["runtime"] in ("runtime/framework/formal_TEST_composite_resume_v1041", "runtime/framework/formal_TEST_reboot_resume_v1056"):
            require(owner.get("terminal") is None, "旧重启中断进程不应产生成功终态")
        elif owner["runtime"] == OWNER:
            require(owner.get("terminal") == ptr[OWNER + "/terminal.json"], "1072没有绑定真实1071终态")
    for name in (OWNER + "/plan.json", OWNER + "/status.json", OWNER + "/terminal.json"):
        require(declared_sha(result.get("inputs_sha256"), name) == ptr[name]["sha256"], "1072读取的元数据SHA不同：" + name)
    for name in (RESULT, OWNER + "/plan.json", OWNER + "/status.json", OWNER + "/terminal.json",
                 CONTROLLER + "/terminal.json"):
        require(declared_sha(motion.get("inputs_sha256"), name) == ptr[name]["sha256"], "1075读取的元数据SHA不同：" + name)
    return plan, status, result, motion


def tabulate(plan, status, result, motion):
    jobs = plan.get("jobs")
    require(isinstance(jobs, list) and len(jobs) == 1920, "计划不是1920条")
    by_id = {row["job_id"]: row for row in jobs}
    completed = status.get("completed")
    require(len(by_id) == 1920 and isinstance(completed, dict) and set(completed) == set(by_id), "完整结果键集合不一致")
    grid, successes, motion_false, motion_unknown = Counter(), Counter(), Counter(), Counter()
    sources = {(task, method): set() for task in LABELS for method in METHODS}
    coordinates = set()
    coordinate_ids = {}
    for key, job in by_id.items():
        task, method, raw = job["task"], job["method"], job["raw"]
        require(task in LABELS and method in METHODS and isinstance(raw, str), "计划坐标不在固定三组任务内")
        coordinate = task, raw, method
        require(coordinate not in coordinates, "重复任务/来源/方法")
        coordinates.add(coordinate)
        coordinate_ids[coordinate] = key
        record = completed[key]
        require(record.get("complete") is True and type(record.get("native_success")) is bool
                and "motion_pass" in record and (record["motion_pass"] is None or type(record["motion_pass"]) is bool),
                "最终记录缺少原生bool或明确的运动bool/None")
        grid[task, method] += 1
        sources[task, method].add(raw)
        successes[task, method] += int(record["native_success"])
        motion_false[task, method] += int(record["motion_pass"] is False)
        motion_unknown[task, method] += int(record["motion_pass"] is None)
    for task in LABELS:
        require(all(grid[task, method] == len(sources[task, method]) == 40 for method in METHODS)
                and sources[task, METHODS[0]] == sources[task, METHODS[1]] == sources[task, METHODS[2]],
                "每任务必须是相同40来源×3组：" + task)
    totals = {method: sum(successes[task, method] for task in LABELS) for method in METHODS}
    failed_motion = {method: sum(motion_false[task, method] for task in LABELS) for method in METHODS}
    unknown_motion = {method: sum(motion_unknown[task, method] for task in LABELS) for method in METHODS}
    require(totals == result.get("native_successes_by_method")
            and failed_motion == result.get("measured_motion_violations_by_method")
            and unknown_motion == result.get("measured_motion_unknown_by_method"), "展示计数与1072正式结果不同")
    require(sum(totals.values()) == status_native_total(status), "原生成功总数不一致")
    rows = motion.get("case_evidence")
    require(isinstance(rows, list) and len(rows) == 1920, "运动逐条证据不是1920条")
    seen = set()
    zero_ticks, unknown_ticks = Counter(), Counter()
    for row in rows:
        coordinate = row["task"], row["source_raw_episode_id"], row["method"]
        require(coordinate in coordinates and coordinate not in seen, "运动汇总坐标缺失或重复")
        seen.add(coordinate)
        key = coordinate_ids[coordinate]
        require(key in completed and row.get("official_task_success") is completed[key]["native_success"]
                and row.get("case") == completed[key].get("case"), "运动汇总没有保留相同Native结果")
        metrics = row.get("metrics")
        require(isinstance(metrics, dict) and "counted_ticks" in metrics, "运动汇总缺少实际counted_ticks字段")
        ticks = metrics["counted_ticks"]
        require(ticks is None or (type(ticks) in (int, float) and math.isfinite(ticks)),
                "保存的counted_ticks必须是有限数或None")
        zero_ticks[row["method"]] += int(type(ticks) in (int, float) and ticks == 0)
        unknown_ticks[row["method"]] += int(ticks is None)
    require(seen == coordinates, "运动汇总不是同一完整网格")
    grouped = motion.get("by_task_method")
    require(isinstance(grouped, list) and len(grouped) == 48
            and {(row["task"], row["method"]) for row in grouped} == set(grid), "运动逐任务48格不完整")
    for row in grouped:
        coordinate = row["task"], row["method"]
        require(row.get("case_count") == row.get("source_count") == 40
                and row.get("native_success_cases") == successes[coordinate]
                and row.get("native_false_cases") == 40 - successes[coordinate], "运动逐任务原生分母或成功数不同")
    overall = motion.get("overall_by_method")
    require(isinstance(overall, dict) and set(overall) == set(METHODS), "缺少三组运动汇总")
    for method in METHODS:
        row = overall[method]
        require(row.get("case_count") == row.get("source_count") == 640
                and row.get("native_success_cases") == totals[method]
                and row.get("native_false_cases") == 640 - totals[method], "运动总体原生分母或成功数不同")
    return dict(native_by_task={task: {method: successes[task, method] for method in METHODS} for task in LABELS},
                native_by_method=totals, motion_false=failed_motion, motion_unknown=unknown_motion,
                zero_counted_ticks={method: zero_ticks[method] for method in METHODS},
                unknown_counted_ticks={method: unknown_ticks[method] for method in METHODS})


def status_native_total(status):
    return sum(int(row["native_success"]) for row in status["completed"].values())


def number(value):
    if value is None:
        return "None（未知）"
    require(type(value) in (int, float) and math.isfinite(value), "已保存统计值不是有限数或None")
    return format(value, ".6g")


def render_text(table, motion, ptr):
    stamp = datetime.now(timezone.utc).astimezone(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")
    lines = ["16任务正式TEST最终结果", "展示生成时间：" + stamp,
             "来源：已完成的1072独立结果与1075保存运动汇总；本脚本仅展示，不执行评测或另授放行。",
             "固定16任务×每任务40来源×三组；Native成功按原任务判定，运动诊断不作为成功率阈值。", "",
             "任务｜CV+几何 成功/40｜Context+几何 成功/40｜PhysMani 成功/40"]
    for task, label in LABELS.items():
        values = table["native_by_task"][task]
        lines.append(label + "｜" + "｜".join(f"{values[method]}/40（{100 * values[method] / 40:.1f}%）" for method in METHODS))
    lines += ["总体（按640来源合并）｜" + "｜".join(
        f"{table['native_by_method'][method]}/640（{100 * table['native_by_method'][method] / 640:.2f}%）" for method in METHODS), "",
        "保存的运动诊断（均保留NativeFalse；False和None分别计数）",
        "方法｜motion_pass=False案例｜motion_pass=None案例｜counted_ticks=0案例｜counted_ticks=None案例"]
    for method in METHODS:
        lines.append(method + "｜" + "｜".join(str(table[key][method]) for key in
            ("motion_false", "motion_unknown", "zero_counted_ticks", "unknown_counted_ticks")))
    lines += ["", "保存的运动数值（各case峰值的P95；0保留为0，None表示未知，不补零）",
              "方法｜关节速度峰值P95(rad/s)｜未知case｜关节加速度峰值P95(rad/s²)｜未知case"]
    for method in METHODS:
        metrics = motion["overall_by_method"][method]["metrics"]
        speed = metrics["peak_window_joint_speed_rad_s"]
        acceleration = metrics["peak_window_joint_acceleration_rad_s2"]
        lines.append(method + "｜" + "｜".join((number(speed["p95"]), str(speed["unknown_or_nonfinite_cases"]),
            number(acceleration["p95"]), str(acceleration["unknown_or_nonfinite_cases"]))))
    lines += ["", "诊断定义沿用1075的metric_definitions、diagnostic_definition与statistical_definition；不重新差分或筛选成功案例。",
              "两次重启历史保留：1419条已有完整结果＋501条本次结果；旧1041/1056中断终态仍为None。",
              "结果文件：" + RESULT, "结果SHA256：" + ptr[RESULT]["sha256"],
              "运动汇总：" + MOTION, "运动SHA256：" + ptr[MOTION]["sha256"], ""]
    return "\n".join(lines)


def unchanged(ptr):
    for name in FILES:
        require(checksum((ROOT / name).read_bytes()) == ptr[name]["sha256"], "展示期间产物发生变化：" + name)


def write_display(text):
    target = ROOT / OUTPUT
    require(target.resolve() == target and not target.is_symlink(), "最终展示TXT不能是别名")
    temp = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=ROOT,
                                         prefix=".TEST_16任务最终结果.", suffix=".tmp", delete=False) as stream:
            temp = Path(stream.name)
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, target)
        dfd = os.open(ROOT, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
    finally:
        if temp is not None:
            temp.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="仅写TEST_16任务最终结果.txt；默认只输出JSON预览")
    args = parser.parse_args()
    write_attempted = False
    try:
        data, ptr = read_products()
        plan, status, result, motion = check_products(data, ptr)
        table = tabulate(plan, status, result, motion)
        require(data[OWNER + "/terminal.json"].get("native_successes") == sum(table["native_by_method"].values()),
                "1071终态原生成功总数与正式结果不一致")
        text = render_text(table, motion, ptr)
        unchanged(ptr)
        if args.write:
            write_attempted = True
            write_display(text)
        print(json.dumps(dict(ready=True, written=args.write, output=OUTPUT if args.write else None,
                              metadata_consistency_check_only=True, independent_audit_rerun=False,
                              source_products=ptr, **table, text=text), ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    except NotReady as error:
        print(json.dumps(dict(ready=False, written=False, status="not_ready", reason=str(error)), ensure_ascii=False))
        return 0
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as error:
        print(json.dumps(dict(ready=False, written=None if write_attempted else False,
                              status="display_write_failed" if write_attempted else "saved_metadata_inconsistent",
                              reason=str(error)), ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
