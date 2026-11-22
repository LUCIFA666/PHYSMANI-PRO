"""Read-only ten-minute NGFF summaries and optional desktop notifications.

This reporter reads existing validation results and does not run a GPU
evaluation, modify checkpoints, or change the training process.
"""
import argparse
import json
import math
import subprocess
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "runtime/continuous/state"
OUT = ROOT / "artifacts/continuous"
REPORTS = OUT / "reports"
LOCAL_TIME = ZoneInfo("Asia/Shanghai")


def read(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError):
        return default


def atomic_write(path, content):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def clock(timestamp):
    return datetime.fromtimestamp(timestamp, LOCAL_TIME).strftime("%Y-%m-%d %H:%M:%S")


def number(value):
    return f"{value:.4f}" if isinstance(value, (int, float)) and math.isfinite(value) else "暂无"


def percent_error(score, baseline):
    if (not isinstance(score, (int, float)) or not isinstance(baseline, (int, float))
            or not math.isfinite(score) or not math.isfinite(baseline) or baseline <= 0):
        return None
    return 100 * (score / baseline - 1)


def comparison(delta):
    if delta is None:
        return "暂无可比值"
    return f"高 {delta:.1f}%" if delta >= 0 else f"低 {-delta:.1f}%"


def validation_history(path):
    rows = []
    if path.exists():
        with path.open() as stream:
            for line in stream:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue  # A concurrent append may not yet have a newline.
                if row.get("kind") == "validation":
                    rows.append(row)
    return rows


def summarize(now, supervisor, trainer, validation, history, previous, service_active):
    alerts = []
    if service_active != "active":
        alerts.append(f"训练服务状态为 {service_active}。")
    for label, data, limit in (("后台状态", supervisor, 120), ("训练状态", trainer, 300)):
        if not data or now - data.get("time", 0) > limit:
            alerts.append(f"{label}超过 {limit} 秒未更新。")
    if trainer.get("status") not in ("running", "validating", "waiting_for_data", "waiting_for_full_resolution_data"):
        alerts.append("训练状态：" + str(trainer.get("status", "未知")))
    if validation and now - validation.get("time", 0) > 600:
        alerts.append("验证结果超过 10 分钟未更新，以下是已有结果。")
    signature = validation.get("cohort_signature")
    cohort = [x for x in history if signature and x.get("cohort_signature") == signature
              and x.get("time", 0) <= validation.get("time", now)]
    valid_best = [x for x in cohort if isinstance(x.get("mean_scene_mse_excluding_initial_frame"), (int, float))]
    best = min(valid_best, key=lambda x: x["mean_scene_mse_excluding_initial_frame"]) if valid_best else validation
    ngff = validation.get("ngff", {})
    baseline = validation.get("gravity_with_floor_stop", {})
    deltas = {key: percent_error(ngff.get(key), baseline.get(key)) for key in ("rmse", "fpe")}

    if not validation:
        conclusion = "尚无验证结果，不能判断预测质量。"
    elif deltas["rmse"] is None or deltas["fpe"] is None:
        conclusion = "验证指标不完整，不能判断相对基线的效果。"
    elif deltas["rmse"] < 0 and deltas["fpe"] <= 0:
        conclusion = "本批验证的位置误差和终点误差均优于物理基线，仍需观察稳定性。"
    elif abs(deltas["rmse"]) <= 1 and deltas["fpe"] <= 1:
        conclusion = "位置精度与物理基线接近，尚无明确优势。"
    else:
        conclusion = "尚未同时超过物理基线的位置精度和终点精度。"

    previous_validation = (previous or {}).get("validation", {})
    if previous_validation.get("cohort_signature") == signature and signature:
        start = previous_validation
        trend_basis = "与上次定时报告的相同验证集相比"
    else:
        recent = [x for x in cohort if x.get("time", 0) >= now - 600]
        start = recent[0] if recent else validation
        trend_basis = "本轮相同验证集内"
    if start.get("update") == validation.get("update"):
        trend = "本轮暂无新的同组验证结果，不判断趋势。"
    else:
        delta = percent_error(ngff.get("rmse"), start.get("ngff", {}).get("rmse"))
        trend = (f"{trend_basis}，RMSE 从 {number(start.get('ngff', {}).get('rmse'))} "
                 f"到 {number(ngff.get('rmse'))}（{comparison(delta)}）。")
    previous_signature = previous_validation.get("cohort_signature")
    if previous_signature and previous_signature != signature:
        trend = (f"验证集从 {previous_validation.get('scenes')} 增至 {validation.get('scenes')} 个场景，"
                 "不跨集合比较总体均值。" + trend)
    rows = []
    for label, key in (("RMSE ↓", "rmse"), ("终点误差 FPE ↓", "fpe"),
                       ("轨迹相关性 R ↑", "r"), ("质心终点误差 ↓", "centroid_final_error")):
        rows.append(f"  {label}：最新 {number(ngff.get(key))} / "
                    f"本验证集最佳 {number(best.get('ngff', {}).get(key))} / "
                    f"物理基线 {number(baseline.get(key))}")
    cache = supervisor.get("canonical_cache", {})
    last_train = trainer.get("last_train", {})
    progress = (f"完整分辨率更新 {trainer.get('canonical_updates', 0)} 次；"
                f"数据 {cache.get('train_ready', 0)}/{cache.get('planned_train', 0)} 训练、"
                f"{cache.get('validation_ready', 0)}/{cache.get('planned_validation', 0)} 验证。")
    batch = (f"目标 batch={trainer.get('target_batch_size', 32)}；"
             f"最近实际 batch={last_train.get('actual_batch_size', '暂无')} "
             "（轮末批次可能较小）。")
    text = "\n".join([
        "NGFF 每 10 分钟效果报告", f"生成时间：{clock(now)}（北京时间）", "",
        "运行异常：" + " ".join(alerts) if alerts else "运行状态：正常",
        "效果判断：" + conclusion,
        (f"最近验证：{clock(validation['time'])}，总更新 {validation.get('update')}，"
         f"{validation.get('scenes')} 个场景，预测未来 {validation.get('horizon_s')} 秒。")
        if validation else "最近验证：暂无",
        *rows,
        f"最新 RMSE 比基线{comparison(deltas['rmse'])}，终点误差比基线{comparison(deltas['fpe'])}。",
        (f"本验证集最佳权重：总更新 {best.get('update')}；按平均场景 MSE 选择，"
         "各项指标均取同一权重。") if best else "最佳权重：暂无",
        "趋势：" + trend, "", progress, batch,
        f"分辨率：每物体 {trainer.get('keypoints_per_object')} 点，{trainer.get('frames')} 帧。",
        "误差单位：仿真原始坐标，不能直接当作米；结果是验证表现，不是正式测试或抓取成功率。",
        "报告位置：" + str(REPORTS / "latest.txt"),
    ]) + "\n"
    notification = "\n".join([
        " ".join(alerts) if alerts else conclusion,
        f"RMSE {number(ngff.get('rmse'))} / 基线 {number(baseline.get('rmse'))}（{comparison(deltas['rmse'])}）",
        f"FPE {number(ngff.get('fpe'))} / 基线 {number(baseline.get('fpe'))}",
        f"R {number(ngff.get('r'))}；验证 {validation.get('scenes', 0)} 场景",
        progress,
        "完整报告：桌面 our-mani/每10分钟效果报告.txt",
    ])
    return {"time": now, "time_local": clock(now), "interval_minutes": 10,
            "delivery": "local report; desktop notification when enabled; no chat push",
            "service_active": service_active, "alerts": alerts,
            "conclusion": conclusion, "trend": trend, "baseline_error_difference_percent": deltas,
            "validation": validation, "best_for_current_cohort": best,
            "trainer": {k: v for k, v in trainer.items() if k not in ("queue", "validation")},
            "canonical_cache": cache, "report_text": text, "notification_text": notification}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notify", action="store_true")
    args = parser.parse_args()
    REPORTS.mkdir(parents=True, exist_ok=True)
    previous = read(REPORTS / "latest.json", {})
    service = subprocess.run(["systemctl", "--user", "is-active", "our-mani-training.service"],
                             text=True, capture_output=True, timeout=5).stdout.strip() or "unknown"
    report = summarize(time.time(), read(STATE / "status.json", {}),
                       read(STATE / "trainer_status.json", {}),
                       read(OUT / "validation_latest.json", {}),
                       validation_history(OUT / "history.jsonl"), previous, service)
    if args.notify:
        try:
            result = subprocess.run([
                "notify-send", "--app-name=our-mani", "--icon=dialog-information",
                "--urgency=normal", "--expire-time=30000", "--print-id", "--",
                "NGFF 效果报告 · " + report["time_local"][11:16], report["notification_text"],
            ], text=True, capture_output=True, timeout=10)
            report["notification"] = {"sent": result.returncode == 0,
                                      "id": result.stdout.strip(), "error": result.stderr.strip()}
        except (OSError, subprocess.SubprocessError) as exc:
            report["notification"] = {"sent": False, "error": type(exc).__name__}
    else:
        report["notification"] = {"sent": False, "reason": "not requested"}
    stamp = datetime.fromtimestamp(report["time"], LOCAL_TIME).strftime("%Y%m%d-%H%M%S")
    content = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    for name in ("latest", stamp):
        atomic_write(REPORTS / f"{name}.txt", report["report_text"])
        atomic_write(REPORTS / f"{name}.json", content)
    with (REPORTS / "history.jsonl").open("a") as stream:
        stream.write(json.dumps(report, ensure_ascii=False) + "\n")
    shortcut = ROOT / "每10分钟效果报告.txt"
    if not shortcut.exists() and not shortcut.is_symlink():
        shortcut.symlink_to("artifacts/continuous/reports/latest.txt")
    print(report["report_text"], flush=True)
    print(json.dumps(report["notification"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
