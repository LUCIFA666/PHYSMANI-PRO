"""Concise local status without dumping the full optimizer or scene queue."""
import json
import time
from pathlib import Path

root = Path(__file__).resolve().parents[1]
state = root / "runtime/continuous/state"
def read(name):
    try:
        return json.loads((state / name).read_text())
    except (OSError, json.JSONDecodeError):
        return {}

s = read("status.json")
t = read("trainer_status.json")
d = read("data_status.json")
print("任务：", s.get("status", "启动中"), "状态更新距今：", round(time.time() - s.get("time", time.time()), 1), "秒")
print("训练：", t.get("status", "启动中"), "阶段：", t.get("phase"), "更新次数：", t.get("updates"))
print("实际 batch：", t.get("actual_batch_size"), "/ 目标：", t.get("target_batch_size"),
      "每物体点数：", t.get("keypoints_per_object"), "帧数：", t.get("frames"))
print("完整分辨率数据：", s.get("canonical_cache", {}))
print("当前下载：", d.get("current"), "帧进度：", d.get("completed_frames"), "/", d.get("frames"))
print("最近训练：", t.get("last_train", {}))
v = t.get("validation") or {}
print("最近验证：", {key: v.get(key) for key in ("update", "phase", "scenes", "ngff", "gravity_with_floor_stop")})
print("权重：", root / "artifacts/continuous/latest.pt")
print("日志：", root / "logs/continuous")
