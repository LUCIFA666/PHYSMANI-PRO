"""Render real exported NGFF rollouts as a labeled three-way comparison video."""
import argparse
import json
import math
import subprocess
from pathlib import Path

import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont

WIDTH, HEIGHT = 1536, 896
FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
BOLD = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
COLORS = [(45, 110, 205), (224, 112, 52), (42, 153, 115)]
INK, SECONDARY, LINE = (29, 38, 49), (88, 103, 118), (216, 224, 231)
CN = {"cloth": "布料", "bowl": "碗", "duck": "鸭子", "rope": "绳子", "panda": "熊猫",
      "ball": "球", "phone": "手机", "can": "易拉罐", "pillow": "枕头", "miku": "初音模型"}


def font(size, bold=False):
    return ImageFont.truetype(BOLD if bold else FONT, size)


def project(points):
    yaw, pitch = math.radians(-42), math.radians(23)
    x, y, z = np.moveaxis(np.asarray(points), -1, 0)
    horizontal = x * math.cos(yaw) + y * math.sin(yaw)
    away = -x * math.sin(yaw) + y * math.cos(yaw)
    vertical = z * math.cos(pitch) - away * math.sin(pitch)
    depth = away * math.cos(pitch) + z * math.sin(pitch)
    return np.stack([horizontal, vertical, depth], axis=-1)


def text(draw, pos, value, size=18, fill=INK, bold=False, anchor=None):
    draw.text(pos, value, font=font(size, bold), fill=fill, anchor=anchor)


def scene_frames(directory, scene, checkpoint_update):
    with np.load(directory / (scene["kind"] + ".npz")) as f:
        tracks = {key: f[key].copy() for key in ("truth", "ngff", "baseline")}
        errors = {key: f["error_" + key].copy() for key in ("ngff", "baseline")}
    all_points = np.concatenate([x[:, :, :512].reshape(-1, 3) for x in tracks.values()])
    low, high = all_points.min(axis=0), all_points.max(axis=0)
    low[2] = min(low[2], -.94)
    margin = np.maximum((high - low) * .08, .04)
    floor = np.array([[low[0]-margin[0], low[1]-margin[1], -.94],
                      [high[0]+margin[0], low[1]-margin[1], -.94],
                      [high[0]+margin[0], high[1]+margin[1], -.94],
                      [low[0]-margin[0], high[1]+margin[1], -.94]])
    projected = {key: project(values[:, :, :512]) for key, values in tracks.items()}
    combined = np.concatenate([x.reshape(-1, 3) for x in projected.values()] + [project(floor)])
    pmin, pmax = combined[:, :2].min(axis=0), combined[:, :2].max(axis=0)
    center = (pmin + pmax) / 2
    scale = min(440 / max(.01, pmax[0] - pmin[0]), 365 / max(.01, pmax[1] - pmin[1]))
    panel_x = [24, 528, 1032]

    def screen(points, panel):
        p = np.asarray(points)
        return np.stack([panel_x[panel] + 240 + (p[..., 0] - center[0]) * scale,
                         410 - (p[..., 1] - center[1]) * scale], axis=-1)

    base = Image.new("RGB", (WIDTH, HEIGHT), (250, 252, 254))
    draw = ImageDraw.Draw(base)
    text(draw, (28, 20), f"NGFF 动力学预测对比  ·  第 {checkpoint_update} 步最佳权重", 21, SECONDARY)
    text(draw, (28, 55), scene["label"], 29, bold=True)
    text(draw, (28, 100), scene["name"], 20, SECONDARY)
    lx = 28
    for i, obj in enumerate(scene["objects"]):
        draw.ellipse((lx, 139, lx+12, 151), fill=COLORS[i])
        name = CN.get(obj, obj)
        text(draw, (lx+20, 131), f"物体 {i+1}：{name}", 18)
        lx += 220
    text(draw, (1508, 105), "4 倍慢放 · 初始状态预测未来 1.58 秒", 18, SECONDARY, anchor="ra")

    for panel, title in enumerate(("真实仿真轨迹", "NGFF 预测", "重力＋地面停止基线")):
        text(draw, (panel_x[panel]+240, 169), title, 24, bold=True, anchor="ma")
        pts = screen(project(floor), panel)
        draw.polygon([tuple(p) for p in pts], fill=(239, 243, 247), outline=LINE, width=1)
        for fraction in (.25, .5, .75):
            for a, b, c, d in ((floor[0], floor[1], floor[3], floor[2]),
                               (floor[0], floor[3], floor[1], floor[2])):
                ends = screen(project(np.stack([a+(b-a)*fraction, c+(d-c)*fraction])), panel)
                draw.line([tuple(p) for p in ends], fill=LINE, width=1)
        origin = np.array([low[0], low[1], -.94])
        axis_size = max(high - low) * .18
        for axis, label in enumerate(("x", "y", "z")):
            end = origin.copy(); end[axis] += axis_size
            xy = screen(project(np.stack([origin, end])), panel)
            draw.line([tuple(p) for p in xy], fill=SECONDARY, width=1)
            text(draw, tuple(xy[1]), label, 15, SECONDARY)
        text(draw, (panel_x[panel]+240, 625), "空间坐标：仿真单位", 16, SECONDARY, anchor="ma")

    chart = (84, 716, 968, 825)
    y_max = max(float(max(errors["ngff"])), float(max(errors["baseline"]))) * 1.15
    def chart_xy(times, values):
        return np.stack([chart[0] + np.asarray(times) / 1.58 * (chart[2] - chart[0]),
                         chart[3] - np.asarray(values) / y_max * (chart[3] - chart[1])], axis=-1)
    text(draw, (28, 672), "预测误差随时间变化", 21, bold=True)
    for j, (key, label) in enumerate((("ngff", "NGFF"), ("baseline", "物理基线"))):
        x = 540 + j * 190
        draw.line((x, 688, x+35, 688), fill=COLORS[j], width=3)
        text(draw, (x+45, 674), label, 18)
    for y in (0, y_max / 2, y_max):
        py = chart_xy([0], [y])[0, 1]
        draw.line((chart[0], py, chart[2], py), fill=LINE)
        text(draw, (chart[0]-10, py), f"{y:.2f}", 15, SECONDARY, anchor="rm")
    for x in (0., .4, .8, 1.2, 1.58):
        px = chart_xy([x], [0])[0, 0]
        text(draw, (px, chart[3]+8), f"{x:.2f}", 15, SECONDARY, anchor="ma")
    text(draw, (525, 866), "预测时长 / 秒", 16, SECONDARY, anchor="ma")
    for j, key in enumerate(("ngff", "baseline")):
        pts = chart_xy(np.arange(80) * .02, errors[key])
        draw.line([tuple(p) for p in pts], fill=COLORS[j], width=3)
    text(draw, (1050, 673), "完整 2048 点 / 80 帧指标", 20, bold=True)
    text(draw, (1232, 710), "NGFF", 18, anchor="ma")
    text(draw, (1417, 710), "物理基线", 18, anchor="ma")
    for j, (label, key) in enumerate((("RMSE ↓", "rmse"), ("FPE ↓", "fpe"), ("R ↑", "r"))):
        y = 747 + j * 35
        text(draw, (1050, y), label, 18)
        for x, method in ((1232, "ngff"), (1417, "baseline")):
            text(draw, (x, y), f"{scene['metrics'][method][key]:.4f}", 19, anchor="ma")

    for frame in range(80):
        image = base.copy(); painter = ImageDraw.Draw(image)
        text(painter, (1508, 48), f"t = {frame*.02:4.2f} 秒", 30, bold=True, anchor="ra")
        for panel, key in enumerate(("truth", "ngff", "baseline")):
            if key != "truth":
                xy = screen(projected["truth"][frame].reshape(-1, 3), panel)
                for x, y in xy[::2]:
                    painter.ellipse((x-.8, y-.8, x+.8, y+.8), fill=(185, 194, 203))
            points = projected[key][frame]
            xy = screen(points.reshape(-1, 3), panel)
            color_idx = np.repeat(np.arange(3), 512)
            order = np.argsort(points.reshape(-1, 3)[:, 2])
            for idx in order:
                x, y = xy[idx]
                painter.ellipse((x-1.35, y-1.35, x+1.35, y+1.35), fill=COLORS[color_idx[idx]])
            if key != "truth":
                text(painter, (panel_x[panel]+240, 653), f"当前帧 RMSE {errors[key][frame]:.4f}  ·  灰点为真实位置", 16, SECONDARY, anchor="ma")
        px = chart_xy([frame * .02], [0])[0, 0]
        painter.line((px, chart[1], px, chart[3]), fill=SECONDARY, width=2)
        for j, key in enumerate(("ngff", "baseline")):
            x, y = chart_xy([frame * .02], [errors[key][frame]])[0]
            painter.ellipse((x-4, y-4, x+4, y+4), fill=COLORS[j])
        if frame == 55:
            image.save(directory / (scene["kind"] + "_poster.png"))
        yield image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()
    directory = args.input
    report = json.loads((directory / "report.json").read_text())
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    outputs = []
    for scene in report["scenes"]:
        path = directory / (scene["kind"] + ".mp4")
        writer = imageio_ffmpeg.write_frames(str(path), (WIDTH, HEIGHT), fps=25,
                    codec="libx264", pix_fmt_out="yuv420p", output_params=[
                        "-crf", "19", "-preset", "fast", "-threads", "4", "-movflags", "+faststart"])
        writer.send(None)
        for frame, image in enumerate(scene_frames(directory, scene, report["checkpoint_update"])):
            raw = image.tobytes()
            if frame == 0:
                for _ in range(25):writer.send(raw)
            for _ in range(2):writer.send(raw)
            if frame == 79:
                for _ in range(50):writer.send(raw)
            if frame % 20 == 0:print("FRAME", scene["kind"], frame, flush=True)
        writer.close()
        outputs.append(path)
        print("VIDEO", path, path.stat().st_size, flush=True)
    concat = directory / "video-inputs.txt"
    concat.write_text("\n".join(f"file '{p.name}'" for p in outputs) + "\n")
    subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i",
                    str(concat), "-c", "copy", "-movflags", "+faststart", str(directory / "demo.mp4")], check=True)
    print("DEMO", directory / "demo.mp4", flush=True)


if __name__ == "__main__":
    main()
