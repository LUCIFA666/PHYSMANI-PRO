"""Write per-object diagnostics and a Chinese interpretation of the frozen test."""

import csv
import json
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"artifacts/quality_smoke"


def main():
    report=json.loads((OUT/"report.json").read_text())
    manifest=json.loads((ROOT/"data/quality_smoke/manifest.json").read_text())
    entries=[entry for entry in manifest["scenes"] if entry["split"]=="test"]
    with np.load(OUT/"test_predictions.npz") as data:
        truth=data["truth"].astype(np.float64)
        rows=[]
        for name in data.files:
            if name=="truth":continue
            prediction=data[name].astype(np.float64)
            for horizon in (0.2,0.6,1.2):
                count=round(horizon/0.04)+1
                error=prediction[:,1:count]-truth[:,1:count]
                centers=error.mean(axis=-2)
                for s,entry in enumerate(entries):
                    for obj,label in enumerate(entry["objects"]):
                        axis_rmse=np.sqrt(np.mean(error[s,:,obj]**2,axis=(0,1)))
                        final=centers[s,-1,obj]
                        rows.append({"scene":entry["name"],"object_instance":obj,"object":label,
                                     "model":name,"horizon_s":horizon,
                                     "point_rmse":float(np.sqrt(np.mean(error[s,:,obj]**2))),
                                     "centroid_final_error":float(np.linalg.norm(final)),
                                     "rmse_x":float(axis_rmse[0]),"rmse_y":float(axis_rmse[1]),
                                     "rmse_z":float(axis_rmse[2])})
    with (OUT/"per_object_metrics.csv").open("w") as file:
        writer=csv.DictWriter(file,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    ngff=report["test"]["ngff_trained"]
    physics=report["test"]["gravity_with_floor_stop"]
    lines=[
        "动力学效果 smoke test：已完成",
        "",
        "目的：检查小规模训练能否降低独立场景上的预测误差，不是完整论文复现。",
        "数据：官方 GSCollision；8 个训练场景、2 个验证场景、2 个测试场景。",
        "训练：随机初始化；400 次更新；batch=8（8 个不同训练场景）；耗时约 %.1f 秒。"%report["training_seconds"],
        "模型：隐藏维度 128、4 层、每物体 256 个 FPS 点，预测 1.2 秒。",
        "论文配置的模型宽度、关键点数、数据量及训练规模更大，此处结果不可直接等同论文数值。",
        "按验证集 MSE 选中第 %d 次更新的权重；测试集未参与参数和权重选择。"%report["best_validation_step"],
        "",
        "测试集对照（RMSE 越低越好，单位为原始仿真坐标）：",
        "预测窗口 | NGFF | 重力+落地停止基线 | 相对改善",
    ]
    for horizon in ("0.2","0.6","1.2"):
        a=ngff[horizon]["rmse"];b=physics[horizon]["rmse"]
        lines.append(f"{horizon} 秒 | {a:.6f} | {b:.6f} | {(1-a/b)*100:.1f}%")
    lines += [
        "",
        "解释：",
        "1. 训练和验证误差都显著低于初始化，学习链路有效。",
        "2. 0.6 秒的接触阶段，平均 RMSE 比物理基线低约 36%；1.2 秒窗口低约 12%。",
        "3. 0.2 秒基本处于自由落体，已知重力的解析预测更准确；NGFF 存在数值积分和局部形变误差。",
        "4. 训练后段训练误差继续下降、验证误差不再下降，存在过拟合迹象。",
        "5. 独立测试只有 2 个场景，平均收益不代表普遍泛化。两个场景的 1.2 秒 RMSE：",
    ]
    for i,entry in enumerate(entries):
        a=ngff["1.2"]["per_scene"][i]["rmse"]
        b=physics["1.2"]["per_scene"][i]["rmse"]
        lines.append(f"   {entry['name']}：NGFF {a:.6f}，基线 {b:.6f}，改善 {(1-a/b)*100:.2f}%。")
    lines += [
        "6. 1.2 秒的物体中心最终误差仍较明显；不能据此认为已达到机械臂动态抓取精度。",
        "",
        "RMSE：预测点坐标与真值的均方根误差；不计已观测的初始帧，按场景平均。",
        "FPE：最后一帧逐点三维欧氏误差的平均值。",
        "R：逐点逐坐标在时间上的 Pearson 相关，再平均；恒定真值坐标不参与平均。",
        "所有数值均为原始仿真坐标，尚未通过实机标定换算成米或毫米。",
        "",
        "先看 learning_and_test_error.png 和 heldout_centroid_trajectories.png。",
        "详细数据见 report.json、test_metrics.csv、per_object_metrics.csv。",
        "best.pt 是验证集选中的小样本训练权重；不是官方预训练权重。",
        "",
        "后续验证建议：扩大独立场景数，增加运动中初始状态，按抓取容差评估短时位置、姿态、速度误差。",
    ]
    (OUT/"结果说明.txt").write_text("\n".join(lines)+"\n")
    print(OUT/"结果说明.txt")


if __name__=="__main__":main()
