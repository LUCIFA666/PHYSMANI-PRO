# 普通／高速 Cube：新训练、验证、测试周期

用户授权：开始训练、测试和验证；本轮先跑普通速度、高速方块完整流程。

预先固定 `config/framework/cube_training_cycle_20261111_v1/plan.json`：12 个 fit、4 个 TRAIN-selection、4 个新生成 validation、4 个新生成 test，每类速度各占一半。种子、整场景身份、角色互不重叠。所有实际任务均最多 60 秒 / 1200 个实际 50 ms 步；第一次原生 `Task.success()` 为真立即停止。未修改原生物理、任务、运动分布、夹爪或成功条件。

本轮保持修复后的 RGB-D 测量器及 CNN 权重不变；从当前 `703bc...` GRU 与 `ea38...` matched Transformer 完整权重继续训练，不回退 legacy 初始化、不重置头。GRU 200 epoch / 1600 updates；Transformer 150 epoch / 1200 updates。每 5 epoch 用仅 TRAIN-selection 的整场景平衡损失选择已训练权重。两个速度至少各 2 个 fit 正例场景、各 1 个 selection 正例场景，才允许候选训练；不足时明确失败并保留全部分母，不伪造标签或静默补样本。

只有 fit/selection 生成独立的 `private_TRAIN_labels.jsonl`。公开 RGB-D、标定、当前/过去测量和机器人传感器用于模型输入；Native 目标位置仅作为离线监督。先用新 GRU 重放并冻结所有公开特征，之后才读取候选接触标签。候选标签只接受实际原生成功的抓持段；未执行候选不作负样本。超出 0.8 秒的真实接触时间保留，屏蔽时间损失，不裁剪。

运行顺序为采集结束并实际 wait/reap → 编译运动数据 → GPU 训练 GRU → 公开重放 → 编译接触标签 → GPU 训练 Transformer → 原有运行时守卫检查 → 冻结两模型 → Native validation → 同一冻结模型 Native test。GPU 训练独占两个 Native 锁和训练锁；Native 串行最多一个。各阶段保存实际退出码，未完成/失败保持原样，无自动重试。旧模型、旧队列和默认执行入口保留。

```bash
# 采集（本轮已启动；输出必须是新目录）
/home/ubuntu/physmani-conda/envs/physmani5090/bin/python -B scripts/cube_cycle_native_owner_20261111_v1.py \
  --plan config/framework/cube_training_cycle_20261111_v1/collection.json \
  --output artifacts/framework/cube_training_cycle_20261111_v1/collection_v1

# 仅在采集 owner 的 actual_exit.json 完整且真实退出后运行一次
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
/home/ubuntu/physmani-conda/envs/physmani5090/bin/python -B \
  -m our_mani.cube_training_cycle_20261111_v1.pipeline --execute
```

状态：`artifacts/framework/cube_training_cycle_20261111_v1/status.json`；阶段日志及实际退出码：`stage_exits/`；最终证据：`report.json`。本轮为小规模新生成场景，不能当作旧官方归档 VAL/TEST 结果。测试不参与调参或权重选择。

停止：创建 `runtime/framework/cube_training_cycle_20261111_v1/STOP`，或修改/替换全局 `runtime/framework/STOP`。历史全局 STOP 原文件完整保留，豁免仅限本轮明确授权的新周期；旧暂停队列不恢复。采集也可使用 `collection_v1/STOP`；管线运行时可使用本周期根输出 `STOP`。

## 本轮实际结果（已结束）

24 个 Native 场景全部实际关闭；8 个管线子进程实际 wait 完成；94 项检查通过。训练、采集、验证和测试均无未回收子进程，资源锁已释放，旧模型与历史 STOP 原文件不变。

| 场景 | 普通 | 高速 |
|---|---:|---:|
| TRAIN 采集（fit+selection） | 8/8 | 4/8 |
| 新生成 validation | 2/2 | 2/2 |
| 新生成 test | 2/2 | 0/2 |

GRU 完成 200 epoch / 1600 updates，选择第 10 轮；selection loss 0.136907 → 0.170466，未改善。候选 Transformer 完成 150 epoch / 1200 updates，选择第 85 轮；selection loss 0.118345 → 0.052638。

高速 test 的 seed 2611118700 在 13.45 秒因缺少满足原碰撞/运动约束的停止后续动作而退出；seed 2611118701 在 2.95 秒后持续缺失有效公开目标观测，跑满 60 秒失败。这些测试结果不用于本轮调参或重新选择权重。

本轮高速尚不稳定，未替换默认模型。完整统计见 `artifacts/framework/cube_training_cycle_20261111_v1/report.json`，失败诊断与最终回收检查见 `artifacts/framework/cube_training_cycle_20261111_v1/final_audit.json`。

- motion_training: `/home/ubuntu/桌面/our-mani/artifacts/framework/cube_training_cycle_20261111_v1/motion_fit/selected.pt` SHA256 `7419c24acb6bd83d67965db8ab09edba2138a7904e9bc03a8930721eda9a5851`
- candidate_training: `/home/ubuntu/桌面/our-mani/artifacts/framework/cube_training_cycle_20261111_v1/candidate_fit/selected.pt` SHA256 `7ad59c481ec7ce896a07d050392fab97fc08ce757df1dccd8e4f69ef29aa2fb2`
