# 所有任务统一 60 秒

用户最新设置为 **60 秒仿真时间、1200 个 50 ms 物理步**。成功或原生终止条件触发后立即停止。控制频率、物理参数、抓持 API 和原生成功条件均未改变。

统一默认值在 `budget.py`，16 类任务与入口登记在 `config/framework/task_time_limits_20261111_v1/active_runtime.json`。这是仿真回合预算，进程保护用的墙钟超时独立保留。

## 当前入口

- 旧系统 16 类任务：`scripts/evaluate_multitask_case.py`，参数沿用 R10 单回合入口，可用 `--time-limit-seconds` 显式覆盖默认值。
- 普通／高速方块新执行器：`scripts/run_physmani_cube.py`。
- 16 任务通用反馈后端：`feedback_backend.ExtendedTimeFeedbackBackend`，默认 1200 步。

方块运行示例（输出目录必须不存在）：

```bash
.conda-env/bin/python -B scripts/run_physmani_cube.py \
  --task high_speed --seed 2611116200 \
  --output artifacts/framework/highspeed_control_20261111_v1/manual_6200
```

增加 `--prepare-only` 只生成可检查的请求，不启动 Native。普通速度使用 `--task ordinary`。旧系统 16 类任务用新入口替换旧命令中的 `scripts/evaluate_multitask_case_r10.py`，其余模型、来源清单、冻结输入和单任务参数沿用；输出放在 `artifacts/framework/multitask_60s_20261111_v1/` 下的新目录。

旧 16 任务不仅有 10 秒协议，还有 200 步外层上限。新入口在第一次控制之前为实际 ledger 安装新预算，外层循环、PhysMani 每次最多 10 步的分发检查、完成结果校验均读取相同预算。旧模型的 10 秒来源协议继续用于核对模型身份，实际运行覆盖值另有 SHA 身份和结果记录。60 秒结果与原 10 秒正式比较分开。

旧冻结入口、历史 8／16 秒实验和旧队列均保留。使用本页的新入口；本次没有恢复训练或原 VAL／TEST 队列。

## 验证范围

`tests/test_all16_time_limits_20261111_v1.py` 覆盖全部 16 个任务的统一预算、1200 步完整超时、超过旧 200 步后的成功、基线多步命令及禁止中途偷偷延长预算。

真实 Native 完整计时检查位于 `artifacts/framework/task_time_limits_20261111_v1/native60s_v1/`。它检查通用后端的实际物理时钟，不是 16 个任务的策略成功率。方块控制器的成功率和剩余问题见 `our_mani/highspeed_control_20261111_v1/README.md`。
