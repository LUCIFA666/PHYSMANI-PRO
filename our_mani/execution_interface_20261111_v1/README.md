# PhysMani 执行接口修复版

本模块在独立命名空间提供执行接口，已经连接当前 Native-contact Cube 网络与 MPC。旧实验源码、权重和成功判定保持原样。导入模块不会启动仿真。

## 两种命令语义

| 命令 | 输入的含义 | 实际执行 |
| --- | --- | --- |
| `JointGoal` | 绝对关节终点，单位 rad | 使用与旧 MPC 响应模型相同的参考生成函数，生成每个 50 ms 的位置参考 |
| `JointTrajectory` | 已采样的绝对关节位置参考 | 按给定顺序原样下发到原生位置伺服，不再运行终点参考生成器 |

每次执行一个或两个物理步。轨迹的 `start_time` 必须匹配当前仿真时钟，`times_from_start` 为 `[0.05]` 或 `[0.05, 0.10]`。时间戳不匹配会拒绝，不会自动平移时间。关节顺序使用构造时声明的 `joint_names`；命令提供的名称必须完全匹配。

完整轨迹前缀会在第一次下发前检查维度、有限值、关节边界，以及离散参考速度、加速度和 jerk。超限时拒绝整段，不裁剪轨迹。这里校验的是参考约束，不保证实际电机跟踪误差为零。场景碰撞资格仍由调用方负责；当前 Cube 接入保留了原来的碰撞检查。

## 调用

```python
from our_mani.execution_interface_20261111_v1 import (
    PhysManiExecutor, JointGoal, JointTrajectory,
)

# backend 为现有 Native 后端；limits 为原生 JointLimits。
executor = PhysManiExecutor(
    backend, limits,
    initial_gripper_opening=1.0,
    joint_names=joint_names,
)

state = executor.read_feedback()  # 不推进物理
receipt = executor.execute_command(JointGoal(
    positions=joint_destination,
    opening=0.0,                 # 0=CLOSE，1=OPEN，None=保留请求
    ticks=2,
    expected_time=state['timestamp'],
    joint_names=joint_names,
))

# 外部规划器提供轨迹时，每次取最前面的一个或两个 50 ms 参考点。
state = executor.read_feedback()
receipt = executor.execute_command(JointTrajectory(
    start_time=state['timestamp'],
    positions=[q_at_50ms, q_at_100ms],
    times_from_start=[0.05, 0.10],
    joint_names=joint_names,
))
```

底层仍使用 `set_joint_target_positions`，实际关节状态由原生动力学推进。轨迹参考的离散速度用于校验和日志，不代表向电机发送了速度控制命令。

旧调用 `execute_joint_target(q, opening=...)` 保留为 **JointGoal 终点语义**。当前 CEM 的适配调用为：

```python
receipt = executor.execute_planner_action(
    action,
    velocity_reference_role='objective_only',
    expected_time=robot.timestamp,
)
```

`tcp_velocity_reference` 和 `angular_velocity_reference` 在当前 CEM 中是跟踪目标。适配入口要求明确声明它们的角色，并记录在回执中。关节速度、力矩等未支持的电机命令会被拒绝。将规划器输出的轨迹点交给 `JointTrajectory`，即可避免再经过终点参考生成器。

## 时钟、反馈和故障

- 只有 `backend.tick()` 可以推进物理，每次必须恰好推进 50 ms。
- 每步回执分别记录下发参考和实际测得的 q/dq，不把参考当测量。
- 闭爪完成、实际 held 和原生任务成功为三个独立字段。
- 原生成功首次为真时立即结束当前命令，第二步不再执行。
- 已完成的夹爪请求会保持，不在每个 tick 重启同一请求。
- 下发后发生异常会记录实际已推进时间并锁住当前执行器，避免隐式重放半条命令。恢复后使用新的执行器实例。
- 没有具体 Observer 类限制，后端通过所需方法和数据协议接入。

当前网络与 MPC 的接入函数为 `producer.run()`。该入口复用原 coordinator、响应模型和碰撞资格检查，只明确执行动作合同。轨迹模式若用于新的 MPC，需要相应的轨迹响应模型；旧终点响应模型不能自动被视为轨迹模式的响应模型。

## 已完成的验证

2026-11-11：21 项新接口测试和 12 项旧执行器回归测试通过。真实 Native 验证包含：

| 检查 | 结果 |
| --- | --- |
| 接口专项，27 个物理步 | 通过；轨迹原样下发、越限前缀拒绝、过期轨迹拒绝、单步执行均实际运行 |
| 当前网络＋MPC，普通任务，seed 2611111000 | 4.95 s 首次原生成功；最后一次命令只执行一步 |
| 当前网络＋MPC，高速任务，seed 2611111100 | 运行到 8 s 预算，无接口异常；任务未成功，没有发出 CLOSE |

三项合计 286 个物理步，无额外物理步，实际原生伺服目标与请求的最大差约 `1.20e-7 rad`。这证明本轮接口检查和集成案例通过，不构成总体成功率或高速抓取改善结论。

结果与实际退出凭证：`artifacts/framework/execution_interface_20261111_v1/native_validation_v1/`。

验证入口：

```bash
.conda-env/bin/python -B scripts/validate_execution_interface_20261111_v1.py \
  --output artifacts/framework/execution_interface_20261111_v1/a_fresh_validation_directory
```

该入口仅运行计划中的有限 TRAIN 开发检查。旧全局 STOP 保留，旧训练与评估队列保持暂停；计划只豁免当时那个确切的历史暂停标记，新建或更新 STOP 仍会中止本轮验证。各 Native 串行运行，共享资源锁覆盖至关闭，主进程实际 wait 并回收后代进程。
