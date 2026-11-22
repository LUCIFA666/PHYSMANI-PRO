# 其余14任务的小规模执行器修复周期

用户授权修复其余任务执行器，并像方块一样做小型TRAIN／VAL／TEST。覆盖电线穿环、插杆、篮球、挂杯、按钮、垃圾桶、取杯七类操作及普通／高速，共14版本。原方块代码、权重、已完成结果和默认入口保留。

本轮复用已有的多任务公开RGB-D编码器和世界坐标指令Transformer。它与方块的目标GRU／相对接触候选网络是不同分支；本轮不宣称已经训练多物体GRU接触模型或语言任务网络。旧模型的初始资格只覆盖公开反馈上的离线forward，未证明任务闭环。

修复内容：

- 机械臂执行目标后，按实际公开TCP位置／旋转确认到达，再发出预测的夹爪开合意图；计步完成或RML完成不当作实际到达。
- 所有任务继承各自Native可抓物体API，夹爪调用完成后保持，不重复启动同一动作；开爪后的原生10步掉落等待被显式计入预算。
- 原生RML只有速度审计，不能证明离散参考加速度／jerk合法。本轮对采样关节折线显式重参数化时间，包含到原路径的连接和最终停止，再按不变的约束全路径预检。新执行器逐50ms原样发送准备好的参考，读回POSITION伺服目标确认。
- 每个实际50ms查询原任务`task.success()`，首次为真即锁存、停止。二值Native抓持反馈只在NN之外使用；无Native目标位姿、实例mask或未来信息进入NN。

RRT/RML保留原生机械参数和`EndEffectorPoseViaPlanning`的默认碰撞规划设置。`ignore_collisions=True`表示规划器不作额外碰撞否决；实际物理接触与任务失败条件仍原样保留。这条作者式世界指令执行路线不同于方块带碰撞停止资格的局部MPC，不将两者冒称同一控制器。

固定小规模计划位于`config/framework/multitask_executor_cycle_20261112_v1/plan.json`：每个任务版本选3条录制TRAIN来源拟合、1条TRAIN来源选择，共42／14父轨迹，保留每父16个阶段窗口。复用冻结的公开特征，只微调四个动作输出头，冻结视觉编码器和世界指令主干，150轮／1200步。标签为原录制专家的目标指令，不是实际接触位姿或时间。新选择集与拟合集按父分离，但初始预训练模型见过这些TRAIN来源；原基座曾由旧VAL选型，均明确披露。新训练不解码旧VAL特征或标签。

原生闭环另用固定新种子：42场TRAIN回归、14场TRAIN-selection回归、14场VAL、14场TEST，共84场，每场60秒／1200实际步。它们均为Native新生成开发场景，TRAIN相机校准可复用，不冒称官方归档VAL／TEST。微调权重在全部原生回归前冻结，TEST不用于继续训练或调参。第一轮按钮开发诊断及其失败源码快照单独保留，不混入84例分母。

主程序：

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
 /home/ubuntu/physmani-conda/envs/physmani5090/bin/python -B \
 -m our_mani.multitask_executor_cycle_20261112_v1.pipeline
```

新命名空间只执行一次，无隐式重试。Native最多一个；GPU拟合持有原两把Native锁和训练锁，不与仿真重叠。owner对实际子进程wait，清理并回收已知后代，保存逐例退出收据。状态为`artifacts/framework/multitask_executor_cycle_20261112_v1/status.json`，最终报告为同目录`report.json`。

停止本轮：创建`runtime/framework/multitask_executor_cycle_20261112_v1/STOP`或根输出目录`STOP`；修改／替换全局STOP也生效。历史STOP原文件继续保留，只豁免该原文件身份用于本次明确授权的新周期，不恢复旧队列。
