# 其余14任务：小型TRAIN／VAL／TEST当前版本

服务：`our-mani-multitask-small-20261112-v5.service`。状态：`artifacts/framework/multitask_executor_cycle_20261112_v5/status.json`。固定计划仍为每个任务版本3场TRAIN回归、1场TRAIN-selection、1场新生成VAL、1场新生成TEST，共84例。TRAIN按重复编号轮转14任务以先覆盖每个适配器；顺序与结果无关。VAL／TEST尚未在此前修复版本中运行。最终84例统一使用v5，不混合旧版诊断。

复用v1已完成的8459参数输出头微调：42个录制TRAIN拟合父、14个TRAIN选择父，每父16窗口，150轮／1200步，epoch80，损失4.997612→3.983306。选中权重SHA`65a906fc96ffe8d7b453fa22b47199bf153d91ff712235f001c368682e1c36af`，骨干／编码器冻结，标签是专家未来目标指令，不是实际接触时间。旧基座见过这些TRAIN父且曾由旧VAL选型，新生成VAL／TEST不参与选型。当前路线为旧多任务公开RGB-D世界指令网络，尚未接入语言分支，也不是已扩展方块专用GRU。

执行链修复：

1. 实测TCP到达才执行开合意图，Native夹爪完成后保持同一命令；开爪原10步等待计入预算。
2. RML速度重试每次从路径起点重置预览状态，保留原机械参数；对采样折线显式时间重参数化，完整检查参考v/a/jerk。
3. Native get_path查询无论成功／异常均零时间回滚实际q／伺服／IK目标，核对dq和时钟。实际关节在上限外的微小动态超调先通过合法POSITION参考真实计步恢复，不能用setter截断实测值或放宽关节限制。
4. RML标量位置完成时可能略超终点，v5明确取原始折线终点，消除越界外推。真正越过原生关节限位的规划参考在执行前拒绝，记录为任务控制失败，不使整个评测进程崩溃；不裁剪关节值或模型目标。
5. 原Task.success每实际50ms一次，首次True锁存并停止。模型只接当前／过去公开RGB-D、标定、本体状态；Native抓持反馈独立于NN。RRT使用作者式允许接触规划设置；原物理、运动分布采样器和原任务判定不变。

每个案例在任何VAL／TEST前已固定5个冷启动seed；仅第一次公开输入／动作前的原生TaskEnvironmentError允许新进程重置，其余失败不换seed。无效reset单列，成功率分母不伪造NativeFalse；最终开发分布是原生采样器通过原生reset检查的场景，不冒称官方归档VAL／TEST。

此前v1～v4的TRAIN故障、回归、真实退出及退役收据保留。新版本未修改旧模型／旧源码／Native源码或默认入口。每例保存公开输入hash和稀疏RGB-D、逐步参考／反馈、原生终态、回滚／越限恢复记录、真实wait和后代回收凭据。Native最多一个，不与已完成的GPU拟合重叠。

停止：创建`runtime/framework/multitask_executor_cycle_20261112_v5/STOP`或当前根输出`STOP`。上游本轮v1／v3／v4的STOP以及被修改的全局STOP也会停止该周期。历史全局STOP原文件保持不变，不恢复旧队列。
