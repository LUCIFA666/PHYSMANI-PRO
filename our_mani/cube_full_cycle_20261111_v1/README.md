# PhysMani 普通／高速方块：全量来源周期

用户明确选择“全部 TRAIN 重新训练，再跑完整 VAL／TEST”。固定来源：普通／高速各 100 TRAIN、20 VAL、40 TEST；每一类速度均覆盖 variation 0、1。TRAIN 按任务与 variation 分层，160 个父场景用于 fit，40 个父场景用于模型选择。官方 VAL40／TEST80 完全不参与拟合或选择。

真实含义：TRAIN／VAL 发布包是 HDF，包含公开相机、机器人初始状态及专家命令等信息，没有可完整恢复原生运动/场景的 reset 状态，也没有逐帧实际物理时钟与 held 标签。因此每个 HDF 原始来源各执行一次原生任务生成，采用该来源的标定、home 与 variation，种子由来源身份预先确定；不把 HDF 帧号、专家前瞻命令冒充真实运动/接触标签。TEST80 使用作者发布的低维 Demo、target_state、variation 及验证后的作者 alias，严格原生 reset 序列；不追加 home 重置或人工目标轨迹。这是全量来源覆盖和完整官方 TEST 存档评估，不能声称 VAL HDF 被完整场景重放，也不与原论文10秒协议混为一谈。

每个回合最多60秒／1200个真实50ms步。每步读取原始 Task.success，首次 true 即停止。Native 二值 membership 可用于执行阶段，Native 目标位姿、速度或未来轨迹不进入 NN。只有 TRAIN fit/selection 保存独立 private_TRAIN_labels.jsonl。失败、零窗口、无接触正例的父场景均保留在完整分母里。无静默重试或成功筛选。

顺序：全200来源采集及实际 wait/reap → 编译运动序列 → GPU GRU训练200epoch/1600updates → 新GRU公开重放且冻结全部特征 → 读取实际接触标签 → GPU候选Transformer训练150epoch/1200updates → 原有运行时守卫 → 冻结完整模型对 → 全VAL40 → 记录VAL release → 才打开/核验TEST存档全部19项输入 → 同一模型全TEST80 → 最终报告。

全程串行最多1个 Native，每来源独立 Python/Xvfb/仿真实例，实际 wait 后检查清理；GPU阶段独占两个 Native 锁和训练锁。训练不与 Native 重叠。捕获每个实际 RGBD 帧的内容哈希与标定／时钟证据，原始图像每20步保存一次，避免200训练及120评估回合撑满磁盘。3D公开测量和机器人输入逐50ms完整保留。

启动器：scripts/run_cube_full_cycle_20261111_v1.sh。此运行由单次 systemd 用户服务 our-mani-cube-full-20261111-v1.service 托管，无自动重启，不恢复任何历史队列。User linger 已启用；关闭当前对话不会主动停止此服务。

输出：artifacts/framework/cube_full_cycle_20261111_v1/。status.json 提供当前阶段及完成来源数，stage_exits 提供各子进程实际退出码；各 case 的 actual_exit.json 记录 Native 真实 wait/reap。最终 report.json 仅在全部320回合和训练阶段成功结束后产生。

停止：创建 runtime/framework/cube_full_cycle_20261111_v1/STOP 或本周期输出根 STOP；也响应替换/修改后的全局 STOP。历史全局 STOP 精确实例仅豁免当前明确授权的新周期，旧文件与旧暂停队列不动。协议/来源/时钟/资源/清理错误会留下失败回执并停止，未完成不得报告全量成功率。
