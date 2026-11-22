# 14任务小规模周期：原生路径查询原子回滚

服务：`our-mani-multitask-small-20261112-v4.service`。状态：`artifacts/framework/multitask_executor_cycle_20261112_v4/status.json`。计划：`config/framework/multitask_executor_cycle_20261112_v4/plan.json`。沿用v2在任何VAL／TEST之前固定的84个案例和420个冷启动seed；最终84例均用v4执行，不混合v1～v3的TRAIN诊断。

继承v3的RML预览修复、关节折线明确时间重参数化、v/a/jerk完整预检、实际到达后夹爪开合、完成保持、原生首次成功停止。v3在插杆TRAIN发现原生get_path异常退出可留下临时关节配置。v4只新增机器人查询回滚代理：对原生路径查询，无论成功或抛错，都恢复查询前实际q、伺服目标和IK目标，核对dq和物理／任务时钟；不重置动力学，不移动到预测目标，不增加物理步。逐次回滚凭据保存为`atomic_path_queries.jsonl`。不能恢复则报错，不能伪造速度继续执行。

模型仍是v1完成的8459参数世界指令输出头微调，150轮／1200步，epoch80，SHA`65a906fc96ffe8d7b453fa22b47199bf153d91ff712235f001c368682e1c36af`。TRAIN选择损失4.997612→3.983306；从旧公开RGB-D特征缓存选42个拟合／14个选择TRAIN父，每父16个窗口。旧基座已见过这些TRAIN父并曾使用旧VAL选型，均披露；新的Native VAL／TEST种子不参与训练或选择。此路线不是方块专用GRU／接触候选网络，语言分支尚未微调或参与执行。

原生闭环共42场TRAIN回归、14场TRAIN-selection、14场新生成VAL、14场新生成TEST。60秒／1200实际50ms步，成功仅原Task.success，第一次True立即停止。模型不读Native目标状态、未来、实例mask或抓持，二值抓持只在NN之外。RRT沿用作者式允许接触的设置，物理参数／任务分布采样器／原生判定未修改。这是开发场景小样本，不是官方归档VAL／TEST或论文同协议成绩。

仅动作前、首次NN输入前的TaskEnvironmentError可按预先固定seed序列启用新进程reset。动作失败、不可达目标、原生失败不重试。所有无效reset和前几版故障／真实退出都保留，不计作NativeFalse，不借此挑选成功场景。原v1模型与源码全部冻结，v4只增加适配器。

停止：创建`runtime/framework/multitask_executor_cycle_20261112_v4/STOP`或本轮根输出`STOP`，也可更新全局STOP。历史STOP原文件完整保留，旧队列不恢复，旧方块结果及默认入口不变。

插杆真实回归还确认：关节7物理实测可比原生上限多约13微弧度；直接回滚setter会被SDK截到上限。因此在查询前先检查实测关节，必要时通过原POSITION伺服发送向内2mrad的合法参考、真实计步恢复，再发起运动学查询；不裁剪实测值或放宽约束。850个实际步的同seed回归已完成，包含2步真实恢复、74次路径查询、73次实际TCP到达，原异常消失。该例42.5秒到达等待超时，任务仍为NativeFalse。
