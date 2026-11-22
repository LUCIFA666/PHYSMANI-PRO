# 原生初始化重试包装：84例小规模周期

沿用v1完成的150轮／1200步TRAIN输出头微调，选中epoch80，TRAIN-selection loss从4.997612降至3.983306。视觉编码器、世界指令主干和执行器源码／权重均保持冻结，模型SHA为`65a906fc96ffe8d7b453fa22b47199bf153d91ff712235f001c368682e1c36af`。v1首个穿环TRAIN在任何策略输入或动作前因Native waypoint0不可达而reset失败，已真实wait／清理，不能计为NativeFalse。该尝试及v1退役收据保留。

v2仅补充基础设施层：在任何VAL／TEST结果之前，为原84个场景各固定5个唯一冷启动seed。只有`TaskEnvironmentError`发生于首次公开观测、模型决策和动作之前，且旧进程已wait／回收，才允许下一个完整新进程reset。复用v1首个失败reset作为该例attempt0，不重复运行；其他场景按已固定顺序进行。策略不可达、未抓住、原生失败、运行时错误或超时均不换seed。所有无效reset单列保存，成功率分母仅包括完成原生reset的固定84例；其分布为原生采样器通过原生reset检查的开发分布，不冒称官方归档测试。

每个任务版本：3场TRAIN回归、1场TRAIN-selection回归、1场新生成VAL、1场新生成TEST，共14×6=84场。物理预算60秒，原始成功首次True即结束。原录制TRAIN拟合／选择来源为42／14个父轨迹、每父16窗口。当前policy使用原多任务RGB-D世界指令分支，不使用尚未微调的语言分支，也不称为已扩展方块专用GRU。

运行服务：`our-mani-multitask-small-20261112-v2.service`。状态：`artifacts/framework/multitask_executor_cycle_20261112_v2/status.json`。逐例实际退出凭据、原生结果、公开RGB-D稀疏帧及全部输入帧hash、关节参考／反馈均保留。最终报告为同目录`report.json`。

停止：创建本轮输出根`STOP`，或`runtime/framework/multitask_executor_cycle_20261112_v2/STOP`。v1同名STOP和新改动的全局STOP也会停止本轮；历史全局STOP原文件完整保留。不会恢复旧队列。
