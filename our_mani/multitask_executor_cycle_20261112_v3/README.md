# 14任务执行器修复：固定小型TRAIN／VAL／TEST

当前服务为`our-mani-multitask-small-20261112-v3.service`，最终状态在`artifacts/framework/multitask_executor_cycle_20261112_v3/status.json`。使用v1完成的150轮／1200步输出头微调，epoch80 checkpoint SHA为`65a906fc96ffe8d7b453fa22b47199bf153d91ff712235f001c368682e1c36af`，TRAIN选择损失4.997612→3.983306，仅8459参数更新。v1训练源代码和原模型完全保留。

沿用v2在VAL／TEST之前固定的84个案例和420个潜在冷启动seed：每个任务版本3场TRAIN回归、1场TRAIN-selection、1场VAL、1场TEST。只有动作／首次公开模型输入之前的Native reset失败才尝试下一个预定seed；真实动作失败不重试。初始化分布为原生采样器通过原生reset检查的开发分布，不是官方归档VAL／TEST。无效reset尝试单列，若全部5次无效则明确停止，不冒充NativeFalse。

v2的第三个穿环TRAIN暴露原生RML离线速度检查中的状态错误：第二次预览沿用了前一次终点，产生虚假首步跳变，导致速度缩小直至10000步保护。v3仅在新适配器中修正，每次预览从原路径起点开始；保持原生RML原语、机械参数及关节限制，不修改PyRep／Native源码。真实同seed回归已确认原比值1.002881经一次校正变为1.000056，没有再次错误缩小至0.0031；该回归在3.25秒触发原生碰线失败，真实失败原样保留。

其余动作接法保持v1：公开RGB-D世界指令模型→原RRT路径→修正的RML离线采样→明确时间重参数化与离散v/a/jerk完整预检→逐50ms绝对关节POSITION参考。实际公开TCP到达后才执行预测开合，Native夹爪完成后保持；原开爪10步等待计入预算。默认作者式RRT规划允许接触，实际物理和原任务终止条件不变。这与方块的碰撞停止资格MPC属于不同执行路线。

每个实际50ms读取原Task.success，首次True即停止。公开网络不读取Native目标位姿、速度、实例mask或抓持；二值抓持反馈单列于NN之外。当前多任务模型仍是既有RGB-D世界指令分支，未使用尚未微调的语言分支，也不称为方块专用GRU已支持14任务。

每个NN输入帧保留数组hash，每10个查询保存一份真实公开RGB-D／标定／本体状态。TRAIN相机HDF14个文件独立固定SHA。源／模型、逐步关节参考与真实反馈、RML校正细节、原生终态、实际wait／后代回收凭据均保存。v1和v2失败、成功完成的开发诊断及退出记录保留，最终84例统一使用v3，不把旧控制版本混入最终分母。

停止：创建`runtime/framework/multitask_executor_cycle_20261112_v3/STOP`或本轮根输出`STOP`。v1同名STOP和被改动的全局STOP也生效。历史全局STOP原文件、旧队列、旧方块结果与默认入口保持不变。
