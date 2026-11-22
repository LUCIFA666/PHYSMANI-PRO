释放顺序控制策略对照（用户已授权运行，统一报告）

用户已确认测试新顺序，并要求作为控制策略选择统一报告。既有 4 个 TRAIN 种子、两种顺序共 8 个成对案例保持；报告逐行注明 gripper_release_order，以相同任务、种子、控制参数和原生 Task.success() 比较。

native_detach_before_open：调用原始 backend，先解除挂接、再张开。
open_fingers_then_native_release：逐个真实 50 ms 步张开，Native actuate 返回完成且两根实际手指开度都达到 95% 后，调用原始 release()。Native 的完成布尔值可能表示停滞或反向，不能单独当成张开完成；第一次反向完成可零步重启一次，仍保持所有物理步由持续执行器计数。

两种策略均使用原始 grasp、actuate、release API，没有修改任务场景、物体动力学、速度、关节限位、时间步或 Task.success()。开夹过程的挂接存续时机由控制策略决定；报告保留该字段，统一汇总，不另设排除类别。

本轮仍是 TRAIN 专家目标诊断，NN=0，不是模型成功率；旧模型权重与方块完整 TEST 保留。每个案例最多 60 秒仿真时间，只按原生判定记成功，并保留失败与实际 wait／回收证据。

计划：artifacts/framework/executor_release_order_proposal_20261113_v19/predeclared_plan.json
用户决定：同目录 user_control_strategy_decision.json
