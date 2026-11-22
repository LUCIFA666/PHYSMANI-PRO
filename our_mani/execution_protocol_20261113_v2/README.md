执行协议 v2

原动作张量保持 8 维：世界坐标位置 3、单位 xyzw 四元数 4、已解码的二值开合 1。新增元数据独立封装，不改变旧权重形状。

兼容调用：

    bridge = ProtocolExecutor(timed_driver, legacy_driver=existing_destination_driver)
    receipt = bridge.execute(decoded_pose8, observation_time=actual_observation_time)

两个驱动必须共用 backend 和 PhysManiExecutor。旧动作交给已配置的原终点驱动；协议不猜预测时长。底层既有执行模式切换语义保留。

扩展调用：

    packet = ExecutionRequest(
        tuple(decoded_pose8), actual_observation_time,
        target_semantics='timed_sample',
        target_time_offset_seconds=declared_target_offset,
        stage_id=declared_stage,
        completion='continuous',
        metadata_origin='explicit_caller',
        release_policy='driver_default')
    receipt = bridge.execute(packet)

observation_time 是这次观测的实际时间；target_time_offset_seconds 是目标点对应的未来时间。K8 是姿态候选，不是时间步；不能把候选下标或推理耗时当预测时长。控制器的 control_lead_seconds 是机器人跟踪补偿，与目标点预测时长分开。

continuous 允许连续更新运动；static_pose 只接受零提前量静态路点，并等待位姿完成；gripper 等待物理开合完成，连续目标刷新不会清除尚未完成的事件。调用方可选择开夹期间继续退让；持续执行器仍保持已接受的开夹动作。开合完成、已抓取标志与原生任务成功是不同的回执。

release_policy 默认保留驱动配置。terminal_pose、position_window、caller_intent 显式区分不同释放合同。候选 aligned_drop 还要求驱动声明 supports_aligned_drop：横向 1.5 mm、旋转 0.02 rad、垂直 4 cm、有效当前目标。它尚未通过稳定性验证，不作为默认。

当前网络仍输出旧动作，没有训练完成的阶段或时间输出。新元数据在本轮原生诊断中来自明确标注的 TRAIN 专家目标调用方。56 个 TRAIN 来源的阶段标签已与 896 个缓存窗口对齐，详见 artifacts/framework/execution_protocol_20261113_v2/TRAIN_stage_label_audit.json。历史物理采样间隔尚未核验，没有编造 dt。

后续网络可保留旧 pose8 头，增加单独的阶段分类与目标时间输出；需要在 TRAIN 拟合、独立选择并记录置信度。字段缺失时禁用对应阶段特性，不能伪造阶段预测。当前尚未切换默认运行入口，也没有新增正式模型 VAL／TEST 成绩。
