# 复用已有全量 TRAIN，再完整 VAL／TEST

用户已授权持续完成普通／高速方块的全部训练、验证和测试，并提醒 TRAIN 已经采集。已核对旧 block_manifest.json：200 个 TRAIN 缓存与官方来源精确对应，文件齐全、SHA 全部一致。旧缓存是公开 RGB-D 轨迹的离线提取，times 由帧号乘0.05生成，并非直接记录的实际物理时钟，也没有 held/native-success 字段。因此不冒充真实50ms／实际接触监督。

本周期复用这些200来源（按任务及variation分层160 fit／40 TRAIN-selection）：

1. 200来源公开轨迹的记录帧域预训练，不读取 truth_metric_only 或专家标签作为输入。
2. 复用已完成的19个真实50ms Native TRAIN采集，校准运动物理时间。包括上一轮16个和本轮已完成3个；被取消的第4个留在原目录，不当作完整训练采集。
3. 先冻结全部200来源公开预测特征，再读取HDF专家期望抓取TCP相对于其专家目标参考点的位姿，仅作弱位姿先验。不是实际接触标签，不训练接触时刻。原HDF没有两指真实传感器，弱阶段机器人条件明确为缺失占位，冻结robot projection和time head，不声称测得这些通道。
4. 用19个既有Native采集中原生成功段的真实接触标签微调候选模型；在这一步恢复真实机器人特征和时间监督。弱预训练检查点使用独立schema，不能直接接入运行时。运行时原类型／源码／checkpoint守卫保持原样。
5. 冻结模型对，完整VAL40；完整release后再打开作者存档并完整TEST80。同一模型完成测试，不据TEST调参。

TRAIN仍覆盖全部200官方来源，Native时间／接触微调的有效父场景另报子集，绝不称200条全是实际接触正例。所有失败、零正例、已取消样本保留原始证据。官方VAL HDF缺少完整原生reset，采用每个来源的相机／home／variation生成一个原生任务；官方TEST80严格恢复作者存档，不追加home或人工运动。全部60秒上限，原生Task.success逐实际50ms首次成功停止；不与原论文10秒协议直接混同。

运行由systemd用户服务 our-mani-cube-full-reuse-20261112-v1.service 托管。状态在 artifacts/framework/cube_full_reuse_20261112_v1/status.json，各阶段真实退出记录在stage_exits，各Native case实际wait/reap在对应actual_exit.json。最终report.json仅在训练与全部120个评估来源完成后生成。

此前新采集周期 cube_full_cycle_20261111_v1 已被root按用户提醒主动退役，保留3个完整和1个取消回合；其服务失败状态是有记录的阶段转向，不是用户取消总体任务。不得重启它重复采200。旧全局STOP仍原样保留，仅为当前明确授权的新周期豁免精确历史实例；旧队列不恢复。

停止本周期：创建 runtime/framework/cube_full_reuse_20261112_v1/STOP 或本周期输出根STOP。替换／修改全局STOP也会停止。本周期无自动重试，无自动改默认部署权重。

## 最终结果

2026-11-12 02:43（北京时间）全部完成。复用200个TRAIN来源及19个真实时钟采集回合；两个网络的记录域预训练和物理／接触微调均完成。

| 阶段 | 普通 | 高速 |
|---|---:|---:|
| 全VAL来源40回合 | 20/20（100%） | 13/20（65%） |
| 官方存档TEST80回合 | 39/40（97.5%） | 27/40（67.5%） |

TEST合计66/80（82.5%）。VAL仍是全部HDF来源的原生生成场景，TEST为全部作者存档重置。60秒上限、原生首次成功判定。全部120个Native owner实际wait、worker关闭且无遗留子进程；模型对跨VAL／TEST不变。systemd主进程正常退出（ExecMainStatus0），资源锁释放；默认入口未替换。完整结果report.json，最终源／权重／退出核验final_audit.json。
