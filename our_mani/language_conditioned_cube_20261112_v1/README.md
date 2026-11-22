# 把任务指令接入方块候选网络

现有候选 Transformer 已有独立的 `task_e[16]` 条件 token，但普通／高速方块都传入固定的 `1/8`。旧的 semantic task 编码器读取场景 RGB-D，不能区分同一场景里的不同语言意图。本目录新增真正读取文字的分支，保留旧实验和默认运行入口。

```text
任务指令 ── 冻结 CLIP RN50 文字塔 ── L2归一化[1024]
                                      │
                              可训练 MLP 1024→128→16
                                      │
                               SimNorm(2×8) = task_e
                                      │
RGB-D历史 → GRU物理预测 → future tokens ├→ 候选Transformer → K8 位姿／接触时间
                     └→ motion c ─────┤
当前机器人本体状态、公开几何 ──────────┘

task_e ── 辅助操作分类(8) + 辅助速度类别(2)
```

操作类别是方块抓取、按钮、插杆等八类语义；普通／高速作为另一属性。分类标签只进入训练损失，不进入网络 forward。指令中的 high speed 不设置物体真实速度或控制速度上限；实际运动仍来自公开视觉历史。物理积分、执行约束和 Native 原始成功判定继续使用原来的职责划分。

文字编码器复用本机 `/home/ubuntu/.cache/clip/RN50.pt`，校验官方 SHA256，只保留文字塔。冻结参数，以 FP32 运行；不下载权重。文字最长77个 token，超长明确报错。缓存冻结的文字特征，训练时重新计算可训练 `task_e`，避免缓存阻断梯度。新增适配层及辅助头共 **135,482** 参数；候选网络仍为 **3,190,538** 参数。

## 接口

```python
text_encoder = FrozenCLIPTextEncoder('/home/ubuntu/.cache/clip/RN50.pt')
candidate = InstructionConditionedCubeCandidate.from_matched_checkpoint(
    checkpoint, expected_sha256=candidate_sha, forecast_sha256=forecast_sha)
policy = LanguageConditionedCubePolicy(text_encoder, candidate)
output = policy(
    instructions=['pick up the red block and lift it up to the target'],
    future_tokens=future_tokens, robot=robot, geometry=geometry,
    context_c=context_c)
```

实际接环境时，取 `task.reset()` 返回的 descriptions 中的一条作为任务指令，在同一 episode 内保持不变。不要用任务目录名、episode ID、Native 目标状态替代文字。示例的三类需要从本目录的 `text` 和 `model` 模块导入。

语言分支末层以零初始化，初始 `task_e` 严格等于原来的常量。从已训练 matched checkpoint 载入候选权重后，输出与原模型完全一致。该初始化只提供可回归的起点；**未微调时语言不会改变动作，辅助分类头也未训练**。此模型是独立开发家族，旧执行 runtime 不接受它。

## 训练接法与验证范围

1. 在现有 TRAIN 数据索引旁增加真实 `instruction` 文本，保留父轨迹和来源哈希；同一轨迹的三种真实表述可随机选用。先按完整父轨迹划分 fit/selection，再做文本增强。VAL/TEST 不用于拟合或选择新指令。
2. 用新网络读取原来的四组物理输入，替换常量 task_e。保持原位姿匹配和实际接触时间 mask 损失，通过 `instruction_candidate_loss` 加入 `0.1*操作交叉熵 + 0.1*速度类别交叉熵`。仍需按父轨迹加权，超出预测窗口的时间标签不裁剪。
3. 第一阶段冻结 CLIP 和候选主体，训练语言适配层；第二阶段以更低学习率联合微调候选主体。CLIP 始终冻结。只用两个方块任务时，操作标签只有一个值，无法训练八类任务识别；八类识别需要对应任务的 TRAIN 指令。会识别八类文字也不等于已训练八类动作。
4. 检查真实同义指令的一致性、未见表述、指令互换及同场景不同目标，再按原60秒协议做方块闭环验证，并与常量 task_e 基线比较。普通／高速指令只是同一抓取操作的变体，单靠这两类数据不能证明模型理解不同任务意图。

当前方块感知仍只支持原来的红色方块／目标标记。语言指定其他颜色、其他物体或不同操作时，还需要语言目标与视觉对象的关联，以及相应动作数据和执行入口。本分支不自动按未训练分类头派发机器人动作。中文指令质量尚未验证。

## 已完成的接线验证

运行环境为 `/home/ubuntu/physmani-conda/envs/physmani5090/bin/python`，只使用 CPU，无 Native 仿真或优化器更新。

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
 /home/ubuntu/physmani-conda/envs/physmani5090/bin/python \
 -m unittest discover -s tests -p test_language_conditioned_cube_20261112_v1.py -v

OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
 /home/ubuntu/physmani-conda/envs/physmani5090/bin/python \
 scripts/verify_language_conditioned_cube_20261112_v1.py \
 --clip-checkpoint /home/ubuntu/.cache/clip/RN50.pt --output /tmp/new_language_wiring_report.json
```

9项新增检查验证语言梯度、旧输出一致性、物理 context 独立性、因果时钟和接触时间 mask。真实权重验证读取完成训练的方块 checkpoint 与一个 TRAIN batch：文字塔与原版 CLIP 输出逐值一致，动作损失对适配层梯度非零。随机修改适配层可改变动作，仅验证通路；它不是训练结果。语言微调与新的闭环成功率尚未运行，已有方块 VAL/TEST 结果仍属于无语言基线。
