# 在线模型重训与 KG-MSTCA 创新设计

## 1. 设计目标

本轮工作只聚焦在线实时检测模型，不再调整离线三分类主模型。

当前项目已经具备可运行的在线检测链路：

- 数据缓存构建：[src/data/build_online_cache.py](src/data/build_online_cache.py)
- 训练入口：[src/models/train_online_detector.py](src/models/train_online_detector.py)
- 在线推理：[src/online/fault_detector.py](src/online/fault_detector.py)
- 事件聚合与实时服务：
  - [src/online/event_aggregator.py](src/online/event_aggregator.py)
  - [src/online/risk_evaluator.py](src/online/risk_evaluator.py)
  - [scripts/eval_telemetry_live.py](scripts/eval_telemetry_live.py)

在时间有限的前提下，本轮目标不是直接把任务升级成真正的 proactive prediction，而是采用“先稳后新”的路线：

1. 先系统重训在线检测模型，选出当前任务定义下的最强 baseline。
2. 再在最强 baseline 的基础上做一个具有论文叙事价值、实现复杂度可控的创新模型。
3. 最终输出一个可以直接用于比赛展示、答辩和系统部署的在线主模型。

## 2. 已验证的当前基线

### 2.1 当前任务定义

当前在线任务是窗口级二分类：

- 标签：`normal / fault`
- 输入频率：`50 Hz`
- 窗口长度：`150`
- 当前保留模型合同中的步长：`stride = 10`
- 特征维度：`31`

这一合同由 [models/online_detector_cnn1d_rfly/norm_stats.json](models/online_detector_cnn1d_rfly/norm_stats.json) 和 [src/models/train_online_detector.py](src/models/train_online_detector.py) 共同约束。

### 2.2 当前最好在线模型

当前保留主模型为：

- [models/online_detector_cnn1d_rfly/config.json](models/online_detector_cnn1d_rfly/config.json)

其窗口级测试结果为：

- `accuracy = 0.9729`
- `macro_f1 = 0.9729`
- `f1_fault = 0.9727`

见：

- [models/online_detector_cnn1d_rfly/metrics_test.json](models/online_detector_cnn1d_rfly/metrics_test.json)

这说明现有 `cnn1d` 已经是一个强 baseline，但当前仍有三个问题：

1. 缺少统一重训后的 backbone 对比，难以证明 `cnn1d` 真的是当前最优选择。
2. 当前在线主模型的创新点偏弱，比赛答辩时更像“工程实现”而不是“算法创新”。
3. 当前“预警”更多依赖后处理规则层，不适合作为本轮主创新目标。

## 3. 设计决策

本轮采用以下总路线：

> 不改变在线任务定义，不主攻真正的“提前预警标签学习”，而是在当前在线检测任务上做一次完整重训，并在最优 baseline 上设计一个知识引导的创新模型 `KG-MSTCA`。

这样做的原因是：

- 当前缓存、训练、推理、评测链路都已围绕在线二分类建立，复用成本最低。
- “提前预警”需要重新定义标签与评价指标，风险过高，不适合作为时间紧张情况下的主线。
- `MSTCA + knowledge-guided relation branch` 可以兼顾可实现性、可解释性和答辩叙事。

## 4. 范围与非目标

### 4.1 本轮范围

- 重训在线检测模型。
- 做 backbone 对比实验。
- 做少量关键超参实验。
- 设计并训练 `KG-MSTCA` 创新模型。
- 用窗口级和事件级评测共同选择最终模型。

### 4.2 明确不做

- 不改离线三分类主线。
- 不把主任务改成多分类在线定位。
- 不把主任务改成真正的 time-to-fault 预测。
- 不开启大规模超参搜索。
- 不引入复杂的 domain adaptation 主线。

## 5. 数据与缓存合同

### 5.1 固定输入合同

本轮创新模型必须兼容现有在线推理链路，因此固定以下输入合同：

- `feature_schema = rfly_seq_columns_v1`
- `feature_count = 31`
- `target_hz = 50`
- `window_size = 150`
- `stride = 10`

原因：

1. 现有训练脚本对缓存合同有硬校验。
2. 现有推理服务和 `FaultDetector` 已能直接消费该合同。
3. 比赛展示时需要强调“模型升级，不改部署接口”。

### 5.2 标签定义

继续使用当前窗口标签定义：

- 当窗口内 `fault-active` 占比大于 `fault_threshold` 时标记为 `fault`
- 否则标记为 `normal`

其中 `fault-active` 的来源继续沿用 [src/data/build_online_cache.py](src/data/build_online_cache.py) 中的优先级：

1. `flight_phase == fault`
2. `UAVState_data_fault_state / TrueState_data_fault_state`
3. `rfly fault id`

### 5.3 本轮建议的数据产物

为了避免和历史缓存混淆，本轮建议新建一版在线缓存与结果目录：

- 缓存目录：`D:\UAV-Data\cache_online_rfly_v2`
- 结果目录：`D:\UAV-Data\results_online_v2`

要求：

- 缓存一旦生成，本轮所有 backbone 和创新模型都使用同一份缓存。
- 不允许每个模型使用不同缓存，以免对比失真。

## 6. 实验矩阵

## 6.1 Phase A：复现基线

目标：确认现有 `cnn1d_rfly` 在新缓存或重训环境下可重复。

实验：

1. `A0_cnn1d_reproduce`

固定设置：

- 模型：`cnn1d`
- 训练脚本：[src/models/train_online_detector.py](src/models/train_online_detector.py)
- 训练参数尽量保持当前主模型默认值：
  - `epochs = 100`
  - `batch_size = 256`
  - `lr = 3e-4`
  - `weight_decay = 1e-4`
  - `gamma = 2.0`
  - `hidden_size = 64`
  - `dropout = 0.2`
  - `warmup = 5`
  - `patience = 20`
  - `min_epochs = 30`
  - `seed = 42`

复现通过标准：

- `f1_fault` 与当前保留模型相差不超过 `0.005`
- `accuracy` 与当前保留模型相差不超过 `0.005`

若复现失败，则优先排查缓存、步长、训练样本数和随机种子，而不是立刻修改模型结构。

## 6.2 Phase B：backbone 对比

目标：在同一缓存和尽量统一的训练预算下找出最优 backbone。

实验列表：

1. `B1_cnn1d`
2. `B2_bilstm`
3. `B3_transformer`
4. `B4_mstca`

统一原则：

- 使用同一缓存目录
- 使用同一随机种子
- 使用同一训练轮数上限
- 使用同一主评估指标

轻微的结构差异允许保留：

- `transformer` 可使用 `num_layers = 2`
- 其余模型维持默认 `num_layers = 1`

主排序指标：

1. `f1_fault`
2. `macro_f1`
3. 事件级 `file_recall / hit_rate`
4. 正常集 `false_alarm_rate_normal`

### 6.3 Phase C：窗口长度小范围实验

目标：只对前两名 backbone 验证窗口长度是否存在明显收益。

窗口候选：

1. `100`
2. `150`
3. `200`

约束：

- 只对 Phase B 的前两名执行
- 不再同时扫更多超参

如果 `150` 仍然最优，则维持现有窗口长度，减少额外改动。

## 6.4 Phase D：创新模型

目标：在最强 baseline 上做一版比赛可讲、代码可落地、部署可兼容的创新模型。

实验列表：

1. `D1_mstca_strong_baseline`
2. `D2_kg_mstca`

其中：

- `D1` 用于确认创新提升是否真的来自知识引导分支，而不是单纯来自 `MSTCA`
- `D2` 为最终创新模型候选

## 7. KG-MSTCA 模型设计

## 7.1 设计动机

现有在线模型的主要问题不是“不能做二分类”，而是缺少一层面向无人机飞行机理的关系建模。

从论文和现有数据合同出发，故障往往不仅表现为单通道异常，还表现为多个通道之间的协同失配，例如：

- 电调输出之间的不平衡
- 控制输入与执行输出之间的偏差
- 执行器努力程度与机体加速度响应之间的不一致
- 姿态、角速度、速度和气压高度之间的耦合异常

因此本轮创新不从“更深的 backbone”切入，而从“显式关系建模”切入。

## 7.2 总体结构

`KG-MSTCA` 采用双分支结构：

```text
Input (31 x T)
 -> Branch A: MSTCA raw temporal encoder
 -> Branch B: Knowledge-guided relation encoder
 -> Gated fusion
 -> Classification head
```

其中：

- `Branch A` 复用当前 `OnlineMSTCA` 的多尺度时序通道建模能力
- `Branch B` 不增加原始输入维度，而是在模型内部基于已有 31 维输入构造关系序列
- `Gated fusion` 用于自适应决定当前窗口更依赖原始时序模式还是关系异常模式

## 7.3 Knowledge-guided relation branch

本分支内部从现有 31 维输入中构造一组低维关系序列，不改缓存合同。

建议首版关系信号包括：

1. `actuator_out_cross_std_t`
   - 四路 `actuator_out` 的时刻标准差
   - 表示执行输出是否失衡

2. `actuator_out_cross_range_t`
   - 四路 `actuator_out` 的 `max - min`
   - 对单路异常更敏感

3. `ctrl_out_gap_mean_t`
   - 四路 `actuator_ctrl` 与归一化 `actuator_out` 的平均绝对差
   - 表示控制命令与实际执行的偏差

4. `ctrl_out_gap_std_t`
   - 四路命令-执行差的标准差
   - 表示偏差是否只集中在局部通道

5. `actuator_effort_t`
   - 继续使用输入中的已有派生通道
   - 表示执行输出与加速度响应的关系

6. `acc_norm_deviation_t`
   - `|acc_norm - g|`
   - 表示整体动力学响应偏离

7. `gyro_norm_t`
   - 三轴 `gyro` 的范数
   - 表示姿态变化剧烈程度

8. `vertical_consistency_t`
   - 由 `baro_alt` 的时间差分与 `vel_z` 构造的垂向一致性残差
   - 表示高度变化与速度估计是否匹配

9. `attitude_energy_t`
   - 由四元数构造的简化姿态能量或倾斜幅度
   - 表示机体姿态整体偏离

以上关系序列再经过一个轻量级多尺度卷积编码器：

- `Conv1d(k=3/7/15)` 并联
- `SE` 或简化通道注意力
- `AdaptiveAvgPool`

这样设计的原因是：

- 可解释，答辩时容易讲清楚
- 不需要修改缓存文件格式
- 复杂度明显低于重做一整套 GNN 或 Transformer relation model

## 7.4 融合方式

融合层使用门控融合而非简单拼接：

```text
z = alpha * h_raw + (1 - alpha) * h_rel
alpha = sigmoid(MLP([h_raw, h_rel]))
```

原因：

- 正常窗口更可能依赖原始稳定时序模式
- 异常窗口更可能依赖关系失配模式
- 门控比直接拼接更适合做“何时相信关系分支”的自适应选择

## 7.5 分类头与损失

首版保持简单：

- 分类头：`Dropout + Linear`
- 损失：继续使用 `FocalLoss`

本轮不再额外引入复杂 auxiliary loss，避免时间失控。

## 8. 评测设计

## 8.1 窗口级指标

统一记录：

- `accuracy`
- `macro_f1`
- `weighted_f1`
- `precision`
- `recall`
- `f1_fault`
- `confusion_matrix`

对应现有训练输出：

- `metrics_val.json`
- `metrics_test.json`

## 8.2 事件级指标

必须使用 [scripts/eval_telemetry_live.py](scripts/eval_telemetry_live.py) 对候选模型做 replay 评测。

重点关注：

- `file_recall`
- `hit_rate`
- `false_alarm_rate_all`
- `false_alarm_rate_normal`
- `false_alarm_count`
- `detection_latency_s`

原因：

- 比赛展示中，“窗口分类精度”不如“是否少误报、是否能稳定报出故障事件”更容易让评委理解。
- 当前系统本身就有事件聚合逻辑，因此模型选择不能只看窗口级指标。

## 8.3 效率指标

至少记录：

- 参数量
- 单窗口推理延迟
- replay 总耗时

若 `KG-MSTCA` 性能略优但推理明显变慢，需要慎重作为最终主模型。

## 9. 成功标准

### 9.1 强 baseline 成功标准

在本轮重训后，若某个 baseline 同时满足：

1. `f1_fault >= 当前 cnn1d_rfly`
2. 事件级 `false_alarm_rate_normal` 不劣于当前模型
3. 事件级 `file_recall` 不低于当前模型

则可视为新的强 baseline。

### 9.2 KG-MSTCA 成功标准

`KG-MSTCA` 视为成功，至少满足以下条件中的两条，且第三条不明显退化：

1. `f1_fault` 超过最强 baseline
2. `file_recall / hit_rate` 超过最强 baseline
3. `false_alarm_rate_normal` 低于最强 baseline

额外约束：

- 参数量不超过最强 baseline 的 `1.5x`
- 推理延迟不超过最强 baseline 的 `1.5x`

## 10. 风险与回退策略

### 10.1 风险 1：创新模型不稳定

表现：

- `KG-MSTCA` 验证波动大
- 训练指标不稳定
- 事件级误报反而变多

回退：

- 直接保留最强 baseline 作为比赛主模型
- `KG-MSTCA` 作为创新尝试写入论文或答辩补充，不强行上线

### 10.2 风险 2：窗口长度扫描耗时过高

回退：

- 只保留 `150`
- 不再继续扫 `100/200`

### 10.3 风险 3：Transformer 明显弱或慢

回退：

- 终止 Transformer 后续扩展
- 资源集中到 `cnn1d / mstca / kg-mstca`

## 11. 最终交付物

本轮最终应产出：

1. 在线缓存一版固定实验合同
2. backbone 对比结果表
3. `KG-MSTCA` 模型代码与模型目录
4. 窗口级与事件级评测报告
5. 用于比赛材料的结果摘要

推荐最终模型目录命名：

- `models/online_detector_kg_mstca_rfly`

推荐实验结果目录命名：

- `D:\UAV-Data\results_online_v2`

## 12. 答辩表述建议

本轮创新建议统一表述为：

> 在保持现有在线实时检测输入合同与部署接口不变的前提下，提出一种知识引导的多尺度时序通道注意力模型 `KG-MSTCA`。该模型在原始时序建模分支之外，显式构造执行器一致性、控制-执行偏差、动力学响应和姿态垂向耦合等关系特征，并通过门控融合机制联合判别故障模式，从而提升在线故障检测的稳定性、可解释性与比赛展示价值。
