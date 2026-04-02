# transfer3 CNN-LSTM v2 结构设计

## 1. 设计目标

当前 `transfer3` 的旧版 `CNN-LSTM` 在 `few-shot Real` 协议下仍然具备竞争力，但在 B 层实验中表现出以下问题：

- `Real Macro F1` 低于 `BiLSTM`
- `Real Balanced Accuracy` 低于 `BiLSTM`
- `normal recall` 明显偏低

与此同时，它在 `motor / sensor` 类上仍然具有较强优势，说明问题不在于“CNN + LSTM 的混合思路错误”，而在于当前实现方式还不够适合真实域。

因此，本设计的目标不是替换 `CNN-LSTM`，而是升级为 `CNN-LSTM v2`，让它：

1. 保留局部故障模式提取能力；
2. 降低对 `normal` 类的过度压缩；
3. 在真实域上提升 `balanced_accuracy` 与 `normal recall`；
4. 形成一条可解释、可答辩的结构创新路线。

## 2. 对旧版问题的判断

当前旧版 `CNN-LSTM` 存在 3 个主要结构问题：

### 2.1 前端为单尺度卷积

旧版三层卷积统一使用：

- `kernel_size = 5`

这意味着模型只能在单一局部感受野上提取模式，不利于同时建模：

- 短时局部突变
- 中等时长振荡
- 较慢的趋势变化

### 2.2 时间压缩过强

旧版每层都带 `MaxPool1d(2)`，总长度变化为：

- `2048 -> 1024 -> 512 -> 256`

这会导致：

- 局部故障特征被增强；
- 但正常动态的细粒度时序结构被过早抹平；
- 对 `normal` 类尤其不利。

### 2.3 缺少原始序列保真路径

旧版 LSTM 后端只接收 CNN 提取后的高层特征，不再能访问原始时序的低级信息。若前端卷积在真实域上提取偏移，后端无法修正。

## 3. CNN-LSTM v2 总体结构

`CNN-LSTM v2` 采用双路径融合结构：

```text
Input (31, 2048)
  ├─ Raw Skip Path
  │    2048 -> 512
  │    Linear projection -> 32 dims
  ├─ Multi-scale CNN Path
  │    Conv branches: k=3 / 7 / 15
  │    concat -> residual conv block
  │    mild pooling only to 512 steps
  └─ Fusion
       concat(raw skip, cnn features)
       -> BiLSTM
       -> Attention Pooling
       -> Classifier
```

核心思想：

- 用多尺度卷积提取不同时间尺度的局部故障模式；
- 用 raw skip path 保留原始序列的基础动态信息；
- 用 BiLSTM 统一建模长程时序依赖；
- 用 attention pooling 聚焦最有诊断价值的时间区域。

## 4. 模块设计

### 4.1 Raw Skip Path

目的：

- 保留原始时序中的低级动态信息；
- 为 `normal` 类提供更稳定的全局动态参考。

固定设计：

- 对输入 `(N, 31, 2048)` 做时间下采样到 `512` 步；
- 方式可使用等间隔采样或轻量线性投影；
- 将每个时间步投影到 `32` 维特征；
- 输出形状为 `(N, 512, 32)`。

### 4.2 Multi-scale CNN Path

目的：

- 增强对不同宽度局部模式的表达能力。

固定设计：

- 并行 3 个 `Conv1d` 分支：
  - `k=3, out=32`
  - `k=7, out=32`
  - `k=15, out=32`
- 每个分支包含：
  - `Conv1d`
  - `Norm`
  - `ReLU`
- 拼接后得到 `96` 通道特征。

### 4.3 Residual Conv Block

目的：

- 在不过度加深网络的前提下，提高卷积特征表达稳定性；
- 让多尺度分支输出具备进一步融合能力。

固定设计：

- 轻量残差块 1 个；
- 主分支将通道从 `96` 提升到 `128`；
- shortcut 做必要的通道匹配；
- 保留非线性和归一化；
- 不做超过两次的总池化。

### 4.4 Mild Pooling Strategy

目的：

- 保留更多时序信息，避免旧版过度压缩。

固定约束：

- 最终长度只压到 `512`
- 不允许再压到 `256`

即：

- 旧版：`2048 -> 256`
- v2：`2048 -> 512`

### 4.5 Fusion

融合方式固定为：

- `concat(raw_skip, cnn_path)`，时间维对齐后沿特征维拼接

输出形状：

- `(N, 512, 160)`
  - `32` 来自 raw skip
  - `128` 来自 CNN path

### 4.6 BiLSTM Back-end

保持与旧版接近，避免把本轮变成大规模超参搜索。

固定参数：

- `hidden_size = 128`
- `num_layers = 2`
- `bidirectional = True`
- `dropout = 0.3`

### 4.7 Attention Pooling

继续保留当前 attention pooling，不在本轮改为更复杂机制。

原因：

- 本轮结构创新重点在前端表示与保真路径；
- 不把 attention 改动混入同一轮变量。

### 4.8 Classification Head

固定为：

- `LayerNorm`
- `Dropout`
- `Linear`

分类数仍为：

- `3`

## 5. 归一化与稳定性策略

旧版卷积前端使用 `BatchNorm1d`。由于本任务核心是 `simulation-to-real`，跨域统计偏移可能导致 BN 稳定性不足。

本轮建议：

- v2 优先尝试 `GroupNorm` 取代 CNN 前端 `BatchNorm1d`

原因：

- `GroupNorm` 不依赖 batch 统计；
- 对 batch size 和域偏移更稳定；
- 更符合真实域小样本适配场景。

本轮不再引入 `AdaBN` 到 v2 主实验中。

## 6. 训练协议

### 6.1 任务固定

- `transfer3 = normal / motor / sensor`

### 6.2 协议固定

- 统一使用 `few-shot`
- cache 固定复用 A 层确认后的 few-shot cache

即：

- `cache_transfer3_a_fewshot_v1`

### 6.3 损失函数固定

- `FocalLoss`

理由：

- 保持和旧版 `CNN-LSTM` 可比；
- 不把 loss 设计变成额外变量。

### 6.4 训练超参

除非验证明显不稳定，否则尽量沿用当前 `train_transfer3.py` 的默认超参：

- `epochs = 120`
- `batch_size = 32`
- `lr = 3e-4`
- `weight_decay = 1e-4`
- `gamma = 2.0`
- `patience = 20`
- `min_epochs = 30`
- `min_delta = 0.002`
- `warmup = 5`

## 7. 实验对照

本轮不直接拿 v2 去替代全体 backbone，而是只回答一个问题：

> `CNN-LSTM v2` 是否优于旧版 `CNN-LSTM`？

实验组固定为：

- `B4_old: cnnlstm_focal_fewshot`
- `B5_new: cnnlstm_v2_focal_fewshot`

若资源允许，可再补：

- `B2: bilstm_focal_fewshot`

作为基础模型参考，但不是本轮首要比较对象。

## 8. 判定标准

若 `CNN-LSTM v2` 同时满足以下条件，则视为结构升级成立：

1. `Real macro_f1` 高于旧版 `CNN-LSTM`
2. `Real balanced_accuracy` 高于旧版 `CNN-LSTM`
3. `normal recall` 明显高于旧版 `CNN-LSTM`
4. `test macro_f1` 不出现明显退化

其中第 3 条是本轮最关键指标。

若仅 `motor / sensor` 提升，但 `normal recall` 仍持续偏低，则不视为真正成功。

## 9. 论文与答辩表述

若本设计验证成功，建议将创新点表述为：

> 针对传统串行 CNN-LSTM 在真实域正常样本识别不足的问题，提出一种面向无人机故障诊断的多尺度残差 CNN-LSTM 结构。该结构通过多尺度卷积提取局部故障模式，通过原始序列跳连保留全局正常动态，再结合 BiLSTM 建模长程时序依赖，从而提升 simulation-to-real 场景下的分类鲁棒性。

## 10. 当前边界

本设计当前**不包含**以下内容：

- domain alignment
- normal calibration loss
- attention 机制升级
- 自编码器 / 预训练模块

这些内容若要加入，应放在 `CNN-LSTM v2` 结构验证成功之后，作为下一阶段增强。
