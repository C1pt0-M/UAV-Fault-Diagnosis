# transfer3 CNN-LSTM v2 + GRL 设计

## 1. 设计目标

在当前实验链中：

- `BiLSTM + FocalLoss + few-shot` 是当前最强稳定基线；
- `CNN-LSTM v2 + FocalLoss + few-shot` 已经优于旧版 `CNN-LSTM`，但仍未完全超越 `BiLSTM`，尤其在：
  - `normal recall`
  - `balanced_accuracy`

因此下一步不再继续更换 backbone，而是在 `CNN-LSTM v2` 上引入跨域对齐机制，验证：

> `CNN-LSTM v2 + domain adversarial alignment` 是否能够进一步提升 `Real` 域性能，并逼近或超过当前 `BiLSTM` 基线。

本轮设计只做一件事：

- 在 `CNN-LSTM v2` 上加入 `GRL + domain classifier`

不引入其它增强项。

## 2. 核心思想

当前任务的主要难点不是仿真域内分类，而是：

- `simulation-to-real domain gap`

在 few-shot 场景下，模型已经获得了少量 `Real` 支撑，但 backbone 学到的表示仍可能带有显著域特征，导致：

- `SIL/HIL` 上表现很好
- `Real` 上尤其 `normal` 类边界不稳

因此本轮方法核心是：

1. 保留 `CNN-LSTM v2` 的故障模式提取能力；
2. 通过对抗式域分类，让 backbone 学到更难区分域、更有泛化性的表示。

## 3. 总体结构

在 `CNN-LSTM v2` 的 attention pooling 之后，加入 domain 分支：

```text
Input
 -> CNN-LSTM v2 backbone
 -> pooled feature
   ├─ class head   -> 3-class logits
   └─ GRL
        -> domain head -> 3-domain logits (SIL / HIL / Real)
```

其中：

- `class head` 继续完成 `normal / motor / sensor` 分类；
- `domain head` 只用于训练时约束 backbone；
- 推理时仍只使用 `class head` 输出。

## 4. 模型设计

### 4.1 Backbone

继续复用：

- `CNN-LSTM v2`

要求：

- Backbone 必须支持输出 pooled feature；
- 分类头与特征提取层解耦；
- `forward()` 至少支持两种模式：
  - 仅输出分类 logits
  - 输出 `class logits + domain logits`

### 4.2 Domain Head

结构固定为轻量 MLP：

- `Linear(feature_dim, feature_dim // 2)`
- `ReLU`
- `Dropout`
- `Linear(feature_dim // 2, 3)`

domain 类别固定：

- `SIL`
- `HIL`
- `Real`

### 4.3 GRL

复用现有仓库中已验证过的实现思路：

- `ReverseLayerF`

即：

- 前向恒等
- 反向乘以 `-alpha`

不重新发明新的对抗层实现。

## 5. 训练设计

### 5.1 任务与协议

固定不变：

- 任务：`transfer3 = normal / motor / sensor`
- 协议：`few-shot`

### 5.2 cache

直接复用：

- `cache_transfer3_a_fewshot_v1`

原因：

- 该 cache 已包含 `domain_y`
- 已被 A 层验证可用
- 本轮不需要重建 cache

### 5.3 训练脚本

新增独立脚本：

- `src\models\train_transfer3_dg.py`

不将 `GRL / domain loss / alpha` 等逻辑继续堆入现有 `train_transfer3.py`。

新增脚本职责：

- 读取 few-shot cache
- 支持 `cnn_lstm_v2_dg` 模式
- 训练并输出 `metrics_val / test / external_test`

### 5.4 损失函数

总损失：

```text
L_total = L_cls + λ_d * L_domain
```

其中：

- `L_cls = FocalLoss`
- `L_domain = CrossEntropy(domain_logits, domain_y)`

### 5.5 alpha schedule

采用 DANN 风格 schedule：

```text
alpha = 2 / (1 + exp(-10 * p)) - 1
```

其中：

- `p = current_step / total_steps`

### 5.6 λ_d

第一轮固定：

- `λ_d = 0.2`

原因：

- 避免过强 domain loss 直接破坏分类能力；
- 先验证方向是否有效。

### 5.7 优化与训练超参

除非训练明显不稳定，否则沿用当前 `train_transfer3.py` 的默认超参：

- `epochs = 120`
- `batch_size = 32`
- `lr = 3e-4`
- `weight_decay = 1e-4`
- `gamma = 2.0`
- `patience = 20`
- `min_epochs = 30`
- `min_delta = 0.002`
- `warmup = 5`

## 6. 对照实验

本轮只看 3 个实验：

1. `B2_bilstm_focal_fewshot_v1`
2. `B5_cnnlstm_v2_focal_fewshot_v1`
3. `C1_cnnlstm_v2_grl_fewshot_v1`

其中：

- `B2` 是当前最强 few-shot 基线
- `B5` 是结构升级但未做域对齐
- `C1` 是结构升级 + 域对齐

本轮不再引入：

- `TextCNN`
- `Transformer`
- `baseline`

## 7. 评估重点

主指标优先级固定：

1. `Real holdout macro_f1`
2. `Real holdout balanced_accuracy`
3. `normal recall`
4. `SIL/HIL test macro_f1`

理由：

- 当前短板最集中在真实域分类边界稳定性；
- 其中 `normal` 类最具代表性。

## 8. 成功标准

`C1` 视为成功，必须满足：

1. `real_macro_f1 > B5`
2. `real_balanced_accuracy > B5`
3. `normal_recall >= B5`
4. `test_macro_f1` 不明显退化

若进一步满足：

5. `real_macro_f1 >= B2`
6. `real_balanced_accuracy >= B2`

则可以将：

- `CNN-LSTM v2 + GRL`

正式提升为主创新候选。

## 9. 失败判定

若出现以下情况之一，则视为 `GRL` 方向本轮失败：

- `real_macro_f1 <= B5`
- `real_balanced_accuracy <= B5`
- `normal_recall` 进一步下降
- `test_macro_f1` 明显退化

若失败，不继续堆 `normal calibration`，先回到：

- 调整 `λ_d`
- 调整 domain head 容量
- 检查 feature 接入位置

## 10. 论文与答辩表述

若本轮成功，建议表述为：

> 在结构升级后的多尺度残差 CNN-LSTM 基础上，进一步引入面向 `simulation-to-real` 的对抗式域对齐机制。通过在 pooled feature 层同时进行故障分类与域混淆学习，模型获得了更强的跨域泛化能力，从而提升了真实飞行域上的故障诊断性能。

## 11. 当前边界

本轮明确**不包含**：

- `normal prototype`
- `class-wise margin`
- `CORAL / MMD`
- attention 机制升级
- 自编码器预训练

这些属于下一阶段增强，只在 `C1` 成功后再考虑。
