# transfer3 CNN-LSTM v2 + Normal Prototype Calibration 设计

## 1. 设计目标

在当前实验链中：

- `CNN-LSTM v2 + FocalLoss + few-shot` 已成为当前最强结果之一；
- 其主要剩余短板集中在：
  - `normal recall`
  - `balanced_accuracy`

`GRL` 实验表明：

- `normal recall` 可以被拉高；
- 但全局域对齐会破坏 `sensor` 类边界。

因此，本轮不再做全局 domain alignment，而是改为更定向的类边界校准：

> 只围绕 `normal` 类构建 prototype 约束，在不伤害 `motor / sensor` 表达的前提下，提高真实域 `normal` 类稳定性。

## 2. 核心思想

当前 `CNN-LSTM v2` 的 pooled feature 已具备较强故障表征能力，但 `normal` 类在真实域中仍容易被压向故障类边界。

`Normal Prototype Calibration` 的核心思想是：

1. 在特征空间中为 `normal` 类建立一个稳定中心；
2. 让正常样本靠近这个中心；
3. 让故障样本与这个中心保持安全间隔；
4. 只对 `normal` 边界做定向修正，而不是整体拉平所有域差异。

这类方法比纯调 class weight 更有结构性，也比全局 GRL 更可控。

## 3. 总体结构

本轮 backbone 继续固定为：

- `CNN-LSTM v2`

整体结构：

```text
Input
 -> CNN-LSTM v2 backbone
 -> pooled feature
   ├─ class head -> 3-class logits
   └─ normal prototype regularization
```

其中：

- 分类头继续执行 `normal / motor / sensor` 分类；
- prototype 正则只作用于 pooled feature 层；
- 不新增额外 domain 分支。

## 4. Prototype 设计

### 4.1 prototype 数量

本轮只使用一个 prototype：

- `normal prototype`

不为 `motor / sensor` 单独建 prototype。

原因：

- 当前任务短板明确集中在 `normal`
- 先用最小复杂度验证是否有效

### 4.2 prototype 位置

prototype 建立在：

- `CNN-LSTM v2` 的 pooled feature

即 attention pooling 后、分类头前的特征向量。

### 4.3 prototype 表示

使用可学习向量：

- `normal_center ∈ R^d`

其中：

- `d = pooled feature dim`

不使用 batch mean 直接代替中心，以避免训练噪声过大。

## 5. 损失设计

### 5.1 总损失

总损失固定为：

```text
L_total = L_cls + λ_n * L_proto
```

其中：

- `L_cls = FocalLoss`
- `L_proto = L_pull + L_push`

### 5.2 Pull loss

对 `normal` 样本：

- 最小化其 feature 到 `normal_center` 的距离

可采用：

- `L2 distance`

形式上：

```text
L_pull = mean(||f_i - c_normal||^2),  y_i = normal
```

### 5.3 Push loss

对故障样本（`motor / sensor`）：

- 要求其与 `normal_center` 至少保持 margin

形式建议为：

```text
L_push = mean(max(0, m - ||f_j - c_normal||)^2),  y_j != normal
```

其中：

- `m` 为 margin

### 5.4 λ_n

首轮固定：

- `λ_n = 0.1`

原因：

- 先做轻量正则
- 避免 prototype loss 盖过分类损失

### 5.5 margin

首轮固定：

- `m = 1.0`

若首轮无效，再做小范围调参，但第一版设计不引入多个 margin 候选。

## 6. 训练协议

### 6.1 任务与协议

固定：

- 任务：`transfer3 = normal / motor / sensor`
- 协议：`few-shot`

### 6.2 cache

继续复用：

- `cache_transfer3_a_fewshot_v1`

### 6.3 训练脚本

建议新增独立脚本：

- `src\models\train_transfer3_normalproto.py`

不把 prototype 逻辑继续堆到：

- `train_transfer3.py`
- `train_transfer3_dg.py`

### 6.4 backbone

固定：

- `CNN-LSTM v2`

即本轮实验名为：

- `C2_cnnlstm_v2_normalproto_fewshot_v1`

## 7. 对照实验

本轮只看 3 个实验：

1. `B2_bilstm_focal_fewshot_v1`
2. `B5_cnnlstm_v2_focal_fewshot_v1`
3. `C2_cnnlstm_v2_normalproto_fewshot_v1`

其中：

- `B2` 是当前基础最强基线
- `B5` 是当前不加校准的最优 CNN-LSTM 结构
- `C2` 是在 `B5` 上做 normal calibration

本轮不和 `C1` 混合比较主结论，`C1` 只作为失败路线记录。

## 8. 评估重点

主指标优先级固定为：

1. `normal recall`
2. `Real balanced_accuracy`
3. `Real macro_f1`
4. `test macro_f1`

原因：

- 本轮唯一目标就是修正 `normal` 边界；
- 其它指标作为约束，避免“用提升 normal 换整体崩盘”。

## 9. 成功标准

`C2` 视为成功，必须满足：

1. `normal_recall > B5`
2. `real_balanced_accuracy >= B5`
3. `real_macro_f1 >= B5`
4. `test_macro_f1` 无明显退化

若进一步满足：

5. `normal_recall` 接近或超过 `B2`
6. `real_macro_f1 >= B2` 或非常接近

则可以把：

- `CNN-LSTM v2 + Normal Prototype Calibration`

视为主创新候选。

## 10. 失败判定

若出现以下情况之一，则视为本轮失败：

- `normal_recall` 未提升
- `motor / sensor` 明显受损，导致 `macro_f1` 下跌
- `balanced_accuracy` 不升反降

若失败，再考虑：

- 调整 `λ_n`
- 调整 margin
- 改为 class-wise margin 而不是 prototype

## 11. 论文与答辩表述

若本轮成功，建议表述为：

> 针对无人机故障诊断中真实域正常样本易被误判的问题，在多尺度残差 CNN-LSTM 主干上进一步引入 Normal Prototype Calibration 机制。该机制通过约束正常样本聚集到稳定中心、并使故障样本与正常中心保持间隔，从而在不破坏故障判别能力的前提下，提高真实域的正常态识别稳定性。

## 12. 当前边界

本轮明确**不包含**：

- domain adversarial alignment
- class-wise prototype
- CORAL / MMD
- temperature scaling
- hierarchical normal/fault 二阶段分类

本轮只验证最小可行的 `normal prototype` 方案。
