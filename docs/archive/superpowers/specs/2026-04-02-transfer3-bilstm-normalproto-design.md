# transfer3 BiLSTM + Normal Prototype Calibration 设计

## 1. 设计目标

当前实验结果表明：

- `BiLSTM + FocalLoss + few-shot` 在 `normal recall` 上表现较强；
- `CNN-LSTM v2 + Normal Prototype Calibration` 在整体主指标上已取得当前最好结果；
- 但 `C2` 的 `normal recall` 仍未超过 `BiLSTM` 基线。

因此，需要增加一个补充层实验，回答：

> `Normal Prototype Calibration` 这一创新点，是更适合 `BiLSTM`，还是更适合 `CNN-LSTM v2`？

本轮不再讨论新的 backbone，也不引入新的跨域模块，只做：

- `BiLSTM + Normal Prototype Calibration`

作为与 `C2` 的平行对照。

## 2. 角色定位

本实验建议命名为：

- `C3_bilstm_normalproto_fewshot_v1`

它的作用不是替代 A 层，而是补充回答：

1. `normal prototype` 是否具备 backbone 可迁移性；
2. 若该创新点放到 `BiLSTM` 上，是否会优于 `CNN-LSTM v2 + normalproto`；
3. 最终主模型应选：
   - `BiLSTM + NormalProto`
   - 还是 `CNN-LSTM v2 + NormalProto`

## 3. 核心思想

`BiLSTM` 当前最大的优势是：

- 对 `normal` 类更稳

因此一个自然假设是：

- 若将 `normal prototype` 这一类边界校准机制直接作用在 `BiLSTM` 的 pooled feature 上，
- 可能能够在保持其正常态优势的同时，进一步提升整体 `balanced_accuracy / macro_f1`。

本轮的意义就是验证这个假设。

## 4. 总体结构

整体结构非常简单：

```text
Input
 -> BiLSTM backbone
 -> pooled feature
   ├─ class head -> 3-class logits
   └─ normal prototype regularization
```

其中：

- backbone 为 `BiLSTM`
- 正则逻辑与 `C2` 保持同构
- 不引入 domain 分支

## 5. Prototype 设计

### 5.1 prototype 数量

固定为：

- 单一 `normal prototype`

### 5.2 prototype 位置

建立在：

- `BiLSTM` 的 attention pooling 之后、分类头之前的特征向量

### 5.3 prototype 表示

固定为：

- 可学习参数 `normal_center`

维度：

- 与 `BiLSTM` pooled feature 一致

## 6. 损失设计

为保证与 `C2` 可比，本轮损失形式保持一致。

### 6.1 总损失

```text
L_total = L_cls + λ_n * (L_pull + L_push)
```

### 6.2 分类损失

- `L_cls = FocalLoss`

### 6.3 Pull loss

对 `normal` 样本：

```text
L_pull = mean(||f_i - c_normal||^2), y_i = normal
```

### 6.4 Push loss

对 `motor / sensor` 样本：

```text
L_push = mean(max(0, m - ||f_j - c_normal||)^2), y_j != normal
```

### 6.5 超参

与 `C2` 保持一致，首轮固定：

- `λ_n = 0.1`
- `m = 1.0`

这样才能真正比较：

- 创新点
- backbone

谁才是结果变化主因。

## 7. 训练协议

### 7.1 任务与协议

固定：

- `transfer3 = normal / motor / sensor`
- `few-shot`

### 7.2 cache

继续复用：

- `cache_transfer3_a_fewshot_v1`

### 7.3 backbone

固定：

- `BiLSTM`

### 7.4 训练脚本

建议新增独立脚本：

- `src\models\train_transfer3_bilstm_normalproto.py`

不把这一逻辑堆入 `C2` 的脚本。

## 8. 对照实验

本轮比较 3 个实验：

1. `B2_bilstm_focal_fewshot_v1`
2. `C2_cnnlstm_v2_normalproto_fewshot_v1`
3. `C3_bilstm_normalproto_fewshot_v1`

目的：

- `B2 -> C3`：看 `BiLSTM` 加 prototype 是否有效
- `C3 vs C2`：看该创新点更适合哪个 backbone

## 9. 评估重点

本轮优先级固定为：

1. `normal recall`
2. `Real balanced_accuracy`
3. `Real macro_f1`
4. `test macro_f1`

因为本轮最重要的问题是：

- `BiLSTM` 的正常态优势是否能在 prototype 机制下进一步转化为整体优势。

## 10. 成功标准

`C3` 视为有效，至少需要：

1. `normal_recall > B2`
2. `real_balanced_accuracy >= B2`
3. `real_macro_f1 >= B2`

若进一步满足：

4. `real_macro_f1 > C2`
5. `real_balanced_accuracy > C2`

则说明：

- `Normal Prototype Calibration` 更适合 `BiLSTM`

此时最终主模型应重新考虑切回：

- `BiLSTM + Normal Prototype`

若 `C3` 只提升 `normal recall`，但整体不如 `C2`，则：

- `C2` 仍保留为当前主创新结果
- `C3` 作为“创新点跨 backbone 验证”补充结果

## 11. 论文与答辩表述

若 `C3` 成功，建议表述为：

> 为验证所提出的 Normal Prototype Calibration 机制是否具有 backbone 可迁移性，进一步将其迁移到 BiLSTM 基线模型上进行验证。实验结果表明，该机制在不同时序主干上均可改善正常态识别稳定性，说明其并非特定于某一 backbone 的偶然增益，而是具有普适性的类边界校准作用。

## 12. 当前边界

本轮明确**不包含**：

- domain adversarial alignment
- class-wise prototype
- class-wise margin
- 二阶段 normal/fault 分层分类
- 新的 CNN/LSTM 结构升级

本轮只验证：

- `BiLSTM + Normal Prototype`
