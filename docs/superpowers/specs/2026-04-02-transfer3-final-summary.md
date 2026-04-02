# transfer3 最终实验总表与结果总表

## 1. 最终结论

当前 `transfer3 = normal / motor / sensor` 主线实验已经可以定版。

最终保留方案：

- **主任务**：`transfer3 = normal / motor / sensor`
- **主协议**：`few-shot Real adaptation`
- **最终主模型**：`CNN-LSTM v2 + Normal Prototype Calibration`

最终不保留为主结果的路线：

- `纯 SIL/HIL baseline`：仅作为迁移必要性对照
- `TextCNN`：性能偏弱
- `Transformer`：不如最终方案
- `CNN-LSTM v2 + GRL`：失败路线
- `BiLSTM + Normal Prototype`：补充对照，不作为最终主模型

## 2. 实验总表

### 2.1 A层：few-shot 必要性验证

| 层级 | 实验ID | 模型 | 协议 | 目的 | 结论 |
|---|---|---|---|---|---|
| A | `A1_bilstm_focal_baseline_v1` | `BiLSTM + FocalLoss` | `baseline` | 验证纯 `SIL+HIL` 是否足够 | 不足以支撑 `Real` 泛化 |
| A | `A2_bilstm_focal_fewshot_v1` | `BiLSTM + FocalLoss` | `few-shot` | 验证少量 `Real` 是否必要 | 成立，后续统一采用 `few-shot` |

### 2.2 B层：few-shot 条件下 backbone 对比

| 层级 | 实验ID | 模型 | 协议 | 目的 | 结论 |
|---|---|---|---|---|---|
| B | `B1_textcnn_focal_fewshot_v1` | `TextCNN + FocalLoss` | `few-shot` | 单 `CNN` 对照 | 较弱，淘汰 |
| B | `B2_bilstm_focal_fewshot_v1` | `BiLSTM + FocalLoss` | `few-shot` | 单 `LSTM` 基线 | 强基线 |
| B | `B3_transformer_fewshot_v1` | `Transformer` | `few-shot` | 全局时序建模对照 | 中等，不作为主线 |
| B | `B4_cnnlstm_focal_fewshot_v1` | `CNN-LSTM + FocalLoss` | `few-shot` | 初版混合结构 | 有优势但 `normal recall` 不足 |
| B | `B5_cnnlstm_v2_focal_fewshot_v1` | `CNN-LSTM v2 + FocalLoss` | `few-shot` | 升级混合结构 | 优于旧版 `CNN-LSTM` |

### 2.3 C层：创新策略验证

| 层级 | 实验ID | 模型 | 协议 | 创新点 | 结论 |
|---|---|---|---|---|---|
| C | `C1_cnnlstm_v2_grl_fewshot_v1` | `CNN-LSTM v2 + GRL` | `few-shot` | 全局对抗式域对齐 | 失败，`sensor` 受损 |
| C | `C2_cnnlstm_v2_normalproto_fewshot_v1` | `CNN-LSTM v2 + Normal Prototype` | `few-shot` | 定向 normal calibration | 成功，当前最佳 |
| C | `C3_bilstm_normalproto_fewshot_v1` | `BiLSTM + Normal Prototype` | `few-shot` | 跨 backbone 验证 | 不如 `C2` |

## 3. 结果总表

### 3.1 A层结果

| 实验ID | Test Accuracy | Test Balanced Accuracy | Test Macro F1 | Real Accuracy | Real Balanced Accuracy | Real Macro F1 | Normal Recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| `A1_bilstm_focal_baseline_v1` | 0.9879 | 0.9784 | 0.9826 | 0.4116 | 0.3183 | 0.2858 | 0.0000 |
| `A2_bilstm_focal_fewshot_v1` | 0.9825 | 0.9713 | 0.9733 | 0.8174 | 0.7522 | 0.7498 | 0.5833 |

### 3.2 B层核心结果

| 实验ID | 模型 | Real Accuracy | Real Balanced Accuracy | Real Macro F1 |
|---|---|---:|---:|---:|
| `B1_textcnn_focal_fewshot_v1` | `TextCNN + FocalLoss` | 0.7478 | 0.6354 | 0.6417 |
| `B2_bilstm_focal_fewshot_v1` | `BiLSTM + FocalLoss` | 0.8174 | 0.7522 | 0.7498 |
| `B3_transformer_fewshot_v1` | `Transformer` | 0.7855 | 0.7020 | 0.6883 |
| `B4_cnnlstm_focal_fewshot_v1` | `CNN-LSTM + FocalLoss` | 0.8377 | 0.7078 | 0.7232 |
| `B5_cnnlstm_v2_focal_fewshot_v1` | `CNN-LSTM v2 + FocalLoss` | 0.8435 | 0.7390 | 0.7570 |

### 3.3 C层核心结果

| 实验ID | 模型 | Real Accuracy | Real Balanced Accuracy | Real Macro F1 | Normal Recall | 结论 |
|---|---|---:|---:|---:|---:|---|
| `B2_bilstm_focal_fewshot_v1` | `BiLSTM + FocalLoss` | 0.8174 | 0.7522 | 0.7498 | 0.5833 | 强基线 |
| `B5_cnnlstm_v2_focal_fewshot_v1` | `CNN-LSTM v2 + FocalLoss` | 0.8435 | 0.7390 | 0.7570 | 0.4444 | 结构升级有效 |
| `C1_cnnlstm_v2_grl_fewshot_v1` | `CNN-LSTM v2 + GRL` | 0.7623 | 0.7104 | 0.6828 | 0.6111 | 失败，整体退化 |
| `C2_cnnlstm_v2_normalproto_fewshot_v1` | `CNN-LSTM v2 + Normal Prototype` | **0.8609** | **0.7600** | **0.7840** | 0.4722 | **最终主模型** |
| `C3_bilstm_normalproto_fewshot_v1` | `BiLSTM + Normal Prototype` | 0.8058 | 0.7347 | 0.7210 | 0.5556 | 不如 `C2` |

## 4. 核心发现

### 4.1 `few-shot Real` 是必要条件

`A2` 相比 `A1` 的提升非常显著：

- `Real Macro F1`: `0.2858 -> 0.7498`
- `Real Balanced Accuracy`: `0.3183 -> 0.7522`
- `Normal Recall`: `0.0000 -> 0.5833`

说明：

- 后续所有主实验都应统一采用 `few-shot` 协议

### 4.2 backbone 不是唯一决定因素

在 `few-shot` 条件下，单纯更换 backbone 的增益有限，但仍能看出：

- `TextCNN` 最弱
- `BiLSTM` 是强基础模型
- `CNN-LSTM` 初版对 `motor / sensor` 较强，但 `normal` 偏弱
- `CNN-LSTM v2` 通过结构升级已优于旧版 `CNN-LSTM`

### 4.3 全局 GRL 不适合当前主线

`C1` 虽然一度改善 `normal recall`，但显著破坏 `sensor` 边界，导致整体退化。

因此：

- 不应继续把全局对抗式域对齐作为当前主创新主线

### 4.4 定向 `Normal Prototype Calibration` 有效

`C2` 相比 `B5` 实现了整体提升：

- `Real Accuracy`: `0.8435 -> 0.8609`
- `Real Balanced Accuracy`: `0.7390 -> 0.7600`
- `Real Macro F1`: `0.7570 -> 0.7840`

并且没有像 `C1` 那样把 `sensor` 类拖垮。

### 4.5 `Normal Prototype` 更适合 `CNN-LSTM v2`

`C3` 证明：

- 同样的 `Normal Prototype Calibration` 放到 `BiLSTM` 上，并未优于 `C2`

说明：

- 该创新点与 `CNN-LSTM v2` 的多尺度局部特征表达更匹配

## 5. 当前定版结论

当前 `transfer3` 主线最终定版为：

> 主任务采用 `transfer3 = normal / motor / sensor`，主协议采用 `few-shot Real adaptation`，主模型采用 `CNN-LSTM v2 + Normal Prototype Calibration`。该方案在 `Real holdout` 上取得 `Accuracy = 0.8609`、`Balanced Accuracy = 0.7600`、`Macro F1 = 0.7840`，为当前最优结果。

## 6. 展示建议

比赛展示中建议只保留 3 类模型：

1. `BiLSTM + FocalLoss`
2. `CNN-LSTM v2 + FocalLoss`
3. `CNN-LSTM v2 + Normal Prototype`

展示逻辑：

- 先展示基础模型
- 再展示结构升级
- 最后展示最终创新模型

这三者最能支撑“基础模型 -> 结构创新 -> 算法创新”的完整叙事。
