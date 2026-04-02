# transfer3 A层实验设计

## 1. 目标

A层只回答一个问题：

> 在 `transfer3 = normal / motor / sensor` 的跨域任务中，少量 `Real` 样本是否值得引入训练？

本层不比较 backbone，不引入任何新模块，不讨论算法创新。A层的唯一目的，是先把后续主线中最核心的迁移前提确认下来：

- `pure simulation/semi-real training` 是否不足以支撑 `Real` 泛化；
- `few-shot Real adaptation` 是否能显著改善真实域性能；
- 后续 B 层及以后是否默认统一采用 `few-shot` 协议。

## 2. 任务与数据协议

### 2.1 固定任务

A层任务固定为：

- `transfer3 = normal / motor / sensor`

类别定义不变：

- `normal`
- `motor`
- `sensor`

不引入 `propeller / wind / load`，不回切 `main_6class`。

### 2.2 固定 split 协议

继续复用现有 `transfer3` split：

- `train`
- `val`
- `test`
- `real_adapt_train`
- `real_adapt_val`
- `real_holdout_test`

数据来源固定为：

- split: `D:\UAV-Data\splits_transfer3_v1`
- processed data: `D:\UAV-Data\processed_SIL_50hz`
- processed data: `D:\UAV-Data\processed_HIL_50hz`
- processed data: `D:\UAV-Data\processed_Real_50hz`

### 2.3 cache 策略

A层采用**严格协议**，重新构建两套干净 cache，不复用历史 `cache_transfer3_v1`，原因是 A层要验证 `baseline` 与 `few-shot` 的迁移增益，必须避免共享 `norm_stats` 或训练口径混淆。

两套 cache 固定为：

- `cache_transfer3_a_baseline_v1`
  - `train_splits_for_norm = ["train"]`
- `cache_transfer3_a_fewshot_v1`
  - `train_splits_for_norm = ["train", "real_adapt_train"]`

统一参数：

- `label_column = label_main_3class`
- `seq_len = 2048`
- 通道集合保持当前 `31` 维，不做特征工程变更

### 2.4 新实验根目录

本轮重启实验不写入旧目录，统一放到：

- `D:\UAV-Data\transfer3_restart_v1\`

其下建议固定为：

- `cache\`
- `upload\`
- `results\`
- `summary\`

## 3. 模型与实验矩阵

### 3.1 A层基础模型

A层基础模型固定为：

- `BiLSTM + FocalLoss`

理由：

- `BiLSTM` 是基础时序模型，不是混合结构；
- 它足以代表时序建模能力，又不会像 `CNN-LSTM` 一样提前引入结构创新；
- 相比 `TextCNN`，它更适合作为迁移前提验证模型。

### 3.2 A层实验矩阵

只跑两组：

#### A1: `A1_bilstm_focal_baseline_v1`

- 模型：`BiLSTM + FocalLoss`
- cache：`cache_transfer3_a_baseline_v1`
- 训练 split：
  - `train`
- 验证 split：
  - `val`
- 测试 split：
  - `test`
- 真实域评估：
  - `real_holdout_test`

该组表示：

- 不引入任何 `Real` 样本参与训练
- 仅使用 `SIL + HIL`

#### A2: `A2_bilstm_focal_fewshot_v1`

- 模型：`BiLSTM + FocalLoss`
- cache：`cache_transfer3_a_fewshot_v1`
- 训练 split：
  - `train + real_adapt_train`
- 验证 split：
  - `val + real_adapt_val`
- 测试 split：
  - `test`
- 真实域评估：
  - `real_holdout_test`

该组表示：

- 在 `SIL + HIL` 基础上引入少量 `Real`
- 目标是验证真实域适配是否显著提升

## 4. 指标、输出与判定规则

### 4.1 主指标

A层指标优先级固定为：

1. `Real holdout macro_f1`
2. `Real holdout balanced_accuracy`
3. `normal / motor / sensor` 的逐类 `recall`
4. `SIL/HIL test macro_f1`

### 4.2 必须输出的文件

每个实验目录必须至少包含：

- `best.pt`
- `config.json`
- `metrics_val.json`
- `metrics_test.json`
- `metrics_external_test.json`

### 4.3 A层汇总

在 `D:\UAV-Data\transfer3_restart_v1\summary\` 下生成：

- `a_layer_summary.csv`
- `a_layer_summary.md`

汇总字段至少包括：

- `experiment_id`
- `model`
- `protocol`
- `test_accuracy`
- `test_balanced_accuracy`
- `test_macro_f1`
- `real_accuracy`
- `real_balanced_accuracy`
- `real_macro_f1`
- `normal_recall`
- `motor_recall`
- `sensor_recall`
- `epochs_trained`
- `best_val_f1`

### 4.4 判定规则

若 `A2` 同时满足以下条件，则正式确认后续统一默认采用 `few-shot` 协议：

- `real_macro_f1` 显著高于 `A1`
- `real_balanced_accuracy` 显著高于 `A1`
- 至少一个弱类（优先关注 `normal`）的 `recall` 明显提升

若 `A2` 仅提升 `accuracy`，但 `macro_f1 / balanced_accuracy` 不提升，则不视为迁移策略成立。

## 5. 与后续实验的关系

A层完成后只允许得出以下结论之一：

### 结论 A

`few-shot Real` 显著优于 `baseline`。

则：

- B层所有 backbone 对比默认统一采用 `few-shot`
- 后续不再重复为每个 backbone 跑 `baseline`

### 结论 B

`few-shot Real` 提升不明显。

则：

- 先暂停 B层
- 重新检查 `few-shot` 配比、cache 口径或 `Real` 样本质量

在 A层完成前，不进入 B层，不进入算法创新层。

## 6. 当前默认假设

本设计默认以下前提成立：

- 现有 `splits_transfer3_v1` 口径有效，不重新切分；
- A层需要**重新构建**两套严格 cache；
- A层只使用 `BiLSTM + FocalLoss`；
- A层只跑 `A1 / A2` 两组；
- A层完成后再决定 B层是否统一改为 few-shot 主协议。
