# transfer3 CNN-LSTM v2 实施计划

## 1. 目标

本计划只覆盖 `CNN-LSTM v2` 的结构实现与验证，不引入其它增强策略，不与 `domain alignment / normal calibration / attention upgrade` 混合。

本轮唯一问题：

> `CNN-LSTM v2` 是否能够在 `few-shot Real` 协议下优于旧版 `CNN-LSTM`？

## 2. 约束

固定约束如下：

- 任务固定：`transfer3 = normal / motor / sensor`
- 协议固定：`few-shot`
- cache 固定复用：
  - `D:\UAV-Data\transfer3_restart_v1\cache\cache_transfer3_a_fewshot_v1`
- 损失函数固定：
  - `FocalLoss`
- 训练入口继续使用：
  - `src\models\train_transfer3.py`
- 对照组固定为：
  - `B4_cnnlstm_focal_fewshot_v1`
- 不新增其它 backbone 实验

## 3. 实现范围

### 3.1 新模型文件

新增一个独立模型文件：

- `src\models\cnn\cnn_lstm_v2.py`

职责：

- 定义 `CNNLSTMv2Classifier`
- 不修改旧版 `CNNLSTMClassifier`

### 3.2 训练入口扩展

修改：

- `src\models\train_transfer3.py`

增加新模型选项：

- `--model cnn_lstm_v2`

并将其映射到：

- `CNNLSTMv2Classifier`

### 3.3 汇总脚本兼容

复用现有：

- `scripts\compare_transfer3_backbones.py`

通过显式 `--experiment` 参数追加：

- `cnn_lstm_v2_focal=B5_cnnlstm_v2_focal_fewshot_v1`

不再单独新写一套比较脚本。

## 4. 模型实现分解

### 4.1 Raw Skip Path

实现要求：

- 输入 `(N, C, T)`
- 将原始序列下采样到 `512` 步
- 转换为 `(N, 512, 32)` 表示

建议实现方式：

- 先 `x.transpose(1, 2)` 得到 `(N, T, C)`
- 使用步长采样 `::4`
- 再用 `Linear(in_channels, 32)` 做投影

### 4.2 Multi-scale CNN Path

实现要求：

- 3 个并行分支：
  - `k=3`
  - `k=7`
  - `k=15`
- 每个分支输出 `32` 通道
- 分支结构统一：
  - `Conv1d`
  - `GroupNorm`
  - `ReLU`

拼接后：

- 通道维为 `96`

### 4.3 Residual Conv Block

实现要求：

- 输入 `96` 通道
- 输出 `128` 通道
- 包含 shortcut
- 使用 `GroupNorm`
- 池化最多两次

目标：

- 最终长度到 `512`
- 不允许压到 `256`

### 4.4 Fusion + BiLSTM

融合要求：

- `raw_skip` 与 `cnn_path` 对齐到同一时间长度 `512`
- 按特征维拼接
- 输入到 `BiLSTM`

固定参数：

- `hidden_size=128`
- `num_layers=2`
- `bidirectional=True`
- `dropout=0.3`

### 4.5 Attention Head

继续沿用旧版实现风格：

- `Linear -> Tanh -> Linear -> Softmax`
- `LayerNorm + Dropout + Linear`

## 5. 测试计划

### 5.1 单元测试

新增测试文件：

- `tests\test_cnn_lstm_v2.py`

至少覆盖：

1. 前向输出 shape 正确：
   - 输入 `(batch, 31, 2048)`
   - 输出 `(batch, 3)`
2. Raw skip path 与 CNN path 时间长度对齐为 `512`
3. `cnn_lstm_v2` 已接入 `train_transfer3.py` 的模型选择

### 5.2 训练入口回归

继续跑现有相关测试，至少包括：

- `tests\test_transfer3_backbone_entrypoints.py`
- `tests\test_compare_transfer3_a_layer.py`
- `tests\test_build_splits_v2.py`

### 5.3 实验验证

只新增一组实验：

- `B5_cnnlstm_v2_focal_fewshot_v1`

然后与旧版：

- `B4_cnnlstm_focal_fewshot_v1`

做对照。

## 6. 实验目录与命名

本轮新增结果目录：

- `D:\UAV-Data\transfer3_restart_v1\results\B5_cnnlstm_v2_focal_fewshot_v1`

AutoDL 远端建议目录：

- `/root/transfer3_run/20260402_cnnlstm_v2`

上传目录建议：

- `D:\UAV-Data\transfer3_restart_v1\upload\cnnlstm_v2`

上传文件：

1. `cache_transfer3_a_fewshot_v1.zip`
2. 最新 `train_transfer3.zip`
3. 最新 `cnn_models.zip`
4. `requirements.txt`

## 7. 判定标准

若 `B5` 同时满足以下条件，则视为 v2 成立：

1. `real_macro_f1 > B4`
2. `real_balanced_accuracy > B4`
3. `normal_recall > B4`
4. `test_macro_f1` 无明显退化

其中优先级最高的是：

- `normal_recall`
- `real_balanced_accuracy`

若仅 `motor / sensor` 更高，但 `normal` 仍无改善，则不进入下一阶段。

## 8. 执行顺序

1. 新增 `cnn_lstm_v2.py`
2. 为 v2 写失败测试
3. 接入 `train_transfer3.py`
4. 跑本地测试
5. 重打包训练入口与 `cnn/` 目录
6. 上传 AutoDL
7. 训练 `B5`
8. 回传结果
9. 用 `compare_transfer3_backbones.py` 生成新旧对比表
10. 输出结论：保留旧版 / 切换 v2 / 放弃 `CNN-LSTM` 主线

## 9. 本计划不做的事

当前不做：

- `CNN-LSTM v2 + domain alignment`
- `CNN-LSTM v2 + normal calibration`
- `CNN-LSTM v2 + AdaBN`
- 更换 loss
- 更换任务定义

本轮只验证结构升级本身。
