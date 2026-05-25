# transfer3 CNN-LSTM v2 + GRL 实施计划

## 1. 目标

本计划只实现并验证一个实验：

- `C1_cnnlstm_v2_grl_fewshot_v1`

目标是判断：

- `CNN-LSTM v2 + GRL` 是否优于 `B5_cnnlstm_v2_focal_fewshot_v1`
- 是否逼近或超过 `B2_bilstm_focal_fewshot_v1`

## 2. 固定输入

### 2.1 任务

- `transfer3 = normal / motor / sensor`

### 2.2 协议

- `few-shot`

### 2.3 cache

固定复用：

- `D:\UAV-Data\transfer3_restart_v1\cache\cache_transfer3_a_fewshot_v1`

本轮不重建 cache。

### 2.4 对照实验

固定对照：

- `B2_bilstm_focal_fewshot_v1`
- `B5_cnnlstm_v2_focal_fewshot_v1`

## 3. 代码改动范围

### 3.1 新模型封装

建议新增：

- `src\models\cnn\cnn_lstm_v2_dg.py`

职责：

- 基于 `CNN-LSTM v2` backbone 输出 pooled feature
- 添加 `class head`
- 添加 `domain head`
- 支持 `forward(x)` 与 `forward(x, alpha, return_domain=True)`

### 3.2 训练脚本

新增：

- `src\models\train_transfer3_dg.py`

职责：

- 读取 few-shot cache
- 训练 `cnn_lstm_v2_dg`
- 计算分类损失与域损失
- 输出：
  - `metrics_val.json`
  - `metrics_test.json`
  - `metrics_external_test.json`
  - `config.json`

### 3.3 现有脚本保持不动

以下文件不作为本轮主改动对象：

- `src\models\train_transfer3.py`
- `src\models\cnn\cnn_lstm.py`
- `src\models\cnn\cnn_lstm_v2.py`

除非为了最小复用做极小接口补充，否则不回写复杂逻辑。

## 4. 训练逻辑

### 4.1 Domain label 来源

直接使用 cache 中已有字段：

- `domain_y`

domain mapping 固定：

- `SIL -> 0`
- `HIL -> 1`
- `Real -> 2`

### 4.2 输出模式

模型需支持：

1. `return_domain=False`
   - 仅返回分类 logits
2. `return_domain=True`
   - 返回 `(class_logits, domain_logits)`

### 4.3 损失

固定总损失：

```text
L_total = L_cls + λ_d * L_domain
```

其中：

- `L_cls = FocalLoss`
- `L_domain = CrossEntropyLoss`

### 4.4 alpha 调度

沿用 DANN 风格：

```text
alpha = 2 / (1 + exp(-10 * p)) - 1
```

### 4.5 λ_d

首轮固定：

- `0.2`

如首轮无效，再做小范围调参：

- `0.1`
- `0.3`

但第一版实现只做 `0.2`。

## 5. 测试计划

### 5.1 新增测试

新增测试文件：

- `tests\test_transfer3_dg.py`

至少覆盖：

1. `cnn_lstm_v2_dg` 模型文件存在并可编译；
2. `forward(return_domain=True)` 返回二元输出；
3. `domain head` 输出类别数为 `3`；
4. `train_transfer3_dg.py` 至少支持单步 smoke 运行或入口解析。

### 5.2 回归测试

继续跑：

- `tests\test_cnn_lstm_v2.py`
- `tests\test_transfer3_backbone_entrypoints.py`
- `tests\test_compare_transfer3_a_layer.py`
- `tests\test_build_splits_v2.py`

## 6. 结果目录

本轮结果目录固定为：

- `D:\UAV-Data\transfer3_restart_v1\results\C1_cnnlstm_v2_grl_fewshot_v1`

AutoDL 远端建议目录：

- `/root/transfer3_run/20260402_cnnlstm_v2_grl`

上传目录建议：

- `D:\UAV-Data\transfer3_restart_v1\upload\cnnlstm_v2_grl`

上传文件：

1. `cache_transfer3_a_fewshot_v1.zip`
2. `train_transfer3_dg.py`
3. `cnn_models.zip`
4. `requirements.txt`

## 7. 本地汇总

继续复用：

- `scripts\compare_transfer3_backbones.py`

显式传入：

- `single_lstm_focal=B2_bilstm_focal_fewshot_v1`
- `cnn_lstm_v2_focal=B5_cnnlstm_v2_focal_fewshot_v1`
- `cnn_lstm_v2_grl=C1_cnnlstm_v2_grl_fewshot_v1`

输出目录建议：

- `D:\UAV-Data\transfer3_restart_v1\summary\c1_compare`

## 8. 成功标准

若 `C1` 同时满足：

1. `real_macro_f1 > B5`
2. `real_balanced_accuracy > B5`
3. `normal_recall >= B5`
4. `test_macro_f1` 无明显退化

则 `GRL` 路线成立。

若进一步满足：

5. `real_macro_f1 >= B2`
6. `real_balanced_accuracy >= B2`

则 `CNN-LSTM v2 + GRL` 进入主创新候选。

## 9. 失败后分支

若 `C1` 失败，优先按以下顺序检查：

1. `λ_d` 是否过大；
2. domain head 是否过强；
3. feature 接入位置是否过晚；
4. `Real` 少样本在 domain 分支中是否被过度放大。

失败时不直接进入 `normal calibration`。

## 10. 执行顺序

1. 新增 `cnn_lstm_v2_dg.py`
2. 新增 `train_transfer3_dg.py`
3. 写失败测试
4. 跑本地测试
5. 打包上传
6. AutoDL 训练 `C1`
7. 回传结果
8. 与 `B5 / B2` 做对比
9. 输出结论：进入下一阶段 / 回调参数 / 暂停 `GRL` 路线
