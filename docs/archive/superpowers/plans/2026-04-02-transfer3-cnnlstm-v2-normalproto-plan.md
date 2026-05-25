# transfer3 CNN-LSTM v2 + Normal Prototype Calibration 实施计划

## 1. 目标

本计划只实现并验证：

- `C2_cnnlstm_v2_normalproto_fewshot_v1`

目标是判断：

- 在 `CNN-LSTM v2 + few-shot` 基础上加入 `normal prototype` 后，
- 是否能够提升 `normal recall`
- 且不破坏 `motor / sensor` 故障判别能力。

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

`C1` 不作为本轮主对照，只作为失败路线记录。

## 3. 代码改动范围

### 3.1 新模型封装

建议新增：

- `src\models\cnn\cnn_lstm_v2_normalproto.py`

职责：

- 基于 `CNN-LSTM v2` 输出 pooled feature
- 添加 `class head`
- 持有可学习参数：
  - `normal_center`

### 3.2 训练脚本

新增：

- `src\models\train_transfer3_normalproto.py`

职责：

- 读取 few-shot cache
- 训练 `cnn_lstm_v2_normalproto`
- 计算分类损失与 prototype 正则
- 输出：
  - `metrics_val.json`
  - `metrics_test.json`
  - `metrics_external_test.json`
  - `config.json`

### 3.3 现有脚本不堆逻辑

以下文件不作为本轮主改动对象：

- `src\models\train_transfer3.py`
- `src\models\train_transfer3_dg.py`
- `src\models\cnn\cnn_lstm_v2.py`

## 4. 训练逻辑

### 4.1 特征来源

prototype 正则作用于：

- `CNN-LSTM v2` 的 pooled feature

即 attention pooling 后、分类头前的特征。

### 4.2 prototype 参数

仅维护一个可学习中心：

- `normal_center`

维度：

- 与 pooled feature 维度一致

### 4.3 输出模式

模型至少支持：

1. `forward(x)` 返回分类 logits
2. `forward_features(x)` 返回 pooled feature

不需要额外 domain 输出。

### 4.4 总损失

固定为：

```text
L_total = L_cls + λ_n * (L_pull + L_push)
```

其中：

- `L_cls = FocalLoss`
- `L_pull`：normal 样本向 `normal_center` 聚拢
- `L_push`：fault 样本与 `normal_center` 保持 margin

### 4.5 λ_n

首轮固定：

- `0.1`

### 4.6 margin

首轮固定：

- `1.0`

若失败，再调这两个参数，但第一版只实现这一组。

## 5. 测试计划

### 5.1 新增测试

新增测试文件：

- `tests\test_transfer3_normalproto.py`

至少覆盖：

1. `cnn_lstm_v2_normalproto.py` 存在并可编译；
2. 模型声明 `normal_center`；
3. 存在 `forward_features()`；
4. `train_transfer3_normalproto.py` 存在并支持 `cnn_lstm_v2_normalproto`；
5. 至少存在 `lambda_normal` / `normal_margin` 配置项。

### 5.2 回归测试

继续跑：

- `tests\test_transfer3_dg.py`
- `tests\test_cnn_lstm_v2.py`
- `tests\test_transfer3_backbone_entrypoints.py`
- `tests\test_compare_transfer3_a_layer.py`
- `tests\test_build_splits_v2.py`

## 6. 结果目录

本轮结果目录固定为：

- `D:\UAV-Data\transfer3_restart_v1\results\C2_cnnlstm_v2_normalproto_fewshot_v1`

AutoDL 远端建议目录：

- `/root/transfer3_run/20260402_cnnlstm_v2_normalproto`

上传目录建议：

- `D:\UAV-Data\transfer3_restart_v1\upload\cnnlstm_v2_normalproto`

上传文件：

1. `cache_transfer3_a_fewshot_v1.zip`
2. `train_transfer3_normalproto.py`
3. `cnn_models.zip`
4. `requirements.txt`

## 7. 本地汇总

继续复用：

- `scripts\compare_transfer3_backbones.py`

显式传入：

- `single_lstm_focal=B2_bilstm_focal_fewshot_v1`
- `cnn_lstm_v2_focal=B5_cnnlstm_v2_focal_fewshot_v1`
- `cnn_lstm_v2_normalproto=C2_cnnlstm_v2_normalproto_fewshot_v1`

输出目录建议：

- `D:\UAV-Data\transfer3_restart_v1\summary\c2_compare`

## 8. 成功标准

若 `C2` 同时满足：

1. `normal_recall > B5`
2. `real_balanced_accuracy >= B5`
3. `real_macro_f1 >= B5`
4. `test_macro_f1` 无明显退化

则 `normal prototype` 路线成立。

若进一步满足：

5. `normal_recall >= B2`
6. `real_macro_f1 >= B2` 或极接近

则 `C2` 可进入主创新候选。

## 9. 失败后分支

若 `C2` 失败，优先按以下顺序检查：

1. `λ_n` 是否过大；
2. margin 是否过强；
3. `L_push` 是否把 fault 类压坏；
4. `normal_center` 是否初始化/更新不稳定。

失败时可再考虑：

- class-wise margin
- class-wise prototype

但不直接回到 GRL。

## 10. 执行顺序

1. 新增 `cnn_lstm_v2_normalproto.py`
2. 新增 `train_transfer3_normalproto.py`
3. 写失败测试
4. 跑本地测试
5. 打包上传
6. AutoDL 训练 `C2`
7. 回传结果
8. 与 `B5 / B2` 做对比
9. 输出结论：成立 / 调参 / 改方案
