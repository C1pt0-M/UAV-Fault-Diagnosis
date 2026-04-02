# transfer3 BiLSTM + Normal Prototype Calibration 实施计划

## 1. 目标

本计划只实现并验证：

- `C3_bilstm_normalproto_fewshot_v1`

目标是判断：

- `BiLSTM + Normal Prototype`
- 是否优于 `B2_bilstm_focal_fewshot_v1`
- 是否优于或接近 `C2_cnnlstm_v2_normalproto_fewshot_v1`

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
- `C2_cnnlstm_v2_normalproto_fewshot_v1`

## 3. 代码改动范围

### 3.1 新模型封装

建议新增：

- `src\models\cnn\bilstm_normalproto.py`

职责：

- 定义 `BiLSTMNormalProtoClassifier`
- 输出 pooled feature
- 添加 `class head`
- 持有可学习参数：
  - `normal_center`

### 3.2 训练脚本

新增：

- `src\models\train_transfer3_bilstm_normalproto.py`

职责：

- 读取 few-shot cache
- 训练 `bilstm_normalproto`
- 计算分类损失与 prototype 正则
- 输出：
  - `metrics_val.json`
  - `metrics_test.json`
  - `metrics_external_test.json`
  - `config.json`

### 3.3 现有文件保持不动

以下文件不作为本轮主改动对象：

- `src\models\train_transfer3.py`
- `src\models\train_transfer3_normalproto.py`
- `src\models\cnn\lstm.py`

除非为最小复用添加极小辅助函数，否则不回写复杂逻辑。

## 4. 训练逻辑

### 4.1 特征来源

prototype 正则作用于：

- `BiLSTM` 的 attention pooling 输出

### 4.2 prototype 参数

固定：

- 单一 `normal_center`

### 4.3 总损失

与 `C2` 保持一致：

```text
L_total = L_cls + λ_n * (L_pull + L_push)
```

其中：

- `L_cls = FocalLoss`
- `L_pull`：normal 样本靠近 `normal_center`
- `L_push`：fault 样本远离 `normal_center`

### 4.4 超参

为保证横向可比，首轮固定与 `C2` 一致：

- `λ_n = 0.1`
- `normal_margin = 1.0`

### 4.5 backbone

固定：

- `BiLSTM`

不改变其 hidden size、层数和 attention 结构。

## 5. 测试计划

### 5.1 新增测试

新增测试文件：

- `tests\test_transfer3_bilstm_normalproto.py`

至少覆盖：

1. `bilstm_normalproto.py` 存在并可编译；
2. 模型声明 `normal_center`；
3. 存在 `forward_features()`；
4. `train_transfer3_bilstm_normalproto.py` 存在并支持 `bilstm_normalproto`；
5. 至少存在 `lambda_normal / normal_margin` 参数。

### 5.2 回归测试

继续跑：

- `tests\test_transfer3_normalproto.py`
- `tests\test_transfer3_dg.py`
- `tests\test_cnn_lstm_v2.py`
- `tests\test_transfer3_backbone_entrypoints.py`
- `tests\test_compare_transfer3_a_layer.py`
- `tests\test_build_splits_v2.py`

## 6. 结果目录

本轮结果目录固定为：

- `D:\UAV-Data\transfer3_restart_v1\results\C3_bilstm_normalproto_fewshot_v1`

AutoDL 远端建议目录：

- `/root/transfer3_run/20260402_bilstm_normalproto`

上传目录建议：

- `D:\UAV-Data\transfer3_restart_v1\upload\bilstm_normalproto`

上传文件：

1. `cache_transfer3_a_fewshot_v1.zip`
2. `train_transfer3_bilstm_normalproto.py`
3. `cnn_models.zip`
4. `requirements.txt`

## 7. 本地汇总

继续复用：

- `scripts\compare_transfer3_backbones.py`

显式传入：

- `single_lstm_focal=B2_bilstm_focal_fewshot_v1`
- `cnn_lstm_v2_normalproto=C2_cnnlstm_v2_normalproto_fewshot_v1`
- `bilstm_normalproto=C3_bilstm_normalproto_fewshot_v1`

输出目录建议：

- `D:\UAV-Data\transfer3_restart_v1\summary\c3_compare`

## 8. 成功标准

若 `C3` 同时满足：

1. `normal_recall > B2`
2. `real_balanced_accuracy >= B2`
3. `real_macro_f1 >= B2`

则说明：

- `BiLSTM + Normal Prototype` 这条线有效

若进一步满足：

4. `real_macro_f1 > C2`
5. `real_balanced_accuracy > C2`

则最终主模型建议切换为：

- `BiLSTM + Normal Prototype`

若只提升 `normal recall`，但整体不如 `C2`，则：

- `C2` 保持主结果
- `C3` 作为创新点跨 backbone 验证结果

## 9. 失败后分支

若 `C3` 失败，优先检查：

1. `λ_n` 是否过大；
2. `normal_margin` 是否过强；
3. `BiLSTM` 本身是否已足够偏向 `normal`，再加强 prototype 反而造成 fault 类受损。

失败时不继续堆新模块，直接停止该线。

## 10. 执行顺序

1. 新增 `bilstm_normalproto.py`
2. 新增 `train_transfer3_bilstm_normalproto.py`
3. 写失败测试
4. 跑本地测试
5. 打包上传
6. AutoDL 训练 `C3`
7. 回传结果
8. 与 `B2 / C2` 做对比
9. 输出结论：保留 `C2` / 切换 `C3` / 并列保留
