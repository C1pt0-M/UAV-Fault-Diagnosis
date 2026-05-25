# transfer3 A层实施计划

## 1. 目标与完成标准

本计划只覆盖 A 层实验：

- `A1_bilstm_focal_baseline_v1`
- `A2_bilstm_focal_fewshot_v1`

完成标准：

1. 在独立目录下重建两套严格 cache；
2. 在 AutoDL 上完成 A1 / A2 训练；
3. 回传结果到本地独立结果目录；
4. 输出 A 层汇总表与结论；
5. 明确 B 层是否统一采用 `few-shot` 协议。

## 2. 目录与命名

统一使用新实验根目录：

- `D:\UAV-Data\transfer3_restart_v1`

其下固定结构：

- `cache\`
- `upload\`
- `results\`
- `summary\`

具体目录：

- `D:\UAV-Data\transfer3_restart_v1\cache\cache_transfer3_a_baseline_v1`
- `D:\UAV-Data\transfer3_restart_v1\cache\cache_transfer3_a_fewshot_v1`
- `D:\UAV-Data\transfer3_restart_v1\results\A1_bilstm_focal_baseline_v1`
- `D:\UAV-Data\transfer3_restart_v1\results\A2_bilstm_focal_fewshot_v1`
- `D:\UAV-Data\transfer3_restart_v1\summary\a_layer_summary.csv`
- `D:\UAV-Data\transfer3_restart_v1\summary\a_layer_summary.md`
- `D:\UAV-Data\transfer3_restart_v1\summary\experiment_manifest.csv`

## 3. 本机数据准备

### 3.1 split 复用

继续使用：

- `D:\UAV-Data\splits_transfer3_v1`

不重新切分，不改类别协议。

### 3.2 严格 cache 重建

输入脚本：

- `D:\UAV-Fault-Diagnosis\src\data\build_sequence_cache_3class.py`

统一参数：

- `label_column = label_main_3class`
- `seq_len = 2048`

baseline cache：

- 输出：`D:\UAV-Data\transfer3_restart_v1\cache\cache_transfer3_a_baseline_v1`
- `train_splits_for_norm = ["train"]`

few-shot cache：

- 输出：`D:\UAV-Data\transfer3_restart_v1\cache\cache_transfer3_a_fewshot_v1`
- `train_splits_for_norm = ["train", "real_adapt_train"]`

### 3.3 cache 验证

重建后必须检查：

1. `build_summary.json` 存在；
2. `norm_stats.json` 存在；
3. baseline 的 `train_splits_for_norm` 仅包含 `train`；
4. few-shot 的 `train_splits_for_norm` 包含 `train + real_adapt_train`；
5. split 样本数与原协议一致。

## 4. 上传准备

本轮上传目录：

- `D:\UAV-Data\transfer3_restart_v1\upload`

建议上传文件：

1. `cache_transfer3_a_baseline_v1.zip`
2. `cache_transfer3_a_fewshot_v1.zip`
3. `train_transfer3.zip`
4. `cnn_models.zip`
5. `requirements.txt`

训练入口代码必须使用本地最新版本：

- `src\models\train_transfer3.py`

模型定义包必须包含：

- `src\models\cnn\lstm.py`
- `src\models\cnn\focal_loss.py`
- `src\models\cnn\__init__.py`

如 AutoDL 端没有旧缓存，上传两套 cache；如已有且确认未污染，可跳过重复上传。

## 5. AutoDL 训练执行

远端工作目录建议固定为：

- `/root/transfer3_run/20260402_a_layer`

### 5.1 A1

- 模型：`lstm`
- mode：`baseline`
- cache：`cache_transfer3_a_baseline_v1`
- 输出：`results/A1_bilstm_focal_baseline_v1`

### 5.2 A2

- 模型：`lstm`
- mode：`fewshot`
- cache：`cache_transfer3_a_fewshot_v1`
- 输出：`results/A2_bilstm_focal_fewshot_v1`

### 5.3 训练环境约束

训练前统一设置：

- `OMP_NUM_THREADS`
- `MKL_NUM_THREADS`

避免 `libgomp` 非法线程告警影响记录。

## 6. 结果回传与本地整理

从 AutoDL 回传：

- `results/A1_bilstm_focal_baseline_v1`
- `results/A2_bilstm_focal_fewshot_v1`

放回本地：

- `D:\UAV-Data\transfer3_restart_v1\results\`

然后本地生成：

- `a_layer_summary.csv`
- `a_layer_summary.md`

汇总字段：

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

## 7. 判定规则

若 `A2` 同时满足：

- `real_macro_f1 > A1`
- `real_balanced_accuracy > A1`
- `normal recall` 或其他弱类 `recall` 有明确提升

则确认：

- B 层默认统一采用 `few-shot` 协议

否则：

- 暂停进入 B 层
- 回查 cache、few-shot 配比和 `Real` 样本质量

## 8. 风险与边界

### 8.1 主要风险

1. baseline / few-shot 共用历史 cache 导致结论污染；
2. AutoDL 目录混乱导致结果覆盖；
3. 输出目录命名不统一，后续难以汇总；
4. 只看 accuracy，忽略 `macro_f1 / balanced_accuracy / class recall`。

### 8.2 本计划不做的事

- 不比较 backbone；
- 不做 `CNN-LSTM`；
- 不做 `Transformer`；
- 不做 domain alignment；
- 不做 normal calibration；
- 不做任何 C 层创新实验。

## 9. 执行清单

1. 创建 `transfer3_restart_v1` 目录结构；
2. 重建 `baseline / few-shot` 两套 cache；
3. 校验 `build_summary.json` 与 split 计数；
4. 打包上传文件到 AutoDL；
5. 训练 `A1`；
6. 训练 `A2`；
7. 回传结果；
8. 生成 `a_layer_summary.csv / md`；
9. 输出 A 层结论，决定是否进入 B 层。
