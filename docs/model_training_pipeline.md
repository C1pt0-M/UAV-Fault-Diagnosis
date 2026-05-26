# 模型训练流水线

这份说明只覆盖当前主线：RflyMAD/RflySim 数据上的离线三分类诊断和在线二分类检测。旧的 6 分类实验先作为历史材料保留，不放进默认流水线。

## 入口

统一入口：

```bash
python3 -B scripts/train_pipeline.py --task both --data-root /mnt/d/UAV-Data
```

默认是 dry-run，只打印将要执行的命令，不写入训练结果。确认路径和参数后再加 `--execute`。

Windows 原生环境可使用：

```powershell
python -B scripts\train_pipeline.py --task both --data-root D:\UAV-Data
```

完整训练需要当前 Python 环境安装 `torch`、`numpy`、`pandas`、`scikit-learn`、`matplotlib` 等训练依赖。当前 WSL 系统 Python 没有 `torch`，因此更适合作为审计和 dry-run 环境；实际训练可以放在 Windows GPU 环境或 AutoDL/Linux 训练机上跑。

## 默认产物

默认输出到：

```text
D:\UAV-Data\training_runs\pipeline_YYYYmmdd_HHMMSS\
```

在 WSL 中对应：

```text
/mnt/d/UAV-Data/training_runs/pipeline_YYYYmmdd_HHMMSS/
```

目录结构：

```text
training_runs/<run>/
├─ manifest/
│  └─ manifest_v2.local.csv
├─ offline_transfer3/
│  ├─ splits_transfer3_v1/
│  ├─ cache_transfer3_fewshot_v1/
│  ├─ results/transfer3_cnnlstm_normalproto/
│  └─ summary/
└─ online_real3_binary/
   ├─ splits_online_real3_binary_v1/
   ├─ cache_online_real3_binary_v1/
   ├─ models/
   └─ summary/
```

`manifest_v2.local.csv` 会把 `source_csv_path` 转成本机可访问的路径，例如把 `D:\UAV-Data\...` 转成 `/mnt/d/UAV-Data/...`。如果训练环境已经使用正确路径，可以加 `--skip-local-manifest`。

## 离线三分类

默认任务是 `normal/motor/sensor` 三分类，复现当前系统使用的主线模型：

```bash
python3 -B scripts/train_pipeline.py --task offline --data-root /mnt/d/UAV-Data
```

执行时会串联：

1. `build_transfer3_splits.py`
2. `build_sequence_cache_3class.py`
3. `train_transfer3_normalproto.py`
4. `compare_transfer3_backbones.py`

默认模型是 `cnn_lstm_v2_normalproto`，few-shot 使用 `train + real_adapt_train`，验证使用 `val + real_adapt_val`，Real holdout 使用 `real_holdout_test`。

## 在线二分类

默认任务是 `normal/fault` 短窗口检测，但数据范围压缩到 Real 可验证的三类：`normal/motor/sensor`。其中 `normal` 作为负类，`motor/sensor` 合并为 `fault` 正类，协议名为 `rflymad_online_real3_binary_v1`。

```bash
python3 -B scripts/train_pipeline.py --task online --data-root /mnt/d/UAV-Data
```

执行时会串联：

1. `build_online_real3_splits.py`
2. `build_online_cache.py`
3. `run_online_backbone_suite.py`

默认在线模型为当前运行资产对应的三个 backbone：

```text
kg_mstca, mstca, cnn1d
```

如需额外比较 BiLSTM 或 Transformer：

```bash
python3 -B scripts/train_pipeline.py --task online --data-root /mnt/d/UAV-Data --online-models kg_mstca mstca cnn1d bilstm transformer
```

## 常用参数

只生成离线命令：

```bash
python3 -B scripts/train_pipeline.py --task offline --data-root /mnt/d/UAV-Data
```

只重跑训练和汇总，复用当前 run 下已有 split/cache：

```bash
python3 -B scripts/train_pipeline.py --task offline --work-root /mnt/d/UAV-Data/training_runs/<run> --stages train summary
```

指定训练 Python：

```bash
python3 -B scripts/train_pipeline.py --task both --python-bin /path/to/python --data-root /mnt/d/UAV-Data
```

真正执行：

```bash
python3 -B scripts/train_pipeline.py --task both --data-root /mnt/d/UAV-Data --execute
```

如果 `work_root` 非空，默认会停止，避免覆盖旧结果。确实要复用该目录时再加：

```bash
--force
```

## 当前不纳入默认流水线的内容

- 旧 6 分类实验：Real 数据只有 `normal/motor/sensor`，不适合作为当前交付主线。
- ULG 转 CSV：RflySim 平台的转换链路仍需单独研究。
- 复制到 `assets/models`：训练结果确认后再手动或另写发布脚本执行，避免误覆盖当前运行资产。
