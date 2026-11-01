# 模型训练流水线

> 文档定位更新（2026-10-28）：**旧训练入口参考**。
> 原文包含旧 Real3 few-shot 训练路线，不是本轮“仅目标域正常数据适配”的新实验协议。不要直接执行旧命令开始新主实验；先核对协议、特征和适配／测试隔离。
> 最新方向、进度和后续任务统一见 [《项目进程与后续计划》](../项目进程.md)。

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

## Buaa/RflySim No Fault 额外 normal 样本

当实时链路或 BuaaAutoTestPlatform 的 No Fault case 被在线模型误报时，先把 case 合并为 RflyMAD-style CSV，再作为 extra normal 加入在线 cache：

```powershell
cd D:\UAV-Fault-Diagnosis
.\apps\dashboard\backend\.venv\Scripts\python.exe .\scripts\data\merge_buaa_ulog_case.py `
  --case-dir D:\BuaaAutoTestPlatForm\Data\TestCase_189_1419000188 `
  --out .\runtime\extra_normals\nf_buaa_case_189_1419000188.csv `
  --target-hz 50
```

合并后先诊断分布漂移：

```powershell
.\apps\dashboard\backend\.venv\Scripts\python.exe .\scripts\eval\diagnose_live_schema_drift.py `
  --csv .\runtime\extra_normals\nf_buaa_case_189_1419000188.csv `
  --model-dir .\assets\models\online_detector_kg_mstca_rfly `
  --backend replay `
  --every 50 `
  --out .\runtime\buaa_case_189_schema_drift_summary.csv `
  --windows-out .\runtime\buaa_case_189_schema_drift_windows.csv
```

重建在线 cache 时传入 extra normal：

```powershell
.\apps\dashboard\backend\.venv\Scripts\python.exe .\scripts\train_pipeline.py `
  --task online `
  --stages splits cache train summary `
  --data-root <processed_RflyMAD root with manifest> `
  --work-root <training run dir> `
  --online-extra-normal-dir .\runtime\extra_normals `
  --online-extra-normal-glob nf_*.csv `
  --online-extra-normal-max-windows-per-file 128 `
  --execute
```

这条路径用于系统修正 Buaa/RflySim 正常工况覆盖不足；不要把 `runtime/extra_normals` 当作仓库交付数据提交。

### 复用本机 processed_RflyMAD split

如果本机已经存在按历史实验固定好的 split 目录：

```text
data/processed_RflyMAD/
├─ train/
├─ val/
├─ test/
└─ external_test/
```

不要再用 `build_online_real3_splits.py` 随机重分割，先把现有 split materialize 成 manifest + splits：

```powershell
.\apps\dashboard\backend\.venv\Scripts\python.exe .\scripts\data\materialize_processed_splits.py `
  --processed-root .\data\processed_RflyMAD `
  --output-dir .\runtime\training_full
```

如果 `manifest_from_processed_splits.csv` 已经扫过完整 CSV，可复用它快速重写 splits/summary：

```powershell
.\apps\dashboard\backend\.venv\Scripts\python.exe .\scripts\data\materialize_processed_splits.py `
  --processed-root .\data\processed_RflyMAD `
  --output-dir .\runtime\training_full `
  --reuse-manifest
```

本机已验证的 split 分布：

```text
train         2368 files: motor 1177, normal 308, sensor 883
val            592 files: motor 295, normal 76, sensor 221
test           741 files: motor 369, normal 96, sensor 276
external_test  345 files: motor 172, normal 36, sensor 137
```

所有 4046 个样本均为 `quality_flag=ok` 且 `has_required_online_columns=1`。使用现有 split 和 Buaa extra normal 构建正式候选 cache：

```powershell
.\apps\dashboard\backend\.venv\Scripts\python.exe .\packages\uav_fault\uav_fault\data\build_online_cache.py `
  --csv-dir .\data\processed_RflyMAD `
  --splits .\runtime\training_full\splits_from_processed_splits.json `
  --output .\runtime\training_full\cache_online_real3_binary_with_buaa_nf `
  --manifest-path .\runtime\training_full\manifest_from_processed_splits.csv `
  --window-size 150 `
  --stride 10 `
  --target-hz 50 `
  --fault-threshold 0.5 `
  --max-windows-per-file 8 `
  --extra-normal-dir .\runtime\extra_normals `
  --extra-normal-glob nf_*.csv `
  --extra-normal-split train `
  --extra-normal-max-windows-per-file 128 `
  --protocol-name rflymad_online_real3_binary_existing_splits_with_buaa_nf_v1 `
  --label-space real3 `
  --source-fault-classes motor sensor
```

本机已验证该 cache 可成功构建，窗口统计为：

```text
split          files  src_nf  src_ff  normal  fault  total
train          2369     309    2060    9963   9109  19072
val             592      76     516    2455   2281   4736
test            741      96     645    3054   2874   5928
external_test   345      36     309    1924    794   2718
```

当前 probe 结论：

- 原始 `online_detector_kg_mstca_rfly` 在 `TestCase_189_1419000188` No Fault 上 `mean_fault_prob≈0.984`，属于明确 false alarm。
- 小型 extra-normal probe 可把该 case 压到 `mean_fault_prob≈0.081`，说明 Buaa/RflySim normal domain alignment 有效。
- 同一个小型 probe 会漏掉 `TS-416582` 的故障段，说明不能把 probe 模型发布；正式修正必须使用完整 train/val/test/external_test 覆盖后重训，并同时验收 No Fault false alarm 和 fault recall。
- 发布候选模型最低验收：`TestCase_189_1419000188` 不产生事件，原有 labeled fault replay 不出现系统性漏检，`metrics_external_test.json` 不退化到 fault recall 接近 0。

## 当前不纳入默认流水线的内容

- 旧 6 分类实验：Real 数据只有 `normal/motor/sensor`，不适合作为当前交付主线。
- ULG 转 CSV：RflySim 平台的转换链路仍需单独研究。
- 复制到 `assets/models`：训练结果确认后再手动或另写发布脚本执行，避免误覆盖当前运行资产。
